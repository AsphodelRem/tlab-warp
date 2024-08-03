import copy

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)

        
class WarpTrainer:
    def __init__(
        self, 
        sft: AutoModelForCausalLM, 
        sft_tokenizer: AutoTokenizer,
        reward_model: AutoModelForSequenceClassification, 
        reward_model_tokenizer: AutoTokenizer,
        warp_config: map,
        dataset: list[str]
    ):
        """
        Initializes the WarpTrainer.
    
        Args:
        - sft (AutoModelForCausalLM): The model to be aligned.
        - sft_tokenizer (AutoTokenizer): The tokenizer for the sft model.
        - reward_model (AutoModelForSequenceClassification): The reward model.
        - reward_model_tokenizer (AutoTokenizer): The tokenizer for the reward model.
        - warp_config (map): Configuration for the training process.
        - dataset (list[str]): Dataset used for training.
        """
        self.config = warp_config
        self.sft = sft
        self.sft_tokenizer = sft_tokenizer
        self.reward_model_tokenizer = reward_model_tokenizer
        self.reward_model = reward_model
        self.dataset = dataset
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.sft.to(self.device)
        self.reward_model.to(self.device)
      
    def train(self):
        """
        Run the alignment process.
        """
        theta_init = copy.deepcopy(self.sft)
        theta_list = []
        for i in tqdm(range(self.config['warp']['iterations'])):
            for run in range(self.config['warp']['ml_runs']):
                theta = copy.deepcopy(theta_init)
                theta_ema = copy.deepcopy(theta_init)
                optimizer = torch.optim.Adam(theta.parameters(), lr=self.config['warp']['lr'])
                data_loader = self._get_dataloader(self.dataset)
                for step in tqdm(range(self.config['warp']['training_steps'])):
                    batched_prompts = next(iter(data_loader))
                    
                    optimizer.zero_grad()
                    
                    theta_completions, theta_log_probs = self._generate_completion(
                        theta, 
                        self.sft_tokenizer, 
                        batched_prompts
                    )
                    
                    sft_completions, sft_log_probs = self._generate_completion(
                        theta_ema, 
                        self.sft_tokenizer, 
                        batched_prompts
                    )

                    reward = self._compute_reward(
                        self.reward_model, 
                        self.reward_model_tokenizer, 
                        theta_completions
                    )

                    kl_reg_reward = self._apply_kl_regularization(
                        reward, 
                        theta_log_probs, 
                        sft_log_probs
                    )
        
                    gp = self._compute_policy_gradient(theta_log_probs, kl_reg_reward)
                    gp.backward()
                    optimizer.step()
                    
                theta_list.append(theta)
                    
            theta_slerp = self._slerp(theta_init, theta_list, 1 / self.config['warp']['ml_runs'])
            theta_list.clear()
            theta_init = self._models_averaging(theta_init, theta_slerp, self.config['warp']['liti_update_rate'])

        self.sft = self._models_averaging(self.sft, theta_slerp, self.config['warp']['liti_update_rate'])

    def save_model(self, save_dir: str=None) -> None:
        """
        Save the aligned model.
        """
        if save_dir is None:
            save_dir = self.config['warp']['output_dir']
        self.sft.save_pretrained(save_dir, from_pt=True)
        self.sft_tokenizer.save_pretrained(save_dir)

    def _get_dataloader(
        self, 
        dataset: list, 
        data_collator: callable=None
    ) -> DataLoader:
        """
        Create DataLoader for the given dataset.
    
        Args:
        dataset (torch.utils.data.Dataset): The dataset to load.
        data_collator (optional): Function to collate data.
    
        Returns:
        torch.utils.data.DataLoader: The DataLoader for the dataset.
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['warp']['batch_size'],
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader
    
    def _generate_completion(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        batched_prompts: list[str]
    ) -> tuple[list[str], torch.Tensor]:
        """
        Generate text completions and computes log probabilities.
    
        Args:
        - model (AutoModelForCausalLM): The model used for text generation.
        - tokenizer (AutoTokenizer): The tokenizer for the model.
        - batched_prompts (list[str]): Batch of initial text prompts.
    
        Returns:
        - list[str]: Generated text completions.
        - torch.Tensor: Log probabilities of the generated words.
        """
        
        model_input = tokenizer(
            batched_prompts, 
            return_tensors='pt', 
            truncation=True, 
            padding=True,
        ).to(self.device)
        
        outputs = model.generate(**model_input, max_length=20)
        decoded_completions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        log_probs = self._get_word_probs(model, outputs)
        
        return decoded_completions, log_probs

    def _models_averaging(
        self, 
        model_a: AutoModelForCausalLM, 
        model_b: AutoModelForCausalLM, 
        alpha: float
    ) -> AutoModelForCausalLM:
        """
        Average parameters of two models.
    
        Args:
        - model_a (AutoModelForCausalLM): The first model.
        - model_b (AutoModelForCausalLM): The second model.
        - alpha (float): Averaging coefficient.
    
        Returns:
        - AutoModelForCausalLM: The model with averaged parameters.
        """
        
        averaged_state_dict = {}
        for key in model_a.state_dict().keys():
            averaged_state_dict[key] = (1 - alpha) * model_a.state_dict()[key] + alpha * model_b.state_dict()[key]
            
        avg_model = AutoModelForCausalLM.from_pretrained(
                self.config['warp']['sft_model_name_or_path'], 
                state_dict=averaged_state_dict
        )
        return avg_model

    def _get_word_probs(
        self, 
        model: AutoModelForCausalLM, 
        batched_completions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities of words for a batch of completions.
    
        Args:
        model (AutoModelForCausalLM): The model to compute probabilities.
        batched_completions (torch.Tensor): Batch of generated completions.
    
        Returns:
        torch.Tensor: Log probabilities of the words.
        """
        logits = model(batched_completions).logits
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def _kl_div(self, alignment_log_probs: torch.Tensor, sft_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the Kullback-Leibler divergence between two distributions.
    
        Args:
        - alignment_log_probs (torch.Tensor): Log probabilities of the first distribution.
        - sft_log_probs (torch.Tensor): Log probabilities of the second distribution.
    
        Returns:
        - torch.Tensor: The Kullback-Leibler divergence.
        """
        
        return torch.mean(alignment_log_probs - sft_log_probs)
        
    def _compute_reward(
        self, 
        reward_model: AutoModelForSequenceClassification, 
        reward_model_tokenizer: AutoTokenizer, 
        batched_completions: list[str]
    ) -> torch.Tensor:
        """
        Compute a reward for a batch of completions.
    
        Args:
        - reward_model (AutoModelForSequenceClassification): The reward model.
        - reward_model_tokenizer (AutoTokenizer): The tokenizer for the reward model.
        - batched_completions (list[str]): Batch of text completions.
    
        Returns:
        - torch.Tensor: Rewards for each completion.
        """
        
        model_input = reward_model_tokenizer(
            batched_completions, 
            truncation=True, 
            padding=True, 
            return_tensors='pt'
        ).to(self.device)
        
        return torch.sigmoid(reward_model(**model_input).logits)

    def _apply_kl_regularization(
        self, 
        reward: torch.Tensor, 
        alignment_log_probs: torch.Tensor, 
        sft_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply KL divergence regularization to the rewards.
    
        Args:
        - reward (torch.Tensor): Rewards for the completions.
        - alignment_log_probs (torch.Tensor): Log probabilities of the alignment model.
        - sft_log_probs (torch.Tensor): Log probabilities of the sft model.
    
        Returns:
        - torch.Tensor: KL-regularized rewards.
        """
        kl_div = self._kl_div(alignment_log_probs, sft_log_probs)
        
        return torch.mean(reward - self.config['warp']['beta'] * kl_div)
        
    def _compute_policy_gradient(
        self, 
        log_probs: torch.Tensor, 
        kl_reg_reward: torch.Tensor, 
        reduce: str='mean'
    ) -> torch.Tensor:
        """
        Compute the policy gradient.
    
        Args:
        - log_probs (torch.Tensor): Log probabilities of the actions.
        - kl_reg_reward (torch.Tensor): Regularized rewards.
        - reduce (str): Aggregation method ('sum' or 'mean').
    
        Returns:
        - torch.Tensor: Policy gradient value.
        """
        
        if reduce == 'sum':
            return -torch.sum(log_probs * kl_reg_reward)
        elif reduce == 'mean':
            return -torch.mean(log_probs * kl_reg_reward)
        else:
            raise NotImplementedError

    def _slerp(
        self, 
        theta_init: AutoModelForCausalLM, 
        theta_m_list: list[AutoModelForCausalLM], 
        lam: float
    ) -> AutoModelForCausalLM:
        """
        Perform Spherical Linear Interpolation (SLERP) between models.
    
        Args:
        - theta_init (AutoModelForCausalLM): The initial model.
        - theta_m_list (list[AutoModelForCausalLM]): List of models to interpolate.
        - lam (float): Interpolation parameter.
    
        Returns:
        - AutoModelForCausalLM: The interpolated model.
        """
        for key in theta_init.state_dict().keys(): 
            theta_init_vec = theta_init.state_dict()[key]
            v_0 = theta_m_list[0].state_dict()[key]
            v_1 = theta_m_list[1].state_dict()[key]
    
            delta_0 = v_0 - theta_init_vec
            delta_1 = v_1 - theta_init_vec
            
            omega = torch.arccos((delta_0 * delta_1).sum() / (delta_0.norm() * delta_1.norm()))
            delta = torch.sin((1.0 - lam) * omega) / \
                torch.sin(omega) * delta_0 + torch.sin(lam * omega) / torch.sin(omega) * delta_1
            
            theta_init.state_dict()[key] += delta
            
        return AutoModelForCausalLM.from_pretrained(
            self.config['warp']['sft_model_name_or_path'],
            state_dict=theta_init.state_dict()
        )

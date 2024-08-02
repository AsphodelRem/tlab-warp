import copy
from dataclasses import dataclass, field

from tqdm import tqdm
import torch
import torch.nn.functional as F

from transformers import (
    HfArgumentParser,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
        
class WarpTrainer:
    def __init__(
        self, 
        sft, 
        sft_tokenizer,
        reward_model, 
        reward_model_tokenizer,
        warp_config,
        dataset
    ):
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
        theta_init = copy.deepcopy(self.sft)
        theta_list = []
        for i in tqdm(range(self.config.iterations)):
            for run in range(self.config.ml_runs):
                theta = copy.deepcopy(theta_init)
                theta_ema = copy.deepcopy(theta_init)
                optimizer = torch.optim.Adam(theta.parameters(), lr=1e-6)
                data_loader = self.get_dataloader(self.dataset)
                for step in tqdm(range(self.config.training_steps)):
                    batched_prompts = next(iter(data_loader))
                    
                    theta_completions, theta_log_probs = self.generate_completion(
                        theta, 
                        self.sft_tokenizer, 
                        batched_prompts
                    )
                    sft_completions, sft_log_probs = self.generate_completion(
                        theta_ema, 
                        self.sft_tokenizer, 
                        batched_prompts
                    )

                    reward = self.compute_reward(
                        self.reward_model, 
                        self.reward_model_tokenizer, 
                        theta_completions
                    )

                    kl_reg_reward = self.apply_kl_regularization(
                        reward, 
                        theta_log_probs, 
                        sft_log_probs
                    )
                    gp = self.compute_policy_gradient(theta_log_probs, kl_reg_reward)
                    gp.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                theta_list.append(theta)
                    
            theta_slerp = self.slerp(theta_init, theta_list, 1 / self.config.ml_runs)
            theta_list.clear()
            theta_init = self.models_averaging(theta_init, theta_slerp, self.config.liti_update_rate)

        output_model = self.models_averaging(self.sft, theta_slerp, self.config.liti_update_rate)
        return output_model

    def get_dataloader(self, dataset, data_collator=None):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config._batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader
    
    def generate_completion(self, model, tokenizer, batched_prompts):
        model_input = tokenizer(
            batched_prompts, 
            return_tensors='pt', 
            truncation=True, 
            padding=True
        ).to(self.device)
        outputs = model.generate(**model_input, max_length=20)
        decoded_completions = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        log_probs = self.get_word_probs(model, outputs)
        return decoded_completions, log_probs

    def models_averaging(self, model_a, model_b, alpha: float):
        averaged_state_dict = {}
        avg_model = copy.deepcopy(model_a)
        for key in model_a.state_dict().keys():
            averaged_state_dict[key] = (1 - alpha) * model_a.state_dict()[key] + alpha * model_b.state_dict()[key]
            
        avg_model = GPT2LMHeadModel.from_pretrained(
                'lvwerra/gpt2-imdb', 
                state_dict=averaged_state_dict
        )
        return avg_model

    def get_word_probs(self, model, batched_completions):
        with torch.no_grad():
            logits = model(batched_completions).logits
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def kl_div(self, alignment_log_probs, sft_log_probs):
        return torch.sum(alignment_log_probs - sft_log_probs)
        
    def compute_reward(
        self, 
        reward_model, 
        reward_model_tokenizer, 
        batched_completions
    ):
        model_input = reward_model_tokenizer(
            batched_completions, 
            truncation=True, 
            padding=True, 
            return_tensors='pt'
        ).to(self.device)
        
        return torch.sigmoid(reward_model(**model_input).logits)

    def apply_kl_regularization(self, reward, alignment_log_probs, sft_log_probs):
        kl_div = self.kl_div(alignment_log_probs, sft_log_probs)
        
        return torch.sum(reward - self.config.beta * kl_div)
        
    def compute_policy_gradient(self, log_probs, kl_reg_reward, reduce: str='sum'):
        if reduce == 'sum':
            return -torch.sum(log_probs * kl_reg_reward)
        elif reduce == 'mean':
            return -torch.mean(log_probs * kl_reg_reward)
        else:
            raise NotImplementedError

    def slerp(self, theta_init, theta_m_list, lam):
        def slerp_two(theta_0, theta_1, lam):
            slerp_state_dict = {}
            for key in theta_0.state_dict().keys():
                v0 = theta_0.state_dict()[key]
                v1 = theta_1.state_dict()[key]
                omega = torch.arccos((v0 * v1).sum() / (v0.norm() * v1.norm()))
                sin_omega = torch.sin(omega)
                
                slerp_state_dict[key] = torch.sin((1.0 - lam) * omega) / \
                    sin_omega * v0 + torch.sin(lam * omega) / sin_omega * v1
                
            return GPT2LMHeadModel.from_pretrained(
                'lvwerra/gpt2-imdb', 
                state_dict=slerp_state_dict
            )
        
        result = theta_init
        for theta_m in theta_m_list:
            result = slerp_two(result, theta_m, lam)
    
        return result

class WarpConfig:
    def __init__(
        self, 
        iterations: int=1, 
        ml_runs: int=2,
        training_steps: int=100,
        ema_update_rate: float=0.01,
        liti_update_rate: float=0.01,
        beta: float=0.5,
        batch_size: int=16
    ):
        self._iterations = iterations
        self._ml_runs = ml_runs
        self._training_steps = training_steps
        self._ema_update_rate = ema_update_rate
        self._liti_update_rate = liti_update_rate
        self._beta = beta
        self._batch_size = batch_size 

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        if value < 0:
            raise ValueError('Iterations cannot be negative')
        self._iterations = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if not (0 <= value <= 1):
            raise ValueError('Beta must be between 0 and 1')
        self._beta = value

    @property
    def ml_runs(self):
        return self._ml_runs

    @ml_runs.setter
    def ml_runs(self, value):
        if value < 0:
            raise ValueError('ML runs cannot be negative')
        self._ml_runs = value

    @property
    def training_steps(self):
        return self._training_steps

    @training_steps.setter
    def training_steps(self, value):
        if value < 0:
            raise ValueError('Training steps cannot be negative')
        self._training_steps = value

    @property
    def ema_update_rate(self):
        return self._ema_update_rate

    @ema_update_rate.setter
    def ema_update_rate(self, value):
        if not (0 <= value <= 1):
            raise ValueError('EMA update rate must be between 0 and 1')
        self._ema_update_rate = value

    @property
    def liti_update_rate(self):
        return self._liti_update_rate

    @liti_update_rate.setter
    def liti_update_rate(self, value):
        if not (0 <= value <= 1):
            raise ValueError('LITI update rate must be between 0 and 1')
        self._liti_update_rate = value

    @property
    def batch_size(self):
        return self._batch_size

    @iterations.setter
    def iterations(self, value):
        if value < 0:
            raise ValueError('Batch size cannot be negative')
        self._batch_size = value
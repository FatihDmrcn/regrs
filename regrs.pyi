import numpy as np


class OLS:


    def __init__(self, exog: np.ndarray, endog: np.ndarray, add_const = False):
        self.size_samples: int
        self.size_params: int
        self.rss: float
        self.r2: float
        self.r2_predicted: float

    def summary(self) -> None:
        '''
        Prints out the summary of the least square analysis!
        '''
import numpy as np


class OLS:

    def __init__(self, exog: np.ndarray, endog: np.ndarray):
        self.exog = exog
        self.endog = endog
        self.r2_predicted: float
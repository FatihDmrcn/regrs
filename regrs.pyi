import numpy as np


class OLS:
    def __init__(self, exog: np.ndarray, endog: np.ndarray):
        self.exog = exog
        self.endog = endog

    def r2_predicted(self) -> float:
        """
        Computing the predicted R² by iteratively
        leaving one row from, exog. and endog. data
        and computing the error based on the left out
        data row. 

        :return: Predicted R² as float
        """
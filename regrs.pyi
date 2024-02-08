def r2_predicted(exog, endog) -> float:
    """
    Computing the predicted R² by iteratively
    leaving one row from, exog. and endog. data
    and computing the error based on the left out
    data row. 

    :return: Predicted R² as float
    """
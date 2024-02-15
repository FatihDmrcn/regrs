import regrs
import numpy as np
import os 

os.environ["RUST_BACKTRACE"] = "1"

endog = np.arange(0,100,1).astype(float)
exog = np.array([endog,]*3).T

ols = regrs.OLS(exog, endog, True)
print(ols)
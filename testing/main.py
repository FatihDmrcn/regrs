import regrs
import numpy as np

endog = np.arange(0,100,1).astype(float)
exog = np.array([endog,]*3).T
print(endog, exog)

ols = regrs.OLS(exog=exog, endog=endog)
r2_predicted = ols.r2_predicted()
print(r2_predicted)
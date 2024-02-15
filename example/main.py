import regrs
import numpy as np
import matplotlib.pyplot as plt

# import os 
# os.environ["RUST_BACKTRACE"] = "1"
 
exog = 0.3*np.arange(0,100,1) + 0.6
endog = exog + np.random.normal(0, 2, len(exog))

# plt.scatter(exog, endog)
# plt.show()

ols = regrs.OLS(exog.reshape(-1,1), endog, add_const=True)
print(ols)
import regrs
import numpy as np
import matplotlib.pyplot as plt
 
exog = 0.3*np.arange(0,100,1) + 0.6
endog = exog + np.random.normal(0, 4, len(exog))

ols = regrs.OLS(exog.reshape(-1,1), endog, add_const=True)
print(ols)

plt.scatter(exog, endog)
plt.show()
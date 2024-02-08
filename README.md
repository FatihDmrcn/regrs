# regrs

> Computing the predicted R² by iteratively leaving one row from, exog. and endog. data and computing the error based on the left out data row. And since each iteration step is independent it is very much suitable to be parallilized. Hence, I wrote a corresponding function in Rust and compiled in to a .whl-File so that it can be easily used in Python. 
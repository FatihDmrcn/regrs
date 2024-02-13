use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::*;
use ndarray_linalg::LeastSquaresSvd;


#[pyclass]
struct OLS{
    exog: Array2<f64>,
    endog: Array1<f64>,
    // r2: f64,
    // r2_adjusted: f64,
    // r2_predicted: f64,
}

#[pymethods]
impl OLS{
    #[new]
    fn new(exog: PyReadonlyArray2<f64>, endog: PyReadonlyArray1<f64>) -> Self {
        let _exog: Array2<f64> = exog.to_owned_array();
        let _endog: Array1<f64> = endog.to_owned_array();
        Self{
            exog: _exog, 
            endog: _endog, 
        }
    }

    fn r2_predicted(&self) -> PyResult<f64> {
        let len = self.exog.nrows();
        let mut errors: Vec<f64> = Vec::new();
        (0..len).into_par_iter()
            .map(|_i| {
                let mut a_train = self.exog.clone();
                a_train.remove_index(Axis(0), _i);
                let mut b_train = self.endog.clone();
                b_train.remove_index(Axis(0), _i);
                let fit = a_train.least_squares(&b_train).unwrap().solution;
                let a_test = self.exog.row(_i);
                let b_test = self.endog.get(_i);
                (b_test.unwrap() - fit.dot(&a_test)).powf(2.0)
            })
            .collect_into_vec(&mut errors);
        let press = Array1::from_vec(errors).sum();
        Ok(1. - (press / self.endog.var(0.)) / len as f64)
    }
}

#[pymodule]
fn regrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OLS>()?;
    Ok(())
}

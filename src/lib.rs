use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{ndarray::*,PyReadonlyArray1,PyReadonlyArray2,};
use ndarray_linalg::LeastSquaresSvd;


#[pyfunction]
fn r2_predicted(exog: PyReadonlyArray2<f64>, endog: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let exog = exog.to_owned_array();
    let endog = endog.to_owned_array();
    let len = exog.nrows();
    let mut errors: Vec<f64> = Vec::new();
    (0..len).into_par_iter()
        .map(|_i| {
            let mut a_train = exog.clone();
            a_train.remove_index(Axis(0), _i);
            let mut b_train = endog.clone();
            b_train.remove_index(Axis(0), _i);
            let fit = a_train.least_squares(&b_train).unwrap().solution;
            let a_test = exog.row(_i);
            let b_test = endog.get(_i);
            (b_test.unwrap() - fit.dot(&a_test)).powf(2.0)
        })
        .collect_into_vec(&mut errors);
    let press = Array1::from_vec(errors).sum();
    let r2_predicted = 1. - (press / endog.var(0.)) / len as f64;
    Ok(r2_predicted)
}


#[pymodule]
fn regrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(r2_predicted, m)?)?;
    Ok(())
}

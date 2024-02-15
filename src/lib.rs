use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::*;
use ndarray_linalg::{LeastSquaresResult, LeastSquaresSvd};


fn _r2_predicted(exog: &Array2<f64>, endog: &Array1<f64>, n: &usize) -> f64 {
    let len = n;
    let mut errors: Vec<f64> = Vec::new();
    (0..*len).into_par_iter()
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
    1. - (press / endog.var(0.)) / *len as f64
}

fn _least_squares(exog: &mut Array2<f64>, endog: &Array1<f64>) -> LeastSquaresResult<f64, Dim<[usize; 1]>> {
    exog.least_squares(endog).unwrap()
}


#[pyclass]
struct OLS{
    size_samples: usize,
    size_params: usize,
    // rss: f64,
    // r2: f64,
    // r2_adjusted: f64,
    r2_predicted: f64,
}


#[pymethods]
impl OLS{
    #[new]
    fn new(exog: PyReadonlyArray2<f64>, endog: PyReadonlyArray1<f64>, add_const: Option<bool>) -> Self {
        // EXOG
        let mut _exog: Array2<f64> = exog.to_owned_array();
        let _n = _exog.nrows();
        if add_const.unwrap_or(false) {
            let _ones = Array2::ones((_n, 1));
            _exog = concatenate(Axis(1), &[_exog.view(), _ones.view()]).unwrap();
        }
        let _p = _exog.ncols();
        // ENDOG
        let _endog: Array1<f64> = endog.to_owned_array();
        //
        let _r2_predicted: f64 = _r2_predicted(&_exog, &_endog, &_n);
        let _least_squares_res = _least_squares(&mut _exog, &_endog);
        //
        // let _rss: f64 = _least_squares_res.residual_sum_of_squares.unwrap().into_scalar() as f64;
        let _var: f64 = _endog.var(0.);
        // let _r2: f64 = 1. - (_rss/(_var * _n as f64));
        Self{
            size_samples: _n,
            size_params: _p,
            // rss: _rss,
            // r2: _r2,
            r2_predicted: _r2_predicted
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SUMMARY OF OLS\n
            Sample Size: {}\n
            Parameter Size: {}\n
            R² (predicted): {}",
            self.size_samples, self.size_params, self.r2_predicted) //self.rss, self.r2,)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[getter]
    fn size_samples(&self) -> PyResult<usize> {Ok(self.size_samples)}

    #[getter]
    fn size_params(&self) -> PyResult<usize> {Ok(self.size_params)}

    #[getter]
    fn r2_predicted(&self) -> PyResult<f64> {Ok(self.r2_predicted)}
    /*
    #[getter]
    fn r2(&self) -> PyResult<f64> {Ok(self.r2)}

    #[getter]
    fn rss(&self) -> PyResult<f64> {Ok(self.rss)}
    */
}


#[pymodule]
fn regrs<'py>(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OLS>()?;
    Ok(())
}
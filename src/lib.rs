use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::*;
use ndarray_linalg::{LeastSquaresResult, LeastSquaresSvd};


fn _least_squares(exog: &mut Array2<f64>, endog: &Array1<f64>) -> LeastSquaresResult<f64, Dim<[usize; 1]>> {
    exog.least_squares(endog).unwrap()
}

fn _tss(endog: &Array1<f64>) -> f64 {
    let _mean = endog.mean().unwrap();
    endog.map(|f| (f-_mean).powf(2.0)).sum()
}

fn _rss(exog: &mut Array2<f64>, endog: &Array1<f64>, beta: &Array1<f64>) -> f64 {
    (endog - exog.dot(beta)).into_iter().map(|f| f.powf(2.0)).sum()
}

fn _press(exog: &Array2<f64>, endog: &Array1<f64>, n: &usize) -> f64 {
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
    Array1::from_vec(errors).sum()
}

fn _r2(tss: &f64, rss: &f64) -> f64{
    1. - (rss/tss)
}

fn _r2_adjusted(tss: &f64, rss: &f64, n: &usize, p: &usize) -> f64{
    let df_res = *n-*p;
    let df_tot = *n-1;
    1. - ((rss/df_res as f64)/(tss/df_tot as f64))
}

fn _r2_predicted(press: &f64, var: &f64, n: &usize) -> f64 {
    1. - (press / var) / *n as f64
}

#[pyclass]
struct OLS{
    size_samples: usize,
    size_params: usize,
    tss: f64,
    rss: f64,
    press: f64,
    r2: f64,
    r2_adjusted: f64,
    r2_predicted: f64,
    var_endog: f64
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
        // OLS
        let _least_squares_res = _least_squares(&mut _exog, &_endog);
        let _beta = _least_squares_res.solution;
        // STATS
        let _tss: f64 = _tss(&_endog);
        let _rss: f64 = _rss(&mut _exog, &_endog, &_beta);
        let _press: f64 = _press(&_exog, &_endog, &_n);
        let _var: f64 = _endog.var(0.);
        let _r2: f64 = _r2(&_tss, &_rss);
        let _r2_adjusted: f64 = _r2_adjusted(&_tss, &_rss, &_n, &_p);
        let _r2_predicted: f64 = _r2_predicted(&_press, &_var, &_n);
        // STRUCT
        Self{
            size_samples: _n,
            size_params: _p,
            tss: _tss,
            rss: _rss,
            press: _press,
            r2: _r2,
            r2_adjusted: _r2_adjusted,
            r2_predicted: _r2_predicted,
            var_endog: _var
        }
    }

    fn __repr__(&self) -> String {
        let _hline_dstroke: String = "=".repeat(84);
        let _hline_stroke: String = "-".repeat(84);
        let mut _lines: Vec<String> = vec![format!("|{: ^82}|", "RegRS SUMMARY")];
        _lines.push(_hline_dstroke.clone());
        _lines.push(format!("|{0: <20}{1: >20}||{2: <20}{3: >20}|",
        "Sample Size",self.size_samples,"Parameter Size",self.size_params));
        _lines.push(_hline_stroke.clone());
        let _elements = vec![
            ("TSS",self.tss,"R²",self.r2),
            ("RSS",self.rss,"R² (adjusted)",self.r2_adjusted),
            ("PRESS",self.press,"R² (predicted)",self.r2_predicted),];
        for (_l_label, _l_value, _r_label, _r_value) in _elements {
            _lines.push(format!("|{0: <20}{1: >20.4}||{2: <20}{3: >20.4}|",_l_label, _l_value, _r_label, _r_value));
        }
        _lines.push(_hline_dstroke.clone());
        _lines.join("\n")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[getter]
    fn size_samples(&self) -> PyResult<usize> {Ok(self.size_samples)}

    #[getter]
    fn size_params(&self) -> PyResult<usize> {Ok(self.size_params)}

    #[getter]
    fn tss(&self) -> PyResult<f64> {Ok(self.tss)}

    #[getter]
    fn rss(&self) -> PyResult<f64> {Ok(self.rss)}

    #[getter]
    fn press(&self) -> PyResult<f64> {Ok(self.press)}

    #[getter]
    fn r2(&self) -> PyResult<f64> {Ok(self.r2)}

    #[getter]
    fn r2_adjusted(&self) -> PyResult<f64> {Ok(self.r2_adjusted)}

    #[getter]
    fn r2_predicted(&self) -> PyResult<f64> {Ok(self.r2_predicted)}

    #[getter]
    fn var(&self) -> PyResult<f64> {Ok(self.var_endog)}
}


#[pymodule]
fn regrs<'py>(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OLS>()?;
    Ok(())
}
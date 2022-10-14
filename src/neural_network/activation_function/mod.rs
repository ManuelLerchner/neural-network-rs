use ndarray::Array;

pub mod linear;
pub mod relu;
pub mod sigmoid;

pub struct ActivationFunction {
    pub f: fn(f64) -> f64,
    pub d: fn(f64) -> f64,
}

impl ActivationFunction {
    pub fn function<D>(&self, x: &Array<f64, D>) -> Array<f64, D>
    where
        D: ndarray::Dimension,
    {
        x.mapv(self.f)
    }

    pub fn derivative<D>(&self, x: &Array<f64, D>) -> Array<f64, D>
    where
        D: ndarray::Dimension,
    {
        x.mapv(self.d)
    }
}

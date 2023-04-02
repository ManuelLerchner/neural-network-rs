use ndarray::Array;
//Sized

pub trait ActivationFunction {
    fn f(&self, x: f64) -> f64;
    fn d(&self, x: f64) -> f64;
}

impl dyn ActivationFunction {
    pub fn f_array<D: ndarray::Dimension>(&self, x: &Array<f64, D>) -> Array<f64, D> {
        x.mapv(|x| self.f(x))
    }

    pub fn d_array<D: ndarray::Dimension>(&self, x: &Array<f64, D>) -> Array<f64, D> {
        x.mapv(|x| self.d(x))
    }
}

pub struct Relu;

impl ActivationFunction for Relu {
    fn f(&self, x: f64) -> f64 {
        x.max(0.0)
    }

    fn d(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn f(&self, x: f64) -> f64 {
        if x > 45.0 {
            1.0
        } else if x < -45.0 {
            0.0
        } else {
            1.0 / (1.0 + (-x).exp())
        }
    }

    fn d(&self, x: f64) -> f64 {
        let x = self.f(x);
        x * (1.0 - x)
    }
}

pub struct Linear;

impl ActivationFunction for Linear {
    fn f(&self, x: f64) -> f64 {
        x
    }

    fn d(&self, _: f64) -> f64 {
        1.0
    }
}

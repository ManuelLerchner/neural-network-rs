use ndarray::Array;

pub struct ActivationFunction {
    f: fn(f64) -> f64,
    d: fn(f64) -> f64,
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

pub static SIGMOID: ActivationFunction = ActivationFunction {
    f: |x| {
        if x > 45.0 {
            1.0
        } else if x < -45.0 {
            0.0
        } else {
            1.0 / (1.0 + (-x).exp())
        }
    },

    d: |x| {
        let x = (SIGMOID.f)(x);
        x * (1.0 - x)
    },
};

pub static RELU: ActivationFunction = ActivationFunction {
    f: (|x: f64| x.max(0.0)),
    d: (|x: f64| if x > 0.0 { 1.0 } else { 0.0 }),
};

pub static ID: ActivationFunction = ActivationFunction {
    f: (|x: f64| x),
    d: (|_x: f64| 1.0),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_simple() {
        assert_eq!(4.0, (RELU.f)(4.0));
        assert_eq!(0.0, (RELU.f)(-4.0));
    }

    #[test]
    fn test_relu_derivative() {
        assert_eq!(1.0, (RELU.d)(4.0));
        assert_eq!(0.0, (RELU.d)(-4.0));
    }

    #[test]
    fn test_sigmoid_simple() {
        assert_eq!(0.5, (SIGMOID.f)(0.0));
        assert_eq!(0.7310585786300049, (SIGMOID.f)(1.0));
        assert_eq!(0.2689414213699951, (SIGMOID.f)(-1.0));
    }

    #[test]
    fn test_sigmoid_derivative() {
        assert_eq!(0.25, (SIGMOID.d)(0.0));
        assert_eq!(0.19661193324148185, (SIGMOID.d)(1.0));
        assert_eq!(0.19661193324148185, (SIGMOID.d)(-1.0));
    }

    #[test]
    fn test_sigmoid_overflow() {
        assert_eq!(0.0, (SIGMOID.f)(-100.0));
        assert_eq!(1.0, (SIGMOID.f)(100.0));
    }

    #[test]
    fn test_sigmoid_derivative_overflow() {
        assert_eq!(0.0, (SIGMOID.d)(-100.0));
        assert_eq!(0.0, (SIGMOID.d)(100.0));
    }

    #[test]
    fn test_apply_on_matrix() {
        let x = ndarray::arr2(&[[1., -0.1], [-3.5, 4.]]);

        let x_after_sigmoid = ndarray::arr2(&[
            [0.7310585786300049, 0.47502081252106],
            [0.02931223075135632, 0.9820137900379085],
        ]);
        let x_after_relu = ndarray::arr2(&[[1., 0.], [0., 4.]]);

        assert_eq!(x_after_sigmoid, SIGMOID.function(&x));
        assert_eq!(x_after_relu, RELU.function(&x));
    }
}

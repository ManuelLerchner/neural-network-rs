use ndarray::Array;

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

#[cfg(test)]
mod tests {
    use crate::activation_function::{RELU, SIGMOID};

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

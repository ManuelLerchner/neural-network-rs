use crate::activation_function;

use activation_function::activation_function::ActivationFunction;
use ndarray::Array2;
use ndarray_rand::{rand_distr::Normal, RandomExt};

pub struct Layer<'a> {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: &'a ActivationFunction,
}

impl Layer<'_> {
    pub fn new(input_size: usize, output_size: usize, activation: &ActivationFunction) -> Layer {
        let weights = Array2::random((input_size, output_size), Normal::new(0.0, 1.0).unwrap())
            / (input_size as f64).sqrt();
        let biases = Array2::random((1, output_size), Normal::new(0.0, 0.1).unwrap());

        Layer {
            weights,
            biases,
            activation,
        }
    }

    // Predicts the output of the layer given an input
    pub fn predict(&self, input: &Array2<f64>) -> Array2<f64> {
        self.activation.function(&self.forward(input))
    }

    // Calculates the weighted sum of the input
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.dot(&self.weights) + &self.biases
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use crate::activation_function::activation_function::SIGMOID;

    use super::*;

    #[test]
    fn test_forward_batch() {
        let mut layer = Layer::new(2, 1, &SIGMOID);
        layer.weights = arr2(&[[1.], [-2.]]);
        layer.biases = arr2(&[[1.]]);

        let input = arr2(&[[1., 2.], [3., 4.]]);
        let expected = arr2(&[[-2.], [-4.]]);

        let output = layer.forward(&input);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward() {
        let mut layer = Layer::new(2, 1, &SIGMOID);
        layer.weights = arr2(&[[1.], [-2.]]);
        layer.biases = arr2(&[[1.]]);

        let input = arr2(&[[1., 2.]]);
        let expected = arr2(&[[-2.]]);

        let output = layer.forward(&input);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_predict_batch() {
        let mut layer = Layer::new(2, 1, &SIGMOID);
        layer.weights = arr2(&[[1.], [-2.]]);
        layer.biases = arr2(&[[1.]]);

        let input = arr2(&[[1., 2.], [3., 4.]]);
        let expected = arr2(&[[0.11920292202211755], [0.01798620996209156]]);

        let output = layer.predict(&input);

        assert_eq!(output, expected);
    }
}

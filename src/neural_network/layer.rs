use ndarray::Array2;
use ndarray_rand::{rand_distr::Normal, RandomExt};

use super::activation_function::ActivationFunction;

pub trait Layer {
    fn new(input_size: usize, activation: &'static dyn ActivationFunction) -> Self
    where
        Self: Sized;

    fn initialize(&mut self, input_size: usize, output_size: usize);

    fn predict(&self, input: &Array2<f64>) -> Array2<f64>;
    fn forward(&self, input: &Array2<f64>) -> Array2<f64>;

    fn get_size(&self) -> usize;
    fn get_activation(&self) -> &'static dyn ActivationFunction;
    fn get_weights(&self) -> &Array2<f64>;
    fn set_weights(&mut self, weights: Array2<f64>);
    fn get_bias(&self) -> &Array2<f64>;
    fn set_bias(&mut self, biases: Array2<f64>);
}

pub struct DenseLayer {
    pub input_size: usize,
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: &'static dyn ActivationFunction,
}

impl Layer for DenseLayer {
    fn new(input_size: usize, activation: &'static dyn ActivationFunction) -> DenseLayer {
        DenseLayer {
            input_size,
            activation,
            weights: Array2::zeros((0, 0)),
            biases: Array2::zeros((0, 0)),
        }
    }

    fn initialize(&mut self, input_size: usize, output_size: usize) {
        self.weights = Array2::random((input_size, output_size), Normal::new(0.0, 1.0).unwrap())
            / (input_size as f64).sqrt();
        self.biases = Array2::random((1, output_size), Normal::new(0.0, 0.1).unwrap());
    }

    fn get_size(&self) -> usize {
        self.input_size
    }

    fn get_activation(&self) -> &'static dyn ActivationFunction {
        self.activation
    }

    fn get_weights(&self) -> &Array2<f64> {
        &self.weights
    }

    fn get_bias(&self) -> &Array2<f64> {
        &self.biases
    }

    fn set_weights(&mut self, weights: Array2<f64>) {
        self.weights = weights;
    }

    fn set_bias(&mut self, biases: Array2<f64>) {
        self.biases = biases;
    }

    // Predicts the output of the layer given an input
    fn predict(&self, input: &Array2<f64>) -> Array2<f64> {
        let a = &self.forward(input);
        self.activation.f_array(a)
    }

    // Calculates the weighted sum of the input
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.dot(&self.weights) + &self.biases
    }
}

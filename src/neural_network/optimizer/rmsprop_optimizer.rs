use itertools::izip;
use ndarray::Array2;

use crate::neural_network::{layer::Layer, Summary};

use super::Optimizer;

#[allow(non_camel_case_types)]
pub struct RMS_PROP {
    learning_rate: f64,
    decay: f64,
    iteration: usize,
    current_learning_rate: f64,
    epsilon: f64,
    rho: f64,
    weights_cache: Vec<Array2<f64>>,
    biases_cache: Vec<Array2<f64>>,
}

impl RMS_PROP {
    pub fn new(learning_rate: f64, decay: f64, epsilon: f64, rho: f64) -> RMS_PROP {
        RMS_PROP {
            learning_rate,
            decay,
            iteration: 0,
            current_learning_rate: learning_rate,
            weights_cache: Vec::new(),
            biases_cache: Vec::new(),
            epsilon,
            rho,
        }
    }

    pub fn default() -> RMS_PROP {
        RMS_PROP::new(0.001, 1e-4, 1e-7, 0.9)
    }
}

impl Optimizer for RMS_PROP {
    fn update_params(
        &mut self,
        layers: &mut Vec<Layer>,

        nabla_bs: &Vec<Array2<f64>>,
        nabla_ws: &Vec<Array2<f64>>,
    ) {
        for (i, (layer, nabla_b, nabla_w)) in izip!(layers, nabla_bs, nabla_ws).enumerate() {
            //update cache
            self.weights_cache[i] =
                self.rho * &self.weights_cache[i] + (1.0 - self.rho) * (nabla_w * nabla_w);
            self.biases_cache[i] =
                self.rho * &self.biases_cache[i] + (1.0 - self.rho) * (nabla_b * nabla_b);

            //calculate updates
            let weights_update = -self.current_learning_rate * nabla_w
                / (self.weights_cache[i].mapv(|x| x.sqrt()) + self.epsilon);
            let biases_update = -self.current_learning_rate * nabla_b
                / (self.biases_cache[i].mapv(|x| x.sqrt()) + self.epsilon);

            //Update weights and biases
            layer.weights = &layer.weights + weights_update;
            layer.biases = &layer.biases + biases_update;
        }
    }

    fn initialize(&mut self, layers: &Vec<Layer>) {
        for layer in layers {
            self.weights_cache.push(Array2::zeros(layer.weights.dim()));
            self.biases_cache.push(Array2::zeros(layer.biases.dim()));
        }
    }

    fn pre_update(&mut self) {
        if self.decay > 0.0 {
            self.current_learning_rate =
                self.learning_rate * (1.0 / (1.0 + self.decay * self.iteration as f64));
        }
    }

    fn post_update(&mut self) {
        self.iteration += 1;
    }
}

impl Summary for RMS_PROP {
    fn summerize(&self) -> String {
        "RMS_PROP".to_string()
    }
}

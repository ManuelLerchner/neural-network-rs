use itertools::izip;
use ndarray::Array2;

use crate::neural_network::{layer::Layer, Summary};

use super::Optimizer;

#[allow(non_camel_case_types)]
pub struct ADAM {
    learning_rate: f64,
    decay: f64,
    iteration: usize,
    current_learning_rate: f64,
    epsilon: f64,
    beta_1: f64,
    beta_2: f64,
    weights_cache: Vec<Array2<f64>>,
    biases_cache: Vec<Array2<f64>>,
    weights_momentum: Vec<Array2<f64>>,
    biases_momentum: Vec<Array2<f64>>,
}

impl ADAM {
    pub fn new(learning_rate: f64, decay: f64, epsilon: f64, beta_1: f64, beta_2: f64) -> ADAM {
        ADAM {
            learning_rate,
            decay,
            iteration: 0,
            current_learning_rate: learning_rate,
            weights_cache: Vec::new(),
            biases_cache: Vec::new(),
            weights_momentum: Vec::new(),
            biases_momentum: Vec::new(),
            epsilon,
            beta_1,
            beta_2,
        }
    }

    pub fn default() -> ADAM {
        ADAM::new(0.002, 1e-5, 1e-7, 0.9, 0.999)
    }
}

impl Optimizer for ADAM {
    fn update_params(
        &mut self,
        layers: &mut Vec<Layer>,

        nabla_bs: &Vec<Array2<f64>>,
        nabla_ws: &Vec<Array2<f64>>,
    ) {
        for (i, (layer, nabla_b, nabla_w)) in izip!(layers, nabla_bs, nabla_ws).enumerate() {
            //update momentum
            self.weights_momentum[i] =
                self.beta_1 * &self.weights_momentum[i] + (1.0 - self.beta_1) * nabla_w;

            self.biases_momentum[i] =
                self.beta_1 * &self.biases_momentum[i] + (1.0 - self.beta_1) * nabla_b;

            //update cache
            self.weights_cache[i] =
                self.beta_2 * &self.weights_cache[i] + (1.0 - self.beta_2) * (nabla_w * nabla_w);

            self.biases_cache[i] =
                self.beta_2 * &self.biases_cache[i] + (1.0 - self.beta_2) * (nabla_b * nabla_b);

            //corrections
            let weights_momentum_corrected =
                &self.weights_momentum[i] / (1.0 - self.beta_1.powi(i as i32 + 1));
            let biases_momentum_corrected =
                &self.biases_momentum[i] / (1.0 - self.beta_1.powi(i as i32 + 1));
            let weights_cache_corrected =
                &self.weights_cache[i] / (1.0 - self.beta_2.powi(i as i32 + 1));
            let biases_cache_corrected =
                &self.biases_cache[i] / (1.0 - self.beta_2.powi(i as i32 + 1));

            let weights_update = self.current_learning_rate * weights_momentum_corrected
                / (weights_cache_corrected.mapv(f64::sqrt) + self.epsilon);
            let biases_update = self.current_learning_rate * biases_momentum_corrected
                / (biases_cache_corrected.mapv(f64::sqrt) + self.epsilon);

            //updates
            layer.weights = &layer.weights - &weights_update;
            layer.biases = &layer.biases - &biases_update;
        }
    }

    fn initialize(&mut self, layers: &Vec<Layer>) {
        for layer in layers {
            self.weights_cache.push(Array2::zeros(layer.weights.dim()));
            self.biases_cache.push(Array2::zeros(layer.biases.dim()));
            self.weights_momentum
                .push(Array2::zeros(layer.weights.dim()));
            self.biases_momentum.push(Array2::zeros(layer.biases.dim()));
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

impl Summary for ADAM {
    fn summerize(&self) -> String {
        "ADAM".to_string()
    }
}

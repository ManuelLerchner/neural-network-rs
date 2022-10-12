use itertools::izip;
use ndarray::Array2;

use crate::neural_network::{layer::Layer, Summary};

use super::Optimizer;

pub struct SGD {
    momentum: f64,
    learning_rate: f64,
    decay: f64,
    iteration: usize,
    current_learning_rate: f64,
    weights_momentum: Vec<Array2<f64>>,
    biases_momentum: Vec<Array2<f64>>,
}

impl SGD {
    pub fn new(learning_rate: f64, momentum: f64, decay: f64) -> SGD {
        SGD {
            learning_rate,
            momentum,
            decay,
            iteration: 0,
            current_learning_rate: learning_rate,
            weights_momentum: Vec::new(),
            biases_momentum: Vec::new(),
        }
    }
}

impl Optimizer for SGD {
    fn update_params(
        &mut self,
        layers: &mut Vec<Layer>,
        batch_size: usize,
        nabla_bs: &Vec<Array2<f64>>,
        nabla_ws: &Vec<Array2<f64>>,
    ) {
        for (i, (layer, nabla_b, nabla_w)) in izip!(layers, nabla_bs, nabla_ws).enumerate() {
            //Calculate standart update_params
            let eta = self.current_learning_rate / batch_size as f64;

            let mut weights_update = -eta * nabla_w;
            let mut biases_update = -eta * nabla_b;

            //Add momentum
            if self.momentum > 0.0 {
                let weights_momentum = &self.weights_momentum[i];
                let biases_momentum = &self.biases_momentum[i];

                weights_update = weights_update + self.momentum * weights_momentum;
                biases_update = biases_update + self.momentum * biases_momentum;

                self.weights_momentum[i] = weights_update.clone();
                self.biases_momentum[i] = biases_update.clone();
            }

            //Update weights and biases
            layer.weights = &layer.weights + weights_update;
            layer.biases = &layer.biases + biases_update;
        }
    }

    fn initialize(&mut self, layers: &Vec<Layer>) {
        for layer in layers {
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

impl Summary for SGD {
    fn summerize(&self) -> String {
        "SGD".to_string()
    }
}

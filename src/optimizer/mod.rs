pub mod sgd_optimzer;

use ndarray::Array2;

use crate::neural_network::{layer::Layer, Summary};

pub trait Optimizer: Summary {
    fn update_params(
        &mut self,
        layers: &mut Vec<Layer>,
        batch_size: usize,
        nabla_bs: &Vec<Array2<f64>>,
        nabla_ws: &Vec<Array2<f64>>,
    );

    fn initialize(&mut self, layers: &Vec<Layer>);

    fn pre_update(&mut self);

    fn post_update(&mut self);
}

pub mod dataset;
pub mod neural_network;
pub mod plotter;

use neural_network::layer::{DenseLayer, Layer};

#[allow(unused_imports)]
use crate::{
    dataset::example_datasets::{CIRCLE, RGB_DONUT, RGB_TEST, XOR},
    neural_network::{
        activation_function::{ActivationFunction, Linear, Relu, Sigmoid},
        cost_function::QuadraticCost,
        optimizer::{adam_optimizer::ADAM, rmsprop_optimizer::RMS_PROP, sgd_optimzer::SGD},
        Network, Summary,
    },
    plotter::{graph_plotter::plot_graph, png_plotter::plot_png},
};

#[allow(dead_code)]
fn main() {
    //Define Network Shape
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(DenseLayer::new(2, &Relu)),
        Box::new(DenseLayer::new(32, &Relu)),
        Box::new(DenseLayer::new(32, &Sigmoid)),
        Box::new(DenseLayer::new(32, &Relu)),
        Box::new(DenseLayer::new(3, &Linear)),
    ];

    //Define Optimizer
    let mut optimizer = ADAM::default();

    //Create Network
    let mut network = Network::new(layers, &mut optimizer, &QuadraticCost);

    //Define Dataset
    let dataset = &RGB_DONUT;

    //Train
    let cost_history = network.train_and_log(dataset, 128, 512, 10000);

    //Prepare Plot-data
    let (dim, unit_square_prediction) = network.predict_unit_square(512);
    let name = String::from(dataset.name) + "_" + &network.summerize();

    //Plot
    plot_png(&name, dim, &unit_square_prediction, png::ColorType::Rgb).unwrap();
    plot_graph(&name, &cost_history).unwrap();
}

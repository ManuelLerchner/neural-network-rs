pub mod dataset;
pub mod neural_network;
pub mod plotter;

#[allow(unused_imports)]
use crate::{
    dataset::example_datasets::{CIRCLE, RGB_DONUT, RGB_TEST, XOR},
    neural_network::{
        activation_function::{linear::LINEAR, relu::RELU, sigmoid::SIGMOID},
        cost_function::quadratic_cost::QUADRATIC_COST,
        optimizer::{adam_optimizer::ADAM, rmsprop_optimizer::RMS_PROP, sgd_optimzer::SGD},
        Network, Summary,
    },
    plotter::{graph_plotter::plot_graph, png_plotter::plot_png},
};

fn main() {
    //Define Network Shape
    let network_shape = [
        (&RELU, 2),
        (&RELU, 32),
        (&RELU, 32),
        (&RELU, 32),
        (&RELU, 3),
    ];

    //Define Optimizer
    let mut optimizer = ADAM::default();

    //Create Network
    let mut network = Network::new(&network_shape, &mut optimizer, &QUADRATIC_COST);

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

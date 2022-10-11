pub mod activation_function;
pub mod cost_function;
pub mod data;
pub mod neural_network;
pub mod plotter;

use data::data::RGB_DONUT;
use neural_network::network::Summary;

#[allow(unused_imports)]
use crate::{
    activation_function::activation_function::{ID, RELU, SIGMOID},
    cost_function::cost_function::QUADRATIC_COST,
    data::data::{Dataset, CIRCLE, RGB_TEST, XOR},
    neural_network::network::Network,
    plotter::{graph_plotter::plot_graph, png_plotter::plot_png},
};

fn main() {
    let mut network = Network::new(&[2, 64, 128, 64, 3], 0.15, &RELU, &QUADRATIC_COST);

    //Train
    let dataset = &RGB_DONUT;
    let cost_history = network.train_and_log(dataset, 16, 64, 6000);

    //Plot
    let (dim, unit_square_prediction) = network.predict_unit_square(1024);

    let name = String::from(dataset.name) + "_" + &network.summerize();
    plot_png(&name, dim, &unit_square_prediction, png::ColorType::Rgb).unwrap();
    plot_graph(format!("{}_history", name), &cost_history).unwrap();
}

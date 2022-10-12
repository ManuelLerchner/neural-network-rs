pub mod activation_function;
pub mod cost_function;
pub mod data;
pub mod neural_network;
pub mod optimizer;
pub mod plotter;

#[allow(unused_imports)]
use crate::{
    activation_function::{ID, RELU, SIGMOID},
    cost_function::QUADRATIC_COST,
    data::RGB_DONUT,
    data::{CIRCLE, XOR},
    neural_network::{Network, Summary},
    optimizer::sgd_optimzer::SGD,
    plotter::{graph_plotter::plot_graph, png_plotter::plot_png},
};

fn main() {
    let mut optimizer = SGD::new(0.1, 0.3, 0.0001);
    let mut network = Network::new(
        &[2, 64, 64, 64, 64, 64, 3],
        &mut optimizer,
        &RELU,
        &QUADRATIC_COST,
    );

    //Train
    let dataset = &RGB_DONUT;
    let cost_history = network.train_and_log(dataset, 256, 512, 10000);

    //Plot
    let (dim, unit_square_prediction) = network.predict_unit_square(128);

    let name = String::from(dataset.name) + "_" + &network.summerize();
    plot_png(&name, dim, &unit_square_prediction, png::ColorType::Rgb).unwrap();
    plot_graph(format!("{}_history", name), &cost_history).unwrap();
}

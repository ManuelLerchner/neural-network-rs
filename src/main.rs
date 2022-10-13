pub mod activation_function;
pub mod cost_function;
pub mod data;
pub mod neural_network;
pub mod optimizer;
pub mod plotter;

#[allow(unused_imports)]
use crate::{
    activation_function::{LINEAR, RELU, SIGMOID},
    cost_function::QUADRATIC_COST,
    data::{CIRCLE, RGB_DONUT, XOR},
    neural_network::{Network, Summary},
    optimizer::sgd_optimzer::SGD,
    plotter::{graph_plotter::plot_graph, png_plotter::plot_png},
};

fn main() {
    let mut optimizer = SGD::new(0.1, 0.5, 0.0005);

    let network_shape = [
        (&RELU, 2),
        (&RELU, 64),
        (&RELU, 64),
        (&RELU, 64),
        (&LINEAR, 3),
    ];

    let mut network = Network::new(&network_shape, &mut optimizer, &QUADRATIC_COST);

    //Train
    let dataset = &RGB_DONUT;
    let cost_history = network.train_and_log(dataset, 128, 512, 10000);

    //Plot
    let (dim, unit_square_prediction) = network.predict_unit_square(512);

    let name = String::from(dataset.name) + "_" + &network.summerize();


    println!("name: {}", name);

    plot_png(&name, dim, &unit_square_prediction, png::ColorType::Rgb).unwrap();
    plot_graph(format!("{}_history", name), &cost_history).unwrap();
}

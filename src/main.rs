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
    let network_shape = [(&RELU, 2), (&RELU, 32), (&RELU, 3)];

    let mut optimizer = SGD::default();
    let mut network = Network::new(&network_shape, &mut optimizer, &QUADRATIC_COST);

    //Train
    let dataset = &RGB_TEST;
    let cost_history = network.train_and_log(dataset, 128, 512, 10000);

    //Plot
    let (dim, unit_square_prediction) = network.predict_unit_square(512);

    let name = String::from(dataset.name) + "_" + &network.summerize();
    plot_png(
        &name,
        dim,
        &unit_square_prediction,
        png::ColorType::Rgb,
    )
    .unwrap();
    plot_graph(&name, &cost_history).unwrap();
}

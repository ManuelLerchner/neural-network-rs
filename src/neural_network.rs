pub mod activation_function;
pub mod cost_function;
pub mod layer;
pub mod optimizer;

use crate::dataset::Dataset;

use ndarray::Array2;

use self::{cost_function::CostFunction, layer::Layer, optimizer::Optimizer};

pub struct Network<'a> {
    input_size: usize,
    output_size: usize,
    layers: Vec<Box<dyn Layer>>,
    optimizer: &'a mut dyn Optimizer,
    cost_function: &'static dyn CostFunction,
}

#[allow(non_snake_case)]
impl Network<'_> {
    pub fn new<'a>(
        mut layers: Vec<Box<dyn Layer>>,
        optimizer: &'a mut dyn Optimizer,
        cost_function: &'static dyn CostFunction,
    ) -> Network<'a> {
        // Initialize the layers
        let network_shape = layers.iter().map(|l| l.get_size()).collect::<Vec<_>>();

        //remove last layer
        layers.pop();

        //initialize the layers
        for (i, boxedLeyer) in layers.iter_mut().enumerate() {
            let layer = boxedLeyer.as_mut();
            layer.initialize(network_shape[i], network_shape[i + 1]);
        }

        optimizer.initialize(&layers);

        Network {
            input_size: network_shape[0],
            output_size: network_shape[network_shape.len() - 1],
            layers,
            optimizer,
            cost_function,
        }
    }

    // Predicts the output of the network given an input
    pub fn predict(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.predict(&output);
        }
        output
    }

    // Calculates the needed adjustments to the weights and biases for a given input and expected output
    pub fn backprop(
        &self,
        X: &Array2<f64>,
        y: &Array2<f64>,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_bs = Vec::new();
        let mut nabla_ws = Vec::new();

        // Forward pass
        let mut activation = X.clone();
        let mut activations = vec![activation.clone()];
        let mut zs = Vec::new();
        for layer in &self.layers {
            let z = layer.forward(&activation);
            zs.push(z.clone());
            activation = layer.get_activation().f_array(&z);
            activations.push(activation.clone());
        }

        // Calculate the cost
        let nabla_c = self.cost_function.cost_derivative(&activation, &y);

        // Calculate sensitivity
        let sig_prime = self.layers[self.layers.len() - 1]
            .get_activation()
            .d_array(&zs[zs.len() - 1]);

        // Calculate delta for last layer
        let mut delta = nabla_c * sig_prime;

        // Calculate nabla_b and nabla_w for last layer
        nabla_bs.push(delta.clone());
        nabla_ws.push((&activations[activations.len() - 2]).t().dot(&delta));

        // Loop backwards through the layers, calculating delta, nabla_b and nabla_w
        for i in 2..self.layers.len() + 1 {
            let sig_prime = self.layers[self.layers.len() - i]
                .get_activation()
                .d_array(&zs[zs.len() - i]);

            let nabla_c = &delta.dot(&self.layers[self.layers.len() - i + 1].get_weights().t());

            delta = nabla_c * sig_prime;

            nabla_bs.push(delta.clone());
            nabla_ws.push((&activations[activations.len() - i - 1].t()).dot(&delta));
        }

        // restore correct ordering
        nabla_bs.reverse();
        nabla_ws.reverse();

        //Adjust for batch size
        let batch_size = X.nrows() as f64;
        for (nabla_b, nabla_w) in nabla_bs.iter_mut().zip(nabla_ws.iter_mut()) {
            *nabla_b = nabla_b
                .sum_axis(ndarray::Axis(0))
                .into_shape((1, nabla_b.ncols()))
                .unwrap();

            *nabla_b /= batch_size;
            *nabla_w /= batch_size;
        }

        (nabla_bs, nabla_ws)
    }

    // Trains the network using a minibatch
    pub fn train_minibatch(&mut self, (X, y): &(Array2<f64>, Array2<f64>)) {
        //assert iput shape is the same as the data
        assert_eq!(
            X.ncols(),
            self.input_size,
            "Input shape does not match data"
        );
        assert_eq!(
            y.ncols(),
            self.output_size,
            "Output shape does not match data"
        );

        let (nabla_bs, nabla_ws) = self.backprop(X, y);

        self.optimizer.pre_update();

        self.optimizer
            .update_params(&mut self.layers, &nabla_bs, &nabla_ws);

        self.optimizer.post_update();
    }

    // Trains the network using a dataset, records the cost for each epoch
    pub fn train_and_log(
        &mut self,
        data: &Dataset,
        batch_size: usize,
        verification_samples: usize,
        epochs: i32,
    ) -> Vec<(i32, f64)> {
        let mut cost_history = Vec::new();

        for epoch in 0..epochs {
            self.train_minibatch(&data.get_batch(batch_size));

            if epoch % (epochs / 100 + 1) == 0 {
                let cost = self.eval(data, verification_samples);
                cost_history.push((epoch, cost));

                println!("Epoch: {}, Cost: {:.8}", epoch, cost);
            }
        }

        cost_history
    }

    // Evaluates the network on a given dataset
    pub fn eval(&self, data: &Dataset, sample_size: usize) -> f64 {
        let (x, y) = data.get_batch(sample_size);

        let prediction = self.predict(&x);
        let cost = self.cost_function.cost(&prediction, &y);
        cost
    }

    // evaluates the prediction-results for the unit-square, returns a list
    // containing the result for each point in a row by row fashion
    pub fn predict_unit_square(&self, resolution: usize) -> ((usize, usize), Vec<Vec<f64>>) {
        let unit_square = Dataset::get_2d_unit_square(resolution);
        let pred = self.predict(&unit_square);

        let res = pred
            .lanes(ndarray::Axis(1))
            .into_iter()
            .map(|x| x.to_vec())
            .collect();

        ((resolution, resolution), res)
    }
}

pub trait Summary {
    fn summerize(&self) -> String;
}

impl Summary for Network<'_> {
    fn summerize(&self) -> String {
        let shape = self.layers.iter().map(|x| x.get_size()).collect::<Vec<_>>();

        format!("{}_{:?}", self.optimizer.summerize(), shape).replace(" ", "")
    }
}

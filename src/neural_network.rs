pub mod layer;

use crate::{
    activation_function::activation_function::ActivationFunction,
    cost_function::cost_function::CostFunction, data::dataset::Dataset, optimizer::Optimizer,
};

use ndarray::Array2;

use self::layer::Layer;

pub struct Network<'a> {
    pub layers: Vec<Layer<'a>>,
    optimizer: &'a mut dyn Optimizer,
    pub shape: &'a [(&'a ActivationFunction, usize)],
    pub cost_function: &'a CostFunction,
}

#[allow(non_snake_case)]
impl Network<'_> {
    pub fn new<'a>(
        shape: &'a [(&ActivationFunction, usize)],
        optimizer: &'a mut dyn Optimizer,
        cost_function: &'a CostFunction,
    ) -> Network<'a> {
        let mut layers = Vec::new();
        for i in 0..shape.len() - 1 {
            let (activation_function, input_size) = shape[i];
            let (_, output_size) = shape[i + 1];
            layers.push(Layer::new(input_size, output_size, activation_function));
        }

        optimizer.initialize(&layers);

        Network {
            layers,
            optimizer,
            shape,
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
            activation = layer.activation.function(&z);
            activations.push(activation.clone());
        }

        // Calculate the cost
        let nabla_c = self.cost_function.cost_derivative(&activation, &y);

        // Calculate sensitivity
        let sig_prime = self.layers[self.layers.len() - 1]
            .activation
            .derivative(&zs[zs.len() - 1]);

        // Calculate delta for last layer
        let mut delta = nabla_c * sig_prime;

        // Calculate nabla_b and nabla_w for last layer
        nabla_bs.push(delta.clone());
        nabla_ws.push((&activations[activations.len() - 2]).t().dot(&delta));

        // Loop backwards through the layers, calculating delta, nabla_b and nabla_w
        for i in 2..self.shape.len() {
            let sig_prime = self.layers[self.layers.len() - i]
                .activation
                .derivative(&zs[zs.len() - i]);

            let nabla_c = &delta.dot(&self.layers[self.layers.len() - i + 1].weights.t());

            delta = nabla_c * sig_prime;

            nabla_bs.push(delta.clone());
            nabla_ws.push((&activations[activations.len() - i - 1].t()).dot(&delta));
        }

        // restore correct ordering
        nabla_bs.reverse();
        nabla_ws.reverse();

        // collect all bias adjustments for the given batch into a single vector
        for nabla_b in &mut nabla_bs {
            *nabla_b = nabla_b
                .sum_axis(ndarray::Axis(0))
                .into_shape((1, nabla_b.ncols()))
                .unwrap();
        }

        (nabla_bs, nabla_ws)
    }

    // Trains the network using a minibatch
    pub fn train_minibatch(&mut self, (X, y): &(Array2<f64>, Array2<f64>)) {
        let (nabla_bs, nabla_ws) = self.backprop(X, y);

        let batch_size = X.nrows();

        self.optimizer.pre_update();

        self.optimizer
            .update_params(&mut self.layers, batch_size, &nabla_bs, &nabla_ws);

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
        let shape = self.shape.iter().map(|x| x.1).collect::<Vec<_>>();

        format!("{}_{:?}", self.optimizer.summerize(), shape).replace(" ", "")
    }
}

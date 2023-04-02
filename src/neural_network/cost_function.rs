use ndarray::{Array1, Array2};

pub trait CostFunction {
    fn f(&self, a: &Array1<f64>, expected: &Array1<f64>) -> f64;
    fn d(&self, a: &Array1<f64>, expected: &Array1<f64>) -> Array1<f64>;
}

impl dyn CostFunction {
    pub fn cost(&self, a: &Array2<f64>, expected: &Array2<f64>) -> f64 {
        let mut cost = 0.0;
        for (a, expected) in a.outer_iter().zip(expected.outer_iter()) {
            cost += self.f(&a.to_owned(), &expected.to_owned());
        }
        cost / a.nrows() as f64
    }

    pub fn cost_derivative(&self, a: &Array2<f64>, expected: &Array2<f64>) -> Array2<f64> {
        let mut cost_derivative = Array2::zeros(a.raw_dim());

        for i in 0..a.ncols() {
            cost_derivative
                .column_mut(i)
                .assign(&(self.d(&a.column(i).to_owned(), &expected.column(i).to_owned())));
        }

        cost_derivative
    }
}

pub struct QuadraticCost;

impl CostFunction for QuadraticCost {
    fn f(&self, a: &Array1<f64>, expected: &Array1<f64>) -> f64 {
        0.5 * (a - expected).mapv(|x| x.powi(2)).sum()
    }

    fn d(&self, a: &Array1<f64>, expected: &Array1<f64>) -> Array1<f64> {
        a - expected
    }
}

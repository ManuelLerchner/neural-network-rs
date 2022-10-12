use ndarray::{Array1, Array2};

pub struct CostFunction {
    pub f: fn(&Array1<f64>, &Array1<f64>) -> f64,
    pub d: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
}

impl CostFunction {
    pub fn cost(&self, a: &Array2<f64>, expected: &Array2<f64>) -> f64 {
        let mut cost = 0.0;
        for (a, expected) in a.outer_iter().zip(expected.outer_iter()) {
            cost += (self.f)(&a.to_owned(), &expected.to_owned());
        }
        cost / a.nrows() as f64
    }

    pub fn cost_derivative(&self, a: &Array2<f64>, expected: &Array2<f64>) -> Array2<f64> {
        let mut cost_derivative = Array2::zeros(a.raw_dim());

        for i in 0..a.ncols() {
            cost_derivative
                .column_mut(i)
                .assign(&((self.d)(&a.column(i).to_owned(), &expected.column(i).to_owned())));
        }

        cost_derivative
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use crate::cost_function::QUADRATIC_COST;

    #[test]
    fn test_quadratic_cost() {
        let a = arr1(&[0.25, 1.0, 0.4]);
        let expected = arr1(&[0.2, 1.5, 0.5]);
        assert_eq!(0.13125, (QUADRATIC_COST.f)(&a, &expected));
    }

    #[test]
    fn test_quadratic_cost_derivative() {
        let a = arr1(&[0.25, 1.0, 2.]);
        let expected = arr1(&[0.5, 1.5, 0.5]);
        assert_eq!(arr1(&[-0.25, -0.5, 1.5]), (QUADRATIC_COST.d)(&a, &expected));
    }

    #[test]
    fn test_quadratic_cost_nabla() {
        let a = arr2(&[[0.25, 1.0, 2.], [0.25, 1.0, 2.], [0.25, 1.0, 2.]]);
        let expected = arr2(&[[0.5, 1.5, 0.5], [0.5, 1.5, 0.5], [0.5, 1.5, 0.5]]);

        let diff = &a - &expected;

        let nabla = QUADRATIC_COST.cost_derivative(&a, &expected);

        assert_eq!(diff, nabla);
    }
}

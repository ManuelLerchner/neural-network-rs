use super::CostFunction;

pub static QUADRATIC_COST: CostFunction = CostFunction {
    f: (|a, expected| (a - expected).iter().fold(0.0, |acc, x| acc + x.powi(2)) / 2.0),
    d: (|a, expected| a - expected),
};

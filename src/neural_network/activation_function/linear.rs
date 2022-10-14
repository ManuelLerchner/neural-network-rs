use super::ActivationFunction;

// The linear activation function: f(x) = x
pub static LINEAR: ActivationFunction = ActivationFunction {
    f: (|x: f64| x),
    d: (|_: f64| 1.0),
};

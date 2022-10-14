use super::ActivationFunction;

// The relu activation function: f(x) = max(0, x)
pub static RELU: ActivationFunction = ActivationFunction {
    f: (|x: f64| x.max(0.0)),
    d: (|x: f64| if x > 0.0 { 1.0 } else { 0.0 }),
};

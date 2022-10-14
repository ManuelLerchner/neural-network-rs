use super::ActivationFunction;

// The sigmoid activation function: f(x) = 1 / (1 + e^(-x))
pub static SIGMOID: ActivationFunction = ActivationFunction {
    f: |x| {
        if x > 45.0 {
            1.0
        } else if x < -45.0 {
            0.0
        } else {
            1.0 / (1.0 + (-x).exp())
        }
    },

    d: |x| {
        let x = (SIGMOID.f)(x);
        x * (1.0 - x)
    },
};

use self::activation_function::ActivationFunction;

pub mod activation_function;

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

pub static RELU: ActivationFunction = ActivationFunction {
    f: (|x: f64| x.max(0.0)),
    d: (|x: f64| if x > 0.0 { 1.0 } else { 0.0 }),
};

pub static ID: ActivationFunction = ActivationFunction {
    f: (|x: f64| x),
    d: (|_x: f64| 1.0),
};

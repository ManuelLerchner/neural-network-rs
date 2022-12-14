use ndarray::array;

use super::{Dataset, DatasetType};

// The XOR dataset: [0, 0] -> 0, [0, 1] -> 1, [1, 0] -> 1, [1, 1] -> 0
pub static XOR: Dataset = Dataset {
    name: "XOR",
    dataset_type: DatasetType::Static(|| {
        let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = array![[0.0], [1.0], [1.0], [0.0]];
        (x, y)
    }),
};

// The Circle dataset: [x, y] -> 1 if (x-0.5)^2 + (y-0.5)^2 < 0.25, 0 otherwise
pub static CIRCLE: Dataset = Dataset {
    name: "Circle",
    dataset_type: DatasetType::Dynamic(
        |x| {
            let dist_from_center = ((x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2)).sqrt();
            let y = if dist_from_center < 0.25 { 1.0 } else { 0.0 };
            array![y]
        },
        (2, 1),
    ),
};

// The RGB_Test dataset: [x, y] -> [r=x, g=y, b=1-x]
pub static RGB_TEST: Dataset = Dataset {
    name: "RGB_TEST",
    dataset_type: DatasetType::Dynamic(
        |x| {
            let r = x[0];
            let g = x[1];
            let b = 1.0 - r;
            array![r, g, b]
        },
        (2, 3),
    ),
};

// The RGB_DONUT dataset: represents a colorful donut-shape in RGB unit-square
pub static RGB_DONUT: Dataset = Dataset {
    name: "RGB_DONUT",
    dataset_type: DatasetType::Dynamic(
        |x| {
            let dist_from_center = ((x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2)).sqrt();

            let r = x[0];
            let g = x[1];
            let b = 1.0 - r;

            if dist_from_center > 0.25 && dist_from_center < 0.45 {
                array![r, g, b]
            } else {
                array![0.0, 0.0, 0.0]
            }
        },
        (2, 3),
    ),
};

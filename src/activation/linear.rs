use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use typetag;

use super::ActivationFunctionClone;

/// Linear Activation Function
///
/// Linear (or Identity) activation function does not transform the input at all. It is typically used in the output layer
/// of a regression model, where we want to predict a numeric value.
///
/// Range: (-∞, +∞)
/// Best for: Output layers where prediction of continuous values is required.
#[derive(Serialize, Deserialize, Clone)]
struct LinearActivation;

/// Linear Activation Function
///
/// Linear (or Identity) activation function does not transform the input at all. It is typically used in the output layer
/// of a regression model, where we want to predict a numeric value.
///
/// Range: (-∞, +∞)
/// Best for: Output layers where prediction of continuous values is required.
pub struct Linear;

impl Linear {
    pub fn new() -> Box<dyn ActivationFunction> {
        Box::new(LinearActivation {})
    }
}

#[typetag::serde]
impl ActivationFunction for LinearActivation {
    fn forward(&self, _input: &mut DenseMatrix) {
        // Linear activation: no change to input
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix, _output: &DenseMatrix) {
        *input = d_output.clone();
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        |_, _| 1.0f32
    }
}

impl ActivationFunctionClone for LinearActivation {
    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod linear_tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_linear_forward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let linear = Linear::new();
        linear.forward(&mut input);

        // Expected output: same as input
        let expected = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert!(equal_approx(&input, &expected, 1e-4), "Linear forward pass failed");
    }

    #[test]
    fn test_linear_backward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let d_output = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let output: DenseMatrix = DenseMatrix::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output

        let linear = Linear::new();
        linear.backward(&d_output, &mut input, &output);

        // Expected output: same as d_output
        let expected = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);

        assert!(equal_approx(&input, &expected, 1e-4), "Linear backward pass failed");
    }
}

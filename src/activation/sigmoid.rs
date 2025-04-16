use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;

use serde::{Deserialize, Serialize};
use typetag;

use super::xavier_initialization;

#[derive(Serialize, Deserialize, Clone)]
pub struct Sigmoid;

impl Sigmoid {
    // Constructor for Sigmoid
    pub fn new() -> Self {
        Sigmoid {}
    }
}

#[typetag::serde]
impl ActivationFunction for Sigmoid {
    // Forward pass: Apply Sigmoid element-wise to the input matrix
    fn forward(&mut self, input: &mut DenseMatrix) {
        input.apply(|x| 1.0 / (1.0 + (-x).exp()));
    }

    // Backward pass: Compute the derivative of Sigmoid
    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix) {
        input.apply(|x| x * (1.0 - x));
        input.mul_elem(d_output);
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        xavier_initialization
    }
}

#[cfg(test)]
mod sigmoid_tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_sigmoid_forward_zero_input() {
        let mut input = DenseMatrix::new(1, 1, &[0.0f32]);
        let mut sigmoid = Sigmoid;
        sigmoid.forward(&mut input);

        let expected = DenseMatrix::new(1, 1, &[0.5f32]);
        assert!(equal_approx(&input, &expected, 1e-6), "Sigmoid forward pass with zero input failed");
    }

    #[test]
    fn test_sigmoid_forward_mixed_values() {
        let mut input = DenseMatrix::new(2, 3, &[-1.0f32, 0.0, 2.0, -3.5, 4.2, 0.0]);
        let mut sigmoid = Sigmoid;
        sigmoid.forward(&mut input);

        // Expected outputs calculated manually for verification
        let expected = DenseMatrix::new(
            2,
            3,
            &[
                1.0 / (1.0 + (-(-1.0f32)).exp()),
                0.5,
                1.0 / (1.0 + (-2.0f32).exp()),
                1.0 / (1.0 + (-(-3.5f32)).exp()),
                1.0 / (1.0 + (-4.2f32).exp()),
                0.5,
            ],
        );

        assert!(equal_approx(&input, &expected, 1e-6), "Sigmoid forward pass with mixed values failed");
    }

    #[test]
    fn test_sigmoid_backward() {
        let mut input = DenseMatrix::new(2, 3, &[-1.0f32, 0.0, 2.0, -3.5, 4.2, 0.0]);
        let d_output = DenseMatrix::new(2, 3, &[0.5f32, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let mut sigmoid = Sigmoid;

        sigmoid.forward(&mut input); // First apply forward pass
        let original_input = input.clone();

        sigmoid.backward(&d_output, &mut input);

        // Compute expected gradient: sigmoid(x) * (1 - sigmoid(x)) * d_output
        let expected = DenseMatrix::new(
            2,
            3,
            &[
                original_input.at(0, 0) * (1.0 - original_input.at(0, 0)) * d_output.at(0, 0),
                original_input.at(0, 1) * (1.0 - original_input.at(0, 1)) * d_output.at(0, 1),
                original_input.at(0, 2) * (1.0 - original_input.at(0, 2)) * d_output.at(0, 2),
                original_input.at(1, 0) * (1.0 - original_input.at(1, 0)) * d_output.at(1, 0),
                original_input.at(1, 1) * (1.0 - original_input.at(1, 1)) * d_output.at(1, 1),
                original_input.at(1, 2) * (1.0 - original_input.at(1, 2)) * d_output.at(1, 2),
            ],
        );

        assert!(equal_approx(&input, &expected, 1e-6), "Sigmoid backward pass failed");
    }

    #[test]
    fn test_sigmoid_bounds() {
        let test_cases = [(f32::NEG_INFINITY, 0.0f32), (f32::INFINITY, 1.0f32)];

        let mut sigmoid = Sigmoid;

        for (input_value, expected_output) in test_cases {
            let mut input = DenseMatrix::new(1, 1, &[input_value]);
            sigmoid.forward(&mut input);

            let expected = DenseMatrix::new(1, 1, &[expected_output]);
            assert!(equal_approx(&input, &expected, 1e-6), "Sigmoid forward pass at extreme bounds failed");
        }
    }
}

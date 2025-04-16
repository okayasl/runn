use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;

use serde::{Deserialize, Serialize};
use typetag;

use super::he_initialization;

// ELU Activation Function
#[derive(Serialize, Deserialize, Clone)]
pub struct ELU {
    alpha: f32,
}

impl ELU {
    // Constructor for ELU
    pub fn new(alpha: f32) -> Self {
        ELU { alpha }
    }
}

#[typetag::serde]
impl ActivationFunction for ELU {
    fn forward(&mut self, input: &mut DenseMatrix) {
        input.apply(|x| {
            if x > 0.0 {
                x // ELU(x) = x for x > 0
            } else {
                self.alpha * ((x).exp() - 1.0) // ELU(x) = alpha * (e^x - 1) for x <= 0
            }
        });
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix) {
        input.apply(|x| {
            if x > 0.0 {
                1.0 // dELU(x) = 1 for x > 0
            } else {
                self.alpha * (x).exp() // dELU(x) = alpha * e^x for x <= 0
            }
        });
        input.mul_elem(d_output); // Multiply the derivative of the output with the derivative of the input
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        he_initialization
    }
}

//ELU Builder for a user-friendly interface
pub struct ELUBuilder {
    alpha: f32,
}

impl ELUBuilder {
    pub fn new() -> Self {
        ELUBuilder { alpha: 1.0 } // Default alpha = 1.0
    }

    // Method to set the alpha value
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    // Method to build the ELU instance
    pub fn build(self) -> ELU {
        ELU::new(self.alpha)
    }
}

#[cfg(test)]
mod elu_tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_elu_forward_positive_values() {
        let mut elu = ELU::new(1.0);

        let mut input = DenseMatrix::new(1, 3, &[1.0, 2.0, 3.0]);

        elu.forward(&mut input);

        // Positive values should remain unchanged
        let expected = DenseMatrix::new(1, 3, &[1.0, 2.0, 3.0]);

        assert!(equal_approx(&input, &expected, 1e-6), "ELU forward pass with positive values failed");
    }

    #[test]
    fn test_elu_forward_mixed_values() {
        // Create an ELU with default alpha (typically 1.0)
        let mut elu = ELU::new(1.0);

        // Create a matrix with mixed positive and negative values
        let mut input = DenseMatrix::new(2, 3, &[-1.0, 0.0, 2.0, -3.5, 4.2, 0.0]);

        elu.forward(&mut input);

        // Expected output:
        // For x > 0: x remains unchanged
        // For x <= 0: alpha * (e^x - 1)
        let expected = DenseMatrix::new(
            2,
            3,
            &[
                elu.alpha * ((-1.0_f32).exp() - 1.0),
                0.0,
                2.0,
                elu.alpha * ((-3.5_f32).exp() - 1.0),
                4.2,
                0.0,
            ],
        );

        // Compare using approximate equality
        assert!(equal_approx(&input, &expected, 1e-6), "ELU forward pass with mixed values failed");
    }

    #[test]
    fn test_elu_backward_positive_values() {
        let elu = ELU::new(1.0);

        // Original input matrix with positive values
        let mut input = DenseMatrix::new(1, 3, &[1.0, 2.0, 3.0]);

        // Downstream gradient
        let d_output = DenseMatrix::new(1, 3, &[0.5, 1.0, 0.7]);

        elu.backward(&d_output, &mut input);

        // Expected output for positive values: gradient is 1.0
        let expected = DenseMatrix::new(1, 3, &[0.5, 1.0, 0.7]);

        assert!(equal_approx(&input, &expected, 1e-6), "ELU backward pass with positive values failed");
    }

    #[test]
    fn test_elu_backward_mixed_values() {
        // Create an ELU with default alpha (typically 1.0)
        let elu = ELU::new(1.0);

        // Original input matrix
        let mut input = DenseMatrix::new(2, 3, &[-1.0, 0.0, 2.0, -3.5, 4.2, 0.0]);

        // Downstream gradient
        let d_output = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);

        elu.backward(&d_output, &mut input);

        // Expected output:
        // For x > 0: 1.0 * d_output
        // For x <= 0: alpha * e^x * d_output
        let expected = DenseMatrix::new(
            2,
            3,
            &[
                elu.alpha * (-1.0_f32).exp() * 0.5,
                1.0,
                0.7,
                elu.alpha * (-3.5_f32).exp() * 0.2,
                0.3,
                0.1,
            ],
        );

        // Compare using approximate equality
        assert!(equal_approx(&input, &expected, 1e-6), "ELU backward pass with mixed values failed");
    }

    #[test]
    fn test_different_elu_alpha() {
        // Test with different alpha values
        let test_cases = [
            (0.5, -2.0, 0.5 * ((-2.0_f32).exp() - 1.0)),
            (1.5, -3.0, 1.5 * ((-3.0_f32).exp() - 1.0)),
        ];

        for (alpha, input_value, expected_output) in test_cases {
            let mut elu = ELU::new(alpha);

            let mut input = DenseMatrix::new(1, 1, &[input_value]);

            elu.forward(&mut input);

            let expected = DenseMatrix::new(1, 1, &[expected_output]);

            assert!(equal_approx(&input, &expected, 1e-6), "ELU forward pass with alpha failed");
        }
    }
}

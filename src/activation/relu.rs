use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;

use serde::{Deserialize, Serialize};
use typetag;

use super::he_initialization;

// ReLU (Rectified Linear Unit) Activation Function
//
// ReLU is a piecewise linear function that outputs zero for negative inputs and raw input for positive inputs.
// It is the most commonly used activation due to its simplicity and efficiency.
//
// Range: [0, +∞)
// Best for: General use in most neural networks, especially in hidden layers,
// as it helps to alleviate the vanishing gradient problem.
#[derive(Serialize, Deserialize, Clone)]
pub struct ReLUActivation;

pub struct ReLU;

impl ReLU {
    pub fn new() -> ReLUActivation {
        ReLUActivation {}
    }
}

#[typetag::serde]
impl ActivationFunction for ReLUActivation {
    // Forward pass: Apply ReLU element-wise to the input matrix.
    fn forward(&self, input: &mut DenseMatrix) {
        input.apply(|x| x.max(0.0)); // ReLU: max(0, x)
    }

    // Backward pass: Compute the derivative of ReLU.
    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix, _output: &DenseMatrix) {
        input.apply(|x| if x < 0.0 { 0.0 } else { 1.0 });
        input.mul_elem(d_output);
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        he_initialization
    }
}

#[cfg(test)]
mod relu_tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_relu_forward_positive_values() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let relu = ReLUActivation;
        relu.forward(&mut input);

        // Positive values should remain unchanged
        let expected = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert!(equal_approx(&input, &expected, 1e-6), "ReLU forward pass with positive values failed");
    }

    #[test]
    fn test_relu_forward_mixed_values() {
        let mut input = DenseMatrix::new(2, 3, &[-1.0, 0.0, 2.0, -3.5, 4.2, 0.0]);

        let relu = ReLUActivation;
        relu.forward(&mut input);

        // Expected output: zeros for negative values, unchanged for non-negative
        let expected = DenseMatrix::new(2, 3, &[0.0, 0.0, 2.0, 0.0, 4.2, 0.0]);

        assert!(equal_approx(&input, &expected, 1e-6), "ReLU forward pass with mixed values failed");
    }

    #[test]
    fn test_relu_backward() {
        // Original input matrix
        let mut input = DenseMatrix::new(2, 3, &[-1.0, 0.0, 2.0, -3.5, 4.2, 0.0]);

        // Downstream gradient
        let d_output = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);

        let relu = ReLUActivation;
        let output = input.clone();
        relu.backward(&d_output, &mut input, &output);

        // Expected output: zeros for inputs < 0, gradient otherwise
        let expected = DenseMatrix::new(2, 3, &[0.0, 1.0, 0.7, 0.0, 0.3, 0.1]);

        assert!(equal_approx(&input, &expected, 1e-6), "ReLU backward pass failed");
    }

    #[test]
    fn test_relu_backward_zero_gradient() {
        // Matrix with all negative values
        let mut input = DenseMatrix::new(1, 3, &[-1.0, -2.0, -3.0]);
        // Downstream gradient
        let d_output = DenseMatrix::new(1, 3, &[0.5, 1.0, 0.7]);
        let output: DenseMatrix = DenseMatrix::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output

        let relu = ReLUActivation;
        relu.backward(&d_output, &mut input, &output);

        // Expected output: all zeros
        let expected = DenseMatrix::new(1, 3, &[0.0, 0.0, 0.0]);

        assert!(equal_approx(&input, &expected, 1e-6), "ReLU backward pass with all negative values failed");
    }
}

use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use typetag;

use super::{he_initialization, ActivationFunctionClone};

/// LeakyReLU (Leaky Rectified Linear Unit) Activation Function
///
/// LeakyReLU is a variation of ReLU that allows a small, non-zero gradient when the input is less than zero.
/// This helps to mitigate the "dying ReLU" problem, where neurons can get stuck in a permanently inactive state.
///
/// Range: (-∞, +∞)
/// Best for: Improving learning in networks where the "dying ReLU" problem is a concern.
#[derive(Serialize, Deserialize, Clone)]
struct LeakyReLUActivation {
    alpha: f32,
}

/// LeakyReLU Builder for a user-friendly interface
/// LeakyReLU (Leaky Rectified Linear Unit) Activation Function
///
/// LeakyReLU is a variation of ReLU that allows a small, non-zero gradient when the input is less than zero.
/// This helps to mitigate the "dying ReLU" problem, where neurons can get stuck in a permanently inactive state.
///
/// Range: (-∞, +∞)
/// Best for: Improving learning in networks where the "dying ReLU" problem is a concern.
pub struct LeakyReLU {
    alpha: f32,
}

impl LeakyReLU {
    pub fn new() -> Self {
        LeakyReLU { alpha: 0.01 } // Default alpha = 0.01
    }

    /// Method to set the alpha value
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    // Method to build the LeakyReLU instanc
    pub fn build(self) -> Box<dyn ActivationFunction> {
        Box::new(LeakyReLUActivation { alpha: self.alpha })
    }
}

#[typetag::serde]
impl ActivationFunction for LeakyReLUActivation {
    fn forward(&self, input: &mut DenseMatrix) {
        input.apply(|x| if x > 0.0 { x } else { self.alpha * x });
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix, _output: &DenseMatrix) {
        input.apply(|x| if x > 0.0 { 1.0 } else { self.alpha });
        input.mul_elem(d_output);
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        he_initialization
    }
}

impl ActivationFunctionClone for LeakyReLUActivation {
    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod leakyrelu_tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_leakyrelu_forward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);

        let leakyrelu = LeakyReLU::new().alpha(0.01).build();
        leakyrelu.forward(&mut input);

        // Expected output: approximate values
        let expected = DenseMatrix::new(2, 3, &[1.0, -0.02, 3.0, -0.04, 5.0, -0.06]);

        assert!(equal_approx(&input, &expected, 1e-4), "LeakyReLU forward pass failed");
    }

    #[test]
    fn test_leakyrelu_backward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
        let d_output = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let output: DenseMatrix = DenseMatrix::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output

        let leakyrelu = LeakyReLU::new().alpha(0.01).build();
        leakyrelu.backward(&d_output, &mut input, &output);

        // Expected output: approximate values
        let expected = DenseMatrix::new(2, 3, &[0.5, 0.01, 0.7, 0.002, 0.3, 0.001]);

        assert!(equal_approx(&input, &expected, 1e-4), "LeakyReLU backward pass failed");
    }
}

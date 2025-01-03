use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use typetag;

#[derive(Serialize, Deserialize, Clone)]
pub struct Linear;

impl Linear {
    pub fn new() -> Self {
        Linear {}
    }
}

#[typetag::serde]
impl ActivationFunction for Linear {
    fn forward(&mut self, _input: &mut DenseMatrix) {
        // Linear activation: no change to input
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix) {
        *input = d_output.clone();
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        |_, _| 1.0f32
    }
}

#[cfg(test)]
mod linear_tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_linear_forward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut linear = Linear::new();
        linear.forward(&mut input);

        // Expected output: same as input
        let expected = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert!(
            equal_approx(&input, &expected, 1e-4),
            "Linear forward pass failed"
        );
    }

    #[test]
    fn test_linear_backward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let d_output = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);

        let linear = Linear::new();
        linear.backward(&d_output, &mut input);

        // Expected output: same as d_output
        let expected = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);

        assert!(
            equal_approx(&input, &expected, 1e-4),
            "Linear backward pass failed"
        );
    }
}

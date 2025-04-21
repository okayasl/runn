use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use typetag;

// Softplus Activation Function
//
// Softplus is a smooth approximation to the ReLU function, returning a positive output for any input.
// It is more differentiable than ReLU and can be useful in scenarios where a non-zero gradient is always necessary.
//
// Range: (0, +∞)
// Best for: Situations where a non-zero gradient is beneficial, providing a smooth approximation to ReLU.
#[derive(Serialize, Deserialize, Clone)]
pub struct SoftplusActivation;

pub struct Softplus;
impl Softplus {
    pub fn new() -> SoftplusActivation {
        SoftplusActivation {}
    }
}

#[typetag::serde]
impl ActivationFunction for SoftplusActivation {
    fn forward(&self, input: &mut DenseMatrix) {
        input.apply(|x| (1.0 + x.exp()).ln());
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix, _output: &DenseMatrix) {
        input.apply(|x| 1.0 / (1.0 + (-x).exp()));
        input.mul_elem(d_output);
    }
}

#[cfg(test)]
mod softplus_tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_softplus_forward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);

        let softplus = Softplus::new();
        softplus.forward(&mut input);

        // Expected output: approximate values
        let expected = DenseMatrix::new(2, 3, &[1.3133, 0.1269, 3.0486, 0.0181, 5.0067, 0.0025]);

        assert!(equal_approx(&input, &expected, 1e-4), "Softplus forward pass failed");
    }

    #[test]
    fn test_softplus_backward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
        let d_output = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let output: DenseMatrix = DenseMatrix::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output

        let softplus = Softplus::new();
        softplus.backward(&d_output, &mut input, &output);

        // Expected output: approximate values
        let expected = DenseMatrix::new(2, 3, &[0.365529, 0.119203, 0.666802, 0.003597, 0.297992, 0.000247]);
        assert!(equal_approx(&input, &expected, 1e-4), "Softplus backward pass failed");
    }
}

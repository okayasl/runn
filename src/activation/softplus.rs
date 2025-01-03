use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use typetag;

#[derive(Serialize, Deserialize, Clone)]
pub struct Softplus;

impl Softplus {
    pub fn new() -> Self {
        Softplus {}
    }
}

#[typetag::serde]
impl ActivationFunction for Softplus {
    fn forward(&mut self, input: &mut DenseMatrix) {
        input.apply(|x| (1.0 + x.exp()).ln());
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix) {
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

        let mut softplus = Softplus::new();
        softplus.forward(&mut input);

        // Expected output: approximate values
        let expected = DenseMatrix::new(2, 3, &[1.3133, 0.1269, 3.0486, 0.0181, 5.0067, 0.0025]);

        assert!(
            equal_approx(&input, &expected, 1e-4),
            "Softplus forward pass failed"
        );
    }

    #[test]
    fn test_softplus_backward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
        let d_output = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);

        let softplus = Softplus::new();
        softplus.backward(&d_output, &mut input);

        // Expected output: approximate values
        let expected = DenseMatrix::new(
            2,
            3,
            &[0.365529, 0.119203, 0.666802, 0.003597, 0.297992, 0.000247],
        );
        assert!(
            equal_approx(&input, &expected, 1e-4),
            "Softplus backward pass failed"
        );
    }
}

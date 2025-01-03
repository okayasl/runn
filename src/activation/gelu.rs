use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use special::Primitive;
use typetag;

use super::he_initialization;

#[derive(Serialize, Deserialize, Clone)]
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        GELU {}
    }
}

#[typetag::serde]
impl ActivationFunction for GELU {
    fn forward(&mut self, input: &mut DenseMatrix) {
        input.apply(|x| 0.5 * x * (1.0 + (x / (2.0_f32.sqrt())).erf()));
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix) {
        input.apply(|x| {
            let cdf = 0.5 * (1.0 + (x / (2.0_f32.sqrt())).erf());
            let pdf = (-(x * x) / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt();
            cdf + x * pdf
        });
        input.mul_elem(d_output);
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        he_initialization
    }
}

#[cfg(test)]
mod gelu_tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_gelu_forward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut gelu = GELU::new();
        gelu.forward(&mut input);

        // Expected output: approximate values
        let expected =
            DenseMatrix::new(2, 3, &[0.84130001, 1.9545, 2.9964, 3.9999, 4.9999, 5.9999]);

        assert!(
            equal_approx(&input, &expected, 1e-3),
            "GELU forward pass failed"
        );
    }

    #[test]
    fn test_gelu_backward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let d_output = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);

        let gelu = GELU::new();
        gelu.backward(&d_output, &mut input);

        // Expected output: approximate values
        let expected = DenseMatrix::new(
            2,
            3,
            &[0.541658, 1.085232, 0.708362, 0.200101, 0.300002, 0.100000],
        );

        assert!(
            equal_approx(&input, &expected, 1e-3),
            "GELU backward pass failed"
        );
    }
}

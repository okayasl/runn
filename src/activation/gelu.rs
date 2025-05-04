use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use typetag;

use super::{he_initialization, ActivationFunctionClone};

/// GeLU (Gaussian Error Linear Unit) Activation Function
///
/// GeLU is a smooth activation function that approximates the behavior of a gate,
/// using the input's magnitude to decide the neuron's output.
/// It uses the standard Gaussian cumulative distribution function.
///
/// Range: (0, +∞)
/// Best for: Transformer models (such as BERT) where it has been shown to improve performance
/// and convergence over standard ReLU.
#[derive(Serialize, Deserialize, Clone)]
struct GELUActivation;

/// GeLU (Gaussian Error Linear Unit) Activation Function
///
/// GeLU is a smooth activation function that approximates the behavior of a gate,
/// using the input's magnitude to decide the neuron's output.
/// It uses the standard Gaussian cumulative distribution function.
///
/// Range: (0, +∞)
/// Best for: Transformer models (such as BERT) where it has been shown to improve performance
/// and convergence over standard ReLU.
pub struct GELU;

impl GELU {
    pub fn new() -> Box<dyn ActivationFunction> {
        Box::new(GELUActivation {})
    }
}

#[typetag::serde]
impl ActivationFunction for GELUActivation {
    fn forward(&self, input: &mut DenseMatrix) {
        input.apply(|x| 0.5 * x * (1.0 + special::Primitive::erf(x / (2.0_f32.sqrt()))));
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix, _output: &DenseMatrix) {
        input.apply(|x| {
            let cdf = 0.5 * (1.0 + special::Primitive::erf(x / (2.0_f32.sqrt())));
            let pdf = (-(x * x) / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt();
            cdf + x * pdf
        });
        input.mul_elem(d_output);
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        he_initialization
    }
}

impl ActivationFunctionClone for GELUActivation {
    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod gelu_tests {

    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_gelu_forward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let gelu = GELU::new();
        gelu.forward(&mut input);

        // Expected output: approximate values
        let expected = DenseMatrix::new(2, 3, &[0.84130001, 1.9545, 2.9964, 3.9999, 4.9999, 5.9999]);

        assert!(equal_approx(&input, &expected, 1e-3), "GELU forward pass failed");
    }

    #[test]
    fn test_gelu_backward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let d_output = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let output: DenseMatrix = DenseMatrix::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output

        let gelu = GELU::new();
        gelu.backward(&d_output, &mut input, &output);

        // Expected output: approximate values
        let expected = DenseMatrix::new(2, 3, &[0.541658, 1.085232, 0.708362, 0.200101, 0.300002, 0.100000]);

        assert!(equal_approx(&input, &expected, 1e-3), "GELU backward pass failed");
    }
}

use crate::common::matrix::DMat;
use crate::{activation::ActivationFunction, error::NetworkError};
use serde::{Deserialize, Serialize};
use typetag;

use super::{he_initialization, ActivationFunctionClone};

// GeLU (Gaussian Error Linear Unit) Activation Function
//
// GeLU is a smooth activation function that approximates the behavior of a gate,
// using the input's magnitude to decide the neuron's output.
// It uses the standard Gaussian cumulative distribution function.
//
// Range: (0, +∞)
// Best for: Transformer models (such as BERT) where it has been shown to improve performance
// and convergence over standard ReLU.
#[derive(Serialize, Deserialize, Clone)]
struct GELUActivation;

/// GELU is a builder for GeLU (Gaussian Error Linear Unit) Activation Function
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
    /// Creates a new GELU activation function
    /// GELU weight initialization factor is set to He initialization.
    pub fn new() -> Result<Box<dyn ActivationFunction>, NetworkError> {
        Ok(Box::new(GELUActivation {}))
    }
}

#[typetag::serde]
impl ActivationFunction for GELUActivation {
    fn forward(&self, input: &mut DMat) {
        input.apply(|x| 0.5 * x * (1.0 + special::Primitive::erf(x / (2.0_f32.sqrt()))));
    }

    fn backward(&self, d_output: &DMat, input: &mut DMat, _output: &DMat) {
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
    use crate::{common::matrix::DMat, util::equal_approx};

    #[test]
    fn test_gelu_forward() {
        let mut input = DMat::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let gelu = GELU::new().unwrap();
        gelu.forward(&mut input);

        // Expected output: approximate values
        let expected = DMat::new(2, 3, &[0.84130001, 1.9545, 2.9964, 3.9999, 4.9999, 5.9999]);

        assert!(equal_approx(&input, &expected, 1e-3), "GELU forward pass failed");
    }

    #[test]
    fn test_gelu_backward() {
        let mut input = DMat::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let d_output = DMat::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let output: DMat = DMat::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output

        let gelu = GELU::new().unwrap();
        gelu.backward(&d_output, &mut input, &output);

        // Expected output: approximate values
        let expected = DMat::new(2, 3, &[0.541658, 1.085232, 0.708362, 0.200101, 0.300002, 0.100000]);

        assert!(equal_approx(&input, &expected, 1e-3), "GELU backward pass failed");
    }

    #[test]
    fn test_gelu_weight_initialization() {
        let gelu = GELU::new().unwrap();
        let factor = gelu.weight_initialization_factor()(2, 3);
        assert_eq!(factor, 0.8164966, "GELU weight initialization factor should be 0.8164966");
    }

    #[test]
    fn test_gelu_clone() {
        let gelu = GELU::new().unwrap();
        let _cloned_gelu = gelu.clone();
        assert!(true, "GELU clone failed");
    }
}

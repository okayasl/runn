use crate::common::matrix::DMat;
use crate::{activation::ActivationFunction, error::NetworkError};
use serde::{Deserialize, Serialize};
use typetag;

use super::ActivationFunctionClone;

// Linear Activation Function
//
// Linear (or Identity) activation function does not transform the input at all. It is typically used in the output layer
// of a regression model, where we want to predict a numeric value.
//
// Range: (-∞, +∞)
// Best for: Output layers where prediction of continuous values is required.
#[derive(Serialize, Deserialize, Clone)]
struct LinearActivation;

/// Linear is a builder for Linear Activation Function
///
/// Linear (or Identity) activation function does not transform the input at all. It is typically used in the output layer
/// of a regression model, where we want to predict a numeric value.
///
/// Range: (-∞, +∞)
/// Best for: Output layers where prediction of continuous values is required.
pub struct Linear;

impl Linear {
    /// Creates a new Linear activation function
    /// Linear weight initialization factor is set to 1.0.
    pub fn new() -> Result<Box<dyn ActivationFunction>, NetworkError> {
        Ok(Box::new(LinearActivation {}))
    }
}

#[typetag::serde]
impl ActivationFunction for LinearActivation {
    fn forward(&self, _input: &mut DMat) {
        // Linear activation: no change to input
    }

    fn backward(&self, d_output: &DMat, input: &mut DMat, _output: &DMat) {
        *input = d_output.clone();
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        |_, _| 1.0f32
    }
}

impl ActivationFunctionClone for LinearActivation {
    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod linear_tests {
    use super::*;
    use crate::{common::matrix::DMat, util::equal_approx};

    #[test]
    fn test_linear_forward() {
        let mut input = DMat::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let linear = Linear::new().unwrap();
        linear.forward(&mut input);

        // Expected output: same as input
        let expected = DMat::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert!(equal_approx(&input, &expected, 1e-4), "Linear forward pass failed");
    }

    #[test]
    fn test_linear_backward() {
        let mut input = DMat::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let d_output = DMat::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let output: DMat = DMat::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output

        let linear = Linear::new().unwrap();
        linear.backward(&d_output, &mut input, &output);

        // Expected output: same as d_output
        let expected = DMat::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);

        assert!(equal_approx(&input, &expected, 1e-4), "Linear backward pass failed");
    }

    #[test]
    fn test_linear_weight_initialization() {
        let linear = Linear::new().unwrap();
        let factor = linear.weight_initialization_factor()(2, 3);
        assert_eq!(factor, 1.0, "Linear weight initialization factor should be 1.0");
    }

    #[test]
    fn test_linear_clone() {
        let linear = Linear::new().unwrap();
        let _cloned = linear.clone_box();
        assert!(true, "Linear clone failed");
    }
}

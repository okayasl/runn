use crate::common::matrix::DMat;
use crate::{activation::ActivationFunction, error::NetworkError};
use serde::{Deserialize, Serialize};
use typetag;

use super::{he_initialization, ActivationFunctionClone};

// LeakyReLU (Leaky Rectified Linear Unit) Activation Function
//
// LeakyReLU is a variation of ReLU that allows a small, non-zero gradient when the input is less than zero.
// This helps to mitigate the "dying ReLU" problem, where neurons can get stuck in a permanently inactive state.
//
// Range: (-∞, +∞)
// Best for: Improving learning in networks where the "dying ReLU" problem is a concern.
#[derive(Serialize, Deserialize, Clone)]
struct LeakyReLUActivation {
    alpha: f32,
}

/// LeakyReLU Builder for LeakyReLU (Leaky Rectified Linear Unit) Activation Function
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
    /// Creates a new LeakyReLU activation function builder with default parameters.
    /// The default alpha value is typically set to 0.01.
    /// You can set a different alpha value using the `alpha` method.
    /// LeakyReLU weight initialization factor is set to He initialization.
    pub fn new() -> Self {
        LeakyReLU { alpha: 0.01 }
    }

    /// Sets the alpha parameter for the LeakyReLU activation function.
    /// Alpha controls the slope of the function for negative values.
    /// # Parameters
    /// - `alpha`: positive value.
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.alpha <= 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Alpha for LeakyReLU must be greater than 0.0, but was {}",
                self.alpha
            )));
        }
        Ok(())
    }

    // Method to build the LeakyReLU instance
    pub fn build(self) -> Result<Box<dyn ActivationFunction>, NetworkError> {
        self.validate()?;
        Ok(Box::new(LeakyReLUActivation { alpha: self.alpha }))
    }
}

#[typetag::serde]
impl ActivationFunction for LeakyReLUActivation {
    fn forward(&self, input: &mut DMat) {
        input.apply(|x| if x > 0.0 { x } else { self.alpha * x });
    }

    fn backward(&self, d_output: &DMat, input: &mut DMat, _output: &DMat) {
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
    use crate::{common::matrix::DMat, util::equal_approx};

    #[test]
    fn test_leakyrelu_forward() {
        let mut input = DMat::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);

        let leakyrelu = LeakyReLU::new().alpha(0.01).build().unwrap();
        leakyrelu.forward(&mut input);

        // Expected output: approximate values
        let expected = DMat::new(2, 3, &[1.0, -0.02, 3.0, -0.04, 5.0, -0.06]);

        assert!(equal_approx(&input, &expected, 1e-4), "LeakyReLU forward pass failed");
    }

    #[test]
    fn test_leakyrelu_backward() {
        let mut input = DMat::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
        let d_output = DMat::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let output: DMat = DMat::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output

        let leakyrelu = LeakyReLU::new().alpha(0.01).build().unwrap();
        leakyrelu.backward(&d_output, &mut input, &output);

        // Expected output: approximate values
        let expected = DMat::new(2, 3, &[0.5, 0.01, 0.7, 0.002, 0.3, 0.001]);

        assert!(equal_approx(&input, &expected, 1e-4), "LeakyReLU backward pass failed");
    }

    #[test]
    fn test_leakyrelu_invalid_alpha() {
        let leakyrelu = LeakyReLU::new().alpha(-0.01);
        let result = leakyrelu.build();
        assert!(result.is_err(), "LeakyReLU should not allow negative alpha");
    }
}

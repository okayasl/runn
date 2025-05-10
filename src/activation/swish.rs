use crate::common::matrix::DMat;
use crate::{activation::ActivationFunction, error::NetworkError};
use serde::{Deserialize, Serialize};
use typetag;

use super::{he_initialization, ActivationFunctionClone};

// Swish Activation Function
//
// Swish is a self-gated activation function defined as x * sigmoid(βx). It has been found to sometimes outperform ReLU
// in deeper networks due to its non-monotonic form.
//
// Range: (-∞, +∞)
// Best for: Deeper networks where traditional functions like ReLU tend to underperform.
#[derive(Serialize, Deserialize, Clone)]
struct SwishActivation {
    beta: f32,
}

/// Swish is a builder for Swish Activation Function
///
/// Swish is a self-gated activation function defined as x * sigmoid(βx). It has been found to sometimes outperform ReLU
/// in deeper networks due to its non-monotonic form.
///
/// Range: (-∞, +∞)
/// Best for: Deeper networks where traditional functions like ReLU tend to underperform.
pub struct Swish {
    beta: f32,
}

impl Swish {
    // Creates a new Swish activation function builder with default parameters.
    // The default beta value is typically set to 1.0.
    // You can set a different beta value using the `beta` method.
    // Swish weight initialization factor is set to He initialization.
    fn new() -> Self {
        Swish { beta: 1.0 } // Default beta = 1.0
    }

    /// Sets the beta parameter for the Swish activation function.
    /// Beta controls the steepness of the curve.
    /// #Parameters
    /// - `beta`: positive value.
    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.beta <= 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Beta for Swish must be greater than 0.0, but was {}",
                self.beta
            )));
        }
        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn ActivationFunction>, NetworkError> {
        self.validate()?;
        Ok(Box::new(SwishActivation { beta: self.beta }))
    }
}

impl Default for Swish {
    /// Creates a new Swish activation function builder with default parameters.
    /// Swish weight initialization factor is set to He initialization.
    /// Default values:
    /// - `beta`: 1.0
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl ActivationFunction for SwishActivation {
    fn forward(&self, input: &mut DMat) {
        input.apply(|x| x / (1.0 + (-self.beta * x).exp()));
    }

    fn backward(&self, d_output: &DMat, input: &mut DMat, _output: &DMat) {
        input.apply(|x| {
            let sigmoid = 1.0 / (1.0 + (-self.beta * x).exp());
            sigmoid + x * self.beta * sigmoid * (1.0 - sigmoid)
        });
        input.mul_elem(d_output);
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        he_initialization
    }
}

impl ActivationFunctionClone for SwishActivation {
    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod swish_tests {
    use super::*;
    use crate::{common::matrix::DMat, util::equal_approx};

    #[test]
    fn test_swish_forward() {
        let mut input = DMat::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);

        let swish = Swish::new().beta(1.0).build().unwrap();
        swish.forward(&mut input);

        // Expected output: approximate values
        let expected = DMat::new(2, 3, &[0.7311, -0.2384, 2.8577, -0.0728, 4.9665, -0.0147]);
        assert!(equal_approx(&input, &expected, 1e-3), "Swish forward pass failed");
    }

    #[test]
    fn test_swish_backward() {
        let mut input = DMat::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
        let d_output = DMat::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let output: DMat = DMat::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output

        let swish = Swish::new().beta(1.0).build().unwrap();
        swish.backward(&d_output, &mut input, &output);

        // Expected output: approximate values
        let expected = DMat::new(2, 3, &[0.463835, -0.090784, 0.761673, -0.010533, 0.307964, -0.001233]);
        assert!(equal_approx(&input, &expected, 1e-3), "Swish backward pass failed");
    }

    #[test]
    fn test_swish_invalid_beta() {
        let swish = Swish::new().beta(-1.0);
        assert!(swish.build().is_err(), "Swish activation function should not allow negative beta");
    }
}

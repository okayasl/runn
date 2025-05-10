use crate::common::matrix::DMat;
use crate::{activation::ActivationFunction, error::NetworkError};

use serde::{Deserialize, Serialize};
use typetag;

use super::{xavier_initialization, ActivationFunctionClone};

// Sigmoid Activation Function
//
// Sigmoid function outputs a value between 0 and 1, making it suitable for probabilities or as a gate in certain neural network architectures.
// It has a characteristic S-shaped curve.
//
// Range: (0, 1)
// Best for: Binary classification tasks in the output layer of a network.
#[derive(Serialize, Deserialize, Clone)]
struct SigmoidActivation;

/// Sigmoid is a builder for Sigmoid Activation Function
///
/// Sigmoid function outputs a value between 0 and 1, making it suitable for probabilities or as a gate in certain neural network architectures.
/// It has a characteristic S-shaped curve.
///
/// Range: (0, 1)
/// Best for: Binary classification tasks in the output layer of a network.
pub struct Sigmoid;

impl Sigmoid {
    // Creates a new Sigmoid activation function
    // Sigmoid weight initialization factor is set to Xavier initialization.
    fn new() -> Self {
        Self {}
    }

    pub fn build() -> Result<Box<dyn ActivationFunction>, NetworkError> {
        Ok(Box::new(SigmoidActivation {}))
    }
}

impl Default for Sigmoid {
    /// Creates a new Sigmoid activation function
    /// Sigmoid weight initialization factor is set to Xavier initialization.
    fn default() -> Self {
        Self::new()
    }
}

#[typetag::serde]
impl ActivationFunction for SigmoidActivation {
    // Forward pass: Apply Sigmoid element-wise to the input matrix
    fn forward(&self, input: &mut DMat) {
        input.apply(|x| 1.0 / (1.0 + (-x).exp()));
    }

    // Backward pass: Compute the derivative of Sigmoid
    fn backward(&self, d_output: &DMat, input: &mut DMat, _output: &DMat) {
        input.apply(|x| x * (1.0 - x));
        input.mul_elem(d_output);
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        xavier_initialization
    }
}

impl ActivationFunctionClone for SigmoidActivation {
    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod sigmoid_tests {
    use super::*;
    use crate::{common::matrix::DMat, util::equal_approx};

    #[test]
    fn test_sigmoid_forward_zero_input() {
        let mut input = DMat::new(1, 1, &[0.0f32]);
        let sigmoid = Sigmoid::build().unwrap();
        sigmoid.forward(&mut input);

        let expected = DMat::new(1, 1, &[0.5f32]);
        assert!(equal_approx(&input, &expected, 1e-6), "Sigmoid forward pass with zero input failed");
    }

    #[test]
    fn test_sigmoid_forward_mixed_values() {
        let mut input = DMat::new(2, 3, &[-1.0f32, 0.0, 2.0, -3.5, 4.2, 0.0]);
        let sigmoid = Sigmoid::build().unwrap();
        sigmoid.forward(&mut input);

        // Expected outputs calculated manually for verification
        let expected = DMat::new(
            2,
            3,
            &[
                1.0 / (1.0 + (-(-1.0f32)).exp()),
                0.5,
                1.0 / (1.0 + (-2.0f32).exp()),
                1.0 / (1.0 + (-(-3.5f32)).exp()),
                1.0 / (1.0 + (-4.2f32).exp()),
                0.5,
            ],
        );

        assert!(equal_approx(&input, &expected, 1e-6), "Sigmoid forward pass with mixed values failed");
    }

    #[test]
    fn test_sigmoid_backward() {
        let mut input = DMat::new(2, 3, &[-1.0f32, 0.0, 2.0, -3.5, 4.2, 0.0]);
        let d_output = DMat::new(2, 3, &[0.5f32, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let output: DMat = DMat::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output

        let sigmoid = Sigmoid::build().unwrap();
        sigmoid.forward(&mut input); // First apply forward pass
        let original_input = input.clone();

        sigmoid.backward(&d_output, &mut input, &output);

        // Compute expected gradient: sigmoid(x) * (1 - sigmoid(x)) * d_output
        let expected = DMat::new(
            2,
            3,
            &[
                original_input.at(0, 0) * (1.0 - original_input.at(0, 0)) * d_output.at(0, 0),
                original_input.at(0, 1) * (1.0 - original_input.at(0, 1)) * d_output.at(0, 1),
                original_input.at(0, 2) * (1.0 - original_input.at(0, 2)) * d_output.at(0, 2),
                original_input.at(1, 0) * (1.0 - original_input.at(1, 0)) * d_output.at(1, 0),
                original_input.at(1, 1) * (1.0 - original_input.at(1, 1)) * d_output.at(1, 1),
                original_input.at(1, 2) * (1.0 - original_input.at(1, 2)) * d_output.at(1, 2),
            ],
        );

        assert!(equal_approx(&input, &expected, 1e-6), "Sigmoid backward pass failed");
    }

    #[test]
    fn test_sigmoid_bounds() {
        let test_cases = [(f32::NEG_INFINITY, 0.0f32), (f32::INFINITY, 1.0f32)];

        let sigmoid = Sigmoid::build().unwrap();

        for (input_value, expected_output) in test_cases {
            let mut input = DMat::new(1, 1, &[input_value]);
            sigmoid.forward(&mut input);

            let expected = DMat::new(1, 1, &[expected_output]);
            assert!(equal_approx(&input, &expected, 1e-6), "Sigmoid forward pass at extreme bounds failed");
        }
    }
}

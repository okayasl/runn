use crate::common::matrix::DenseMatrix;
use crate::{activation::ActivationFunction, error::NetworkError};
use serde::{Deserialize, Serialize};
use typetag;

use super::{xavier_initialization, ActivationFunctionClone};

// Tanh (Hyperbolic Tangent) Activation Function
//
// Tanh outputs values between -1 and 1, effectively scaling the input data. It is symmetric around the origin, which can help
// keep the mean activations close to zero and potentially improve convergence rates.
//
// Range: (-1, 1)
// Best for: Hidden layers in a network where data normalization is beneficial, such as in certain types of autoencoders.
#[derive(Serialize, Deserialize, Clone)]
struct TanhActivation;

/// Tanh is a builder for Tanh (Hyperbolic Tangent) Activation Function
///
/// Tanh outputs values between -1 and 1, effectively scaling the input data. It is symmetric around the origin, which can help
/// keep the mean activations close to zero and potentially improve convergence rates.
///
/// Range: (-1, 1)
/// Best for: Hidden layers in a network where data normalization is beneficial, such as in certain types of autoencoders.
pub struct Tanh;

impl Tanh {
    /// Creates a new Tanh activation function
    pub fn new() -> Result<Box<dyn ActivationFunction>, NetworkError> {
        Ok(Box::new(TanhActivation {}))
    }
}

#[typetag::serde]
impl ActivationFunction for TanhActivation {
    // Forward pass: Apply Tanh element-wise to the input matrix
    fn forward(&self, input: &mut DenseMatrix) {
        input.apply(|x| x.tanh());
    }

    // Backward pass: Compute the derivative of Tanh
    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix, _output: &DenseMatrix) {
        input.apply(|x| {
            //let tanh_x = x.tanh();
            x * (1.0 - x * x) // derivative of tanh is 1 - tanh^2
        });
        input.mul_elem(d_output);
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        xavier_initialization
    }
}

impl ActivationFunctionClone for TanhActivation {
    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tanh_tests {

    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_tanh_forward_zero_input() {
        let mut input = DenseMatrix::new(1, 1, &[0.0f32]);
        let tanh = TanhActivation;
        tanh.forward(&mut input);

        let expected = DenseMatrix::new(1, 1, &[0.0f32]);
        assert!(equal_approx(&input, &expected, 1e-6), "Tanh forward pass with zero input failed");
    }

    #[test]
    fn test_tanh_forward_mixed_values() {
        let mut input = DenseMatrix::new(2, 3, &[-1.0f32, 0.0, 2.0, -3.5, 4.2, 0.0]);
        let tanh = TanhActivation;
        tanh.forward(&mut input);

        // Expected outputs using tanh function
        let expected = DenseMatrix::new(
            2,
            3,
            &[
                (-1.0f32).tanh(),
                0.0,
                2.0f32.tanh(),
                (-3.5f32).tanh(),
                4.2f32.tanh(),
                0.0,
            ],
        );

        assert!(equal_approx(&input, &expected, 1e-6), "Tanh forward pass with mixed values failed");
    }

    #[test]
    fn test_tanh_backward() {
        let mut input = DenseMatrix::new(2, 3, &[-1.0f32, 0.0, 2.0, -3.5, 4.2, 0.0]);
        let d_output = DenseMatrix::new(2, 3, &[0.5f32, 1.0, 0.7, 0.2, 0.3, 0.1]);
        let tanh = TanhActivation;

        tanh.forward(&mut input); // First apply forward pass
        let original_input = input.clone();
        let output: DenseMatrix = DenseMatrix::new(2, 3, &[0.0; 6]); // Create an empty DenseMatrix for output
        tanh.backward(&d_output, &mut input, &output);

        // Compute expected gradient: (1 - tanh(x)^2) * d_output
        let expected = DenseMatrix::new(
            2,
            3,
            &[
                original_input.at(0, 0) * (1.0 - original_input.at(0, 0).powi(2)) * d_output.at(0, 0),
                original_input.at(0, 1) * (1.0 - original_input.at(0, 1).powi(2)) * d_output.at(0, 1),
                original_input.at(0, 2) * (1.0 - original_input.at(0, 2).powi(2)) * d_output.at(0, 2),
                original_input.at(1, 0) * (1.0 - original_input.at(1, 0).powi(2)) * d_output.at(1, 0),
                original_input.at(1, 1) * (1.0 - original_input.at(1, 1).powi(2)) * d_output.at(1, 1),
                original_input.at(1, 2) * (1.0 - original_input.at(1, 2).powi(2)) * d_output.at(1, 2),
            ],
        );

        assert!(equal_approx(&input, &expected, 1e-6), "Tanh backward pass failed");
    }

    #[test]
    fn test_tanh_bounds() {
        let test_cases = [(f32::NEG_INFINITY, -1.0f32), (f32::INFINITY, 1.0f32)];

        let tanh = TanhActivation;

        for (input_value, expected_output) in test_cases {
            let mut input = DenseMatrix::new(1, 1, &[input_value]);
            tanh.forward(&mut input);

            let expected = DenseMatrix::new(1, 1, &[expected_output]);
            assert!(equal_approx(&input, &expected, 1e-6), "Tanh forward pass at extreme bounds failed");
        }
    }

    #[test]
    fn test_tanh_symmetry() {
        let test_cases = [
            (-2.0f32, -2.0f32.tanh()),
            (2.0f32, 2.0f32.tanh()),
            (-0.5f32, -0.5f32.tanh()),
            (0.5f32, 0.5f32.tanh()),
        ];

        let tanh = TanhActivation;

        for (input_value, expected_output) in test_cases {
            let mut input = DenseMatrix::new(1, 1, &[input_value]);
            tanh.forward(&mut input);

            let expected = DenseMatrix::new(1, 1, &[expected_output]);
            assert!(equal_approx(&input, &expected, 1e-6), "Tanh forward pass symmetry test failed");
        }
    }
}

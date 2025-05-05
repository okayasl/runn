use crate::common::matrix::DenseMatrix;
use crate::{activation::ActivationFunction, error::NetworkError};

use serde::{Deserialize, Serialize};
use typetag;

use super::{xavier_initialization, ActivationFunctionClone};

// Softmax Activation Function converts a vector of values into a normalized probability distribution,
// where each element is in the range (0, 1) and all elements sum to 1.
// It is typically used in the output layer of a classification model to represent
// confidence scores across multiple classes.
//
// Range: (0, 1) for each output element
// Best for: Output layers of multi-class classification models.
#[derive(Serialize, Deserialize, Clone)]
struct SoftmaxActivation {}

/// Softmax  is a builder for Softmax Activation Function
///
/// It converts a vector of values into a normalized probability distribution,
/// where each element is in the range (0, 1) and all elements sum to 1.
/// It is typically used in the output layer of a classification model to represent
/// confidence scores across multiple classes.
///
/// Range: (0, 1) for each output element
/// Best for: Output layers of multi-class classification models.
pub struct Softmax;

impl Softmax {
    /// Creates a new Softmax activation function
    pub fn new() -> Result<Box<dyn ActivationFunction>, NetworkError> {
        Ok(Box::new(SoftmaxActivation {}))
    }
}

#[typetag::serde]
impl ActivationFunction for SoftmaxActivation {
    fn forward(&self, input: &mut DenseMatrix) {
        let (rows, cols) = (input.rows(), input.cols());
        let mut result = DenseMatrix::zeros(rows, cols);

        for i in 0..rows {
            // Step 1: Find the maximum value to stabilize the exponentials
            let max_val = (0..cols).fold(f32::NEG_INFINITY, |max, j| max.max(input.at(i, j)));

            // Step 2: Compute the exponentials and their sum
            let mut exp_sum = 0.0;
            for j in 0..cols {
                let exp_val = (input.at(i, j) - max_val).exp();
                result.set(i, j, exp_val);
                exp_sum += exp_val;
            }

            // Step 3: Normalize the exponentials
            for j in 0..cols {
                result.set(i, j, result.at(i, j) / exp_sum);
            }
        }

        *input = result;
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix, output: &DenseMatrix) {
        let (rows, cols) = (input.rows(), input.cols());
        let mut result = DenseMatrix::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                let mut gradient = 0.0;
                for k in 0..cols {
                    if j == k {
                        gradient += output.at(i, k) * (1.0 - output.at(i, j)) * d_output.at(i, k);
                    } else {
                        gradient += -output.at(i, k) * output.at(i, j) * d_output.at(i, k);
                    }
                }
                result.set(i, j, gradient);
            }
        }

        *input = result;
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        xavier_initialization
    }
}

impl ActivationFunctionClone for SoftmaxActivation {
    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::matrix::DenseMatrix;

    #[test]
    fn test_softmax_forward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let softmax = Softmax::new().unwrap();
        softmax.forward(&mut input);

        // Check if the output sums to 1 for each row
        for i in 0..input.rows() {
            let sum: f32 = (0..input.cols()).map(|j| input.at(i, j)).sum();
            assert!((sum - 1.0).abs() < 1e-6, "Softmax output does not sum to 1");
        }
    }
    #[test]
    fn test_softmax_backward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let output = DenseMatrix::new(2, 3, &[0.1, 0.2, 0.7, 0.1, 0.2, 0.7]);
        let d_output = DenseMatrix::new(2, 3, &[0.01, 0.02, 0.03, 0.01, 0.02, 0.03]);

        let softmax = Softmax::new().unwrap();
        softmax.forward(&mut input);
        softmax.backward(&d_output, &mut input, &output);

        // Compute expected gradients manually for verification
        let expected_gradients = DenseMatrix::new(2, 3, &[-0.0016, -0.0011, 0.0028, -0.0016, -0.0011, 0.0028]);

        for i in 0..input.rows() {
            for j in 0..input.cols() {
                let computed = input.at(i, j);
                let expected = expected_gradients.at(i, j);
                assert!(
                    (computed - expected).abs() < 1e-4,
                    "Gradient mismatch at ({}, {}): computed = {}, expected = {}",
                    i,
                    j,
                    computed,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_softmax_small_input() {
        let mut input = DenseMatrix::new(1, 3, &[1.0, 2.0, 3.0]);
        let softmax = Softmax::new().unwrap();
        softmax.forward(&mut input);

        let expected = DenseMatrix::new(1, 3, &[0.0900, 0.2447, 0.6652]);

        for j in 0..input.cols() {
            let computed = input.at(0, j);
            let expected_val = expected.at(0, j);
            assert!(
                (computed - expected_val).abs() < 1e-4,
                "Mismatch at column {}: computed = {}, expected = {}",
                j,
                computed,
                expected_val
            );
        }
    }

    #[test]
    fn test_softmax_large_positive_values() {
        let mut input = DenseMatrix::new(1, 3, &[100.0, 200.0, 300.0]);
        let softmax = Softmax::new().unwrap();
        softmax.forward(&mut input);

        // The largest value should dominate
        assert!((input.at(0, 2) - 1.0).abs() < 1e-6, "Expected probability close to 1 for largest input");
        assert!(input.at(0, 0) < 1e-6, "Expected probability close to 0 for smaller input");
        assert!(input.at(0, 1) < 1e-6, "Expected probability close to 0 for smaller input");
    }

    #[test]
    fn test_softmax_large_negative_values() {
        let mut input = DenseMatrix::new(1, 3, &[-100.0, -200.0, -300.0]);
        let softmax = Softmax::new().unwrap();
        softmax.forward(&mut input);

        // The least negative value should dominate
        assert!((input.at(0, 0) - 1.0).abs() < 1e-6, "Expected probability close to 1 for least negative input");
        assert!(input.at(0, 1) < 1e-6, "Expected probability close to 0 for more negative input");
        assert!(input.at(0, 2) < 1e-6, "Expected probability close to 0 for more negative input");
    }

    #[test]
    fn test_softmax_equal_values() {
        let mut input = DenseMatrix::new(1, 3, &[1.0, 1.0, 1.0]);
        let softmax = Softmax::new().unwrap();
        softmax.forward(&mut input);

        // All probabilities should be equal
        let expected = 1.0 / 3.0;
        for j in 0..input.cols() {
            let computed = input.at(0, j);
            assert!(
                (computed - expected).abs() < 1e-6,
                "Mismatch at column {}: computed = {}, expected = {}",
                j,
                computed,
                expected
            );
        }
    }

    #[test]
    fn test_softmax_empty_input() {
        let mut input = DenseMatrix::new(0, 0, &[]);
        let softmax = Softmax::new().unwrap();
        softmax.forward(&mut input);

        // Ensure no panic and output remains empty
        assert_eq!(input.rows(), 0);
        assert_eq!(input.cols(), 0);
    }

    #[test]
    fn test_softmax_single_element() {
        let mut input = DenseMatrix::new(1, 1, &[42.0]);
        let softmax = Softmax::new().unwrap();
        softmax.forward(&mut input);

        // Single element should always have probability 1
        assert!((input.at(0, 0) - 1.0).abs() < 1e-6, "Expected probability of 1 for single element");
    }
}

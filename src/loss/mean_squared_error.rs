use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use typetag;

use super::LossFunction;

// MeanSquaredErrorLoss is a commonly used loss function for regression tasks,
// measuring the average squared difference between the predicted values and the true target values.
// It penalizes larger errors more heavily than smaller ones, making it sensitive to outliers.
// The loss is computed as the mean of the squared differences between the predicted and target values.
// Forward pass:
// loss = (1 / N) * Σ((predicted - target) ** 2)
// where N is the number of samples, and the summation is over all elements in the matrices.
// Backward pass:
// gradient = (2 / N) * (predicted - target)
// The gradient represents the scaled difference between the predicted and target values,
// which can be used to update the model parameters during optimization.
#[derive(Serialize, Deserialize, Clone)]
pub struct MeanSquaredErrorLoss;

pub struct MeanSquared;

impl MeanSquared {
    /// Creates a new builder for CrossEntropyLoss
    pub fn new() -> MeanSquaredErrorLoss {
        MeanSquaredErrorLoss {}
    }
}

#[typetag::serde]
impl LossFunction for MeanSquaredErrorLoss {
    fn forward(&self, predicted: &DenseMatrix, target: &DenseMatrix) -> f32 {
        let mut loss = 0.0;
        let rows = predicted.rows();

        for i in 0..rows {
            let diff = predicted.at(i, 0) - target.at(i, 0);
            loss += diff * diff;
        }

        loss / rows as f32
    }

    fn backward(&self, predicted: &DenseMatrix, target: &DenseMatrix) -> DenseMatrix {
        let (rows, cols) = (predicted.rows(), predicted.cols());
        let mut gradient = DenseMatrix::zeros(rows, cols);

        gradient.apply_with_indices(|i, j, v| {
            let diff = predicted.at(i, j) - target.at(i, j);
            *v = 2.0 * diff / rows as f32;
        });

        gradient
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util};

    #[test]
    fn test_forward() {
        let loss = MeanSquared::new();
        let predicted = DenseMatrix::new(2, 1, &[0.9, 0.2]);
        let target = DenseMatrix::new(2, 1, &[1.0, 0.0]);
        let result = loss.forward(&predicted, &target);
        assert!((result - 0.025).abs() < 1e-6);
    }

    #[test]
    fn test_backward() {
        let loss = MeanSquared::new();
        let predicted = DenseMatrix::new(2, 1, &[0.9, 0.2]);
        let target = DenseMatrix::new(2, 1, &[1.0, 0.0]);
        let gradient = loss.backward(&predicted, &target);
        let expected_gradient = DenseMatrix::new(2, 1, &[-0.1, 0.2]);
        assert!(util::equal_approx(&gradient, &expected_gradient, 1e-6));
    }
}

use serde::{Deserialize, Serialize};
use typetag;

use crate::{
    classification::ClassificationEvaluator, error::NetworkError, matrix::DMat, MetricEvaluator, MetricResult,
};

use super::{LossFunction, LossFunctionClone};

// CrossEntropyLoss is a commonly used loss function for classification tasks,
// particularly in scenarios involving probabilistic outputs such as those from a softmax layer.
// It measures the dissimilarity between the predicted probability distribution and the true labels,
// penalizing predictions that deviate from the target distribution.
// The loss is computed as the negative log-likelihood of the true labels given the predicted probabilities.
// To prevent numerical instability (e.g., log(0)), the predicted probabilities are clamped
// to the range [epsilon, 1-epsilon], where epsilon is a small positive constant.
// This ensures that the logarithm operation remains well-defined and avoids issues like division by zero.
// Forward pass:
// loss = -Σ(target * log(predicted))
// where the summation is over all elements in the matrices.
// Backward pass:
// gradient = predicted - target
// The gradient represents the difference between the predicted probabilities and the true labels,
// which can be used to update the model parameters during optimization.
#[derive(Serialize, Deserialize, Clone)]
struct CrossEntropyLoss {
    epsilon: f32,
}

/// CrossEntropy is a builder for Cross-Entropy Loss which is a commonly used loss function for classification tasks,
/// particularly in scenarios involving probabilistic outputs such as those from a softmax layer.
///
/// It measures the dissimilarity between the predicted probability distribution and the true labels,
/// penalizing predictions that deviate from the target distribution.
/// The loss is computed as the negative log-likelihood of the true labels given the predicted probabilities.
/// To prevent numerical instability (e.g., log(0)), the predicted probabilities are clamped
/// to the range [epsilon, 1-epsilon], where epsilon is a small positive constant.
/// This ensures that the logarithm operation remains well-defined and avoids issues like division by zero.
///
/// Forward pass:
/// loss = -Σ(target * log(predicted))
/// where the summation is over all elements in the matrices.
///
/// Backward pass:
/// gradient = predicted - target
///
/// The gradient represents the difference between the predicted probabilities and the true labels,
/// which can be used to update the model parameters during optimization.
pub struct CrossEntropy {
    epsilon: f32,
}

impl CrossEntropy {
    /// Creates a new builder for CrossEntropyLoss
    /// The default epsilon value is set to f32::EPSILON
    pub fn new() -> Self {
        Self { epsilon: f32::EPSILON }
    }

    /// Sets the epsilon value for the loss function
    /// Epsilon is a small positive constant used to prevent numerical instability
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Validates the parameters of the loss function
    /// Ensures that epsilon is set and within a valid range
    fn validate(&self) -> Result<(), NetworkError> {
        if self.epsilon <= 0.0 || self.epsilon >= 1.0 {
            return Err(NetworkError::ConfigError(format!(
                "Epsilon for CrossEntropy must be in the range (0, 1), but was {}",
                self.epsilon
            )));
        }
        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn LossFunction>, NetworkError> {
        self.validate()?;
        Ok(Box::new(CrossEntropyLoss { epsilon: self.epsilon }))
    }
}

impl LossFunctionClone for CrossEntropyLoss {
    fn clone_box(&self) -> Box<dyn LossFunction> {
        Box::new(self.clone())
    }
}

#[typetag::serde]
impl LossFunction for CrossEntropyLoss {
    fn forward(&self, predicted: &DMat, target: &DMat) -> f32 {
        let mut total_loss = 0.0;
        let (rows, cols) = (predicted.rows(), predicted.cols());

        let (pred_rows, pred_cols) = (predicted.rows(), predicted.cols());
        let (targ_rows, targ_cols) = (target.rows(), target.cols());
        // Ensure the dimensions of predicted and target matrices match
        if pred_rows != targ_rows || pred_cols != targ_cols {
            panic!(
                "Dimension mismatch: predicted matrix is {}x{}, but target matrix is {}x{}",
                pred_rows, pred_cols, targ_rows, targ_cols
            );
        }

        for i in 0..rows {
            for j in 0..cols {
                let mut v = predicted.at(i, j);
                let t = target.at(i, j);
                // Clamp v to [epsilon, 1-epsilon] to prevent numerical issues like log(0)
                v = v.max(self.epsilon).min(1.0 - self.epsilon);
                // Accumulate the loss
                total_loss -= t * v.ln();
            }
        }

        // Return the average loss
        total_loss / rows as f32
    }

    fn backward(&self, predicted: &DMat, target: &DMat) -> DMat {
        let (rows, cols) = (predicted.rows(), predicted.cols());
        let mut gradient = DMat::zeros(rows, cols);

        gradient.apply_with_indices(|i, j, v| {
            let t = target.at(i, j);
            let p = predicted.at(i, j);
            // Compute the gradient
            *v = p - t;
        });

        gradient
    }
    fn calculate_metrics(&self, targets: &DMat, predictions: &DMat) -> MetricResult {
        ClassificationEvaluator.evaluate(targets, predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{common::matrix::DMat, util};

    #[test]
    fn test_forward() {
        let loss = CrossEntropy::new().epsilon(1e-7).build().unwrap();
        let predicted = DMat::new(2, 2, &[0.9, 0.1, 0.2, 0.8]);
        let target = DMat::new(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = loss.forward(&predicted, &target);
        assert!((result - 0.164252033486018).abs() < 1e-6);
    }

    #[test]
    fn test_backward() {
        let loss = CrossEntropy::new().epsilon(1e-7).build().unwrap();
        let predicted = DMat::new(2, 2, &[0.9, 0.1, 0.2, 0.8]);
        let target = DMat::new(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let gradient = loss.backward(&predicted, &target);
        let expected_gradient = DMat::new(2, 2, &[-0.1, 0.1, 0.2, -0.2]);
        assert!(util::equal_approx(&gradient, &expected_gradient, 1e-6));
    }

    #[test]
    fn test_crossentropy_validate() {
        let loss = CrossEntropy::new().epsilon(1e-7);
        assert!(loss.validate().is_ok());

        let loss = CrossEntropy::new().epsilon(0.0);
        assert!(loss.validate().is_err());

        let loss = CrossEntropy::new().epsilon(1.0);
        assert!(loss.validate().is_err());
    }
}

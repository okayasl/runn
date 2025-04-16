use serde::{Deserialize, Serialize};
use typetag;

use crate::matrix::DenseMatrix;

use super::LossFunction;

#[derive(Serialize, Deserialize, Clone)]
pub struct CrossEntropyLoss {
    epsilon: f32,
}

impl CrossEntropyLoss {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }
}
#[typetag::serde]
impl LossFunction for CrossEntropyLoss {
    fn forward(&self, predicted: &DenseMatrix, target: &DenseMatrix) -> f32 {
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

    fn backward(&self, predicted: &DenseMatrix, target: &DenseMatrix) -> DenseMatrix {
        let (rows, cols) = (predicted.rows(), predicted.cols());
        let mut gradient = DenseMatrix::zeros(rows, cols);

        gradient.apply_with_indices(|i, j, v| {
            let t = target.at(i, j);
            let p = predicted.at(i, j);
            // Compute the gradient
            *v = p - t;
        });

        gradient
    }
}

pub struct CrossEntropy {
    epsilon: Option<f32>,
}

impl CrossEntropy {
    /// Creates a new builder for CrossEntropyLoss
    pub fn new() -> Self {
        Self { epsilon: None }
    }

    /// Sets the epsilon value for the loss function
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = Some(epsilon);
        self
    }

    /// Builds the CrossEntropyLoss instance
    pub fn build(self) -> CrossEntropyLoss {
        CrossEntropyLoss {
            epsilon: self.epsilon.unwrap_or(1e-8), // Default epsilon value if not set
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util};

    #[test]
    fn test_forward() {
        let loss = CrossEntropyLoss::new(1e-7);
        let predicted = DenseMatrix::new(2, 2, &[0.9, 0.1, 0.2, 0.8]);
        let target = DenseMatrix::new(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = loss.forward(&predicted, &target);
        assert!((result - 0.164252033486018).abs() < 1e-6);
    }

    #[test]
    fn test_backward() {
        let loss = CrossEntropyLoss::new(1e-7);
        let predicted = DenseMatrix::new(2, 2, &[0.9, 0.1, 0.2, 0.8]);
        let target = DenseMatrix::new(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let gradient = loss.backward(&predicted, &target);
        let expected_gradient = DenseMatrix::new(2, 2, &[-0.1, 0.1, 0.2, -0.2]);
        assert!(util::equal_approx(&gradient, &expected_gradient, 1e-6));
    }
}

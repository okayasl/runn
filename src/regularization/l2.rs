use crate::{common::matrix::DenseMatrix, error::NetworkError};

use serde::{Deserialize, Serialize};
use typetag;

use super::{Regularization, RegularizationClone};

// L2 regularization(also known as Ridge regularization) adds a penalty term to
// the loss function that is proportional to the square of the weights.
// This encourages the weights to be small but non-zero, effectively shrinking the weights
// towards zero but not driving them to exactly zero.
//
// L2 regularization can help to prevent overfitting by reducing the complexity of
// the model and forcing it to learn simpler patterns. It works by adding a term
// to the loss function that penalizes large weights, which can lead to overly
// complex models that fit the training data too closely.
//
// The lambda parameter controls the strength of the regularization. A higher lambda
// value will result in smaller weights and a simpler model, potentially reducing
// overfitting but also increasing the risk of underfitting if the value is too high.
//
// L2 regularization is commonly used in neural networks and other machine learning
// models, especially when dealing with high-dimensional data or when there is a risk
// of overfitting due to the complexity of the model.
//
// Unlike L1 regularization, which can drive some weights to exactly zero (leading
// to sparse models), L2 regularization tends to keep all weights non-zero but small.
// This can be advantageous when all features are potentially relevant and feature
// selection is not a primary concern.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct L2Regularization {
    lambda: f32,
}

#[typetag::serde]
impl Regularization for L2Regularization {
    fn apply(&self, params: &mut [&mut DenseMatrix], grads: &mut [&mut DenseMatrix]) {
        for (param, grad) in params.iter().zip(grads.iter_mut()) {
            grad.apply_with_indices(|i, j, v| {
                let p = param.at(i, j);
                *v += self.lambda * p * p;
            });
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl RegularizationClone for L2Regularization {
    fn clone_box(&self) -> Box<dyn Regularization> {
        Box::new(self.clone())
    }
}

/// L2 is a Builder for L2 regularization(also known as Ridge regularization) which adds
/// a penalty term to the loss function that is proportional to the square of the weights.
/// This encourages the weights to be small but non-zero, effectively shrinking the weights
/// towards zero but not driving them to exactly zero.
///
/// L2 regularization can help to prevent overfitting by reducing the complexity of
/// the model and forcing it to learn simpler patterns. It works by adding a term
/// to the loss function that penalizes large weights, which can lead to overly
/// complex models that fit the training data too closely.
///
/// The lambda parameter controls the strength of the regularization. A higher lambda
/// value will result in smaller weights and a simpler model, potentially reducing
/// overfitting but also increasing the risk of underfitting if the value is too high.
///
/// L2 regularization is commonly used in neural networks and other machine learning
/// models, especially when dealing with high-dimensional data or when there is a risk
/// of overfitting due to the complexity of the model.
///
/// Unlike L1 regularization, which can drive some weights to exactly zero (leading
/// to sparse models), L2 regularization tends to keep all weights non-zero but small.
/// This can be advantageous when all features are potentially relevant and feature
/// selection is not a primary concern.
pub struct L2 {
    lambda: f32,
}

impl L2 {
    /// Creates a new builder for L2Regularization
    /// Default values:
    /// - `lambda`: 0.01
    pub fn new() -> Self {
        Self { lambda: 0.01 }
    }

    /// Set the L2 regularization strength (lambda).
    ///
    /// Controls the penalty applied to the square of weights. Higher values reduce weight magnitudes but may affect model accuracy.
    /// # Parameters
    /// - `lambda`: Regularization strength (e.g., 0.01).
    pub fn lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.lambda < 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Lambda for L2 regularization must be positive, but was {}",
                self.lambda
            )));
        }
        Ok(())
    }

    /// Builds the L2Regularization instance
    pub fn build(self) -> Result<Box<dyn Regularization>, NetworkError> {
        self.validate()?;
        Ok(Box::new(L2Regularization { lambda: self.lambda }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::matrix::DenseMatrix;
    use crate::util::equal_approx;

    #[test]
    fn test_l2_regularization() {
        let mut params = vec![DenseMatrix::new(2, 2, &[1.0, -2.0, 3.0, -4.0])];
        let mut grads = vec![DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1])];
        let l2 = L2::new().lambda(0.01).build().unwrap();

        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        l2.apply(&mut params_refs, &mut grads_refs);

        let expected_grads = DenseMatrix::new(2, 2, &[0.11, 0.14, 0.19, 0.26]);
        equal_approx(&grads[0], &expected_grads, 1e-6);
    }

    #[test]
    fn test_l2_regularization_invalid_lambda() {
        let l2 = L2::new().lambda(-0.01);
        let result = l2.build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(
                e.to_string(),
                "Configuration error: Lambda for L2 regularization must be positive, but was -0.01"
            );
        }
    }
}

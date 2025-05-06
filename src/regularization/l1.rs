use super::{Regularization, RegularizationClone};
use crate::{common::matrix::DenseMatrix, error::NetworkError};

use serde::{Deserialize, Serialize};
use typetag;

// L1 regularization(also known as Lasso regularization) adds a penalty term
// to the loss function that is proportional to the absolute value of the weights.
// This encourages the weights to be sparse, meaning that some weights will be driven
// to exactly zero, effectively removing some connections from the network.
//
// L1 regularization can help to prevent overfitting and improve the interpretability
// of the model by identifying and removing irrelevant features. It is particularly
// useful when the input data has many features, and some of them are redundant or
// not relevant to the problem.
//
// The lambda parameter controls the strength of the regularization. A higher lambda
// value will result in more weights being driven to zero, potentially leading to a
// sparser model but also increasing the risk of underfitting.
//
// L1 regularization is commonly used in linear models and sparse models, where
// feature selection and interpretability are important.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct L1Regularization {
    lambda: f32,
}

#[typetag::serde]
impl Regularization for L1Regularization {
    fn apply(&self, params: &mut [&mut DenseMatrix], grads: &mut [&mut DenseMatrix]) {
        for (param, grad) in params.iter().zip(grads.iter_mut()) {
            grad.apply_with_indices(|i, j, v| {
                let p = param.at(i, j);
                *v += self.lambda * p.abs();
            });
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl RegularizationClone for L1Regularization {
    fn clone_box(&self) -> Box<dyn Regularization> {
        Box::new(self.clone())
    }
}

/// L1 is a builder for L1 regularization(also known as Lasso regularization) which
/// adds a penalty term to the loss function that is proportional to the absolute value
/// of the weights. This encourages the weights to be sparse, meaning that some weights
/// will be driven to exactly zero, effectively removing some connections from the network.
///
/// L1 regularization can help to prevent overfitting and improve the interpretability
/// of the model by identifying and removing irrelevant features. It is particularly
/// useful when the input data has many features, and some of them are redundant or
/// not relevant to the problem.
///
/// The lambda parameter controls the strength of the regularization. A higher lambda
/// value will result in more weights being driven to zero, potentially leading to a
/// sparser model but also increasing the risk of underfitting.
///
/// L1 regularization is commonly used in linear models and sparse models, where
/// feature selection and interpretability are important.
pub struct L1 {
    lambda: f32,
}

impl L1 {
    /// Creates a new builder for L1Regularization
    /// Default values:
    /// - `lambda`: 0.01
    pub fn new() -> Self {
        Self { lambda: 0.01 }
    }

    /// Set the L1 regularization strength (lambda).
    ///
    /// Controls the penalty applied to the absolute value of weights. Higher values increase sparsity but may reduce model accuracy.
    /// # Parameters
    /// - `lambda`: Regularization strength (e.g., 0.01).
    pub fn lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.lambda < 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Lambda for L1 regularization must be positive, but was {}",
                self.lambda
            )));
        }
        Ok(())
    }

    /// Builds the L1Regularization instance
    pub fn build(self) -> Result<Box<dyn Regularization>, NetworkError> {
        self.validate()?;
        Ok(Box::new(L1Regularization { lambda: self.lambda }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::matrix::DenseMatrix;
    use crate::util::equal_approx;

    #[test]
    fn test_l1_regularization() {
        let mut params = vec![DenseMatrix::new(2, 2, &[1.0, -2.0, 3.0, -4.0])];
        let mut grads = vec![DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1])];
        let l1 = L1::new().lambda(0.01).build().unwrap();

        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        l1.apply(&mut params_refs, &mut grads_refs);

        let expected_grads = DenseMatrix::new(2, 2, &[0.11, 0.12, 0.13, 0.14]);
        equal_approx(&grads[0], &expected_grads, 1e-6);
    }

    #[test]
    fn test_l1_regularization_invalid_lambda() {
        let l1 = L1::new().lambda(-0.01);
        let result = l1.build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(
                e.to_string(),
                "Configuration error: Lambda for L1 regularization must be positive, but was -0.01"
            );
        }
    }
}

use crate::common::matrix::DenseMatrix;

use serde::{Deserialize, Serialize};
use typetag;

use super::Regularization;

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
pub struct L2Regularization {
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

pub struct L2 {
    lambda: Option<f32>,
}

impl L2 {
    /// Creates a new builder for L2Regularization
    pub fn new() -> Self {
        Self { lambda: None }
    }

    /// Sets the lambda value for L2 regularization
    pub fn lambda(mut self, lambda: f32) -> Self {
        self.lambda = Some(lambda);
        self
    }

    /// Builds the L2Regularization instance
    pub fn build(self) -> L2Regularization {
        L2Regularization {
            lambda: self.lambda.expect("Lambda must be set"),
        }
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
        let l2 = L2::new().lambda(0.01).build();

        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        l2.apply(&mut params_refs, &mut grads_refs);

        let expected_grads = DenseMatrix::new(2, 2, &[0.11, 0.14, 0.19, 0.26]);
        equal_approx(&grads[0], &expected_grads, 1e-6);
    }
}

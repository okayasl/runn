use super::Regularization;
use crate::common::matrix::DenseMatrix;

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
pub struct L1Regularization {
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

pub struct L1 {
    lambda: Option<f32>,
}

impl L1 {
    /// Creates a new builder for L1Regularization
    pub fn new() -> Self {
        Self { lambda: None }
    }

    /// Sets the lambda value for L1 regularization
    pub fn lambda(mut self, lambda: f32) -> Self {
        self.lambda = Some(lambda);
        self
    }

    /// Builds the L1Regularization instance
    pub fn build(self) -> L1Regularization {
        L1Regularization {
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
    fn test_l1_regularization() {
        let mut params = vec![DenseMatrix::new(2, 2, &[1.0, -2.0, 3.0, -4.0])];
        let mut grads = vec![DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1])];
        let l1 = L1::new().lambda(0.01).build();

        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        l1.apply(&mut params_refs, &mut grads_refs);

        let expected_grads = DenseMatrix::new(2, 2, &[0.11, 0.12, 0.13, 0.14]);
        equal_approx(&grads[0], &expected_grads, 1e-6);
    }
}

use super::{Regularization, RegularizationClone};
use crate::common::matrix::DenseMatrix;
use crate::common::random::Randomizer;

use serde::{Deserialize, Serialize};
use typetag;

//// Dropout regularization is a technique that randomly sets a fraction of the weights
/// to zero during training, effectively "dropping out" some neurons. This helps to
/// prevent overfitting by introducing noise and forcing the network to learn more
/// robust features.
///
/// Dropout is typically applied after the activation function in each layer during
/// the forward propagation step.
///
/// The dropoutRate parameter determines the fraction of weights to be set to zero.
/// A higher dropout rate means more weights will be dropped, which can help reduce
/// overfitting but may also make the training process slower and more difficult.
///
/// Dropout is commonly used in deep neural networks with many layers and parameters,
/// as these networks are more prone to overfitting.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct DropoutRegularization {
    dropout_rate: f32,
    randomizer: Randomizer,
}

#[typetag::serde]
impl Regularization for DropoutRegularization {
    fn apply(&self, params: &mut [&mut DenseMatrix], _grads: &mut [&mut DenseMatrix]) {
        for param in params.iter_mut() {
            param.apply_with_indices(|_, _, v| {
                if self.randomizer.float32() < self.dropout_rate {
                    *v = 0.0;
                }
            });
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl RegularizationClone for DropoutRegularization {
    fn clone_box(&self) -> Box<dyn Regularization> {
        Box::new(self.clone())
    }
}

/// Builder for Dropout regularization
/// Dropout regularization is a technique that randomly sets a fraction of the weights
/// to zero during training, effectively "dropping out" some neurons. This helps to
/// prevent overfitting by introducing noise and forcing the network to learn more
/// robust features.
///
/// Dropout is typically applied after the activation function in each layer during
/// the forward propagation step.
///
/// The dropoutRate parameter determines the fraction of weights to be set to zero.
/// A higher dropout rate means more weights will be dropped, which can help reduce
/// overfitting but may also make the training process slower and more difficult.
///
/// Dropout is commonly used in deep neural networks with many layers and parameters,
/// as these networks are more prone to overfitting.
pub struct Dropout {
    dropout_rate: f32,
    seed: Option<u64>,
}

impl Dropout {
    /// Creates a new builder for DropoutRegularization
    pub fn new() -> Self {
        Self {
            dropout_rate: 0.5,
            seed: None,
        }
    }

    /// Set the dropout rate.
    ///
    /// Specifies the fraction of input units to randomly set to zero during training. A higher rate increases regularization strength.
    /// # Parameters
    /// - `dropout_rate`: Fraction of units to drop, in [0.0, 1.0] (e.g., 0.3 for 30%).
    pub fn dropout_rate(mut self, dropout_rate: f32) -> Self {
        self.dropout_rate = dropout_rate;
        self
    }

    /// Set the random seed for reproducibility.
    ///
    /// Fixes the random number generator used for dropout to ensure consistent results across runs.
    /// # Parameters
    /// - `seed`: Random seed value.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn build(self) -> Box<dyn Regularization> {
        Box::new(DropoutRegularization {
            dropout_rate: self.dropout_rate,
            randomizer: Randomizer::new(self.seed),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::matrix::DenseMatrix;
    use crate::util;

    #[test]
    fn test_dropout_regularization() {
        let mut params = vec![DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0])];
        let mut grads = vec![DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1])];
        let dropout = Dropout::new().dropout_rate(0.5).seed(42).build();

        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        dropout.apply(&mut params_refs, &mut grads_refs);

        // Since dropout is random, we can't assert exact values, but we can check if some values are zero
        let flattened = util::flatten(&params[0]);
        assert!(flattened.iter().any(|&v| v == 0.0));
    }
}

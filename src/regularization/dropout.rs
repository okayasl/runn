use super::Regularization;
use crate::common::matrix::DenseMatrix;
use crate::common::random::Randomizer;

use serde::{Deserialize, Serialize};
use typetag;

#[derive(Serialize, Deserialize, Clone)]
pub struct DropoutRegularization {
    dropout_rate: f32,
    // #[serde(skip)] // Skip serialization if needed
    randomizer: Randomizer,
}

impl DropoutRegularization {
    pub(crate) fn new(dropout_rate: f32, randomizer: Randomizer) -> Self {
        Self {
            dropout_rate,
            randomizer,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::matrix::DenseMatrix;
    use crate::common::random::Randomizer;
    use crate::util;

    #[test]
    fn test_dropout_regularization() {
        let mut params = vec![DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0])];
        let mut grads = vec![DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1])];
        let rnd = Randomizer::new(Some(42));
        let dropout = DropoutRegularization::new(0.5, rnd);

        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        dropout.apply(&mut params_refs, &mut grads_refs);

        // Since dropout is random, we can't assert exact values, but we can check if some values are zero
        let flattened = util::flatten(&params[0]);
        assert!(flattened.iter().any(|&v| v == 0.0));
    }
}

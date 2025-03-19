use super::Regularization;
use crate::common::matrix::DenseMatrix;

use serde::{Deserialize, Serialize};
use typetag;

#[derive(Serialize, Deserialize, Clone)]
pub struct L1Regularization {
    lambda: f32,
}

impl L1Regularization {
    pub(crate) fn new(lambda: f32) -> Self {
        Self { lambda }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::matrix::DenseMatrix;
    use crate::util::equal_approx;

    #[test]
    fn test_l1_regularization() {
        let mut params = vec![DenseMatrix::new(2, 2, &[1.0, -2.0, 3.0, -4.0])];
        let mut grads = vec![DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1])];
        let l1 = L1Regularization::new(0.01);

        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        l1.apply(&mut params_refs, &mut grads_refs);

        let expected_grads = DenseMatrix::new(2, 2, &[0.11, 0.12, 0.13, 0.14]);
        equal_approx(&grads[0], &expected_grads, 1e-6);
    }
}

use crate::common::matrix::DenseMatrix;

use serde::{Deserialize, Serialize};
use typetag;

use super::Optimizer;

#[derive(Serialize, Deserialize, Clone)]
pub struct SGDOptimizer {
    learning_rate: f32,
}

impl SGDOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

#[typetag::serde]
impl Optimizer for SGDOptimizer {
    fn initialize(&mut self, _params: &[DenseMatrix]) {
        // No initialization needed for basic SGD.
    }

    fn update(
        &mut self,
        params: &mut [&mut DenseMatrix],
        grads: &[&mut DenseMatrix],
        _epoch: usize,
    ) {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            param.apply_with_indices(|i, j, v| {
                *v -= self.learning_rate * grad.at(i, j);
            });
        }
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
}

pub struct SGDBuilder {
    learning_rate: f32,
}

impl SGDBuilder {
    pub fn new() -> SGDBuilder {
        SGDBuilder {
            learning_rate: 0.01,
        }
    }
}

impl SGDBuilder {
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn build(self) -> SGDOptimizer {
        SGDOptimizer::new(self.learning_rate)
    }
}

#[cfg(test)]
mod tests {
    use crate::util::equal_approx;

    use super::*;

    #[test]
    fn test_sgd_optimizer() {
        let mut params = vec![
            DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            DenseMatrix::new(2, 2, &[5.0, 6.0, 7.0, 8.0]),
        ];

        let mut grads = vec![
            DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]),
            DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]),
        ];

        let mut optimizer = SGDOptimizer::new(0.01);
        optimizer.initialize(&params);
        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        optimizer.update(&mut params_refs, &mut grads_refs, 1);

        let expected_params = vec![
            DenseMatrix::new(2, 2, &[0.999, 1.999, 2.999, 3.999]),
            DenseMatrix::new(2, 2, &[4.999, 5.999, 6.999, 7.999]),
        ];

        for (param, expected) in params.iter().zip(expected_params.iter()) {
            assert!(equal_approx(&param, &expected, 1e-6));
        }
    }
}

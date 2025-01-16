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
    fn initialize(&mut self, _weights: &DenseMatrix, _biases: &DenseMatrix) {
        // No initialization needed for basic SGD
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix,
        biases: &mut DenseMatrix,
        d_weights: &DenseMatrix,
        d_biases: &DenseMatrix,
        _epoch: usize,
    ) {
        weights.apply_with_indices(|i, j, v| {
            *v -= self.learning_rate * d_weights.at(i, j);
        });
        biases.apply_with_indices(|i, j, v| {
            *v -= self.learning_rate * d_biases.at(i, j);
        });
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
    fn test_initialize() {
        let mut optimizer = SGDOptimizer::new(0.01);
        let weights = DenseMatrix::new(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        let biases = DenseMatrix::new(2, 1, &[0.1, 0.2]);
        optimizer.initialize(&weights, &biases);
        // No specific assertions needed for initialization
    }

    #[test]
    fn test_update() {
        let mut optimizer = SGDOptimizer::new(0.01);
        let mut weights = DenseMatrix::new(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let mut biases = DenseMatrix::new(2, 1, &[1.0, 1.0]);
        let d_weights = DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]);
        let d_biases = DenseMatrix::new(2, 1, &[0.1, 0.1]);
        optimizer.initialize(&weights, &biases);

        optimizer.update(&mut weights, &mut biases, &d_weights, &d_biases, 1);
        assert!(weights.at(0, 0) < 1.0);
        assert!(biases.at(0, 0) < 1.0);
    }

    #[test]
    fn test_update_learning_rate() {
        let mut optimizer = SGDOptimizer::new(0.01);
        optimizer.update_learning_rate(0.02);
        assert_eq!(optimizer.learning_rate, 0.02);
    }

    #[test]
    fn test_sgd_optimizer() {
        // Create mock parameter matrices
        let mut weights = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut biases = DenseMatrix::new(2, 1, &[1.0, 2.0]);

        // Create mock gradient matrices
        let d_weights = DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]);
        let d_biases = DenseMatrix::new(2, 1, &[0.1, 0.1]);

        // Create an instance of the SGD optimizer
        let mut optimizer = SGDOptimizer::new(0.01);
        optimizer.initialize(&weights, &biases);

        // Update the parameters using the mock gradients
        optimizer.update(&mut weights, &mut biases, &d_weights, &d_biases, 1);

        let expected_weights = DenseMatrix::new(2, 2, &[0.999, 1.999, 2.999, 3.999]);

        assert!(equal_approx(&weights, &expected_weights, 1e-6));
    }
}

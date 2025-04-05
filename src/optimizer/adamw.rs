use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use typetag;

use super::{Optimizer, OptimizerConfig};

#[derive(Serialize, Deserialize, Clone)]
pub struct AdamWConfig {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
}

impl OptimizerConfig for AdamWConfig {
    fn create_optimizer(self: Box<Self>) -> Box<dyn Optimizer> {
        Box::new(AdamWOptimizer::new(*self))
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AdamWOptimizer {
    config: AdamWConfig,
    moment1_weights: DenseMatrix,
    moment2_weights: DenseMatrix,
    moment1_biases: DenseMatrix,
    moment2_biases: DenseMatrix,
    t: usize,
    m_hat_factor: f32,
    v_hat_factor: f32,
}

impl AdamWOptimizer {
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            moment1_weights: DenseMatrix::zeros(0, 0),
            moment1_biases: DenseMatrix::zeros(0, 0),
            moment2_weights: DenseMatrix::zeros(0, 0),
            moment2_biases: DenseMatrix::zeros(0, 0),
            t: 0,
            m_hat_factor: 1.0,
            v_hat_factor: 1.0,
        }
    }

    fn update_moments(&mut self, d_weights: &DenseMatrix, d_biases: &DenseMatrix) {
        self.moment1_weights.apply_with_indices(|i, j, v| {
            *v = self.config.beta1 * *v + (1.0 - self.config.beta1) * d_weights.at(i, j);
        });

        self.moment2_weights.apply_with_indices(|i, j, v| {
            let g = d_weights.at(i, j);
            *v = self.config.beta2 * *v + (1.0 - self.config.beta2) * g * g;
        });

        self.moment1_biases.apply_with_indices(|i, j, v| {
            *v = self.config.beta1 * *v + (1.0 - self.config.beta1) * d_biases.at(i, j);
        });

        self.moment2_biases.apply_with_indices(|i, j, v| {
            let g = d_biases.at(i, j);
            *v = self.config.beta2 * *v + (1.0 - self.config.beta2) * g * g;
        });
    }

    fn update_parameters(
        &mut self,
        weights: &mut DenseMatrix,
        biases: &mut DenseMatrix,
        step_size: f32,
    ) {
        weights.apply_with_indices(|i, j, v| {
            let m_hat = self.moment1_weights.at(i, j) / self.m_hat_factor;
            let v_hat = self.moment2_weights.at(i, j) / self.v_hat_factor;
            *v -= step_size * m_hat / (v_hat.sqrt() + self.config.epsilon);
            *v -= self.config.weight_decay * *v; // Weight decay
        });

        biases.apply_with_indices(|i, j, v| {
            let m_hat = self.moment1_biases.at(i, j) / self.m_hat_factor;
            let v_hat = self.moment2_biases.at(i, j) / self.v_hat_factor;
            *v -= step_size * m_hat / (v_hat.sqrt() + self.config.epsilon);
        });
    }
}

#[typetag::serde]
impl Optimizer for AdamWOptimizer {
    fn initialize(&mut self, weights: &DenseMatrix, biases: &DenseMatrix) {
        self.moment1_weights = DenseMatrix::zeros(weights.rows(), weights.cols());
        self.moment1_biases = DenseMatrix::zeros(biases.rows(), biases.cols());
        self.moment2_weights = DenseMatrix::zeros(weights.rows(), weights.cols());
        self.moment2_biases = DenseMatrix::zeros(biases.rows(), biases.cols());
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix,
        biases: &mut DenseMatrix,
        d_weights: &DenseMatrix,
        d_biases: &DenseMatrix,
        _epoch: usize,
    ) {
        self.t += 1;
        self.m_hat_factor = 1.0 - self.config.beta1.powi(self.t as i32);
        self.v_hat_factor = 1.0 - self.config.beta2.powi(self.t as i32);
        let step_size = self.config.learning_rate * self.m_hat_factor / self.v_hat_factor.sqrt();

        self.update_moments(&d_weights, &d_biases);
        self.update_parameters(weights, biases, step_size);
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.config.learning_rate = learning_rate;
    }
}

pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
}

impl AdamW {
    pub fn new() -> AdamW {
        AdamW {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl AdamW {
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn build(self) -> Box<AdamWConfig> {
        Box::new(AdamWConfig {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::{self, equal_approx}};

    #[test]
    fn test_initialize() {
        let adamw_config = AdamW::new()
            .learning_rate(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .weight_decay(0.01)
            .build();
        let mut optimizer = AdamWOptimizer::new(adamw_config.as_ref().clone());
        let weights = DenseMatrix::new(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        let biases = DenseMatrix::new(2, 1, &[0.1, 0.2]);
        optimizer.initialize(&weights, &biases);
        assert_eq!(optimizer.moment1_weights.rows(), 2);
        assert_eq!(optimizer.moment1_weights.cols(), 2);
        assert_eq!(optimizer.moment1_biases.rows(), 2);
        assert_eq!(optimizer.moment1_biases.cols(), 1);
    }

    #[test]
    fn test_update() {
        let config = AdamWConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        };
        let mut optimizer = AdamWOptimizer::new(config);
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
        let adamw_config = AdamW::new()
            .learning_rate(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .weight_decay(0.01)
            .build();
        let mut optimizer = AdamWOptimizer::new(adamw_config.as_ref().clone());
        optimizer.update_learning_rate(0.01);
        assert_eq!(optimizer.config.learning_rate, 0.01);
    }

    /// Test the AdamW optimizer.
    ///
    /// This test creates mock parameter and gradient matrices, creates an instance of the AdamW
    /// optimizer, and updates the parameters using the mock gradients. It then computes the
    /// expected values using the AdamW update rule and checks that the updated parameters match
    /// the expected values.

    #[test]
    fn test_adamw_optimizer() {
        // Create mock parameter matrices
        let mut weights = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut biases = DenseMatrix::new(2, 1, &[1.0, 2.0]);

        // Create mock gradient matrices
        let d_weights = DenseMatrix::new(2, 2, &[10.0, 11.0, 12.0, 13.0]);
        let d_biases = DenseMatrix::new(2, 1, &[10.0, 11.0]);

        // Create an instance of the AdamW optimizer
        let adamw_config = AdamW::new()
            .learning_rate(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .weight_decay(0.01)
            .build();
        let mut optimizer = AdamWOptimizer::new(adamw_config.as_ref().clone());
        optimizer.initialize(&weights, &biases);

        // Update the parameters using the mock gradients
        optimizer.update(&mut weights, &mut biases, &d_weights, &d_biases, 1);

        // Manually compute the expected values with the AdamW update rule
        let expected_weights =
            DenseMatrix::new(2, 2, &[0.9868693, 1.9768693, 2.9668694, 3.9568691]);
        let expected_biases = DenseMatrix::new(2, 1, &[0.9968377, 1.9968377]);

        println!("Updated weights: {:?}", util::flatten(&weights));
        println!("Expected weights: {:?}", util::flatten(&expected_weights));
        println!("Updated biases: {:?}", util::flatten(&biases));
        println!("Expected biases: {:?}", util::flatten(&expected_biases));

        assert!(equal_approx(&weights, &expected_weights, 1e-2));
        assert!(equal_approx(&biases, &expected_biases, 1e-2));
    }
}

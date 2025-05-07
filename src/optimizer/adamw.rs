use crate::{common::matrix::DMat, error::NetworkError, LearningRateScheduler};
use serde::{Deserialize, Serialize};
use typetag;

use super::{Optimizer, OptimizerConfig, OptimizerConfigClone};

// AdamW (Adam with Weight Decay) is an optimization algorithm that extends the Adam optimizer
// by incorporating weight decay directly into the parameter update rule. This modification
// improves generalization by penalizing large weights, which helps prevent overfitting.
// Similar to Adam, AdamW maintains two moments:
// - moment1 (m_t): The first moment, which captures the exponentially decaying average of past gradients,
//   acting as a momentum term to accelerate optimization in the direction of historical gradients.
// - moment2 (v_t): The second moment, which tracks the exponentially decaying average of squared gradients,
//   providing an adaptive learning rate that scales based on the magnitude of recent gradients.
// AdamW introduces an additional term:
// - weight_decay: A regularization term that penalizes large weights by scaling them down during updates.
// This modification ensures that weight decay is applied independently of the adaptive learning rate,
// addressing issues with traditional L2 regularization in Adam.
// Update rules:
// momentum = beta1 * momentum + (1 - beta1) * gradient
// accumulated_gradient = beta2 * accumulated_gradient + (1 - beta2) * gradient ** 2
// weight = weight - (learning_rate / sqrt(accumulated_gradient + epsilon)) * momentum
// weight = weight - weight_decay * weight /// Weight decay
// bias = bias - (learning_rate / sqrt(accumulated_gradient + epsilon)) * momentum
#[derive(Serialize, Deserialize, Clone)]
struct AdamWOptimizer {
    config: AdamWConfig,
    moment1_weights: DMat,
    moment2_weights: DMat,
    moment1_biases: DMat,
    moment2_biases: DMat,
    t: usize,
    m_hat_factor: f32,
    v_hat_factor: f32,
}

impl AdamWOptimizer {
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            moment1_weights: DMat::zeros(0, 0),
            moment1_biases: DMat::zeros(0, 0),
            moment2_weights: DMat::zeros(0, 0),
            moment2_biases: DMat::zeros(0, 0),
            t: 0,
            m_hat_factor: 1.0,
            v_hat_factor: 1.0,
        }
    }

    fn update_moments(&mut self, d_weights: &DMat, d_biases: &DMat) {
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

    fn update_parameters(&mut self, weights: &mut DMat, biases: &mut DMat, step_size: f32) {
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
    fn initialize(&mut self, weights: &DMat, biases: &DMat) {
        self.moment1_weights = DMat::zeros(weights.rows(), weights.cols());
        self.moment1_biases = DMat::zeros(biases.rows(), biases.cols());
        self.moment2_weights = DMat::zeros(weights.rows(), weights.cols());
        self.moment2_biases = DMat::zeros(biases.rows(), biases.cols());
    }

    fn update(&mut self, weights: &mut DMat, biases: &mut DMat, d_weights: &DMat, d_biases: &DMat, epoch: usize) {
        if self.config.scheduler.is_some() {
            let scheduler = self.config.scheduler.as_ref().unwrap();
            self.config.learning_rate = scheduler.schedule(epoch, self.config.learning_rate);
        }
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

#[derive(Serialize, Deserialize, Clone)]
struct AdamWConfig {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
}

#[typetag::serde]
impl OptimizerConfig for AdamWConfig {
    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
    fn create_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(AdamWOptimizer::new(self.clone()))
    }
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

/// AdamW is a Builder for AAdamW (Adam with Weight Decay) optimizer which  is an optimization algorithm
/// that extends the Adam optimizer by incorporating weight decay directly into the parameter update rule.
/// This modification improves generalization by penalizing large weights, which helps prevent overfitting.
///
/// Similar to Adam, AdamW maintains two moments:
/// - moment1 (m_t): The first moment, which captures the exponentially decaying average of past gradients,
///   acting as a momentum term to accelerate optimization in the direction of historical gradients.
/// - moment2 (v_t): The second moment, which tracks the exponentially decaying average of squared gradients,
///   providing an adaptive learning rate that scales based on the magnitude of recent gradients.
///
/// AdamW introduces an additional term:
/// - weight_decay: A regularization term that penalizes large weights by scaling them down during updates.
///
/// This modification ensures that weight decay is applied independently of the adaptive learning rate,
/// addressing issues with traditional L2 regularization in Adam.
///
/// Update rules:
/// momentum = beta1 * momentum + (1 - beta1) * gradient
/// accumulated_gradient = beta2 * accumulated_gradient + (1 - beta2) * gradient ** 2
/// weight = weight - (learning_rate / sqrt(accumulated_gradient + epsilon)) * momentum
/// weight = weight - weight_decay * weight /// Weight decay
/// bias = bias - (learning_rate / sqrt(accumulated_gradient + epsilon)) * momentum
pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    scheduler: Option<Result<Box<dyn LearningRateScheduler>, NetworkError>>,
}

impl AdamW {
    /// Creates a new AdamW optimizer builder with default parameters.
    /// Default values:
    /// - learning_rate: 0.01
    /// - beta1: 0.9
    /// - beta2: 0.999
    /// - epsilon: f32::EPSILON
    /// - weight_decay: 0.01
    /// - scheduler: None
    pub fn new() -> AdamW {
        AdamW {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: f32::EPSILON,
            weight_decay: 0.01,
            scheduler: None,
        }
    }
}

impl AdamW {
    /// Set the learning rate.
    ///
    /// Controls the step size for parameter updates. Smaller values lead to slower but more stable convergence.
    /// # Parameters
    /// - `learning_rate`: The learning rate value (e.g., 0.001).
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the first moment decay rate (beta1).
    ///
    /// Controls the exponential decay rate for the moving average of gradients. Typically close to 1.0 (e.g., 0.9).
    /// # Parameters
    /// - `beta1`: First moment decay rate, in [0.0, 1.0].
    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set the second moment decay rate (beta2).
    ///
    /// Controls the exponential decay rate for the moving average of squared gradients. Typically very close to 1.0 (e.g., 0.999).
    /// # Parameters
    /// - `beta2`: Second moment decay rate, in [0.0, 1.0].
    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set the epsilon value for numerical stability.
    ///
    /// Prevents division by zero in the update rule. Typically a very small value (e.g., 1e-8).
    /// # Parameters
    /// - `epsilon`: Small constant for numerical stability.
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the weight decay strength.
    ///
    /// Controls the L2 regularization penalty applied to weights, encouraging smaller weights to prevent overfitting.
    /// # Parameters
    /// - `weight_decay`: Weight decay coefficient (e.g., 0.01).
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set a learning rate scheduler.
    ///
    /// Optionally applies a scheduler to adjust the learning rate during training (e.g., exponential, step).
    /// # Parameters
    /// - `scheduler`: Learning rate scheduler to use.
    pub fn scheduler(mut self, scheduler: Result<Box<dyn LearningRateScheduler>, NetworkError>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.learning_rate <= 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Learning rate for AdamW must be greater than 0.0, but was {}",
                self.learning_rate
            )));
        }
        if self.beta1 <= 0.0 || self.beta1 >= 1.0 {
            return Err(NetworkError::ConfigError(format!(
                "Beta1 for AdamW must be in the range (0, 1), but was {}",
                self.beta1
            )));
        }
        if self.beta2 <= 0.0 || self.beta2 >= 1.0 {
            return Err(NetworkError::ConfigError(format!(
                "Beta2 for AdamW must be in the range (0, 1), but was {}",
                self.beta2
            )));
        }
        if self.epsilon <= 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Epsilon for AdamW must be greater than 0.0, but was {}",
                self.epsilon
            )));
        }
        if self.weight_decay < 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Weight decay for AdamW must be greater than or equal to 0.0, but was {}",
                self.weight_decay
            )));
        }
        if let Some(ref scheduler) = self.scheduler {
            scheduler.as_ref().map_err(|e| e.clone())?;
        }
        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn OptimizerConfig>, NetworkError> {
        self.validate()?;

        Ok(Box::new(AdamWConfig {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
            scheduler: self.scheduler.map(|s| s.unwrap()),
        }))
    }
}

impl OptimizerConfigClone for AdamWConfig {
    fn clone_box(&self) -> Box<dyn OptimizerConfig> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common::matrix::DMat,
        util::{self, equal_approx},
    };

    #[test]
    fn test_initialize() {
        let adamw_config = AdamWConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            scheduler: None,
        };
        let mut optimizer = AdamWOptimizer::new(adamw_config);
        let weights = DMat::new(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        let biases = DMat::new(2, 1, &[0.1, 0.2]);
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
            scheduler: None,
        };
        let mut optimizer = AdamWOptimizer::new(config);
        let mut weights = DMat::new(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let mut biases = DMat::new(2, 1, &[1.0, 1.0]);
        let d_weights = DMat::new(2, 2, &[0.1, 0.1, 0.1, 0.1]);
        let d_biases = DMat::new(2, 1, &[0.1, 0.1]);
        optimizer.initialize(&weights, &biases);

        optimizer.update(&mut weights, &mut biases, &d_weights, &d_biases, 1);
        assert!(weights.at(0, 0) < 1.0);
        assert!(biases.at(0, 0) < 1.0);
    }

    #[test]
    fn test_update_learning_rate() {
        let adamw_config = AdamWConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            scheduler: None,
        };
        let mut optimizer = AdamWOptimizer::new(adamw_config);
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
        let mut weights = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut biases = DMat::new(2, 1, &[1.0, 2.0]);

        // Create mock gradient matrices
        let d_weights = DMat::new(2, 2, &[10.0, 11.0, 12.0, 13.0]);
        let d_biases = DMat::new(2, 1, &[10.0, 11.0]);

        // Create an instance of the AdamW optimizer
        let adamw_config = AdamWConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            scheduler: None,
        };
        let mut optimizer = AdamWOptimizer::new(adamw_config);
        optimizer.initialize(&weights, &biases);

        // Update the parameters using the mock gradients
        optimizer.update(&mut weights, &mut biases, &d_weights, &d_biases, 1);

        // Manually compute the expected values with the AdamW update rule
        let expected_weights = DMat::new(2, 2, &[0.9868693, 1.9768693, 2.9668694, 3.9568691]);
        let expected_biases = DMat::new(2, 1, &[0.9968377, 1.9968377]);

        println!("Updated weights: {:?}", util::flatten(&weights));
        println!("Expected weights: {:?}", util::flatten(&expected_weights));
        println!("Updated biases: {:?}", util::flatten(&biases));
        println!("Expected biases: {:?}", util::flatten(&expected_biases));

        assert!(equal_approx(&weights, &expected_weights, 1e-2));
        assert!(equal_approx(&biases, &expected_biases, 1e-2));
    }
}

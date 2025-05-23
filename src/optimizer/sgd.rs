use crate::{common::matrix::DMat, error::NetworkError, LearningRateScheduler};

use serde::{Deserialize, Serialize};
use typetag;

use super::{Optimizer, OptimizerConfig, OptimizerConfigClone};

// Stochastic Gradient Descent (SGD) optimizer is a simple and popular optimization algorithm
// that updates model parameters in the direction of the negative gradient of the loss function.
// It is widely used due to its simplicity and effectiveness, especially when the dataset is large.
// However, it can be slow to converge, especially for complex models or noisy data.
// weight = weight - learning_rate * gradient_of_weight
// bias = bias - learning_rate * gradient_of_bias
#[derive(Serialize, Deserialize, Clone)]
struct SGDOptimizer {
    config: SGDConfig,
}

impl SGDOptimizer {
    pub fn new(config: SGDConfig) -> Self {
        Self { config }
    }
}

#[typetag::serde]
impl Optimizer for SGDOptimizer {
    fn initialize(&mut self, _weights: &DMat, _biases: &DMat) {
        // No initialization needed for basic SGD
    }

    fn update(&mut self, weights: &mut DMat, biases: &mut DMat, d_weights: &DMat, d_biases: &DMat, epoch: usize) {
        if self.config.scheduler.is_some() {
            let scheduler = self.config.scheduler.as_ref().unwrap();
            self.config.learning_rate = scheduler.schedule(epoch, self.config.learning_rate);
        }
        weights.apply_with_indices(|i, j, v| {
            *v -= self.config.learning_rate * d_weights.at(i, j);
        });
        biases.apply_with_indices(|i, j, v| {
            *v -= self.config.learning_rate * d_biases.at(i, j);
        });
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.config.learning_rate = learning_rate;
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct SGDConfig {
    learning_rate: f32,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
}

#[typetag::serde]
impl OptimizerConfig for SGDConfig {
    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
    fn create_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(SGDOptimizer::new(self.clone()))
    }
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

/// SGD is a builder for Stochastic Gradient Descent (SGD) optimizer which is a simple and
/// popular optimization algorithm that updates model parameters in the direction of the
/// negative gradient of the loss function.
///
/// It is widely used due to its simplicity and effectiveness, especially when the dataset is large.
/// However, it can be slow to converge, especially for complex models or noisy data.
///
/// weight = weight - learning_rate * gradient_of_weight
/// bias = bias - learning_rate * gradient_of_bias
pub struct SGD {
    learning_rate: f32,
    scheduler: Option<Result<Box<dyn LearningRateScheduler>, NetworkError>>,
}

impl SGD {
    // Creates a new SGD optimizer builder
    // Default values:
    // - learning_rate: 0.01
    // - scheduler: None
    fn new() -> Self {
        Self {
            learning_rate: 0.01,
            scheduler: None,
        }
    }

    /// Set the learning rate.
    ///
    /// Controls the step size for parameter updates. Smaller values lead to slower but more stable convergence.
    /// # Parameters
    /// - `learning_rate`: The learning rate value (e.g., 0.01).
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
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
                "Learning rate for SGD must be greater than 0.0, but was {}",
                self.learning_rate
            )));
        }
        if let Some(ref scheduler) = self.scheduler {
            scheduler.as_ref().map_err(|e| e.clone())?;
        }
        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn OptimizerConfig>, NetworkError> {
        self.validate()?;
        Ok(Box::new(SGDConfig {
            learning_rate: self.learning_rate,
            scheduler: self.scheduler.map(|s| s.unwrap()),
        }))
    }
}

impl Default for SGD {
    /// Creates a new SGD optimizer builder with default values.
    /// Default values:
    /// - `learning_rate`: 0.01
    /// - `scheduler`: None
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizerConfigClone for SGDConfig {
    fn clone_box(&self) -> Box<dyn OptimizerConfig> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::util::equal_approx;

    use super::*;

    #[test]
    fn test_initialize() {
        let config = SGDConfig {
            learning_rate: 0.01,
            scheduler: None,
        };
        let mut optimizer = SGDOptimizer::new(config);
        let weights = DMat::new(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        let biases = DMat::new(2, 1, &[0.1, 0.2]);
        optimizer.initialize(&weights, &biases);
        // No specific assertions needed for initialization
    }

    #[test]
    fn test_update() {
        let config = SGDConfig {
            learning_rate: 0.01,
            scheduler: None,
        };
        let mut optimizer = SGDOptimizer::new(config);
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
        let config = SGDConfig {
            learning_rate: 0.01,
            scheduler: None,
        };
        let mut optimizer = SGDOptimizer::new(config);
        optimizer.update_learning_rate(0.02);
        assert_eq!(optimizer.config.learning_rate, 0.02);
    }

    #[test]
    fn test_sgd_optimizer() {
        let config = SGDConfig {
            learning_rate: 0.01,
            scheduler: None,
        };
        // Create mock parameter matrices
        let mut weights = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut biases = DMat::new(2, 1, &[1.0, 2.0]);

        // Create mock gradient matrices
        let d_weights = DMat::new(2, 2, &[0.1, 0.1, 0.1, 0.1]);
        let d_biases = DMat::new(2, 1, &[0.1, 0.1]);

        // Create an instance of the SGD optimizer
        let mut optimizer = SGDOptimizer::new(config);
        optimizer.initialize(&weights, &biases);

        // Update the parameters using the mock gradients
        optimizer.update(&mut weights, &mut biases, &d_weights, &d_biases, 1);

        let expected_weights = DMat::new(2, 2, &[0.999, 1.999, 2.999, 3.999]);

        assert!(equal_approx(&weights, &expected_weights, 1e-6));
    }

    #[test]
    fn test_sgd_builder() {
        let optimizer = SGD::default().learning_rate(0.01).build().unwrap();
        assert_eq!(optimizer.learning_rate(), 0.01);
    }
    #[test]
    fn test_sgd_builder_invalid() {
        let result = SGD::default().learning_rate(-0.01).build();
        assert!(result.is_err());
        if let Err(err) = result {
            assert_eq!(
                err.to_string(),
                "Configuration error: Learning rate for SGD must be greater than 0.0, but was -0.01"
            );
        }
    }
}

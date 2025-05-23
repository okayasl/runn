use super::{Optimizer, OptimizerConfig, OptimizerConfigClone};
use crate::{common::matrix::DMat, error::NetworkError, LearningRateScheduler};
use serde::{Deserialize, Serialize};
use typetag;

/// RmsPropOptimizer is an implementation of the RMSProp(Root Mean Squared Propagation) optimization algorithm.
/// RMSProp is an adaptive learning rate optimization algorithm that divides the
/// learning rate by a running average of the magnitudes of recent gradients.
/// accumulated_gradient = (decay_rate * accumulated_gradient) + ((1 - decay_rate) * gradient * gradien)
/// weight = weight - (learning_rate * gradient) / sqrt(accumulated_gradient + epsilon)
/// bias = bias - (learning_rate * gradient) / sqrt(accumulated_gradient + epsilon)
#[derive(Serialize, Deserialize, Clone)]
struct RMSPropOptimizer {
    config: RMSPropConfig,
    accumulated_squared_grad_weights: DMat,
    accumulated_squared_grad_biases: DMat,
}

impl RMSPropOptimizer {
    pub fn new(config: RMSPropConfig) -> Self {
        Self {
            config,
            accumulated_squared_grad_weights: DMat::zeros(0, 0),
            accumulated_squared_grad_biases: DMat::zeros(0, 0),
        }
    }
}

#[typetag::serde]
impl Optimizer for RMSPropOptimizer {
    fn initialize(&mut self, weights: &DMat, biases: &DMat) {
        self.accumulated_squared_grad_weights = DMat::zeros(weights.rows(), weights.cols());
        self.accumulated_squared_grad_biases = DMat::zeros(biases.rows(), biases.cols());
    }

    fn update(&mut self, weights: &mut DMat, biases: &mut DMat, d_weights: &DMat, d_biases: &DMat, epoch: usize) {
        if self.config.scheduler.is_some() {
            let scheduler = self.config.scheduler.as_ref().unwrap();
            self.config.learning_rate = scheduler.schedule(epoch, self.config.learning_rate);
        }
        weights.apply_with_indices(|r, c, v| {
            let grad = d_weights.at(r, c);
            let ema_grad = &mut self.accumulated_squared_grad_weights;
            let previous_ema_grad = ema_grad.at(r, c);
            let new_ema_grad =
                self.config.decay_rate * previous_ema_grad + (1.0 - self.config.decay_rate) * grad * grad;
            ema_grad.set(r, c, new_ema_grad);
            *v -= self.config.learning_rate * grad / (new_ema_grad.sqrt() + self.config.epsilon);
        });

        biases.apply_with_indices(|r, c, v| {
            let grad = d_biases.at(r, c);
            let ema_grad = &mut self.accumulated_squared_grad_biases;
            let previous_ema_grad = ema_grad.at(r, c);
            let new_ema_grad =
                self.config.decay_rate * previous_ema_grad + (1.0 - self.config.decay_rate) * grad * grad;
            ema_grad.set(r, c, new_ema_grad);
            *v -= self.config.learning_rate * grad / (new_ema_grad.sqrt() + self.config.epsilon);
        });
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.config.learning_rate = learning_rate;
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct RMSPropConfig {
    learning_rate: f32,
    decay_rate: f32,
    epsilon: f32,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
}

#[typetag::serde]
impl OptimizerConfig for RMSPropConfig {
    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
    fn create_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(RMSPropOptimizer::new(self.clone()))
    }
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

/// builder for RmsProp optimizer
/// RmsPropOptimizer is an implementation of the RMSProp(Root Mean Squared Propagation) optimization algorithm.
/// RMSProp is an adaptive learning rate optimization algorithm that divides the
/// learning rate by a running average of the magnitudes of recent gradients.
/// accumulated_gradient = (decay_rate * accumulated_gradient) + ((1 - decay_rate) * gradient * gradien)
/// weight = weight - (learning_rate * gradient) / sqrt(accumulated_gradient + epsilon)
/// bias = bias - (learning_rate * gradient) / sqrt(accumulated_gradient + epsilon)
pub struct RMSProp {
    learning_rate: f32,
    decay_rate: f32,
    epsilon: f32,
    scheduler: Option<Result<Box<dyn LearningRateScheduler>, NetworkError>>,
}

impl RMSProp {
    // Creates a new builder for RMSProp optimizer
    // Default values:
    // - learning_rate: 0.001
    // - decay_rate: 0.9
    // - epsilon: f32::EPSILON
    // - scheduler: None
    fn new() -> Self {
        Self {
            learning_rate: 0.001,
            decay_rate: 0.9,
            epsilon: f32::EPSILON,
            scheduler: None,
        }
    }

    /// Set the learning rate.
    ///
    /// Controls the step size for parameter updates. Smaller values lead to slower but more stable convergence.
    /// # Parameters
    /// - `lr`: The learning rate value (e.g., 0.001).
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the decay rate for squared gradients.
    ///
    /// Controls the exponential decay rate for the moving average of squared gradients. Typically close to 1.0 (e.g., 0.9).
    /// # Parameters
    /// - `rate`: Decay rate, in [0.0, 1.0].
    pub fn decay_rate(mut self, rate: f32) -> Self {
        self.decay_rate = rate;
        self
    }

    /// Set the epsilon value for numerical stability.
    ///
    /// Prevents division by zero in the update rule. Typically a very small value (e.g., 1e-8).
    /// # Parameters
    /// - `eps`: Small constant for numerical stability.
    pub fn epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
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
                "Learning rate for RMSProp must be greater than 0.0, but was {}",
                self.learning_rate
            )));
        }
        if self.decay_rate < 0.0 || self.decay_rate > 1.0 {
            return Err(NetworkError::ConfigError(format!(
                "Decay rate for RMSProp must be in [0.0, 1.0], but was {}",
                self.decay_rate
            )));
        }
        if self.epsilon <= 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Epsilon for RMSProp must be greater than 0.0, but was {}",
                self.epsilon
            )));
        }
        if let Some(ref scheduler) = self.scheduler {
            scheduler.as_ref().map_err(|e| e.clone())?;
        }
        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn OptimizerConfig>, NetworkError> {
        self.validate()?;
        Ok(Box::new(RMSPropConfig {
            learning_rate: self.learning_rate,
            decay_rate: self.decay_rate,
            epsilon: self.epsilon,
            scheduler: self.scheduler.map(|s| s.unwrap()),
        }))
    }
}

impl Default for RMSProp {
    /// Creates a new RMSProp builder with default values.
    /// Default values:
    /// - `learning_rate`: 0.001
    /// - `decay_rate`: 0.9
    /// - `epsilon`: f32::EPSILON
    /// - `scheduler`: None
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizerConfigClone for RMSPropConfig {
    fn clone_box(&self) -> Box<dyn OptimizerConfig> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::{exponential::Exponential, util::equal_approx};

    use super::*;

    #[test]
    fn test_initialize() {
        let config = RMSPropConfig {
            learning_rate: 0.001,
            decay_rate: 0.9,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = RMSPropOptimizer::new(config);
        let weights = DMat::new(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        let biases = DMat::new(2, 1, &[0.1, 0.2]);
        optimizer.initialize(&weights, &biases);
        assert_eq!(optimizer.accumulated_squared_grad_weights.rows(), 2);
        assert_eq!(optimizer.accumulated_squared_grad_weights.cols(), 2);
        assert_eq!(optimizer.accumulated_squared_grad_biases.rows(), 2);
        assert_eq!(optimizer.accumulated_squared_grad_biases.cols(), 1);
    }

    #[test]
    fn test_update() {
        let config = RMSPropConfig {
            learning_rate: 0.1,
            decay_rate: 0.9,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = RMSPropOptimizer::new(config);
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
        let config = RMSPropConfig {
            learning_rate: 0.001,
            decay_rate: 0.9,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = RMSPropOptimizer::new(config);
        optimizer.update_learning_rate(0.01);
        assert_eq!(optimizer.config.learning_rate, 0.01);
    }

    #[test]
    fn test_rmsprop_optimizer() {
        // Create mock parameter matrices
        let mut weights = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut biases = DMat::new(2, 1, &[1.0, 2.0]);

        // Create mock gradient matrices
        let d_weights = DMat::new(2, 2, &[0.1, 0.1, 0.1, 0.1]);
        let d_biases = DMat::new(2, 1, &[0.1, 0.1]);

        // Create an instance of the RMSProp optimizer
        let config = RMSPropConfig {
            learning_rate: 0.01,
            decay_rate: 0.9,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = RMSPropOptimizer::new(config);
        optimizer.initialize(&weights, &biases);

        // Update the parameters using the mock gradients
        optimizer.update(&mut weights, &mut biases, &d_weights, &d_biases, 1);

        let expected_weights = DMat::new(2, 2, &[0.96837723, 1.9683772, 2.9683774, 3.9683774]);

        assert!(equal_approx(&weights, &expected_weights, 1e-6));
    }

    #[test]
    fn test_rmsprop_builder() {
        let optimizer = RMSProp::default()
            .learning_rate(0.01)
            .decay_rate(0.9)
            .epsilon(1e-8)
            .build()
            .unwrap();

        assert_eq!(optimizer.learning_rate(), 0.01);
    }

    #[test]
    fn test_rmsprop_builder_invalid() {
        let optimizer = RMSProp::new().learning_rate(0.0).decay_rate(1.5).epsilon(-1e-8).build();

        assert!(optimizer.is_err());
        if let Err(err) = optimizer {
            assert_eq!(
                err.to_string(),
                "Configuration error: Learning rate for RMSProp must be greater than 0.0, but was 0"
            );
        }
    }

    #[test]
    fn test_rmsprop_builder_with_scheduler() {
        let scheduler = Exponential::default().build().unwrap();
        let optimizer = RMSProp::default()
            .learning_rate(0.01)
            .decay_rate(0.9)
            .epsilon(1e-8)
            .scheduler(Ok(Box::new(scheduler)))
            .build()
            .unwrap();

        assert_eq!(optimizer.learning_rate(), 0.01);
    }

    #[test]
    fn test_rmsprop_builder_with_invalid_scheduler() {
        let scheduler = Exponential::default().initial_lr(0.0).build();
        let optimizer = RMSProp::default()
            .learning_rate(0.01)
            .decay_rate(0.9)
            .epsilon(1e-8)
            .scheduler(scheduler)
            .build();

        assert!(optimizer.is_err());
        if let Err(err) = optimizer {
            assert_eq!(
                err.to_string(),
                "Configuration error: Initial learning rate for Exponential must be greater than 0.0, but was 0"
            );
        }
    }

    #[test]
    fn test_rmsprop_builder_with_invalid_decay_rate() {
        let optimizer = RMSProp::default()
            .learning_rate(0.01)
            .decay_rate(1.5)
            .epsilon(1e-8)
            .build();

        assert!(optimizer.is_err());
        if let Err(err) = optimizer {
            assert_eq!(
                err.to_string(),
                "Configuration error: Decay rate for RMSProp must be in [0.0, 1.0], but was 1.5"
            );
        }
    }
    #[test]
    fn test_rmsprop_builder_with_invalid_epsilon() {
        let optimizer = RMSProp::default()
            .learning_rate(0.01)
            .decay_rate(0.9)
            .epsilon(-1.0)
            .build();

        assert!(optimizer.is_err());
        if let Err(err) = optimizer {
            assert_eq!(
                err.to_string(),
                "Configuration error: Epsilon for RMSProp must be greater than 0.0, but was -1"
            );
        }
    }
}

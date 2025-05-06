use super::{Optimizer, OptimizerConfig, OptimizerConfigClone};
use crate::{common::matrix::DenseMatrix, error::NetworkError, LearningRateScheduler};
use serde::{Deserialize, Serialize};
use typetag;

/// MomentumOptimizer is an implementation of the Momentum optimization algorithm.
/// Momentum is an optimization algorithm that accelerates gradients in the direction
/// of previous gradients, helping to overcome local minima and speed up convergence.
/// velocity = momentum * velocity + learning_rate * gradient
/// weight = weight - velocity
/// bias = bias - velocity
#[derive(Serialize, Deserialize, Clone)]
struct MomentumOptimizer {
    config: MomentumConfig,
    velocity_weights: DenseMatrix,
    velocity_biases: DenseMatrix,
}

impl MomentumOptimizer {
    pub fn new(config: MomentumConfig) -> Self {
        Self {
            config,
            velocity_weights: DenseMatrix::zeros(0, 0),
            velocity_biases: DenseMatrix::zeros(0, 0),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct MomentumConfig {
    learning_rate: f32,
    momentum: f32,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
}

#[typetag::serde]
impl OptimizerConfig for MomentumConfig {
    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
    fn create_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(MomentumOptimizer::new(self.clone()))
    }
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

#[typetag::serde]
impl Optimizer for MomentumOptimizer {
    fn initialize(&mut self, weights: &DenseMatrix, biases: &DenseMatrix) {
        self.velocity_weights = DenseMatrix::zeros(weights.rows(), weights.cols());
        self.velocity_biases = DenseMatrix::zeros(biases.rows(), biases.cols());
    }

    fn update(
        &mut self, weights: &mut DenseMatrix, biases: &mut DenseMatrix, d_weights: &DenseMatrix,
        d_biases: &DenseMatrix, epoch: usize,
    ) {
        if self.config.scheduler.is_some() {
            let scheduler = self.config.scheduler.as_ref().unwrap();
            self.config.learning_rate = scheduler.schedule(epoch, self.config.learning_rate);
        }
        weights.apply_with_indices(|r, c, v| {
            let velocity = &mut self.velocity_weights;
            let previous_velocity = velocity.at(r, c);
            // Calculate the new velocity using the Momentum update rule
            // velocity = (momentum * previous_velocity) + (learning_rate * gradient)
            let new_velocity =
                self.config.momentum * previous_velocity + self.config.learning_rate * d_weights.at(r, c);
            velocity.set(r, c, new_velocity);
            *v -= new_velocity
        });
        biases.apply_with_indices(|r, c, v| {
            let velocity = &mut self.velocity_biases;
            let previous_velocity = velocity.at(r, c);
            // Calculate the new velocity using the Momentum update rule
            // velocity = (momentum * previous_velocity) + (learning_rate * gradient)
            let new_velocity = self.config.momentum * previous_velocity + self.config.learning_rate * d_biases.at(r, c);
            velocity.set(r, c, new_velocity);
            *v -= new_velocity
        });
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.config.learning_rate = learning_rate;
    }
}

///Builder for Momentum optimizer
/// MomentumOptimizer is an implementation of the Momentum optimization algorithm.
/// Momentum is an optimization algorithm that accelerates gradients in the direction
/// of previous gradients, helping to overcome local minima and speed up convergence.
/// velocity = momentum * velocity + learning_rate * gradient
/// weight = weight - velocity
/// bias = bias - velocity
pub struct Momentum {
    learning_rate: f32,
    momentum: f32,
    scheduler: Option<Result<Box<dyn LearningRateScheduler>, NetworkError>>,
}

impl Momentum {
    /// Creates a new builder for Momentum optimizer.
    /// Default values:
    /// - learning_rate: 0.01
    /// - momentum: 0.9
    /// - scheduler: None
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            scheduler: None,
        }
    }

    /// Set the learning rate.
    ///
    /// Controls the step size for parameter updates. Smaller values lead to slower but more stable convergence.
    /// # Parameters
    /// - `lr`: The learning rate value (e.g., 0.01).
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the momentum factor.
    ///
    /// Determines how much past gradients influence the current update. Higher values increase the effect of momentum.
    /// # Parameters
    /// - `momentum`: Momentum factor, typically in [0.0, 1.0] (e.g., 0.9).
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
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
                "Learning rate for Momentum must be greater than 0.0, but was {}",
                self.learning_rate
            )));
        }
        if self.momentum < 0.0 || self.momentum > 1.0 {
            return Err(NetworkError::ConfigError(format!(
                "Momentum for Momentum must be in [0.0, 1.0], but was {}",
                self.momentum
            )));
        }
        if let Some(ref scheduler) = self.scheduler {
            scheduler.as_ref().map_err(|e| e.clone())?;
        }
        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn OptimizerConfig>, NetworkError> {
        self.validate()?;
        Ok(Box::new(MomentumConfig {
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            scheduler: self.scheduler.map(|s| s.unwrap()),
        }))
    }
}

impl OptimizerConfigClone for MomentumConfig {
    fn clone_box(&self) -> Box<dyn OptimizerConfig> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::{step::Step, util::equal_approx};

    use super::*;

    #[test]
    fn test_initialize() {
        let config = MomentumConfig {
            learning_rate: 0.01,
            momentum: 0.9,
            scheduler: None,
        };
        let mut optimizer = MomentumOptimizer::new(config);
        let weights = DenseMatrix::new(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        let biases = DenseMatrix::new(2, 1, &[0.1, 0.2]);
        optimizer.initialize(&weights, &biases);
        assert_eq!(optimizer.velocity_weights.rows(), 2);
        assert_eq!(optimizer.velocity_weights.cols(), 2);
        assert_eq!(optimizer.velocity_biases.rows(), 2);
        assert_eq!(optimizer.velocity_biases.cols(), 1);
    }

    #[test]
    fn test_update() {
        let config = MomentumConfig {
            learning_rate: 0.1,
            momentum: 0.9,
            scheduler: None,
        };
        let mut optimizer = MomentumOptimizer::new(config);
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
        let config = MomentumConfig {
            learning_rate: 0.01,
            momentum: 0.9,
            scheduler: None,
        };
        let mut optimizer = MomentumOptimizer::new(config);
        optimizer.update_learning_rate(0.02);
        assert_eq!(optimizer.config.learning_rate, 0.02);
    }

    #[test]
    fn test_momentum_optimizer() {
        // Create mock parameter matrices
        let mut weights = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut biases = DenseMatrix::new(2, 1, &[1.0, 2.0]);

        // Create mock gradient matrices
        let d_weights = DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]);
        let d_biases = DenseMatrix::new(2, 1, &[0.1, 0.1]);

        // Create an instance of the Momentum optimizer
        let config = MomentumConfig {
            learning_rate: 0.1,
            momentum: 0.9,
            scheduler: None,
        };
        let mut optimizer = MomentumOptimizer::new(config);
        optimizer.initialize(&weights, &biases);

        // Update the parameters using the mock gradients
        optimizer.update(&mut weights, &mut biases, &d_weights, &d_biases, 1);

        let expected_weights = DenseMatrix::new(2, 2, &[0.99, 1.99, 2.99, 3.99]);

        assert!(equal_approx(&weights, &expected_weights, 1e-3));
    }

    #[test]
    fn test_momentum_builder() {
        let optimizer = Momentum::new().learning_rate(0.01).momentum(0.9).build().unwrap();
        assert_eq!(optimizer.learning_rate(), 0.01);
    }

    #[test]
    fn test_momentum_builder_invalid_learning_rate() {
        let result = Momentum::new().learning_rate(-0.01).build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(
                e.to_string(),
                "Configuration error: Learning rate for Momentum must be greater than 0.0, but was -0.01"
            );
        }
    }

    #[test]
    fn test_momentum_builder_invalid_scheduler() {
        let result = Momentum::new()
            .momentum(0.5)
            .scheduler(Step::new().decay_rate(5.0).build())
            .build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(
                e.to_string(),
                "Configuration error: Decay rate for Step must be in the range (0, 1), but was 5"
            );
        }
    }
}

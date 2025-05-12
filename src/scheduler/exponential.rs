use serde::{Deserialize, Serialize};

use crate::error::NetworkError;

use super::{LearningRateScheduler, LearningRateSchedulerClone};

// ExponentialLRScheduler implements an exponential decay learning rate scheduler which continuously
// decreases the learning rate after each epoch. This scheduler is useful for smoothing the training process
// by gradually decreasing the step size of updates to the model's parameters,
// allowing for more precise convergence as training progresses. The rate of decay per epoch is
// controlled by `decay_rate` raised to the power of the product of the epoch number and `decay_factor`.
#[derive(Serialize, Deserialize, Clone)]
struct ExponentialLRScheduler {
    initial_lr: f32,   // Initial learning rate
    decay_rate: f32,   // Base for the exponential decay
    decay_factor: f32, // Decay factor
}

impl ExponentialLRScheduler {
    // Creates a new `ExponentialLRScheduler` with the given initial learning rate, decay rate, and decay factor.
    fn new(initial_lr: f32, decay_rate: f32, decay_factor: f32) -> Self {
        Self {
            initial_lr,
            decay_rate,
            decay_factor,
        }
    }
}

#[typetag::serde]
impl LearningRateScheduler for ExponentialLRScheduler {
    // Computes the new learning rate by applying exponential decay based on the epoch number.
    // The decay factor controls how quickly the learning rate decreases;
    // a smaller decay factor results in slower decay.
    fn schedule(&self, epoch: usize, _current_learning_rate: f32) -> f32 {
        self.initial_lr * self.decay_rate.powf(epoch as f32 * self.decay_factor)
    }
}

impl LearningRateSchedulerClone for ExponentialLRScheduler {
    fn clone_box(&self) -> Box<dyn LearningRateScheduler> {
        Box::new(self.clone())
    }
}

/// Exponential is a builder for Exponential Learning rate scheduler which allows for more flexible and readable construction.
///
/// It implements an exponential decay learning rate scheduler which continuously
/// decreases the learning rate after each epoch. This scheduler is useful for smoothing the training process
/// by gradually decreasing the step size of updates to the model's parameters,
/// allowing for more precise convergence as training progresses. The rate of decay per epoch is
/// controlled by `decay_rate` raised to the power of the product of the epoch number and `decay_factor`.
pub struct Exponential {
    initial_lr: f32,
    decay_rate: f32,
    decay_factor: f32,
}

impl Exponential {
    // Creates a new `ExponentialLRSchedulerBuilder`.
    // Default values:
    // - `initial_lr`: 0.01
    // - `decay_rate`: 0.95
    // - `decay_factor`: 0.1
    fn new() -> Self {
        Self {
            initial_lr: 0.01,
            decay_rate: 0.95,
            decay_factor: 0.1,
        }
    }

    /// Sets the initial learning rate.
    /// This is the starting learning rate before any decay is applied.
    /// # Parameters
    /// - `initial_lr`: The initial learning rate.
    pub fn initial_lr(mut self, initial_lr: f32) -> Self {
        self.initial_lr = initial_lr;
        self
    }

    /// Sets the decay rate.
    /// This is the base for the exponential decay.
    /// # Parameters
    /// - `decay_rate`: The decay rate.
    pub fn decay_rate(mut self, decay_rate: f32) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    /// Sets the decay factor.
    /// This controls how quickly the learning rate decreases.
    /// A smaller decay factor results in slower decay.
    /// # Parameters
    /// - `decay_factor`: The decay factor.
    pub fn decay_factor(mut self, decay_factor: f32) -> Self {
        self.decay_factor = decay_factor;
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.initial_lr <= 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Initial learning rate for Exponential must be greater than 0.0, but was {}",
                self.initial_lr
            )));
        }
        if self.decay_rate <= 0.0 || self.decay_rate >= 1.0 {
            return Err(NetworkError::ConfigError(format!(
                "Decay rate for Exponential must be in the range (0, 1), but was {}",
                self.decay_rate
            )));
        }
        if self.decay_factor <= 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Decay factor for Exponential  must be greater than 0.0, but was {}",
                self.decay_factor
            )));
        }
        Ok(())
    }

    /// Builds the `ExponentialLRScheduler` if all required fields are set.
    pub fn build(self) -> Result<Box<dyn LearningRateScheduler>, NetworkError> {
        self.validate()?;
        Ok(Box::new(ExponentialLRScheduler::new(self.initial_lr, self.decay_rate, self.decay_factor)))
    }
}

impl Default for Exponential {
    /// Creates a new `ExponentialLRSchedulerBuilder` with default values.
    /// Default values:
    /// - `initial_lr`: 0.01
    /// - `decay_rate`: 0.95
    /// - `decay_factor`: 0.1
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_lr_scheduler() {
        let scheduler = ExponentialLRScheduler::new(0.1, 0.9, 0.5);
        assert!((scheduler.schedule(0, 0.0) - 0.1).abs() < 1e-6);
        assert!((scheduler.schedule(1, 0.0) - 0.09486833).abs() < 1e-6);
        assert!((scheduler.schedule(2, 0.0) - 0.09).abs() < 1e-6);
    }
    #[test]
    fn test_exponential_lr_scheduler_builder() {
        let scheduler = Exponential::new()
            .initial_lr(0.1)
            .decay_rate(0.9)
            .decay_factor(0.5)
            .build()
            .unwrap();

        assert!((scheduler.schedule(0, 0.0) - 0.1).abs() < 1e-6);
        assert!((scheduler.schedule(1, 0.0) - 0.09486833).abs() < 1e-6);
        assert!((scheduler.schedule(2, 0.0) - 0.09).abs() < 1e-6);
    }
    #[test]
    fn test_exponential_lr_scheduler_invalid_lr() {
        let scheduler = Exponential::new()
            .initial_lr(0.0)
            .decay_rate(1.5)
            .decay_factor(-0.5)
            .build();

        assert!(scheduler.is_err());
        if let Err(err) = scheduler {
            assert_eq!(
                err.to_string(),
                "Configuration error: Initial learning rate for Exponential must be greater than 0.0, but was 0"
            );
        }
    }

    #[test]
    fn test_exponential_lr_scheduler_invalid_decay_rate() {
        let scheduler = Exponential::new()
            .initial_lr(0.1)
            .decay_rate(1.5)
            .decay_factor(0.5)
            .build();

        assert!(scheduler.is_err());
        if let Err(err) = scheduler {
            assert_eq!(
                err.to_string(),
                "Configuration error: Decay rate for Exponential must be in the range (0, 1), but was 1.5"
            );
        }
    }

    #[test]
    fn test_exponential_lr_scheduler_invalid_decay_factor() {
        let scheduler = Exponential::new()
            .initial_lr(0.1)
            .decay_rate(0.9)
            .decay_factor(-0.5)
            .build();

        assert!(scheduler.is_err());
        if let Err(err) = scheduler {
            assert_eq!(
                err.to_string(),
                "Configuration error: Decay factor for Exponential  must be greater than 0.0, but was -0.5"
            );
        }
    }
}

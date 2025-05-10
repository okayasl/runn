use serde::{Deserialize, Serialize};

use crate::error::NetworkError;

use super::{LearningRateScheduler, LearningRateSchedulerClone};

// StepLRScheduler implements a step decay learning rate scheduler which periodically
// reduces the learning rate by a fixed factor.
// This is useful for training neural networks where you might want to
// decrease the learning rate as the model converges,
// helping to refine the weights with smaller updates as training progresses.
// The decay occurs every `step_size` number of epochs,
// and the amount by which the learning rate decreases is controlled by `decay_rate`.
#[derive(Serialize, Deserialize, Clone)]
struct StepLRScheduler {
    decay_rate: f32,  // Factor by which the learning rate is decayed
    step_size: usize, // Number of epochs between two updates to the learning rate
}

impl StepLRScheduler {
    // Creates a new StepLRScheduler with the given decay rate and step size.
    pub fn new(decay_rate: f32, step_size: usize) -> Self {
        Self { decay_rate, step_size }
    }
}

#[typetag::serde]
impl LearningRateScheduler for StepLRScheduler {
    // Updates the learning rate if the current epoch is a multiple of `step_size`,
    // applying the decay rate to reduce the learning rate.
    // Otherwise, it returns the current learning rate unchanged.
    fn schedule(&self, epoch: usize, current_learning_rate: f32) -> f32 {
        if epoch % self.step_size == 0 {
            current_learning_rate * self.decay_rate
        } else {
            current_learning_rate
        }
    }
}

impl LearningRateSchedulerClone for StepLRScheduler {
    fn clone_box(&self) -> Box<dyn LearningRateScheduler> {
        Box::new(self.clone())
    }
}

/// Step is a builder for StepLRScheduler which allows step-by-step construction.
///
/// StepLRScheduler implements a step decay learning rate scheduler which periodically
/// reduces the learning rate by a fixed factor.
/// This is useful for training neural networks where you might want to
/// decrease the learning rate as the model converges,
/// helping to refine the weights with smaller updates as training progresses.
///
/// The decay occurs every `step_size` number of epochs,
/// and the amount by which the learning rate decreases is controlled by `decay_rate`.
pub struct Step {
    decay_rate: f32,
    step_size: usize,
}

impl Step {
    /// Creates a new builder instance.
    /// The default decay rate is 0.9 and the step size is 10.
    fn new() -> Self {
        Self {
            decay_rate: 0.9,
            step_size: 10,
        }
    }

    /// Sets the decay rate for the scheduler.
    /// The decay rate should be in the range (0, 1).
    /// # Parameters
    /// - `decay_rate`: The factor by which the learning rate is decayed.
    pub fn decay_rate(mut self, decay_rate: f32) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    /// Sets the step size for the scheduler.
    /// This is the number of epochs between two updates to the learning rate.
    /// The step size should be greater than 0.
    /// # Parameters
    /// - `step_size`: The number of epochs between two updates to the learning rate.
    pub fn step_size(mut self, step_size: usize) -> Self {
        self.step_size = step_size;
        self
    }

    /// Validates the parameters of the scheduler.
    fn validate(&self) -> Result<(), NetworkError> {
        if self.decay_rate <= 0.0 || self.decay_rate >= 1.0 {
            return Err(NetworkError::ConfigError(format!(
                "Decay rate for Step must be in the range (0, 1), but was {}",
                self.decay_rate
            )));
        }
        if self.step_size == 0 {
            return Err(NetworkError::ConfigError(format!(
                "Step size for Step must be greater than 0, but was {}",
                self.step_size
            )));
        }
        Ok(())
    }

    /// Builds the StepLRScheduler, returning an error if any required fields are missing.
    pub fn build(self) -> Result<Box<dyn LearningRateScheduler>, NetworkError> {
        self.validate()?;
        Ok(Box::new(StepLRScheduler::new(self.decay_rate, self.step_size)))
    }
}

impl Default for Step {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_lr_scheduler() {
        let scheduler = StepLRScheduler::new(0.5, 10);
        assert_eq!(scheduler.schedule(10, 0.1), 0.05); // Decay applied
        assert_eq!(scheduler.schedule(15, 0.1), 0.1); // No decay
        assert_eq!(scheduler.schedule(20, 0.1), 0.05); // Decay applied
    }

    #[test]
    fn test_step_builder() {
        let scheduler = Step::new()
            .decay_rate(0.8)
            .step_size(5)
            .build()
            .expect("Failed to build StepLRScheduler");

        assert_eq!(scheduler.schedule(5, 0.1), 0.080000006); // Decay applied
        assert_eq!(scheduler.schedule(6, 0.1), 0.1); // No decay
        assert_eq!(scheduler.schedule(10, 0.1), 0.080000006); // Decay applied
    }
    #[test]
    fn test_step_builder_invalid_decay_rate() {
        let step = Step::new().decay_rate(1.0).step_size(5).build();
        // Expect an error due to invalid decay rate
        assert!(step.is_err());
        if let Err(err) = step {
            assert_eq!(
                err.to_string(),
                "Configuration error: Decay rate for Step must be in the range (0, 1), but was 1"
            );
        }
    }
}

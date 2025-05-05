use serde::{Deserialize, Serialize};

use super::{LearningRateScheduler, LearningRateSchedulerClone};

/// StepLRScheduler implements a step decay learning rate scheduler which periodically
/// reduces the learning rate by a fixed factor.
/// This is useful for training neural networks where you might want to
/// decrease the learning rate as the model converges,
/// helping to refine the weights with smaller updates as training progresses.
/// The decay occurs every `step_size` number of epochs,
/// and the amount by which the learning rate decreases is controlled by `decay_rate`.
#[derive(Serialize, Deserialize, Clone)]
struct StepLRScheduler {
    decay_rate: f32,  // Factor by which the learning rate is decayed
    step_size: usize, // Number of epochs between two updates to the learning rate
}

impl StepLRScheduler {
    /// Creates a new StepLRScheduler with the given decay rate and step size.
    pub fn new(decay_rate: f32, step_size: usize) -> Self {
        Self { decay_rate, step_size }
    }
}

#[typetag::serde]
impl LearningRateScheduler for StepLRScheduler {
    /// Updates the learning rate if the current epoch is a multiple of `step_size`,
    /// applying the decay rate to reduce the learning rate.
    /// Otherwise, it returns the current learning rate unchanged.
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

/// Builder for StepLRScheduler to allow step-by-step construction.
/// StepLRScheduler implements a step decay learning rate scheduler which periodically
/// reduces the learning rate by a fixed factor.
/// This is useful for training neural networks where you might want to
/// decrease the learning rate as the model converges,
/// helping to refine the weights with smaller updates as training progresses.
/// The decay occurs every `step_size` number of epochs,
/// and the amount by which the learning rate decreases is controlled by `decay_rate`.
pub struct Step {
    decay_rate: f32,
    step_size: usize,
}

impl Step {
    /// Creates a new builder instance.
    pub fn new() -> Self {
        Self {
            decay_rate: 0.9,
            step_size: 10,
        }
    }

    /// Sets the decay rate for the scheduler.
    pub fn decay_rate(mut self, decay_rate: f32) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    /// Sets the step size for the scheduler.
    pub fn step_size(mut self, step_size: usize) -> Self {
        self.step_size = step_size;
        self
    }

    /// Builds the StepLRScheduler, returning an error if any required fields are missing.
    pub fn build(self) -> Box<dyn LearningRateScheduler> {
        Box::new(StepLRScheduler::new(self.decay_rate, self.step_size))
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
}

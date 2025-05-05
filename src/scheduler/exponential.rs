use serde::{Deserialize, Serialize};

use super::{LearningRateScheduler, LearningRateSchedulerClone};

/// ExponentialLRScheduler implements an exponential decay learning rate scheduler which continuously
/// decreases the learning rate after each epoch. This scheduler is useful for smoothing the training process
/// by gradually decreasing the step size of updates to the model's parameters,
/// allowing for more precise convergence as training progresses. The rate of decay per epoch is
/// controlled by `decay_rate` raised to the power of the product of the epoch number and `decay_factor`.
#[derive(Serialize, Deserialize, Clone)]
struct ExponentialLRScheduler {
    initial_lr: f32,   // Initial learning rate
    decay_rate: f32,   // Base for the exponential decay
    decay_factor: f32, // Decay factor
}

impl ExponentialLRScheduler {
    /// Creates a new `ExponentialLRScheduler` with the given initial learning rate, decay rate, and decay factor.
    pub fn new(initial_lr: f32, decay_rate: f32, decay_factor: f32) -> Self {
        Self {
            initial_lr,
            decay_rate,
            decay_factor,
        }
    }
}

#[typetag::serde]
impl LearningRateScheduler for ExponentialLRScheduler {
    /// Computes the new learning rate by applying exponential decay based on the epoch number.
    /// The decay factor controls how quickly the learning rate decreases;
    /// a smaller decay factor results in slower decay.
    fn schedule(&self, epoch: usize, _current_learning_rate: f32) -> f32 {
        self.initial_lr * self.decay_rate.powf(epoch as f32 * self.decay_factor)
    }
}

impl LearningRateSchedulerClone for ExponentialLRScheduler {
    fn clone_box(&self) -> Box<dyn LearningRateScheduler> {
        Box::new(self.clone())
    }
}

/// Builder for `ExponentialLRScheduler` to allow for more flexible and readable construction.
/// ExponentialLRScheduler implements an exponential decay learning rate scheduler which continuously
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
    /// Creates a new `ExponentialLRSchedulerBuilder`.
    pub fn new() -> Self {
        Self {
            initial_lr: 0.01,
            decay_rate: 0.95,
            decay_factor: 0.1,
        }
    }

    /// Sets the initial learning rate.
    pub fn initial_lr(mut self, initial_lr: f32) -> Self {
        self.initial_lr = initial_lr;
        self
    }

    /// Sets the decay rate.
    pub fn decay_rate(mut self, decay_rate: f32) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    /// Sets the decay factor.
    pub fn decay_factor(mut self, decay_factor: f32) -> Self {
        self.decay_factor = decay_factor;
        self
    }

    /// Builds the `ExponentialLRScheduler` if all required fields are set.
    pub fn build(self) -> Box<dyn LearningRateScheduler> {
        Box::new(ExponentialLRScheduler::new(self.initial_lr, self.decay_rate, self.decay_factor))
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
}

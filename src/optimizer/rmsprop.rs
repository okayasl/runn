use super::{Optimizer, OptimizerConfig};
use crate::{common::matrix::DenseMatrix, LearningRateScheduler};
use serde::{Deserialize, Serialize};
use typetag;

#[derive(Serialize, Deserialize, Clone)]
pub struct RMSPropConfig {
    learning_rate: f32,
    decay_rate: f32,
    epsilon: f32,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
}

#[typetag::serde]
impl OptimizerConfig for RMSPropConfig {
    fn create_optimizer(self: Box<Self>) -> Box<dyn Optimizer> {
        Box::new(RMSPropOptimizer::new(*self))
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RMSPropOptimizer {
    config: RMSPropConfig,
    accumulated_squared_grad_weights: DenseMatrix,
    accumulated_squared_grad_biases: DenseMatrix,
}

impl RMSPropOptimizer {
    pub fn new(config: RMSPropConfig) -> Self {
        Self {
            config,
            accumulated_squared_grad_weights: DenseMatrix::zeros(0, 0),
            accumulated_squared_grad_biases: DenseMatrix::zeros(0, 0),
        }
    }
}

#[typetag::serde]
impl Optimizer for RMSPropOptimizer {
    fn initialize(&mut self, weights: &DenseMatrix, biases: &DenseMatrix) {
        self.accumulated_squared_grad_weights = DenseMatrix::zeros(weights.rows(), weights.cols());
        self.accumulated_squared_grad_biases = DenseMatrix::zeros(biases.rows(), biases.cols());
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix,
        biases: &mut DenseMatrix,
        d_weights: &DenseMatrix,
        d_biases: &DenseMatrix,
        epoch: usize,
    ) {
        if self.config.scheduler.is_some() {
            let scheduler = self.config.scheduler.as_ref().unwrap();
            self.config.learning_rate = scheduler.schedule(epoch, self.config.learning_rate);
        }
        weights.apply_with_indices(|r, c, v| {
            let grad = d_weights.at(r, c);
            let ema_grad = &mut self.accumulated_squared_grad_weights;
            let previous_ema_grad = ema_grad.at(r, c);
            let new_ema_grad = self.config.decay_rate * previous_ema_grad
                + (1.0 - self.config.decay_rate) * grad * grad;
            ema_grad.set(r, c, new_ema_grad);
            *v -= self.config.learning_rate * grad / (new_ema_grad.sqrt() + self.config.epsilon);
        });

        biases.apply_with_indices(|r, c, v| {
            let grad = d_biases.at(r, c);
            let ema_grad = &mut self.accumulated_squared_grad_biases;
            let previous_ema_grad = ema_grad.at(r, c);
            let new_ema_grad = self.config.decay_rate * previous_ema_grad
                + (1.0 - self.config.decay_rate) * grad * grad;
            ema_grad.set(r, c, new_ema_grad);
            *v -= self.config.learning_rate * grad / (new_ema_grad.sqrt() + self.config.epsilon);
        });
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.config.learning_rate = learning_rate;
    }
}

pub struct RMSProp {
    learning_rate: f32,
    decay_rate: f32,
    epsilon: f32,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
}

impl RMSProp {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.001,
            decay_rate: 0.9,
            epsilon: 1e-8,
            scheduler: None,
        }
    }

    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn decay_rate(mut self, rate: f32) -> Self {
        self.decay_rate = rate;
        self
    }

    pub fn epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    pub fn scheduler(mut self, scheduler: Box<dyn LearningRateScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    pub fn build(self) -> RMSPropConfig {
        RMSPropConfig {
            learning_rate: self.learning_rate,
            decay_rate: self.decay_rate,
            epsilon: self.epsilon,
            scheduler: self.scheduler,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::util::equal_approx;

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
        let weights = DenseMatrix::new(2, 2, &[0.1, 0.2, 0.3, 0.4]);
        let biases = DenseMatrix::new(2, 1, &[0.1, 0.2]);
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
        let mut weights = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut biases = DenseMatrix::new(2, 1, &[1.0, 2.0]);

        // Create mock gradient matrices
        let d_weights = DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]);
        let d_biases = DenseMatrix::new(2, 1, &[0.1, 0.1]);

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

        let expected_weights =
            DenseMatrix::new(2, 2, &[0.96837723, 1.9683772, 2.9683774, 3.9683774]);

        assert!(equal_approx(&weights, &expected_weights, 1e-6));
    }
}

use crate::{common::matrix::DenseMatrix, LearningRateScheduler};
use serde::{Deserialize, Serialize};
use typetag;

use super::{Optimizer, OptimizerConfig};

// AMSGrad (Adaptive Moment Estimation with Maximum Moment) is a variant of the Adam optimizer
// designed to improve convergence in scenarios where Adam may fail due to rapidly changing gradients.
// Similar to Adam, AMSGrad maintains two moments:
// - moment1 (m_t): The first moment, which captures the exponentially decaying average of past gradients,
//   acting as a momentum term to accelerate optimization in the direction of historical gradients.
// - moment2 (v_t): The second moment, which tracks the exponentially decaying average of squared gradients,
//   providing an adaptive learning rate that scales based on the magnitude of recent gradients.
// AMSGrad introduces an additional term:
// - max_moment2 (v_max): This tracks the maximum value of the second moment (v_t) observed so far,
//   ensuring that the denominator in the parameter update rule does not decrease over time.
// This modification addresses issues with Adam's convergence by enforcing a more stable learning rate.
// Update rules:
// momentum = beta1 * momentum + (1 - beta1) * gradient
// accumulated_gradient = beta2 * accumulated_gradient + (1 - beta2) * gradient ** 2
// max_accumulated_gradient = max(max_accumulated_gradient, accumulated_gradient)
// weight = weight - (learning_rate / sqrt(max_accumulated_gradient + epsilon)) * momentum
// bias = bias - (learning_rate / sqrt(max_accumulated_gradient + epsilon)) * momentum
#[derive(Serialize, Deserialize, Clone)]
pub struct AMSGradOptimizer {
    config: AMSGradConfig,
    moment1_weights: DenseMatrix,
    moment2_weights: DenseMatrix,
    max_moment2_weights: DenseMatrix,
    moment1_biases: DenseMatrix,
    moment2_biases: DenseMatrix,
    max_moment2_biases: DenseMatrix,
    t: usize,
}

impl AMSGradOptimizer {
    pub fn new(config: AMSGradConfig) -> Self {
        Self {
            config,
            moment1_weights: DenseMatrix::zeros(0, 0),
            moment2_weights: DenseMatrix::zeros(0, 0),
            max_moment2_weights: DenseMatrix::zeros(0, 0),
            moment1_biases: DenseMatrix::zeros(0, 0),
            moment2_biases: DenseMatrix::zeros(0, 0),
            max_moment2_biases: DenseMatrix::zeros(0, 0),
            t: 0,
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

        self.max_moment2_weights.apply_with_indices(|i, j, v| {
            *v = v.max(self.moment2_weights.at(i, j));
        });

        self.moment1_biases.apply_with_indices(|i, j, v| {
            *v = self.config.beta1 * *v + (1.0 - self.config.beta1) * d_biases.at(i, j);
        });

        self.moment2_biases.apply_with_indices(|i, j, v| {
            let g = d_biases.at(i, j);
            *v = self.config.beta2 * *v + (1.0 - self.config.beta2) * g * g;
        });

        self.max_moment2_biases.apply_with_indices(|i, j, v| {
            *v = v.max(self.moment2_biases.at(i, j));
        });
    }

    fn update_parameters(&mut self, weights: &mut DenseMatrix, biases: &mut DenseMatrix, step_size: f32) {
        weights.apply_with_indices(|i, j, v| {
            let m_hat = self.moment1_weights.at(i, j) / (1.0 - self.config.beta1.powi(self.t as i32));
            let v_hat = self.max_moment2_weights.at(i, j) / (1.0 - self.config.beta2.powi(self.t as i32));
            *v -= step_size * m_hat / (v_hat.sqrt() + self.config.epsilon);
        });

        biases.apply_with_indices(|i, j, v| {
            let m_hat = self.moment1_biases.at(i, j) / (1.0 - self.config.beta1.powi(self.t as i32));
            let v_hat = self.max_moment2_biases.at(i, j) / (1.0 - self.config.beta2.powi(self.t as i32));
            *v -= step_size * m_hat / (v_hat.sqrt() + self.config.epsilon);
        });
    }
}

#[typetag::serde]
impl Optimizer for AMSGradOptimizer {
    fn initialize(&mut self, weights: &DenseMatrix, biases: &DenseMatrix) {
        self.moment1_weights = DenseMatrix::zeros(weights.rows(), weights.cols());
        self.moment2_weights = DenseMatrix::zeros(weights.rows(), weights.cols());
        self.max_moment2_weights = DenseMatrix::zeros(weights.rows(), weights.cols());
        self.moment1_biases = DenseMatrix::zeros(biases.rows(), biases.cols());
        self.moment2_biases = DenseMatrix::zeros(biases.rows(), biases.cols());
        self.max_moment2_biases = DenseMatrix::zeros(biases.rows(), biases.cols());
    }

    fn update(
        &mut self, weights: &mut DenseMatrix, biases: &mut DenseMatrix, d_weights: &DenseMatrix,
        d_biases: &DenseMatrix, epoch: usize,
    ) {
        if self.config.scheduler.is_some() {
            let scheduler = self.config.scheduler.as_ref().unwrap();
            self.config.learning_rate = scheduler.schedule(epoch, self.config.learning_rate);
        }
        self.t += 1;
        let step_size = self.config.learning_rate * (1.0 - self.config.beta1.powi(self.t as i32))
            / (1.0 - self.config.beta2.powi(self.t as i32)).sqrt();

        self.update_moments(d_weights, d_biases);
        self.update_parameters(weights, biases, step_size);
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.config.learning_rate = learning_rate;
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AMSGradConfig {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
}

#[typetag::serde]
impl OptimizerConfig for AMSGradConfig {
    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
    fn create_optimizer(self: Box<Self>) -> Box<dyn Optimizer> {
        Box::new(AMSGradOptimizer::new(*self))
    }
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

pub struct AMSGrad {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
}

impl AMSGrad {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            scheduler: None,
        }
    }

    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
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
    pub fn scheduler(mut self, scheduler: Box<dyn LearningRateScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    pub fn build(self) -> AMSGradConfig {
        AMSGradConfig {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            scheduler: self.scheduler,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_initialize() {
        let config = AMSGradConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = AMSGradOptimizer::new(config);
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
        let config = AMSGradConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = AMSGradOptimizer::new(config);
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
        let config = AMSGradConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = AMSGradOptimizer::new(config);
        optimizer.update_learning_rate(0.01);
        assert_eq!(optimizer.config.learning_rate, 0.01);
    }

    #[test]
    fn test_amsgrad_optimizer() {
        // Create mock parameter matrices
        let mut weights = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut biases = DenseMatrix::new(2, 1, &[1.0, 2.0]);

        // Create mock gradient matrices
        let d_weights = DenseMatrix::new(2, 2, &[10.0, 11.0, 12.0, 13.0]);
        let d_biases = DenseMatrix::new(2, 1, &[10.0, 11.0]);

        // Create an instance of the AMSGrad optimizer
        let config = AMSGradConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = AMSGradOptimizer::new(config);
        optimizer.initialize(&weights, &biases);

        // Update the parameters using the mock gradients
        optimizer.update(&mut weights, &mut biases, &d_weights, &d_biases, 1);

        // Manually compute the expected values with the AMSGrad update rule
        let mut expected_params = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        let (rows, cols) = (weights.rows(), weights.cols());
        let mut m_t;
        let mut v_t;

        // Compute m_t and v_t
        m_t = optimizer.moment1_weights.clone();
        m_t.scale(optimizer.config.beta1);
        let mut m_tmp;
        m_tmp = d_weights.clone();
        m_tmp.scale(1.0 - optimizer.config.beta1);
        m_t.add(&m_tmp);

        v_t = optimizer.moment2_weights.clone();
        v_t.scale(optimizer.config.beta2);
        let mut v_tmp = DenseMatrix::zeros(rows, cols);
        v_tmp.apply_with_indices(|r, c, v| {
            let g = d_weights.at(r, c);
            *v = g * g;
        });
        v_tmp.scale(1.0 - optimizer.config.beta2);
        v_t.add(&v_tmp);

        // Bias-corrected first and second moment estimates
        let mut m_hat;
        let mut v_hat;

        m_hat = m_t.clone();
        m_hat.scale(1.0 / (1.0 - optimizer.config.beta1.powi(optimizer.t as i32)));

        v_hat = v_t.clone();
        v_hat.scale(1.0 / (1.0 - optimizer.config.beta2.powi(optimizer.t as i32)));

        // AMSGrad parameter update
        expected_params.apply_with_indices(|r, c, v| {
            let m_h = m_hat.at(r, c);
            let v_h = v_hat.at(r, c);
            let update = optimizer.config.learning_rate * m_h / (v_h.sqrt() + optimizer.config.epsilon);
            *v = weights.at(r, c) - update;
        });

        assert!(equal_approx(&weights, &expected_params, 1e-2));
    }
}

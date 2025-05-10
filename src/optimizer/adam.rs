use crate::{common::matrix::DMat, error::NetworkError, LearningRateScheduler};
use serde::{Deserialize, Serialize};
use typetag;

use super::{Optimizer, OptimizerConfig, OptimizerConfigClone};

// Adam(Adaptive Moment Estimation) is an optimization algorithm that
// adapts learning rates for each parameter based on the magnitude of the gradient.
// moment1 (often referred to as m or v_t in literature) acts as the Momentum,
// capturing the direction of the gradients over time and
// accelerating the optimization in the direction of the combined historical gradients.
// moment2 (typically denoted as v or s_t) represents the RMS component,
// tracking the magnitude (or scale) of the gradients,
// allowing for an adaptive learning rate that scales according to the recent gradient history,
// helping in managing oscillations and stabilizing learning.
// momentum = beta1 * momentum + (1 - beta1) * gradient
// accumulated_gradient = beta2 * accumulated_gradient + (1 - beta2) * gradient ** 2
// weight = weight - (learning_rate / sqrt(accumulated_gradient + epsilon)) * momentum
// bias = bias - (learning_rate / sqrt(accumulated_gradient + epsilon)) * momentum
#[derive(Serialize, Deserialize, Clone)]
struct AdamOptimizer {
    config: AdamConfig,
    moment1_weights: DMat,
    moment2_weights: DMat,
    moment1_biases: DMat,
    moment2_biases: DMat,
    t: usize,
    m_hat_factor: f32,
    v_hat_factor: f32,
}

impl AdamOptimizer {
    pub(crate) fn new(config: AdamConfig) -> Self {
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

    // Updates the parameters (weights and biases) using the Adam optimization algorithm.
    //
    // This function applies the Adam update rule to the provided weights and biases matrices.
    // It uses the pre-computed first and second moment estimates (m_hat and v_hat) to adjust
    // the parameters, ensuring a more adaptive learning rate for each parameter. The update
    // is performed in place, modifying the original matrices. The step size is a scaling
    // factor that influences the magnitude of the update.
    //
    // # Arguments
    //
    // * `weights` - A mutable reference to a DenseMatrix representing the weights to be updated.
    // * `biases` - A mutable reference to a DenseMatrix representing the biases to be updated.
    // * `step_size` - A floating-point value that represents the step size for scaling the update.
    fn update_parameters(&self, weights: &mut DMat, biases: &mut DMat, step_size: f32) {
        weights.apply_with_indices(|i, j, v| {
            let m_hat = self.moment1_weights.at(i, j) / self.m_hat_factor;
            let v_hat = self.moment2_weights.at(i, j) / self.v_hat_factor;
            *v -= step_size * m_hat / (v_hat.sqrt() + self.config.epsilon);
        });

        biases.apply_with_indices(|i, j, v| {
            let m_hat = self.moment1_biases.at(i, j) / self.m_hat_factor;
            let v_hat = self.moment2_biases.at(i, j) / self.v_hat_factor;
            *v -= step_size * m_hat / (v_hat.sqrt() + self.config.epsilon);
        });
    }
}

#[typetag::serde]
impl Optimizer for AdamOptimizer {
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

        self.update_moments(d_weights, d_biases);
        self.update_parameters(weights, biases, step_size);
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.config.learning_rate = learning_rate;
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct AdamConfig {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
}

#[typetag::serde]
impl OptimizerConfig for AdamConfig {
    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
    fn create_optimizer(&self) -> Box<dyn Optimizer> {
        Box::new(AdamOptimizer::new(self.clone()))
    }
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

/// Adam is a builder for Adam(Adaptive Moment Estimation) which is an optimization algorithm that
/// adapts learning rates for each parameter based on the magnitude of the gradient.
///
/// moment1 (often referred to as m or v_t in literature) acts as the Momentum,
/// capturing the direction of the gradients over time and
/// accelerating the optimization in the direction of the combined historical gradients.
///
/// moment2 (typically denoted as v or s_t) represents the RMS component,
/// tracking the magnitude (or scale) of the gradients,
/// allowing for an adaptive learning rate that scales according to the recent gradient history,
/// helping in managing oscillations and stabilizing learning.
///
/// momentum = beta1 * momentum + (1 - beta1) * gradient
/// accumulated_gradient = beta2 * accumulated_gradient + (1 - beta2) * gradient ** 2
/// weight = weight - (learning_rate / sqrt(accumulated_gradient + epsilon)) * momentum
/// bias = bias - (learning_rate / sqrt(accumulated_gradient + epsilon)) * momentum
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    scheduler: Option<Result<Box<dyn LearningRateScheduler>, NetworkError>>,
}

impl Adam {
    /// Creates a new builder for Adam(Adaptive Moment Estimation) optimizer.
    /// Default values:
    /// - learning_rate: 0.01
    /// - beta1: 0.9
    /// - beta2: 0.999
    /// - epsilon: f32::EPSILON
    /// - scheduler: None
    fn new() -> Adam {
        Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: f32::EPSILON,
            scheduler: None,
        }
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self::new()
    }
}

impl Adam {
    /// Set the learning rate.
    ///
    /// The learning rate controls the step size at each iteration while moving toward a minimum of the loss function.
    /// # Parameters
    /// - `learning_rate`: Learning rate, typically a small positive value (e.g., 0.001).
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
                "Learning rate for Adam must be greater than 0.0, but was {}",
                self.learning_rate
            )));
        }
        if self.beta1 <= 0.0 || self.beta1 >= 1.0 {
            return Err(NetworkError::ConfigError(format!(
                "Beta1 for Adam must be in the range (0, 1), but was {}",
                self.beta1
            )));
        }
        if self.beta2 <= 0.0 || self.beta2 >= 1.0 {
            return Err(NetworkError::ConfigError(format!(
                "Beta2 for Adam must be in the range (0, 1), but was {}",
                self.beta2
            )));
        }
        if self.epsilon <= 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Epsilon for Adam must be greater than 0.0, but was {}",
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
        Ok(Box::new(AdamConfig {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            scheduler: self.scheduler.map(|s| s.unwrap()),
        }))
    }
}

impl OptimizerConfigClone for AdamConfig {
    fn clone_box(&self) -> Box<dyn OptimizerConfig> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{common::matrix::DMat, util::equal_approx};

    #[test]
    fn test_initialize() {
        let adam_config = AdamConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = AdamOptimizer::new(adam_config);
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
        let config = AdamConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = AdamOptimizer::new(config);
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
        let adam_config = AdamConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = AdamOptimizer::new(adam_config);
        optimizer.update_learning_rate(0.01);
        assert_eq!(optimizer.config.learning_rate, 0.01);
    }

    #[test]
    fn test_adam_optimizer() {
        // Create mock parameter matrices
        let mut weights = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut biases = DMat::new(2, 1, &[1.0, 2.0]);

        // Create mock gradient matrices
        let d_weights = DMat::new(2, 2, &[10.0, 11.0, 12.0, 13.0]);
        let d_biases = DMat::new(2, 1, &[10.0, 11.0]);

        // Create an instance of the Adam optimizer
        let adam_config = AdamConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            scheduler: None,
        };
        let mut optimizer = AdamOptimizer::new(adam_config);
        optimizer.initialize(&weights, &biases);

        // Update the parameters using the mock gradients
        optimizer.update(&mut weights, &mut biases, &d_weights, &d_biases, 1);

        // Manually compute the expected values with the Adam update rule
        let mut expected_params = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);

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
        let mut v_tmp = DMat::zeros(rows, cols);
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

        // Adam parameter update
        expected_params.apply_with_indices(|r, c, v| {
            let m_h = m_hat.at(r, c);
            let v_h = v_hat.at(r, c);
            let update = optimizer.config.learning_rate * m_h / (v_h.sqrt() + optimizer.config.epsilon);
            *v = weights.at(r, c) - update;
        });

        assert!(equal_approx(&weights, &expected_params, 1e-2));
    }

    #[test]
    fn test_clone_adam_optimizer() {
        let adam_config = AdamConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            scheduler: None,
        };
        let optimizer = AdamOptimizer::new(adam_config);
        let cloned_optimizer = optimizer.clone();
        assert_eq!(optimizer.config.learning_rate, cloned_optimizer.config.learning_rate);
        assert_eq!(optimizer.config.beta1, cloned_optimizer.config.beta1);
        assert_eq!(optimizer.config.beta2, cloned_optimizer.config.beta2);
    }

    #[test]
    fn test_clone_adam_optimizer_config() {
        let adam_config = Adam::new()
            .learning_rate(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .build()
            .unwrap();
        let cloned_config = adam_config.clone();
        assert_eq!(adam_config.learning_rate(), cloned_config.learning_rate());
    }
}

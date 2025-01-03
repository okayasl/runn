use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use typetag;

use super::Optimizer;

#[derive(Serialize, Deserialize, Clone)]
pub struct AdamOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    moment1: Vec<DenseMatrix>,
    moment2: Vec<DenseMatrix>,
    t: usize,
    m_hat_factor: f32,
    v_hat_factor: f32,
}

impl AdamOptimizer {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            moment1: Vec::new(),
            moment2: Vec::new(),
            t: 0,
            m_hat_factor: 1.0,
            v_hat_factor: 1.0,
        }
    }

    fn update_moments(&mut self, index: usize, grad: &DenseMatrix) {
        self.moment1[index].apply_with_indices(|i, j, v| {
            *v = self.beta1 * *v + (1.0 - self.beta1) * grad.at(i, j);
        });

        self.moment2[index].apply_with_indices(|i, j, v| {
            let g = grad.at(i, j);
            *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
        });
    }

    fn update_parameters(&self, index: usize, param: &mut DenseMatrix, step_size: f32) {
        param.apply_with_indices(|i, j, v| {
            let m_hat = self.moment1[index].at(i, j) / self.m_hat_factor;
            let v_hat = self.moment2[index].at(i, j) / self.v_hat_factor;
            *v -= step_size * m_hat / (v_hat.sqrt() + self.epsilon);
        });
    }
}

#[typetag::serde]
impl Optimizer for AdamOptimizer {
    fn initialize(&mut self, params: &[DenseMatrix]) {
        self.moment1 = params
            .iter()
            .map(|p| DenseMatrix::zeros(p.rows(), p.cols()))
            .collect();
        self.moment2 = params
            .iter()
            .map(|p| DenseMatrix::zeros(p.rows(), p.cols()))
            .collect();
        self.t = 0;
        self.m_hat_factor = 1.0;
        self.v_hat_factor = 1.0;
    }

    fn update(
        &mut self,
        params: &mut [&mut DenseMatrix],
        grads: &[&mut DenseMatrix],
        _epoch: usize,
    ) {
        self.t += 1;
        self.m_hat_factor = 1.0 - self.beta1.powi(self.t as i32);
        self.v_hat_factor = 1.0 - self.beta2.powi(self.t as i32);
        let step_size = self.learning_rate * self.m_hat_factor / self.v_hat_factor.sqrt();

        for i in 0..params.len() {
            self.update_moments(i, &grads[i]);
            self.update_parameters(i, &mut params[i], step_size);
        }
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
}

pub struct AdamBuilder {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl AdamBuilder {
    pub fn new() -> AdamBuilder {
        AdamBuilder {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

impl AdamBuilder {
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
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

    pub fn build(self) -> AdamOptimizer {
        AdamOptimizer::new(self.learning_rate, self.beta1, self.beta2, self.epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_initialize() {
        let mut optimizer = AdamOptimizer::new(0.001, 0.9, 0.999, 1e-8);
        let params = vec![DenseMatrix::new(2, 2, &[0.1, 0.2, 0.3, 0.4])];
        optimizer.initialize(&params);
        assert_eq!(optimizer.moment1.len(), 1);
        assert_eq!(optimizer.moment2.len(), 1);
    }

    #[test]
    fn test_update() {
        let mut optimizer = AdamOptimizer::new(0.001, 0.9, 0.999, 1e-8);
        let mut params = vec![DenseMatrix::new(2, 2, &[0.1, 0.2, 0.3, 0.4])];
        let mut grads = vec![DenseMatrix::new(2, 2, &[0.01, 0.01, 0.01, 0.01])];
        optimizer.initialize(&params);
        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        optimizer.update(&mut params_refs, &mut grads_refs, 1);
        // Check if params are updated correctly
        assert!(params[0].at(0, 0) < 0.1);
    }

    #[test]
    fn test_update_learning_rate() {
        let mut optimizer = AdamOptimizer::new(0.001, 0.9, 0.999, 1e-8);
        optimizer.update_learning_rate(0.01);
        assert_eq!(optimizer.learning_rate, 0.01);
    }

    #[test]
    fn test_adam_optimizer() {
        // Create a mock parameter matrix
        let mut params = vec![
            DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            DenseMatrix::new(2, 2, &[5.0, 6.0, 7.0, 8.0]),
        ];

        // Create a mock gradient matrix
        let mut grads = vec![
            DenseMatrix::new(2, 2, &[10.0, 11.0, 12.0, 13.0]),
            DenseMatrix::new(2, 2, &[14.0, 15.0, 16.0, 17.0]),
        ];

        // Create an instance of the Adam optimizer
        let mut optimizer = AdamOptimizer::new(0.001, 0.9, 0.999, 1e-8);
        optimizer.initialize(&params);

        // Update the parameters using the mock gradients
        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        optimizer.update(&mut params_refs, &mut grads_refs, 1);

        // Manually compute the expected values with the Adam update rule
        let mut expected_params = vec![
            DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            DenseMatrix::new(2, 2, &[5.0, 6.0, 7.0, 8.0]),
        ];

        for i in 0..params.len() {
            let (rows, cols) = (params[i].rows(), params[i].cols());
            let mut m_t;
            let mut v_t;

            // Compute m_t and v_t
            m_t = optimizer.moment1[i].clone();
            m_t.scale(optimizer.beta1);
            let mut m_tmp;
            m_tmp = grads[i].clone();
            m_tmp.scale(1.0 - optimizer.beta1);
            m_t.add(&m_tmp);

            v_t = optimizer.moment2[i].clone();
            v_t.scale(optimizer.beta2);
            let mut v_tmp = DenseMatrix::zeros(rows, cols);
            v_tmp.apply_with_indices(|r, c, v| {
                let g = grads[i].at(r, c);
                *v = g * g;
            });
            v_tmp.scale(1.0 - optimizer.beta2);
            v_t.add(&v_tmp);

            // Bias-corrected first and second moment estimates
            let mut m_hat;
            let mut v_hat;

            m_hat = m_t.clone();
            m_hat.scale(1.0 / (1.0 - optimizer.beta1.powi(optimizer.t as i32)));

            v_hat = v_t.clone();
            v_hat.scale(1.0 / (1.0 - optimizer.beta2.powi(optimizer.t as i32)));

            // Adam parameter update
            expected_params[i].apply_with_indices(|r, c, v| {
                let m_h = m_hat.at(r, c);
                let v_h = v_hat.at(r, c);
                let update = optimizer.learning_rate * m_h / (v_h.sqrt() + optimizer.epsilon);
                *v = params[i].at(r, c) - update;
            });
        }

        // Check that the parameters have been updated correctly
        for i in 0..params.len() {
            assert!(equal_approx(&params[i], &expected_params[i], 1e-2));
        }
    }
}

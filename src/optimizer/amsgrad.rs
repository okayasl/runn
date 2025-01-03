use crate::common::matrix::DenseMatrix;

use serde::{Deserialize, Serialize};
use std::f32;
use typetag;

use super::Optimizer;

#[derive(Serialize, Deserialize, Clone)]
pub struct AMSGradOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    moment1: Vec<DenseMatrix>,
    moment2: Vec<DenseMatrix>,
    max_moment2: Vec<DenseMatrix>,
    t: usize,
}

impl AMSGradOptimizer {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            moment1: Vec::new(),
            moment2: Vec::new(),
            max_moment2: Vec::new(),
            t: 0,
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

        self.max_moment2[index].apply_with_indices(|i, j, v| {
            *v = v.max(self.moment2[index].at(i, j));
        });
    }

    fn update_parameters(&self, index: usize, param: &mut DenseMatrix, step_size: f32) {
        param.apply_with_indices(|i, j, v| {
            let m_hat = self.moment1[index].at(i, j) / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = self.max_moment2[index].at(i, j) / (1.0 - self.beta2.powi(self.t as i32));
            *v -= step_size * m_hat / (v_hat.sqrt() + self.epsilon);
        });
    }
}

#[typetag::serde]
impl Optimizer for AMSGradOptimizer {
    fn initialize(&mut self, params: &[DenseMatrix]) {
        self.moment1 = params
            .iter()
            .map(|p| DenseMatrix::zeros(p.rows(), p.cols()))
            .collect();
        self.moment2 = params
            .iter()
            .map(|p| DenseMatrix::zeros(p.rows(), p.cols()))
            .collect();
        self.max_moment2 = params
            .iter()
            .map(|p| DenseMatrix::zeros(p.rows(), p.cols()))
            .collect();
        self.t = 0;
    }

    fn update(
        &mut self,
        params: &mut [&mut DenseMatrix],
        grads: &[&mut DenseMatrix],
        _epoch: usize,
    ) {
        self.t += 1;
        let step_size = self.learning_rate * (1.0 - self.beta2.powi(self.t as i32)).sqrt()
            / (1.0 - self.beta1.powi(self.t as i32));

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            self.update_moments(i, grad);
            self.update_parameters(i, param, step_size);
        }
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
}

pub struct AMSGradBuilder {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl AMSGradBuilder {
    pub fn new() -> AMSGradBuilder {
        AMSGradBuilder {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

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

    pub fn build(self) -> AMSGradOptimizer {
        AMSGradOptimizer::new(self.learning_rate, self.beta1, self.beta2, self.epsilon)
    }
}

#[cfg(test)]
mod tests {
    use crate::util::equal_approx;

    use super::*;

    #[test]
    fn test_amsgrad_optimizer() {
        let mut params = vec![
            DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            DenseMatrix::new(2, 2, &[5.0, 6.0, 7.0, 8.0]),
        ];

        let mut grads = vec![
            DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]),
            DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]),
        ];

        let mut optimizer = AMSGradOptimizer::new(0.01, 0.9, 0.999, 1e-8);
        optimizer.initialize(&params);
        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        optimizer.update(&mut params_refs, &grads_refs, 1);

        let expected_params = vec![
            DenseMatrix::new(2, 2, &[0.99683774, 1.9968377, 2.9968379, 3.9968379]),
            DenseMatrix::new(2, 2, &[4.9968376, 5.9968376, 6.9968376, 7.9968376]),
        ];

        //write a code that prints params
        params.iter().for_each(|p| println!("{:?}", p.flatten()));

        for (param, expected) in params.iter().zip(expected_params.iter()) {
            assert!(equal_approx(&param, &expected, 1e-6));
        }
    }
}

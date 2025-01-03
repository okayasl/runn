use crate::common::matrix::DenseMatrix;

use serde::{Deserialize, Serialize};
use typetag;

use super::Optimizer;

#[derive(Serialize, Deserialize, Clone)]
pub struct RMSPropOptimizer {
    learning_rate: f32,
    decay_rate: f32,
    epsilon: f32,
    accumulated_ema_squared_gradient: Vec<DenseMatrix>,
}

impl RMSPropOptimizer {
    pub fn new(learning_rate: f32, decay_rate: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            decay_rate,
            epsilon,
            accumulated_ema_squared_gradient: Vec::new(),
        }
    }
}

#[typetag::serde]
impl Optimizer for RMSPropOptimizer {
    fn initialize(&mut self, params: &[DenseMatrix]) {
        self.accumulated_ema_squared_gradient = params
            .iter()
            .map(|p| DenseMatrix::zeros(p.rows(), p.cols()))
            .collect();
    }

    fn update(
        &mut self,
        params: &mut [&mut DenseMatrix],
        grads: &[&mut DenseMatrix],
        _epoch: usize,
    ) {
        for (i, param) in params.iter_mut().enumerate() {
            param.apply_with_indices(|r, c, v| {
                let grad = grads[i].at(r, c);
                let ema_grad = &mut self.accumulated_ema_squared_gradient[i];
                let previous_ema_grad = ema_grad.at(r, c);
                let new_ema_grad =
                    self.decay_rate * previous_ema_grad + (1.0 - self.decay_rate) * grad * grad;
                ema_grad.set(r, c, new_ema_grad);
                *v -= self.learning_rate * grad / (new_ema_grad.sqrt() + self.epsilon);
            });
        }
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
}

pub struct RMSPropBuilder {
    learning_rate: f32,
    decay_rate: f32,
    epsilon: f32,
}

impl RMSPropBuilder {
    pub fn new() -> RMSPropBuilder {
        RMSPropBuilder {
            learning_rate: 0.01,
            decay_rate: 0.9,
            epsilon: 1e-8,
        }
    }

    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn decay_rate(mut self, decay_rate: f32) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn build(self) -> RMSPropOptimizer {
        RMSPropOptimizer::new(self.learning_rate, self.decay_rate, self.epsilon)
    }
}

#[cfg(test)]
mod tests {
    use crate::util::equal_approx;

    use super::*;

    #[test]
    fn test_rmsprop_optimizer() {
        let mut params = vec![
            DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            DenseMatrix::new(2, 2, &[5.0, 6.0, 7.0, 8.0]),
        ];

        let mut grads = vec![
            DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]),
            DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]),
        ];

        let mut optimizer = RMSPropOptimizer::new(0.01, 0.9, 1e-8);
        optimizer.initialize(&params);
        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        optimizer.update(&mut params_refs, &mut grads_refs, 1);

        let expected_params = vec![
            DenseMatrix::new(2, 2, &[0.96837723, 1.9683772, 2.9683774, 3.9683774]),
            DenseMatrix::new(2, 2, &[4.968377, 5.968377, 6.968377, 7.968377]),
        ];

        for (param, expected) in params.iter().zip(expected_params.iter()) {
            assert!(equal_approx(&param, &expected, 1e-6));
        }
    }
}

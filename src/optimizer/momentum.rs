use crate::common::matrix::DenseMatrix;

use serde::{Deserialize, Serialize};
use typetag;

use super::Optimizer;

#[derive(Serialize, Deserialize, Clone)]
pub struct MomentumOptimizer {
    learning_rate: f32,
    momentum: f32,
    velocity: Vec<DenseMatrix>,
}

impl MomentumOptimizer {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: Vec::new(),
        }
    }
}

#[typetag::serde]
impl Optimizer for MomentumOptimizer {
    fn initialize(&mut self, params: &[DenseMatrix]) {
        self.velocity = params
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
                let grad = &grads[i];
                let velocity = &mut self.velocity[i];
                let previous_velocity = velocity.at(r, c);
                // Calculate the new velocity using the Momentum update rule
                // velocity = (momentum * previous_velocity) + (learning_rate * gradient)
                let new_velocity =
                    self.momentum * previous_velocity + self.learning_rate * grad.at(r, c);
                velocity.set(r, c, new_velocity);
                *v -= new_velocity
            });
        }
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }
}

pub struct MomentumBuilder {
    learning_rate: f32,
    momentum: f32,
}

impl MomentumBuilder {
    pub fn new() -> MomentumBuilder {
        MomentumBuilder {
            learning_rate: 0.01,
            momentum: 0.9,
        }
    }

    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn build(self) -> MomentumOptimizer {
        MomentumOptimizer::new(self.learning_rate, self.momentum)
    }
}

#[cfg(test)]
mod tests {
    use crate::util::equal_approx;

    use super::*;

    #[test]
    fn test_momentum_optimizer() {
        let mut params = vec![
            DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            DenseMatrix::new(2, 2, &[5.0, 6.0, 7.0, 8.0]),
        ];

        let mut grads = vec![
            DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]),
            DenseMatrix::new(2, 2, &[0.1, 0.1, 0.1, 0.1]),
        ];

        let mut optimizer = MomentumOptimizer::new(0.01, 0.9);
        optimizer.initialize(&params);
        let mut params_refs: Vec<&mut DenseMatrix> = params.iter_mut().collect();
        let mut grads_refs: Vec<&mut DenseMatrix> = grads.iter_mut().collect();
        optimizer.update(&mut params_refs, &mut grads_refs, 1);

        let expected_params = vec![
            DenseMatrix::new(2, 2, &[0.999, 1.999, 2.999, 3.999]),
            DenseMatrix::new(2, 2, &[4.999, 5.999, 6.999, 7.999]),
        ];

        for (param, expected) in params.iter().zip(expected_params.iter()) {
            assert!(equal_approx(&param, &expected, 1e-6));
        }
    }
}

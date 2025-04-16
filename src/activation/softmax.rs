use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;

use serde::{Deserialize, Serialize};
use typetag;

use super::xavier_initialization;

#[derive(Serialize, Deserialize, Clone)]
pub struct Softmax {
    original_output: Option<DenseMatrix>,
}

impl Softmax {
    pub fn new() -> Self {
        Self { original_output: None }
    }
}

#[typetag::serde]
impl ActivationFunction for Softmax {
    fn forward(&mut self, input: &mut DenseMatrix) {
        let (rows, cols) = (input.rows(), input.cols());
        let mut result = DenseMatrix::zeros(rows, cols);

        for i in 0..rows {
            // Step 1: Find the maximum value to stabilize the exponentials
            let max_val = (0..cols).fold(f32::NEG_INFINITY, |max, j| max.max(input.at(i, j)));

            // Step 2: Compute the exponentials and their sum
            let mut exp_sum = 0.0;
            for j in 0..cols {
                let exp_val = (input.at(i, j) - max_val).exp();
                result.set(i, j, exp_val);
                exp_sum += exp_val;
            }

            // Step 3: Normalize the exponentials
            for j in 0..cols {
                result.set(i, j, result.at(i, j) / exp_sum);
            }
        }

        self.original_output = Some(result.clone());
        *input = result;
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix) {
        let (rows, cols) = (input.rows(), input.cols());
        let mut result = DenseMatrix::zeros(rows, cols);
        let output = self.original_output.as_ref().unwrap();

        for i in 0..rows {
            for j in 0..cols {
                let mut gradient = 0.0;
                for k in 0..cols {
                    if j == k {
                        gradient += output.at(i, k) * (1.0 - output.at(i, j)) * d_output.at(i, k);
                    } else {
                        gradient += -output.at(i, k) * output.at(i, j) * d_output.at(i, k);
                    }
                }
                result.set(i, j, gradient);
            }
        }

        *input = result;
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        xavier_initialization
    }
}

use crate::activation::ActivationFunction;
use crate::common::matrix::DenseMatrix;
use serde::{Deserialize, Serialize};
use typetag;

use super::he_initialization;

// Swish Activation Function
//
// Swish is a self-gated activation function defined as x * sigmoid(βx). It has been found to sometimes outperform ReLU
// in deeper networks due to its non-monotonic form.
//
// Range: (-∞, +∞)
// Best for: Deeper networks where traditional functions like ReLU tend to underperform.
#[derive(Serialize, Deserialize, Clone)]
pub struct Swish {
    beta: f32,
}

impl Swish {
    pub fn new(beta: f32) -> Self {
        Swish { beta }
    }
}

#[typetag::serde]
impl ActivationFunction for Swish {
    fn forward(&mut self, input: &mut DenseMatrix) {
        input.apply(|x| x / (1.0 + (-self.beta * x).exp()));
    }

    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix) {
        input.apply(|x| {
            let sigmoid = 1.0 / (1.0 + (-self.beta * x).exp());
            sigmoid + x * self.beta * sigmoid * (1.0 - sigmoid)
        });
        input.mul_elem(d_output);
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        he_initialization
    }
}

#[cfg(test)]
mod swish_tests {
    use super::*;
    use crate::{common::matrix::DenseMatrix, util::equal_approx};

    #[test]
    fn test_swish_forward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);

        let mut swish = Swish::new(1.0);
        swish.forward(&mut input);

        // Expected output: approximate values
        let expected = DenseMatrix::new(2, 3, &[0.7311, -0.2384, 2.8577, -0.0728, 4.9665, -0.0147]);
        assert!(equal_approx(&input, &expected, 1e-3), "Swish forward pass failed");
    }

    #[test]
    fn test_swish_backward() {
        let mut input = DenseMatrix::new(2, 3, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
        let d_output = DenseMatrix::new(2, 3, &[0.5, 1.0, 0.7, 0.2, 0.3, 0.1]);

        let swish = Swish::new(1.0);
        swish.backward(&d_output, &mut input);

        // Expected output: approximate values
        let expected = DenseMatrix::new(2, 3, &[0.463835, -0.090784, 0.761673, -0.010533, 0.307964, -0.001233]);
        assert!(equal_approx(&input, &expected, 1e-3), "Swish backward pass failed");
    }
}

pub mod cross_entropy;
pub mod mean_squared_error;

use crate::{common::matrix::DenseMatrix, MetricResult};
use typetag;

#[typetag::serde]
pub trait LossFunction: LossFunctionClone + Send + Sync {
    fn forward(&self, predicted: &DenseMatrix, target: &DenseMatrix) -> f32;
    fn backward(&self, predicted: &DenseMatrix, target: &DenseMatrix) -> DenseMatrix;
    fn calculate_metrics(&self, targets: &DenseMatrix, predictions: &DenseMatrix) -> MetricResult;
}

pub trait LossFunctionClone {
    fn clone_box(&self) -> Box<dyn LossFunction>;
}

impl LossFunctionClone for Box<dyn LossFunction> {
    fn clone_box(&self) -> Box<dyn LossFunction> {
        (**self).clone_box()
    }
}

impl Clone for Box<dyn LossFunction> {
    fn clone(&self) -> Box<dyn LossFunction> {
        self.clone_box()
    }
}

#[typetag::serde]
impl LossFunction for Box<dyn LossFunction> {
    fn forward(&self, predicted: &DenseMatrix, target: &DenseMatrix) -> f32 {
        (**self).forward(predicted, target)
    }

    fn backward(&self, predicted: &DenseMatrix, target: &DenseMatrix) -> DenseMatrix {
        (**self).backward(predicted, target)
    }

    fn calculate_metrics(&self, targets: &DenseMatrix, predictions: &DenseMatrix) -> MetricResult {
        (**self).calculate_metrics(targets, predictions)
    }
}

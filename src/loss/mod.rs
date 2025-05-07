pub mod cross_entropy;
pub mod mean_squared_error;

use crate::{common::matrix::DMat, MetricResult};
use typetag;

#[typetag::serde]
pub trait LossFunction: LossFunctionClone + Send + Sync {
    fn forward(&self, predicted: &DMat, target: &DMat) -> f32;
    fn backward(&self, predicted: &DMat, target: &DMat) -> DMat;
    fn calculate_metrics(&self, targets: &DMat, predictions: &DMat) -> MetricResult;
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
    fn forward(&self, predicted: &DMat, target: &DMat) -> f32 {
        (**self).forward(predicted, target)
    }

    fn backward(&self, predicted: &DMat, target: &DMat) -> DMat {
        (**self).backward(predicted, target)
    }

    fn calculate_metrics(&self, targets: &DMat, predictions: &DMat) -> MetricResult {
        (**self).calculate_metrics(targets, predictions)
    }
}

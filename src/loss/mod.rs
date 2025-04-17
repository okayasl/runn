pub mod cross_entropy;
pub mod mean_squared_error;

use crate::common::matrix::DenseMatrix;
use typetag;

#[typetag::serde]
pub trait LossFunction: LossFunctionClone + Send + Sync {
    fn forward(&self, predicted: &DenseMatrix, target: &DenseMatrix) -> f32;
    fn backward(&self, predicted: &DenseMatrix, target: &DenseMatrix) -> DenseMatrix;
}

pub trait LossFunctionClone {
    fn clone_box(&self) -> Box<dyn LossFunction>;
}

impl<T> LossFunctionClone for T
where
    T: 'static + LossFunction + Clone,
{
    fn clone_box(&self) -> Box<dyn LossFunction> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn LossFunction> {
    fn clone(&self) -> Box<dyn LossFunction> {
        self.clone_box()
    }
}

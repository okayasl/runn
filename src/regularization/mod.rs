pub mod dropout;
pub mod l1;
pub mod l2;

use crate::common::matrix::DenseMatrix;

#[typetag::serde]
pub trait Regularization: RegularizationClone + Send + Sync {
    fn apply(&self, params: &mut [&mut DenseMatrix], grads: &mut [&mut DenseMatrix]);
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait RegularizationClone {
    fn clone_box(&self) -> Box<dyn Regularization>;
}

impl<T> RegularizationClone for T
where
    T: 'static + Regularization + Clone,
{
    fn clone_box(&self) -> Box<dyn Regularization> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Regularization> {
    fn clone(&self) -> Box<dyn Regularization> {
        self.clone_box()
    }
}

pub mod dropout;
pub mod l1;
pub mod l2;

use crate::common::matrix::DenseMatrix;

use super::dropout::DropoutRegularization;
use super::l1::L1Regularization;
use super::l2::L2Regularization;

#[typetag::serde]
pub trait Regularization: RegularizationClone + Send {
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

pub struct RegularizationHolder {
    regularizations: Vec<Box<dyn Regularization>>,
}

impl RegularizationHolder {
    pub fn new(regularizations: Vec<Box<dyn Regularization>>) -> Self {
        Self { regularizations }
    }

    pub fn apply_forward(&self, params: &mut [&mut DenseMatrix], grads: &mut [&mut DenseMatrix]) {
        for reg in &self.regularizations {
            if let Some(dropout) = reg.as_any().downcast_ref::<DropoutRegularization>() {
                dropout.apply(params, grads);
            }
        }
    }

    pub fn apply_backward(&self, params: &mut [&mut DenseMatrix], grads: &mut [&mut DenseMatrix]) {
        for reg in &self.regularizations {
            if reg.as_any().downcast_ref::<L1Regularization>().is_some()
                || reg.as_any().downcast_ref::<L2Regularization>().is_some()
            {
                reg.apply(params, grads);
            }
        }
    }
}

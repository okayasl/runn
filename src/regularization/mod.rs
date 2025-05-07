pub mod dropout;
pub mod l1;
pub mod l2;

use crate::common::matrix::DMat;

#[typetag::serde]
pub trait Regularization: RegularizationClone + Send + Sync {
    fn apply(&self, params: &mut [&mut DMat], grads: &mut [&mut DMat]);
    fn as_any(&self) -> &dyn std::any::Any;
}

#[typetag::serde]
impl Regularization for Box<dyn Regularization> {
    fn apply(&self, params: &mut [&mut DMat], grads: &mut [&mut DMat]) {
        (**self).apply(params, grads);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        (**self).as_any()
    }
}

impl RegularizationClone for Box<dyn Regularization> {
    fn clone_box(&self) -> Box<dyn Regularization> {
        (**self).clone_box()
    }
}

pub trait RegularizationClone {
    fn clone_box(&self) -> Box<dyn Regularization>;
}

impl Clone for Box<dyn Regularization> {
    fn clone(&self) -> Box<dyn Regularization> {
        self.clone_box()
    }
}

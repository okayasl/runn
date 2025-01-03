pub mod adam;
pub mod amsgrad;
pub mod momentum;
pub mod rmsprop;
pub mod sgd;

use crate::common::matrix::DenseMatrix;

use typetag;

#[typetag::serde]
pub trait Optimizer: OptimizerClone + Send {
    fn initialize(&mut self, params: &[DenseMatrix]);
    fn update(&mut self, params: &mut [&mut DenseMatrix], grads: &[&mut DenseMatrix], epoch: usize);
    fn update_learning_rate(&mut self, learning_rate: f32);
}

pub trait OptimizerClone {
    fn clone_box(&self) -> Box<dyn Optimizer>;
}

impl<T> OptimizerClone for T
where
    T: 'static + Optimizer + Clone,
{
    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Optimizer> {
    fn clone(&self) -> Box<dyn Optimizer> {
        self.clone_box()
    }
}

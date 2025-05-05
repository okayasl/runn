pub mod adam;
pub mod adamw;
pub mod amsgrad;
pub mod momentum;
pub mod rmsprop;
pub mod sgd;

use crate::common::matrix::DenseMatrix;

use typetag;

#[typetag::serde]
pub trait Optimizer: OptimizerClone + Send + Sync {
    fn initialize(&mut self, weights: &DenseMatrix, biases: &DenseMatrix);
    fn update(
        &mut self, weights: &mut DenseMatrix, biases: &mut DenseMatrix, d_weights: &DenseMatrix,
        d_biases: &DenseMatrix, epoch: usize,
    );
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

#[typetag::serde]
pub trait OptimizerConfig: OptimizerConfigClone + Send + Sync {
    fn create_optimizer(&mut self) -> Box<dyn Optimizer>;
    fn update_learning_rate(&mut self, learning_rate: f32);
    fn learning_rate(&self) -> f32;
}

#[typetag::serde]
impl OptimizerConfig for Box<dyn OptimizerConfig> {
    fn create_optimizer(&mut self) -> Box<dyn Optimizer> {
        (**self).create_optimizer()
    }

    fn update_learning_rate(&mut self, learning_rate: f32) {
        (**self).update_learning_rate(learning_rate)
    }

    fn learning_rate(&self) -> f32 {
        (**self).learning_rate()
    }
}

impl OptimizerConfigClone for Box<dyn OptimizerConfig> {
    fn clone_box(&self) -> Box<dyn OptimizerConfig> {
        (**self).clone_box() // Call clone_box on the inner concrete type
    }
}

pub trait OptimizerConfigClone {
    fn clone_box(&self) -> Box<dyn OptimizerConfig>;
}

impl Clone for Box<dyn OptimizerConfig> {
    fn clone(&self) -> Box<dyn OptimizerConfig> {
        self.clone_box()
    }
}

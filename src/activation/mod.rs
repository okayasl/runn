use crate::common::matrix::DenseMatrix;

use typetag;

#[typetag::serde]
pub trait ActivationFunction: ActivationFunctionClone + Send {
    fn forward(&mut self, input: &mut DenseMatrix);
    fn backward(&self, d_output: &DenseMatrix, input: &mut DenseMatrix);
    fn weight_initialization_function(&self) -> fn(usize, usize) -> f32;
}

pub trait ActivationFunctionClone {
    fn clone_box(&self) -> Box<dyn ActivationFunction>;
}

impl<T> ActivationFunctionClone for T
where
    T: 'static + ActivationFunction + Clone,
{
    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ActivationFunction> {
    fn clone(&self) -> Box<dyn ActivationFunction> {
        self.clone_box()
    }
}

fn he_initialization(_: usize, cols: usize) -> f32 {
    (2.0 / cols as f32).sqrt()
}

fn xavier_initialization(rows: usize, cols: usize) -> f32 {
    (2.0 / (rows as f32 + cols as f32)).sqrt()
}

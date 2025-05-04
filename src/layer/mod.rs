use crate::{matrix::DenseMatrix, random::Randomizer, ActivationFunction, Optimizer, Regularization, SummaryWriter};

pub mod dense_layer;

#[typetag::serde]
pub trait Layer: LayerClone + Send + Sync {
    fn forward(&self, input: &DenseMatrix) -> (DenseMatrix, DenseMatrix);
    fn backward(
        &self, d_output: &DenseMatrix, input: &DenseMatrix, pre_activated_output: &DenseMatrix,
        activated_output: &DenseMatrix,
    ) -> (DenseMatrix, DenseMatrix, DenseMatrix);
    fn activation_function(&self) -> &dyn ActivationFunction;
    fn regulate(
        &mut self, d_weights: &mut DenseMatrix, d_biases: &mut DenseMatrix, regularization: &Box<dyn Regularization>,
    );
    fn update(&mut self, d_weights: &DenseMatrix, d_biases: &DenseMatrix, epoch: usize);
    fn summarize(&self, epoch: usize, summary_writer: &mut dyn SummaryWriter);
    fn visualize(&self);
    fn input_output_size(&self) -> (usize, usize);
}

pub trait LayerClone {
    fn clone_box(&self) -> Box<dyn Layer>;
}

impl<T> LayerClone for T
where
    T: 'static + Layer + Clone,
{
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Layer> {
    fn clone(&self) -> Box<dyn Layer> {
        self.clone_box()
    }
}

pub trait LayerConfig {
    fn size(&self) -> usize;
    fn create_layer(
        &mut self, name: String, input_size: usize, optimizer: Box<dyn Optimizer>, randomizer: &Randomizer,
    ) -> Box<dyn Layer>;
}

impl LayerConfig for Box<dyn LayerConfig> {
    fn size(&self) -> usize {
        (**self).size() // Dereference the Box to call the method on the inner type
    }
    fn create_layer(
        &mut self, name: String, input_size: usize, optimizer: Box<dyn Optimizer>, randomizer: &Randomizer,
    ) -> Box<dyn Layer> {
        (**self).create_layer(name, input_size, optimizer, randomizer) // Dereference the Box to call the method on the inner type
    }
}

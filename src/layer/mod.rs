use dense_layer::DenseLayer;

use crate::{matrix::DenseMatrix, random::Randomizer, ActivationFunction, Optimizer, Regularization, SummaryWriter};

pub mod dense_layer;

#[typetag::serde]
pub trait Layer: LayerClone + Send + Sync {
    fn forward(&self, input: &DenseMatrix) -> (DenseMatrix, DenseMatrix);
    fn backward(
        &self, d_output: &DenseMatrix, input: &DenseMatrix, pre_activated_output: &mut DenseMatrix,
        activated_output: &DenseMatrix,
    ) -> (DenseMatrix, DenseMatrix, DenseMatrix);
    // fn get_params_and_grads(&mut self) -> ([&mut DenseMatrix; 2], [&mut DenseMatrix; 2]);
    // fn get_size(&self) -> usize;
    fn activation_function(&self) -> &dyn ActivationFunction;
    //fn reset(&mut self);
    fn regulate(
        &mut self, d_weights: &mut DenseMatrix, d_biases: &mut DenseMatrix, regularization: &Box<dyn Regularization>,
    );
    fn update(&mut self, d_weights: &DenseMatrix, d_biases: &DenseMatrix, epoch: usize);
    fn summarize(&self, epoch: usize, summary_writer: &mut dyn SummaryWriter);
    fn visualize(&self);
    fn get_input_output_size(&self) -> (usize, usize);
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
    fn get_size(&self) -> usize;
    fn create_layer(
        self: Box<Self>, name: String, input_size: usize, optimizer: Box<dyn Optimizer>, randomizer: &Randomizer,
    ) -> Box<dyn Layer>;
}

pub struct DenseConfig {
    pub size: usize,
    pub activation_function: Option<Box<dyn ActivationFunction>>,
}

impl LayerConfig for DenseConfig {
    fn get_size(&self) -> usize {
        self.size
    }
    fn create_layer(
        self: Box<Self>, name: String, input_size: usize, optimizer: Box<dyn Optimizer>, randomizer: &Randomizer,
    ) -> Box<dyn Layer> {
        Box::new(DenseLayer::new(name, input_size, self.size, self.activation_function.unwrap(), optimizer, randomizer))
    }
}

pub struct Dense {
    size: Option<usize>,
    activation_function: Option<Box<dyn ActivationFunction>>,
}

impl Dense {
    pub fn new() -> Self {
        Self {
            size: None,
            activation_function: None,
        }
    }

    pub fn size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    pub fn activation(mut self, activation_function: impl ActivationFunction + 'static) -> Self {
        self.activation_function = Some(Box::new(activation_function));
        self
    }

    pub(crate) fn from(mut self, size: usize, af: Box<dyn ActivationFunction>) -> Self {
        self.size = Some(size);
        self.activation_function = Some(af);
        self
    }

    pub fn build(self) -> DenseConfig {
        DenseConfig {
            size: self.size.expect("Size must be set"),
            activation_function: Some(self.activation_function.expect("Activation function must be set")),
        }
    }
}

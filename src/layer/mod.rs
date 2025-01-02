pub mod dense_layer;

use crate::{
    activation::activation_function::ActivationFunction,
    common::{matrix::DenseMatrix, random::Randomizer},
};

use super::dense_layer::DenseLayer;

#[typetag::serde]
pub trait Layer: LayerClone + Send {
    fn forward(&mut self, input: &DenseMatrix) -> DenseMatrix;
    fn backward(&mut self, d_output: &DenseMatrix, input: &DenseMatrix) -> DenseMatrix;
    fn get_params_and_grads(&mut self) -> ([&mut DenseMatrix; 2], [&mut DenseMatrix; 2]);
    fn get_size(&self) -> usize;
    fn get_activation_function(&self) -> &dyn ActivationFunction;
    fn reset(&mut self);
    fn visualize(&self, layer_name: &str);
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
    fn get_size(self) -> usize;
    fn create_layer(self: Box<Self>, input_size: usize, randomizer: &Randomizer) -> Box<dyn Layer>;
}

pub struct DenseConfig {
    pub size: usize,
    pub activation_function: Option<Box<dyn ActivationFunction>>,
}

impl LayerConfig for DenseConfig {
    fn get_size(self) -> usize {
        self.size
    }
    fn create_layer(self: Box<Self>, input_size: usize, randomizer: &Randomizer) -> Box<dyn Layer> {
        Box::new(DenseLayer::new(
            input_size,
            self.size,
            self.activation_function.unwrap(),
            randomizer,
        ))
    }
}

pub struct DenseConfigBuilder {
    size: Option<usize>,
    activation_function: Option<Box<dyn ActivationFunction>>,
}

impl DenseConfigBuilder {
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
            activation_function: Some(
                self.activation_function
                    .expect("Activation function must be set"),
            ),
        }
    }
}


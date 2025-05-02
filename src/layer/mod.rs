use dense_layer::DenseLayer;

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
        self: Box<Self>, name: String, input_size: usize, optimizer: Box<dyn Optimizer>, randomizer: &Randomizer,
    ) -> Box<dyn Layer>;
}

pub struct DenseConfig {
    pub size: usize,
    pub activation_function: Option<Box<dyn ActivationFunction>>,
}

impl LayerConfig for DenseConfig {
    fn size(&self) -> usize {
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

/// A builder for configuring a dense (fully connected) neural network layer.
///
/// This struct sets up a dense layer with a specified number of neurons and an activation function.
/// Default settings:
/// - size: None (must be set)
/// - activation_function: None (must be set)
impl Dense {
    pub fn new() -> Self {
        Self {
            size: None,
            activation_function: None,
        }
    }

    /// Set the number of neurons in the dense layer.
    ///
    /// Defines the output size of the layer (i.e., the number of neurons).
    /// # Parameters
    /// - `size`: Number of neurons in the layer (e.g., 64).
    pub fn size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// Set the activation function for the dense layer.
    ///
    /// Specifies the non-linear function applied to the layer’s output (e.g., ReLU, Sigmoid).
    /// # Parameters
    /// - `activation_function`: Activation function to apply (e.g., `ReLU`, `Sigmoid`).
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

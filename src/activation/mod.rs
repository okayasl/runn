pub mod elu;
pub mod gelu;
pub mod leaky_relu;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod softplus;
pub mod swish;
pub mod tanh;

use crate::common::matrix::DMat;

use typetag;

#[typetag::serde]
pub trait ActivationFunction: ActivationFunctionClone + Send + Sync {
    fn forward(&self, input: &mut DMat);
    fn backward(&self, d_output: &DMat, input: &mut DMat, output: &DMat);
    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        he_initialization
    }
}

#[typetag::serde]
impl ActivationFunction for Box<dyn ActivationFunction> {
    fn forward(&self, input: &mut DMat) {
        (**self).forward(input); // Dereference the Box to call the method
    }

    fn backward(&self, d_output: &DMat, input: &mut DMat, output: &DMat) {
        (**self).backward(d_output, input, output); // Dereference the Box to call the method
    }

    fn weight_initialization_factor(&self) -> fn(usize, usize) -> f32 {
        (**self).weight_initialization_factor() // Dereference the Box to call the method
    }
}

impl ActivationFunctionClone for Box<dyn ActivationFunction> {
    fn clone_box(&self) -> Box<dyn ActivationFunction> {
        (**self).clone_box() // Call clone_box on the inner concrete type
    }
}

pub trait ActivationFunctionClone {
    fn clone_box(&self) -> Box<dyn ActivationFunction>;
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

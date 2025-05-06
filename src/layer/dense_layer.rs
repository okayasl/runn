use log::{error, info};
use serde::{Deserialize, Serialize};
use std::fmt::Write;
use typetag;

use crate::{
    error::NetworkError, matrix::DenseMatrix, random::Randomizer, util, ActivationFunction, Optimizer, Regularization,
};

use super::{Layer, LayerConfig};

#[derive(Serialize, Deserialize, Clone)]
struct DenseLayer {
    name: String,
    input_size: usize,
    output_size: usize,
    weights: DenseMatrix,
    biases: DenseMatrix,
    activation: Box<dyn ActivationFunction>,
    optimizer: Box<dyn Optimizer>,
}

impl DenseLayer {
    pub(crate) fn new(
        name: String, input_size: usize, output_size: usize, activation: Box<dyn ActivationFunction>,
        mut optimizer: Box<dyn Optimizer>, randomizer: &Randomizer,
    ) -> Self {
        let mut weights = DenseMatrix::zeros(output_size, input_size);
        let biases = DenseMatrix::zeros(output_size, 1);
        // Initialize weights with random values
        weights.apply(|_| randomizer.float32() * activation.weight_initialization_factor()(output_size, input_size));
        optimizer.initialize(&weights, &biases);
        Self {
            name: name,
            input_size,
            output_size,
            weights,
            biases,
            optimizer: optimizer,
            activation,
        }
    }
}

#[typetag::serde]
impl Layer for DenseLayer {
    fn forward(&self, input: &DenseMatrix) -> (DenseMatrix, DenseMatrix) {
        let mut weighted_sum = DenseMatrix::mul_new(input, &self.weights.transpose());

        // Add the biases to the weighted sum using the Apply function
        // This approach avoids the need for broadcasting the biases and performs the addition in-place
        // The bias value (l.Biases.At(j, 0)) is added to each element of the weightedSum matrix
        weighted_sum.apply_with_indices(|_i, j, v| *v += self.biases.at(j, 0));

        let pre_activated_output = weighted_sum.clone();
        self.activation.forward(&mut weighted_sum);
        (weighted_sum, pre_activated_output)
    }

    fn backward(
        &self, d_output: &DenseMatrix, input: &DenseMatrix, pre_activated_output: &DenseMatrix,
        activated_output: &DenseMatrix,
    ) -> (DenseMatrix, DenseMatrix, DenseMatrix) {
        // Compute the gradient of the loss with respect to the activation (dZ)
        // This line computes the local gradient (also known as the derivative) of the loss
        // with respect to the pre-activation output of the dense layer.
        // This local gradient is used to update the weights and biases of the layer.

        let mut pre_activated_output = pre_activated_output.clone();
        //let pao: &mut DenseMatrix = pre_activated_output;
        self.activation
            .backward(d_output, &mut pre_activated_output, activated_output);

        // after backward method pao becomes gradient of activation function
        let act_grad = &pre_activated_output;

        let d_weights = DenseMatrix::mul_new(&act_grad.transpose(), input);

        // The gradient of the biases (dB) is computed by summing the gradients over the batch dimension,
        // resulting in the gradient of the loss with respect to
        // each bias being the sum of the corresponding gradient across the entire batch.
        // This operation is performed because the biases are shared across all the examples in a batch.
        let mut d_biases = DenseMatrix::zeros(self.biases.rows(), 1);
        d_biases.set_column_sum(act_grad);

        let d_input: DenseMatrix = DenseMatrix::mul_new(act_grad, &self.weights);
        (d_input, d_weights, d_biases)
    }

    fn regulate(
        &mut self, d_weights: &mut DenseMatrix, d_biases: &mut DenseMatrix, regularization: &Box<dyn Regularization>,
    ) {
        // Apply the single regularization technique
        regularization.apply(&mut [&mut self.weights, &mut self.biases], &mut [&mut *d_weights, &mut *d_biases]);
    }

    fn update(&mut self, d_weights: &DenseMatrix, d_biases: &DenseMatrix, epoch: usize) {
        self.optimizer
            .update(&mut self.weights, &mut self.biases, d_weights, d_biases, epoch);
    }

    fn activation_function(&self) -> &dyn ActivationFunction {
        &*self.activation
    }

    fn input_output_size(&self) -> (usize, usize) {
        (self.input_size, self.output_size)
    }

    fn visualize(&self) {
        info!("----- {} Layer (Dense) -----", self.name);
        info!("\nWeights:\n{}", format_matrix(&self.weights));
        info!("\nBiases:\n{}", format_matrix(&self.biases));
    }

    fn summarize(&self, epoch: usize, summary_writer: &mut dyn crate::summary::SummaryWriter) {
        if let Err(e) =
            summary_writer.write_histogram(&format!("{}-weights", self.name), epoch, &util::flatten(&self.weights))
        {
            error!("Failed to write weights histogram: {}", e);
        }

        if let Err(e) =
            summary_writer.write_histogram(&format!("{}-biases", self.name), epoch, &util::flatten(&self.biases))
        {
            error!("Failed to write biases histogram: {}", e);
        }
    }
}

/// Returns a pretty-printed single-matrix string with Unicode borders.
fn format_matrix(matrix: &DenseMatrix) -> String {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let mut out = String::with_capacity(rows * (cols * 10 + 4));

    for i in 0..rows {
        let borders = match (i, rows) {
            (0, 1) => ('[', ']'),
            (0, _) => ('⎡', '⎤'),
            (i, r) if i + 1 == r => ('⎣', '⎦'),
            _ => ('⎢', '⎥'),
        };
        out.push(borders.0);
        for j in 0..cols {
            write!(out, " {:9.6}", matrix.at(i, j)).unwrap();
        }
        out.push(borders.1);
        out.push('\n');
    }
    out
}

struct DenseConfig {
    pub(crate) size: usize,
    pub(crate) activation_function: Box<dyn ActivationFunction>,
}

impl LayerConfig for DenseConfig {
    fn size(&self) -> usize {
        self.size
    }
    fn create_layer(
        &mut self, name: String, input_size: usize, optimizer: Box<dyn Optimizer>, randomizer: &Randomizer,
    ) -> Box<dyn Layer> {
        Box::new(DenseLayer::new(name, input_size, self.size, self.activation_function.clone(), optimizer, randomizer))
    }
}

pub struct Dense {
    size: usize,
    activation_function: Result<Box<dyn ActivationFunction>, NetworkError>,
}

/// A builder for configuring a dense (fully connected) neural network layer.
///
/// This struct sets up a dense layer with a specified number of neurons and an activation function.
impl Dense {
    /// Creates a new Dense layer builder with default settings.
    /// - size: 0 (must be set)
    /// - activation_function: Error (must be set)
    pub fn new() -> Self {
        Self {
            size: 0,
            activation_function: Err(NetworkError::ConfigError(
                "Activation function must be specified for Dense Layer.".to_string(),
            )),
        }
    }

    /// Set the number of neurons in the dense layer.
    ///
    /// Defines the output size of the layer (i.e., the number of neurons).
    /// # Parameters
    /// - `size`: Number of neurons in the layer (e.g., 64).
    pub fn size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    /// Set the activation function for the dense layer.
    ///
    /// Specifies the non-linear function applied to the layer’s output (e.g., ReLU, Sigmoid).
    /// # Parameters
    /// - `activation_function`: Activation function to apply (e.g., `ReLU`, `Sigmoid`).
    pub fn activation(mut self, activation_function: Result<Box<dyn ActivationFunction>, NetworkError>) -> Self {
        self.activation_function = activation_function;
        self
    }

    pub(crate) fn from(mut self, size: usize, af: Box<dyn ActivationFunction>) -> Self {
        self.size = size;
        self.activation_function = Ok(af);
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.size == 0 {
            return Err(NetworkError::ConfigError("Dense layer size must be greater than 0".to_string()));
        }
        if self.activation_function.is_err() {
            return Err(NetworkError::ConfigError("Dense layer activation function must be set".to_string()));
        }
        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn LayerConfig>, NetworkError> {
        self.validate()?;
        Ok(Box::new(DenseConfig {
            size: self.size,
            activation_function: self.activation_function?,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adam::Adam;
    use crate::common::matrix::DenseMatrix;
    use crate::random::Randomizer;
    use crate::relu::ReLU;
    use crate::sigmoid::Sigmoid;
    use crate::OptimizerConfig;

    #[test]
    fn test_dense_layer_forward() {
        let randomizer = Randomizer::new(Some(42));
        let activation = ReLU::new().unwrap();
        let optimizer_config = Adam::new()
            .learning_rate(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .build()
            .unwrap();

        let layer =
            DenseLayer::new("layer".to_owned(), 3, 2, activation, optimizer_config.create_optimizer(), &randomizer);

        let input = DenseMatrix::new(1, 3, &[1.0, 2.0, 3.0]);
        let (output, pre_activated_output) = layer.forward(&input);

        assert_eq!(output.rows(), 1);
        assert_eq!(output.cols(), 2);
        assert_eq!(pre_activated_output.rows(), 1);
        assert_eq!(pre_activated_output.cols(), 2);
    }

    #[test]
    fn test_dense_layer_backward() {
        let randomizer = Randomizer::new(Some(42));
        let activation = Sigmoid::new().unwrap();
        let optimizer_config = Adam::new()
            .learning_rate(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .build()
            .unwrap();

        let layer =
            DenseLayer::new("layer".to_owned(), 3, 2, activation, optimizer_config.create_optimizer(), &randomizer);

        let input = DenseMatrix::new(1, 3, &[1.0, 2.0, 3.0]);
        let (output, mut pre_activated_output) = layer.forward(&input);

        let d_output = DenseMatrix::new(1, 2, &[0.1, 0.2]);
        let (d_input, d_weights, d_biases) = layer.backward(&d_output, &input, &mut pre_activated_output, &output);

        assert_eq!(d_input.rows(), 1);
        assert_eq!(d_input.cols(), 3);
        assert_eq!(d_weights.rows(), 2);
        assert_eq!(d_weights.cols(), 3);
        assert_eq!(d_biases.rows(), 2);
        assert_eq!(d_biases.cols(), 1);
    }

    #[test]
    fn test_dense_layer_update() {
        let randomizer = Randomizer::new(Some(42));
        let activation = ReLU::new().unwrap();
        let optimizer_config = Adam::new()
            .learning_rate(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .build()
            .unwrap();

        let mut layer =
            DenseLayer::new("layer".to_owned(), 3, 2, activation, optimizer_config.create_optimizer(), &randomizer);

        let input = DenseMatrix::new(1, 3, &[1.0, 2.0, 3.0]);
        let (output, mut pre_activated_output) = layer.forward(&input);

        let d_output = DenseMatrix::new(1, 2, &[0.1, 0.2]);
        let (_d_input, d_weights, d_biases) = layer.backward(&d_output, &input, &mut pre_activated_output, &output);

        layer.update(&d_weights, &d_biases, 1);
    }

    #[test]
    fn test_dense_validate() {
        let dense = Dense::new().size(10).activation(Ok(Box::new(ReLU::new().unwrap())));
        assert!(dense.validate().is_ok());

        let dense_invalid = Dense::new().size(0).activation(Ok(Box::new(ReLU::new().unwrap())));
        assert!(dense_invalid.validate().is_err());

        let dense_invalid_activation = Dense::new().size(10).activation(Err(NetworkError::ConfigError(
            "Activation function must be specified for Dense Layer.".to_string(),
        )));
        assert!(dense_invalid_activation.validate().is_err());
    }
}

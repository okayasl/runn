use log::info;
use serde::{Deserialize, Serialize};
use typetag;

use crate::{matrix::DenseMatrix, random::Randomizer, util, ActivationFunction, Optimizer, Regularization};

use super::Layer;

#[derive(Serialize, Deserialize, Clone)]
pub struct DenseLayer {
    name: String,
    input_size: usize,
    output_size: usize,
    weights: DenseMatrix,
    biases: DenseMatrix,
    //d_weights: DenseMatrix,
    //d_biases: DenseMatrix,
    activation: Box<dyn ActivationFunction>,
    optimizer: Box<dyn Optimizer>,
    //pre_activated_output: Option<DenseMatrix>,
}

impl DenseLayer {
    pub(crate) fn new(
        name: String,
        input_size: usize,
        output_size: usize,
        activation: Box<dyn ActivationFunction>,
        mut optimizer: Box<dyn Optimizer>,
        randomizer: &Randomizer,
    ) -> Self {
        let mut weights = DenseMatrix::zeros(output_size, input_size);
        let biases = DenseMatrix::zeros(output_size, 1);
        //let std_dev = activation.weight_initialization_factor()(weights.rows(), weights.cols());
        weights.apply(|_| {
            randomizer.float32()
                * activation.weight_initialization_factor()(output_size, input_size)
        }); // Initialize weights with random values
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
    fn forward(&mut self, input: &DenseMatrix) -> (DenseMatrix, DenseMatrix) {
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
        &mut self,
        d_output: &DenseMatrix,
        input: &DenseMatrix,
        pre_activated_output: &mut DenseMatrix,
    ) -> (DenseMatrix, DenseMatrix, DenseMatrix) {
        // Compute the gradient of the loss with respect to the activation (dZ)
        // This line computes the local gradient (also known as the derivative) of the loss
        // with respect to the pre-activation output of the dense layer.
        // This local gradient is used to update the weights and biases of the layer.
        //let pao: &mut DenseMatrix = pre_activated_output;
        self.activation.backward(d_output, pre_activated_output);

        // after backward method pao becomes gradient of activation function
        let act_grad = pre_activated_output;

        let d_weights = DenseMatrix::mul_new(&act_grad.transpose(), input);

        // The gradient of the weights (dWeights) calculated in the previous step is added to the weights' gradient accumulator (DWeights).
        // This addition is performed to accumulate gradients when dealing with minibatches.
        // Once all the minibatches are processed, the gradients will be used to update the weights.
        //self.d_weights.add(&d_weights);

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
        &mut self,
        d_weights: &mut DenseMatrix,
        d_biases: &mut DenseMatrix,
        regularization: &Box<dyn Regularization>,
    ) {
        // Apply the single regularization technique
        regularization.apply(
            &mut [&mut self.weights, &mut self.biases],
            &mut [&mut *d_weights, &mut *d_biases],
        );
    }

    fn update(&mut self, d_weights: &DenseMatrix, d_biases: &DenseMatrix, epoch: usize) {
        self.optimizer.update(
            &mut self.weights,
            &mut self.biases,
            d_weights,
            d_biases,
            epoch,
        );
    }
    // fn get_parameters_and_gradients(&mut self) -> (Vec<&mut DenseMatrix>, Vec<&mut DenseMatrix>) {
    //     (
    //         vec![&mut self.weights, &mut self.biases],
    //         vec![&mut self.d_weights, &mut self.d_biases],
    //     )
    // }

    // fn get_params_and_grads(&mut self) -> ([&mut DenseMatrix; 2], [&mut DenseMatrix; 2]) {
    //     (
    //         [&mut self.weights, &mut self.biases],
    //         [&mut self.d_weights, &mut self.d_biases],
    //     )
    // }

    // fn get_size(&self) -> usize {
    //     self.weights.rows()
    // }

    fn activation_function(&self) -> &dyn ActivationFunction {
        &*self.activation
    }

    // fn reset(&mut self) {
    //     self.d_weights.zero();
    //     self.d_biases.zero();
    // }

    fn get_input_output_size(&self) -> (usize, usize) {
        (self.input_size, self.output_size)
    }
    fn visualize(&self) {
        info!("----- {} Layer (Dense) -----", self.name);
        info!("Weights: {}", util::format_matrix(&self.weights));
        info!("Biases: {}", util::format_matrix(&self.biases));
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
        let activation = Box::new(ReLU::new());
        let optimizer_config = Adam::new()
            .learning_rate(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .build();

        let mut layer = DenseLayer::new("layer".to_owned(),3, 2, activation, Box::new(optimizer_config).create_optimizer(), &randomizer);

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
        let activation = Box::new(Sigmoid::new());
        let optimizer_config = Adam::new()
            .learning_rate(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .build();

        let mut layer = DenseLayer::new("layer".to_owned(),3, 2, activation, Box::new(optimizer_config).create_optimizer(), &randomizer);

        let input = DenseMatrix::new(1, 3, &[1.0, 2.0, 3.0]);
        let (_output, mut pre_activated_output) = layer.forward(&input);

        let d_output = DenseMatrix::new(1, 2, &[0.1, 0.2]);
        let (d_input, d_weights, d_biases) =
            layer.backward(&d_output, &input, &mut pre_activated_output);

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
        let activation = Box::new(ReLU::new());
        let optimizer_config = Adam::new()
            .learning_rate(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .build();

        let mut layer = DenseLayer::new("layer".to_owned(),3, 2, activation, Box::new(optimizer_config).create_optimizer(), &randomizer);

        let input = DenseMatrix::new(1, 3, &[1.0, 2.0, 3.0]);
        let (_output, mut pre_activated_output) = layer.forward(&input);

        let d_output = DenseMatrix::new(1, 2, &[0.1, 0.2]);
        let (_d_input, d_weights, d_biases) =
            layer.backward(&d_output, &input, &mut pre_activated_output);

        layer.update(&d_weights, &d_biases, 1);

        // Add assertions to verify the updated weights and biases
        // For example:
        // assert_eq!(layer.weights, expected_weights);
        // assert_eq!(layer.biases, expected_biases);
    }
}

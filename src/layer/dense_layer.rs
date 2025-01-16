use log::info;
use serde::{Deserialize, Serialize};
use typetag;

use crate::{matrix::DenseMatrix, random::Randomizer, util, ActivationFunction, Optimizer};

use super::Layer;

#[derive(Serialize, Deserialize, Clone)]
pub struct DenseLayer {
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

    // fn get_activation_function(&self) -> &dyn ActivationFunction {
    //     &*self.activation
    // }

    // fn reset(&mut self) {
    //     self.d_weights.zero();
    //     self.d_biases.zero();
    // }
    fn visualize(&self, layer_name: &str) {
        info!("----- {} Layer (Dense) -----", layer_name);
        info!("Weights: {}", util::format_matrix(&self.weights));
        info!("Biases: {}", util::format_matrix(&self.biases));
    }
}

// fn init_weights(
//     weights: &mut DenseMatrix,
//     randomizer: &Randomizer,
//     activation: &Box<dyn ActivationFunction>,
// ) {
//     let std_dev = match activation.activation_type() {
//         ActivationType::RELU
//         | ActivationType::GELU
//         | ActivationType::ELU
//         | ActivationType::LEAKYRELU
//         | ActivationType::SWISH => (2.0 / weights.cols() as f32).sqrt(),

//         ActivationType::TANH
//         | ActivationType::SIGMOID
//         | ActivationType::SOFTMAX => (2.0 / (weights.rows() as f32 + weights.cols() as f32)).sqrt(),

//         ActivationType::LINEAR => 1.0,
//         _ => 1.0,
//     };
//     weights.apply(|_| randomizer.float32() * activation.weight_initialization_factor());
// }

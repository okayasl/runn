use std::error::Error;

use log::{error, info};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator},
    ThreadPoolBuilder,
};

use crate::{
    dropout::DropoutRegularization,
    l1::L1Regularization,
    l2::L2Regularization,
    layer::{Layer, LayerConfig},
    matrix::DenseMatrix,
    metric::{calculate_confusion_matrix, calculate_f1_score, calculate_precision, calculate_recall},
    random::Randomizer,
    regularization::Regularization,
    util::{self},
    EarlyStopper, LossFunction, OptimizerConfig, SummaryWriter,
};

use super::network_io::{load_network, save_network, NetworkIO, SerializationFormat};

pub struct NetworkBuilder {
    input_size: usize,
    output_size: usize,
    layer_configs: Vec<Box<dyn LayerConfig>>,
    loss_function: Option<Box<dyn LossFunction>>,
    optimizer_config: Option<Box<dyn OptimizerConfig>>,
    regularization: Vec<Box<dyn Regularization>>,
    batch_size: usize,
    batch_group_size: usize,
    epochs: usize,
    clip_threshold: f32,
    seed: u64,
    early_stopper: Option<Box<dyn EarlyStopper>>,
    debug: bool,
    normalized: bool,
    summary_writer: Option<Box<dyn SummaryWriter>>,
    parallelize: usize,
}

impl NetworkBuilder {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            layer_configs: Vec::new(),
            loss_function: None,
            optimizer_config: None,
            regularization: Vec::new(),
            batch_size: usize::MAX,
            batch_group_size: 1,
            epochs: 0,
            clip_threshold: 0.0,
            seed: 0,
            early_stopper: None,
            debug: false,
            normalized: false,
            summary_writer: None,
            parallelize: 1,
        }
    }

    pub fn layer(mut self, layer_config: impl LayerConfig + 'static) -> Self {
        self.layer_configs.push(Box::new(layer_config));
        self
    }

    pub fn regularization(mut self, reg: impl Regularization + 'static) -> Self {
        self.regularization.push(Box::new(reg));
        self
    }

    pub fn loss_function(mut self, loss_function: impl LossFunction + 'static) -> Self {
        self.loss_function = Some(Box::new(loss_function));
        self
    }

    pub fn optimizer(mut self, optimizer_config: impl OptimizerConfig + 'static) -> Self {
        self.optimizer_config = Some(Box::new(optimizer_config));
        self
    }

    pub fn early_stopper(mut self, early_stopper: impl EarlyStopper + 'static) -> Self {
        self.early_stopper = Some(Box::new(early_stopper));
        self
    }

    pub fn batch_group_size(mut self, batch_group_size: usize) -> Self {
        self.batch_group_size = batch_group_size;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn clip_threshold(mut self, clip_threshold: f32) -> Self {
        self.clip_threshold = clip_threshold;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn normalize(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    pub fn summary(mut self, summary_writer: Box<dyn SummaryWriter>) -> Self {
        self.summary_writer = Some(summary_writer);
        self
    }

    pub fn parallelize(mut self, paralelize: usize) -> Self {
        self.parallelize = paralelize;
        self
    }

    pub(crate) fn from_network(mut self, nw: &Network) -> Self {
        self.loss_function = Some(nw.loss_function.clone());
        self.optimizer_config = Some(nw.optimizer_config.clone());
        self.batch_size = nw.batch_size;
        self.batch_group_size = nw.batch_group_size;
        self.epochs = nw.epochs;
        self.seed = nw.seed;
        self.clip_threshold = nw.clip_threshold;
        self.debug = nw.debug;
        self.parallelize = nw.parallelize;

        if let Some(early_stopper) = &nw.early_stopper {
            self.early_stopper = Some(early_stopper.clone());
        }

        if let Some(summary_writer) = &nw.summary_writer {
            self.summary_writer = Some(summary_writer.clone());
        }

        self.regularization = nw.regularization.iter().map(|reg| (**reg).clone_box()).collect();

        self
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.input_size == 0 || self.output_size == 0 {
            return Err("Input and output sizes must be greater than zero".into());
        }
        if self.loss_function.is_none() {
            return Err("Loss function is not set".into());
        }
        if self.optimizer_config.is_none() {
            return Err("Optimizer is not set".into());
        }
        if self.epochs == 0 {
            return Err("Epochs must be greater than zero".into());
        }
        if self.parallelize <= 0 {
            return Err("Parallelization factor must be greater than zero".into());
        }
        Ok(())
    }

    pub fn build(self) -> Result<Network, Box<dyn Error>> {
        self.validate()?;

        let randomizer = Randomizer::new(Some(self.seed));
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        let mut input_size = self.input_size; // Initialize with input_size
        let layer_size = self.layer_configs.len();
        for (i, layer_config) in self.layer_configs.into_iter().enumerate() {
            let size = layer_config.get_size(); // Get size via &self
            let mut name = format!("Hidden {}", i);
            if i == layer_size - 1 {
                name = String::from("Output");
            }
            let layer = layer_config.create_layer(
                name,
                input_size,
                self.optimizer_config.as_ref().unwrap().clone().create_optimizer(),
                &randomizer,
            );
            input_size = size; // Update input_size for the next layer
            layers.push(layer);
        }
        // configure_global_thread_pool(self.parallelize);
        Ok(Network {
            input_size: self.input_size,
            output_size: self.output_size,
            layers,
            loss_function: self.loss_function.unwrap(),
            optimizer_config: self.optimizer_config.unwrap(),
            regularization: self.regularization,
            batch_size: self.batch_size,
            batch_group_size: self.batch_group_size,
            epochs: self.epochs,
            clip_threshold: self.clip_threshold,
            seed: self.seed,
            early_stopper: self.early_stopper,
            debug: self.debug,
            normalized: self.normalized,
            mins: None,
            maxs: None,
            randomizer,
            search: false,
            summary_writer: self.summary_writer,
            parallelize: self.parallelize,
        })
    }
}

pub struct NetworkResult {
    pub predictions: DenseMatrix,
    pub accuracy: f32,
    pub loss: f32,
    pub metrics: Metrics,
}

pub struct Metrics {
    pub micro_precision: f32,
    pub micro_recall: f32,
    pub macro_f1_score: f32,
    pub micro_f1_score: f32,
    pub metrics_by_class: Vec<Metric>,
}

pub struct Metric {
    pub f1_score: f32,
    pub recall: f32,
    pub precision: f32,
}

pub struct Network {
    pub(crate) input_size: usize,
    pub(crate) output_size: usize,
    pub(crate) layers: Vec<Box<dyn Layer>>,
    pub(crate) loss_function: Box<dyn LossFunction>,
    pub(crate) optimizer_config: Box<dyn OptimizerConfig>,
    pub(crate) regularization: Vec<Box<dyn Regularization>>,
    pub(crate) batch_size: usize,
    pub(crate) batch_group_size: usize,
    pub(crate) epochs: usize,
    pub(crate) clip_threshold: f32,
    pub(crate) seed: u64,
    pub(crate) randomizer: Randomizer,
    pub(crate) debug: bool,
    pub(crate) normalized: bool,
    pub(crate) mins: Option<Vec<f32>>,
    pub(crate) maxs: Option<Vec<f32>>,
    pub(crate) search: bool,
    pub(crate) early_stopper: Option<Box<dyn EarlyStopper>>,
    pub(crate) summary_writer: Option<Box<dyn SummaryWriter>>,
    pub(crate) parallelize: usize,
}

impl Network {
    pub fn train(&mut self, inputs: &DenseMatrix, targets: &DenseMatrix) -> Result<NetworkResult, Box<dyn Error>> {
        let training_inputs = self.prepare_inputs(inputs);
        let sample_size = training_inputs.rows();
        self.log_start_info(sample_size);
        let mut shuffled_inputs = DenseMatrix::zeros(sample_size, self.input_size);
        let mut shuffled_targets = DenseMatrix::zeros(sample_size, self.output_size);

        self.initialize_threadpool();
        self.init_summary_writer();
        let mut last_epoch = 0;
        for epoch in 1..=self.epochs {
            last_epoch = epoch;
            self.visualize_layers();
            self.shuffle(inputs, targets, &mut shuffled_inputs, &mut shuffled_targets);
            let (batch_inputs, batch_targets) = self.create_batches(&shuffled_inputs, &shuffled_targets);
            let (group_inputs, group_targets) = self.create_groups(&batch_inputs, &batch_targets);

            for (grp_id, (grp_inputs, grp_targets)) in group_inputs.iter().zip(group_targets.iter()).enumerate() {
                let (grp_predictions, mut group_layer_inputs) = self.forward(grp_inputs);
                self.log_group_training_info(
                    epoch,
                    group_inputs.len(),
                    grp_id,
                    grp_inputs,
                    grp_targets,
                    &grp_predictions,
                );
                self.backward(&grp_predictions, grp_targets, &mut group_layer_inputs, epoch);
            }

            let epoch_result = self.predict(&training_inputs, targets);
            let epoch_accuracy = epoch_result.accuracy;
            let epoch_loss = self.loss_function.forward(&epoch_result.predictions, targets);
            self.log_epoch_training_info(epoch, epoch_accuracy, epoch_loss);
            self.summarize(epoch, epoch_loss, epoch_accuracy);
            if self.early_stopped(epoch, epoch_loss, epoch_accuracy) {
                break;
            }
        }

        let final_result = self.predict(&training_inputs, targets);
        self.log_finish_info(last_epoch, &final_result);
        self.close_summary_writer();
        Ok(final_result)
    }
    
    fn forward(&self, batch_inputs: &[DenseMatrix]) -> (Vec<DenseMatrix>, Vec<Vec<LayerParams>>) {
        let results: Vec<(DenseMatrix, Vec<LayerParams>)> = batch_inputs
            .par_iter()
            .map(|input| {
                let mut layer_params = Vec::with_capacity(self.layers.len());
                let mut current_input: DenseMatrix = input.clone();
                for layer in &self.layers {
                    // Get both the activated output and the pre-activated output from the layer
                    let (mut activated_output, pre_activated_output) = layer.forward(&current_input);
                    // Apply forward regularization
                    for reg in &self.regularization {
                        if let Some(dropout) = reg.as_any().downcast_ref::<DropoutRegularization>() {
                            dropout.apply(&mut [&mut activated_output], &mut Vec::new());
                        }
                    }
                    // Store the layer input and pre-activated output in LayerParams
                    layer_params.push(LayerParams::new(
                        current_input.clone(),
                        pre_activated_output.clone(),
                        activated_output.clone(),
                    ));
                    // Update the current input for the next layer
                    current_input = activated_output;
                }
                // Return the final activated output and the layer parameters for this input
                (current_input, layer_params)
            })
            .collect();

        // Split the results into batch_predictions and batch_layer_params
        let (batch_predictions, batch_layer_params): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        (batch_predictions, batch_layer_params)
    }

    fn backward(
        &mut self,
        predicteds: &[DenseMatrix],                  // Predictions for each batch
        targets: &[DenseMatrix],                     // Targets for each batch
        batch_layer_params: &mut [Vec<LayerParams>], // LayerParams for each batch
        epoch: usize,
    ) {
        // Initialize aggregated gradients for weights and biases
        let mut aggregated_d_weights = Vec::new();
        let mut aggregated_d_biases = Vec::new();

        for layer in &self.layers {
            let (input_size, output_size) = layer.get_input_output_size();
            aggregated_d_weights.push(DenseMatrix::zeros(output_size, input_size)); // Initialize weights
            aggregated_d_biases.push(DenseMatrix::zeros(output_size, 1)); // Initialize biases
        }

        let layer_idx_length = self.layers.len() - 1;

        // Parallelize the computation of gradients for each batch
        let batch_gradients: Vec<(Vec<DenseMatrix>, Vec<DenseMatrix>)> = predicteds
            .par_iter()
            .zip(targets.par_iter())
            .zip(batch_layer_params.par_iter_mut())
            .map(|((predicted, target), layer_params)| {
                let mut d_output = self.loss_function.backward(predicted, target);

                let mut batch_d_weights = Vec::new();
                let mut batch_d_biases = Vec::new();

                // Initialize gradients for this batch
                for layer in &self.layers {
                    let (input_size, output_size) = layer.get_input_output_size();
                    batch_d_weights.push(DenseMatrix::zeros(output_size, input_size));
                    batch_d_biases.push(DenseMatrix::zeros(output_size, 1));
                }

                // Iterate over the layers in reverse order
                for (i, (layer, params)) in self.layers.iter().rev().zip(layer_params.iter_mut().rev()).enumerate() {
                    let (d_input, d_weights, d_biases) = layer.backward(
                        &d_output,
                        &params.layer_input,
                        &mut params.pre_activated_output,
                        &params.activated_output,
                    );

                    let grad_i = layer_idx_length - i;

                    // Accumulate gradients for this batch
                    batch_d_weights[grad_i].add(&d_weights);
                    batch_d_biases[grad_i].add(&d_biases);

                    // Propagate the gradient to the previous layer
                    d_output = d_input;
                }

                (batch_d_weights, batch_d_biases)
            })
            .collect();

        // Aggregate gradients across all batches
        for (batch_d_weights, batch_d_biases) in batch_gradients {
            for (i, (d_weights, d_biases)) in batch_d_weights.into_iter().zip(batch_d_biases).enumerate() {
                aggregated_d_weights[i].add(&d_weights);
                aggregated_d_biases[i].add(&d_biases);
            }
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Regulate gradients for the current layer
            for reg in &self.regularization {
                if let Some(_l1) = reg.as_any().downcast_ref::<L1Regularization>() {
                    layer.regulate(&mut aggregated_d_weights[i], &mut aggregated_d_biases[i], reg);
                }
                if let Some(_l2) = reg.as_any().downcast_ref::<L2Regularization>() {
                    layer.regulate(&mut aggregated_d_weights[i], &mut aggregated_d_biases[i], reg);
                }
            }

            // Clip gradients for the current layer
            let gradients = vec![&mut aggregated_d_weights[i], &mut aggregated_d_biases[i]];
            if self.clip_threshold > 0.0 {
                for grad in gradients {
                    grad.clip(self.clip_threshold);
                }
            }

            // Update the parameters for the current layer
            layer.update(&aggregated_d_weights[i], &aggregated_d_biases[i], epoch);
            if self.debug {
                layer.visualize();
            }
        }
    }

    pub fn predict(&mut self, inputs: &DenseMatrix, targets: &DenseMatrix) -> NetworkResult {
        let mut output = inputs.clone();
        for layer in &mut self.layers {
            (output, _) = layer.forward(&output);
        }

        let accuracy = util::calculate_accuracy(&output, targets);
        let loss = self.loss_function.forward(&output, targets);
        let metrics = calculate_metrics(targets, &output);

        NetworkResult {
            predictions: output,
            accuracy,
            loss,
            metrics,
        }
    }

    fn initialize_threadpool(&mut self) {
        if self.parallelize <= 0 || self.search{
            return;
        }
        ThreadPoolBuilder::new()
            .num_threads(self.parallelize)
            .build_global()
            .expect("Failed to build global thread pool");
    }

    fn close_summary_writer(&mut self) {
        if let Some(summary_writer) = self.summary_writer.as_mut() {
            if let Err(e) = summary_writer.close() {
                error!("Failed to close summary writer: {}", e);
            }
        }
    }

    pub fn reset(&mut self) {
        if let Some(early_stopper) = &mut self.early_stopper {
            early_stopper.reset();
        }
    }

    pub fn visualize_layers(&mut self) {
        if !self.debug {
            return;
        }
        self.layers.iter().for_each(|layer| {
            layer.visualize();
        });
    }

    fn shuffle(
        &self, inputs: &DenseMatrix, targets: &DenseMatrix, shuffling_inputs: &mut DenseMatrix,
        shuffling_targets: &mut DenseMatrix,
    ) {
        let sample_size = inputs.rows();
        let shuffle_indices = self.randomizer.perm(sample_size);

        shuffle_indices.iter().enumerate().for_each(|(i, &idx)| {
            shuffling_inputs.set_row(i, &inputs.get_row(idx));
            shuffling_targets.set_row(i, &targets.get_row(idx));
        });
    }

    fn prepare_inputs(&mut self, inputs: &DenseMatrix) -> DenseMatrix {
        let mut training_inputs = inputs.clone();
        if self.normalized {
            let (mins, maxs) = util::find_min_max(&inputs);
            util::normalize_in_place(&mut training_inputs, &mins, &maxs);
            self.mins = Some(mins);
            self.maxs = Some(maxs);
        }
        training_inputs
    }

    fn create_batches<'a>(
        &mut self, inputs: &'a DenseMatrix, targets: &'a DenseMatrix,
    ) -> (Vec<DenseMatrix>, Vec<DenseMatrix>) {
        let sample_size = inputs.rows();
        let (batch_size, batch_count) = self.calculate_batches(sample_size);
        self.get_all_batch_inputs_targets(sample_size, batch_size, batch_count, inputs, targets)
    }

    fn log_epoch_training_info(&mut self, epoch: usize, epoch_accuracy: f32, epoch_loss: f32) {
        if self.debug {
            info!("Epoch [{}/{}], Loss:{:.4}, Accuracy:{:.2}%", epoch, self.epochs, epoch_loss, epoch_accuracy * 100.0);
        }
    }

    fn log_group_training_info(
        &mut self, epoch: usize, group_count: usize, group_id: usize, group_batch_inputs: &&[DenseMatrix],
        group_batch_targets: &&[DenseMatrix], group_predictions: &Vec<DenseMatrix>,
    ) {
        if self.debug {
            // Optionally compute the loss and accuracy for this group.
            let group_losses = self.forward_loss(group_predictions, group_batch_targets);
            let ave_group_loss: f32 = group_losses.iter().sum::<f32>() / group_losses.len() as f32;
            let ave_group_accuracy =
                calculate_group_accuracy(group_batch_inputs, group_batch_targets, group_predictions);
            if !self.search {
                info!(
                    "Epoch [{}/{}], Group [{}/{}], Avg Group Loss: {:.4}, Avg Group Accuracy: {:.2}%",
                    epoch,
                    self.epochs,
                    group_id,
                    group_count,
                    ave_group_loss,
                    ave_group_accuracy * 100.0
                );
            }
        }
    }

    fn log_finish_info(&mut self, last_epoch: usize, final_result: &NetworkResult) {
        if !self.search {
            info!(
                "Network training finished: epoch:{}, Loss:{:.4}, Accuracy:{:.2}%",
                last_epoch,
                final_result.loss,
                final_result.accuracy * 100.0
            );
        }
    }

    fn log_start_info(&mut self, sample_size: usize) {
        let (batch_size, batch_count) = self.calculate_batches(sample_size);
        let group_count = (batch_count + self.batch_group_size - 1) / self.batch_group_size;

        if !self.search {
            info!("Network training started: sample_size:{}, group_size:{}, group_count:{}, batch_size:{}, batch_count:{}, epoch:{}", sample_size,group_count,self.batch_group_size, batch_size, batch_count, self.epochs);
        }
    }

    fn create_groups<'a>(
        &self, all_batch_inputs: &'a [DenseMatrix], all_batch_targets: &'a [DenseMatrix],
    ) -> (Vec<&'a [DenseMatrix]>, Vec<&'a [DenseMatrix]>) {
        let batch_group_size = self.batch_group_size;
        let batch_count = all_batch_inputs.len();
        let mut all_group_batch_inputs = Vec::with_capacity((batch_count + batch_group_size - 1) / batch_group_size);
        let mut all_group_batch_targets = Vec::with_capacity((batch_count + batch_group_size - 1) / batch_group_size);

        for group_start in (0..batch_count).step_by(batch_group_size) {
            let group_end = std::cmp::min(group_start + batch_group_size, batch_count);

            // Use slices instead of copying data
            let group_batch_inputs = &all_batch_inputs[group_start..group_end];
            let group_batch_targets = &all_batch_targets[group_start..group_end];

            all_group_batch_inputs.push(group_batch_inputs);
            all_group_batch_targets.push(group_batch_targets);
        }

        (all_group_batch_inputs, all_group_batch_targets)
    }

    fn calculate_batches(&mut self, sample_size: usize) -> (usize, usize) {
        let batch_size = if self.batch_size > 0 {
            std::cmp::min(self.batch_size, sample_size)
        } else {
            sample_size
        };

        let batch_count = (sample_size + batch_size - 1) / batch_size; // Round up
        (batch_size, batch_count)
    }

    fn get_all_batch_inputs_targets(
        &mut self, sample_size: usize, batch_size: usize, batch_count: usize, shuffled_inputs: &DenseMatrix,
        shuffled_targets: &DenseMatrix,
    ) -> (Vec<DenseMatrix>, Vec<DenseMatrix>) {
        let mut all_batch_inputs = Vec::with_capacity(batch_count);
        let mut all_batch_targets = Vec::with_capacity(batch_count);

        for start_idx in (0..sample_size).step_by(batch_size) {
            let end_idx = std::cmp::min(start_idx + batch_size, sample_size);

            let batch_inputs = shuffled_inputs.slice(start_idx, end_idx, 0, self.input_size);
            let batch_targets = shuffled_targets.slice(start_idx, end_idx, 0, self.output_size);

            all_batch_inputs.push(batch_inputs);
            all_batch_targets.push(batch_targets);
        }
        (all_batch_inputs, all_batch_targets)
    }

    fn forward_loss(&mut self, batch_predictions: &[DenseMatrix], batch_targets: &[DenseMatrix]) -> Vec<f32> {
        let mut all_losses = Vec::with_capacity(batch_predictions.len());

        for (predicted, target) in batch_predictions.iter().zip(batch_targets.iter()) {
            let loss = self.loss_function.forward(predicted, target);
            all_losses.push(loss);
        }

        all_losses
    }

    fn summarize(&mut self, epoch: usize, epoch_loss: f32, epoch_accuracy: f32) {
        if self.search {
            return;
        }
        if let Some(summary_writer) = &mut self.summary_writer {
            summary_writer
                .write_scalar("Training/Loss", epoch, epoch_loss as f32)
                .unwrap();
            summary_writer
                .write_scalar("Training/Accuracy", epoch, epoch_accuracy as f32)
                .unwrap();
            for layer in &self.layers {
                layer.summarize(epoch, &mut **summary_writer);
            }
        }
    }

    fn early_stopped(&mut self, epoch: usize, val_loss: f32, val_accuracy: f32) -> bool {
        if let Some(early_stopper) = &mut self.early_stopper {
            early_stopper.update(epoch, val_loss, val_accuracy);
            if early_stopper.is_training_stopped() {
                if !self.search {
                    info!("Network training early stopped: epoch:{}", epoch,);
                }
                return true;
            }
        }
        false
    }

    pub fn save(&self, filename: &str, format: SerializationFormat) {
        save_network(&self.to_io(), filename, format);
    }

    pub fn load(filename: &str, format: SerializationFormat) -> Self {
        let network_io = load_network(filename, format);
        Network::from_io(network_io)
    }

    fn from_io(network_io: NetworkIO) -> Self {
        Network {
            input_size: network_io.input_size,
            output_size: network_io.output_size,
            layers: network_io.layers,
            loss_function: network_io.loss_function as Box<dyn LossFunction>,
            optimizer_config: network_io.optimizer_config as Box<dyn OptimizerConfig>,
            regularization: network_io.regularization,
            batch_size: network_io.batch_size,
            batch_group_size: network_io.batch_group_size,
            epochs: network_io.epochs,
            clip_threshold: network_io.clip_threshold,
            seed: network_io.seed,
            early_stopper: network_io.early_stopper,
            debug: network_io.debug,
            normalized: network_io.normalized,
            mins: network_io.mins,
            maxs: network_io.maxs,
            randomizer: Randomizer::new(Some(network_io.seed)),
            search: false,
            summary_writer: network_io.summary_writer as Option<Box<dyn SummaryWriter>>,
            parallelize: network_io.parallelize,
        }
    }

    fn to_io(&self) -> NetworkIO {
        NetworkIO {
            input_size: self.input_size,
            output_size: self.output_size,
            layers: self.layers.clone(),
            loss_function: self.loss_function.clone(),
            optimizer_config: self.optimizer_config.clone(),
            regularization: self.regularization.clone(),
            batch_size: self.batch_size,
            batch_group_size: self.batch_group_size,
            epochs: self.epochs,
            clip_threshold: self.clip_threshold,
            seed: self.seed,
            early_stopper: self.early_stopper.clone(),
            debug: self.debug,
            normalized: self.normalized,
            mins: self.mins.clone(),
            maxs: self.maxs.clone(),
            summary_writer: self.summary_writer.clone(),
            parallelize: self.parallelize,
        }
    }

    fn init_summary_writer(&mut self) {
        if let Some(summary_writer) = &mut self.summary_writer {
            summary_writer.init()
        }
    }
}

fn calculate_group_accuracy(
    group_batch_inputs: &&[DenseMatrix], group_batch_targets: &&[DenseMatrix], group_predictions: &Vec<DenseMatrix>,
) -> f32 {
    let group_accuracy: f32 = group_batch_inputs
        .iter()
        .zip(group_predictions.iter())
        .zip(group_batch_targets.iter())
        .map(|((_, prediction), target)| util::calculate_accuracy(prediction, target))
        .sum::<f32>()
        / (group_batch_targets.len() as f32);
    group_accuracy
}

struct LayerParams {
    pub(crate) layer_input: DenseMatrix,
    pub(crate) pre_activated_output: DenseMatrix,
    pub(crate) activated_output: DenseMatrix,
}
impl LayerParams {
    pub fn new(layer_input: DenseMatrix, pre_activated_output: DenseMatrix, activated_output: DenseMatrix) -> Self {
        LayerParams {
            layer_input,
            pre_activated_output,
            activated_output,
        }
    }
}

pub fn clip_gradients(grads: &mut [&mut DenseMatrix], clip_threshold: f32) {
    if clip_threshold > 0.0 {
        for grad in grads {
            grad.clip(clip_threshold);
        }
    }
}

pub fn calculate_metrics(targets: &DenseMatrix, predictions: &DenseMatrix) -> Metrics {
    let (true_positives_map, false_positives_map, false_negatives_map) =
        calculate_confusion_matrix(targets, predictions);

    // Initialize sums for micro-average calculations
    let mut sum_tp = 0;
    let mut sum_fp = 0;
    let mut sum_fn = 0;

    // Initialize sums for macro-average calculations
    let mut sum_f1_macro = 0.0;

    // Calculate metrics for each class
    let num_classes = true_positives_map.len();
    let mut metrics_by_class = Vec::with_capacity(num_classes);

    for class in 0..num_classes {
        let tp = *true_positives_map.get(&class).unwrap_or(&0);
        let f_pos = *false_positives_map.get(&class).unwrap_or(&0);
        let f_neg = *false_negatives_map.get(&class).unwrap_or(&0);

        // Update sums for micro-average
        sum_tp += tp;
        sum_fp += f_pos;
        sum_fn += f_neg;

        let precision = calculate_precision(tp, f_pos);
        let recall = calculate_recall(tp, f_neg);
        let f1_score = calculate_f1_score(precision, recall);

        let metric = Metric {
            precision,
            recall,
            f1_score,
        };
        metrics_by_class.push(metric);

        sum_f1_macro += f1_score;
    }

    // Calculate macro-average F1 score
    let macro_f1 = sum_f1_macro / num_classes as f32;

    // Calculate micro-average F1 score
    let micro_precision = calculate_precision(sum_tp, sum_fp);
    let micro_recall = calculate_recall(sum_tp, sum_fn);
    let micro_f1 = calculate_f1_score(micro_precision, micro_recall);

    Metrics {
        micro_precision,
        micro_recall,
        macro_f1_score: macro_f1,
        micro_f1_score: micro_f1,
        metrics_by_class,
    }
}

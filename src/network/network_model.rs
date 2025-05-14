use std::sync::{Arc, RwLock};

use log::{error, info};

use crate::{
    dropout::DropoutRegularization,
    error::NetworkError,
    layer::{Layer, LayerConfig},
    matrix::DMat,
    parallel::ThreadPool,
    random::Randomizer,
    regularization::Regularization,
    EarlyStopper, LossFunction, Metrics, Normalization, OptimizerConfig, SummaryWriter,
};

use super::network_io::{NetworkIO, NetworkSerialized};

type ForwardResult = Result<(Vec<DMat>, Vec<Arc<Vec<LayerParams>>>), NetworkError>;

/// A builder for constructing a neural network with customizable layers, loss functions, optimizers, and training parameters.
///
/// Use this struct to configure the architecture and training settings of a neural network, then call `build()` to create a `Network` instance.
pub struct NetworkBuilder {
    input_size: usize,
    output_size: usize,
    layer_configs: Vec<Result<Box<dyn LayerConfig>, NetworkError>>,
    loss_function: Result<Box<dyn LossFunction>, NetworkError>,
    optimizer_config: Result<Box<dyn OptimizerConfig>, NetworkError>,
    regularization: Vec<Result<Box<dyn Regularization>, NetworkError>>,
    batch_size: usize,
    batch_group_size: usize,
    epochs: usize,
    clip_threshold: f32,
    seed: u64,
    early_stopper: Option<Result<Box<dyn EarlyStopper>, NetworkError>>,
    debug: bool,
    normalize_input: Option<Box<dyn Normalization>>,
    normalize_output: Option<Box<dyn Normalization>>,
    summary_writer: Option<Result<Box<dyn SummaryWriter>, NetworkError>>,
    parallelize: usize,
}

impl NetworkBuilder {
    /// Create a new `NetworkBuilder` with the specified input and output sizes.
    ///
    /// # Parameters
    /// - `input_size`: Number of features in the input data.
    /// - `output_size`: Number of outputs (e.g., classes for classification).
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            layer_configs: Vec::new(),
            loss_function: Err(NetworkError::ConfigError("Loss function for NetworkBuilder is not set".to_string())),
            optimizer_config: Err(NetworkError::ConfigError("Optimizer for NetworkBuilder is not set".to_string())),
            regularization: Vec::new(),
            batch_size: usize::MAX,
            batch_group_size: 1,
            epochs: 0,
            clip_threshold: 0.0,
            seed: 0,
            early_stopper: None,
            debug: false,
            normalize_input: None,
            normalize_output: None,
            summary_writer: None,
            parallelize: 1,
        }
    }

    /// Add a layer to the neural network.
    ///
    /// Layers are added in the order they are specified and will be executed sequentially during forward and backward passes.
    /// # Parameters
    /// - `layer_config`: Configuration for the layer (e.g., Dense with specific size and activation).
    pub fn layer(mut self, layer: Result<Box<dyn LayerConfig>, NetworkError>) -> Self {
        self.layer_configs.push(layer);
        self
    }

    /// Add a regularization method to the network.
    ///
    /// Regularization (e.g., L1, L2, dropout) is applied during training to prevent overfitting.
    /// Multiple regularizations can be added and will be applied in order.
    /// # Parameters
    /// - `reg`: Regularization method to apply (e.g., `Dropout`).
    pub fn regularization(mut self, reg: Result<Box<dyn Regularization>, NetworkError>) -> Self {
        self.regularization.push(reg);
        self
    }

    /// Set the loss function for training.
    ///
    /// The loss function measures the error between predictions and targets. It must be set before building the network.
    /// # Parameters
    /// - `loss_function`: Loss function to use (e.g., mean squared error, cross-entropy).
    pub fn loss_function(mut self, loss_function: Result<Box<dyn LossFunction>, NetworkError>) -> Self {
        self.loss_function = loss_function;
        self
    }

    /// Set the optimizer for training.
    ///
    /// The optimizer defines how weights are updated during backpropagation. It must be set before building the network.
    /// # Parameters
    /// - `optimizer_config`: Optimizer configuration (e.g., SGD, Adam).
    pub fn optimizer(mut self, optimizer_config: Result<Box<dyn OptimizerConfig>, NetworkError>) -> Self {
        self.optimizer_config = optimizer_config;
        self
    }

    /// Set an early stopping mechanism.
    ///
    /// Early stopping halts training when performance stops improving, based on the provided stopper's criteria.
    /// # Parameters
    /// - `early_stopper`: Early stopping strategy (e.g., `Loss`).
    pub fn early_stopper(mut self, early_stopper: Result<Box<dyn EarlyStopper>, NetworkError>) -> Self {
        self.early_stopper = Some(early_stopper);
        self
    }

    /// Set the batch group size for training.
    ///
    /// Batches are grouped to balance memory usage and parallelism. A larger group size may improve throughput but requires more memory.
    /// Default is 1 (no grouping).
    /// # Parameters
    /// - `batch_group_size`: Number of batches to process together.
    pub fn batch_group_size(mut self, batch_group_size: usize) -> Self {
        self.batch_group_size = batch_group_size;
        self
    }

    /// Set the batch size for training.
    ///
    /// Smaller batch sizes may improve generalization but increase training time. Default is `usize::MAX` (full dataset).
    /// # Parameters
    /// - `batch_size`: Number of samples per batch.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the number of training epochs.
    ///
    /// An epoch is one full pass through the training data. Must be greater than 0.
    /// # Parameters
    /// - `epochs`: Number of epochs to train.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set the gradient clipping threshold.
    ///
    /// Clips gradients to prevent exploding gradients during training.
    /// # Parameters
    /// - `clip_threshold`: Maximum allowed gradient magnitude.
    pub fn clip_threshold(mut self, clip_threshold: f32) -> Self {
        self.clip_threshold = clip_threshold;
        self
    }

    /// Set the random seed for reproducibility.
    ///
    /// Controls weight initialization and data shuffling. Default is 0.
    /// # Parameters
    /// - `seed`: Random seed value.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable or disable debug logging.
    ///
    /// When enabled, detailed training information (e.g., epoch losses, layer visualizations) is logged. Default is false.
    /// # Parameters
    /// - `debug`: Whether to enable debug logging.
    pub fn debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Set input data normalization.
    ///
    /// Normalizes input data (e.g., min-max scaling, zscore) before feeding it to the network.
    /// # Parameters
    /// - `normalization`: Normalization method to apply to inputs(e.g., `MinMax`, 'ZScore' ).
    pub fn normalize_input(mut self, normalization: impl Normalization + 'static) -> Self {
        self.normalize_input = Some(Box::new(normalization));
        self
    }

    /// Set output data normalization.
    ///
    /// Normalizes target data (e.g., min-max scaling, standardization) before computing the loss.
    /// # Parameters
    /// - `normalization`: Normalization method to apply to outputs(e.g., `MinMax`, 'ZScore' ).
    pub fn normalize_output(mut self, normalization: impl Normalization + 'static) -> Self {
        self.normalize_output = Some(Box::new(normalization));
        self
    }

    /// Set a summary writer for logging training metrics.
    ///
    /// Writes training statistics (e.g., loss, metrics) to a specified output (e.g., TensorBoard).
    /// # Parameters
    /// - `summary_writer`: Summary writer for logging metrics.
    pub fn summary(mut self, summary_writer: Result<Box<dyn SummaryWriter>, NetworkError>) -> Self {
        self.summary_writer = Some(summary_writer);
        self
    }

    /// Set the number of threads for parallel processing.
    ///
    /// Controls the degree of parallelism for forward and backward passes. Must be greater than 0. Default is 1 (single-threaded).
    /// # Parameters
    /// - `parallelize`: Number of threads to use.
    pub fn parallelize(mut self, parallelize: usize) -> Self {
        self.parallelize = parallelize;
        self
    }

    pub(crate) fn update_learning_rate(mut self, learning_rate: f32) -> Self {
        if let Ok(optimizer_config) = &mut self.optimizer_config {
            optimizer_config.update_learning_rate(learning_rate);
        }
        self
    }

    pub(crate) fn with_network(mut self, nw: &Network) -> Self {
        self.loss_function = Ok(nw.loss_function.clone_box());
        self.optimizer_config = Ok(nw.optimizer_config.clone());
        self.batch_size = nw.batch_size;
        self.batch_group_size = nw.batch_group_size;
        self.parallelize = nw.parallelize;
        self.epochs = nw.epochs;
        self.seed = nw.seed;
        self.clip_threshold = nw.clip_threshold;
        self.debug = nw.debug;

        if let Some(early_stopper) = &nw.early_stopper {
            self.early_stopper = Some(Ok(early_stopper.as_ref().clone_box()));
        }

        if let Some(summary_writer) = &nw.summary_writer {
            self.summary_writer = Some(Ok(summary_writer.as_ref().clone_box()));
        }

        self.regularization = nw.regularizations.iter().map(|reg| Ok((**reg).clone_box())).collect();

        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.input_size == 0 || self.output_size == 0 {
            return Err(NetworkError::ConfigError(format!(
                "Input:{} and output:{} sizes must be greater than zero",
                self.input_size, self.output_size
            )));
        }
        if let Some(err) = self.loss_function.as_ref().err() {
            return Err(err.clone());
        }
        if let Some(err) = self.optimizer_config.as_ref().err() {
            return Err(err.clone());
        }
        if let Some(Err(err)) = self.summary_writer.as_ref() {
            return Err(err.clone());
        }
        if self.epochs == 0 {
            return Err(NetworkError::ConfigError(format!(
                "Epochs must be greater than zero, but was: {}",
                self.epochs
            )));
        }
        if self.parallelize == 0 {
            return Err(NetworkError::ConfigError(format!(
                "Parallelization factor must be greater than zero, but was: {}",
                self.parallelize
            )));
        }
        if self.batch_size == 0 {
            return Err(NetworkError::ConfigError(format!(
                "Batch size must be greater than zero, but was: {}",
                self.batch_size
            )));
        }
        if self.batch_group_size == 0 {
            return Err(NetworkError::ConfigError(format!(
                "Batch group size must be greater than zero, but was: {}",
                self.batch_group_size
            )));
        }
        if self.clip_threshold < 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Clip threshold must be non-negative, but was: {}",
                self.clip_threshold
            )));
        }
        if self.layer_configs.is_empty() {
            return Err(NetworkError::ConfigError("At least one layer must be added".to_string()));
        }
        if self.layer_configs.len() == 1 {
            return Err(NetworkError::ConfigError("At least two layers are required".to_string()));
        }
        Ok(())
    }

    /// Build the neural network.
    ///
    /// Validates the configuration and creates a `Network` instance ready for training or inference.
    pub fn build(self) -> Result<Network, NetworkError> {
        self.validate()?;

        let layer_configs = self.layer_configs.into_iter().collect::<Result<Vec<_>, _>>()?;
        let regularizations = self.regularization.into_iter().collect::<Result<Vec<_>, _>>()?;
        let randomizer = Randomizer::new(Some(self.seed));
        let mut layers: Vec<Arc<RwLock<Box<dyn Layer + Send + Sync>>>> = Vec::new();
        let mut input_size = self.input_size; // Initialize with input_size
        let layer_count = layer_configs.len();
        let opt = self.optimizer_config.as_ref().unwrap().clone();
        for (i, mut layer_config) in layer_configs.into_iter().enumerate() {
            let size = layer_config.size(); // Get size via &self
            let mut name = format!("Hidden {}", i);
            if i == layer_count - 1 {
                name = String::from("Output");
                if size != self.output_size {
                    return Err(NetworkError::ConfigError(format!(
                        "Output size of the last layer must match the network output size: {}",
                        self.output_size
                    )));
                }
            }
            let layer = layer_config.create_layer(name, input_size, opt.create_optimizer(), &randomizer);
            input_size = size; // Update input_size for the next layer
            layers.push(Arc::new(RwLock::new(layer)));
        }

        // configure_global_thread_pool(self.parallelize);
        Ok(Network {
            input_size: self.input_size,
            output_size: self.output_size,
            layers,
            loss_function: self.loss_function?,
            optimizer_config: self.optimizer_config.unwrap(),
            regularizations: Arc::new(regularizations),
            batch_size: self.batch_size,
            batch_group_size: self.batch_group_size,
            epochs: self.epochs,
            clip_threshold: self.clip_threshold,
            seed: self.seed,
            early_stopper: self.early_stopper.transpose()?,
            debug: self.debug,
            normalize_input: self.normalize_input,
            normalize_output: self.normalize_output,
            randomizer,
            search: false,
            summary_writer: self.summary_writer.transpose()?,
            parallelize: self.parallelize,
            forward_pool: ThreadPool::new(self.parallelize)?,
            backward_pool: ThreadPool::new(self.parallelize)?,
        })
    }
}

pub struct NetworkResult {
    pub predictions: DMat,
    pub loss: f32,
    pub metrics: Metrics,
}

impl NetworkResult {
    pub fn display_metrics(&self) -> String {
        format!("Loss:{:.4}, {}", self.loss, self.metrics.display())
    }
}

pub struct Network {
    pub(crate) input_size: usize,
    pub(crate) output_size: usize,
    pub(crate) layers: Vec<Arc<RwLock<Box<dyn Layer + Send + Sync>>>>,
    pub(crate) loss_function: Box<dyn LossFunction>,
    pub(crate) optimizer_config: Box<dyn OptimizerConfig>,
    pub(crate) regularizations: Arc<Vec<Box<dyn Regularization>>>,
    pub(crate) batch_size: usize,
    pub(crate) batch_group_size: usize,
    pub(crate) epochs: usize,
    pub(crate) clip_threshold: f32,
    pub(crate) seed: u64,
    pub(crate) randomizer: Randomizer,
    pub(crate) debug: bool,
    pub(crate) normalize_input: Option<Box<dyn Normalization>>,
    pub(crate) normalize_output: Option<Box<dyn Normalization>>,
    pub(crate) search: bool,
    pub(crate) early_stopper: Option<Box<dyn EarlyStopper>>,
    pub(crate) summary_writer: Option<Box<dyn SummaryWriter>>,
    pub(crate) parallelize: usize,
    pub(crate) forward_pool: ThreadPool,
    pub(crate) backward_pool: ThreadPool,
}

impl Network {
    pub fn train(&mut self, inputs: &DMat, targets: &DMat) -> Result<NetworkResult, NetworkError> {
        self.validate_input_target(inputs, targets)?;
        //let training_inputs = self.prepare_inputs(inputs);
        let (training_inputs, training_targets) = self.prepare_data(inputs, targets);
        let sample_size = training_inputs.rows();
        self.log_start_info(sample_size);
        let mut shuffled_inputs = DMat::zeros(sample_size, self.input_size);
        let mut shuffled_targets = DMat::zeros(sample_size, self.output_size);
        // let (backward_pool, forward_pool) = self.create_thread_pools();

        //        self.initialize_threadpool();
        self.init_summary_writer();
        let mut last_epoch = 0;
        for epoch in 1..=self.epochs {
            last_epoch = epoch;
            self.visualize_layers();
            self.shuffle(&training_inputs, &training_targets, &mut shuffled_inputs, &mut shuffled_targets);
            let (batch_inputs, batch_targets) = self.create_batches(&shuffled_inputs, &shuffled_targets);
            let (group_inputs, group_targets) = self.create_groups(&batch_inputs, &batch_targets);

            for (grp_id, (grp_inputs, grp_targets)) in group_inputs.iter().zip(group_targets.iter()).enumerate() {
                let (grp_predictions, mut grp_layer_params) = self.forward(grp_inputs)?;
                self.log_group_training_info(epoch, group_inputs.len(), grp_id, grp_targets, &grp_predictions);
                self.backward(&grp_predictions, grp_targets, &mut grp_layer_params, epoch)?;
            }

            if self.debug || self.summary_writer.is_some() || self.early_stopper.is_some() {
                let epoch_result = self.predict_internal(&training_inputs, &training_targets);
                let epoch_loss = epoch_result.loss;
                self.log_epoch_training_info(epoch, epoch_loss, &epoch_result.metrics);
                self.summarize(epoch, epoch_loss, &epoch_result.metrics);
                if self.early_stopped(epoch, epoch_loss, &epoch_result.metrics) {
                    break;
                }
            }
        }

        let final_result = self.predict_internal(&training_inputs, &training_targets);
        self.log_finish_info(last_epoch, &final_result);
        self.close_summary_writer();
        Ok(final_result)
    }

    fn forward(&self, group_inputs: &[DMat]) -> ForwardResult {
        let mut receivers = Vec::new();
        let base_layers: Vec<_> = self.layers.iter().map(Arc::clone).collect();
        for input in group_inputs.iter() {
            let layers = base_layers.clone();
            let input = input.clone();
            let regularizations = Arc::clone(&self.regularizations);
            let receiver = self
                .forward_pool
                .submit(move || forward_job(&input, &layers, &regularizations))
                .map_err(|e| NetworkError::TrainingError(format!("Failed to submit forward job: {}", e)))?;
            receivers.push(receiver);
        }
        self.forward_pool
            .join()
            .map_err(|e| NetworkError::TrainingError(format!("Failed to join forward pool: {}", e)))?;

        let results: Result<Vec<_>, _> = receivers
            .into_iter()
            .map(|receiver| {
                match receiver.recv() {
                    Ok(result) => Ok(result), // Successfully received the result
                    Err(err) => Err(NetworkError::TrainingError(format!("Forward job error: {}", err))), // Inner error
                }
            })
            .collect();

        let (outputs, layer_params): (Vec<_>, Vec<_>) = results?.into_iter().unzip();
        Ok((outputs, layer_params))
    }

    fn backward(
        &mut self, group_predictions: &[DMat], group_targets: &[DMat],
        group_layer_params: &mut [Arc<Vec<LayerParams>>], epoch: usize,
    ) -> Result<(), NetworkError> {
        // clone layer handles once
        let base_layers = self.layers.to_vec();

        // initialize (dW, dB) for each layer in one go
        let mut grads = self
            .layers
            .iter()
            .map(|layer| {
                let (input_size, output_size) = layer.read().unwrap().input_output_size();
                (DMat::zeros(output_size, input_size), DMat::zeros(output_size, 1))
            })
            .collect::<Vec<_>>();

        // spawn one job per batch
        let receivers: Vec<_> = group_predictions
            .iter()
            .zip(group_targets)
            .zip(group_layer_params.iter_mut())
            .map(|((pred, tgt), params)| {
                let d_out = self.loss_function.backward(pred, tgt);
                let params = Arc::clone(params);
                let layers = base_layers.clone();

                // Handle errors from ThreadPool::submit
                self.backward_pool
                    .submit(move || backward_job(&layers, d_out, &params))
                    .map_err(|e| NetworkError::TrainingError(format!("Failed to submit backward job: {}", e)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Handle errors from ThreadPool::join
        self.backward_pool
            .join()
            .map_err(|e| NetworkError::TrainingError(format!("Failed to join backward pool: {}", e)))?;

        // accumulate per-batch grads into our `grads` vec
        for recv in receivers {
            let (w_batch, b_batch) = recv
                .recv()
                .map_err(|e| NetworkError::TrainingError(format!("Failed to receive backward result: {}", e)))?
                .map_err(|e| NetworkError::TrainingError(format!("Backward job error: {}", e)))?;

            for ((acc_weight, acc_bias), (weight, bias)) in grads.iter_mut().zip(w_batch.into_iter().zip(b_batch)) {
                acc_weight.add(&weight);
                acc_bias.add(&bias);
            }
        }

        // apply regularization, clipping and update in one pass
        for (layer_arc, (mut d_weight, mut d_bias)) in self.layers.iter_mut().zip(grads.into_iter()) {
            let mut layer = layer_arc.write().unwrap();

            self.regularizations.iter().for_each(|reg| {
                // skip dropout here
                if reg.as_any().downcast_ref::<DropoutRegularization>().is_none() {
                    layer.regulate(&mut d_weight, &mut d_bias, reg);
                }
            });

            if self.clip_threshold > 0.0 {
                d_weight.clip(self.clip_threshold);
                d_bias.clip(self.clip_threshold);
            }

            layer.update(&d_weight, &d_bias, epoch);

            if self.debug {
                layer.visualize();
            }
        }

        Ok(())
    }

    pub(crate) fn predict_internal(&mut self, training_inputs: &DMat, training_targets: &DMat) -> NetworkResult {
        let mut output = training_inputs.clone();
        self.layers.iter().for_each(|layer| {
            (output, _) = layer.read().unwrap().forward(&output);
        });

        let loss = self.loss_function.forward(&output, training_targets);
        let metrics = self.loss_function.calculate_metrics(training_targets, &output);

        NetworkResult {
            predictions: output,
            loss,
            metrics,
        }
    }

    fn validate_input_target(&self, inputs: &DMat, targets: &DMat) -> Result<(), NetworkError> {
        if inputs.rows() != targets.rows() {
            return Err(NetworkError::ConfigError(format!(
                "Input and target matrices must have the same number of rows. Inputs: {}, Targets: {}",
                inputs.rows(),
                targets.rows()
            )));
        }
        if inputs.cols() != self.input_size {
            return Err(NetworkError::ConfigError(format!(
                "Input matrix must have {} columns. Found: {}",
                self.input_size,
                inputs.cols()
            )));
        }
        if targets.cols() != self.output_size {
            return Err(NetworkError::ConfigError(format!(
                "Target matrix must have {} columns. Found: {}",
                self.output_size,
                targets.cols()
            )));
        }
        Ok(())
    }

    /// Predict the output for given inputs and targets.
    pub fn predict(&mut self, inputs: &DMat, targets: &DMat) -> Result<NetworkResult, NetworkError> {
        self.validate_input_target(inputs, targets)?;
        let (inputs, targets) = self.prepare_data(inputs, targets);
        let mut output = inputs;
        self.layers.iter().for_each(|layer| {
            (output, _) = layer.read().unwrap().forward(&output);
        });

        let loss = self.loss_function.forward(&output, &targets);
        let metrics = self.loss_function.calculate_metrics(&targets, &output);

        Ok(NetworkResult {
            predictions: output,
            loss,
            metrics,
        })
    }

    fn close_summary_writer(&mut self) {
        if let Some(summary_writer) = self.summary_writer.as_mut() {
            if let Err(e) = summary_writer.close() {
                error!("Failed to close summary writer: {}", e);
            }
        }
    }

    fn visualize_layers(&mut self) {
        if !self.debug {
            return;
        }
        self.layers.iter().for_each(|layer| {
            layer.read().unwrap().visualize();
        });
    }

    fn shuffle(&self, inputs: &DMat, targets: &DMat, shuffling_inputs: &mut DMat, shuffling_targets: &mut DMat) {
        let sample_size = inputs.rows();
        let shuffle_indices = self.randomizer.perm(sample_size);

        shuffle_indices.iter().enumerate().for_each(|(i, &idx)| {
            shuffling_inputs.set_row(i, &inputs.get_row(idx));
            shuffling_targets.set_row(i, &targets.get_row(idx));
        });
    }

    fn prepare_data(&mut self, inputs: &DMat, targets: &DMat) -> (DMat, DMat) {
        let mut training_inputs = inputs.clone();

        if self.normalize_input.is_some() {
            let normalize_input = self.normalize_input.as_mut().unwrap();
            normalize_input.normalize(&mut training_inputs).unwrap();
        }

        let mut training_targets = targets.clone();
        if self.normalize_output.is_some() {
            let normalize_output = self.normalize_output.as_mut().unwrap();
            normalize_output.normalize(&mut training_targets).unwrap();
        }

        (training_inputs, training_targets)
    }

    fn create_batches<'a>(&mut self, inputs: &'a DMat, targets: &'a DMat) -> (Vec<DMat>, Vec<DMat>) {
        let sample_size = inputs.rows();
        let (batch_size, batch_count) = self.calculate_batches(sample_size);
        self.get_all_batch_inputs_targets(sample_size, batch_size, batch_count, inputs, targets)
    }

    fn log_epoch_training_info(&mut self, epoch: usize, epoch_loss: f32, metric_result: &Metrics) {
        if self.debug {
            info!("Epoch [{}/{}], Loss:{:.4}, {}%", epoch, self.epochs, epoch_loss, metric_result.display());
        }
    }

    fn log_group_training_info(
        &mut self, epoch: usize, group_count: usize, group_id: usize, group_targets: &&[DMat],
        group_predictions: &[DMat],
    ) {
        if self.debug {
            let group_losses = self.forward_loss(group_predictions, group_targets);
            let ave_group_loss: f32 = group_losses.iter().sum::<f32>() / group_losses.len() as f32;
            if !self.search {
                info!(
                    "Epoch [{}/{}], Group [{}/{}], Avg Group Loss: {:.4}%",
                    epoch, self.epochs, group_id, group_count, ave_group_loss,
                );
            }
        }
    }

    fn log_finish_info(&mut self, last_epoch: usize, final_result: &NetworkResult) {
        if !self.search {
            info!("Network training finished: epoch:{}, Loss:{:.4}", last_epoch, final_result.loss,);
        }
    }

    fn log_start_info(&mut self, sample_size: usize) {
        let (batch_size, batch_count) = self.calculate_batches(sample_size);
        let group_count = batch_count.div_ceil(self.batch_group_size);

        if !self.search {
            info!("Network training started: sample_size:{}, group_size:{}, group_count:{}, batch_size:{}, batch_count:{}, epoch:{}", sample_size,group_count,self.batch_group_size, batch_size, batch_count, self.epochs);
        }
    }

    fn create_groups<'a>(
        &self, all_batch_inputs: &'a [DMat], all_batch_targets: &'a [DMat],
    ) -> (Vec<&'a [DMat]>, Vec<&'a [DMat]>) {
        let batch_group_size = self.batch_group_size;
        let batch_count = all_batch_inputs.len();
        let mut all_group_batch_inputs = Vec::with_capacity(batch_count.div_ceil(batch_group_size));
        let mut all_group_batch_targets = Vec::with_capacity(batch_count.div_ceil(batch_group_size));

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

        let batch_count = sample_size.div_ceil(batch_size); // Round up
        (batch_size, batch_count)
    }

    fn get_all_batch_inputs_targets(
        &mut self, sample_size: usize, batch_size: usize, batch_count: usize, shuffled_inputs: &DMat,
        shuffled_targets: &DMat,
    ) -> (Vec<DMat>, Vec<DMat>) {
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

    fn forward_loss(&mut self, group_predictions: &[DMat], group_targets: &[DMat]) -> Vec<f32> {
        let mut all_losses = Vec::with_capacity(group_predictions.len());

        for (predicted, target) in group_predictions.iter().zip(group_targets.iter()) {
            let loss = self.loss_function.forward(predicted, target);
            all_losses.push(loss);
        }

        all_losses
    }

    fn summarize(&mut self, epoch: usize, epoch_loss: f32, metric_result: &Metrics) {
        if self.search {
            return;
        }
        if let Some(summary_writer) = &mut self.summary_writer {
            summary_writer.write_scalar("Training/Loss", epoch, epoch_loss).unwrap();
            metric_result
                .headers()
                .iter()
                .zip(metric_result.values())
                .for_each(|(header, value)| {
                    summary_writer
                        .write_scalar(&format!("Training/{}", header), epoch, value)
                        .unwrap();
                });
            self.layers.iter().for_each(|layer| {
                layer.read().unwrap().summarize(epoch, &mut **summary_writer);
            });
        }
    }

    fn early_stopped(&mut self, epoch: usize, val_loss: f32, metric_result: &Metrics) -> bool {
        if let Some(early_stopper) = &mut self.early_stopper {
            early_stopper.update(epoch, val_loss, metric_result);
            if early_stopper.is_training_stopped() {
                if !self.search {
                    info!("Network training early stopped: epoch:{}", epoch,);
                }
                return true;
            }
        }
        false
    }

    pub fn save(&self, network_io: impl NetworkIO) -> Result<(), NetworkError> {
        network_io.save(self.to_io())
    }

    pub fn load(network_io: impl NetworkIO) -> Result<Self, NetworkError> {
        Network::from_io(network_io.load()?)
    }

    fn from_io(network_io: NetworkSerialized) -> Result<Self, NetworkError> {
        let layers = network_io
            .layers
            .into_iter()
            .map(|layer| {
                let layer: Box<dyn Layer + Send + Sync> = layer; // Upcast to match runtime
                Arc::new(RwLock::new(layer))
            })
            .collect();

        Ok(Network {
            input_size: network_io.input_size,
            output_size: network_io.output_size,
            layers,
            loss_function: network_io.loss_function,
            optimizer_config: network_io.optimizer_config as Box<dyn OptimizerConfig>,
            regularizations: Into::into(network_io.regularizations),
            batch_size: network_io.batch_size,
            batch_group_size: network_io.batch_group_size,
            epochs: network_io.epochs,
            clip_threshold: network_io.clip_threshold,
            seed: network_io.seed,
            early_stopper: network_io.early_stopper,
            debug: network_io.debug,
            normalize_input: network_io.normalize_input,
            normalize_output: network_io.normalize_output,
            randomizer: Randomizer::new(Some(network_io.seed)),
            search: false,
            summary_writer: network_io.summary_writer as Option<Box<dyn SummaryWriter>>,
            parallelize: network_io.parallelize,
            forward_pool: ThreadPool::new(network_io.parallelize)?,
            backward_pool: ThreadPool::new(network_io.parallelize)?,
        })
    }

    fn to_io(&self) -> NetworkSerialized {
        let layers: Vec<Box<dyn Layer>> = self
            .layers
            .iter()
            .map(|arc_lock| {
                let lock = arc_lock.read().unwrap();
                lock.clone_box()
            })
            .collect();

        NetworkSerialized {
            input_size: self.input_size,
            output_size: self.output_size,
            layers,
            loss_function: self.loss_function.clone(),
            optimizer_config: self.optimizer_config.clone(),
            regularizations: self.regularizations.to_vec(),
            batch_size: self.batch_size,
            batch_group_size: self.batch_group_size,
            epochs: self.epochs,
            clip_threshold: self.clip_threshold,
            seed: self.seed,
            early_stopper: self.early_stopper.clone(),
            debug: self.debug,
            normalize_input: self.normalize_input.clone(),
            normalize_output: self.normalize_output.clone(),
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

fn forward_job(
    input: &DMat, layers: &Vec<Arc<RwLock<Box<dyn Layer + Send + Sync>>>>,
    regularizations: &Vec<Box<dyn Regularization>>,
) -> (DMat, Arc<Vec<LayerParams>>) {
    let mut layer_params = Vec::with_capacity(layers.len());
    let mut current_input: DMat = input.clone();

    layers.iter().for_each(|layer| {
        let layer = layer.read().unwrap();
        // Get both the activated output and the pre-activated output from the layer
        let (mut activated_output, pre_activated_output) = layer.forward(&current_input);

        // Apply forward regularization
        regularizations
            .iter()
            .filter_map(|r| r.as_any().downcast_ref::<DropoutRegularization>())
            .for_each(|dropout| {
                dropout.apply(&mut [&mut activated_output], &mut Vec::new());
            });

        // Store the layer input and pre-activated output in LayerParams
        layer_params.push(LayerParams::new(current_input.clone(), pre_activated_output, activated_output.clone()));

        // Update the current input for the next layer
        current_input = activated_output;
    });
    (current_input, Arc::new(layer_params))
}

fn backward_job(
    layers: &Vec<Arc<RwLock<Box<dyn Layer + Send + Sync>>>>, d_output: DMat, layer_params: &Arc<Vec<LayerParams>>,
) -> Result<(Vec<DMat>, Vec<DMat>), String> {
    let mut d_output = d_output;
    let mut batch_d_weights = Vec::new();
    let mut batch_d_biases = Vec::new();

    for (layer, params) in layers.iter().rev().zip(layer_params.iter().rev()) {
        let layer = layer.read().map_err(|e| format!("Lock poisoned: {}", e))?;
        let (d_input, d_weights, d_biases) =
            layer.backward(&d_output, &params.layer_input, &params.pre_activated_output, &params.activated_output);

        batch_d_weights.push(d_weights);
        batch_d_biases.push(d_biases);
        d_output = d_input;
    }

    //reverse the gradients to match the original order
    batch_d_weights.reverse();
    batch_d_biases.reverse();
    Ok((batch_d_weights, batch_d_biases))
}

struct LayerParams {
    pub(crate) layer_input: DMat,
    pub(crate) pre_activated_output: DMat,
    pub(crate) activated_output: DMat,
}
impl LayerParams {
    pub fn new(layer_input: DMat, pre_activated_output: DMat, activated_output: DMat) -> Self {
        LayerParams {
            layer_input,
            pre_activated_output,
            activated_output,
        }
    }
}

pub fn clip_gradients(grads: &mut [&mut DMat], clip_threshold: f32) {
    if clip_threshold > 0.0 {
        for grad in grads {
            grad.clip(clip_threshold);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adam::Adam,
        cross_entropy::CrossEntropy,
        dense_layer::Dense,
        dropout::Dropout,
        elu::ELU,
        exponential::Exponential,
        flexible::{Flexible, MonitorMetric},
        l1::L1,
        l2::L2,
        matrix::{DMat, DenseMatrix},
        mean_squared_error::MeanSquared,
        min_max::MinMax,
        numbers::{Numbers, RandomNumbers},
        relu::ReLU,
        sgd::SGD,
        sigmoid::Sigmoid,
        softmax::Softmax,
        swish::Swish,
    };

    #[test]
    fn test_network_builder_with_minimal_configuration() {
        let builder = NetworkBuilder::new(4, 3)
            .layer(Dense::default().size(5).activation(ReLU::build()).build())
            .layer(Dense::default().size(3).activation(Softmax::build()).build())
            .loss_function(MeanSquared.build())
            .optimizer(SGD::default().learning_rate(0.01).build())
            .seed(42)
            .epochs(10)
            .batch_size(2);

        let network = builder.build();
        assert!(network.is_ok(), "Network should build successfully with minimal configuration");
    }

    #[test]
    fn test_network_builder_with_regularization() {
        let builder = NetworkBuilder::new(4, 3)
            .layer(Dense::default().size(5).activation(ReLU::build()).build())
            .layer(Dense::default().size(3).activation(Softmax::build()).build())
            .loss_function(MeanSquared.build())
            .optimizer(SGD::default().learning_rate(0.01).build())
            .regularization(L1::default().lambda(0.01).build())
            .regularization(L2::default().lambda(0.01).build())
            .seed(42)
            .epochs(10)
            .batch_size(2);

        let network = builder.build();
        assert!(network.is_ok(), "Network should build successfully with regularization");
    }

    #[test]
    fn test_network_builder_with_dropout() {
        let builder = NetworkBuilder::new(4, 3)
            .layer(Dense::default().size(5).activation(ReLU::build()).build())
            .layer(Dense::default().size(3).activation(Softmax::build()).build())
            .loss_function(MeanSquared.build())
            .optimizer(SGD::default().learning_rate(0.01).build())
            .regularization(Dropout::default().dropout_rate(0.5).seed(42).build())
            .seed(42)
            .epochs(10)
            .batch_size(2);

        let network = builder.build();
        assert!(network.is_ok(), "Network should build successfully with dropout regularization");
    }

    #[test]
    fn test_network_training_with_simple_data() {
        let mut network = NetworkBuilder::new(2, 1)
            .layer(Dense::default().size(4).activation(ReLU::build()).build())
            .layer(Dense::default().size(1).activation(Sigmoid::build()).build())
            .loss_function(CrossEntropy::default().build())
            .optimizer(Adam::default().build())
            .seed(42)
            .epochs(5)
            .batch_size(1)
            .build()
            .unwrap();

        let inputs = DMat::new(4, 2, &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        let targets = DMat::new(4, 1, &[0.0, 1.0, 1.0, 0.0]);

        let result = network.train(&inputs, &targets);
        assert!(result.is_ok(), "Training should complete without errors");
        let loss = result.unwrap().loss;
        assert!(loss < 0.01, "Network should achieve low loss on XOR problem. Loss({}) in not lower than 0.01", loss);
    }

    #[test]
    fn test_network_training_with_multiple_layers_and_regularization() {
        let mut network = NetworkBuilder::new(4, 3)
            .layer(Dense::default().size(8).activation(ReLU::build()).build())
            .layer(Dense::default().size(8).activation(ReLU::build()).build())
            .layer(Dense::default().size(3).activation(Softmax::build()).build())
            .loss_function(MeanSquared.build())
            .optimizer(SGD::default().learning_rate(0.01).build())
            .seed(42)
            .regularization(L2::default().lambda(0.01).build())
            .epochs(1000)
            .batch_size(2)
            .build()
            .unwrap();
        let input_data = RandomNumbers::new()
            .lower_limit(0.0)
            .upper_limit(1.0)
            .size(100)
            .seed(42)
            .floats();
        let target_data = RandomNumbers::new()
            .lower_limit(0.0)
            .upper_limit(1.0)
            .size(75)
            .seed(42)
            .floats();

        let inputs = DMat::new(25, 4, &input_data);
        let targets = DMat::new(25, 3, &target_data);

        let result = network.train(&inputs, &targets);
        assert!(result.is_ok(), "Training should complete without errors");

        let loss = result.unwrap().loss;
        assert!(loss > 0.01, "Network should achieve reasonable loss with dropout regularization.Test loss: {}", loss);
    }

    #[test]
    fn test_network_training_with_dropout_regularization() {
        let mut network = NetworkBuilder::new(4, 3)
            .layer(Dense::default().size(8).activation(ReLU::build()).build())
            .layer(Dense::default().size(8).activation(ReLU::build()).build())
            .layer(Dense::default().size(3).activation(Softmax::build()).build())
            .loss_function(MeanSquared.build())
            .optimizer(SGD::default().learning_rate(0.01).build())
            .regularization(Dropout::default().dropout_rate(0.01).seed(42).build())
            .seed(42)
            .epochs(400)
            .batch_size(2)
            .build()
            .unwrap();

        let input_data = RandomNumbers::new()
            .lower_limit(0.0)
            .upper_limit(1.0)
            .size(100)
            .seed(42)
            .floats();
        let target_data = RandomNumbers::new()
            .lower_limit(0.0)
            .upper_limit(1.0)
            .size(75)
            .seed(42)
            .floats();

        let inputs = DMat::new(25, 4, &input_data);
        let targets = DMat::new(25, 3, &target_data);

        let result = network.train(&inputs, &targets);
        assert!(result.is_ok(), "Training should complete without errors");

        let loss = result.unwrap().loss;
        assert!(loss < 0.11, "Network should achieve reasonable loss with dropout regularization.Test loss: {}", loss);
    }

    #[test]
    fn test_network_training_with_early_stopping() {
        let mut network = NetworkBuilder::new(4, 3)
            .layer(Dense::default().size(8).activation(ReLU::build()).build())
            .layer(Dense::default().size(8).activation(ReLU::build()).build())
            .layer(Dense::default().size(3).activation(Softmax::build()).build())
            .loss_function(MeanSquared.build())
            .optimizer(SGD::default().learning_rate(0.01).build())
            .regularization(Dropout::default().dropout_rate(0.01).seed(42).build())
            .seed(42)
            .epochs(300)
            .batch_size(2)
            .early_stopper(
                Flexible::default()
                    .patience(20)
                    .target(0.11)
                    .monitor_metric(MonitorMetric::Loss)
                    .min_delta(0.002)
                    .build(),
            )
            .build()
            .unwrap();

        let input_data = RandomNumbers::new()
            .lower_limit(0.0)
            .upper_limit(1.0)
            .size(100)
            .seed(42)
            .floats();
        let target_data = RandomNumbers::new()
            .lower_limit(0.0)
            .upper_limit(1.0)
            .size(75)
            .seed(42)
            .floats();

        let inputs = DMat::new(25, 4, &input_data);
        let targets = DMat::new(25, 3, &target_data);

        let result = network.train(&inputs, &targets);
        assert!(result.is_ok(), "Training should complete without errors");

        let loss = result.unwrap().loss;
        assert!(loss < 0.11, "Network should achieve reasonable loss with early stopping.Test loss: {}", loss);
    }

    #[test]
    fn test_network_prediction() {
        let mut network = NetworkBuilder::new(4, 3)
            .layer(Dense::default().size(8).activation(ReLU::build()).build())
            .layer(Dense::default().size(8).activation(ReLU::build()).build())
            .layer(Dense::default().size(3).activation(Softmax::build()).build())
            .loss_function(MeanSquared.build())
            .optimizer(SGD::default().learning_rate(0.01).build())
            .seed(42)
            .epochs(10)
            .batch_size(2)
            .build()
            .unwrap();

        let input_data = RandomNumbers::new()
            .lower_limit(0.0)
            .upper_limit(1.0)
            .size(100)
            .seed(42)
            .floats();
        let target_data = RandomNumbers::new()
            .lower_limit(0.0)
            .upper_limit(1.0)
            .size(75)
            .seed(42)
            .floats();

        let inputs = DMat::new(25, 4, &input_data);
        let targets = DMat::new(25, 3, &target_data);

        let result = network.predict(&inputs, &targets);
        assert!(result.is_ok(), "Prediction should complete without errors");
    }

    #[test]
    fn test_prediction_with_invalid_input_target() {
        let mut network = NetworkBuilder::new(4, 3)
            .layer(Dense::default().size(8).activation(ReLU::build()).build())
            .layer(Dense::default().size(8).activation(ReLU::build()).build())
            .layer(Dense::default().size(3).activation(Softmax::build()).build())
            .loss_function(MeanSquared.build())
            .optimizer(SGD::default().learning_rate(0.01).build())
            .seed(42)
            .epochs(10)
            .batch_size(2)
            .build()
            .unwrap();

        let inputs = DMat::zeros(25, 4);
        let targets = DMat::zeros(25, 5); // Invalid target size

        let result = network.predict(&inputs, &targets);
        assert!(result.is_err(), "Prediction should fail with invalid input/target sizes");
        if let Err(NetworkError::ConfigError(msg)) = result {
            assert_eq!(msg, "Target matrix must have 3 columns. Found: 5");
        } else {
            panic!("Expected ConfigError");
        }
    }

    #[test]
    fn test_network_simple() {
        let mut network = NetworkBuilder::new(2, 1)
            .layer(Dense::default().size(4).activation(ReLU::build()).build())
            .layer(Dense::default().size(1).activation(Softmax::build()).build())
            .loss_function(CrossEntropy::default().build()) // Cross-entropy loss function with default values
            .optimizer(Adam::default().build()) // Adam optimizer with default values
            .batch_size(2) // Number of batches
            .seed(42) // Optional seed for reproducibility
            .epochs(5)
            .build()
            .unwrap(); // Handle error in production use

        let inputs = DenseMatrix::new(4, 2)
            .data(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .build()
            .unwrap();
        let targets = DenseMatrix::new(4, 1).data(&[0.0, 1.0, 1.0, 0.0]).build().unwrap();

        let result = network.train(&inputs, &targets);

        match result {
            Ok(result) => println!("Training successfully completed.\nResults: {}", result.display_metrics()),
            Err(e) => eprintln!("Training failed: {}", e),
        }
    }

    #[test]
    fn test_network_train() {
        let mut network = NetworkBuilder::new(2, 1)
            .layer(
                Dense::default()
                    .size(12)
                    .activation(ELU::default().alpha(0.9).build())
                    .build(),
            )
            .layer(
                Dense::default()
                    .size(24)
                    .activation(Swish::default().beta(1.0).build())
                    .build(),
            )
            .layer(Dense::default().size(1).activation(Softmax::build()).build())
            .loss_function(CrossEntropy::default().epsilon(0.99).build()) // loss function with epsilon
            .optimizer(
                Adam::default() // Adam optimizer with custom parameters
                    .beta1(0.98)
                    .beta2(0.990)
                    .learning_rate(0.0035)
                    .scheduler(Exponential::default().decay_factor(0.2).build()) // scheduler for learning rate
                    .build(),
            )
            .seed(42) // seed for reproducibility
            .early_stopper(
                Flexible::default()
                    .monitor_metric(MonitorMetric::Loss) // early stopping based on loss
                    .patience(500) // number of epochs with no improvement after which training will be stopped
                    .min_delta(0.1) // minimum change to be considered an improvement
                    .smoothing_factor(0.5) // factor to smooth the loss
                    .build(),
            )
            .regularization(L2::default().lambda(0.01).build()) // L2 regularization
            .regularization(Dropout::default().dropout_rate(0.2).seed(42).build()) // Dropout regularization
            .epochs(5000)
            .batch_size(4)
            .batch_group_size(4) // number of batches to process in groups
            .parallelize(4) // number of threads to use for parallel process the batch groups
            .normalize_input(MinMax::default()) // normalization of the input data
            .build()
            .unwrap();

        let inputs = DenseMatrix::new(4, 2)
            .data(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .build()
            .unwrap();
        let targets = DenseMatrix::new(4, 1).data(&[0.0, 1.0, 1.0, 0.0]).build().unwrap();

        let res = network.train(&inputs, &targets);
        assert!(res.is_ok())
    }
}

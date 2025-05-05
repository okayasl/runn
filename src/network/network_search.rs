use std::{
    collections::BTreeMap,
    fs::{self, File},
    io,
    sync::Arc,
    time::Instant,
};

use csv::Writer;
use log::info;
use rand::{seq::SliceRandom, thread_rng};

use crate::{
    dense_layer::Dense, error::NetworkError, matrix::DenseMatrix, parallel::ThreadPool, ActivationFunction,
    MetricResult, Normalization,
};

use super::network::{Network, NetworkBuilder};

/// A builder for configuring a hyperparameter search over neural network architectures and training settings.
///
/// Use this struct to define ranges of hyperparameters (e.g., layer sizes, learning rates) and a base network configuration.
/// The search will evaluate all specified combinations, training each network and reporting performance metrics.
pub struct NetworkSearchBuilder {
    network: Option<Network>,
    activation_functions: Vec<Result<Box<dyn ActivationFunction>, NetworkError>>,
    hidden_layer_sizes: Vec<Vec<usize>>,
    batch_sizes: Vec<usize>,
    learning_rates: Vec<f32>,
    filename: String,
    normalize_input: Option<Box<dyn Normalization>>,
    normalize_output: Option<Box<dyn Normalization>>,
    parallelize: usize,
}

impl NetworkSearchBuilder {
    /// Create a new `NetworkSearchBuilder` with default settings.
    ///
    /// Initialize an empty search configuration that must be populated with a network, hyperparameters, and optional settings.
    pub fn new() -> Self {
        Self {
            network: None,
            activation_functions: Vec::new(),
            hidden_layer_sizes: Vec::new(),
            batch_sizes: Vec::new(),
            learning_rates: Vec::new(),
            filename: String::new(),
            normalize_input: None,
            normalize_output: None,
            parallelize: 1,
        }
    }

    /// Set the base network configuration for the search.
    ///
    /// The provided network’s settings (e.g., loss function, optimizer) are used as a template, with search parameters overriding specific aspects.
    /// # Parameters
    /// - `network`: The base `Network` to modify during the search.
    pub fn network(mut self, network: Network) -> Self {
        self.network = Some(network);
        self
    }

    /// Add a hidden layer configuration with its activation function.
    ///
    /// Specifies a set of possible neuron counts for a hidden layer and the activation function to use. Multiple calls add additional layers.
    /// # Parameters
    /// - `layer_sizes`: Vector of possible neuron counts for this layer (e.g., `[64, 128]`).
    /// - `af`: Activation function for the layer (e.g., ReLU, Sigmoid).
    pub fn hidden_layer(
        mut self, layer_sizes: Vec<usize>, activation_function: Result<Box<dyn ActivationFunction>, NetworkError>,
    ) -> Self {
        self.hidden_layer_sizes.push(layer_sizes);
        self.activation_functions.push(activation_function);
        self
    }

    /// Set the batch sizes to evaluate.
    ///
    /// Defines the range of batch sizes to test during the search. Smaller batch sizes may improve generalization but increase training time.
    /// # Parameters
    /// - `bs`: Vector of batch sizes to try (e.g., `[32, 64, 128]`).
    pub fn batch_sizes(mut self, bs: Vec<usize>) -> Self {
        self.batch_sizes = bs;
        self
    }

    /// Set the learning rates to evaluate.
    ///
    /// Defines the range of learning rates to test during the search. Lower rates may lead to slower but more stable convergence.
    /// # Parameters
    /// - `lrs`: Vector of learning rates to try (e.g., `[0.001, 0.01, 0.1]`).
    pub fn learning_rates(mut self, lrs: Vec<f32>) -> Self {
        self.learning_rates = lrs;
        self
    }

    /// Set the filename for exporting search results.
    ///
    /// If specified, search results (e.g., losses, metrics) are saved to a CSV file in the `.out` directory.
    /// # Parameters
    /// - `filename`: Name of the output CSV file (without path or extension).
    pub fn export(mut self, filename: String) -> Self {
        self.filename = filename;
        self
    }

    /// Set input data normalization.
    ///
    /// Normalizes input data (e.g., min-max scaling, standardization) before feeding it to networks during the search.
    /// # Parameters
    /// - `normalization`: Normalization method to apply to inputs(e.g., `MinMax`, 'ZScore' ).
    pub fn normalize_input(mut self, normalization: impl Normalization + 'static) -> Self {
        self.normalize_input = Some(Box::new(normalization));
        self
    }

    /// Set output data normalization.
    ///
    /// Normalizes target data (e.g., min-max scaling, standardization) before computing losses during the search.
    /// # Parameters
    /// - `normalization`: Normalization method to apply to outputs(e.g., `MinMax`, 'ZScore' ).
    pub fn normalize_output(mut self, normalization: impl Normalization + 'static) -> Self {
        self.normalize_output = Some(Box::new(normalization));
        self
    }

    /// Set the number of threads for parallel processing.
    ///
    /// Controls the degree of parallelism for training multiple networks concurrently. Must be greater than 0. Default is 1 (single-threaded).
    /// # Parameters
    /// - `parallelize`: Number of threads to use.
    pub fn parallelize(mut self, parallelize: usize) -> Self {
        self.parallelize = parallelize;
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.parallelize <= 0 {
            return Err(NetworkError::ConfigError(format!(
                "Parallelize value for network search must be greater than zero, but was {}",
                self.parallelize
            )));
        }
        if self.batch_sizes.is_empty() {
            return Err(NetworkError::ConfigError("No batch sizes provided".into()));
        }
        if self.learning_rates.is_empty() {
            return Err(NetworkError::ConfigError("No learning rates provided".into()));
        }
        if self.network.is_none() {
            return Err(NetworkError::ConfigError("No network provided".into()));
        }
        if self.hidden_layer_sizes.is_empty() {
            return Err(NetworkError::ConfigError("No hidden layer sizes provided".into()));
        }
        if self.activation_functions.is_empty() {
            return Err(NetworkError::ConfigError("No activation functions provided".into()));
        }
        if self.activation_functions.len() != self.hidden_layer_sizes.len() {
            return Err(NetworkError::ConfigError(
                "Mismatch between activation functions and hidden layer sizes".into(),
            ));
        }
        Ok(())
    }

    /// Build the hyperparameter search.
    ///
    /// Validates the configuration and creates a `NetworkSearch` instance that will evaluate all specified hyperparameter combinations.
    fn generate_network_combinations(&self) -> Result<Vec<Network>, NetworkError> {
        // let mut activation_functions: Vec<Box<dyn ActivationFunction>> = Vec::new();
        // for activation_function in &self.activation_functions {
        //     match activation_function {
        //         Ok(activation_function) => activation_functions.push(activation_function.clone_box()),
        //         Err(e) => return Err(e.clone()),
        //     }
        // }

        let activation_functions: Vec<Box<dyn ActivationFunction>> = self
            .activation_functions
            .iter()
            .map(|af| af.as_ref().map(|f| f.clone_box()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.clone())?; // Convert &NetworkError to NetworkError

        // let activation_functions: Vec<Box<dyn ActivationFunction>> = self.activation_functions.into_iter().collect::<Result<Vec<_>, _>>()?; // Fix: Use into_iter() instead of iter()

        let mut networks = Vec::new();
        let nw = self.network.as_ref().unwrap();
        let nw_is = nw.input_size;
        let nw_os = nw.output_size;
        let output_layer = nw.layers.last().unwrap().read().unwrap();
        let (_, output_layer_size) = output_layer.input_output_size();
        let hidden_layer_sizes_groups = generate_layer_size_combinations(&self.hidden_layer_sizes);
        let balanced_configs =
            balance_network_configs(&hidden_layer_sizes_groups, &self.batch_sizes, &self.learning_rates);
        for (hlsg, bs, lr) in balanced_configs {
            let mut new_nwb: NetworkBuilder = NetworkBuilder::new(nw_is, nw_os)
                .from_network(nw)
                .batch_size(bs)
                .update_learning_rate(lr);
            for (i, &size) in hlsg.iter().enumerate() {
                new_nwb = new_nwb.layer(Dense::new().from(size, activation_functions[i].clone_box()).build());
            }
            new_nwb = new_nwb.layer(
                Dense::new()
                    .from(output_layer_size, output_layer.activation_function().clone_box())
                    .build(),
            );
            let mut network = new_nwb.build().unwrap();
            network.search = true;
            networks.push(network);
        }
        Ok(networks)
    }

    pub fn build(self) -> Result<NetworkSearch, NetworkError> {
        self.validate()?;

        let nc = self.generate_network_combinations()?;
        Ok(NetworkSearch {
            networks: nc,
            filename: self.filename,
            normalize_input: self.normalize_input,
            normalize_output: self.normalize_output,
            parallelize: self.parallelize,
        })
    }
}

fn generate_layer_size_combinations(layer_sizes: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    if layer_sizes.is_empty() {
        return vec![vec![]];
    }

    let first_range = &layer_sizes[0];
    let rest_combinations = generate_layer_size_combinations(&layer_sizes[1..].to_vec());

    let mut combinations = Vec::new();
    for &v in first_range {
        for rest in &rest_combinations {
            let mut new_combination = vec![v];
            new_combination.extend(rest);
            combinations.push(new_combination);
        }
    }
    combinations
}

/// Each combination of (hidden_layers, batch_size, learning_rate)
/// is grouped and interleaved based on a simple cost metric.
pub fn balance_network_configs(
    hidden_layer_groups: &[Vec<usize>], batch_sizes: &[usize], learning_rates: &[f32],
) -> Vec<(Vec<usize>, usize, f32)> {
    let mut rng = thread_rng();
    let mut map: BTreeMap<usize, Vec<(Vec<usize>, usize, f32)>> = BTreeMap::new();

    for hlg in hidden_layer_groups {
        let total_neurons = hlg.iter().sum::<usize>();
        for &bs in batch_sizes {
            for &lr in learning_rates {
                // Estimated cost: total_neurons × batch_size
                let cost = total_neurons * (10_000 / bs);
                map.entry(cost).or_default().push((hlg.clone(), bs, lr));
            }
        }
    }

    // Shuffle within each cost group
    for group in map.values_mut() {
        group.shuffle(&mut rng);
    }

    // Interleave across cost groups
    let mut balanced = Vec::new();
    loop {
        let mut added = false;
        for group in map.values_mut() {
            if let Some(config) = group.pop() {
                balanced.push(config);
                added = true;
            }
        }
        if !added {
            break;
        }
    }

    balanced
}

pub struct NetworkSearchConfig {
    learning_rate: f32,
    batch_size: usize,
    layer_sizes: Vec<usize>,
}

fn extract_config_from_network(nw: &Network) -> NetworkSearchConfig {
    let learning_rate = nw.optimizer_config.learning_rate();
    let batch_size = nw.batch_size;
    let mut layer_sizes = Vec::new();

    for layer in nw.layers.iter().take(nw.layers.len() - 1) {
        let (_, output_size) = layer.read().unwrap().input_output_size();
        layer_sizes.push(output_size);
    }

    NetworkSearchConfig {
        learning_rate,
        batch_size,
        layer_sizes,
    }
}

pub struct NetworkSearch {
    networks: Vec<Network>,
    filename: String,
    normalize_input: Option<Box<dyn Normalization>>,
    normalize_output: Option<Box<dyn Normalization>>,
    parallelize: usize,
}

impl NetworkSearch {
    pub fn search(
        &mut self, training_inputs: &DenseMatrix, training_targets: &DenseMatrix, validation_inputs: &DenseMatrix,
        validation_targets: &DenseMatrix,
    ) -> Vec<SearchResult> {
        let (training_inputs, training_targets) = self.prepare_data(training_inputs, training_targets);
        let (validation_inputs, validation_targets) = self.prepare_data(validation_inputs, validation_targets);

        let number_of_networks = self.networks.len();
        info!("Total number of networks to train: {}", number_of_networks);

        let training_inputs = Arc::new(training_inputs);
        let training_targets = Arc::new(training_targets);
        let validation_inputs = Arc::new(validation_inputs);
        let validation_targets = Arc::new(validation_targets);

        let pool = ThreadPool::new(self.parallelize);
        let mut receivers = Vec::new();
        for network in self.networks.drain(..) {
            let training_inputs = Arc::clone(&training_inputs);
            let training_targets = Arc::clone(&training_targets);
            let validation_inputs = Arc::clone(&validation_inputs);
            let validation_targets = Arc::clone(&validation_targets);

            receivers.push(pool.submit(move || {
                run(network, &training_inputs, &training_targets, &validation_inputs, &validation_targets)
            }));
        }
        let tracker = track_progress(number_of_networks, pool.progress());

        pool.join();
        tracker.join().unwrap();
        let search_results: Vec<_> = receivers.into_iter().map(|r| r.recv().unwrap()).collect();
        if !search_results.is_empty() && !self.filename.is_empty() {
            write_search_results(&self.filename, &search_results).unwrap();
        }

        search_results
    }

    fn prepare_data(&mut self, inputs: &DenseMatrix, targets: &DenseMatrix) -> (DenseMatrix, DenseMatrix) {
        let mut inputs = inputs.clone();

        if self.normalize_input.is_some() {
            let normalize_input = self.normalize_input.as_mut().unwrap();
            normalize_input.normalize(&mut inputs).unwrap();
        }

        let mut targets = targets.clone();
        if self.normalize_output.is_some() {
            let normalize_output = self.normalize_output.as_mut().unwrap();
            normalize_output.normalize(&mut targets).unwrap();
        }

        (inputs, targets)
    }
}

fn track_progress(
    number_of_networks: usize, progress_rx: crossbeam_channel::Receiver<()>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let start = Instant::now();
        let mut last_update = start;
        let mut last_completed = 0;
        let mut completed = 0;
        let mut next_milestone = 10;

        for _ in 0..number_of_networks {
            progress_rx.recv().unwrap();
            completed += 1;

            let percentage = (completed * 100) / number_of_networks;
            if percentage >= next_milestone {
                let now = Instant::now();
                let jobs_since_last = completed - last_completed;
                let seconds_since_last = now.duration_since(last_update).as_secs();
                let total_elapsed = now.duration_since(start).as_secs();

                // Estimate remaining time
                let eta = if completed > 0 {
                    let avg_time_per_job = total_elapsed as f64 / completed as f64;
                    let remaining_jobs = number_of_networks - completed;
                    Some((avg_time_per_job * remaining_jobs as f64).round() as u64)
                } else {
                    None
                };

                info!(
                    "Progress: {:>3}% | Total: {:>3}/{:<3} | +{:>2} jobs in {:>3}s | {:>3}s elapsed | ETA: {:>3}s",
                    percentage,
                    completed,
                    number_of_networks,
                    jobs_since_last,
                    seconds_since_last,
                    total_elapsed,
                    eta.unwrap_or(0)
                );

                last_update = now;
                last_completed = completed;
                next_milestone += 10;
            }
        }

        // Final 100% update if not already printed
        if completed == number_of_networks && (next_milestone - 10) < 100 {
            let now = Instant::now();
            let jobs_since_last = completed - last_completed;
            let seconds_since_last = now.duration_since(last_update).as_secs();
            let total_elapsed = now.duration_since(start).as_secs();

            info!(
                "Progress: 100% | Total: {:>3}/{:<3} | +{:>2} jobs in {:>3}s | {:>3}s elapsed | ETA:   0s",
                completed, number_of_networks, jobs_since_last, seconds_since_last, total_elapsed,
            );
        }
    })
}

fn run(
    mut network: Network, training_inputs: &DenseMatrix, training_targets: &DenseMatrix,
    validation_inputs: &DenseMatrix, validation_targets: &DenseMatrix,
) -> SearchResult {
    let start_time = Instant::now();
    let train_res = network.train(training_inputs, training_targets).unwrap();

    let elapsed_time_in_sec = start_time.elapsed().as_secs_f32();
    let validation_res = network.predict_internal(validation_inputs, validation_targets);

    SearchResult {
        config: extract_config_from_network(&network),
        training_metrics: train_res.metrics,
        validation_metrics: validation_res.metrics,
        t_loss: train_res.loss,
        v_loss: validation_res.loss,
        elapsed_time: elapsed_time_in_sec,
    }
}

pub struct SearchResult {
    elapsed_time: f32,
    config: NetworkSearchConfig,
    training_metrics: MetricResult,
    validation_metrics: MetricResult,
    t_loss: f32,
    v_loss: f32,
}

impl SearchResult {
    fn values(&self) -> Vec<String> {
        let size_string: Vec<String> = self.config.layer_sizes.iter().map(|&size| size.to_string()).collect();
        vec![
            format!("{:.5}", self.config.learning_rate),
            self.config.batch_size.to_string(),
            size_string.join(","),
            format!("{:.5}", self.t_loss),
            self.training_metrics.values_str().join(", "),
            format!("{:.5}", self.v_loss),
            self.validation_metrics.values_str().join(", "),
            format!("{:.3}", self.elapsed_time),
        ]
    }

    fn default_headers(&self) -> Vec<String> {
        vec![
            "Learning_Rate",
            "Batch_Size",
            "Hidden_Layer_Sizes",
            "Training_Loss",
            &self
                .training_metrics
                .headers()
                .into_iter()
                .map(|header| format!("Training_{}", header))
                .collect::<Vec<String>>()
                .join(", "),
            "Validation_Loss",
            &self
                .validation_metrics
                .headers()
                .into_iter()
                .map(|header| format!("Validation_{}", header))
                .collect::<Vec<String>>()
                .join(", "),
            "Elapsed_Time",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }
}

pub fn write_search_results(name: &str, results: &[SearchResult]) -> io::Result<()> {
    if !std::path::Path::new(".out").exists() {
        fs::create_dir(".out")?;
    }
    let file_path = format!(".out/{}-result.csv", name);
    let file = File::create(file_path)?;
    let mut writer = Writer::from_writer(file);

    writer.write_record(results[0].default_headers())?;

    for result in results {
        writer.write_record(result.values())?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {

    use crate::{
        adam::Adam, cross_entropy::CrossEntropy, elu::ELU, error::NetworkError, min_max::MinMax, relu::ReLU,
        softmax::Softmax,
    };

    use super::*;

    #[test]
    #[should_panic(expected = "No batch sizes provided")]
    fn test_build_no_batch_sizes() {
        let builder = NetworkSearchBuilder::new()
            .network(get_network().unwrap())
            .hidden_layer(vec![10], ReLU::new())
            .learning_rates(vec![0.01]);
        builder.build().unwrap();
    }

    #[test]
    #[should_panic(expected = "No learning rates provided")]
    fn test_build_no_learning_rates() {
        let builder = NetworkSearchBuilder::new()
            .network(get_network().unwrap())
            .hidden_layer(vec![10], ReLU::new())
            .batch_sizes(vec![32]);
        builder.build().unwrap();
    }

    #[test]
    #[should_panic(expected = "No network provided")]
    fn test_build_no_network() {
        let builder = NetworkSearchBuilder::new()
            .hidden_layer(vec![10], ReLU::new())
            .batch_sizes(vec![32])
            .learning_rates(vec![0.01]);
        builder.build().unwrap();
    }

    #[test]
    fn test_build_success() {
        let builder = NetworkSearchBuilder::new()
            .network(get_network().unwrap())
            .hidden_layer(vec![10], ReLU::new())
            .batch_sizes(vec![32, 64])
            .learning_rates(vec![0.01, 0.02])
            .export("test_file".to_string())
            .normalize_input(MinMax::new())
            .parallelize(4);

        let network_search = builder.build().unwrap();

        assert_eq!(network_search.filename, "test_file");
        assert!(network_search.normalize_input.is_some());
        assert_eq!(network_search.parallelize, 4);
        assert!(!network_search.networks.is_empty());
    }

    fn get_network() -> Result<Network, NetworkError> {
        let network = NetworkBuilder::new(1, 3)
            .layer(Dense::new().size(16).activation(ReLU::new()).build())
            .layer(Dense::new().size(3).activation(Softmax::new()).build())
            .loss_function(CrossEntropy::new().epsilon(1e-8).build())
            .optimizer(Adam::new().beta1(0.99).beta2(0.999).learning_rate(0.0035).build())
            .batch_size(10)
            .epochs(300)
            .build();
        network
    }

    #[test]
    fn validate_network_search() {
        let network = get_network().unwrap();
        let search = NetworkSearchBuilder::new()
            .network(network)
            .hidden_layer(vec![10], ReLU::new())
            .batch_sizes(vec![32])
            .learning_rates(vec![0.01])
            .build();

        assert!(search.is_ok());
    }

    #[test]
    fn validate_network_search_with_invalid_activation_function() {
        let network = get_network().unwrap();
        let search = NetworkSearchBuilder::new()
            .network(network)
            .hidden_layer(vec![10], ELU::new().alpha(-1.0).build())
            .batch_sizes(vec![32])
            .learning_rates(vec![0.01])
            .build();

        assert!(search.is_err());
        if let Err(e) = search {
            assert_eq!(e.to_string(), "Configuration error: Alpha for ELU must be greater than 0.0, but was -1");
        }
    }
}

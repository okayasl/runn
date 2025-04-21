use std::{
    sync::{
        atomic::{AtomicI64, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::Instant,
};

use log::info;
use rayon::ThreadPoolBuilder;

use crate::{matrix::DenseMatrix, network_search_io::write_search_results, util, ActivationFunction, Dense};

use super::{
    network::{Network, NetworkBuilder},
    network_search_io::NetworkResult,
};

pub struct NetworkSearchBuilder {
    network: Option<Network>,
    activation_functions: Vec<Box<dyn ActivationFunction>>,
    hidden_layer_sizes: Vec<Vec<usize>>,
    batch_sizes: Vec<usize>,
    learning_rates: Vec<f32>,
    filename: String,
    normalize: bool,
    parallelize: usize,
}

impl NetworkSearchBuilder {
    pub fn new() -> Self {
        Self {
            network: None,
            activation_functions: Vec::new(),
            hidden_layer_sizes: Vec::new(),
            batch_sizes: Vec::new(),
            learning_rates: Vec::new(),
            filename: String::new(),
            normalize: false,
            parallelize: 1,
        }
    }
    pub fn network(mut self, network: Network) -> Self {
        self.network = Some(network);
        self
    }

    pub fn hidden_layer(mut self, layer_sizes: Vec<usize>, af: impl ActivationFunction + 'static) -> Self {
        self.hidden_layer_sizes.push(layer_sizes);
        self.activation_functions.push(Box::new(af));
        self
    }

    pub fn batch_sizes(mut self, bs: Vec<usize>) -> Self {
        self.batch_sizes = bs;
        self
    }

    pub fn learning_rates(mut self, lrs: Vec<f32>) -> Self {
        self.learning_rates = lrs;
        self
    }

    pub fn export(mut self, filename: String) -> Self {
        self.filename = filename;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
    pub fn parallelize(mut self, parallelize: usize) -> Self {
        self.parallelize = parallelize;
        self
    }

    fn validate(&self) {
        if self.activation_functions.is_empty() {
            panic!("No activation functions provided");
        }
        if self.hidden_layer_sizes.is_empty() {
            panic!("No hidden layer sizes provided");
        }
        if self.batch_sizes.is_empty() {
            panic!("No batch sizes provided");
        }
        if self.learning_rates.is_empty() {
            panic!("No learning rates provided");
        }
        if self.network.is_none() {
            panic!("No network provided");
        }
    }

    fn generate_network_combinations(&self) -> Vec<Network> {
        let mut networks = Vec::new();
        let nw = self.network.as_ref().unwrap();
        let output_layer = nw.layers.last().unwrap();
        let (_, output_layer_size) = output_layer.input_output_size();
        let hidden_layer_sizes_groups = generate_layer_size_combinations(&self.hidden_layer_sizes);
        for hlsg in &hidden_layer_sizes_groups {
            for lr in &self.learning_rates {
                for bs in &self.batch_sizes {
                    let mut new_nwb: NetworkBuilder = NetworkBuilder::new(nw.input_size, nw.output_size)
                        .from_network(nw)
                        .batch_size(*bs)
                        .update_learning_rate(*lr);
                    for (i, &size) in hlsg.iter().enumerate() {
                        new_nwb = new_nwb.layer(Dense::new().from(size, self.activation_functions[i].clone()).build());
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
            }
        }
        networks
    }

    pub fn build(self) -> NetworkSearch {
        self.validate();
        NetworkSearch {
            networks: self.generate_network_combinations(),
            filename: self.filename,
            normalize: self.normalize,
            parallelize: self.parallelize,
        }
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

pub struct NetworkConfig {
    learning_rate: f32,
    batch_size: usize,
    layer_sizes: Vec<usize>,
}

fn extract_config_from_network(nw: &Network) -> NetworkConfig {
    let learning_rate = nw.optimizer_config.learning_rate();
    let batch_size = nw.batch_size;
    let mut layer_sizes = Vec::new();

    for layer in nw.layers.iter().take(nw.layers.len() - 1) {
        let (_, output_size) = layer.input_output_size();
        layer_sizes.push(output_size);
    }

    NetworkConfig {
        learning_rate,
        batch_size,
        layer_sizes,
    }
}

pub struct NetworkSearch {
    networks: Vec<Network>,
    filename: String,
    normalize: bool,
    parallelize: usize,
}

impl NetworkSearch {
    fn prepare_inputs(&mut self, t_inputs: &DenseMatrix, v_inputs: &DenseMatrix) -> (DenseMatrix, DenseMatrix) {
        let mut training_inputs = t_inputs.clone();
        let mut validation_inputs = v_inputs.clone();
        if self.normalize {
            let (mins, maxs) = util::find_min_max(&t_inputs);
            util::normalize_in_place(&mut training_inputs, &mins, &maxs);
            util::normalize_in_place(&mut validation_inputs, &mins, &maxs);
        }
        (training_inputs, validation_inputs)
    }
    fn build_thread_pool(&self, worker_count: usize) {
        ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build_global()
            .expect("Failed to build global thread pool");
    }

    pub fn search(
        &mut self, training_inputs: &DenseMatrix, training_targets: &DenseMatrix, validation_inputs: &DenseMatrix,
        validation_targets: &DenseMatrix,
    ) -> Vec<SearchResult> {
        let (training_inputs, validation_inputs) = self.prepare_inputs(training_inputs, validation_inputs);
        self.build_thread_pool(self.parallelize);
        let number_of_networks = self.networks.len();
        info!("Total number of network to train: {}", number_of_networks);

        let (nc_jobs_tx, nc_jobs_rx) = mpsc::channel();
        let (results_tx, results_rx) = mpsc::channel();
        let network_count = Arc::new(AtomicI64::new(0));
        let start_time = Instant::now();
        let period = Arc::new(Mutex::new(start_time));

        let nc_jobs_rx = Arc::new(Mutex::new(nc_jobs_rx)); // Wrap the Receiver in Arc<Mutex<>>
        let mut handles = vec![];

        for _ in 0..self.parallelize {
            let nc_jobs_rx = Arc::clone(&nc_jobs_rx); // Clone the Arc<Mutex<Receiver>>
            let results_tx = results_tx.clone();
            let network_count = Arc::clone(&network_count);
            let period = Arc::clone(&period);
            let training_inputs = training_inputs.clone();
            let training_targets = training_targets.clone();
            let validation_inputs = validation_inputs.clone();
            let validation_targets = validation_targets.clone();

            let handle = thread::spawn(move || {
                while let Ok(job) = nc_jobs_rx.lock().unwrap().recv() {
                    let result = run(job, &training_inputs, &training_targets, &validation_inputs, &validation_targets);
                    results_tx.send(result).unwrap();

                    let completed_jobs = network_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if completed_jobs % 100 == 0 {
                        let current_period = period.lock().unwrap().elapsed().as_secs_f64() / 60.0;
                        let time_passed = start_time.elapsed().as_secs_f64() / 60.0;
                        let guessed_remaining_time =
                            (time_passed / completed_jobs as f64) * (number_of_networks as f64 - completed_jobs as f64);
                        info!(
                            "[{}/{}] trained. Training period/total(m):[{:.2}/{:.2}]. Guessed remaining time(m):{:.2}",
                            completed_jobs, number_of_networks, current_period, time_passed, guessed_remaining_time
                        );
                        *period.lock().unwrap() = Instant::now();
                    }
                }
            });
            handles.push(handle);
        }

        for network in self.networks.drain(..) {
            nc_jobs_tx.send(network).unwrap(); // Move the `Network` into the channel
        }

        drop(nc_jobs_tx);
        handles.into_iter().for_each(|handle| {
            handle.join().unwrap();
        });

        drop(results_tx); // Close the results channel

        let mut search_results = Vec::new();
        while let Ok(res) = results_rx.recv() {
            search_results.push(res);
        }

        if !search_results.is_empty() && !self.filename.is_empty() {
            write_search_results(&self.filename, &convert_results(&search_results)).unwrap();
        }

        search_results
    }
}

fn run(
    mut network: Network, training_inputs: &DenseMatrix, training_targets: &DenseMatrix,
    validation_inputs: &DenseMatrix, validation_targets: &DenseMatrix,
) -> SearchResult {
    let start_time = Instant::now();
    let train_res = network.train(training_inputs, training_targets).unwrap();

    let elapsed_time_in_sec = start_time.elapsed().as_secs_f32();
    let validation_res = network.predict(validation_inputs, validation_targets);

    SearchResult {
        config: extract_config_from_network(&network),
        t_accuracy: train_res.accuracy * 100.0,
        t_loss: train_res.loss,
        v_accuracy: validation_res.accuracy * 100.0,
        v_loss: validation_res.loss,
        elapsed_time: elapsed_time_in_sec,
    }
}

pub struct SearchResult {
    elapsed_time: f32,
    config: NetworkConfig,
    t_accuracy: f32,
    t_loss: f32,
    v_accuracy: f32,
    v_loss: f32,
}

fn convert_results(search_results: &[SearchResult]) -> Vec<NetworkResult> {
    search_results
        .iter()
        .map(|sr| NetworkResult {
            learning_rate: sr.config.learning_rate,
            batch_size: sr.config.batch_size,
            layer_sizes: sr.config.layer_sizes.clone(),
            t_loss: sr.t_loss,
            t_accuracy: sr.t_accuracy,
            v_loss: sr.v_loss,
            v_accuracy: sr.v_accuracy,
            elapsed_time: sr.elapsed_time,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use crate::{
        adam::Adam,
        cross_entropy::CrossEntropy,
        relu::ReLU,
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
        builder.build();
    }

    #[test]
    #[should_panic(expected = "No learning rates provided")]
    fn test_build_no_learning_rates() {
        let builder = NetworkSearchBuilder::new()
            .network(get_network().unwrap())
            .hidden_layer(vec![10], ReLU::new())
            .batch_sizes(vec![32]);
        builder.build();
    }

    #[test]
    #[should_panic(expected = "No network provided")]
    fn test_build_no_network() {
        let builder = NetworkSearchBuilder::new()
            .hidden_layer(vec![10], ReLU::new())
            .batch_sizes(vec![32])
            .learning_rates(vec![0.01]);
        builder.build();
    }

    #[test]
    fn test_build_success() {
        let builder = NetworkSearchBuilder::new()
            .network(get_network().unwrap())
            .hidden_layer(vec![10], ReLU::new())
            .batch_sizes(vec![32, 64])
            .learning_rates(vec![0.01, 0.02])
            .export("test_file".to_string())
            .normalize(true)
            .parallelize(4);

        let network_search = builder.build();

        assert_eq!(network_search.filename, "test_file");
        assert!(network_search.normalize);
        assert_eq!(network_search.parallelize, 4);
        assert!(!network_search.networks.is_empty());
    }

    fn get_network() -> Result<Network, Box<dyn Error>> {
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
}

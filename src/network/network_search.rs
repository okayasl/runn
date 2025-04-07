use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

use log::info;

use crate::layer::Dense;
use crate::matrix::DenseMatrix;
use crate::network::network::{Network, NetworkBuilder};
use crate::network_search_io::write_search_results;
use crate::{util, ActivationFunction};

use super::network_search_io::NetworkResult;

pub struct SearchConfigs {
    activation_functions: Vec<Box<dyn ActivationFunction>>,
    hidden_layer_sizes: Vec<Vec<usize>>,
    batch_sizes: Vec<usize>,
    learning_rates: Vec<f32>,
    filename: String,
    normalize: bool,
}

pub struct SearchConfigsBuilder {
    activation_functions: Vec<Box<dyn ActivationFunction>>,
    hidden_layer_sizes: Vec<Vec<usize>>,
    batch_sizes: Vec<usize>,
    learning_rates: Vec<f32>,
    filename: String,
    normalize: bool,
}

impl SearchConfigsBuilder {
    pub fn new() -> Self {
        Self {
            activation_functions: Vec::new(),
            hidden_layer_sizes: Vec::new(),
            batch_sizes: Vec::new(),
            learning_rates: Vec::new(),
            filename: String::new(),
            normalize: false,
        }
    }

    pub fn hidden_layer(
        mut self,
        layer_sizes: Vec<usize>,
        af: impl ActivationFunction + 'static,
    ) -> Self {
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

    pub fn build(self) -> SearchConfigs {
        SearchConfigs {
            activation_functions: self.activation_functions,
            hidden_layer_sizes: generate_layer_size_combinations(self.hidden_layer_sizes),
            batch_sizes: self.batch_sizes,
            learning_rates: self.learning_rates,
            filename: self.filename,
            normalize: self.normalize,
        }
    }
}

fn generate_layer_size_combinations(layer_sizes: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    if layer_sizes.is_empty() {
        return vec![vec![]];
    }

    let first_range = &layer_sizes[0];
    let rest_combinations = generate_layer_size_combinations(layer_sizes[1..].to_vec());

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
    activation_functions: Vec<Box<dyn ActivationFunction>>,
}

impl NetworkConfig {
    pub fn values(&self) -> Vec<String> {
        let size_string: Vec<String> = self
            .layer_sizes
            .iter()
            .map(|&size| size.to_string())
            .collect();
        vec![
            format!("{:.5}", self.learning_rate),
            self.batch_size.to_string(),
            size_string.join(","),
        ]
    }
}

pub fn default_headers() -> Vec<&'static str> {
    vec![
        "LearningRate",
        "BatchSize",
        "HiddenLayerSizes",
        "Loss",
        "Accuracy",
    ]
}

pub struct SearchResult {
    config: NetworkConfig,
    t_accuracy: f32,
    t_loss: f32,
    elapsed_time: f32,
    v_accuracy: f32,
    v_loss: f32,
}

impl SearchResult {
    pub fn values(&self) -> Vec<String> {
        vec![
            format!("{:.5}", self.t_loss),
            format!("{:.3}", self.t_accuracy),
        ]
    }
}

pub struct SearchJob {
    network: Network,
    config: NetworkConfig,
}

pub fn search(
    nw: Network,
    np: SearchConfigs,
    worker_count: usize,
    mut training_inputs: DenseMatrix,
    training_targets: DenseMatrix,
    mut validation_inputs: DenseMatrix,
    validation_targets: DenseMatrix,
) -> Vec<SearchResult> {
    if np.normalize {
        let (mins, maxs) = util::find_min_max(&training_inputs);
        training_inputs = util::normalize(&training_inputs,&mins, &maxs).unwrap();
        validation_inputs = util::normalize(&validation_inputs,&mins, &maxs).unwrap();
    }

    let ncs = generate_network_configurations(&np);
    let number_of_networks = ncs.len();
    info!(
        "Total number of network configurations: {}",
        number_of_networks
    );

    let (nc_jobs_tx, nc_jobs_rx) = mpsc::channel();
    let (results_tx, results_rx) = mpsc::channel();
    let network_count = Arc::new(AtomicI64::new(0));
    let start_time = Instant::now();
    let period = Arc::new(Mutex::new(start_time));

    let nc_jobs_rx = Arc::new(Mutex::new(nc_jobs_rx)); // Wrap the Receiver in Arc<Mutex<>>
    let mut handles = vec![];

    for _ in 0..worker_count {
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
                let result = run(
                    job,
                    &training_inputs,
                    &training_targets,
                    &validation_inputs,
                    &validation_targets,
                );
                results_tx.send(result).unwrap();

                let completed_jobs = network_count.fetch_add(1, Ordering::SeqCst) + 1;
                if completed_jobs % 100 == 0 {
                    let current_period = period.lock().unwrap().elapsed().as_secs_f64() / 60.0;
                    let time_passed = start_time.elapsed().as_secs_f64() / 60.0;
                    let guessed_remaining_time = (time_passed / completed_jobs as f64)
                        * (number_of_networks as f64 - completed_jobs as f64);
                    info!(
                        "[{}/{}] trained. Training period/total(m):[{:.2}/{:.2}]. Guessed remaining time(m):{:.2}",
                        completed_jobs,
                        number_of_networks,
                        current_period,
                        time_passed,
                        guessed_remaining_time
                    );
                    *period.lock().unwrap() = Instant::now();
                }
            }
        });

        handles.push(handle);
    }

    let mut search_jobs = Vec::new();
    for nc in ncs {
        let new_nt = generate_tuned_network(&nw, &nc);
        search_jobs.push(SearchJob {
            network: new_nt,
            config: nc,
        });
    }

    for sc in search_jobs {
        nc_jobs_tx.send(sc).unwrap();
    }

    drop(nc_jobs_tx);
    for handle in handles {
        handle.join().unwrap();
    }

    drop(results_tx); // Close the results channel

    let mut search_results = Vec::new();
    while let Ok(res) = results_rx.recv() {
        search_results.push(res);
    }

    if !search_results.is_empty() && !np.filename.is_empty() {
        write_search_results(&np.filename, &convert_results(&search_results)).unwrap();
    }

    search_results
}

fn generate_network_configurations(np: &SearchConfigs) -> Vec<NetworkConfig> {
    let mut configs = Vec::new();

    for hls in &np.hidden_layer_sizes {
        for &lr in &np.learning_rates {
            for &bs in &np.batch_sizes {
                configs.push(NetworkConfig {
                    layer_sizes: hls.clone(),
                    learning_rate: lr,
                    batch_size: bs,
                    activation_functions: np.activation_functions.clone(),
                });
            }
        }
    }
    configs
}

fn generate_tuned_network(nw: &Network, nc: &NetworkConfig) -> Network {
    let mut new_nwb = NetworkBuilder::new(nw.input_size, nw.output_size).from_network(nw);

    for (i, &size) in nc.layer_sizes.iter().enumerate() {
        new_nwb = new_nwb.layer(
            Dense::new()
                .from(size, nc.activation_functions[i].clone())
                .build(),
        );
    }

    let output_layer = nw.layers.last().unwrap();
    let (_,output_layer_size) = output_layer.get_input_output_size();
    new_nwb = new_nwb.layer(
        Dense::new()
            .from(
                output_layer_size,
                output_layer.activation_function().clone_box(),
            )
            .build(),
    );

    new_nwb.build().unwrap()
}

fn run(
    mut search_job: SearchJob,
    training_inputs: &DenseMatrix,
    training_targets: &DenseMatrix,
    validation_inputs: &DenseMatrix,
    validation_targets: &DenseMatrix,
) -> SearchResult {
    let start_time = Instant::now();
    let train_res = search_job
        .network
        .train(training_inputs, training_targets)
        .unwrap();
    let elapsed_time_in_sec = start_time.elapsed().as_secs_f32();
    let validation_res = search_job
        .network
        .predict(validation_inputs, validation_targets);

    SearchResult {
        config: search_job.config,
        t_accuracy: train_res.accuracy * 100.0,
        t_loss: train_res.loss,
        v_accuracy: validation_res.accuracy * 100.0,
        v_loss: validation_res.loss,
        elapsed_time: elapsed_time_in_sec,
    }
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

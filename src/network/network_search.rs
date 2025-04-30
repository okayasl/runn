use std::{
    fs::{self, File},
    io,
    sync::Arc,
    time::Instant,
};

use csv::Writer;
use log::info;

use crate::{matrix::DenseMatrix, parallel::ThreadPool, ActivationFunction, Dense, MetricResult, Normalization};

use super::network::{Network, NetworkBuilder};

pub struct NetworkSearchBuilder {
    network: Option<Network>,
    activation_functions: Vec<Box<dyn ActivationFunction>>,
    hidden_layer_sizes: Vec<Vec<usize>>,
    batch_sizes: Vec<usize>,
    learning_rates: Vec<f32>,
    filename: String,
    normalize_input: Option<Box<dyn Normalization>>,
    normalize_output: Option<Box<dyn Normalization>>,
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
            normalize_input: None,
            normalize_output: None,
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

    pub fn normalize_input(mut self, normalization: impl Normalization + 'static) -> Self {
        self.normalize_input = Some(Box::new(normalization));
        self
    }

    pub fn normalize_output(mut self, normalization: impl Normalization + 'static) -> Self {
        self.normalize_output = Some(Box::new(normalization));
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
        let output_layer = nw.layers.last().unwrap().read().unwrap();
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
            normalize_input: self.normalize_input,
            normalize_output: self.normalize_output,
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
        let (_, output_size) = layer.read().unwrap().input_output_size();
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
        pool.join();
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
    config: NetworkConfig,
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
            self.training_metrics.values().join(", "),
            format!("{:.5}", self.v_loss),
            self.validation_metrics.values().join(", "),
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
    use std::error::Error;

    use crate::{adam::Adam, cross_entropy::CrossEntropy, min_max::MinMax, relu::ReLU, softmax::Softmax};

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
            .normalize_input(MinMax::new())
            .parallelize(4);

        let network_search = builder.build();

        assert_eq!(network_search.filename, "test_file");
        assert!(network_search.normalize_input.is_some());
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

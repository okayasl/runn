use csv::ReaderBuilder;
use env_logger::{Builder, Target};
use log::{error, info};
use runn::{
    adam::Adam,
    csv::CSV,
    dense_layer::Dense,
    flexible::{Flexible, MonitorMetric},
    helper,
    linear::Linear,
    matrix::{DMat, DenseMatrix},
    mean_squared_error::MeanSquared,
    min_max::MinMax,
    network::network_model::{Network, NetworkBuilder},
    network_io::JSON,
    network_search::NetworkSearchBuilder,
    numbers::{Numbers, SequentialNumbers},
    relu::ReLU,
};
use std::error::Error;
use std::fs::File;
use std::{env, fs};

const EXP_NAME: &str = "energy_efficiency";

/// This example demonstrates how to train and validate a neural network on the Energy Efficiency dataset.
/// The Energy Efficiency dataset is a regression dataset used for predicting energy efficiency.
/// The code includes functions to load the dataset, build the neural network,
/// train the network, validate its performance, and perform a hyperparameter search.
///
/// to run the example:
/// ```bash
/// cargo run --example energy_efficiency
/// ```
/// to run the hyperparameter search:
/// ```bash
/// cargo run --example energy_efficiency -- -search
/// ```
/// The hyperparameter search will create a CSV file with the results in the `energy_efficiency` directory.
/// The training and validation results will be logged in the `energy_efficiency` directory.
fn main() {
    initialize_logger(EXP_NAME);

    let (training_inputs, training_targets, validation_inputs, validation_targets) =
        energy_efficiency_inputs_targets("ENB2012_data", 10, 8).unwrap();

    let args: Vec<String> = env::args().collect();
    if args.contains(&"-search".to_string()) {
        search(&training_inputs, &training_targets, &validation_inputs, &validation_targets);
    } else {
        train_and_validate(&training_inputs, &training_targets, &validation_inputs, &validation_targets);
    }
}

fn train_and_validate(
    training_inputs: &DMat, training_targets: &DMat, validation_inputs: &DMat, validation_targets: &DMat,
) {
    let network_file = format!("{}_network", EXP_NAME);

    let mut network = energy_efficiency_network(training_inputs.cols(), training_targets.cols());

    let training_result = network.train(training_inputs, training_targets);
    match training_result {
        Ok(_) => {
            info!("Training successfully completed");
            network
                .save(
                    JSON::default()
                        .directory(EXP_NAME)
                        .file_name(&network_file)
                        .build()
                        .unwrap(),
                )
                .unwrap();
            let net_results = network.predict(training_inputs, training_targets).unwrap();
            info!(
                "{}",
                helper::pretty_compare_matrices(
                    training_inputs,
                    training_targets,
                    &net_results.predictions,
                    helper::CompareMode::Regression
                )
            );
            info!("Training: {}", net_results.display_metrics());
        }
        Err(e) => {
            error!("Training failed: {}", e);
        }
    }

    network = Network::load(
        JSON::default()
            .directory(EXP_NAME)
            .file_name(&network_file)
            .build()
            .unwrap(),
    )
    .unwrap();
    let net_results = network.predict(validation_inputs, validation_targets).unwrap();
    log::info!(
        "{}",
        helper::pretty_compare_matrices(
            validation_inputs,
            validation_targets,
            &net_results.predictions,
            helper::CompareMode::Regression
        )
    );
    info!("Validation: {}", net_results.display_metrics());
}

fn energy_efficiency_network(inp_size: usize, targ_size: usize) -> Network {
    let network = NetworkBuilder::new(inp_size, targ_size)
        .layer(Dense::default().size(18).activation(ReLU::build()).build())
        .layer(Dense::default().size(14).activation(ReLU::build()).build())
        .layer(Dense::default().size(targ_size).activation(Linear::build()).build())
        .optimizer(Adam::default().beta1(0.99).beta2(0.999).learning_rate(0.0030).build())
        .loss_function(MeanSquared.build())
        .early_stopper(
            Flexible::default()
                .monitor_metric(MonitorMetric::Loss)
                .patience(500)
                .min_delta(0.1)
                .smoothing_factor(0.5)
                .build(),
        )
        .batch_size(7)
        .batch_group_size(2)
        .parallelize(2)
        .normalize_input(MinMax::default())
        .epochs(500)
        .seed(55)
        .build();

    match network {
        Ok(net) => net,
        Err(e) => {
            eprintln!("Failed to build network: {}", e);
            std::process::exit(1);
        }
    }
}

fn search(training_inputs: &DMat, training_targets: &DMat, validation_inputs: &DMat, validation_targets: &DMat) {
    let start_time = std::time::Instant::now();
    info!("Energy Efficieny network search started");
    let network = energy_efficiency_network(training_inputs.cols(), training_targets.cols());

    let network_search = NetworkSearchBuilder::new()
        .network(network)
        .parallelize(4)
        .normalize_input(MinMax::default())
        .learning_rates(
            SequentialNumbers::new()
                .lower_limit(0.0025)
                .upper_limit(0.0035)
                .increment(0.0005)
                .floats(),
        )
        .batch_sizes(
            SequentialNumbers::new()
                .lower_limit(6.0)
                .upper_limit(9.0)
                .increment(1.0)
                .ints(),
        )
        .hidden_layer(
            SequentialNumbers::new()
                .lower_limit(14.0)
                .upper_limit(18.0)
                .increment(2.0)
                .ints(),
            ReLU::build(),
        )
        .hidden_layer(
            SequentialNumbers::new()
                .lower_limit(14.0)
                .upper_limit(18.0)
                .increment(2.0)
                .ints(),
            ReLU::build(),
        )
        .export(
            CSV::default()
                .directory(EXP_NAME)
                .file_name(&format!("{}_search", EXP_NAME))
                .build(),
        )
        .build();
    let mut network_search = match network_search {
        Ok(ns) => ns,
        Err(e) => {
            error!("Failed to build network_search: {}", e);
            std::process::exit(1);
        }
    };

    let search_res = network_search
        .search(training_inputs, training_targets, validation_inputs, validation_targets)
        .unwrap();

    info!("Energy Efficieny network search finished in {} seconds", start_time.elapsed().as_secs());
    info!("Num Results: {}", search_res.len());
}

pub fn energy_efficiency_inputs_targets(
    name: &str, fields_count: usize, input_count: usize,
) -> Result<(DMat, DMat, DMat, DMat), Box<dyn Error>> {
    let target_count = fields_count - input_count;

    let file_path = format!("./examples/energy_efficiency/{}.csv", name);
    let file = File::open(&file_path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut inputs_data = Vec::new();
    let mut targets_data = Vec::new();

    for (index, result) in reader.records().enumerate() {
        let record = result?;
        // Skip if record is empty
        if record.is_empty() {
            error!("Skipping empty record at line {}", index + 2); // +2 because of header + 0-indexed
            continue;
        }
        // If record has wrong number of fields, print detailed info
        if record.len() != fields_count {
            error!(
                "Bad record at line {}: expected {} fields, but got {} fields",
                index + 2,
                fields_count,
                record.len()
            );
            error!("Record content: {:?}", record);
            return Err("Unexpected number of fields".to_string().into());
        }
        for (i, value) in record.iter().enumerate() {
            let parsed_val: f32 = value.parse()?;
            if i >= fields_count - target_count {
                targets_data.push(parsed_val);
            } else {
                inputs_data.push(parsed_val);
            }
        }
    }

    let all_inputs = DenseMatrix::new(inputs_data.len() / input_count, input_count)
        .data(&inputs_data)
        .build()
        .unwrap();
    let all_targets = DenseMatrix::new(targets_data.len() / target_count, target_count)
        .data(&targets_data)
        .build()
        .unwrap();

    let (training_inputs, training_targets, validation_inputs, validation_targets) =
        helper::random_split(&all_inputs, &all_targets, 0.2, 55);

    Ok((training_inputs, training_targets, validation_inputs, validation_targets))
}

/// Initializes the logger for the application.
/// The LOG environment variable is used to define the log level (e.g., info, debug, warn, error).
/// If the LOG variable is not set, it defaults to info.
fn initialize_logger(name: &str) {
    // Check if the directory exists, and attempt to create it if it doesn't
    if !std::path::Path::new(name).exists() {
        let _res = fs::create_dir_all(name).map_err(|e| {
            eprintln!("Failed to create log directory: {}", e);
        });
    }

    // Attempt to create a log file
    let log_file = match File::create(format!("./{}/{}.log", name, name)) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Failed to create log file: {}", e);
            return;
        }
    };

    // Check if the "LOG" environment variable is set
    let log_level = env::var("LOG").unwrap_or_else(|_| "info".to_string()); // Default to "info"

    // Initialize the logger with the specified log level
    Builder::new()
        .target(Target::Pipe(Box::new(log_file)))
        .parse_filters(&log_level) // Use the log level from the environment variable
        .init();
}

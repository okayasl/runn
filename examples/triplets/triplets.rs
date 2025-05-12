mod data;

use std::{
    env,
    fs::{self, File},
};

use env_logger::{Builder, Target};
use log::{error, info};
use runn::{
    adam::Adam,
    cross_entropy::CrossEntropy,
    csv::CSV,
    dense_layer::Dense,
    helper,
    network::network_model::{Network, NetworkBuilder},
    network_io::JSON,
    network_search::NetworkSearchBuilder,
    numbers::{Numbers, SequentialNumbers},
    relu::ReLU,
    softmax::Softmax,
};

const EXP_NAME: &str = "triplets";

// Triplets is a Multi-class classification problem.
// One-hot encoding problem with 3 classes.
// predict 1,0,0 if all input elements are same
// predict 0,1,0 if only two of the input elements are same
// predict 0,0,1 if none of the input elements are same

/// This example demonstrates how to train and validate a neural network on the Triplets dataset.
/// Triplets is a Multi-class classification problem.
///
///  - predict 1,0,0 if all input elements are same
///  - predict 0,1,0 if only two of the input elements are same
///  - predict 0,0,1 if none of the input elements are same
///
/// The code includes functions to load the dataset, build the neural network,
/// train the network, validate its performance, and perform a hyperparameter search.
///
/// to run the example:
/// ```bash
/// cargo run --example triplets
/// ```
/// to run the hyperparameter search:
/// ```bash
/// cargo run --example triplets -- -search
/// ```
/// The hyperparameter search will create a CSV file with the results in the `triplets` directory.
/// The training and validation results will be logged in the `triplets` directory.
fn main() {
    initialize_logger(EXP_NAME);

    let args: Vec<String> = env::args().collect();
    if args.contains(&"-search".to_string()) {
        search();
    } else {
        train_and_validate();
    }
}

fn train_and_validate() {
    let network_file = format!("{}_network", EXP_NAME);
    let training_inputs = data::training_inputs();
    let training_targets = data::training_targets();
    let mut network = triplets_network(training_inputs.cols(), training_targets.cols());

    let train_result = network.train(&training_inputs, &training_targets);
    match train_result {
        Ok(_) => {
            println!("Training completed successfully");
            network
                .save(
                    JSON::default()
                        .directory(EXP_NAME)
                        .file_name(&network_file)
                        .build()
                        .unwrap(),
                )
                .unwrap();
            let net_results = network.predict(&training_inputs, &training_targets).unwrap();
            log::info!(
                "{}",
                helper::pretty_compare_matrices(
                    &training_inputs,
                    &training_targets,
                    &net_results.predictions,
                    helper::CompareMode::Classification
                )
            );
            info!("Training: {}", net_results.display_metrics());
        }
        Err(e) => {
            eprintln!("Training failed: {}", e);
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
    let validation_inputs = data::validation_inputs();
    let validation_targets = data::validation_targets();
    let net_results = network.predict(&validation_inputs, &validation_targets).unwrap();
    log::info!(
        "{}",
        helper::pretty_compare_matrices(
            &validation_inputs,
            &validation_targets,
            &net_results.predictions,
            helper::CompareMode::Classification
        )
    );
    info!("Validation: {}", net_results.display_metrics());
}

fn search() {
    let training_inputs = data::training_inputs();
    let training_targets = data::training_targets();

    let validation_inputs = data::validation_inputs();
    let validation_targets = data::validation_targets();

    let network = triplets_network(training_inputs.cols(), training_targets.cols());

    let network_search = NetworkSearchBuilder::new()
        .network(network)
        .parallelize(4)
        .learning_rates(
            SequentialNumbers::new()
                .lower_limit(0.0025)
                .upper_limit(0.0035)
                .increment(0.0005)
                .floats(),
        )
        .batch_sizes(
            SequentialNumbers::new()
                .lower_limit(5.0)
                .upper_limit(10.0)
                .increment(1.0)
                .ints(),
        )
        .hidden_layer(
            SequentialNumbers::new()
                .lower_limit(12.0)
                .upper_limit(24.0)
                .increment(4.0)
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
        .search(&training_inputs, &training_targets, &validation_inputs, &validation_targets)
        .unwrap();

    info!("Num Results: {}", search_res.len());
}

fn triplets_network(inp_size: usize, targ_size: usize) -> Network {
    let network = NetworkBuilder::new(inp_size, targ_size)
        .layer(Dense::default().size(24).activation(ReLU::build()).build())
        .layer(Dense::default().size(targ_size).activation(Softmax::build()).build())
        .loss_function(CrossEntropy::default().epsilon(1e-8).build())
        .optimizer(Adam::default().beta1(0.99).beta2(0.999).learning_rate(0.0035).build())
        .batch_size(8)
        .batch_group_size(2)
        .parallelize(2)
        .epochs(1000)
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

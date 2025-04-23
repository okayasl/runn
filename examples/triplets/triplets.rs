mod data;

use std::{env, fs::File};

use env_logger::{Builder, Target};
use log::info;
use rayon::ThreadPoolBuilder;
use runn::{
    adam::Adam,
    cross_entropy::CrossEntropy,
    network::network::{Network, NetworkBuilder},
    network_search::NetworkSearchBuilder,
    numbers::{Numbers, SequentialNumbers},
    relu::ReLU,
    softmax::Softmax,
    util, Dense,
};

// Triplets is a Multi-class classification problem.
// One-hot encoding problem with 3 classes.
// predict 1,0,0 if all input elements are same
// predict 0,1,0 if only two of the input elements are same
// predict 0,0,1 if none of the input elements are same
fn main() {
    initialize_logger();
    initialize_thread_pool();

    let args: Vec<String> = env::args().collect();
    if args.contains(&"-search".to_string()) {
        search();
    } else {
        train_and_validate();
    }
}

fn train_and_validate() {
    let triplets_file = String::from("triplets_network.json");
    let training_inputs = data::training_inputs();
    let training_targets = data::training_targets();
    let mut network = triplets_network(training_inputs.cols(), training_targets.cols());

    let train_result = network.train(&training_inputs, &training_targets);
    match train_result {
        Ok(_) => {
            println!("Training completed successfully");
            network.save(&triplets_file, runn::network_io::SerializationFormat::Json);
            let net_results = network.predict(&training_inputs, &training_targets);
            util::print_matrices_comparison(&training_inputs, &training_targets, &net_results.predictions);
            util::print_metrics(
                net_results.accuracy,
                net_results.loss,
                net_results.metrics.macro_f1_score,
                net_results.metrics.micro_f1_score,
                net_results.metrics.micro_recall,
                net_results.metrics.micro_precision,
            );
        }
        Err(e) => {
            eprintln!("Training failed: {}", e);
        }
    }

    network = Network::load(&triplets_file, runn::network_io::SerializationFormat::Json);
    let validation_inputs = data::validation_inputs();
    let validation_targets = data::validation_targets();
    let net_results = network.predict(&validation_inputs, &validation_targets);
    util::print_matrices_comparison(&validation_inputs, &validation_targets, &net_results.predictions);
    util::print_metrics(
        net_results.accuracy,
        net_results.loss,
        net_results.metrics.macro_f1_score,
        net_results.metrics.micro_f1_score,
        net_results.metrics.micro_recall,
        net_results.metrics.micro_precision,
    );
}

fn search() {
    let training_inputs = data::training_inputs();
    let training_targets = data::training_targets();

    let validation_inputs = data::validation_inputs();
    let validation_targets = data::validation_targets();

    let network = triplets_network(training_inputs.cols(), training_targets.cols());

    let mut network_search = NetworkSearchBuilder::new()
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
            ReLU::new(),
        )
        .export("triplets_search".to_string())
        .build();

    let search_res =
        network_search.search(&training_inputs, &training_targets, &validation_inputs, &validation_targets);

    info!("Num Results: {}", search_res.len());
}

fn triplets_network(inp_size: usize, targ_size: usize) -> Network {
    let network = NetworkBuilder::new(inp_size, targ_size)
        .layer(Dense::new().size(24).activation(ReLU::new()).build())
        .layer(Dense::new().size(targ_size).activation(Softmax::new()).build())
        .loss_function(CrossEntropy::new().epsilon(1e-8).build())
        .optimizer(Adam::new().beta1(0.99).beta2(0.999).learning_rate(0.0035).build())
        // .early_stopper(
        //     Flexible::new()
        //         .patience(1000)
        //         .min_delta(0.000001)
        //         .monitor_accuracy(true)
        //         .build(),
        // )
        .batch_size(8)
        .batch_group_size(2)
        //.parallelize(2)
        .epochs(1500)
        .seed(55)
        //.summary(TensorBoard::new().logdir("summary").build())
        //.debug(true)
        .build();

    match network {
        Ok(net) => net,
        Err(e) => {
            eprintln!("Failed to build network: {}", e);
            std::process::exit(1);
        }
    }
}

/// Initializes the global thread pool with a specified number of threads.
/// The number of threads is set to 2 in this example.
/// The thread pool is used for parallel processing in the application.
/// It is built using the `ThreadPoolBuilder` from the `rayon` crate.
/// The `build_global` method creates a global thread pool that can be used across the application.
fn initialize_thread_pool() {
    ThreadPoolBuilder::new()
        .num_threads(2)
        .build_global()
        .expect("Failed to build global thread pool");
}

/// Initializes the logger for the application.
/// It creates a log file named "app.log" and sets the log level to "info" by default.
/// The LOG environment variable is used to define the log level (e.g., info, debug, warn, error).
/// If the LOG variable is not set, it defaults to info.
fn initialize_logger() {
    // Attempt to create a log file
    let log_file = match File::create("triplets.log") {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Failed to create log file: {}", e);
            return; // Exit the function if the log file cannot be created
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

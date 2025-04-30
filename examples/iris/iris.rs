use csv::ReaderBuilder;
use env_logger::{Builder, Target};
use log::info;
use runn::{
    adam::Adam,
    cross_entropy::CrossEntropy,
    layer::Dense,
    matrix::DenseMatrix,
    network::network::{Network, NetworkBuilder},
    network_search::NetworkSearchBuilder,
    numbers::{Numbers, SequentialNumbers},
    relu::ReLU,
    softmax::Softmax,
    util,
};
use std::env;
use std::error::Error;
use std::fs::File;

// This example demonstrates how to train a neural network on the Iris dataset using the runn library.
// It includes functions for training, validation, and hyperparameter search.
// The Iris dataset is a classic dataset for classification tasks, and this example shows how to
// use the runn library to build and train a neural network on this dataset.
fn main() {
    initialize_logger();

    let args: Vec<String> = env::args().collect();
    if args.contains(&"-search".to_string()) {
        test_search();
    } else {
        train_and_validate();
    }
}

fn train_and_validate() {
    let filed = String::from("iris_network.json");
    let (training_inputs, training_targets) = iris_inputs_outputs("train", 7, 4).unwrap();
    let mut network = iris_network(training_inputs.cols(), training_targets.cols());

    let training_result = network.train(&training_inputs, &training_targets);
    match training_result {
        Ok(_) => {
            println!("Training completed successfully");
            network.save(&filed, runn::network_io::SerializationFormat::Json);
            let net_results = network.predict(&training_inputs, &training_targets);
            util::print_matrices_comparison(&training_inputs, &training_targets, &net_results.predictions);
            info!("Training: {}", net_results.display_metrics());
        }
        Err(e) => {
            eprintln!("Training failed: {}", e);
        }
    }

    network = Network::load(&filed, runn::network_io::SerializationFormat::Json);
    let (validation_inputs, validation_targets) = iris_inputs_outputs("test", 7, 4).unwrap();
    let net_results = network.predict(&validation_inputs, &validation_targets);
    util::print_matrices_comparison(&validation_inputs, &validation_targets, &net_results.predictions);
    info!("Validation: {}", net_results.display_metrics());
}

fn iris_network(inp_size: usize, targ_size: usize) -> Network {
    let network = NetworkBuilder::new(inp_size, targ_size)
        .layer(Dense::new().size(12).activation(ReLU::new()).build())
        .layer(Dense::new().size(12).activation(ReLU::new()).build())
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
        .batch_size(9)
        // .batch_group_size(2)
        // .parallelize(2)
        .epochs(500)
        .seed(55)
        //.summary(TensorBoard::new().logdir("iris_summary").build())
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

fn test_search() {
    let (training_inputs, training_targets) = iris_inputs_outputs("train", 7, 4).unwrap();
    let (validation_inputs, validation_targets) = iris_inputs_outputs("test", 7, 4).unwrap();

    let network = iris_network(training_inputs.cols(), training_targets.cols());

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
                .lower_limit(7.0)
                .upper_limit(10.0)
                .increment(1.0)
                .ints(),
        )
        .hidden_layer(
            SequentialNumbers::new()
                .lower_limit(12.0)
                .upper_limit(20.0)
                .increment(4.0)
                .ints(),
            ReLU::new(),
        )
        .hidden_layer(
            SequentialNumbers::new()
                .lower_limit(12.0)
                .upper_limit(20.0)
                .increment(4.0)
                .ints(),
            ReLU::new(),
        )
        .export("iris_search".to_string())
        .build();

    let search_res =
        network_search.search(&training_inputs, &training_targets, &validation_inputs, &validation_targets);

    info!("Num Results: {}", search_res.len());
}

pub fn iris_inputs_outputs(
    name: &str, fields_count: usize, input_count: usize,
) -> Result<(DenseMatrix, DenseMatrix), Box<dyn Error>> {
    let target_count = fields_count - input_count;

    let file_path = format!("./examples/iris/{}.csv", name);
    let file = File::open(&file_path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut inputs_data = Vec::new();
    let mut labels_data = Vec::new();

    for result in reader.records() {
        let record = result?;
        for (i, value) in record.iter().enumerate() {
            let parsed_val: f32 = value.parse()?;
            if i >= fields_count - target_count {
                labels_data.push(parsed_val);
            } else {
                inputs_data.push(parsed_val);
            }
        }
    }

    let data_length = inputs_data.len() / input_count;

    let inputs = DenseMatrix::new(data_length, input_count, &inputs_data);
    let labels = DenseMatrix::new(data_length, target_count, &labels_data);

    Ok((inputs, labels))
}

/// Initializes the logger for the application.
/// It creates a log file named "app.log" and sets the log level to "info" by default.
/// The LOG environment variable is used to define the log level (e.g., info, debug, warn, error).
/// If the LOG variable is not set, it defaults to info.
fn initialize_logger() {
    // Attempt to create a log file
    let log_file = match File::create("iris.log") {
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

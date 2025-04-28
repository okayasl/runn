use csv::ReaderBuilder;
use env_logger::{Builder, Target};
use log::info;
use runn::{
    adam::Adam,
    cross_entropy::CrossEntropy,
    layer::Dense,
    matrix::DenseMatrix,
    min_max::MinMax,
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

// This example demonstrates how to train a neural network on the wine dataset using the runn library.
// It includes functions for training, validation, and hyperparameter search.
// The wine dataset is a classic dataset for classification tasks, and this example shows how to
// use the runn library to build and train a neural network on this dataset.
fn main() {
    initialize_logger();

    let (red_training_inputs, red_training_targets, red_validation_inputs, red_validation_targets) =
        wine_inputs_targets("winequality-red", 12, 11).unwrap();
    // let (white_training_inputs, white_training_targets, white_validation_inputs, white_validation_targets) =
    //     wine_inputs_targets("winequality-white", 11, 1).unwrap();

    let red_training_targets = util::one_hot_encode(&red_training_targets);
    let red_validation_targets = util::one_hot_encode(&red_validation_targets);

    let args: Vec<String> = env::args().collect();
    if args.contains(&"-search".to_string()) {
        test_search(&red_training_inputs, &red_training_targets, &red_validation_inputs, &red_validation_targets);
    } else {
        train_and_validate(
            &red_training_inputs,
            &red_training_targets,
            &red_validation_inputs,
            &red_validation_targets,
        );
    }
}

fn train_and_validate(
    training_inputs: &DenseMatrix, training_targets: &DenseMatrix, validation_inputs: &DenseMatrix,
    validation_targets: &DenseMatrix,
) {
    let filed = String::from("wine_network.json");

    let mut network = one_hot_encode_network(training_inputs.cols(), training_targets.cols());

    let training_result = network.train(&training_inputs, &training_targets);
    match training_result {
        Ok(_) => {
            println!("Training completed successfully");
            network.save(&filed, runn::network_io::SerializationFormat::Json);
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

    network = Network::load(&filed, runn::network_io::SerializationFormat::Json);
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

fn one_hot_encode_network(inp_size: usize, targ_size: usize) -> Network {
    let network = NetworkBuilder::new(inp_size, targ_size)
        .layer(Dense::new().size(19).activation(ReLU::new()).build())
        .layer(Dense::new().size(13).activation(ReLU::new()).build())
        .layer(Dense::new().size(targ_size).activation(Softmax::new()).build())
        .optimizer(Adam::new().beta1(0.99).beta2(0.999).learning_rate(0.0025).build())
        .loss_function(CrossEntropy::new().epsilon(1e-8).build())
        // .early_stopper(
        //     Flexible::new()
        //         .patience(100)
        //         .min_delta(0.000001)
        //         .monitor_metric(MonitorMetric::Accuracy)
        //         .build(),
        // )
        .batch_size(8)
        .batch_group_size(2)
        .parallelize(2)
        .normalize_input(MinMax::new())
        .epochs(7000)
        .seed(55)
        //.summary(TensorBoard::new().logdir("wine_summary").build())
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

fn test_search(
    training_inputs: &DenseMatrix, training_targets: &DenseMatrix, validation_inputs: &DenseMatrix,
    validation_targets: &DenseMatrix,
) {
    let network = one_hot_encode_network(training_inputs.cols(), training_targets.cols());

    let mut network_search = NetworkSearchBuilder::new()
        .network(network)
        .parallelize(4)
        .normalize_input(MinMax::new())
        .learning_rates(
            SequentialNumbers::new()
                .lower_limit(0.0015)
                .upper_limit(0.0025)
                .increment(0.0005)
                .floats(),
        )
        .batch_sizes(
            SequentialNumbers::new()
                .lower_limit(8.0)
                .upper_limit(8.0)
                .increment(1.0)
                .ints(),
        )
        .hidden_layer(
            SequentialNumbers::new()
                .lower_limit(10.0)
                .upper_limit(19.0)
                .increment(3.0)
                .ints(),
            ReLU::new(),
        )
        .hidden_layer(
            SequentialNumbers::new()
                .lower_limit(13.0)
                .upper_limit(16.0)
                .increment(3.0)
                .ints(),
            ReLU::new(),
        )
        .export("wine_search".to_string())
        .build();

    let search_res =
        network_search.search(&training_inputs, &training_targets, &validation_inputs, &validation_targets);

    info!("Num Results: {}", search_res.len());
}

pub fn wine_inputs_targets(
    name: &str, fields_count: usize, input_count: usize,
) -> Result<(DenseMatrix, DenseMatrix, DenseMatrix, DenseMatrix), Box<dyn Error>> {
    let target_count = fields_count - input_count;

    let file_path = format!("./examples/wine_quality/{}.csv", name);
    let file = File::open(&file_path)?;
    let mut reader = ReaderBuilder::new().delimiter(b';').has_headers(true).from_reader(file);

    let mut inputs_data = Vec::new();
    let mut targets_data = Vec::new();

    for (index, result) in reader.records().enumerate() {
        let record = result?;
        // Skip if record is empty
        if record.len() == 0 {
            println!("Skipping empty record at line {}", index + 2); // +2 because of header + 0-indexed
            continue;
        }
        // If record has wrong number of fields, print detailed info
        if record.len() != fields_count {
            println!(
                "Bad record at line {}: expected {} fields, but got {} fields.",
                index + 2,
                fields_count,
                record.len()
            );
            println!("Record content: {:?}", record);
            return Err(format!("Unexpected number of fields").into());
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

    let data_length = inputs_data.len() / input_count;

    // set first 80 percent of the data for training
    let training_data_length = (data_length as f32 * 0.8).round() as usize;
    let training_inputs_data = &inputs_data[0..training_data_length * input_count];
    let training_targets_data = &targets_data[0..training_data_length * target_count];
    let training_inputs = DenseMatrix::new(training_data_length, input_count, &training_inputs_data);
    let training_targets = DenseMatrix::new(training_data_length, target_count, &training_targets_data);
    // set last 20 percent of the data for validation
    let validation_data_length = data_length - training_data_length;
    let validation_inputs_data = &inputs_data[training_data_length * input_count..];
    let validation_targets_data = &targets_data[training_data_length * target_count..];
    let validation_inputs = DenseMatrix::new(validation_data_length, input_count, &validation_inputs_data);
    let validation_targets = DenseMatrix::new(validation_data_length, target_count, &validation_targets_data);

    Ok((training_inputs, training_targets, validation_inputs, validation_targets))
}

/// Initializes the logger for the application.
/// It creates a log file named "app.log" and sets the log level to "info" by default.
/// The LOG environment variable is used to define the log level (e.g., info, debug, warn, error).
/// If the LOG variable is not set, it defaults to info.
fn initialize_logger() {
    // Attempt to create a log file
    let log_file = match File::create("wine_quality.log") {
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

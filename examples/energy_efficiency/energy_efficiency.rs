use csv::ReaderBuilder;
use env_logger::{Builder, Target};
use log::info;
use runn::{
    adam::Adam,
    layer::Dense,
    linear::Linear,
    matrix::DenseMatrix,
    mean_squared_error::MeanSquared,
    min_max::MinMax,
    network::network::{Network, NetworkBuilder},
    network_search::NetworkSearchBuilder,
    numbers::{Numbers, SequentialNumbers},
    relu::ReLU,
    util,
};
use std::env;
use std::error::Error;
use std::fs::File;

/// This example demonstrates how to use the runn library to train a neural network on the energy efficiency dataset.
/// The dataset is used for predicting the energy efficiency of buildings based on various features.
/// The goal is to classify the energy efficiency into different classes.
/// In this dataset energy analysis using 12 different building shapes simulated in Ecotect is performed.
/// The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters.
/// The dataset comprises 768 samples and 8 features, aiming to predict two real valued responses.
/// It can also be used as a multi-class classification problem if the response is rounded to the nearest integer.
fn main() {
    initialize_logger();

    let (training_inputs, training_targets, validation_inputs, validation_targets) =
        energy_efficiency_inputs_targets("ENB2012_data", 10, 8).unwrap();

    let args: Vec<String> = env::args().collect();
    if args.contains(&"-search".to_string()) {
        test_search(&training_inputs, &training_targets, &validation_inputs, &validation_targets);
    } else {
        train_and_validate(&training_inputs, &training_targets, &validation_inputs, &validation_targets);
    }
}

fn train_and_validate(
    training_inputs: &DenseMatrix, training_targets: &DenseMatrix, validation_inputs: &DenseMatrix,
    validation_targets: &DenseMatrix,
) {
    let filed = String::from("energy_efficiency_network.json");

    let mut network = energy_efficiency_network(training_inputs.cols(), training_targets.cols());

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
    let net_results = network.predict(&validation_inputs, &validation_targets);
    util::print_matrices_comparison(&validation_inputs, &validation_targets, &net_results.predictions);
    info!("Validation: {}", net_results.display_metrics());
}

fn energy_efficiency_network(inp_size: usize, targ_size: usize) -> Network {
    let network = NetworkBuilder::new(inp_size, targ_size)
        .layer(Dense::new().size(7).activation(ReLU::new()).build())
        .layer(Dense::new().size(7).activation(ReLU::new()).build())
        .layer(Dense::new().size(targ_size).activation(Linear::new()).build())
        .optimizer(Adam::new().beta1(0.99).beta2(0.999).learning_rate(0.0025).build())
        .loss_function(MeanSquared::new())
        // .early_stopper(
        //     Flexible::new()
        //         .patience(100)
        //         .min_delta(0.000001)
        //         .monitor_metric(MonitorMetric::Accuracy)
        //         .build(),
        // )
        .batch_size(6)
        .batch_group_size(2)
        .parallelize(2)
        .normalize_input(MinMax::new())
        .epochs(1000)
        .seed(55)
        //.summary(TensorBoard::new().logdir("energy_efficiency_summary").build())
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
    let start_time = std::time::Instant::now();
    info!("Energy Efficieny network search started.");
    let network = energy_efficiency_network(training_inputs.cols(), training_targets.cols());

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
                .lower_limit(6.0)
                .upper_limit(9.0)
                .increment(1.0)
                .ints(),
        )
        .hidden_layer(
            SequentialNumbers::new()
                .lower_limit(8.0)
                .upper_limit(18.0)
                .increment(2.0)
                .ints(),
            ReLU::new(),
        )
        .hidden_layer(
            SequentialNumbers::new()
                .lower_limit(8.0)
                .upper_limit(18.0)
                .increment(2.0)
                .ints(),
            ReLU::new(),
        )
        .export("energy_efficiency_search".to_string())
        .build();

    let search_res =
        network_search.search(&training_inputs, &training_targets, &validation_inputs, &validation_targets);

    info!("Energy Efficieny network search finished in {} seconds.", start_time.elapsed().as_secs());
    info!("Num Results: {}", search_res.len());
}

pub fn energy_efficiency_inputs_targets(
    name: &str, fields_count: usize, input_count: usize,
) -> Result<(DenseMatrix, DenseMatrix, DenseMatrix, DenseMatrix), Box<dyn Error>> {
    let target_count = fields_count - input_count;

    let file_path = format!("./examples/energy_efficiency/{}.csv", name);
    let file = File::open(&file_path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

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

    let all_inputs = DenseMatrix::new(inputs_data.len() / input_count, input_count, &inputs_data);
    let all_targets = DenseMatrix::new(targets_data.len() / target_count, target_count, &targets_data);

    let (training_inputs, training_targets, validation_inputs, validation_targets) =
        util::random_split(&all_inputs, &all_targets, 0.2, 55);

    Ok((training_inputs, training_targets, validation_inputs, validation_targets))
}

/// Initializes the logger for the application.
/// It creates a log file named "app.log" and sets the log level to "info" by default.
/// The LOG environment variable is used to define the log level (e.g., info, debug, warn, error).
/// If the LOG variable is not set, it defaults to info.
fn initialize_logger() {
    // Attempt to create a log file
    let log_file = match File::create("energy_efficiency.log") {
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

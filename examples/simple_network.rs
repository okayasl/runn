use env_logger::{Builder, Target};
use log::info;
use runn::{
    adam::Adam,
    cross_entropy::CrossEntropy,
    layer::Dense,
    matrix::DenseMatrix,
    network::network::{Network, NetworkBuilder},
    network_search::{search, SearchConfigsBuilder},
    relu::ReLU,
    search_param::{Parameters, RangeParameters},
    softmax::Softmax,
    util,
};

use std::{env, fs::File};

fn main() {
    // Create a log file
    let log_file = File::create("app.log").expect("Could not create log file");

    // // Initialize the logger to write to the log file
    // Builder::new()
    //     .target(Target::Pipe(Box::new(log_file)))
    //     .init();

    // Initialize the logger with a default level of "info"
    if env::var("LOG").is_err() {
        Builder::from_default_env()
            .target(Target::Pipe(Box::new(log_file)))
            .filter_level(log::LevelFilter::Info)
            .init();
    } else {
        env_logger::init();
    }

    let args: Vec<String> = env::args().collect();
    if args.contains(&"-grs".to_string()) {
        test_search();
    } else {
        train_and_validate();
    }
}

// fn test_training() {
//     let training_inputs = get_training_input_matrix();
//     let training_targets = get_training_target_matrix();
//     let mut network = generate_network(training_inputs.cols(), training_targets.cols());

//     let training_result = network.train(&training_inputs, &training_targets);
//     match training_result {
//         Ok(_) => {
//             network.save("network.json", runn::network_io::SerializationFormat::Json);
//             println!("Training completed successfully");
//             let net_results = network.predict(&training_inputs, &training_targets);
//             print_matrices_comparisons(
//                 &training_inputs,
//                 &training_targets,
//                 &net_results.predictions,
//             );
//             print_metrics(
//                 net_results.accuracy,
//                 net_results.loss,
//                 net_results.metrics.macro_f1_score,
//                 net_results.metrics.micro_f1_score,
//                 net_results.metrics.micro_recall,
//                 net_results.metrics.micro_precision,
//             );
//         }
//         Err(e) => {
//             eprintln!("Training failed: {}", e);
//         }
//     }
// }

// fn test_validation() {
//     let validation_inputs = get_validation_input_matrix();
//     let validation_targets = get_validation_target_matrix();
//     let mut network = generate_network(validation_inputs.cols(), validation_targets.cols());

//     let training_result = network.train(&validation_inputs, &validation_targets);
//     match training_result {
//         Ok(_) => {
//             println!("Training completed successfully");
//             let net_results = network.predict(&validation_inputs, &validation_targets);
//             print_matrices_comparisons(
//                 &validation_inputs,
//                 &validation_targets,
//                 &net_results.predictions,
//             );
//             print_metrics(
//                 net_results.accuracy,
//                 net_results.loss,
//                 net_results.metrics.macro_f1_score,
//                 net_results.metrics.micro_f1_score,
//                 net_results.metrics.micro_recall,
//                 net_results.metrics.micro_precision,
//             );
//         }
//         Err(e) => {
//             eprintln!("Training failed: {}", e);
//         }
//     }
// }

fn train_and_validate() {
    let filed = String::from("network.json");
    let training_inputs = get_training_input_matrix();
    let training_targets = get_training_target_matrix();
    let mut network = generate_network(training_inputs.cols(), training_targets.cols());

    let training_result = network.train(&training_inputs, &training_targets);
    match training_result {
        Ok(_) => {
            println!("Training completed successfully");
            network.save(&filed, runn::network_io::SerializationFormat::Json);
            let net_results = network.predict(&training_inputs, &training_targets);
            util::print_matrices_comparisons(
                &training_inputs,
                &training_targets,
                &net_results.predictions,
            );
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
    let validation_inputs = get_validation_input_matrix();
    let validation_targets = get_validation_target_matrix();
    let net_results = network.predict(&validation_inputs, &validation_targets);
    util::print_matrices_comparisons(
        &validation_inputs,
        &validation_targets,
        &net_results.predictions,
    );
    util::print_metrics(
        net_results.accuracy,
        net_results.loss,
        net_results.metrics.macro_f1_score,
        net_results.metrics.micro_f1_score,
        net_results.metrics.micro_recall,
        net_results.metrics.micro_precision,
    );
}

fn generate_network(inp_size: usize, targ_size: usize) -> Network {
    let network = NetworkBuilder::new(inp_size, targ_size)
        .layer(Dense::new().size(16).activation(ReLU::new()).build())
        .layer(
            Dense::new()
                .size(targ_size)
                .activation(Softmax::new())
                .build(),
        )
        .loss_function(CrossEntropy::new().epsilon(1e-8).build())
        .optimizer(
            Adam::new()
                .beta1(0.99)
                .beta2(0.999)
                .learning_rate(0.0035)
                .build(),
        )
        // .early_stopper(
        //     Flexible::new()
        //         .patience(1000)
        //         .min_delta(0.000001)
        //         .monitor_accuracy(true)
        //         .build(),
        // )
        .batch_size(10)
        .batch_group_size(1)
        .epochs(300)
        .seed(55)
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

fn get_training_input_matrix() -> DenseMatrix {
    let inps = vec![
        5.0, 4.0, 4.0, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 1.0, 6.0, 2.0, 1.0, 3.0, 6.0,
        5.0, 5.0, 5.0, 3.0, 2.0, 6.0, 4.0, 9.0, 4.0, 0.0, 0.0, 0.0, 2.0, 6.0, 4.0, 2.0, 6.0, 2.0,
        8.0, 8.0, 8.0, 2.0, 7.0, 2.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 6.0, 6.0, 6.0, 1.0, 1.0, 1.0,
        1.0, 3.0, 3.0, 2.0, 2.0, 2.0, 6.0, 7.0, 7.0, 8.0, 4.0, 7.0, 4.0, 4.0, 4.0, 4.0, 7.0, 5.0,
        1.0, 2.0, 8.0, 1.0, 9.0, 3.0, 0.0, 4.0, 0.0, 3.0, 7.0, 7.0, 2.0, 5.0, 1.0, 8.0, 1.0, 1.0,
        4.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 7.0, 0.0, 0.0,
        6.0, 6.0, 6.0, 7.0, 1.0, 4.0, 2.0, 2.0, 2.0, 8.0, 7.0, 0.0, 9.0, 2.0, 2.0, 7.0, 7.0, 7.0,
        0.0, 0.0, 0.0, 6.0, 5.0, 5.0, 4.0, 4.0, 4.0, 3.0, 1.0, 6.0, 5.0, 1.0, 5.0, 2.0, 2.0, 2.0,
        8.0, 1.0, 9.0, 5.0, 8.0, 8.0, 0.0, 6.0, 1.0, 3.0, 6.0, 6.0, 1.0, 4.0, 7.0, 6.0, 6.0, 6.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 6.0, 3.0, 0.0, 4.0, 4.0, 4.0, 9.0, 9.0, 9.0, 1.0, 3.0, 0.0,
        0.0, 5.0, 5.0, 6.0, 5.0, 5.0, 0.0, 5.0, 5.0, 0.0, 1.0, 1.0, 5.0, 1.0, 7.0, 7.0, 6.0, 7.0,
        2.0, 0.0, 7.0, 3.0, 6.0, 0.0, 8.0, 5.0, 9.0, 0.0, 4.0, 0.0, 9.0, 8.0, 9.0, 4.0, 7.0, 6.0,
        2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 6.0, 7.0, 8.0, 3.0, 9.0, 5.0, 1.0, 0.0, 8.0, 9.0, 9.0,
        3.0, 3.0, 3.0, 9.0, 7.0, 1.0, 9.0, 7.0, 5.0, 7.0, 7.0, 7.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
        8.0, 3.0, 3.0, 9.0, 9.0, 9.0, 2.0, 6.0, 2.0, 4.0, 2.0, 8.0, 5.0, 9.0, 9.0, 3.0, 9.0, 7.0,
        7.0, 8.0, 4.0, 9.0, 8.0, 3.0, 7.0, 3.0, 9.0, 7.0, 2.0, 7.0, 5.0, 5.0, 5.0, 7.0, 2.0, 7.0,
        0.0, 6.0, 6.0, 9.0, 1.0, 7.0, 9.0, 6.0, 6.0, 6.0, 9.0, 9.0, 2.0, 6.0, 9.0, 9.0, 8.0, 8.0,
        7.0, 4.0, 5.0, 0.0, 5.0, 7.0, 2.0, 2.0, 2.0, 7.0, 3.0, 9.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0,
        1.0, 1.0, 1.0, 7.0, 7.0, 7.0, 2.0, 2.0, 2.0, 8.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        8.0, 1.0, 0.0, 7.0, 7.0, 7.0, 4.0, 2.0, 4.0, 9.0, 9.0, 9.0, 2.0, 2.0, 2.0, 6.0, 0.0, 5.0,
        3.0, 7.0, 8.0, 6.0, 6.0, 6.0, 7.0, 2.0, 0.0, 1.0, 4.0, 8.0, 5.0, 0.0, 5.0, 1.0, 6.0, 6.0,
        9.0, 3.0, 3.0, 3.0, 5.0, 2.0, 3.0, 2.0, 4.0, 8.0, 8.0, 8.0, 9.0, 6.0, 6.0, 6.0, 6.0, 6.0,
        3.0, 8.0, 8.0, 5.0, 2.0, 9.0, 9.0, 8.0, 9.0, 2.0, 0.0, 0.0, 6.0, 8.0, 8.0, 2.0, 2.0, 2.0,
        1.0, 1.0, 1.0, 5.0, 4.0, 4.0, 4.0, 0.0, 4.0, 0.0, 2.0, 0.0, 7.0, 6.0, 3.0, 1.0, 8.0, 8.0,
        3.0, 1.0, 3.0, 6.0, 7.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 7.0, 1.0, 1.0, 1.0, 4.0, 5.0, 2.0,
        7.0, 7.0, 7.0, 2.0, 5.0, 5.0, 7.0, 8.0, 8.0, 6.0, 5.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        3.0, 8.0, 7.0, 9.0, 9.0, 9.0, 7.0, 7.0, 7.0, 3.0, 0.0, 6.0, 9.0, 4.0, 7.0, 0.0, 1.0, 0.0,
        1.0, 3.0, 1.0, 5.0, 1.0, 3.0, 1.0, 6.0, 6.0, 9.0, 6.0, 3.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0,
        1.0, 6.0, 6.0, 1.0, 2.0, 3.0, 9.0, 4.0, 4.0, 2.0, 2.0, 2.0, 9.0, 5.0, 5.0, 9.0, 8.0, 0.0,
        1.0, 9.0, 1.0, 8.0, 5.0, 9.0, 3.0, 3.0, 3.0, 5.0, 9.0, 2.0, 5.0, 8.0, 5.0, 7.0, 3.0, 3.0,
        1.0, 3.0, 3.0, 0.0, 7.0, 9.0, 0.0, 0.0, 0.0, 2.0, 5.0, 0.0, 7.0, 9.0, 1.0, 4.0, 5.0, 5.0,
        8.0, 1.0, 1.0, 9.0, 8.0, 9.0, 5.0, 5.0, 5.0, 3.0, 1.0, 3.0, 1.0, 3.0, 8.0, 2.0, 4.0, 6.0,
        5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 8.0, 4.0, 4.0, 6.0, 6.0, 6.0, 7.0, 3.0, 9.0,
    ];
    let inp_size = 3;
    let inps_row = inps.len() / inp_size;
    DenseMatrix::new(inps_row, inp_size, &inps)
}

fn get_training_target_matrix() -> DenseMatrix {
    let tar = vec![
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let targ_size = 3;
    let targ_row = tar.len() / targ_size;
    DenseMatrix::new(targ_row, targ_size, &tar)
}

fn get_validation_input_matrix() -> DenseMatrix {
    let inps = vec![
        1.0, 6.0, 5.0, 0.0, 8.0, 9.0, 7.0, 6.0, 6.0, 0.0, 9.0, 9.0, 0.0, 7.0, 0.0, 8.0, 3.0, 8.0,
        4.0, 9.0, 5.0, 7.0, 5.0, 6.0, 6.0, 5.0, 5.0, 0.0, 9.0, 8.0, 3.0, 0.0, 3.0, 2.0, 5.0, 7.0,
        1.0, 1.0, 1.0, 6.0, 2.0, 0.0, 7.0, 5.0, 5.0, 0.0, 2.0, 1.0, 4.0, 9.0, 3.0, 7.0, 9.0, 7.0,
        0.0, 0.0, 0.0, 3.0, 5.0, 7.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0,
        4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 7.0, 7.0, 7.0,
    ];

    let inp_size = 3;
    let inps_row = inps.len() / inp_size;
    DenseMatrix::new(inps_row, inp_size, &inps)
}

fn get_validation_target_matrix() -> DenseMatrix {
    let tar = vec![
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];

    let targ_size = 3;
    let targ_row = tar.len() / targ_size;
    DenseMatrix::new(targ_row, targ_size, &tar)
}

fn test_search() {
    let training_inputs = get_training_input_matrix();
    let training_targets = get_training_target_matrix();

    let validation_inputs = get_validation_input_matrix();
    let validation_targets = get_validation_target_matrix();

    let nw = generate_network(training_inputs.cols(), training_targets.cols());

    let sp = SearchConfigsBuilder::new()
        .learning_rates(
            RangeParameters::new()
                .lower_limit(0.0025)
                .upper_limit(0.0045)
                .increment(0.0005)
                .float_parameters(),
        )
        .batch_sizes(
            RangeParameters::new()
                .lower_limit(5.0)
                .upper_limit(10.0)
                .increment(1.0)
                .int_parameters(),
        )
        .hidden_layer(
            RangeParameters::new()
                .lower_limit(12.0)
                .upper_limit(24.0)
                .increment(4.0)
                .int_parameters(),
            ReLU::new(),
        )
        .export("search".to_string())
        .build();

    let search_res = search(
        nw,
        sp,
        4,
        training_inputs,
        training_targets,
        validation_inputs,
        validation_targets,
    );

    info!("Num Results: {}", search_res.len());
}

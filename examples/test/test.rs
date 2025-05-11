use runn::{
    adam::Adam,
    cross_entropy::CrossEntropy,
    csv::CSV,
    dense_layer::Dense,
    dropout::Dropout,
    elu::ELU,
    exponential::Exponential,
    flexible::{Flexible, MonitorMetric},
    l2::L2,
    matrix::DenseMatrix,
    min_max::MinMax,
    network::network_model::NetworkBuilder,
    network_io::JSON,
    network_model::Network,
    network_search::NetworkSearchBuilder,
    relu::ReLU,
    softmax::Softmax,
    swish::Swish,
    tensor_board::TensorBoard,
};

fn main() {
    let mut network = NetworkBuilder::new(2, 1)
        .layer(Dense::default().size(4).activation(ReLU::build()).build())
        .layer(Dense::default().size(1).activation(Softmax::build()).build())
        .loss_function(CrossEntropy::default().build()) // Cross-entropy loss function with default values
        .optimizer(Adam::default().build()) // Adam optimizer with default values
        .batch_size(2) // Number of batches
        .seed(42) // Optional seed for reproducibility
        .epochs(5)
        .build()
        .unwrap(); // Handle error in production use

    let inputs = DenseMatrix::new(4, 2)
        .data(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .build()
        .unwrap();
    let targets = DenseMatrix::new(4, 1).data(&[0.0, 1.0, 1.0, 0.0]).build().unwrap();

    let result = network.train(&inputs, &targets);

    match result {
        Ok(result) => println!("Training completed successfully.\nResults: {}", result.display_metrics()),
        Err(e) => eprintln!("Training failed: {}", e),
    }

    network
        .save(JSON::default().file_name("model").build().unwrap())
        .unwrap();
    let _loaded_network = Network::load(JSON::default().file_name("model").build().unwrap()).unwrap();
    //let results = loaded_network.predict(&validation_inputs, &validation_targets).unwrap();

    let _network_search = NetworkSearchBuilder::new()
        .network(network)
        .parallelize(4)
        .learning_rates(vec![0.0025, 0.0035])
        .batch_sizes(vec![1, 2, 4, 7])
        .hidden_layer(vec![1, 3, 4, 7], ReLU::build())
        .hidden_layer(vec![1, 3, 7, 9], ReLU::build())
        .export(CSV::default().file_name("hp_search").build())
        .build();

    //let ns=     network_search.unwrap().search(training_inputs, training_targets, validation_inputs, validation_targets);

    let _network = NetworkBuilder::new(5, 3)
        .layer(
            Dense::default()
                .size(12)
                .activation(ELU::default().alpha(0.9).build())
                .build(),
        )
        .layer(
            Dense::default()
                .size(24)
                .activation(Swish::default().beta(1.0).build())
                .build(),
        )
        .layer(Dense::default().size(3).activation(Softmax::build()).build())
        .loss_function(CrossEntropy::default().epsilon(0.99).build()) // loss function with epsilon
        .optimizer(
            Adam::default() // Adam optimizer with custom parameters
                .beta1(0.98)
                .beta2(0.990)
                .learning_rate(0.0035)
                .scheduler(Exponential::default().decay_factor(0.2).build()) // scheduler for learning rate
                .build(),
        )
        .seed(42) // seed for reproducibility
        .early_stopper(
            Flexible::default()
                .monitor_metric(MonitorMetric::Loss) // early stopping based on loss
                .patience(500) // number of epochs with no improvement after which training will be stopped
                .min_delta(0.1) // minimum change to be considered an improvement
                .smoothing_factor(0.5) // factor to smooth the loss
                .build(),
        )
        .regularization(L2::default().lambda(0.01).build()) // L2 regularization
        .regularization(Dropout::default().dropout_rate(0.2).seed(42).build()) // Dropout regularization
        .epochs(5000)
        .batch_size(4)
        .batch_group_size(4) // number of batches to process in groups
        .parallelize(4) // number of threads to use for parallel process the batch groups
        .summary(TensorBoard::default().directory("summary").build()) // tensorboard summary
        .normalize_input(MinMax::default()) // normalization of the input data
        .build()
        .unwrap();
}

#[cfg(test)]
mod relu_tests {

    use super::*;

    #[test]
    fn test_network_search() {
        let net = NetworkBuilder::new(2, 1)
            .layer(Dense::default().size(4).activation(ReLU::build()).build())
            .layer(Dense::default().size(1).activation(Softmax::build()).build())
            .loss_function(CrossEntropy::default().build()) // Cross-entropy loss function with default values
            .optimizer(Adam::default().build()) // Adam optimizer with default values
            .batch_size(2) // Number of batches
            .seed(42) // Optional seed for reproducibility
            .epochs(5)
            .build()
            .unwrap();

        let mut ns = NetworkSearchBuilder::new()
            .network(net)
            .parallelize(4)
            .learning_rates(vec![0.0025, 0.0035])
            .batch_sizes(vec![1, 2, 4, 7])
            .hidden_layer(vec![1, 3, 4, 7], ReLU::build())
            .hidden_layer(vec![1, 3, 7, 9], ReLU::build())
            .export(CSV::default().file_name("hp_search").build())
            .build()
            .unwrap();

        let inputs = DenseMatrix::new(4, 2)
            .data(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .build()
            .unwrap();
        let targets = DenseMatrix::new(4, 1).data(&[0.0, 1.0, 1.0, 0.0]).build().unwrap();

        let res = ns.search(&inputs, &targets, &inputs, &targets);
        assert!(res.is_ok())
    }

    #[test]
    fn test_network_train() {
        let mut network = NetworkBuilder::new(2, 1)
            .layer(
                Dense::default()
                    .size(12)
                    .activation(ELU::default().alpha(0.9).build())
                    .build(),
            )
            .layer(
                Dense::default()
                    .size(24)
                    .activation(Swish::default().beta(1.0).build())
                    .build(),
            )
            .layer(Dense::default().size(1).activation(Softmax::build()).build())
            .loss_function(CrossEntropy::default().epsilon(0.99).build()) // loss function with epsilon
            .optimizer(
                Adam::default() // Adam optimizer with custom parameters
                    .beta1(0.98)
                    .beta2(0.990)
                    .learning_rate(0.0035)
                    .scheduler(Exponential::default().decay_factor(0.2).build()) // scheduler for learning rate
                    .build(),
            )
            .seed(42) // seed for reproducibility
            .early_stopper(
                Flexible::default()
                    .monitor_metric(MonitorMetric::Loss) // early stopping based on loss
                    .patience(500) // number of epochs with no improvement after which training will be stopped
                    .min_delta(0.1) // minimum change to be considered an improvement
                    .smoothing_factor(0.5) // factor to smooth the loss
                    .build(),
            )
            .regularization(L2::default().lambda(0.01).build()) // L2 regularization
            .regularization(Dropout::default().dropout_rate(0.2).seed(42).build()) // Dropout regularization
            .epochs(5000)
            .batch_size(4)
            .batch_group_size(4) // number of batches to process in groups
            .parallelize(4) // number of threads to use for parallel process the batch groups
            .summary(TensorBoard::default().directory("summary").build()) // tensorboard summary
            .normalize_input(MinMax::default()) // normalization of the input data
            .build()
            .unwrap();

        let inputs = DenseMatrix::new(4, 2)
            .data(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .build()
            .unwrap();
        let targets = DenseMatrix::new(4, 1).data(&[0.0, 1.0, 1.0, 0.0]).build().unwrap();

        let res = network.train(&inputs, &targets);
        assert!(res.is_ok())
    }
}

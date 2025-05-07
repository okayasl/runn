use runn::{
    adam::Adam,
    cross_entropy::CrossEntropy,
    dense_layer::Dense,
    dropout::Dropout,
    elu::ELU,
    exponential::Exponential,
    flexible::{Flexible, MonitorMetric},
    l2::L2,
    matrix::DenseMatrix,
    min_max::MinMax,
    network::network::NetworkBuilder,
    network_search::NetworkSearchBuilder,
    relu::ReLU,
    softmax::Softmax,
    swish::Swish,
    tensor_board::TensorBoard,
};

fn main() {
    let mut network = NetworkBuilder::new(2, 1)
        .layer(Dense::new().size(4).activation(ReLU::new()).build())
        .layer(Dense::new().size(1).activation(Softmax::new()).build())
        .loss_function(CrossEntropy::new().build())
        .optimizer(Adam::new().build())
        .seed(42)
        .epochs(5)
        .batch_size(2)
        .build()
        .unwrap();

    let inputs = DenseMatrix::new(4, 2)
        .data(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .build()
        .unwrap();
    let targets = DenseMatrix::new(4, 1).data(&[0.0, 1.0, 1.0, 0.0]).build().unwrap();

    let result = network.train(&inputs, &targets);

    match result {
        Ok(_) => println!("Training completed successfully.\nResults: {}", result.unwrap().display_metrics()),
        Err(e) => eprintln!("Training failed: {}", e),
    }

    // network.save("model.json", SerializationFormat::Json);
    // let mut loaded_network = Network::load("model.json", SerializationFormat::Json);
    // let results = loaded_network.predict(&validation_inputs, &validation_targets);

    let network_search = NetworkSearchBuilder::new()
        .network(network)
        .parallelize(4)
        .learning_rates(vec![0.0025, 0.0035])
        .batch_sizes(vec![1, 2, 4, 7])
        .hidden_layer(vec![1, 3, 4, 7], ReLU::new())
        .hidden_layer(vec![1, 3, 7, 9], ReLU::new())
        .export("hp_search".to_string())
        .build();

    // let ns=     network_search.unwrap().search(training_inputs, training_targets, validation_inputs, validation_targets);

    let network = NetworkBuilder::new(5, 3)
        .layer(Dense::new().size(12).activation(ELU::new().alpha(0.9).build()).build())
        .layer(Dense::new().size(24).activation(Swish::new().beta(1.0).build()).build())
        .layer(Dense::new().size(3).activation(Softmax::new()).build())
        .loss_function(CrossEntropy::new().epsilon(0.99).build()) // loss function with epsilon
        .optimizer(
            Adam::new() // Adam optimizer with custom parameters
                .beta1(0.98)
                .beta2(0.990)
                .learning_rate(0.0035)
                .scheduler(Exponential::new().decay_factor(0.2).build()) // scheduler for learning rate
                .build(),
        )
        .seed(42) // seed for reproducibility
        .early_stopper(
            Flexible::new()
                .monitor_metric(MonitorMetric::Loss) // early stopping based on loss
                .patience(500) // number of epochs with no improvement after which training will be stopped
                .min_delta(0.1) // minimum change to be considered an improvement
                .smoothing_factor(0.5) // factor to smooth the loss
                .build(),
        )
        .regularization(L2::new().lambda(0.01).build()) // L2 regularization
        .regularization(Dropout::new().dropout_rate(0.2).seed(42).build()) // Dropout regularization
        .epochs(5000)
        .batch_size(4)
        .batch_group_size(4) // number of batches to process in groups
        .parallelize(4) // number of threads to use for parallel process the batch groups
        .summary(TensorBoard::new().logdir("summary").build()) // tensorboard summary
        .normalize_input(MinMax::new()) // normalization of the input data
        .build()
        .unwrap();
}

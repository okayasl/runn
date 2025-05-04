use runn::{
    adam::Adam,
    cross_entropy::CrossEntropy,
    matrix::DenseMatrix,
    network::network::{Network, NetworkBuilder},
    network_io::SerializationFormat,
    network_search::NetworkSearchBuilder,
    relu::ReLU,
    softmax::Softmax,
    Dense,
};
use serde::ser;

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

    let inputs = DenseMatrix::new(4, 2, &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let targets = DenseMatrix::new(4, 1, &[0.0, 1.0, 1.0, 0.0]);

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
}

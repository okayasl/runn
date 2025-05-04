<!-- Badges -->
[![Crates.io](https://img.shields.io/crates/v/runn)](https://crates.io/crates/runn)
[![docs.rs](https://img.shields.io/docsrs/runn)](https://docs.rs/runn)
[![Build Status](https://img.shields.io/github/actions/workflow/status/okayasl/runn/ci.yml?branch=main)](https://github.com/okayasl/runn/actions)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](LICENSE-MIT)

# 📦 RUNN

**A Compact Rust Neural Network Library**

`runn` is a feature-rich, easy to use library for building, training, and evaluating feed forward neural networks in Rust. It supports a wide range of activation functions, optimizers, regularization techniques, fine grained parallelization, hyperparameter search and more, with a user friendly api.

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Features](#-features)
- [Examples](#-examples)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors & Acknowledgments](#-authors--acknowledgments)

---

## 💾 Installation

Add `runn` to your project:

```bash
cargo add runn
```


Or in Cargo.toml:

```bash
[dependencies]
runn = "0.1"
```

⚙️ Quickstart

Here’s how to build and train a simple neural network:

```rust

use runn::{
    adam::Adam, cross_entropy::CrossEntropy, matrix::DenseMatrix, network::network::NetworkBuilder, relu::ReLU,
    softmax::Softmax, Dense,
};

fn main() {
    let mut network = NetworkBuilder::new(2, 1)
        .layer(Dense::new().size(4).activation(ReLU::new()).build())
        .layer(Dense::new().size(1).activation(Softmax::new()).build())
        .loss_function(CrossEntropy::new().build())
        .optimizer(Adam::new().build())
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
}
```
and you would see something like that:

```bash
Training completed successfully.
Results: Loss:0.0000, Classification Metrics: Accuracy:100.0000, Micro Precision:1.0000, Micro Recall:1.0000, Macro F1 Score:1.0000, Micro F1 Score:1.0000
  Metrics by Class:
    Class 0:    Precision:1.0000    Recall:1.0000    F1 Score:1.0000
```

You can save & load networks like

```rust
    network.save("model.json", SerializationFormat::Json);
    let mut loaded_network = Network::load("model.json", SerializationFormat::Json);
    let results = loaded_network.predict(&validation_inputs, &validation_targets);
```

You can simply search for parameters(learning rate, batch size, layer size) like:

```rust
    let network_search = NetworkSearchBuilder::new()
        .network(network)
        .parallelize(4) // run for 4 thread
        .learning_rates(vec![0.0025,0.0035]) 
        .batch_sizes(vec![1,2,4,7])
        .hidden_layer(vec![1,3,4,7],ReLU::new())
        .hidden_layer(vec![1,3,7,9],ReLU::new())
        .export("hp_search".to_string()) // export results
        .build();

    // search takes validation inputs outpus as well to make predictions as well   
    let ns = network_search.unwrap().search(training_inputs, training_targets, validation_inputs, validation_targets);
```

## ✨ Features
| Feature	|Built in Support|
| ------------- | ------------- |
Activations |	ELU, GeLU, ReLU, LeakyReLU, Linear, Sigmoid, Softmax, Softplus, Swish, Tanh
Optimizers |	SGD, Momentum, RMSProp, Adam, AdamW, AMSGrad
Loss Functions|	Cross-Entropy, Mean Squared Error
Regularization|	L1, L2, Dropout
Schedulers|	Exponential, Step
Early Stopping|	Loss
Save&load network | JSON & MessagePack
Logging Summary | TensorBoard
Hyperparameter Search | Layer size, batch size, learning rate
Normalization | Minmax, Zscore


## 📂 Examples

With `runn`, you can write fairly complex networks according to your needs:

```rust

    let mut network = NetworkBuilder::new(5, 3)
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
            Loss::new() // early stopping based on loss
                .patience(500) // number of epochs with no improvement after which training will be stopped
                .min_delta(0.1) // minimum change to be considered an improvement
                .smoothing_factor(0.5) // factor to smooth the loss
                .build(),
        )
        .regularization(L2::new().lambda(0.01).build()) // L2 regularization
        .regularization(Dropout::new().dropout_rate(0.2).seed(42).build()) // Dropout regularization with seed for reproducibility
        .epochs(5000) // number of epoch to run
        .batch_size(4) // number of batches 
        .batch_group_size(4) // number of batches to process in groups
        .parallelize(4) // number of threads to use for parallel process the batch groups
        .summary(TensorBoard::new().logdir("summary").build()) // tensorboard summary
        .normalize_input(MinMax::new()) // normalization of the input data
        .build() // Build network
        .unwrap();

```

Bonus:
`runn` provides some handy utility methods:

| method	|description| usage|
| ------------- | ------------- | ------------- |
helper::one_hot_encode |	Provides one hot encoding for target.  |   ```let training_targets = helper::one_hot_encode(&training_targets);```
helper::stratified_split |	Provides stratified split for matrix inputs and single-column targets  |   ```let (training_inputs, training_targets, validation_inputs, validation_targets) = helper::stratified_split(&all_inputs, &all_targets, 0.2, 11);```
helper::random_split |	Provides random split for matrix inputs and multi-columbn targets  |   ```let (training_inputs, training_targets, validation_inputs, validation_targets) = helper::random(&all_inputs, &all_targets, 0.2, 11);```
helper::pretty_compare_matrices |	 Pretty prints three matrices side-by-side as an ASCII-art comparison  |   ```helper::pretty_compare_matrices(&training_inputs, &training_targets, &predictions, helper::CompareMode::Regression)```



See the examples/ directory for end‑to‑end demos:

    - triplets: Metric learning with triplet loss
    - iris: Iris classification
    - wine: Wine quality regression
    - energy_efficiency: Energy efficiency regression

Run an example:

```bash
cargo run --example iris
```

## 📖 Documentation

Full API docs on docs.rs/runn.

For local docs
```bash
cargo doc --open
```

## 🤝 Contributing

We welcome all contributions! Please see CONTRIBUTING.md for:

    Issue templates and PR guidelines

    Code style (rustfmt, Clippy) and testing (cargo test)

    Code of Conduct

## 📜 License

Dual‑licensed under MIT OR Apache‑2.0. See LICENSE-MIT and LICENSE-APACHE.    

## 👤 Authors & Acknowledgments

    Okay Aslan (okayasl) – Creator & maintainer

    Thanks to the Rust community for nalgebra, serde, tensorboard-rs, and other tooling.
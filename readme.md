<!-- Badges -->
[![Crates.io](https://img.shields.io/crates/v/runn)](https://crates.io/crates/runn)
[![docs.rs](https://img.shields.io/docsrs/runn)](https://docs.rs/runn)
[![Build Status](https://img.shields.io/github/actions/workflow/status/okayasl/runn/ci.yml?branch=main)](https://github.com/okayasl/runn/actions)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](LICENSE-MIT)

# 📦 RUNN

**A Compact Rust Neural Network Library**

`runn` is a feature-rich, easy to use library for building, training, and evaluating feed forward neural networks in Rust. It supports a wide range of activation functions, optimizers, regularization techniques, hyperparameter search and more, with a user friendly api.

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
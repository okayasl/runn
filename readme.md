<!-- Badges -->
[![Crates.io](https://img.shields.io/crates/v/runn)](https://crates.io/crates/runn)
[![docs.rs](https://img.shields.io/docsrs/runn)](https://docs.rs/runn)
[![Build Status](https://img.shields.io/github/actions/workflow/status/okayasl/runn/ci.yml?branch=main)](https://github.com/okayasl/runn/actions)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](LICENSE-MIT)

# 📦 runn

**A Pure-Rust Neural Network Library**  
`runn` is a feature-rich library for building, training, and evaluating neural networks in Rust. It supports a wide range of activation functions, optimizers, regularization techniques, and more.

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Features](#-features)
- [Examples](#-examples)
- [Usage](#-usage)
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
    adam::Adam,
    cross_entropy::CrossEntropy,
    helper,
    network::network::{Network, NetworkBuilder},
    network_search::NetworkSearchBuilder,
    numbers::{Numbers, SequentialNumbers},
    relu::ReLU,
    softmax::Softmax,
    Dense,
};

    let network = NetworkBuilder::new(3, 3)
        .layer(Dense::new().size(12).activation(ReLU::new()).build())
        .layer(Dense::new().size(8).activation(ReLU::new()).build())
        .layer(Dense::new().size(3).activation(Softmax::new()).build())
        .loss_function(CrossEntropy::new().epsilon(1e-8).build())
        .optimizer(Adam::new().beta1(0.99).beta2(0.999).learning_rate(0.0035).build())
        .batch_size(8)
        .epochs(150)
        .seed(55)
        .build()
        .unwrap();

    let train_result = network.train(&training_inputs, &training_targets);
```


✨ Features
|Category	|Options|
| ------------- | ------------- |
Activations |	ELU, GeLU, ReLU, LeakyReLU, Linear, Sigmoid, Softmax, Softplus, Swish, Tanh
Optimizers |	SGD, Momentum, RMSProp, Adam, AdamW, AMSGrad
Loss Functions|	Cross-Entropy, Mean Squared Error
Regularization|	L1, L2, Dropout
Schedulers|	Exponential, Step
Early Stopping|	Loss
Save&load network | JSON & MessagePack
Logging Summary | TensorBoard
Hyperparameter Search | Grid search




📂 Examples

See the examples/ directory for end‑to‑end demos:

    - triplets: Metric learning with triplet loss
    - iris: Iris classification
    - wine: Wine quality regression
    - energy_efficiency: Energy efficiency regression

Run an example:

```bash
cargo run --example iris
```

📖 Documentation

Full API docs on docs.rs/runn. Run cargo doc --open for local docs.


🤝 Contributing

We welcome all contributions! Please see CONTRIBUTING.md for:

    Issue templates and PR guidelines

    Code style (rustfmt, Clippy) and testing (cargo test)

    Code of Conduct

📜 License

Dual‑licensed under MIT OR Apache‑2.0. See LICENSE-MIT and LICENSE-APACHE.    

👤 Authors & Acknowledgments

    Okay Aslan (okayasl) – Creator & maintainer

    Thanks to the Rust community for nalgebra, serde, tensorboard-rs, and other tooling.
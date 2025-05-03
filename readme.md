<!-- Badges -->
[![Crates.io](https://img.shields.io/crates/v/runn)](https://crates.io/crates/runn)
[![docs.rs](https://img.shields.io/docsrs/runn)](https://docs.rs/runn)
[![Build Status](https://img.shields.io/github/actions/workflow/status/okayasl/runn/ci.yml?branch=main)](https://github.com/okayasl/runn/actions)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](LICENSE-MIT)
[![Lines of Code](https://img.shields.io/tokei/lines/github/okayasl/runn)]()

# 📦 runn

**Pure‑Rust neural‑network library** backed by `nalgebra`, featuring:

- 10 activation functions (ELU, GeLU, ReLU, LeakyReLU, Linear, Sigmoid, Softmax, Softplus, Swish, Tanh)  
- Optimizers: SGD, Momentum, RMSProp, Adam, AdamW, AMSGrad  
- Regularization: L1, L2, Dropout; LR schedulers: exponential, step  
- Losses: Cross‑Entropy, Mean Squared Error  
- Early stopping (target loss, patience, smoothing)  
- TensorBoard metrics via `tensorboard-rs`  
- Hyperparameter grid search (batch size, layer sizes, learning rate)  
- Save/load model (JSON, MessagePack)  

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

Add **runn** to your project:

```bash
cargo add runn

Or in Cargo.toml:

[dependencies]
runn = "0.1"

⚙️ Quickstart

use runn::{NetworkBuilder, Dense, ActivationFunction, LossFunction, OptimizerConfig};

// Build & train a simple network
let network = NetworkBuilder::new(784, 10)
    .layer(Dense::new().from(128, ActivationFunction::ReLU).build())
    .layer(Dense::new().from(64,  ActivationFunction::ReLU).build())
    .layer(Dense::new().from(10,  ActivationFunction::Softmax).build())
    .loss_function(LossFunction::CrossEntropy)
    .optimizer(OptimizerConfig::Adam { learning_rate: 0.001, weight_decay: 0.0 })
    .epochs(20)
    .batch_size(32)
    .build()
    .unwrap();

network.train(&train_images, &train_labels);


✨ Features
Category	Options
Activations	ELU, GeLU, ReLU, LeakyReLU, Linear, Sigmoid, Softmax, …
Optimizers	SGD, Momentum, RMSProp, Adam, AdamW, AMSGrad
Loss Functions	Cross‑Entropy, MSE
Regularization	L1, L2, Dropout
Schedulers	Exponential, Step
EarlyStopping Loss
I/O	JSON & MessagePack via serde/rmp-serde
Logging	TensorBoard via tensorboard-rs
HyperparameterSearch	Grid search over batch size, layer sizes, learning rate
📂 Examples

See the examples/ directory for end‑to‑end demos:

    triplets: Metric learning with triplet loss

    iris: Iris classification

    wine: Wine quality regression

    energy_efficiency: Energy efficiency regression

Run an example:

cargo run --example iris


🛠️ Usage

    Load data (e.g. via the csv crate).

    Normalize with built‑in Normalization traits.

    Configure network via NetworkBuilder.

    Train with network.train(...) and visualize in TensorBoard.

    Save model: network.save("model.json").

    Load & infer: Network::load("model.json").

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
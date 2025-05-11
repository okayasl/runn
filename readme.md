<!-- Badges -->

[![Crates.io](https://img.shields.io/crates/v/runn)](https://crates.io/crates/runn)
[![docs.rs](https://img.shields.io/docsrs/runn)](https://docs.rs/runn)
[![Build Status](https://img.shields.io/github/actions/workflow/status/okayasl/runn/ci.yml?branch=main)](https://github.com/okayasl/runn/actions)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](LICENSE-MIT)

# 📦 RUNN

**A Compact Rust Neural Network Library**

`runn` is a feature-rich, easy-to-use library for building, training, and evaluating feed-forward neural networks in Rust. It supports a wide range of activation functions, optimizers, regularization techniques, fine-grained parallelization, hyperparameter search, and more—all with a user-friendly API.

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

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
runn = "0.1"
```

---

## ⚡ Quickstart

`runn` adopts a fluent interface design pattern for ease of use. Most components are initialized with sensible defaults, which you can override as needed.

Here's how to build and train a simple neural network:

```rust
use runn::{
    adam::Adam, cross_entropy::CrossEntropy, dense_layer::Dense, matrix::DenseMatrix,
    network::network_model::NetworkBuilder, relu::ReLU, softmax::Softmax,
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
}
```

**Sample Output:**
```
Training completed successfully.
Results: Loss:0.0000, Classification Metrics: Accuracy:100.0000, Micro Precision:1.0000, Micro Recall:1.0000, Macro F1 Score:1.0000, Micro F1 Score:1.0000
  Metrics by Class:
    Class 0:    Precision:1.0000    Recall:1.0000    F1 Score:1.0000
```

---

## 💾 Save & Load

You can save and load your trained networks:

```rust
network.save(JSON::default().file_name("model").build().unwrap()).unwrap();
let mut loaded_network = Network::load(JSON::default().file_name("model").build().unwrap()).unwrap();
let results = loaded_network.predict(&validation_inputs, &validation_targets).unwrap();
```

---

## 🔍 Hyperparameter Search

Easily perform hyperparameter search for learning rate, batch size, and layer size:

```rust
let network_search = NetworkSearchBuilder::new()
    .network(network)
    .parallelize(4)
    .learning_rates(vec![0.0025, 0.0035])
    .batch_sizes(vec![1, 2, 4, 7])
    .hidden_layer(vec![1, 3, 4, 7], ReLU::build())
    .hidden_layer(vec![1, 3, 7, 9], ReLU::build())
    .export(CSV::default().file_name("hp_search").build())
    .build();

let ns = network_search.search(training_inputs, training_targets, validation_inputs, validation_targets);
```

Results are exported to a CSV file, including loss and training metrics.

---

## ✨ Features

| Feature               | Built-in Support                                                            |
|-----------------------|-----------------------------------------------------------------------------|
| **Activations**       | ELU, GeLU, ReLU, LeakyReLU, Linear, Sigmoid, Softmax, Softplus, Swish, Tanh |
| **Optimizers**        | SGD, Momentum, RMSProp, Adam, AdamW, AMSGrad                                |
| **Loss Functions**    | Cross-Entropy, Mean Squared Error                                           |
| **Regularization**    | L1, L2, Dropout                                                             |
| **Schedulers**        | Exponential, Step                                                           |
| **Early Stopping**    | Loss, Accuracy, R2                                                          |
| **Save & Load**       | JSON & MessagePack                                                          |
| **Logging Summary**   | TensorBoard                                                                 |
| **Hyperparameter Search** | Layer size, batch size, learning rate                                   |
| **Normalization**     | MinMax, Zscore                                                              |
| **Parallelization**   | Forward & Backward runs for batch groups                                    |

---

## 📂 Examples

With `runn`, you can build complex networks tailored to your needs:

```rust
let network = NetworkBuilder::new(5, 3)
    .layer(Dense::default().size(12).activation(ELU::default().alpha(0.9).build()).build())
    .layer(Dense::default().size(24).activation(Swish::default().beta(1.0).build()).build())
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
```

---

### 🛠️ Utility Methods

`runn` provides handy utility functions:

- **`helper::one_hot_encode`**  
  Converts categorical labels into one-hot encoded vectors.
  ```rust
  let training_targets = helper::one_hot_encode(&training_targets);
  ```

- **`helper::stratified_split`**  
  Stratified split for matrix inputs and single-column targets.
  ```rust
  let (training_inputs, training_targets, validation_inputs, validation_targets) =
      helper::stratified_split(&all_inputs, &all_targets, 0.2, 11);
  ```

- **`helper::random_split`**  
  Random split for matrix inputs and multi-column targets.
  ```rust
  let (training_inputs, training_targets, validation_inputs, validation_targets) =
      helper::random_split(&all_inputs, &all_targets, 0.2, 11);
  ```

- **`helper::pretty_compare_matrices`**  
  Pretty-prints three matrices side-by-side as an ASCII-art comparison.
  ```rust
  helper::pretty_compare_matrices(&training_inputs, &training_targets, &predictions, helper::CompareMode::Regression)
  ```

---

### 📦 Example Projects

| Example | Description | Train | Hyperparameter Search |
|---------|-------------|-------|----------------------|
| [triplets](examples/triplets) | Multi-class classification | `cargo run --example triplets` | `cargo run --example triplets -- -search` |
| [iris](examples/iris) | Multi-class classification | `cargo run --example iris` | `cargo run --example iris -- -search` |
| [wine](examples/wine) | Multi-class classification | `cargo run --example wine` | `cargo run --example wine -- -search` |
| [energy efficiency](examples/energy_efficiency) | Regression | `cargo run --example energy_efficiency` | `cargo run --example energy_efficiency -- -search` |

---

## 📖 Documentation

- [Full API docs on docs.rs/runn](https://docs.rs/runn)
- Generate local docs:
  ```bash
  cargo doc --open
  ```

---

## 🤝 Contributing

We welcome all contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Issue templates and PR guidelines
- Code style (rustfmt, Clippy) and testing (`cargo test`)

---

## 📜 License

Dual-licensed under MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

---

## 👤 Authors & Acknowledgments

**Okay Aslan (okayasl)** – Creator & Maintainer

---

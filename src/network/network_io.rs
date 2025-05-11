use rmp_serde::{decode, encode};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};

use crate::error::NetworkError;
use crate::layer::Layer;
use crate::{EarlyStopper, LossFunction, Normalization, OptimizerConfig, Regularization};

pub trait NetworkIO {
    fn save(&self, network: NetworkSerialized) -> Result<(), NetworkError>;
    fn load(&self) -> Result<NetworkSerialized, NetworkError>;
}

#[derive(Clone)]
struct JSONNetworkIO {
    filename: String,
    directory: String,
}

impl NetworkIO for JSONNetworkIO {
    fn save(&self, network_s: NetworkSerialized) -> Result<(), NetworkError> {
        let serialized_data = serde_json::to_vec(&network_s);
        match serialized_data {
            Ok(data) => save(self.filename.clone(), self.directory.clone(), data)?,
            Err(_) => return Err(NetworkError::IoError("Failed to serialize to JSON".to_string())),
        };
        Ok(())
    }

    fn load(&self) -> Result<NetworkSerialized, NetworkError> {
        let serialized_data = load(self.filename.clone(), self.directory.clone())?;
        let network_s = serde_json::from_slice(&serialized_data);
        if network_s.is_err() {
            return Err(NetworkError::IoError("Failed to deserialize from JSON".to_string()));
        }
        Ok(network_s.unwrap())
    }
}

/// A builder for configuring and creating a JSON network I/O exporter.
///
/// This struct provides a fluent interface to customize the file name
/// and output directory for a JSON-encoded network representation.
/// Use the `build` method to validate the configuration and return
/// a concrete implementation of `NetworkIO`.
pub struct JSON {
    pub file_name: String,
    pub directory: String,
}

impl JSON {
    // Creates a new JSON builder.
    // Default values:
    // - File name: `"network"`
    // - Directory: `"."` (current directory)
    fn new() -> Self {
        JSON {
            file_name: "network".to_string(),
            directory: ".".to_string(),
        }
    }

    /// Sets the base name for the output JSON file.
    pub fn file_name(mut self, filename: &str) -> Self {
        self.file_name = filename.to_string();
        self
    }

    /// Sets the output directory for the JSON file.
    pub fn directory(mut self, directory: &str) -> Self {
        self.directory = directory.to_string();
        self
    }

    // Validates the configuration and checks if the output directory is writable.
    //
    // # Errors
    // Returns a `NetworkError` if any field is invalid or the file system check fails.
    fn validate(&self) -> Result<(), NetworkError> {
        if self.file_name.is_empty() {
            return Err(NetworkError::ConfigError("Filename cannot be empty".to_string()));
        }
        if self.directory.is_empty() {
            return Err(NetworkError::ConfigError("Directory cannot be empty".to_string()));
        }

        if !std::path::Path::new(&self.directory).exists() {
            fs::create_dir_all(&self.directory).map_err(|e| {
                NetworkError::IoError(format!("Failed to create output directory '{}': {}", self.directory, e))
            })?;
        }

        Ok(())
    }

    /// Finalizes the builder and constructs a `JSONNetworkIO` if the configuration is valid.
    pub fn build(self) -> Result<impl NetworkIO, NetworkError> {
        self.validate()?;
        Ok(JSONNetworkIO {
            filename: self.file_name,
            directory: self.directory,
        })
    }
}

impl Default for JSON {
    /// Creates a new JSON builder with default values.
    /// Default values:
    /// - File name: `"network"`
    /// - Directory: `"."` (current directory)
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct MessagePackNetworkIO {
    filename: String,
    directory: String,
}

impl NetworkIO for MessagePackNetworkIO {
    fn save(&self, network_s: NetworkSerialized) -> Result<(), NetworkError> {
        let serialized_data = encode::to_vec(&network_s);
        match serialized_data {
            Ok(data) => save(self.filename.clone(), self.directory.clone(), data)?,
            Err(_) => return Err(NetworkError::IoError("Failed to serialize to MessagePack".to_string())),
        };
        Ok(())
    }

    fn load(&self) -> Result<NetworkSerialized, NetworkError> {
        let serialized_data = load(self.filename.clone(), self.directory.clone())?;
        let network_s = decode::from_slice(&serialized_data);
        if network_s.is_err() {
            return Err(NetworkError::IoError("Failed to deserialize from MessagePack".to_string()));
        }
        Ok(network_s.unwrap())
    }
}
/// A builder for configuring and creating a MessagePack network I/O exporter.
///
/// This struct provides a fluent interface to customize the file name
/// and output directory for a MessagePack-encoded network representation.
/// Use the `build` method to validate the configuration and return
/// a concrete implementation of `NetworkIO`.
pub struct MessagePack {
    pub file_name: String,
    pub directory: String,
}

impl MessagePack {
    // Creates a new MessagePack builder.
    // Default values:
    // - File name: `"network"`
    // - Directory: `"."` (current directory)
    fn new() -> Self {
        MessagePack {
            file_name: "network".to_string(),
            directory: ".".to_string(),
        }
    }

    /// Sets the base name for the output MessagePack file.
    pub fn file_name(mut self, filename: &str) -> Self {
        self.file_name = filename.to_string();
        self
    }

    /// Sets the output directory for the MessagePack file.
    pub fn directory(mut self, directory: &str) -> Self {
        self.directory = directory.to_string();
        self
    }

    // Validates the configuration and checks if the output directory is writable.
    //
    // # Errors
    // Returns a `NetworkError` if any field is invalid or the file system check fails.
    fn validate(&self) -> Result<(), NetworkError> {
        if self.file_name.is_empty() {
            return Err(NetworkError::ConfigError("Filename cannot be empty".to_string()));
        }
        if self.directory.is_empty() {
            return Err(NetworkError::ConfigError("Directory cannot be empty".to_string()));
        }

        if !std::path::Path::new(&self.directory).exists() {
            fs::create_dir_all(&self.directory).map_err(|e| {
                NetworkError::IoError(format!("Failed to create output directory '{}': {}", self.directory, e))
            })?;
        }

        Ok(())
    }

    /// Finalizes the builder and constructs a `MessagePackNetworkIO` if the configuration is valid.
    pub fn build(self) -> Result<impl NetworkIO, NetworkError> {
        self.validate()?;
        Ok(MessagePackNetworkIO {
            filename: self.file_name,
            directory: self.directory,
        })
    }
}

impl Default for MessagePack {
    /// Creates a new MessagePack builder.
    /// Default values:
    /// - File name: `"network"`
    /// - Directory: `"."` (current directory)
    fn default() -> Self {
        Self::new()
    }
}

fn save(file_name: String, directory: String, serialized_data: Vec<u8>) -> Result<(), NetworkError> {
    let file = File::create(format!("{}/{}.json", directory, file_name));
    if file.is_err() {
        return Err(NetworkError::IoError("Failed to create file".to_string()));
    }
    let res = file.unwrap().write_all(&serialized_data);
    if res.is_err() {
        return Err(NetworkError::IoError("Failed to write to file".to_string()));
    }
    Ok(())
}

fn load(file_name: String, directory: String) -> Result<Vec<u8>, NetworkError> {
    let file = File::open(format!("{}/{}.json", directory, file_name));
    if file.is_err() {
        return Err(NetworkError::IoError("Failed to open file".to_string()));
    }
    let mut buffer = Vec::new();
    let res = file.unwrap().read_to_end(&mut buffer);
    if res.is_err() {
        return Err(NetworkError::IoError("Failed to read file".to_string()));
    }
    Ok(buffer)
}

#[derive(Serialize, Deserialize)]
pub struct NetworkSerialized {
    pub(crate) input_size: usize,
    pub(crate) output_size: usize,
    pub(crate) layers: Vec<Box<dyn Layer>>,
    pub(crate) loss_function: Box<dyn LossFunction>,
    pub(crate) optimizer_config: Box<dyn OptimizerConfig>,
    pub(crate) regularizations: Vec<Box<dyn Regularization>>,
    pub(crate) batch_size: usize,
    pub(crate) batch_group_size: usize,
    pub(crate) epochs: usize,
    pub(crate) clip_threshold: f32,
    pub(crate) seed: u64,
    pub(crate) early_stopper: Option<Box<dyn EarlyStopper>>,
    pub(crate) debug: bool,
    pub(crate) normalize_input: Option<Box<dyn Normalization>>,
    pub(crate) normalize_output: Option<Box<dyn Normalization>>,
    pub(crate) summary_writer: Option<Box<dyn crate::summary::SummaryWriter>>,
    pub(crate) parallelize: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_layer::Dense;
    use crate::dropout::Dropout;
    use crate::mean_squared_error::MeanSquared;
    use crate::network::network_model::Network;
    use crate::network::network_model::NetworkBuilder;
    use crate::relu::ReLU;
    use crate::sgd::SGD;
    use crate::softmax::Softmax;

    #[test]
    fn test_json_io() {
        let json_io = JSON::new()
            .file_name("test_network")
            .directory("./test_dir_123")
            .build()
            .unwrap();

        let network = NetworkBuilder::new(4, 3)
            .layer(Dense::default().size(5).activation(ReLU::build()).build())
            .layer(Dense::default().size(3).activation(Softmax::build()).build())
            .loss_function(MeanSquared.build())
            .optimizer(SGD::default().learning_rate(0.01).build())
            .regularization(Dropout::default().dropout_rate(0.5).seed(42).build())
            .seed(42)
            .epochs(10)
            .batch_size(2)
            .build()
            .unwrap();

        let _res = network.save(json_io);
        let loaded_network = Network::load(
            JSON::new()
                .file_name("test_network")
                .directory("./test_dir_123")
                .build()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(loaded_network.input_size, 4);
        assert_eq!(loaded_network.output_size, 3);
        //remove file and directory
        let _res = fs::remove_dir_all("./test_dir_123");
        assert!(_res.is_ok());
    }

    #[test]
    fn test_message_pack_io() {
        let msgpack_io = MessagePack::new()
            .file_name("test_network")
            .directory("./test_dir_1234")
            .build()
            .unwrap();

        let network = NetworkBuilder::new(4, 3)
            .layer(Dense::default().size(5).activation(ReLU::build()).build())
            .layer(Dense::default().size(3).activation(Softmax::build()).build())
            .loss_function(MeanSquared.build())
            .optimizer(SGD::default().learning_rate(0.01).build())
            .regularization(Dropout::default().dropout_rate(0.5).seed(42).build())
            .seed(42)
            .epochs(10)
            .batch_size(2)
            .build()
            .unwrap();

        let _res = network.save(msgpack_io);
        let loaded_network = Network::load(
            MessagePack::new()
                .file_name("test_network")
                .directory("./test_dir_1234")
                .build()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(loaded_network.input_size, 4);
        assert_eq!(loaded_network.output_size, 3);
        //remove file and directory
        let _res = fs::remove_dir_all("./test_dir_1234");
        assert!(_res.is_ok());
    }

    #[test]
    fn test_save_load_invalid_file() {
        let json_io = JSON::new()
            .file_name("invalid_network")
            .directory("./invalid_dir")
            .build()
            .unwrap();

        let result = json_io.load();
        assert!(result.is_err());
        if let Err(NetworkError::IoError(msg)) = result {
            assert_eq!(msg, "Failed to open file");
        } else {
            panic!("Expected ConfigError");
        }
        //remove file and directory
        let _res = fs::remove_dir_all("./invalid_dir");
    }

    #[test]
    fn test_save_load_invalid_directory() {
        let msgpack_io = MessagePack::new()
            .file_name("invalid_network")
            .directory("./invalid_dir")
            .build()
            .unwrap();

        let result = msgpack_io.load();
        assert!(result.is_err());
        if let Err(NetworkError::IoError(msg)) = result {
            assert_eq!(msg, "Failed to open file");
        } else {
            panic!("Expected ConfigError");
        }
        //remove file and directory
        let _res = fs::remove_dir_all("./invalid_dir");
        assert!(_res.is_ok());
    }
}

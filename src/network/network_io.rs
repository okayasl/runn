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

pub struct JSON {
    pub file_name: String,
    pub directory: String,
}

impl JSON {
    pub fn new() -> Self {
        JSON {
            file_name: "network".to_string(),
            directory: ".".to_string(),
        }
    }

    pub fn filename(mut self, filename: &str) -> Self {
        self.file_name = filename.to_string();
        self
    }

    pub fn directory(mut self, directory: &str) -> Self {
        self.directory = directory.to_string();
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.file_name.is_empty() {
            return Err(NetworkError::ConfigError("Filename cannot be empty".to_string()));
        }
        if self.directory.is_empty() {
            return Err(NetworkError::ConfigError("Directory cannot be empty".to_string()));
        }
        // Check if the directory exists, and attempt to create it if it doesn't
        if !std::path::Path::new(&self.directory).exists() {
            fs::create_dir_all(&self.directory).map_err(|e| {
                NetworkError::IoError(format!("Failed to create output directory '{}': {}", self.directory, e))
            })?;
        }

        Ok(())
    }

    pub fn build(self) -> Result<impl NetworkIO, NetworkError> {
        self.validate()?;
        Ok(JSONNetworkIO {
            filename: self.file_name,
            directory: self.directory,
        })
    }
}

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
pub struct MessagePack {
    pub file_name: String,
    pub directory: String,
}

impl MessagePack {
    pub fn new() -> Self {
        MessagePack {
            file_name: "network".to_string(),
            directory: ".".to_string(),
        }
    }

    pub fn filename(mut self, filename: &str) -> Self {
        self.file_name = filename.to_string();
        self
    }

    pub fn directory(mut self, directory: &str) -> Self {
        self.directory = directory.to_string();
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.file_name.is_empty() {
            return Err(NetworkError::ConfigError("Filename cannot be empty".to_string()));
        }
        if self.directory.is_empty() {
            return Err(NetworkError::ConfigError("Directory cannot be empty".to_string()));
        }
        // Check if the directory exists, and attempt to create it if it doesn't
        if !std::path::Path::new(&self.directory).exists() {
            fs::create_dir_all(&self.directory).map_err(|e| {
                NetworkError::IoError(format!("Failed to create output directory '{}': {}", self.directory, e))
            })?;
        }

        Ok(())
    }

    pub fn build(self) -> Result<impl NetworkIO, NetworkError> {
        self.validate()?;
        Ok(MessagePackNetworkIO {
            filename: self.file_name,
            directory: self.directory,
        })
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

// pub enum SerializationFormat {
//     Json,
//     MessagePack,
// }
// pub(crate) fn save_network(network_io: &NetworkSerialized, filename: &str, format: SerializationFormat) {
//     let serialized_data = match format {
//         SerializationFormat::Json => serde_json::to_vec(&network_io).expect("Failed to serialize to JSON"),
//         SerializationFormat::MessagePack => encode::to_vec(&network_io).expect("Failed to serialize to MessagePack"),
//     };

//     let mut file = File::create(filename).expect("Failed to create file");
//     file.write_all(&serialized_data).expect("Failed to write to file");
// }

// pub(crate) fn load_network(filename: &str, format: SerializationFormat) -> NetworkSerialized {
//     let mut file = File::open(filename).expect("Failed to open file");
//     let mut buffer = Vec::new();
//     file.read_to_end(&mut buffer).expect("Failed to read file");

//     match format {
//         SerializationFormat::Json => serde_json::from_slice(&buffer).expect("Failed to deserialize from JSON"),
//         SerializationFormat::MessagePack => {
//             decode::from_slice(&buffer).expect("Failed to deserialize from MessagePack")
//         }
//     }
// }

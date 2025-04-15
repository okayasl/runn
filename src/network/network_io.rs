use rmp_serde::{decode, encode};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

use crate::layer::Layer;
use crate::{EarlyStopper, LossFunction, OptimizerConfig, Regularization};

pub enum SerializationFormat {
    Json,
    MessagePack,
}

#[derive(Serialize, Deserialize)]
pub struct NetworkIO {
    pub(crate) input_size: usize,
    pub(crate) output_size: usize,
    pub(crate) layers: Vec<Box<dyn Layer>>,
    pub(crate) loss_function: Box<dyn LossFunction>,
    pub(crate) optimizer_config: Box<dyn OptimizerConfig>,
    pub(crate) regularization: Vec<Box<dyn Regularization>>,
    pub(crate) batch_size: usize,
    pub(crate) batch_group_size: usize,
    pub(crate) epochs: usize,
    pub(crate) clip_threshold: f32,
    pub(crate) seed: u64,
    pub(crate) early_stopper: Option<Box<dyn EarlyStopper>>,
    pub(crate) debug: bool,
    pub(crate) normalized: bool,
    pub(crate) mins: Option<Vec<f32>>,
    pub(crate) maxs: Option<Vec<f32>>,
    pub(crate) summary_writer: Option<Box<dyn crate::summary::SummaryWriter>>,
}

pub fn save_network(network_io: &NetworkIO, filename: &str, format: SerializationFormat) {
    let serialized_data = match format {
        SerializationFormat::Json => {
            serde_json::to_vec(&network_io).expect("Failed to serialize to JSON")
        }
        SerializationFormat::MessagePack => {
            encode::to_vec(&network_io).expect("Failed to serialize to MessagePack")
        }
    };

    let mut file = File::create(filename).expect("Failed to create file");
    file.write_all(&serialized_data)
        .expect("Failed to write to file");
}

pub fn load_network(filename: &str, format: SerializationFormat) -> NetworkIO {
    let mut file = File::open(filename).expect("Failed to open file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");

    match format {
        SerializationFormat::Json => {
            serde_json::from_slice(&buffer).expect("Failed to deserialize from JSON")
        }
        SerializationFormat::MessagePack => {
            decode::from_slice(&buffer).expect("Failed to deserialize from MessagePack")
        }
    }
}

use csv::Writer;
use std::fs;
use std::fs::File;
use std::io::{self};

#[derive(Debug)]
pub struct NetworkResult {
    pub(crate) learning_rate: f32,
    pub(crate) batch_size: usize,
    pub(crate) layer_sizes: Vec<usize>,
    pub(crate) t_accuracy: f32,
    pub(crate) t_loss: f32,
    pub(crate) v_accuracy: f32,
    pub(crate) v_loss: f32,
    pub(crate) elapsed_time: f32,
}

impl NetworkResult {
    fn values(&self) -> Vec<String> {
        let size_string: Vec<String> = self
            .layer_sizes
            .iter()
            .map(|&size| size.to_string())
            .collect();
        vec![
            format!("{:.5}", self.learning_rate),
            self.batch_size.to_string(),
            size_string.join(","),
            format!("{:.5}", self.t_loss),
            format!("{:.3}", self.t_accuracy),
            format!("{:.5}", self.v_loss),
            format!("{:.3}", self.v_accuracy),
            format!("{:.3}", self.elapsed_time),
        ]
    }
}

fn default_headers() -> Vec<&'static str> {
    vec![
        "LearningRate",
        "BatchSize",
        "HiddenLayerSizes",
        "TrainingLoss",
        "TrainingAccuracy",
        "ValidationLoss",
        "ValidationAccuracy",
        "ElapsedTime",
    ]
}

pub(crate) fn write_search_results(name: &str, results: &[NetworkResult]) -> io::Result<()> {
    if !std::path::Path::new(".out").exists() {
        fs::create_dir(".out")?;
    }
    let file_path = format!(".out/{}-result.csv", name);
    let file = File::create(file_path)?;
    let mut writer = Writer::from_writer(file);

    writer.write_record(default_headers())?;

    for result in results {
        writer.write_record(result.values())?;
    }

    writer.flush()?;
    Ok(())
}

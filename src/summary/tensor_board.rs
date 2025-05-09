use std::{error::Error, fs};

use serde::{Deserialize, Serialize};
use tensorboard_rs::summary_writer::SummaryWriter as InnerWriter;

use crate::error::NetworkError;

use super::{SummaryWriter, SummaryWriterClone};

#[derive(Serialize, Deserialize)]
struct TensorBoardSummaryWriter {
    logdir: String,
    #[serde(skip)] // Skip serialization of the InnerWriter itself
    inner: Option<InnerWriter>,
}

impl TensorBoardSummaryWriter {
    pub fn new(logdir: &str) -> Self {
        TensorBoardSummaryWriter {
            logdir: logdir.to_string(),
            inner: None,
        }
    }

    fn compute_histogram_stats(&self, values: &[f32]) -> (f64, f64, f64, f64, f64) {
        let (min, max, sum, sum_squares) =
            values
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0), |(min, max, sum, ss), &v| {
                    let v = v as f64;
                    (min.min(v), max.max(v), sum + v, ss + v * v)
                });
        let (min, max) = if min == max { (min, min + 1.0) } else { (min, max) };
        (min, max, values.len() as f64, sum, sum_squares)
    }

    fn generate_bucket_limits(&self, min: f64, max: f64, bucket_count: usize) -> Vec<f64> {
        let bucket_width = (max - min) / bucket_count as f64;
        (0..bucket_count).map(|i| min + (i + 1) as f64 * bucket_width).collect()
    }

    fn compute_bucket_counts(&self, values: &[f32], min: f64, max: f64, bucket_count: usize) -> Vec<f64> {
        let bucket_width = (max - min) / bucket_count as f64;
        let mut bucket_counts = vec![0.0; bucket_count];
        values.iter().for_each(|&v| {
            let index = ((v as f64 - min) / bucket_width).floor() as usize;
            bucket_counts[index.min(bucket_count - 1)] += 1.0;
        });
        bucket_counts
    }
}

#[typetag::serde]
impl SummaryWriter for TensorBoardSummaryWriter {
    fn write_scalar(&mut self, tag: &str, step: usize, value: f32) -> Result<(), Box<dyn Error>> {
        if let Some(inner_writer) = &mut self.inner {
            inner_writer.add_scalar(tag, value, step);
        }
        Ok(())
    }

    fn write_histogram(&mut self, tag: &str, step: usize, values: &[f32]) -> Result<(), Box<dyn Error>> {
        if values.is_empty() {
            return Err("values slice is empty".into());
        }
        let bucket_count = 10;
        let (min, max, num, sum, sum_squares) = self.compute_histogram_stats(values);
        let bucket_limits = self.generate_bucket_limits(min, max, bucket_count);
        let bucket_counts = self.compute_bucket_counts(values, min, max, bucket_count);
        if let Some(inner_writer) = &mut self.inner {
            inner_writer.add_histogram_raw(
                tag,
                min,
                max,
                num,
                sum,
                sum_squares,
                &bucket_limits,
                &bucket_counts,
                step as usize,
            );
        }
        Ok(())
    }

    fn close(&mut self) -> Result<(), Box<dyn Error>> {
        if let Some(inner_writer) = &mut self.inner {
            inner_writer.flush();
        }
        Ok(())
    }
    fn init(&mut self) {
        self.inner = Some(InnerWriter::new(&self.logdir));
    }
}

impl SummaryWriterClone for TensorBoardSummaryWriter {
    fn clone_box(&self) -> Box<dyn SummaryWriter> {
        Box::new(self.clone())
    }
}

impl Clone for TensorBoardSummaryWriter {
    fn clone(&self) -> Self {
        TensorBoardSummaryWriter {
            logdir: self.logdir.clone(),
            inner: None, // Recreate the `InnerWriter`
        }
    }
}

/// TensorBoard is builder for configuring a TensorBoard summary writer.
///
/// This struct sets up a TensorBoard logger to write training metrics (e.g., scalars, histograms) to a specified log directory for visualization.
/// Default settings:
/// - directory: `"."`
pub struct TensorBoard {
    directory: String,
}

impl TensorBoard {
    pub fn new() -> Self {
        TensorBoard {
            directory: ".".to_string(),
        }
    }

    /// Set the log directory for TensorBoard output.
    ///
    /// Specifies the directory where TensorBoard event files will be written for visualization.
    /// # Parameters
    /// - `directory`: Path to the log directory (e.g., "./logs").
    pub fn directory(mut self, logdir: &str) -> Self {
        self.directory = logdir.to_string();
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        // Check if the directory is set
        if self.directory.is_empty() {
            return Err(NetworkError::ConfigError("Log directory cannot be empty".to_string()));
        }

        // Check if the directory exists, and attempt to create it if it doesn't
        if !std::path::Path::new(&self.directory).exists() {
            fs::create_dir_all(&self.directory).map_err(|e| {
                NetworkError::IoError(format!("Failed to create output directory '{}': {}", self.directory, e))
            })?;
        }

        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn SummaryWriter>, NetworkError> {
        self.validate()?;
        let logdir = self.directory.clone();
        Ok(Box::new(TensorBoardSummaryWriter::new(&logdir)))
    }
}

#[cfg(test)]
mod tests {

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_write_scalar() {
        let temp_dir = tempdir().unwrap();
        let logdir = temp_dir.path().to_str().unwrap();
        let mut writer = TensorBoardSummaryWriter::new(logdir);
        writer.init();

        let result = writer.write_scalar("test_scalar", 1, 42.0);
        assert!(result.is_ok());

        writer.close().unwrap();
    }

    #[test]
    fn test_write_histogram() {
        let temp_dir = tempdir().unwrap();
        let logdir = temp_dir.path().to_str().unwrap();
        let mut writer = TensorBoardSummaryWriter::new(logdir);
        writer.init();

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = writer.write_histogram("test_histogram", 1, &values);
        assert!(result.is_ok());

        writer.close().unwrap();
    }

    #[test]
    fn test_write_histogram_empty_values() {
        let temp_dir = tempdir().unwrap();
        let logdir = temp_dir.path().to_str().unwrap();
        let mut writer = TensorBoardSummaryWriter::new(logdir);

        let values: Vec<f32> = vec![];
        let result = writer.write_histogram("test_histogram_empty", 1, &values);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_histogram_stats() {
        let writer = TensorBoardSummaryWriter::new("/tmp");
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (min, max, num, sum, sum_squares) = writer.compute_histogram_stats(&values);

        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert_eq!(num, 5.0);
        assert_eq!(sum, 15.0);
        assert_eq!(sum_squares, 55.0);
    }

    #[test]
    fn test_generate_bucket_limits() {
        let writer = TensorBoardSummaryWriter::new("/tmp");
        let bucket_limits = writer.generate_bucket_limits(0.0, 10.0, 5);

        assert_eq!(bucket_limits, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_compute_bucket_counts() {
        let writer = TensorBoardSummaryWriter::new("/tmp");
        let values = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let bucket_counts = writer.compute_bucket_counts(&values, 0.0, 1.0, 5);

        assert_eq!(bucket_counts, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_tensor_board() {
        let temp_dir = tempdir().unwrap();
        let logdir = temp_dir.path().to_str().unwrap();
        let tensor_board = TensorBoard::new().directory(logdir);
        let result = tensor_board.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tensor_board_invalid_logdir() {
        let tensor_board = TensorBoard::new().directory("");
        let result = tensor_board.build();
        assert!(result.is_err());
    }
}

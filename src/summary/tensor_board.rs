use crate::error::NetworkError;

use super::gen::tensorboard;
use super::{r#gen, SummaryWriter, SummaryWriterClone};
use crc::{Crc, CRC_32_ISCSI};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write}; // Added io for io::Error
use std::path::PathBuf; // For potentially cleaner path construction
use std::time::{SystemTime, UNIX_EPOCH};
use tensorboard::summary::Value;
use tensorboard::{Event, Summary};

#[derive(Serialize, Deserialize)]
struct TensorBoardSummaryWriter {
    directory: String,
    #[serde(skip)]
    inner: Option<InnerWriter>,
}

// Helper to get current wall time
fn current_wall_time_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("SystemTime before UNIX EPOCH!") // Or handle this error more gracefully
        .as_secs_f64()
}

impl TensorBoardSummaryWriter {
    fn new(logdir: &str) -> Self {
        TensorBoardSummaryWriter {
            directory: logdir.to_string(),
            inner: None,
        }
    }

    // No changes needed in histogram computation logic, it's quite standard.
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
        if bucket_count == 0 {
            return vec![];
        } // Avoid division by zero
        let bucket_width = (max - min) / bucket_count as f64;
        let mut bucket_counts = vec![0.0; bucket_count];
        values.iter().for_each(|&v| {
            let v_f64 = v as f64;
            // Handle edge case where v_f64 might be slightly less than min due to precision
            let index = if v_f64 <= min {
                0
            } else {
                ((v_f64 - min) / bucket_width).floor() as usize
            };
            bucket_counts[index.min(bucket_count - 1)] += 1.0;
        });
        bucket_counts
    }
}

#[typetag::serde]
impl SummaryWriter for TensorBoardSummaryWriter {
    fn write_scalar(&mut self, tag: &str, step: usize, value: f32) -> Result<(), Box<dyn Error>> {
        if let Some(inner_writer) = &mut self.inner {
            inner_writer.add_scalar(tag, value, step)?; // Propagate error
        }
        Ok(())
    }

    fn write_histogram(&mut self, tag: &str, step: usize, values: &[f32]) -> Result<(), Box<dyn Error>> {
        if values.is_empty() {
            return Err("values slice is empty".into());
        }
        let bucket_count = 10; // Consider making this configurable
        let (min, max, num, sum, sum_squares) = self.compute_histogram_stats(values);
        let bucket_limits = self.generate_bucket_limits(min, max, bucket_count);
        let bucket_counts = self.compute_bucket_counts(values, min, max, bucket_count);

        if let Some(inner_writer) = &mut self.inner {
            let hc = HistogramConfigBuilder::new()
                .min(min)
                .max(max)
                .num(num)
                .sum(sum)
                .sum_squares(sum_squares)
                .bucket_limits(bucket_limits)
                .bucket_counts(bucket_counts)
                .step(step)
                .build(); // Assuming build() panics on missing fields for crate-internal struct
            inner_writer.add_histogram_raw(tag, hc)?; // Propagate error
        }
        Ok(())
    }

    fn close(&mut self) -> Result<(), Box<dyn Error>> {
        if let Some(inner_writer) = &mut self.inner {
            inner_writer.flush()?; // Propagate error
        }
        Ok(())
    }

    fn init(&mut self) {
        // Since SummaryWriter::init does not return Result, we expect InnerWriter::new to succeed
        // or we panic, consistent with the original code's implicit panics on file errors.
        self.inner = Some(InnerWriter::new(&self.directory).expect("Failed to initialize TensorBoard InnerWriter"));
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
            directory: self.directory.clone(),
            // InnerWriter is not cloned; it will be recreated (initialized)
            // when `init()` is called on the new cloned instance.
            inner: None,
        }
    }
}

pub struct TensorBoard {
    directory: String,
}

impl TensorBoard {
    fn new() -> Self {
        TensorBoard {
            directory: ".".to_string(),
        }
    }

    pub fn logdir(mut self, logdir: &str) -> Self {
        self.directory = logdir.to_string();
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.directory.is_empty() {
            return Err(NetworkError::ConfigError("Log directory cannot be empty".to_string()));
        }
        let path = PathBuf::from(&self.directory);
        if !path.exists() {
            fs::create_dir_all(&path).map_err(|e| {
                NetworkError::IoError(format!("Failed to create output directory '{}': {}", self.directory, e))
            })?;
        }
        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn SummaryWriter>, NetworkError> {
        self.validate()?;
        // No need to clone directory here if TensorBoardSummaryWriter::new takes String
        // but it takes &str, so clone is fine.
        Ok(Box::new(TensorBoardSummaryWriter::new(&self.directory)))
    }
}

impl Default for TensorBoard {
    fn default() -> Self {
        Self::new()
    }
}

struct InnerWriter {
    writer: BufWriter<File>,
    // start_time is only used for the initial file_version event.
    // Each subsequent event should get its own fresh wall_time.
}

impl InnerWriter {
    fn new(logdir: &str) -> Result<Self, io::Error> {
        // Return Result
        let wall_time = current_wall_time_secs();
        let filename = format!(
            "{}/events.out.tfevents.{:.0}", // Consider PathBuf::join for robustness
            logdir, wall_time
        );

        let file = OpenOptions::new()
            .create(true)
            .append(true) // Changed from write+truncate to append for robustness if multiple writers are (incorrectly) pointed to same file or for restarts
            // .truncate(true) // If truncate is essential, ensure only one writer instance exists per file.
            .open(filename)?; // Use ?

        let mut writer = InnerWriter {
            writer: BufWriter::new(file),
        };
        writer.write_file_version(wall_time)?; // Pass wall_time, use ?
        Ok(writer)
    }

    fn write_file_version(&mut self, start_time: f64) -> Result<(), io::Error> {
        // Return Result
        let event = Event {
            wall_time: start_time,
            step: 0,
            what: Some(gen::tensorboard::event::What::FileVersion("brain.Event:2".to_string())),
            ..Default::default()
        };
        self.write_event(&event)?; // Use ?
        self.writer.flush() // Use ?, flush is important here
    }

    fn add_scalar(&mut self, tag: &str, value: f32, step: usize) -> Result<(), io::Error> {
        // Return Result
        let event = Event {
            wall_time: current_wall_time_secs(),
            step: step as i64,
            what: Some(gen::tensorboard::event::What::Summary(Summary {
                value: vec![Value {
                    tag: tag.to_string(),
                    value: Some(gen::tensorboard::summary::value::Value::Simple(value)),
                    ..Default::default()
                }],
            })),
            ..Default::default()
        };
        self.write_event(&event) // Use ?
    }

    fn add_histogram_raw(&mut self, tag: &str, config: HistogramConfig) -> Result<(), io::Error> {
        // Return Result
        let hist = gen::tensorboard::HistogramProto {
            min: config.min,
            max: config.max,
            num: config.num,
            sum: config.sum,
            sum_squares: config.sum_squares,
            bucket_limit: config.bucket_limits,
            bucket: config.bucket_counts,
        };
        let event = Event {
            wall_time: current_wall_time_secs(),
            step: config.step as i64,
            what: Some(gen::tensorboard::event::What::Summary(Summary {
                value: vec![Value {
                    tag: tag.to_string(),
                    value: Some(gen::tensorboard::summary::value::Value::Histo(hist)),
                    ..Default::default()
                }],
            })),
            ..Default::default()
        };
        self.write_event(&event) // Use ?
    }

    fn write_event(&mut self, event: &Event) -> Result<(), io::Error> {
        // Return Result
        let mut buf = Vec::new();
        event.encode(&mut buf).map_err(|e| {
            // Handle prost encode error
            io::Error::new(io::ErrorKind::InvalidData, format!("Failed to encode event: {}", e))
        })?;
        let len = buf.len() as u64;

        const CRC32C_ALGORITHM: Crc<u32> = Crc::<u32>::new(&CRC_32_ISCSI);

        let mut len_digest = CRC32C_ALGORITHM.digest();
        len_digest.update(&len.to_le_bytes());
        let len_crc32 = len_digest.finalize();
        let masked_len_crc32 = mask_crc32(len_crc32);

        let mut data_digest = CRC32C_ALGORITHM.digest();
        data_digest.update(&buf);
        let data_crc32 = data_digest.finalize();
        let masked_data_crc32 = mask_crc32(data_crc32);

        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&masked_len_crc32.to_le_bytes())?;
        self.writer.write_all(&buf)?;
        self.writer.write_all(&masked_data_crc32.to_le_bytes())?;
        self.writer.flush() // Flush after each event to ensure it's written
                            // Consider if flushing less frequently is acceptable for performance.
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        // Return Result
        self.writer.flush()
    }
}

fn mask_crc32(crc: u32) -> u32 {
    crc.rotate_right(15).wrapping_add(0xa282ead8)
}

// HistogramConfig and HistogramConfigBuilder are mostly fine.
// Using `expect` in the builder's `build` method is common for crate-internal
// builders where you control all calls. If it were public, Result would be better.
pub(crate) struct HistogramConfig {
    min: f64,
    max: f64,
    num: f64,
    sum: f64,
    sum_squares: f64,
    bucket_limits: Vec<f64>,
    bucket_counts: Vec<f64>,
    step: usize,
}

pub(crate) struct HistogramConfigBuilder {
    min: Option<f64>,
    max: Option<f64>,
    num: Option<f64>,
    sum: Option<f64>,
    sum_squares: Option<f64>,
    bucket_limits: Option<Vec<f64>>,
    bucket_counts: Option<Vec<f64>>,
    step: Option<usize>,
}

impl HistogramConfigBuilder {
    pub(crate) fn new() -> Self {
        Self {
            min: None,
            max: None,
            num: None,
            sum: None,
            sum_squares: None,
            bucket_limits: None,
            bucket_counts: None,
            step: None,
        }
    }

    // Setter methods are fine as they are (chainable)
    pub(crate) fn min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }
    pub(crate) fn max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }
    pub(crate) fn num(mut self, num: f64) -> Self {
        self.num = Some(num);
        self
    }
    pub(crate) fn sum(mut self, sum: f64) -> Self {
        self.sum = Some(sum);
        self
    }
    pub(crate) fn sum_squares(mut self, sum_squares: f64) -> Self {
        self.sum_squares = Some(sum_squares);
        self
    }
    pub(crate) fn bucket_limits(mut self, bucket_limits: Vec<f64>) -> Self {
        self.bucket_limits = Some(bucket_limits);
        self
    }
    pub(crate) fn bucket_counts(mut self, bucket_counts: Vec<f64>) -> Self {
        self.bucket_counts = Some(bucket_counts);
        self
    }
    pub(crate) fn step(mut self, step: usize) -> Self {
        self.step = Some(step);
        self
    }

    pub(crate) fn build(self) -> HistogramConfig {
        HistogramConfig {
            min: self.min.expect("min must be set for HistogramConfig"),
            max: self.max.expect("max must be set for HistogramConfig"),
            num: self.num.expect("num must be set for HistogramConfig"),
            sum: self.sum.expect("sum must be set for HistogramConfig"),
            sum_squares: self.sum_squares.expect("sum_squares must be set for HistogramConfig"),
            bucket_limits: self
                .bucket_limits
                .expect("bucket_limits must be set for HistogramConfig"),
            bucket_counts: self
                .bucket_counts
                .expect("bucket_counts must be set for HistogramConfig"),
            step: self.step.expect("step must be set for HistogramConfig"),
        }
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
        let tensor_board = TensorBoard::default().logdir(logdir);
        let result = tensor_board.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tensor_board_invalid_directory() {
        let tensor_board = TensorBoard::default().logdir("");
        let result = tensor_board.build();
        assert!(result.is_err());
    }
}

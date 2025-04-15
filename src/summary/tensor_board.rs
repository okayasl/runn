use std::error::Error;

use tensorboard_rs::summary_writer::SummaryWriter as InnerWriter;

use super::SummaryWriter;

pub struct TensorBoardSummaryWriter {
    inner: InnerWriter,
}

impl TensorBoardSummaryWriter {
    pub fn new(logdir: &str) -> Self {
        TensorBoardSummaryWriter {
            inner: InnerWriter::new(logdir),
        }
    }

    fn compute_histogram_stats(&self, values: &[f32]) -> (f64, f64, f64, f64, f64) {
        let (min, max, sum, sum_squares) = values.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0),
            |(min, max, sum, ss), &v| {
                let v = v as f64;
                (min.min(v), max.max(v), sum + v, ss + v * v)
            },
        );
        let (min, max) = if min == max {
            (min, min + 1.0)
        } else {
            (min, max)
        };
        (min, max, values.len() as f64, sum, sum_squares)
    }

    fn generate_bucket_limits(&self, min: f64, max: f64, bucket_count: usize) -> Vec<f64> {
        let bucket_width = (max - min) / bucket_count as f64;
        (0..bucket_count)
            .map(|i| min + (i + 1) as f64 * bucket_width)
            .collect()
    }

    fn compute_bucket_counts(
        &self,
        values: &[f32],
        min: f64,
        bucket_width: f64,
        bucket_count: usize,
    ) -> Vec<f64> {
        let mut bucket_counts = vec![0.0; bucket_count];
        values.iter().for_each(|&v| {
            let index = ((v as f64 - min) / bucket_width).floor() as usize;
            bucket_counts[index.min(bucket_count - 1)] += 1.0;
        });
        bucket_counts
    }
}

impl SummaryWriter for TensorBoardSummaryWriter {
    fn write_scalar(&mut self, tag: &str, step: usize, value: f32) -> Result<(), Box<dyn Error>> {
        self.inner.add_scalar(tag, value, step);
        Ok(())
    }

    fn write_histogram(
        &mut self,
        tag: &str,
        step: usize,
        values: &[f32],
    ) -> Result<(), Box<dyn Error>> {
        if values.is_empty() {
            return Err("values slice is empty".into());
        }
        let bucket_count = 10;
        let (min, max, num, sum, sum_squares) = self.compute_histogram_stats(values);
        let bucket_limits = self.generate_bucket_limits(min, max, bucket_count);
        let bucket_counts = self.compute_bucket_counts(
            values,
            min,
            (max - min) / bucket_count as f64,
            bucket_count,
        );
        self.inner.add_histogram_raw(
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
        Ok(())
    }

    fn close(&mut self) -> Result<(), Box<dyn Error>> {
        self.inner.flush();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_write_scalar() {
        let temp_dir = tempdir().unwrap();
        let logdir = temp_dir.path().to_str().unwrap();
        let mut writer = TensorBoardSummaryWriter::new(logdir);

        let result = writer.write_scalar("test_scalar", 1, 42.0);
        assert!(result.is_ok());

        writer.close().unwrap();
        assert!(fs::read_dir(logdir).unwrap().count() > 0);
    }

    #[test]
    fn test_write_histogram() {
        let temp_dir = tempdir().unwrap();
        let logdir = temp_dir.path().to_str().unwrap();
        let mut writer = TensorBoardSummaryWriter::new(logdir);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = writer.write_histogram("test_histogram", 1, &values);
        assert!(result.is_ok());

        writer.close().unwrap();
        assert!(fs::read_dir(logdir).unwrap().count() > 0);
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
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bucket_counts = writer.compute_bucket_counts(&values, 0.0, 1.0, 5);

        assert_eq!(bucket_counts, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }
}

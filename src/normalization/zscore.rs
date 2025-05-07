use serde::{Deserialize, Serialize};

use crate::matrix::DMat;

use super::Normalization;

/// A builder for z-score normalization, standardizing input data to have zero mean and unit variance.
///
/// Z-score normalization transforms each feature by subtracting its mean and dividing by its standard deviation.
#[derive(Serialize, Deserialize, Clone)]
pub struct ZScore {
    means: Option<Vec<f32>>,
    std_devs: Option<Vec<f32>>,
}

impl ZScore {
    pub fn new() -> Self {
        Self {
            means: None,
            std_devs: None,
        }
    }

    fn compute_mean_std(&mut self, matrix: &DMat) {
        let (rows, cols) = (matrix.rows(), matrix.cols());
        let mut means = vec![0.0; cols];
        let mut std_devs = vec![0.0; cols];

        // Calculate means
        for j in 0..cols {
            let mut sum = 0.0;
            for i in 0..rows {
                sum += matrix.at(i, j);
            }
            means[j] = sum / rows as f32;
        }

        // Calculate standard deviations
        for j in 0..cols {
            let mut sum = 0.0;
            for i in 0..rows {
                let diff = matrix.at(i, j) - means[j];
                sum += diff * diff;
            }
            std_devs[j] = (sum / rows as f32).sqrt();
        }

        self.means = Some(means);
        self.std_devs = Some(std_devs);
    }
}

#[typetag::serde]
impl Normalization for ZScore {
    fn normalize(&mut self, matrix: &mut DMat) -> Result<(), String> {
        if self.means.is_none() || self.std_devs.is_none() {
            self.compute_mean_std(matrix);
        }

        let (rows, cols) = (matrix.rows(), matrix.cols());

        let means = self.means.as_ref().ok_or_else(|| "Means not initialized".to_string())?;
        let std_devs = self
            .std_devs
            .as_ref()
            .ok_or_else(|| "Standard deviations not initialized".to_string())?;

        if means.len() != cols || std_devs.len() != cols {
            return Err("Matrix column size does not match the initialized mean/std dev sizes.".to_string());
        }

        for i in 0..rows {
            for j in 0..cols {
                let val = matrix.at(i, j);
                let mean = means[j];
                let std_dev = std_devs[j];

                if std_dev.abs() < f32::EPSILON {
                    matrix.set(i, j, 0.0); // If standard deviation is zero, set to 0
                } else {
                    matrix.set(i, j, (val - mean) / std_dev); // Normalize value
                }
            }
        }
        Ok(())
    }

    fn denormalize(&self, matrix: &mut DMat) -> Result<(), String> {
        let (rows, cols) = (matrix.rows(), matrix.cols());

        let means = self.means.as_ref().ok_or_else(|| "Means not initialized".to_string())?;
        let std_devs = self
            .std_devs
            .as_ref()
            .ok_or_else(|| "Standard deviations not initialized".to_string())?;

        if means.len() != cols || std_devs.len() != cols {
            return Err("Matrix column size does not match the initialized mean/std dev sizes.".to_string());
        }

        for i in 0..rows {
            for j in 0..cols {
                let val = matrix.at(i, j);
                let mean = means[j];
                let std_dev = std_devs[j];

                if std_dev.abs() < f32::EPSILON {
                    matrix.set(i, j, mean); // If std dev is zero, set to mean
                } else {
                    matrix.set(i, j, val * std_dev + mean); // Denormalize value
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::DMat; // Import DenseMatrix

    // Test for Z-Score normalization and denormalization
    #[test]
    fn test_z_score_normalization() {
        let matrix_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut matrix = DMat::new(3, 3, &matrix_data);

        // Z-Score Normalization
        let mut z_score = ZScore::new();
        z_score.normalize(&mut matrix).unwrap();

        // After normalization, the mean of each column should be close to 0 and std dev close to 1
        for j in 0..matrix.cols() {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for i in 0..matrix.rows() {
                let val = matrix.at(i, j);
                sum += val;
                sum_sq += val * val;
            }

            let mean = sum / matrix.rows() as f32;
            let std_dev = (sum_sq / matrix.rows() as f32 - mean * mean).sqrt();

            // Assert that the mean is close to 0 and standard deviation close to 1
            assert!((mean).abs() < 0.1); // Mean should be close to 0
            assert!((std_dev - 1.0).abs() < 0.1); // Std dev should be close to 1
        }

        // Denormalization
        z_score.denormalize(&mut matrix).unwrap();

        // Ensure the denormalized values match the original matrix
        let original = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];

        // Compare denormalized matrix with the original matrix
        for i in 0..matrix.rows() {
            for j in 0..matrix.cols() {
                assert!((matrix.at(i, j) - original[i][j]).abs() < f32::EPSILON);
            }
        }
    }
}

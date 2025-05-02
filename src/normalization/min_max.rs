use serde::{Deserialize, Serialize};

use crate::matrix::DenseMatrix;

use super::Normalization;

/// A builder for min-max normalization, scaling input data to a specified range (typically [0, 1]).
///
/// Min-max normalization transforms each feature by subtracting its minimum value and dividing by the range (max - min).
/// The minimum and maximum values are computed from the input data when `compute_min_max` is called.
#[derive(Serialize, Deserialize, Clone)]
pub struct MinMax {
    mins: Option<Vec<f32>>,
    maxs: Option<Vec<f32>>,
}

impl MinMax {
    pub fn new() -> Self {
        Self { mins: None, maxs: None }
    }

    fn compute_min_max(&mut self, matrix: &DenseMatrix) {
        let (rows, cols) = (matrix.rows(), matrix.cols());
        let mut mins = vec![f32::INFINITY; cols];
        let mut maxs = vec![f32::NEG_INFINITY; cols];

        for j in 0..cols {
            for i in 0..rows {
                let val = matrix.at(i, j);
                if val < mins[j] {
                    mins[j] = val;
                }
                if val > maxs[j] {
                    maxs[j] = val;
                }
            }
        }
        self.mins = Some(mins);
        self.maxs = Some(maxs);
    }
}

#[typetag::serde]
impl Normalization for MinMax {
    fn normalize(&mut self, matrix: &mut DenseMatrix) -> Result<(), String> {
        // If mins and maxs are not computed, do it now
        if self.mins.is_none() || self.maxs.is_none() {
            self.compute_min_max(matrix);
        }

        let (rows, cols) = (matrix.rows(), matrix.cols());

        // Check if dimensions match
        let mins = self.mins.as_ref().ok_or_else(|| "Mins not initialized".to_string())?;
        let maxs = self.maxs.as_ref().ok_or_else(|| "Maxs not initialized".to_string())?;

        if mins.len() != cols || maxs.len() != cols {
            return Err("Matrix column size does not match the initialized min/max sizes.".to_string());
        }

        for i in 0..rows {
            for j in 0..cols {
                let val = matrix.at(i, j);
                let min = mins[j];
                let max = maxs[j];

                if (max - min).abs() < f32::EPSILON {
                    matrix.set(i, j, 0.0); // Set to 0 if min and max are the same
                } else {
                    matrix.set(i, j, (val - min) / (max - min)); // Normalize value
                }
            }
        }
        Ok(())
    }

    fn denormalize(&self, matrix: &mut DenseMatrix) -> Result<(), String> {
        let (rows, cols) = (matrix.rows(), matrix.cols());

        // Check if dimensions match
        let mins = self.mins.as_ref().ok_or_else(|| "Mins not initialized".to_string())?;
        let maxs = self.maxs.as_ref().ok_or_else(|| "Maxs not initialized".to_string())?;

        if mins.len() != cols || maxs.len() != cols {
            return Err("Matrix column size does not match the initialized min/max sizes.".to_string());
        }

        for i in 0..rows {
            for j in 0..cols {
                let val = matrix.at(i, j);
                let min = mins[j];
                let max = maxs[j];

                if (max - min).abs() < f32::EPSILON {
                    matrix.set(i, j, min); // Set to min if min and max are the same
                } else {
                    matrix.set(i, j, val * (max - min) + min); // Denormalize value
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::DenseMatrix; // Import DenseMatrix

    // Test for Min-Max normalization and denormalization
    #[test]
    fn test_min_max_normalization() {
        let matrix_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut matrix = DenseMatrix::new(3, 3, &matrix_data);

        // Min-Max Normalization
        let mut min_max = MinMax::new();
        min_max.normalize(&mut matrix).unwrap();

        // Expected normalized values for each column in the matrix
        let normalized = vec![vec![0.0, 0.0, 0.0], vec![0.5, 0.5, 0.5], vec![1.0, 1.0, 1.0]];

        // Compare the normalized matrix with expected values
        for i in 0..matrix.rows() {
            for j in 0..matrix.cols() {
                assert!((matrix.at(i, j) - normalized[i][j]).abs() < f32::EPSILON);
            }
        }

        // Denormalization
        min_max.denormalize(&mut matrix).unwrap();

        // Expected values after denormalization
        let original = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];

        // Compare denormalized matrix with the original
        for i in 0..matrix.rows() {
            for j in 0..matrix.cols() {
                assert!((matrix.at(i, j) - original[i][j]).abs() < f32::EPSILON);
            }
        }
    }
}

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use super::error::NetworkError;

/// A structure encapsulating a dense matrix.
#[derive(Clone, Serialize, Deserialize)]
pub struct DMat {
    data: DMatrix<f32>,
}

impl DMat {
    /// Creates a new dense matrix with given rows, columns, and data. Always create in row major order.
    pub(crate) fn new(rows: usize, cols: usize, data: &[f32]) -> Self {
        Self {
            data: DMatrix::from_row_slice(rows, cols, data),
        }
    }

    pub(crate) fn mul_new(other: &DMat, another: &DMat) -> DMat {
        DMat {
            data: &other.data * &another.data,
        }
    }

    /// Creates a new dense matrix with given rows and columns, initialized with zeros.
    pub(crate) fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: DMatrix::zeros(rows, cols),
        }
    }

    /// Returns a transposed version of the matrix.
    pub(crate) fn transpose(&self) -> DMat {
        Self {
            data: self.data.transpose(),
        }
    }

    /// Returns the number of rows in the matrix.
    #[inline]
    pub fn rows(&self) -> usize {
        self.data.nrows()
    }

    /// Returns the number of columns in the matrix.
    #[inline]
    pub fn cols(&self) -> usize {
        self.data.ncols()
    }

    /// Gets the value at position (i, j).
    #[inline]
    pub(crate) fn at(&self, i: usize, j: usize) -> f32 {
        self.data[(i, j)]
    }

    /// Sets the value at position (i, j).
    #[inline]
    pub(crate) fn set(&mut self, i: usize, j: usize, value: f32) {
        self.data[(i, j)] = value;
    }

    /// Adds another matrix to the current matrix.
    #[inline]
    pub(crate) fn add(&mut self, other: &DMat) {
        self.data += &other.data;
    }

    /// Scales the matrix by a scalar factor.
    #[inline]
    pub(crate) fn scale(&mut self, factor: f32) {
        self.data *= factor;
    }

    /// Element-wise multiplication with another matrix.
    #[inline]
    pub(crate) fn mul_elem(&mut self, other: &DMat) {
        self.data.component_mul_assign(&other.data);
    }

    // Extracts a submatrix as a new DenseMatrix.
    pub(crate) fn slice(&self, i: usize, k: usize, j: usize, l: usize) -> DMat {
        let rows = k - i;
        let cols = l - j;
        DMat {
            data: DMatrix::from_fn(rows, cols, |row, col| self.data[(i + row, j + col)]),
        }
    }

    pub(crate) fn get_row(&self, i: usize) -> Vec<f32> {
        self.data.row(i).iter().cloned().collect()
    }

    // Sets the values of a specific row.
    pub(crate) fn set_row(&mut self, i: usize, src: &[f32]) {
        for (j, &value) in src.iter().enumerate() {
            self.set(i, j, value);
        }
    }

    // Clips the norm of the matrix to a given threshold, by scaling the matrix down if needed.
    pub(crate) fn clip(&mut self, threshold: f32) {
        if threshold > 0.0 {
            let norm = self.norm(2.0);
            if norm > threshold {
                let scale = threshold / norm;
                self.scale(scale);
            }
        }
    }

    // Calculates the norm of the matrix (L1 or L2 supported).
    pub(crate) fn norm(&self, norm_type: f32) -> f32 {
        if norm_type == 1.0 {
            self.data.iter().map(|x| x.abs()).sum::<f32>()
        } else if norm_type == 2.0 {
            (self.data.iter().map(|x| x * x).sum::<f32>()).sqrt()
        } else {
            panic!("Unsupported norm type");
        }
    }

    pub(crate) fn set_column_sum(&mut self, other: &DMat) {
        for i in 0..self.rows() {
            self.data[(i, 0)] = other.data.column(i).iter().sum();
        }
    }

    // Applies a function to each element of the matrix in place.
    pub(crate) fn apply<F>(&mut self, func: F)
    where
        F: Fn(f32) -> f32,
    {
        self.data.apply(|x| *x = func(*x));
    }

    // Applies a function to each element of the matrix, with access to the element's indices.
    pub(crate) fn apply_with_indices<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, usize, &mut f32),
    {
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                f(i, j, &mut self.data[(i, j)]);
            }
        }
    }
}

// impl fmt::Display for DenseMatrix {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         for i in 0..self.rows() {
//             // Decide the row border type
//             let left_border = if i == 0 {
//                 "⎡"
//             } else if i == self.rows() - 1 {
//                 "⎣"
//             } else {
//                 "⎢"
//             };

//             let right_border = if i == 0 {
//                 "⎤"
//             } else if i == self.rows() - 1 {
//                 "⎦"
//             } else {
//                 "⎥"
//             };

//             let row: Vec<String> = (0..self.cols())
//                 .map(|j| format!("{:7.6}", self.at(i, j))) // Format each element to 4 decimal places
//                 .collect();

//             // Format the row with borders
//             writeln!(f, "{} {} {}", left_border, row.join(" "), right_border)?;
//         }
//         Ok(())
//     }
// }

pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}
impl DenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: Vec::new(),
        }
    }

    pub fn data(mut self, data: &[f32]) -> Self {
        self.data = data.to_vec();
        self
    }

    pub fn random(mut self) -> Self {
        // Initialize the data vector with the correct size
        self.data = vec![0.0; self.rows * self.cols];
        // Fill the data with random values
        for i in 0..self.rows * self.cols {
            self.data[i] = rand::random();
        }
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.rows == 0 || self.cols == 0 {
            return Err(NetworkError::ConfigError(format!(
                "Rows:{} and columns:{} must be greater than zero",
                self.rows, self.cols
            )));
        }
        if self.data.len() != self.rows * self.cols {
            return Err(NetworkError::ConfigError(format!(
                "Data length:{} does not match matrix dimensions:{}",
                self.data.len(),
                self.rows * self.cols
            )));
        }
        Ok(())
    }

    pub fn build(self) -> Result<DMat, NetworkError> {
        self.validate()?;
        Ok(DMat::new(self.rows, self.cols, &self.data))
    }
}

// Example usage in tests
#[cfg(test)]
mod tests {
    use crate::util;

    use super::*;

    #[test]
    fn test_at() {
        let matrix = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.at(1, 1), 4.0);
    }

    #[test]
    fn test_add() {
        let mut matrix = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let other = DMat::new(2, 2, &[4.0, 3.0, 2.0, 1.0]);
        matrix.add(&other);
        assert_eq!(util::flatten(&matrix), &[5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_scale() {
        let mut matrix = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        matrix.scale(2.0);
        assert_eq!(util::flatten(&matrix), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_transpose() {
        let matrix = DMat::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = matrix.transpose();
        assert_eq!((transposed.rows(), transposed.cols()), (3, 2));
        assert_eq!(transposed.at(0, 1), 4.0);
    }

    #[test]
    fn test_norm() {
        let matrix = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.norm(1.0), 10.0); // L1 norm
        assert!((matrix.norm(2.0) - 5.477).abs() < 0.001); // L2 norm
    }

    #[test]
    fn test_slice() {
        let matrix = DMat::new(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let submatrix = matrix.slice(1, 2, 1, 2);
        assert_eq!(util::flatten(&submatrix), &[5.0]);
    }

    #[test]
    fn test_mul_elem() {
        // Case 1: Element-wise multiplication for square matrices (2x2)
        let mut matrix_a = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let matrix_b = DMat::new(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        matrix_a.mul_elem(&matrix_b);
        assert_eq!(util::flatten(&matrix_a), &[5.0, 12.0, 21.0, 32.0]);

        // Case 2: Element-wise multiplication for rectangular matrices (2x3)
        let mut matrix_c = DMat::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_d = DMat::new(2, 3, &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        matrix_c.mul_elem(&matrix_d);
        assert_eq!(util::flatten(&matrix_c), &[6.0, 10.0, 12.0, 12.0, 10.0, 6.0]);

        // Case 3: Element-wise multiplication with all ones (identity for element-wise mul)
        let mut matrix_e = DMat::new(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let matrix_ones = DMat::new(3, 3, &[1.0; 9]); // 3x3 matrix of all ones
        matrix_e.mul_elem(&matrix_ones);
        assert_eq!(util::flatten(&matrix_e), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Case 4: Element-wise multiplication with a zero matrix (result should be zero matrix)
        let mut matrix_f = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let zero_matrix = DMat::zeros(2, 2);
        matrix_f.mul_elem(&zero_matrix);
        assert_eq!(util::flatten(&matrix_f), &[0.0, 0.0, 0.0, 0.0]);

        // Case 5: Element-wise multiplication for single-row matrices (1x4)
        let mut matrix_g = DMat::new(1, 4, &[1.0, 2.0, 3.0, 4.0]);
        let matrix_h = DMat::new(1, 4, &[4.0, 3.0, 2.0, 1.0]);
        matrix_g.mul_elem(&matrix_h);
        assert_eq!(util::flatten(&matrix_g), &[4.0, 6.0, 6.0, 4.0]);

        // Case 6: Element-wise multiplication for single-column matrices (4x1)
        let mut matrix_i = DMat::new(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let matrix_j = DMat::new(4, 1, &[4.0, 3.0, 2.0, 1.0]);
        matrix_i.mul_elem(&matrix_j);
        assert_eq!(util::flatten(&matrix_i), &[4.0, 6.0, 6.0, 4.0]);
    }

    #[test]
    fn test_clone() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut matrix = DMat::new(2, 2, &data);
        let mut clone = matrix.clone();

        // Assert that the dimensions and initial data match
        assert_eq!((matrix.rows(), matrix.cols()), (clone.rows(), clone.cols()));
        assert_eq!(matrix.at(0, 0), clone.at(0, 0));

        // Modify the cloned matrix and verify it does not affect the original
        clone.data[(0, 0)] = 5.0;
        assert_eq!(clone.at(0, 0), 5.0); // Cloned matrix is updated
        assert_eq!(matrix.at(0, 0), 1.0); // Original matrix remains unchanged

        // Modify the original matrix and verify it does not affect the clone
        matrix.data[(1, 1)] = 6.0;
        assert_eq!(matrix.at(1, 1), 6.0); // Original matrix is updated
        assert_eq!(clone.at(1, 1), 4.0); // Cloned matrix remains unchanged
    }

    #[test]
    fn test_set_column_sum() {
        let mut result_matrix = DMat::new(3, 1, &[0.0, 0.0, 0.0]);
        let source_matrix = DMat::new(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        result_matrix.set_column_sum(&source_matrix);
        assert_eq!(util::flatten(&result_matrix), &[12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_zeros() {
        let matrix = DMat::zeros(2, 2);
        assert_eq!((matrix.rows(), matrix.cols()), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(matrix.at(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_clip() {
        let mut matrix = DMat::new(2, 2, &[3.0, 4.0, 0.0, 0.0]);
        matrix.clip(5.0);
        let clipped = DMat::new(2, 2, &[3.0, 4.0, 0.0, 0.0]);
        assert_eq!(util::flatten(&matrix), util::flatten(&clipped)); // or a slightly scaled version of this

        let mut matrix = DMat::new(2, 2, &[6.0, 8.0, 0.0, 0.0]);
        matrix.clip(5.0);
        let clipped = DMat::new(2, 2, &[3.0, 4.0, 0.0, 0.0]);
        assert_eq!(util::flatten(&matrix), util::flatten(&clipped)); // or a slightly scaled version of this
    }

    #[test]
    fn test_apply() {
        let mut matrix = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        matrix.apply(|x| x * 2.0);
        assert_eq!(util::flatten(&matrix), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_apply_with_indices() {
        let mut matrix = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        matrix.apply_with_indices(|i, j, v| {
            *v = (i + j) as f32;
        });
        assert_eq!(util::flatten(&matrix), &[0.0, 1.0, 1.0, 2.0]);
    }
    #[test]
    fn test_set_row() {
        let mut matrix = DMat::new(3, 3, &[0.0; 9]);
        matrix.set_row(1, &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.get_row(1), vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_get_row() {
        let matrix = DMat::new(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(matrix.get_row(1), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul_new() {
        let matrix_a = DMat::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let matrix_b = DMat::new(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let result = DMat::mul_new(&matrix_a, &matrix_b);
        assert_eq!(util::flatten(&result), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_dense_matrix() {
        let matrix = DenseMatrix::new(2, 2).data(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
    }
    #[test]
    fn test_dense_matrix_random() {
        let matrix = DenseMatrix::new(2, 2).random().build().unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.data.len(), 4);
    }
    #[test]
    fn test_dense_matrix_validate() {
        let matrix = DenseMatrix::new(2, 2).data(&[1.0, 2.0, 3.0, 4.0]).build();
        assert!(matrix.is_ok());

        let invalid_matrix = DenseMatrix::new(2, 2).data(&[1.0, 2.0]).build();
        assert!(invalid_matrix.is_err());
        if let Err(err) = invalid_matrix {
            assert_eq!(err.to_string(), "Configuration error: Data length:2 does not match matrix dimensions:4");
        }
    }
}

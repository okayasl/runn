use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A structure encapsulating a dense matrix.
#[derive(Clone, Serialize, Deserialize)]
pub struct DenseMatrix {
    data: DMatrix<f32>,
}

impl DenseMatrix {
    /// Creates a new dense matrix with given rows, columns, and data. Always create in row major order.
    pub fn new(rows: usize, cols: usize, data: &[f32]) -> Self {
        Self {
            data: DMatrix::from_row_slice(rows, cols, data),
        }
    }

    /// Creates a new dense matrix with given rows and columns, initialized with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: DMatrix::zeros(rows, cols),
        }
    }

    /// Fills the matrix with zeros.
    pub fn zero(&mut self) {
        self.data.fill(0.0);
    }

    /// Returns a transposed version of the matrix.
    pub fn transpose(&self) -> DenseMatrix {
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
    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.data[(i, j)]
    }

    /// Sets the value at position (i, j).
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f32) {
        self.data[(i, j)] = value;
    }

    /// Adds another matrix to the current matrix.
    #[inline]
    pub fn add(&mut self, other: &Self) {
        self.data += &other.data;
    }

    /// Subtracts another matrix from the current matrix.
    #[inline]
    pub fn sub(&mut self, other: &Self) {
        self.data -= &other.data;
    }

    /// Scales the matrix by a scalar factor.
    #[inline]
    pub fn scale(&mut self, factor: f32) {
        self.data *= factor;
    }

    /// Element-wise multiplication with another matrix.
    #[inline]
    pub fn mul_elem(&mut self, other: &Self) {
        self.data.component_mul_assign(&other.data);
    }

    /// Multiplies the current matrix with another matrix.
    pub fn matrix_multiply_in_place(&mut self, other: &DenseMatrix) {
        self.data = &self.data * &other.data;
    }

    /// Returns a slice of the matrix.
    pub fn slice(&self, i: usize, k: usize, j: usize, l: usize) -> Self {
        let rows = k - i;
        let cols = l - j;
        let view = self.data.view((i, j), (rows, cols));
        Self { data: view.into() }
    }

    /// Sets the values of a specific row.
    pub fn set_row(&mut self, i: usize, src: &[f32]) {
        for (j, &value) in src.iter().enumerate() {
            self.set(i, j, value);
        }
    }

    /// Clips all values to be within [-threshold, threshold]
    pub fn clip(&mut self, threshold: f32) {
        if threshold > 0.0 {
            let norm = self.norm(2.0);
            if norm > threshold {
                let scale = threshold / norm;
                self.scale(scale);
            }
        }
    }

    /// Flattens the matrix into a single vector of elements in row major layout.
    pub fn flatten(&self) -> Vec<f32> {
        self.data
            .row_iter() // Iterate over rows
            .flat_map(|row| row.into_iter().cloned()) // Flatten each row into the resulting vector
            .collect()
    }

    /// Calculates the norm of the matrix (L1 or L2 supported).
    pub fn norm(&self, norm_type: f32) -> f32 {
        if norm_type == 1.0 {
            self.data.iter().map(|x| x.abs()).sum::<f32>()
        } else if norm_type == 2.0 {
            (self.data.iter().map(|x| x * x).sum::<f32>()).sqrt()
        } else {
            panic!("Unsupported norm type");
        }
    }

    /// Sets each element in the first column to the sum of the corresponding row in the other matrix.
    pub fn set_column_sum(&mut self, other: &DenseMatrix) {
        for i in 0..self.rows() {
            let sum = (0..other.cols()).map(|j| other.at(i, j)).sum::<f32>();
            self.set(i, 0, sum);
        }
    }
}

// Example usage in tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_at() {
        let matrix = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.at(1, 1), 4.0);
    }

    #[test]
    fn test_zero() {
        let mut matrix = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        matrix.zero();
        assert_eq!(matrix.flatten(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add() {
        let mut matrix = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let other = DenseMatrix::new(2, 2, &[4.0, 3.0, 2.0, 1.0]);
        matrix.add(&other);
        assert_eq!(matrix.flatten(), &[5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_sub() {
        let mut matrix = DenseMatrix::new(2, 2, &[5.0, 5.0, 5.0, 5.0]);
        let other = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        matrix.sub(&other);
        assert_eq!(matrix.flatten(), &[4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_scale() {
        let mut matrix = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        matrix.scale(2.0);
        assert_eq!(matrix.flatten(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_transpose() {
        let matrix = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = matrix.transpose();
        assert_eq!((transposed.rows(), transposed.cols()), (3, 2));
        assert_eq!(transposed.at(0, 1), 4.0);
    }

    #[test]
    fn test_norm() {
        let matrix = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.norm(1.0), 10.0); // L1 norm
        assert!((matrix.norm(2.0) - 5.477).abs() < 0.001); // L2 norm
    }

    #[test]
    fn test_slice() {
        let matrix = DenseMatrix::new(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let submatrix = matrix.slice(1, 1, 2, 2);
        assert_eq!(submatrix.flatten(), &[5.0, 6.0, 8.0, 9.0]);
    }

    #[test]
    fn test_mul() {
        // Case 1: Square Matrix Multiplication (2x2 * 2x2)
        let mut matrix = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let identity = DenseMatrix::new(2, 2, &[1.0, 0.0, 0.0, 1.0]); // Identity matrix
        matrix.matrix_multiply_in_place(&identity);
        assert_eq!(matrix.flatten(), &[1.0, 2.0, 3.0, 4.0]);

        // Case 2: Rectangular Matrix Multiplication (2x3 * 3x2)
        let mut matrix_a = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_b = DenseMatrix::new(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        matrix_a.matrix_multiply_in_place(&matrix_b);
        assert_eq!(
            matrix_a.flatten(),
            &[22.0, 28.0, 49.0, 64.0] // Result of 2x3 * 3x2 multiplication
        );

        // Case 3: Multiplying with a Zero Matrix (3x2 * 2x3)
        let mut matrix_c = DenseMatrix::new(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let zero_matrix = DenseMatrix::zeros(2, 3);
        matrix_c.matrix_multiply_in_place(&zero_matrix);
        assert_eq!(
            matrix_c.flatten(),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] // Result is a 3x3 zero matrix
        );

        // Case 4: Rectangular Identity Matrix Multiplication (3x3 * 3x3)
        let mut matrix_d = DenseMatrix::new(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let identity_3x3 = DenseMatrix::new(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        matrix_d.matrix_multiply_in_place(&identity_3x3);
        assert_eq!(
            matrix_d.flatten(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] // Result is the same matrix
        );

        // Case 5: Single Row and Single Column Multiplication (1x3 * 3x1)
        let mut row_matrix = DenseMatrix::new(1, 3, &[1.0, 2.0, 3.0]);
        let col_matrix = DenseMatrix::new(3, 1, &[4.0, 5.0, 6.0]);
        row_matrix.matrix_multiply_in_place(&col_matrix);
        assert_eq!(row_matrix.flatten(), &[32.0]); // Dot product result: 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_mul_elem() {
        // Case 1: Element-wise multiplication for square matrices (2x2)
        let mut matrix_a = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let matrix_b = DenseMatrix::new(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        matrix_a.mul_elem(&matrix_b);
        assert_eq!(matrix_a.flatten(), &[5.0, 12.0, 21.0, 32.0]);

        // Case 2: Element-wise multiplication for rectangular matrices (2x3)
        let mut matrix_c = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_d = DenseMatrix::new(2, 3, &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        matrix_c.mul_elem(&matrix_d);
        assert_eq!(matrix_c.flatten(), &[6.0, 10.0, 12.0, 12.0, 10.0, 6.0]);

        // Case 3: Element-wise multiplication with all ones (identity for element-wise mul)
        let mut matrix_e = DenseMatrix::new(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let matrix_ones = DenseMatrix::new(3, 3, &[1.0; 9]); // 3x3 matrix of all ones
        matrix_e.mul_elem(&matrix_ones);
        assert_eq!(
            matrix_e.flatten(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );

        // Case 4: Element-wise multiplication with a zero matrix (result should be zero matrix)
        let mut matrix_f = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let zero_matrix = DenseMatrix::zeros(2, 2);
        matrix_f.mul_elem(&zero_matrix);
        assert_eq!(matrix_f.flatten(), &[0.0, 0.0, 0.0, 0.0]);

        // Case 5: Element-wise multiplication for single-row matrices (1x4)
        let mut matrix_g = DenseMatrix::new(1, 4, &[1.0, 2.0, 3.0, 4.0]);
        let matrix_h = DenseMatrix::new(1, 4, &[4.0, 3.0, 2.0, 1.0]);
        matrix_g.mul_elem(&matrix_h);
        assert_eq!(matrix_g.flatten(), &[4.0, 6.0, 6.0, 4.0]);

        // Case 6: Element-wise multiplication for single-column matrices (4x1)
        let mut matrix_i = DenseMatrix::new(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let matrix_j = DenseMatrix::new(4, 1, &[4.0, 3.0, 2.0, 1.0]);
        matrix_i.mul_elem(&matrix_j);
        assert_eq!(matrix_i.flatten(), &[4.0, 6.0, 6.0, 4.0]);
    }

    #[test]
    fn test_flatten() {
        let matrix = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(matrix.flatten(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_clone() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut matrix = DenseMatrix::new(2, 2, &data);
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
        // Create a test matrix for setting column sums
        let mut result_matrix = DenseMatrix::new(3, 1, &[0.0, 0.0, 0.0]);

        // Source matrix to sum columns from
        let source_matrix = DenseMatrix::new(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Apply the method
        result_matrix.set_column_sum(&source_matrix);

        // Expected results after summing columns
        assert_eq!(result_matrix.flatten(), &[12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_zeros() {
        let matrix = DenseMatrix::zeros(2, 2);
        assert_eq!((matrix.rows(), matrix.cols()), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(matrix.at(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_clip() {
        let mut matrix = DenseMatrix::new(2, 2, &vec![3.0, 4.0, 0.0, 0.0]);
        matrix.clip(5.0);
        assert_eq!(matrix.data.as_slice(), &[3.0, 4.0, 0.0, 0.0]);

        let mut matrix = DenseMatrix::new(2, 2, &vec![6.0, 8.0, 0.0, 0.0]);
        matrix.clip(5.0);
        assert_eq!(matrix.data.as_slice(), &[3.0, 4.0, 0.0, 0.0]);
    }
}

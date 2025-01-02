use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, MulAssign, SubAssign};

use nalgebra::DMatrix;
use rand_distr::num_traits::{Float, One, Zero};
use serde::{Deserialize, Serialize};

/// A structure encapsulating a dense matrix with generic floating point type.
#[derive(Clone, Serialize, Deserialize)]
pub struct DenseMatrix<T>
where
    T: Float + Debug + 'static,
{
    data: DMatrix<T>,
}

impl<T> DenseMatrix<T>
where
    T: Float + AddAssign + SubAssign + MulAssign + Zero + One + Send + Sync + Debug + Sum + 'static,
    DMatrix<T>: Clone,
{
    /// Creates a new dense matrix with given rows, columns, and data.
    pub fn new(rows: usize, cols: usize, data: &[T]) -> Self {
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
        self.data.fill(T::zero());
    }

    /// Returns a transposed version of the matrix.
    pub fn transpose(&self) -> DenseMatrix<T> {
        DenseMatrix {
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
    pub fn at(&self, i: usize, j: usize) -> T {
        self.data[(i, j)]
    }

    /// Sets the value at position (i, j).
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: T) {
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
    pub fn scale(&mut self, factor: T) {
        self.data *= factor;
    }

    /// Element-wise multiplication with another matrix.
    #[inline]
    pub fn mul_elem(&mut self, other: &Self) {
        self.data.component_mul_assign(&other.data);
    }

    /// Multiplies the current matrix with another matrix.
    pub fn matrix_multiply_in_place(&mut self, other: &DenseMatrix<T>) {
        self.data = &self.data * &other.data;
    }

    /// Returns a slice of the matrix.
    pub fn slice(&self, i: usize, k: usize, j: usize, l: usize) -> Self {
        let rows = k - i;
        let cols = l - j;
        let view = self.data.view((i, j), (rows, cols));
        Self {
            data: view.into_owned(),
        }
    }

    /// Sets the values of a specific row.
    pub fn set_row(&mut self, i: usize, src: &[T]) {
        for (j, &value) in src.iter().enumerate() {
            self.set(i, j, value);
        }
    }

    /// Clips all values to be within [-threshold, threshold]
    pub fn clip(&mut self, threshold: T) {
        self.data.iter_mut().for_each(|x| {
            if *x > threshold {
                *x = threshold;
            } else if *x < -threshold {
                *x = -threshold;
            }
        });
    }

    /// Flattens the matrix into a single vector of elements in row major layout.
    pub fn flatten(&self) -> Vec<T> {
        self.data
            .row_iter() // Iterate over rows
            .flat_map(|row| row.into_iter().cloned()) // Flatten each row into the resulting vector
            .collect()
    }

    /// Calculates the norm of the matrix (L1 or L2 supported).
    pub fn norm(&self, norm: T) -> T {
        if norm == T::one() {
            // L1 norm
            self.data.iter().map(|x| x.abs()).sum()
        } else if norm + norm == T::one() + T::one() {
            // L2 norm
            let sum_squares = self.data.iter().map(|x| (*x) * (*x)).sum::<T>();
            sum_squares.sqrt()
        } else {
            panic!("Norm type not implemented")
        }
    }

    /// Sets each element in the first column to the sum of the corresponding row in the other matrix.
    pub fn set_column_sum(&mut self, other: &DenseMatrix<T>) {
        for i in 0..self.rows() {
            let sum = other.data.row(i).iter().copied().sum();
            self.set(i, 0, sum);
        }
    }
}

// Type aliases for common float types
pub type DenseMatrix32 = DenseMatrix<f32>;
pub type DenseMatrix64 = DenseMatrix<f64>;

// Implement conversion between different float types
impl<T> DenseMatrix<T>
where
    T: Float + AddAssign + SubAssign + MulAssign + Zero + One + Send + Sync + Debug + Sum + 'static,
    DMatrix<T>: Clone,
{
    pub fn convert<U>(&self) -> DenseMatrix<U>
    where
        U: Float
            + AddAssign
            + SubAssign
            + MulAssign
            + Zero
            + One
            + Send
            + Sync
            + Debug
            + Sum
            + 'static,
        DMatrix<U>: Clone,
        T: Into<U>,
    {
        let mut new_data = Vec::with_capacity(self.rows() * self.cols());
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                new_data.push(self.at(i, j).into());
            }
        }
        DenseMatrix::new(self.rows(), self.cols(), &new_data)
    }
}

// Example usage in tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_different_float_types() {
        // f32 matrix
        let matrix32 = DenseMatrix32::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix32.at(0, 0), 1.0f32);

        // f64 matrix
        let matrix64 = DenseMatrix64::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix64.at(0, 0), 1.0f64);

        // Convert from f32 to f64
        let converted = matrix32.convert::<f64>();
        assert_eq!(converted.at(0, 0), 1.0f64);
    }

    #[test]
    fn test_at() {
        let matrix = DenseMatrix::<f32>::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.at(1, 1), 4.0);
    }

    #[test]
    fn test_zero() {
        let mut matrix = DenseMatrix::<f64>::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
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
        let mut matrix = DenseMatrix::<f32>::new(2, 2, &[5.0, 5.0, 5.0, 5.0]);
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

    // #[test]
    // fn test_apply() {
    //     let mut matrix = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    //     matrix.apply(|x| x * 2.0);
    //     assert_eq!(matrix.flatten(), &[2.0, 4.0, 6.0, 8.0]);
    // }

    // #[test]
    // fn test_max_in_row() {
    //     let matrix = DenseMatrix::new(2, 3, &[1.0, 3.0, 2.0, 4.0, 5.0, 6.0]);
    //     assert_eq!(matrix.max_in_row(0), Some(3.0));
    //     assert_eq!(matrix.max_in_row(1), Some(6.0));
    // }

    // #[test]
    // fn test_max_value_index_in_row() {
    //     let matrix = DenseMatrix::new(2, 3, &[1.0, 3.0, 2.0, 4.0, 5.0, 6.0]);
    //     assert_eq!(matrix.max_value_index_in_row(0), Some(1));
    //     assert_eq!(matrix.max_value_index_in_row(1), Some(2));
    // }

    // #[test]
    // fn test_sum_column() {
    //     let matrix = DenseMatrix::new(3, 2, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    //     assert_eq!(matrix.sum_column(0), 6.0); // Sum of column 0
    //     assert_eq!(matrix.sum_column(1), 15.0); // Sum of column 1
    // }

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

    // #[test]
    // fn test_normalize_success() {
    //     let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    //     let matrix = DenseMatrix::new(2, 3, &data);
    //     let mins = vec![1.0, 2.0, 3.0];
    //     let maxs = vec![4.0, 5.0, 6.0];
    //     let normalized = matrix.normalize(&mins, &maxs).unwrap();
    //     assert_eq!(normalized.at(0, 0), 0.0);
    //     assert_eq!(normalized.at(1, 2), 1.0);
    // }

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

    // #[test]
    // fn test_equal_approx() {
    //     let data1 = vec![1.0, 2.0, 3.0, 4.0];
    //     let data2 = vec![1.001, 2.001, 2.999, 4.0];
    //     let matrix1 = DenseMatrix::new(2, 2, &data1);
    //     let matrix2 = DenseMatrix::new(2, 2, &data2);
    //     assert!(matrix1.equal_approx(&matrix2, 0.01));
    //     assert!(!matrix1.equal_approx(&matrix2, 0.0001));
    // }

    #[test]
    fn test_zeros() {
        let matrix = DenseMatrix::<f32>::zeros(2, 2);
        assert_eq!((matrix.rows(), matrix.cols()), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(matrix.at(i, j), 0.0);
            }
        }
    }

    // #[test]
    // fn test_calculate_accuracy_against_by_max_in_row() {
    //     let targets = DenseMatrix::new(2, 2, &vec![0.0, 1.0, 1.0, 0.0]);
    //     let predictions = DenseMatrix::new(2, 2, &vec![0.0, 1.0, 0.0, 1.0]);
    //     let accuracy = targets.calculate_accuracy_against_by_max_in_row(&predictions);
    //     assert_eq!(accuracy, 0.5);
    // }

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

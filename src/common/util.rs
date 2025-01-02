use matrix::DenseMatrix;
use rand_distr::num_traits::{Float, One, Zero};

use std::{
    fmt::Debug,
    iter::Sum,
    ops::{AddAssign, MulAssign, SubAssign},
};

use super::matrix;

/// Find minimum and maximum values for each column
pub fn find_min_max<T>(matrix: &DenseMatrix<T>) -> (Vec<T>, Vec<T>)
where
    T: Float + AddAssign + SubAssign + MulAssign + Zero + One + Send + Sync + Debug + Sum + 'static,
{
    let (rows, cols) = (matrix.rows(), matrix.cols());
    let mut mins = vec![T::infinity(); cols];
    let mut maxs = vec![T::neg_infinity(); cols];

    for j in 0..cols {
        for i in 0..rows {
            let value = matrix.at(i, j);
            mins[j] = mins[j].min(value);
            maxs[j] = maxs[j].max(value);
        }
    }
    (mins, maxs)
}

/// Normalize matrix values to range [0, 1] using min-max normalization
pub fn normalize<T>(matrix: &DenseMatrix<T>, mins: &[T], maxs: &[T]) -> Option<DenseMatrix<T>>
where
    T: Float
        + AddAssign
        + SubAssign
        + MulAssign
        + Zero
        + One
        + Send
        + Sync
        + Clone
        + Debug
        + Sum
        + 'static,
{
    let (rows, cols) = (matrix.rows(), matrix.cols());
    if mins.len() != cols || maxs.len() != cols {
        return None;
    }

    let mut normalized = DenseMatrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let normalized_value = (matrix.at(i, j) - mins[j]) / (maxs[j] - mins[j]);
            normalized.set(i, j, normalized_value);
        }
    }
    Some(normalized)
}

/// Calculate accuracy by comparing max values in each row
pub fn calculate_accuracy<T>(predictions: &DenseMatrix<T>, targets: &DenseMatrix<T>) -> T
where
    T: Float
        + AddAssign
        + SubAssign
        + MulAssign
        + Zero
        + One
        + Send
        + Sync
        + Clone
        + Debug
        + Sum
        + 'static,
{
    let rows = predictions.rows();
    let mut correct = T::zero();

    for i in 0..rows {
        if find_max_index_in_row(predictions, i) == find_max_index_in_row(targets, i) {
            correct = correct + T::one();
        }
    }
    correct / T::from(rows).unwrap()
}

/// Helper function to find index of maximum value in a row
fn find_max_index_in_row<T>(matrix: &DenseMatrix<T>, row: usize) -> usize
where
    T: Float
        + AddAssign
        + SubAssign
        + MulAssign
        + Zero
        + One
        + Send
        + Sync
        + Clone
        + Debug
        + Sum
        + 'static,
{
    let cols = matrix.cols();
    let mut max_idx = 0;
    let mut max_val = matrix.at(row, 0);

    for j in 1..cols {
        let val = matrix.at(row, j);
        if val > max_val {
            max_val = val;
            max_idx = j;
        }
    }
    max_idx
}

/// Check if two matrices are approximately equal within a tolerance
pub fn equal_approx<T>(a: &DenseMatrix<T>, b: &DenseMatrix<T>, tolerance: T) -> bool
where
    T: Float
        + AddAssign
        + SubAssign
        + MulAssign
        + Zero
        + One
        + Send
        + Sync
        + Clone
        + Debug
        + Sum
        + 'static,
{
    if (a.rows(), a.cols()) != (b.rows(), b.cols()) {
        return false;
    }
    let (rows, cols) = (a.rows(), a.cols());
    for i in 0..rows {
        for j in 0..cols {
            if (a.at(i, j) - b.at(i, j)).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

/// Flatten matrix into a vector in row-major order
pub fn flatten<T>(matrix: &DenseMatrix<T>) -> Vec<T>
where
    T: Float
        + AddAssign
        + SubAssign
        + MulAssign
        + Zero
        + One
        + Send
        + Sync
        + Clone
        + Debug
        + Sum
        + 'static,
{
    let (rows, cols) = (matrix.rows(), matrix.cols());
    let mut result = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            result.push(matrix.at(i, j));
        }
    }
    result
}

/// Apply a function to each element of the matrix
pub fn apply<T, F>(matrix: &mut DenseMatrix<T>, f: F)
where
    T: Float
        + AddAssign
        + SubAssign
        + MulAssign
        + Zero
        + One
        + Send
        + Sync
        + Clone
        + Debug
        + Sum
        + 'static,
    F: Fn(T) -> T,
{
    let (rows, cols) = (matrix.rows(), matrix.cols());
    for i in 0..rows {
        for j in 0..cols {
            let val = matrix.at(i, j);
            let new_val = f(val);
            matrix.set(i, j, new_val);
        }
    }
}

/// Apply a function with indices to each element of the matrix
pub fn apply_with_indices<T, F>(matrix: &mut DenseMatrix<T>, mut f: F)
where
    T: Float
        + AddAssign
        + SubAssign
        + MulAssign
        + Zero
        + One
        + Send
        + Sync
        + Clone
        + Debug
        + Sum
        + 'static,
    F: FnMut(usize, usize, T) -> T,
{
    let (rows, cols) = (matrix.rows(), matrix.cols());
    for i in 0..rows {
        for j in 0..cols {
            let new_val = f(i, j, matrix.at(i, j));
            matrix.set(i, j, new_val);
        }
    }
}

// Tests for utility functions
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matrix = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let (mins, maxs) = find_min_max(&matrix);
        let normalized = normalize(&matrix, &mins, &maxs).unwrap();
        let expected = DenseMatrix::new(2, 2, &[0.0, 0.0, 1.0, 1.0]);

        assert!(equal_approx(&normalized, &expected, 1e-6));

        let input1 = DenseMatrix::new(4, 2, &[0.0, -10.0, 5.0, -7.0, 7.0, -5.0, 10.0, 0.0]);
        let mins1 = vec![0.0, -10.0];
        let maxs1 = vec![10.0, 0.0];
        let expected1 = DenseMatrix::new(4, 2, &[0.0, 0.0, 0.5, 0.3, 0.7, 0.5, 1.0, 1.0]);
        let result1 = normalize(&input1, &mins1, &maxs1).unwrap();
        assert!(equal_approx(&result1, &expected1, 1e-6));

        // Test case with zero range in a column
        let input2 = DenseMatrix::new(3, 2, &[5.0, 1.0, 5.0, 2.0, 5.0, 3.0]);
        let mins2 = vec![5.0, 1.0];
        let maxs2 = vec![5.0, 3.0];
        let result2 = normalize(&input2, &mins2, &maxs2).unwrap();
        let expected2 = DenseMatrix::new(3, 2, &[0.0, 0.0, 0.0, 0.5, 0.0, 1.0]);
        assert!(equal_approx(&result2, &expected2, 1e-6));

        // Test case with negative numbers
        let input3 = DenseMatrix::new(2, 2, &[-10.0, -5.0, -2.0, -1.0]);
        let mins3 = vec![-10.0, -5.0];
        let maxs3 = vec![-2.0, -1.0];
        let result3 = normalize(&input3, &mins3, &maxs3).unwrap();
        let expected3 = DenseMatrix::new(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        assert!(equal_approx(&result3, &expected3, 1e-6));
    }

    #[test]
    fn test_calculate_accuracy() {
        let predictions = DenseMatrix::new(2, 2, &[0.9f32, 0.1, 0.2, 0.8]);
        let targets = DenseMatrix::new(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let accuracy = calculate_accuracy(&predictions, &targets);
        assert!((accuracy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply() {
        let mut matrix = DenseMatrix::new(2, 2, &[1.0f32, 2.0, 3.0, 4.0]);
        apply(&mut matrix, |x| x * 2.0);
        assert!(equal_approx(
            &matrix,
            &DenseMatrix::new(2, 2, &[2.0, 4.0, 6.0, 8.0]),
            1e-6
        ));
    }

    #[test]
    fn test_apply_with_indices() {
        let mut matrix = DenseMatrix::new(2, 2, &[1.0f32, 2.0, 3.0, 4.0]);
        apply_with_indices(&mut matrix, |i, j, x| x + (i + j) as f32);
        assert!(equal_approx(
            &matrix,
            &DenseMatrix::new(2, 2, &[1.0, 3.0, 4.0, 6.0]),
            1e-6
        ));
    }
}

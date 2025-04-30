use matrix::DenseMatrix;

use super::matrix;

/// Helper function to find index of maximum value in a row
pub(crate) fn find_max_index_in_row(matrix: &DenseMatrix, row: usize) -> usize {
    let cols = matrix.cols();
    if cols == 0 {
        return 0;
    }

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
#[cfg(test)]
pub(crate) fn equal_approx(a: &DenseMatrix, b: &DenseMatrix, tolerance: f32) -> bool {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return false;
    }

    let rows = a.rows();
    let cols = a.cols();

    for i in 0..rows {
        for j in 0..cols {
            let diff = (a.at(i, j) - b.at(i, j)).abs();
            if diff > tolerance {
                return false;
            }
        }
    }

    true
}

/// Flatten matrix into a vector in row-major order
pub(crate) fn flatten(matrix: &DenseMatrix) -> Vec<f32> {
    let (rows, cols) = (matrix.rows(), matrix.cols());
    let mut result = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        for j in 0..cols {
            result.push(matrix.at(i, j));
        }
    }

    result
}

// Tests for utility functions
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten() {
        let matrix = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(flatten(&matrix), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

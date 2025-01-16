use matrix::DenseMatrix;

use super::matrix;

/// Find minimum and maximum values for each column
pub(crate) fn find_min_max(matrix: &DenseMatrix) -> (Vec<f32>, Vec<f32>) {
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

    (mins, maxs)
}

/// Normalize matrix values to range [0, 1] using min-max normalization
pub(crate) fn normalize(matrix: &DenseMatrix, mins: &[f32], maxs: &[f32]) -> Option<DenseMatrix> {
    let (rows, cols) = (matrix.rows(), matrix.cols());

    // Check if dimensions match
    if mins.len() != cols || maxs.len() != cols {
        return None;
    }

    let mut normalized = DenseMatrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let val = matrix.at(i, j);
            let min = mins[j];
            let max = maxs[j];

            if (max - min).abs() < f32::EPSILON {
                normalized.set(i, j, 0.0);
            } else {
                normalized.set(i, j, (val - min) / (max - min));
            }
        }
    }

    Some(normalized)
}

/// Calculate accuracy by comparing max values in each row
pub(crate) fn calculate_accuracy(predictions: &DenseMatrix, targets: &DenseMatrix) -> f32 {
    let rows = predictions.rows();
    if rows == 0 || predictions.rows() != targets.rows() || predictions.cols() != targets.cols() {
        return 0.0;
    }

    let mut correct = 0;
    for i in 0..rows {
        let pred_max_idx = find_max_index_in_row(predictions, i);
        let target_max_idx = find_max_index_in_row(targets, i);
        if pred_max_idx == target_max_idx {
            correct += 1;
        }
    }

    correct as f32 / rows as f32
}

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

pub fn format_matrix(matrix: &DenseMatrix) -> String {
    let mut result = String::new();

    for i in 0..matrix.rows() {
        let left_border = if i == 0 {
            "⎡"
        } else if i == matrix.rows() - 1 {
            "⎣"
        } else {
            "⎢"
        };

        let right_border = if i == 0 {
            "⎤"
        } else if i == matrix.rows() - 1 {
            "⎦"
        } else {
            "⎥"
        };

        let row: Vec<String> = (0..matrix.cols())
            .map(|j| format!("{:7.6}", matrix.at(i, j)))
            .collect();

        result.push_str(&format!(
            "{} {} {}\n",
            left_border,
            row.join(" "),
            right_border
        ));
    }
    result
}

/// Apply a function to each element of the matrix
// pub(crate) fn apply<F>(matrix: &mut DenseMatrix, mut f: F)
// where
//     F: FnMut(f32) -> f32,
// {
//     let (rows, cols) = (matrix.rows(), matrix.cols());
//     for i in 0..rows {
//         for j in 0..cols {
//             let val = matrix.at(i, j);
//             matrix.set(i, j, f(val));
//         }
//     }
// }

// /// Apply a function with indices to each element of the matrix
// pub(crate) fn apply_with_indices<F>(matrix: &mut DenseMatrix, mut f: F)
// where
//     F: FnMut(usize, usize, f32) -> f32,
// {
//     let (rows, cols) = (matrix.rows(), matrix.cols());
//     for i in 0..rows {
//         for j in 0..cols {
//             let val = matrix.at(i, j);
//             matrix.set(i, j, f(i, j, val));
//         }
//     }
// }

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

    // #[test]
    // fn test_apply() {
    //     let mut matrix = DenseMatrix::new(2, 2, &[1.0f32, 2.0, 3.0, 4.0]);
    //     apply(&mut matrix, |x| x * 2.0);
    //     assert!(equal_approx(
    //         &matrix,
    //         &DenseMatrix::new(2, 2, &[2.0, 4.0, 6.0, 8.0]),
    //         1e-6
    //     ));
    // }

    // #[test]
    // fn test_apply_with_indices() {
    //     let mut matrix = DenseMatrix::new(2, 2, &[1.0f32, 2.0, 3.0, 4.0]);
    //     apply_with_indices(&mut matrix, |i, j, x| x + (i + j) as f32);
    //     assert!(equal_approx(
    //         &matrix,
    //         &DenseMatrix::new(2, 2, &[1.0, 3.0, 4.0, 6.0]),
    //         1e-6
    //     ));
    // }
}

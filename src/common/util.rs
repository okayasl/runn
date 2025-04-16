use log::info;
use matrix::DenseMatrix;
use std::fmt::Write as FmtWrite;

use crate::util;

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

/// Normalize matrix values to range [0, 1] in-place using min-max normalization
pub(crate) fn normalize_in_place(matrix: &mut DenseMatrix, mins: &[f32], maxs: &[f32]) {
    let (rows, cols) = (matrix.rows(), matrix.cols());

    // Check if dimensions match
    if mins.len() != cols || maxs.len() != cols {
        return;
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

pub fn print_metrics(
    accuracy: f32,
    loss: f32,
    macro_f1: f32,
    micro_f1: f32,
    micro_recall: f32,
    micro_precision: f32,
) {
    info!(
        "Accuracy: {:.2}%, Loss: {:.5}, MacroF1: {:.4}, MicroF1: {:.4}, MicroRecall: {:.4}, MicroPrecision: {:.4}",
        accuracy * 100.0,
        loss,
        macro_f1,
        micro_f1,
        micro_recall,
        micro_precision
    );
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

pub fn print_matrices_comparisons(
    input: &DenseMatrix,
    target: &DenseMatrix,
    prediction: &DenseMatrix,
) {
    let r = input.rows();
    let tc = target.cols();
    let pc = prediction.cols();

    let mut buf = String::new();

    writeln!(buf, "\nInput    Target    Prediction").unwrap();

    if r == 0 {
        info!("{}", buf);
        return;
    }

    if r == 1 {
        // Prepare the top border
        let mut input_str = String::new();
        let mut target_str = String::new();
        let mut prediction_str = String::new();

        input_str.push('[');
        for j in 0..input.cols() {
            write!(input_str, "{:.2} ", input.at(0, j)).unwrap();
        }
        input_str.push(']');

        target_str.push_str("  [");
        for j in 0..tc {
            write!(target_str, "{:.2} ", target.at(0, j)).unwrap();
        }
        target_str.push(']');

        if util::find_max_index_in_row(target, 0) != util::find_max_index_in_row(prediction, 0) {
            target_str.push_str(" <>");
        } else {
            target_str.push_str("   ");
        }

        prediction_str.push_str(" [");
        for j in 0..pc {
            write!(prediction_str, "{:.2} ", prediction.at(0, j)).unwrap();
        }
        prediction_str.push(']');

        // Append the line
        writeln!(buf, "{}{}{}", input_str, target_str, prediction_str).unwrap();
        info!("{}", buf);
        return;
    }

    // Prepare the top border
    let mut input_str = String::new();
    let mut target_str = String::new();
    let mut prediction_str = String::new();

    input_str.push('⎡');
    for j in 0..input.cols() {
        write!(input_str, "{:.2} ", input.at(0, j)).unwrap();
    }
    input_str.push('⎤');

    target_str.push_str("  ⎡");
    for j in 0..tc {
        write!(target_str, "{:.2} ", target.at(0, j)).unwrap();
    }
    target_str.push('⎤');

    if util::find_max_index_in_row(target, 0) != util::find_max_index_in_row(prediction, 0) {
        target_str.push_str(" <>");
    } else {
        target_str.push_str("   ");
    }

    prediction_str.push_str(" ⎡");
    for j in 0..pc {
        write!(prediction_str, "{:.2} ", prediction.at(0, j)).unwrap();
    }
    prediction_str.push('⎡');

    // Append the top border
    writeln!(buf, "{}{}{}", input_str, target_str, prediction_str).unwrap();

    // Append the middle rows
    for i in 1..r - 1 {
        input_str.clear();
        target_str.clear();
        prediction_str.clear();

        input_str.push('⎢');
        for j in 0..input.cols() {
            write!(input_str, "{:.2} ", input.at(i, j)).unwrap();
        }
        input_str.push('⎥');

        target_str.push_str("  ⎢");
        for j in 0..tc {
            write!(target_str, "{:.2} ", target.at(i, j)).unwrap();
        }
        target_str.push('⎥');

        if util::find_max_index_in_row(target, i) != util::find_max_index_in_row(prediction, i) {
            target_str.push_str(" <>");
        } else {
            target_str.push_str("   ");
        }

        prediction_str.push_str(" ⎢");
        for j in 0..pc {
            write!(prediction_str, "{:.2} ", prediction.at(i, j)).unwrap();
        }
        prediction_str.push('⎢');

        // Append the line
        writeln!(buf, "{}{}{}", input_str, target_str, prediction_str).unwrap();
    }

    // Prepare the bottom border
    input_str.clear();
    target_str.clear();
    prediction_str.clear();

    input_str.push('⎣');
    for j in 0..input.cols() {
        write!(input_str, "{:.2} ", input.at(r - 1, j)).unwrap();
    }
    input_str.push('⎦');

    target_str.push_str("  ⎣");
    for j in 0..tc {
        write!(target_str, "{:.2} ", target.at(r - 1, j)).unwrap();
    }
    target_str.push('⎦');

    if util::find_max_index_in_row(target, r - 1) != util::find_max_index_in_row(prediction, r - 1)
    {
        target_str.push_str(" <>");
    } else {
        target_str.push_str("   ");
    }

    prediction_str.push_str(" ⎣");
    for j in 0..pc {
        write!(prediction_str, "{:.2} ", prediction.at(r - 1, j)).unwrap();
    }
    prediction_str.push('⎣');

    // Append the bottom border
    writeln!(buf, "{}{}{}", input_str, target_str, prediction_str).unwrap();

    // Log the entire output
    info!("{}", buf);
}

// Tests for utility functions
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_in_place() {
        // Test case 1: Normal range
        let mut matrix = DenseMatrix::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let (mins, maxs) = find_min_max(&matrix);
        normalize_in_place(&mut matrix, &mins, &maxs);
        let expected = DenseMatrix::new(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        assert!(equal_approx(&matrix, &expected, 1e-6));

        // Test case 2: Custom range
        let mut input1 = DenseMatrix::new(4, 2, &[0.0, -10.0, 5.0, -7.0, 7.0, -5.0, 10.0, 0.0]);
        let mins1 = vec![0.0, -10.0];
        let maxs1 = vec![10.0, 0.0];
        let expected1 = DenseMatrix::new(4, 2, &[0.0, 0.0, 0.5, 0.3, 0.7, 0.5, 1.0, 1.0]);
        normalize_in_place(&mut input1, &mins1, &maxs1);
        assert!(equal_approx(&input1, &expected1, 1e-6));

        // Test case 3: Zero range in a column
        let mut input2 = DenseMatrix::new(3, 2, &[5.0, 1.0, 5.0, 2.0, 5.0, 3.0]);
        let mins2 = vec![5.0, 1.0];
        let maxs2 = vec![5.0, 3.0];
        let expected2 = DenseMatrix::new(3, 2, &[0.0, 0.0, 0.0, 0.5, 0.0, 1.0]);
        normalize_in_place(&mut input2, &mins2, &maxs2);
        assert!(equal_approx(&input2, &expected2, 1e-6));

        // Test case 4: Negative numbers
        let mut input3 = DenseMatrix::new(2, 2, &[-10.0, -5.0, -2.0, -1.0]);
        let mins3 = vec![-10.0, -5.0];
        let maxs3 = vec![-2.0, -1.0];
        let expected3 = DenseMatrix::new(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        normalize_in_place(&mut input3, &mins3, &maxs3);
        assert!(equal_approx(&input3, &expected3, 1e-6));
    }

    #[test]
    fn test_calculate_accuracy() {
        let predictions = DenseMatrix::new(2, 2, &[0.9f32, 0.1, 0.2, 0.8]);
        let targets = DenseMatrix::new(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let accuracy = calculate_accuracy(&predictions, &targets);
        assert!((accuracy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_flatten() {
        let matrix = DenseMatrix::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(flatten(&matrix), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

use std::collections::HashMap;

use super::{matrix::DenseMatrix, util};

pub fn calculate_precision(true_positives: usize, false_positives: usize) -> f32 {
    if true_positives + false_positives == 0 {
        return 0.0;
    }
    true_positives as f32 / (true_positives + false_positives) as f32
}

pub fn calculate_recall(true_positives: usize, false_negatives: usize) -> f32 {
    if true_positives + false_negatives == 0 {
        return 0.0;
    }
    true_positives as f32 / (true_positives + false_negatives) as f32
}

pub fn calculate_f1_score(precision: f32, recall: f32) -> f32 {
    if precision + recall == 0.0 {
        return 0.0;
    }
    2.0 * (precision * recall) / (precision + recall)
}

pub fn calculate_accuracy_by_max_in_row(targets: &DenseMatrix, predictions: &DenseMatrix) -> f32 {
    let mut correct = 0;
    let rows = targets.rows();

    for i in 0..rows {
        let true_idx = util::find_max_index_in_row(targets, i);
        let pred_idx = util::find_max_index_in_row(predictions, i);
        if true_idx == pred_idx {
            correct += 1;
        }
    }
    correct as f32 / rows as f32
}

pub fn calculate_accuracy_per_element(targets: &DenseMatrix, predictions: &DenseMatrix) -> f32 {
    let mut correct = 0;
    let rows = targets.rows();

    for i in 0..rows {
        let true_label = targets.at(i, 0);
        let pred_label = predictions.at(i, 0);
        if true_label == pred_label {
            correct += 1;
        }
    }
    correct as f32 / rows as f32
}

pub fn calculate_confusion_matrix(
    targets: &DenseMatrix,
    predictions: &DenseMatrix,
) -> (
    HashMap<usize, usize>,
    HashMap<usize, usize>,
    HashMap<usize, usize>,
) {
    let mut true_positives: HashMap<usize, usize> = HashMap::new();
    let mut false_positives: HashMap<usize, usize> = HashMap::new();
    let mut false_negatives: HashMap<usize, usize> = HashMap::new();

    let rows = targets.rows();
    let cols = targets.cols();

    for i in 0..rows {
        let true_idx = util::find_max_index_in_row(targets, i);
        let pred_idx = util::find_max_index_in_row(predictions, i);

        if true_idx == pred_idx {
            *true_positives.entry(true_idx).or_insert(0) += 1; // Correctly predicted class
        } else {
            *false_positives.entry(pred_idx).or_insert(0) += 1; // Incorrectly predicted as another class
            *false_negatives.entry(true_idx).or_insert(0) += 1; // Failed to predict the actual class
        }
    }

    // For classes not predicted at all or not having any positives, set to 0
    for c in 0..cols {
        true_positives.entry(c).or_insert(0);
        false_positives.entry(c).or_insert(0);
        false_negatives.entry(c).or_insert(0);
    }

    (true_positives, false_positives, false_negatives)
}

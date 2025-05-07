use std::collections::HashMap;

use crate::{matrix::DMat, util};

use super::{MetricEvaluator, MetricResult};

pub struct ClassificationMetrics {
    pub accuracy: f32,
    pub micro_precision: f32,
    pub micro_recall: f32,
    pub macro_f1_score: f32,
    pub micro_f1_score: f32,
    pub metrics_by_class: Vec<Metric>,
}

pub struct Metric {
    pub f1_score: f32,
    pub recall: f32,
    pub precision: f32,
}

impl ClassificationMetrics {
    pub fn display(&self) -> String {
        format!(
            "Classification Metrics: Accuracy:{:.4}, Micro Precision:{:.4}, Micro Recall:{:.4}, Macro F1 Score:{:.4}, Micro F1 Score:{:.4}\n  Metrics by Class:\n{}",
            self.accuracy * 100.0,
            self.micro_precision,
            self.micro_recall,
            self.macro_f1_score,
            self.micro_f1_score,
            self.metrics_by_class
                .iter()
                .enumerate()
                .map(|(i, metric)| {
                    format!(
                        "    Class {}:    Precision:{:.4}    Recall:{:.4}    F1 Score:{:.4}\n",
                        i, metric.precision, metric.recall, metric.f1_score
                    )
                })
                .collect::<String>()
        )
    }

    pub(crate) fn headers(&self) -> Vec<&'static str> {
        vec!["Accuracy"]
    }

    pub(crate) fn values_str(&self) -> Vec<String> {
        vec![format!("{:.5}", self.accuracy * 100.0)]
    }

    pub(crate) fn values(&self) -> Vec<f32> {
        vec![self.accuracy * 100.0]
    }
}

pub(crate) struct ClassificationEvaluator;

impl MetricEvaluator for ClassificationEvaluator {
    fn evaluate(&self, targets: &DMat, predictions: &DMat) -> MetricResult {
        let classification_metrics = calculate_classification_metrics(targets, predictions);
        MetricResult::Classification(classification_metrics)
    }
}

fn calculate_classification_metrics(targets: &DMat, predictions: &DMat) -> ClassificationMetrics {
    let (true_positives_map, false_positives_map, false_negatives_map) =
        calculate_confusion_matrix(targets, predictions);

    // Initialize sums for micro-average calculations
    let mut sum_tp = 0;
    let mut sum_fp = 0;
    let mut sum_fn = 0;

    // Initialize sums for macro-average calculations
    let mut sum_f1_macro = 0.0;

    // Calculate metrics for each class
    let num_classes = true_positives_map.len();
    let mut metrics_by_class = Vec::with_capacity(num_classes);

    for class in 0..num_classes {
        let tp = *true_positives_map.get(&class).unwrap_or(&0);
        let f_pos = *false_positives_map.get(&class).unwrap_or(&0);
        let f_neg = *false_negatives_map.get(&class).unwrap_or(&0);

        // Update sums for micro-average
        sum_tp += tp;
        sum_fp += f_pos;
        sum_fn += f_neg;

        let precision = calculate_precision(tp, f_pos);
        let recall = calculate_recall(tp, f_neg);
        let f1_score = calculate_f1_score(precision, recall);

        let metric = Metric {
            precision,
            recall,
            f1_score,
        };
        metrics_by_class.push(metric);

        sum_f1_macro += f1_score;
    }
    let accuracy = calculate_accuracy(targets, predictions);

    // Calculate macro-average F1 score
    let macro_f1 = sum_f1_macro / num_classes as f32;

    // Calculate micro-average F1 score
    let micro_precision = calculate_precision(sum_tp, sum_fp);
    let micro_recall = calculate_recall(sum_tp, sum_fn);
    let micro_f1 = calculate_f1_score(micro_precision, micro_recall);

    ClassificationMetrics {
        accuracy,
        micro_precision,
        micro_recall,
        macro_f1_score: macro_f1,
        micro_f1_score: micro_f1,
        metrics_by_class,
    }
}

/// Calculate accuracy by comparing max values in each row
fn calculate_accuracy(predictions: &DMat, targets: &DMat) -> f32 {
    let rows = predictions.rows();
    if rows == 0 || predictions.rows() != targets.rows() || predictions.cols() != targets.cols() {
        return 0.0;
    }

    let mut correct = 0;
    for i in 0..rows {
        let pred_max_idx = util::find_max_index_in_row(predictions, i);
        let target_max_idx = util::find_max_index_in_row(targets, i);
        if pred_max_idx == target_max_idx {
            correct += 1;
        }
    }

    correct as f32 / rows as f32
}

fn calculate_confusion_matrix(
    targets: &DMat, predictions: &DMat,
) -> (HashMap<usize, usize>, HashMap<usize, usize>, HashMap<usize, usize>) {
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

// Tests for utility functions
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_accuracy() {
        let predictions = DMat::new(2, 2, &[0.9f32, 0.1, 0.2, 0.8]);
        let targets = DMat::new(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let accuracy = calculate_accuracy(&predictions, &targets);
        assert!((accuracy - 1.0).abs() < 1e-6);
    }
}

use crate::matrix::DMat;

use super::{MetricEvaluator, Metrics};

pub struct RegressionMetrics {
    pub rmse: f32,
    pub r2: f32,
}

impl RegressionMetrics {
    pub fn display(&self) -> String {
        format!("Regression Metrics: RMSE:{:.4}, R-squared:{:.4}", self.rmse, self.r2)
    }

    pub(crate) fn headers(&self) -> Vec<&'static str> {
        vec!["R2"]
    }

    pub(crate) fn values_str(&self) -> Vec<String> {
        vec![format!("{:.5}", self.r2)]
    }

    pub(crate) fn values(&self) -> Vec<f32> {
        vec![self.r2]
    }
}

pub(crate) struct RegressionEvaluator;

impl MetricEvaluator for RegressionEvaluator {
    fn evaluate(&self, targets: &DMat, predictions: &DMat) -> Metrics {
        let rmse = rmse(targets, predictions);
        let r2 = r2_score(targets, predictions);

        Metrics::Regression(RegressionMetrics { rmse, r2 })
    }
}

fn rmse(y_true: &DMat, y_pred: &DMat) -> f32 {
    let mut sum = 0.0;
    let n = y_true.rows();

    for i in 0..n {
        for j in 0..y_true.cols() {
            let diff = y_true.at(i, j) - y_pred.at(i, j);
            sum += diff * diff;
        }
    }

    (sum / (n as f32)).sqrt()
}

fn r2_score(y_true: &DMat, y_pred: &DMat) -> f32 {
    let n = y_true.rows();
    let m = y_true.cols();

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for j in 0..m {
        let mean = (0..n).map(|i| y_true.at(i, j)).sum::<f32>() / (n as f32);

        for i in 0..n {
            let actual = y_true.at(i, j);
            let predicted = y_pred.at(i, j);
            ss_res += (actual - predicted).powi(2);
            ss_tot += (actual - mean).powi(2);
        }
    }

    1.0 - (ss_res / ss_tot)
}

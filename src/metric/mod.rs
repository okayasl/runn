use classification::ClassificationMetrics;
use regression::RegressionMetrics;

use crate::matrix::DenseMatrix;

pub mod classification;
pub mod regression;

pub trait MetricEvaluator {
    fn evaluate(&self, targets: &DenseMatrix, predictions: &DenseMatrix) -> MetricResult;
}

pub enum MetricResult {
    Classification(ClassificationMetrics),
    Regression(RegressionMetrics),
}

impl MetricResult {
    pub fn display(&self) -> String {
        match self {
            MetricResult::Classification(metrics) => metrics.display(),
            MetricResult::Regression(metrics) => metrics.display(),
        }
    }
}

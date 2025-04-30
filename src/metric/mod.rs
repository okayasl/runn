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

    pub(crate) fn headers(&self) -> Vec<&'static str> {
        match self {
            MetricResult::Classification(metrics) => metrics.headers(),
            MetricResult::Regression(metrics) => metrics.headers(),
        }
    }

    pub(crate) fn values_str(&self) -> Vec<String> {
        match self {
            MetricResult::Classification(metrics) => metrics.values_str(),
            MetricResult::Regression(metrics) => metrics.values_str(),
        }
    }

    pub(crate) fn values(&self) -> Vec<f32> {
        match self {
            MetricResult::Classification(metrics) => metrics.values(),
            MetricResult::Regression(metrics) => metrics.values(),
        }
    }
}

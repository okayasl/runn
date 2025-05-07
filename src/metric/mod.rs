use classification::ClassificationMetrics;
use regression::RegressionMetrics;

use crate::matrix::DMat;

pub mod classification;
pub mod regression;

pub(crate) trait MetricEvaluator {
    fn evaluate(&self, targets: &DMat, predictions: &DMat) -> Metrics;
}

pub enum Metrics {
    Classification(ClassificationMetrics),
    Regression(RegressionMetrics),
}

impl Metrics {
    pub fn display(&self) -> String {
        match self {
            Metrics::Classification(metrics) => metrics.display(),
            Metrics::Regression(metrics) => metrics.display(),
        }
    }

    pub(crate) fn headers(&self) -> Vec<&'static str> {
        match self {
            Metrics::Classification(metrics) => metrics.headers(),
            Metrics::Regression(metrics) => metrics.headers(),
        }
    }

    pub(crate) fn values_str(&self) -> Vec<String> {
        match self {
            Metrics::Classification(metrics) => metrics.values_str(),
            Metrics::Regression(metrics) => metrics.values_str(),
        }
    }

    pub(crate) fn values(&self) -> Vec<f32> {
        match self {
            Metrics::Classification(metrics) => metrics.values(),
            Metrics::Regression(metrics) => metrics.values(),
        }
    }
}

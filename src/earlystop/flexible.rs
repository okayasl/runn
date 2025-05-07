use log::{info, warn};
use serde::{Deserialize, Serialize};
use typetag;

use crate::{error::NetworkError, Metrics};

use super::EarlyStopper;

#[derive(Serialize, Deserialize, Clone)]
struct FlexibleEarlyStopper {
    patience: usize,
    min_delta: f32,
    best: f32,
    wait: usize,
    stopped_epoch: isize,
    stop_training: bool,
    monitor_metric: MonitorMetric,
    target: Option<f32>,
    smoothing_factor: Option<f32>,
    smoothed_value: Option<f32>,
}

impl FlexibleEarlyStopper {
    fn get_current_value(&self, loss: f32, metric_result: &Metrics) -> Option<f32> {
        match self.monitor_metric {
            MonitorMetric::Loss => Some(loss), // Loss is always valid
            MonitorMetric::Accuracy => {
                if let Metrics::Classification(metrics) = metric_result {
                    Some(metrics.accuracy)
                } else {
                    warn!("Early stopping is set to monitor accuracy, but the metric result is not classification.");
                    None // Invalid if MetricResult is not Classification
                }
            }
            MonitorMetric::R2 => {
                if let Metrics::Regression(metrics) = metric_result {
                    Some(metrics.r2)
                } else {
                    warn!("Early stopping is set to monitor R2, but the metric result is not regression.");
                    None // Invalid if MetricResult is not Regression
                }
            }
        }
    }
}

#[typetag::serde]
impl EarlyStopper for FlexibleEarlyStopper {
    fn update(&mut self, epoch: usize, loss: f32, metric_result: &Metrics) {
        let raw_value = match self.get_current_value(loss, metric_result) {
            Some(value) => value,
            None => {
                return; // Exit if the metric result is invalid
            }
        };

        // Apply smoothing if a smoothing factor is set
        let effective_value = if let Some(factor) = self.smoothing_factor {
            let smoothed = match self.smoothed_value {
                Some(prev) => factor * raw_value + (1.0 - factor) * prev,
                None => raw_value,
            };
            self.smoothed_value = Some(smoothed);
            smoothed
        } else {
            raw_value
        };

        // Check if the target value is reached
        if let Some(target) = self.target {
            let target_reached = match self.monitor_metric {
                MonitorMetric::Loss => effective_value <= target, // Minimize loss
                MonitorMetric::Accuracy | MonitorMetric::R2 => effective_value >= target, // Maximize accuracy or R2
            };

            if target_reached {
                self.stop_training = true;
                self.stopped_epoch = epoch as isize;
                info!(
                    "Early stopping triggered: {:?} reached target value of {} at epoch {}.",
                    self.monitor_metric, target, epoch
                );
                return;
            }
        }

        // Check for improvement
        if self
            .monitor_metric
            .is_improvement(effective_value, self.best, self.min_delta)
        {
            self.best = effective_value; // Update the best value
            self.wait = 0; // Reset the wait counter
        } else {
            self.wait += 1; // Increment the wait counter if no improvement
            if self.wait >= self.patience {
                self.stopped_epoch = epoch as isize;
                self.stop_training = true; // Stop training if patience is exceeded
                info!(
                    "Early stopping triggered: no improvement in {:?} for {} epochs.",
                    self.monitor_metric, self.patience
                );
            }
        }
    }

    fn is_training_stopped(&self) -> bool {
        self.stop_training
    }

    fn reset(&mut self) {
        self.best = self.monitor_metric.initial_best();
        self.wait = 0;
        self.stopped_epoch = -1;
        self.stop_training = false;
        self.target = None;
        self.smoothing_factor = None;
    }
}

/// Flexible Early Stopping
/// This early stopping strategy allows you to monitor different metrics (like loss, accuracy, etc.)
/// and set a target value for the monitored metric.
/// It stops training if the monitored metric does not improve for a specified number of epochs (patience).
/// It also allows for a minimum delta to consider an improvement.
/// The target value can be used to stop training if the monitored metric reaches a certain threshold.
/// The smoothing factor can be used to smooth the monitored metric over epochs.
/// This is useful for scenarios where the monitored metric may fluctuate a lot.
pub struct Flexible {
    patience: usize,
    min_delta: f32,
    monitor_metric: MonitorMetric,
    target: Option<f32>,
    smoothing_factor: Option<f32>,
}

impl Flexible {
    /// Creates a new instance of the Flexible early stopping strategy.
    /// Default values are:
    /// - patience: 10
    /// - min_delta: 0.0
    /// - monitor_metric: Loss
    /// - target: None
    /// - smoothing_factor: None
    pub fn new() -> Self {
        Self {
            patience: 10,
            min_delta: 0.0,
            monitor_metric: MonitorMetric::Loss, // Default to monitoring loss
            target: None,
            smoothing_factor: None,
        }
    }

    /// Sets the patience for early stopping.
    /// The patience is the number of epochs with no improvement after which training will be stopped.
    /// #parameters
    /// - `patience`: The number of epochs with no improvement after which training will be stopped.
    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Sets the minimum delta for early stopping.
    /// The minimum delta is the minimum change in the monitored metric to consider it an improvement.
    /// #parameters
    /// - `min_delta`: The minimum change in the monitored metric to consider it an improvement.
    pub fn min_delta(mut self, min_delta: f32) -> Self {
        self.min_delta = min_delta;
        self
    }

    /// Sets the metric to monitor for early stopping.
    /// The metric can be one of the following:
    /// - Loss
    /// - Accuracy if the model is a classifier
    /// - R2 if the model is a regressor
    /// #parameters
    /// - `monitor_metric`: The metric to monitor for early stopping.
    pub fn monitor_metric(mut self, monitor_metric: MonitorMetric) -> Self {
        self.monitor_metric = monitor_metric;
        self
    }

    /// Sets the target value for early stopping.
    /// The target value is the value of the monitored metric at which training will be stopped.
    /// This is useful for scenarios where the monitored metric may fluctuate a lot.
    /// #parameters
    /// - `target`: The target value for early stopping.
    pub fn target(mut self, target: f32) -> Self {
        self.target = Some(target);
        self
    }

    /// Sets the smoothing factor for early stopping.
    /// The smoothing factor is used to smooth the monitored metric over epochs.
    /// It uses exponential moving average (EMA) smoothing on the loss before comparing to best
    /// This is useful for scenarios where the monitored metric may fluctuate a lot.
    /// The smoothing factor should be in the range [0, 1].
    /// A value of 0 means no smoothing, and a value of 1 means maximum smoothing.
    /// #parameters
    /// - `smoothing_factor`: The smoothing factor for early stopping.
    pub fn smoothing_factor(mut self, smoothing_factor: f32) -> Self {
        self.smoothing_factor = Some(smoothing_factor);
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.patience == 0 {
            return Err(NetworkError::ConfigError(format!(
                "Patience for flexible early stopper must be greater than 0, but was {}",
                self.patience
            )));
        }
        if self.min_delta < 0.0 {
            return Err(NetworkError::ConfigError(format!(
                "Min delta for flexible early stopper must be non-negative, but was {}",
                self.min_delta
            )));
        }
        if let Some(target) = self.target {
            if target < 0.0 {
                return Err(NetworkError::ConfigError(format!(
                    "Target for flexible early stopper must be non-negative, but was {}",
                    target
                )));
            }
        }
        if let Some(smoothing_factor) = self.smoothing_factor {
            if smoothing_factor < 0.0 || smoothing_factor > 1.0 {
                return Err(NetworkError::ConfigError(format!(
                    "Smoothing factor for flexible early stopper must be in the range [0, 1], but was {}",
                    smoothing_factor
                )));
            }
        }

        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn EarlyStopper>, NetworkError> {
        self.validate()?;
        Ok(Box::new(FlexibleEarlyStopper {
            patience: self.patience,
            min_delta: self.min_delta,
            best: self.monitor_metric.initial_best(),
            wait: 0,
            stopped_epoch: -1,
            stop_training: false,
            monitor_metric: self.monitor_metric,
            target: self.target,
            smoothing_factor: self.smoothing_factor,
            smoothed_value: None,
        }))
    }
}

/// Enum representing the different metrics that can be monitored for early stopping.
/// The metrics can be one of the following:
/// - Loss
/// - Accuracy
/// - R2
/// The metrics are used to determine if the training should be stopped.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum MonitorMetric {
    Loss,
    Accuracy,
    R2,
}

impl MonitorMetric {
    /// Determines if the current value is an improvement over the best value
    pub(crate) fn is_improvement(&self, current: f32, best: f32, min_delta: f32) -> bool {
        match self {
            MonitorMetric::Loss => (best - current) > min_delta, // Minimize loss
            MonitorMetric::Accuracy => (current - best) > min_delta, // Maximize accuracy
            MonitorMetric::R2 => (current - best) > min_delta,   // Maximize R2
        }
    }

    /// Provides the initial best value for the metric
    pub(crate) fn initial_best(&self) -> f32 {
        match self {
            MonitorMetric::Loss => f32::MAX, // Start with a very high loss
            MonitorMetric::Accuracy => 0.0,  // Start with a very low accuracy
            MonitorMetric::R2 => f32::MIN,   // Start with a very low R2
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        classification::ClassificationMetrics,
        flexible::{Flexible, MonitorMetric},
        metric,
        regression::RegressionMetrics,
        Metrics,
    };

    #[test]
    fn test_flexible_builder_default_values() {
        let stopper = Flexible::new().build();
        assert!(stopper.is_ok());
    }

    #[test]
    fn test_flexible_builder_custom_values() {
        let stopper = Flexible::new()
            .patience(5)
            .min_delta(0.01)
            .monitor_metric(MonitorMetric::Accuracy)
            .target(0.95)
            .smoothing_factor(0.5)
            .build();
        assert!(stopper.is_ok());
    }

    #[test]
    fn test_flexible_builder_validation_errors() {
        // Invalid patience
        let result = Flexible::new().patience(0).build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(
                e.to_string(),
                "Configuration error: Patience for flexible early stopper must be greater than 0, but was 0"
            );
        }

        // Invalid min_delta
        let result = Flexible::new().min_delta(-0.01).build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(
                e.to_string(),
                "Configuration error: Min delta for flexible early stopper must be non-negative, but was -0.01"
            );
        }

        // Invalid target
        let result = Flexible::new().target(-1.0).build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(
                e.to_string(),
                "Configuration error: Target for flexible early stopper must be non-negative, but was -1"
            );
        }

        // Invalid smoothing factor
        let result = Flexible::new().smoothing_factor(1.5).build();
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(
                e.to_string(),
                "Configuration error: Smoothing factor for flexible early stopper must be in the range [0, 1], but was 1.5"
            );
        }
    }

    #[test]
    fn test_flexible_early_stopper_loss() {
        let mut stopper = Flexible::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_metric(MonitorMetric::Loss)
            .build()
            .unwrap();

        let losses = vec![0.5, 0.4, 0.39, 0.39, 0.39, 0.38];
        for (epoch, &loss) in losses.iter().enumerate() {
            stopper.update(epoch, loss, &Metrics::Regression(RegressionMetrics { rmse: loss, r2: 0.0 }));
            if stopper.is_training_stopped() {
                assert_eq!(epoch, 5); // Training should stop after 4 epochs with no improvement
                break;
            }
        }
    }

    #[test]
    fn test_flexible_early_stopper_accuracy() {
        let mut stopper = Flexible::new()
            .patience(2)
            .min_delta(0.01)
            .monitor_metric(MonitorMetric::Accuracy)
            .target(0.95)
            .build()
            .unwrap();

        let accuracies = vec![0.7, 0.75, 0.76, 0.77, 0.95, 0.96];
        for (epoch, &accuracy) in accuracies.iter().enumerate() {
            stopper.update(
                epoch,
                0.0,
                &Metrics::Classification(ClassificationMetrics {
                    accuracy,
                    micro_precision: 0.0,
                    micro_recall: 0.0,
                    macro_f1_score: 0.0,
                    micro_f1_score: 0.0,
                    metrics_by_class: vec![],
                }),
            );
            if stopper.is_training_stopped() {
                assert_eq!(epoch, 4); // Training should stop when accuracy reaches 0.95
                break;
            }
        }
    }

    #[test]
    fn test_flexible_early_stopper_with_smoothing() {
        let mut stopper = Flexible::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_metric(MonitorMetric::Loss)
            .smoothing_factor(0.5)
            .build()
            .unwrap();

        let losses = vec![0.5, 0.4, 0.39, 0.39, 0.39, 0.38];
        for (epoch, &loss) in losses.iter().enumerate() {
            stopper.update(epoch, loss, &Metrics::Regression(RegressionMetrics { rmse: loss, r2: 0.0 }));
            if stopper.is_training_stopped() {
                assert_eq!(epoch, 4); // Training should stop after 3 epochs with no improvement
                break;
            }
        }
    }

    #[test]
    fn test_flexible_early_stopper_invalid_metric_result() {
        let mut stopper = Flexible::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_metric(MonitorMetric::Accuracy)
            .build()
            .unwrap();

        stopper.update(0, 0.0, &Metrics::Regression(RegressionMetrics { rmse: 0.5, r2: 0.0 }));
        assert!(!stopper.is_training_stopped()); // Should not stop as MetricResult is invalid for Accuracy
    }

    #[test]
    fn test_early_stopping_by_loss() {
        let mut early_stopper = Flexible::new()
            .monitor_metric(MonitorMetric::Loss)
            .patience(3)
            .min_delta(0.01)
            .build()
            .unwrap();

        let losses = vec![0.5, 0.4, 0.35, 0.36, 0.37, 0.38];
        for (epoch, &val_loss) in losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss, &get_dummy_metric_result());
            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 5);
                break;
            }
        }
    }

    fn get_dummy_metric_result() -> Metrics {
        let dummy_metric = metric::classification::ClassificationMetrics {
            accuracy: 0.0,
            micro_precision: 0.0,
            micro_recall: 0.0,
            macro_f1_score: 0.0,
            micro_f1_score: 0.0,
            metrics_by_class: vec![],
        };
        let metric_result = Metrics::Classification(dummy_metric);
        metric_result
    }

    #[test]
    fn test_no_early_stopping() {
        let mut early_stopper = Flexible::new()
            .monitor_metric(MonitorMetric::Loss)
            .patience(3)
            .min_delta(0.01)
            .build()
            .unwrap();

        let val_losses = vec![0.5, 0.4, 0.35, 0.34, 0.33, 0.32];
        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss, &get_dummy_metric_result());
        }
        assert!(!early_stopper.is_training_stopped());
    }

    #[test]
    fn test_reset() {
        let mut early_stopper = Flexible::new()
            .monitor_metric(MonitorMetric::Loss)
            .patience(3)
            .min_delta(0.01)
            .build()
            .unwrap();

        let val_losses = vec![0.5, 0.4, 0.35, 0.36, 0.37, 0.38];

        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss, &get_dummy_metric_result());

            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 5);
                break;
            }
        }
        early_stopper.reset();
        assert!(!early_stopper.is_training_stopped());
    }

    #[test]
    fn test_target_loss_triggers_stop() {
        let mut early_stopper = Flexible::new()
            .monitor_metric(MonitorMetric::Loss)
            .target(0.33)
            .patience(10)
            .build()
            .unwrap();

        let val_losses = vec![0.5, 0.4, 0.35, 0.34, 0.33, 0.32];
        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss, &get_dummy_metric_result());
            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 4); // Should stop at 0.33
                return;
            }
        }
        panic!("Expected early stopping by target loss");
    }

    #[test]
    fn test_smoothing_affects_stopping() {
        let mut early_stopper = Flexible::new()
            .monitor_metric(MonitorMetric::Loss)
            .patience(2)
            .min_delta(0.01)
            .smoothing_factor(0.5)
            .build()
            .unwrap();

        let losses = vec![0.5, 0.4, 0.45, 0.46, 0.47]; // Raw loss is going up but slowly
        let mut stopped = false;
        for (epoch, &val_loss) in losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss, &get_dummy_metric_result());
            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 3);
                stopped = true;
                break;
            }
        }
        assert!(stopped, "Expected early stopping with smoothing applied");
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_no_smoothing_vs_smoothing() {
            // A slowly rising loss curve
            let val_losses = vec![0.30, 0.31, 0.32, 0.33, 0.34, 0.35];

            // Raw stopper: high patience so it won't stop in 6 epochs
            let mut no_smoothing = Flexible::new()
                .monitor_metric(MonitorMetric::Loss)
                .patience(10)
                .min_delta(0.01)
                .build()
                .unwrap();

            // EMA stopper: low patience, smoothing_factor => will detect rising trend
            let mut with_smoothing = Flexible::new()
                .monitor_metric(MonitorMetric::Loss)
                .patience(2)
                .min_delta(0.01)
                .smoothing_factor(0.6)
                .build()
                .unwrap();

            for (epoch, &loss) in val_losses.iter().enumerate() {
                no_smoothing.update(epoch, loss, &get_dummy_metric_result());
                with_smoothing.update(epoch, loss, &get_dummy_metric_result());
            }

            // EMA-based stopper should have halted
            assert!(with_smoothing.is_training_stopped(), "With smoothing should stop on the rising trend");
            // Raw stopper still within patience window
            assert!(!no_smoothing.is_training_stopped(), "Without smoothing (high patience) should continue");
        }
    }
}

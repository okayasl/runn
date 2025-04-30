use serde::{Deserialize, Serialize};
use typetag;

use super::EarlyStopper;

#[derive(Serialize, Deserialize, Clone)]
pub struct FlexibleEarlyStopper {
    patience: usize,
    min_delta: f32,
    best: f32,
    wait: usize,
    stopped_epoch: isize,
    stop_training: bool,
    monitor_metric: MonitorMetric,
}

impl FlexibleEarlyStopper {
    pub fn new(patience: usize, min_delta: f32, monitor_metric: MonitorMetric) -> Self {
        Self {
            patience,
            min_delta,
            best: match monitor_metric {
                MonitorMetric::Loss => f32::MAX, // Initialize for loss
                MonitorMetric::Accuracy => 0.0,  // Initialize for accuracy
            },
            wait: 0,
            stopped_epoch: -1,
            stop_training: false,
            monitor_metric,
        }
    }
}

#[typetag::serde]
impl EarlyStopper for FlexibleEarlyStopper {
    fn update(&mut self, epoch: usize, loss: f32) {
        let current_value = match self.monitor_metric {
            MonitorMetric::Loss => loss,
            MonitorMetric::Accuracy => 0.0,
        };

        if self
            .monitor_metric
            .is_improvement(current_value, self.best, self.min_delta)
        {
            self.best = current_value; // Update the best value
            self.wait = 0; // Reset the wait counter
        } else {
            self.wait += 1; // Increment the wait counter if no improvement
            if self.wait >= self.patience {
                self.stopped_epoch = epoch as isize;
                self.stop_training = true; // Stop training if patience is exceeded
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
    }
}

pub struct Flexible {
    patience: usize,
    min_delta: f32,
    monitor_metric: MonitorMetric,
}

impl Flexible {
    pub fn new() -> Self {
        Self {
            patience: 10,
            min_delta: 0.0,
            monitor_metric: MonitorMetric::Loss, // Default to monitoring loss
        }
    }

    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    pub fn min_delta(mut self, min_delta: f32) -> Self {
        self.min_delta = min_delta;
        self
    }

    pub fn monitor_metric(mut self, monitor_metric: MonitorMetric) -> Self {
        self.monitor_metric = monitor_metric;
        self
    }

    pub fn build(self) -> FlexibleEarlyStopper {
        FlexibleEarlyStopper::new(self.patience, self.min_delta, self.monitor_metric)
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum MonitorMetric {
    Loss,
    Accuracy,
}

impl MonitorMetric {
    /// Determines if the current value is an improvement over the best value
    pub fn is_improvement(&self, current: f32, best: f32, min_delta: f32) -> bool {
        match self {
            MonitorMetric::Loss => (best - current) > min_delta, // Minimize loss
            MonitorMetric::Accuracy => (current - best) > min_delta, // Maximize accuracy
        }
    }

    /// Provides the initial best value for the metric
    pub fn initial_best(&self) -> f32 {
        match self {
            MonitorMetric::Loss => f32::MAX, // Start with a very high loss
            MonitorMetric::Accuracy => 0.0,  // Start with a very low accuracy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_by_loss() {
        let mut early_stopper = Flexible::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_metric(MonitorMetric::Loss)
            .build();

        let losses = vec![0.5, 0.4, 0.35, 0.36, 0.37, 0.38];
        for (epoch, &val_loss) in losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss);
            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 5);
                break;
            }
        }
    }

    #[test]
    fn test_early_stopping_by_accuracy() {
        let mut early_stopper = Flexible::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_metric(MonitorMetric::Accuracy)
            .build();

        let val_accuracies = vec![0.7, 0.75, 0.76, 0.75, 0.74, 0.73];
        for (epoch, &val_accuracy) in val_accuracies.iter().enumerate() {
            early_stopper.update(epoch, val_accuracy);
            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 4);
                break;
            }
        }
    }

    #[test]
    fn test_no_early_stopping() {
        let mut early_stopper = Flexible::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_metric(MonitorMetric::Loss)
            .build();

        let val_losses = vec![0.5, 0.4, 0.35, 0.34, 0.33, 0.32];
        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss);
        }
        assert!(!early_stopper.is_training_stopped());
    }

    #[test]
    fn test_reset() {
        let mut early_stopper = Flexible::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_metric(MonitorMetric::Loss)
            .build();

        let val_losses = vec![0.5, 0.4, 0.35, 0.36, 0.37, 0.38];

        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss);

            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 5);
                break;
            }
        }
        early_stopper.reset();
        assert!(!early_stopper.is_training_stopped());
        assert_eq!(early_stopper.best, f32::MAX);
        assert_eq!(early_stopper.wait, 0);
        assert_eq!(early_stopper.stopped_epoch, -1);
    }
}

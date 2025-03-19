use serde::{Deserialize, Serialize};
use typetag;

use super::EarlyStopper;

#[derive(Serialize, Deserialize, Clone)]
pub struct FlexibleEarlyStopper {
    patience: usize,
    min_delta: f32,
    best_loss: f32,
    best_accuracy: f32,
    wait: usize,
    stopped_epoch: isize,
    stop_training: bool,
    monitor_accuracy: bool,
}

impl FlexibleEarlyStopper {
    pub fn new(patience: usize, min_delta: f32, monitor_accuracy: bool) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f32::INFINITY,
            best_accuracy: f32::NEG_INFINITY,
            wait: 0,
            stopped_epoch: -1,
            stop_training: false,
            monitor_accuracy,
        }
    }
}

#[typetag::serde]
impl EarlyStopper for FlexibleEarlyStopper {
    fn update(&mut self, epoch: usize, val_loss: f32, val_accuracy: f32) {
        if self.monitor_accuracy {
            if (val_accuracy - self.best_accuracy) > self.min_delta {
                self.best_accuracy = val_accuracy;
                self.wait = 0;
            } else {
                self.wait += 1;
                if self.wait >= self.patience {
                    self.stopped_epoch = epoch as isize;
                    self.stop_training = true;
                }
            }
        } else {
            if (self.best_loss - val_loss) > self.min_delta {
                self.best_loss = val_loss;
                self.wait = 0;
            } else {
                self.wait += 1;
                if self.wait >= self.patience {
                    self.stopped_epoch = epoch as isize;
                    self.stop_training = true;
                }
            }
        }
    }

    fn is_training_stopped(&self) -> bool {
        self.stop_training
    }

    fn reset(&mut self) {
        self.best_loss = f32::INFINITY;
        self.best_accuracy = f32::NEG_INFINITY;
        self.wait = 0;
        self.stopped_epoch = -1;
        self.stop_training = false;
    }
}

pub struct FlexibleBuilder {
    patience: usize,
    min_delta: f32,
    monitor_accuracy: bool,
}

impl FlexibleBuilder {
    pub fn new() -> Self {
        Self {
            patience: 10,
            min_delta: 0.0,
            monitor_accuracy: false,
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

    pub fn monitor_accuracy(mut self, monitor_accuracy: bool) -> Self {
        self.monitor_accuracy = monitor_accuracy;
        self
    }

    pub fn build(self) -> FlexibleEarlyStopper {
        FlexibleEarlyStopper::new(self.patience, self.min_delta, self.monitor_accuracy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_by_loss() {
        let mut early_stopper = FlexibleBuilder::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_accuracy(false)
            .build();

        let val_losses = vec![0.5, 0.4, 0.35, 0.36, 0.37, 0.38];

        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss, 0.0);

            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 5);
                break;
            }
        }
    }

    #[test]
    fn test_early_stopping_by_accuracy() {
        let mut early_stopper = FlexibleBuilder::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_accuracy(true)
            .build();

        let val_accuracies = vec![0.7, 0.75, 0.76, 0.75, 0.74, 0.73];

        for (epoch, &val_accuracy) in val_accuracies.iter().enumerate() {
            early_stopper.update(epoch, 0.0, val_accuracy);

            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 4);
                break;
            }
        }
    }

    #[test]
    fn test_no_early_stopping() {
        let mut early_stopper = FlexibleBuilder::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_accuracy(false)
            .build();

        let val_losses = vec![0.5, 0.4, 0.35, 0.34, 0.33, 0.32];

        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss, 0.0);
        }

        assert!(!early_stopper.is_training_stopped());
    }

    #[test]
    fn test_reset() {
        let mut early_stopper = FlexibleBuilder::new()
            .patience(3)
            .min_delta(0.01)
            .monitor_accuracy(false)
            .build();

        let val_losses = vec![0.5, 0.4, 0.35, 0.36, 0.37, 0.38];

        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss, 0.0);

            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 5);
                break;
            }
        }

        early_stopper.reset();

        assert!(!early_stopper.is_training_stopped());
        assert_eq!(early_stopper.best_loss, f32::INFINITY);
        assert_eq!(early_stopper.best_accuracy, f32::NEG_INFINITY);
        assert_eq!(early_stopper.wait, 0);
        assert_eq!(early_stopper.stopped_epoch, -1);
    }
}

use serde::{Deserialize, Serialize};

use super::EarlyStopper;

#[derive(Serialize, Deserialize, Clone)]
pub struct LossEarlyStopper {
    patience: usize,
    min_delta: f32,
    best: f32,
    wait: usize,
    stopped_epoch: isize,
    stop_training: bool,
}

impl LossEarlyStopper {
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            best: f32::MAX,
            wait: 0,
            stopped_epoch: -1,
            stop_training: false,
        }
    }
}

#[typetag::serde]
impl EarlyStopper for LossEarlyStopper {
    fn update(&mut self, epoch: usize, loss: f32) {
        if (self.best - loss) > self.min_delta {
            self.best = loss;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped_epoch = epoch as isize;
                self.stop_training = true;
            }
        }
    }

    fn is_training_stopped(&self) -> bool {
        self.stop_training
    }

    fn reset(&mut self) {
        self.best = f32::MAX;
        self.wait = 0;
        self.stopped_epoch = -1;
        self.stop_training = false;
    }
}

pub struct Loss {
    patience: usize,
    min_delta: f32,
}

impl Loss {
    pub fn new() -> Self {
        Self {
            patience: 10,
            min_delta: 0.0,
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

    pub fn build(self) -> LossEarlyStopper {
        LossEarlyStopper::new(self.patience, self.min_delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_by_loss() {
        let mut early_stopper = Loss::new().patience(3).min_delta(0.01).build();

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
    fn test_no_early_stopping() {
        let mut early_stopper = Loss::new().patience(3).min_delta(0.01).build();

        let val_losses = vec![0.5, 0.4, 0.35, 0.34, 0.33, 0.32];
        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss);
        }
        assert!(!early_stopper.is_training_stopped());
    }

    #[test]
    fn test_reset() {
        let mut early_stopper = Loss::new().patience(3).min_delta(0.01).build();

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

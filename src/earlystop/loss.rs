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
    target_loss: Option<f32>,
    smoothed_loss: Option<f32>,
    smoothing_factor: Option<f32>,
}

impl LossEarlyStopper {
    pub fn new(patience: usize, min_delta: f32, target_loss: Option<f32>, smoothing_factor: Option<f32>) -> Self {
        Self {
            patience,
            min_delta,
            best: f32::MAX,
            wait: 0,
            stopped_epoch: -1,
            stop_training: false,
            target_loss,
            smoothed_loss: None,
            smoothing_factor,
        }
    }
}

#[typetag::serde]
impl EarlyStopper for LossEarlyStopper {
    fn update(&mut self, epoch: usize, loss: f32) {
        if let Some(target) = self.target_loss {
            if loss <= target {
                self.stop_training = true;
                self.stopped_epoch = epoch as isize;
                return;
            }
        }

        let effective_loss = if let Some(factor) = self.smoothing_factor {
            let smoothed = match self.smoothed_loss {
                Some(prev) => factor * loss + (1.0 - factor) * prev,
                None => loss,
            };
            self.smoothed_loss = Some(smoothed);
            smoothed
        } else {
            loss
        };

        if (self.best - effective_loss) >= (self.min_delta + f32::EPSILON) {
            self.best = effective_loss;
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
        self.smoothed_loss = None;
    }
}

pub struct Loss {
    patience: usize,
    min_delta: f32,
    target_loss: Option<f32>,
    smoothing_factor: Option<f32>,
}

impl Loss {
    /// Create a new `Loss` builder with **default** parameters:
    ///
    /// - `patience = 10`  
    /// - `min_delta = 0.0` (any improvement counts)  
    /// - `target_loss = None` (no target threshold)  
    /// - `smoothing_factor = None` (no EMA smoothing)  
    pub fn new() -> Self {
        Self {
            patience: 10,
            min_delta: 0.0,
            target_loss: None,
            smoothing_factor: None,
        }
    }

    /// Stop training as soon as `loss <= target`.
    /// This is an **optional** hard threshold: if the reported loss
    /// ever falls to or below `target`, training halts immediately,
    /// regardless of patience or smoothing.  
    /// # Parameters
    /// - `target`: the loss value at which you consider the model “good enough.”  
    pub fn target_loss(mut self, target: f32) -> Self {
        self.target_loss = Some(target);
        self
    }

    /// How many epochs to wait **without** sufficient improvement
    /// before stopping.
    /// The default is 10; set a smaller value to end training sooner
    /// when progress stalls.  
    /// # Parameters
    /// - `patience`: number of consecutive non‑improving epochs allowed.  
    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Minimum required decrease in loss to count as “improvement.”
    /// If `(previous_best − current_loss) < min_delta`, the epoch
    /// counts toward patience. Defaults to `0.0`, meaning **any**
    /// decrease resets the patience counter.  
    /// # Parameters
    /// - `min_delta`: smallest loss drop to treat as genuine progress.  
    pub fn min_delta(mut self, min_delta: f32) -> Self {
        self.min_delta = min_delta;
        self
    }

    /// Use **exponential moving average** (EMA) smoothing on the loss
    /// before comparing to `best`.  
    /// This filters out noisy spikes (tiny batch‑to‑batch jitter)
    /// so that patience measures a genuine plateau rather than
    /// momentary blips :contentReference[oaicite:3]{index=3}.  
    /// # Parameters
    /// - `factor` ∈ [0.0, 1.0]:  
    ///   - Closer to 1.0 → heavy smoothing (slow to react).  
    ///   - Closer to 0.0 → light smoothing (nearly raw loss).  
    pub fn smoothing_factor(mut self, factor: f32) -> Self {
        self.smoothing_factor = Some(factor.clamp(0.0, 1.0));
        self
    }

    /// Finalize the builder and create a `LossEarlyStopper`.
    pub fn build(self) -> LossEarlyStopper {
        LossEarlyStopper::new(self.patience, self.min_delta, self.target_loss, self.smoothing_factor)
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

    #[test]
    fn test_target_loss_triggers_stop() {
        let mut early_stopper = Loss::new().target_loss(0.33).patience(10).build();

        let val_losses = vec![0.5, 0.4, 0.35, 0.34, 0.33, 0.32];
        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss);
            if early_stopper.is_training_stopped() {
                assert_eq!(epoch, 4); // Should stop at 0.33
                return;
            }
        }
        panic!("Expected early stopping by target loss");
    }

    #[test]
    fn test_smoothing_affects_stopping() {
        let mut early_stopper = Loss::new().patience(2).min_delta(0.01).smoothing_factor(0.5).build();

        let losses = vec![0.5, 0.4, 0.45, 0.46, 0.47]; // Raw loss is going up but slowly
        let mut stopped = false;
        for (epoch, &val_loss) in losses.iter().enumerate() {
            early_stopper.update(epoch, val_loss);
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
            let mut no_smoothing = Loss::new().patience(10).min_delta(0.01).build();

            // EMA stopper: low patience, smoothing_factor => will detect rising trend
            let mut with_smoothing = Loss::new().patience(2).min_delta(0.01).smoothing_factor(0.6).build();

            for (epoch, &loss) in val_losses.iter().enumerate() {
                no_smoothing.update(epoch, loss);
                with_smoothing.update(epoch, loss);
            }

            // EMA-based stopper should have halted
            assert!(with_smoothing.is_training_stopped(), "With smoothing should stop on the rising trend");
            // Raw stopper still within patience window
            assert!(!no_smoothing.is_training_stopped(), "Without smoothing (high patience) should continue");
        }
    }
}

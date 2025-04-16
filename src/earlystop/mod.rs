pub mod flexible;

use typetag;

#[typetag::serde]
pub trait EarlyStopper: EarlyStopperClone + Send {
    fn update(&mut self, epoch: usize, val_loss: f32, val_accuracy: f32);
    fn is_training_stopped(&self) -> bool;
    fn reset(&mut self);
}

pub trait EarlyStopperClone {
    fn clone_box(&self) -> Box<dyn EarlyStopper>;
}

impl<T> EarlyStopperClone for T
where
    T: 'static + EarlyStopper + Clone,
{
    fn clone_box(&self) -> Box<dyn EarlyStopper> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn EarlyStopper> {
    fn clone(&self) -> Box<dyn EarlyStopper> {
        self.clone_box()
    }
}

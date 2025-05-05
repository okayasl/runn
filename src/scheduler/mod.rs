pub mod exponential;
pub mod step;

#[typetag::serde]
pub trait LearningRateScheduler: LearningRateSchedulerClone + Send + Sync {
    fn schedule(&self, epoch: usize, current_learning_rate: f32) -> f32;
}

#[typetag::serde]
impl LearningRateScheduler for Box<dyn LearningRateScheduler> {
    fn schedule(&self, epoch: usize, current_learning_rate: f32) -> f32 {
        (**self).schedule(epoch, current_learning_rate)
    }
}

impl LearningRateSchedulerClone for Box<dyn LearningRateScheduler> {
    fn clone_box(&self) -> Box<dyn LearningRateScheduler> {
        (**self).clone_box()
    }
}

pub trait LearningRateSchedulerClone {
    fn clone_box(&self) -> Box<dyn LearningRateScheduler>;
}

impl Clone for Box<dyn LearningRateScheduler> {
    fn clone(&self) -> Box<dyn LearningRateScheduler> {
        self.clone_box()
    }
}

pub mod exponential;
pub mod step;

#[typetag::serde]
pub trait LearningRateScheduler: LearningRateSchedulerClone + Send + Sync {
    fn schedule(&self, epoch: usize, current_learning_rate: f32) -> f32;
}

pub trait LearningRateSchedulerClone {
    fn clone_box(&self) -> Box<dyn LearningRateScheduler>;
}

impl<T> LearningRateSchedulerClone for T
where
    T: 'static + LearningRateScheduler + Clone,
{
    fn clone_box(&self) -> Box<dyn LearningRateScheduler> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn LearningRateScheduler> {
    fn clone(&self) -> Box<dyn LearningRateScheduler> {
        self.clone_box()
    }
}

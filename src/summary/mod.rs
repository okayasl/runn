pub mod tensor_board;
use std::error::Error;

#[typetag::serde]
pub trait SummaryWriter: SummaryWriterClone + Send + Sync {
    fn write_scalar(&mut self, tag: &str, step: usize, value: f32) -> Result<(), Box<dyn Error>>;
    fn write_histogram(&mut self, tag: &str, step: usize, values: &[f32]) -> Result<(), Box<dyn Error>>;
    fn close(&mut self) -> Result<(), Box<dyn Error>>;
    fn init(&mut self);
}
pub trait SummaryWriterClone {
    fn clone_box(&self) -> Box<dyn SummaryWriter>;
}

impl<T> SummaryWriterClone for T
where
    T: 'static + SummaryWriter + Clone,
{
    fn clone_box(&self) -> Box<dyn SummaryWriter> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn SummaryWriter> {
    fn clone(&self) -> Box<dyn SummaryWriter> {
        self.clone_box()
    }
}

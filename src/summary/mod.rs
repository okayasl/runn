pub mod tensor_board;
use std::error::Error;

#[typetag::serde]
pub trait SummaryWriter: SummaryWriterClone + Send + Sync {
    fn write_scalar(&mut self, tag: &str, step: usize, value: f32) -> Result<(), Box<dyn Error>>;
    fn write_histogram(&mut self, tag: &str, step: usize, values: &[f32]) -> Result<(), Box<dyn Error>>;
    fn close(&mut self) -> Result<(), Box<dyn Error>>;
    fn init(&mut self);
}

#[typetag::serde]
impl SummaryWriter for Box<dyn SummaryWriter> {
    fn write_scalar(&mut self, tag: &str, step: usize, value: f32) -> Result<(), Box<dyn Error>> {
        (**self).write_scalar(tag, step, value)
    }

    fn write_histogram(&mut self, tag: &str, step: usize, values: &[f32]) -> Result<(), Box<dyn Error>> {
        (**self).write_histogram(tag, step, values)
    }

    fn close(&mut self) -> Result<(), Box<dyn Error>> {
        (**self).close()
    }

    fn init(&mut self) {
        (**self).init()
    }
}
impl SummaryWriterClone for Box<dyn SummaryWriter> {
    fn clone_box(&self) -> Box<dyn SummaryWriter> {
        (**self).clone_box()
    }
}

pub trait SummaryWriterClone {
    fn clone_box(&self) -> Box<dyn SummaryWriter>;
}

impl Clone for Box<dyn SummaryWriter> {
    fn clone(&self) -> Box<dyn SummaryWriter> {
        self.clone_box()
    }
}

pub mod tensor_board;
use std::error::Error;

pub trait SummaryWriter {
    fn write_scalar(&mut self, tag: &str, step: usize, value: f32) -> Result<(), Box<dyn Error>>;
    fn write_histogram(
        &mut self,
        tag: &str,
        step: usize,
        values: &[f32],
    ) -> Result<(), Box<dyn Error>>;
    fn close(&mut self) -> Result<(), Box<dyn Error>>;
}

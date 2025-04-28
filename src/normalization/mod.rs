use crate::matrix::DenseMatrix;

pub mod min_max;
pub mod zscore;

#[typetag::serde]
pub trait Normalization: NormalizationClone + Send + Sync {
    fn normalize(&mut self, matrix: &mut DenseMatrix) -> Result<(), String>;
    fn denormalize(&self, matrix: &mut DenseMatrix) -> Result<(), String>;
}

pub trait NormalizationClone {
    fn clone_box(&self) -> Box<dyn Normalization>;
}
impl<T> NormalizationClone for T
where
    T: 'static + Normalization + Clone,
{
    fn clone_box(&self) -> Box<dyn Normalization> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Normalization> {
    fn clone(&self) -> Box<dyn Normalization> {
        self.clone_box()
    }
}

use crate::error::NetworkError;

pub mod cvs;

pub trait Exporter {
    fn export(&self, headers: Vec<String>, values: Vec<Vec<String>>) -> Result<(), NetworkError>;
}

pub(crate) trait Exportable {
    fn header(&self) -> Vec<String>;
    fn values(&self) -> Vec<String>;
}

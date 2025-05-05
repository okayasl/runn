use thiserror::Error;

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

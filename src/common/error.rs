use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum NetworkError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

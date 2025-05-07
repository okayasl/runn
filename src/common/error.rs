use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum NetworkError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Search error: {0}")]
    SearchError(String),

    #[error("ThreadPool error: {0}")]
    ThreadPoolError(String),

    #[error("IO error: {0}")]
    IoError(String),
}

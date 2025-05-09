use std::fs::{self, File};

use crate::error::NetworkError;

use super::Exporter;

struct CSVExporter {
    delimiter: char,
    file_name: String,
    directory: String,
    file_extension: String,
}

impl Exporter for CSVExporter {
    fn export(&self, headers: Vec<String>, values: Vec<Vec<String>>) -> Result<(), NetworkError> {
        if !std::path::Path::new(&self.directory).exists() {
            fs::create_dir_all(&self.directory).map_err(|e| {
                NetworkError::IoError(format!("Failed to create output directory '{}': {}", self.directory, e))
            })?;
        }

        let file_path = format!("{}/{}{}", self.directory, self.file_name, self.file_extension);
        let file = File::create(&file_path)
            .map_err(|e| NetworkError::IoError(format!("Failed to create file '{}': {}", file_path, e)))?;

        // Use the delimiter property to configure the CSV writer
        let mut writer = csv::WriterBuilder::new()
            .delimiter(self.delimiter as u8) // Set the custom delimiter
            .from_writer(file);

        writer
            .write_record(headers)
            .map_err(|e| NetworkError::IoError(format!("Failed to write headers to '{}': {}", file_path, e)))?;

        for val in values {
            writer
                .write_record(val)
                .map_err(|e| NetworkError::IoError(format!("Failed to write result to '{}': {}", file_path, e)))?;
        }

        writer
            .flush()
            .map_err(|e| NetworkError::IoError(format!("Failed to flush writer for '{}': {}", file_path, e)))?;

        Ok(())
    }
}

pub struct CSV {
    delimiter: char,
    file_name: String,
    directory: String,
    file_extension: String,
}

impl CSV {
    pub fn new() -> Self {
        CSV {
            delimiter: ',',
            file_name: "result".to_string(),
            directory: ".".to_string(),
            file_extension: ".csv".to_string(),
        }
    }

    pub fn delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = delimiter;
        self
    }

    pub fn file_name(mut self, file_name: &str) -> Self {
        self.file_name = file_name.to_string();
        self
    }

    pub fn directory(mut self, directory: &str) -> Self {
        self.directory = directory.to_string();
        self
    }

    pub fn file_extension(mut self, file_extension: &str) -> Self {
        self.file_extension = file_extension.to_string();
        self
    }

    fn validate(&self) -> Result<(), NetworkError> {
        if self.file_name.is_empty() {
            return Err(NetworkError::ConfigError("File name cannot be empty".to_string()));
        }
        if self.directory.is_empty() {
            return Err(NetworkError::ConfigError("Directory cannot be empty".to_string()));
        }
        if self.delimiter == '\0' {
            return Err(NetworkError::ConfigError("Delimiter cannot be null character".to_string()));
        }
        if self.file_extension.is_empty() {
            return Err(NetworkError::ConfigError("File extension cannot be empty".to_string()));
        }

        // Check if the directory exists, and attempt to create it if it doesn't
        if !std::path::Path::new(&self.directory).exists() {
            fs::create_dir_all(&self.directory).map_err(|e| {
                NetworkError::IoError(format!("Failed to create output directory '{}': {}", self.directory, e))
            })?;
        }

        // Check if file can be created in the directory
        let file_path = format!("{}/{}{}", self.directory, self.file_name, self.file_extension);
        let file = File::create(&file_path).map_err(|e| {
            NetworkError::IoError(format!("Failed to create a test file in directory '{}': {}", self.directory, e))
        })?;

        // Clean up the test file
        drop(file);
        let _res = fs::remove_file(&file_path);

        Ok(())
    }

    pub fn build(self) -> Result<Box<dyn Exporter>, NetworkError> {
        self.validate()?;
        Ok(Box::new(CSVExporter {
            delimiter: self.delimiter,
            file_name: self.file_name,
            directory: self.directory,
            file_extension: self.file_extension,
        }))
    }
}

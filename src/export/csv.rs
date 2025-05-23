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

/// A builder for configuring and creating a CSV exporter.
///
/// This struct provides a fluent interface to customize various
/// options such as delimiter, file name, output directory, and file extension.
/// Once configured, the `build` method validates the configuration and
/// returns a CSV exporter.
pub struct CSV {
    delimiter: char,
    file_name: String,
    directory: String,
    file_extension: String,
}

impl CSV {
    /// Creates a new CSV builder.
    /// Default values:
    /// - Delimiter: `,`
    /// - File name: `"result"`
    /// - Directory: `"."` (current directory)
    /// - File extension: `".csv"`
    fn new() -> Self {
        CSV {
            delimiter: ',',
            file_name: "result".to_string(),
            directory: ".".to_string(),
            file_extension: ".csv".to_string(),
        }
    }

    /// Sets the delimiter character for the CSV file.
    pub fn delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Sets the base name for the output CSV file.
    pub fn file_name(mut self, file_name: &str) -> Self {
        self.file_name = file_name.to_string();
        self
    }

    /// Sets the output directory for the CSV file.
    pub fn directory(mut self, directory: &str) -> Self {
        self.directory = directory.to_string();
        self
    }

    /// Sets the file extension for the output file.
    pub fn file_extension(mut self, file_extension: &str) -> Self {
        self.file_extension = file_extension.to_string();
        self
    }

    // Validates the configuration and checks if the output directory and file are writable.
    //
    // # Errors
    // Returns a `NetworkError` if any field is invalid or the file system check fails.
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

        if !std::path::Path::new(&self.directory).exists() {
            fs::create_dir_all(&self.directory).map_err(|e| {
                NetworkError::IoError(format!("Failed to create output directory '{}': {}", self.directory, e))
            })?;
        }

        let file_path = format!("{}/{}{}", self.directory, self.file_name, self.file_extension);
        let file = File::create(&file_path).map_err(|e| {
            NetworkError::IoError(format!("Failed to create a test file in directory '{}': {}", self.directory, e))
        })?;

        drop(file);
        let _res = fs::remove_file(&file_path);

        Ok(())
    }

    /// Finalizes the builder and constructs a `CSVExporter` if the configuration is valid.
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

impl Default for CSV {
    /// Creates a new CSV builder with default values.
    /// Default values:
    /// - Delimiter: `,`
    /// - File name: `"result"`
    /// - Directory: `"."` (current directory)
    /// - File extension: `".csv"`
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_exporter() {
        let exporter = CSVExporter {
            delimiter: ',',
            file_name: "test".to_string(),
            directory: ".".to_string(),
            file_extension: ".csv".to_string(),
        };

        let headers = vec!["Header1".to_string(), "Header2".to_string()];
        let values = vec![vec!["Value1".to_string(), "Value2".to_string()]];

        let result = exporter.export(headers, values);
        assert!(result.is_ok());
        //remove the test file after the test
        let file_path = format!("{}/{}{}", ".", "test", ".csv");
        let _res = fs::remove_file(&file_path);
    }

    #[test]
    fn test_csv_exporter_invalid_directory() {
        let exporter = CSVExporter {
            delimiter: ',',
            file_name: "test".to_string(),
            directory: "/invalid/directory".to_string(),
            file_extension: ".csv".to_string(),
        };

        let headers = vec!["Header1".to_string(), "Header2".to_string()];
        let values = vec![vec!["Value1".to_string(), "Value2".to_string()]];

        let result = exporter.export(headers, values);
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_exporter_invalid_headers() {
        let exporter = CSVExporter {
            delimiter: ',',
            file_name: "test".to_string(),
            directory: ".".to_string(),
            file_extension: ".csv".to_string(),
        };

        let headers = vec![];
        let values = vec![vec!["Value1".to_string(), "Value2".to_string()]];

        let result = exporter.export(headers, values);
        assert!(result.is_err());

        //remove the test file after the test
        let file_path = format!("{}/{}{}", ".", "test", ".csv");
        let _res = fs::remove_file(&file_path);
    }

    #[test]
    fn test_csv_builder() {
        let csv = CSV::new()
            .delimiter(',')
            .file_name("test")
            .directory(".")
            .file_extension(".csv");

        let result = csv.build();
        assert!(result.is_ok());

        //remove the test file after the test
        let file_path = format!("{}/{}{}", ".", "test", ".csv");
        let _res = fs::remove_file(&file_path);
    }

    #[test]
    fn test_csv_builder_invalid_directory() {
        let csv = CSV::default()
            .delimiter(',')
            .file_name("test")
            .directory("/invalid/directory")
            .file_extension(".csv");

        let result = csv.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_builder_invalid_file_name() {
        let csv = CSV::default()
            .delimiter(',')
            .file_name("")
            .directory(".")
            .file_extension(".csv");

        let result = csv.build();
        assert!(result.is_err());
    }
}

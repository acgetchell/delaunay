#![forbid(unsafe_code)]

//! Validated artifact destinations and contextual output errors for the CLI.

use std::{
    fmt::{self, Display, Formatter},
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use serde::Serialize;

/// A validated file destination for one artifact write.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactPath {
    requested: PathBuf,
}

impl ArtifactPath {
    /// Parse a raw output path into a destination suitable for artifact writes.
    pub fn try_new(requested: PathBuf) -> Result<Self, ArtifactOutputError> {
        if requested.as_os_str().is_empty() {
            return Err(ArtifactOutputError::EmptyPath);
        }

        Ok(Self { requested })
    }

    /// Return the user-requested path used for diagnostics and file creation.
    pub fn as_path(&self) -> &Path {
        &self.requested
    }

    /// Create this destination's parent directories and truncate the output file.
    pub fn create_writer(&self) -> Result<BufWriter<File>, ArtifactOutputError> {
        if let Some(parent) = self
            .requested
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|source| ArtifactOutputError::CreateParent {
                path: parent.to_owned(),
                source,
            })?;
        }

        let file = File::create(&self.requested).map_err(|source| ArtifactOutputError::Open {
            path: self.requested.clone(),
            source,
        })?;
        Ok(BufWriter::new(file))
    }

    /// Attach this destination to a final artifact-flush failure.
    pub fn flush_error(&self, source: io::Error) -> ArtifactOutputError {
        ArtifactOutputError::Flush {
            path: self.requested.clone(),
            source,
        }
    }
}

/// Write pretty JSON to a validated path or to stdout.
pub fn write_json_output(
    value: &impl Serialize,
    path: Option<&ArtifactPath>,
) -> Result<(), ArtifactOutputError> {
    if let Some(path) = path {
        let mut writer = path.create_writer()?;
        serde_json::to_writer_pretty(&mut writer, value).map_err(|source| {
            ArtifactOutputError::Serialize {
                destination: ArtifactDestination::Path(path.as_path().to_owned()),
                source,
            }
        })?;
        writer.flush().map_err(|source| path.flush_error(source))?;
    } else {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        serde_json::to_writer_pretty(&mut handle, value).map_err(|source| {
            ArtifactOutputError::Serialize {
                destination: ArtifactDestination::Stdout,
                source,
            }
        })?;
        writeln!(handle).map_err(|source| ArtifactOutputError::WriteStdout { source })?;
    }
    Ok(())
}

/// Human-readable JSON destination carried by serialization diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactDestination {
    /// Standard output.
    Stdout,
    /// A requested file path.
    Path(PathBuf),
}

impl Display for ArtifactDestination {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stdout => formatter.write_str("stdout"),
            Self::Path(path) => write!(formatter, "{}", path.display()),
        }
    }
}

/// Typed failures while validating or writing a CLI artifact destination.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ArtifactOutputError {
    /// An empty output path cannot identify a file.
    #[error("artifact path must not be empty")]
    EmptyPath,
    /// An artifact parent directory could not be created.
    #[error("failed to create artifact directory {path:?}: {source}")]
    CreateParent {
        /// Parent directory requested by the caller.
        path: PathBuf,
        /// Underlying operating-system error.
        #[source]
        source: io::Error,
    },
    /// An artifact file could not be opened for writing.
    #[error("failed to open artifact {path:?} for writing: {source}")]
    Open {
        /// Requested artifact path.
        path: PathBuf,
        /// Underlying operating-system error.
        #[source]
        source: io::Error,
    },
    /// JSON serialization failed for the selected destination.
    #[error("failed to serialize JSON to {destination}: {source}")]
    Serialize {
        /// Requested output destination.
        destination: ArtifactDestination,
        /// Underlying JSON serialization error.
        #[source]
        source: serde_json::Error,
    },
    /// A final artifact flush failed.
    #[error("failed to flush artifact {path:?}: {source}")]
    Flush {
        /// Requested artifact path.
        path: PathBuf,
        /// Underlying operating-system error.
        #[source]
        source: io::Error,
    },
    /// Writing the final newline to stdout failed.
    #[error("failed to write JSON to stdout: {source}")]
    WriteStdout {
        /// Underlying standard-output error.
        #[source]
        source: io::Error,
    },
}

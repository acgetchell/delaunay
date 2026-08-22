#![forbid(unsafe_code)]

//! Shared validated artifact destinations plus tests owned by the artifact binary.

#[path = "../shared/cli_output.rs"]
mod shared;

pub use shared::{ArtifactOutputError, ArtifactPath, write_json_output};

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{ArtifactOutputError, ArtifactPath};

    #[test]
    fn empty_artifact_path_is_rejected_before_storage() {
        let error = ArtifactPath::try_new(Path::new("").to_owned())
            .expect_err("empty artifact path should fail validation");
        std::assert_matches!(error, ArtifactOutputError::EmptyPath);
    }
}

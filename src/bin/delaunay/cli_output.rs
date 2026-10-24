#![forbid(unsafe_code)]

//! Shared validated artifact destinations plus tests owned by the artifact binary.

#[path = "../shared/cli_output.rs"]
mod shared;

pub use shared::{ArtifactOutputError, ArtifactPath, write_json_output};

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::{
        fs::{PermissionsExt, symlink},
        net::UnixListener,
    };
    use std::{fs, path::Path};

    use serde::{Serialize, Serializer, ser::SerializeSeq};

    use super::{
        ArtifactOutputError, ArtifactPath, shared::ArtifactDestination, write_json_output,
    };

    struct FailingJson;

    impl Serialize for FailingJson {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut sequence = serializer.serialize_seq(Some(2))?;
            // Exceed BufWriter's capacity so bytes reach the temporary file before
            // the later serialization failure.
            sequence.serialize_element(&"x".repeat(32 * 1024))?;
            Err(serde::ser::Error::custom("injected serialization failure"))
        }
    }

    struct JsonWithSideEffect<F>(F);

    impl<F: Fn()> Serialize for JsonWithSideEffect<F> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            (self.0)();
            serializer.serialize_str("replacement")
        }
    }

    #[test]
    fn empty_artifact_path_is_rejected_before_storage() {
        let error = ArtifactPath::try_new(Path::new("").to_owned())
            .expect_err("empty artifact path should fail validation");
        std::assert_matches!(error, ArtifactOutputError::EmptyPath);
    }

    #[test]
    fn serialization_failure_preserves_destination_and_cleans_temporary_file() {
        for existing in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let path = ArtifactPath::try_new(directory.path().join("output.json")).unwrap();
            if existing {
                fs::write(path.as_path(), b"original artifact").unwrap();
            }

            let error = write_json_output(&FailingJson, Some(&path)).unwrap_err();
            std::assert_matches!(
                error,
                ArtifactOutputError::Serialize {
                    destination: ArtifactDestination::Path(ref requested),
                    ref source,
                } if requested == path.as_path() && source.is_data()
            );
            if existing {
                assert_eq!(fs::read(path.as_path()).unwrap(), b"original artifact");
            } else {
                assert!(!path.as_path().exists());
            }
            assert_eq!(
                fs::read_dir(directory.path()).unwrap().count(),
                usize::from(existing)
            );
        }
    }

    #[test]
    fn json_publication_creates_parents_and_replaces_only_the_named_file() {
        let directory = tempfile::tempdir().unwrap();
        let parent = directory.path().join("new").join("nested");
        let path = ArtifactPath::try_new(parent.join("output.json")).unwrap();
        write_json_output(&[1, 2], Some(&path)).unwrap();
        let original = fs::read(path.as_path()).unwrap();
        assert_eq!(original, b"[\n  1,\n  2\n]");
        let alias = parent.join("original.json");
        fs::hard_link(path.as_path(), &alias).unwrap();

        write_json_output(&[3], Some(&path)).unwrap();

        assert_eq!(fs::read(path.as_path()).unwrap(), b"[\n  3\n]");
        assert_eq!(fs::read(alias).unwrap(), original);
        assert_eq!(fs::read_dir(parent).unwrap().count(), 2);
    }

    #[test]
    fn destination_is_rechecked_after_serialization() {
        let directory = tempfile::tempdir().unwrap();
        let path = ArtifactPath::try_new(directory.path().join("output.json")).unwrap();
        let value = JsonWithSideEffect(|| fs::create_dir(path.as_path()).unwrap());

        let error = write_json_output(&value, Some(&path)).unwrap_err();

        std::assert_matches!(error, ArtifactOutputError::UnsupportedDestination { .. });
        assert!(path.as_path().is_dir());
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn publication_failure_preserves_destination_and_cleans_temporary_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = ArtifactPath::try_new(directory.path().join("output.json")).unwrap();
        fs::write(path.as_path(), b"original artifact").unwrap();
        let value = JsonWithSideEffect(|| {
            // Removing the open temporary file deterministically makes the later
            // atomic rename fail, without changing the destination or permissions.
            let temporary = fs::read_dir(directory.path())
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .find(|entry| entry != path.as_path())
                .expect("serialization should use a separate temporary file");
            fs::remove_file(temporary).unwrap();
        });

        let error = write_json_output(&value, Some(&path)).unwrap_err();

        std::assert_matches!(
            error,
            ArtifactOutputError::Publish { ref path, ref source }
                if path.ends_with("output.json") && source.kind() == std::io::ErrorKind::NotFound
        );
        assert_eq!(fs::read(path.as_path()).unwrap(), b"original artifact");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn json_publication_preserves_existing_permissions() {
        let directory = tempfile::tempdir().unwrap();
        let path = ArtifactPath::try_new(directory.path().join("output.json")).unwrap();
        fs::write(path.as_path(), b"original artifact").unwrap();
        fs::set_permissions(path.as_path(), fs::Permissions::from_mode(0o640)).unwrap();

        write_json_output(&[3], Some(&path)).unwrap();

        assert_eq!(
            fs::metadata(path.as_path()).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    #[cfg(unix)]
    #[test]
    fn json_rejects_destination_symlinks_but_allows_symlinked_parents() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("target.json");
        fs::write(&file, b"original artifact").unwrap();
        let subdirectory = directory.path().join("directory");
        fs::create_dir(&subdirectory).unwrap();
        let missing = directory.path().join("missing.json");
        let path = ArtifactPath::try_new(directory.path().join("alias")).unwrap();
        for target in [&file, &subdirectory, &missing] {
            symlink(target, path.as_path()).unwrap();
            let error = write_json_output(&[3], Some(&path)).unwrap_err();
            std::assert_matches!(error, ArtifactOutputError::UnsupportedDestination { .. });
            assert_eq!(fs::read_link(path.as_path()).unwrap(), *target);
            fs::remove_file(path.as_path()).unwrap();
        }
        assert_eq!(fs::read(file).unwrap(), b"original artifact");
        assert!(!missing.exists());

        symlink(&subdirectory, path.as_path()).unwrap();
        let nested = ArtifactPath::try_new(path.as_path().join("output.json")).unwrap();
        write_json_output(&[3], Some(&nested)).unwrap();
        assert_eq!(
            fs::read(subdirectory.join("output.json")).unwrap(),
            b"[\n  3\n]"
        );
        assert_eq!(fs::read_dir(subdirectory).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn json_rejects_other_nonregular_files_before_serializing() {
        let directory = tempfile::tempdir().unwrap();
        let path = ArtifactPath::try_new(directory.path().join("socket")).unwrap();
        let _listener = UnixListener::bind(path.as_path()).unwrap();

        let error = write_json_output(&FailingJson, Some(&path)).unwrap_err();

        std::assert_matches!(error, ArtifactOutputError::UnsupportedDestination { .. });
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}

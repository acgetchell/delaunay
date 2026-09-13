//! Isolated downstream regression for checkpoint JSON without float-roundtrip features.

#![forbid(unsafe_code)]

#[cfg(unix)]
use std::{ffi::OsString, os::unix::ffi::OsStringExt};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// Preserve native checkout paths when preparing the isolated Cargo consumer.
fn downstream_command(repository: &Path) -> Command {
    let mut command = Command::new(env!("CARGO"));
    command
        .args(["run", "--quiet", "--offline", "--locked", "--manifest-path"])
        .arg(repository.join("tests/fixtures/checkpoint_no_float_roundtrip/Cargo.toml"))
        .env(
            "CARGO_TARGET_DIR",
            repository.join("target/checkpoint-no-float-roundtrip"),
        );
    command
}

#[test]
fn downstream_json_without_float_roundtrip_preserves_checkpoint_bits() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest = repository.join("tests/fixtures/checkpoint_no_float_roundtrip/Cargo.toml");
    if !manifest.is_file() {
        assert!(
            repository.join(".cargo_vcs_info.json").is_file(),
            "isolated downstream fixture is missing from the repository checkout: {}",
            manifest.display()
        );
        return;
    }
    let output = downstream_command(&repository)
        .output()
        .expect("isolated downstream fixture should run");
    assert!(
        output.status.success(),
        "fixture failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[test]
fn downstream_command_preserves_non_utf8_checkout_paths() {
    let repository = PathBuf::from(OsString::from_vec(b"checkout with spaces-\xff".to_vec()));
    let command = downstream_command(&repository);
    let manifest = repository.join("tests/fixtures/checkpoint_no_float_roundtrip/Cargo.toml");
    assert_eq!(command.get_args().last(), Some(manifest.as_os_str()));
}

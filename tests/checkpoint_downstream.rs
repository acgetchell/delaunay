//! Isolated downstream regression for checkpoint JSON without float-roundtrip features.

#![forbid(unsafe_code)]

use std::{path::PathBuf, process::Command};

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
    let target = repository.join("target/checkpoint-no-float-roundtrip");
    let output = Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--offline",
            "--locked",
            "--manifest-path",
            manifest.to_str().expect("fixture path should be UTF-8"),
        ])
        .env("CARGO_TARGET_DIR", target)
        .output()
        .expect("isolated downstream fixture should run");
    assert!(
        output.status.success(),
        "fixture failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

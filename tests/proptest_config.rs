#![forbid(unsafe_code)]

//! Integration tests for suite-specific Proptest case-count defaults.

#[macro_use]
#[path = "common/proptest_config.rs"]
mod proptest_config;

use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;
use proptest_config::{REPOSITORY_DEFAULT_CASES, repository_default, with_default_cases};
use std::process::Command;

const CHILD_EXPECTED_CASES: &str = "DELAUNAY_PROPTEST_CONFIG_CHILD_EXPECTED_CASES";
const LOCAL_DEFAULT_CASES: u32 = 12;

repo_proptest! {
    #[test]
    fn repository_wrapper_runs_with_shared_configuration(value in Just(42_u8)) {
        prop_assert_eq!(value, 42);
    }
}

/// Reads the repository declaration without adding a TOML parser dependency to
/// the integration-test support surface.
fn declared_repository_default_cases() -> u32 {
    include_str!("../proptest.toml")
        .lines()
        .find_map(|line| line.strip_prefix("cases = "))
        .expect("proptest.toml must declare `cases = <u32>`")
        .parse()
        .expect("the proptest.toml case count must be a valid u32")
}

/// Runs each precedence check in a fresh process so Proptest reads the intended
/// environment before its default configuration is cached.
fn assert_cases_in_isolated_process(
    test_name: &str,
    config: fn() -> ProptestConfig,
    proptest_cases: Option<&str>,
    expected_cases: u32,
) {
    if let Some(expected_cases) = std::env::var_os(CHILD_EXPECTED_CASES) {
        let expected_cases = expected_cases
            .to_string_lossy()
            .parse::<u32>()
            .expect("child expected case count must be a valid u32");
        assert_eq!(config().cases, expected_cases);
        return;
    }

    let test_executable =
        std::env::current_exe().expect("the current integration-test executable must exist");
    let mut command = Command::new(test_executable);
    command
        .arg("--exact")
        .arg(test_name)
        .env(CHILD_EXPECTED_CASES, expected_cases.to_string());
    if let Some(proptest_cases) = proptest_cases {
        command.env("PROPTEST_CASES", proptest_cases);
    } else {
        command.env_remove("PROPTEST_CASES");
    }

    let output = command
        .output()
        .expect("the isolated integration-test process must run");
    assert!(
        output.status.success(),
        "isolated case-count check failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn repository_default_matches_toml_declaration() {
    assert_eq!(
        REPOSITORY_DEFAULT_CASES,
        declared_repository_default_cases()
    );
}

#[test]
fn repository_default_applies_without_proptest_cases() {
    assert_cases_in_isolated_process(
        "repository_default_applies_without_proptest_cases",
        repository_default,
        None,
        REPOSITORY_DEFAULT_CASES,
    );
}

#[test]
fn suite_default_applies_without_proptest_cases() {
    assert_cases_in_isolated_process(
        "suite_default_applies_without_proptest_cases",
        || with_default_cases(LOCAL_DEFAULT_CASES),
        None,
        LOCAL_DEFAULT_CASES,
    );
}

#[test]
fn proptest_cases_takes_precedence_over_suite_default() {
    const OVERRIDE_CASES: u32 = 37;
    assert_cases_in_isolated_process(
        "proptest_cases_takes_precedence_over_suite_default",
        || with_default_cases(LOCAL_DEFAULT_CASES),
        Some("37"),
        OVERRIDE_CASES,
    );
}

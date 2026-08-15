#![forbid(unsafe_code)]

use proptest::test_runner::Config as ProptestConfig;

/// Repository-wide fallback for property tests without a narrower local budget.
pub const REPOSITORY_DEFAULT_CASES: u32 = 32;

/// Applies the repository fallback to an otherwise unconfigured `proptest!` block.
pub fn repository_default() -> ProptestConfig {
    with_default_cases(REPOSITORY_DEFAULT_CASES)
}

/// Preserves a suite-specific local default while honoring `PROPTEST_CASES`.
pub fn with_default_cases(default_cases: u32) -> ProptestConfig {
    let config = ProptestConfig::default();
    if std::env::var_os("PROPTEST_CASES").is_some() {
        config
    } else {
        ProptestConfig {
            cases: default_cases,
            ..config
        }
    }
}

/// Gives every integration property a repository fallback while preserving
/// explicit suite configuration and Proptest's environment overrides.
macro_rules! repo_proptest {
    (#![proptest_config($config:expr)] $($tokens:tt)*) => {
        ::proptest::proptest! {
            #![proptest_config($config)]
            $($tokens)*
        }
    };
    ($($tokens:tt)*) => {
        ::proptest::proptest! {
            #![proptest_config($crate::proptest_config::repository_default())]
            $($tokens)*
        }
    };
}

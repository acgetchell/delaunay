//! Shared benchmark setup helpers for fatal setup failures.
//!
//! Criterion benchmark targets cannot return [`Result`] from ordinary setup
//! helpers. These adapters keep benchmark setup code concise while preserving
//! the original error message from fallible constructors and setup routines.

use std::{fmt::Display, process};

#[cfg(feature = "bench-logging")]
use std::sync::Once;
#[cfg(feature = "bench-logging")]
use tracing_subscriber::EnvFilter;

#[cfg(feature = "bench-logging")]
fn init_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("error"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    });
}

/// Emits a benchmark setup failure through tracing, then exits.
#[cfg(feature = "bench-logging")]
pub fn abort_benchmark(message: impl Display) -> ! {
    init_tracing();
    tracing::error!("{message}");
    process::exit(1);
}

/// Prints a benchmark setup failure to standard error, then exits.
#[cfg(not(feature = "bench-logging"))]
pub fn abort_benchmark(message: impl Display) -> ! {
    eprintln!("{message}");
    process::exit(1);
}

/// Converts fallible [`Result`] benchmark setup values into abort-on-failure values.
pub trait OrAbort {
    /// The successful setup value.
    type Output;

    /// Returns the setup value or aborts the benchmark with the underlying error.
    fn or_abort(self) -> Self::Output;
}

impl<T, E: Display> OrAbort for Result<T, E> {
    type Output = T;

    fn or_abort(self) -> Self::Output {
        match self {
            Ok(value) => value,
            Err(error) => abort_benchmark(error),
        }
    }
}

/// Converts optional [`Option`] benchmark setup values into abort-on-missing values.
pub trait OrAbortWithContext {
    /// The successful setup value.
    type Output;

    /// Returns the setup value or aborts the benchmark with context.
    fn or_abort(self, context: impl Display) -> Self::Output;
}

impl<T> OrAbortWithContext for Option<T> {
    type Output = T;

    fn or_abort(self, context: impl Display) -> Self::Output {
        self.unwrap_or_else(|| abort_benchmark(context))
    }
}

//! Shared benchmark setup helpers for fatal setup failures.
//!
//! Criterion benchmark targets cannot return [`Result`] from ordinary setup
//! helpers. These adapters keep benchmark setup code concise while preserving
//! the original error message from fallible constructors and setup routines.

use std::{fmt::Display, process};

/// Prints fatal setup or measured-operation failures regardless of logging configuration.
pub fn abort_benchmark(message: impl Display) -> ! {
    eprintln!("benchmark failed: {message}");
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

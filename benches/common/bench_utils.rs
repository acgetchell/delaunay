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

/// Logs a benchmark setup failure when bench logging is enabled, then exits.
#[cfg(feature = "bench-logging")]
pub fn abort_benchmark(message: impl Display) -> ! {
    init_tracing();
    tracing::error!("{message}");
    process::exit(1);
}

/// Exits after a benchmark setup failure when bench logging is disabled.
#[cfg(not(feature = "bench-logging"))]
pub fn abort_benchmark(_message: impl Display) -> ! {
    process::exit(1);
}

/// Unwraps a benchmark setup result or aborts with context.
pub fn bench_result<T, E: Display>(result: Result<T, E>, context: impl Display) -> T {
    match result {
        Ok(value) => value,
        Err(error) => abort_benchmark(format_args!("{context}: {error}")),
    }
}

/// Unwraps a benchmark setup option or aborts with context.
pub fn bench_option<T>(option: Option<T>, context: impl Display) -> T {
    option.unwrap_or_else(|| abort_benchmark(context))
}

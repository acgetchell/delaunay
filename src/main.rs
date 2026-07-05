#![forbid(unsafe_code)]

//! Companion command-line entrypoint for notebook and diagnostic workflows.

mod config;

use std::process::ExitCode;

fn main() -> ExitCode {
    let command = match config::DelaunayCliArgs::from_args().into_validated() {
        Ok(command) => command,
        Err(error) => return config::exit_with_error(error),
    };

    command
        .run()
        .map_or_else(config::exit_with_error, |()| ExitCode::SUCCESS)
}

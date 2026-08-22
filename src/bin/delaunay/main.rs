#![forbid(unsafe_code)]
#![deny(dead_code_pub_in_binary)]

//! Companion command-line entrypoint for notebook and scriptable artifact workflows.

mod cli_output;
mod config;
mod generate;
mod spherical_hero;
mod validation_demo;

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

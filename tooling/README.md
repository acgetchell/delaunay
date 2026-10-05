# Tooling configuration

Declare reusable workflows here using the published `research-repo-tools==0.1.7`
contracts. Keep scientific logic in its owning Rust or Python module and
consumer policy checks in `scripts/tests/`.

`examples.toml` replaces the example runner script. Run it with `just examples`:
the shared validator builds the examples once in release mode, runs ordinary
examples with default features, then builds and runs the diagnostics example
with its feature enabled. Cargo uses the lockfile. Each build has a 1,800-second
deadline; each example has a 600-second deadline on every supported platform.
Successful commands print their captured output after completion. Expected
stdout markers guard against skipped workflows, including the diagnostics stub.

The timeout values live in TOML; the former `EXAMPLE_TIMEOUT` environment
override and optional coreutils dependency are retired. A consumer test compares
the declarations with Cargo metadata, so adding, removing, or renaming an
example requires updating this inventory. Add feature-specific build steps
before their corresponding execution checks.

Notebook configuration remains in `pyproject.toml` under
`tool.research-repo-tools.notebooks`. `just notebook-check` invokes shared
lint/advice directly and runs the consumer's lowercase kebab-case ID policy test.
There is no local notebook-check CLI or orchestration module.

The [adoption inventory](../docs/dev/tooling_adoption.md) records capabilities
still needed before performance, paper/PDF, hardware, and SARIF wrappers can be
deleted. Add configuration only when a published shared command consumes it;
do not move Python wrappers here merely to change their directory.

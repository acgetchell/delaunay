# Development Commands

Development commands and validation steps for the repository.

Agents must run appropriate checks after modifying code.

---

## Contents

- [Core Workflow](#core-workflow)
- [Justfile Usage](#justfile-usage)
- [Formatting](#formatting)
- [Linting](#linting)
- [Documentation Validation](#documentation-validation)
- [Full CI Validation](#full-ci-validation)
- [Benchmark Profiles](#benchmark-profiles)
- [Examples](#examples)
- [Spell Checking](#spell-checking)
- [TOML Formatting](#toml-formatting)
- [Shell Script Validation](#shell-script-validation)
- [JSON Validation](#json-validation)
- [GitHub Actions Validation](#github-actions-validation)
- [Recommended Command Matrix](#recommended-command-matrix)
- [CI Expectations](#ci-expectations)
- [Changelog](#changelog)

---

## Core Workflow

Typical development loop:

```bash
just fix
just check
just test
just ci
```

These commands ensure:

- formatting
- linting
- static analysis
- tests

## Justfile Usage

This repository standardizes development tasks through the `justfile`.

Agents should **prefer running `just` commands instead of invoking the
underlying tools directly**. The justfile ensures the correct flags,
configuration, and tool ordering are used.

Examples:

- prefer `just fix` instead of running `cargo fmt` directly
- prefer `just check` instead of running `cargo clippy` directly
- prefer `just ci` instead of manually running multiple validation steps

Direct tool invocation should only be used when a corresponding `just`
command does not exist.

---

## Formatting

Rust formatting:

```bash
cargo fmt
```

Typically run through:

```bash
just fix
```

Formatting must always be applied before committing changes.

---

## Linting

Lint checks include:

```bash
cargo clippy
```

Repository warnings are denied through the manifest lint policy in
`Cargo.toml`; explicitly configured lint exceptions remain warnings. Clippy
invocations also deny warnings.

Run via:

```bash
just check
```

`just check` is the non-mutating lint/validator bundle. It does not run tests,
examples, or benchmarks.

`just check` runs the default DenseSlotMap backend checks and an
`--all-features` pass. The justfile runs Clippy for the default feature set and
for `--all-features`; the legacy SlotMap backend is kept as an optional
compatibility canary. Run it explicitly with:

```bash
just check-storage-backends
```

---

## Documentation Validation

Documentation must build successfully.

Verify with:

```bash
just doc-check
```

or

```bash
cargo doc
```

---

## Full CI Validation

Before large changes, run the full CI command:

```bash
just ci
```

This runs:

- formatting checks
- lint checks
- benchmark and release-test compile checks using Cargo's default release profile
- unit tests
- integration tests
- documentation builds
- example builds

---

## Benchmark Profiles

`just ci` is the comprehensive error-catching validation path. It runs the
`check`, `test`, and `examples` recipes. The `test` recipe already depends on
`bench-test-compile`, so CI compiles benchmark harnesses and release tests
through that path; `bench-compile` is a standalone recipe that is **not**
executed by `just ci`:

```bash
just ci
just test              # includes just bench-test-compile
just bench-test-compile
```

Commands that run benchmarks and produce performance data use the `perf`
profile:

```bash
just bench
just bench-ci
just bench-baseline
just bench-compare
just bench-perf-summary
cargo bench --profile perf --bench ci_performance_suite
```

The `perf` profile inherits from release and restores ThinLTO with one codegen
unit. Use it for measured benchmark output; `just ci` does not need it to catch
compile, lint, test, documentation, example, or benchmark-harness build errors.
Use `just bench-smoke` only for quick harness validation with minimal samples;
do not treat smoke output as performance data.

---

## Examples

Example programs live in:

```text
examples/
```

Validate with:

```bash
just examples
```

Examples must:

- compile
- run successfully
- demonstrate correct API usage

---

## Spell Checking

Documentation and comments are spell‑checked.

Run:

```bash
just spell-check
```

If a legitimate technical word fails:

Add it to:

```text
typos.toml
```

under:

```toml
[default.extend-words]
```

---

## TOML Formatting

TOML files should be validated and formatted using Taplo.

Commands:

```bash
just toml-lint
just toml-fmt
```

---

## Shell Script Validation

Shell scripts must pass:

```text
shfmt
shellcheck
```

Run via CI or `just` commands.

---

## JSON Validation

JSON files should be validated after edits.

Example:

```bash
jq empty file.json
```

---

## GitHub Actions Validation

Workflows must pass `actionlint`.

Run with:

```bash
just action-lint
```

---

## Recommended Command Matrix

| Task | Command |
|-----|-----|
| Format code | `just fix` |
| Run lints | `just check` |
| Run tests + compile smoke | `just test` |
| Run unit/doc tests only | `just test-unit` |
| Run integration tests | `just test-integration` |
| Run all tests | `just test-all` |
| Run examples | `just examples` |
| Run full CI | `just ci` |
| Run perf-profile benchmarks | `just bench` |

---

## CI Expectations

CI enforces:

- formatting
- clippy lints
- documentation build
- tests

Rust warnings are denied by the manifest lint policy and Clippy warnings are
denied by the `just clippy` invocations. `just check-storage-backends` separately
checks the SlotMap backend with `--no-default-features`. Keep any intentional warning-level
exceptions explicit in `Cargo.toml`.

Agents must ensure changes pass CI locally before proposing patches.

---

## Changelog

The changelog is **auto-generated**.

Never edit manually.

Regenerate with:

```bash
just changelog
```

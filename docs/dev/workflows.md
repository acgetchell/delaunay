# Development Workflow

Development commands and validation steps for the repository.

Agents must run appropriate checks after modifying code.

---

## Core Workflow

Typical development loop:

```bash
just fix
just check
just test
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

Warnings are treated as errors in CI.

Run via:

```bash
just check
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
- unit tests
- integration tests
- documentation builds
- example builds

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
| Run unit tests | `just test` |
| Run integration tests | `just test-integration` |
| Run all tests | `just test-all` |
| Run examples | `just examples` |
| Run full CI | `just ci` |

---

## CI Expectations

CI enforces:

- formatting
- clippy lints
- documentation build
- tests

All warnings are treated as errors.

Agents must ensure changes pass CI locally before proposing patches.

---

## Changelog

The changelog is **auto-generated**.

Never edit manually.

Regenerate with:

```bash
just changelog
```

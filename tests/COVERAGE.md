# Coverage

This repository uses `cargo-llvm-cov` for local and CI coverage. Coverage runs
use Rust's LLVM source-based instrumentation and keep the coverage command in
the `justfile` so local runs and CI stay aligned.

## Local HTML

Generate the local developer report with:

```bash
just coverage
```

The HTML report is written to:

```text
target/llvm-cov/html/index.html
```

## CI XML

Generate the CI-compatible Cobertura report with:

```bash
just coverage-ci
```

The XML report is written to:

```text
coverage/cobertura.xml
```

The Codecov workflow installs Rust's `llvm-tools-preview` component, installs
`cargo-llvm-cov`, caches the installed cargo binary by version, runs
`just coverage-ci`, verifies `coverage/cobertura.xml`, uploads that file to
Codecov and Codacy, and archives the full `coverage/` directory.

## Coverage Surface

Both local and CI coverage use:

```bash
cargo llvm-cov --workspace --lib --tests
```

The common arguments also ignore `benches/` and `examples/` source paths in
reports. Integration tests still run and exercise library code, but coverage
metrics focus on the library implementation rather than benchmark, example, or
test harness source.

`just coverage-ci` runs through `cargo llvm-cov nextest` with the nextest
`coverage` profile. That single instrumented pass produces both
`coverage/cobertura.xml` and Codecov test-analytics JUnit XML. Property tests
remain part of the CI coverage run; their case budget is controlled by
`proptest.toml`, and longer deterministic coverage belongs behind the
`slow-tests` feature rather than being hidden from coverage.

Doc-test coverage remains intentionally disabled because `cargo-llvm-cov` marks
that path as unstable.

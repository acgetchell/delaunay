# Project Structure

This document maps the repository layout and top-level packaging shape. For
module ownership inside `src/`, see [`module_map.md`](module_map.md). For
development guidance, start with [`../dev/README.md`](../dev/README.md); for
developer commands, see [`../dev/commands.md`](../dev/commands.md).

## Directory Snapshot

The tree below is a human-maintained orientation aid, not a generated artifact.
Refresh it when files or major directories move.

```text
delaunay/
├── .cargo/
│   └── config.toml
├── .config/
│   └── nextest.toml
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── instructions/
│   └── workflows/
├── benches/
│   ├── common/
│   ├── allocation_hot_paths.rs
│   ├── boundary_uuid_iter.rs
│   ├── ci_performance_suite.rs
│   ├── circumsphere_containment.rs
│   ├── cold_path_predicates.rs
│   ├── delete_vertex.rs
│   ├── pachner_stress.rs
│   ├── pl_manifold_repair.rs
│   ├── profiling_suite.rs
│   ├── tds_clone.rs
│   └── topology_guarantee_construction.rs
├── docs/
│   ├── architecture/
│   │   ├── README.md
│   │   ├── module_map.md
│   │   ├── module_patterns.md
│   │   ├── prelude_reference.md
│   │   └── project_structure.md
│   ├── archive/
│   ├── dev/
│   ├── templates/
│   ├── api_design.md
│   ├── code_organization.md
│   ├── diagnostics.md
│   ├── invariants.md
│   ├── limitations.md
│   ├── numerical_robustness_guide.md
│   ├── property_testing_summary.md
│   ├── topology.md
│   ├── validation.md
│   └── workflows.md
├── examples/
├── scripts/
│   ├── ci/
│   ├── tests/
│   ├── archive_changelog.py
│   ├── benchmark_models.py
│   ├── benchmark_utils.py
│   ├── hardware_utils.py
│   ├── postprocess_changelog.py
│   ├── subprocess_utils.py
│   └── tag_release.py
├── src/
│   ├── core/
│   ├── delaunay/
│   ├── geometry/
│   ├── io/
│   ├── topology/
│   └── lib.rs
├── tests/
│   ├── semgrep/
│   ├── proptest_*.rs
│   ├── pachner_roundtrip.rs
│   ├── prelude_exports.rs
│   └── regressions.rs
├── AGENTS.md
├── Cargo.toml
├── Cargo.lock
├── README.md
├── REFERENCES.md
├── justfile
├── pyproject.toml
├── rust-toolchain.toml
├── rustfmt.toml
├── semgrep.yaml
└── uv.lock
```

To generate a full tree locally:

```bash
git --no-pager ls-files | LC_ALL=C sort | \
  LC_ALL=C tree -a --charset UTF-8 --dirsfirst --noreport \
    -I 'target|.git|**/*.png|**/*.svg' -F --fromfile
```

When `tree` is unavailable, use a read-only `find` command:

```bash
find . -type f \( -name "*.rs" -o -name "*.md" -o -name "*.toml" -o -name "*.yml" -o -name "*.yaml" \) | LC_ALL=C sort
```

## Top-Level Areas

- `src/` is the Rust library implementation. See
  [`module_map.md`](module_map.md) for ownership and layering.
- `tests/` contains integration tests, property tests, regression tests, and
  repository-owned Semgrep fixtures.
- `benches/` contains Criterion benchmark harnesses, shared benchmark fixtures,
  and performance-result documentation. Timing-based measurements belong here,
  not in unit tests.
- `examples/` contains user-facing API demos and workflow examples.
- `docs/` contains user documentation, contributor guidance, architecture
  references, archived design notes, and templates.
- `docs/dev/` contains operational rules for agents and contributors, indexed
  by [`../dev/README.md`](../dev/README.md).
- `docs/architecture/` contains focused architecture references.
- `scripts/` contains typed Python utilities for changelog, benchmark,
  hardware, SARIF, subprocess, and release workflows.
- `.github/` contains issue templates, workflow definitions, and
  repository-integrated automation.

## Packaging And Tooling Shape

- `Cargo.toml` uses an explicit package allowlist so crates.io artifacts carry
  the public library surface, examples, benchmarks, integration tests, and
  active documentation without bundling CI-only tooling or archived notes.
- `rust-toolchain.toml` pins the MSRV toolchain and uses a lean profile with
  only repository-required components.
- `pyproject.toml` owns Python support-tooling dependencies and validation
  configuration.
- `justfile` is the command entry point. Architecture docs should link to
  [`../dev/README.md`](../dev/README.md) or
  [`../dev/commands.md`](../dev/commands.md) rather than repeating command
  matrices.

## Special Purpose Areas

- `tests/semgrep/` mirrors repository-owned rule fixtures. Normal Semgrep scans
  exclude those fixture violations; `just semgrep-test` validates the rules.
- `docs/archive/` stores historical plans, completed changelog series, and old
  design notes. Do not update archived docs as active guidance unless an
  explicit archive-maintenance task asks for it.
- `baseline-artifact/` and `baseline-artifacts/` are ignored local benchmark
  baseline paths used by performance comparison tooling.

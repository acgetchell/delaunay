# delaunay

[![DOI](https://zenodo.org/badge/729897852.svg)](https://doi.org/10.5281/zenodo.16931097)
[![Crates.io](https://img.shields.io/crates/v/delaunay.svg)](https://crates.io/crates/delaunay)
[![Downloads](https://img.shields.io/crates/d/delaunay.svg)](https://crates.io/crates/delaunay)
[![License](https://img.shields.io/crates/l/delaunay.svg)](LICENSE)
[![Docs.rs](https://docs.rs/delaunay/badge.svg)](https://docs.rs/delaunay)
[![CI](https://github.com/acgetchell/delaunay/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/delaunay/actions/workflows/ci.yml)
[![rust-clippy analyze](https://github.com/acgetchell/delaunay/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/delaunay/actions/workflows/rust-clippy.yml)
[![codecov](https://codecov.io/gh/acgetchell/delaunay/graph/badge.svg?token=WT7qZGT9bO)](https://codecov.io/gh/acgetchell/delaunay)
[![Audit dependencies](https://github.com/acgetchell/delaunay/actions/workflows/audit.yml/badge.svg)](https://github.com/acgetchell/delaunay/actions/workflows/audit.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3cad94f994f5434d877ae77f0daee692)](https://app.codacy.com/gh/acgetchell/delaunay/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

D-dimensional Delaunay triangulations in [Rust], inspired by [CGAL].

## 📐 Introduction

This library implements d-dimensional Delaunay triangulations in [Rust]. It is
inspired by [CGAL], which is a [C++] library for computational geometry,
and [Spade], a [Rust] library that implements 2D [Delaunay triangulations],
[Constrained Delaunay triangulations], and [Voronoi diagrams]. The goal of this library is to provide a
lightweight alternative to [CGAL] for the [Rust] ecosystem.

## ✨ Features

- [x]  Copy-able data types associated with vertices and cells (integers, floats, chars, custom enums)
- [x]  d-dimensional [Delaunay triangulations]
- [x]  d-dimensional [Convex hulls]
- [x]  Geometry quality metrics for simplices: radius ratio and normalized volume (dimension-agnostic)
- [x]  Serialization/Deserialization of all data structures to/from [JSON]
- [x]  Tested for 2-, 3-, 4-, and 5-dimensional triangulations
- [x]  Local topology validation ([Pseudomanifold] default, [PL-manifold] opt-in)
- [x]  [Bistellar k-flips] for k = 1, 2, 3 plus inverse moves (repair uses k=2/k=3; inverse edge/triangle queues in 4D/5D)

See [CHANGELOG.md](CHANGELOG.md) for details.

## ⚠️ Delaunay Property

The triangulation uses flip-based [Delaunay repair] (k=2 facet queues, k=3 ridge queues,
and inverse edge/triangle queues in 4D/5D) after insertion by default via
`DelaunayRepairPolicy`. Flip-based repair **requires** `TopologyGuarantee::PLManifold`
(automatic repair is skipped under `Pseudomanifold`, and manual repair returns
`InvalidTopology`). This restores the local Delaunay property for most configurations
and preserves all structural invariants (TDS validity).

Repair is **bounded to two attempts**: attempt 1 uses FIFO ordering with fast predicates;
on non-convergence it retries once with LIFO ordering and robust predicates **only for
ambiguous boundary classifications**. If it still fails, the error includes diagnostics
(checked counts, ambiguous predicate samples, max queue depth, etc.). Highly degenerate
inputs or duplicate-handling edge cases can still require additional filtering.
For persistent failures, an **optional heuristic fallback** is available via
`repair_delaunay_with_flips_advanced`. This runs the standard two-pass repair, and
if it still fails, rebuilds the triangulation from the current vertex set using a
shuffled insertion order and a fresh perturbation seed, then runs a final flip-repair
pass. This fallback is heuristic and **non-reproducible by default**; the returned
`DelaunayRepairOutcome` includes the seeds used so you can replay the exact run.

```rust
use delaunay::prelude::*;

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
dt.set_topology_guarantee(TopologyGuarantee::PLManifold);

// Insert some vertices to build an initial triangulation.
// dt.insert(vertex!([0.0, 0.0, 0.0]))?;
// dt.insert(vertex!([1.0, 0.0, 0.0]))?;
// dt.insert(vertex!([0.0, 1.0, 0.0]))?;
// dt.insert(vertex!([0.0, 0.0, 1.0]))?;

let outcome = dt.repair_delaunay_with_flips_advanced(
    DelaunayRepairHeuristicConfig::default(),
)?;

if outcome.used_heuristic() {
    eprintln!("Heuristic rebuild used: {:?}", outcome.heuristic);
}
```

For details, see: [Issue #120 Investigation](docs/issue_120_investigation.md)

**Validation**: You can verify your triangulation meets your requirements using the library's
[4-level validation hierarchy](docs/validation.md):

- **Level 2** (`dt.tds().is_valid()`) - Structural correctness (expected to pass when using public APIs; not affected by Issue #120)
- **Level 3** (`dt.as_triangulation().is_valid()`) - Manifold topology + Euler characteristic
- **Level 4** (`dt.is_valid()`) - Delaunay property only (may fail if repair is disabled or non-convergent)
- **All levels (1–4)** (`dt.validate()`) - Elements + structure + topology + Delaunay property

Level 3 topology validation is parameterized by `TopologyGuarantee` (default: `Pseudomanifold`).
To enable stricter PL-manifold checks, set `TopologyGuarantee::PLManifold` (adds vertex-link validation).

During incremental insertion, the automatic Level 3 validation pass is controlled by
`ValidationPolicy` (default: `OnSuspicion`).

```rust
use delaunay::prelude::*;

let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

// Strictest topology checks (adds vertex-link validation):
dt.set_topology_guarantee(TopologyGuarantee::PLManifold);

// In tests/debugging, validate Level 3 after every insertion:
dt.set_validation_policy(ValidationPolicy::Always);
```

For applications requiring strict Delaunay guarantees:

- Keep `DelaunayRepairPolicy::EveryInsertion` (default) or call `repair_delaunay_with_flips()` after batch edits
- If standard repair does not converge, consider `repair_delaunay_with_flips_advanced()` for the heuristic rebuild fallback
- Use `dt.is_valid()` (Level 4 only) or `dt.validate()` (Levels 1–4) to check your triangulation
- Filter degenerate configurations when possible
- Monitor for additional flip types in future releases

## 🚧 Project History

This crate was originally maintained at [https://github.com/oovm/shape-rs](https://github.com/oovm/shape-rs) through version `0.1.0`.
The original implementation provided basic Delaunay triangulation functionality.

Starting with version `0.3.4`, maintenance transferred to [this repository](https://github.com/acgetchell/delaunay), which hosts a completely
rewritten d-dimensional implementation focused on computational geometry research applications.

- 📚 Docs for old versions (≤ 0.1.0): <https://docs.rs/delaunay/0.1.0/delaunay/>
- 📚 Docs for current version (≥ 0.3.4): <https://docs.rs/delaunay>

## 🤝 How to Contribute

We welcome contributions! Here's a 30-second quickstart:

```bash
# Clone and setup
git clone https://github.com/acgetchell/delaunay.git
cd delaunay

# Setup development environment (installs tools, builds project)
cargo install just
just setup            # Installs all development tools and dependencies

# Development workflow
just fix              # Apply formatters/auto-fixes (mutating)
just check            # Lint/validators (non-mutating)
just ci               # Full CI run (checks + all tests + examples + bench compile)
just ci-slow          # CI + slow tests (100+ vertices)
just --list           # See all available commands
just help-workflows   # Show common workflow patterns
```

**Try the examples:**

```bash
just examples         # Run all examples
# Or run specific examples:
cargo run --release --example triangulation_3d_20_points
cargo run --release --example convex_hull_3d_20_points
```

## 📋 Examples

The `examples/` directory contains several demonstrations:

- **`triangulation_3d_20_points`**: 3D Delaunay triangulation with a stable 20-point random configuration
- **`convex_hull_3d_20_points`**: 3D convex hull extraction and analysis on the same 20-point configuration
- **`into_from_conversions`**: Demonstrates Into/From trait conversions and utilities
- **`point_comparison_and_hashing`**: Demonstrates point comparison and hashing behavior
- **`memory_analysis`**: Memory usage analysis for triangulations across dimensions with allocation tracking
- **`zero_allocation_iterator_demo`**: Performance comparison between allocation and zero-allocation iterators

For detailed documentation, sample output, and usage instructions for each example, see [examples/README.md](examples/README.md).

For comprehensive guidelines on development environment setup,
testing, benchmarking, performance analysis, and development workflow, please see [CONTRIBUTING.md](CONTRIBUTING.md).

This includes information about:

- Building and testing the library
- Running benchmarks and performance analysis
- Code style and standards
- Submitting changes and pull requests
- Project structure and development tools

## 📖 Documentation

- **[Code Organization](docs/code_organization.md)** - Project structure and module patterns
- **[Topology integration design](docs/topology.md)** - Design notes on topology integration (includes historical sections)
- **[Validation Guide](docs/validation.md)** - Comprehensive 4-level validation hierarchy guide (element → structural → manifold → Delaunay)
- **[Issue #120 Investigation](docs/issue_120_investigation.md)** - Known Delaunay property limitations

## 📚 References

For a comprehensive list of academic references and bibliographic citations used throughout the library, see [REFERENCES.md](REFERENCES.md).

## 🤖 AI Agents

This repository contains an `AGENTS.md` file, which defines the canonical rules and invariants
for all AI coding assistants and autonomous agents working on this codebase.

AI tools (including ChatGPT, Claude, GitHub Copilot, Cursor, Warp, and CI repair agents) are
expected to read and follow `AGENTS.md` when proposing or applying changes.

Portions of this library were developed with the assistance of these AI tools:

- [ChatGPT]
- [Claude]
- [CodeRabbit]
- [GitHub Copilot]
- [KiloCode]
- [WARP]
>
> All code was written and/or reviewed and validated by the author.

[Rust]: https://rust-lang.org
[CGAL]: https://www.cgal.org/
[C++]: https://isocpp.org
[Spade]: https://crates.io/crates/spade
[JSON]: https://www.json.org/json-en.html
[Delaunay triangulations]: https://grokipedia.com/page/Delaunay_triangulation
[Constrained Delaunay triangulations]: https://grokipedia.com/page/Constrained_Delaunay_triangulation
[Voronoi diagrams]: https://grokipedia.com/page/Voronoi_diagram
[Convex hulls]: https://grokipedia.com/page/Convex_hull
[ChatGPT]: https://openai.com/chatgpt
[Claude]: https://www.anthropic.com/claude
[CodeRabbit]: https://coderabbit.ai/
[GitHub Copilot]: https://github.com/features/copilot
[KiloCode]: https://kilocode.ai/
[WARP]: https://www.warp.dev
[Pseudomanifold]: https://grokipedia.com/page/Pseudomanifold
[PL-manifold]: https://grokipedia.com/page/Piecewise_linear_manifold
[Delaunay repair]: https://link.springer.com/article/10.1007/BF01975867
[Bistellar k-flips]: https://grokipedia.com/page/Bistellar_flip

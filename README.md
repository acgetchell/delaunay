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

- [x]  Arbitrary data types associated with vertices and cells
- [x]  d-dimensional [Delaunay triangulations]
- [x]  d-dimensional [Convex hulls]
- [x]  Serialization/Deserialization of all data structures to/from [JSON]
- [x]  Tested for 2-, 3-, 4-, and 5-dimensional triangulations

See [CHANGELOG.md](CHANGELOG.md) for details.

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

# Build and test
cargo build
cargo test

# Format and lint
cargo fmt --all
cargo clippy --all-targets --all-features

# Run examples
./scripts/run_all_examples.sh

# Run a specific example
cargo run --example triangulation_3d_50_points
cargo run --example convex_hull_3d_50_points
cargo run --example into_from_conversions
cargo run --example point_comparison_and_hashing
```

## 📋 Examples

The `examples/` directory contains several demonstrations:

- **`triangulation_3d_50_points`**: Complete 3D Delaunay triangulation with 50 random points
- **`convex_hull_3d_50_points`**: 3D convex hull extraction and analysis with performance benchmarks
- **`into_from_conversions`**: Demonstrates Into/From trait conversions and utilities
- **`point_comparison_and_hashing`**: Demonstrates point comparison and hashing behavior

For comprehensive guidelines on development environment setup,
testing, benchmarking, performance analysis, and development workflow, please see [CONTRIBUTING.md](CONTRIBUTING.md).

This includes information about:

- Building and testing the library
- Running benchmarks and performance analysis
- Code style and standards
- Submitting changes and pull requests
- Project structure and development tools

## 📚 References

For a comprehensive list of academic references and bibliographic citations used throughout the library, see [REFERENCES.md](REFERENCES.md).

[Rust]: https://rust-lang.org
[CGAL]: https://www.cgal.org/
[C++]: https://isocpp.org
[Spade]: https://github.com/Stoeoef/spade
[JSON]: https://www.json.org/json-en.html
[Delaunay triangulations]: https://en.wikipedia.org/wiki/Delaunay_triangulation
[Constrained Delaunay triangulations]: https://en.wikipedia.org/wiki/Constrained_Delaunay_triangulation
[Voronoi diagrams]: https://en.wikipedia.org/wiki/Voronoi_diagram
[Convex hulls]: https://en.wikipedia.org/wiki/Convex_hull

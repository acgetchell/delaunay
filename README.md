# delaunay

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

## Introduction

This library implements d-dimensional Delaunay triangulations in [Rust]. It is
inspired by [CGAL], which is a [C++] library for computational geometry,
and [Spade], a [Rust] library implementing 2D [Delaunay triangulations],
[Constrained Delaunay triangulations], and [Voronoi diagrams]. The goal of this library is to provide a
lightweight alternative to [CGAL] for the [Rust] ecosystem.

## Features

- [x]  Arbitrary data types associated with vertices and cells
- [x]  d-dimensional [Delaunay triangulations]
- [x]  Serialization/Deserialization of all data structures to/from [JSON]
- [x]  Tested for 3-, 4-, and 5-dimensional triangulations

See [CHANGELOG.md](CHANGELOG.md) for details.

## 🚧 Project History

This crate was originally maintained at [https://github.com/oovm/shape-rs](https://github.com/oovm/shape-rs) through version `0.1.0`.
The original implementation provided basic Delaunay triangulation functionality.

Starting with version `0.3.3`, maintenance transferred to [this repository](https://github.com/acgetchell/delaunay), which hosts a completely
rewritten d-dimensional implementation focused on computational geometry research applications.

- 📚 Docs for old versions (≤ 0.1.0): <https://docs.rs/delaunay/0.1.0/delaunay/>
- 📚 Docs for current version (≥ 0.3.3): <https://docs.rs/delaunay>

## Development Tools

The repository includes utility scripts for development, testing, and performance analysis:

### Benchmarking and Performance

- **Automated benchmark execution**: Run triangulation benchmarks across multiple dimensions with GitHub Actions
- **Baseline generation**: Create performance baselines for regression detection
- **Performance regression testing**: Compare benchmark results against baseline metrics

See [benches/README.md](benches/README.md) for detailed performance results, baseline metrics, and analysis.

### Testing and Validation

- **Example execution**: Run all example programs to verify functionality
- **Comprehensive testing**: Automated discovery and execution of test suites

See [scripts/README.md](scripts/README.md) for detailed documentation on available development scripts and their usage.

## References

The library's geometric predicates and algorithms are based on established computational geometry literature:

### Circumcenter and Circumradius Calculations

- Lévy, Bruno, and Yang Liu. "Lp Centroidal Voronoi Tessellation and Its Applications." *ACM Transactions on Graphics* 29, no. 4 (July 26, 2010):
  119:1-119:11. DOI: [10.1145/1778765.1778856](https://doi.org/10.1145/1778765.1778856)

### Robust Geometric Predicates

- Shewchuk, J. R. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates." *Discrete & Computational Geometry* 18,
  no. 3 (1997): 305-363. DOI: [10.1007/PL00009321](https://doi.org/10.1007/PL00009321)
- Shewchuk, J. R. "Robust Adaptive Floating-Point Geometric Predicates." *Proceedings of the Twelfth Annual Symposium on Computational Geometry* (1996): 141-150.

### Lifted Paraboloid Method

- Preparata, Franco P., and Michael Ian Shamos. "Computational Geometry: An Introduction." Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
- Edelsbrunner, Herbert. "Algorithms in Combinatorial Geometry." EATCS Monographs on Theoretical Computer Science. Berlin: Springer-Verlag, 1987.

### Triangulation Data Structures and Algorithms

- [CGAL Triangulation Documentation](https://doc.cgal.org/latest/Triangulation/index.html)
- Bowyer, A. "Computing Dirichlet tessellations." *The Computer Journal* 24, no. 2 (1981): 162-166. DOI: [10.1093/comjnl/24.2.162](https://doi.org/10.1093/comjnl/24.2.162)
- Watson, D.F. "Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes." *The Computer Journal* 24, no. 2 (1981):
  167-172. DOI: [10.1093/comjnl/24.2.167](https://doi.org/10.1093/comjnl/24.2.167)
- de Berg, M., et al. "Computational Geometry: Algorithms and Applications." 3rd ed. Berlin: Springer-Verlag, 2008. DOI: [10.1007/978-3-540-77974-2](https://doi.org/10.1007/978-3-540-77974-2)

[Rust]: https://rust-lang.org
[CGAL]: https://www.cgal.org/
[C++]: https://isocpp.org
[Spade]: https://github.com/Stoeoef/spade
[JSON]: https://www.json.org/json-en.html
[Delaunay triangulations]: https://en.wikipedia.org/wiki/Delaunay_triangulation
[Constrained Delaunay triangulations]: https://en.wikipedia.org/wiki/Constrained_Delaunay_triangulation
[Voronoi diagrams]: https://en.wikipedia.org/wiki/Voronoi_diagram

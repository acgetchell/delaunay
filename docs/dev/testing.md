# Testing Guidelines

Testing rules for the Delaunay triangulation library.

Agents must follow these expectations when adding or modifying Rust code.

---

## Testing Philosophy

This project is a **scientific computational geometry library**.

Tests should verify:

- mathematical correctness
- geometric invariants
- topological consistency
- algorithm stability

When possible, prefer **property-based testing** over single-case tests.

Tests should focus on validating invariants rather than merely executing code.

---

## Test Types

The project uses several categories of tests.

### Unit Tests

Location:

```text
src/**
```

Defined inline using:

```rust
#[cfg(test)]
mod tests {
```

Unit tests validate:

- small internal algorithms
- helper utilities
- invariants within modules

They should be small, deterministic, and fast.

---

### Integration Tests

Location:

```text
tests/
```

Integration tests compile as **separate crates** and test the public API.

Each integration test crate should include a crate-level documentation comment:

```rust
//! Integration tests for triangulation invariants.
```

This satisfies `clippy::missing_docs` in CI.

Integration tests should validate:

- full triangulation construction
- public API behavior
- cross-module interactions

---

### Property Tests

Property tests are strongly preferred for geometric structures.

The project uses the **proptest** crate.

Example pattern:

```rust
proptest! {
    #[test]
    fn triangulation_is_valid(points in point_cloud_strategy()) {
        let tri = build_triangulation(points);
        assert!(tri.is_valid());
    }
}
```

Property tests should validate invariants rather than specific outputs.

Typical invariants include:

- Euler characteristic
- simplex adjacency consistency
- vertex-star topology
- manifold link conditions
- orientation predicate correctness

---

## Floating-Point Comparisons

Never compare floating-point values using `assert_eq!`.

Use the **approx** crate for tolerant comparisons.

Preferred macros:

```rust
use approx::{assert_relative_eq, assert_abs_diff_eq};

assert_relative_eq!(a, b, epsilon = 1e-12);
```

Floating-point arithmetic is not exact and direct equality comparisons will
produce fragile tests.

For geometric predicates you might also allow **ULP comparisons** from the same crate:

```rust
assert_ulps_eq!(a, b, max_ulps = 4);
```

---

## Degenerate Geometry

Tests should include degenerate or near-degenerate configurations.

Important cases include:

- duplicate vertices
- collinear points
- coplanar point sets
- nearly coincident points
- extremely large coordinate values
- extremely small coordinate values

Robust geometry code must handle these cases gracefully.

---

## Dimension Coverage (2D–5D)

This library supports d-dimensional triangulations. Tests for
dimension-generic code **must cover 2D through 5D** whenever possible.

### Use macros for per-dimension test generation

Define a macro that accepts a dimension literal and generates the full set
of test functions for that dimension. Invoke it once per dimension:

```rust
macro_rules! gen_tests {
    ($dim:literal) => {
        pastey::paste! {
            #[test]
            fn [<test_foo_ $dim d>]() {
                let points = build_points::<$dim>();
                // assertions …
            }
        }
    };
}

gen_tests!(2);
gen_tests!(3);
gen_tests!(4);
gen_tests!(5);
```

### Keep core logic in generic helper functions

The macro body should be thin — primarily calling generic helpers and
asserting results. Dimension-specific point construction, translation, and
other setup belongs in `const`-generic helper functions:

```rust
fn build_degenerate_points<const D: usize>() -> Vec<Point<f64, D>> { … }
fn translate_point<const D: usize>(p: &Point<f64, D>) -> Point<f64, D> { … }
```

This keeps the macro readable and the helpers independently testable.

### Reference examples

- Unit tests: `src/geometry/sos.rs` — `gen_sos_dim_tests!`
- Property tests: `tests/proptest_sos.rs` — `gen_sos_tests!`

### When single-dimension tests are acceptable

Some tests are inherently dimension-specific (e.g. 1D edge cases,
matrix-level tests for a fixed size, error-handling tests). These do not
need macro-ification.

---

## Deterministic Randomness

Tests must be deterministic.

If randomness is required, use a seeded RNG.

Example:

```rust
use rand::{SeedableRng, rngs::StdRng};

let rng = StdRng::seed_from_u64(1234);
```

Do **not** use:

```rust
thread_rng()
```

Deterministic seeds allow failures to be reproduced.

---

## Error Handling in Tests

Tests may freely use `unwrap()` or `expect()` when a failure should cause the
 test to fail immediately.

Examples:

```rust
let tri = build_triangulation(points).unwrap();
```

or

```rust
let tri = build_triangulation(points)
    .expect("triangulation construction failed");
```

Explicit error handling is usually unnecessary in tests unless the test is
specifically verifying error behavior.

Clippy's `unwrap_used` lint may be relaxed or allowed in test code when
appropriate.

---

## Triangulation Validation

Whenever possible, prefer validating triangulations using invariant checks.

Example:

```rust
assert!(tri.is_valid());
```

Validation helpers are preferred over writing manual assertions about
internal state.

Tests should verify structural correctness of the triangulation.

---

## Core Geometry Invariants

The formal triangulation invariants are defined in:

```text
docs/invariants.md
```

Tests should verify behavior consistent with that specification.

For details on validation helpers such as `tri.is_valid()`, see:

```text
docs/validation.md
```

Tests should prefer calling these validation helpers instead of
re‑implementing invariant logic.

Tests should verify core invariants such as:

- every simplex references valid vertices
- adjacency relationships are symmetric
- vertex stars are topologically consistent
- no duplicate simplices exist
- Euler characteristic is correct
- orientation predicates produce consistent signs

## Triangulation Validity Checklist

When writing tests that construct or modify a triangulation, agents should
prefer validating the following checklist rather than writing ad‑hoc
assertions:

- `tri.is_valid()` returns true
- every simplex references existing vertices
- adjacency relationships are symmetric
- vertex stars form closed topological neighborhoods
- no duplicate simplices exist
- orientation predicates are consistent across neighbors

Whenever possible, prefer a single invariant validation call (e.g.
`tri.is_valid()`) rather than duplicating these checks manually.

Invariant-based testing is the most reliable way to validate geometric
algorithms.

---

## Test Commands

Tests should pass using the repository command set.

Run standard tests:

```bash
just test
```

Run integration tests:

```bash
just test-integration
```

Run all tests:

```bash
just test-all
```

---

## Documentation Tests

Public documentation examples must compile.

Validate with:

```bash
just doc-check
```

or:

```bash
cargo test --doc
```

---

## Performance-Sensitive Tests

Tests should remain fast.

Avoid:

- extremely large random inputs
- quadratic or worse scaling test loops
- heavy allocations

Large-scale performance validation belongs in **benchmarks**, not tests.

---

## CI Expectations

All tests must pass under CI.

Before proposing patches agents should run:

```bash
just ci
```

CI enforces:

- formatting
- linting
- documentation builds
- unit tests
- integration tests

---

## Preferred Test Style

Tests should be:

- deterministic
- focused
- invariant-driven
- easy to reproduce

Avoid large monolithic tests or tests that do not verify correctness.

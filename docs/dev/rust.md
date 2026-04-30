# Rust Development Guidelines

Rust coding conventions for this repository.

Agents must follow these rules when modifying or adding Rust code.

---

## Contents

- [Core Principles](#core-principles)
- [Safety](#safety)
- [Dimension Generic Architecture](#dimension-generic-architecture)
- [Numeric Conversions](#numeric-conversions)
- [Borrowing and Ownership](#borrowing-and-ownership)
- [Error Handling](#error-handling)
- [Panic Policy](#panic-policy)
- [Error Types](#error-types)
  - [Orthogonal variants](#orthogonal-variants)
  - [Struct‑with‑named‑fields throughout](#structwithnamedfields-throughout)
  - [Preserve typed sources — no boxing, no `dyn Error`](#preserve-typed-sources--no-boxing-no-dyn-error)
  - [Do not stringify; carry typed context instead](#do-not-stringify-carry-typed-context-instead)
  - [Derive `Clone, Debug, Error, PartialEq, Eq`](#derive-clone-debug-error-partialeq-eq)
- [Naming and Paths](#naming-and-paths)
- [Imports](#imports)
- [Module Layout](#module-layout)
- [Prelude Design](#prelude-design)
- [Documentation](#documentation)
- [Integration Tests](#integration-tests)
- [Testing Expectations](#testing-expectations)
- [Performance](#performance)
- [External Dependencies](#external-dependencies)
- [Formatting and Lints](#formatting-and-lints)
- [API Stability](#api-stability)
- [Logging and Diagnostics](#logging-and-diagnostics)
- [Preferred Patch Style](#preferred-patch-style)

---

## Core Principles

This project is a **scientific computational geometry library**.

Key goals:

- Correctness
- Predictable performance
- API stability
- Zero unsafe code
- Dimension-generic architecture

All design decisions should prioritize these goals.

---

## Safety

Unsafe Rust is forbidden.

The crate enforces:

```rust
#![forbid(unsafe_code)]
```

Agents must never introduce:

- `unsafe`
- `unsafe fn`
- `unsafe impl`
- `unsafe` blocks

---

## Dimension Generic Architecture

The library is generic over dimension using const generics:

```rust
const D: usize
```

Code must remain compatible with:

- 2D
- 3D
- 4D
- 5D

Avoid hard‑coding dimension assumptions unless they are explicitly isolated.

Prefer patterns like:

```rust
struct Point<const D: usize> {
    coords: [f64; D],
}
```

Algorithms should operate generically over `D` whenever practical.

---

## Numeric Conversions

Avoid unchecked numeric casts in geometry, topology, tests, and benchmarks when
precision or range can matter.

Prefer repository helpers from `crate::geometry::util`, for example:

- `safe_usize_to_scalar::<T>(value)`
- `safe_scalar_to_f64(value)`
- `safe_scalar_from_f64::<T>(value)`
- `safe_coords_to_f64(coords)`
- `safe_coords_from_f64::<T, D>(coords)`

Do not silence `clippy::cast_precision_loss` with `#[expect(...)]` simply
because the current values are small. Use a safe conversion helper and handle
or justify the `Result` at the call site. A lint expectation is appropriate only
when no safe conversion applies and the invariant is documented in the code.

Avoid fallback conversions such as `unwrap_or(f64::NAN)`,
`unwrap_or(f64::INFINITY)`, or silently clamping failed conversions. These hide
the numerical state that geometric predicates and validation layers need in
order to fail explicitly.

---

## Borrowing and Ownership

Prefer **borrowing APIs** whenever possible.

### Function arguments

Prefer:

```rust
fn foo(points: &[Point<D>])
```

Instead of:

```rust
fn foo(points: Vec<Point<D>>)
```

### Return values

Prefer borrowed results:

```rust
fn vertex(&self, key: VertexKey) -> Option<&Vertex<D>>
```

Avoid unnecessary allocations.

Public APIs should also avoid unnecessary cloning. Prefer returning references
or iterators over internal data instead of cloning structures.

Avoid patterns like:

```rust
fn vertices(&self) -> Vec<Vertex<D>> {
    self.vertices.clone()
}
```

Prefer borrowed views instead:

```rust
fn vertices(&self) -> &[Vertex<D>] {
    &self.vertices
}
```

Cloning large structures in public APIs can introduce hidden performance
costs and should only be done when ownership transfer is required.

Only return owned values (`Vec`, `String`, etc.) when necessary.

---

## Error Handling

Public APIs must **not panic**.

Use explicit error propagation.

### Fallible public functions

Return `Result`:

```rust
pub fn insert_vertex(...) -> Result<VertexKey, InsertError>
```

### Lookup functions

Return `Option`:

```rust
pub fn vertex(&self, key: VertexKey) -> Option<&Vertex<D>>
```

### Infallible APIs

These should return values directly:

Infallible functions **must not return `Result`**.
If a function cannot fail under normal operation, it should return its value
directly rather than wrapping it in `Result`. Returning `Result` from
infallible APIs is considered unidiomatic and unnecessarily complicates
callers.

- `len()`
- `is_empty()`
- iterators
- accessors
- builder setters

Example:

```rust
pub fn len(&self) -> usize
```

Examples of infallible APIs include:

- accessors (`len`, `dimension`, `capacity`)
- iterators and views
- builder setters
- simple queries over internal state

If a function may fail due to invalid input or algorithmic conditions, it
should return `Result`. If the value may or may not exist (e.g. lookup by key),
return `Option`.

Do not introduce artificial error types simply to satisfy a `Result` return type.

### Builder pattern

Builder setters return `Self`.

Errors occur in `build()`.

Example:

```rust
builder
    .with_capacity(100)
    .with_seed(seed)
    .build()?;
```

---

## Panic Policy

Panics should be avoided in library code.

Acceptable panic situations:

- internal invariants violated
- unreachable logic errors
- debugging assertions

Prefer returning:

- `Result`
- `Option`

instead of panicking.

---

## Error Types

Errors should be defined **within the module where they are used**.

Avoid large centralized error enums.

Example:

```rust
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertError {
    #[error("duplicate vertex")]
    DuplicateVertex,
}
```

The sub‑sections below spell out the conventions that keep error values
**debuggable, composable, and stable**. They apply to every new error enum
and to edits of existing ones.

### Orthogonal variants

Every variant represents a **distinct failure mode**. Two variants must not
overlap in meaning: if a caller can't decide which one to match on, the
taxonomy is wrong.

When the same underlying condition occurs in two different contexts
(e.g. primary failure vs. failure during fallback), model it with
**separate variants that each carry the full typed context**, not with a
single variant and a free‑form `context: String` field.

Good:

```rust
pub enum DelaunayizeError {
    TopologyRepairFailed {
        source: PlManifoldRepairError,
    },
    TopologyRepairFailedWithRebuild {
        source: PlManifoldRepairError,
        rebuild_error: DelaunayTriangulationConstructionError,
    },
    DelaunayRepairFailed {
        source: DelaunayRepairError,
    },
    DelaunayRepairFailedWithRebuild {
        source: DelaunayRepairError,
        rebuild_error: DelaunayTriangulationConstructionError,
    },
}
```

Each pair `Failed` / `FailedWithRebuild` is **orthogonal**: the caller
always knows whether a fallback was attempted, and if so which specific
rebuild error was produced.

### Struct‑with‑named‑fields throughout

Prefer **struct variants with named fields** over positional (tuple) variants,
even for single‑field carriers. Named fields:

- document the semantics of each payload at the declaration site,
- keep `Display` format strings readable (`{source}`, `{rebuild_error}`),
- let downstream code pattern‑match by field name without caring about
  positional order,
- remain additive: adding a new field is a compile‑error surface that
  forces callers to consider it.

Prefer:

```rust
#[error("Invalid facet index {index} for cell with {facet_count} facets")]
InvalidFacetIndex {
    index: u8,
    facet_count: usize,
},
```

Avoid:

```rust
#[error("Invalid facet index {0} for cell with {1} facets")]
InvalidFacetIndex(u8, usize),
```

### Preserve typed sources — no boxing, no `dyn Error`

Source and "secondary" errors must be stored **by value as typed enums**,
not as `Box<dyn Error>`, not as `anyhow::Error`, and not stringified into a
`message: String` field. The whole point of the taxonomy is that consumers
can pattern-match the full structured error, while [`Error::source`]
exposes whichever field is annotated as the primary source.

- Use `#[source]` (and `#[from]` where the conversion is unambiguous) on
  the typed field so `thiserror` wires up the source chain.
- Use `Box<T>` only when the **typed** payload would make the enum
  unbalanced in size (e.g. `NonConvergent` carries a fat diagnostics
  struct); the inner type is still fully typed.
- Never replace a typed error with a `String` just because the enum lived
  in a different crate — that erases variant and source information.

```rust
// Good: typed rebuild error preserved by value; primary source chain intact.
TopologyRepairFailedWithRebuild {
    #[source]
    source: PlManifoldRepairError,
    rebuild_error: DelaunayTriangulationConstructionError,
},
```

```rust
// Bad: stringification erases the typed variant.
TopologyRepairFailedWithRebuild {
    source: PlManifoldRepairError,
    rebuild_message: String,
},
```

### Do not stringify; carry typed context instead

Free‑form `message: String` fields are only acceptable when the context is
genuinely unstructured prose (rare). In practice, **most** "context" is
structured — indices, counts, keys, UUIDs, other enums — and belongs in
named fields of a struct variant.

Prefer:

```rust
#[error("Ridge indices ({omit_a}, {omit_b}) out of bounds for cell {cell_key:?} with {vertex_count} vertices")]
InvalidRidgeIndex {
    cell_key: CellKey,
    omit_a: u8,
    omit_b: u8,
    vertex_count: usize,
},
```

Avoid:

```rust
#[error("Ridge indices out of bounds: {message}")]
InvalidRidgeIndex {
    message: String,
},
```

Structured payloads support:

- test assertions via `assert_eq!` / `matches!` without string parsing,
- diagnostic tools that filter or aggregate by field,
- localization and richer `Display` implementations without rewriting
  call‑sites.

### Derive `Clone, Debug, Error, PartialEq, Eq`

All error enums should derive the standard set:

```rust
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FooError { ... }
```

- `Clone` — lets callers attach the error to multiple diagnostics paths
  and lets tests construct expected values once and compare them.
- `Debug` — required for `Error`.
- `thiserror::Error` — wires up `Display` and `source()`.
- `PartialEq, Eq` — deriveable whenever all payload types are `Eq`
  (integers, strings, UUIDs, keys, other `Eq` enums, `Arc<T>` /
  `Box<T>` where `T: Eq`). All error enums in this crate satisfy
  this today. Skip these only when a payload genuinely cannot be `Eq`
  (e.g. `f64`, `io::Error`, `Box<dyn Error>`) — none of which belong in
  error values anyway.
- `#[non_exhaustive]` — new variants must remain additive; downstream
  matches need a `_` arm.

Use `assert_eq!` for fixed‑shape variants in tests; prefer `matches!` for
"just check the variant" when the payload contains long free‑form strings
or nondeterministic samples.

---

## Naming and Paths

Function names should be concise but specific. Prefer short verbs and domain
terms over names that restate the module, type, or every implementation detail.

Prefer:

```rust
fn align_offsets(...)
fn validate_link(...)
fn rebuild_candidate(...)
```

Avoid:

```rust
fn align_periodic_vertex_offsets_for_source_cell_to_target_cell(...)
fn validate_manifold_link_consistency_for_all_ridges(...)
fn rebuild_delaunay_triangulation_candidate_after_repair_failure(...)
```

Use short, unqualified paths inside function bodies. If a function needs a type,
trait, constant, or helper from another module, import it at the top of the
module and refer to the item by its short name locally.

---

## Imports

Always import types at the top of the module rather than using fully‑qualified
paths inline. This keeps code readable and consistent.

Prefer:

```rust
use crate::core::tds::TdsError;

fn check(err: &TdsError) -> bool { ... }
```

Instead of:

```rust
fn check(err: &crate::core::tds::TdsError) -> bool { ... }
```

Group imports from the same module into a single `use` statement with braces:

```rust
use crate::core::tds::{
    CellKey, EntityKind, Tds, TdsError, VertexKey,
};
```

Do not add `use` statements inside function bodies just to shorten a path.
Move those imports to the top of the module. Local imports are acceptable only
when they are intentionally scoped for conditional compilation, tests, macro
expansion, or to avoid a documented name collision.

If a test module already has `use super::*;`, do not re‑import items that are
already brought into scope by the parent module's imports.

---

## Module Layout

Never use `mod.rs`.

Modules are declared from `src/lib.rs`.

Example:

```rust
pub mod core;
pub mod geometry;
pub mod algorithms;
```

Nested modules may use inline declaration:

```rust
pub mod core {
    pub mod triangulation;
    pub mod vertex;
}
```

---

## Prelude Design

Focused preludes should remain **small and purpose-specific**.

A focused prelude should import only the items needed for a specific task.
Prefer these focused preludes in doctests, integration tests, examples, and
benchmarks because they make intent visible at the import site.

Examples:

```text
delaunay::prelude::triangulation
delaunay::prelude::triangulation::flips
delaunay::prelude::triangulation::repair
delaunay::prelude::triangulation::delaunayize
delaunay::prelude::query
delaunay::prelude::geometry
delaunay::prelude::generators
delaunay::prelude::ordering
delaunay::prelude::collections
delaunay::prelude::tds
delaunay::prelude::topology::validation
delaunay::prelude::topology::spaces
```

The root `delaunay::prelude::*` is intentionally available as a convenience for
new users and quick experiments. Avoid using it in committed examples,
benchmarks, and doctests when a focused prelude communicates the workflow more
clearly.

---

## Documentation

All public items must have documentation. Public functions must include a
doctest in their documentation.

Example:

```rust
/// Inserts a vertex into the triangulation.
///
/// Returns the key of the inserted vertex.
///
/// # Examples
///
/// ```rust
/// # use delaunay::prelude::triangulation::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut triangulation = DelaunayTriangulation::<_, _, _, 2>::default();
/// let key = triangulation.insert_vertex([0.0, 0.0])?;
/// assert!(triangulation.contains_vertex(key));
/// # Ok(())
/// # }
/// ```
pub fn insert_vertex(...)
```

### Private functions

Private functions must have a brief doc comment (`///`) explaining **why they
exist** — what problem they solve or what invariant they maintain. The *what*
is often clear from the signature; the *why* is not.

Prefer:

```rust
/// Aligns source-cell periodic offsets into the target-cell frame so
/// cross-cell insphere predicates see consistent lifted coordinates.
fn align_periodic_offset<const D: usize>(...) -> Result<[i8; D], FlipError>
```

Use normal comments (`//`) for documentation inside function bodies or other
implementation-local notes:

```rust
fn align_periodic_offset<const D: usize>(...) -> Result<[i8; D], FlipError> {
    // Compare deltas in each coordinate so conflicting frame translations are
    // rejected before lifted coordinates are constructed.
    ...
}
```

A bare signature with no context forces readers to reverse-engineer
intent from the implementation.

After Rust changes, verify documentation builds:

```bash
just doc-check
```

or

```bash
cargo doc
```

---

## Integration Tests

Integration tests live in:

```text
tests/
```

Each integration test crate should include a crate‑level doc comment:

```rust
//! Integration tests for triangulation invariants.
```

This satisfies `clippy::missing_docs` in CI.

Fixed-bug regression integration tests belong in `tests/regressions.rs` unless
they need separate crate-level configuration, feature flags, or profile
isolation.

---

## Testing Expectations

Rust changes should be validated with:

```bash
just test
just test-integration
```

Property tests are preferred for geometric invariants such as:

- Euler characteristic checks
- simplex adjacency invariants
- manifold consistency

---

## Performance

Avoid unnecessary allocations.

Prefer:

- iterators
- slices
- stack arrays `[T; D]`
- fixed‑size containers

Avoid cloning large structures unless necessary.

---

## External Dependencies

Dependencies should be minimal.

Before adding a dependency, consider:

1. compile time impact
2. MSRV compatibility
3. maintenance status
4. dependency tree size

---

## Formatting and Lints

Code must pass:

```bash
cargo fmt
cargo clippy
```

Typically run via:

```bash
just fix
just check
```

CI treats warnings as errors.

### Lint Suppression

When suppressing a lint, use `#[expect(...)]` instead of `#[allow(...)]`.

`expect` causes a compiler warning if the lint is no longer triggered,
ensuring suppressions are removed when they become unnecessary.

Always include a `reason`:

```rust
#[expect(clippy::too_many_lines, reason = "test covers multiple cases")]
fn test_large_dataset_performance() { ... }
```

---

## API Stability

The crate is intended for external use.

Agents must avoid:

- breaking public APIs
- renaming public types
- removing public functions

If an API change is necessary, prefer:

```rust
#[deprecated]
```

with migration guidance.

---

## Logging and Diagnostics

Use `tracing` for committed diagnostics across production code, tests,
and benchmarks. This includes library/runtime code, non-trivial test
diagnostics, and debugging of numerical instability or topological
invariants. Prefer `tracing::debug!`, `tracing::trace!`, etc. over
ad-hoc printing.

This ensures all diagnostic output is:

- filterable via `RUST_LOG` / `tracing-subscriber`
- structured and machine-parseable
- suppressible in production builds

`eprintln!` is acceptable only for short-lived local debugging while
investigating an issue. Do not leave it in committed code when `tracing`
or a typed error path is more appropriate.

Debug hooks gated on environment variables should still use `tracing`:

```rust
#[cfg(debug_assertions)]
if std::env::var_os("DELAUNAY_DEBUG_FOO").is_some() {
    tracing::debug!("diagnostic message: {value}");
}
```

### Tests and Benchmarks

- Use `tracing` for non-trivial test diagnostics rather than
  `eprintln!`, especially when diagnosing geometric predicate behavior,
  invariant failures, or shrink/reproduction context.
- Never log inside hot benchmark loops or Criterion-measured closures.
  Emit diagnostics before or after the measured path so measurements stay
  meaningful.
- Gate non-essential test and benchmark diagnostics behind feature flags.
  In this repository, use `test-debug` for test diagnostics and
  `bench-logging` for benchmark diagnostics:

```rust
#[cfg(feature = "test-debug")]
tracing::debug!("test diagnostic");

#[cfg(feature = "bench-logging")]
tracing::debug!("benchmark diagnostic");
```

---

## Preferred Patch Style

When modifying Rust code:

- make **small focused changes**
- avoid large refactors
- maintain existing naming conventions
- preserve module boundaries

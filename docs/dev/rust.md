# Rust Development Guidelines

Rust coding conventions for this repository.

Agents must follow these rules when modifying or adding Rust code.

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
#[derive(Debug, thiserror::Error)]
pub enum InsertError {
    #[error("duplicate vertex")]
    DuplicateVertex,
}
```

---

## Imports

Always import types at the top of the module rather than using fully‑qualified
paths inline. This keeps code readable and consistent.

Prefer:

```rust
use crate::core::triangulation_data_structure::TdsValidationError;

fn check(err: &TdsValidationError) -> bool { ... }
```

Instead of:

```rust
fn check(err: &crate::core::triangulation_data_structure::TdsValidationError) -> bool { ... }
```

Group imports from the same module into a single `use` statement with braces:

```rust
use crate::core::triangulation_data_structure::{
    CellKey, EntityKind, Tds, TdsValidationError, VertexKey,
};
```

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

Preludes should remain **small and purpose‑specific**.

A prelude should import only the items needed for a specific task.

Example:

```text
delaunay::prelude::triangulation
```

Avoid giant catch‑all preludes.

---

## Documentation

All public items must have documentation.

Example:

```rust
/// Inserts a vertex into the triangulation.
///
/// Returns the key of the inserted vertex.
pub fn insert_vertex(...)
```

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

Use `tracing` for all runtime diagnostics. **Never use `eprintln!`** or
`println!` for debug output — use `tracing::debug!`, `tracing::trace!`, etc.

This ensures all diagnostic output is:

- filterable via `RUST_LOG` / `tracing-subscriber`
- structured and machine-parseable
- suppressible in production builds

Debug hooks gated on environment variables should still use `tracing`:

```rust
#[cfg(debug_assertions)]
if std::env::var_os("DELAUNAY_DEBUG_FOO").is_some() {
    tracing::debug!("diagnostic message: {value}");
}
```

---

## Preferred Patch Style

When modifying Rust code:

- make **small focused changes**
- avoid large refactors
- maintain existing naming conventions
- preserve module boundaries

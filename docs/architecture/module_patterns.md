# Module Organization Patterns

This document records common file-internal Rust organization patterns. For
ownership of modules under `src/`, see [`module_map.md`](module_map.md).

## Canonical Section Sequence

Large modules generally follow this order:

1. Module documentation (`//!`).
2. Imports.
3. Error types.
4. Convenience macros and helper functions.
5. Struct definitions.
6. Deserialization implementations.
7. Core implementation blocks.
8. Advanced implementation blocks with specialized trait bounds.
9. Standard trait implementations.
10. Specialized trait implementations such as `Hash` and equality contracts.
11. Tests.

Small modules can compress this sequence, but should preserve the same relative
ordering where the sections exist.

## Comment Separators

Use this separator for major sections in large modules:

```rust
// =============================================================================
// SECTION NAME
// =============================================================================
```

Within test modules, use the indented form for major subsections:

```rust
    // =============================================================================
    // SUBSECTION NAME TESTS
    // =============================================================================
```

## Imports

Organize imports at module scope. Prefer:

1. `super::` imports.
2. `crate::` imports.
3. External crate imports.
4. Standard library imports.

Move imports to the top of the module instead of adding function-local imports,
unless the import is intentionally scoped for conditional compilation, tests,
macro expansion, or a documented name collision.

## Error Types

Error enums should follow the repository conventions in
[`../dev/rust.md`](../dev/rust.md):

- derive `Clone`, `Debug`, `thiserror::Error`, `PartialEq`, and `Eq` when
  payloads allow it.
- use `#[non_exhaustive]` for public error enums.
- prefer struct variants with named fields.
- carry typed context instead of free-form strings.
- preserve typed sources instead of erasing errors behind `dyn Error`.

## Implementation Blocks

Keep implementation blocks grouped by trait-bound requirements. Basic
operations should live in the least-constrained impl block that can support
them; methods requiring stronger numeric, topology, or data-trait bounds should
be isolated in a later specialized impl block.

## Tests

Within a `#[cfg(test)] mod tests { ... }` block, use this order:

1. `use` imports.
2. Test-only types.
3. Helper functions.
4. Macros.
5. `#[test]` functions and `proptest!` blocks.

Keep helper functions above the tests that use them, and prefer
dimension-generic helpers plus thin per-dimension macros when the behavior
should cover 2D through 5D.

## Module-Specific Notes

- `simplex.rs` is a large module with extensive geometric predicate
  integration and detailed equality/hash contract tests.
- `vertex.rs` centers coordinate validation, equality, data preservation, and
  serialization behavior.
- `facet.rs` centers codimension-1 face relationships, key derivation, and
  adjacency behavior.
- `facet_incidence.rs` is trait-implementation focused and supports TDS
  one-sided/two-sided incidence analysis.
- `src/core/util/` modules are function-focused. Keep utility concerns split by
  purpose rather than recreating a single broad `util.rs`.

## Documentation Standards

- Use `///` for item documentation and `//!` for module documentation.
- Include examples in public API documentation.
- Include `# Errors`, `# Panics`, and `# Safety` sections where applicable.
- Reference other types with intra-doc links where possible.
- Private helpers should explain why they exist or what invariant they
  maintain when that is not obvious from the signature.

# Phase 3A: Key-based Cell and Facet Refactor (Branch-only)

This document tracks the Phase 3A refactor to store keys in Cell and Facet (similar to Vertex), remove UUID neighbor usage, and redesign
serialization to reconstruct keys during deserialization. This is a branch-only effort; backward compatibility is not required for this phase.

**Status**: 🔄 In progress

**Started**: 2025-10-11

## Guiding Constraints

- [x] Branch-only: Breaking changes are allowed in this branch
- [ ] No dual storage: Do not keep both object- and key-based representations simultaneously
- [ ] Serde strategy: Do not serialize VertexKey/CellKey; reconstruct keys on deserialization
- [x] Update Cell and Facet in the same style as Vertex (Vertex already done)

## High-Level Goals

- [ ] Refactor Cell to key-based storage
- [ ] Refactor Facet to key-based storage
- [ ] Eliminate UUID (neighbor) usage in core types
- [ ] Provide Tds-based accessors to retrieve objects from keys
- [ ] Redesign serialization/deserialization to be keyless on the wire
- [ ] Update builders, macros, validation, and trait impls
- [ ] Update tests, examples, and docs
- [ ] Validate with quality, lint, and build steps

---

## Task Checklist

### 1) Baseline in the working branch

- [ ] Confirm current branch and ensure all changes are isolated from main
- [ ] Run baseline checks and tests
  - [ ] `just test`
  - [ ] `just doc-check`
  - [ ] `just clippy`

### 2) Type and dependency confirmations

- [x] Ensure VertexKey and CellKey are defined and stable
- [x] Confirm `SmallBuffer<T, 8>` (or equivalent) is available and appropriate
- [ ] Remove uuid dependency usage from Cell and Facet (keep for now, remove later)

### 3) Cell struct refactor (src/core/cell.rs)

- [ ] Replace `vertices: Vec<Vertex<T, U, D>>` with `vertex_keys: SmallBuffer<VertexKey, 8>`
- [ ] Replace `neighbors: Option<Vec<Option<Uuid>>>` with `neighbor_keys: Option<SmallBuffer<Option<CellKey>, 8>>`
- [ ] Maintain existing generics: `T: CoordinateScalar`, `U: DataType` (vertex data), `V: DataType` (cell data), `const D: usize`
- [ ] Update Default/constructors and internal invariants to align with key-based fields

### 4) Facet refactor (src/core/facet.rs)

- [ ] Replace any Vertex or Cell object storage with key-based fields
- [ ] Ensure Facet construction and comparison logic work with keys
- [ ] Provide Tds-based access where object references are required

### 5) Tds access layer updates

- [ ] Add/get APIs to map `VertexKey -> &Vertex<T, U, D>` and `CellKey -> &Cell<T, U, V, D>`
- [ ] Add creation APIs returning keys when inserting vertices/cells
- [ ] Provide neighbor set/get by key (no UUIDs)

### 6) Cell methods rewritten to operate via keys

- [ ] Replace direct vertex access with key-based lookups (require `&Tds` where needed)
- [ ] Ensure all neighbor operations use `CellKey`
- [ ] Update any orientation/ordering or facet derivations to key-based logic
- [ ] Validate invariants: no duplicate vertex keys, correct arity D+1, consistent neighbor counts when present

### 7) Update the `cell!` macro and CellBuilder

- [ ] `cell!` macro accepts keys (VertexKey, CellKey) only
- [ ] Provide friendly compile errors if object types are passed
- [ ] Refactor `CellBuilder` to be key-native
- [ ] Consider providing helper constructors: from vertex keys; from vertex indices via Tds; etc.

### 8) Trait implementations for key-based types

- [ ] Update `Debug` to print keys and helpful summaries
- [ ] Update `PartialEq/Eq` based on key identity and structural equivalence
- [ ] Update `Hash` to hash a canonicalized representation (stable order of vertex keys)
- [ ] Update `Clone`, `Ord/PartialOrd` if applicable

### 9) Validation and utilities

- [ ] Update validation methods to verify key existence in Tds and structural correctness
- [ ] Update facet/adjacency utilities to use keys throughout
- [ ] Ensure boundary conditions and convex hull checks work with keys

### 10) Serialization/deserialization redesign (keyless on the wire)

- [ ] Define new schema (breaking change allowed in this branch)
  - [ ] Do not serialize `VertexKey` or `CellKey`
  - [ ] Serialize vertices as an array of payloads (point + U)
  - [ ] Serialize cells as:
    - [ ] Vertex indices (usize) referring into the serialized vertex array, in canonical order
    - [ ] Neighbor references as cell indices (usize) or `None`
    - [ ] Cell payloads (V) and any required metadata if present
- [ ] Deserialization plan:
  - [ ] Recreate vertices in Tds in serialized order and collect new `VertexKey`s
  - [ ] Create cells using the mapped `VertexKey`s (collect new `CellKey`s)
  - [ ] Resolve neighbor relationships using cell indices mapped to newly created `CellKey`s
- [ ] Add schema/version marker to the serialized form and update crate docs accordingly

### 11) Tests and verification

- [ ] Unit tests for Cell key-based storage/API
- [ ] Unit tests for Facet key-based storage/API
- [ ] Tests for Tds lookups and neighbor wiring
- [ ] Serde round-trip tests: ensure keys are reconstructed and structure is preserved
- [ ] Property tests where feasible (e.g., random triangulations maintain invariants)

### 12) Update call sites and examples

- [ ] Refactor all call sites in core to the new key-based APIs
- [ ] Update examples to construct with keys (and/or through Tds helpers)
- [ ] Update benchmarks to compile with new APIs (do not run full benches yet)

### 13) Documentation updates

- [ ] Update `docs/code_organization.md` to reflect the new key-based design
- [ ] Document serde schema changes and migration notes
- [ ] Update any public API docs and module-level docs
- [ ] Ensure `just doc-check` passes (crates.io doc publishing constraints)

### 14) Quality and validation passes

- [ ] `just fmt`
- [ ] `just clippy`
- [ ] `just doc-check`
- [ ] `just markdown-lint`
- [ ] `just spell-check`
- [ ] `just validate-toml`
- [ ] `just validate-json`
- [ ] `just test` (and `just test-release` when appropriate)
- [ ] `just bench-compile` (compile-only verification for benches)

### 15) Cleanup

- [ ] Remove uuid dependency if no longer used anywhere
- [ ] Remove obsolete code paths and deprecated APIs
- [ ] Open follow-up issues for remaining migrations (outside cell/facet), performance, and documentation polish

---

## Notes and Acceptance Criteria

- All core operations in Cell and Facet must function using keys with object access via Tds only.
- No UUIDs in Cell/Facet storage; neighbors use `Option<CellKey>`.
- Serde does not persist keys; keys are deterministically reconstructed on load.
- Tests and examples must compile and pass in this branch.
- Documentation updated prior to any potential publishing.

## Progress Log

### 2025-10-11

- ✅ Created Phase 3A tracking document
- ✅ Confirmed Vertex already migrated to use `CellKey` for `incident_cell`
- 🔄 Ready to begin Cell refactoring

---

## Related Documentation

- `docs/OPTIMIZATION_ROADMAP.md` - Overall Phase 1-4 optimization plan
- `docs/Cell_refactoring.md` - Original architectural design document
- `src/core/vertex.rs` - Example of key-based storage pattern (already implemented)

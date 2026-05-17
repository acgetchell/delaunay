# Coherent Orientation Technical Specification

This document describes the current coherent-orientation machinery in the
`delaunay` crate. It is an implementation-focused companion to
[`invariants.md`](invariants.md), which gives the broader mathematical rationale
for the library's topological and geometric invariants.

Keep these documents separate:

- `invariants.md` explains the whole invariant model: simplicial complexes,
  PL-manifold checks, Delaunay validation, and convergence rationale.
- `ORIENTATION_SPEC.md` documents one invariant in detail: how simplex orderings,
  facet parity, and geometric signs are maintained in the implementation.

## Orientation Invariants

The crate maintains two related but distinct orientation properties.

### Coherent Combinatorial Orientation

At the TDS layer, neighboring simplices must induce compatible orientations on their
shared facet. This is a structural invariant over stored vertex orderings and
neighbor slots; it does not require coordinate predicates.

Implementation entry points:

- [`Tds::is_coherently_oriented`](../src/core/tds.rs) returns `true` when all
  non-periodic adjacent simplex pairs satisfy the facet-parity convention.
- `Tds::validate_coherent_orientation()` is called by `Tds::is_valid()` and by
  `Tds::validation_report()` as `InvariantKind::CoherentOrientation`.
- `TdsError::OrientationViolation` reports the first incoherent adjacent pair,
  including both facet orderings and the observed/expected permutation parity.
- `Tds::normalize_coherent_orientation()` computes a BFS flip assignment over
  each connected component and swaps vertex slots `0 <-> 1` for simplices that must
  flip.

Periodic image-point triangulations are special: periodic-lifted adjacencies do
not have a single canonical structural orientation independent of the chosen
lattice representative. The TDS orientation validator still checks reciprocal
neighbor wiring, but skips combinatorial facet-parity checks when either simplex in
the adjacent pair carries periodic offsets.

### Positive Geometric Simplex Orientation

At the `Triangulation` layer, stored simplices are promoted toward the positive
geometric orientation convention using exact orientation predicates. This is a
coordinate-dependent invariant.

Implementation entry points:

- `Triangulation::validate_geometric_simplex_orientation()` evaluates each simplex
  with `robust_orientation` and rejects negative signs as
  `GeometricError::NegativeOrientation`.
- `Triangulation::normalize_and_promote_positive_orientation()` first runs
  coherent-orientation normalization, then canonicalizes the global sign, then
  performs a bounded per-simplex promotion fallback for near-degenerate edge cases.
- `Triangulation::canonicalize_positive_orientation_for_simplices()` is used on
  newly inserted simplices so incremental insertion does not leave negative simplices in
  the mutation frontier.
- `Triangulation::validate_geometric_nondegeneracy()` is stricter: it rejects
  zero-determinant simplices. Explicit user-provided simplex construction calls this
  after structural/topological validation.

`validate_geometric_simplex_orientation()` rejects negative signs, but it does not
treat an exact zero determinant as a sign mismatch. That distinction matters:
construction and explicit-builder paths reject truly degenerate initial or
user-provided simplices, while some repair paths can encounter near-degenerate
intermediate simplices where coherent combinatorial orientation remains the primary
safety property.

## Mathematical Background

A `D`-simplex with vertices `v_0, v_1, ..., v_D` has an orientation determined
by the sign of the homogeneous determinant:

```text
| x_0,0  x_0,1  ...  x_0,D-1  1 |
| x_1,0  x_1,1  ...  x_1,D-1  1 |
|  ...    ...   ...    ...     1 |
| x_D,0  x_D,1  ...  x_D,D-1  1 |
```

- Positive orientation: determinant > 0.
- Negative orientation: determinant < 0.
- Degenerate orientation: determinant = 0.

Swapping any two vertices flips the orientation sign. Even permutations preserve
the sign; odd permutations reverse it.

The implementation uses [`robust_orientation`](../src/geometry/robust_predicates.rs)
for orientation canonicalization and validation. This is intentional:
`AdaptiveKernel::orientation()` applies Simulation of Simplicity (SoS) and may
return a deterministic ±1 for degenerate finite inputs. Simplex orientation
maintenance needs the true geometric sign, including `DEGENERATE`, so it uses
the robust predicate directly and bypasses SoS.

Dimension notes:

- The f64 fast filter is available through D ≤ 4.
- Exact orientation determinants are available through D ≤ 6 because the
  orientation matrix is `(D + 1) × (D + 1)` and the stack-matrix limit is 7×7.
- The routine triangulation envelope remains 2D through 5D; see
  [`limitations.md`](limitations.md) for scale and predicate limits.

## Facet-Parity Convention

For adjacent simplices, the shared facet is represented by the vertex ordering
obtained from each simplex after omitting the opposite vertex slot.

The coherent boundary-orientation convention is not simply "the two facet
orders must always be an odd permutation." The omitted slot contributes a sign,
so the expected parity depends on both facet indices:

```rust
let expected_odd_permutation = (facet_idx + mirror_idx).is_multiple_of(2);
```

`Tds::facet_permutation_parity()` computes:

- `observed_odd_permutation`: whether the permutation from the first simplex's
  facet ordering to the neighbor simplex's facet ordering is odd.
- `expected_odd_permutation`: the parity required by the facet-index convention.
- `currently_coherent`: whether the observed and expected parities match.

For ordinary Euclidean simplices, facet identities are just `VertexKey`s in
simplex-local order. For periodic simplices, the helper can build lifted identities
`(VertexKey, offset)`, normalize offsets by a deterministic anchor, and compare
them when a periodic-aware caller needs lifted facet identity.

## Implementation Map

### TDS Layer

`src/core/tds.rs` owns the combinatorial invariant:

- `TdsError::OrientationViolation`
- `InvariantKind::CoherentOrientation`
- `Tds::is_coherently_oriented()`
- `Tds::normalize_coherent_orientation()`
- `Tds::validate_coherent_orientation_for_simplices(simplices)`
- `Tds::facet_permutation_parity(...)`
- `Tds::permutation_is_odd(...)`

`Tds::normalize_coherent_orientation()` walks each connected component of the
simplex-neighbor graph. It assigns a boolean "flip this simplex" state by propagating
facet-parity constraints. If it finds contradictory constraints, it returns
`TdsError::InconsistentDataStructure`; otherwise it swaps vertex slots `0` and
`1` in every simplex assigned `true`.

`Tds::is_valid()` includes coherent orientation as a Level 2 structural check
after simplex vertex keys, duplicate simplices, facet sharing, and neighbor consistency
have been validated. The validation report records it as
`InvariantKind::CoherentOrientation`.

### Triangulation Layer

`src/core/triangulation.rs` owns geometric sign handling:

- Initial simplex construction calls `robust_orientation`, rejects
  `DEGENERATE`, and swaps two vertex keys when the sign is negative.
- Incremental cavity filling inserts new simplices, then calls
  `canonicalize_positive_orientation_for_simplices(&new_simplices)`.
- Local insertion validation calls both
  `tds.validate_coherent_orientation_for_simplices(simplices)` and
  `validate_geometric_simplex_orientation_for_simplices(simplices)`.
- Whole-triangulation topology validation runs `validate_topology_core()` and
  then `validate_geometric_simplex_orientation()`.
- Explicit combinatorial construction normalizes orientation, validates TDS
  structure, validates topology, and then calls
  `validate_geometric_nondegeneracy()` before enforcing the Delaunay property.

The geometric orientation path is exact-sign based and kernel-independent. It
does not use `Kernel::orientation()` because that method may apply SoS depending
on the selected kernel.

### Flip Operations

`src/core/algorithms/flips.rs` preserves orientation during bistellar flips:

- `apply_bistellar_flip_with_k()` asserts coherent orientation before and after
  the trial mutation in debug builds.
- Candidate replacement simplices are checked with `robust_orientation`.
- `Orientation::DEGENERATE` becomes `FlipError::DegenerateSimplex`.
- `Orientation::NEGATIVE` swaps two vertex slots before insertion.
- `orient_replacement_simplices()` chooses replacement-simplex parity from the oriented
  cavity boundary and internal replacement-simplex adjacencies.
- `ReplacementOrientationPolicy::RequirePositive` validates positive geometric
  orientation for Delaunay-repair replacement simplices.
- The flip commits only after full-TDS or local-cavity validation succeeds.

The flip code is generic over the scalar type, not over a kernel, for orientation
canonicalization. Kernels are needed by the Delaunay violation predicates that
drive repair, but replacement-simplex orientation itself uses `robust_orientation`.

### Builder Paths

`src/triangulation/builder.rs` normalizes explicit and periodic construction:

- `from_vertices_and_simplices(...)` accepts user-provided simplex orderings, assembles
  the TDS, calls `normalize_and_promote_positive_orientation()`, validates TDS
  structure/topology, rejects geometrically degenerate simplices, and then enforces
  the Delaunay property.
- `.toroidal_periodic([..])` builds an image-point triangulation and then runs
  orientation normalization plus lifted geometric orientation validation before
  returning the quotient triangulation.

## Degenerate Simplices

Degeneracy handling depends on context:

- Initial simplex construction rejects `Orientation::DEGENERATE`.
- Explicit user-provided simplices are rejected by
  `validate_geometric_nondegeneracy()`.
- Flip replacement simplices are rejected as `FlipError::DegenerateSimplex`.
- `validate_geometric_simplex_orientation()` does not reject zero determinants by
  itself; it only rejects negative signs. This allows topology diagnostics and
  repair paths to distinguish "wrong sign" from "geometrically collapsed."

When adding a new construction or mutation path, decide explicitly whether a
zero determinant should be rejected immediately, allowed temporarily for repair,
or surfaced through a typed validation error.

## Testing

Focused orientation coverage lives in:

- [`tests/proptest_orientation.rs`](../tests/proptest_orientation.rs): property
  tests for construction coherence, serialized tamper detection, and incremental
  insertion coherence.
- [`tests/triangulation_builder.rs`](../tests/triangulation_builder.rs):
  explicit construction normalizes incoherent user simplex orderings.
- [`tests/regressions.rs`](../tests/regressions.rs): regression coverage for the
  4D bulk-repair orientation cleanup that previously left negative simplices.
- Unit tests inside `src/core/tds.rs`, `src/core/triangulation.rs`, and
  `src/core/algorithms/flips.rs`: parity helpers, normalization behavior,
  geometric-orientation diagnostics, and flip replacement orientation.

## Performance Notes

Orientation maintenance is deliberately scoped:

- Full coherent-orientation validation is O(simplices × D³): each simplex has O(D)
  facets, each facet parity check uses O(D²) inversion counting.
- Local mutation paths validate only the simplices touched by insertion or repair
  when they can do so safely.
- Geometric sign checks use exact predicates and can be much more expensive than
  raw f64 orientation on near-degenerate inputs, especially for D ≥ 5 where the
  f64 fast filter is unavailable.
- Debug assertions around flips compile out in release builds, but the actual
  replacement-simplex orientation checks are part of correctness and remain active.

Do not add full-TDS orientation validation inside hot loops unless the caller is
explicitly requesting validation or diagnostic output.

## Contributor Checklist

When adding or changing a path that creates, removes, rewires, or reorders simplices:

- Preserve simplex-local vertex/neighbor/periodic-offset slot alignment when
  swapping vertex slots.
- Use `robust_orientation`, not `Kernel::orientation()`, when detecting
  geometric simplex sign or degeneracy.
- Normalize coherent orientation after assembling simplices whose relative parity is
  not already fixed by construction.
- Validate the local mutation frontier with
  `validate_coherent_orientation_for_simplices` and geometric orientation checks
  when the path can avoid a full-TDS scan.
- Use typed errors: `TdsError::OrientationViolation` for combinatorial
  incoherence, `GeometricError::NegativeOrientation` for negative geometric
  simplices, and `FlipError::DegenerateSimplex` for degenerate flip replacements.
- Add tests that tamper with simplex orderings or exercise the new mutation path in
  at least the dimensions affected by the change.

## References

- CGAL Triangulation documentation:
  <https://doc.cgal.org/latest/Triangulation/index.html>
- Orientation predicates: [`src/geometry/predicates.rs`](../src/geometry/predicates.rs)
  and [`src/geometry/robust_predicates.rs`](../src/geometry/robust_predicates.rs)
- TDS orientation implementation: [`src/core/tds.rs`](../src/core/tds.rs)
- Geometric orientation promotion: [`src/core/triangulation.rs`](../src/core/triangulation.rs)
- Flip replacement orientation: [`src/core/algorithms/flips.rs`](../src/core/algorithms/flips.rs)
- Permutation parity: <https://en.wikipedia.org/wiki/Parity_of_a_permutation>

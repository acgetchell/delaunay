# Scientific Basis

This overview connects the crate's scientific scope and methods to their detailed
contracts and supporting literature. Start with the [README](../README.md) to
evaluate the crate and run a first example; use [REFERENCES](../REFERENCES.md)
for bibliographic records and stable citation keys.

## Contents

- [API selection and scope](#api-selection-and-scope)
- [Shared assumptions and validation layers](#shared-assumptions-and-validation-layers)
- [Construction and editing methods](#construction-and-editing-methods)
- [Numerical and realization methods](#numerical-and-realization-methods)
- [Correctness evidence and performance measurements](#correctness-evidence-and-performance-measurements)

## API selection and scope

| Goal or model | Entry point | Scientific scope and detailed owner |
|---|---|---|
| Euclidean construction | `DelaunayTriangulationBuilder` | Finite `f64` point sets, routine 2D–5D coverage: [contracts][construction] |
| Generic triangulation/editing | `Triangulation`, `PachnerMoves` | Topology and realization separate from Delaunay: [API design](api_design.md) |
| Periodic tori | `DelaunayTriangulationBuilder::try_toroidal` | Image-point quotients on `T^2` and compact `T^3`: [limits][domains] |
| Spheres | `SphericalDelaunayBuilder` | Bounded `S^2`/`S^3` prototype in ambient `R^3`/`R^4`: [topology](topology.md) |
| Stored connectivity | `Tds`, then checked promotion | Levels 1–2 alone do not prove a manifold, realization, or Delaunay: [promotion][construction] |

[construction]: construction_and_validation.md
[domains]: limitations.md#topology-and-domain-limits

Use `DelaunayIncrementalBuilder` for incremental construction. Toroidal
`T^4`/`T^5` construction fails fast. The spherical prototype checks topology,
realization, and empty-cap predicates separately.

The [workflow guide](USING_TRIANGULATIONS.md) and [runnable examples](../examples/README.md)
teach API composition. The [import selector](https://docs.rs/delaunay/latest/delaunay/#which-import-do-i-need)
identifies focused preludes. The optional `diagnostics` feature adds diagnostic
helpers; the repository's notebook/binary workflows require `cli`.

Exact orientation, relative in-sphere signs, and complete symbolic tie-breaking
are available through D=6; that predicate boundary is broader than routine
construction coverage. D≥7 distance-based fallbacks lack the same exact-sign
contract. Full spherical editing integration, hyperbolic triangulation semantics,
constrained meshing, and arbitrary PL-manifold realization are outside the
implemented contract. [Limitations](limitations.md) owns the complete scope.

## Shared assumptions and validation layers

A triangulation combines an oriented abstract simplicial complex with a coordinate
realization in a supported geometric model. Finite binary64 coordinates are the
public input model; exact predicates reason about those stored values, not an
unrecorded ideal measurement. Unrepresentable derived coordinates must surface
typed errors rather than silently erase a predicate sign.

The proof owners separate five obligations:

| Level | Obligation | Owner and scope |
|---|---|---|
| 1 | Element Validity | Vertices, simplices, facets, coordinate storage, and local invariants |
| 2 | Combinatorial Consistency | `Tds`: incidence, adjacency, indexes, and coherent stored simplex orderings |
| 3 | Intrinsic PL Topology | `Triangulation`: links, components, Euler consistency, and supported orientability checks |
| 4 | Valid Realization | `Triangulation` or the spherical backend: model-specific nondegeneracy and realization conditions |
| 5 | Geometric Predicates | `DelaunayTriangulation` or the spherical backend: empty-circumsphere or empty-cap predicates |

Levels 2–3 are coordinate-independent. Level 3 certifies intrinsic orientability
for supported 2D/3D PL-manifold guarantees, including periodic quotients.
In D≥4, local incidence tests alone do not decide whether an arbitrary vertex
link is a PL sphere or ball; the `PLManifold` contract also requires crate-held
construction provenance. See the [topology contract](topology.md) and
[PL-topology background](../REFERENCES.md#topological-manifolds-realizations-and-pl-topology-levels-3-4-validation).

Euclidean/toroidal Level 4 checks require positively oriented, nondegenerate
maximal simplices whose realizations intersect only in shared abstract faces.
Toroidal checks use lifted covering-space charts. The bounded spherical backend
separately checks simplex nondegeneracy on `S^D` in `R^(D+1)`; do not infer the
full affine overlap contract from that prototype. Level 5 then adds the selected
geometric predicate. Detailed definitions and costs belong in
[construction and validation](construction_and_validation.md), with orientation
conventions in the [orientation specification](orientation_spec.md).

Checked TDS transitions protect canonical storage. Promotion adds the next
owner's proof without permitting higher layers to bypass lower invariants.
Fallible published mutations preserve their promised layers or restore the prior
valid state. `TopologyGuarantee` selects topology obligations;
`ValidationPolicy` selects insertion-time audit cadence. Layer-local checks and
cumulative `validate`/report APIs have different scopes; see the
[validation API pattern](construction_and_validation.md#validation-api-pattern).

## Construction and editing methods

### Bistellar moves and bounded repair

Admissible Pachner moves provide local topology edits; the Delaunay refinement
workflow uses bounded flip repair and final certification. Edelsbrunner and Shah
are an implementation source for incremental topological flipping; Pachner
supplies the PL-homeomorphism context. Neither is an unconditional convergence
bound for every local schedule used here. Budgets, rollback, and typed
non-convergence remain part of the [repair workflow](USING_TRIANGULATIONS.md#builder-api-flip-based-delaunay-repair-details).
See [bistellar-move sources](../REFERENCES.md#bistellar-pachner-moves-and-delaunay-repair).

### Cavity-based insertion

Incremental insertion locates a point, identifies a conflict cavity, and replaces
it while maintaining the promised invariants. Bowyer and Watson [2, 3] supply
the construction basis; walking point-location literature supplies related
algorithmic context. See the [insertion rationale](invariants.md#incremental-insertion-algorithm-cavity-based),
[construction sources](../REFERENCES.md#triangulation-construction-algorithms),
and [point-location sources](../REFERENCES.md#point-location-in-triangulations).

### Convex hull extraction and spherical construction

`ConvexHull::try_from_triangulation` extracts and certifies the triangulation's
boundary facets. Brown is the provenance for the Delaunay–hull relationship;
Quickhull is related background, not this crate's hull algorithm. The spherical
prototype uses ambient convex-hull duality and empty-cap predicates on bounded
inputs. See [hull sources](../REFERENCES.md#convex-hull-from-delaunay-triangulations),
[lifting background](../REFERENCES.md#lifted-paraboloid-method), and the
[spherical example](../examples/spherical_construction.rs).

### Hilbert insertion ordering

Hilbert ordering improves spatial locality without replacing invariant checks.
Skilling [9] is the index-algorithm source; Moon et al. [8] provide locality
analysis. See [ordering conventions](invariants.md#hilbert-ordering) and
[Hilbert sources](../REFERENCES.md#spatial-ordering-and-hilbert-curves).

### Periodic image-point construction

Toroidal construction uses `3^D` images and a checked quotient. It includes a
deterministic coordinate perturbation bounded relative to each domain period;
this coordinate change is distinct from symbolic predicate tie-breaking.
Caroli and Teillaud provide periodic-covering provenance, while CGAL is a
reference implementation. See [periodic sources](../REFERENCES.md#periodic-and-toroidal-triangulations)
and the exact [domain restrictions](limitations.md#topology-and-domain-limits).

## Numerical and realization methods

### Filtered-exact predicates

Provable floating-point filters accept only certified signs; unresolved cases
use exact determinant arithmetic through `la-stack`. Shewchuk [1] supplies
filter/error-bound analysis and Bareiss [6] fraction-free elimination. Relative
in-sphere cold paths form differences and squared norms exactly from the original
binary64 inputs: an exact determinant of already-rounded derived entries would
answer a different question. See [predicate contracts](numerical_robustness_guide.md#exact-predicates-v071),
[robust-predicate sources](../REFERENCES.md#robust-geometric-predicates), and
[exact-determinant sources](../REFERENCES.md#exact-determinant-sign-computation).

### Realized-simplex overlap detection

The affine Level 4 validator finds candidate pairs by sweeping axis-aligned
bounding boxes, then applies exact rational barycentric intersection checks.
Overlapping projections on every coordinate axis are necessary for simplex
intersection, so the broad phase must retain every potentially intersecting pair.
The narrow phase determines whether intersection exceeds the shared abstract
face. Baraff and I-COLLIDE provide sweep-and-prune provenance; Ericson supplies
related collision-detection background. See [overlap sources](../REFERENCES.md#realized-simplex-overlap-detection-level-4-validation)
and the [realization contract](construction_and_validation.md#level-4-valid-realization).

### Simulation of Simplicity

Edelsbrunner and Mücke [7] supply the symbolic perturbation method. The crate
computes a complete exact determinant-polynomial expansion for supported
dimensions and uses canonical vertex identity ordering at kernel call sites.
`AdaptiveKernel` resolves ties consistently; `FastKernel` and `RobustKernel`
retain explicit degeneracy signals. See [SoS conventions](numerical_robustness_guide.md#identity-based-sos-perturbation-via-canonical-vertex-ordering)
and [SoS sources](../REFERENCES.md#simulation-of-simplicity-and-degeneracy-handling).

## Correctness evidence and performance measurements

The [invariant model](invariants.md), exact arithmetic, property tests, regression
tests, and public examples supply distinct forms of correctness evidence. The
[property-testing summary](property_testing_summary.md) records tested invariants;
the [validation gallery](construction_and_validation.md#notebook-generated-validation-gallery)
illustrates failures and links executable evidence. Passing bounded fixtures does
not prove unsupported dimensions or geometries correct.

Hilbert ordering, allocation-conscious storage, validation-level benchmarks,
math-kernel benchmarks, and Criterion release comparisons characterize cost.
They do not replace correctness checks or establish portable runtime guarantees.
Use the [benchmark guide](../benches/README.md) for workloads and provenance, and
[performance tuning](dev/TUNING-PERFORMANCE.md) for the measurement workflow.
The retained [performance report](performance.md) currently records legacy,
provenance-limited measurements; preserve that qualification until new evidence
is published through its owner.

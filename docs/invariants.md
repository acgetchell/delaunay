# Topological and Geometric Invariants

This document provides the **theoretical background, formal definitions, and rationale**
for the invariants enforced by the [`delaunay`](https://crates.io/crates/delaunay) crate.

Whereas the crate-level [documentation](https://docs.rs/delaunay/) specifies the **semantic contract**
— what invariants are enforced and when —
this document explains **why** those invariants are sufficient, how they relate to
PL-manifold theory, and what assumptions underlie convergence guarantees.

- Rendered public API docs: <https://docs.rs/delaunay>.
- Crate page: <https://crates.io/crates/delaunay>.
- Source: <https://github.com/acgetchell/delaunay>.
- External reference implementation: [CGAL](https://www.cgal.org/) (<https://doc.cgal.org/latest/>).

This file is intentionally non-normative: it complements, but does not override,
the guarantees stated in the public API documentation.

---

## Table of Contents

- [Topological and Geometric Invariants](#topological-and-geometric-invariants)
  - [Table of Contents](#table-of-contents)
  - [Simplicial complexes and manifolds](#simplicial-complexes-and-manifolds)
    - [Simplicial complex model](#simplicial-complex-model)
  - [Validation layering](#validation-layering)
    - [Builder publication contracts](#builder-publication-contracts)
  - [Orientation contracts](#orientation-contracts)
  - [Geometric invariants](#geometric-invariants)
    - [Valid realization](#valid-realization)
    - [Geometric predicates and the Delaunay condition](#geometric-predicates-and-the-delaunay-condition)
    - [Robust predicate envelope](#robust-predicate-envelope)
  - [PL-manifold conditions](#pl-manifold-conditions)
    - [PL-manifolds vs pseudomanifolds](#pl-manifolds-vs-pseudomanifolds)
  - [Link-based manifold validation](#link-based-manifold-validation)
    - [Vertex links](#vertex-links)
    - [Ridge links](#ridge-links)
  - [Topological domains](#topological-domains)
  - [Incremental validation strategy](#incremental-validation-strategy)
    - [Incremental insertion algorithm (cavity-based)](#incremental-insertion-algorithm-cavity-based)
    - [Degenerate input and initial simplex construction](#degenerate-input-and-initial-simplex-construction)
    - [Tradeoffs](#tradeoffs)
  - [Insertion ordering and locality heuristics](#insertion-ordering-and-locality-heuristics)
    - [Hilbert ordering](#hilbert-ordering)
  - [Convergence considerations](#convergence-considerations)
  - [Limitations and pathological cases](#limitations-and-pathological-cases)
  - [Footnotes](#footnotes)

Readers primarily interested in **how to use the library** should start with:

- [`README.md`](../README.md)
- [`docs/workflows.md`](workflows.md)
- [`docs/construction_and_validation.md`](construction_and_validation.md)
- [`docs/limitations.md`](limitations.md)
- [`docs/numerical_robustness_guide.md`](numerical_robustness_guide.md)
- [`docs/api_design.md`](api_design.md)
- [`docs/topology.md`](topology.md)

For the implementation details of the coherent-orientation invariant specifically,
see [`ORIENTATION_SPEC.md`](ORIENTATION_SPEC.md).

---

## Simplicial complexes and manifolds

### Simplicial complex model

At the data-structure level, the crate models a triangulation as a **finite simplicial complex**[^edelsbrunner2001]
represented by its **minimal** ("vertices") and **maximal** simplices (“simplices”). In dimension `D`, a maximal simplex is a
`D`-simplex with exactly `D+1` vertices.

Key combinatorial objects:

- **Vertices**: 0-simplices. In the implementation, a vertex has coordinates plus an internal key
  and a UUID (used for stable referencing, e.g. serialization to files).
- **Simplices**: maximal `D`-simplices. Each simplex stores an ordered list of
  `D+1` vertex keys, and also has an internal key and an externally accessible
  UUID.
- **Facets**: codimension-1 faces of a simplex. A `D`-simplex has `D+1` facets, each missing exactly one
  vertex.
- **Adjacency / neighbors**: two simplices are neighbors if they share a facet. The triangulation data
  structure (TDS) stores neighbor pointers across facets (see
  [`src/core/tds/model.rs`](../src/core/tds/model.rs),
  [`src/core/tds/mutation.rs`](../src/core/tds/mutation.rs),
  [`src/core/tds/validation.rs`](../src/core/tds/validation.rs), and CGAL’s
  [TDS_3](https://doc.cgal.org/latest/TDS_3/index.html)).[^cgal-tds3][^impl-tds]
- **Boundary vs interior facets**:
  - An **interior facet** is incident to exactly two simplices.
  - A **boundary facet** is one-sided and not an admissible periodic
    self-identification.
  - A periodic quotient facet may be incident to one stored simplex while a
    self-neighbor pointer identifies it as closed topology rather than boundary.

These are **combinatorial** notions: they depend only on incidence and adjacency relationships.
Geometric predicates (orientation / in-sphere tests) are used to construct and validate the
**geometric** Delaunay property, but the topology checks are expressed in terms of the simplicial
complex.

The order of a simplex's vertices is part of the data-structure contract, not presentation detail.
It determines the simplex's combinatorial orientation and the facet index used for neighbor links.
This is why the TDS validation layer checks coherent orientation alongside neighbor reciprocity.

---

## Validation layering

The implementation separates invariants into five validation levels. Keeping these layers distinct
prevents geometric checks from leaking into purely combinatorial validation and makes it clear which
operation has certified which part of the structure:

1. **Level 1 — Element Validity**: individual vertices, simplices, facets, coordinates, resolved
   simplex-local coordinate uniqueness, and local orientation data are internally consistent.
2. **Level 2 — Combinatorial Consistency**: the triangulation data structure has valid
   vertex/simplex mappings, reciprocal neighbor pointers, bounded facet sharing, no duplicate
   simplices, simplex/ridge connectivity, and coherent combinatorial orientation.
3. **Level 3 — Intrinsic PL Topology**: the abstract simplicial complex satisfies the requested
   `TopologyGuarantee` (pseudomanifold or PL manifold) through incidence,
   connected components, Euler-characteristic, link, and supported 2D/3D orientability checks.
4. **Level 4 — Valid Realization**: the complex is geometrically valid in the chosen coordinate
   model. `Triangulation::is_valid_realization()` owns realization-only fast-fail validation, and
   `Triangulation::validate_realization()` owns cumulative Levels 1–4 certification. Euclidean and
   toroidal affine-chart realizations are checked by these validator APIs, with toroidal checks
   lifted to periodic covering-space charts; spherical realization validation is owned by the
   spherical backend for simplices on `S^D \subset R^(D+1)`.
5. **Level 5 — Geometric Predicates**: the valid realization satisfies the selected
   geometry-specific predicate family. The implemented family today is Delaunay, with
   Euclidean/toroidal empty-circumsphere predicates and spherical empty-cap / ambient-hull-facet
   predicates.

`Triangulation::is_valid_topology()` is the Level 3 intrinsic PL-topology check.
`Triangulation::is_valid_realization()` is the Level 4 realization-only check, and
`Triangulation::validate_realization()` is the cumulative Levels 1–4 check.
`DelaunayTriangulation::is_valid_delaunay()` is the implemented Level 5 geometric-predicate check
for an already-formed Delaunay triangulation.
Cumulative validation is exposed through the `validate` / `validation_report` APIs described in
[`docs/construction_and_validation.md`](construction_and_validation.md).

The domain owners follow the same layering. A published `Tds` carries the
Levels 1–2 proof and never represents a partial bootstrap. The empty complex is
a valid, constructed TDS because its element, mapping, incidence, adjacency,
and orientation obligations hold vacuously. A draft containing some vertices
but no full-dimensional simplex is different: it remains unpublished until it
can prove the complete TDS contract. Canonical storage fields are visible only
inside the TDS module; every edit that crosses a published domain boundary is a
checked transition that either preserves Levels 1–2 or reports a typed error
without publishing the failed state.

This distinction is encoded by a private unpublished wrapper, not only by a
runtime flag. `TdsDraft` owns `UnverifiedTds<_, _, D>` internally; only its
consuming `finish()` path can validate and remove that wrapper to return the
public `Tds`. Snapshot restoration follows the same unpublished path before
returning a `Tds`. The wrapper is confined to `core::tds`, and the public owner
therefore keeps its ordinary three-parameter API without exposing publication
typestate to callers.

`TriangulationBuilder` is the only public TDS → `Triangulation` proof boundary.
Its default strict mode consumes the Levels 1–2 proof and checks only the missing
Level 3 topology and Level 4 realization conditions. Strict mode does not
mutate the input or allocate a rollback snapshot; success preserves the TDS
owner identity, generation, simplex ordering, and stored orientation exactly.
The explicit `.canonicalizing()` mode may normalize orientation before the same
Levels 3–4 certification. It therefore keeps one TDS snapshot and restores the
exact input on every failure. The generic `TriangulationDraft` is internal
publication state, not another public route across the same proof boundary. It
stores a crate-private `UnverifiedTriangulation` wrapper until successful
Levels 3–4 certification consumes that wrapper and returns the public
`Triangulation`. The public owner likewise keeps its ordinary four-parameter
API.

`DelaunayRefinementBuilder` is the only public `Triangulation` →
`DelaunayTriangulation` proof boundary. Its default strict terminal consumes the
Levels 1–4 proof and checks only Level 5, without mutation or a rollback
snapshot. Its `.repair_by_flips()` type state enables bounded transactional
repair, `.max_flips(...)`, and `.fallback_rebuild(...)`; the original Levels
1–4 owner is restored on every failure. A failed refinement returns that
unchanged owner with its typed rejection reason, so callers can retry with a different policy
without defensively cloning canonical storage.

Explicit cumulative audit methods intentionally recheck lower layers for
diagnostics; proof-consuming constructors do not.

Every construction mutation enforces changed-scope Levels 1–4 postconditions. `ValidationPolicy`
independently controls when construction repeats a full global Levels 1–4 audit. Level 5
geometric-predicate validation remains a separate certification step.

### Builder publication contracts

Builders are configurable workflows, drafts are unpublished single-boundary
representations, workspaces are private mutable algorithm state, and owners are
verified domain values. The canonical vocabulary and complete promotion ladder
are defined in
[`construction_and_validation.md`](construction_and_validation.md#construction-vocabulary).

- `TdsBuilder`, `TriangulationBuilder`, `DelaunayRefinementBuilder`, and
  `DelaunayTriangulationBuilder` collect their layer's input and multiplying
  options.
- `DelaunayIncrementalBuilder` is the public, long-lived stateful workflow for
  point-by-point construction.
- `TdsDraft`, the crate-internal `TriangulationDraft`, and the crate-private
  `DelaunayTriangulationDraft` each belong to one promotion boundary. A draft
  is not a second public promotion route merely because a builder uses it.
- `DelaunayBootstrapWorkspace` and `DelaunayBatchWorkspace` own mutable
  algorithm state, caches, retries, and repair scheduling that are not proofs.
- `Tds`, `Triangulation`, and `DelaunayTriangulation` are published owners.

`DelaunayIncrementalBuilder` has two disjoint internal states. Its bootstrap
state owns a `TdsDraft` plus the kernel, topology options, staged vertex keys,
and Delaunay insertion caches;
it never stores a partially valid `DelaunayTriangulation`. The insertion that
first creates a maximal simplex finishes Levels 1–2, attempts canonical Levels
3–4 publication, and then performs Level 5 certification. Direct TDS draft
assembly preserves every `VertexKey` returned during bootstrap. The transition
keeps a rollback copy of the small bootstrap state and restores its exact owner
identity, generation, vertices, keys, and caches if any proof fails. After
success, the builder stores a Levels 1–5
`DelaunayTriangulation`, and every later insertion uses the owner's normal
transactional invariant-preserving path. `finish()` publishes the verified
empty complex, rejects a nonempty partial bootstrap, or cumulatively audits the
verified state before returning it.

A verified empty `DelaunayTriangulation` remains a valid published owner, but a
single vertex insertion in positive dimension would turn it into a nonempty
partial bootstrap. Published-owner insertion therefore rejects that transition
with `InsertionError::PublishedOwnerBootstrapRequiresBuilder` before changing
storage or caches. Callers that want to grow a triangulation from empty start
with `DelaunayIncrementalBuilder`; once its first maximal simplex is certified,
the same stateful builder API delegates later insertions to the owner.

The crate-private `DelaunayTriangulationDraft` is a different, narrower type:
it can contain only an already valid Levels 1–4 `Triangulation` plus the caches
that must survive publication. It cannot represent partial connectivity and has
no mutation API. Strict refinement, repaired refinement, batch construction,
incremental-builder publication, and TDS restoration all cross this same final Level 5
publication primitive. Construction paths with complete-point-set provenance
may use the linear local flip-predicate certificate. That evidence is an
unforgeable token created only by Level 5 validation functions and bound to the
exact topology owner identity, structural generation, and global topology;
detached or stale evidence falls back to a new Level 5 check. A report-domain
enum is not accepted as publication evidence. Strict refinement,
transactional repair, and incremental-builder publication cross the same
certificate-backed publication boundary after their selected Level 5 check.
This keeps the proof transition orthogonal without repeating a successful
predicate pass.

Calling `build()` or `finish()` on a builder, or `finish()` on a staged draft,
is a publication boundary. It must either return an owner carrying every invariant assigned to
that layer or return a typed error without exposing the failed intermediate
state. An empty draft may publish because the empty complex is valid
vacuously; `TdsBuilder` likewise accepts empty vertex and simplex inputs as the
empty complex. A non-empty bootstrap with no maximal simplex cannot publish.

The construction hierarchy follows the validation hierarchy:

1. `TdsBuilder` owns explicit combinatorial input: vertices, maximal-simplex
   connectivity, and TDS payload types. It checks index arity and bounds,
   repeated vertices within a simplex, duplicate maximal simplices, bounded
   facet sharing, key mappings, incidence, reciprocal neighbor links, and
   coherent stored orientation. It returns a constructed `Tds` only after
   cumulative Levels 1–2 validation succeeds. Coordinates alone are not enough
   to determine a TDS: the core layer must not infer connectivity through a
   Delaunay or other geometry-specific policy. When a higher builder parses the
   same explicit input boundary, it passes `ParsedTdsInput` to `TdsBuilder` so
   arity, bounds, and within-simplex uniqueness are proved once rather than
   revalidated from raw vectors.
2. `TriangulationBuilder` consumes a TDS together with the kernel, topology
   guarantee, global topology, validation policy, and build mode. It preserves
   the Levels 1–2 proof, establishes the requested intrinsic topology, and
   validates the realization. Canonicalizing mode first normalizes positive
   geometric orientation where required; strict mode rejects a representation
   that needs normalization. It publishes a `Triangulation` only after
   cumulative Levels 1–4 validation succeeds. `TriangulationBuildFailure`
   returns the exact input TDS plus the typed `TriangulationBuilderError`.
3. `DelaunayRefinementBuilder` consumes a valid `Triangulation`. Strict mode
   checks only the geometry-specific Level 5 predicate. Flip-repair mode may
   change connectivity inside a rollback transaction before the same final
   certification. It publishes a `DelaunayTriangulation` only after Level 5
   succeeds in addition to the lower proofs.
4. `DelaunayTriangulationBuilder` is the separate point-set construction
   workflow. It infers connectivity, establishes the lower proofs, applies its
   Delaunay construction policy, and publishes only after Levels 1–5 succeed.
   Its private batch workspace and `DelaunayIncrementalBuilder` have different
   mutation schedules, but both hand an established Levels 1–4 owner to the
   same private Level 5 draft.

This dependency direction is strict:

```text
explicit input -> TdsBuilder -> Tds
                              |
                              v
             TriangulationBuilder -> Triangulation
                                      |
                                      v
             DelaunayRefinementBuilder -> DelaunayTriangulation

point set -> DelaunayTriangulationBuilder -> DelaunayTriangulation
```

Incremental strategies expose staged mutation only where a layer has meaningful
unfinished state:

```text
TdsDraft::finish()                    -> Tds
DelaunayIncrementalBuilder::finish()  -> DelaunayTriangulation
```

For a nonempty `DelaunayIncrementalBuilder`, the first full-dimensional simplex is an
internal early publication checkpoint: failure returns
`DelaunayIncrementalBuilderError` and leaves the pre-insertion bootstrap state
reusable. The public owner is still returned only by `finish()`.

The generic and Delaunay layers each have one public promotion builder with
explicit modes:

```text
Tds -> TriangulationBuilder
       |-- .build()                    -> Triangulation by strict certification
       `-- .canonicalizing().build()   -> Triangulation after transactional canonicalization

Triangulation -> DelaunayRefinementBuilder
                 |-- .build()                   -> DelaunayTriangulation
                 `-- .repair_by_flips().build() -> DelaunayTriangulation + repair outcome
```

`TriangulationBuilder` default strict mode preserves owner identity, generation, and
simplex ordering on success and avoids a full rollback snapshot. Its
explicit canonicalizing mode may change simplex ordering and generation on success; its
snapshot and copy cost are linear in the TDS representation. Both modes return
the unchanged TDS on failure and invoke the same non-mutating Levels 3–4
certification proof. `DelaunayRefinementBuilder` applies the equivalent rule at
Level 5: strict mode is non-mutating and snapshot-free, while repair mode pays
for rollback only after the caller explicitly selects it. The type-state split
keeps each terminal's result and error type precise.

Lower-layer builders and drafts must not call higher-layer constructors to
obtain their own fixtures or implementation state. Specialized assembly paths
may stage an incomplete lower-layer draft when an algorithm cannot express its
work as explicit connectivity up front, but they must cross the same checked
completion boundary before a higher-layer owner is published.

Options belong to the lowest layer that can interpret them without importing a
higher invariant: explicit connectivity and TDS payload shape belong to
`TdsBuilder`; kernel, topology, realization, and validation-policy choices
belong to `TriangulationBuilder`; insertion order, predicate-family, and
Delaunay repair choices belong to `DelaunayTriangulationBuilder`. Adding an
option must not weaken an already established lower-layer proof.

Failure is atomic at every boundary. A failed build never returns a partially
proven higher-layer owner. When a builder starts from raw input, it may drop its
unpublished workspace and return only a typed diagnostic. When
`TriangulationBuilder` consumes an already proof-bearing TDS, failure returns
that original lower-layer owner alongside the rejection reason. Rollback must
restore its canonical fields, runtime owner identity, and structural generation
exactly; callers may repair it or retry with different options without a
defensive clone. Strict TDS refinement returns the unchanged owner without
needing rollback. Strict refinement and canonicalizing builder publication must
share one Levels 3-4 certification implementation so their proof criteria
cannot drift.

---

## Orientation contracts

Orientation has three related but independently validated meanings in this crate:

- **Intrinsic PL orientability (Level 3)**: the shared-facet parity constraints
  must admit a coherent simplex-orientation assignment independently of the
  orderings currently stored in the TDS. `Triangulation::orientation_witness()`
  returns the opaque assignment for supported pure 2D/3D complexes.
- **Stored TDS coherence (Level 2)**: adjacent simplices must induce opposite orientations on their shared facet. In
  practice this is checked by comparing the facet index in one simplex with the reciprocal mirror index
  in its neighbor.
- **Geometric orientation (Level 4)**: Euclidean/toroidal maximal simplices
  should have positive orientation in their active affine charts, while the
  spherical backend enforces its model-specific realization conditions. An
  operation may handle a degenerate or intermediate state only inside a
  failure-atomic transaction.

The orientation checker uses the robust orientation predicate directly instead of a kernel-level
predicate that may apply Simulation of Simplicity. That preserves the distinction between an
actually degenerate simplex and a deterministically tie-broken predicate result. At the TDS layer,
periodic-image simplices compare lifted `(vertex, offset)` facet identities after translation
normalization, so quotient facets participate in stored-coherence and intrinsic-orientability
constraints without conflating either property with geometric sign.

Pachner and bistellar-editing transactions must keep coherent combinatorial orientation and positive
geometric simplex orientation separate. A move can leave the TDS coherently oriented while affected
simplices have negative signed volume, or can make individual simplices positive while breaking
shared-facet coherence. The ordinary realized-geometry-preserving edit path repairs negative
orientation by reordering simplex vertex slots where possible, then relies on topology and Level 4
realization validation before the edited state is accepted. If the repaired state still violates the
contract, the transaction rolls back with a typed error instead of leaving a partially repaired
triangulation behind.

See [`ORIENTATION_SPEC.md`](ORIENTATION_SPEC.md) for the exact parity convention, implementation
map, and test expectations.

---

## Geometric invariants

### Valid realization

Levels 1-3 validate the abstract simplicial complex with coherent orientation: elements, incidence,
neighbor reciprocity, manifoldness, links, connectedness, and Euler consistency. These checks do not
by themselves prove that the complex is faithfully realized in its coordinates. A topologically valid
complex can still fold over itself, contain a zero-volume maximal simplex, or identify simplices in a
way that overlaps in the chosen geometric chart.

Level 4 is the valid-realization check. It is independent of Level 5 geometric predicates and
enforces:

- every Euclidean/toroidal maximal simplex has positive geometric orientation in its affine chart;
- every Euclidean/toroidal maximal simplex has nonzero `D`-volume under the robust orientation predicate;
- every pair of Euclidean/toroidal maximal simplices intersects only in the realization of the face
  spanned by their shared vertices;
- Euclidean and toroidal triangulations use valid affine-chart realization checks, with toroidal
  triangulations checked in periodic covering-space charts, including translated
  images that can overlap across the fundamental-domain boundary.
- the bounded spherical prototype validates `S^2`/`S^3` maximal simplices as nondegenerate spherical simplices
  in `S^D \subset R^(D+1)`.

This is intentionally separate from topology. Non-orientable spaces are valid objects in topology in
general, but the crate's 2D/3D PL-manifold guarantees require an intrinsic orientation witness, and
its TDS contract separately maintains coherent stored orderings for the complexes its construction,
flip, and predicate machinery operate on. Level 4 then asks whether that oriented complex is a valid
realization in the active realization model. For Euclidean/toroidal affine-chart models, this means
the vertex map is injective, every abstract simplex is realized as a nondegenerate affine simplex,
and realized simplex intersections satisfy `|sigma| ∩ |tau| = |sigma ∩ tau|`. General spherical
integration with the ordinary mutable triangulation surface and hyperbolic topology need
model-specific chart validators before they can offer the same Level 4 guarantee. The crate
validates finite coordinate realizations; it does not construct a realization for an arbitrary
abstract PL-manifold.

### Geometric predicates and the Delaunay condition

Level 5 is the geometric-predicate layer: geometric optimality or predicate satisfaction on top of
a valid realization. Its implemented predicate family is Delaunay: Euclidean and toroidal
triangulations use empty-circumsphere predicates, while the bounded spherical prototype uses the
equivalent empty-cap / ambient convex-hull-facet predicate.

A Euclidean Delaunay triangulation is characterized by the **empty circumsphere**
condition:[^deberg2008][^edelsbrunner2001]

- for each `D`-simplex (simplex), no non-simplex vertex lies *strictly inside* that simplex’s
  circumsphere.

This is a **geometric** invariant: it depends on the realization coordinates and on robust evaluation
of orientation / in-sphere predicates.[^shewchuk1997]

Edelsbrunner and Shah establish incremental topological flipping for regular
triangulations; Delaunay triangulations are regular, so that result supplies the
mathematical basis for the crate's flip-repair stage.[^edelshah1996] Pachner's
theorem separately characterizes PL-homeomorphic manifolds through bistellar
moves, which is why the repair boundary requires a PL-manifold proof rather than
mere pseudomanifold incidence.[^pachner1991] These results do not promise that
every arbitrary local queue order on every supported higher-dimensional input
will reach a Delaunay triangulation within a fixed engineering budget. The crate
therefore treats repair as bounded and fallible, rolls back failed attempts, and
publishes the Level 5 owner only after certification.

The smallest executable check of that move system is the crate’s n=1 ergodicity contract: for a
selected admissible Pachner move, applying the move and then its inverse must recover the same
triangulation, including vertex identity and top-dimensional simplex incidence. A merely valid
post-roundtrip triangulation is not sufficient. Jaccard similarity is useful in the failure path to
quantify how vertex and simplex-incidence sets diverged, but exact equality is the invariant.

In practice, floating-point degeneracy matters:

- For near-degenerate configurations, robust predicates (and/or retry/repair strategies) may be
  required to construct or certify the Delaunay property.
- Validation can be performed explicitly via the Level 5 Delaunay-predicate check
  (`DelaunayTriangulation::is_valid_delaunay`) when a workflow requires
  certainty.

Internally, the crate’s Level 5 verifier prefers fast, local flip-based checks over the naive
O(simplices × vertices) brute-force test. This reflects the standard theoretical relationship between
Delaunay optimality and local flip predicates.[^edelshah1996][^impl-flips][^impl-delaunay-validation]

### Robust predicate envelope

The default kernel is designed around a staged predicate model:

- fast floating-point filters where a sign can be certified cheaply,
- exact arithmetic fallback for supported dimensions,
- and deterministic symbolic perturbation for exact degeneracies.

The current release envelope is intentionally finite. The fast `f64` predicate filters are available
through `D <= 4`; exact orientation is available through `D <= 6`; exact in-sphere support, and the
`ExactPredicates` contract used by flip repair, is available through `D <= 5`. Higher-dimensional
experiments may still be useful, but they are outside the strongest predicate and repair contract.

For practical coordinate hygiene, kernel selection, and dimensional limits, see
[`docs/numerical_robustness_guide.md`](numerical_robustness_guide.md) and
[`docs/limitations.md`](limitations.md).

---

## PL-manifold conditions

### PL-manifolds vs pseudomanifolds

This crate distinguishes between two common “manifoldness” tiers that arise in practice when using
simplicial complexes for geometry:

- **Pseudomanifold / manifold-with-boundary (codimension-1)**: enforce that each facet has the
  expected incidence count:
  - one-sided facets are incident to exactly 1 simplex
  - two-sided facets are incident to exactly 2 simplices
  Boundary classification then excludes admissible periodic self-identifications.
  This rules out the most obvious non-manifold failures (branching facets).

- **Closed boundary condition (codimension-2 on the boundary)**: enforce “no boundary of boundary”
  (intuitively: the boundary itself is a (D−1)-manifold with no boundary). This rules out hanging
  boundary ridges.

- **Connectedness + isolated vertices**: enforce that the simplex-neighbor graph is a single component
  and that every vertex is incident to at least one simplex.

- **Euler characteristic**: check χ against expected classifications where available. This is a
  global consistency check that catches some classes of topological corruption.

Piecewise-linear (PL) manifoldness is strictly stronger than the pseudomanifold conditions. The public API exposes this
via `TopologyGuarantee`, re-exported at the crate root and in
`delaunay::prelude::construction` (source:
[`src/triangulation/validation.rs`](../src/triangulation/validation.rs)):

- `TopologyGuarantee::Pseudomanifold`
  checks the codimension-1 incidence conditions (plus boundary consistency, connectedness,
  isolated-vertex checks, and Euler characteristic checks).
- `TopologyGuarantee::PLManifold`
  adds **link-based** conditions (ridge and vertex links) that are characteristic of
  PL-manifolds. In PL topology, requiring the links of simplices to be spheres (or balls at the
  boundary) is equivalent to the standard manifold condition that every point has a locally
  Euclidean neighborhood (up to PL homeomorphism).[^hatcher2002][^rourke-sanderson]

`ValidationPolicy` independently controls when a complete global audit is repeated. It does not
change which mathematical invariants the domain type carries.

---

## Link-based manifold validation

### Vertex links

A **vertex link** is the simplicial complex formed by taking all simplices incident
to a given vertex and removing that vertex from each simplex. Intuitively, the
vertex link represents the local neighborhood around the vertex, abstracted
away from the realization space.

For a PL-manifold, the link of every interior vertex must be homeomorphic to a
(d−1)-sphere, where d is the dimension of the triangulation. Boundary vertices
must have links homeomorphic to a (d−1)-ball.[^hatcher2002][^rourke-sanderson] These conditions characterize local
manifoldness at vertices and rule out singularities such as cones, pinched points,
or branching neighborhoods.

Vertex-link validation is strictly stronger than ridge-link validation: it can
detect global or vertex-local pathologies that are invisible to codimension-2
checks alone. However, constructing and validating full vertex links is
computationally expensive, as it requires enumerating the complete star of each
vertex and verifying topological properties of the resulting complex.

Construction proves vertex-link validity before publishing a `PLManifold` triangulation. Legal
bistellar moves preserve that PL-manifold class while changed-scope postconditions guard every
commit. `ValidationPolicy::Always` trades performance for a complete global Levels 1–4 audit after
every mutation, providing earlier detection and improved diagnosability without inventing a second
mathematical guarantee.

The owner-level public API exposes this certification as
`Triangulation::validate_vertex_links()` and
`DelaunayTriangulation::validate_vertex_links()`. These methods use the
triangulation's declared topology metadata when deciding whether a one-sided
facet is a true boundary or an admissible periodic identification.

### Ridge links

A **ridge** is a codimension‑2 simplex (e.g. an edge in 3D, a triangle in 4D).
The *link* of a ridge consists of the set of simplices incident to that ridge,
with the ridge itself removed.

For a PL‑manifold, the link of every interior ridge must be homeomorphic to a
circle (in 3D) or, more generally, a 1‑sphere. Boundary ridges must have links
homeomorphic to an interval.[^hatcher2002][^rourke-sanderson] Violations of this condition indicate local
non‑manifold behavior such as branching or pinching.

The `delaunay` crate exploits this property during **incremental insertion**:
ridge‑link validation is applied as cavities are created and retriangulated.
This detects the majority of topological failures early, while keeping per‑step
cost low. Because ridge links are small and localized, they can be checked
efficiently without scanning the full star of each vertex.

Ridge‑link validation is *necessary but not sufficient* to guarantee
PL‑manifoldness. Certain global or vertex‑local pathologies are only detectable
via vertex‑link validation. Construction therefore certifies vertex links before
publishing a `PLManifold` value, and every later full Level 3 audit repeats that
certification. Within a full audit, ridge links run first as a cheap fail-fast;
between audits, mutations check ridge links over their changed scope before
committing. Ridge links never substitute for vertex-link certification.

The public owner methods are `Triangulation::validate_ridge_links()`,
`Triangulation::validate_ridge_links_for_simplices()`, and matching
`DelaunayTriangulation` forwarding methods. The localized form is intended for
post-insertion and post-flip diagnostics where the touched simplex frontier is
known.

---

## Topological domains

The default model is a finite Euclidean triangulation of a point set. In that setting, boundary
facets are expected unless the simplex complex represents a closed manifold by construction.

Toroidal workflows are integrated as a first-class topology option:

- `.try_toroidal(...)` constructs a periodic image-point triangulation over neighboring
  fundamental domains. The `T^2` and compact `T^3` paths are validated periodic quotients; `T^4`/`T^5`
  periodic construction fails fast until quotient selection scales to routine release validation
  under issue #416.

Spherical topology models provide unit-sphere coordinate projection, and the bounded spherical
prototype adds `S^2`/`S^3` construction plus spherical Level 4/5 validation through a separate
backend. Hyperbolic topology models remain metadata and validation scaffolding. Treat full
non-Euclidean integration as ongoing extension work rather than a completed ordinary-triangulation
domain.

---

## Incremental validation strategy

The implementation uses a **hybrid validation strategy** intended to balance:

- fast incremental construction,
- early detection of common topological failures,
- and the ability to certify stronger PL-manifold conditions.

At a high level:

- **Ridge-link validation during mutations** is used as an inexpensive, local safety check. Ridge
  links are small, local objects, and validating them catches many PL-manifold violations early.
  Full Level 3 audits also run this check first so common failures avoid the more expensive
  vertex-link construction.
- **Vertex-link validation** is stronger but significantly more expensive. It is part of every full
  `PLManifold` Level 3 audit.
- **Always-on auditing** (`ValidationPolicy::Always`) repeats the complete Levels 1–4 audit after
  each mutation, trading performance for earlier detection and improved diagnosability.

### Incremental insertion algorithm (cavity-based)

The crate’s incremental construction follows the standard cavity-based approach (CGAL-style; see
[CGAL Triangulation_3](https://doc.cgal.org/latest/Triangulation_3/index.html) and
[`src/core/algorithms/insertion.rs`](../src/core/algorithms/insertion.rs)):
[^bowyer1981][^watson1981][^cgal-triangulation3][^impl-insertion-primitives]

1. **Locate** the simplex containing the query point (facet walking / scan fallback;
   [`src/core/algorithms/locate.rs`](../src/core/algorithms/locate.rs)).[^devillers-walking][^impl-locate]
2. **Find the conflict region**: the set of simplices whose circumspheres contain the point.
3. **Extract the cavity boundary** (a set of boundary facets separating conflicting from
   non-conflicting simplices).
4. **Remove** the conflicting simplices.
5. **Fill** the cavity by connecting the new vertex to the cavity boundary.
6. **Wire neighbors** locally (without global recomputation).

For points outside the current convex hull, an exterior insertion path extends the hull by
identifying *visible* boundary facets and retriangulating the visible region.

### Degenerate input and initial simplex construction

Construction begins by creating an initial simplex from `D+1` affinely independent real input
vertices. The default batch constructor searches a bounded pool of extreme vertices for a
large-volume simplex before falling back to the selected insertion order. If no non-degenerate
simplex can be formed (e.g., collinear points in 2D, coplanar in 3D), construction fails with a
geometric degeneracy error.

This early degeneracy detection is intentional: it prevents building a combinatorial structure whose
geometric interpretation is undefined.

### Tradeoffs

- Ridge-link checks are “cheap and local” and therefore viable as an insertion-time safety-net.
- Vertex-link checks are “expensive and global”; `ValidationPolicy` controls how often a complete
  audit repeats after the construction boundary has already certified the PL-manifold proof.
- Ordering heuristics (Hilbert) can improve locality and reduce cavity size, improving
  robustness in practice without changing the formal correctness contract.

---

## Insertion ordering and locality heuristics

### Hilbert ordering

Hilbert ordering refers to sorting vertices along a space-filling Hilbert curve
prior to incremental insertion.[^moon2001][^cgal-spatial-sorting][^impl-hilbert]
Hilbert curves have strong locality-preserving properties: points that are close in Euclidean space tend to be close along the
curve parameterization.

In the context of incremental Delaunay construction, improved locality reduces
the spatial extent of insertion cavities, leading to fewer affected simplices,
smaller flip cascades, and more stable intermediate triangulations. This can
significantly improve cache behavior and reduce the likelihood of numerically
fragile configurations during construction.

Hilbert ordering does not change the formal correctness guarantees of the
triangulation. Its impact is strictly on performance, robustness, and practical
convergence behavior, particularly in higher dimensions where cavity growth and
flip complexity can otherwise become large.

In this crate, Hilbert indices are computed using Skilling’s algorithm and used for batch preprocessing
(see [`src/core/util/hilbert.rs`](../src/core/util/hilbert.rs)).[^skilling2004][^impl-hilbert]

---

## Convergence considerations

Many “repair” and “editing” workflows in high dimensions rely on sequences of **bistellar flips**
(Pachner moves) to improve topology or restore the Delaunay property (see
[`src/core/algorithms/flips/repair.rs`](../src/core/algorithms/flips/repair.rs)).[^pachner1991][^edelshah1996][^impl-flips]

Important caveats:

- Convergence of local flip sequences is sensitive to both geometry (near-degeneracy) and topology.
- Relaxing topology guarantees (e.g., allowing only pseudomanifold checks) can admit intermediate
  states in which flip sequences are ill-posed or fail to converge.
- Even under PL-manifold constraints, numerical predicates can be borderline for ill-conditioned
  inputs, which can lead to non-progressing local operations.

Since v0.7.3, the default `AdaptiveKernel` applies **Simulation of Simplicity (SoS)** to both
orientation and insphere predicates, breaking exact-degeneracy ties deterministically and eliminating
the most common source of non-progressing flip cycles. The `ExactPredicates` marker trait ensures
that flip repair entry points only accept kernels with provably correct in-sphere sign decisions in
the supported dimensions. Exact orientation extends farther than exact in-sphere support, but repair
entry points are intentionally gated by the stronger Delaunay-predicate contract. Remaining
convergence risks at large scale are primarily cavity/topology interactions, pathological input
conditioning, or workflows outside the supported predicate envelope rather than ambiguous predicate
ties.

The crate therefore treats flip/repair as a best-effort procedure with explicit validation hooks:

- Prefer to validate Level 3 intrinsic PL topology (`Triangulation::validate` /
  `TopologyGuarantee`) when running flip-heavy workflows.
- Validate Level 4 realization and the relevant Level 5 geometric predicate explicitly when inputs
  are near-degenerate (`Triangulation::validate_realization` and
  `DelaunayTriangulation::is_valid_delaunay` for the Delaunay predicate family).

See the public API docs (<https://docs.rs/delaunay>) and [`docs/workflows.md`](workflows.md) for practical guidance.

---

## Limitations and pathological cases

Some limitations are inherent to incremental high-dimensional computational geometry:

- **Degenerate geometry in higher dimensions**: highly degenerate point configurations (many
  nearly coplanar / collinear subsets) can cause insertion to fail or require perturbation.
- **Predicate support envelope**: the strongest exact Delaunay-predicate and flip-repair contract is
  currently `D <= 5`; dimensions above that should be treated as experimental unless a workflow
  performs its own validation.
- **Topological-domain scope**: Euclidean and toroidal workflows are the active ordinary
  triangulation domains. The bounded spherical prototype supports `S^2`/`S^3` construction and
  spherical Level 4/5 validation through a separate backend; full spherical integration and
  hyperbolic models remain future work.
- **Iterative refinement constraints**: cavity-based insertion and flip-based repair are local
  procedures. In rare cases, local refinement can be blocked by topology or by non-progressing
  numerical predicates.
- **Numerical precision**: floating-point robustness is a fundamental constraint. Exact predicates
  with SoS (via `AdaptiveKernel`) substantially reduce failures, but extreme coordinate magnitudes
  or ill-conditioned point sets can still trigger edge cases.[^shewchuk1997]

Ordering and preprocessing can mitigate (but not eliminate) these issues:

- Locality-preserving orders (Hilbert) tend to keep cavities small and reduce flip cascades.
- Deduplication / near-duplicate rejection avoids many “almost coincident” degeneracies.

For concrete failure modes and recommended workflows, see [`docs/limitations.md`](limitations.md),
[`docs/workflows.md`](workflows.md), [`docs/construction_and_validation.md`](construction_and_validation.md), and the issue
investigation notes in [`docs/archive/`](archive/).

---

## Footnotes

For the project-wide bibliography (including references not cited here), see [`REFERENCES.md`](../REFERENCES.md).

[^edelsbrunner2001]: Herbert Edelsbrunner. *Geometry and Topology for Mesh Generation*. Cambridge University Press, 2001.
    DOI: <https://doi.org/10.1017/CBO9780511530067>.
[^cgal-tds3]: CGAL Project. *Triangulation Data Structure* (TDS_3) documentation.
    <https://doc.cgal.org/latest/TDS_3/index.html>.
[^impl-tds]: Implementation: [src/core/tds/model.rs](../src/core/tds/model.rs),
    [src/core/tds/mutation.rs](../src/core/tds/mutation.rs),
    [src/core/tds/incidence.rs](../src/core/tds/incidence.rs), and
    [src/core/tds/validation.rs](../src/core/tds/validation.rs).
[^deberg2008]: Mark de Berg, Otfried Cheong, Marc van Kreveld, Mark Overmars.
    *Computational Geometry: Algorithms and Applications*, 3rd ed. Springer, 2008.
    DOI: <https://doi.org/10.1007/978-3-540-77974-2>.
[^shewchuk1997]: Jonathan Richard Shewchuk. “Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates.”
    *Discrete & Computational Geometry* 18(3), 1997. DOI: <https://doi.org/10.1007/PL00009321>.
[^edelshah1996]: Herbert Edelsbrunner and Nimish R. Shah.
    “Incremental Topological Flipping Works for Regular Triangulations.” *Algorithmica* 15(3), 1996.
    DOI: <https://doi.org/10.1007/BF01975867>.
[^pachner1991]: Udo Pachner. “P.L. Homeomorphic Manifolds Are Equivalent by Elementary Shellings.”
    *European Journal of Combinatorics* 12(2), 1991. DOI: <https://doi.org/10.1016/S0195-6698(13)80080-7>.
[^impl-flips]: Implementation: [src/core/algorithms/flips/engine.rs](../src/core/algorithms/flips/engine.rs)
    and [src/core/algorithms/flips/repair.rs](../src/core/algorithms/flips/repair.rs).
[^impl-delaunay-validation]: Implementation: [src/delaunay/property_validation.rs](../src/delaunay/property_validation.rs).
[^hatcher2002]: Allen Hatcher. *Algebraic Topology*. Cambridge University Press, 2002.
    Free online version: <https://pi.math.cornell.edu/~hatcher/AT/ATpage>. (See Appendix A: “PL Manifolds and Links”.)
[^rourke-sanderson]: C. P. Rourke and B. J. Sanderson. *Introduction to Piecewise-Linear Topology*. Springer, 1972.
    DOI: <https://doi.org/10.1007/978-3-642-81735-9>.
[^bowyer1981]: A. Bowyer. “Computing Dirichlet Tessellations.” *The Computer Journal* 24(2), 1981.
    DOI: <https://doi.org/10.1093/comjnl/24.2.162>.
[^watson1981]: D. F. Watson. “Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes.”
    *The Computer Journal* 24(2), 1981. DOI: <https://doi.org/10.1093/comjnl/24.2.167>.
[^cgal-triangulation3]: CGAL Project. *3D Triangulations* (Triangulation_3) documentation.
    <https://doc.cgal.org/latest/Triangulation_3/>.
[^impl-insertion-primitives]: Implementation: [src/core/algorithms/insertion.rs](../src/core/algorithms/insertion.rs).
[^devillers-walking]: Olivier Devillers, Sylvain Pion, Monique Teillaud. “Walking in a Triangulation.”
    *International Journal of Foundations of Computer Science* 13(2), 2002. DOI: <https://doi.org/10.1142/S0129054102001047>.
[^impl-locate]: Implementation: [src/core/algorithms/locate.rs](../src/core/algorithms/locate.rs).
[^moon2001]: Bongki Moon, H. V. Jagadish, Christos Faloutsos, Joel H. Saltz.
    “Analysis of the Clustering Properties of the Hilbert Space-Filling Curve.”
    *IEEE Transactions on Knowledge and Data Engineering* 13(1), 2001. DOI: <https://doi.org/10.1109/69.908985>.
[^cgal-spatial-sorting]: CGAL Project. *Spatial Sorting* documentation.
    <https://doc.cgal.org/latest/Spatial_sorting/index.html>.
[^impl-hilbert]: Implementation: [src/core/util/hilbert.rs](../src/core/util/hilbert.rs) (Skilling’s Hilbert index).
[^skilling2004]: John Skilling. “Programming the Hilbert curve.” *AIP Conference Proceedings* 707, 2004.
    DOI: <https://doi.org/10.1063/1.1751381>.

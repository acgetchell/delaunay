# Topological and Geometric Invariants

This document provides the **theoretical background, formal definitions, and rationale**
for the invariants enforced by the `delaunay` crate.

Whereas the crate-level documentation (`lib.rs`) specifies the **semantic contract**
— what invariants are enforced and when —
this document explains **why** those invariants are sufficient, how they relate to
PL-manifold theory, and what assumptions underlie convergence guarantees.

This file is intentionally non-normative: it complements, but does not override,
the guarantees stated in the public API documentation.

---

## Scope and intent

This document covers:

- Formal definitions of simplicial-complex and PL-manifold invariants
- Link-based manifold conditions (ridge links vs vertex links)
- Rationale for incremental validation strategies
- Ordering heuristics used during insertion (Hilbert, Morton)
- Convergence assumptions for bistellar / flip-based repair
- Known limitations and pathological cases
- References to relevant computational geometry and CDT literature

Readers primarily interested in **how to use the library** should start with:

- `README.md`
- `docs/workflows.md`
- `docs/validation.md`

---

## Simplicial complexes and manifolds

### Simplicial complex model

At the data-structure level, the crate models a triangulation as a **finite simplicial complex**
represented by its **maximal** simplices (“cells”). In dimension `D`, a maximal cell is a
`D`-simplex with exactly `D+1` vertices.

Key combinatorial objects:

- **Vertices**: 0-simplices. In the implementation, a vertex has coordinates plus an internal key
  (and a UUID) used for stable referencing.
- **Cells**: maximal `D`-simplices. Each cell stores a set of `D+1` vertex keys.
- **Facets**: codimension-1 faces of a cell. A `D`-simplex has `D+1` facets, each missing exactly one
  vertex.
- **Adjacency / neighbors**: two cells are neighbors if they share a facet. The TDS stores neighbor
  pointers across facets.
- **Boundary vs interior facets**:
  - An **interior facet** is incident to exactly two cells.
  - A **boundary facet** is incident to exactly one cell.

These are **combinatorial** notions: they depend only on incidence and adjacency relationships.
Geometric predicates (orientation / in-sphere tests) are used to construct and validate the
**geometric** Delaunay property, but the topology checks are expressed in terms of the simplicial
complex.

---

## Geometric invariants

### Delaunay condition (empty circumsphere property)

A Delaunay triangulation is characterized by the **empty circumsphere** condition:

- for each `D`-simplex (cell), no other input vertex lies *strictly inside* that simplex’s
  circumsphere.

This is a **geometric** invariant: it depends on the embedding coordinates and on robust evaluation
of orientation / in-sphere predicates.

In practice, floating-point degeneracy matters:

- For near-degenerate configurations, robust predicates (and/or retry/repair strategies) may be
  required to construct or certify the Delaunay property.
- Validation can be performed explicitly via the Level 4 check (`DelaunayTriangulation::is_valid`)
  when a workflow requires certainty.

Internally, the crate’s Level 4 verifier prefers fast, local flip-based checks over the naive
O(cells × vertices) brute-force test. This reflects the standard theoretical relationship between
Delaunay optimality and local flip predicates.

---

## PL-manifold conditions

### PL-manifolds vs pseudomanifolds

This crate distinguishes between two common “manifoldness” tiers that arise in practice when using
simplicial complexes for geometry:

- **Pseudomanifold / manifold-with-boundary (codimension-1)**: enforce that each facet has the
  expected incidence count:
  - boundary facets are incident to exactly 1 cell
  - interior facets are incident to exactly 2 cells
  This rules out the most obvious non-manifold failures (branching facets).

- **Closed boundary condition (codimension-2 on the boundary)**: enforce “no boundary of boundary”
  (intuitively: the boundary itself is a (D−1)-manifold with no boundary). This rules out hanging
  boundary ridges.

- **Connectedness + isolated vertices**: enforce that the cell-neighbor graph is a single component
  and that every vertex is incident to at least one cell.

- **Euler characteristic**: check χ against expected classifications where available. This is a
  global consistency check that catches some classes of topological corruption.

Piecewise-linear (PL) manifoldness is strictly stronger than the pseudomanifold conditions. The public API exposes this
via `TopologyGuarantee`:

- `TopologyGuarantee::Pseudomanifold` checks the codimension-1 incidence conditions (plus boundary
- consistency, connectedness, isolated-vertex, and Euler characteristic checks).
- `TopologyGuarantee::PLManifold` and `PLManifoldStrict` add **link-based** conditions (ridge links
  and/or vertex links) that are characteristic of PL-manifolds. In PL topology, requiring the links
  of simplices to be spheres (or balls at the boundary) is equivalent to the standard manifold condition that
  every point has a locally Euclidean neighborhood (up to PL homeomorphism).

The precise **when/where** of these checks (during insertion vs at completion) is described in the
crate-level API docs and implemented by the validation stack; this document focuses on the rationale
and intuition.

---

## Link-based manifold validation

### Vertex links

A **vertex link** is the simplicial complex formed by taking all simplices incident
to a given vertex and removing that vertex from each simplex. Intuitively, the
vertex link represents the local neighborhood “around” the vertex, abstracted
away from the embedding space.

For a PL-manifold, the link of every interior vertex must be homeomorphic to a
(d−1)-sphere, where d is the dimension of the triangulation. Boundary vertices
must have links homeomorphic to a (d−1)-ball. These conditions characterize local
manifoldness at vertices and rule out singularities such as cones, pinched points,
or branching neighborhoods.

Vertex-link validation is strictly stronger than ridge-link validation: it can
detect global or vertex-local pathologies that are invisible to codimension-2
checks alone. However, constructing and validating full vertex links is
computationally expensive, as it requires enumerating the complete star of each
vertex and verifying topological properties of the resulting complex.

For this reason, the `delaunay` crate defers vertex-link validation until
construction completion by default. When stronger guarantees are required,
`TopologyGuarantee::PLManifoldStrict` enables vertex-link validation after every
insertion, trading performance for earlier detection and improved diagnosability.

### Ridge links

A **ridge** is a codimension‑2 simplex (e.g. an edge in 3D, a triangle in 4D).
The *link* of a ridge consists of the set of simplices incident to that ridge,
with the ridge itself removed.

For a PL‑manifold, the link of every interior ridge must be homeomorphic to a
circle (in 3D) or, more generally, a 1‑sphere. Boundary ridges must have links
homeomorphic to an interval. Violations of this condition indicate local
non‑manifold behavior such as branching or pinching.

The `delaunay` crate exploits this property during **incremental insertion**:
ridge‑link validation is applied as cavities are created and retriangulated.
This detects the majority of topological failures early, while keeping per‑step
cost low. Because ridge links are small and localized, they can be checked
efficiently without scanning the full star of each vertex.

Ridge‑link validation is *necessary but not sufficient* to fully guarantee
PL‑manifoldness. Certain global or vertex‑local pathologies are only detectable
via vertex‑link validation, which is why vertex‑link checks are deferred until
construction completion by default.

---

## Incremental validation strategy

The implementation uses a **hybrid validation strategy** intended to balance:

- fast incremental construction,
- early detection of common topological failures,
- and the ability to certify stronger PL-manifold conditions.

At a high level:

- **Ridge-link validation during insertion** is used as an inexpensive, local safety check. Ridge
  links are small, local objects, and validating them catches many PL-manifold violations early.
- **Vertex-link validation** is stronger but significantly more expensive. The default strategy is
  to defer full vertex-link certification until construction completion.
- **Strict mode** (`TopologyGuarantee::PLManifoldStrict`) runs vertex-link validation after each
  insertion, trading performance for earlier detection and improved diagnosability.

### Incremental insertion algorithm (cavity-based)

The crate’s incremental construction follows the standard cavity-based approach (CGAL-style):

1. **Locate** the simplex containing the query point (facet walking / scan fallback).
2. **Find the conflict region**: the set of cells whose circumspheres contain the point.
3. **Extract the cavity boundary** (a set of boundary facets separating conflicting from
   non-conflicting cells).
4. **Remove** the conflicting cells.
5. **Fill** the cavity by connecting the new vertex to the cavity boundary.
6. **Wire neighbors** locally (without global recomputation).

For points outside the current convex hull, an exterior insertion path extends the hull by
identifying *visible* boundary facets and retriangulating the visible region.

### Degenerate input and initial simplex construction

Construction begins by creating an initial simplex from the first `D+1` affinely independent
vertices. If no non-degenerate simplex can be formed (e.g., collinear points in 2D, coplanar in 3D),
construction fails with a geometric degeneracy error.

This early degeneracy detection is intentional: it prevents building a combinatorial structure whose
geometric interpretation is undefined.

### Tradeoffs

- Ridge-link checks are “cheap and local” and therefore viable as an insertion-time safety-net.
- Vertex-link checks are “expensive and global” and therefore better suited to completion-time
  certification unless strict guarantees are required.
- Ordering heuristics (Hilbert/Morton) can improve locality and reduce cavity size, improving
  robustness in practice without changing the formal correctness contract.

---

## Insertion ordering and locality heuristics

### Hilbert ordering

Hilbert ordering refers to sorting vertices along a space-filling Hilbert curve
prior to incremental insertion. Hilbert curves have strong locality-preserving
properties: points that are close in Euclidean space tend to be close along the
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

### Morton (Z-order) ordering

Morton (Z-order) ordering sorts points by interleaving coordinate bits (after an appropriate
normalization). Like Hilbert ordering, it is a space-filling curve that tends to preserve locality,
but typically has weaker locality properties than Hilbert.

In practice:

- Morton is often simpler/faster to compute than Hilbert ordering.
- It can still reduce cavity growth and improve cache behavior compared to raw input order.
- It may be preferred in performance-sensitive preprocessing paths where “good enough” locality is
  sufficient.

As with Hilbert ordering, this affects **performance and robustness**, not the formal correctness
guarantees: the same invariants are enforced regardless of insertion order.

---

## Convergence considerations

Many “repair” and “editing” workflows in high dimensions rely on sequences of **bistellar flips**
(Pachner moves) to improve topology or restore the Delaunay property.

Important caveats:

- Convergence of local flip sequences is sensitive to both geometry (near-degeneracy) and topology.
- Relaxing topology guarantees (e.g., allowing only pseudomanifold checks) can admit intermediate
  states in which flip sequences are ill-posed or fail to converge.
- Even under PL-manifold constraints, numerical predicates can be borderline for ill-conditioned
  inputs, which can lead to non-progressing local operations.

The crate therefore treats flip/repair as a best-effort procedure with explicit validation hooks:

- Prefer to validate Level 3 topology (`Triangulation::validate` / `TopologyGuarantee`) when running
  flip-heavy workflows.
- Validate the Delaunay property (Level 4) explicitly when inputs are near-degenerate.

See the public API docs and `docs/workflows.md` for practical guidance.

---

## Limitations and pathological cases

Some limitations are inherent to incremental high-dimensional computational geometry:

- **Degenerate geometry in higher dimensions**: highly degenerate point configurations (many
  nearly coplanar / collinear subsets) can cause insertion to fail or require perturbation.
- **Iterative refinement constraints**: cavity-based insertion and flip-based repair are local
  procedures. In rare cases, local refinement can be blocked by topology or by non-progressing
  numerical predicates.
- **Numerical precision**: floating-point robustness is a fundamental constraint. Robust predicates
  substantially reduce failures, but extreme coordinate magnitudes or ill-conditioned point sets can
  still trigger edge cases.

Ordering and preprocessing can mitigate (but not eliminate) these issues:

- Locality-preserving orders (Hilbert / Morton) tend to keep cavities small and reduce flip cascades.
- Deduplication / near-duplicate rejection avoids many “almost coincident” degeneracies.

For concrete failure modes and recommended workflows, see `docs/workflows.md`, `docs/validation.md`,
and the issue investigation notes in `docs/archive/`.

---

## Further reading

This section is a short, curated list of references most directly relevant to the invariants and
validation strategy described in this document.

For the full project bibliography (including many more specialized topics), see `REFERENCES.md`:

- <https://github.com/acgetchell/delaunay/blob/main/REFERENCES.md>

### Selected references for invariants and validation

- The CGAL Project. "CGAL User and Reference Manual." CGAL Editorial Board, 6.0.1 edition, 2024.
  <https://doc.cgal.org/6.0.1/Manual/packages.html>
- de Berg, M., et al. "Computational Geometry: Algorithms and Applications." 3rd ed.
  Springer-Verlag, 2008. DOI: <https://doi.org/10.1007/978-3-540-77974-2>
- Edelsbrunner, H. *Geometry and Topology for Mesh Generation*. Cambridge University Press, 2001.
- Shewchuk, J. R. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates."
  *Discrete & Computational Geometry* 18, no. 3 (1997): 305-363. DOI: <https://doi.org/10.1007/PL00009321>
- Edelsbrunner, H., and Shah, N. R. "Incremental Topological Flipping Works for Regular Triangulations."
  *Algorithmica* 15, no. 3 (1996): 223-241. DOI: <https://doi.org/10.1007/BF02523408>
- Pachner, U. "PL Homeomorphic Manifolds Are Equivalent by Elementary Shellings."
  *European Journal of Combinatorics* 12, no. 2 (1991): 129-145.
  DOI: <https://doi.org/10.1016/S0195-6698(13)80028-2>
- Hatcher, A. *Algebraic Topology*. Cambridge University Press, 2002.
  (Appendix A: PL Manifolds and Links.)
- Rourke, C. P., and Sanderson, B. J. *Introduction to Piecewise-Linear Topology*. Springer, 1972.

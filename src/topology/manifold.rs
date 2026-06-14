//! # PL-Manifold Validation (Topology Only)
//!
//! This module implements *topological* (combinatorial) invariants that are useful
//! for certifying that a finite simplicial complex is a **piecewise-linear (PL) manifold
//! with boundary**, assuming the underlying triangulation data structure (TDS) is
//! structurally consistent.
//!
//! ## What is being validated
//!
//! Let `K` be a pure D-dimensional simplicial complex. Classical PL topology shows that
//! `K` is a PL-manifold with boundary if and only if the following local conditions hold:
//!
//! 1. **Codimension-1 (facet) condition**
//!    Every (D−1)-simplex is incident to exactly one or two D-simplices.
//!
//! 2. **Boundary codimension-2 condition**
//!    Every (D−2)-simplex lying on the boundary is incident to exactly two boundary facets
//!    (equivalently, ∂² = ∅).
//!
//! 3. **Codimension-2 link condition**
//!    The link of every (D−2)-simplex is a 1-dimensional PL-manifold:
//!    - a cycle (S¹) for interior ridges, or
//!    - a path (I) for boundary ridges.
//!
//! These conditions are enforced respectively by:
//!
//! - [`validate_facet_degree`]
//! - [`validate_closed_boundary`]
//! - [`validate_ridge_links`]
//!
//! These conditions are necessary and catch many common non-manifold singularities, but they
//! are **not sufficient** in general (for D≥3) to certify full PL-manifoldness.
//!
//! To certify that `K` is a **PL-manifold with boundary**, the canonical condition is:
//!
//! 4. **Vertex-link condition (canonical PL-manifold test)**
//!    For every vertex `v`, the link `Lk(v)` is a (D−1)-sphere if `v` is an interior //!    or a (D−1)-ball if `v` lies on the boundary.
//!
//! This condition is enforced by [`validate_vertex_links`].
//!
//! ## What is *not* checked here
//!
//! - Geometric predicates (e.g. Delaunay conditions)
//! - Metric properties
//! - Global topology classification (e.g. genus, Euler characteristic)
//! - TDS structural invariants (neighbor pointers, index validity, etc.)
//!
//! Those concerns are handled elsewhere:
//!
//! - TDS structural correctness is validated at **Level 2**
//! - Global invariants (Euler characteristic, classification) live in
//!   `topology::characteristics`
//!
//! ## Ridge links vs vertex links
//!
//! PL-manifoldness is canonically defined in terms of **vertex links**. This module provides
//! both:
//!
//! - **Ridge-link validation** ([`validate_ridge_links`]) as an efficient, highly-local check
//!   that detects many codimension-2 branching/wedge singularities.
//! - **Vertex-link validation** ([`validate_vertex_links`]) as the definitive (topological)
//!   certificate for PL-manifoldness.
//!
//! In debug/test builds, or when a strong `TopologyGuarantee` is requested, callers should
//! prefer running [`validate_vertex_links`] (optionally in addition to ridge links).
//!
//! ## References
//!
//! - J. R. Munkres, *Elements of Algebraic Topology*, Addison–Wesley, 1984.
//!   (Chapter 9: Simplicial Manifolds and Links.)
//!
//! - C. P. Rourke & B. J. Sanderson, *Introduction to Piecewise-Linear Topology*,
//!   Springer, 1972.
//!
//! - A. Hatcher, *Algebraic Topology*, Cambridge University Press, 2002.
//!   (Appendix A: PL Manifolds and Links.)
//!
//! - H. Edelsbrunner & J. Harer, *Computational Topology*, AMS, 2010.
//!   (Sections on pseudomanifolds and manifold conditions in simplicial complexes.)
//!
//! These invariants are standard in computational geometry, Regge calculus, and
//! Causal Dynamical Triangulations (CDT), where curvature and local neighborhood
//! structure are only well-defined for simplicial PL-manifolds.

#![forbid(unsafe_code)]

use crate::core::{
    collections::{
        FacetToSimplicesMap, FastHashMap, FastHashSet, FastHasher, SimplexKeySet, SmallBuffer,
        VertexKeyBuffer, fast_hash_map_with_capacity, fast_hash_set_with_capacity,
    },
    facet::{FacetHandle, facet_key_from_vertices},
    tds::{SimplexKey, Tds, TdsError, VertexKey},
};
use crate::topology::characteristics::euler::{
    triangulated_surface_boundary_component_count, triangulated_surface_euler_characteristic,
};
use slotmap::Key;
use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};
use thiserror::Error;

// =============================================================================
// Periodic-aware vertex identity
// =============================================================================

/// Vertex identity in a periodic covering space.
///
/// This deliberately is not a `VertexKey`: lifted periodic images are graph
/// identities used by topology validators, not entries in the TDS vertex store.
#[derive(Clone, Debug, Eq, PartialEq)]
struct LiftedVertexId {
    vertex_key: VertexKey,
    offset: SmallBuffer<i16, 8>,
}

type LiftedVertexBuffer = SmallBuffer<LiftedVertexId, 8>;
type LinkSimplexBuffer = SmallBuffer<LiftedVertexBuffer, 8>;

impl LiftedVertexId {
    fn base(vertex_key: VertexKey) -> Self {
        Self {
            vertex_key,
            offset: SmallBuffer::new(),
        }
    }

    fn is_base(&self) -> bool {
        self.offset.is_empty()
    }
}

impl Ord for LiftedVertexId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.vertex_key
            .data()
            .as_ffi()
            .cmp(&other.vertex_key.data().as_ffi())
            .then_with(|| self.offset.as_slice().cmp(other.offset.as_slice()))
    }
}

impl PartialOrd for LiftedVertexId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for LiftedVertexId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_key.data().as_ffi().hash(state);
        self.offset.as_slice().hash(state);
    }
}

/// Creates a lifted vertex identity from a real TDS vertex key and periodic
/// lattice offset.
fn lifted_vertex_id<O>(vk: VertexKey, offset: &[O]) -> LiftedVertexId
where
    O: Copy + Into<i16>,
{
    if offset.is_empty() || offset.iter().all(|&o| o.into() == 0) {
        return LiftedVertexId::base(vk);
    }
    LiftedVertexId {
        vertex_key: vk,
        offset: offset.iter().map(|&component| component.into()).collect(),
    }
}

/// Computes a periodic-aware simplex key from lifted vertex IDs.
fn periodic_simplex_key(lifted_vertices: &[LiftedVertexId]) -> u64 {
    if lifted_vertices.iter().all(LiftedVertexId::is_base) {
        let bare_vertices: VertexKeyBuffer =
            lifted_vertices.iter().map(|id| id.vertex_key).collect();
        return facet_key_from_vertices(&bare_vertices);
    }

    let keys = normalize_lifted_vertices(lifted_vertices);
    let mut hasher = FastHasher::default();
    for key in &keys {
        key.hash(&mut hasher);
    }
    hasher.finish()
}

/// Computes an exact lifted simplex key without quotient translation normalization.
///
/// Vertex links already express every lifted vertex relative to the linked
/// anchor so applying an additional global translation quotient can
/// collapse distinct link simplices.
fn anchored_lifted_simplex_key(lifted_vertices: &[LiftedVertexId]) -> u64 {
    if lifted_vertices.iter().all(LiftedVertexId::is_base) {
        let bare_vertices: VertexKeyBuffer =
            lifted_vertices.iter().map(|id| id.vertex_key).collect();
        return facet_key_from_vertices(&bare_vertices);
    }

    let mut keys: LiftedVertexBuffer = lifted_vertices.iter().cloned().collect();
    keys.sort_unstable();
    let mut hasher = FastHasher::default();
    for key in &keys {
        key.hash(&mut hasher);
    }
    hasher.finish()
}

/// Normalizes lifted vertices by subtracting the offset of the first sorted
/// lifted making periodic simplex identities translation invariant.
fn normalize_lifted_vertices(lifted_vertices: &[LiftedVertexId]) -> LiftedVertexBuffer {
    let mut keys: LiftedVertexBuffer = lifted_vertices.iter().cloned().collect();
    keys.sort_unstable();
    let anchor_offset: SmallBuffer<i16, 8> = keys
        .first()
        .map_or_else(SmallBuffer::new, |key| key.offset.clone());
    let axes = keys
        .iter()
        .map(|key| key.offset.len())
        .max()
        .unwrap_or(0)
        .max(anchor_offset.len());

    let mut normalized = LiftedVertexBuffer::with_capacity(keys.len());
    for key in keys {
        let mut offset: SmallBuffer<i16, 8> = SmallBuffer::with_capacity(axes);
        for axis in 0..axes {
            let component = key.offset.get(axis).copied().unwrap_or(0)
                - anchor_offset.get(axis).copied().unwrap_or(0);
            offset.push(component);
        }
        normalized.push(lifted_vertex_id(key.vertex_key, &offset));
    }
    normalized
}

fn ordered_lifted_edge(a: &LiftedVertexId, b: &LiftedVertexId) -> (LiftedVertexId, LiftedVertexId) {
    if b < a {
        (b.clone(), a.clone())
    } else {
        (a.clone(), b.clone())
    }
}

/// Errors that can occur during manifold (topology) validation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::validation::ManifoldError;
///
/// let err = ManifoldError::BoundaryRidgeMultiplicity {
///     ridge_key: 1,
///     boundary_facet_count: 3,
/// };
/// std::assert_matches!(err, ManifoldError::BoundaryRidgeMultiplicity { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ManifoldError {
    /// The underlying triangulation data structure is internally inconsistent.
    #[error(transparent)]
    Tds(#[from] TdsError),

    /// A facet belongs to an unexpected number of simplices for a manifold-with-boundary.
    #[error(
        "Non-manifold facet: facet {facet_key:016x} belongs to {simplex_count} simplices (expected 1 or 2)"
    )]
    ManifoldFacetMultiplicity {
        /// The facet key with invalid multiplicity.
        facet_key: u64,
        /// The number of incident simplices observed.
        simplex_count: usize,
    },

    /// Boundary is not a closed (D-1)-manifold: a ridge on the boundary is incident to the
    /// wrong number of boundary facets.
    ///
    /// This detects "boundary of boundary" issues (codimension-2 manifoldness of the boundary).
    #[error(
        "Boundary is not closed: boundary ridge {ridge_key:016x} is incident to {boundary_facet_count} boundary facets (expected 2)"
    )]
    BoundaryRidgeMultiplicity {
        /// Canonical key for the (D-2)-simplex (ridge) on the boundary.
        ridge_key: u64,
        /// Number of incident boundary facets observed.
        boundary_facet_count: usize,
    },

    /// A ridge's link graph is not a 1-manifold (path or cycle).
    ///
    /// In a PL-manifold (with boundary), the link of every (D-2)-simplex is:
    /// - a cycle (S¹) for interior ridges, or
    /// - a path (I) for boundary ridges.
    #[error(
        "Ridge link is not a 1-manifold: ridge {ridge_key:016x} has link graph with {link_vertex_count} vertices, {link_edge_count} edges, max degree {max_degree}, degree-1 vertices {degree_one_vertices}, connected={connected} (expected connected cycle or path)"
    )]
    RidgeLinkNotManifold {
        /// Canonical key for the (D-2)-simplex (ridge).
        ridge_key: u64,
        /// Number of vertices in the ridge's link graph.
        link_vertex_count: usize,
        /// Number of edges in the ridge's link graph.
        link_edge_count: usize,
        /// Maximum vertex degree observed in the link graph.
        max_degree: usize,
        /// Number of vertices of degree 1 observed in the link graph.
        degree_one_vertices: usize,
        /// Whether the link graph is connected.
        connected: bool,
    },
    /// A vertex link is not a (D-1)-manifold (sphere/ball) as required for PL-manifoldness.
    #[error(
        "Vertex link is not a PL (D-1)-manifold: vertex {vertex_key:?} has link with {link_vertex_count} vertices, {link_simplex_count} simplices, boundary_facets={boundary_facet_count}, max_degree={max_degree}, connected={connected}, interior_vertex={interior_vertex}"
    )]
    VertexLinkNotManifold {
        /// The vertex whose link failed validation.
        vertex_key: VertexKey,
        /// Number of vertices in the link (0-simplices of the link).
        link_vertex_count: usize,
        /// Number of (D-1)-simplices (simplices) in the link.
        link_simplex_count: usize,
        /// Number of boundary facets in the link (facets of degree 1).
        boundary_facet_count: usize,
        /// Maximum degree in the link 1-skeleton.
        max_degree: usize,
        /// Whether the link 1-skeleton is connected.
        connected: bool,
        /// Whether the vertex was classified as an interior vertex of the original complex.
        interior_vertex: bool,
    },
}

/// Errors returned when parsing raw vertex keys into a ridge vertex set.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum RidgeVerticesError {
    /// Ridge vertices are only meaningful for dimensions `D >= 2`.
    #[error("ridge vertices require D >= 2, got D={dimension}")]
    UnsupportedDimension {
        /// Requested triangulation dimension.
        dimension: usize,
    },

    /// The supplied vertex count does not match the ridge arity `D - 1`.
    #[error("ridge vertex count mismatch for {dimension}D: expected {expected}, got {actual}")]
    WrongArity {
        /// Requested triangulation dimension.
        dimension: usize,
        /// Expected number of ridge vertices.
        expected: usize,
        /// Actual number of supplied vertices.
        actual: usize,
    },

    /// A ridge cannot contain the same vertex more than once.
    #[error("ridge vertices contain duplicate vertex key {vertex_key:?}")]
    DuplicateVertex {
        /// Duplicate vertex key.
        vertex_key: VertexKey,
    },
}

/// Validated vertex keys for a `(D - 2)`-simplex ridge.
///
/// This proof-bearing wrapper encodes the arity and uniqueness invariants for
/// ridge-star queries before they reach topology computation. It stores vertex
/// keys in canonical sorted order so the same ridge has the same identity
/// regardless of input order. It does not prove that the vertices exist in a
/// particular [`Tds`]; that dynamic check remains part of
/// [`ridge_star_simplices`].
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayTriangulation, DelaunayTriangulationConstructionError,
/// };
/// use delaunay::prelude::topology::validation::{RidgeVertices, RidgeVerticesError};
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)] Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)] Ridge(#[from] RidgeVerticesError),
/// #     #[error("constructed triangulation has no vertex keys")] Empty,
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let triangulation = DelaunayTriangulation::new(&vertices)?;
/// let Some(v0) = triangulation.tds().vertex_keys().next() else {
///     return Err(ExampleError::Empty);
/// };
///
/// // In 2D, a ridge is a so the validated ridge set has arity 1.
/// let ridge = RidgeVertices::<2>::try_from_vertices([v0])?;
/// assert_eq!(ridge.as_slice(), &[v0]);
/// assert_eq!(ridge.iter().collect::<Vec<_>>(), vec![v0]);
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RidgeVertices<const D: usize> {
    vertices: VertexKeyBuffer,
}

impl<const D: usize> RidgeVertices<D> {
    /// Parses raw vertex keys into a validated ridge vertex set.
    ///
    /// Stored vertex keys are canonicalized into sorted order.
    ///
    /// # Errors
    ///
    /// Returns [`RidgeVerticesError::UnsupportedDimension`] when `D < 2`,
    /// [`RidgeVerticesError::WrongArity`] when the input length is not `D - 1`,
    /// or [`RidgeVerticesError::DuplicateVertex`] when a vertex key is repeated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayTriangulation, DelaunayTriangulationConstructionError,
    /// };
    /// use delaunay::prelude::topology::validation::{RidgeVertices, RidgeVerticesError};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)] Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)] Ridge(#[from] RidgeVerticesError),
    /// #     #[error("constructed triangulation has fewer than two vertex keys")] TooFewVertices,
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
    ///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
    /// ];
    /// let triangulation = DelaunayTriangulation::new(&vertices)?;
    /// let keys = triangulation.tds().vertex_keys().collect::<Vec<_>>();
    /// let [v0, v1, ..] = keys.as_slice() else {
    ///     return Err(ExampleError::TooFewVertices);
    /// };
    ///
    /// // In 3D, a ridge is an edge and therefore has two vertices.
    /// let ridge = RidgeVertices::<3>::try_from_vertices([*v1, *v0])?;
    /// let mut expected = vec![*v0, *v1];
    /// expected.sort_unstable();
    /// assert_eq!(ridge.as_slice(), expected.as_slice());
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_from_vertices(
        vertices: impl IntoIterator<Item = VertexKey>,
    ) -> Result<Self, RidgeVerticesError> {
        if D < 2 {
            return Err(RidgeVerticesError::UnsupportedDimension { dimension: D });
        }

        let mut vertices: VertexKeyBuffer = vertices.into_iter().collect();
        let expected = D - 1;
        if vertices.len() != expected {
            return Err(RidgeVerticesError::WrongArity {
                dimension: D,
                expected,
                actual: vertices.len(),
            });
        }

        vertices.sort_unstable();
        for duplicate_pair in vertices.windows(2) {
            if duplicate_pair[0] == duplicate_pair[1] {
                return Err(RidgeVerticesError::DuplicateVertex {
                    vertex_key: duplicate_pair[0],
                });
            }
        }

        Ok(Self { vertices })
    }

    /// Returns the validated ridge vertex keys.
    #[must_use]
    pub fn as_slice(&self) -> &[VertexKey] {
        &self.vertices
    }

    /// Iterates over the validated ridge vertex keys.
    pub fn iter(&self) -> impl Iterator<Item = VertexKey> + '_ {
        self.vertices.iter().copied()
    }
}

/// Validates that each (D-1)-facet has degree 1 (boundary) or 2 (interior).
///
/// This enforces the codimension-1 pseudomanifold condition and is not sufficient by itself
/// to guarantee full PL-manifoldness.
///
/// # Errors
///
/// Returns [`ManifoldError::ManifoldFacetMultiplicity`] if any facet is incident
/// to a number of simplices other than 1 or 2.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::*;
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::validate_facet_degree;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)] Construction(#[from] delaunay::prelude::triangulation::TriangulationConstructionError),
/// #     #[error(transparent)] Manifold(#[from] delaunay::prelude::topology::validation::ManifoldError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices)?;
/// let facet_to_simplices = tds.build_facet_to_simplices_map()?;
///
/// validate_facet_degree(&facet_to_simplices)?;
/// # Ok(())
/// # }
/// ```
pub fn validate_facet_degree(
    facet_to_simplices: &FacetToSimplicesMap,
) -> Result<(), ManifoldError> {
    for (facet_key, simplex_facet_pairs) in facet_to_simplices {
        match simplex_facet_pairs.as_slice() {
            [_] | [_, _] => {}
            _ => {
                return Err(ManifoldError::ManifoldFacetMultiplicity {
                    facet_key: *facet_key,
                    simplex_count: simplex_facet_pairs.len(),
                });
            }
        }
    }

    Ok(())
}

/// Validates that the boundary (if present) is a closed (D-1)-manifold.
///
/// This is the codimension-2 pseudomanifold / manifold-with-boundary condition for
/// triangulations: every (D-2)-simplex (ridge) that lies on the boundary must be
/// incident to exactly 2 boundary facets.
/// This enforces the identity ∂² = ∅ (the boundary of a manifold has no boundary).
///
/// # Errors
///
/// Returns:
/// - [`ManifoldError::Tds`] if the underlying triangulation data structure is internally inconsistent.
/// - [`ManifoldError::BoundaryRidgeMultiplicity`] if a boundary ridge is incident to
///   a number of boundary facets other than 2.
///
/// Notes:
/// - Interior ridges can have arbitrary degree; this check only counts incidence among
///   boundary facets (facets with exactly 1 incident D-simplex).
/// - If the triangulation has no boundary facets, this check is a no-op.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::*;
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::validate_closed_boundary;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)] Construction(#[from] delaunay::prelude::triangulation::TriangulationConstructionError),
/// #     #[error(transparent)] Manifold(#[from] delaunay::prelude::topology::validation::ManifoldError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices)?;
/// let facet_to_simplices = tds.build_facet_to_simplices_map()?;
///
/// validate_closed_boundary(&tds, &facet_to_simplices)?;
/// # Ok(())
/// # }
/// ```
pub fn validate_closed_boundary<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesMap,
) -> Result<(), ManifoldError> {
    // The boundary is a (D-1)-complex. Codimension-2 manifoldness is only meaningful for D>=2.
    if D < 2 {
        return Ok(());
    }

    // First count boundary facets so we can reserve reasonably. Periodic
    // self-neighbor facets are closed quotient identifications, not boundary.
    let mut boundary_facet_count = 0usize;
    for simplex_facet_pairs in facet_to_simplices.values() {
        let [handle] = simplex_facet_pairs.as_slice() else {
            continue;
        };
        if is_boundary_facet_handle(tds, *handle)? {
            boundary_facet_count = boundary_facet_count.saturating_add(1);
        }
    }

    if boundary_facet_count == 0 {
        return Ok(());
    }

    // Each boundary facet contributes D ridges; each boundary ridge is shared by exactly 2
    // boundary facets in a closed boundary manifold.
    let estimated_boundary_ridges = boundary_facet_count
        .saturating_mul(D)
        .saturating_div(2)
        .max(1);

    let mut ridge_to_boundary_facet_count: FastHashMap<u64, usize> =
        fast_hash_map_with_capacity(estimated_boundary_ridges);

    let mut facet_vertices: VertexKeyBuffer = VertexKeyBuffer::with_capacity(D);
    let mut ridge_vertices: VertexKeyBuffer = VertexKeyBuffer::with_capacity(D.saturating_sub(1));

    for simplex_facet_pairs in facet_to_simplices.values() {
        // Only boundary facets (exactly one incident simplex).
        let [handle] = simplex_facet_pairs.as_slice() else {
            continue;
        };

        let simplex_key = handle.simplex_key();
        let facet_index = handle.facet_index() as usize;
        if !is_boundary_facet_handle(tds, *handle)? {
            continue;
        }

        // Derive the facet's vertex keys from the owning simplex.
        let simplex_vertices = tds.simplex_vertices(simplex_key)?;
        if facet_index >= simplex_vertices.len() {
            return Err(TdsError::IndexOutOfBounds {
                index: facet_index,
                bound: simplex_vertices.len(),
                context: format!("boundary facet index for simplex {simplex_key:?}"),
            }
            .into());
        }

        facet_vertices.clear();
        for (i, &vk) in simplex_vertices.iter().enumerate() {
            if i == facet_index {
                continue;
            }
            facet_vertices.push(vk);
        }

        if facet_vertices.len() != D {
            return Err(TdsError::DimensionMismatch {
                expected: D,
                actual: facet_vertices.len(),
                context: format!(
                    "boundary facet vertex count (simplex_key={simplex_key:?}, facet_index={facet_index})"
                ),
            }
            .into());
        }

        // Enumerate the (D-2)-faces (ridges) of this boundary facet by excluding each
        // facet vertex in turn.
        for omit in 0..facet_vertices.len() {
            ridge_vertices.clear();
            for (j, &vk) in facet_vertices.iter().enumerate() {
                if j == omit {
                    continue;
                }
                ridge_vertices.push(vk);
            }

            let ridge_key = facet_key_from_vertices(&ridge_vertices);
            *ridge_to_boundary_facet_count.entry(ridge_key).or_insert(0) += 1;
        }
    }

    for (ridge_key, boundary_facet_count) in ridge_to_boundary_facet_count {
        if boundary_facet_count != 2 {
            return Err(ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            });
        }
    }

    Ok(())
}

/// Validates pseudomanifold conditions for facets and boundary ridges touched
/// by `simplices`.
///
/// This is the local counterpart to [`validate_facet_degree`] plus
/// [`validate_closed_boundary`]. It expands each touched facet to its full
/// incident-simplex star, then checks only boundary ridges incident to those
/// touched facets. This keeps post-insertion checks local while preserving the
/// same codimension-1 and codimension-2 invariants for the mutated region.
pub(crate) fn validate_local_pseudomanifold_for_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: &[SimplexKey],
) -> Result<(), ManifoldError> {
    if D == 0 || simplices.is_empty() {
        return Ok(());
    }

    let facet_to_simplices = build_local_facet_star_map(tds, simplices)?;
    validate_facet_degree(&facet_to_simplices)?;
    validate_closed_boundary_for_local_facets(tds, &facet_to_simplices)
}

/// Builds full facet-incidence entries for facets owned by the supplied simplices.
fn build_local_facet_star_map<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: &[SimplexKey],
) -> Result<FacetToSimplicesMap, ManifoldError> {
    let mut facet_to_simplices = FacetToSimplicesMap::default();
    let mut seen_facets: FastHashSet<u64> = FastHashSet::default();

    for &simplex_key in simplices {
        let simplex_vertices = tds.simplex_vertices(simplex_key)?;
        for facet_index in 0..simplex_vertices.len() {
            let (facet_vertices, facet_vertices_bare) =
                simplex_facet_vertex_ids(tds, simplex_key, facet_index)?;
            let facet_key = periodic_simplex_key(&facet_vertices);
            if !seen_facets.insert(facet_key) {
                continue;
            }

            let incident = facet_incident_handles(tds, facet_key, &facet_vertices_bare)?;
            facet_to_simplices.insert(facet_key, incident);
        }
    }

    Ok(facet_to_simplices)
}

/// Returns lifted and bare vertices of one simplex facet by omitting `facet_index`.
fn simplex_facet_vertex_ids<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    facet_index: usize,
) -> Result<(LiftedVertexBuffer, VertexKeyBuffer), ManifoldError> {
    let simplex_vertices = tds.simplex_vertices(simplex_key)?;
    if facet_index >= simplex_vertices.len() {
        return Err(TdsError::IndexOutOfBounds {
            index: facet_index,
            bound: simplex_vertices.len(),
            context: "local lifted facet vertex extraction".to_string(),
        }
        .into());
    }

    let offsets = tds
        .simplex(simplex_key)
        .and_then(|simplex| simplex.periodic_vertex_offsets());
    let mut lifted_vertices =
        LiftedVertexBuffer::with_capacity(simplex_vertices.len().saturating_sub(1));
    let mut bare_vertices =
        VertexKeyBuffer::with_capacity(simplex_vertices.len().saturating_sub(1));

    for (idx, &vertex_key) in simplex_vertices.iter().enumerate() {
        if idx == facet_index {
            continue;
        }
        bare_vertices.push(vertex_key);
        let lifted = offsets.map_or_else(
            || LiftedVertexId::base(vertex_key),
            |simplex_offsets| lifted_vertex_id(vertex_key, &simplex_offsets[idx]),
        );
        lifted_vertices.push(lifted);
    }

    Ok((lifted_vertices, bare_vertices))
}

/// Finds all simplex/facet handles whose facet has the requested key.
fn facet_incident_handles<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_key: u64,
    facet_vertices_bare: &[VertexKey],
) -> Result<SmallBuffer<FacetHandle, 2>, ManifoldError> {
    let candidate_simplices = simplex_star_simplices(tds, facet_vertices_bare)?;
    let mut handles: SmallBuffer<FacetHandle, 2> =
        SmallBuffer::with_capacity(candidate_simplices.len().max(1));

    for simplex_key in candidate_simplices {
        let simplex_vertices = tds.simplex_vertices(simplex_key)?;
        for candidate_facet_index in 0..simplex_vertices.len() {
            let (candidate_vertices, _candidate_vertices_bare) =
                simplex_facet_vertex_ids(tds, simplex_key, candidate_facet_index)?;
            if periodic_simplex_key(&candidate_vertices) != facet_key {
                continue;
            }
            let Ok(facet_index) = u8::try_from(candidate_facet_index) else {
                return Err(TdsError::IndexOutOfBounds {
                    index: candidate_facet_index,
                    bound: u8::MAX as usize + 1,
                    context: "local facet incident handle".to_string(),
                }
                .into());
            };
            handles.push(FacetHandle::new(simplex_key, facet_index));
        }
    }

    Ok(handles)
}

/// Validates boundary closure for boundary facets present in a local facet map.
fn validate_closed_boundary_for_local_facets<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesMap,
) -> Result<(), ManifoldError> {
    if D < 2 {
        return Ok(());
    }

    let mut checked_ridges: FastHashSet<u64> = FastHashSet::default();
    for simplex_facet_pairs in facet_to_simplices.values() {
        let [handle] = simplex_facet_pairs.as_slice() else {
            continue;
        };
        if !is_boundary_facet_handle(tds, *handle)? {
            continue;
        }

        let (facet_vertices, facet_vertices_bare) =
            simplex_facet_vertex_ids(tds, handle.simplex_key(), handle.facet_index() as usize)?;
        for (ridge_vertices, ridge_vertices_bare) in
            ridge_vertices_for_facet::<D>(&facet_vertices, &facet_vertices_bare)?
        {
            let ridge_key = periodic_simplex_key(&ridge_vertices);
            if !checked_ridges.insert(ridge_key) {
                continue;
            }
            let boundary_facet_count =
                boundary_facet_count_for_ridge(tds, &ridge_vertices, &ridge_vertices_bare)?;
            if boundary_facet_count != 2 {
                return Err(ManifoldError::BoundaryRidgeMultiplicity {
                    ridge_key,
                    boundary_facet_count,
                });
            }
        }
    }

    Ok(())
}

/// Enumerates all ridges of a boundary facet.
fn ridge_vertices_for_facet<const D: usize>(
    facet_vertices: &[LiftedVertexId],
    facet_vertices_bare: &[VertexKey],
) -> Result<SmallBuffer<(LiftedVertexBuffer, VertexKeyBuffer), 8>, ManifoldError> {
    if facet_vertices.len() != D {
        return Err(TdsError::DimensionMismatch {
            expected: D,
            actual: facet_vertices.len(),
            context: "local boundary facet vertex count".to_string(),
        }
        .into());
    }
    if facet_vertices_bare.len() != D {
        return Err(TdsError::DimensionMismatch {
            expected: D,
            actual: facet_vertices_bare.len(),
            context: "local boundary facet bare vertex count".to_string(),
        }
        .into());
    }

    let mut ridges: SmallBuffer<(LiftedVertexBuffer, VertexKeyBuffer), 8> =
        SmallBuffer::with_capacity(D);
    for omit in 0..facet_vertices.len() {
        let mut ridge_vertices = LiftedVertexBuffer::with_capacity(D.saturating_sub(1));
        let mut ridge_vertices_bare = VertexKeyBuffer::with_capacity(D.saturating_sub(1));
        for (idx, vertex_key) in facet_vertices.iter().enumerate() {
            if idx != omit {
                ridge_vertices.push(vertex_key.clone());
                ridge_vertices_bare.push(facet_vertices_bare[idx]);
            }
        }
        ridges.push((ridge_vertices, ridge_vertices_bare));
    }
    Ok(ridges)
}

/// Counts boundary facets in the full star of a ridge.
fn boundary_facet_count_for_ridge<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge_vertices: &[LiftedVertexId],
    ridge_vertices_bare: &[VertexKey],
) -> Result<usize, ManifoldError> {
    let star_simplices = simplex_star_simplices(tds, ridge_vertices_bare)?;
    let mut count = 0usize;
    let mut seen_boundary_facets: FastHashSet<u64> = FastHashSet::default();

    for simplex_key in star_simplices {
        let simplex_vertices = tds.simplex_vertices(simplex_key)?;
        for facet_index in 0..simplex_vertices.len() {
            let (facet_vertices, facet_vertices_bare) =
                simplex_facet_vertex_ids(tds, simplex_key, facet_index)?;
            if !ridge_vertices
                .iter()
                .all(|ridge_vertex| facet_vertices.contains(ridge_vertex))
            {
                continue;
            }

            let facet_key = periodic_simplex_key(&facet_vertices);
            if !seen_boundary_facets.insert(facet_key) {
                continue;
            }
            let handles = facet_incident_handles(tds, facet_key, &facet_vertices_bare)?;
            match handles.len() {
                1 => {
                    if is_boundary_facet_handle(tds, handles[0])? {
                        count = count.saturating_add(1);
                    }
                }
                2 => {}
                other => {
                    return Err(ManifoldError::ManifoldFacetMultiplicity {
                        facet_key,
                        simplex_count: other,
                    });
                }
            }
        }
    }

    Ok(count)
}

/// Returns true when a one-sided facet occurrence is an actual boundary facet.
///
/// Periodic quotient TDSs may encode a closed facet identification as a single
/// facet occurrence whose neighbor slot points back to the owning simplex. That
/// is valid closed-topology metadata, not a boundary facet.
fn is_boundary_facet_handle<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    handle: FacetHandle,
) -> Result<bool, ManifoldError> {
    let simplex_key = handle.simplex_key();
    let facet_index = handle.facet_index() as usize;
    let simplex = tds
        .simplex(simplex_key)
        .ok_or_else(|| TdsError::SimplexNotFound {
            simplex_key,
            context: "boundary facet classification".to_string(),
        })?;

    let is_periodic_self_identified = simplex
        .neighbor_key(facet_index)
        .is_some_and(|neighbor| neighbor == Some(simplex_key))
        && simplex.periodic_vertex_offsets().is_some_and(|offsets| {
            !offsets.is_empty() && offsets.len() == simplex.number_of_vertices()
        });

    Ok(!is_periodic_self_identified)
}

/// Computes the star of a simplex (a set of vertices) as the set of incident D-simplices.
///
/// This is a local combinatorial query intended for reuse by topology validation and
/// (future) local topology mutations (e.g. bistellar flips).
///
/// This helper does **not** call `tds.is_valid()`; it performs lightweight checks and
/// returns [`ManifoldError::Tds`] if the underlying TDS is internally inconsistent.
fn simplex_star_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_vertices: &[VertexKey],
) -> Result<SmallBuffer<SimplexKey, 8>, ManifoldError> {
    if simplex_vertices.is_empty() {
        return Err(TdsError::InconsistentDataStructure {
            message: "simplex_star_simplices requires at least one vertex".to_string(),
        }
        .into());
    }

    // Defensive: ensure all simplex vertices exist in the vertex store.
    //
    // Note: This is cheaper than `tds.is_valid()` and provides a clearer error when
    // callers use this helper on stale keys.
    for &vk in simplex_vertices {
        if !tds.contains_vertex_key(vk) {
            return Err(TdsError::VertexNotFound {
                vertex_key: vk,
                context: "simplex star computation".to_string(),
            }
            .into());
        }
    }

    // Use the first simplex vertex to get a small candidate set (local star walk when possible).
    let candidates: SimplexKeySet =
        tds.find_simplices_containing_vertex_by_key(simplex_vertices[0]);

    let mut star_simplices: SmallBuffer<SimplexKey, 8> =
        SmallBuffer::with_capacity(candidates.len());

    for simplex_key in candidates {
        let candidate_vertices = tds.simplex_vertices(simplex_key)?;
        if simplex_vertices
            .iter()
            .all(|&sv| candidate_vertices.contains(&sv))
        {
            star_simplices.push(simplex_key);
        }
    }

    Ok(star_simplices)
}

/// Computes the link simplices induced by a simplex star.
///
/// For each incident D-simplex, this returns the complementary vertex set (the vertices in the
/// D-simplex that are not in `simplex_vertices`). This is the standard combinatorial definition of
/// the link of a simplex in a pure simplicial complex.
fn simplex_link_simplices_from_star<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_vertices: &[VertexKey],
    star_simplices: &[SimplexKey],
) -> Result<LinkSimplexBuffer, ManifoldError> {
    if simplex_vertices.is_empty() {
        return Err(TdsError::InconsistentDataStructure {
            message: "simplex_link_simplices_from_star requires at least one vertex".to_string(),
        }
        .into());
    }

    let expected_link_vertices = (D + 1).saturating_sub(simplex_vertices.len());

    let mut link_simplices: LinkSimplexBuffer = SmallBuffer::with_capacity(star_simplices.len());
    let mut seen_link_simplices: FastHashSet<u64> =
        fast_hash_set_with_capacity(star_simplices.len().max(1));

    for &simplex_key in star_simplices {
        let candidate_vertices = tds.simplex_vertices(simplex_key)?;
        let offsets = tds
            .simplex(simplex_key)
            .and_then(|c| c.periodic_vertex_offsets());

        // Find the reference offset: the first simplex vertex's offset in
        // this simplex.  All link vertex offsets are computed relative to this
        // so that shared vertices across TDS-adjacent simplices get the same
        // lifted ID (adjacent simplices may store different absolute offsets
        // for the same quotient-space vertex).
        let ref_slot = offsets.and_then(|_| {
            candidate_vertices
                .iter()
                .position(|cv| simplex_vertices.contains(cv))
        });

        let mut link_vertices: LiftedVertexBuffer =
            LiftedVertexBuffer::with_capacity(expected_link_vertices);
        for (i, &vk) in candidate_vertices.iter().enumerate() {
            // Membership test on bare key: the input simplex (e.g. a single
            // vertex) IS the same vertex regardless of periodic offset.
            if !simplex_vertices.contains(&vk) {
                // Lift with *relative* offset so adjacent simplices agree.
                let lifted = match (offsets, ref_slot) {
                    (Some(offs), Some(r)) => {
                        let rel: SmallBuffer<i16, 8> = offs[i]
                            .iter()
                            .zip(offs[r].iter())
                            .map(|(&a, &b)| i16::from(a) - i16::from(b))
                            .collect();
                        lifted_vertex_id(vk, &rel)
                    }
                    _ => LiftedVertexId::base(vk),
                };
                link_vertices.push(lifted);
            }
        }

        if link_vertices.len() != expected_link_vertices {
            return Err(TdsError::DimensionMismatch {
                expected: expected_link_vertices,
                actual: link_vertices.len(),
                context: format!(
                    "simplex link vertex count for {D}D (simplex_key={simplex_key:?})"
                ),
            }
            .into());
        }

        let link_key = anchored_lifted_simplex_key(&link_vertices);
        if seen_link_simplices.insert(link_key) {
            link_simplices.push(link_vertices);
        }
    }

    Ok(link_simplices)
}

/// Computes the star of a ridge (a (D-2)-simplex) as the set of incident D-simplices.
///
/// This is a local combinatorial query intended for reuse by topology validation and
/// (future) local topology mutations (e.g. bistellar flips).
///
/// This helper does **not** call `tds.is_valid()`; it performs lightweight checks and
/// returns [`ManifoldError::Tds`] if the underlying TDS is internally inconsistent.
///
/// # Errors
///
/// Returns [`ManifoldError::Tds`] when any ridge vertex is missing from the
/// [`Tds`] or a candidate star simplex cannot resolve its vertex keys.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayTriangulation, DelaunayTriangulationConstructionError,
/// };
/// use delaunay::prelude::topology::validation::{
///     ManifoldError, RidgeVertices, RidgeVerticesError, ridge_star_simplices,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)] Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)] Ridge(#[from] RidgeVerticesError),
/// #     #[error(transparent)] Manifold(#[from] ManifoldError),
/// #     #[error("constructed triangulation has no vertex keys")] Empty,
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let triangulation = DelaunayTriangulation::new(&vertices)?;
/// let Some(v0) = triangulation.tds().vertex_keys().next() else {
///     return Err(ExampleError::Empty);
/// };
///
/// let ridge = RidgeVertices::<2>::try_from_vertices([v0])?;
/// let star = ridge_star_simplices(triangulation.tds(), &ridge)?;
/// assert_eq!(star.len(), 1);
/// # Ok(())
/// # }
/// ```
pub fn ridge_star_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge_vertices: &RidgeVertices<D>,
) -> Result<SmallBuffer<SimplexKey, 8>, ManifoldError> {
    simplex_star_simplices(tds, ridge_vertices.as_slice())
}

fn ridge_link_edges_from_star<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge_vertices: &[LiftedVertexId],
    star_simplices: &[SimplexKey],
) -> Result<SmallBuffer<(LiftedVertexId, LiftedVertexId), 8>, ManifoldError> {
    // Ridge links are only meaningful for D>=2.
    if D < 2 {
        return Ok(SmallBuffer::new());
    }

    let expected_ridge_vertices = D.saturating_sub(1);
    if ridge_vertices.len() != expected_ridge_vertices {
        return Err(TdsError::DimensionMismatch {
            expected: expected_ridge_vertices,
            actual: ridge_vertices.len(),
            context: format!("ridge vertex count for {D}D (link edges)"),
        }
        .into());
    }

    let mut link_edges: SmallBuffer<(LiftedVertexId, LiftedVertexId), 8> =
        SmallBuffer::with_capacity(star_simplices.len());

    let mut link_vertices: LiftedVertexBuffer = LiftedVertexBuffer::with_capacity(2);

    for &simplex_key in star_simplices {
        let Some(simplex_vertices) =
            normalized_simplex_vertices_for_lifted_target(tds, simplex_key, ridge_vertices)?
        else {
            return Err(TdsError::InconsistentDataStructure {
                message: format!(
                    "ridge star simplex {simplex_key:?} does not contain normalized ridge vertices \
                     {ridge_vertices:?}"
                ),
            }
            .into());
        };

        link_vertices.clear();
        for lifted in simplex_vertices {
            if !ridge_vertices.contains(&lifted) {
                link_vertices.push(lifted);
            }
        }

        if link_vertices.len() != 2 {
            return Err(TdsError::DimensionMismatch {
                expected: 2,
                actual: link_vertices.len(),
                context: format!("ridge link vertex count for {D}D (simplex_key={simplex_key:?})"),
            }
            .into());
        }

        if link_vertices[0] == link_vertices[1] {
            return Err(TdsError::InconsistentDataStructure {
                message: format!(
                    "Ridge link edge is a self-loop: link vertex {vk:?} repeated (simplex_key={simplex_key:?})",
                    vk = &link_vertices[0],
                ),
            }
            .into());
        }

        link_edges.push((link_vertices[0].clone(), link_vertices[1].clone()));
    }

    Ok(link_edges)
}

#[derive(Clone, Debug)]
struct RidgeStar {
    ridge_vertices: LiftedVertexBuffer,
    star_simplices: SmallBuffer<SimplexKey, 8>,
}

// Performance: This builds a ridge → star incidence map by visiting every simplex and
// enumerating its ridges.
//
// In terms of D, each simplex contributes C(D+1, 2) = O(D²) ridges, each with O(D) vertices.
// Therefore this pass is O(#simplices × C(D+1,2) × D) time (i.e., O(#simplices × D³) in D) and
// O(#simplices × C(D+1,2)) additional memory for the incidence map.
//
// This is appropriate for Level 3 topology validation / debugging, but it can be expensive
// for extremely large triangulations (e.g., millions of simplices) or higher-dimensional complexes.
fn build_ridge_star_map<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Result<FastHashMap<u64, RidgeStar>, ManifoldError> {
    let simplex_count = tds.number_of_simplices();
    if simplex_count == 0 {
        return Ok(FastHashMap::default());
    }

    // Each D-simplex has C(D+1, 2) ridges (omit two vertices).
    let ridges_per_simplex = (D + 1).saturating_mul(D) / 2;

    // A crude-but-safe estimate: in a manifold, ridges are typically incident to ~2 simplices, so the
    // number of unique ridges is often around half the total ridge incidences.
    let estimated_unique_ridges = simplex_count
        .saturating_mul(ridges_per_simplex)
        .saturating_div(2)
        .max(1);

    // Map ridge key -> ridge star (incident simplices).
    let mut ridge_to_star: FastHashMap<u64, RidgeStar> =
        fast_hash_map_with_capacity(estimated_unique_ridges);

    let mut ridge_vertices: LiftedVertexBuffer =
        LiftedVertexBuffer::with_capacity(D.saturating_sub(1));

    for (simplex_key, simplex) in tds.simplices() {
        let simplex_vertices = tds.simplex_vertices(simplex_key)?;
        let offsets = simplex.periodic_vertex_offsets();

        if simplex_vertices.len() != D + 1 {
            return Err(TdsError::DimensionMismatch {
                expected: D + 1,
                actual: simplex_vertices.len(),
                context: format!("simplex {simplex_key:?} vertex count for {D}D"),
            }
            .into());
        }

        // Enumerate ridges in this simplex by omitting two vertices.
        for omit_a in 0..simplex_vertices.len() {
            for omit_b in (omit_a + 1)..simplex_vertices.len() {
                ridge_vertices.clear();
                for (i, &vk) in simplex_vertices.iter().enumerate() {
                    if i == omit_a || i == omit_b {
                        continue;
                    }
                    // Use lifted vertex ID when periodic offsets are present.
                    let lifted = offsets.map_or_else(
                        || LiftedVertexId::base(vk),
                        |offs| lifted_vertex_id(vk, &offs[i]),
                    );
                    ridge_vertices.push(lifted);
                }

                if ridge_vertices.len() != D.saturating_sub(1) {
                    return Err(TdsError::DimensionMismatch {
                        expected: D.saturating_sub(1),
                        actual: ridge_vertices.len(),
                        context: format!("ridge vertex count for {D}D (simplex_key={simplex_key:?}, omit_a={omit_a}, omit_b={omit_b})"),
                    }
                    .into());
                }

                let normalized_ridge_vertices = normalize_lifted_vertices(&ridge_vertices);
                let ridge_key = periodic_simplex_key(&normalized_ridge_vertices);
                let star = ridge_to_star.entry(ridge_key).or_insert_with(|| RidgeStar {
                    ridge_vertices: normalized_ridge_vertices.clone(),
                    star_simplices: SmallBuffer::new(),
                });
                star.star_simplices.push(simplex_key);
            }
        }
    }

    Ok(ridge_to_star)
}

fn build_ridge_star_map_for_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: &[SimplexKey],
) -> Result<FastHashMap<u64, RidgeStar>, ManifoldError> {
    if D < 2 {
        return Ok(FastHashMap::default());
    }

    if simplices.is_empty() {
        return Ok(FastHashMap::default());
    }

    // Each D-simplex has C(D+1, 2) ridges (omit two vertices).
    let ridges_per_simplex = (D + 1).saturating_mul(D) / 2;
    let estimated_unique_ridges = simplices.len().saturating_mul(ridges_per_simplex).max(1);

    // Build a set of ridges touched by the specified simplices.
    // For periodic simplices we store both lifted vertices (for ridge identity and
    // downstream link computation) and bare vertices (for `simplex_star_simplices`
    // which looks up real TDS vertex keys).
    let mut ridge_to_vertices: FastHashMap<u64, (LiftedVertexBuffer, VertexKeyBuffer)> =
        fast_hash_map_with_capacity(estimated_unique_ridges);

    let mut ridge_vertices_bare: VertexKeyBuffer =
        VertexKeyBuffer::with_capacity(D.saturating_sub(1));
    let mut ridge_vertices_lifted: LiftedVertexBuffer =
        LiftedVertexBuffer::with_capacity(D.saturating_sub(1));

    for &simplex_key in simplices {
        if !tds.contains_simplex(simplex_key) {
            continue;
        }

        let simplex_vertices = tds.simplex_vertices(simplex_key)?;
        let offsets = tds
            .simplex(simplex_key)
            .and_then(|c| c.periodic_vertex_offsets());

        if simplex_vertices.len() != D + 1 {
            return Err(TdsError::DimensionMismatch {
                expected: D + 1,
                actual: simplex_vertices.len(),
                context: format!("simplex {simplex_key:?} vertex count for {D}D (local ridge map)"),
            }
            .into());
        }

        // Enumerate ridges in this simplex by omitting two vertices.
        for omit_a in 0..simplex_vertices.len() {
            for omit_b in (omit_a + 1)..simplex_vertices.len() {
                ridge_vertices_bare.clear();
                ridge_vertices_lifted.clear();
                for (i, &vk) in simplex_vertices.iter().enumerate() {
                    if i == omit_a || i == omit_b {
                        continue;
                    }
                    ridge_vertices_bare.push(vk);
                    // Use lifted vertex ID when periodic offsets are present.
                    let lifted = offsets.map_or_else(
                        || LiftedVertexId::base(vk),
                        |offs| lifted_vertex_id(vk, &offs[i]),
                    );
                    ridge_vertices_lifted.push(lifted);
                }

                if ridge_vertices_bare.len() != D.saturating_sub(1) {
                    return Err(TdsError::DimensionMismatch {
                        expected: D.saturating_sub(1),
                        actual: ridge_vertices_bare.len(),
                        context: format!("ridge vertex count for {D}D (simplex_key={simplex_key:?}, omit_a={omit_a}, omit_b={omit_b})"),
                    }
                    .into());
                }

                let normalized_ridge_vertices = normalize_lifted_vertices(&ridge_vertices_lifted);
                let ridge_key = periodic_simplex_key(&normalized_ridge_vertices);
                ridge_to_vertices
                    .entry(ridge_key)
                    .or_insert_with(|| (normalized_ridge_vertices, ridge_vertices_bare.clone()));
            }
        }
    }

    // For each ridge touched by the local simplex set, compute its full star.
    // Use bare ridge vertices for `simplex_star_simplices` (which looks up real TDS
    // keys), then filter to simplices sharing the same periodic image, and store
    // lifted ridge vertices in the `RidgeStar` for downstream link computation.
    let mut ridge_to_star: FastHashMap<u64, RidgeStar> =
        fast_hash_map_with_capacity(ridge_to_vertices.len().max(1));

    for (ridge_key, (lifted_vertices, bare_vertices)) in ridge_to_vertices {
        let star_simplices =
            periodic_aware_ridge_star(tds, ridge_key, &lifted_vertices, &bare_vertices)?;

        ridge_to_star.insert(
            ridge_key,
            RidgeStar {
                ridge_vertices: lifted_vertices,
                star_simplices,
            },
        );
    }

    Ok(ridge_to_star)
}

fn normalized_simplex_vertices_for_lifted_target<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    target_vertices: &[LiftedVertexId],
) -> Result<Option<LiftedVertexBuffer>, ManifoldError> {
    let simplex_vertices = tds.simplex_vertices(simplex_key)?;
    let offsets = tds
        .simplex(simplex_key)
        .and_then(|simplex| simplex.periodic_vertex_offsets());

    let Some(offsets) = offsets else {
        let vertices: LiftedVertexBuffer = simplex_vertices
            .iter()
            .copied()
            .map(LiftedVertexId::base)
            .collect();
        return Ok(Some(vertices));
    };

    let Some(anchor) = target_vertices.first() else {
        return Ok(Some(LiftedVertexBuffer::new()));
    };
    let Some(anchor_index) = simplex_vertices
        .iter()
        .position(|&vertex_key| vertex_key == anchor.vertex_key)
    else {
        return Ok(None);
    };
    let anchor_offset = offsets[anchor_index];
    let mut normalized = LiftedVertexBuffer::with_capacity(simplex_vertices.len());
    for (idx, &vertex_key) in simplex_vertices.iter().enumerate() {
        let mut relative_offset: SmallBuffer<i16, 8> = SmallBuffer::with_capacity(D);
        for axis in 0..D {
            relative_offset.push(i16::from(offsets[idx][axis]) - i16::from(anchor_offset[axis]));
        }
        normalized.push(lifted_vertex_id(vertex_key, &relative_offset));
    }
    Ok(Some(normalized))
}

/// Computes the periodic-aware star of a ridge from its lifted and bare vertex
/// representations.
///
/// Uses bare keys to find candidate simplices via [`simplex_star_simplices`], then
/// filters to simplices whose lifted ridge vertices match `lifted_vertices`.
/// For non-periodic simplices (no offsets) all candidates pass.
///
/// # Errors
///
/// Returns [`ManifoldError::Tds`] if:
/// - `simplex_star_simplices` fails (vertex not found, etc.).
/// - `simplex_vertices` fails for any candidate simplex.
/// - Periodic offset filtering produces an empty star, indicating inconsistent
///   offsets in the TDS.
fn periodic_aware_ridge_star<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge_key: u64,
    lifted_vertices: &[LiftedVertexId],
    bare_vertices: &VertexKeyBuffer,
) -> Result<SmallBuffer<SimplexKey, 8>, ManifoldError> {
    let all_star_simplices = simplex_star_simplices(tds, bare_vertices)?;

    // For periodic simplices, keep only simplices whose lifted ridge vertices agree
    // with this ridge's lifted vertices.  For non-periodic simplices the check is
    // a no-op (offsets are `None`).  We use an explicit loop instead of
    // `.filter()` so that `simplex_vertices` errors propagate.
    let mut star_simplices: SmallBuffer<SimplexKey, 8> =
        SmallBuffer::with_capacity(all_star_simplices.len());
    for &ck in &all_star_simplices {
        let Some(normalized_vertices) =
            normalized_simplex_vertices_for_lifted_target(tds, ck, lifted_vertices)?
        else {
            continue;
        };
        if lifted_vertices
            .iter()
            .all(|lv| normalized_vertices.contains(lv))
        {
            star_simplices.push(ck);
        }
    }

    // A ridge enumerated from a simplex must be incident to at least that simplex.
    // An empty star after periodic filtering indicates inconsistent offsets.
    if star_simplices.is_empty() {
        return Err(TdsError::InconsistentDataStructure {
            message: format!(
                "periodic offset filtering produced empty star for ridge \
                 {ridge_key:016x}: {count} candidate simplices were all excluded \
                 (lifted ridge vertices: {lifted:?})",
                count = all_star_simplices.len(),
                lifted = lifted_vertices,
            ),
        }
        .into());
    }

    Ok(star_simplices)
}

/// Validates the ridge-link condition for a PL-manifold (with boundary).
///
/// For a D-dimensional simplicial complex, the link of any (D-2)-simplex is a
/// 1-dimensional simplicial complex. In a PL-manifold (with boundary), this link must
/// be a 1-manifold:
/// - a **cycle** for interior ridges, or
/// - a **path** for boundary ridges.
///
/// This check rules out wedge and branching singularities that are not detected by
/// facet-degree or boundary-closure checks alone.
///
/// It is a strict refinement of codimension-1 manifoldness, and detects common
/// singularities like “two surface components glued at a single vertex” in 2D.
///
/// # Performance
///
/// This is intentionally more expensive than basic codimension-1 manifold validation
/// (e.g., [`validate_facet_degree`]) because it must inspect ridge stars/links across the
/// entire complex.
///
/// Roughly speaking, this requires a full pass over all simplices to build a ridge → star
/// incidence map, which is O(#simplices × C(D+1,2) × D) time (linear in #simplices for fixed small D)
/// and uses additional memory proportional to total ridge incidences.
///
/// Prefer reserving this check for debug/test builds or on-demand validation in production,
/// especially for very large triangulations or higher-dimensional complexes.
///
/// # Errors
///
/// Returns:
/// - [`ManifoldError::Tds`] if the underlying triangulation data structure is internally inconsistent.
/// - [`ManifoldError::RidgeLinkNotManifold`] if any ridge has a link graph that is not a connected
///   cycle or path.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::*;
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::validate_ridge_links;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)] Construction(#[from] delaunay::prelude::triangulation::TriangulationConstructionError),
/// #     #[error(transparent)] Manifold(#[from] delaunay::prelude::topology::validation::ManifoldError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices)?;
///
/// validate_ridge_links(&tds)?;
/// # Ok(())
/// # }
/// ```
pub fn validate_ridge_links<U, V, const D: usize>(tds: &Tds<U, V, D>) -> Result<(), ManifoldError> {
    // Ridge links are only meaningful for D>=2.
    if D < 2 {
        return Ok(());
    }

    if tds.number_of_simplices() == 0 {
        return Ok(());
    }

    // Algorithm: ridge -> star (incident simplices) -> link (edges) -> validate link graph.
    let ridge_to_star = build_ridge_star_map(tds)?;
    for (ridge_key, star) in ridge_to_star {
        let link_edges =
            ridge_link_edges_from_star(tds, &star.ridge_vertices, &star.star_simplices)?;
        if let Err(err) = validate_ridge_link_graph(ridge_key, &link_edges) {
            #[cfg(debug_assertions)]
            if std::env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some() {
                let mut star_simplex_vertices: Vec<(SimplexKey, VertexKeyBuffer)> =
                    Vec::with_capacity(star.star_simplices.len());
                for &simplex_key in &star.star_simplices {
                    match tds.simplex_vertices(simplex_key) {
                        Ok(vertices) => star_simplex_vertices.push((simplex_key, vertices)),
                        Err(_) => star_simplex_vertices.push((simplex_key, VertexKeyBuffer::new())),
                    }
                }

                tracing::warn!(
                    ridge_key = ridge_key,
                    ridge_vertices = ?star.ridge_vertices,
                    star_simplices = ?star.star_simplices,
                    star_simplex_vertices = ?star_simplex_vertices,
                    link_edges = ?link_edges,
                    "validate_ridge_links: ridge link validation failed"
                );
            }
            return Err(err);
        }
    }

    Ok(())
}

/// Validates ridge links for a specific set of simplices.
///
/// This is a localized version of [`validate_ridge_links`] that only checks ridges
/// incident to the specified simplices. This is useful for post-insertion validation
/// without needing to re-validate the entire triangulation.
///
/// # Arguments
/// - `tds` - The triangulation data structure
/// - `simplices` - The specific simplices to check ridge links for
///
/// # Errors
/// Returns [`ManifoldError::RidgeLinkNotManifold`] if any ridge incident to the
/// specified simplices has a disconnected or invalid link graph.
///
/// # Examples
/// ```rust
/// use delaunay::prelude::geometry::*;
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::validate_ridge_links_for_simplices;
/// use delaunay::prelude::collections::SimplexKeyBuffer;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)] Construction(#[from] delaunay::prelude::triangulation::TriangulationConstructionError),
/// #     #[error(transparent)] Manifold(#[from] delaunay::prelude::topology::validation::ManifoldError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices)?;
/// let simplices: SimplexKeyBuffer = tds.simplices().map(|(k, _)| k).collect();
///
/// validate_ridge_links_for_simplices(&tds, &simplices)?;
/// # Ok(())
/// # }
/// ```
pub fn validate_ridge_links_for_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: &[SimplexKey],
) -> Result<(), ManifoldError> {
    // Ridge links are only meaningful for D>=2.
    if D < 2 {
        return Ok(());
    }

    if simplices.is_empty() {
        return Ok(());
    }

    // Build ridge -> star map only for ridges touching the specified simplices.
    let ridge_to_star = build_ridge_star_map_for_simplices(tds, simplices)?;

    for (ridge_key, star) in ridge_to_star {
        let link_edges =
            ridge_link_edges_from_star(tds, &star.ridge_vertices, &star.star_simplices)?;
        if let Err(err) = validate_ridge_link_graph(ridge_key, &link_edges) {
            #[cfg(debug_assertions)]
            if std::env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some() {
                let mut star_simplex_vertices: Vec<(SimplexKey, VertexKeyBuffer)> =
                    Vec::with_capacity(star.star_simplices.len());
                for &simplex_key in &star.star_simplices {
                    match tds.simplex_vertices(simplex_key) {
                        Ok(vertices) => star_simplex_vertices.push((simplex_key, vertices)),
                        Err(_) => star_simplex_vertices.push((simplex_key, VertexKeyBuffer::new())),
                    }
                }

                tracing::warn!(
                    ridge_key = ridge_key,
                    ridge_vertices = ?star.ridge_vertices,
                    star_simplices = ?star.star_simplices,
                    star_simplex_vertices = ?star_simplex_vertices,
                    link_edges = ?link_edges,
                    "validate_ridge_links_for_simplices: ridge link validation failed"
                );
            }
            return Err(err);
        }
    }

    Ok(())
}

/// Validates the vertex-link condition for PL-manifoldness (with boundary).
///
/// A pure D-dimensional simplicial complex is a PL-manifold with boundary if and only if
/// for every vertex `v`, the link `Lk(v)` is a (D-1)-sphere when `v` is interior, or a
/// (D-1)-ball when `v` lies on the boundary.
///
/// This validator treats a vertex as a *boundary vertex* if it participates in any
/// boundary facet of the original complex (a facet incident to exactly one D-simplex).
///
/// # Performance
///
/// For fixed D, this runs in time proportional to the total size of all vertex stars
/// (roughly O(#simplices × (D+1))) plus local work to build and validate each link.
///
/// # Errors
///
/// Returns:
/// - [`ManifoldError::Tds`] if the underlying triangulation data structure is internally inconsistent.
/// - [`ManifoldError::VertexLinkNotManifold`] if any vertex has a link that is not a connected
///   (D-1)-manifold with the expected boundary behavior.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::*;
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::{
///     validate_closed_boundary, validate_facet_degree, validate_vertex_links,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)] Construction(#[from] delaunay::prelude::triangulation::TriangulationConstructionError),
/// #     #[error(transparent)] Manifold(#[from] delaunay::prelude::topology::validation::ManifoldError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices)?;
/// let facet_to_simplices = tds.build_facet_to_simplices_map()?;
///
/// validate_facet_degree(&facet_to_simplices)?;
/// validate_closed_boundary(&tds, &facet_to_simplices)?;
/// validate_vertex_links(&tds, &facet_to_simplices)?;
/// # Ok(())
/// # }
/// ```
pub fn validate_vertex_links<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesMap,
) -> Result<(), ManifoldError> {
    // Vertex links are only meaningful for D>=1.
    if D < 1 {
        return Ok(());
    }

    if tds.number_of_simplices() == 0 {
        return Ok(());
    }

    let boundary_vertices = build_boundary_vertex_set(tds, facet_to_simplices)?;

    for (vertex_key, _vertex) in tds.vertices() {
        let interior_vertex = !boundary_vertices.contains(&vertex_key);
        validate_single_vertex_link(tds, vertex_key, interior_vertex)?;
    }

    Ok(())
}

fn build_boundary_vertex_set<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesMap,
) -> Result<FastHashSet<VertexKey>, ManifoldError> {
    // Single pass: collect all vertices that appear on a boundary facet (a facet incident to exactly 1 D-simplex).
    //
    // NOTE: We intentionally avoid a pre-count pass over `facet_to_simplices` since Level-3 validation is already
    // expensive and we only need a coarse set (it can grow dynamically).
    let mut boundary_vertices: FastHashSet<VertexKey> = FastHashSet::default();

    for simplex_facet_pairs in facet_to_simplices.values() {
        let [handle] = simplex_facet_pairs.as_slice() else {
            continue;
        };

        let simplex_key = handle.simplex_key();
        let facet_index = handle.facet_index() as usize;

        let simplex_vertices = tds.simplex_vertices(simplex_key)?;
        if facet_index >= simplex_vertices.len() {
            return Err(TdsError::IndexOutOfBounds {
                index: facet_index,
                bound: simplex_vertices.len(),
                context: format!("boundary facet index for simplex {simplex_key:?}"),
            }
            .into());
        }

        let mut facet_vertex_count = 0usize;
        for (i, &vk) in simplex_vertices.iter().enumerate() {
            if i == facet_index {
                continue;
            }
            boundary_vertices.insert(vk);
            facet_vertex_count += 1;
        }

        if facet_vertex_count != D {
            return Err(TdsError::DimensionMismatch {
                expected: D,
                actual: facet_vertex_count,
                context: format!(
                    "boundary facet vertex count (simplex_key={simplex_key:?}, facet_index={facet_index})"
                ),
            }
            .into());
        }
    }

    Ok(boundary_vertices)
}

fn validate_vertex_link_d1(
    vertex_key: VertexKey,
    interior_vertex: bool,
    link_simplices: &LinkSimplexBuffer,
) -> Result<(), ManifoldError> {
    let mut link_vertices: FastHashSet<LiftedVertexId> =
        fast_hash_set_with_capacity(link_simplices.len().max(1));
    for simplex in link_simplices {
        for vk in simplex {
            link_vertices.insert(vk.clone());
        }
    }

    let link_vertex_count = link_vertices.len();

    // For D=1: the link is a 0-manifold.
    // - Interior vertex: Lk(v) ≅ S⁰ (2 isolated points) ⇒ exactly 2 neighbors.
    // - Boundary vertex:  Lk(v) ≅ B⁰ (1 point)           ⇒ exactly 1 neighbor.
    let ok = if interior_vertex {
        link_vertex_count == 2
    } else {
        link_vertex_count == 1
    };

    if ok {
        Ok(())
    } else {
        Err(ManifoldError::VertexLinkNotManifold {
            vertex_key,
            link_vertex_count,
            link_simplex_count: link_simplices.len(),
            boundary_facet_count: usize::from(!interior_vertex),
            max_degree: 0,
            connected: true,
            interior_vertex,
        })
    }
}

fn validate_vertex_link_d2(
    vertex_key: VertexKey,
    interior_vertex: bool,
    link_simplices: &LinkSimplexBuffer,
    link_vertex_count: usize,
    link_simplex_count: usize,
) -> Result<(), ManifoldError> {
    // In D=2, the link is a 1-manifold.
    //
    // For PL 2-manifolds, the correct local condition is:
    // - Interior vertex: Lk(v) is a cycle (S¹)            ⇒ degree-1 vertices = 0.
    // - Boundary vertex:  Lk(v) is a path (I)             ⇒ degree-1 vertices = 2.
    let boundary_facet_count = 0; // not used for 1D links

    let Some((connected, max_degree, degree_one_vertices, _vertex_count)) =
        link_1d_graph_stats(link_simplices)
    else {
        return Err(ManifoldError::VertexLinkNotManifold {
            vertex_key,
            link_vertex_count,
            link_simplex_count,
            boundary_facet_count,
            max_degree: 0,
            connected: false,
            interior_vertex,
        });
    };

    let expected_degree_one_vertices = if interior_vertex { 0 } else { 2 };
    let ok_1d = connected && max_degree <= 2 && degree_one_vertices == expected_degree_one_vertices;

    if ok_1d {
        Ok(())
    } else {
        Err(ManifoldError::VertexLinkNotManifold {
            vertex_key,
            link_vertex_count,
            link_simplex_count,
            boundary_facet_count,
            max_degree,
            connected,
            interior_vertex,
        })
    }
}

fn validate_single_vertex_link<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertex_key: VertexKey,
    interior_vertex: bool,
) -> Result<(), ManifoldError> {
    // Collect the star of the vertex.
    let star_simplices = simplex_star_simplices(tds, &[vertex_key])?;
    if star_simplices.is_empty() {
        // A vertex with empty star violates purity for a non-empty triangulation.
        return Err(ManifoldError::VertexLinkNotManifold {
            vertex_key,
            link_vertex_count: 0,
            link_simplex_count: 0,
            boundary_facet_count: 0,
            max_degree: 0,
            connected: true,
            interior_vertex,
        });
    }

    let link_simplices = simplex_link_simplices_from_star(tds, &[vertex_key], &star_simplices)?;

    // D=1: the link is a 0-manifold (S^0 for interior vertices, B^0 for boundary vertices).
    if D == 1 {
        return validate_vertex_link_d1(vertex_key, interior_vertex, &link_simplices);
    }

    let mut link_vertex_set: FastHashSet<LiftedVertexId> =
        fast_hash_set_with_capacity(link_simplices.len().saturating_mul(D).max(1));

    for simplex in &link_simplices {
        for vk in simplex {
            link_vertex_set.insert(vk.clone());
        }
    }

    let link_simplex_count = link_simplices.len();
    let link_vertex_count = link_vertex_set.len();

    // D=2: the link is a 1-manifold.
    if D == 2 {
        return validate_vertex_link_d2(
            vertex_key,
            interior_vertex,
            &link_simplices,
            link_vertex_count,
            link_simplex_count,
        );
    }

    // Connectivity + max-degree in the link 1-skeleton.
    let (connected, max_degree) = link_1_skeleton_connectivity_and_max_degree(&link_simplices);

    // D>=3: validate the (D-1)-dimensional link via facet degrees and boundary closure.
    let (boundary_facet_count, link_is_manifold) =
        validate_link_facets_and_boundary::<D>(&link_simplices, interior_vertex);

    // For D=3, the link is a triangulated 2D surface. In this case we can enforce the
    // canonical PL-manifoldness condition (sphere/ball) via Euler characteristic.
    //
    // For D>=4, Euler characteristic is not sufficient to distinguish spheres from other
    // closed manifolds in general, so we fall back to manifoldness-only checks.
    let link_topology_ok = if D == 3 {
        let (chi, boundary_components) = if link_simplices_are_base(&link_simplices) {
            let bare_simplices = bare_link_simplices(&link_simplices);
            (
                triangulated_surface_euler_characteristic(&bare_simplices),
                triangulated_surface_boundary_component_count(&bare_simplices),
            )
        } else {
            (
                triangulated_surface_euler_characteristic_for_link(&link_simplices),
                triangulated_surface_boundary_component_count_for_link(&link_simplices),
            )
        };
        if interior_vertex {
            chi == 2 && boundary_components == 0
        } else {
            chi == 1 && boundary_components == 1
        }
    } else {
        true
    };

    let ok = connected && link_is_manifold && link_topology_ok;

    if ok {
        Ok(())
    } else {
        Err(ManifoldError::VertexLinkNotManifold {
            vertex_key,
            link_vertex_count,
            link_simplex_count,
            boundary_facet_count,
            max_degree,
            connected,
            interior_vertex,
        })
    }
}

fn link_1_skeleton_connectivity_and_max_degree(
    link_simplices: &LinkSimplexBuffer,
) -> (bool, usize) {
    // Build adjacency from the 1-skeleton of the link.
    let mut unique_edges: FastHashSet<(LiftedVertexId, LiftedVertexId)> =
        fast_hash_set_with_capacity(link_simplices.len().max(1));
    let mut adjacency: FastHashMap<LiftedVertexId, LiftedVertexBuffer> =
        fast_hash_map_with_capacity(link_simplices.len().saturating_mul(2).max(1));

    for simplex in link_simplices {
        // Add all edges in the simplex.
        for i in 0..simplex.len() {
            for j in (i + 1)..simplex.len() {
                let edge = ordered_lifted_edge(&simplex[i], &simplex[j]);
                if !unique_edges.insert(edge.clone()) {
                    continue;
                }
                let (a, b) = edge;
                adjacency.entry(a.clone()).or_default().push(b.clone());
                adjacency.entry(b).or_default().push(a);
            }
        }

        // Ensure isolated vertices are present in adjacency.
        for vk in simplex {
            adjacency.entry(vk.clone()).or_default();
        }
    }

    let mut max_degree = 0usize;
    for neighbors in adjacency.values() {
        max_degree = max_degree.max(neighbors.len());
    }

    // Connectivity check.
    let connected = match adjacency.iter().next() {
        None => true,
        Some((start, _)) => {
            let mut visited: FastHashSet<LiftedVertexId> =
                fast_hash_set_with_capacity(adjacency.len().max(1));
            let mut stack: LiftedVertexBuffer =
                LiftedVertexBuffer::with_capacity(adjacency.len().max(1));
            stack.push(start.clone());

            while let Some(v) = stack.pop() {
                if !visited.insert(v.clone()) {
                    continue;
                }
                let Some(neigh) = adjacency.get(&v) else {
                    continue;
                };
                for n in neigh {
                    if !visited.contains(n) {
                        stack.push(n.clone());
                    }
                }
            }

            visited.len() == adjacency.len()
        }
    };

    (connected, max_degree)
}

fn link_1d_graph_stats(link_simplices: &LinkSimplexBuffer) -> Option<(bool, usize, usize, usize)> {
    // In D=2, link simplices are edges (2 vertices). Build an undirected graph and compute:
    // - connectivity
    // - max degree
    // - number of degree-1 vertices
    // - vertex count
    let mut unique_edges: FastHashSet<(LiftedVertexId, LiftedVertexId)> =
        fast_hash_set_with_capacity(link_simplices.len().max(1));
    let mut adjacency: FastHashMap<LiftedVertexId, SmallBuffer<LiftedVertexId, 2>> =
        fast_hash_map_with_capacity(link_simplices.len().saturating_mul(2).max(1));

    for e in link_simplices {
        if e.len() != 2 {
            return None;
        }
        let edge = ordered_lifted_edge(&e[0], &e[1]);
        if !unique_edges.insert(edge.clone()) {
            continue;
        }
        let (a, b) = edge;
        adjacency.entry(a.clone()).or_default().push(b.clone());
        adjacency.entry(b).or_default().push(a);
    }

    let vertex_count = adjacency.len();
    if vertex_count == 0 {
        return None;
    }

    let mut max_degree = 0usize;
    let mut degree_one_vertices = 0usize;
    for neighbors in adjacency.values() {
        max_degree = max_degree.max(neighbors.len());
        if neighbors.len() == 1 {
            degree_one_vertices += 1;
        }
    }

    // Connectivity.
    let connected = match adjacency.iter().next() {
        None => true,
        Some((start, _)) => {
            let mut visited: FastHashSet<LiftedVertexId> =
                fast_hash_set_with_capacity(vertex_count);
            let mut stack: LiftedVertexBuffer = LiftedVertexBuffer::with_capacity(vertex_count);
            stack.push(start.clone());

            while let Some(v) = stack.pop() {
                if !visited.insert(v.clone()) {
                    continue;
                }
                let Some(neigh) = adjacency.get(&v) else {
                    continue;
                };
                for n in neigh {
                    if !visited.contains(n) {
                        stack.push(n.clone());
                    }
                }
            }

            visited.len() == vertex_count
        }
    };

    Some((connected, max_degree, degree_one_vertices, vertex_count))
}

fn validate_link_facets_and_boundary<const D: usize>(
    link_simplices: &LinkSimplexBuffer,
    interior_vertex: bool,
) -> (usize, bool) {
    // For a vertex link in D>=3, the link dimension is (D-1) >= 2.
    // Validate that every (D-2)-facet has degree 1 or 2 (manifold-with-boundary), and
    // that the boundary (if present) has empty boundary (∂² = ∅).

    #[derive(Clone, Debug)]
    struct FacetInfo {
        vertices: LiftedVertexBuffer,
        count: usize,
    }

    let mut facet_map: FastHashMap<u64, FacetInfo> =
        fast_hash_map_with_capacity(link_simplices.len().saturating_mul(D).max(1));

    let mut facet_vertices: LiftedVertexBuffer =
        LiftedVertexBuffer::with_capacity(D.saturating_sub(1));

    for simplex in link_simplices {
        if simplex.len() != D {
            return (0, false);
        }

        for omit in 0..simplex.len() {
            facet_vertices.clear();
            for (j, vk) in simplex.iter().enumerate() {
                if j == omit {
                    continue;
                }
                facet_vertices.push(vk.clone());
            }

            if facet_vertices.len() != D.saturating_sub(1) {
                return (0, false);
            }

            let key = anchored_lifted_simplex_key(&facet_vertices);
            let entry = facet_map.entry(key).or_insert_with(|| FacetInfo {
                vertices: facet_vertices.clone(),
                count: 0,
            });
            entry.count += 1;
        }
    }

    let mut boundary_facet_count = 0usize;
    for info in facet_map.values() {
        match info.count {
            1 => boundary_facet_count += 1,
            2 => {}
            _ => return (boundary_facet_count, false),
        }
    }

    // Interior vertex => link must be closed (no boundary facets).
    if interior_vertex && boundary_facet_count != 0 {
        return (boundary_facet_count, false);
    }

    // If boundary exists in the link, validate that boundary is closed: every (D-3)-ridge
    // on the boundary is incident to exactly 2 boundary facets.
    if boundary_facet_count > 0 {
        // Only meaningful when (D-1) >= 2 => D>=3, which is our caller contract.
        let mut ridge_map: FastHashMap<u64, usize> = fast_hash_map_with_capacity(
            boundary_facet_count
                .saturating_mul(D.saturating_sub(1))
                .max(1),
        );

        let mut ridge_vertices: LiftedVertexBuffer =
            LiftedVertexBuffer::with_capacity(D.saturating_sub(2));

        for info in facet_map.values() {
            if info.count != 1 {
                continue;
            }

            let f = &info.vertices;
            for omit in 0..f.len() {
                ridge_vertices.clear();
                for (j, vk) in f.iter().enumerate() {
                    if j == omit {
                        continue;
                    }
                    ridge_vertices.push(vk.clone());
                }
                if ridge_vertices.len() != D.saturating_sub(2) {
                    return (boundary_facet_count, false);
                }
                let ridge_key = anchored_lifted_simplex_key(&ridge_vertices);
                *ridge_map.entry(ridge_key).or_insert(0) += 1;
            }
        }

        for c in ridge_map.values() {
            if *c != 2 {
                return (boundary_facet_count, false);
            }
        }
    }

    (boundary_facet_count, true)
}

fn link_simplices_are_base(link_simplices: &LinkSimplexBuffer) -> bool {
    link_simplices
        .iter()
        .flat_map(|simplex| simplex.iter())
        .all(LiftedVertexId::is_base)
}

fn bare_link_simplices(link_simplices: &LinkSimplexBuffer) -> SmallBuffer<VertexKeyBuffer, 8> {
    link_simplices
        .iter()
        .map(|simplex| {
            simplex
                .iter()
                .map(|vertex| vertex.vertex_key)
                .collect::<VertexKeyBuffer>()
        })
        .collect()
}

fn triangulated_surface_euler_characteristic_for_link(link_simplices: &LinkSimplexBuffer) -> isize {
    let mut vertices: FastHashSet<LiftedVertexId> =
        fast_hash_set_with_capacity(link_simplices.len().saturating_mul(3).max(1));
    let mut edges: FastHashSet<(LiftedVertexId, LiftedVertexId)> =
        fast_hash_set_with_capacity(link_simplices.len().saturating_mul(3).max(1));

    for simplex in link_simplices {
        for vertex in simplex {
            vertices.insert(vertex.clone());
        }
        for i in 0..simplex.len() {
            for j in (i + 1)..simplex.len() {
                edges.insert(ordered_lifted_edge(&simplex[i], &simplex[j]));
            }
        }
    }

    vertices.len().cast_signed() - edges.len().cast_signed() + link_simplices.len().cast_signed()
}

fn triangulated_surface_boundary_component_count_for_link(
    link_simplices: &LinkSimplexBuffer,
) -> usize {
    let mut edge_counts: FastHashMap<(LiftedVertexId, LiftedVertexId), usize> =
        fast_hash_map_with_capacity(link_simplices.len().saturating_mul(3).max(1));

    for simplex in link_simplices {
        if simplex.len() < 2 {
            continue;
        }
        for i in 0..simplex.len() {
            for j in (i + 1)..simplex.len() {
                *edge_counts
                    .entry(ordered_lifted_edge(&simplex[i], &simplex[j]))
                    .or_insert(0) += 1;
            }
        }
    }

    let boundary_edges: SmallBuffer<(LiftedVertexId, LiftedVertexId), 8> = edge_counts
        .into_iter()
        .filter_map(|(edge, count)| (count == 1).then_some(edge))
        .collect();
    if boundary_edges.is_empty() {
        return 0;
    }

    let mut adjacency: FastHashMap<LiftedVertexId, LiftedVertexBuffer> =
        fast_hash_map_with_capacity(boundary_edges.len().saturating_mul(2));
    for (a, b) in boundary_edges {
        adjacency.entry(a.clone()).or_default().push(b.clone());
        adjacency.entry(b).or_default().push(a);
    }

    let mut visited: FastHashSet<LiftedVertexId> = fast_hash_set_with_capacity(adjacency.len());
    let mut components = 0usize;
    for start in adjacency.keys() {
        if visited.contains(start) {
            continue;
        }
        components += 1;
        let mut stack: LiftedVertexBuffer = LiftedVertexBuffer::with_capacity(adjacency.len());
        stack.push(start.clone());
        while let Some(vertex) = stack.pop() {
            if !visited.insert(vertex.clone()) {
                continue;
            }
            let Some(neighbors) = adjacency.get(&vertex) else {
                continue;
            };
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    stack.push(neighbor.clone());
                }
            }
        }
    }

    components
}

fn validate_ridge_link_graph(
    ridge_key: u64,
    link_edges: &[(LiftedVertexId, LiftedVertexId)],
) -> Result<(), ManifoldError> {
    // De-duplicate parallel edges defensively: if the underlying TDS contains duplicate
    // simplices/edges, the ridge link can contain repeated edges which would otherwise inflate
    // degrees and edge counts.
    let mut unique_edges: FastHashSet<(LiftedVertexId, LiftedVertexId)> =
        fast_hash_set_with_capacity(link_edges.len().max(1));

    // Build adjacency lists for the (simple) link graph.
    let estimated_link_vertices = link_edges.len().saturating_mul(2).max(1);
    let mut adjacency: FastHashMap<LiftedVertexId, SmallBuffer<LiftedVertexId, 2>> =
        fast_hash_map_with_capacity(estimated_link_vertices);

    let mut max_degree = 0usize;
    let mut link_edge_count = 0usize;

    for (a, b) in link_edges {
        let edge = ordered_lifted_edge(a, b);
        if !unique_edges.insert(edge.clone()) {
            continue;
        }
        link_edge_count += 1;

        let (a, b) = edge;

        let a_neighbors = adjacency.entry(a.clone()).or_default();
        a_neighbors.push(b.clone());
        max_degree = max_degree.max(a_neighbors.len());

        let b_neighbors = adjacency.entry(b).or_default();
        b_neighbors.push(a);
        max_degree = max_degree.max(b_neighbors.len());
    }

    let link_vertex_count = adjacency.len();

    let degree_one_vertices = adjacency.values().filter(|n| n.len() == 1).count();

    // Connectivity check: traverse the link graph.
    let connected = match adjacency.iter().next() {
        None => true,
        Some((start, _)) => {
            let mut visited: FastHashSet<LiftedVertexId> =
                fast_hash_set_with_capacity(link_vertex_count);
            let mut stack: LiftedVertexBuffer =
                LiftedVertexBuffer::with_capacity(link_vertex_count);
            stack.push(start.clone());

            while let Some(v) = stack.pop() {
                if !visited.insert(v.clone()) {
                    continue;
                }

                let Some(neighbors) = adjacency.get(&v) else {
                    continue;
                };

                for n in neighbors {
                    if !visited.contains(n) {
                        stack.push(n.clone());
                    }
                }
            }

            visited.len() == link_vertex_count
        }
    };

    // A 1-manifold graph is a connected union of cycles and paths.
    // For a connected graph with max degree <=2, that reduces to:
    // - degree_one_vertices == 0  => cycle
    // - degree_one_vertices == 2  => path
    let ok = connected && max_degree <= 2 && (degree_one_vertices == 0 || degree_one_vertices == 2);

    if ok {
        Ok(())
    } else {
        Err(ManifoldError::RidgeLinkNotManifold {
            ridge_key,
            link_vertex_count,
            link_edge_count,
            max_degree,
            degree_one_vertices,
            connected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::assert_matches;

    use crate::core::facet::FacetHandle;
    use crate::core::simplex::Simplex;
    use crate::core::triangulation::Triangulation;
    use crate::geometry::kernel::FastKernel;

    use slotmap::KeyData;

    fn vk(id: u64) -> VertexKey {
        VertexKey::from(KeyData::from_ffi(id))
    }

    fn simplex(vertices: &[VertexKey]) -> LiftedVertexBuffer {
        let mut s: LiftedVertexBuffer = LiftedVertexBuffer::with_capacity(vertices.len());
        s.extend(vertices.iter().copied().map(LiftedVertexId::base));
        s
    }

    fn build_closed_surface_s2_tds_2d() -> (Tds<(), (), 2>, [VertexKey; 4]) {
        // Closed 2D simplicial complex (topologically S²): boundary of a tetrahedron.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        for tri in [[v0, v1, v2], [v0, v1, v3], [v0, v2, v3], [v1, v2, v3]] {
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![tri[0], tri[1], tri[2]], None).unwrap(),
            )
            .unwrap();
        }

        (tds, [v0, v1, v2, v3])
    }

    fn build_non_manifold_boundary_ridge_tds_3d() -> (Tds<(), (), 3>, SimplexKey, u64) {
        // Two tetrahedra that share an edge but not a facet create a non-manifold boundary:
        // the shared edge is incident to 4 boundary triangles.
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let shared_edge_v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let shared_edge_v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();

        let tet1_v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let tet1_v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let tet2_v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let tet2_v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, -1.0]).unwrap(),
            )
            .unwrap();

        let touched_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(
                    vec![shared_edge_v0, shared_edge_v1, tet1_v2, tet1_v3],
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(
                    vec![shared_edge_v0, shared_edge_v1, tet2_v2, tet2_v3],
                    None,
                )
                .unwrap(),
            )
            .unwrap();

        let expected_ridge_key = facet_key_from_vertices(&[shared_edge_v0, shared_edge_v1]);
        (tds, touched_simplex, expected_ridge_key)
    }

    #[test]
    fn test_validate_facet_degree_ok_for_single_tetrahedron() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        assert!(validate_facet_degree(&facet_to_simplices).is_ok());
    }

    #[test]
    fn test_validate_facet_degree_ok_for_two_tetrahedra_sharing_facet() {
        // Two tetrahedra share a facet => that facet has degree 2, all others degree 1.
        let mut tds: Tds<(), (), 3> = Tds::empty();

        // Shared triangle.
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();

        // Apex points on opposite sides.
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, -1.0]).unwrap(),
            )
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v4], None).unwrap(),
            )
            .unwrap();

        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();
        assert!(validate_facet_degree(&facet_to_simplices).is_ok());
    }

    #[test]
    fn test_validate_facet_degree_errors_on_non_manifold_facet_multiplicity() {
        // Three tetrahedra share a single facet -> not a manifold-with-boundary.
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();

        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 2.0]).unwrap(),
            )
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 3.0]).unwrap(),
            )
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v4], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![v0, v1, v2, v5], None).unwrap(),
            )
            .unwrap();

        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        let expected_facet_key = facet_key_from_vertices(&[v0, v1, v2]);
        match validate_facet_degree(&facet_to_simplices) {
            Err(ManifoldError::ManifoldFacetMultiplicity {
                facet_key,
                simplex_count,
            }) => {
                assert_eq!(facet_key, expected_facet_key);
                assert_eq!(simplex_count, 3);
            }
            other => panic!("Expected ManifoldFacetMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_closed_boundary_ok_for_single_tetrahedron() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        assert!(validate_closed_boundary(&tds, &facet_to_simplices).is_ok());
    }

    #[test]
    fn test_validate_closed_boundary_errors_on_out_of_bounds_facet_index() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplices().next().unwrap().0;

        // Synthesize an invalid boundary facet handle: facet indices must be < D+1.
        let mut facet_to_simplices: FacetToSimplicesMap = FacetToSimplicesMap::default();
        let mut handles: SmallBuffer<FacetHandle, 2> = SmallBuffer::new();
        handles.push(FacetHandle::new(simplex_key, u8::MAX));
        facet_to_simplices.insert(0_u64, handles);

        match validate_closed_boundary(&tds, &facet_to_simplices) {
            Err(ManifoldError::Tds(TdsError::IndexOutOfBounds { index, bound, .. })) => {
                assert!(
                    index >= bound,
                    "Expected index ({index}) >= bound ({bound})"
                );
            }
            other => panic!("Expected IndexOutOfBounds error, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_vertex_link_d1_accepts_interior_vertex_with_two_neighbors() {
        let vertex_key = vk(0);

        let mut link_simplices: LinkSimplexBuffer = SmallBuffer::new();
        link_simplices.push(simplex(&[vk(1)]));
        link_simplices.push(simplex(&[vk(2)]));

        validate_vertex_link_d1(vertex_key, true, &link_simplices).unwrap();
    }

    #[test]
    fn test_validate_vertex_link_d1_rejects_interior_vertex_with_one_neighbor() {
        let vertex_key = vk(0);

        let mut link_simplices: LinkSimplexBuffer = SmallBuffer::new();
        link_simplices.push(simplex(&[vk(1)]));

        match validate_vertex_link_d1(vertex_key, true, &link_simplices) {
            Err(ManifoldError::VertexLinkNotManifold {
                vertex_key: got,
                link_vertex_count,
                link_simplex_count,
                boundary_facet_count,
                interior_vertex,
                ..
            }) => {
                assert_eq!(got, vertex_key);
                assert!(interior_vertex);
                assert_eq!(link_vertex_count, 1);
                assert_eq!(link_simplex_count, 1);
                assert_eq!(boundary_facet_count, 0);
            }
            other => panic!("Expected VertexLinkNotManifold, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_vertex_link_d1_accepts_boundary_vertex_with_one_neighbor() {
        let vertex_key = vk(0);

        let mut link_simplices: LinkSimplexBuffer = SmallBuffer::new();
        link_simplices.push(simplex(&[vk(1)]));

        validate_vertex_link_d1(vertex_key, false, &link_simplices).unwrap();
    }

    #[test]
    fn test_validate_vertex_link_d1_rejects_boundary_vertex_with_two_neighbors() {
        let vertex_key = vk(0);

        let mut link_simplices: LinkSimplexBuffer = SmallBuffer::new();
        link_simplices.push(simplex(&[vk(1)]));
        link_simplices.push(simplex(&[vk(2)]));

        match validate_vertex_link_d1(vertex_key, false, &link_simplices) {
            Err(ManifoldError::VertexLinkNotManifold {
                vertex_key: got,
                link_vertex_count,
                link_simplex_count,
                boundary_facet_count,
                interior_vertex,
                ..
            }) => {
                assert_eq!(got, vertex_key);
                assert!(!interior_vertex);
                assert_eq!(link_vertex_count, 2);
                assert_eq!(link_simplex_count, 2);
                assert_eq!(boundary_facet_count, 1);
            }
            other => panic!("Expected VertexLinkNotManifold, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_vertex_link_d2_accepts_cycle_for_interior_vertex() {
        let vertex_key = vk(0);
        let a = vk(1);
        let b = vk(2);
        let c = vk(3);

        let mut link_simplices: LinkSimplexBuffer = SmallBuffer::new();
        link_simplices.push(simplex(&[a, b]));
        link_simplices.push(simplex(&[b, c]));
        link_simplices.push(simplex(&[c, a]));

        validate_vertex_link_d2(vertex_key, true, &link_simplices, 3, 3).unwrap();
    }

    #[test]
    fn test_validate_vertex_link_d2_accepts_path_for_boundary_vertex() {
        let vertex_key = vk(0);
        let a = vk(1);
        let b = vk(2);
        let c = vk(3);

        let mut link_simplices: LinkSimplexBuffer = SmallBuffer::new();
        link_simplices.push(simplex(&[a, b]));
        link_simplices.push(simplex(&[b, c]));

        validate_vertex_link_d2(vertex_key, false, &link_simplices, 3, 2).unwrap();
    }

    #[test]
    fn test_validate_vertex_link_d2_rejects_path_for_interior_vertex() {
        let vertex_key = vk(0);
        let a = vk(1);
        let b = vk(2);
        let c = vk(3);

        let mut link_simplices: LinkSimplexBuffer = SmallBuffer::new();
        link_simplices.push(simplex(&[a, b]));
        link_simplices.push(simplex(&[b, c]));

        match validate_vertex_link_d2(vertex_key, true, &link_simplices, 3, 2) {
            Err(ManifoldError::VertexLinkNotManifold {
                vertex_key: got,
                link_vertex_count,
                link_simplex_count,
                boundary_facet_count,
                connected,
                max_degree,
                interior_vertex,
            }) => {
                assert_eq!(got, vertex_key);
                assert!(interior_vertex);
                assert_eq!(link_vertex_count, 3);
                assert_eq!(link_simplex_count, 2);
                assert_eq!(boundary_facet_count, 0);
                assert!(connected);
                assert_eq!(max_degree, 2);
            }
            other => panic!("Expected VertexLinkNotManifold, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_vertex_link_d2_rejects_cycle_for_boundary_vertex() {
        let vertex_key = vk(0);
        let a = vk(1);
        let b = vk(2);
        let c = vk(3);

        let mut link_simplices: LinkSimplexBuffer = SmallBuffer::new();
        link_simplices.push(simplex(&[a, b]));
        link_simplices.push(simplex(&[b, c]));
        link_simplices.push(simplex(&[c, a]));

        assert_matches!(
            validate_vertex_link_d2(vertex_key, false, &link_simplices, 3, 3),
            Err(ManifoldError::VertexLinkNotManifold { .. })
        );
    }

    #[test]
    fn test_validate_vertex_link_d2_rejects_non_edge_link_simplices() {
        let vertex_key = vk(0);

        let mut link_simplices: LinkSimplexBuffer = SmallBuffer::new();
        link_simplices.push(simplex(&[vk(1)]));

        assert_matches!(
            validate_vertex_link_d2(vertex_key, true, &link_simplices, 1, 1),
            Err(ManifoldError::VertexLinkNotManifold { .. })
        );
    }

    #[test]
    fn test_validate_single_vertex_link_d1_accepts_path_middle_and_endpoints() {
        // 0--1--2 : middle vertex is interior (2 neighbors), endpoints are boundary (1 neighbor).
        let mut tds: Tds<(), (), 1> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v0, v1], None).unwrap())
            .unwrap();
        tds.insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v1, v2], None).unwrap())
            .unwrap();

        validate_single_vertex_link(&tds, v0, false).unwrap();
        validate_single_vertex_link(&tds, v1, true).unwrap();
        validate_single_vertex_link(&tds, v2, false).unwrap();
    }

    #[test]
    fn test_validate_single_vertex_link_d1_rejects_endpoint_classified_as_interior() {
        let mut tds: Tds<(), (), 1> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v0, v1], None).unwrap())
            .unwrap();

        assert_matches!(
            validate_single_vertex_link(&tds, v0, true),
            Err(ManifoldError::VertexLinkNotManifold { .. })
        );
    }

    #[test]
    fn test_validate_single_vertex_link_d2_accepts_boundary_vertex_in_single_triangle() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();

        validate_single_vertex_link(&tds, v0, false).unwrap();
        assert_matches!(
            validate_single_vertex_link(&tds, v0, true),
            Err(ManifoldError::VertexLinkNotManifold { .. })
        );
    }

    #[test]
    fn test_validate_single_vertex_link_d2_accepts_interior_vertex_in_closed_surface() {
        let (tds, [v0, ..]) = build_closed_surface_s2_tds_2d();

        validate_single_vertex_link(&tds, v0, true).unwrap();
        assert_matches!(
            validate_single_vertex_link(&tds, v0, false),
            Err(ManifoldError::VertexLinkNotManifold { .. })
        );
    }

    #[test]
    fn test_validate_vertex_links_in_2d_disk_distinguishes_boundary_and_interior_vertices() {
        // Triangulated 2D disk: a 4-cycle boundary with a single interior vertex.
        //
        // Boundary vertices must have path links (degree-one vertices = 2), while the interior
        // vertex has a cycle link (degree-one vertices = 0).
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let va = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let vb = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let vc = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();
        let vd = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let center = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5]).unwrap(),
            )
            .unwrap();

        for tri in [
            [center, va, vb],
            [center, vb, vc],
            [center, vc, vd],
            [center, vd, va],
        ] {
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![tri[0], tri[1], tri[2]], None).unwrap(),
            )
            .unwrap();
        }

        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        // Sanity: pseudomanifold-with-boundary checks pass.
        validate_facet_degree(&facet_to_simplices).unwrap();
        validate_closed_boundary(&tds, &facet_to_simplices).unwrap();

        let boundary_vertices = build_boundary_vertex_set(&tds, &facet_to_simplices).unwrap();
        for boundary_vertex in [va, vb, vc, vd] {
            assert!(boundary_vertices.contains(&boundary_vertex));
        }
        assert!(!boundary_vertices.contains(&center));

        // Boundary vertex link is a path (two degree-1 vertices).
        let star_a = simplex_star_simplices(&tds, &[va]).unwrap();
        let link_a = simplex_link_simplices_from_star(&tds, &[va], &star_a).unwrap();
        let Some((connected, max_degree, degree_one_vertices, vertex_count)) =
            link_1d_graph_stats(&link_a)
        else {
            panic!("Expected 1D link stats for boundary vertex");
        };
        assert!(connected);
        assert_eq!(max_degree, 2);
        assert_eq!(degree_one_vertices, 2);
        assert_eq!(vertex_count, 3);

        // Interior vertex link is a cycle (no degree-1 vertices).
        let star_o = simplex_star_simplices(&tds, &[center]).unwrap();
        let link_o = simplex_link_simplices_from_star(&tds, &[center], &star_o).unwrap();
        let Some((connected, max_degree, degree_one_vertices, vertex_count)) =
            link_1d_graph_stats(&link_o)
        else {
            panic!("Expected 1D link stats for interior vertex");
        };
        assert!(connected);
        assert_eq!(max_degree, 2);
        assert_eq!(degree_one_vertices, 0);
        assert_eq!(vertex_count, 4);

        // Full vertex-link validation should succeed.
        validate_vertex_links(&tds, &facet_to_simplices).unwrap();

        // And misclassifications should be rejected (guards the interior/boundary distinction).
        assert_matches!(
            validate_single_vertex_link(&tds, va, true),
            Err(ManifoldError::VertexLinkNotManifold { .. })
        );
        assert_matches!(
            validate_single_vertex_link(&tds, center, false),
            Err(ManifoldError::VertexLinkNotManifold { .. })
        );
    }

    #[test]
    fn test_validate_closed_boundary_noop_for_d_lt_2() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 1>::build_initial_simplex(&vertices).unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        // Codimension-2 boundary manifoldness is only meaningful for D>=2.
        assert!(validate_closed_boundary(&tds, &facet_to_simplices).is_ok());
    }

    #[test]
    fn test_validate_closed_boundary_noop_for_closed_2d_surface() {
        let (tds, _vertices) = build_closed_surface_s2_tds_2d();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        // Sanity: no boundary facets (every edge has exactly 2 incident triangles).
        assert!(
            facet_to_simplices
                .values()
                .all(|handles| handles.len() == 2)
        );

        assert!(validate_closed_boundary(&tds, &facet_to_simplices).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_ok_for_closed_2d_surface() {
        let (tds, _vertices) = build_closed_surface_s2_tds_2d();

        // Sanity: pseudomanifold checks pass.
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();
        validate_facet_degree(&facet_to_simplices).unwrap();
        validate_closed_boundary(&tds, &facet_to_simplices).unwrap();

        assert!(validate_ridge_links(&tds).is_ok());
    }

    #[test]
    fn test_simplex_star_simplices_errors_on_empty_simplex() {
        let tds: Tds<(), (), 2> = Tds::empty();

        match simplex_star_simplices(&tds, &[]) {
            Err(ManifoldError::Tds(TdsError::InconsistentDataStructure { ref message }))
                if message.contains("at least one vertex") => {}
            other => panic!("Expected InconsistentDataStructure for empty simplex, got {other:?}"),
        }
    }

    #[test]
    fn test_simplex_star_simplices_returns_empty_for_isolated_vertex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();

        let star = simplex_star_simplices(&tds, &[v0]).unwrap();
        assert!(star.is_empty());
    }

    #[test]
    fn test_simplex_link_simplices_from_star_errors_on_empty_simplex() {
        let tds: Tds<(), (), 2> = Tds::empty();

        match simplex_link_simplices_from_star(&tds, &[], &[]) {
            Err(ManifoldError::Tds(TdsError::InconsistentDataStructure { ref message }))
                if message.contains("at least one vertex") => {}
            other => panic!("Expected InconsistentDataStructure for empty simplex, got {other:?}"),
        }
    }

    #[test]
    fn test_simplex_link_simplices_from_star_errors_on_unrelated_vertex() {
        // Defensive: a simplex vertex that exists in the TDS but is not part of a star simplex should
        // trigger a link-size mismatch (this should not happen when star simplices are produced by
        // `simplex_star_simplices`, but is a robustness check for corrupted inputs).
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, 10.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        match simplex_link_simplices_from_star(&tds, &[v0, v3], &[simplex_key]) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 1,
                actual,
                ..
            })) => {
                assert_ne!(actual, 1, "Expected actual != 1 for unrelated vertex");
            }
            other => panic!("Expected DimensionMismatch for link-size mismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_vertices_rejects_d_lt_2() {
        match RidgeVertices::<1>::try_from_vertices([VertexKey::from(KeyData::from_ffi(0))]) {
            Err(RidgeVerticesError::UnsupportedDimension { dimension }) => {
                assert_eq!(dimension, 1);
            }
            other => panic!("Expected UnsupportedDimension for D<2, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_link_edges_from_star_noop_for_d_lt_2() {
        let tds: Tds<(), (), 1> = Tds::empty();

        let edges = ridge_link_edges_from_star(&tds, &[], &[]).unwrap();
        assert!(edges.is_empty());
    }

    #[test]
    fn test_ridge_vertices_rejects_too_few_vertices_in_3d() {
        let v0 = VertexKey::from(KeyData::from_ffi(1));

        // In 3D, ridges are edges (2 vertices). Passing a single vertex is invalid.
        match RidgeVertices::<3>::try_from_vertices([v0]) {
            Err(RidgeVerticesError::WrongArity {
                dimension: 3,
                expected: 2,
                actual: 1,
            }) => {}
            other => panic!("Expected WrongArity(2, 1) for wrong ridge size, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_link_edges_from_star_errors_on_wrong_vertex_count_in_3d() {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        // In 3D, ridges are edges (2 vertices). Passing a single vertex is invalid.
        match ridge_link_edges_from_star(&tds, &simplex(&[v0]), &[simplex_key]) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 2,
                actual: 1,
                ..
            })) => {}
            other => panic!("Expected DimensionMismatch(2, 1) for wrong ridge size, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_star_simplices_returns_incident_simplices_for_vertex_ridge_in_2d() {
        // In 2D, a ridge is a vertex and its star is the set of incident triangles.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let c012 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let c013 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();
        let c023 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
            )
            .unwrap();
        let _c123 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        let ridge_vertices = RidgeVertices::<2>::try_from_vertices([v0]).unwrap();
        let star = ridge_star_simplices(&tds, &ridge_vertices).unwrap();
        let star_set: SimplexKeySet = star.iter().copied().collect();

        let expected: SimplexKeySet = [c012, c013, c023].into_iter().collect();
        assert_eq!(star_set, expected);
    }

    #[test]
    fn test_ridge_star_simplices_returns_full_edge_star_in_3d() {
        // In 3D, a ridge is an edge. This regression protects k=3 support
        // collection from using only the anchor simplex.
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0, 0.0]).unwrap(),
            )
            .unwrap();

        let c0123 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let c0134 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3, v4], None).unwrap(),
            )
            .unwrap();
        let c0142 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v4, v2], None).unwrap(),
            )
            .unwrap();

        let ridge_vertices = RidgeVertices::<3>::try_from_vertices([v0, v1]).unwrap();
        let star = ridge_star_simplices(&tds, &ridge_vertices).unwrap();
        let star_set: SimplexKeySet = star.iter().copied().collect();

        let expected: SimplexKeySet = [c0123, c0134, c0142].into_iter().collect();
        assert_eq!(star_set, expected);
    }

    #[test]
    fn test_ridge_star_simplices_errors_on_missing_vertex_key() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let missing = VertexKey::from(KeyData::from_ffi(u64::MAX));

        let ridge_vertices = RidgeVertices::<2>::try_from_vertices([missing]).unwrap();
        match ridge_star_simplices(&tds, &ridge_vertices) {
            Err(ManifoldError::Tds(TdsError::VertexNotFound { vertex_key, .. })) => {
                assert_eq!(vertex_key, missing);
            }
            other => panic!("Expected VertexNotFound error, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_link_edges_from_star_rejects_self_loop_edge() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Corrupt the simplex in-place: keep length == D+1 but introduce a duplicate link vertex.
        {
            let simplex = tds
                .simplex_mut(simplex_key)
                .expect("simplex key should be valid in test");
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v1);
            simplex.push_vertex_key(v1);
        }

        // For ridge (vertex) v0, the link edge becomes (v1, v1), which is not a simplicial edge.
        match ridge_link_edges_from_star(&tds, &simplex(&[v0]), &[simplex_key]) {
            Err(ManifoldError::Tds(TdsError::InconsistentDataStructure { message })) => {
                assert!(
                    message.contains("self-loop"),
                    "Unexpected message: {message}"
                );
            }
            other => panic!("Expected self-loop edge error, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_ridge_link_graph_deduplicates_parallel_edges() {
        // Triangle cycle a-b-c-a, but with a duplicated edge.
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));

        let edges = vec![
            (LiftedVertexId::base(a), LiftedVertexId::base(b)),
            (LiftedVertexId::base(b), LiftedVertexId::base(c)),
            (LiftedVertexId::base(c), LiftedVertexId::base(a)),
            (LiftedVertexId::base(a), LiftedVertexId::base(b)),
        ];
        assert!(validate_ridge_link_graph(0_u64, &edges).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_errors_on_corrupted_simplex_with_duplicate_vertices() {
        // This is a defensive robustness test: a corrupted simplex with duplicate vertices can
        // produce a malformed ridge link (wrong number of link vertices).
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Corrupt the simplex in-place: keep length == D+1 but introduce a duplicate vertex.
        {
            let simplex = tds
                .simplex_mut(simplex_key)
                .expect("simplex key should be valid in test");
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v0);
        }

        match validate_ridge_links(&tds) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 2,
                actual,
                ..
            })) => {
                assert_ne!(actual, 2, "Expected actual != 2 for corrupted link");
            }
            other => {
                panic!("Expected DimensionMismatch for ridge-link structural error, got {other:?}")
            }
        }
    }

    #[test]
    fn test_validate_closed_boundary_errors_on_non_manifold_boundary_ridge() {
        let (tds, _touched_simplex, expected_ridge_key) =
            build_non_manifold_boundary_ridge_tds_3d();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        match validate_closed_boundary(&tds, &facet_to_simplices) {
            Err(ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            }) => {
                assert_eq!(ridge_key, expected_ridge_key);
                assert_eq!(boundary_facet_count, 4);
            }
            other => panic!("Expected BoundaryRidgeMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_local_pseudomanifold_for_simplices_errors_on_non_manifold_boundary_ridge() {
        let (tds, touched_simplex, expected_ridge_key) = build_non_manifold_boundary_ridge_tds_3d();

        match validate_local_pseudomanifold_for_simplices(&tds, &[touched_simplex]) {
            Err(ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            }) => {
                assert_eq!(ridge_key, expected_ridge_key);
                assert_eq!(boundary_facet_count, 4);
            }
            other => panic!("Expected BoundaryRidgeMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_local_pseudomanifold_for_simplices_errors_on_missing_scope_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        assert_eq!(tds.remove_simplices_by_keys(&[simplex_key]), 1);

        match validate_local_pseudomanifold_for_simplices(&tds, &[simplex_key]) {
            Err(ManifoldError::Tds(TdsError::SimplexNotFound {
                simplex_key: missing_key,
                ..
            })) => assert_eq!(missing_key, simplex_key),
            other => panic!("Expected SimplexNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_ridge_links_ok_for_single_tetrahedron() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();

        assert!(validate_ridge_links(&tds).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_noop_for_empty_tds() {
        let tds: Tds<(), (), 2> = Tds::empty();
        assert!(validate_ridge_links(&tds).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_rejects_wedge_at_vertex_in_2d() {
        let (tds, v0, _incident, _nonincident) = build_wedge_two_spheres_share_vertex_tds_2d();

        // Sanity: pseudomanifold-with-boundary checks pass (in fact, this complex is closed).
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();
        validate_facet_degree(&facet_to_simplices).unwrap();
        validate_closed_boundary(&tds, &facet_to_simplices).unwrap();

        let expected_ridge_key = facet_key_from_vertices(&[v0]);

        match validate_ridge_links(&tds) {
            Err(ManifoldError::RidgeLinkNotManifold {
                ridge_key,
                connected,
                degree_one_vertices,
                max_degree,
                ..
            }) => {
                assert_eq!(ridge_key, expected_ridge_key);
                assert!(!connected);
                assert_eq!(degree_one_vertices, 0);
                assert_eq!(max_degree, 2);
            }
            other => panic!("Expected RidgeLinkNotManifold, got {other:?}"),
        }
    }

    fn build_two_tetrahedra_sharing_facet_tds_3d()
    -> (Tds<(), (), 3>, [VertexKey; 5], [SimplexKey; 2]) {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        // Shared triangle.
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();

        // Opposite vertices.
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, -1.0]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v4], None).unwrap(),
            )
            .unwrap();

        (tds, [v0, v1, v2, v3, v4], [c1, c2])
    }

    fn build_wedge_two_spheres_share_vertex_tds_2d()
    -> (Tds<(), (), 2>, VertexKey, SimplexKey, SimplexKey) {
        // Two closed 2D spheres (boundaries of tetrahedra) that share a single vertex.
        // This is a pseudomanifold (every edge has degree 2), but not a PL 2-manifold:
        // the shared vertex has a disconnected link (two disjoint cycles).
        let mut tds: Tds<(), (), 2> = Tds::empty();

        // Shared vertex.
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();

        // First tetrahedron boundary (4 triangles on 4 vertices).
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let c012 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let _c013 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();
        let _c023 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
            )
            .unwrap();
        let c123 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        // Second tetrahedron boundary (shares only v0).
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, 10.0]).unwrap(),
            )
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([11.0, 10.0]).unwrap(),
            )
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([10.0, 11.0]).unwrap(),
            )
            .unwrap();

        let _c045 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v4, v5], None).unwrap(),
            )
            .unwrap();
        let _c046 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v4, v6], None).unwrap(),
            )
            .unwrap();
        let _c056 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v5, v6], None).unwrap(),
            )
            .unwrap();
        let _c456 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v4, v5, v6], None).unwrap(),
            )
            .unwrap();

        (tds, v0, c012, c123)
    }

    #[test]
    fn test_build_ridge_star_map_empty_returns_empty() {
        let tds: Tds<(), (), 3> = Tds::empty();

        let map = build_ridge_star_map(&tds).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_build_ridge_star_map_errors_on_corrupted_simplex_vertex_count() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Corrupt the simplex in-place: change it to have only 2 vertices.
        {
            let simplex = tds
                .simplex_mut(simplex_key)
                .expect("simplex key should be valid in test");
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v1);
        }

        match build_ridge_star_map(&tds) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 3,
                actual: 2,
                ..
            })) => {}
            other => {
                panic!("Expected DimensionMismatch(3, 2) for corrupted simplex, got {other:?}")
            }
        }
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_noop_for_d_lt_2() {
        let tds: Tds<(), (), 1> = Tds::empty();
        let simplex_key = SimplexKey::from(KeyData::from_ffi(0));

        let map = build_ridge_star_map_for_simplices(&tds, &[simplex_key]).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_empty_returns_empty() {
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        let map = build_ridge_star_map_for_simplices(&tds, &[]).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_3d_single_simplex_includes_only_its_ridges_and_full_stars()
     {
        let (tds, [v0, v1, v2, v3, v4], [c1, c2]) = build_two_tetrahedra_sharing_facet_tds_3d();

        // Include a missing simplex key to ensure it is skipped, not treated as an error.
        let missing = SimplexKey::from(KeyData::from_ffi(u64::MAX));

        let map = build_ridge_star_map_for_simplices(&tds, &[c1, missing]).unwrap();

        // In 3D, ridges are edges: a tetrahedron has C(4,2) = 6 edges.
        assert_eq!(map.len(), 6);

        let star_set_for_edge = |a: VertexKey, b: VertexKey| -> SimplexKeySet {
            let key = facet_key_from_vertices(&[a, b]);
            let star = map
                .get(&key)
                .expect("expected ridge key in local ridge-star map");

            // RidgeStar stores the ridge vertices; ensure its canonical key matches the map key.
            assert_eq!(periodic_simplex_key(&star.ridge_vertices), key);
            assert_eq!(star.ridge_vertices.len(), 2);

            star.star_simplices.iter().copied().collect()
        };

        let shared_star: SimplexKeySet = [c1, c2].into_iter().collect();
        let c1_only: SimplexKeySet = std::iter::once(c1).collect();

        // Shared-facet edges should have a 2-simplex star (full star across the whole TDS).
        assert_eq!(star_set_for_edge(v0, v1), shared_star);
        assert_eq!(star_set_for_edge(v0, v2), shared_star);
        assert_eq!(star_set_for_edge(v1, v2), shared_star);

        // Edges incident to the first tetrahedron's opposite vertex should have a 1-simplex star.
        assert_eq!(star_set_for_edge(v0, v3), c1_only);
        assert_eq!(star_set_for_edge(v1, v3), c1_only);
        assert_eq!(star_set_for_edge(v2, v3), c1_only);

        // Edges involving v4 belong only to c2, so they should not appear when selecting only c1.
        assert!(!map.contains_key(&facet_key_from_vertices(&[v0, v4])));
        assert!(!map.contains_key(&facet_key_from_vertices(&[v1, v4])));
        assert!(!map.contains_key(&facet_key_from_vertices(&[v2, v4])));
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_3d_two_simplices_includes_union_of_ridges() {
        let (tds, [v0, v1, v2, v3, v4], [c1, c2]) = build_two_tetrahedra_sharing_facet_tds_3d();

        let map = build_ridge_star_map_for_simplices(&tds, &[c1, c2]).unwrap();

        // Each tetrahedron has 6 edges and they share 3 edges on the shared facet => 6+6-3=9.
        assert_eq!(map.len(), 9);

        let star_size_for_edge = |a: VertexKey, b: VertexKey| -> usize {
            let key = facet_key_from_vertices(&[a, b]);
            map.get(&key)
                .expect("expected ridge key in local ridge-star map")
                .star_simplices
                .len()
        };

        // Shared edges have a 2-simplex star.
        assert_eq!(star_size_for_edge(v0, v1), 2);
        assert_eq!(star_size_for_edge(v0, v2), 2);
        assert_eq!(star_size_for_edge(v1, v2), 2);

        // Opposite-vertex edges are unique to each tetrahedron.
        assert_eq!(star_size_for_edge(v0, v3), 1);
        assert_eq!(star_size_for_edge(v1, v3), 1);
        assert_eq!(star_size_for_edge(v2, v3), 1);

        assert_eq!(star_size_for_edge(v0, v4), 1);
        assert_eq!(star_size_for_edge(v1, v4), 1);
        assert_eq!(star_size_for_edge(v2, v4), 1);
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_2d_includes_full_star_for_shared_vertex() {
        let (tds, v0, incident, _nonincident) = build_wedge_two_spheres_share_vertex_tds_2d();

        // In 2D, ridges are vertices. A single triangle touches 3 ridges.
        let map = build_ridge_star_map_for_simplices(&tds, &[incident]).unwrap();
        assert_eq!(map.len(), 3);

        // The shared vertex v0 should have a star consisting of 6 incident triangles (3 from each sphere).
        let ridge_key = facet_key_from_vertices(&[v0]);
        let star = map
            .get(&ridge_key)
            .expect("expected ridge key for shared vertex");
        assert_eq!(star.star_simplices.len(), 6);
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_errors_on_corrupted_simplex_vertex_count() {
        // Corrupt a simplex's vertex list to violate the (D+1)-vertices invariant.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        // Corrupt the simplex in-place: change it to have only 2 vertices.
        {
            let simplex = tds
                .simplex_mut(simplex_key)
                .expect("simplex key should be valid in test");
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v1);
        }

        match build_ridge_star_map_for_simplices(&tds, &[simplex_key]) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 3,
                actual: 2,
                ..
            })) => {}
            other => {
                panic!("Expected DimensionMismatch(3, 2) for corrupted simplex, got {other:?}")
            }
        }
    }

    #[test]
    fn test_validate_ridge_links_for_simplices_ok_for_single_tetrahedron_in_3d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();

        let simplices: Vec<SimplexKey> = tds.simplices().map(|(k, _)| k).collect();
        validate_ridge_links_for_simplices(&tds, &simplices).unwrap();

        // And it should be a no-op on empty simplex lists.
        validate_ridge_links_for_simplices(&tds, &[]).unwrap();
    }

    #[test]
    fn test_validate_ridge_links_for_simplices_ok_for_missing_simplex_keys() {
        // Defensive: local validation should ignore missing simplex keys.
        let tds: Tds<(), (), 3> = Tds::empty();
        let missing = SimplexKey::from(KeyData::from_ffi(u64::MAX));

        assert!(validate_ridge_links_for_simplices(&tds, &[missing]).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_for_simplices_noop_for_d_lt_2() {
        let mut tds: Tds<(), (), 1> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0]).unwrap(),
            )
            .unwrap();

        let c01 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v0, v1], None).unwrap())
            .unwrap();

        assert!(validate_ridge_links_for_simplices(&tds, &[c01]).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_for_simplices_rejects_wedge_at_vertex_in_2d() {
        let (tds, v0, incident, _nonincident) = build_wedge_two_spheres_share_vertex_tds_2d();

        let expected_ridge_key = facet_key_from_vertices(&[v0]);

        match validate_ridge_links_for_simplices(&tds, &[incident]) {
            Err(ManifoldError::RidgeLinkNotManifold {
                ridge_key,
                link_vertex_count,
                link_edge_count,
                max_degree,
                degree_one_vertices,
                connected,
            }) => {
                assert_eq!(ridge_key, expected_ridge_key);
                assert!(!connected);
                assert_eq!(link_vertex_count, 6);
                assert_eq!(link_edge_count, 6);
                assert_eq!(max_degree, 2);
                assert_eq!(degree_one_vertices, 0);
            }
            other => panic!("Expected RidgeLinkNotManifold, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_ridge_links_for_simplices_only_checks_ridges_touched_by_input_simplices() {
        // The wedge complex is globally invalid, but local ridge-link validation should only
        // consider ridges incident to the provided simplices.
        let (tds, _v0, _incident, nonincident) = build_wedge_two_spheres_share_vertex_tds_2d();

        // Validate a triangle that does NOT touch the shared vertex; this should not detect the wedge.
        assert!(validate_ridge_links_for_simplices(&tds, &[nonincident]).is_ok());
    }

    fn build_cone_on_torus_tds() -> (Tds<(), (), 3>, VertexKey) {
        // Construct a 3D simplicial complex that is a cone over a triangulated 2-torus.
        //
        // This is a pseudomanifold and passes ridge-link validation, but is NOT a PL 3-manifold:
        // the apex vertex has link homeomorphic to T^2 instead of S^2.

        const N: usize = 3;
        const M: usize = 3;

        let mut tds: Tds<(), (), 3> = Tds::empty();

        // Build a small triangulated torus using a periodic 3x3 grid.
        let mut v: [[VertexKey; M]; N] = [[VertexKey::from(KeyData::from_ffi(0)); M]; N];
        for (i, row) in v.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                let i_f = <f64 as std::convert::From<u32>>::from(u32::try_from(i).unwrap());
                let j_f = <f64 as std::convert::From<u32>>::from(u32::try_from(j).unwrap());
                *slot = tds
                    .insert_vertex_with_mapping(
                        crate::core::vertex::Vertex::<(), _>::try_new([i_f, j_f, 0.0]).unwrap(),
                    )
                    .unwrap();
            }
        }

        // Apex of the cone (interior vertex; not on any boundary facet).
        let apex = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5, 1.0]).unwrap(),
            )
            .unwrap();

        // Triangulate each periodic square into two triangles, then cone to the apex.
        for i in 0..N {
            for j in 0..M {
                let i1 = (i + 1) % N;
                let j1 = (j + 1) % M;

                let v00 = v[i][j];
                let v10 = v[i1][j];
                let v01 = v[i][j1];
                let v11 = v[i1][j1];

                for tri in [[v00, v10, v01], [v10, v11, v01]] {
                    let _ = tds
                        .insert_simplex_with_mapping(
                            Simplex::try_new_with_data(vec![tri[0], tri[1], tri[2], apex], None)
                                .unwrap(),
                        )
                        .unwrap();
                }
            }
        }

        (tds, apex)
    }

    #[test]
    fn test_ridge_links_insufficient_for_pl_manifold() {
        // Classic counterexample: a cone over a 2-torus.
        //
        // NOTE: This test is the canonical coverage for the cone-on-torus singularity and replaces the
        // previously-duplicated `test_validate_vertex_links_rejects_cone_on_torus_in_3d` after consolidation.
        let (tds, apex) = build_cone_on_torus_tds();

        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();
        validate_facet_degree(&facet_to_simplices).unwrap();
        validate_closed_boundary(&tds, &facet_to_simplices).unwrap();

        // Ridge-link validation should *not* detect this singularity.
        assert!(validate_ridge_links(&tds).is_ok());

        // Vertex-link validation MUST reject it: apex link is T^2, not S^2.
        match validate_vertex_links(&tds, &facet_to_simplices) {
            Err(ManifoldError::VertexLinkNotManifold {
                vertex_key,
                interior_vertex,
                ..
            }) => {
                assert_eq!(vertex_key, apex);
                assert!(interior_vertex);
            }
            Ok(()) => panic!("Expected VertexLinkNotManifold for cone apex, got Ok(())"),
            other => panic!("Expected VertexLinkNotManifold for cone apex, got {other:?}"),
        }
    }

    #[test]
    fn test_simplex_star_simplices_rejects_missing_vertex() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let stale_key = VertexKey::from(KeyData::from_ffi(0xDEAD));
        match simplex_star_simplices(&tds, &[stale_key]) {
            Err(ManifoldError::Tds(TdsError::VertexNotFound {
                vertex_key,
                ref context,
            })) => {
                assert_eq!(vertex_key, stale_key);
                assert!(context.contains("simplex star"));
            }
            other => panic!("Expected VertexNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_vertices_rejects_too_many_vertices_in_3d() {
        // For D=3, ridges have D-1=2 vertices; pass 3 vertices instead.
        let v0 = VertexKey::from(KeyData::from_ffi(1));
        let v1 = VertexKey::from(KeyData::from_ffi(2));
        let v2 = VertexKey::from(KeyData::from_ffi(3));
        match RidgeVertices::<3>::try_from_vertices([v0, v1, v2]) {
            Err(RidgeVerticesError::WrongArity {
                expected, actual, ..
            }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 3);
            }
            other => panic!("Expected WrongArity, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_vertices_rejects_duplicate_vertices() {
        let v0 = VertexKey::from(KeyData::from_ffi(1));

        match RidgeVertices::<3>::try_from_vertices([v0, v0]) {
            Err(RidgeVerticesError::DuplicateVertex { vertex_key }) => {
                assert_eq!(vertex_key, v0);
            }
            other => panic!("Expected DuplicateVertex, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_vertices_canonicalizes_permuted_vertices() {
        let v0 = VertexKey::from(KeyData::from_ffi(1));
        let v1 = VertexKey::from(KeyData::from_ffi(2));

        let forward = RidgeVertices::<3>::try_from_vertices([v0, v1]).unwrap();
        let reversed = RidgeVertices::<3>::try_from_vertices([v1, v0]).unwrap();

        assert_eq!(forward, reversed);
        assert_eq!(reversed.as_slice(), &[v0, v1]);
    }

    #[test]
    fn test_validate_closed_boundary_dimension_mismatch_on_corrupted_simplex() {
        // Create a 3D TDS with a simplex that has too few vertices (corrupted state),
        // then trigger the DimensionMismatch path in validate_closed_boundary.
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        // Build a facet-to-simplices map with a synthetic boundary facet pointing at facet_index=0
        // but then corrupt the simplex to only have 2 vertices.
        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            // Replace with only 2 vertices so the facet subtraction produces wrong count.
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v1);
        }

        let mut facet_to_simplices: FacetToSimplicesMap = FacetToSimplicesMap::default();
        let mut handles: SmallBuffer<FacetHandle, 2> = SmallBuffer::new();
        handles.push(FacetHandle::new(simplex_key, 0));
        facet_to_simplices.insert(0_u64, handles);

        match validate_closed_boundary(&tds, &facet_to_simplices) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected, actual, ..
            })) => {
                assert_eq!(expected, 3, "D=3: boundary facet should have 3 vertices");
                assert!(
                    actual != 3,
                    "Corrupted simplex should produce wrong vertex count"
                );
            }
            other => panic!("Expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_manifold_error_display_variants() {
        let err = ManifoldError::ManifoldFacetMultiplicity {
            facet_key: 0xABCD,
            simplex_count: 3,
        };
        assert!(err.to_string().contains("Non-manifold facet"));

        let err = ManifoldError::BoundaryRidgeMultiplicity {
            ridge_key: 0x1234,
            boundary_facet_count: 4,
        };
        assert!(err.to_string().contains("Boundary is not closed"));

        let tds_err = TdsError::InconsistentDataStructure {
            message: "inner".to_string(),
        };
        let err = ManifoldError::from(tds_err);
        assert!(err.to_string().contains("inner"));
    }

    #[test]
    fn test_validate_vertex_links_accepts_cone_on_sphere_in_3d() {
        // Cone on the boundary of a tetrahedron (S^2).
        // The apex link is S^2, so this IS a valid PL 3-manifold (a 3-ball).

        let mut tds: Tds<(), (), 3> = Tds::empty();

        // Base tetrahedron vertices (triangulated S^2 boundary)
        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            )
            .unwrap();

        // Apex of the cone
        let apex = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.3, 0.3, 0.3]).unwrap(),
            )
            .unwrap();

        // Each boundary triangle of the tetrahedron, coned to apex
        let sphere_faces = vec![
            vec![v0, v1, v2],
            vec![v0, v1, v3],
            vec![v0, v2, v3],
            vec![v1, v2, v3],
        ];

        for tri in sphere_faces {
            let mut verts = tri;
            verts.push(apex);
            tds.insert_simplex_with_mapping(Simplex::try_new_with_data(verts, None).unwrap())
                .unwrap();
        }

        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        // Sanity: pseudomanifold + ridge links pass
        validate_facet_degree(&facet_to_simplices).unwrap();
        validate_closed_boundary(&tds, &facet_to_simplices).unwrap();
        validate_ridge_links(&tds).unwrap();

        // Vertex-link validation should ACCEPT this complex
        validate_vertex_links(&tds, &facet_to_simplices).unwrap();
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_identifies_translated_periodic_images() {
        // Two 2D simplices share bare vertex keys but differ in periodic offsets.
        // The ridge map identifies globally translated quotient ridges.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
            )
            .unwrap();

        // c1: all vertices at base image [0,0].
        let mut simplex1 = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        simplex1
            .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
            .unwrap();
        let c1 = tds.insert_simplex_with_mapping(simplex1).unwrap();

        // c2: v0 at periodic image [1,0]; v1 and v2 at base image.
        let mut simplex2 = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        simplex2
            .set_periodic_vertex_offsets(vec![[1, 0], [0, 0], [0, 0]])
            .unwrap();
        let c2 = tds.insert_simplex_with_mapping(simplex2).unwrap();

        let map = build_ridge_star_map_for_simplices(&tds, &[c1, c2]).unwrap();

        // In 2D, ridges have D-1 = 1 vertex. Single-vertex ridges are identified
        // modulo global periodic translation, so v0@base and v0@[1,0] represent
        // the same quotient ridge.
        assert_eq!(map.len(), 3, "expected 3 quotient-aware ridges");

        // All quotient ridges should have a 2-simplex star (both c1 and c2).
        let shared_count = map.values().filter(|s| s.star_simplices.len() == 2).count();
        assert_eq!(shared_count, 3, "three ridges should be shared");
    }

    #[test]
    fn test_anchored_lifted_simplex_key_preserves_vertex_link_offsets() {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            )
            .unwrap();

        let first_link_triangle: LiftedVertexBuffer = [
            lifted_vertex_id(v0, &[1_i16, 0, 0]),
            lifted_vertex_id(v1, &[1_i16, 0, 0]),
            lifted_vertex_id(v2, &[1_i16, 0, 0]),
        ]
        .into_iter()
        .collect();
        let shifted_link_triangle: LiftedVertexBuffer = [
            lifted_vertex_id(v0, &[2_i16, 0, 0]),
            lifted_vertex_id(v1, &[2_i16, 0, 0]),
            lifted_vertex_id(v2, &[2_i16, 0, 0]),
        ]
        .into_iter()
        .collect();

        assert_eq!(
            periodic_simplex_key(&first_link_triangle),
            periodic_simplex_key(&shifted_link_triangle),
            "quotient simplex keys intentionally identify global translations"
        );
        assert_ne!(
            anchored_lifted_simplex_key(&first_link_triangle),
            anchored_lifted_simplex_key(&shifted_link_triangle),
            "vertex-link keys must preserve offsets relative to the linked vertex"
        );
    }

    #[test]
    fn test_periodic_aware_ridge_star_empty_star_returns_error() {
        // Call periodic_aware_ridge_star with lifted vertices that don't match
        // any simplex's offsets, forcing an empty star after filtering.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.simplex_mut(c1)
            .unwrap()
            .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
            .unwrap();

        // Bare [v0] finds c1, but lifted_vertex_id(v0, [99,99]) won't match
        // c1's lifted v0 (which is bare v0 since offset is [0,0]).
        let synthetic = lifted_vertex_id(v0, &[99_i16, 99_i16]);
        let bare: VertexKeyBuffer = std::iter::once(v0).collect();
        let lifted: LiftedVertexBuffer = std::iter::once(synthetic).collect();

        match periodic_aware_ridge_star(&tds, 0x42, &lifted, &bare) {
            Err(ManifoldError::Tds(TdsError::InconsistentDataStructure { ref message })) => {
                assert!(
                    message.contains("empty star"),
                    "error should mention empty star: {message}"
                );
            }
            other => panic!("Expected InconsistentDataStructure (empty star), got {other:?}"),
        }
    }

    #[test]
    fn test_validate_ridge_links_for_simplices_rejects_split_periodic_link() {
        // These two lifted triangles share quotient ridge vertices but leave a split
        // periodic link, so quotient-aware ridge-link validation must reject them.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 1.0]).unwrap(),
            )
            .unwrap();

        let mut simplex1 = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        simplex1
            .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
            .unwrap();
        let c1 = tds.insert_simplex_with_mapping(simplex1).unwrap();

        let mut simplex2 = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        simplex2
            .set_periodic_vertex_offsets(vec![[1, 0], [0, 0], [0, 0]])
            .unwrap();
        let c2 = tds.insert_simplex_with_mapping(simplex2).unwrap();

        match validate_ridge_links_for_simplices(&tds, &[c1, c2]) {
            Err(ManifoldError::RidgeLinkNotManifold { .. }) => {}
            other => panic!("Expected RidgeLinkNotManifold, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_vertex_links_boundary_vertex_has_ball_link_in_3d() {
        // A single tetrahedron is a 3-ball.
        // Each boundary vertex has a link homeomorphic to a 2-ball.

        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        // All vertices are boundary vertices in a single tetrahedron
        validate_facet_degree(&facet_to_simplices).unwrap();
        validate_closed_boundary(&tds, &facet_to_simplices).unwrap();

        // Vertex-link validation must succeed (links are 2-balls)
        validate_vertex_links(&tds, &facet_to_simplices).unwrap();
    }
}

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
//! - [`Tds::build_facet_to_simplices_index`](crate::tds::Tds::build_facet_to_simplices_index)
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
        FacetToSimplicesMap, FastHashMap, FastHashSet, SmallBuffer, VertexKeyBuffer,
        VertexSecondaryMap, fast_hash_map_with_capacity, fast_hash_set_with_capacity,
    },
    facet::{
        FacetHandle, FacetIncidenceView, FacetToSimplicesIndex, OneSidedFacetAdjacency,
        classify_one_sided_facet_adjacency, facet_key_from_vertices,
    },
    tds::{SimplexKey, Tds, TdsError, VertexKey},
};
use crate::topology::characteristics::euler::{
    triangulated_surface_boundary_component_count, triangulated_surface_euler_characteristic,
};
use crate::topology::ridge::{
    build_ridge_star_map, build_ridge_star_map_for_simplices, ridge_link_edges_from_star,
    simplex_star_simplices,
};
use crate::topology::spaces::toroidal::{
    LiftedVertexBuffer, LiftedVertexId, LinkSimplexBuffer, anchored_lifted_simplex_key,
    lifted_vertex_id, ordered_lifted_edge, periodic_simplex_key,
};
use crate::topology::traits::topological_space::{GlobalTopology, TopologyKind};
#[cfg(debug_assertions)]
use std::env;
use thiserror::Error;

// =============================================================================
// Manifold validation errors
// =============================================================================

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

    /// A live ridge candidate does not occur in any D-simplex.
    #[error("Ridge candidate {ridge_vertices:?} is not present in the TDS")]
    RidgeNotFound {
        /// Canonical quotient-space ridge vertices that had an empty simplex star.
        ridge_vertices: VertexKeyBuffer,
    },

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

    /// A topology declared as closed contains a raw open one-sided facet.
    #[error(
        "Closed {topology:?} topology contains open boundary facet {facet_key:016x} at simplex {simplex_uuid}[{facet_index}]"
    )]
    BoundaryFacetInClosedTopology {
        /// Declared global topology kind.
        topology: TopologyKind,
        /// Canonical facet key with open one-sided incidence.
        facet_key: u64,
        /// Simplex containing the open facet.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the open facet.
        simplex_uuid: uuid::Uuid,
        /// Facet index in the simplex.
        facet_index: usize,
    },

    /// A non-periodic topology contains a periodic self-identification facet.
    #[error(
        "{topology:?} topology contains periodic self-identified facet {facet_key:016x} at simplex {simplex_uuid}[{facet_index}]"
    )]
    PeriodicIdentificationInNonPeriodicTopology {
        /// Declared global topology kind.
        topology: TopologyKind,
        /// Canonical facet key with periodic self-identification.
        facet_key: u64,
        /// Simplex containing the periodic self-identification.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the periodic self-identification.
        simplex_uuid: uuid::Uuid,
        /// Facet index in the simplex.
        facet_index: usize,
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

/// Validates raw facet multiplicities for crate-internal validation caches.
pub(crate) fn validate_facet_degree_map(
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

/// Topology-aware boundary classification for one canonical facet key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum BoundaryFacetClassification {
    /// The facet is a true manifold boundary facet.
    Boundary(FacetHandle),
    /// The facet is shared by two D-simplices.
    Interior,
    /// The facet is a closed periodic self-identification, not boundary.
    ClosedIdentification,
}

/// Classifies parsed facet incidence under the declared global topology.
///
/// This is the semantic boundary classifier: the TDS supplies incidence, while
/// the triangulation/topology layer decides whether one-sided incidence is
/// boundary, a closed periodic identification, or an invalid open facet in a
/// closed space.
///
/// # Errors
///
/// Returns [`ManifoldError::Tds`] if the facet handle references corrupt TDS
/// state, [`ManifoldError::BoundaryFacetInClosedTopology`] when an open
/// one-sided facet appears in closed topology, or
/// [`ManifoldError::PeriodicIdentificationInNonPeriodicTopology`] when a
/// periodic self-identification appears in non-periodic topology metadata.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::{
///     BoundaryFacetClassification, classify_boundary_facet,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)]
/// #     Manifold(#[from] delaunay::prelude::topology::validation::ManifoldError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.5, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let facet_index = dt.tds().build_facet_to_simplices_index()?;
///
/// let Some(incidence) = facet_index.iter().find(|incidence| incidence.is_one_sided())
/// else {
///     return Ok(());
/// };
///
/// let classification = classify_boundary_facet(incidence, dt.global_topology())?;
/// std::assert_matches!(
///     classification,
///     BoundaryFacetClassification::Boundary(_)
/// );
/// # Ok(())
/// # }
/// ```
pub fn classify_boundary_facet<U, V, const D: usize>(
    incidence: FacetIncidenceView<'_, '_, U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<BoundaryFacetClassification, ManifoldError> {
    let Some(handle) = incidence.one_sided_handle() else {
        return Ok(BoundaryFacetClassification::Interior);
    };

    classify_boundary_facet_handle(
        incidence.tds(),
        global_topology,
        incidence.facet_key(),
        handle,
    )
}

/// Applies topology-specific semantics to a parsed one-sided facet handle.
///
/// This helper keeps the public [`classify_boundary_facet`] contract aligned
/// with validation paths that still operate on raw facet maps for performance.
fn classify_boundary_facet_handle<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
    facet_key: u64,
    handle: FacetHandle,
) -> Result<BoundaryFacetClassification, ManifoldError> {
    let simplex_key = handle.simplex_key();
    let facet_index = usize::from(handle.facet_index());
    let simplex = tds
        .simplex(simplex_key)
        .ok_or_else(|| TdsError::SimplexNotFound {
            simplex_key,
            context: "topology-aware boundary classification".to_string(),
        })?;
    let simplex_uuid = simplex.uuid();

    match classify_one_sided_facet_adjacency(tds, facet_key, handle)? {
        OneSidedFacetAdjacency::Open if global_topology.allows_boundary() => {
            Ok(BoundaryFacetClassification::Boundary(handle))
        }
        OneSidedFacetAdjacency::Open => Err(ManifoldError::BoundaryFacetInClosedTopology {
            topology: global_topology.kind(),
            facet_key,
            simplex_key,
            simplex_uuid,
            facet_index,
        }),
        OneSidedFacetAdjacency::PeriodicSelfIdentification if global_topology.is_periodic() => {
            Ok(BoundaryFacetClassification::ClosedIdentification)
        }
        OneSidedFacetAdjacency::PeriodicSelfIdentification => {
            Err(ManifoldError::PeriodicIdentificationInNonPeriodicTopology {
                topology: global_topology.kind(),
                facet_key,
                simplex_key,
                simplex_uuid,
                facet_index,
            })
        }
    }
}

/// Builds the canonical set of true boundary facet keys for a triangulation.
///
/// # Errors
///
/// Returns [`ManifoldError::Tds`] if facet handles reference corrupt TDS state,
/// [`ManifoldError::BoundaryFacetInClosedTopology`] when the declared topology
/// is closed but an open one-sided facet is present, or
/// [`ManifoldError::PeriodicIdentificationInNonPeriodicTopology`] when a
/// periodic self-identification is observed in non-periodic topology metadata.
pub(crate) fn boundary_facet_keys_from_index<U, V, const D: usize>(
    facet_to_simplices: &FacetToSimplicesIndex<'_, U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<FastHashSet<u64>, ManifoldError> {
    let mut boundary_facet_keys: FastHashSet<u64> = FastHashSet::default();
    for incidence in facet_to_simplices.iter() {
        let facet_key = incidence.facet_key();
        if matches!(
            classify_boundary_facet(incidence, global_topology)?,
            BoundaryFacetClassification::Boundary(_)
        ) {
            boundary_facet_keys.insert(facet_key);
        }
    }
    Ok(boundary_facet_keys)
}

/// Returns whether a raw facet map contains any true boundary facet.
///
/// This is the map-reuse counterpart to [`boundary_facet_keys_from_index`].
/// Euler validation uses it so a raw one-sided facet is never mistaken for a
/// semantic boundary without first checking [`GlobalTopology`].
///
/// # Errors
///
/// Returns [`ManifoldError::ManifoldFacetMultiplicity`] if the raw map contains
/// a facet with a non-manifold incident-simplex count. It also returns the same
/// topology-classification errors as [`boundary_facet_keys_from_index`] when a
/// one-sided facet is incompatible with the declared topology.
pub(crate) fn has_boundary_facets_in_map<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesMap,
    global_topology: GlobalTopology<D>,
) -> Result<bool, ManifoldError> {
    validate_facet_degree_map(facet_to_simplices)?;

    for (facet_key, simplex_facet_pairs) in facet_to_simplices {
        let [handle] = simplex_facet_pairs.as_slice() else {
            continue;
        };
        if matches!(
            classify_boundary_facet_handle(tds, global_topology, *facet_key, *handle)?,
            BoundaryFacetClassification::Boundary(_)
        ) {
            return Ok(true);
        }
    }

    Ok(false)
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
/// use delaunay::prelude::topology::spaces::GlobalTopology;
/// use delaunay::prelude::topology::validation::validate_closed_boundary;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)] Construction(#[from] delaunay::prelude::triangulation::TriangulationConstructionError),
/// #     #[error(transparent)] Manifold(#[from] delaunay::prelude::topology::validation::ManifoldError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices)?;
/// let facet_to_simplices = tds.build_facet_to_simplices_index()?;
///
/// validate_closed_boundary(&facet_to_simplices, GlobalTopology::Euclidean)?;
/// # Ok(())
/// # }
/// ```
pub fn validate_closed_boundary<U, V, const D: usize>(
    facet_to_simplices: &FacetToSimplicesIndex<'_, U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<(), ManifoldError> {
    validate_closed_boundary_index(facet_to_simplices, global_topology)
}

fn validate_closed_boundary_index<U, V, const D: usize>(
    facet_to_simplices: &FacetToSimplicesIndex<'_, U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<(), ManifoldError> {
    let tds = facet_to_simplices.tds();
    // The boundary is a (D-1)-complex. Codimension-2 manifoldness is only meaningful for D>=2.
    if D < 2 {
        return Ok(());
    }

    // First count boundary facets so we can reserve reasonably. Periodic
    // self-neighbor facets are closed quotient identifications, not boundary.
    let mut boundary_facet_count = 0usize;
    for incidence in facet_to_simplices.iter() {
        if matches!(
            classify_boundary_facet(incidence, global_topology)?,
            BoundaryFacetClassification::Boundary(_)
        ) {
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

    for incidence in facet_to_simplices.iter() {
        let BoundaryFacetClassification::Boundary(handle) =
            classify_boundary_facet(incidence, global_topology)?
        else {
            continue;
        };

        count_boundary_facet_ridges(
            tds,
            handle,
            &mut facet_vertices,
            &mut ridge_vertices,
            &mut ridge_to_boundary_facet_count,
        )?;
    }

    validate_boundary_ridge_counts(ridge_to_boundary_facet_count)
}

/// Validates closed-boundary invariants from a raw facet map shared by validation internals.
pub(crate) fn validate_closed_boundary_map<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesMap,
    global_topology: GlobalTopology<D>,
) -> Result<(), ManifoldError> {
    // The boundary is a (D-1)-complex. Codimension-2 manifoldness is only meaningful for D>=2.
    if D < 2 {
        return Ok(());
    }

    // First count boundary facets so we can reserve reasonably. Periodic
    // self-neighbor facets are closed quotient identifications, not boundary.
    let mut boundary_facet_count = 0usize;
    for (facet_key, simplex_facet_pairs) in facet_to_simplices {
        let [handle] = simplex_facet_pairs.as_slice() else {
            continue;
        };
        if matches!(
            classify_boundary_facet_handle(tds, global_topology, *facet_key, *handle)?,
            BoundaryFacetClassification::Boundary(_)
        ) {
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

    for (facet_key, simplex_facet_pairs) in facet_to_simplices {
        // Only boundary facets (exactly one incident simplex).
        let [handle] = simplex_facet_pairs.as_slice() else {
            continue;
        };

        let BoundaryFacetClassification::Boundary(handle) =
            classify_boundary_facet_handle(tds, global_topology, *facet_key, *handle)?
        else {
            continue;
        };
        count_boundary_facet_ridges(
            tds,
            handle,
            &mut facet_vertices,
            &mut ridge_vertices,
            &mut ridge_to_boundary_facet_count,
        )?;
    }

    validate_boundary_ridge_counts(ridge_to_boundary_facet_count)
}

fn count_boundary_facet_ridges<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    handle: FacetHandle,
    facet_vertices: &mut VertexKeyBuffer,
    ridge_vertices: &mut VertexKeyBuffer,
    ridge_to_boundary_facet_count: &mut FastHashMap<u64, usize>,
) -> Result<(), ManifoldError> {
    let simplex_key = handle.simplex_key();
    let facet_index = handle.facet_index() as usize;

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

        let ridge_key = facet_key_from_vertices(ridge_vertices.as_slice());
        *ridge_to_boundary_facet_count.entry(ridge_key).or_insert(0) += 1;
    }

    Ok(())
}

fn validate_boundary_ridge_counts(
    ridge_to_boundary_facet_count: FastHashMap<u64, usize>,
) -> Result<(), ManifoldError> {
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
/// This is the local counterpart to facet-index construction plus
/// [`validate_closed_boundary`]. It expands each touched facet to its full
/// incident-simplex star, then checks only boundary ridges incident to those
/// touched facets. This keeps post-insertion checks local while preserving the
/// same codimension-1 and codimension-2 invariants for the mutated region.
pub(crate) fn validate_local_pseudomanifold_for_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
    simplices: &[SimplexKey],
) -> Result<(), ManifoldError> {
    if D == 0 || simplices.is_empty() {
        return Ok(());
    }

    let facet_to_simplices = build_local_facet_star_map(tds, simplices)?;
    validate_facet_degree_map(&facet_to_simplices)?;
    validate_closed_boundary_for_local_facets(tds, global_topology, &facet_to_simplices)
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
    if let Some(simplex_offsets) = offsets
        && simplex_offsets.len() != simplex_vertices.len()
    {
        return Err(TdsError::DimensionMismatch {
            expected: simplex_vertices.len(),
            actual: simplex_offsets.len(),
            context: format!(
                "periodic offset count for {D}D simplex {simplex_key:?} \
                 (local lifted facet vertex extraction)"
            ),
        }
        .into());
    }

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
            |simplex_offsets| {
                lifted_vertex_id(
                    vertex_key,
                    simplex_offsets[idx].iter().copied().map(i16::from),
                )
            },
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
            handles.push(FacetHandle::from_validated(simplex_key, facet_index));
        }
    }

    Ok(handles)
}

/// Validates boundary closure for boundary facets present in a local facet map.
fn validate_closed_boundary_for_local_facets<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
    facet_to_simplices: &FacetToSimplicesMap,
) -> Result<(), ManifoldError> {
    if D < 2 {
        return Ok(());
    }

    let mut checked_ridges: FastHashSet<u64> = FastHashSet::default();
    for (facet_key, simplex_facet_pairs) in facet_to_simplices {
        let [handle] = simplex_facet_pairs.as_slice() else {
            continue;
        };
        let BoundaryFacetClassification::Boundary(handle) =
            classify_boundary_facet_handle(tds, global_topology, *facet_key, *handle)?
        else {
            continue;
        };

        let (facet_vertices, facet_vertices_bare) =
            simplex_facet_vertex_ids(tds, handle.simplex_key(), handle.facet_index() as usize)?;
        for (ridge_vertices, ridge_vertices_bare) in
            ridge_vertices_for_facet::<D>(&facet_vertices, &facet_vertices_bare)?
        {
            let ridge_key = periodic_simplex_key(&ridge_vertices);
            if !checked_ridges.insert(ridge_key) {
                continue;
            }
            let boundary_facet_count = boundary_facet_count_for_ridge(
                tds,
                global_topology,
                &ridge_vertices,
                &ridge_vertices_bare,
            )?;
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
    global_topology: GlobalTopology<D>,
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
                    if matches!(
                        classify_boundary_facet_handle(
                            tds,
                            global_topology,
                            facet_key,
                            handles[0]
                        )?,
                        BoundaryFacetClassification::Boundary(_)
                    ) {
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
        return Err(TdsError::DimensionMismatch {
            expected: 1,
            actual: 0,
            context: "simplex_link_simplices_from_star requires at least one vertex".to_string(),
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
        if let Some(simplex_offsets) = offsets
            && simplex_offsets.len() != candidate_vertices.len()
        {
            return Err(TdsError::DimensionMismatch {
                expected: candidate_vertices.len(),
                actual: simplex_offsets.len(),
                context: format!(
                    "periodic offset count for {D}D simplex {simplex_key:?} \
                     (simplex link extraction)"
                ),
            }
            .into());
        }

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
                        lifted_vertex_id(vk, rel)
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
/// This is intentionally more expensive than basic codimension-1 manifold
/// validation during facet-index construction because it must inspect ridge
/// stars/links across the entire complex.
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
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
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
            if env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some() {
                let mut star_simplex_vertices: Vec<(SimplexKey, VertexKeyBuffer)> =
                    Vec::with_capacity(star.star_simplices.len());
                for &simplex_key in &star.star_simplices {
                    match tds.simplex_vertices(simplex_key) {
                        Ok(vertices) => {
                            star_simplex_vertices.push((simplex_key, vertices.into()));
                        }
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
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices)?;
/// let simplices: SimplexKeyBuffer = tds.simplices().map(|(k, _)| k).collect();
///
/// validate_ridge_links_for_simplices(&tds, simplices.iter().copied())?;
/// # Ok(())
/// # }
/// ```
pub fn validate_ridge_links_for_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: impl IntoIterator<Item = SimplexKey>,
) -> Result<(), ManifoldError> {
    // Ridge links are only meaningful for D>=2.
    if D < 2 {
        return Ok(());
    }

    // Build ridge -> star map only for ridges touching the specified simplices.
    let ridge_to_star = build_ridge_star_map_for_simplices(tds, simplices)?;

    for (ridge_key, star) in ridge_to_star {
        let link_edges =
            ridge_link_edges_from_star(tds, &star.ridge_vertices, &star.star_simplices)?;
        if let Err(err) = validate_ridge_link_graph(ridge_key, &link_edges) {
            #[cfg(debug_assertions)]
            if env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some() {
                let mut star_simplex_vertices: Vec<(SimplexKey, VertexKeyBuffer)> =
                    Vec::with_capacity(star.star_simplices.len());
                for &simplex_key in &star.star_simplices {
                    match tds.simplex_vertices(simplex_key) {
                        Ok(vertices) => {
                            star_simplex_vertices.push((simplex_key, vertices.into()));
                        }
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
/// This validator treats a vertex as a *boundary vertex* if it participates in
/// any actual boundary facet of the original complex. One-sided periodic
/// self-identifications are closed topology and do not make a vertex boundary.
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
/// use delaunay::prelude::topology::spaces::GlobalTopology;
/// use delaunay::prelude::topology::validation::{
///     validate_closed_boundary, validate_vertex_links,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)] Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)] Construction(#[from] delaunay::prelude::triangulation::TriangulationConstructionError),
/// #     #[error(transparent)] Manifold(#[from] delaunay::prelude::topology::validation::ManifoldError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices)?;
/// let facet_to_simplices = tds.build_facet_to_simplices_index()?;
///
/// validate_closed_boundary(&facet_to_simplices, GlobalTopology::Euclidean)?;
/// validate_vertex_links(&facet_to_simplices, GlobalTopology::Euclidean)?;
/// # Ok(())
/// # }
/// ```
pub fn validate_vertex_links<U, V, const D: usize>(
    facet_to_simplices: &FacetToSimplicesIndex<'_, U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<(), ManifoldError> {
    validate_vertex_links_index(facet_to_simplices, global_topology)
}

fn validate_vertex_links_index<U, V, const D: usize>(
    facet_to_simplices: &FacetToSimplicesIndex<'_, U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<(), ManifoldError> {
    let tds = facet_to_simplices.tds();
    // Vertex links are only meaningful for D>=1.
    if D < 1 {
        return Ok(());
    }

    if tds.number_of_simplices() == 0 {
        return Ok(());
    }

    let boundary_vertices =
        build_boundary_vertex_labels_from_index(facet_to_simplices, global_topology)?;

    for (vertex_key, _vertex) in tds.vertices() {
        let interior_vertex = !boundary_vertices.contains_key(vertex_key);
        validate_single_vertex_link(tds, vertex_key, interior_vertex)?;
    }

    Ok(())
}

/// Validates vertex links from a raw facet map shared by validation internals.
pub(crate) fn validate_vertex_links_map<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesMap,
    global_topology: GlobalTopology<D>,
) -> Result<(), ManifoldError> {
    // Vertex links are only meaningful for D>=1.
    if D < 1 {
        return Ok(());
    }

    if tds.number_of_simplices() == 0 {
        return Ok(());
    }

    let boundary_vertices =
        build_boundary_vertex_labels_from_map(tds, facet_to_simplices, global_topology)?;

    for (vertex_key, _vertex) in tds.vertices() {
        let interior_vertex = !boundary_vertices.contains_key(vertex_key);
        validate_single_vertex_link(tds, vertex_key, interior_vertex)?;
    }

    Ok(())
}

fn build_boundary_vertex_labels_from_map<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesMap,
    global_topology: GlobalTopology<D>,
) -> Result<VertexSecondaryMap<()>, ManifoldError> {
    // Single pass: collect all vertices that appear on a boundary facet (a facet incident to exactly 1 D-simplex).
    //
    // NOTE: We intentionally avoid a pre-count pass over `facet_to_simplices` since Level-3 validation is already
    // expensive and we only need a coarse set (it can grow dynamically).
    let mut boundary_vertices: VertexSecondaryMap<()> = VertexSecondaryMap::new();

    for (facet_key, simplex_facet_pairs) in facet_to_simplices {
        let [handle] = simplex_facet_pairs.as_slice() else {
            continue;
        };
        let BoundaryFacetClassification::Boundary(handle) =
            classify_boundary_facet_handle(tds, global_topology, *facet_key, *handle)?
        else {
            continue;
        };

        insert_boundary_facet_vertices(tds, handle, &mut boundary_vertices)?;
    }

    Ok(boundary_vertices)
}

fn build_boundary_vertex_labels_from_index<U, V, const D: usize>(
    facet_to_simplices: &FacetToSimplicesIndex<'_, U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<VertexSecondaryMap<()>, ManifoldError> {
    let tds = facet_to_simplices.tds();
    let mut boundary_vertices: VertexSecondaryMap<()> = VertexSecondaryMap::new();

    for incidence in facet_to_simplices.iter() {
        let BoundaryFacetClassification::Boundary(handle) =
            classify_boundary_facet(incidence, global_topology)?
        else {
            continue;
        };
        insert_boundary_facet_vertices(tds, handle, &mut boundary_vertices)?;
    }

    Ok(boundary_vertices)
}

fn insert_boundary_facet_vertices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    handle: FacetHandle,
    boundary_vertices: &mut VertexSecondaryMap<()>,
) -> Result<(), ManifoldError> {
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
        boundary_vertices.insert(vk, ());
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

    Ok(())
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

    let mut adjacency: FastHashMap<LiftedVertexId, LiftedVertexBuffer> =
        fast_hash_map_with_capacity(edge_counts.len().saturating_mul(2));
    for ((a, b), count) in edge_counts {
        if count != 1 {
            continue;
        }
        adjacency.entry(a.clone()).or_default().push(b.clone());
        adjacency.entry(b).or_default().push(a);
    }
    if adjacency.is_empty() {
        return 0;
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
    use std::{assert_matches, iter};

    use crate::core::facet::{FacetHandle, FacetView};
    use crate::core::simplex::Simplex;
    use crate::core::triangulation::Triangulation;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::FastKernel;
    use crate::topology::traits::topological_space::ToroidalConstructionMode;

    use slotmap::KeyData;

    fn vk(id: u64) -> VertexKey {
        VertexKey::from(KeyData::from_ffi(id))
    }

    fn simplex(vertices: &[VertexKey]) -> LiftedVertexBuffer {
        let mut s: LiftedVertexBuffer = LiftedVertexBuffer::with_capacity(vertices.len());
        s.extend(vertices.iter().copied().map(LiftedVertexId::base));
        s
    }

    fn build_single_triangle_tds(periodic_self_neighbor: bool) -> (Tds<(), (), 2>, SimplexKey) {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(Vertex::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::try_new([0.0, 1.0]).unwrap())
            .unwrap();

        let mut simplex = Simplex::try_new(vec![v0, v1, v2]).unwrap();
        if periodic_self_neighbor {
            simplex
                .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [1, 0]])
                .unwrap();
        }
        let simplex_key = tds.insert_simplex_with_mapping(simplex).unwrap();
        if periodic_self_neighbor {
            tds.simplex_mut(simplex_key)
                .unwrap()
                .set_neighbors_from_keys([Some(simplex_key), None, None])
                .unwrap();
        }

        (tds, simplex_key)
    }

    fn facet_key_for_simplex_facet(
        tds: &Tds<(), (), 2>,
        simplex_key: SimplexKey,
        facet_index: u8,
    ) -> u64 {
        let facet = FacetView::try_new(tds, simplex_key, facet_index).unwrap();
        facet.key()
    }

    fn build_closed_surface_s2_tds_2d() -> (Tds<(), (), 2>, [VertexKey; 4]) {
        // Closed 2D simplicial complex (topologically S²): boundary of a tetrahedron.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 1.0]).unwrap())
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
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let shared_edge_v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap())
            .unwrap();

        let tet1_v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let tet1_v3 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let tet2_v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, -1.0, 0.0]).unwrap())
            .unwrap();
        let tet2_v3 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, -1.0]).unwrap())
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
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        assert!(validate_facet_degree_map(&facet_to_simplices).is_ok());
    }

    #[test]
    fn test_validate_facet_degree_ok_for_two_tetrahedra_sharing_facet() {
        // Two tetrahedra share a facet => that facet has degree 2, all others degree 1.
        let mut tds: Tds<(), (), 3> = Tds::empty();

        // Shared triangle.
        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap())
            .unwrap();

        // Apex points on opposite sides.
        let v3 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, -1.0]).unwrap())
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
        assert!(validate_facet_degree_map(&facet_to_simplices).is_ok());
    }

    #[test]
    fn test_validate_facet_degree_errors_on_non_manifold_facet_multiplicity() {
        // Three tetrahedra share a single facet -> not a manifold-with-boundary.
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap())
            .unwrap();

        let v3 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 2.0]).unwrap())
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 3.0]).unwrap())
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
        match validate_facet_degree_map(&facet_to_simplices) {
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
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        assert!(
            validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean)
                .is_ok()
        );
    }

    #[test]
    fn test_validate_closed_boundary_errors_on_out_of_bounds_facet_index() {
        let vertices = vec![
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplices().next().unwrap().0;

        // Synthesize an invalid boundary facet handle: facet indices must be < D+1.
        let mut facet_to_simplices: FacetToSimplicesMap = FacetToSimplicesMap::default();
        let mut handles: SmallBuffer<FacetHandle, 2> = SmallBuffer::new();
        handles.push(FacetHandle::from_validated(simplex_key, u8::MAX));
        facet_to_simplices.insert(0_u64, handles);

        match validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean) {
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
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([2.0]).unwrap())
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
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0]).unwrap())
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
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
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
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let vb = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let vc = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 1.0]).unwrap())
            .unwrap();
        let vd = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
            .unwrap();
        let center = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.5, 0.5]).unwrap())
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
        validate_facet_degree_map(&facet_to_simplices).unwrap();
        validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean).unwrap();

        let boundary_vertices = build_boundary_vertex_labels_from_map(
            &tds,
            &facet_to_simplices,
            GlobalTopology::Euclidean,
        )
        .unwrap();
        for boundary_vertex in [va, vb, vc, vd] {
            assert!(boundary_vertices.contains_key(boundary_vertex));
        }
        assert!(!boundary_vertices.contains_key(center));

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
        validate_vertex_links_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean).unwrap();

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
            Vertex::<(), _>::try_new([0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 1>::build_initial_simplex(&vertices).unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        // Codimension-2 boundary manifoldness is only meaningful for D>=2.
        assert!(
            validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean)
                .is_ok()
        );
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

        assert!(
            validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean)
                .is_ok()
        );
    }

    #[test]
    fn test_validate_ridge_links_ok_for_closed_2d_surface() {
        let (tds, _vertices) = build_closed_surface_s2_tds_2d();

        // Sanity: pseudomanifold checks pass.
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();
        validate_facet_degree_map(&facet_to_simplices).unwrap();
        validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean).unwrap();

        assert!(validate_ridge_links(&tds).is_ok());
    }

    #[test]
    fn test_simplex_link_simplices_from_star_errors_on_empty_simplex() {
        let tds: Tds<(), (), 2> = Tds::empty();

        match simplex_link_simplices_from_star(&tds, &[], &[]) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 1,
                actual: 0,
                ..
            })) => {}
            other => panic!("Expected DimensionMismatch for empty simplex, got {other:?}"),
        }
    }

    #[test]
    fn test_simplex_link_simplices_from_star_errors_on_unrelated_vertex() {
        // Defensive: a simplex vertex that exists in the TDS but is not part of a star simplex should
        // trigger a link-size mismatch (this should not happen when star simplices are produced by
        // `simplex_star_simplices`, but is a robustness check for corrupted inputs).
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([10.0, 10.0]).unwrap())
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
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
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

        match validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean) {
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

        match validate_local_pseudomanifold_for_simplices(
            &tds,
            GlobalTopology::Euclidean,
            &[touched_simplex],
        ) {
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
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        assert_eq!(tds.remove_simplices_by_keys(&[simplex_key]).unwrap(), 1);

        match validate_local_pseudomanifold_for_simplices(
            &tds,
            GlobalTopology::Euclidean,
            &[simplex_key],
        ) {
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
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
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
        validate_facet_degree_map(&facet_to_simplices).unwrap();
        validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean).unwrap();

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

    fn build_wedge_two_spheres_share_vertex_tds_2d()
    -> (Tds<(), (), 2>, VertexKey, SimplexKey, SimplexKey) {
        // Two closed 2D spheres (boundaries of tetrahedra) that share a single vertex.
        // This is a pseudomanifold (every edge has degree 2), but not a PL 2-manifold:
        // the shared vertex has a disconnected link (two disjoint cycles).
        let mut tds: Tds<(), (), 2> = Tds::empty();

        // Shared vertex.
        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();

        // First tetrahedron boundary (4 triangles on 4 vertices).
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 1.0]).unwrap())
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
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([10.0, 10.0]).unwrap())
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([11.0, 10.0]).unwrap())
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([10.0, 11.0]).unwrap())
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
    fn test_validate_ridge_links_for_simplices_ok_for_single_tetrahedron_in_3d() {
        let vertices = vec![
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();

        let simplices: Vec<SimplexKey> = tds.simplices().map(|(k, _)| k).collect();
        validate_ridge_links_for_simplices(&tds, simplices.iter().copied()).unwrap();

        // And it should be a no-op on empty simplex lists.
        validate_ridge_links_for_simplices(&tds, iter::empty::<SimplexKey>()).unwrap();
    }

    #[test]
    fn test_validate_ridge_links_for_simplices_ok_for_missing_simplex_keys() {
        // Defensive: local validation should ignore missing simplex keys.
        let tds: Tds<(), (), 3> = Tds::empty();
        let missing = SimplexKey::from(KeyData::from_ffi(u64::MAX));

        assert!(validate_ridge_links_for_simplices(&tds, [missing]).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_for_simplices_noop_for_d_lt_2() {
        let mut tds: Tds<(), (), 1> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0]).unwrap())
            .unwrap();

        let c01 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v0, v1], None).unwrap())
            .unwrap();

        assert!(validate_ridge_links_for_simplices(&tds, [c01]).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_for_simplices_rejects_wedge_at_vertex_in_2d() {
        let (tds, v0, incident, _nonincident) = build_wedge_two_spheres_share_vertex_tds_2d();

        let expected_ridge_key = facet_key_from_vertices(&[v0]);

        match validate_ridge_links_for_simplices(&tds, [incident]) {
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
        assert!(validate_ridge_links_for_simplices(&tds, [nonincident]).is_ok());
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
                let i_f = f64::from(u32::try_from(i).unwrap());
                let j_f = f64::from(u32::try_from(j).unwrap());
                *slot = tds
                    .insert_vertex_with_mapping(Vertex::<(), _>::try_new([i_f, j_f, 0.0]).unwrap())
                    .unwrap();
            }
        }

        // Apex of the cone (interior vertex; not on any boundary facet).
        let apex = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.5, 0.5, 1.0]).unwrap())
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
        validate_facet_degree_map(&facet_to_simplices).unwrap();
        validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean).unwrap();

        // Ridge-link validation should *not* detect this singularity.
        assert!(validate_ridge_links(&tds).is_ok());

        // Vertex-link validation MUST reject it: apex link is T^2, not S^2.
        match validate_vertex_links_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean) {
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
    fn test_validate_closed_boundary_dimension_mismatch_on_corrupted_simplex() {
        // Create a 3D TDS with a simplex that has too few vertices (corrupted state),
        // then trigger the DimensionMismatch path in validate_closed_boundary.
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap())
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
        handles.push(FacetHandle::from_validated(simplex_key, 0));
        facet_to_simplices.insert(0_u64, handles);

        match validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean) {
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
    fn classify_boundary_facet_euclidean_open_one_sided_is_boundary() {
        let (tds, simplex_key) = build_single_triangle_tds(false);
        let facet_key = facet_key_for_simplex_facet(&tds, simplex_key, 0);
        let facet_to_simplices = tds.build_facet_to_simplices_index().unwrap();
        let incidence = facet_to_simplices.get(&facet_key).unwrap();

        let classification = classify_boundary_facet(incidence, GlobalTopology::Euclidean).unwrap();

        assert_matches!(
            classification,
            BoundaryFacetClassification::Boundary(handle)
                if handle.simplex_key() == simplex_key && handle.facet_index() == 0
        );
    }

    #[test]
    fn classify_boundary_facet_closed_topology_rejects_open_one_sided() {
        let (tds, simplex_key) = build_single_triangle_tds(false);
        let facet_key = facet_key_for_simplex_facet(&tds, simplex_key, 0);
        let facet_to_simplices = tds.build_facet_to_simplices_index().unwrap();
        let incidence = facet_to_simplices.get(&facet_key).unwrap();

        let err = classify_boundary_facet(incidence, GlobalTopology::Spherical).unwrap_err();

        assert_matches!(
            err,
            ManifoldError::BoundaryFacetInClosedTopology {
                topology: TopologyKind::Spherical,
                facet_key: found_facet_key,
                simplex_key: found_simplex_key,
                facet_index: 0,
                ..
            } if found_facet_key == facet_key && found_simplex_key == simplex_key
        );
    }

    #[test]
    fn classify_boundary_facet_periodic_self_identification_requires_periodic_topology() {
        let (tds, simplex_key) = build_single_triangle_tds(true);
        let facet_key = facet_key_for_simplex_facet(&tds, simplex_key, 0);
        let facet_to_simplices = tds.build_facet_to_simplices_index().unwrap();
        let incidence = facet_to_simplices.get(&facet_key).unwrap();
        let periodic_topology =
            GlobalTopology::try_toroidal([1.0, 1.0], ToroidalConstructionMode::PeriodicImagePoint)
                .unwrap();

        let classification = classify_boundary_facet(incidence, periodic_topology).unwrap();
        assert_eq!(
            classification,
            BoundaryFacetClassification::ClosedIdentification
        );

        let err = classify_boundary_facet(incidence, GlobalTopology::Euclidean).unwrap_err();
        assert_matches!(
            err,
            ManifoldError::PeriodicIdentificationInNonPeriodicTopology {
                topology: TopologyKind::Euclidean,
                facet_key: found_facet_key,
                simplex_key: found_simplex_key,
                facet_index: 0,
                ..
            } if found_facet_key == facet_key && found_simplex_key == simplex_key
        );
    }

    #[test]
    fn test_validate_vertex_links_accepts_cone_on_sphere_in_3d() {
        // Cone on the boundary of a tetrahedron (S^2).
        // The apex link is S^2, so this IS a valid PL 3-manifold (a 3-ball).

        let mut tds: Tds<(), (), 3> = Tds::empty();

        // Base tetrahedron vertices (triangulated S^2 boundary)
        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap())
            .unwrap();

        // Apex of the cone
        let apex = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.3, 0.3, 0.3]).unwrap())
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
        validate_facet_degree_map(&facet_to_simplices).unwrap();
        validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean).unwrap();
        validate_ridge_links(&tds).unwrap();

        // Vertex-link validation should ACCEPT this complex
        validate_vertex_links_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean).unwrap();
    }

    #[test]
    fn test_validate_ridge_links_for_simplices_rejects_split_periodic_link() {
        // These two lifted triangles share quotient ridge vertices but leave a split
        // periodic link, so quotient-aware ridge-link validation must reject them.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.5, 1.0]).unwrap())
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

        match validate_ridge_links_for_simplices(&tds, [c1, c2]) {
            Err(ManifoldError::RidgeLinkNotManifold { .. }) => {}
            other => panic!("Expected RidgeLinkNotManifold, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_vertex_links_boundary_vertex_has_ball_link_in_3d() {
        // A single tetrahedron is a 3-ball.
        // Each boundary vertex has a link homeomorphic to a 2-ball.

        let vertices = vec![
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let facet_to_simplices = tds.build_facet_to_simplices_map().unwrap();

        // All vertices are boundary vertices in a single tetrahedron
        validate_facet_degree_map(&facet_to_simplices).unwrap();
        validate_closed_boundary_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean).unwrap();

        // Vertex-link validation must succeed (links are 2-balls)
        validate_vertex_links_map(&tds, &facet_to_simplices, GlobalTopology::Euclidean).unwrap();
    }
}

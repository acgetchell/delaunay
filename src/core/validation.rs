//! Generic validation orchestration for [`Triangulation`](crate::Triangulation).
//!
//! This module owns the generic validation vocabulary and the cumulative
//! triangulation-level validation pipeline:
//!
//! - **Level 1** element validity remains implemented next to the element types:
//!   [`Vertex::is_valid`](crate::prelude::Vertex::is_valid) /
//!   [`Vertex::vertex_report`](crate::prelude::Vertex::vertex_report) and
//!   [`Simplex::is_valid`](crate::prelude::tds::Simplex::is_valid) /
//!   [`Simplex::simplex_report`](crate::prelude::tds::Simplex::simplex_report).
//! - **Level 2** structural validation remains implemented by
//!   [`Tds`](crate::prelude::tds::Tds).
//! - **Level 3** topological validation is orchestrated here for
//!   [`Triangulation`](crate::Triangulation).
//! - **Level 4** faithful embedded-geometry validation is implemented by
//!   [`Triangulation::validate_embedding`](crate::Triangulation::validate_embedding).
//!
//! Delaunay-specific Level 5 validation lives in [`crate::validation`]. Keeping
//! the module boundary at the generic triangulation layer avoids one file per
//! validation level while still making the layering explicit.
//!
//! # Validation Hierarchy
//!
//! The library provides **five levels** of validation, each building on the previous:
//!
//! ## Level 1: Element Validity
//!
//! - **Methods**:
//!   [`Simplex::is_valid()`](crate::prelude::tds::Simplex::is_valid),
//!   [`Simplex::simplex_report()`](crate::prelude::tds::Simplex::simplex_report),
//!   [`Vertex::is_valid()`](crate::prelude::Vertex::is_valid),
//!   [`Vertex::vertex_report()`](crate::prelude::Vertex::vertex_report)
//! - **Checks**: Basic data integrity (coordinate validity, UUID presence, proper initialization)
//! - **Cost**: O(1) per element
//!
//! ## Level 2: TDS Structural Validity
//!
//! - **Method**: [`Tds::is_valid()`](crate::prelude::tds::Tds::is_valid)
//! - **Checks**:
//!   - UUID ↔ Key mapping consistency
//!   - No duplicate simplices (same vertex sets)
//!   - Facet sharing invariant (≤2 simplices per facet)
//!   - Neighbor consistency (mutual relationships)
//! - **Cost**: O(N×D²) where N = simplices, D = dimension
//!
//! Use [`Tds::validate()`](crate::prelude::tds::Tds::validate) for cumulative
//! Levels 1–2 (element + structural) validation.
//!
//! ## Level 3: Manifold Topology
//!
//! - **Method**: [`Triangulation::is_valid_topology()`](crate::prelude::triangulation::Triangulation::is_valid_topology)
//! - **Checks**:
//!   - **Codimension-1 incidence**: each facet is one-sided or two-sided
//!   - **Topology-aware boundary manifoldness**: true boundary facets are closed ("no boundary of boundary")
//!   - Connectedness (single connected component in the simplex neighbor graph)
//!   - No isolated vertices (every vertex must be incident to at least one simplex)
//!   - Euler characteristic (χ = V - E + F - C matches expected topology)
//! - **Cost**: O(N×D²) dominated by simplex counting
//!
//! Use [`Triangulation::validate()`](crate::prelude::triangulation::Triangulation::validate)
//! for cumulative Levels 1–3.
//!
//! ## Level 4: Faithful Embedding
//!
//! - **Method**: [`Triangulation::validate_embedding`](crate::prelude::triangulation::Triangulation::validate_embedding)
//! - **Checks**: Nondegenerate maximal simplices and no intersections outside shared faces
//! - **Cost**: O(N²) worst case, dominated by pairwise simplex-intersection checks
//!
//! Use [`Triangulation::validate_embedding`](crate::prelude::triangulation::Triangulation::validate_embedding)
//! for cumulative Levels 1–4.
//!
//! ## Level 5: Delaunay Property
//!
//! - **Method**: [`DelaunayTriangulation::is_valid_delaunay()`](crate::DelaunayTriangulation::is_valid_delaunay)
//! - **Checks**: Empty circumsphere property (no vertex inside any simplex's circumsphere)
//! - **Cost**: O(N×V) where N = simplices, V = vertices
//!
//! Use [`DelaunayTriangulation::validate()`](crate::DelaunayTriangulation::validate)
//! for cumulative Levels 1–5.
//!
//! ## Topology guarantees
//!
//! [`TopologyGuarantee`](crate::prelude::validation::TopologyGuarantee) selects
//! which **manifoldness** invariants are checked by Level 3 topology validation.
//! Whether those checks run automatically after insertion is controlled by
//! [`ValidationPolicy`](crate::prelude::validation::ValidationPolicy).
//!
//! Level 3 validation always checks:
//! - Codimension-1 facet degree (pseudomanifold condition: one-sided or two-sided incidence)
//! - Topology-aware boundary manifoldness (closed true boundary: "no boundary of boundary")
//! - Connectedness (single connected component in the simplex neighbor graph)
//! - No isolated vertices (every vertex must be incident to at least one simplex)
//! - Euler characteristic
//!
//! With
//! [`TopologyGuarantee::PLManifold`](crate::prelude::validation::TopologyGuarantee::PLManifold),
//! Level 3 validation additionally checks the canonical **vertex-link** PL-manifoldness condition via
//! [`crate::topology::manifold::validate_vertex_links`].
//!
//! Note: for **D=3**, the current vertex-link validator additionally enforces that each link
//! has the Euler characteristic / boundary component counts of a sphere/ball (S²/B²).
//! For **D≥4**, it currently checks that each vertex link is a connected (D−1)-manifold
//! with the correct boundary behavior (a necessary condition), but does not attempt to
//! distinguish spheres/balls from other manifolds (not sufficient in general).

use crate::core::algorithms::incremental_insertion::{
    InsertionError, InsertionTopologyValidationContext,
};
use crate::core::collections::{
    FacetToSimplicesMap, FastHashSet, SimplexKeyBuffer, SimplexKeySet, VertexKeyBuffer,
    fast_hash_set_with_capacity,
};
use crate::core::operations::{InsertionTelemetry, InsertionTelemetryMode, SuspicionFlags};
use crate::core::tds::{
    InvariantError, InvariantKind, InvariantViolation, SimplexKey, TdsError,
    TriangulationValidationReport, VertexKey,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::geometry::kernel::Kernel;
use crate::topology::characteristics::euler::TopologyClassification;
use crate::topology::characteristics::validation::validate_triangulation_euler_from_validated_facet_map;
use crate::topology::manifold::{
    ManifoldError, ValidatedFacetDegreeMap, validate_closed_boundary_from_validated_facet_map,
    validate_local_pseudomanifold_for_simplices, validate_ridge_links,
    validate_ridge_links_for_simplices, validate_vertex_links_from_validated_facet_map,
};
use crate::topology::traits::topological_space::{GlobalTopology, TopologyError, TopologyKind};
use std::time::{Duration, Instant};
use thiserror::Error;
use uuid::Uuid;

/// Convert an [`InsertionError`] into the appropriate [`InvariantError`], preserving
/// structured error information across all layers.
///
/// - `TopologyValidation(source)` → `InvariantError::Tds(source)` (Level 1–2 preserved)
/// - `TopologyValidationFailed { source }` → `InvariantError::Triangulation(source)` (Level 3 preserved)
/// - `EmbeddingValidationFailed { source }` → `InvariantError::Embedding(source)` (Level 4 preserved)
/// - `DelaunayValidationFailed { source }` → `InvariantError::Delaunay(source)` (Level 5 preserved)
/// - All other variants → `InvariantError::Tds(InconsistentDataStructure { .. })` with `context`
pub(crate) fn insertion_error_to_invariant_error(
    error: InsertionError,
    context: &str,
) -> InvariantError {
    match error {
        InsertionError::TopologyValidation(source) => InvariantError::Tds(source),
        InsertionError::TopologyValidationFailed { source, .. } => {
            InvariantError::Triangulation(source)
        }
        InsertionError::EmbeddingValidationFailed { source } => InvariantError::Embedding(source),
        InsertionError::DelaunayValidationFailed { source } => InvariantError::Delaunay(source),
        other => InvariantError::Tds(TdsError::InconsistentDataStructure {
            message: format!("{context}: {other}"),
        }),
    }
}

/// Errors that can occur during triangulation topology validation (Level 3).
///
/// This type represents **only** Level 3 (topology) errors. It does not contain
/// TDS-level (Levels 1–2) errors. Cumulative validators that can return errors
/// from any level use [`InvariantError`] instead.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::InvariantError;
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Source(#[from] DelaunayTriangulationConstructionError),
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
/// let dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let result: Result<(), InvariantError> = dt.as_triangulation().validate();
/// assert!(result.is_ok());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TriangulationValidationError {
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

    /// Boundary is not a closed (D-1)-manifold:
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

    /// A closed global topology contains a raw open one-sided facet.
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
        simplex_uuid: Uuid,
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
        simplex_uuid: Uuid,
        /// Facet index in the simplex.
        facet_index: usize,
    },

    /// A live ridge candidate did not occur in any D-simplex.
    #[error("Ridge candidate {ridge_vertices:?} is not present in the triangulation")]
    RidgeNotFound {
        /// Canonical quotient-space ridge vertices that had an empty simplex star.
        ridge_vertices: VertexKeyBuffer,
    },

    /// A ridge's link graph is not a 1-manifold (path or cycle).
    ///
    /// This is required for PL-manifold validation.
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

    /// Euler characteristic does not match the expected value for the classified topology.
    #[error(
        "Euler characteristic mismatch: computed χ={computed}, expected χ={expected} for {classification:?}"
    )]
    EulerCharacteristicMismatch {
        /// Computed Euler characteristic.
        computed: isize,
        /// Expected Euler characteristic for the classification.
        expected: isize,
        /// The topology classification used to determine expectation.
        classification: TopologyClassification,
    },

    /// Positive-orientation promotion did not converge within its bounded pass budget.
    ///
    /// This is reported at the triangulation layer because the TDS may still be
    /// structurally coherent while failing the geometric simplex-orientation
    /// invariant expected by triangulation validation.
    #[error(
        "Positive-orientation promotion did not converge: {residual_count} residual negative-orientation simplices, sampled keys {sampled:?}"
    )]
    OrientationPromotionNonConvergence {
        /// Number of simplices still evaluated as negative after promotion.
        residual_count: usize,
        /// Sample of residual simplex keys for diagnostics.
        sampled: Vec<SimplexKey>,
    },

    /// Vertex is not incident to any simplex.
    ///
    /// An isolated vertex violates manifold invariants at the topology (Level 3) layer
    /// and may indicate a failed insertion or an insertion that was partially rolled back.
    #[error(
        "Isolated vertex: vertex {vertex_uuid} (key {vertex_key:?}) is not incident to any simplex"
    )]
    IsolatedVertex {
        /// Key of the isolated vertex.
        vertex_key: VertexKey,
        /// UUID of the isolated vertex.
        vertex_uuid: Uuid,
    },

    /// The simplex neighbor graph is not a single connected component.
    ///
    /// A valid triangulation-with-boundary must be connected; multiple disconnected
    /// components indicate a structural problem (e.g. simplices that share only a vertex
    /// or edge but no facet, so no neighbor pointers link them).
    #[error(
        "Disconnected triangulation: simplex neighbor graph is not a single connected component ({simplex_count} simplices total)"
    )]
    Disconnected {
        /// Total number of simplices in the triangulation.
        simplex_count: usize,
    },
}

impl TryFrom<ManifoldError> for TriangulationValidationError {
    type Error = TdsError;

    fn try_from(err: ManifoldError) -> Result<Self, Self::Error> {
        match err {
            ManifoldError::Tds(source) => Err(source),
            ManifoldError::ManifoldFacetMultiplicity {
                facet_key,
                simplex_count,
            } => Ok(Self::ManifoldFacetMultiplicity {
                facet_key,
                simplex_count,
            }),
            ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            } => Ok(Self::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            }),
            ManifoldError::BoundaryFacetInClosedTopology {
                topology,
                facet_key,
                simplex_key,
                simplex_uuid,
                facet_index,
            } => Ok(Self::BoundaryFacetInClosedTopology {
                topology,
                facet_key,
                simplex_key,
                simplex_uuid,
                facet_index,
            }),
            ManifoldError::PeriodicIdentificationInNonPeriodicTopology {
                topology,
                facet_key,
                simplex_key,
                simplex_uuid,
                facet_index,
            } => Ok(Self::PeriodicIdentificationInNonPeriodicTopology {
                topology,
                facet_key,
                simplex_key,
                simplex_uuid,
                facet_index,
            }),
            ManifoldError::RidgeNotFound { ridge_vertices } => {
                Ok(Self::RidgeNotFound { ridge_vertices })
            }
            ManifoldError::RidgeLinkNotManifold {
                ridge_key,
                link_vertex_count,
                link_edge_count,
                max_degree,
                degree_one_vertices,
                connected,
            } => Ok(Self::RidgeLinkNotManifold {
                ridge_key,
                link_vertex_count,
                link_edge_count,
                max_degree,
                degree_one_vertices,
                connected,
            }),
            ManifoldError::VertexLinkNotManifold {
                vertex_key,
                link_vertex_count,
                link_simplex_count,
                boundary_facet_count,
                max_degree,
                connected,
                interior_vertex,
            } => Ok(Self::VertexLinkNotManifold {
                vertex_key,
                link_vertex_count,
                link_simplex_count,
                boundary_facet_count,
                max_degree,
                connected,
                interior_vertex,
            }),
        }
    }
}

impl From<ManifoldError> for InvariantError {
    fn from(err: ManifoldError) -> Self {
        match TriangulationValidationError::try_from(err) {
            Ok(source) => Self::Triangulation(source),
            Err(source) => Self::Tds(source),
        }
    }
}

/// Preserves validation-layer error provenance from topology helper failures.
///
/// [`Triangulation::validate`] exposes Level 1-2 failures as [`InvariantError::Tds`]
/// and Level 3 failures as [`InvariantError::Triangulation`]. Euler helpers
/// return [`TopologyError`], so this adapter maps support-data failures back to
/// TDS errors and semantic boundary failures back through [`ManifoldError`].
fn invariant_error_from_topology_error(err: TopologyError) -> InvariantError {
    match err {
        TopologyError::FacetMapBuild { source }
        | TopologyError::BoundaryFacetEnumeration { source }
        | TopologyError::BoundaryFacetCount { source } => InvariantError::Tds(source),
        TopologyError::BoundaryFacetSimplexAccess { source } => InvariantError::Tds(source.into()),
        TopologyError::BoundaryClassification { source } => InvariantError::from(*source),
    }
}

/// Policy controlling when the triangulation runs global validation passes.
///
/// Validation can be expensive (O(N×D²) or worse), so this allows callers to trade
/// performance for stricter correctness checks during incremental operations.
///
/// **Note**: [`TopologyGuarantee::PLManifold`] is incompatible with [`ValidationPolicy::Never`].
/// `PLManifold` requires at least caller-owned completion validation to certify full
/// PL-manifoldness. Use [`ValidationPolicy::ExplicitOnly`] when callers own explicit validation
/// checkpoints, [`ValidationPolicy::OnSuspicion`] for suspicion-triggered validation, or
/// [`ValidationPolicy::Always`] for maximum safety during incremental operations.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::operations::SuspicionFlags;
/// use delaunay::prelude::ValidationPolicy;
///
/// let policy = ValidationPolicy::OnSuspicion;
/// let suspicion = SuspicionFlags { perturbation_used: true, ..SuspicionFlags::default() };
/// assert!(policy.should_validate(suspicion));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationPolicy {
    /// Never run full global validation automatically.
    ///
    /// This is only compatible with [`TopologyGuarantee::Pseudomanifold`]. Use
    /// [`ValidationPolicy::ExplicitOnly`] when a stronger topology guarantee is active and
    /// full validation checkpoints are owned by the caller.
    Never,

    /// Do not run policy-triggered global topology/changed-scope embedding validation during insertion.
    ///
    /// Mandatory local topology checks required by the active [`TopologyGuarantee`] still run
    /// during insertion, but suspicion-triggered global Level 3/changed-scope Level 4 validation is disabled.
    ExplicitOnly,

    /// Validate only if the operation is suspicious (e.g. degeneracy).
    OnSuspicion,

    /// Always validate after insertion.
    Always,

    /// Debug builds: always validate; release builds: [`ValidationPolicy::OnSuspicion`].
    DebugOnly,
}

impl ValidationPolicy {
    /// Returns `true` if a global validation pass should be run given the observed
    /// [`crate::prelude::operations::SuspicionFlags`].
    #[inline]
    #[must_use]
    pub const fn should_validate(&self, suspicion: SuspicionFlags) -> bool {
        match self {
            Self::Never | Self::ExplicitOnly => false,
            Self::Always => true,
            Self::OnSuspicion => suspicion.is_suspicious(),
            Self::DebugOnly => cfg!(debug_assertions) || suspicion.is_suspicious(),
        }
    }
}

impl Default for ValidationPolicy {
    #[inline]
    fn default() -> Self {
        Self::OnSuspicion
    }
}

/// Selects which topological invariants are checked by Level 3 validation.
///
/// This enum specifies *what is checked* about the underlying simplicial complex when
/// Level 3 validation runs. Whether Level 3 validation runs automatically after insertion
/// is controlled by [`ValidationPolicy`].
///
/// - [`TopologyGuarantee::Pseudomanifold`] checks the codimension-1 adjacency condition:
///   each facet is incident to one or two simplices, and the codimension-2 boundary is closed.
///   This is sufficient for many geometric algorithms but does not guarantee local Euclidean structure.
///
/// - [`TopologyGuarantee::PLManifold`] uses ridge-link validation during insertion and
///   requires a vertex-link validation pass at construction completion to certify
///   PL-manifoldness.
/// - [`TopologyGuarantee::PLManifoldStrict`] runs vertex-link validation after every
///   insertion for maximal safety (slowest).
///
/// # Example
///
/// ```rust
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Source(#[from] DelaunayTriangulationConstructionError),
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
/// let mut dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
///
/// // Optional: relax topology checks for speed (weaker guarantees).
/// dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
/// assert!(!dt.topology_guarantee().requires_vertex_links_at_completion());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyGuarantee {
    /// Validate only the pseudomanifold / manifold-with-boundary invariants:
    /// - facet degree (1 or 2 incident simplices per facet)
    /// - closed boundary ("no boundary of boundary")
    Pseudomanifold,

    /// Validate PL-manifold invariants (incremental mode).
    ///
    /// This includes all `Pseudomanifold` checks plus ridge-link validation during
    /// insertion, with a required vertex-link validation at construction completion.
    PLManifold,

    /// Validate PL-manifold invariants with strict per-insertion checks.
    ///
    /// This includes all `Pseudomanifold` checks plus vertex-link validation
    /// after every insertion (slowest, maximum safety).
    PLManifoldStrict,
}

impl Default for TopologyGuarantee {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TopologyGuarantee {
    /// The default topology guarantee used when constructing triangulations.
    ///
    /// This is a `const` alternative to `<Self as Default>::default()` for `const fn` constructors.
    pub const DEFAULT: Self = Self::PLManifold;

    /// Returns `true` if this topology guarantee requires vertex-link validation
    /// after each insertion.
    #[inline]
    #[must_use]
    pub const fn requires_vertex_links_during_insertion(self) -> bool {
        matches!(self, Self::PLManifoldStrict)
    }

    /// Returns `true` if this topology guarantee requires vertex-link validation
    /// at construction completion.
    #[inline]
    #[must_use]
    pub const fn requires_vertex_links_at_completion(self) -> bool {
        matches!(self, Self::PLManifold | Self::PLManifoldStrict)
    }

    /// Returns `true` if this topology guarantee requires pseudomanifold checks
    /// during insertion.
    ///
    /// All current guarantees require the codimension-1 facet-degree and
    /// codimension-2 closed-boundary conditions. Stronger guarantees layer
    /// ridge-link and vertex-link validation on top of these checks.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::TopologyGuarantee;
    ///
    /// assert!(
    ///     TopologyGuarantee::Pseudomanifold
    ///         .requires_pseudomanifold_checks_during_insertion()
    /// );
    /// assert!(
    ///     TopologyGuarantee::PLManifold
    ///         .requires_pseudomanifold_checks_during_insertion()
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub const fn requires_pseudomanifold_checks_during_insertion(self) -> bool {
        matches!(
            self,
            Self::Pseudomanifold | Self::PLManifold | Self::PLManifoldStrict
        )
    }

    /// Returns `true` if this topology guarantee requires ridge-link validation.
    ///
    /// Ridge-link validation is fast (O(local)) and catches many PL-manifold violations,
    /// providing good error detection even with reduced validation frequency.
    #[inline]
    #[must_use]
    pub const fn requires_ridge_links(self) -> bool {
        matches!(self, Self::PLManifold | Self::PLManifoldStrict)
    }

    /// Returns the [`ValidationPolicy`] that should be used by default for this guarantee.
    ///
    /// [`PLManifoldStrict`](Self::PLManifoldStrict) uses [`Always`](ValidationPolicy::Always)
    /// so that full Level-3 global validation (including vertex-link checks) runs
    /// after every insertion — this is the strongest and slowest setting.
    /// [`PLManifold`](Self::PLManifold) uses [`ExplicitOnly`](ValidationPolicy::ExplicitOnly)
    /// because insertion-time ridge-link checks are mandatory but full vertex-link validation
    /// is a caller-owned completion checkpoint. [`Pseudomanifold`](Self::Pseudomanifold)
    /// uses [`OnSuspicion`](ValidationPolicy::OnSuspicion).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{TopologyGuarantee, ValidationPolicy};
    ///
    /// assert_eq!(
    ///     TopologyGuarantee::PLManifoldStrict.default_validation_policy(),
    ///     ValidationPolicy::Always,
    /// );
    /// assert_eq!(
    ///     TopologyGuarantee::PLManifold.default_validation_policy(),
    ///     ValidationPolicy::ExplicitOnly,
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub const fn default_validation_policy(self) -> ValidationPolicy {
        match self {
            Self::PLManifoldStrict => ValidationPolicy::Always,
            Self::PLManifold => ValidationPolicy::ExplicitOnly,
            Self::Pseudomanifold => ValidationPolicy::OnSuspicion,
        }
    }

    /// Returns `true` if this guarantee is compatible with the given validation policy.
    ///
    /// `PLManifold` requires at least end-of-construction validation, so it's incompatible
    /// with `ValidationPolicy::Never`.
    #[inline]
    #[must_use]
    pub const fn is_compatible_with_policy(self, policy: ValidationPolicy) -> bool {
        match self {
            Self::Pseudomanifold => true,
            Self::PLManifold | Self::PLManifoldStrict => !matches!(policy, ValidationPolicy::Never),
        }
    }
}

/// Errors returned when validation scheduling and topology guarantees are incoherent.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::FastKernel;
/// use delaunay::prelude::{
///     TopologyGuarantee, Triangulation, ValidationConfigurationError, ValidationPolicy,
/// };
///
/// let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
///     Triangulation::new_empty(FastKernel::new());
///
/// std::assert_matches!(
///     tri.try_set_validation_policy(ValidationPolicy::Never),
///     Err(ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
///         topology_guarantee: TopologyGuarantee::PLManifold,
///         validation_policy: ValidationPolicy::Never,
///     })
/// );
/// ```
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidationConfigurationError {
    /// The requested validation policy cannot support the requested topology guarantee.
    #[error(
        "validation policy {validation_policy:?} is incompatible with topology guarantee {topology_guarantee:?}"
    )]
    IncompatibleTopologyAndValidationPolicy {
        /// Topology guarantee that would be active.
        topology_guarantee: TopologyGuarantee,
        /// Validation policy that would be active.
        validation_policy: ValidationPolicy,
    },
}

/// Verifies that a topology guarantee and validation policy can coexist.
const fn validate_configuration(
    topology_guarantee: TopologyGuarantee,
    validation_policy: ValidationPolicy,
) -> Result<(), ValidationConfigurationError> {
    if topology_guarantee.is_compatible_with_policy(validation_policy) {
        Ok(())
    } else {
        Err(
            ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                topology_guarantee,
                validation_policy,
            },
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InsertionValidationWork {
    FullValidation,
    RequiredTopologyLinks,
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D> {
    /// Returns the topology guarantee used for Level 3 topology validation.
    #[inline]
    #[must_use]
    pub const fn topology_guarantee(&self) -> TopologyGuarantee {
        self.topology_guarantee
    }

    /// Returns the runtime global topology metadata associated with this triangulation.
    #[inline]
    #[must_use]
    pub const fn global_topology(&self) -> GlobalTopology<D> {
        self.global_topology
    }

    /// Returns the high-level topology kind (`Euclidean`, `Toroidal`, etc.).
    #[inline]
    #[must_use]
    pub const fn topology_kind(&self) -> TopologyKind {
        self.global_topology.kind()
    }

    /// Sets runtime global topology metadata after validating it against current topology.
    ///
    /// The update is atomic: if the current triangulation does not satisfy the
    /// requested global topology, the previous metadata is restored before the
    /// error is returned.
    ///
    /// # Errors
    ///
    /// Returns [`InvariantError::Tds`] if lower-level structure is invalid while
    /// checking topology, or [`InvariantError::Triangulation`] when Level 3
    /// topology violates the requested metadata, for example when Euclidean
    /// boundary facets are relabeled as closed spherical or toroidal topology.
    /// The previous topology metadata is restored before the error is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::Triangulation;
    /// use delaunay::prelude::geometry::FastKernel;
    /// use delaunay::prelude::tds::InvariantError;
    /// use delaunay::prelude::topology::spaces::GlobalTopology;
    ///
    /// # fn main() -> Result<(), InvariantError> {
    /// let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    ///
    /// tri.try_set_global_topology(GlobalTopology::Euclidean)?;
    /// assert_eq!(tri.global_topology(), GlobalTopology::Euclidean);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_set_global_topology(
        &mut self,
        global_topology: GlobalTopology<D>,
    ) -> Result<(), InvariantError> {
        let previous = self.global_topology;
        if previous == global_topology {
            return Ok(());
        }

        self.global_topology = global_topology;
        if let Err(err) = self.validate_topology_core() {
            self.global_topology = previous;
            return Err(err);
        }
        Ok(())
    }

    /// Shared Level-3 topology validation sequence used by both [`is_valid_topology`](Self::is_valid_topology)
    /// and [`is_valid_topology_only`](Self::is_valid_topology_only).
    ///
    /// Checks connectedness, manifold facet degree, closed boundary, ridge/vertex
    /// links (when required by the topology guarantee), isolated vertices, and
    /// Euler characteristic.
    fn validate_topology_core(&self) -> Result<(), InvariantError> {
        // 1. Connectedness
        //
        // Checked first because it is cheaper than building the facet-to-simplices map
        // (which requires O(N·D) hash-map insertions plus allocations) and avoids
        // all subsequent work when the triangulation is disconnected.
        self.validate_global_connectedness()?;

        // 2. Manifold facet multiplicity (codimension-1 pseudomanifold condition)
        //
        // Build the facet map once and reuse it for manifold validation and Euler counting.
        let facet_to_simplices: FacetToSimplicesMap = self.tds.build_facet_to_simplices_map()?;
        let facet_to_simplices = ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices)?;
        self.validate_topology_core_from_validated_facet_map(facet_to_simplices)
    }

    fn validate_topology_core_from_validated_facet_map(
        &self,
        facet_to_simplices: ValidatedFacetDegreeMap<'_>,
    ) -> Result<(), InvariantError> {
        // 2b. Boundary manifoldness in codimension 2: the boundary must be "closed"
        // (i.e., its ridges must have degree 2 within boundary facets).
        validate_closed_boundary_from_validated_facet_map(
            &self.tds,
            facet_to_simplices,
            self.global_topology,
        )?;

        // 2c. Ridge-link validation for PLManifold/PLManifoldStrict (fast, catches many PL issues).
        if self.topology_guarantee.requires_ridge_links() {
            validate_ridge_links(&self.tds)?;
        }
        // 2d. PL-manifold vertex-link condition during insertion (strict mode).
        if self
            .topology_guarantee
            .requires_vertex_links_during_insertion()
        {
            validate_vertex_links_from_validated_facet_map(
                &self.tds,
                facet_to_simplices,
                self.global_topology,
            )?;
        }

        // 3. Vertex incidence (manifold invariant): every vertex must be incident to at least one simplex.
        self.validate_no_isolated_vertices()?;

        // 4. Euler characteristic using the topology module
        let topology_result = validate_triangulation_euler_from_validated_facet_map(
            &self.tds,
            facet_to_simplices,
            self.global_topology,
        )
        .map_err(invariant_error_from_topology_error)?;

        if let Some(exp) = topology_result.expected
            && topology_result.chi != exp
        {
            return Err(TriangulationValidationError::EulerCharacteristicMismatch {
                computed: topology_result.chi,
                expected: exp,
                classification: topology_result.classification,
            }
            .into());
        }

        Ok(())
    }

    /// Validates that the triangulation's simplex neighbor graph is a single connected component.
    ///
    /// Delegates to [`Tds::is_connected`](crate::prelude::tds::Tds::is_connected), an O(N·D) BFS
    /// over neighbor pointers.
    pub(crate) fn validate_global_connectedness(&self) -> Result<(), TriangulationValidationError> {
        if !self.tds.is_connected() {
            return Err(TriangulationValidationError::Disconnected {
                simplex_count: self.tds.number_of_simplices(),
            });
        }
        Ok(())
    }

    /// Validates that every vertex is incident to at least one simplex.
    ///
    /// Isolated vertices are allowed at the TDS (structural) layer, but they violate the
    /// manifold invariants checked at the topology (Level 3) layer.
    pub(crate) fn validate_no_isolated_vertices(&self) -> Result<(), TriangulationValidationError> {
        if self.tds.number_of_vertices() == 0 {
            return Ok(());
        }

        let mut vertices_in_simplices: FastHashSet<VertexKey> =
            fast_hash_set_with_capacity(self.tds.number_of_vertices());

        for (_simplex_key, simplex) in self.tds.simplices() {
            for &vk in simplex.vertices() {
                vertices_in_simplices.insert(vk);
            }
        }

        for (vk, vertex) in self.tds.vertices() {
            if !vertices_in_simplices.contains(&vk) {
                return Err(TriangulationValidationError::IsolatedVertex {
                    vertex_key: vk,
                    vertex_uuid: vertex.uuid(),
                });
            }
        }

        Ok(())
    }

    /// Returns the insertion-time global topology validation policy used by the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{Triangulation, ValidationPolicy};
    /// use delaunay::prelude::geometry::FastKernel;
    ///
    /// let tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    ///
    /// assert_eq!(tri.validation_policy(), ValidationPolicy::ExplicitOnly);
    /// ```
    #[inline]
    #[must_use]
    pub const fn validation_policy(&self) -> ValidationPolicy {
        self.validation_policy
    }

    /// Tries to set the insertion-time global topology validation policy used by the triangulation.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy`] when the
    /// requested policy cannot support the current topology guarantee. For example,
    /// [`ValidationPolicy::Never`] is rejected when the triangulation has
    /// [`TopologyGuarantee::PLManifold`]; use [`ValidationPolicy::ExplicitOnly`] for
    /// caller-owned full validation checkpoints with stronger topology guarantees.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{Triangulation, ValidationPolicy};
    /// use delaunay::prelude::geometry::FastKernel;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::ValidationConfigurationError> {
    /// let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    ///
    /// tri.try_set_validation_policy(ValidationPolicy::Always)?;
    /// assert_eq!(tri.validation_policy(), ValidationPolicy::Always);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn try_set_validation_policy(
        &mut self,
        policy: ValidationPolicy,
    ) -> Result<(), ValidationConfigurationError> {
        validate_configuration(self.topology_guarantee, policy)?;
        self.validation_policy = policy;
        Ok(())
    }

    /// Sets the insertion-time global topology validation policy used by the triangulation.
    ///
    /// Prefer [`try_set_validation_policy`](Self::try_set_validation_policy) when callers need
    /// typed feedback for rejected combinations. This compatibility setter leaves the existing
    /// policy unchanged and emits a warning if the requested combination is incoherent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{Triangulation, ValidationPolicy};
    /// use delaunay::prelude::geometry::FastKernel;
    ///
    /// let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    ///
    /// tri.set_validation_policy(ValidationPolicy::Always);
    /// assert_eq!(tri.validation_policy(), ValidationPolicy::Always);
    /// ```
    #[inline]
    pub fn set_validation_policy(&mut self, policy: ValidationPolicy) {
        if let Err(err) = self.try_set_validation_policy(policy) {
            tracing::warn!("{err}. Validation policy not updated.");
        }
    }

    /// Tries to set the topology guarantee used for Level 3 topology validation.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy`] when the
    /// requested guarantee cannot be represented with the current validation policy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{TopologyGuarantee, Triangulation};
    /// use delaunay::prelude::geometry::FastKernel;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::ValidationConfigurationError> {
    /// let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// tri.try_set_topology_guarantee(TopologyGuarantee::Pseudomanifold)?;
    /// assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn try_set_topology_guarantee(
        &mut self,
        guarantee: TopologyGuarantee,
    ) -> Result<(), ValidationConfigurationError> {
        validate_configuration(guarantee, self.validation_policy)?;
        self.topology_guarantee = guarantee;
        Ok(())
    }

    /// Sets the topology guarantee used for Level 3 topology validation.
    ///
    /// Prefer [`try_set_topology_guarantee`](Self::try_set_topology_guarantee) when callers need
    /// typed feedback for rejected combinations. This compatibility setter leaves the existing
    /// guarantee unchanged and emits a warning if the requested combination is incoherent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::{TopologyGuarantee, Triangulation};
    /// use delaunay::prelude::geometry::FastKernel;
    ///
    /// let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
    /// assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// ```
    #[inline]
    pub fn set_topology_guarantee(&mut self, guarantee: TopologyGuarantee) {
        if let Err(err) = self.try_set_topology_guarantee(guarantee) {
            tracing::warn!("{err}. Topology guarantee not updated.");
        }
    }
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
{
    /// Traverses the simplex neighbor graph for validation without assuming global connectivity.
    ///
    /// If `allowed` is `Some`, traversal is restricted to that set. Neighbors
    /// outside the allowed set are reported through `on_external_neighbor` so
    /// localized validation can still prove the new component attaches to the
    /// existing triangulation.
    #[must_use]
    fn traverse_simplex_neighbor_graph<F>(
        &self,
        start: SimplexKey,
        reserve: usize,
        allowed: Option<&SimplexKeySet>,
        mut on_external_neighbor: F,
    ) -> SimplexKeySet
    where
        F: FnMut(SimplexKey, SimplexKey),
    {
        let mut visited: SimplexKeySet = SimplexKeySet::default();
        visited.reserve(reserve);

        let mut stack: SimplexKeyBuffer = SimplexKeyBuffer::new();
        stack.push(start);

        while let Some(ck) = stack.pop() {
            if !visited.insert(ck) {
                continue;
            }

            let Some(simplex) = self.tds.simplex(ck) else {
                continue;
            };

            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };

            for n_opt in neighbors {
                let Some(nk) = n_opt else {
                    continue;
                };

                if !self.tds.contains_simplex(nk) {
                    continue;
                }

                if allowed.is_some_and(|allowed| !allowed.contains(&nk)) {
                    on_external_neighbor(ck, nk);
                    continue;
                }

                if !visited.contains(&nk) {
                    stack.push(nk);
                }
            }
        }

        visited
    }

    /// Validates topological invariants of the triangulation (Level 3).
    ///
    /// This checks the triangulation/topology layer **only**:
    /// - Codimension-1 pseudomanifold condition: each facet is one-sided or two-sided.
    /// - Topology-aware boundary manifoldness: true boundary facets must be closed ("no boundary of boundary").
    /// - Geometric orientation-sign consistency for stored simplices (signed determinant > 0)
    /// - Ridge-link validation (when `topology_guarantee.requires_ridge_links()`)
    /// - Vertex-link validation during insertion (when `topology_guarantee.requires_vertex_links_during_insertion()`)
    /// - Connectedness (single component in the simplex neighbor graph)
    /// - No isolated vertices (every vertex must be incident to at least one simplex)
    /// - Euler characteristic
    ///
    /// For `TopologyGuarantee::PLManifold`, full PL-manifold certification requires
    /// calling [`Triangulation::validate_at_completion`](Self::validate_at_completion)
    /// (or [`Triangulation::validate`](Self::validate)) after batch construction.
    ///
    /// It intentionally does **not** validate lower layers (vertices/simplices or TDS structure).
    /// For cumulative validation, use [`Triangulation::validate`](Self::validate).
    ///
    /// # Errors
    ///
    /// Returns an [`InvariantError`] if:
    /// - The manifold-with-boundary facet property is violated.
    /// - The triangulation is disconnected (multiple simplex components).
    /// - An isolated vertex is detected (no incident simplex).
    /// - Euler characteristic validation fails.
    /// - The topology module reports an error (treated as inconsistent data structure).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices_4d = [
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulationBuilder::new(&vertices_4d).build::<()>()?;
    ///
    /// // Level 3: topology validation (manifold-with-boundary + Euler characteristic)
    /// assert!(dt.as_triangulation().is_valid_topology().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_valid_topology(&self) -> Result<(), InvariantError> {
        self.validate_topology_core()?;
        // Check geometric orientation after manifold/link checks so topology-specific
        // diagnostics surface first when multiple invariants are violated.
        self.validate_geometric_simplex_orientation()?;
        Ok(())
    }

    /// Returns the first actionable Level 3 topology diagnostic, if any.
    ///
    /// This is the repair/retry-oriented counterpart to
    /// [`is_valid_topology`](Self::is_valid_topology). It preserves the
    /// [`InvariantKind`] grouping used by aggregate reports while returning at
    /// most one local failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, vertex,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// assert!(dt.as_triangulation().topology_diagnostic().is_none());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn topology_diagnostic(&self) -> Option<InvariantViolation> {
        self.topology_report()
            .err()
            .and_then(|report| report.violations.into_iter().next())
    }

    /// Generate a Level 3 topology report.
    ///
    /// This report checks topology-layer invariants only. It assumes the TDS
    /// structure is already valid; use [`validation_report`](Self::validation_report)
    /// for cumulative Levels 1-4 diagnostics.
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationReport)` when one or more checkable
    /// topology-layer invariants fail.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, vertex,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// assert!(dt.as_triangulation().topology_report().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn topology_report(&self) -> Result<(), TriangulationValidationReport> {
        let mut violations = Vec::new();

        if let Err(source) = self.validate_global_connectedness() {
            violations.push(InvariantViolation {
                kind: InvariantKind::Connectedness,
                error: InvariantError::Triangulation(source),
            });
        }

        match self.tds.build_facet_to_simplices_map() {
            Ok(facet_to_simplices) => {
                match ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices) {
                    Ok(validated_facets) => {
                        if let Err(source) =
                            self.validate_topology_core_from_validated_facet_map(validated_facets)
                        {
                            violations.push(InvariantViolation {
                                kind: InvariantKind::Topology,
                                error: source,
                            });
                        }
                    }
                    Err(source) => {
                        violations.push(InvariantViolation {
                            kind: InvariantKind::Topology,
                            error: source.into(),
                        });
                    }
                }
            }
            Err(source) => {
                violations.push(InvariantViolation {
                    kind: InvariantKind::Topology,
                    error: source.into(),
                });
            }
        }

        if let Err(source) = self.validate_geometric_simplex_orientation() {
            violations.push(InvariantViolation {
                kind: InvariantKind::Topology,
                error: source.into(),
            });
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(TriangulationValidationReport { violations })
        }
    }

    /// Validates topological invariants **without** geometric orientation checks.
    ///
    /// This is identical to [`is_valid_topology`](Self::is_valid_topology) but omits the
    /// `validate_geometric_simplex_orientation()` step. It is intended for
    /// explicit combinatorial construction where the user-provided vertex
    /// orderings may produce negative determinants that are nonetheless
    /// topologically valid.
    pub(crate) fn is_valid_topology_only(&self) -> Result<(), InvariantError> {
        self.validate_topology_core()
    }

    /// Validates vertex-link condition at construction completion.
    ///
    /// This should be called once after batch construction is complete to certify
    /// full PL-manifoldness when using `TopologyGuarantee::PLManifold` (incremental mode).
    ///
    /// # Errors
    ///
    /// Returns an [`InvariantError`] if vertex-link validation fails
    /// (e.g. a vertex link is not a PL-sphere/ball as required for PL-manifoldness).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
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
    /// let dt: DelaunayTriangulation<_, (), (), 3> =
    ///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// assert!(dt.as_triangulation().validate_at_completion().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate_at_completion(&self) -> Result<(), InvariantError> {
        if !self
            .topology_guarantee
            .requires_vertex_links_at_completion()
        {
            return Ok(());
        }

        if self.tds.number_of_simplices() == 0 {
            return Ok(());
        }

        let facet_to_simplices: FacetToSimplicesMap = self.tds.build_facet_to_simplices_map()?;
        let facet_to_simplices = ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices)?;
        self.validate_at_completion_from_validated_facet_map(facet_to_simplices)?;
        Ok(())
    }

    fn validate_at_completion_from_validated_facet_map(
        &self,
        facet_to_simplices: ValidatedFacetDegreeMap<'_>,
    ) -> Result<(), InvariantError> {
        if !self
            .topology_guarantee
            .requires_vertex_links_at_completion()
        {
            return Ok(());
        }

        if self.tds.number_of_simplices() == 0 {
            return Ok(());
        }

        validate_vertex_links_from_validated_facet_map(
            &self.tds,
            facet_to_simplices,
            self.global_topology,
        )?;
        Ok(())
    }

    /// Performs cumulative validation for Levels 1–3.
    ///
    /// This validates:
    /// - **Level 1–2** via [`Tds::validate`](crate::prelude::tds::Tds::validate)
    /// - **Level 3** via [`Triangulation::is_valid_topology`](Self::is_valid_topology)
    /// - **Completion-time PL-manifold check** via [`Triangulation::validate_at_completion`](Self::validate_at_completion)
    ///
    /// # Errors
    ///
    /// Returns an [`InvariantError`] if:
    /// - Any vertex/simplex is invalid (Level 1).
    /// - The TDS structural invariants fail (Level 2).
    /// - Topology validation fails (Level 3).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices_4d = [
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulationBuilder::new(&vertices_4d).build::<()>()?;
    ///
    /// // Levels 1–3: elements + TDS structure + topology
    /// assert!(dt.as_triangulation().validate().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate(&self) -> Result<(), InvariantError>
    where
        U: DataType,
        V: DataType,
    {
        self.tds.validate()?;
        self.validate_global_connectedness()?;
        let facet_to_simplices: FacetToSimplicesMap = self.tds.build_facet_to_simplices_map()?;
        let facet_to_simplices = ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices)?;
        self.validate_topology_core_from_validated_facet_map(facet_to_simplices)?;
        // Check geometric orientation after manifold/link checks so topology-specific
        // diagnostics surface first when multiple invariants are violated.
        self.validate_geometric_simplex_orientation()?;
        self.validate_at_completion_from_validated_facet_map(facet_to_simplices)
    }

    /// Generate a comprehensive validation report for Levels 1–3.
    ///
    /// This is intended for debugging, telemetry, tests, and repair planning
    /// where you want to see all checkable violated invariants, not just the
    /// first one.
    ///
    /// # Notes
    /// - If UUID↔key mappings are inconsistent, this returns only mapping failures (other
    ///   checks may produce misleading secondary errors).
    /// - This report is **cumulative** across Levels 1–3.
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationReport)` containing all invariant violations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// assert!(dt.as_triangulation().validation_report().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn validation_report(&self) -> Result<(), TriangulationValidationReport>
    where
        U: DataType,
        V: DataType,
    {
        let mut violations: Vec<InvariantViolation> = Vec::new();

        // Levels 1-2: reuse the TDS cumulative report.
        match self.tds.validation_report() {
            Ok(()) => {}
            Err(report) => {
                if report.violations.iter().any(|v| {
                    matches!(
                        v.kind,
                        InvariantKind::VertexMappings | InvariantKind::SimplexMappings
                    )
                }) {
                    return Err(report);
                }
                violations.extend(report.violations);
            }
        }

        // Level 3: topology.
        if let Err(report) = self.topology_report() {
            violations.extend(report.violations);
        }

        if let Err(source) = self.validate_at_completion() {
            violations.push(InvariantViolation {
                kind: InvariantKind::Topology,
                error: source,
            });
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(TriangulationValidationReport { violations })
        }
    }

    /// Convert an [`InvariantError`] into the appropriate [`InsertionError`] variant.
    ///
    /// - `InvariantError::Tds(e)` → `InsertionError::TopologyValidation(e)`
    /// - `InvariantError::Triangulation(e)` → `InsertionError::TopologyValidationFailed { source: e }`
    /// - `InvariantError::Embedding(e)` → `InsertionError::EmbeddingValidationFailed { source: e }`
    /// - `InvariantError::Delaunay(e)` → `InsertionError::DelaunayValidationFailed { source: e }`
    pub(crate) fn invariant_error_to_insertion_error(err: InvariantError) -> InsertionError {
        match err {
            InvariantError::Tds(tds_err) => InsertionError::TopologyValidation(tds_err),
            InvariantError::Triangulation(tri_err) => InsertionError::TopologyValidationFailed {
                context: InsertionTopologyValidationContext::InvariantConversion,
                source: tri_err,
            },
            InvariantError::Embedding(embedding_err) => InsertionError::EmbeddingValidationFailed {
                source: embedding_err,
            },
            InvariantError::Delaunay(dt_err) => {
                InsertionError::DelaunayValidationFailed { source: dt_err }
            }
        }
    }

    /// Runs mandatory link checks required by the topology guarantee.
    pub(crate) fn validate_required_topology_links(&self) -> Result<(), InvariantError> {
        if self.tds.number_of_simplices() == 0 {
            return Ok(());
        }

        let facet_to_simplices: FacetToSimplicesMap = self.tds.build_facet_to_simplices_map()?;
        let facet_to_simplices = ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices)?;
        validate_closed_boundary_from_validated_facet_map(
            &self.tds,
            facet_to_simplices,
            self.global_topology,
        )?;

        if self.topology_guarantee.requires_ridge_links() {
            validate_ridge_links(&self.tds)?;
        }

        if self
            .topology_guarantee
            .requires_vertex_links_during_insertion()
        {
            validate_vertex_links_from_validated_facet_map(
                &self.tds,
                facet_to_simplices,
                self.global_topology,
            )?;
        }

        // Keep geometric orientation non-negotiable during incremental insertion,
        // even when global validation is throttled. Run this after topology
        // checks so topology diagnostics still surface first.
        self.validate_geometric_simplex_orientation()?;
        let simplex_keys: SimplexKeyBuffer = self.tds.simplex_keys().collect();
        self.validate_local_embedding_nondegeneracy(&simplex_keys)
            .map_err(InvariantError::Embedding)?;

        Ok(())
    }

    /// Runs the localized connectedness guard after insertion.
    ///
    /// This checks that surviving new simplices form one component and, when
    /// older simplices exist, that the new component attaches back to them via
    /// mutual neighbor pointers.
    pub(crate) fn validate_connectedness(
        &self,
        new_simplices: &SimplexKeyBuffer,
    ) -> Result<(), InsertionError> {
        let total_simplices = self.tds.number_of_simplices();
        if total_simplices == 0 {
            return Ok(());
        }

        let mut new_set: SimplexKeySet = SimplexKeySet::default();
        new_set.reserve(new_simplices.len());
        for &ck in new_simplices {
            if self.tds.contains_simplex(ck) {
                new_set.insert(ck);
            }
        }

        if new_set.is_empty() {
            return Err(InsertionError::TopologyValidation(
                TdsError::InconsistentDataStructure {
                    message: "Disconnected triangulation detected after insertion: no surviving new simplices"
                        .to_string(),
                },
            ));
        }

        let expected_new_simplices = new_set.len();

        let Some(&start) = new_set.iter().next() else {
            return Err(InsertionError::TopologyValidation(
                TdsError::InconsistentDataStructure {
                    message:
                        "new_set unexpectedly empty after non-empty check in validate_connectedness"
                            .to_string(),
                },
            ));
        };

        let mut touches_existing_simplices = false;

        let visited = self.traverse_simplex_neighbor_graph(
            start,
            expected_new_simplices,
            Some(&new_set),
            |ck, nk| {
                if touches_existing_simplices {
                    return;
                }

                // For connectivity between new simplices and existing simplices, require *mutual* adjacency.
                // This avoids treating one-way neighbor pointers as “connected”.
                if let Some(neighbor_simplex) = self.tds.simplex(nk)
                    && neighbor_simplex
                        .neighbor_keys()
                        .is_some_and(|mut neighbor_keys| {
                            neighbor_keys.any(|neighbor| neighbor == Some(ck))
                        })
                {
                    touches_existing_simplices = true;
                }
            },
        );

        if visited.len() != expected_new_simplices {
            return Err(InsertionError::TopologyValidation(
                TdsError::InconsistentDataStructure {
                    message: format!(
                        "Disconnected triangulation detected after insertion: new-simplex subgraph visited {} of {} simplices",
                        visited.len(),
                        expected_new_simplices
                    ),
                },
            ));
        }

        if total_simplices > expected_new_simplices && !touches_existing_simplices {
            return Err(InsertionError::TopologyValidation(
                TdsError::InconsistentDataStructure {
                    message: format!(
                        "Disconnected triangulation detected after insertion: new-simplex component ({expected_new_simplices} simplices) is not connected to existing simplices (total_simplices={total_simplices})"
                    ),
                },
            ));
        }

        Ok(())
    }

    /// Runs mandatory topology checks over the local simplices touched by insertion.
    ///
    /// Soundness boundary: the scoped path checks coherent orientation, local
    /// pseudomanifold facet incidence, ridge links, and geometric simplex
    /// orientation. Those local checks are sufficient only when `simplices` is
    /// non-empty and `topology_guarantee` does not require vertex-link checks
    /// during insertion; otherwise this explicitly falls back to
    /// [`validate_required_topology_links`](Self::validate_required_topology_links).
    /// See `REFERENCES.md`, "Scoped Local Validation and Flips" \[1\], for the
    /// local-vs-global validation tradeoff and geometric conditioning context.
    pub(crate) fn validate_required_topology_links_for_simplices(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<(), InvariantError> {
        if self.tds.number_of_simplices() == 0 {
            return Ok(());
        }

        if simplices.is_empty()
            || self
                .topology_guarantee
                .requires_vertex_links_during_insertion()
        {
            return self.validate_required_topology_links();
        }

        self.tds
            .validate_coherent_orientation_for_simplices(simplices)?;
        validate_local_pseudomanifold_for_simplices(&self.tds, self.global_topology, simplices)?;

        if self.topology_guarantee.requires_ridge_links() {
            validate_ridge_links_for_simplices(&self.tds, simplices.iter().copied())?;
        }

        self.validate_geometric_simplex_orientation_for_simplices(simplices)?;
        self.validate_local_embedding_nondegeneracy(simplices)
            .map_err(InvariantError::Embedding)?;

        Ok(())
    }

    pub(crate) fn validation_after_insertion_work(
        &self,
        suspicion: SuspicionFlags,
    ) -> Option<InsertionValidationWork> {
        if self.tds.number_of_simplices() == 0 {
            return None;
        }

        let should_validate = self.validation_policy.should_validate(suspicion);
        let requires_required_topology_checks = self
            .topology_guarantee
            .requires_pseudomanifold_checks_during_insertion();

        if should_validate {
            Some(InsertionValidationWork::FullValidation)
        } else if requires_required_topology_checks {
            Some(InsertionValidationWork::RequiredTopologyLinks)
        } else {
            None
        }
    }

    pub(crate) fn validate_after_insertion_with_scope(
        &self,
        suspicion: SuspicionFlags,
        local_simplices: Option<&[SimplexKey]>,
    ) -> Result<(), InvariantError>
    where
        U: DataType,
        V: DataType,
    {
        let Some(work) = self.validation_after_insertion_work(suspicion) else {
            return Ok(());
        };

        log_validation_trigger_if_enabled(self.validation_policy, suspicion);
        match work {
            InsertionValidationWork::FullValidation => {
                self.validate()?;
                match local_simplices {
                    Some([]) => Ok(()),
                    Some(simplices) => self.validate_embedding_for_simplices(simplices),
                    None => self.is_valid_embedding(),
                }
                .map_err(InvariantError::Embedding)
            }
            InsertionValidationWork::RequiredTopologyLinks => local_simplices.map_or_else(
                || self.validate_required_topology_links(),
                |simplices| self.validate_required_topology_links_for_simplices(simplices),
            ),
        }
    }

    /// Runs post-insertion validation and records count/timing telemetry for the selected work.
    pub(crate) fn validate_after_insertion_and_record_telemetry(
        &self,
        suspicion: SuspicionFlags,
        local_simplices: &[SimplexKey],
        telemetry: &mut InsertionTelemetry,
        telemetry_mode: InsertionTelemetryMode,
    ) -> Result<(), InvariantError>
    where
        U: DataType,
        V: DataType,
    {
        let validation_work = self.validation_after_insertion_work(suspicion);
        let validation_started =
            validation_work.and_then(|_| start_insertion_timing(telemetry_mode));
        let validation_result =
            self.validate_after_insertion_with_scope(suspicion, Some(local_simplices));

        if validation_work.is_some() {
            record_topology_validation_telemetry(
                telemetry,
                validation_started.map(|started| duration_nanos_saturating(started.elapsed())),
            );
        }

        validation_result
    }
}

/// Logs when Level 3 validation is triggered (debug builds only).
#[inline]
fn log_validation_trigger_if_enabled(policy: ValidationPolicy, suspicion: SuspicionFlags) {
    #[cfg(debug_assertions)]
    if policy.should_validate(suspicion) && suspicion.is_suspicious() {
        tracing::debug!("Validation triggered by {suspicion:?}");
    }

    // Keep the parameters "used" in release builds where the debug-only logging
    // is compiled out, so `cargo clippy -D warnings` stays clean across profiles.
    #[cfg(not(debug_assertions))]
    {
        let _ = policy;
        let _ = suspicion;
    }
}

/// Records one topology-validation pass and its optional elapsed time.
#[inline]
fn record_topology_validation_telemetry(
    telemetry: &mut InsertionTelemetry,
    elapsed_nanos: Option<u64>,
) {
    telemetry.topology_validation_calls = telemetry.topology_validation_calls.saturating_add(1);
    if let Some(elapsed_nanos) = elapsed_nanos {
        telemetry.topology_validation_nanos = telemetry
            .topology_validation_nanos
            .saturating_add(elapsed_nanos);
        telemetry.topology_validation_nanos_max =
            telemetry.topology_validation_nanos_max.max(elapsed_nanos);
    }
}

/// Convert a duration to nanoseconds while saturating at `u64::MAX`.
#[inline]
fn duration_nanos_saturating(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Starts a wall-clock timer only when insertion telemetry will publish timings.
#[inline]
fn start_insertion_timing(telemetry_mode: InsertionTelemetryMode) -> Option<Instant> {
    telemetry_mode.records_timings().then(Instant::now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::flips::{DelaunayRepairError, DelaunayRepairPostconditionFailure};
    use crate::core::algorithms::incremental_insertion::CavityFillingError;
    use crate::core::algorithms::incremental_insertion::repair_neighbor_pointers;
    use crate::core::collections::{NeighborBuffer, SimplexVertexKeyBuffer};
    use crate::core::embedding::TriangulationEmbeddingValidationError;
    use crate::core::facet::FacetError;
    use crate::core::operations::InsertionOutcome;
    use crate::core::simplex::Simplex;
    use crate::core::tds::{
        GeometricError, NeighborValidationError, Tds, TriangulationConstructionState,
    };
    use crate::core::vertex::Vertex;
    use crate::geometry::coordinate_range::CoordinateRange;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::point::Point;
    use crate::geometry::util::generate_random_points_in_range_seeded;
    use crate::repair::DelaunayRepairPolicy;
    use crate::triangulation::DelaunayTriangulation;
    use crate::validation::{DelaunayTriangulationValidationError, DelaunayVerificationError};
    use crate::vertex;
    use slotmap::KeyData;
    use std::{assert_matches, iter};

    fn synthetic_delaunay_verification_error(
        message: &str,
    ) -> DelaunayTriangulationValidationError {
        let _ = message;
        DelaunayTriangulationValidationError::VerificationFailed {
            source: DelaunayVerificationError::from(DelaunayRepairError::PostconditionFailed {
                reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected {
                    simplex_count: 1,
                }),
            })
            .into(),
        }
    }

    const fn synthetic_embedding_error() -> TriangulationEmbeddingValidationError {
        TriangulationEmbeddingValidationError::UnsupportedTopology {
            topology: TopologyKind::Spherical,
            dimension: 3,
        }
    }

    fn test_vertex<const D: usize>(coords: [f64; D]) -> Vertex<(), D> {
        vertex!(coords).unwrap()
    }

    fn insert_test_vertex_with_coords<const D: usize>(
        tds: &mut Tds<(), (), D>,
        entries: &[(usize, f64)],
    ) -> VertexKey {
        let mut coords = [0.0_f64; D];
        for &(axis, value) in entries {
            coords[axis] = value;
        }
        tds.insert_vertex_with_mapping(test_vertex(coords)).unwrap()
    }

    fn build_invalid_vertex_link_tds<const D: usize>() -> (Tds<(), (), D>, VertexKey) {
        let mut tds: Tds<(), (), D> = Tds::empty();
        let shared = insert_test_vertex_with_coords(&mut tds, &[]);

        if D == 2 {
            let first_a = insert_test_vertex_with_coords(&mut tds, &[(0, 1.0)]);
            let first_b = insert_test_vertex_with_coords(&mut tds, &[(1, 1.0)]);
            let first_c = insert_test_vertex_with_coords(&mut tds, &[(0, -1.0)]);
            let second_a = insert_test_vertex_with_coords(&mut tds, &[(0, 10.0)]);
            let second_b = insert_test_vertex_with_coords(&mut tds, &[(0, 11.0), (1, 1.0)]);
            let second_c = insert_test_vertex_with_coords(&mut tds, &[(0, 9.0), (1, 1.0)]);

            for simplex_vertices in [
                vec![shared, first_a, first_b],
                vec![shared, first_b, first_c],
                vec![shared, first_c, first_a],
                vec![shared, second_a, second_b],
                vec![shared, second_b, second_c],
                vec![shared, second_c, second_a],
            ] {
                let _ = tds
                    .insert_simplex_with_mapping(
                        Simplex::try_new_with_data(simplex_vertices, None).unwrap(),
                    )
                    .unwrap();
            }

            tds.assign_incident_simplices().unwrap();
            return (tds, shared);
        }

        let mut first_simplex_vertices = vec![shared];
        for axis in 0..D {
            let mut coords = [0.0_f64; D];
            coords[axis] = 1.0;
            first_simplex_vertices
                .push(tds.insert_vertex_with_mapping(test_vertex(coords)).unwrap());
        }

        let mut second_simplex_vertices = vec![shared];
        for axis in 0..D {
            let mut coords = [0.0_f64; D];
            coords[0] = 10.0;
            coords[axis] += 1.0;
            second_simplex_vertices
                .push(tds.insert_vertex_with_mapping(test_vertex(coords)).unwrap());
        }

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(first_simplex_vertices, None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(second_simplex_vertices, None).unwrap(),
            )
            .unwrap();

        tds.assign_incident_simplices().unwrap();

        (tds, shared)
    }

    fn build_disconnected_two_triangles_tds_2d() -> Tds<(), (), 2> {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let a0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let a1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let a2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();

        let b0 = tds
            .insert_vertex_with_mapping(test_vertex([10.0, 0.0]))
            .unwrap();
        let b1 = tds
            .insert_vertex_with_mapping(test_vertex([11.0, 0.0]))
            .unwrap();
        let b2 = tds
            .insert_vertex_with_mapping(test_vertex([10.0, 1.0]))
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![a0, a1, a2], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![b0, b1, b2], None).unwrap(),
            )
            .unwrap();

        tds
    }

    fn build_three_triangles_sharing_edge_tds_2d() -> Tds<(), (), 2> {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, -1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(test_vertex([2.0, 0.0]))
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![v0, v1, v4], None).unwrap(),
            )
            .unwrap();

        tds
    }

    fn build_topologically_valid_self_overlapping_tds_2d() -> (Tds<(), (), 2>, SimplexKeyBuffer) {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let mut vertices = Vec::new();
        for coords in [
            [1.850_341_970_997_476_4, 3.808_736_162_215_642_4],
            [-1.705_108_018_057_679, 3.541_228_835_829_82],
            [-1.151_312_061_387_885_3, 0.227_299_663_756_810_77],
            [0.478_746_443_632_698_25, 2.055_189_799_064_582],
            [-1.383_321_070_900_029, -1.797_028_018_114_396_3],
            [3.030_089_610_961_752_6, 2.181_406_554_808_236_6],
        ] {
            vertices.push(tds.insert_vertex_with_mapping(test_vertex(coords)).unwrap());
        }

        let mut simplex_keys = SimplexKeyBuffer::new();
        for simplex_vertices in [
            [vertices[0], vertices[2], vertices[3]],
            [vertices[5], vertices[3], vertices[2]],
            [vertices[4], vertices[3], vertices[1]],
            [vertices[3], vertices[4], vertices[0]],
            [vertices[3], vertices[5], vertices[1]],
        ] {
            let simplex_vertices: SimplexVertexKeyBuffer = simplex_vertices.into_iter().collect();
            simplex_keys.push(
                tds.insert_simplex_with_mapping(
                    Simplex::try_new_with_data(simplex_vertices, None).unwrap(),
                )
                .unwrap(),
            );
        }

        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();

        (tds, simplex_keys)
    }

    fn unit_simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(test_vertex([0.0_f64; D]));
        for axis in 0..D {
            let mut coords = [0.0_f64; D];
            coords[axis] = 1.0;
            vertices.push(test_vertex(coords));
        }
        vertices
    }

    fn unit_simplex_interior_vertex<const D: usize>() -> Vertex<(), D> {
        test_vertex([0.125_f64; D])
    }

    fn build_single_tet() -> (
        Triangulation<FastKernel<f64>, (), (), 3>,
        [VertexKey; 4],
        SimplexKey,
    ) {
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();
        let ck = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        for vk in [v0, v1, v2, v3] {
            tds.vertex_mut(vk).unwrap().set_incident_simplex(Some(ck));
        }

        (
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds),
            [v0, v1, v2, v3],
            ck,
        )
    }

    fn build_degenerate_tet() -> (Triangulation<FastKernel<f64>, (), (), 3>, SimplexKey) {
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 1.0, 0.0]))
            .unwrap();
        let ck = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        for vk in [v0, v1, v2, v3] {
            tds.vertex_mut(vk).unwrap().set_incident_simplex(Some(ck));
        }

        (
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds),
            ck,
        )
    }

    #[test]
    fn triangulation_validation_error_try_from_manifold_error_preserves_detail() {
        let tds_err = TdsError::InconsistentDataStructure {
            message: "unit test".to_string(),
        };

        assert_eq!(
            TriangulationValidationError::try_from(ManifoldError::Tds(tds_err.clone())),
            Err(tds_err.clone())
        );
        assert_eq!(
            InvariantError::from(ManifoldError::Tds(tds_err.clone())),
            InvariantError::Tds(tds_err)
        );

        assert_matches!(
            TriangulationValidationError::try_from(ManifoldError::ManifoldFacetMultiplicity {
                facet_key: 123,
                simplex_count: 3
            })
            .unwrap(),
            TriangulationValidationError::ManifoldFacetMultiplicity {
                facet_key: 123,
                simplex_count: 3
            }
        );

        assert_matches!(
            TriangulationValidationError::try_from(ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key: 0x00ab_cdef,
                boundary_facet_count: 4
            })
            .unwrap(),
            TriangulationValidationError::BoundaryRidgeMultiplicity {
                ridge_key: 0x00ab_cdef,
                boundary_facet_count: 4
            }
        );

        let ridge_vertex = VertexKey::from(KeyData::from_ffi(7));
        assert_matches!(
            TriangulationValidationError::try_from(ManifoldError::RidgeNotFound {
                ridge_vertices: iter::once(ridge_vertex).collect()
            })
            .unwrap(),
            TriangulationValidationError::RidgeNotFound { ridge_vertices }
                if ridge_vertices.as_slice() == [ridge_vertex]
        );

        assert_matches!(
            TriangulationValidationError::try_from(ManifoldError::RidgeLinkNotManifold {
                ridge_key: 0x00ab_cdef,
                link_vertex_count: 7,
                link_edge_count: 8,
                max_degree: 3,
                degree_one_vertices: 2,
                connected: false
            })
            .unwrap(),
            TriangulationValidationError::RidgeLinkNotManifold {
                ridge_key: 0x00ab_cdef,
                link_vertex_count: 7,
                link_edge_count: 8,
                max_degree: 3,
                degree_one_vertices: 2,
                connected: false
            }
        );

        assert_matches!(
            TriangulationValidationError::try_from(ManifoldError::VertexLinkNotManifold {
                vertex_key: VertexKey::from(KeyData::from_ffi(1)),
                link_vertex_count: 3,
                link_simplex_count: 4,
                boundary_facet_count: 1,
                max_degree: 2,
                connected: false,
                interior_vertex: true,
            })
            .unwrap(),
            TriangulationValidationError::VertexLinkNotManifold {
                link_vertex_count: 3,
                link_simplex_count: 4,
                boundary_facet_count: 1,
                max_degree: 2,
                connected: false,
                interior_vertex: true,
                ..
            }
        );
    }

    #[test]
    fn triangulation_validation_error_preserves_boundary_classification_detail() {
        let closed_topology_simplex_key = SimplexKey::from(KeyData::from_ffi(11));
        let closed_topology_simplex_uuid = Uuid::from_u128(0x1111);
        assert_matches!(
            TriangulationValidationError::try_from(
                ManifoldError::BoundaryFacetInClosedTopology {
                    topology: TopologyKind::Spherical,
                    facet_key: 0xfeed_face,
                    simplex_key: closed_topology_simplex_key,
                    simplex_uuid: closed_topology_simplex_uuid,
                    facet_index: 2
                }
            )
            .unwrap(),
            TriangulationValidationError::BoundaryFacetInClosedTopology {
                topology: TopologyKind::Spherical,
                facet_key: 0xfeed_face,
                simplex_key,
                simplex_uuid,
                facet_index: 2
            } if simplex_key == closed_topology_simplex_key
                && simplex_uuid == closed_topology_simplex_uuid
        );

        let non_periodic_simplex_key = SimplexKey::from(KeyData::from_ffi(12));
        let non_periodic_simplex_uuid = Uuid::from_u128(0x2222);
        assert_matches!(
            TriangulationValidationError::try_from(
                ManifoldError::PeriodicIdentificationInNonPeriodicTopology {
                    topology: TopologyKind::Euclidean,
                    facet_key: 0xdeca_fbad,
                    simplex_key: non_periodic_simplex_key,
                    simplex_uuid: non_periodic_simplex_uuid,
                    facet_index: 1
                }
            )
            .unwrap(),
            TriangulationValidationError::PeriodicIdentificationInNonPeriodicTopology {
                topology: TopologyKind::Euclidean,
                facet_key: 0xdeca_fbad,
                simplex_key,
                simplex_uuid,
                facet_index: 1
            } if simplex_key == non_periodic_simplex_key
                && simplex_uuid == non_periodic_simplex_uuid
        );
    }

    #[test]
    fn topology_error_adapter_preserves_validation_layering() {
        let facet_map_err = TdsError::InconsistentDataStructure {
            message: "facet map".to_string(),
        };
        assert_eq!(
            invariant_error_from_topology_error(TopologyError::FacetMapBuild {
                source: facet_map_err.clone()
            }),
            InvariantError::Tds(facet_map_err)
        );

        let boundary_enumeration_err = TdsError::InconsistentDataStructure {
            message: "boundary enumeration".to_string(),
        };
        assert_eq!(
            invariant_error_from_topology_error(TopologyError::BoundaryFacetEnumeration {
                source: boundary_enumeration_err.clone()
            }),
            InvariantError::Tds(boundary_enumeration_err)
        );

        let boundary_count_err = TdsError::InconsistentDataStructure {
            message: "boundary count".to_string(),
        };
        assert_eq!(
            invariant_error_from_topology_error(TopologyError::BoundaryFacetCount {
                source: boundary_count_err.clone()
            }),
            InvariantError::Tds(boundary_count_err)
        );

        assert_eq!(
            invariant_error_from_topology_error(TopologyError::BoundaryFacetSimplexAccess {
                source: FacetError::SimplexNotFoundInTriangulation
            }),
            InvariantError::Tds(TdsError::FacetError(
                FacetError::SimplexNotFoundInTriangulation
            ))
        );

        let simplex_key = SimplexKey::from(KeyData::from_ffi(13));
        let simplex_uuid = Uuid::from_u128(0x3333);
        assert_matches!(
            invariant_error_from_topology_error(TopologyError::BoundaryClassification {
                source: Box::new(ManifoldError::BoundaryFacetInClosedTopology {
                    topology: TopologyKind::Toroidal,
                    facet_key: 0x1234_5678,
                    simplex_key,
                    simplex_uuid,
                    facet_index: 3
                })
            }),
            InvariantError::Triangulation(
                TriangulationValidationError::BoundaryFacetInClosedTopology {
                    topology: TopologyKind::Toroidal,
                    facet_key: 0x1234_5678,
                    simplex_key: observed_simplex_key,
                    simplex_uuid: observed_simplex_uuid,
                    facet_index: 3
                }
            ) if observed_simplex_key == simplex_key && observed_simplex_uuid == simplex_uuid
        );
    }

    #[test]
    fn try_global_topology_setter_rejects_closed_metadata_for_euclidean_boundary() {
        let vertices: Vec<Vertex<(), 2>> = vec![
            test_vertex([0.0, 0.0]),
            test_vertex([1.0, 0.0]),
            test_vertex([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        let err = tri
            .try_set_global_topology(GlobalTopology::Spherical)
            .unwrap_err();

        assert_matches!(
            err,
            InvariantError::Triangulation(
                TriangulationValidationError::BoundaryFacetInClosedTopology {
                    topology: TopologyKind::Spherical,
                    ..
                }
            )
        );
        assert_eq!(tri.global_topology(), GlobalTopology::Euclidean);
        assert!(tri.is_valid_topology().is_ok());
    }

    #[test]
    fn validation_policy_should_validate_matrix() {
        let clean = SuspicionFlags::default();
        let suspicious = SuspicionFlags {
            perturbation_used: true,
            ..SuspicionFlags::default()
        };

        assert!(!ValidationPolicy::Never.should_validate(clean));
        assert!(!ValidationPolicy::Never.should_validate(suspicious));
        assert!(!ValidationPolicy::ExplicitOnly.should_validate(clean));
        assert!(!ValidationPolicy::ExplicitOnly.should_validate(suspicious));
        assert!(ValidationPolicy::Always.should_validate(clean));
        assert!(ValidationPolicy::Always.should_validate(suspicious));
        assert!(!ValidationPolicy::OnSuspicion.should_validate(clean));
        assert!(ValidationPolicy::OnSuspicion.should_validate(suspicious));
        assert!(ValidationPolicy::DebugOnly.should_validate(suspicious));
        assert_eq!(
            ValidationPolicy::DebugOnly.should_validate(clean),
            cfg!(debug_assertions)
        );
    }

    #[test]
    fn topology_guarantee_helper_matrix_and_policy_compatibility() {
        assert_eq!(TopologyGuarantee::default(), TopologyGuarantee::DEFAULT);
        assert_eq!(TopologyGuarantee::DEFAULT, TopologyGuarantee::PLManifold);
        assert!(!TopologyGuarantee::Pseudomanifold.requires_vertex_links_during_insertion());
        assert!(TopologyGuarantee::PLManifoldStrict.requires_vertex_links_during_insertion());
        assert!(!TopologyGuarantee::Pseudomanifold.requires_vertex_links_at_completion());
        assert!(TopologyGuarantee::PLManifold.requires_vertex_links_at_completion());
        assert!(TopologyGuarantee::PLManifoldStrict.requires_vertex_links_at_completion());
        assert!(
            TopologyGuarantee::Pseudomanifold.requires_pseudomanifold_checks_during_insertion()
        );
        assert!(TopologyGuarantee::PLManifold.requires_pseudomanifold_checks_during_insertion());
        assert!(
            TopologyGuarantee::PLManifoldStrict.requires_pseudomanifold_checks_during_insertion()
        );
        assert!(!TopologyGuarantee::Pseudomanifold.requires_ridge_links());
        assert!(TopologyGuarantee::PLManifold.requires_ridge_links());
        assert!(TopologyGuarantee::PLManifoldStrict.requires_ridge_links());

        assert_eq!(
            TopologyGuarantee::PLManifoldStrict.default_validation_policy(),
            ValidationPolicy::Always
        );
        assert_eq!(
            TopologyGuarantee::PLManifold.default_validation_policy(),
            ValidationPolicy::ExplicitOnly
        );
        assert_eq!(
            TopologyGuarantee::Pseudomanifold.default_validation_policy(),
            ValidationPolicy::OnSuspicion
        );

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_empty(FastKernel::new());
        assert_eq!(
            tri.validation_policy(),
            TopologyGuarantee::DEFAULT.default_validation_policy()
        );

        for policy in [
            ValidationPolicy::Never,
            ValidationPolicy::ExplicitOnly,
            ValidationPolicy::OnSuspicion,
            ValidationPolicy::Always,
            ValidationPolicy::DebugOnly,
        ] {
            assert!(TopologyGuarantee::Pseudomanifold.is_compatible_with_policy(policy));
        }

        assert!(!TopologyGuarantee::PLManifold.is_compatible_with_policy(ValidationPolicy::Never));
        assert!(
            !TopologyGuarantee::PLManifoldStrict.is_compatible_with_policy(ValidationPolicy::Never)
        );
        assert!(
            TopologyGuarantee::PLManifold.is_compatible_with_policy(ValidationPolicy::ExplicitOnly)
        );
        assert!(
            TopologyGuarantee::PLManifoldStrict
                .is_compatible_with_policy(ValidationPolicy::ExplicitOnly)
        );
        assert!(
            TopologyGuarantee::PLManifold.is_compatible_with_policy(ValidationPolicy::OnSuspicion)
        );
        assert!(TopologyGuarantee::PLManifold.is_compatible_with_policy(ValidationPolicy::Always));
        assert!(
            TopologyGuarantee::PLManifoldStrict.is_compatible_with_policy(ValidationPolicy::Always)
        );
    }

    macro_rules! test_validation_configuration_paths {
        ($($dim:expr),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<validation_accessors_and_mutators_round_trip_ $dim d>]() {
                        let mut tri: Triangulation<FastKernel<f64>, (), (), $dim> =
                            Triangulation::new_empty(FastKernel::new());
                        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);
                        assert_eq!(tri.validation_policy(), ValidationPolicy::ExplicitOnly);

                        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
                        tri.set_validation_policy(ValidationPolicy::Always);

                        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
                        assert_eq!(tri.validation_policy(), ValidationPolicy::Always);
                    }

                    #[test]
                    fn [<explicit_only_policy_supports_manual_pl_manifold_validation_ $dim d>]() {
                        let mut tri: Triangulation<FastKernel<f64>, (), (), $dim> =
                            Triangulation::new_empty(FastKernel::new());

                        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);
                        assert_eq!(tri.validation_policy(), ValidationPolicy::ExplicitOnly);

                        tri.try_set_validation_policy(ValidationPolicy::ExplicitOnly)
                            .unwrap();
                        assert_eq!(tri.validation_policy(), ValidationPolicy::ExplicitOnly);
                    }

                    #[test]
                    fn [<incompatible_policy_rejected_even_when_completion_validation_succeeds_ $dim d>]() {
                        let mut tri: Triangulation<FastKernel<f64>, (), (), $dim> =
                            Triangulation::new_empty(FastKernel::new());

                        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);
                        assert!(tri.validate_at_completion().is_ok());

                        assert_eq!(
                            tri.try_set_validation_policy(ValidationPolicy::Never),
                            Err(
                                ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                                    topology_guarantee: TopologyGuarantee::PLManifold,
                                    validation_policy: ValidationPolicy::Never,
                                }
                            )
                        );
                        assert_eq!(tri.validation_policy(), ValidationPolicy::ExplicitOnly);

                        tri.set_validation_policy(ValidationPolicy::Never);
                        assert_eq!(tri.validation_policy(), ValidationPolicy::ExplicitOnly);
                    }

                    #[test]
                    fn [<incompatible_guarantee_rejected_even_when_completion_validation_succeeds_ $dim d>]() {
                        let mut tri: Triangulation<FastKernel<f64>, (), (), $dim> =
                            Triangulation::new_empty(FastKernel::new());

                        tri.try_set_topology_guarantee(TopologyGuarantee::Pseudomanifold)
                            .unwrap();
                        tri.try_set_validation_policy(ValidationPolicy::Never)
                            .unwrap();
                        assert_eq!(tri.validation_policy(), ValidationPolicy::Never);
                        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
                        assert!(tri.validate_at_completion().is_ok());

                        assert_eq!(
                            tri.try_set_topology_guarantee(TopologyGuarantee::PLManifoldStrict),
                            Err(
                                ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                                    topology_guarantee: TopologyGuarantee::PLManifoldStrict,
                                    validation_policy: ValidationPolicy::Never,
                                }
                            )
                        );
                        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);

                        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);
                        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
                    }

                    #[test]
                    fn [<validate_at_completion_skips_for_pseudomanifold_ $dim d>]() {
                        let vertices = unit_simplex_vertices::<$dim>();
                        let tds =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                                .unwrap();
                        let mut tri =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
                        assert!(tri.validate_at_completion().is_ok());
                    }

                    #[test]
                    fn [<validate_at_completion_reports_invalid_vertex_link_ $dim d>]() {
                        let (tds, expected_vertex_key) = build_invalid_vertex_link_tds::<$dim>();

                        let mut tri =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);
                        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);

                        match tri.validate_at_completion() {
                            Err(InvariantError::Triangulation(
                                TriangulationValidationError::VertexLinkNotManifold { vertex_key, .. },
                            )) => assert_eq!(vertex_key, expected_vertex_key),
                            other => panic!("Expected VertexLinkNotManifold, got {other:?}"),
                        }
                    }

                    #[test]
                    fn [<incompatible_policy_rejected_when_completion_validation_fails_ $dim d>]() {
                        let (tds, _) = build_invalid_vertex_link_tds::<$dim>();
                        let mut tri =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                        assert_matches!(
                            tri.validate_at_completion(),
                            Err(InvariantError::Triangulation(
                                TriangulationValidationError::VertexLinkNotManifold { .. }
                            ))
                        );
                        assert_eq!(tri.validation_policy(), ValidationPolicy::ExplicitOnly);
                        assert_eq!(
                            tri.try_set_validation_policy(ValidationPolicy::Never),
                            Err(
                                ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                                    topology_guarantee: TopologyGuarantee::PLManifold,
                                    validation_policy: ValidationPolicy::Never,
                                }
                            )
                        );
                        tri.set_validation_policy(ValidationPolicy::Never);
                        assert_eq!(tri.validation_policy(), ValidationPolicy::ExplicitOnly);
                    }

                    #[test]
                    fn [<incompatible_guarantee_rejected_when_completion_validation_fails_ $dim d>]() {
                        let (tds, _) = build_invalid_vertex_link_tds::<$dim>();
                        let mut tri =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                        tri.try_set_topology_guarantee(TopologyGuarantee::Pseudomanifold)
                            .unwrap();
                        tri.try_set_validation_policy(ValidationPolicy::Never)
                            .unwrap();
                        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
                        assert_eq!(tri.validation_policy(), ValidationPolicy::Never);
                        assert_eq!(
                            tri.try_set_topology_guarantee(TopologyGuarantee::PLManifoldStrict),
                            Err(
                                ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                                    topology_guarantee: TopologyGuarantee::PLManifoldStrict,
                                    validation_policy: ValidationPolicy::Never,
                                }
                            )
                        );
                        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);
                        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
                    }
                )+
            }
        };
    }

    test_validation_configuration_paths!(2, 3, 4, 5);

    #[test]
    fn validate_after_insertion_skips_when_no_simplices() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        tri.set_validation_policy(ValidationPolicy::Always);
        let _ = tri
            .tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        assert_eq!(tri.number_of_simplices(), 0);

        tri.validate_after_insertion_with_scope(SuspicionFlags::default(), None)
            .unwrap();
    }

    #[test]
    fn validate_after_insertion_calls_is_valid_when_policy_triggers() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::Always);

        match tri.validate_after_insertion_with_scope(SuspicionFlags::default(), None) {
            Err(InvariantError::Triangulation(TriangulationValidationError::Disconnected {
                ..
            })) => {}
            other => panic!("Expected Disconnected error, got {other:?}"),
        }
    }

    #[test]
    fn validate_after_insertion_full_validation_includes_tds_layer() {
        let (mut tri, [v0, _, _, _], _) = build_single_tet();
        let uuid = tri.tds.vertex(v0).unwrap().uuid();
        tri.tds.uuid_to_vertex_key.remove(&uuid);
        tri.set_validation_policy(ValidationPolicy::Always);

        match tri.validate_after_insertion_with_scope(SuspicionFlags::default(), None) {
            Err(InvariantError::Tds(TdsError::MappingInconsistency { .. })) => {}
            other => panic!("Expected InvariantError::Tds(MappingInconsistency), got {other:?}"),
        }
    }

    #[test]
    fn validation_after_insertion_work_matches_policy_and_link_requirements() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
        assert_eq!(
            tri.validation_after_insertion_work(SuspicionFlags::default()),
            Some(InsertionValidationWork::RequiredTopologyLinks)
        );

        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);
        assert_eq!(
            tri.validation_after_insertion_work(SuspicionFlags::default()),
            Some(InsertionValidationWork::RequiredTopologyLinks)
        );

        tri.set_validation_policy(ValidationPolicy::Always);
        assert_eq!(
            tri.validation_after_insertion_work(SuspicionFlags::default()),
            Some(InsertionValidationWork::FullValidation)
        );
    }

    #[test]
    fn insertion_error_to_invariant_error_maps_all_arms() {
        let source = TdsError::Geometric(GeometricError::DegenerateOrientation {
            message: "det=0".to_string(),
        });
        let error = InsertionError::TopologyValidation(source.clone());
        assert_eq!(
            insertion_error_to_invariant_error(error, "ctx"),
            InvariantError::Tds(source)
        );

        let inner = TriangulationValidationError::IsolatedVertex {
            vertex_key: VertexKey::from(KeyData::from_ffi(1)),
            vertex_uuid: Uuid::nil(),
        };
        let error = InsertionError::TopologyValidationFailed {
            context: InsertionTopologyValidationContext::InvariantConversion,
            source: inner.clone(),
        };
        assert_eq!(
            insertion_error_to_invariant_error(error, "ctx"),
            InvariantError::Triangulation(inner)
        );

        let embedding_source = synthetic_embedding_error();
        let error = InsertionError::EmbeddingValidationFailed {
            source: embedding_source.clone(),
        };
        assert_eq!(
            insertion_error_to_invariant_error(error, "ctx"),
            InvariantError::Embedding(embedding_source)
        );

        let delaunay_source = synthetic_delaunay_verification_error("delaunay");
        let error = InsertionError::DelaunayValidationFailed {
            source: delaunay_source.clone(),
        };
        assert_eq!(
            insertion_error_to_invariant_error(error, "ctx"),
            InvariantError::Delaunay(delaunay_source)
        );

        let error = InsertionError::CavityFilling {
            reason: CavityFillingError::EmptyFanTriangulation,
        };
        let result = insertion_error_to_invariant_error(error, "ctx");
        assert!(
            matches!(
                result,
                InvariantError::Tds(TdsError::InconsistentDataStructure { ref message })
                    if message.contains("ctx") && message.contains("fan triangulation produced no simplices")
            ),
            "CavityFilling should wrap to InconsistentDataStructure: {result:?}"
        );
    }

    #[test]
    fn invariant_error_to_insertion_error_maps_all_arms() {
        let inv = InvariantError::Tds(TdsError::InconsistentDataStructure {
            message: "test".to_string(),
        });
        let ins =
            Triangulation::<FastKernel<f64>, (), (), 3>::invariant_error_to_insertion_error(inv);
        assert_matches!(ins, InsertionError::TopologyValidation(_));

        let inv = InvariantError::Triangulation(TriangulationValidationError::IsolatedVertex {
            vertex_key: VertexKey::from(KeyData::from_ffi(1)),
            vertex_uuid: Uuid::nil(),
        });
        let ins =
            Triangulation::<FastKernel<f64>, (), (), 3>::invariant_error_to_insertion_error(inv);
        assert_matches!(ins, InsertionError::TopologyValidationFailed { .. });

        let inv = InvariantError::Embedding(synthetic_embedding_error());
        let ins =
            Triangulation::<FastKernel<f64>, (), (), 3>::invariant_error_to_insertion_error(inv);
        assert_matches!(ins, InsertionError::EmbeddingValidationFailed { .. });

        let inv = InvariantError::Delaunay(synthetic_delaunay_verification_error("test"));
        let ins =
            Triangulation::<FastKernel<f64>, (), (), 3>::invariant_error_to_insertion_error(inv);
        assert_matches!(ins, InsertionError::DelaunayValidationFailed { .. });
    }

    #[test]
    fn from_manifold_error_routes_tds_and_topology_layers() {
        let tds_err = TdsError::InconsistentDataStructure {
            message: "underlying TDS issue".to_string(),
        };
        let manifold_err = ManifoldError::Tds(tds_err.clone());
        assert_eq!(
            InvariantError::from(manifold_err),
            InvariantError::Tds(tds_err)
        );

        let err = ManifoldError::ManifoldFacetMultiplicity {
            facet_key: 999,
            simplex_count: 5,
        };
        let inv = InvariantError::from(err);
        assert_matches!(
            inv,
            InvariantError::Triangulation(
                TriangulationValidationError::ManifoldFacetMultiplicity {
                    facet_key: 999,
                    simplex_count: 5
                }
            )
        );

        let ridge_vertex = VertexKey::from(KeyData::from_ffi(42));
        let inv = InvariantError::from(ManifoldError::RidgeNotFound {
            ridge_vertices: iter::once(ridge_vertex).collect(),
        });
        assert_matches!(
            inv,
            InvariantError::Triangulation(TriangulationValidationError::RidgeNotFound {
                ridge_vertices
            }) if ridge_vertices.as_slice() == [ridge_vertex]
        );
    }

    #[test]
    fn isolated_vertex_error_display_is_informative() {
        let err = TriangulationValidationError::IsolatedVertex {
            vertex_key: VertexKey::from(KeyData::from_ffi(42)),
            vertex_uuid: Uuid::nil(),
        };
        let msg = err.to_string();
        assert!(msg.contains("Isolated vertex"));
        assert!(msg.contains("not incident to any simplex"));
    }

    #[test]
    fn is_valid_returns_triangulation_error_for_isolated_vertex() {
        let (mut tri, _, _) = build_single_tet();
        let iso = tri
            .tds
            .insert_vertex_with_mapping(test_vertex([0.5, 0.5, 0.5]))
            .unwrap();

        match tri.is_valid_topology() {
            Err(InvariantError::Triangulation(TriangulationValidationError::IsolatedVertex {
                vertex_key,
                ..
            })) => assert_eq!(vertex_key, iso),
            other => {
                panic!("Expected InvariantError::Triangulation(IsolatedVertex), got {other:?}")
            }
        }
    }

    #[test]
    fn is_valid_returns_triangulation_error_for_disconnected() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        match tri.is_valid_topology() {
            Err(InvariantError::Triangulation(TriangulationValidationError::Disconnected {
                simplex_count,
            })) => assert_eq!(simplex_count, 2),
            other => {
                panic!("Expected InvariantError::Triangulation(Disconnected), got {other:?}")
            }
        }
    }

    #[test]
    fn validate_returns_invariant_error_from_tds_layer() {
        let (mut tri, [v0, _, _, _], _) = build_single_tet();
        let uuid = tri.tds.vertex(v0).unwrap().uuid();
        tri.tds.uuid_to_vertex_key.remove(&uuid);

        match tri.validate() {
            Err(InvariantError::Tds(TdsError::MappingInconsistency { .. })) => {}
            other => panic!("Expected InvariantError::Tds(MappingInconsistency), got {other:?}"),
        }
    }

    #[test]
    fn validate_returns_invariant_error_from_topology_layer() {
        let (mut tri, _, _) = build_single_tet();
        let _ = tri
            .tds
            .insert_vertex_with_mapping(test_vertex([0.5, 0.5, 0.5]))
            .unwrap();

        match tri.validate() {
            Err(InvariantError::Triangulation(TriangulationValidationError::IsolatedVertex {
                ..
            })) => {}
            other => {
                panic!("Expected InvariantError::Triangulation(IsolatedVertex), got {other:?}")
            }
        }
    }

    #[test]
    fn validation_report_ok_for_valid_triangulation() {
        let vertices = [
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
            test_vertex([0.5, 0.5, 0.5]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        assert!(dt.as_triangulation().validation_report().is_ok());
    }

    #[test]
    fn validation_report_reports_isolated_vertex_topology_violation() {
        let (mut tri, _, _) = build_single_tet();
        let _ = tri
            .tds
            .insert_vertex_with_mapping(test_vertex([0.5, 0.5, 0.5]))
            .unwrap();

        let report = tri.validation_report().unwrap_err();
        assert!(!report.is_empty());
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::Topology),
            "Expected Topology violation in report"
        );
    }

    #[test]
    fn validate_global_connectedness_ok_for_connected() {
        let (tri, _, _) = build_single_tet();
        assert!(tri.validate_global_connectedness().is_ok());
    }

    #[test]
    fn validate_no_isolated_vertices_ok_when_no_vertices() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());
        assert!(tri.validate_no_isolated_vertices().is_ok());
    }

    macro_rules! test_is_valid_topology {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<is_valid_topology_ $dim d>]() {
                    let vertices: Vec<Vertex<(), $dim>> = vec![
                        $(test_vertex($simplex_coords)),+
                    ];

                    let expected_vertices = vertices.len();
                    assert_eq!(expected_vertices, $dim + 1);

                    let dt = DelaunayTriangulation::try_new(&vertices)
                        .expect("simplex construction should succeed");
                    let tri = dt.as_triangulation();

                    assert!(tri.is_valid_topology().is_ok());
                    assert_eq!(tri.number_of_vertices(), expected_vertices);
                    assert_eq!(tri.number_of_simplices(), 1);
                }
            }
        };
    }

    test_is_valid_topology!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    test_is_valid_topology!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
    test_is_valid_topology!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    test_is_valid_topology!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    #[test]
    fn is_valid_topology_empty() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        assert!(tri.is_valid_topology().is_ok());
    }

    #[test]
    fn is_valid_pl_manifold_mode_rejects_wedge_at_vertex_in_2d() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 1.0]))
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        let v4 = tds
            .insert_vertex_with_mapping(test_vertex([10.0, 10.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(test_vertex([11.0, 10.0]))
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(test_vertex([10.0, 11.0]))
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v4, v5], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v4, v6], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v5, v6], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v4, v5, v6], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        assert_matches!(
            tri.is_valid_topology(),
            Err(InvariantError::Triangulation(
                TriangulationValidationError::Disconnected { .. }
            ))
        );

        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

        match tri.is_valid_topology() {
            Err(InvariantError::Triangulation(
                TriangulationValidationError::VertexLinkNotManifold { vertex_key, .. },
            )) => assert_eq!(vertex_key, v0),
            Err(InvariantError::Triangulation(
                TriangulationValidationError::RidgeLinkNotManifold { .. }
                | TriangulationValidationError::Disconnected { .. },
            )) => {}
            other => panic!(
                "Expected RidgeLinkNotManifold, VertexLinkNotManifold, or Disconnected, got {other:?}"
            ),
        }
    }

    #[test]
    fn is_valid_pl_manifold_mode_rejects_cone_on_torus_in_3d_even_when_connected() {
        const N: usize = 3;
        const M: usize = 3;

        let mut tds: Tds<(), (), 3> = Tds::empty();
        let mut v: [[VertexKey; M]; N] = [[VertexKey::from(KeyData::from_ffi(0)); M]; N];
        for (i, row) in v.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                let i_f = f64::from(u32::try_from(i).unwrap());
                let j_f = f64::from(u32::try_from(j).unwrap());
                *slot = tds
                    .insert_vertex_with_mapping(test_vertex([i_f, j_f, 0.0]))
                    .unwrap();
            }
        }

        let apex = tds
            .insert_vertex_with_mapping(test_vertex([0.5, 0.5, 1.0]))
            .unwrap();

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

        repair_neighbor_pointers(&mut tds).unwrap();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);
        tri.validate_global_connectedness().unwrap();
        let facet_to_simplices = tri.tds.build_facet_to_simplices_map().unwrap();
        let facet_to_simplices =
            ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices).unwrap();
        validate_closed_boundary_from_validated_facet_map(
            &tri.tds,
            facet_to_simplices,
            GlobalTopology::Euclidean,
        )
        .unwrap();

        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

        match tri.is_valid_topology() {
            Err(InvariantError::Triangulation(
                TriangulationValidationError::VertexLinkNotManifold {
                    vertex_key,
                    connected,
                    interior_vertex,
                    ..
                },
            )) => {
                assert_eq!(vertex_key, apex);
                assert!(connected);
                assert!(interior_vertex);
            }
            other => panic!("Expected VertexLinkNotManifold for cone apex, got {other:?}"),
        }
    }

    #[test]
    fn is_valid_disconnected_detected_before_non_manifold_boundary_ridge() {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let shared_edge_v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let shared_edge_v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let tet1_v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let tet1_v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();
        let tet2_v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, -1.0, 0.0]))
            .unwrap();
        let tet2_v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, -1.0]))
            .unwrap();

        let _ = tds
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

        let tri = Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        match tri.is_valid_topology() {
            Err(InvariantError::Triangulation(TriangulationValidationError::Disconnected {
                simplex_count,
            })) => assert_eq!(simplex_count, 2),
            other => panic!("Expected Disconnected, got {other:?}"),
        }
    }

    #[test]
    fn validate_includes_tds_validation() {
        let vertices = vec![
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        assert!(tri.tds.validate().is_ok());
        assert!(tri.validate().is_ok());
    }

    #[test]
    fn is_valid_rejects_bootstrap_phase_with_isolated_vertex() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());
        let vertex = test_vertex([0.0, 0.0, 0.0]);
        let expected_uuid = vertex.uuid();
        let expected_vk = tri.tds.insert_vertex_with_mapping(vertex).unwrap();

        match tri.is_valid_topology() {
            Err(InvariantError::Triangulation(TriangulationValidationError::IsolatedVertex {
                vertex_key,
                vertex_uuid,
            })) => {
                assert_eq!(vertex_key, expected_vk);
                assert_eq!(vertex_uuid, expected_uuid);
            }
            other => panic!("Expected IsolatedVertex error, got {other:?}"),
        }
    }

    #[test]
    fn is_valid_rejects_isolated_vertex_even_when_simplices_exist() {
        let vertices = vec![
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        let _isolated_vk = tri
            .tds
            .insert_vertex_with_mapping(test_vertex([10.0, 10.0, 10.0]))
            .unwrap();

        assert_matches!(
            tri.is_valid_topology(),
            Err(InvariantError::Triangulation(
                TriangulationValidationError::IsolatedVertex { .. }
            ))
        );
    }

    #[test]
    fn is_valid_rejects_disconnected_even_when_euler_matches() {
        let mut tds: Tds<(), (), 1> = Tds::empty();

        let v0 = tds.insert_vertex_with_mapping(test_vertex([0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(test_vertex([1.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(test_vertex([2.0])).unwrap();
        let e0 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v0, v1], None).unwrap())
            .unwrap();
        let e1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v1, v2], None).unwrap())
            .unwrap();

        let v3 = tds.insert_vertex_with_mapping(test_vertex([10.0])).unwrap();
        let v4 = tds.insert_vertex_with_mapping(test_vertex([11.0])).unwrap();
        let v5 = tds.insert_vertex_with_mapping(test_vertex([12.0])).unwrap();
        let c0 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v3, v4], None).unwrap())
            .unwrap();
        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v4, v5], None).unwrap())
            .unwrap();
        let c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v5, v3], None).unwrap())
            .unwrap();

        for (simplex_key, neighbor_keys) in [
            (e0, [Some(e1), None]),
            (e1, [None, Some(e0)]),
            (c0, [Some(c1), Some(c2)]),
            (c1, [Some(c2), Some(c0)]),
            (c2, [Some(c0), Some(c1)]),
        ] {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
            neighbors.extend(neighbor_keys);
            simplex.set_neighbors_from_keys(neighbors).unwrap();
        }

        tds.assign_incident_simplices().unwrap();

        let tri = Triangulation::<FastKernel<f64>, (), (), 1>::new_with_tds(FastKernel::new(), tds);
        let facet_to_simplices = tri.tds.build_facet_to_simplices_map().unwrap();
        let facet_to_simplices =
            ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices).unwrap();

        let topology = validate_triangulation_euler_from_validated_facet_map(
            &tri.tds,
            facet_to_simplices,
            GlobalTopology::Euclidean,
        )
        .unwrap();
        assert_eq!(topology.classification, TopologyClassification::Ball(1));
        assert_eq!(topology.expected, Some(1));
        assert_eq!(topology.chi, 1);

        match tri.is_valid_topology() {
            Err(InvariantError::Triangulation(TriangulationValidationError::Disconnected {
                simplex_count,
            })) => assert_eq!(simplex_count, 5),
            other => panic!("Expected Disconnected, got {other:?}"),
        }
    }

    #[test]
    fn tds_is_valid_rejects_boundary_facet_has_neighbor() {
        let vertices_simplex_1 = vec![
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices_simplex_1)
                .unwrap();
        let first_simplex_key = tds.simplex_keys().next().unwrap();

        let v4 = tds
            .insert_vertex_with_mapping(test_vertex([10.0, 0.0, 0.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(test_vertex([11.0, 0.0, 0.0]))
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(test_vertex([10.0, 1.0, 0.0]))
            .unwrap();
        let v7 = tds
            .insert_vertex_with_mapping(test_vertex([10.0, 0.0, 1.0]))
            .unwrap();

        let second_simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v4, v5, v6, v7], None).unwrap(),
            )
            .unwrap();

        let first_simplex = tds.simplex_mut(first_simplex_key).unwrap();
        let mut neighbors = NeighborBuffer::<Option<SimplexKey>>::new();
        neighbors.resize(4, None);
        neighbors[0] = Some(second_simplex_key);
        first_simplex.set_neighbors_from_keys(neighbors).unwrap();

        match tds.is_valid() {
            Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::BoundaryFacetHasNeighbor { neighbor_key, .. },
            }) => assert_eq!(neighbor_key, second_simplex_key),
            other => panic!("Expected InvalidNeighbors, got {other:?}"),
        }
    }

    #[test]
    fn tds_is_valid_rejects_interior_facet_neighbor_mismatch() {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 2.0]))
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

        assert_matches!(
            tds.is_valid(),
            Err(TdsError::InvalidNeighbors {
                reason: NeighborValidationError::InteriorFacetNeighborMismatch { .. },
            })
        );
    }

    #[test]
    fn is_valid_non_manifold_facet_multiplicity() {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 2.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 3.0]))
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

        let tri = Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        match tri.is_valid_topology() {
            Err(InvariantError::Triangulation(TriangulationValidationError::Disconnected {
                ..
            })) => {}
            Err(InvariantError::Triangulation(
                TriangulationValidationError::ManifoldFacetMultiplicity { simplex_count, .. },
            )) => assert_eq!(simplex_count, 3),
            other => panic!("Expected Disconnected or ManifoldFacetMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn validation_report_returns_mapping_failures_only() {
        let vertices = vec![
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        let uuid = tri.tds.vertices().next().unwrap().1.uuid();
        tri.tds.uuid_to_vertex_key.remove(&uuid);

        let report = tri.validation_report().unwrap_err();
        assert!(!report.violations.is_empty());
        assert!(report.violations.iter().all(|v| {
            matches!(
                v.kind,
                InvariantKind::VertexMappings | InvariantKind::SimplexMappings
            )
        }));
    }

    #[test]
    fn validation_report_includes_vertex_and_simplex_validity() {
        let vertices = vec![
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        let invalid_vertex: Vertex<(), 3> = Vertex::from_validated_point_with_uuid(
            Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates"),
            Uuid::nil(),
            None,
        );
        let _ = tri.tds.insert_vertex_with_mapping(invalid_vertex).unwrap();

        let simplex_key = tri.tds.simplex_keys().next().unwrap();
        let simplex = tri.tds.simplex_mut(simplex_key).unwrap();
        simplex.ensure_neighbors_buffer_mut().truncate(3);

        let report = tri.validation_report().unwrap_err();
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::VertexValidity)
        );
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::SimplexValidity)
        );
    }

    #[test]
    fn validate_after_insertion_required_checks_do_not_run_global_connectedness() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        assert!(tri.is_valid_topology().is_err());
        tri.validate_after_insertion_with_scope(SuspicionFlags::default(), None)
            .unwrap();
    }

    #[test]
    fn validate_after_insertion_does_not_skip_pseudomanifold_checks() {
        let tds = build_three_triangles_sharing_edge_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        match tri.validate_after_insertion_with_scope(SuspicionFlags::default(), None) {
            Err(InvariantError::Triangulation(
                TriangulationValidationError::ManifoldFacetMultiplicity { simplex_count, .. },
            )) => assert_eq!(simplex_count, 3),
            other => panic!("Expected ManifoldFacetMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn scoped_validation_catches_touched_over_shared_facet() {
        let tds = build_three_triangles_sharing_edge_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let scope: SimplexKeyBuffer = tri.tds.simplex_keys().take(1).collect();

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);

        match tri.validate_after_insertion_with_scope(SuspicionFlags::default(), Some(&scope)) {
            Err(InvariantError::Triangulation(
                TriangulationValidationError::ManifoldFacetMultiplicity { simplex_count, .. },
            )) => assert_eq!(simplex_count, 3),
            other => panic!("Expected ManifoldFacetMultiplicity, got {other:?}"),
        }
    }

    macro_rules! test_scoped_strict_validation_falls_back_to_global_vertex_links {
        ($($dim:expr),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<scoped_strict_validation_falls_back_to_global_vertex_links_ $dim d>]() {
                        let (tds, expected_vertex_key) = build_invalid_vertex_link_tds::<$dim>();
                        let mut tri =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);
                        let scope: SimplexKeyBuffer = tri.tds.simplex_keys().take(1).collect();
                        assert!(!scope.is_empty());

                        tri.validation_policy = ValidationPolicy::OnSuspicion;
                        tri.topology_guarantee = TopologyGuarantee::PLManifoldStrict;

                        match tri.validate_after_insertion_with_scope(SuspicionFlags::default(), Some(&scope)) {
                            Err(InvariantError::Triangulation(
                                TriangulationValidationError::RidgeLinkNotManifold {
                                    connected: false,
                                    ..
                                },
                            )) if $dim == 2 => {}
                            Err(InvariantError::Triangulation(
                                TriangulationValidationError::VertexLinkNotManifold { vertex_key, .. },
                            )) => assert_eq!(vertex_key, expected_vertex_key),
                            other => panic!("Expected VertexLinkNotManifold, got {other:?}"),
                        }
                    }
                )+
            }
        };
    }

    test_scoped_strict_validation_falls_back_to_global_vertex_links!(2, 3, 4, 5);

    macro_rules! test_insertion_scoped_validation_preserves_full_validity {
        ($($dim:expr),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<insertion_scoped_validation_preserves_full_validity_ $dim d>]() {
                        let vertices = unit_simplex_vertices::<$dim>();
                        let tds =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                                .unwrap();
                        let mut tri =
                            Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
                        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

                        let detail = tri
                            .insert_with_statistics_seeded_indexed_detailed(
                                unit_simplex_interior_vertex::<$dim>(),
                                None,
                                None,
                                0,
                                None,
                                None,
                            )
                            .unwrap();

                        assert!(!detail.repair_seed_simplices.is_empty());
                        tri.validate_after_insertion_with_scope(
                            SuspicionFlags::default(),
                            Some(&detail.repair_seed_simplices),
                        )
                        .unwrap();
                        tri.is_valid_topology().unwrap();
                    }
                )+
            }
        };
    }

    test_insertion_scoped_validation_preserves_full_validity!(2, 3, 4, 5);

    #[test]
    fn validation_report_collects_multiple_violations() {
        let (mut tri, _, ck) = build_single_tet();

        let _ = tri
            .tds
            .insert_vertex_with_mapping(test_vertex([5.0, 5.0, 5.0]))
            .unwrap();

        let simplex = tri.tds.simplex_mut(ck).unwrap();
        simplex.ensure_neighbors_buffer_mut().truncate(2);

        let report = tri.validation_report().unwrap_err();
        assert!(
            report.violations.len() >= 2,
            "Expected at least 2 violations, got {}",
            report.violations.len()
        );
    }

    #[test]
    fn validate_connectedness_rejects_empty_new_simplices() {
        let (tri, _, _) = build_single_tet();

        let empty: SimplexKeyBuffer = SimplexKeyBuffer::new();
        let err = tri.validate_connectedness(&empty).unwrap_err();
        assert_matches!(err, InsertionError::TopologyValidation(_));
    }

    #[test]
    fn validate_connectedness_passes_for_valid_new_simplices() {
        let (tri, _, ck) = build_single_tet();

        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.push(ck);
        assert!(tri.validate_connectedness(&new_simplices).is_ok());
    }

    #[test]
    fn validate_after_insertion_ok_for_valid_simplex() {
        let vertices = [
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let tri = dt.as_triangulation();
        let suspicion = SuspicionFlags {
            repair_loop_entered: true,
            ..Default::default()
        };

        assert!(
            tri.validate_after_insertion_with_scope(suspicion, None)
                .is_ok()
        );
    }

    #[test]
    fn validate_after_insertion_rejects_local_degenerate_simplex_embedding() {
        let (tri, ck) = build_degenerate_tet();
        let mut scope = SimplexKeyBuffer::new();
        scope.push(ck);

        let err = tri
            .validate_after_insertion_with_scope(SuspicionFlags::default(), Some(&scope))
            .unwrap_err();

        assert_matches!(
            err,
            InvariantError::Embedding(
                TriangulationEmbeddingValidationError::DegenerateSimplex {
                    simplex_key,
                    dimension: 3,
                    ..
                }
            ) if simplex_key == ck
        );
    }

    #[test]
    fn validate_after_insertion_full_validation_rejects_local_degenerate_simplex_embedding() {
        let (mut tri, ck) = build_degenerate_tet();
        tri.set_validation_policy(ValidationPolicy::Always);
        let mut scope = SimplexKeyBuffer::new();
        scope.push(ck);

        let err = tri
            .validate_after_insertion_with_scope(SuspicionFlags::default(), Some(&scope))
            .unwrap_err();

        assert_matches!(
            err,
            InvariantError::Embedding(
                TriangulationEmbeddingValidationError::DegenerateSimplex {
                    simplex_key,
                    dimension: 3,
                    ..
                }
            ) if simplex_key == ck
        );
    }

    #[test]
    fn validate_after_insertion_full_validation_rejects_global_embedding_intersection() {
        let (tds, scope) = build_topologically_valid_self_overlapping_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        tri.set_validation_policy(ValidationPolicy::Always);

        tri.is_valid_topology()
            .expect("fixture should isolate a Level 4 embedding failure");

        let err = tri
            .validate_after_insertion_with_scope(SuspicionFlags::default(), Some(&scope[..1]))
            .unwrap_err();

        assert_matches!(
            err,
            InvariantError::Embedding(
                TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace { .. }
            )
        );
    }

    #[test]
    fn validate_after_insertion_full_validation_checks_large_raw_embedding_scope() {
        let (tds, scope) = build_topologically_valid_self_overlapping_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        tri.set_validation_policy(ValidationPolicy::Always);

        tri.is_valid_topology()
            .expect("fixture should isolate a Level 4 embedding failure");

        let mut large_scope = SimplexKeyBuffer::new();
        for _ in 0..9 {
            large_scope.push(scope[0]);
        }

        let err = tri
            .validate_after_insertion_with_scope(SuspicionFlags::default(), Some(&large_scope))
            .unwrap_err();

        assert_matches!(
            err,
            InvariantError::Embedding(
                TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace { .. }
            )
        );
    }

    #[test]
    fn validate_at_completion_ok_for_pseudomanifold_empty() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        assert!(tri.validate_at_completion().is_ok());
    }

    #[test]
    fn validate_at_completion_ok_for_pl_manifold_no_simplices() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        assert!(tri.validate_at_completion().is_ok());
    }

    #[test]
    fn pl_manifold_insertion_keeps_valid_topology_with_explicit_only_validation() {
        let bounds = CoordinateRange::try_new(-100.0_f64, 100.0).unwrap();
        let points = generate_random_points_in_range_seeded::<3>(25, bounds, 123)
            .expect("validated range should generate finite points");

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

        dt.try_set_validation_policy(ValidationPolicy::ExplicitOnly)
            .unwrap();
        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

        for (i, point) in points.into_iter().enumerate() {
            let vertex = Vertex::from_validated_point(point, None);
            let (outcome, stats) = dt
                .insert_best_effort_with_statistics(vertex)
                .unwrap_or_else(|err| panic!("Non-retryable insertion error at i={i}: {err:?}"));

            if dt.number_of_simplices() > 0
                && let Err(err) = dt.as_triangulation().validate()
            {
                panic!(
                    "Topology invalid after insertion i={i} (outcome={outcome:?}, attempts={}, used_perturbation={}): {err}",
                    stats.attempts,
                    stats.used_perturbation()
                );
            }
        }
    }

    #[test]
    fn required_topology_validation_records_telemetry() {
        let vertices = vec![
            test_vertex([0.0, 0.0]),
            test_vertex([1.0, 0.0]),
            test_vertex([0.0, 1.0]),
        ];

        for guarantee in [
            TopologyGuarantee::Pseudomanifold,
            TopologyGuarantee::PLManifold,
        ] {
            let tds = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices)
                .unwrap();
            let mut tri =
                Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
            tri.set_validation_policy(ValidationPolicy::OnSuspicion);
            tri.set_topology_guarantee(guarantee);

            let hint = tri.simplices().next().map(|(simplex_key, _)| simplex_key);
            let detail = tri
                .insert_with_statistics_seeded_indexed_detailed(
                    test_vertex([0.25, 0.25]),
                    None,
                    hint,
                    0,
                    None,
                    None,
                )
                .unwrap();

            assert_matches!(detail.outcome, InsertionOutcome::Inserted { .. });
            assert!(
                detail.telemetry.topology_validation_calls > 0,
                "{guarantee:?} insertion should record required topology validation"
            );
            assert_eq!(
                detail.telemetry.topology_validation_nanos, 0,
                "default detailed insertion should not start validation timers"
            );
        }
    }
}

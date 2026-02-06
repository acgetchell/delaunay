//! Generic triangulation combining kernel and combinatorial data structure.
//!
//! Following CGAL's architecture, the `Triangulation` struct combines:
//! - A geometric `Kernel` for predicates
//! - A purely combinatorial `Tds` for topology
//!
//! This layer provides geometric operations while delegating topology to Tds.
//!
//! # Validation Hierarchy
//!
//! The library provides **four levels** of validation, each building on the previous:
//!
//! ## Level 1: Element Validity
//! - **Methods**: [`Cell::is_valid()`], [`Vertex::is_valid()`]
//! - **Checks**: Basic data integrity (coordinate validity, UUID presence, proper initialization)
//! - **Cost**: O(1) per element
//!
//! ## Level 2: TDS Structural Validity
//! - **Method**: [`Tds::is_valid()`]
//! - **Checks**:
//!   - UUID ↔ Key mapping consistency
//!   - No duplicate cells (same vertex sets)
//!   - Facet sharing invariant (≤2 cells per facet)
//!   - Neighbor consistency (mutual relationships)
//! - **Cost**: O(N×D²) where N = cells, D = dimension
//!
//! Use [`Tds::validate()`] for cumulative Levels 1–2 (element + structural) validation.
//!
//! ## Level 3: Manifold Topology
//! - **Method**: [`Triangulation::is_valid()`](crate::core::triangulation::Triangulation::is_valid)
//! - **Checks**:
//!   - **Codimension-1 manifoldness**: exactly 1 boundary cell or 2 interior cells per facet
//!   - **Codimension-2 boundary manifoldness**: the boundary is closed ("no boundary of boundary")
//!   - Connectedness (single connected component in the cell neighbor graph)
//!   - No isolated vertices (every vertex must be incident to at least one cell)
//!   - Euler characteristic (χ = V - E + F - C matches expected topology)
//! - **Cost**: O(N×D²) dominated by simplex counting
//!
//! Use [`Triangulation::validate()`](crate::core::triangulation::Triangulation::validate) for cumulative Levels 1–3.
//!
//! ## Level 4: Delaunay Property
//! - **Method**: [`DelaunayTriangulation::is_valid()`](crate::core::delaunay_triangulation::DelaunayTriangulation::is_valid)
//! - **Checks**: Empty circumsphere property (no vertex inside any cell's circumsphere)
//! - **Cost**: O(N×V) where N = cells, V = vertices
//!
//! Use [`DelaunayTriangulation::validate()`](crate::core::delaunay_triangulation::DelaunayTriangulation::validate) for cumulative Levels 1–4.
//!
//! ## Usage Guidelines
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let dt = DelaunayTriangulation::new(&vertices).unwrap();
//!
//! // Level 2: structural only (fast)
//! assert!(dt.tds().is_valid().is_ok());
//!
//! // Level 3: topology only (assumes structural validity)
//! assert!(dt.as_triangulation().is_valid().is_ok());
//!
//! // Level 4: Delaunay property only (assumes Levels 1–3)
//! assert!(dt.is_valid().is_ok());
//!
//! // Full cumulative validation (Levels 1–4)
//! assert!(dt.validate().is_ok());
//! ```
//!
//! **Performance**: Use Level 2 for most production validation. Reserve Level 3 for
//! tests/debug builds, and Level 4 for critical verification or debugging geometric issues.
//!
//! [`Cell::is_valid()`]: crate::core::cell::Cell::is_valid
//! [`Vertex::is_valid()`]: crate::core::vertex::Vertex::is_valid
//! [`Tds::is_valid()`]: crate::core::triangulation_data_structure::Tds::is_valid
//! [`Tds::validate()`]: crate::core::triangulation_data_structure::Tds::validate
//!
//! ## Topology guarantees
//!
//! [`TopologyGuarantee`](crate::core::triangulation::TopologyGuarantee) selects which **manifoldness**
//! invariants are checked by Level 3 topology validation.
//!
//! Whether these checks run automatically after insertion is controlled by
//! [`ValidationPolicy`](crate::core::triangulation::ValidationPolicy).
//!
//! Level 3 validation always checks:
//! - Codimension-1 facet degree (pseudomanifold condition: 1 boundary or 2 interior cells per facet)
//! - Codimension-2 boundary manifoldness (closed boundary: "no boundary of boundary")
//! - Connectedness (single connected component in the cell neighbor graph)
//! - No isolated vertices (every vertex must be incident to at least one cell)
//! - Euler characteristic
//!
//! With [`TopologyGuarantee::PLManifold`](crate::core::triangulation::TopologyGuarantee::PLManifold),
//! Level 3 validation additionally checks the canonical **vertex-link** PL-manifoldness
//! condition via [`crate::topology::manifold::validate_vertex_links`].
//!
//! Note: for **D=3**, the current vertex-link validator additionally enforces that each link
//! has the Euler characteristic / boundary component counts of a sphere/ball (S²/B²).
//! For **D≥4**, it currently checks that each vertex link is a connected (D−1)-manifold
//! with the correct boundary behavior (a necessary condition), but does not attempt to
//! distinguish spheres/balls from other manifolds (not sufficient in general).
//!
use core::iter::Sum;
use core::ops::{AddAssign, Div, SubAssign};
use std::borrow::Cow;
use std::cmp::Ordering as CmpOrdering;
use std::hash::{Hash, Hasher};
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};

use num_traits::{NumCast, One, Zero};
use thiserror::Error;
use uuid::Uuid;

use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
use crate::core::algorithms::incremental_insertion::{
    HullExtensionReason, InsertionError, extend_hull, fill_cavity, repair_neighbor_pointers,
    wire_cavity_neighbors,
};
#[cfg(debug_assertions)]
use crate::core::algorithms::locate::locate_with_stats;
use crate::core::algorithms::locate::{
    ConflictError, LocateResult, extract_cavity_boundary, find_conflict_region, locate,
    locate_by_scan,
};
use crate::core::cell::{Cell, CellValidationError};
use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::collections::{
    CavityBoundaryBuffer, CellKeyBuffer, CellKeySet, FacetIssuesMap, FacetToCellsMap, FastHashMap,
    FastHashSet, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer, VertexToCellsMap,
    fast_hash_map_with_capacity, fast_hash_set_with_capacity,
};
use crate::core::edge::EdgeKey;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter, FacetHandle, facet_key_from_vertices};
use crate::core::operations::{
    InsertionOutcome, InsertionResult, InsertionStatistics, SuspicionFlags,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{
    CellKey, InvariantError, InvariantKind, InvariantViolation, Tds, TdsConstructionError,
    TdsMutationError, TdsValidationError, TriangulationValidationReport, VertexKey,
};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::quality::radius_ratio;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar};
use crate::geometry::util::safe_scalar_to_f64;
use crate::topology::characteristics::euler::TopologyClassification;
use crate::topology::characteristics::validation::validate_triangulation_euler_with_facet_to_cells_map;
use crate::topology::manifold::{
    ManifoldError, validate_closed_boundary, validate_facet_degree, validate_ridge_links,
    validate_vertex_links,
};
use crate::topology::traits::topological_space::TopologyError;

/// Maximum number of repair iterations for fixing non-manifold topology after insertion.
///
/// This limit prevents infinite loops in the rare case where repair cannot make progress.
/// In practice, most insertions require 0-2 iterations to restore manifold topology.
const MAX_REPAIR_ITERATIONS: usize = 10;

/// Telemetry: counts how often the topology safety-net recovered from a Level 3 validation
/// failure by retrying insertion with a star-split of the containing cell.
///
/// This is a process-wide counter across all triangulation instances.
///
/// This counter is intentionally lightweight and can be polled by production workloads
/// to see whether this recovery path is frequently used.
static TOPOLOGY_SAFETY_NET_STAR_SPLIT_FALLBACK_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_DETECTION_TOTAL: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_DETECTION_GRID_USED: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_DETECTION_GRID_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_DETECTION_GRID_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_DETECTION_ENABLED: OnceLock<bool> = OnceLock::new();

#[cfg(test)]
static DUPLICATE_DETECTION_FORCE_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(debug_assertions)]
static VERTEX_TO_CELLS_SPILL_EVENTS: AtomicU64 = AtomicU64::new(0);

fn duplicate_detection_metrics_enabled() -> bool {
    #[cfg(test)]
    if DUPLICATE_DETECTION_FORCE_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    *DUPLICATE_DETECTION_ENABLED
        .get_or_init(|| std::env::var_os("DELAUNAY_DUPLICATE_METRICS").is_some())
}

/// Telemetry counters for duplicate-coordinate detection.
#[derive(Debug, Clone, Copy, Default)]
pub struct DuplicateDetectionMetrics {
    /// Total number of duplicate-coordinate checks executed.
    pub total_checks: u64,
    /// Number of checks that successfully used the hash grid.
    pub grid_used: u64,
    /// Number of checks that fell back to a non-grid scan.
    pub grid_fallbacks: u64,
    /// Total candidate vertices inspected during grid-based checks.
    pub grid_candidates: u64,
}

pub(crate) fn record_duplicate_detection_metrics(
    used_grid: bool,
    candidate_count: usize,
    fell_back: bool,
) {
    if !duplicate_detection_metrics_enabled() {
        return;
    }
    DUPLICATE_DETECTION_TOTAL.fetch_add(1, Ordering::Relaxed);
    if used_grid {
        DUPLICATE_DETECTION_GRID_USED.fetch_add(1, Ordering::Relaxed);
        DUPLICATE_DETECTION_GRID_CANDIDATES.fetch_add(candidate_count as u64, Ordering::Relaxed);
    }
    if fell_back {
        DUPLICATE_DETECTION_GRID_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Errors that can occur during triangulation construction.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::triangulation::{Triangulation, TriangulationConstructionError};
/// use delaunay::geometry::kernel::FastKernel;
/// use delaunay::vertex;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.0, 1.0]),
/// ];
/// let result: Result<_, TriangulationConstructionError> =
///     Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);
/// assert!(result.is_ok());
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TriangulationConstructionError {
    /// Lower-layer construction error in the TDS.
    #[error(transparent)]
    Tds(#[from] TdsConstructionError),

    /// Failed to create a cell during triangulation construction.
    #[error("Failed to create cell during construction: {message}")]
    FailedToCreateCell {
        /// Description of the cell creation failure.
        message: String,
    },

    /// Insufficient vertices to create a triangulation.
    #[error("Insufficient vertices for {dimension}D triangulation: {source}")]
    InsufficientVertices {
        /// The dimension that was attempted.
        dimension: usize,
        /// The underlying cell validation error.
        source: CellValidationError,
    },

    /// Failed to add vertex during triangulation construction.
    #[error("Failed to add vertex during construction: {message}")]
    FailedToAddVertex {
        /// Description of the vertex addition failure.
        message: String,
    },

    /// Geometric degeneracy prevents triangulation construction.
    #[error("Geometric degeneracy encountered during construction: {message}")]
    GeometricDegeneracy {
        /// Description of the degeneracy issue.
        message: String,
    },

    /// Attempted to insert a vertex with coordinates that already exist.
    #[error(
        "Duplicate coordinates: vertex with coordinates {coordinates} already exists in the triangulation"
    )]
    DuplicateCoordinates {
        /// String representation of the duplicate coordinates.
        coordinates: String,
    },
}

/// Errors that can occur during triangulation topology validation (Level 3).
///
/// # Examples
///
/// ```rust
/// use delaunay::core::triangulation::TriangulationValidationError;
/// use delaunay::prelude::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let result: Result<(), TriangulationValidationError> = dt.as_triangulation().validate();
/// assert!(result.is_ok());
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TriangulationValidationError {
    /// Lower-layer element or TDS structural validation error (Levels 1–2).
    #[error(transparent)]
    Tds(#[from] TdsValidationError),

    /// A facet belongs to an unexpected number of cells for a manifold-with-boundary.
    #[error(
        "Non-manifold facet: facet {facet_key:016x} belongs to {cell_count} cells (expected 1 or 2)"
    )]
    ManifoldFacetMultiplicity {
        /// The facet key with invalid multiplicity.
        facet_key: u64,
        /// The number of incident cells observed.
        cell_count: usize,
    },

    /// A boundary facet unexpectedly has a neighbor pointer across it.
    #[error(
        "Boundary facet {facet_key:016x} unexpectedly has a neighbor across cell {cell_uuid}[{facet_index}] -> {neighbor_key:?}"
    )]
    BoundaryFacetHasNeighbor {
        /// The facet key.
        facet_key: u64,
        /// UUID of the cell that owns the boundary facet.
        cell_uuid: Uuid,
        /// The facet index within the cell.
        facet_index: usize,
        /// The neighbor key that was unexpectedly present.
        neighbor_key: CellKey,
    },

    /// Two cells that share a facet do not point to each other as neighbors across that facet.
    #[error(
        "Interior facet {facet_key:016x} has inconsistent neighbor pointers: {first_cell_uuid}[{first_facet_index}] -> {first_neighbor:?}, {second_cell_uuid}[{second_facet_index}] -> {second_neighbor:?}"
    )]
    InteriorFacetNeighborMismatch {
        /// The facet key.
        facet_key: u64,
        /// The first cell key.
        first_cell_key: CellKey,
        /// The first cell UUID.
        first_cell_uuid: Uuid,
        /// The facet index in the first cell.
        first_facet_index: usize,
        /// The neighbor recorded in the first cell.
        first_neighbor: Option<CellKey>,
        /// The second cell key.
        second_cell_key: CellKey,
        /// The second cell UUID.
        second_cell_uuid: Uuid,
        /// The facet index in the second cell.
        second_facet_index: usize,
        /// The neighbor recorded in the second cell.
        second_neighbor: Option<CellKey>,
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
        "Vertex link is not a PL (D-1)-manifold: vertex {vertex_key:?} has link with {link_vertex_count} vertices, {link_cell_count} cells, boundary_facets={boundary_facet_count}, max_degree={max_degree}, connected={connected}, interior_vertex={interior_vertex}"
    )]
    VertexLinkNotManifold {
        /// The vertex whose link failed validation.
        vertex_key: VertexKey,
        /// Number of vertices in the link (0-simplices of the link).
        link_vertex_count: usize,
        /// Number of (D-1)-simplices (cells) in the link.
        link_cell_count: usize,
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

    /// Topology computation/classification failed.
    #[error(transparent)]
    Topology(#[from] TopologyError),
}

impl From<TdsMutationError> for TriangulationValidationError {
    fn from(err: TdsMutationError) -> Self {
        Self::Tds(err.into())
    }
}

impl From<ManifoldError> for TriangulationValidationError {
    fn from(err: ManifoldError) -> Self {
        match err {
            ManifoldError::Tds(source) => Self::Tds(source),
            ManifoldError::ManifoldFacetMultiplicity {
                facet_key,
                cell_count,
            } => Self::ManifoldFacetMultiplicity {
                facet_key,
                cell_count,
            },
            ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            } => Self::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            },
            ManifoldError::RidgeLinkNotManifold {
                ridge_key,
                link_vertex_count,
                link_edge_count,
                max_degree,
                degree_one_vertices,
                connected,
            } => Self::RidgeLinkNotManifold {
                ridge_key,
                link_vertex_count,
                link_edge_count,
                max_degree,
                degree_one_vertices,
                connected,
            },
            ManifoldError::VertexLinkNotManifold {
                vertex_key,
                link_vertex_count,
                link_cell_count,
                boundary_facet_count,
                max_degree,
                connected,
                interior_vertex,
            } => Self::VertexLinkNotManifold {
                vertex_key,
                link_vertex_count,
                link_cell_count,
                boundary_facet_count,
                max_degree,
                connected,
                interior_vertex,
            },
        }
    }
}

type TryInsertImplOk = ((VertexKey, Option<CellKey>), usize, SuspicionFlags);

/// Policy controlling when the triangulation runs global validation passes.
///
/// Validation can be expensive (O(N×D²) or worse), so this allows callers to trade
/// performance for stricter correctness checks during incremental operations.
///
/// **Note**: [`TopologyGuarantee::PLManifold`] is incompatible with [`ValidationPolicy::Never`].
/// `PLManifold` requires at least end-of-construction validation to certify full
/// PL-manifoldness. Use [`ValidationPolicy::OnSuspicion`] (default) for best performance,
/// or [`ValidationPolicy::Always`] for maximum safety during incremental operations.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::operations::SuspicionFlags;
/// use delaunay::core::triangulation::ValidationPolicy;
///
/// let policy = ValidationPolicy::OnSuspicion;
/// let suspicion = SuspicionFlags { perturbation_used: true, ..SuspicionFlags::default() };
/// assert!(policy.should_validate(suspicion));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationPolicy {
    /// Never run global validation.
    Never,

    /// Validate only if the operation is suspicious (e.g. degeneracy).
    OnSuspicion,

    /// Always validate after insertion.
    Always,

    /// Debug builds: always validate; release builds: [`ValidationPolicy::OnSuspicion`].
    DebugOnly,
}

impl ValidationPolicy {
    /// Returns `true` if a global validation pass should be run given the observed
    /// [`crate::core::operations::SuspicionFlags`].
    #[inline]
    #[must_use]
    pub const fn should_validate(&self, suspicion: SuspicionFlags) -> bool {
        match self {
            Self::Never => false,
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
///   each facet is incident to one or two cells, and the codimension-2 boundary is closed.
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
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
/// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
///
/// // Optional: relax topology checks for speed (weaker guarantees).
/// dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
/// assert!(!dt.topology_guarantee().requires_vertex_links_at_completion());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyGuarantee {
    /// Validate only the pseudomanifold / manifold-with-boundary invariants:
    /// - facet degree (1 or 2 incident cells per facet)
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

    /// Returns `true` if this topology guarantee requires ridge-link validation.
    ///
    /// Ridge-link validation is fast (O(local)) and catches many PL-manifold violations,
    /// providing good error detection even with reduced validation frequency.
    #[inline]
    #[must_use]
    pub const fn requires_ridge_links(self) -> bool {
        matches!(self, Self::PLManifold | Self::PLManifoldStrict)
    }

    /// Legacy method for backward compatibility.
    #[deprecated(
        since = "0.7.0",
        note = "Use requires_vertex_links_during_insertion or requires_vertex_links_at_completion"
    )]
    #[inline]
    #[must_use]
    pub const fn requires_vertex_links(self) -> bool {
        matches!(self, Self::PLManifold | Self::PLManifoldStrict)
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

/// Generic triangulation combining kernel and data structure.
///
/// # Type Parameters
/// - `K`: Geometric kernel implementing predicates
/// - `U`: User data type for vertices
/// - `V`: User data type for cells
/// - `D`: Dimension of the triangulation
///
/// # Phase 2 TODO
/// Add geometric operations that use the kernel for predicates.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::triangulation::Triangulation;
/// use delaunay::geometry::kernel::FastKernel;
///
/// let tri: Triangulation<FastKernel<f64>, (), (), 3> =
///     Triangulation::new_empty(FastKernel::new());
/// assert_eq!(tri.number_of_vertices(), 0);
/// ```
#[derive(Clone, Debug)]
pub struct Triangulation<K, U, V, const D: usize>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    /// The geometric kernel for predicates.
    pub(crate) kernel: K,
    /// The combinatorial triangulation data structure.
    pub(crate) tds: Tds<K::Scalar, U, V, D>,
    // TODO: Add after bistellar flips + robust insertion (v0.7.0+)
    // /// The topological space this triangulation lives in.
    // pub(crate) topology: Box<dyn TopologicalSpace>,
    pub(crate) validation_policy: ValidationPolicy,
    pub(crate) topology_guarantee: TopologyGuarantee,
}

// =============================================================================
// Internal Helpers (Structural / Graph Traversals)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    /// Traverses the cell neighbor graph starting at `start` and returns the set of visited cells.
    ///
    /// If `allowed` is `Some`, traversal is restricted to that set. Neighbors outside the allowed
    /// set are reported via `on_external_neighbor`.
    #[must_use]
    fn traverse_cell_neighbor_graph<F>(
        &self,
        start: CellKey,
        reserve: usize,
        allowed: Option<&CellKeySet>,
        mut on_external_neighbor: F,
    ) -> CellKeySet
    where
        F: FnMut(CellKey, CellKey),
    {
        let mut visited: CellKeySet = CellKeySet::default();
        visited.reserve(reserve);

        let mut stack: CellKeyBuffer = CellKeyBuffer::new();
        stack.push(start);

        while let Some(ck) = stack.pop() {
            if !visited.insert(ck) {
                continue;
            }

            let Some(cell) = self.tds.get_cell(ck) else {
                continue;
            };

            let Some(neighbors) = cell.neighbors() else {
                continue;
            };

            for &n_opt in neighbors {
                let Some(nk) = n_opt else {
                    continue;
                };

                if !self.tds.contains_cell(nk) {
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
}

// =============================================================================
// Basic Accessors (Minimal Bounds)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Create an empty triangulation with the given kernel.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let tri: Triangulation<FastKernel<f64>, (), (), 3> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// assert_eq!(tri.number_of_vertices(), 0);
    /// assert_eq!(tri.number_of_cells(), 0);
    /// assert_eq!(tri.dim(), -1); // Empty triangulation has dimension -1
    /// ```
    #[must_use]
    pub fn new_empty(kernel: K) -> Self {
        Self {
            kernel,
            tds: Tds::empty(),
            validation_policy: ValidationPolicy::default(),
            topology_guarantee: TopologyGuarantee::DEFAULT,
        }
    }

    /// Returns the topology guarantee used for Level 3 topology validation.
    #[inline]
    #[must_use]
    pub const fn topology_guarantee(&self) -> TopologyGuarantee {
        self.topology_guarantee
    }

    /// Returns the insertion-time global topology validation policy used by the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation::{Triangulation, ValidationPolicy};
    /// use delaunay::geometry::kernel::FastKernel;
    ///
    /// let tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    ///
    /// assert_eq!(tri.validation_policy(), ValidationPolicy::OnSuspicion);
    /// ```
    #[inline]
    #[must_use]
    pub const fn validation_policy(&self) -> ValidationPolicy {
        self.validation_policy
    }

    /// Sets the insertion-time global topology validation policy used by the triangulation.
    ///
    /// If the requested policy is incompatible with the current topology guarantee (for example,
    /// `ValidationPolicy::Never` with `TopologyGuarantee::PLManifold`), this runs
    /// [`Triangulation::validate_at_completion`](Self::validate_at_completion) to provide
    /// immediate feedback and emits a warning. Call `validate_at_completion()` after batch
    /// construction when using an incompatible combination.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation::{Triangulation, ValidationPolicy};
    /// use delaunay::geometry::kernel::FastKernel;
    ///
    /// let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    ///
    /// tri.set_validation_policy(ValidationPolicy::Always);
    /// assert_eq!(tri.validation_policy(), ValidationPolicy::Always);
    /// ```
    #[inline]
    pub fn set_validation_policy(&mut self, policy: ValidationPolicy) {
        if !self.topology_guarantee.is_compatible_with_policy(policy) {
            let completion_result = self.validate_at_completion();

            if let Err(err) = completion_result {
                debug_assert!(
                    false,
                    "Validation policy {policy:?} is incompatible with topology guarantee {guarantee:?}; validate_at_completion failed: {err}",
                    guarantee = self.topology_guarantee
                );
                tracing::warn!(
                    "Validation policy {policy:?} is incompatible with topology guarantee {guarantee:?}; validate_at_completion failed: {err}. Validation policy not updated.",
                    guarantee = self.topology_guarantee
                );
                return;
            }

            tracing::warn!(
                "Validation policy {policy:?} is incompatible with topology guarantee {guarantee:?}; call validate_at_completion() after construction to certify PL-manifoldness.",
                guarantee = self.topology_guarantee
            );
        }

        self.validation_policy = policy;
    }

    /// Sets the topology guarantee used for Level 3 topology validation.
    ///
    /// If the requested guarantee is incompatible with the current validation policy (for
    /// example, `ValidationPolicy::Never` with `TopologyGuarantee::PLManifold`), this runs
    /// [`Triangulation::validate_at_completion`](Self::validate_at_completion) to provide
    /// immediate feedback and emits a warning. Call `validate_at_completion()` after batch
    /// construction when using an incompatible combination.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation::{TopologyGuarantee, Triangulation};
    /// use delaunay::geometry::kernel::FastKernel;
    ///
    /// let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
    /// assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
    /// ```
    #[inline]
    pub fn set_topology_guarantee(&mut self, guarantee: TopologyGuarantee) {
        if !guarantee.is_compatible_with_policy(self.validation_policy) {
            let previous = self.topology_guarantee;
            self.topology_guarantee = guarantee;
            let completion_result = self.validate_at_completion();

            if let Err(err) = completion_result {
                self.topology_guarantee = previous;
                debug_assert!(
                    false,
                    "Topology guarantee {guarantee:?} is incompatible with validation policy {policy:?}; validate_at_completion failed: {err}",
                    policy = self.validation_policy
                );
                tracing::warn!(
                    "Topology guarantee {guarantee:?} is incompatible with validation policy {policy:?}; validate_at_completion failed: {err}. Topology guarantee not updated.",
                    policy = self.validation_policy
                );
                return;
            }

            self.topology_guarantee = previous;
            tracing::warn!(
                "Topology guarantee {guarantee:?} is incompatible with validation policy {policy:?}; call validate_at_completion() after construction to certify PL-manifoldness.",
                policy = self.validation_policy
            );
        }

        self.topology_guarantee = guarantee;
    }

    /// Returns the number of times the topology safety-net recovered from a Level 3
    /// validation failure by retrying insertion with a star-split of the containing cell.
    ///
    /// This is a process-wide counter (across all triangulation instances) intended for
    /// production telemetry. A high value suggests the cavity-based insertion frequently
    /// creates transient invalid topology that is being masked by the fallback.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation::Triangulation;
    ///
    /// let count = Triangulation::<delaunay::geometry::kernel::FastKernel<f64>, (), (), 3>
    ///     ::topology_safety_net_star_split_fallback_successes();
    /// assert!(count >= 0);
    /// ```
    #[must_use]
    pub fn topology_safety_net_star_split_fallback_successes() -> u64 {
        TOPOLOGY_SAFETY_NET_STAR_SPLIT_FALLBACK_SUCCESSES.load(Ordering::Relaxed)
    }

    /// Returns duplicate-detection telemetry if enabled via `DELAUNAY_DUPLICATE_METRICS`.
    ///
    /// This is a process-wide counter (across all triangulation instances). It reports how often
    /// duplicate checks used the hash grid versus falling back to linear scans, along with the
    /// total candidate count inspected during grid queries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation::{DuplicateDetectionMetrics, Triangulation};
    ///
    /// let metrics = Triangulation::<delaunay::geometry::kernel::FastKernel<f64>, (), (), 3>
    ///     ::duplicate_detection_metrics();
    /// let _ = metrics; // None unless DELAUNAY_DUPLICATE_METRICS is set
    /// ```
    #[must_use]
    pub fn duplicate_detection_metrics() -> Option<DuplicateDetectionMetrics> {
        if !duplicate_detection_metrics_enabled() {
            return None;
        }
        Some(DuplicateDetectionMetrics {
            total_checks: DUPLICATE_DETECTION_TOTAL.load(Ordering::Relaxed),
            grid_used: DUPLICATE_DETECTION_GRID_USED.load(Ordering::Relaxed),
            grid_fallbacks: DUPLICATE_DETECTION_GRID_FALLBACKS.load(Ordering::Relaxed),
            grid_candidates: DUPLICATE_DETECTION_GRID_CANDIDATES.load(Ordering::Relaxed),
        })
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn new_with_tds(kernel: K, tds: Tds<K::Scalar, U, V, D>) -> Self {
        Self {
            kernel,
            tds,
            validation_policy: ValidationPolicy::default(),
            topology_guarantee: TopologyGuarantee::DEFAULT,
        }
    }

    // TODO: Implement after bistellar flips + robust insertion (v0.7.0+)
    // /// Create a triangulation with a specified topological space.
    // ///
    // /// This is the generic triangulation layer method that constructs
    // /// triangulations on different topological spaces. The Delaunay layer's
    // /// `with_topology` method should delegate to this.
    // ///
    // /// Requires:
    // /// - Bistellar flips for topology-preserving operations
    // /// - Insertion algorithm that respects topology constraints
    // /// - Topology-aware boundary handling
    // ///
    // /// # Examples (future)
    // ///
    // /// ```rust,ignore
    // /// use delaunay::prelude::*;
    // /// use delaunay::topology::spaces::SphericalSpace;
    // ///
    // /// let space = SphericalSpace::new();
    // /// let tri = Triangulation::with_topology(
    // ///     FastKernel::new(),
    // ///     space,
    // ///     tds
    // /// );
    // /// ```
    // #[must_use]
    // pub fn with_topology<T>(
    //     kernel: K,
    //     topology: T,
    //     tds: Tds<K::Scalar, U, V, D>,
    // ) -> Self
    // where
    //     T: TopologicalSpace,
    // {
    //     Self {
    //         kernel,
    //         tds,
    //         // topology: Box::new(topology),
    //     }
    // }

    /// Returns an iterator over all cells in the triangulation.
    ///
    /// Delegates to the underlying Tds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// // Iterate over cells
    /// for (_cell_key, cell) in tri.cells() {
    ///     assert_eq!(cell.number_of_vertices(), 3); // 2D triangle
    /// }
    /// assert_eq!(tri.cells().count(), 1);
    /// ```
    pub fn cells(&self) -> impl Iterator<Item = (CellKey, &Cell<K::Scalar, U, V, D>)> {
        self.tds.cells()
    }

    /// Returns an iterator over all vertices in the triangulation.
    ///
    /// Delegates to the underlying Tds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// // Iterate over vertices
    /// for (_vertex_key, vertex) in tri.vertices() {
    ///     assert_eq!(vertex.dim(), 2); // 2D vertices
    /// }
    /// assert_eq!(tri.vertices().count(), 3);
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<K::Scalar, U, D>)> {
        self.tds.vertices()
    }

    /// Returns the number of vertices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.as_triangulation().number_of_vertices(), 4);
    /// ```
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.tds.number_of_vertices()
    }

    /// Returns the number of cells in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.as_triangulation().number_of_cells(), 1); // Single tetrahedron
    /// ```
    #[must_use]
    pub fn number_of_cells(&self) -> usize {
        self.tds.number_of_cells()
    }

    /// Returns the dimension of the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // Empty triangulation has dimension -1
    /// let empty: Triangulation<FastKernel<f64>, (), (), 3> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// assert_eq!(empty.dim(), -1);
    ///
    /// // 3D tetrahedron has dimension 3
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.as_triangulation().dim(), 3);
    /// ```
    #[must_use]
    pub fn dim(&self) -> i32 {
        self.tds.dim()
    }

    /// Returns an iterator over all facets in the triangulation.
    ///
    /// This provides efficient access to all facets without pre-allocating a vector.
    /// Each facet is represented as a lightweight `FacetView` that references the
    /// underlying triangulation data.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for all facets in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Iterate over all facets
    /// let facet_count = dt.as_triangulation().facets().count();
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 facets
    /// ```
    pub fn facets(&self) -> AllFacetsIter<'_, K::Scalar, U, V, D> {
        AllFacetsIter::new(&self.tds)
    }

    /// Returns an iterator over boundary (hull) facets in the triangulation.
    ///
    /// Boundary facets are those that belong to exactly one cell. This method
    /// computes the facet-to-cells map internally for convenience.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for boundary facets only.
    ///
    /// # Panics
    ///
    /// Panics if the triangulation data structure is corrupted (cells have invalid
    /// neighbor relationships or facet information). This indicates a bug in the
    /// library and should never happen with a properly constructed triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let boundary_count = dt.as_triangulation().boundary_facets().count();
    /// assert_eq!(boundary_count, 4); // All facets are on boundary
    /// ```
    pub fn boundary_facets(&self) -> BoundaryFacetsIter<'_, K::Scalar, U, V, D> {
        // build_facet_to_cells_map only fails if cells have invalid structure,
        // which should never happen in a valid triangulation
        let facet_map = self
            .tds
            .build_facet_to_cells_map()
            .expect("Failed to build facet map - triangulation structure is corrupted");
        BoundaryFacetsIter::new(&self.tds, facet_map)
    }

    // =============================================================================
    // Public Topology Traversal & Adjacency API (Read-only)
    // =============================================================================

    #[inline]
    fn debug_assert_adjacency_index_matches(&self, index: &AdjacencyIndex) {
        // AdjacencyIndex is built from a snapshot of a triangulation. We cannot enforce at
        // compile-time that an index belongs to this triangulation, but we can cheaply catch
        // obvious mix-ups in debug builds.
        debug_assert_eq!(
            index.vertex_to_cells.len(),
            self.tds.number_of_vertices(),
            "AdjacencyIndex vertex_to_cells size does not match triangulation vertex count"
        );
        debug_assert_eq!(
            index.vertex_to_edges.len(),
            self.tds.number_of_vertices(),
            "AdjacencyIndex vertex_to_edges size does not match triangulation vertex count"
        );
    }

    /// Returns an iterator over all unique edges in the triangulation.
    ///
    /// Edges are inferred from the vertex lists of each cell; they are not stored explicitly.
    ///
    /// ## Allocation and iteration order
    ///
    /// This method allocates an internal set to deduplicate edges. The iteration order is
    /// not specified.
    ///
    /// If you need fast repeated topology queries, consider building an
    /// [`AdjacencyIndex`] once via [`Triangulation::build_adjacency_index`](Self::build_adjacency_index).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let edges: std::collections::HashSet<_> = tri.edges().collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = EdgeKey> + '_ {
        self.collect_edges().into_iter()
    }

    /// Returns an iterator over all unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This avoids per-call deduplication and allocations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::query::*;
    ///
    /// // A single 3D tetrahedron has 6 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let index = tri.build_adjacency_index().unwrap();
    /// let edges: std::collections::HashSet<_> = tri.edges_with_index(&index).collect();
    /// assert_eq!(edges.len(), 6);
    /// ```
    pub fn edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.edges()
    }

    /// Returns the number of unique edges in the triangulation.
    ///
    /// This is equivalent to `self.edges().count()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // A single 2D triangle has 3 unique edges.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// assert_eq!(tri.number_of_edges(), 3);
    /// ```
    #[must_use]
    pub fn number_of_edges(&self) -> usize {
        self.collect_edges().len()
    }

    /// Returns the number of unique edges using a precomputed [`AdjacencyIndex`].
    ///
    /// This is equivalent to `self.edges_with_index(index).count()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// assert_eq!(tri.number_of_edges_with_index(&index), 6);
    /// ```
    #[must_use]
    pub fn number_of_edges_with_index(&self, index: &AdjacencyIndex) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_edges()
    }

    /// Returns an iterator over all cells adjacent (incident) to a vertex.
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    ///
    /// Iteration order is not specified.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // Two tetrahedra sharing a triangular facet.
    /// let vertices: Vec<_> = vec![
    ///     // Shared triangle
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([2.0, 0.0, 0.0]),
    ///     vertex!([1.0, 2.0, 0.0]),
    ///     // Two apices
    ///     vertex!([1.0, 0.7, 1.5]),
    ///     vertex!([1.0, 0.7, -1.5]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// // Find a vertex on the shared triangle by coordinates.
    /// let shared_vertex_key = tri
    ///     .vertices()
    ///     .find_map(|(vk, _)| {
    ///         let coords = tri.vertex_coords(vk)?;
    ///         (coords == [0.0, 0.0, 0.0]).then_some(vk)
    ///     })
    ///     .unwrap();
    ///
    /// // The shared vertex is incident to both cells.
    /// assert_eq!(tri.adjacent_cells(shared_vertex_key).count(), 2);
    /// ```
    pub fn adjacent_cells(&self, v: VertexKey) -> impl Iterator<Item = CellKey> + '_ {
        self.tds.find_cells_containing_vertex_by_key(v).into_iter()
    }

    /// Returns an iterator over all cells adjacent (incident) to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// This avoids per-call scans of the triangulation.
    ///
    /// If `v` is not present in the index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([2.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 2.0, 0.0]),
    /// #     // Two apices
    /// #     vertex!([1.0, 0.7, 1.5]),
    /// #     vertex!([1.0, 0.7, -1.5]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let v = tri.vertices().next().unwrap().0;
    /// assert!(tri.adjacent_cells_with_index(&index, v).count() >= 1);
    /// ```
    pub fn adjacent_cells_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> impl Iterator<Item = CellKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.adjacent_cells(v)
    }

    /// Returns the number of cells adjacent (incident) to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// If `v` is not present in the index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let v0 = tri.vertices().next().unwrap().0;
    /// assert_eq!(tri.number_of_adjacent_cells_with_index(&index, v0), 1);
    /// ```
    #[must_use]
    pub fn number_of_adjacent_cells_with_index(
        &self,
        index: &AdjacencyIndex,
        v: VertexKey,
    ) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_adjacent_cells(v)
    }

    /// Returns an iterator over all neighbors of a cell.
    ///
    /// Boundary facets are omitted (only existing neighbors are yielded). If `c` is not
    /// present, the iterator is empty.
    ///
    /// Iteration order is not specified.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // Two tetrahedra sharing a triangular facet => each tetra has exactly one neighbor.
    /// let vertices: Vec<_> = vec![
    ///     // Shared triangle
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([2.0, 0.0, 0.0]),
    ///     vertex!([1.0, 2.0, 0.0]),
    ///     // Two apices
    ///     vertex!([1.0, 0.7, 1.5]),
    ///     vertex!([1.0, 0.7, -1.5]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let cell_keys: Vec<_> = tri.cells().map(|(ck, _)| ck).collect();
    /// assert_eq!(cell_keys.len(), 2);
    ///
    /// for &ck in &cell_keys {
    ///     let neighbors: Vec<_> = tri.cell_neighbors(ck).collect();
    ///     assert_eq!(neighbors.len(), 1);
    ///     assert!(cell_keys.contains(&neighbors[0]));
    ///     assert_ne!(neighbors[0], ck);
    /// }
    /// ```
    pub fn cell_neighbors(&self, c: CellKey) -> impl Iterator<Item = CellKey> + '_ {
        self.tds
            .get_cell(c)
            .and_then(|cell| cell.neighbors())
            .into_iter()
            .flat_map(|neighbors| neighbors.iter().copied().flatten())
    }

    /// Returns an iterator over all neighbors of a cell using a precomputed [`AdjacencyIndex`].
    ///
    /// If `c` is not present in the index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([2.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 2.0, 0.0]),
    /// #     // Two apices
    /// #     vertex!([1.0, 0.7, 1.5]),
    /// #     vertex!([1.0, 0.7, -1.5]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let cell_key = tri.cells().next().unwrap().0;
    /// let neighbors: Vec<_> = tri.cell_neighbors_with_index(&index, cell_key).collect();
    /// assert_eq!(neighbors.len(), 1);
    /// ```
    pub fn cell_neighbors_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        c: CellKey,
    ) -> impl Iterator<Item = CellKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.cell_neighbors(c)
    }

    /// Returns the number of neighbors of a cell using a precomputed [`AdjacencyIndex`].
    ///
    /// If `c` is not present in the index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices: Vec<_> = vec![
    /// #     // Shared triangle
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([2.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 2.0, 0.0]),
    /// #     // Two apices
    /// #     vertex!([1.0, 0.7, 1.5]),
    /// #     vertex!([1.0, 0.7, -1.5]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let cell_key = tri.cells().next().unwrap().0;
    /// assert_eq!(tri.number_of_cell_neighbors_with_index(&index, cell_key), 1);
    /// ```
    #[must_use]
    pub fn number_of_cell_neighbors_with_index(&self, index: &AdjacencyIndex, c: CellKey) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_cell_neighbors(c)
    }

    /// Returns an iterator over all unique edges incident to a vertex.
    ///
    /// If `v` is not present in this triangulation, the iterator is empty.
    ///
    /// ## Allocation and iteration order
    ///
    /// This method allocates an internal set to deduplicate edges. The iteration order is
    /// not specified.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // In a single tetrahedron, each vertex has degree 3.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let v0 = tri.vertices().next().unwrap().0;
    /// let incident: Vec<_> = tri.incident_edges(v0).collect();
    /// assert_eq!(incident.len(), 3);
    /// assert!(incident
    ///     .iter()
    ///     .all(|e| matches!(e.endpoints(), (a, b) if a == v0 || b == v0)));
    /// ```
    pub fn incident_edges(&self, v: VertexKey) -> impl Iterator<Item = EdgeKey> + '_ {
        self.collect_incident_edges(v).into_iter()
    }

    /// Returns an iterator over all unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// If `v` is not present in the index, the iterator is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let v0 = tri.vertices().next().unwrap().0;
    /// assert_eq!(tri.incident_edges_with_index(&index, v0).count(), 3);
    /// ```
    pub fn incident_edges_with_index<'a>(
        &self,
        index: &'a AdjacencyIndex,
        v: VertexKey,
    ) -> impl Iterator<Item = EdgeKey> + 'a {
        self.debug_assert_adjacency_index_matches(index);
        index.incident_edges(v)
    }

    /// Returns the number of unique edges incident to a vertex using a precomputed
    /// [`AdjacencyIndex`].
    ///
    /// If `v` is not present in the index, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use delaunay::prelude::query::*;
    /// # let vertices = vec![
    /// #     vertex!([0.0, 0.0, 0.0]),
    /// #     vertex!([1.0, 0.0, 0.0]),
    /// #     vertex!([0.0, 1.0, 0.0]),
    /// #     vertex!([0.0, 0.0, 1.0]),
    /// # ];
    /// # let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// # let tri = dt.as_triangulation();
    /// # let index = tri.build_adjacency_index().unwrap();
    /// let v0 = tri.vertices().next().unwrap().0;
    /// assert_eq!(tri.number_of_incident_edges_with_index(&index, v0), 3);
    /// ```
    #[must_use]
    pub fn number_of_incident_edges_with_index(
        &self,
        index: &AdjacencyIndex,
        v: VertexKey,
    ) -> usize {
        self.debug_assert_adjacency_index_matches(index);
        index.number_of_incident_edges(v)
    }

    /// Returns the number of unique edges incident to a vertex.
    ///
    /// If `v` is not present in this triangulation, returns 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // In a single tetrahedron, each vertex has degree 3.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let v0 = tri.vertices().next().unwrap().0;
    /// assert_eq!(tri.number_of_incident_edges(v0), 3);
    /// ```
    #[must_use]
    pub fn number_of_incident_edges(&self, v: VertexKey) -> usize {
        self.collect_incident_edges(v).len()
    }

    /// Returns a slice view of a cell's vertex keys.
    ///
    /// This is a zero-allocation accessor. If `c` is not present, returns `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let cell_key = tri.cells().next().unwrap().0;
    /// let cell_vertices = tri.cell_vertices(cell_key).unwrap();
    /// assert_eq!(cell_vertices.len(), 3); // D+1 for a 2D simplex
    /// ```
    #[must_use]
    pub fn cell_vertices(&self, c: CellKey) -> Option<&[VertexKey]> {
        self.tds.get_cell(c).map(Cell::vertices)
    }

    /// Returns a slice view of a vertex's coordinates.
    ///
    /// This is a zero-allocation accessor. If `v` is not present, returns `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// // Find the key for a known vertex by matching coordinates.
    /// let v_key = tri
    ///     .vertices()
    ///     .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [1.0, 0.0]).then_some(vk))
    ///     .unwrap();
    ///
    /// assert_eq!(tri.vertex_coords(v_key).unwrap(), [1.0, 0.0]);
    /// ```
    #[must_use]
    pub fn vertex_coords(&self, v: VertexKey) -> Option<&[K::Scalar]> {
        self.tds
            .get_vertex_by_key(v)
            .map(|vertex| &vertex.point().coords()[..])
    }

    /// Builds an immutable adjacency index for fast repeated topology queries.
    ///
    /// This never stores any cache internally and does not mutate the triangulation.
    ///
    /// ## Notes
    ///
    /// - No sorted-order guarantees are provided for the values.
    /// - The returned collections are optimized for performance.
    /// - The maps include an entry for every vertex currently stored in the triangulation.
    ///   During the bootstrap phase (before the initial simplex is created), vertices have empty
    ///   adjacency lists because no cells exist yet. This is expected and not an error condition.
    /// - Isolated vertices (present in the vertex store but not referenced by any cell) are allowed at
    ///   the TDS structural layer, but violate the Level 3 manifold invariants checked by
    ///   [`Triangulation::is_valid`](Self::is_valid). When present, their adjacency lists are empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the triangulation data structure is internally inconsistent
    /// (e.g., a cell references a missing vertex key or a missing neighbor cell key).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // Two tetrahedra sharing a triangular facet.
    /// let vertices: Vec<_> = vec![
    ///     // Shared triangle
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([2.0, 0.0, 0.0]),
    ///     vertex!([1.0, 2.0, 0.0]),
    ///     // Two apices
    ///     vertex!([1.0, 0.7, 1.5]),
    ///     vertex!([1.0, 0.7, -1.5]),
    /// ];
    ///
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.as_triangulation();
    ///
    /// let index = tri.build_adjacency_index().unwrap();
    ///
    /// // The index exposes adjacency maps keyed by VertexKey / CellKey.
    /// let cell_keys: Vec<_> = tri.cells().map(|(ck, _)| ck).collect();
    /// for &ck in &cell_keys {
    ///     let neighbors = index.cell_to_neighbors.get(&ck).unwrap();
    ///     assert_eq!(neighbors.len(), 1);
    /// }
    /// ```
    pub fn build_adjacency_index(&self) -> Result<AdjacencyIndex, AdjacencyIndexBuildError> {
        let vertex_cap = self.tds.number_of_vertices();
        let cell_cap = self.tds.number_of_cells();

        let mut vertex_to_cells: VertexToCellsMap = fast_hash_map_with_capacity(vertex_cap);
        let mut cell_to_neighbors: FastHashMap<
            CellKey,
            SmallBuffer<CellKey, MAX_PRACTICAL_DIMENSION_SIZE>,
        > = fast_hash_map_with_capacity(cell_cap);
        let mut vertex_to_edges: FastHashMap<
            VertexKey,
            SmallBuffer<EdgeKey, MAX_PRACTICAL_DIMENSION_SIZE>,
        > = fast_hash_map_with_capacity(vertex_cap);

        // Deduplicate edges globally while building the index.
        let edges_per_cell = (D + 1).saturating_mul(D) / 2;
        let mut seen_edges: FastHashSet<EdgeKey> =
            fast_hash_set_with_capacity(cell_cap.saturating_mul(edges_per_cell));

        for (cell_key, cell) in self.tds.cells() {
            let vertices = cell.vertices();

            // Vertex → cells
            for &vk in vertices {
                if !self.tds.contains_vertex_key(vk) {
                    return Err(AdjacencyIndexBuildError::MissingVertexKey {
                        cell_key,
                        vertex_key: vk,
                    });
                }
                let entry = vertex_to_cells.entry(vk).or_default();
                #[cfg(debug_assertions)]
                let was_spilled = entry.spilled();
                entry.push(cell_key);
                #[cfg(debug_assertions)]
                if !was_spilled && entry.spilled() {
                    let spill_count =
                        VERTEX_TO_CELLS_SPILL_EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
                    tracing::debug!(
                        "VertexToCellsMap spill #{spill_count}: vertex={vk:?} len={} cap={} (MAX_PRACTICAL_DIMENSION_SIZE={MAX_PRACTICAL_DIMENSION_SIZE})",
                        entry.len(),
                        entry.capacity()
                    );
                }
            }

            // Cell → neighbors
            if let Some(neighbors) = cell.neighbors() {
                let mut neighs: SmallBuffer<CellKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    SmallBuffer::new();

                for &n_opt in neighbors {
                    let Some(nk) = n_opt else {
                        continue;
                    };

                    if !self.tds.contains_cell(nk) {
                        return Err(AdjacencyIndexBuildError::MissingNeighborCell {
                            cell_key,
                            neighbor_key: nk,
                        });
                    }

                    neighs.push(nk);
                }

                if !neighs.is_empty() {
                    cell_to_neighbors.insert(cell_key, neighs);
                }
            }

            // Vertex → edges (deduped)
            for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    let edge = EdgeKey::new(vertices[i], vertices[j]);
                    if !seen_edges.insert(edge) {
                        continue;
                    }

                    let (a, b) = edge.endpoints();
                    vertex_to_edges.entry(a).or_default().push(edge);
                    vertex_to_edges.entry(b).or_default().push(edge);
                }
            }
        }

        // Ensure every vertex in the triangulation has an entry, even if it is currently
        // not incident to any cell (e.g., bootstrap phase with < D+1 vertices, or TDS-level
        // states with isolated vertices). Level 3 topology validation (`Triangulation::is_valid`)
        // rejects isolated vertices, but this indexing helper remains usable for debugging and
        // intermediate construction states.
        for (vk, _) in self.tds.vertices() {
            vertex_to_cells.entry(vk).or_default();
            vertex_to_edges.entry(vk).or_default();
        }

        Ok(AdjacencyIndex {
            vertex_to_edges,
            vertex_to_cells,
            cell_to_neighbors,
        })
    }

    #[must_use]
    fn collect_edges(&self) -> FastHashSet<EdgeKey> {
        let cell_cap = self.tds.number_of_cells();
        let edges_per_cell = (D + 1).saturating_mul(D) / 2;

        let mut edges: FastHashSet<EdgeKey> =
            fast_hash_set_with_capacity(cell_cap.saturating_mul(edges_per_cell));

        for (_cell_key, cell) in self.tds.cells() {
            let vertices = cell.vertices();
            for i in 0..vertices.len() {
                for j in (i + 1)..vertices.len() {
                    edges.insert(EdgeKey::new(vertices[i], vertices[j]));
                }
            }
        }

        edges
    }

    #[must_use]
    fn collect_incident_edges(&self, v: VertexKey) -> FastHashSet<EdgeKey> {
        let mut edges: FastHashSet<EdgeKey> = FastHashSet::default();

        for cell_key in self.adjacent_cells(v) {
            let Some(cell) = self.tds.get_cell(cell_key) else {
                continue;
            };

            for &other in cell.vertices() {
                if other == v {
                    continue;
                }
                edges.insert(EdgeKey::new(v, other));
            }
        }

        edges
    }

    /// Validates topological invariants of the triangulation (Level 3).
    ///
    /// This checks the triangulation/topology layer **only**:
    /// - Codimension-1 pseudomanifold condition: each facet is incident to 1 (boundary) or 2 (interior) cells
    /// - Codimension-2 boundary manifoldness: the boundary must be closed ("no boundary of boundary")
    /// - Ridge-link validation (when `topology_guarantee.requires_ridge_links()`)
    /// - Vertex-link validation during insertion (when `topology_guarantee.requires_vertex_links_during_insertion()`)
    /// - Connectedness (single component in the cell neighbor graph)
    /// - No isolated vertices (every vertex must be incident to at least one cell)
    /// - Euler characteristic
    ///
    /// For `TopologyGuarantee::PLManifold`, full PL-manifold certification requires
    /// calling [`Triangulation::validate_at_completion`](Self::validate_at_completion)
    /// (or [`Triangulation::validate`](Self::validate)) after batch construction.
    ///
    /// It intentionally does **not** validate lower layers (vertices/cells or TDS structure).
    /// For cumulative validation, use [`Triangulation::validate`](Self::validate).
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationError`] if:
    /// - The manifold-with-boundary facet property is violated.
    /// - The triangulation is disconnected (multiple cell components).
    /// - An isolated vertex is detected (no incident cell).
    /// - Euler characteristic validation fails.
    /// - The topology module reports an error (treated as inconsistent data structure).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices_4d = [
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices_4d).unwrap();
    ///
    /// // Level 3: topology validation (manifold-with-boundary + Euler characteristic)
    /// assert!(dt.as_triangulation().is_valid().is_ok());
    /// ```
    pub fn is_valid(&self) -> Result<(), TriangulationValidationError> {
        // 1. Manifold facet multiplicity (codimension-1 pseudomanifold condition)
        //
        // Build the facet map once and reuse it for manifold validation and Euler counting.
        let facet_to_cells: FacetToCellsMap = self.tds.build_facet_to_cells_map()?;
        validate_facet_degree(&facet_to_cells)?;

        // 1b. Boundary manifoldness in codimension 2: the boundary must be "closed"
        // (i.e., its ridges must have degree 2 within boundary facets).
        validate_closed_boundary(&self.tds, &facet_to_cells)?;

        // 1c. Ridge-link validation for PLManifold/PLManifoldStrict (fast, catches many PL issues).
        if self.topology_guarantee.requires_ridge_links() {
            validate_ridge_links(&self.tds)?;
        }
        // 1d. PL-manifold vertex-link condition during insertion (strict mode).
        if self
            .topology_guarantee
            .requires_vertex_links_during_insertion()
        {
            validate_vertex_links(&self.tds, &facet_to_cells)?;
        }

        // 2. Connectedness (single component in the cell neighbor graph).
        //
        // This is cheaper than Euler characteristic validation and catches cases where χ can
        // still match even though the triangulation is disconnected.
        self.validate_global_connectedness()?;

        // 3. Vertex incidence (manifold invariant): every vertex must be incident to at least one cell.
        self.validate_no_isolated_vertices()?;

        // 4. Euler characteristic using the topology module
        let topology_result =
            validate_triangulation_euler_with_facet_to_cells_map(&self.tds, &facet_to_cells);

        if let Some(expected) = topology_result.expected
            && topology_result.chi != expected
        {
            return Err(TriangulationValidationError::EulerCharacteristicMismatch {
                computed: topology_result.chi,
                expected,
                classification: topology_result.classification,
            });
        }

        Ok(())
    }

    /// Validates vertex-link condition at construction completion.
    ///
    /// This should be called once after batch construction is complete to certify
    /// full PL-manifoldness when using `TopologyGuarantee::PLManifold` (incremental mode).
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationError`] if vertex-link validation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert!(dt.as_triangulation().validate_at_completion().is_ok());
    /// ```
    pub fn validate_at_completion(&self) -> Result<(), TriangulationValidationError> {
        if !self
            .topology_guarantee
            .requires_vertex_links_at_completion()
        {
            return Ok(());
        }

        if self.tds.number_of_cells() == 0 {
            return Ok(());
        }

        let facet_to_cells: FacetToCellsMap = self.tds.build_facet_to_cells_map()?;
        validate_vertex_links(&self.tds, &facet_to_cells)?;
        Ok(())
    }
    /// Performs cumulative validation for Levels 1–3.
    ///
    /// This validates:
    /// - **Level 1–2** via [`Tds::validate`](crate::core::triangulation_data_structure::Tds::validate)
    /// - **Level 3** via [`Triangulation::is_valid`](Self::is_valid)
    /// - **Completion-time PL-manifold check** via [`Triangulation::validate_at_completion`](Self::validate_at_completion)
    ///
    /// # Errors
    ///
    /// Returns a [`TriangulationValidationError`] if:
    /// - Any vertex/cell is invalid (Level 1).
    /// - The TDS structural invariants fail (Level 2).
    /// - Topology validation fails (Level 3).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices_4d = [
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let dt: DelaunayTriangulation<_, (), (), 4> =
    ///     DelaunayTriangulation::new(&vertices_4d).unwrap();
    ///
    /// // Levels 1–3: elements + TDS structure + topology
    /// assert!(dt.as_triangulation().validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), TriangulationValidationError> {
        self.tds.validate()?;
        self.is_valid()?;
        self.validate_at_completion()
    }

    /// Generate a comprehensive validation report for Levels 1–3.
    ///
    /// This is intended for debugging/telemetry where you want to see *all* violated
    /// invariants, not just the first one.
    ///
    /// # Notes
    /// - If UUID↔key mappings are inconsistent, this returns only mapping failures (other
    ///   checks may produce misleading secondary errors).
    /// - This report is **cumulative** across Levels 1–3.
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationReport)` containing all invariant violations.
    pub(crate) fn validation_report(&self) -> Result<(), TriangulationValidationReport> {
        let mut violations: Vec<InvariantViolation> = Vec::new();

        // Level 2 (structural): reuse the TDS report.
        match self.tds.validation_report() {
            Ok(()) => {}
            Err(report) => {
                if report.violations.iter().any(|v| {
                    matches!(
                        v.kind,
                        InvariantKind::VertexMappings | InvariantKind::CellMappings
                    )
                }) {
                    return Err(report);
                }
                violations.extend(report.violations);
            }
        }

        // Level 1 (element validity): vertices
        for (_vertex_key, vertex) in self.tds.vertices() {
            if let Err(source) = (*vertex).is_valid() {
                violations.push(InvariantViolation {
                    kind: InvariantKind::VertexValidity,
                    error: InvariantError::Tds(TdsValidationError::InvalidVertex {
                        vertex_id: vertex.uuid(),
                        source,
                    }),
                });
            }
        }

        // Level 1 (element validity): cells
        for (_cell_key, cell) in self.tds.cells() {
            if let Err(source) = cell.is_valid() {
                violations.push(InvariantViolation {
                    kind: InvariantKind::CellValidity,
                    error: InvariantError::Tds(TdsValidationError::InvalidCell {
                        cell_id: cell.uuid(),
                        source,
                    }),
                });
            }
        }

        // Level 3 (topology)
        if let Err(e) = self.is_valid() {
            violations.push(InvariantViolation {
                kind: InvariantKind::Topology,
                error: e.into(),
            });
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(TriangulationValidationReport { violations })
        }
    }

    /// Validates that the triangulation's cell neighbor graph is a single connected component.
    ///
    /// This is an O(N·D) traversal (equivalently O(N+E) with bounded degree), where N is the
    /// number of cells and each cell has at most D+1 neighbors.
    fn validate_global_connectedness(&self) -> Result<(), TriangulationValidationError> {
        let total_cells = self.tds.number_of_cells();
        if total_cells == 0 {
            return Ok(());
        }

        let start = self.tds.cell_keys().next().ok_or_else(|| {
            TdsValidationError::InconsistentDataStructure {
                message: "Triangulation has non-zero cell count but no cell keys".to_string(),
            }
        })?;

        let visited = self.traverse_cell_neighbor_graph(start, total_cells, None, |_from, _to| {});

        if visited.len() != total_cells {
            return Err(TdsValidationError::InconsistentDataStructure {
                message: format!(
                    "Disconnected triangulation: visited {} of {} cells in the cell neighbor graph",
                    visited.len(),
                    total_cells
                ),
            }
            .into());
        }

        Ok(())
    }

    /// Validates that every vertex is incident to at least one cell.
    ///
    /// Isolated vertices are allowed at the TDS (structural) layer, but they violate the
    /// manifold invariants checked at the topology (Level 3) layer.
    fn validate_no_isolated_vertices(&self) -> Result<(), TriangulationValidationError> {
        if self.tds.number_of_vertices() == 0 {
            return Ok(());
        }

        let mut vertices_in_cells: FastHashSet<VertexKey> =
            fast_hash_set_with_capacity(self.tds.number_of_vertices());

        for (_cell_key, cell) in self.tds.cells() {
            for &vk in cell.vertices() {
                vertices_in_cells.insert(vk);
            }
        }

        for (vk, vertex) in self.tds.vertices() {
            if !vertices_in_cells.contains(&vk) {
                return Err(TdsValidationError::InconsistentDataStructure {
                    message: format!(
                        "Isolated vertex detected during topology validation: vertex {} (key {vk:?}) is not incident to any cell",
                        vertex.uuid()
                    ),
                }
                .into());
            }
        }

        Ok(())
    }
}

// =============================================================================
// Geometric Operations (Requires Numeric Scalar Bounds)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: AddAssign + SubAssign + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    /// Build initial D-simplex from D+1 vertices with degeneracy validation.
    ///
    /// This creates a Tds with a single cell containing all D+1 vertices,
    /// with no neighbor relationships (all boundary facets). The simplex is
    /// validated to ensure it is non-degenerate (vertices span full D-dimensional space).
    ///
    /// **Design Note**: This method uses `K::default()` to construct a kernel instance
    /// for the orientation test, relying on the design principle that kernels are stateless
    /// and reconstructible. If stateful kernels are introduced in the future, this method
    /// should accept an explicit kernel parameter instead.
    ///
    /// # Arguments
    /// - `vertices`: Exactly D+1 vertices to form the initial simplex
    ///
    /// # Returns
    /// A Tds containing one D-cell with all vertices, ready for incremental insertion.
    ///
    /// # Errors
    /// Returns error if:
    /// - Wrong number of vertices (must be exactly D+1)
    /// - Vertices are degenerate (collinear in 2D, coplanar in 3D, etc.)
    /// - Vertex or cell insertion fails
    /// - Duplicate UUIDs detected
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // Create a 2D triangle (initial simplex)
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
    /// assert_eq!(tds.number_of_vertices(), 3);
    /// assert_eq!(tds.number_of_cells(), 1);
    /// assert_eq!(tds.dim(), 2);
    ///
    /// // Error: wrong number of vertices (need exactly D+1)
    /// let bad_vertices = vec![vertex!([0.0, 0.0])];
    /// let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&bad_vertices);
    /// assert!(result.is_err());
    ///
    /// // Error: collinear points in 2D (degenerate simplex)
    /// let collinear = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([2.0, 0.0]),
    /// ];
    /// let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&collinear);
    /// assert!(result.is_err());
    /// ```
    pub fn build_initial_simplex(
        vertices: &[Vertex<K::Scalar, U, D>],
    ) -> Result<Tds<K::Scalar, U, V, D>, TriangulationConstructionError>
    where
        K::Scalar: CoordinateScalar,
    {
        if vertices.len() != D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: crate::core::cell::CellValidationError::InsufficientVertices {
                    actual: vertices.len(),
                    expected: D + 1,
                    dimension: D,
                },
            });
        }

        // Validate that the simplex is non-degenerate using orientation test
        // A degenerate simplex (collinear/coplanar) has zero orientation
        let kernel = K::default();

        // Collect points into stack-allocated buffer (at most 8 points for D ≤ 7)
        let points: SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            vertices.iter().map(|v| *v.point()).collect();

        // Check orientation - zero (0) means degenerate
        // orientation() returns -1 (negative), 0 (degenerate), or +1 (positive)
        let orientation = kernel.orientation(&points[..]).map_err(|e| {
            TriangulationConstructionError::FailedToCreateCell {
                message: format!("Orientation test failed: {e}"),
            }
        })?;

        if orientation == 0 {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Degenerate initial simplex: vertices are collinear/coplanar in {}D space. \
                     The {} input vertices do not span a full {}-dimensional simplex. \
                     Provide non-degenerate vertices to create a valid triangulation.",
                    D,
                    D + 1,
                    D
                ),
            });
        }

        // Create empty Tds
        let mut tds = Tds::empty();

        // Insert all vertices and collect their keys
        let mut vertex_keys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        for vertex in vertices {
            let vkey = tds.insert_vertex_with_mapping(*vertex)?;
            vertex_keys.push(vkey);
        }

        // Create single D-cell from all vertices
        // Note: Cell::new() handles vertex ordering/orientation internally
        let cell = Cell::new(vertex_keys, None).map_err(|e| {
            TriangulationConstructionError::FailedToCreateCell {
                message: format!("Failed to create initial simplex cell: {e}"),
            }
        })?;

        // Insert the cell
        let _cell_key = tds.insert_cell_with_mapping(cell)?;

        // Assign incident cells to vertices (each vertex points to this one cell)
        // This is required for proper Tds structure
        tds.assign_incident_cells()
            .map_err(|e| TdsConstructionError::ValidationError(e.into()))?;

        Ok(tds)
    }

    /// Insert a vertex into the triangulation using cavity-based algorithm.
    ///
    /// This is a generic insertion method that handles:
    /// - **Bootstrap (< D+1 vertices)**: Accumulates vertices without creating cells
    /// - **Initial simplex (D+1 vertices)**: Automatically builds the first D-cell
    /// - **Incremental (> D+1 vertices)**: Cavity-based insertion or hull extension
    ///
    /// # Arguments
    /// - `vertex`: The vertex to insert
    /// - `conflict_cells`: Optional conflict region (cells to be removed). Required for
    ///   interior points, not needed for exterior points (hull extension).
    /// - `hint`: Optional cell hint for point location (improves performance)
    ///
    /// # Algorithm
    /// 1. Insert vertex into Tds
    /// 2. Check vertex count:
    ///    - If < D+1: Return (bootstrap phase)
    ///    - If == D+1: Build initial simplex from all vertices
    ///    - If > D+1: Continue with steps 3-7
    /// 3. Locate cell containing the point
    /// 4. Handle location result:
    ///    - `InsideCell`: Use provided `conflict_cells` for cavity-based insertion
    ///    - `Outside`: Extend hull (no conflict cells needed)
    /// 5. Extract cavity boundary (if interior)
    /// 6. Fill cavity (create new cells)
    /// 7. Wire neighbors locally
    /// 8. Remove conflict cells (if interior)
    /// 9. Repair invalid facet sharing
    ///
    /// # Returns
    /// - `Ok(VertexKey)`: The key of the inserted vertex
    /// - New cell keys via the returned result (for hint caching at higher layers)
    ///
    /// # Errors
    /// Returns error if:
    /// - Duplicate coordinates detected (within 1e-10 tolerance)
    /// - Duplicate UUID detected
    /// - Initial simplex construction fails
    /// - Point location fails
    /// - Interior point without `conflict_cells` parameter
    /// - Cavity operations fail
    /// - Degenerate location (`OnFacet`, `OnEdge`, `OnVertex`) - not yet implemented
    ///
    /// **Note**: For insertions beyond D+1 vertices, use `DelaunayTriangulation::insert()`
    /// instead, which handles conflict region computation automatically.
    #[cfg(test)]
    pub(crate) fn insert(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_cells: Option<&CellKeyBuffer>,
        hint: Option<CellKey>,
    ) -> Result<(VertexKey, Option<CellKey>), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        // Use transactional insertion with a single local-scale perturbation retry.
        let (outcome, _stats) =
            self.insert_transactional(vertex, conflict_cells, hint, 1, 0, None)?;
        match outcome {
            InsertionOutcome::Inserted { vertex_key, hint } => Ok((vertex_key, hint)),
            InsertionOutcome::Skipped { error } => Err(error),
        }
    }

    /// Insert a vertex and return statistics about the operation.
    ///
    /// This method returns detailed statistics about the insertion including:
    /// - Number of attempts (perturbation retries)
    /// - Whether the vertex was skipped
    /// - Number of cells removed during repair
    ///
    /// This is useful for testing, debugging, and understanding how the
    /// triangulation handles geometric degeneracies.
    ///
    /// # Errors
    ///
    /// Returns an error only for non-retryable structural failures (e.g. duplicate UUID).
    /// Retryable geometric degeneracies that exhaust all attempts, and duplicate coordinates,
    /// return `Ok((InsertionOutcome::Skipped { .. }, stats))`.
    #[cfg(test)]
    pub(crate) fn insert_with_statistics(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_cells: Option<&CellKeyBuffer>,
        hint: Option<CellKey>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        // Single perturbation retry (local-scale) for geometric degeneracies.
        self.insert_transactional(vertex, conflict_cells, hint, 1, 0, None)
    }

    /// Insert a vertex with statistics, using a custom perturbation seed and an optional
    /// spatial hash-grid index.
    ///
    /// This is intended for bulk-construction paths that maintain a local index to
    /// accelerate duplicate detection and locate-hint selection.
    pub(in crate::core) fn insert_with_statistics_seeded_indexed(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_cells: Option<&CellKeyBuffer>,
        hint: Option<CellKey>,
        perturbation_seed: u64,
        index: Option<&mut HashGridIndex<K::Scalar, D>>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        self.insert_transactional(vertex, conflict_cells, hint, 1, perturbation_seed, index)
    }

    /// Transactional insertion with automatic rollback and perturbation retry.
    ///
    /// This ensures the triangulation always remains in a valid state by:
    /// 1. Cloning TDS before each insertion attempt (snapshot)
    /// 2. Attempting insertion
    /// 3. On failure: restore TDS from snapshot
    /// 4. If the error is retryable: perturb vertex and retry (up to `max_perturbation_attempts`)
    /// 5. If retryable attempts are exhausted, or the vertex is a duplicate: return
    ///    `Ok((InsertionOutcome::Skipped { error }, stats))`
    /// 6. If the error is non-retryable: return `Err(InsertionError)`
    ///
    /// This guarantees we transition from one valid manifold to another.
    #[expect(
        clippy::too_many_lines,
        reason = "Complex insertion logic; splitting further would harm readability"
    )]
    fn insert_transactional(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_cells: Option<&CellKeyBuffer>,
        hint: Option<CellKey>,
        max_perturbation_attempts: usize,
        perturbation_seed: u64,
        mut index: Option<&mut HashGridIndex<K::Scalar, D>>,
    ) -> Result<(InsertionOutcome, InsertionStatistics), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        let mut stats = InsertionStatistics::default();
        let original_coords = *vertex.point().coords();
        let original_uuid = vertex.uuid();
        let mut current_vertex = vertex;
        let mut last_retryable_error: Option<InsertionError> = None;

        let mut hint = hint;
        if hint.is_none()
            && let Some(index_ref) = index.as_deref()
        {
            hint = self.select_locate_hint_from_hash_grid(&original_coords, index_ref);
        }

        let local_scale = self.estimate_local_perturbation_scale(&original_coords, hint);

        let duplicate_tolerance: K::Scalar =
            <K::Scalar as NumCast>::from(1e-10_f64).unwrap_or_else(K::Scalar::default_tolerance);
        let duplicate_tolerance_sq = duplicate_tolerance * duplicate_tolerance;

        for attempt in 0..=max_perturbation_attempts {
            stats.attempts = attempt + 1;

            // Apply perturbation for retry attempts
            if attempt > 0 {
                let mut perturbed_coords = original_coords;
                // Single local-scale perturbation:
                // - f64: 1e-8 × local scale
                // - f32: 1e-4 × local scale
                let epsilon_value = if K::Scalar::mantissa_digits() <= 24 {
                    1e-4
                } else {
                    1e-8
                };
                let Some(epsilon) = <K::Scalar as NumCast>::from(epsilon_value) else {
                    // We failed to convert the perturbation scale into the scalar type.
                    //
                    // This should not happen for our supported scalar types (`f32`, `f64`), but if it
                    // does (e.g. with a custom scalar), we degrade gracefully by skipping this vertex
                    // rather than aborting the whole insertion.
                    stats.result = InsertionResult::SkippedDegeneracy;
                    let error = last_retryable_error.unwrap_or_else(|| InsertionError::CavityFilling {
                        message: format!(
                            "Failed to convert perturbation scale {epsilon_value} into scalar type"
                        ),
                    });
                    return Ok((InsertionOutcome::Skipped { error }, stats));
                };

                let perturbation_scale = epsilon * local_scale;
                for (idx, coord) in perturbed_coords.iter_mut().enumerate() {
                    let perturbation = if perturbation_seed == 0 {
                        if (attempt + idx) % 2 == 0 {
                            perturbation_scale
                        } else {
                            -perturbation_scale
                        }
                    } else {
                        let mix = perturbation_seed
                            ^ ((attempt as u64) << 32)
                            ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        if mix & 1 == 0 {
                            perturbation_scale
                        } else {
                            -perturbation_scale
                        }
                    };
                    *coord += perturbation;
                }

                // Preserve the caller-provided vertex UUID across perturbation retries.
                // This ensures the inserted vertex retains its original identity even if we have
                // to retry with perturbed coordinates.
                current_vertex =
                    Vertex::new_with_uuid(Point::new(perturbed_coords), original_uuid, vertex.data);
            }

            // Duplicate coordinate detection uses the hash grid when available; otherwise it
            // falls back to a linear scan (O(n·D) per insertion, O(n²·D) worst-case).
            if let Some(error) = self.duplicate_coordinates_error(
                current_vertex.point().coords(),
                duplicate_tolerance_sq,
                index.as_deref(),
            ) {
                stats.result = InsertionResult::SkippedDuplicate;
                #[cfg(debug_assertions)]
                tracing::debug!("SKIPPED: {error}");
                return Ok((InsertionOutcome::Skipped { error }, stats));
            }

            // Clone TDS for rollback (transactional semantics)
            let tds_snapshot = self.tds.clone();

            // Try insertion.
            //
            // Topology safety net: ensure we don't commit an insertion that breaks Level 3 topology.
            // If the cavity-based insertion produces an Euler/topology mismatch, roll back and retry a
            // conservative fallback (star-split of the containing cell) within the same transactional attempt.
            let result = self.try_insert_with_topology_safety_net(
                current_vertex,
                conflict_cells,
                hint,
                attempt,
                &tds_snapshot,
            );

            match result {
                Ok((result, cells_removed, _suspicion)) => {
                    stats.cells_removed_during_repair = cells_removed;
                    stats.result = InsertionResult::Inserted;
                    #[cfg(debug_assertions)]
                    if attempt > 0 {
                        tracing::debug!(
                            "Warning: Geometric degeneracy resolved via perturbation (attempt {attempt})"
                        );
                    }

                    let (vertex_key, hint) = result;
                    if let Some(index) = index.as_deref_mut()
                        && let Some(vertex) = self.tds.get_vertex_by_key(vertex_key)
                    {
                        index.insert_vertex(vertex_key, vertex.point().coords());
                    }

                    return Ok((InsertionOutcome::Inserted { vertex_key, hint }, stats));
                }
                Err(e) => {
                    // Any error - rollback to snapshot
                    self.tds = tds_snapshot;

                    // Handle duplicate coordinates specially - skip immediately without retry
                    if matches!(e, InsertionError::DuplicateCoordinates { .. }) {
                        stats.result = InsertionResult::SkippedDuplicate;
                        #[cfg(debug_assertions)]
                        tracing::debug!("SKIPPED: {e}");
                        return Ok((InsertionOutcome::Skipped { error: e }, stats));
                    }

                    // Check if this is a retryable error (geometric degeneracy)
                    let is_retryable = e.is_retryable();

                    if is_retryable && attempt < max_perturbation_attempts {
                        last_retryable_error = Some(e.clone());
                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            "RETRYING: Attempt {} failed with: {e}. Applying perturbation...",
                            attempt + 1
                        );
                    } else if is_retryable {
                        stats.result = InsertionResult::SkippedDegeneracy;
                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            "SKIPPED: Could not insert vertex after {} attempts (perturbations up to {:.1}%). Last error: {e}. Vertex skipped to maintain manifold.",
                            max_perturbation_attempts + 1,
                            match max_perturbation_attempts {
                                0 => 0.0,
                                1 => 0.01,
                                2 => 0.1,
                                3 => 1.0,
                                4 => 2.0,
                                5 => 5.0,
                                _ => 10.0,
                            }
                        );
                        return Ok((InsertionOutcome::Skipped { error: e }, stats));
                    } else {
                        // Non-retryable structural error (e.g., duplicate UUID)
                        return Err(e);
                    }
                }
            }
        }

        unreachable!("Loop should have returned in all cases");
    }

    fn select_locate_hint_from_hash_grid(
        &self,
        coords: &[K::Scalar; D],
        index: &HashGridIndex<K::Scalar, D>,
    ) -> Option<CellKey>
    where
        K::Scalar: CoordinateScalar,
    {
        let mut best: Option<(K::Scalar, CellKey)> = None;

        index.for_each_candidate_vertex_key(coords, |vkey| {
            let Some(vertex) = self.tds.get_vertex_by_key(vkey) else {
                return true;
            };

            let Some(cell_key) = vertex.incident_cell else {
                return true;
            };

            if !self.tds.contains_cell(cell_key) {
                return true;
            }

            let vcoords = vertex.point().coords();
            let mut dist_sq = K::Scalar::zero();
            for i in 0..D {
                let diff = vcoords[i] - coords[i];
                dist_sq += diff * diff;
            }

            match best {
                Some((best_dist, _)) if dist_sq >= best_dist => {}
                _ => {
                    best = Some((dist_sq, cell_key));
                }
            }

            true
        });

        best.map(|(_, cell_key)| cell_key)
    }

    /// Check for near-duplicate coordinates using the hash grid when available, with a
    /// linear-scan fallback (O(n·D) per insertion) if the index is unavailable/unusable.
    fn duplicate_coordinates_error(
        &self,
        coords: &[K::Scalar; D],
        tolerance_sq: K::Scalar,
        index: Option<&HashGridIndex<K::Scalar, D>>,
    ) -> Option<InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        let mut duplicate_found = false;
        let make_duplicate_error = || {
            let coord_str = coords
                .iter()
                .map(|c| format!("{c:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            InsertionError::DuplicateCoordinates {
                coordinates: format!("[{coord_str}]"),
            }
        };

        if let Some(index) = index {
            let mut candidate_count = 0usize;
            let used_index = index.for_each_candidate_vertex_key(coords, |vkey| {
                candidate_count = candidate_count.saturating_add(1);
                let Some(vertex) = self.tds.get_vertex_by_key(vkey) else {
                    return true;
                };

                let vcoords = vertex.point().coords();
                let mut dist_sq = K::Scalar::zero();
                for i in 0..D {
                    let diff = vcoords[i] - coords[i];
                    dist_sq += diff * diff;
                }

                if dist_sq < tolerance_sq {
                    duplicate_found = true;
                    return false;
                }

                true
            });
            record_duplicate_detection_metrics(used_index, candidate_count, !used_index);

            if duplicate_found {
                return Some(make_duplicate_error());
            }

            if used_index {
                return None;
            }
        } else {
            record_duplicate_detection_metrics(false, 0, true);
        }

        for (_, existing_vertex) in self.tds.vertices() {
            let existing_coords = existing_vertex.point().coords();
            let mut dist_sq = K::Scalar::zero();
            for i in 0..D {
                let diff = coords[i] - existing_coords[i];
                dist_sq += diff * diff;
            }

            if dist_sq < tolerance_sq {
                duplicate_found = true;
                break;
            }
        }

        if duplicate_found {
            Some(make_duplicate_error())
        } else {
            None
        }
    }

    /// Estimate a local length scale for perturbation based on nearby vertices.
    ///
    /// Uses the hint cell when available; otherwise falls back to the closest
    /// existing vertex. This keeps perturbations translation-invariant and
    /// proportional to local feature size.
    fn estimate_local_perturbation_scale(
        &self,
        coords: &[K::Scalar; D],
        hint: Option<CellKey>,
    ) -> K::Scalar
    where
        K::Scalar: CoordinateScalar,
    {
        let mut min_dist_sq: Option<K::Scalar> = None;

        let consider_vertex = |vertex: &Vertex<K::Scalar, U, D>,
                               min_dist_sq: &mut Option<K::Scalar>| {
            let vcoords = vertex.point().coords();
            let mut dist_sq = K::Scalar::zero();
            for i in 0..D {
                let diff = vcoords[i] - coords[i];
                dist_sq += diff * diff;
            }
            match min_dist_sq {
                Some(current) => {
                    if dist_sq < *current {
                        *current = dist_sq;
                    }
                }
                None => {
                    *min_dist_sq = Some(dist_sq);
                }
            }
        };

        if let Some(cell_key) = hint
            && let Some(cell) = self.tds.get_cell(cell_key)
        {
            for &vkey in cell.vertices() {
                if let Some(vertex) = self.tds.get_vertex_by_key(vkey) {
                    consider_vertex(vertex, &mut min_dist_sq);
                }
            }
        }

        if min_dist_sq.is_none() {
            for (_, vertex) in self.tds.vertices() {
                consider_vertex(vertex, &mut min_dist_sq);
            }
        }

        let mut scale = min_dist_sq.map_or_else(K::Scalar::one, num_traits::Float::sqrt);

        let min_scale = K::Scalar::default_tolerance();
        if scale < min_scale {
            scale = min_scale;
        }

        scale
    }

    // -------------------------------------------------------------------------
    // Topology safety net helpers
    // -------------------------------------------------------------------------

    /// Logs when Level 3 validation is triggered (debug builds only).
    #[inline]
    fn log_validation_trigger_if_enabled(&self, suspicion: SuspicionFlags) {
        #[cfg(debug_assertions)]
        if self.validation_policy.should_validate(suspicion) && suspicion.is_suspicious() {
            tracing::debug!("Validation triggered by {suspicion:?}");
        }

        // Keep the parameter "used" in release builds where the debug-only logging
        // is compiled out, so `cargo clippy -D warnings` stays clean across profiles.
        #[cfg(not(debug_assertions))]
        {
            let _ = suspicion;
        }
    }

    /// Runs mandatory link checks required by the topology guarantee.
    fn validate_required_topology_links(&self) -> Result<(), TriangulationValidationError> {
        if self.tds.number_of_cells() == 0 {
            return Ok(());
        }

        if self
            .topology_guarantee
            .requires_vertex_links_during_insertion()
        {
            let facet_to_cells: FacetToCellsMap = self.tds.build_facet_to_cells_map()?;
            validate_facet_degree(&facet_to_cells).map_err(TriangulationValidationError::from)?;
            validate_closed_boundary(&self.tds, &facet_to_cells)
                .map_err(TriangulationValidationError::from)?;
            validate_ridge_links(&self.tds).map_err(TriangulationValidationError::from)?;
            validate_vertex_links(&self.tds, &facet_to_cells)
                .map_err(TriangulationValidationError::from)?;
            return Ok(());
        }

        if self.topology_guarantee.requires_ridge_links() {
            // Ridge-link checks assume the pseudomanifold invariants already hold.
            let facet_to_cells: FacetToCellsMap = self.tds.build_facet_to_cells_map()?;
            validate_facet_degree(&facet_to_cells).map_err(TriangulationValidationError::from)?;
            validate_closed_boundary(&self.tds, &facet_to_cells)
                .map_err(TriangulationValidationError::from)?;
            validate_ridge_links(&self.tds).map_err(TriangulationValidationError::from)?;
        }

        Ok(())
    }

    fn validate_after_insertion(
        &self,
        suspicion: SuspicionFlags,
    ) -> Result<(), TriangulationValidationError> {
        if self.tds.number_of_cells() == 0 {
            return Ok(());
        }

        let should_validate = self.validation_policy.should_validate(suspicion);
        let requires_link_checks = self.topology_guarantee.requires_ridge_links()
            || self
                .topology_guarantee
                .requires_vertex_links_during_insertion();

        if !should_validate && !requires_link_checks {
            return Ok(());
        }

        self.log_validation_trigger_if_enabled(suspicion);
        if should_validate {
            self.is_valid()
        } else {
            self.validate_required_topology_links()
        }
    }

    /// Attempt an insertion, and if Level 3 validation fails, roll back and try a
    /// conservative star-split fallback of the containing cell.
    fn try_insert_with_topology_safety_net(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_cells: Option<&CellKeyBuffer>,
        hint: Option<CellKey>,
        attempt: usize,
        tds_snapshot: &Tds<K::Scalar, U, V, D>,
    ) -> Result<TryInsertImplOk, InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        let (ok, cells_removed, mut suspicion) =
            self.try_insert_impl(vertex, conflict_cells, hint)?;

        if attempt > 0 {
            suspicion.perturbation_used = true;
        }

        // Skip Level 3 validation during bootstrap (vertices but no cells yet).
        if self.tds.number_of_cells() == 0 {
            return Ok((ok, cells_removed, suspicion));
        }

        if let Err(validation_err) = self.validate_after_insertion(suspicion) {
            // Roll back to snapshot and attempt a star-split fallback for interior points.
            self.tds = tds_snapshot.clone();
            return self.try_star_split_fallback_after_topology_failure(
                vertex,
                hint,
                attempt,
                &validation_err,
            );
        }

        Ok((ok, cells_removed, suspicion))
    }

    /// After a Level 3 topology validation failure, try to recover by performing a star-split
    /// of the containing cell (if the point can be re-located inside a cell).
    ///
    /// Notes:
    /// - This fallback is only applicable when the point re-locates to [`LocateResult::InsideCell`].
    /// - We re-run Level 3 validation after the fallback to avoid "recovering" into an invalid state.
    fn try_star_split_fallback_after_topology_failure(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        hint: Option<CellKey>,
        attempt: usize,
        validation_err: &TriangulationValidationError,
    ) -> Result<TryInsertImplOk, InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        let point = *vertex.point();
        let location = locate(&self.tds, &self.kernel, &point, hint);

        let Ok(LocateResult::InsideCell(start_cell)) = location else {
            return Err(InsertionError::TopologyValidationFailed {
                message: "Topology invalid after insertion; star-split fallback requires point to re-locate inside a cell"
                    .to_string(),
                source: validation_err.clone(),
            });
        };

        let mut star_conflict = CellKeyBuffer::new();
        star_conflict.push(start_cell);

        match self.try_insert_impl(vertex, Some(&star_conflict), Some(start_cell)) {
            Ok((fallback_ok, fallback_removed, mut fallback_suspicion)) => {
                fallback_suspicion.fallback_star_split = true;
                if attempt > 0 {
                    fallback_suspicion.perturbation_used = true;
                }

                if let Err(fallback_validation_err) =
                    self.validate_after_insertion(fallback_suspicion)
                {
                    return Err(InsertionError::TopologyValidationFailed {
                        message: "Topology invalid after star-split fallback".to_string(),
                        source: fallback_validation_err,
                    });
                }

                // Telemetry: the fallback succeeded, meaning we recovered from a topology
                // validation failure without surfacing an insertion error to the caller.
                TOPOLOGY_SAFETY_NET_STAR_SPLIT_FALLBACK_SUCCESSES.fetch_add(1, Ordering::Relaxed);

                #[cfg(debug_assertions)]
                tracing::debug!(
                    "Topology safety-net: star-split fallback succeeded (start_cell={start_cell:?})"
                );

                Ok((fallback_ok, fallback_removed, fallback_suspicion))
            }
            Err(fallback_err) => Err(InsertionError::TopologyValidationFailed {
                message: format!(
                    "Topology invalid after insertion; star-split fallback failed: {fallback_err}"
                ),
                source: validation_err.clone(),
            }),
        }
    }

    /// Ensure an interior insertion never proceeds with an empty conflict region.
    ///
    /// An empty conflict region would produce an empty cavity boundary, create no new cells, and
    /// leave the inserted vertex isolated (not incident to any cell), which breaks Level 3 topology
    /// validation via Euler characteristic.
    fn ensure_non_empty_conflict_cells(
        conflict_cells: Cow<'_, CellKeyBuffer>,
        fallback_cell: CellKey,
    ) -> Cow<'_, CellKeyBuffer> {
        if !conflict_cells.is_empty() {
            return conflict_cells;
        }

        if let Cow::Owned(mut owned) = conflict_cells {
            owned.push(fallback_cell);
            Cow::Owned(owned)
        } else {
            let mut owned = CellKeyBuffer::new();
            owned.push(fallback_cell);
            Cow::Owned(owned)
        }
    }

    /// Build the boundary facets for a "star-split" of the containing cell.
    fn star_split_boundary_facets(start_cell: CellKey) -> CavityBoundaryBuffer {
        (0..=D)
            .map(|i| {
                FacetHandle::new(
                    start_cell,
                    u8::try_from(i).expect("facet index must fit in u8"),
                )
            })
            .collect()
    }

    /// Connectedness guard (localized).
    ///
    /// This check is designed to be **O(k·D)**, where `k` is the number of newly created cells and
    /// `D` is the triangulation dimension (each cell has at most `D+1` neighbors).
    ///
    /// It validates two properties that are sufficient to catch the common “disconnected neighbor
    /// graph after insertion” failure modes without walking the entire triangulation:
    ///
    /// 1. The surviving subset of `new_cells` forms a single connected component (via neighbor pointers).
    /// 2. If there are cells outside that component, the new component is attached to at least one
    ///    existing cell (via a *mutual* neighbor relationship).
    fn validate_connectedness(&self, new_cells: &CellKeyBuffer) -> Result<(), InsertionError> {
        let total_cells = self.tds.number_of_cells();
        if total_cells == 0 {
            return Ok(());
        }

        // Build a set of the *surviving* new cells (some may have been removed during repair).
        let mut new_set: CellKeySet = CellKeySet::default();
        new_set.reserve(new_cells.len());
        for &ck in new_cells {
            if self.tds.contains_cell(ck) {
                new_set.insert(ck);
            }
        }

        if new_set.is_empty() {
            return Err(InsertionError::TopologyValidation(
                TdsValidationError::InconsistentDataStructure {
                    message: "Disconnected triangulation detected after insertion: no surviving new cells"
                        .to_string(),
                },
            ));
        }

        let expected_new_cells = new_set.len();

        let start = *new_set
            .iter()
            .next()
            .expect("new_set is non-empty by construction");

        let mut touches_existing_cells = false;

        let visited = self.traverse_cell_neighbor_graph(
            start,
            expected_new_cells,
            Some(&new_set),
            |ck, nk| {
                if touches_existing_cells {
                    return;
                }

                // For connectivity between new cells and existing cells, require *mutual* adjacency.
                // This avoids treating one-way neighbor pointers as “connected”.
                if let Some(neighbor_cell) = self.tds.get_cell(nk)
                    && neighbor_cell
                        .neighbors()
                        .is_some_and(|ns| ns.contains(&Some(ck)))
                {
                    touches_existing_cells = true;
                }
            },
        );

        if visited.len() != expected_new_cells {
            return Err(InsertionError::TopologyValidation(
                TdsValidationError::InconsistentDataStructure {
                    message: format!(
                        "Disconnected triangulation detected after insertion: new-cell subgraph visited {} of {} cells",
                        visited.len(),
                        expected_new_cells
                    ),
                },
            ));
        }

        // If there are cells outside `new_set`, ensure the new component is attached to at least one
        // of them (otherwise we'd be creating a disconnected component).
        if total_cells > expected_new_cells && !touches_existing_cells {
            return Err(InsertionError::TopologyValidation(
                TdsValidationError::InconsistentDataStructure {
                    message: format!(
                        "Disconnected triangulation detected after insertion: new-cell component ({expected_new_cells} cells) is not connected to existing cells (total_cells={total_cells})"
                    ),
                },
            ));
        }

        Ok(())
    }

    /// Find all conflict cells by scanning the entire triangulation.
    ///
    /// This is used for exterior points where the standard BFS conflict-region
    /// search lacks a guaranteed seed cell inside the conflict region.
    fn find_conflict_region_global(
        &self,
        point: &Point<K::Scalar, D>,
    ) -> Result<CellKeyBuffer, ConflictError>
    where
        K::Scalar: CoordinateScalar,
    {
        #[cfg(debug_assertions)]
        let log_enabled = std::env::var_os("DELAUNAY_DEBUG_HULL").is_some()
            || std::env::var_os("DELAUNAY_DEBUG_CONFLICT").is_some();
        #[cfg(debug_assertions)]
        let mut cells_scanned = 0usize;
        #[cfg(debug_assertions)]
        let mut sign_positive = 0usize;
        #[cfg(debug_assertions)]
        let mut sign_zero = 0usize;
        #[cfg(debug_assertions)]
        let mut sign_negative = 0usize;

        let mut conflict_cells = CellKeyBuffer::new();

        for (cell_key, cell) in self.tds.cells() {
            #[cfg(debug_assertions)]
            {
                cells_scanned = cells_scanned.saturating_add(1);
            }
            let simplex_points: SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
                cell.vertices()
                    .iter()
                    .filter_map(|&vkey| self.tds.get_vertex_by_key(vkey).map(|v| *v.point()))
                    .collect();

            if simplex_points.len() != D + 1 {
                return Err(ConflictError::CellDataAccessFailed {
                    cell_key,
                    message: format!("Expected {} vertices, got {}", D + 1, simplex_points.len()),
                });
            }

            let sign = self.kernel.in_sphere(&simplex_points, point)?;
            #[cfg(debug_assertions)]
            {
                if log_enabled {
                    tracing::debug!(
                        cell_key = ?cell_key,
                        sign,
                        "find_conflict_region_global: in_sphere sign"
                    );
                }
                match sign.cmp(&0) {
                    CmpOrdering::Greater => {
                        sign_positive = sign_positive.saturating_add(1);
                    }
                    CmpOrdering::Equal => {
                        sign_zero = sign_zero.saturating_add(1);
                    }
                    CmpOrdering::Less => {
                        sign_negative = sign_negative.saturating_add(1);
                    }
                }
            }
            if sign > 0 {
                conflict_cells.push(cell_key);
            }
        }

        #[cfg(debug_assertions)]
        if log_enabled {
            tracing::debug!(
                point = ?point,
                cells_scanned,
                conflict_cells = conflict_cells.len(),
                sign_positive,
                sign_zero,
                sign_negative,
                "find_conflict_region_global: summary"
            );
        }

        Ok(conflict_cells)
    }

    /// Returns true if any conflict cell has a facet on the hull boundary.
    fn conflict_region_touches_boundary(
        &self,
        conflict_cells: &CellKeyBuffer,
    ) -> Result<bool, InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        if conflict_cells.is_empty() {
            return Ok(false);
        }

        let facet_to_cells = self
            .tds
            .build_facet_to_cells_map()
            .map_err(InsertionError::TopologyValidation)?;

        let mut boundary_facets: FastHashSet<u64> =
            fast_hash_set_with_capacity(facet_to_cells.len());
        for (facet_key, cell_list) in &facet_to_cells {
            if cell_list.len() == 1 {
                boundary_facets.insert(*facet_key);
            }
        }

        if boundary_facets.is_empty() {
            return Ok(false);
        }

        for &cell_key in conflict_cells {
            let cell = self.tds.get_cell(cell_key).ok_or_else(|| {
                InsertionError::TopologyValidation(TdsValidationError::InconsistentDataStructure {
                    message: format!(
                        "Conflict cell {cell_key:?} not found while checking boundary facets"
                    ),
                })
            })?;
            for facet_idx in 0..cell.number_of_vertices() {
                let mut facet_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    SmallBuffer::with_capacity(D);
                for (i, &vkey) in cell.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vertices.push(vkey);
                    }
                }
                let facet_key = facet_key_from_vertices(&facet_vertices);
                if boundary_facets.contains(&facet_key) {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Perform cavity insertion given an explicit conflict region.
    #[expect(
        clippy::too_many_lines,
        reason = "Keep cavity insertion and repair logic together for clarity"
    )]
    fn insert_with_conflict_region(
        &mut self,
        v_key: VertexKey,
        point: &Point<K::Scalar, D>,
        mut conflict_cells: CellKeyBuffer,
        fallback_cell: Option<CellKey>,
        suspicion: &mut SuspicionFlags,
    ) -> Result<(Option<CellKey>, usize), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        if conflict_cells.is_empty() {
            let Some(start_cell) = fallback_cell else {
                return Err(InsertionError::CavityFilling {
                    message: "Empty conflict region for exterior insertion".to_string(),
                });
            };
            suspicion.empty_conflict_region = true;
            suspicion.fallback_star_split = true;
            conflict_cells.push(start_cell);
        }

        // Extract cavity boundary
        let mut boundary_facets = match extract_cavity_boundary(&self.tds, &conflict_cells) {
            Ok(boundary) => boundary,
            Err(err) => {
                let should_fallback = matches!(
                    err,
                    ConflictError::NonManifoldFacet { .. }
                        | ConflictError::RidgeFan { .. }
                        | ConflictError::DisconnectedBoundary { .. }
                        | ConflictError::OpenBoundary { .. }
                );

                if should_fallback {
                    let Some(start_cell) = fallback_cell else {
                        return Err(err.into());
                    };

                    suspicion.fallback_star_split = true;

                    #[cfg(debug_assertions)]
                    tracing::warn!(
                        "Conflict region degeneracy ({err}); falling back to star-split of cell {start_cell:?}"
                    );

                    conflict_cells = {
                        let mut owned = CellKeyBuffer::new();
                        owned.push(start_cell);
                        owned
                    };

                    Self::star_split_boundary_facets(start_cell)
                } else {
                    return Err(err.into());
                }
            }
        };

        // Fallback: never allow an insertion to create a dangling vertex.
        if boundary_facets.is_empty() {
            let Some(start_cell) = fallback_cell else {
                return Err(InsertionError::CavityFilling {
                    message: "Empty cavity boundary for exterior insertion".to_string(),
                });
            };

            suspicion.empty_conflict_region = true;
            suspicion.fallback_star_split = true;

            #[cfg(debug_assertions)]
            tracing::warn!(
                "Empty cavity boundary; falling back to splitting containing cell {start_cell:?}"
            );

            conflict_cells = {
                let mut owned = CellKeyBuffer::new();
                owned.push(start_cell);
                owned
            };
            boundary_facets = Self::star_split_boundary_facets(start_cell);
        }

        // Fill cavity BEFORE removing old cells
        let new_cells = fill_cavity(&mut self.tds, v_key, &boundary_facets)?;

        // Wire neighbors (while both old and new cells exist)
        wire_cavity_neighbors(&mut self.tds, &new_cells, Some(&conflict_cells))?;

        // Remove conflict cells (now that new cells are wired up)
        let _removed_count = self.tds.remove_cells_by_keys(&conflict_cells);

        // Iteratively repair non-manifold topology until facet sharing is valid
        let mut total_removed = 0;
        #[cfg_attr(
            not(debug_assertions),
            expect(
                unused_variables,
                reason = "`iteration` is only used for debug logging",
            )
        )]
        for iteration in 0..MAX_REPAIR_ITERATIONS {
            // Check for non-manifold issues in newly created cells (local scan)
            let cells_to_check: CellKeyBuffer = new_cells
                .iter()
                .copied()
                .filter(|ck| self.tds.contains_cell(*ck))
                .collect();

            if let Some(issues) = self.detect_local_facet_issues(&cells_to_check)? {
                // Only mark this as "suspicious" if we *actually* detected local facet issues
                // and entered the repair path.
                suspicion.repair_loop_entered = true;

                #[cfg(debug_assertions)]
                tracing::debug!(
                    "Repair iteration {}: {} over-shared facets detected, removing cells...",
                    iteration + 1,
                    issues.len()
                );

                let removed = self.repair_local_facet_issues(&issues)?;

                // Early exit if repair made no progress
                if removed == 0 {
                    #[cfg(debug_assertions)]
                    tracing::warn!(
                        "No cells removed in iteration {} - repair cannot make progress",
                        iteration + 1
                    );
                    return Err(InsertionError::TopologyValidation(
                        TdsValidationError::InconsistentDataStructure {
                            message: format!(
                                "Repair stalled: {} over-shared facets remain but no cells could be removed",
                                issues.len()
                            ),
                        },
                    ));
                }

                total_removed += removed;

                if removed > 0 {
                    suspicion.cells_removed = true;
                }

                #[cfg(debug_assertions)]
                tracing::debug!("Removed {removed} cells (total: {total_removed})");

                // Early exit if repair succeeded
                if self.tds.validate_facet_sharing().is_ok() {
                    break;
                }
            } else {
                // No more non-manifold issues - safe to rebuild neighbors
                break;
            }
        }

        // Rebuild neighbor pointers now that topology is manifold
        #[cfg(debug_assertions)]
        tracing::debug!("After repair loop: total_removed={total_removed}");

        // After insertion we ALWAYS removed the conflict region, which can leave broken/None neighbor
        // pointers in surviving cells. Even if the subsequent non-manifold repair loop removed 0 cells,
        // we still must repair neighbor pointers to ensure the cavity is glued.
        let facet_valid = self.tds.validate_facet_sharing().is_ok();
        #[cfg(debug_assertions)]
        tracing::debug!(
            "Before repair_neighbor_pointers: facet_sharing_valid={facet_valid}, cells={}",
            self.tds.number_of_cells()
        );

        if !facet_valid {
            return Err(InsertionError::CavityFilling {
                message:
                    "Facet sharing invalid after insertion/repairs - cannot safely repair neighbors"
                        .to_string(),
            });
        }

        // Surgical reconstruction: fix broken/None pointers by facet matching.
        let repaired =
            repair_neighbor_pointers(&mut self.tds).map_err(|e| InsertionError::CavityFilling {
                message: format!("Failed to repair neighbor pointers after insertion: {e}"),
            })?;
        suspicion.neighbor_pointers_rebuilt = repaired > 0;

        // Validate neighbor pointers by forcing a full facet walk (no hint).
        let _ = locate(&self.tds, &self.kernel, point, None)?;

        // Always rebuild vertex→cell incidence after insertion.
        self.tds
            .assign_incident_cells()
            .map_err(|e| InsertionError::CavityFilling {
                message: format!("Failed to assign incident cells after insertion: {e}"),
            })?;

        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some() {
            match validate_ridge_links(&self.tds) {
                Ok(()) => {
                    tracing::debug!(
                        "insert_with_conflict_region: ridge-link validation passed after insertion"
                    );
                }
                Err(err) => {
                    tracing::warn!(
                        error = ?err,
                        "insert_with_conflict_region: ridge-link validation failed after insertion"
                    );
                }
            }
        }

        // If any vertex is not incident to a cell, topology is not a pure ball anymore.
        let isolated_count = self
            .tds
            .vertices()
            .filter(|(_, v)| v.incident_cell.is_none())
            .count();
        if isolated_count > 0 {
            #[cfg(debug_assertions)]
            if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                tracing::warn!(
                    isolated_count,
                    "insert_with_conflict_region: isolated vertices detected after insertion"
                );
            }
            return Err(InsertionError::TopologyValidation(
                TdsValidationError::InconsistentDataStructure {
                    message: "Isolated vertex detected after insertion (vertex not in any cell)"
                        .to_string(),
                },
            ));
        }

        // Connectedness guard (STRUCTURAL SAFETY, NOT Level 3 validation)
        self.validate_connectedness(&new_cells)?;

        // Return hint for next insertion
        let hint = new_cells
            .iter()
            .copied()
            .find(|ck| self.tds.contains_cell(*ck));
        Ok((hint, total_removed))
    }
    /// Internal implementation of insert without retry logic.
    /// Returns the result and the number of cells removed during repair.
    ///
    /// Note: `conflict_cells` parameter is optional. If `None`, it will be computed automatically
    /// for interior points using `locate()` + `find_conflict_region()`.
    #[expect(
        clippy::too_many_lines,
        reason = "Complex insertion logic; splitting further would harm readability"
    )]
    fn try_insert_impl(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_cells: Option<&CellKeyBuffer>,
        hint: Option<CellKey>,
    ) -> Result<TryInsertImplOk, InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        let mut suspicion = SuspicionFlags::default();

        // CRITICAL: Capture UUID and point BEFORE inserting into TDS
        // Rationale:
        // - inserted_uuid: Needed to remap v_key after TDS rebuild (lines 736-744)
        //   when building initial simplex. The rebuild replaces self.tds entirely,
        //   invalidating all previous VertexKeys.
        // - point: Needed for locate(), find_conflict_region(), and extend_hull() calls
        //   (lines 752, 760, 879, 895). After TDS rebuild, we cannot access the vertex
        //   via the old v_key, so we must have the point value captured.
        let inserted_uuid = vertex.uuid();
        let point = *vertex.point();

        // 1. Insert vertex into Tds
        let mut v_key = self
            .tds
            .insert_vertex_with_mapping(vertex)
            .map_err(TriangulationConstructionError::from)?;

        // 2. Check if we need to bootstrap the initial simplex
        let num_vertices = self.tds.number_of_vertices();

        if num_vertices < D + 1 {
            // Bootstrap phase: just accumulate vertices, no cells yet
            return Ok(((v_key, None), 0, suspicion));
        } else if num_vertices == D + 1 {
            // Build initial simplex from all D+1 vertices
            let all_vertices: Vec<_> = self.tds.vertices().map(|(_, v)| *v).collect();
            let new_tds = Self::build_initial_simplex(&all_vertices).map_err(|e| {
                InsertionError::CavityFilling {
                    message: format!("Failed to build initial simplex: {e}"),
                }
            })?;

            // Replace empty TDS with simplex TDS (preserve kernel)
            self.tds = new_tds;

            // Re-map vertex key to the rebuilt TDS
            v_key = self
                .tds
                .vertex_key_from_uuid(&inserted_uuid)
                .ok_or_else(|| InsertionError::CavityFilling {
                    message: "Inserted vertex not found in rebuilt TDS".to_string(),
                })?;

            // Return first cell key for hint caching
            let first_cell = self.tds.cell_keys().next();
            return Ok(((v_key, first_cell), 0, suspicion));
        }

        // 3. Locate containing cell (for vertex D+2 and beyond)
        #[cfg(debug_assertions)]
        let (location, locate_stats) = {
            #[cfg(debug_assertions)]
            {
                let log_locate = std::env::var_os("DELAUNAY_DEBUG_HULL").is_some()
                    || std::env::var_os("DELAUNAY_DEBUG_LOCATE").is_some();
                if log_locate {
                    let (location, stats) =
                        locate_with_stats(&self.tds, &self.kernel, &point, hint)?;
                    (location, Some(stats))
                } else {
                    (locate(&self.tds, &self.kernel, &point, hint)?, None)
                }
            }
            #[cfg(not(debug_assertions))]
            {
                (locate(&self.tds, &self.kernel, &point, hint)?, None)
            }
        };

        #[cfg(not(debug_assertions))]
        let location = locate(&self.tds, &self.kernel, &point, hint)?;

        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some()
            || std::env::var_os("DELAUNAY_DEBUG_LOCATE").is_some()
        {
            if let Some(stats) = locate_stats {
                tracing::debug!(
                    point = ?point,
                    location = ?location,
                    start_cell = ?stats.start_cell,
                    used_hint = stats.used_hint,
                    walk_steps = stats.walk_steps,
                    fallback = ?stats.fallback,
                    "try_insert_impl: locate stats"
                );
            } else {
                tracing::debug!(
                    point = ?point,
                    location = ?location,
                    "try_insert_impl: locate result"
                );
            }
        }

        // 4. Determine conflict cells (for interior points)
        let conflict_cells = match (location, conflict_cells) {
            (LocateResult::InsideCell(start_cell), None) => {
                // Interior point: compute conflict region automatically.
                //
                // IMPORTANT:
                // `find_conflict_region()` (Bowyer–Watson style) can legitimately return an empty
                // set when the point lies inside the triangulation but is not strictly inside any
                // existing cell circumsphere (e.g., obtuse tetrahedra whose circumsphere does not
                // contain all interior points).
                //
                // An empty conflict region would produce an empty cavity boundary, create no new
                // cells, and leave the inserted vertex isolated (not incident to any cell), which
                // breaks Level 3 topology validation via Euler characteristic.
                //
                // Fallback: treat the containing cell as the conflict region, effectively performing
                // a star-split of that cell to keep the simplicial complex connected.
                let computed = find_conflict_region(&self.tds, &self.kernel, &point, start_cell)?;
                if computed.is_empty() {
                    suspicion.empty_conflict_region = true;
                    suspicion.fallback_star_split = true;
                }
                Some(Self::ensure_non_empty_conflict_cells(
                    Cow::Owned(computed),
                    start_cell,
                ))
            }
            (LocateResult::InsideCell(start_cell), Some(cells)) => {
                // If the caller provided an empty conflict region (can happen if the Delaunay layer
                // computes conflicts using a strict in-sphere test), we must still replace at least
                // one cell; otherwise we'd create no cavity, no new cells, and leave a dangling
                // vertex (χ increases by 1, typically showing up as χ=2 for Ball(3)).
                if cells.is_empty() {
                    suspicion.empty_conflict_region = true;
                    suspicion.fallback_star_split = true;
                }
                Some(Self::ensure_non_empty_conflict_cells(
                    Cow::Borrowed(cells),
                    start_cell,
                ))
            }
            (LocateResult::Outside, None) => {
                // 2D exterior insertions skip the global conflict-region scan and go straight to
                // hull extension, which is cheaper and more reliable in 2D. For D>2 we attempt
                // cavity insertion first using a global conflict scan.
                if D == 2 {
                    #[cfg(debug_assertions)]
                    tracing::debug!(
                        "Outside insertion in 2D: skipping global conflict-region scan; using hull extension"
                    );
                    None
                } else {
                    #[cfg(debug_assertions)]
                    if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                        tracing::debug!("Outside insertion: starting global conflict-region scan");
                    }
                    let computed = self.find_conflict_region_global(&point)?;
                    if computed.is_empty() {
                        #[cfg(debug_assertions)]
                        if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                            tracing::debug!(
                                "Outside insertion: global conflict region empty; will use hull extension"
                            );
                        }
                        None
                    } else if self.conflict_region_touches_boundary(&computed)? {
                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            "Outside insertion (D={D}) conflict region touches hull; skipping cavity insertion"
                        );
                        // Avoid cavity insertion when the conflict region touches the hull.
                        // These mixed boundaries can yield ridge-link singularities in higher dimensions.
                        None
                    } else {
                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            "Outside insertion (D={D}) using global conflict region with {} cells",
                            computed.len()
                        );
                        Some(Cow::Owned(computed))
                    }
                }
            }
            (LocateResult::Outside, Some(cells)) => {
                if cells.is_empty() {
                    #[cfg(debug_assertions)]
                    if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                        tracing::debug!(
                            "Outside insertion: caller provided empty conflict region; will use hull extension"
                        );
                    }
                    None
                } else {
                    #[cfg(debug_assertions)]
                    if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                        tracing::debug!(
                            conflict_cells = cells.len(),
                            "Outside insertion: using caller-provided conflict region"
                        );
                    }
                    Some(Cow::Borrowed(cells))
                }
            }
            (location, _) => {
                // Degenerate locations (OnFacet, OnEdge, OnVertex)
                return Err(InsertionError::CavityFilling {
                    message: format!(
                        "Unhandled degenerate location: {location:?}. Point lies on facet/edge/vertex which is not yet supported."
                    ),
                });
            }
        };

        // 5. Handle different location results
        match location {
            LocateResult::InsideCell(start_cell) => {
                let conflict_cells = conflict_cells
                    .expect("conflict_cells should be computed above")
                    .into_owned();
                let (hint, total_removed) = self.insert_with_conflict_region(
                    v_key,
                    &point,
                    conflict_cells,
                    Some(start_cell),
                    &mut suspicion,
                )?;
                Ok(((v_key, hint), total_removed, suspicion))
            }
            LocateResult::Outside => {
                if let Some(conflict_cells) = conflict_cells {
                    let conflict_cells = conflict_cells.into_owned();
                    #[cfg(debug_assertions)]
                    let conflict_len = conflict_cells.len();
                    #[cfg(debug_assertions)]
                    tracing::debug!(
                        "Outside insertion attempting cavity insertion with conflict region size {conflict_len}"
                    );
                    let result = self.insert_with_conflict_region(
                        v_key,
                        &point,
                        conflict_cells,
                        None,
                        &mut suspicion,
                    );
                    match result {
                        Ok((hint, total_removed)) => {
                            return Ok(((v_key, hint), total_removed, suspicion));
                        }
                        Err(err) => {
                            // For exterior points, a "global" conflict region can intersect the hull,
                            // producing an open/disconnected cavity boundary. In these cases we fall back
                            // to hull extension instead of surfacing an insertion error.
                            let fallback_on_isolated_vertex = matches!(
                                &err,
                                InsertionError::TopologyValidation(
                                    TdsValidationError::InconsistentDataStructure { message }
                                ) if message.contains("Isolated vertex detected after insertion")
                            );
                            let should_fallback = matches!(
                                &err,
                                InsertionError::ConflictRegion(
                                    ConflictError::NonManifoldFacet { .. }
                                        | ConflictError::RidgeFan { .. }
                                        | ConflictError::DisconnectedBoundary { .. }
                                        | ConflictError::OpenBoundary { .. }
                                )
                            ) || fallback_on_isolated_vertex;

                            if should_fallback {
                                #[cfg(debug_assertions)]
                                tracing::warn!(
                                    "Outside insertion conflict boundary degeneracy ({err}) (conflict_cells={conflict_len}); falling back to hull extension"
                                );
                                #[cfg(debug_assertions)]
                                if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                                    tracing::debug!(
                                        error = %err,
                                        fallback_on_isolated_vertex,
                                        "Outside insertion: cavity insertion failed; using hull extension"
                                    );
                                }
                            } else {
                                #[cfg(debug_assertions)]
                                tracing::warn!("Outside insertion cavity insertion failed: {err}");
                                return Err(err);
                            }
                        }
                    }
                }
                // Exterior vertex: extend convex hull
                #[cfg(debug_assertions)]
                if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                    tracing::debug!(
                        point = ?point,
                        "Outside insertion: proceeding to hull extension"
                    );
                }
                let new_cells = match extend_hull(&mut self.tds, &self.kernel, v_key, &point) {
                    Ok(cells) => cells,
                    Err(err) => {
                        let retry_inside = matches!(
                            &err,
                            InsertionError::HullExtension {
                                reason: HullExtensionReason::NoVisibleFacets
                            }
                        );
                        if retry_inside {
                            let fallback_location =
                                locate_by_scan(&self.tds, &self.kernel, &point)?;
                            if let LocateResult::InsideCell(start_cell) = fallback_location {
                                #[cfg(debug_assertions)]
                                if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                                    tracing::warn!(
                                        point = ?point,
                                        start_cell = ?start_cell,
                                        "Outside insertion: no visible facets; retrying as interior with star-split"
                                    );
                                }
                                suspicion.fallback_star_split = true;
                                let mut star_conflict = CellKeyBuffer::new();
                                star_conflict.push(start_cell);
                                let (hint, total_removed) = self.insert_with_conflict_region(
                                    v_key,
                                    &point,
                                    star_conflict,
                                    Some(start_cell),
                                    &mut suspicion,
                                )?;
                                return Ok(((v_key, hint), total_removed, suspicion));
                            }
                        }
                        #[cfg(debug_assertions)]
                        if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                            tracing::warn!(
                                point = ?point,
                                error = %err,
                                "Outside insertion: hull extension failed"
                            );
                        }
                        return Err(err);
                    }
                };
                #[cfg(debug_assertions)]
                if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                    tracing::debug!(
                        new_cells = new_cells.len(),
                        "Outside insertion: hull extension succeeded"
                    );
                }

                #[cfg(debug_assertions)]
                if std::env::var_os("DELAUNAY_DEBUG_NEIGHBORS").is_some() {
                    let mut total_slots = 0usize;
                    let mut neighbor_none = 0usize;
                    let mut neighbor_missing = 0usize;
                    let mut neighbor_mutual = 0usize;
                    let mut neighbor_non_mutual = 0usize;

                    for &cell_key in &new_cells {
                        let Some(cell) = self.tds.get_cell(cell_key) else {
                            continue;
                        };
                        let Some(neighbors) = cell.neighbors() else {
                            continue;
                        };
                        for &neighbor_opt in neighbors {
                            total_slots = total_slots.saturating_add(1);
                            match neighbor_opt {
                                None => {
                                    neighbor_none = neighbor_none.saturating_add(1);
                                }
                                Some(neighbor_key) => {
                                    if !self.tds.contains_cell(neighbor_key) {
                                        neighbor_missing = neighbor_missing.saturating_add(1);
                                    } else if self
                                        .tds
                                        .get_cell(neighbor_key)
                                        .and_then(|neighbor_cell| neighbor_cell.neighbors())
                                        .is_some_and(|ns| ns.contains(&Some(cell_key)))
                                    {
                                        neighbor_mutual = neighbor_mutual.saturating_add(1);
                                    } else {
                                        neighbor_non_mutual = neighbor_non_mutual.saturating_add(1);
                                    }
                                }
                            }
                        }
                    }

                    tracing::debug!(
                        new_cells = new_cells.len(),
                        total_slots,
                        neighbor_none,
                        neighbor_missing,
                        neighbor_mutual,
                        neighbor_non_mutual,
                        "Outside insertion: hull extension neighbor-pointer summary"
                    );
                }

                // Iteratively repair non-manifold topology until facet sharing is valid
                let mut total_removed = 0;
                #[cfg_attr(
                    not(debug_assertions),
                    expect(
                        unused_variables,
                        reason = "`iteration` is only used for debug logging",
                    )
                )]
                for iteration in 0..MAX_REPAIR_ITERATIONS {
                    // Check for non-manifold issues in newly created hull cells (local scan)
                    // This keeps the repair O(k·D) where k is the number of new hull cells, rather than O(N·D)
                    let cells_to_check: CellKeyBuffer = new_cells
                        .iter()
                        .copied()
                        .filter(|ck| self.tds.contains_cell(*ck))
                        .collect();

                    if let Some(issues) = self.detect_local_facet_issues(&cells_to_check)? {
                        // Only mark this as "suspicious" if we *actually* detected local facet issues
                        // and entered the repair path.
                        suspicion.repair_loop_entered = true;

                        #[cfg(debug_assertions)]
                        tracing::debug!(
                            "Hull extension repair iteration {}: {} over-shared facets detected, removing cells...",
                            iteration + 1,
                            issues.len()
                        );

                        let removed = self.repair_local_facet_issues(&issues)?;

                        // Early exit if repair made no progress
                        if removed == 0 {
                            #[cfg(debug_assertions)]
                            tracing::warn!(
                                "No cells removed in iteration {} - repair cannot make progress",
                                iteration + 1
                            );
                            return Err(InsertionError::TopologyValidation(
                                TdsValidationError::InconsistentDataStructure {
                                    message: format!(
                                        "Hull extension repair stalled: {} over-shared facets remain but no cells could be removed",
                                        issues.len()
                                    ),
                                },
                            ));
                        }

                        total_removed += removed;
                        if removed > 0 {
                            suspicion.cells_removed = true;
                        }

                        #[cfg(debug_assertions)]
                        tracing::debug!("Removed {removed} cells (total: {total_removed})");

                        // Early exit if repair succeeded
                        if self.tds.validate_facet_sharing().is_ok() {
                            break;
                        }
                    } else {
                        // No more non-manifold issues - safe to rebuild neighbors
                        break;
                    }
                }

                // Rebuild neighbor pointers now that topology is manifold
                if total_removed > 0 {
                    // Double-check that facet sharing is actually valid
                    let facet_valid = self.tds.validate_facet_sharing().is_ok();
                    #[cfg(debug_assertions)]
                    tracing::debug!(
                        "Before repair_neighbor_pointers: facet_sharing_valid={facet_valid}, cells={}",
                        self.tds.number_of_cells()
                    );

                    if !facet_valid {
                        return Err(InsertionError::CavityFilling {
                            message: "Facet sharing still invalid after repairs - cannot safely rebuild neighbors".to_string(),
                        });
                    }

                    // Use repair_neighbor_pointers for surgical reconstruction
                    // This preserves existing correct pointers and only fixes broken ones
                    let repaired = repair_neighbor_pointers(&mut self.tds).map_err(|e| {
                        InsertionError::CavityFilling {
                            message: format!("Failed to rebuild neighbors after repairs: {e}"),
                        }
                    })?;
                    suspicion.neighbor_pointers_rebuilt = repaired > 0;
                }

                // Always rebuild vertex→cell incidence after insertion.
                self.tds
                    .assign_incident_cells()
                    .map_err(|e| InsertionError::CavityFilling {
                        message: format!("Failed to assign incident cells after insertion: {e}"),
                    })?;

                #[cfg(debug_assertions)]
                if std::env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some() {
                    match validate_ridge_links(&self.tds) {
                        Ok(()) => {
                            tracing::debug!(
                                "extend_hull: ridge-link validation passed after insertion"
                            );
                        }
                        Err(err) => {
                            tracing::warn!(
                                error = ?err,
                                "extend_hull: ridge-link validation failed after insertion"
                            );
                        }
                    }
                }

                // Detect isolated vertices and treat as retryable degeneracy.
                if self.tds.vertices().any(|(_, v)| v.incident_cell.is_none()) {
                    return Err(InsertionError::TopologyValidation(
                        TdsValidationError::InconsistentDataStructure {
                            message:
                                "Isolated vertex detected after insertion (vertex not in any cell)"
                                    .to_string(),
                        },
                    ));
                }

                // Connectedness guard (localized): ensure the newly created cell set is internally
                // connected and attached to the existing triangulation.
                self.validate_connectedness(&new_cells)?;

                // Return vertex key and hint for next insertion
                let hint = new_cells
                    .iter()
                    .copied()
                    .find(|ck| self.tds.contains_cell(*ck));
                Ok(((v_key, hint), total_removed, suspicion))
            }
            LocateResult::OnFacet(_, _) | LocateResult::OnEdge(_) | LocateResult::OnVertex(_) => {
                // These degenerate cases are already handled at lines 772-779 above,
                // so this arm is unreachable. Included only for exhaustiveness.
                unreachable!("Degenerate locations should have been handled earlier")
            }
        }
    }

    /// Removes a vertex and retriangulates the resulting cavity using fan triangulation.
    ///
    /// This operation maintains topological consistency by:
    /// 1. Finding all cells containing the vertex
    /// 2. Removing those cells (creating a cavity)
    /// 3. Extracting the cavity boundary facets
    /// 4. Filling the cavity with a fan triangulation (pick apex, connect to all boundary facets)
    /// 5. Wiring neighbors to maintain consistency
    /// 6. Removing the vertex itself
    ///
    /// **Fan Triangulation**: The cavity is filled by picking one boundary vertex as an apex
    /// and connecting it to all boundary facets. This is fast and maintains all topological
    /// invariants, though it may create poorly-shaped cells in some cases.
    ///
    /// # Arguments
    ///
    /// * `vertex` - Reference to the vertex to remove
    ///
    /// # Returns
    ///
    /// The number of cells that were removed along with the vertex.
    ///
    /// # Errors
    ///
    /// Returns [`TdsMutationError`]
    /// if the removal cannot be completed while maintaining triangulation invariants.
    ///
    /// (Note: `TdsMutationError` is currently a thin wrapper around
    /// [`TdsValidationError`]; the wrapper exists to make mutation call sites/docs more semantically explicit.)
    ///
    pub(crate) fn remove_vertex(
        &mut self,
        vertex: &Vertex<K::Scalar, U, D>,
    ) -> Result<usize, TdsMutationError>
    where
        K::Scalar: CoordinateScalar,
    {
        // Find the vertex key
        let Some(vertex_key) = self.tds.vertex_key_from_uuid(&vertex.uuid()) else {
            return Ok(0); // Vertex not found, nothing to remove
        };

        // Collect all cells containing this vertex by scanning all cells
        let cells_to_remove: CellKeyBuffer = self
            .tds
            .cells()
            .filter_map(|(cell_key, cell)| {
                if cell.vertices().contains(&vertex_key) {
                    Some(cell_key)
                } else {
                    None
                }
            })
            .collect();

        if cells_to_remove.is_empty() {
            // Vertex exists but has no incident cells - use Tds removal
            return self.tds.remove_vertex(vertex);
        }

        // Extract cavity boundary BEFORE removing cells
        let boundary_facets =
            extract_cavity_boundary(&self.tds, &cells_to_remove).map_err(|e| {
                TdsValidationError::InconsistentDataStructure {
                    message: format!("Failed to extract cavity boundary: {e}"),
                }
            })?;

        // If boundary is empty, we're removing the entire triangulation
        if boundary_facets.is_empty() {
            // Use Tds removal for empty boundary case
            return self.tds.remove_vertex(vertex);
        }

        // Pick apex vertex for fan triangulation (first vertex of first boundary facet)
        let apex_vertex_key = self.pick_fan_apex(&boundary_facets).ok_or_else(|| {
            TdsValidationError::InconsistentDataStructure {
                message: "Failed to find apex vertex for fan triangulation".to_string(),
            }
        })?;

        // Fill cavity with fan triangulation BEFORE removing old cells
        // Use fan triangulation that skips boundary facets which already include the apex
        let new_cells = self
            .fan_fill_cavity(apex_vertex_key, &boundary_facets)
            .map_err(|e| TdsValidationError::InconsistentDataStructure {
                message: format!("Fan triangulation failed: {e}"),
            })?;

        // Wire neighbors for the new cells (while both old and new cells exist)
        wire_cavity_neighbors(&mut self.tds, &new_cells, Some(&cells_to_remove)).map_err(|e| {
            TdsValidationError::InconsistentDataStructure {
                message: format!("Neighbor wiring failed: {e}"),
            }
        })?;

        // Remove the cells containing the vertex (now that new cells are wired up)
        // Note: remove_cells_by_keys() automatically clears neighbor pointers in surviving
        // cells that reference removed cells (sets them to None/boundary)
        let mut cells_removed = self.tds.remove_cells_by_keys(&cells_to_remove);

        // Validate facet topology for newly created cells (O(k*D) localized check)
        if let Some(issues) = self.detect_local_facet_issues(&new_cells)? {
            #[cfg(debug_assertions)]
            tracing::warn!(
                "Warning: {} over-shared facets detected after vertex removal, repairing...",
                issues.len()
            );
            let removed = self.repair_local_facet_issues(&issues)?;
            cells_removed += removed;
            #[cfg(debug_assertions)]
            tracing::debug!("Repaired by removing {removed} additional cells");

            // Repair neighbor pointers after removing additional cells
            // This ensures neighbor consistency after repair operations
            if removed > 0 {
                repair_neighbor_pointers(&mut self.tds).map_err(|e| {
                    TdsValidationError::InconsistentDataStructure {
                        message: format!("Neighbor repair after facet issue repair failed: {e}"),
                    }
                })?;
            }
        }

        // Rebuild vertex-cell incidence for all vertices
        self.tds.assign_incident_cells()?;

        // Remove the vertex using Tds method (handles internal bookkeeping)
        self.tds.remove_vertex(vertex)?;

        Ok(cells_removed)
    }

    /// Pick an apex vertex for fan triangulation.
    ///
    /// Selects the first vertex from the first boundary facet as the apex.
    /// The fan will connect this apex to all boundary facets.
    ///
    /// # Arguments
    ///
    /// * `boundary_facets` - The cavity boundary facets
    ///
    /// # Returns
    ///
    /// The vertex key to use as apex, or None if no suitable vertex found.
    fn pick_fan_apex(&self, boundary_facets: &[FacetHandle]) -> Option<VertexKey>
    where
        K::Scalar: CoordinateScalar,
    {
        // Get first boundary facet
        let first_facet = boundary_facets.first()?;
        let cell = self.tds.get_cell(first_facet.cell_key())?;

        // Get the first vertex from this facet (any vertex that's not the opposite one)
        let facet_idx = <usize as From<_>>::from(first_facet.facet_index());
        cell.vertices()
            .iter()
            .enumerate()
            .find(|(i, _)| *i != facet_idx)
            .map(|(_, &vkey)| vkey)
    }

    /// Fan-specific cavity fill: connect an existing apex vertex to boundary facets
    /// that do not already include the apex. This avoids creating degenerate cells
    /// with duplicate vertices when the apex lies on a boundary facet.
    fn fan_fill_cavity(
        &mut self,
        apex_vertex_key: VertexKey,
        boundary_facets: &[FacetHandle],
    ) -> Result<CellKeyBuffer, InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        let mut new_cells = CellKeyBuffer::new();

        for facet_handle in boundary_facets {
            let boundary_cell = self.tds.get_cell(facet_handle.cell_key()).ok_or_else(|| {
                InsertionError::CavityFilling {
                    message: format!(
                        "Boundary facet cell {:?} not found",
                        facet_handle.cell_key()
                    ),
                }
            })?;

            let facet_idx = <usize as From<_>>::from(facet_handle.facet_index());

            // Gather facet vertices (all except the opposite vertex)
            let mut facet_vertices = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in boundary_cell.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vertices.push(vkey);
                }
            }

            // Skip facets that already contain the apex to avoid duplicate vertices
            if facet_vertices.contains(&apex_vertex_key) {
                continue;
            }

            // Build new cell vertices = facet_vertices + apex
            let mut new_cell_vertices = facet_vertices;
            new_cell_vertices.push(apex_vertex_key);

            // Create and insert the new cell
            let new_cell =
                Cell::new(new_cell_vertices, None).map_err(|e| InsertionError::CavityFilling {
                    message: format!("Failed to create cell: {e}"),
                })?;
            let cell_key = self.tds.insert_cell_with_mapping(new_cell).map_err(|e| {
                InsertionError::CavityFilling {
                    message: format!("Failed to insert cell: {e}"),
                }
            })?;

            new_cells.push(cell_key);
        }

        if new_cells.is_empty() {
            return Err(InsertionError::CavityFilling {
                message: "Fan triangulation produced no cells (apex on all boundary facets?)"
                    .to_string(),
            });
        }

        Ok(new_cells)
    }

    // Phase 2 TODO: Add geometric operations using kernel predicates
    // - locate(point) - point location using facet walking

    /// Detects over-shared facets within a specific set of cells (localized check).
    ///
    /// This is an **O(k * D)** operation where k = number of cells to check,
    /// unlike global validation which is O(N * D) for the entire triangulation.
    ///
    /// # Performance
    ///
    /// - **Complexity**: O(k * D) where k = `cells.len()`, D = dimension
    /// - **Use case**: Detect issues in newly created cells after insertion/removal
    /// - **Comparison**: Global detection is O(N * D) where N = total cells
    ///
    /// # Arguments
    ///
    /// * `cells` - Keys of cells to check (typically newly created cells)
    ///
    /// # Returns
    ///
    /// `Ok(None)` if all facets are valid (≤2 cells per facet).
    /// `Ok(Some(issues))` if over-shared facets are detected, where issues is a map
    /// from facet hash to the cells sharing that facet.
    ///
    /// # Errors
    ///
    /// Returns error if cells cannot be accessed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // A single simplex has no over-shared facets.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let cell_keys: Vec<_> = dt.cells().map(|(ck, _)| ck).collect();
    /// let issues = dt
    ///     .as_triangulation()
    ///     .detect_local_facet_issues(&cell_keys)
    ///     .unwrap();
    /// assert!(issues.is_none());
    ///
    /// // Note: This method is most useful for checking newly created cells
    /// // after insertion/removal operations (see usage in insert_transactional).
    /// ```
    pub fn detect_local_facet_issues(
        &self,
        cells: &[CellKey],
    ) -> Result<Option<FacetIssuesMap>, TdsValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        // Build facet map for ONLY the specified cells
        // This is O(k * D) instead of O(N * D)
        let mut facet_to_cells = FacetIssuesMap::default();

        // Index facets from the specified cells
        for &cell_key in cells {
            let Some(cell) = self.tds.get_cell(cell_key) else {
                continue; // Cell was removed, skip
            };

            // For each facet of this cell
            for facet_idx in 0..cell.number_of_vertices() {
                // Compute facet hash from sorted vertex keys
                let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (i, &vkey) in cell.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vkeys.push(vkey);
                    }
                }
                facet_vkeys.sort_unstable();

                // Hash the facet
                let mut hasher = FastHasher::default();
                for &vkey in &facet_vkeys {
                    vkey.hash(&mut hasher);
                }
                let facet_hash = hasher.finish();

                // Track this cell/facet pair
                let facet_idx_u8 = u8::try_from(facet_idx).map_err(|_| {
                    TdsValidationError::InconsistentDataStructure {
                        message: format!(
                            "Facet index {facet_idx} exceeds u8::MAX (dimension too high)"
                        ),
                    }
                })?;
                facet_to_cells
                    .entry(facet_hash)
                    .or_insert_with(SmallBuffer::new)
                    .push((cell_key, facet_idx_u8));
            }
        }

        // Filter to only over-shared facets (> 2 cells) in a single pass
        facet_to_cells.retain(|_, cell_facet_pairs| cell_facet_pairs.len() > 2);

        if facet_to_cells.is_empty() {
            Ok(None)
        } else {
            Ok(Some(facet_to_cells))
        }
    }

    /// Repairs over-shared facets by removing lower-quality cells.
    ///
    /// Uses geometric quality metrics (`radius_ratio`) to select which cells to keep
    /// when a facet is shared by more than 2 cells. UUID ordering is used as a tie-breaker
    /// when cells have equal quality. Errors if quality computation or conversion fails.
    ///
    /// # Performance
    ///
    /// - **Complexity**: O(m * q) where m = number of problematic facets, q = quality computation cost
    /// - **Localized**: Only processes cells involved in detected issues
    ///
    /// # Arguments
    ///
    /// * `issues` - Detected facet issues map from `detect_local_facet_issues()`
    ///
    /// # Returns
    ///
    /// Number of cells removed during repair.
    ///
    /// # Errors
    ///
    /// Returns error if quality evaluation or facet bookkeeping fails while
    /// selecting cells to remove. This function itself does not rebuild neighbors;
    /// callers are responsible for repairing or validating topology after removal
    /// (e.g., via `repair_neighbor_pointers` or a validation pass).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::collections::FacetIssuesMap;
    /// use delaunay::prelude::*;
    ///
    /// // Start with a valid 2D simplex.
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let mut dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Empty issues map => nothing to remove.
    /// let removed = dt
    ///     .as_triangulation_mut()
    ///     .repair_local_facet_issues(&FacetIssuesMap::default())
    ///     .unwrap();
    /// assert_eq!(removed, 0);
    /// ```
    ///
    /// In practice, this method is typically called with issues detected by
    /// [`detect_local_facet_issues`](Self::detect_local_facet_issues) after insertion/removal
    /// operations. See `insert_transactional` for a typical usage pattern.
    pub fn repair_local_facet_issues(
        &mut self,
        issues: &FacetIssuesMap,
    ) -> Result<usize, TdsValidationError>
    where
        K::Scalar: CoordinateScalar + Div<Output = K::Scalar>,
    {
        let mut cells_to_remove = CellKeySet::default();

        // For each over-shared facet, select cells to remove
        for cell_facet_pairs in issues.values() {
            let involved_cells: Vec<CellKey> = cell_facet_pairs.iter().map(|(ck, _)| *ck).collect();

            // Compute quality for each cell - propagate errors from quality evaluation
            let mut cell_qualities: Vec<(CellKey, f64, Uuid)> = Vec::new();
            for &cell_key in &involved_cells {
                let cell = self.tds.get_cell(cell_key).ok_or_else(|| {
                    TdsValidationError::InconsistentDataStructure {
                        message: format!("Cell {cell_key:?} not found during facet repair"),
                    }
                })?;
                let uuid = cell.uuid();

                // Propagate quality evaluation errors
                let ratio = radius_ratio(self, cell_key).map_err(|e| {
                    TdsValidationError::InconsistentDataStructure {
                        message: format!("Quality evaluation failed for cell {cell_key:?}: {e}"),
                    }
                })?;
                let ratio_f64 = safe_scalar_to_f64(ratio).map_err(|_| {
                    TdsValidationError::InconsistentDataStructure {
                        message: format!("Quality ratio conversion failed for cell {cell_key:?}"),
                    }
                })?;

                if ratio_f64.is_finite() {
                    cell_qualities.push((cell_key, ratio_f64, uuid));
                } else {
                    return Err(TdsValidationError::InconsistentDataStructure {
                        message: format!(
                            "Non-finite quality ratio {ratio_f64} for cell {cell_key:?}"
                        ),
                    });
                }
            }

            // Quality-based selection: keep 2 best, remove rest
            // Note: cell_qualities always has all involved_cells at this point since
            // any quality computation failure results in an early error return above
            cell_qualities.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(CmpOrdering::Equal)
                    .then_with(|| a.2.cmp(&b.2))
            });

            // Mark cells beyond the top 2 for removal
            for (cell_key, _, _) in cell_qualities.iter().skip(2) {
                if self.tds.contains_cell(*cell_key) {
                    cells_to_remove.insert(*cell_key);
                }
            }
        }

        // Remove the selected cells - do NOT rebuild neighbors here
        // Neighbor wiring should happen AFTER all non-manifold issues are resolved
        let to_remove: Vec<CellKey> = cells_to_remove.into_iter().collect();
        let removed_count = self.tds.remove_cells_by_keys(&to_remove);

        Ok(removed_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::collections::NeighborBuffer;
    use crate::core::collections::spatial_hash_grid::HashGridIndex;
    use crate::core::delaunay_triangulation::DelaunayTriangulation;
    use crate::core::vertex::VertexBuilder;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar};
    use crate::topology::characteristics::validation::validate_triangulation_euler;
    use crate::vertex;

    use slotmap::KeyData;

    #[test]
    fn test_triangulation_validation_error_from_manifold_error_preserves_detail() {
        let tds_err = TdsValidationError::InvalidNeighbors {
            message: "unit test".to_string(),
        };

        assert_eq!(
            TriangulationValidationError::from(ManifoldError::Tds(tds_err.clone())),
            TriangulationValidationError::Tds(tds_err)
        );

        assert!(matches!(
            TriangulationValidationError::from(ManifoldError::ManifoldFacetMultiplicity {
                facet_key: 123,
                cell_count: 3
            }),
            TriangulationValidationError::ManifoldFacetMultiplicity {
                facet_key: 123,
                cell_count: 3
            }
        ));

        assert!(matches!(
            TriangulationValidationError::from(ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key: 0x00ab_cdef,
                boundary_facet_count: 4
            }),
            TriangulationValidationError::BoundaryRidgeMultiplicity {
                ridge_key: 0x00ab_cdef,
                boundary_facet_count: 4
            }
        ));

        assert!(matches!(
            TriangulationValidationError::from(ManifoldError::RidgeLinkNotManifold {
                ridge_key: 0x00ab_cdef,
                link_vertex_count: 7,
                link_edge_count: 8,
                max_degree: 3,
                degree_one_vertices: 2,
                connected: false
            }),
            TriangulationValidationError::RidgeLinkNotManifold {
                ridge_key: 0x00ab_cdef,
                link_vertex_count: 7,
                link_edge_count: 8,
                max_degree: 3,
                degree_one_vertices: 2,
                connected: false
            }
        ));

        assert!(matches!(
            TriangulationValidationError::from(ManifoldError::VertexLinkNotManifold {
                vertex_key: VertexKey::from(KeyData::from_ffi(1)),
                link_vertex_count: 3,
                link_cell_count: 4,
                boundary_facet_count: 1,
                max_degree: 2,
                connected: false,
                interior_vertex: true,
            }),
            TriangulationValidationError::VertexLinkNotManifold {
                link_vertex_count: 3,
                link_cell_count: 4,
                boundary_facet_count: 1,
                max_degree: 2,
                connected: false,
                interior_vertex: true,
                ..
            }
        ));
    }

    #[test]
    fn test_triangulation_new_empty_and_new_with_tds_default_to_pl_manifold() {
        let tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);

        let tri_with_tds: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_with_tds(FastKernel::new(), Tds::<f64, (), (), 2>::empty());
        assert_eq!(
            tri_with_tds.topology_guarantee(),
            TopologyGuarantee::PLManifold
        );
    }

    #[test]
    fn test_triangulation_set_topology_guarantee_round_trips() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);

        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::Pseudomanifold);

        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);
        assert_eq!(
            tri.topology_guarantee(),
            TopologyGuarantee::PLManifoldStrict
        );

        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);
        assert_eq!(tri.topology_guarantee(), TopologyGuarantee::PLManifold);
    }

    #[test]
    fn test_duplicate_detection_metrics_force_enable() {
        struct DuplicateDetectionGuard;

        impl Drop for DuplicateDetectionGuard {
            fn drop(&mut self) {
                DUPLICATE_DETECTION_FORCE_ENABLED.store(false, Ordering::Relaxed);
            }
        }

        let _guard = DuplicateDetectionGuard;
        DUPLICATE_DETECTION_FORCE_ENABLED.store(true, Ordering::Relaxed);

        let before = Triangulation::<FastKernel<f64>, (), (), 2>::duplicate_detection_metrics()
            .expect("duplicate detection metrics should be enabled");

        record_duplicate_detection_metrics(true, 3, false);
        record_duplicate_detection_metrics(false, 0, true);

        let after = Triangulation::<FastKernel<f64>, (), (), 2>::duplicate_detection_metrics()
            .expect("duplicate detection metrics should be enabled");

        assert!(after.total_checks > before.total_checks);
        assert!(after.grid_used > before.grid_used);
        assert!(after.grid_fallbacks > before.grid_fallbacks);
        assert!(after.grid_candidates >= before.grid_candidates + 3);
    }

    #[test]
    fn test_validate_at_completion_skips_for_pseudomanifold() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
        assert!(tri.validate_at_completion().is_ok());
    }

    #[test]
    fn test_validate_at_completion_reports_invalid_vertex_link() {
        // Two triangles sharing only a single vertex produce a disconnected vertex link.
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 0.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 1.0]))
            .unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v3, v4], None).unwrap())
            .unwrap();

        tds.assign_incident_cells().unwrap();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);

        match tri.validate_at_completion() {
            Err(TriangulationValidationError::VertexLinkNotManifold { vertex_key, .. }) => {
                assert_eq!(vertex_key, v0);
            }
            other => panic!("Expected VertexLinkNotManifold, got {other:?}"),
        }
    }

    fn build_disconnected_two_triangles_tds_2d() -> Tds<f64, (), (), 2> {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let a0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let a1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let a2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let b0 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 0.0]))
            .unwrap();
        let b1 = tds
            .insert_vertex_with_mapping(vertex!([11.0, 0.0]))
            .unwrap();
        let b2 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 1.0]))
            .unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![a0, a1, a2], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![b0, b1, b2], None).unwrap())
            .unwrap();

        tds
    }

    fn build_three_triangles_sharing_edge_tds_2d() -> Tds<f64, (), (), 2> {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0]))
            .unwrap();
        let v4 = tds.insert_vertex_with_mapping(vertex!([2.0, 0.0])).unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v4], None).unwrap())
            .unwrap();

        tds
    }

    #[test]
    fn test_validate_after_insertion_skips_when_no_cells() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Force validation to be enabled if there were any cells.
        tri.set_validation_policy(ValidationPolicy::Always);

        // Insert a vertex without creating any cells (bootstrap phase).
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();
        assert_eq!(tri.number_of_cells(), 0);

        tri.validate_after_insertion(SuspicionFlags::default())
            .unwrap();
    }

    #[test]
    fn test_validate_after_insertion_calls_is_valid_when_policy_triggers() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::Always);

        match tri.validate_after_insertion(SuspicionFlags::default()) {
            Err(TriangulationValidationError::Tds(
                TdsValidationError::InconsistentDataStructure { message },
            )) => {
                assert!(
                    message.contains("Disconnected triangulation"),
                    "Unexpected message: {message}"
                );
            }
            other => panic!("Expected disconnectedness error, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_after_insertion_only_checks_required_links_when_policy_does_not_trigger() {
        let tds = build_disconnected_two_triangles_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);

        // The triangulation is invalid (disconnected), but the required PL-manifold link
        // checks are still satisfied.
        assert!(tri.is_valid().is_err());
        tri.validate_after_insertion(SuspicionFlags::default())
            .unwrap();
    }

    #[test]
    fn test_validate_after_insertion_does_not_skip_required_link_checks_in_pl_manifold_mode() {
        let tds = build_three_triangles_sharing_edge_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::PLManifold);

        match tri.validate_after_insertion(SuspicionFlags::default()) {
            Err(TriangulationValidationError::ManifoldFacetMultiplicity { cell_count, .. }) => {
                assert_eq!(cell_count, 3);
            }
            other => panic!("Expected ManifoldFacetMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_after_insertion_skips_when_policy_does_not_trigger_and_no_required_link_checks()
     {
        let tds = build_disconnected_two_triangles_tds_2d();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        tri.set_validation_policy(ValidationPolicy::OnSuspicion);
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        assert!(tri.is_valid().is_err());
        tri.validate_after_insertion(SuspicionFlags::default())
            .unwrap();
    }

    #[test]
    fn test_select_locate_hint_from_hash_grid_returns_incident_cell() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let cell_key = tri.tds.cell_keys().next().unwrap();

        let mut index: HashGridIndex<f64, 2> = HashGridIndex::new(1.0);
        for (vkey, vertex) in tri.tds.vertices() {
            index.insert_vertex(vkey, vertex.point().coords());
        }

        let hint = tri.select_locate_hint_from_hash_grid(&[0.05, 0.05], &index);
        assert_eq!(hint, Some(cell_key));
    }

    #[test]
    fn test_select_locate_hint_from_hash_grid_skips_missing_cell() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let vkey = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();
        {
            let vertex = tri.tds.get_vertex_by_key_mut(vkey).unwrap();
            vertex.incident_cell = Some(CellKey::default());
        }

        let mut index: HashGridIndex<f64, 2> = HashGridIndex::new(1.0);
        index.insert_vertex(vkey, &[0.0, 0.0]);

        let hint = tri.select_locate_hint_from_hash_grid(&[0.0, 0.0], &index);
        assert!(hint.is_none());
    }

    #[test]
    fn test_duplicate_coordinates_error_uses_hash_grid_index() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let vkey = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();

        let mut index: HashGridIndex<f64, 2> = HashGridIndex::new(1.0);
        index.insert_vertex(vkey, &[0.0, 0.0]);

        let tol = 1e-10_f64;
        let err = tri.duplicate_coordinates_error(&[0.0, 0.0], tol * tol, Some(&index));
        assert!(matches!(
            err,
            Some(InsertionError::DuplicateCoordinates { .. })
        ));
    }

    #[test]
    fn test_duplicate_coordinates_error_falls_back_when_index_unusable() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();

        let index: HashGridIndex<f64, 2> = HashGridIndex::new(0.0); // unusable
        let tol = 1e-10_f64;
        let err = tri.duplicate_coordinates_error(&[0.0, 0.0], tol * tol, Some(&index));
        assert!(matches!(
            err,
            Some(InsertionError::DuplicateCoordinates { .. })
        ));
    }

    #[test]
    fn test_estimate_local_perturbation_scale_uses_hint_cell_vertices() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let cell_key = tri.tds.cell_keys().next().unwrap();

        let scale = tri.estimate_local_perturbation_scale(&[0.1, 0.0], Some(cell_key));
        assert!((scale - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_estimate_local_perturbation_scale_clamps_to_min_scale() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        let _ = tri
            .tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]))
            .unwrap();

        let scale = tri.estimate_local_perturbation_scale(&[0.0, 0.0], None);
        let min_scale = <f64 as CoordinateScalar>::default_tolerance();
        approx::assert_abs_diff_eq!(scale, min_scale, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_pl_manifold_insertion_never_commits_invalid_topology_under_validation_policy_never() {
        // A small deterministic point set (seeded RNG) that exercises degeneracy handling.
        //
        // This test is intentionally small and validates after *each* insertion to ensure
        // we never commit an invalid PL-manifold state, even when the user disables
        // automatic validation via `ValidationPolicy::Never`.
        let points = crate::geometry::util::generate_random_points_seeded::<f64, 3>(
            25,
            (-100.0, 100.0),
            123,
        )
        .unwrap();

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

        dt.set_validation_policy(ValidationPolicy::Never);
        dt.set_delaunay_repair_policy(
            crate::core::delaunay_triangulation::DelaunayRepairPolicy::Never,
        );

        for (i, point) in points.into_iter().enumerate() {
            let vertex = VertexBuilder::default().point(point).build().unwrap();

            let result = dt
                .insert_with_statistics(vertex)
                .unwrap_or_else(|e| panic!("Non-retryable insertion error at i={i}: {e:?}"));

            let (outcome, stats) = result;

            // Skip Level 3 validation during bootstrap (vertices but no cells yet).
            if dt.number_of_cells() > 0
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

    /// Macro to generate `build_initial_simplex` tests across dimensions.
    ///
    /// This macro generates tests that verify `build_initial_simplex` by:
    /// 1. Creating D+1 affinely independent vertices
    /// 2. Calling `build_initial_simplex` directly
    /// 3. Verifying the Tds has correct structure (vertices, cells, dimension)
    ///
    /// # Usage
    /// ```ignore
    /// test_build_initial_simplex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    /// ```
    macro_rules! test_build_initial_simplex {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_build_initial_simplex_ $dim d>]() {
                    // Build initial simplex (D+1 vertices)
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let expected_vertices = vertices.len();
                    assert_eq!(expected_vertices, $dim + 1,
                        "Test must provide exactly D+1 vertices for {}D simplex", $dim);

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();

                    // Verify structure
                    assert_eq!(tds.number_of_vertices(), expected_vertices,
                        "{}D: Expected {} vertices", $dim, expected_vertices);
                    assert_eq!(tds.number_of_cells(), 1,
                        "{}D: Expected 1 cell", $dim);
                    assert_eq!(tds.dim(), $dim as i32,
                        "{}D: Expected dimension {}", $dim, $dim);

                    // Verify all vertices are present
                    assert_eq!(tds.vertices().count(), expected_vertices,
                        "{}D: All vertices should be in Tds", $dim);

                    // Verify the single cell has correct number of vertices
                    let (_, cell) = tds.cells().next()
                        .expect(&format!("{}D: Should have exactly one cell", $dim));
                    assert_eq!(cell.number_of_vertices(), expected_vertices,
                        "{}D: Cell should have {} vertices", $dim, expected_vertices);

                    // Verify incident cells are assigned
                    for (_, vertex) in tds.vertices() {
                        assert!(vertex.incident_cell.is_some(),
                            "{}D: All vertices should have incident cell assigned", $dim);
                    }

                    // Verify initial simplex has no neighbors (all boundary facets)
                    if let Some(neighbors) = cell.neighbors() {
                        assert!(neighbors.iter().all(|n| n.is_none()),
                            "{}D: Initial simplex should have no neighbors (all boundary)", $dim);
                    }
                }
            }
        };
    }

    // 2D: Triangle
    test_build_initial_simplex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

    // 3D: Tetrahedron
    test_build_initial_simplex!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );

    // 4D: 4-simplex
    test_build_initial_simplex!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );

    // 5D: 5-simplex
    test_build_initial_simplex!(
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

    /// Macro to generate Level 3 (topology) validation tests across dimensions.
    ///
    /// This macro generates tests that verify manifold-with-boundary validation by:
    /// 1. Creating a Delaunay triangulation from D+1 affinely independent vertices
    /// 2. Calling `Triangulation::is_valid()` (Level 3)
    /// 3. Verifying that the validation passes
    ///
    /// # Usage
    /// ```ignore
    /// test_is_valid_topology!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    /// ```
    macro_rules! test_is_valid_topology {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_is_valid_topology_ $dim d>]() {
                    // Build triangulation from D+1 vertices (initial simplex)
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let expected_vertices = vertices.len();
                    assert_eq!(expected_vertices, $dim + 1,
                        "Test must provide exactly D+1 vertices for {}D simplex", $dim);

                    let dt = DelaunayTriangulation::new(&vertices)
                        .expect(&format!("Failed to create {}D triangulation", $dim));
                    let tri = dt.as_triangulation();

                    // Level 3: topology validation
                    let result = tri.is_valid();
                    assert!(
                        result.is_ok(),
                        "{}D: Simple simplex should be a valid manifold-with-boundary. Error: {:?}",
                        $dim,
                        result.err()
                    );

                    // Also verify basic properties
                    assert_eq!(tri.number_of_vertices(), expected_vertices,
                        "{}D: Should have {} vertices", $dim, expected_vertices);
                    assert_eq!(tri.number_of_cells(), 1,
                        "{}D: Should have exactly 1 cell", $dim);
                }
            }
        };
    }

    // 2D: Triangle manifold
    test_is_valid_topology!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

    // 3D: Tetrahedron manifold
    test_is_valid_topology!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );

    // 4D: 4-simplex manifold
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

    // 5D: 5-simplex manifold
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
    fn test_is_valid_topology_empty() {
        // Empty triangulation should pass topology validation
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        assert!(
            tri.is_valid().is_ok(),
            "Empty triangulation should be a valid (empty) manifold"
        );
    }

    #[test]
    fn test_is_valid_pl_manifold_mode_rejects_wedge_at_vertex_in_2d() {
        // This builds the same 2D "wedge at a vertex" configuration as the topology-module
        // unit test, but exercises the Level 3 validation pipeline and TopologyGuarantee gating.
        //
        // The complex is a pseudomanifold (every edge has degree 2), but not a PL 2-manifold:
        // the shared vertex has a disconnected link (two disjoint cycles).
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        // Shared vertex.
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();

        // First tetrahedron boundary (4 triangles on 4 vertices).
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v3 = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v2, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v1, v2, v3], None).unwrap())
            .unwrap();

        // Second tetrahedron boundary (shares only v0).
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 10.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(vertex!([11.0, 10.0]))
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 11.0]))
            .unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v4, v5], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v4, v6], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v5, v6], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v4, v5, v6], None).unwrap())
            .unwrap();

        // Ensure neighbor pointers exist so connectedness validation is meaningful.
        repair_neighbor_pointers(&mut tds).unwrap();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        // Default is PL-manifold mode; relax to pseudomanifold for this part of the test.
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        // In pseudomanifold mode, Level 3 validation proceeds past manifold checks and fails at
        // connectedness (two components that share only a vertex).
        assert!(matches!(
            tri.is_valid(),
            Err(TriangulationValidationError::Tds(
                TdsValidationError::InconsistentDataStructure { .. }
            ))
        ));

        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

        // In strict PL-manifold mode, Level 3 validation should fail due to link validation.
        // Depending on validation order, this may surface as ridge-link or vertex-link failure.
        match tri.is_valid() {
            Err(TriangulationValidationError::VertexLinkNotManifold {
                vertex_key,
                link_vertex_count,
                link_cell_count,
                boundary_facet_count,
                max_degree,
                connected,
                interior_vertex,
            }) => {
                assert_eq!(vertex_key, v0);
                assert!(interior_vertex);
                assert!(link_vertex_count > 0);
                assert!(link_cell_count > 0);
                assert_eq!(boundary_facet_count, 0);
                assert_eq!(max_degree, 2);
                assert!(!connected);
            }
            Err(TriangulationValidationError::RidgeLinkNotManifold { .. }) => {}
            other => panic!(
                "Expected RidgeLinkNotManifold or VertexLinkNotManifold in strict PL-manifold mode, got {other:?}"
            ),
        }
    }

    #[test]
    fn test_is_valid_pl_manifold_mode_rejects_cone_on_torus_in_3d_even_when_cell_graph_connected() {
        // Cone over a triangulated torus:
        // - The 3D cell neighbor graph is connected.
        // - Facet-degree and closed-boundary checks pass.
        // - But the apex has link T^2 (χ=0), so PL-manifold vertex-link validation must fail.
        const N: usize = 3;
        const M: usize = 3;

        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        let mut v: [[VertexKey; M]; N] = [[VertexKey::from(KeyData::from_ffi(0)); M]; N];
        for (i, row) in v.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                let i_f = <f64 as std::convert::From<u32>>::from(u32::try_from(i).unwrap());
                let j_f = <f64 as std::convert::From<u32>>::from(u32::try_from(j).unwrap());
                *slot = tds
                    .insert_vertex_with_mapping(vertex!([i_f, j_f, 0.0]))
                    .unwrap();
            }
        }

        let apex = tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.5, 1.0]))
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
                        .insert_cell_with_mapping(
                            Cell::new(vec![tri[0], tri[1], tri[2], apex], None).unwrap(),
                        )
                        .unwrap();
                }
            }
        }

        repair_neighbor_pointers(&mut tds).unwrap();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // Sanity: the cell neighbor graph is connected.
        tri.validate_global_connectedness().unwrap();

        // Sanity: pseudomanifold-with-boundary checks pass.
        let facet_to_cells = tri.tds.build_facet_to_cells_map().unwrap();
        validate_facet_degree(&facet_to_cells).unwrap();
        validate_closed_boundary(&tri.tds, &facet_to_cells).unwrap();

        tri.set_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

        match tri.is_valid() {
            Err(TriangulationValidationError::VertexLinkNotManifold {
                vertex_key,
                link_vertex_count,
                link_cell_count,
                boundary_facet_count,
                connected,
                interior_vertex,
                ..
            }) => {
                assert_eq!(vertex_key, apex);
                assert!(interior_vertex);
                assert!(connected);
                assert_eq!(boundary_facet_count, 0);
                assert!(link_vertex_count > 0);
                assert!(link_cell_count > 0);
            }
            other => panic!("Expected VertexLinkNotManifold for cone apex, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_errors_on_non_manifold_boundary_ridge_before_connectedness() {
        // Two tetrahedra that share an edge but not a facet create a non-manifold boundary:
        // the shared edge is incident to 4 boundary triangles.
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        // Shared edge
        let shared_edge_v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let shared_edge_v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();

        // First tetrahedron
        let tet1_v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let tet1_v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();

        // Second tetrahedron
        let tet2_v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0, 0.0]))
            .unwrap();
        let tet2_v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, -1.0]))
            .unwrap();

        let _ = tds
            .insert_cell_with_mapping(
                Cell::new(vec![shared_edge_v0, shared_edge_v1, tet1_v2, tet1_v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(
                Cell::new(vec![shared_edge_v0, shared_edge_v1, tet2_v2, tet2_v3], None).unwrap(),
            )
            .unwrap();

        let tri = Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // Ordering check: Level 3 should fail for boundary ridge multiplicity before connectedness.
        match tri.is_valid() {
            Err(TriangulationValidationError::BoundaryRidgeMultiplicity {
                boundary_facet_count,
                ..
            }) => {
                assert_eq!(boundary_facet_count, 4);
            }
            other => panic!("Expected BoundaryRidgeMultiplicity, got {other:?}"),
        }
    }
    #[test]
    fn test_validate_includes_tds_validation() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        // Triangulation::validate should pass if the underlying TDS validates.
        assert!(tri.tds.validate().is_ok(), "TDS should validate");
        assert!(
            tri.validate().is_ok(),
            "Triangulation::validate should pass"
        );
    }

    #[test]
    fn test_is_valid_rejects_bootstrap_phase_with_isolated_vertex() {
        // A triangulation with vertices but no cells is not a valid manifold (Level 3).
        // Level 3 requires every vertex to be incident to at least one cell.
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        // Bootstrap insertion (no cells yet)
        tri.insert(vertex!([0.0, 0.0, 0.0]), None, None)
            .expect("bootstrap insertion should succeed");

        match tri.is_valid() {
            Err(TriangulationValidationError::Tds(
                TdsValidationError::InconsistentDataStructure { message },
            )) => {
                assert!(
                    message.contains("Isolated vertex detected"),
                    "Expected isolated-vertex diagnostic, got message: {message}"
                );
            }
            other => panic!("Expected isolated-vertex error, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_rejects_isolated_vertex_even_when_cells_exist() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // Default is PL-manifold mode; use pseudomanifold mode here so the isolated-vertex check
        // triggers before vertex-link validation.
        tri.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);

        // Insert a vertex into the TDS without adding any cells that reference it.
        // This creates an isolated vertex, which violates the Level 3 manifold invariant.
        let _isolated_vk = tri
            .tds
            .insert_vertex_with_mapping(vertex!([10.0, 10.0, 10.0]))
            .unwrap();

        match tri.is_valid() {
            Err(TriangulationValidationError::Tds(
                TdsValidationError::InconsistentDataStructure { message },
            )) => {
                assert!(
                    message.contains("Isolated vertex detected"),
                    "Expected isolated-vertex diagnostic, got message: {message}"
                );
            }
            other => panic!("Expected isolated-vertex error, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_rejects_disconnected_even_when_euler_matches() {
        // Construct a disconnected 1D triangulation made of:
        // - A path (Ball(1)) with χ = 1
        // - A cycle (ClosedSphere(1)) with χ = 0
        //
        // The overall complex has boundary, so it is classified as Ball(1) with expected χ = 1.
        // Euler characteristic alone therefore cannot detect disconnectedness here.
        let mut tds: Tds<f64, (), (), 1> = Tds::empty();

        // Path component: v0 - v1 - v2 (2 edges)
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([2.0])).unwrap();

        let e0 = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1], None).unwrap())
            .unwrap();
        let e1 = tds
            .insert_cell_with_mapping(Cell::new(vec![v1, v2], None).unwrap())
            .unwrap();

        // Cycle component: v3 - v4 - v5 - v3 (3 edges)
        let v3 = tds.insert_vertex_with_mapping(vertex!([10.0])).unwrap();
        let v4 = tds.insert_vertex_with_mapping(vertex!([11.0])).unwrap();
        let v5 = tds.insert_vertex_with_mapping(vertex!([12.0])).unwrap();

        let c0 = tds
            .insert_cell_with_mapping(Cell::new(vec![v3, v4], None).unwrap())
            .unwrap();
        let c1 = tds
            .insert_cell_with_mapping(Cell::new(vec![v4, v5], None).unwrap())
            .unwrap();
        let c2 = tds
            .insert_cell_with_mapping(Cell::new(vec![v5, v3], None).unwrap())
            .unwrap();

        // Set neighbor pointers (1D: each cell has 2 "facets" => 2 neighbor slots).

        // Path neighbors:
        {
            let cell = tds.get_cell_by_key_mut(e0).unwrap();
            let mut neighbors = NeighborBuffer::<Option<CellKey>>::new();
            neighbors.resize(2, None);
            // e0 = [v0, v1]; across v1 is facet_index=0
            neighbors[0] = Some(e1);
            cell.neighbors = Some(neighbors);
        }
        {
            let cell = tds.get_cell_by_key_mut(e1).unwrap();
            let mut neighbors = NeighborBuffer::<Option<CellKey>>::new();
            neighbors.resize(2, None);
            // e1 = [v1, v2]; across v1 is facet_index=1
            neighbors[1] = Some(e0);
            cell.neighbors = Some(neighbors);
        }

        // Cycle neighbors:
        {
            let cell = tds.get_cell_by_key_mut(c0).unwrap();
            let mut neighbors = NeighborBuffer::<Option<CellKey>>::new();
            neighbors.resize(2, None);
            // c0 = [v3, v4]; across v4 is facet_index=0, across v3 is facet_index=1
            neighbors[0] = Some(c1); // at v4
            neighbors[1] = Some(c2); // at v3
            cell.neighbors = Some(neighbors);
        }
        {
            let cell = tds.get_cell_by_key_mut(c1).unwrap();
            let mut neighbors = NeighborBuffer::<Option<CellKey>>::new();
            neighbors.resize(2, None);
            // c1 = [v4, v5]; across v5 is facet_index=0, across v4 is facet_index=1
            neighbors[0] = Some(c2); // at v5
            neighbors[1] = Some(c0); // at v4
            cell.neighbors = Some(neighbors);
        }
        {
            let cell = tds.get_cell_by_key_mut(c2).unwrap();
            let mut neighbors = NeighborBuffer::<Option<CellKey>>::new();
            neighbors.resize(2, None);
            // c2 = [v5, v3]; across v3 is facet_index=0, across v5 is facet_index=1
            neighbors[0] = Some(c0); // at v3
            neighbors[1] = Some(c1); // at v5
            cell.neighbors = Some(neighbors);
        }

        tds.assign_incident_cells().unwrap();

        let tri = Triangulation::<FastKernel<f64>, (), (), 1>::new_with_tds(FastKernel::new(), tds);

        // Sanity: codimension-1 pseudomanifold facet multiplicity passes.
        let facet_to_cells = tri.tds.build_facet_to_cells_map().unwrap();
        validate_facet_degree(&facet_to_cells).unwrap();

        // Sanity: Euler characteristic check would pass for this disconnected complex.
        let topology = validate_triangulation_euler(&tri.tds).unwrap();
        assert_eq!(
            topology.classification,
            TopologyClassification::Ball(1),
            "Classification should be Ball(1) because the complex has boundary"
        );
        assert_eq!(topology.expected, Some(1));
        assert_eq!(topology.chi, 1);

        // Level 3 should still fail due to disconnectedness.
        match tri.is_valid() {
            Err(TriangulationValidationError::Tds(
                TdsValidationError::InconsistentDataStructure { message },
            )) => {
                assert!(
                    message.contains("Disconnected triangulation"),
                    "Expected disconnectedness error, got message: {message}"
                );
            }
            other => panic!("Expected disconnectedness error, got {other:?}"),
        }
    }

    #[test]
    fn test_tds_is_valid_rejects_boundary_facet_has_neighbor() {
        // Create two disjoint tetrahedra and manually introduce an invalid neighbor pointer
        // across a boundary facet.
        let vertices_cell_1 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices_cell_1)
                .unwrap();
        let first_cell_key = tds.cell_keys().next().unwrap();

        // Add a disjoint second tetrahedron.
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 0.0, 0.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(vertex!([11.0, 0.0, 0.0]))
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 1.0, 0.0]))
            .unwrap();
        let v7 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 0.0, 1.0]))
            .unwrap();

        let cell_2 = Cell::new(vec![v4, v5, v6, v7], None).unwrap();
        let second_cell_key = tds.insert_cell_with_mapping(cell_2).unwrap();

        // Invalidate: boundary facet has a neighbor pointer.
        let first_cell = tds.get_cell_by_key_mut(first_cell_key).unwrap();
        let mut neighbors = crate::core::collections::NeighborBuffer::<Option<CellKey>>::new();
        neighbors.resize(4, None);
        neighbors[0] = Some(second_cell_key);
        first_cell.neighbors = Some(neighbors);

        match tds.is_valid() {
            Err(TdsValidationError::InvalidNeighbors { message }) => {
                assert!(
                    message.contains("Boundary facet"),
                    "Unexpected message: {message}"
                );
                assert!(
                    message.contains(&format!("{second_cell_key:?}")),
                    "Expected message to reference {second_cell_key:?}, got: {message}"
                );
            }
            other => panic!("Expected InvalidNeighbors, got {other:?}"),
        }
    }

    #[test]
    fn test_tds_is_valid_rejects_interior_facet_neighbor_mismatch() {
        // Two tetrahedra share a facet, but we leave neighbor pointers unset.
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 2.0]))
            .unwrap();

        let cell_1 = Cell::new(vec![v0, v1, v2, v3], None).unwrap();
        let cell_2 = Cell::new(vec![v0, v1, v2, v4], None).unwrap();
        let _ = tds.insert_cell_with_mapping(cell_1).unwrap();
        let _ = tds.insert_cell_with_mapping(cell_2).unwrap();

        match tds.is_valid() {
            Err(TdsValidationError::InvalidNeighbors { message }) => {
                assert!(
                    message.contains("Interior facet"),
                    "Unexpected message: {message}"
                );
                assert!(
                    message.contains("inconsistent neighbor pointers"),
                    "Unexpected message: {message}"
                );
            }
            other => panic!("Expected InvalidNeighbors, got {other:?}"),
        }
    }

    #[test]
    fn test_is_valid_non_manifold_facet_multiplicity() {
        // Three tetrahedra share a single facet -> not a manifold-with-boundary.
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 2.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 3.0]))
            .unwrap();

        let cell_1 = Cell::new(vec![v0, v1, v2, v3], None).unwrap();
        let cell_2 = Cell::new(vec![v0, v1, v2, v4], None).unwrap();
        let cell_3 = Cell::new(vec![v0, v1, v2, v5], None).unwrap();

        let _ = tds.insert_cell_with_mapping(cell_1).unwrap();
        let _ = tds.insert_cell_with_mapping(cell_2).unwrap();
        let _ = tds.insert_cell_with_mapping(cell_3).unwrap();

        let tri = Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        match tri.is_valid() {
            Err(TriangulationValidationError::ManifoldFacetMultiplicity { cell_count, .. }) => {
                assert_eq!(cell_count, 3);
            }
            other => panic!("Expected ManifoldFacetMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn test_triangulation_validation_report_ok_for_valid_simplex() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        assert!(tri.validation_report().is_ok());
    }

    #[test]
    fn test_triangulation_validation_report_returns_mapping_failures_only() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // Break UUID↔key mappings: remove one vertex UUID entry.
        let uuid = tri.tds.vertices().next().unwrap().1.uuid();
        tri.tds.uuid_to_vertex_key.remove(&uuid);

        let report = tri.validation_report().unwrap_err();
        assert!(!report.violations.is_empty());
        assert!(report.violations.iter().all(|v| {
            matches!(
                v.kind,
                InvariantKind::VertexMappings | InvariantKind::CellMappings
            )
        }));
    }

    #[test]
    fn test_triangulation_validation_report_includes_vertex_and_cell_validity() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);

        // Insert an invalid vertex (nil UUID) to exercise VertexValidity reporting.
        let invalid_vertex: Vertex<f64, (), 3> = Vertex::empty();
        let _ = tri.tds.insert_vertex_with_mapping(invalid_vertex).unwrap();

        // Corrupt one cell locally: neighbors buffer with the wrong length.
        let cell_key = tri.tds.cell_keys().next().unwrap();
        let cell = tri.tds.get_cell_by_key_mut(cell_key).unwrap();
        let mut bad_neighbors = crate::core::collections::NeighborBuffer::<Option<CellKey>>::new();
        bad_neighbors.resize(3, None); // expected D+1 = 4
        cell.neighbors = Some(bad_neighbors);

        let report = tri.validation_report().unwrap_err();

        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::VertexValidity),
            "Report should include a VertexValidity violation"
        );
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::CellValidity),
            "Report should include a CellValidity violation"
        );
    }

    #[test]
    fn test_insert_duplicate_coordinates_skips_with_statistics_and_errors_without() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        // First insertion succeeds.
        tri.insert(vertex!([0.0, 0.0, 0.0]), None, None)
            .expect("first insertion should succeed");
        assert_eq!(tri.number_of_vertices(), 1);

        // Second insertion at same coordinates: insert() returns Err, insert_with_statistics() reports Skipped.
        let err = tri
            .insert(vertex!([0.0, 0.0, 0.0]), None, None)
            .unwrap_err();
        assert!(matches!(err, InsertionError::DuplicateCoordinates { .. }));

        let (outcome, stats) = tri
            .insert_with_statistics(vertex!([0.0, 0.0, 0.0]), None, None)
            .unwrap();
        assert!(stats.skipped());
        assert!(matches!(outcome, InsertionOutcome::Skipped { .. }));

        // No new vertex should have been inserted.
        assert_eq!(tri.number_of_vertices(), 1);
    }

    #[test]
    fn test_insert_duplicate_uuid_is_non_retryable_and_rolls_back() {
        // Insert a vertex, then attempt to insert another vertex with the same UUID.
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        tri.insert(vertex!([0.0, 0.0, 0.0]), None, None)
            .expect("first insertion should succeed");
        assert_eq!(tri.number_of_vertices(), 1);

        let existing_uuid = tri.tds.vertices().next().unwrap().1.uuid();
        let mut dup = vertex!([1.0, 0.0, 0.0]);
        dup.set_uuid(existing_uuid).unwrap();

        let err = tri.insert(dup, None, None).unwrap_err();
        assert!(
            !err.is_retryable(),
            "Duplicate UUID should be non-retryable"
        );

        // Ensure rollback: vertex count unchanged.
        assert_eq!(tri.number_of_vertices(), 1);
    }

    #[test]
    fn test_build_initial_simplex_insufficient_vertices() {
        // Try to build 3D simplex with only 2 vertices (need 4)
        let vertices = vec![vertex!([0.0, 0.0, 0.0]), vertex!([1.0, 0.0, 0.0])];

        let result = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert!(result.is_err());
        match result {
            Err(TriangulationConstructionError::InsufficientVertices { dimension, .. }) => {
                assert_eq!(dimension, 3);
            }
            _ => panic!("Expected InsufficientVertices error"),
        }
    }

    #[test]
    fn test_build_initial_simplex_too_many_vertices() {
        // Try to build 2D simplex with 4 vertices (need exactly 3)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert!(result.is_err());
        match result {
            Err(TriangulationConstructionError::InsufficientVertices { .. }) => {}
            _ => panic!("Expected InsufficientVertices error for wrong count"),
        }
    }

    #[test]
    fn test_build_initial_simplex_with_user_data() {
        // Build vertices with user data
        let v1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .data(42_usize)
            .build()
            .unwrap();
        let v2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .data(43_usize)
            .build()
            .unwrap();
        let v3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .data(44_usize)
            .build()
            .unwrap();

        let vertices = vec![v1, v2, v3];
        let tds = Triangulation::<FastKernel<f64>, usize, (), 2>::build_initial_simplex(&vertices)
            .unwrap();

        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.number_of_cells(), 1);

        // Verify user data is preserved
        let data_values: Vec<_> = tds
            .vertices()
            .filter_map(|(_, v)| v.data.as_ref())
            .copied()
            .collect();
        assert_eq!(data_values.len(), 3);
        assert!(data_values.contains(&42));
        assert!(data_values.contains(&43));
        assert!(data_values.contains(&44));
    }

    // =============================================================================
    // Tests for build_initial_simplex degeneracy validation
    // =============================================================================

    #[test]
    fn test_build_initial_simplex_rejects_collinear_2d() {
        // Collinear points should be rejected by build_initial_simplex
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, 0.0]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert!(result.is_err(), "Collinear points should be rejected");
        match result {
            Err(TriangulationConstructionError::GeometricDegeneracy { message }) => {
                assert!(
                    message.contains("Degenerate"),
                    "Error message should mention degeneracy"
                );
            }
            _ => panic!("Expected GeometricDegeneracy error for collinear points"),
        }
    }

    #[test]
    fn test_build_initial_simplex_rejects_coplanar_3d() {
        // Coplanar points should be rejected by build_initial_simplex
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.5, 0.5, 0.0]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert!(result.is_err(), "Coplanar points should be rejected");
        match result {
            Err(TriangulationConstructionError::GeometricDegeneracy { message }) => {
                assert!(
                    message.contains("Degenerate") || message.contains("coplanar"),
                    "Error message should mention degeneracy or coplanarity"
                );
            }
            _ => panic!("Expected GeometricDegeneracy error for coplanar points"),
        }
    }

    /// Consolidated macro for facet validation tests across dimensions.
    ///
    /// Verifies the manifold topology invariant: each facet shared by at most 2 cells.
    /// Consolidates detection and repair tests into comprehensive suites.
    macro_rules! test_facet_validation {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_detect_local_facet_issues_ $dim d>]() {
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let tri = Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                    // Valid simplex: should have no issues
                    let cell_keys: Vec<_> = tri.tds.cell_keys().collect();
                    assert_eq!(cell_keys.len(), 1);
                    let issues = tri.detect_local_facet_issues(&cell_keys).unwrap();
                    assert!(issues.is_none(), "{}D: Valid simplex should have no facet issues", $dim);

                    // Empty list: should return None
                    let issues = tri.detect_local_facet_issues(&[]).unwrap();
                    assert!(issues.is_none(), "{}D: Empty list should have no issues", $dim);

                    // Nonexistent cells: should be skipped gracefully
                    let fake_keys = vec![CellKey::default()];
                    let issues = tri.detect_local_facet_issues(&fake_keys).unwrap();
                    assert!(issues.is_none(), "{}D: Nonexistent cells should be skipped", $dim);

                    // Verify neighbors (all should be None for single cell)
                    let (_, cell) = tri.tds.cells().next().unwrap();
                    if let Some(neighbors) = cell.neighbors() {
                        assert!(neighbors.iter().all(|n| n.is_none()),
                            "{}D: Single cell should have no neighbors", $dim);
                    }
                }

                #[test]
                fn [<test_repair_local_facet_issues_ $dim d>]() {
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let mut tri = Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(FastKernel::new(), tds);

                    // Empty issues map: should remove nothing
                    let empty_issues = FacetIssuesMap::default();
                    let removed = tri.repair_local_facet_issues(&empty_issues).unwrap();
                    assert_eq!(removed, 0, "{}D: Empty issues should remove 0 cells", $dim);
                    assert_eq!(tri.tds.number_of_cells(), 1, "{}D: Should still have 1 cell", $dim);
                }
            }
        };
    }

    /// Dimension-parametric `remove_vertex` tests.
    ///
    /// Verifies that vertex removal maintains neighbor pointer integrity and
    /// triangulation validity across dimensions.
    macro_rules! test_remove_vertex {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?], $interior_point:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_remove_vertex_neighbor_pointers_ $dim d>]() {
                    // Build triangulation with D+1 simplex vertices + 1 interior point
                    let vertices: Vec<Vertex<f64, (), $dim>> = {
                        let mut v = vec![$(vertex!($simplex_coords)),+];
                        v.push(vertex!($interior_point));
                        v
                    };

                    let mut dt = DelaunayTriangulation::new(&vertices)
                        .expect("Failed to create triangulation");

                    // Find and remove the interior vertex
                    let interior_vertex = dt
                        .vertices()
                        .find(|(_, v)| {
                            let coords = v.point().coords();
                            coords.iter()
                                .zip($interior_point.iter())
                                .all(|(a, b)| (a - b).abs() < 1e-10)
                        })
                        .map(|(_, v)| *v)
                        .expect("Interior vertex not found");

                    let initial_cell_count = dt.tds().number_of_cells();
                    dt.remove_vertex(&interior_vertex)
                        .expect("Failed to remove vertex");

                    // After removal, should have fewer cells (or same if just 1 simplex left)
                    assert!(dt.tds().number_of_cells() <= initial_cell_count,
                        "{}D: Cell count should not increase after removal", $dim);

                    // Verify neighbor pointer consistency:
                    // 1. No dangling pointers (all neighbor keys exist)
                    // 2. Neighbor relationships are symmetric
                    for (cell_key, cell) in dt.tds().cells() {
                        if let Some(neighbors) = cell.neighbors() {
                            for (facet_idx, neighbor_opt) in neighbors.iter().enumerate() {
                                if let Some(neighbor_key) = neighbor_opt {
                                    // Verify neighbor exists
                                    assert!(
                                        dt.tds().contains_cell(*neighbor_key),
                                        "{}D: Cell {cell_key:?} has neighbor pointer to non-existent cell {neighbor_key:?}",
                                        $dim
                                    );

                                    // Verify symmetry: neighbor should point back to us
                                    let neighbor_cell = dt
                                        .tds()
                                        .get_cell(*neighbor_key)
                                        .expect("Neighbor cell should exist");
                                    if let Some(neighbor_neighbors) = neighbor_cell.neighbors() {
                                        let points_back = neighbor_neighbors
                                            .iter()
                                            .any(|n| n.as_ref() == Some(&cell_key));
                                        assert!(
                                            points_back,
                                            "{}D: Cell {cell_key:?} has neighbor {neighbor_key:?} at facet {facet_idx}, but neighbor doesn't point back",
                                            $dim
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Verify triangulation is still valid (Levels 1–3; removal does not guarantee Delaunay)
                    let validation = dt.as_triangulation().validate();
                    assert!(
                        validation.is_ok(),
                        "{}D: Triangulation should be structurally valid after vertex removal: {:?}",
                        $dim,
                        validation.err()
                    );
                }
            }
        };
    }

    /// Basic accessor tests across dimensions.
    macro_rules! test_basic_accessors {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_basic_accessors_ $dim d>]() {
                    // Empty triangulation
                    let empty: Triangulation<FastKernel<f64>, (), (), $dim> =
                        Triangulation::new_empty(FastKernel::new());
                    assert_eq!(empty.number_of_vertices(), 0);
                    assert_eq!(empty.number_of_cells(), 0);
                    assert_eq!(empty.dim(), -1);
                    assert_eq!(empty.cells().count(), 0);
                    assert_eq!(empty.vertices().count(), 0);
                    assert_eq!(empty.facets().count(), 0);
                    assert_eq!(empty.boundary_facets().count(), 0);

                    // Simplex triangulation
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];
                    let expected_vertex_count = vertices.len();

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let tri = Triangulation::<FastKernel<f64>, (), (), $dim>::new_with_tds(
                        FastKernel::new(),
                        tds,
                    );

                    assert_eq!(tri.number_of_vertices(), expected_vertex_count);
                    assert_eq!(tri.number_of_cells(), 1);
                    assert_eq!(tri.dim(), $dim as i32);
                    assert_eq!(tri.cells().count(), 1);
                    assert_eq!(tri.vertices().count(), expected_vertex_count);

                    // D-simplex has D+1 facets, all on boundary
                    let facet_count = tri.facets().count();
                    assert_eq!(facet_count, expected_vertex_count, "{}D: D-simplex should have D+1 facets", $dim);
                    let boundary_count = tri.boundary_facets().count();
                    assert_eq!(boundary_count, expected_vertex_count, "{}D: All facets should be on boundary", $dim);
                }
            }
        };
    }

    // Facet validation tests (2D - 5D)
    test_facet_validation!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    test_facet_validation!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
    test_facet_validation!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    test_facet_validation!(
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

    // Basic accessor tests (2D - 5D)
    test_basic_accessors!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    test_basic_accessors!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
    test_basic_accessors!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    test_basic_accessors!(
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

    // Remove vertex tests (2D - 5D)
    test_remove_vertex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.3, 0.3]);
    test_remove_vertex!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [0.25, 0.25, 0.25]
    );
    test_remove_vertex!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2, 0.2]
    );
    test_remove_vertex!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        [0.16, 0.16, 0.16, 0.16, 0.16]
    );

    // =============================================================================
    // Public Topology Traversal & Adjacency API (Read-only)
    // =============================================================================

    #[test]
    fn test_topology_edges_triangle_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        assert_eq!(tri.number_of_cells(), 1);
        assert_eq!(tri.number_of_vertices(), 3);
        assert_eq!(tri.number_of_edges(), 3);

        let edges: std::collections::HashSet<_> = tri.edges().collect();
        assert_eq!(edges.len(), 3);

        let index = tri.build_adjacency_index().unwrap();
        let edges_with_index: std::collections::HashSet<_> = tri.edges_with_index(&index).collect();
        assert_eq!(edges_with_index, edges);
        assert_eq!(tri.number_of_edges_with_index(&index), 3);

        // Edge endpoints should always be vertex keys from this triangulation.
        assert!(edges.iter().all(|e| {
            let (a, b) = e.endpoints();
            a != b && tri.vertex_coords(a).is_some() && tri.vertex_coords(b).is_some()
        }));
    }

    #[test]
    fn test_topology_edges_and_incident_edges_double_tetrahedron_3d() {
        // Two tetrahedra sharing a triangular facet.
        let vertices: Vec<_> = vec![
            // Shared triangle
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([1.0, 2.0, 0.0]),
            // Two apices
            vertex!([1.0, 0.7, 1.5]),
            vertex!([1.0, 0.7, -1.5]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        assert_eq!(tri.number_of_cells(), 2);
        assert_eq!(tri.number_of_vertices(), 5);

        // This configuration has 9 unique edges (3 base + 6 apex-to-base).
        assert_eq!(tri.number_of_edges(), 9);

        // A base vertex has degree 4: two base edges + two apex edges.
        let base_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [0.0, 0.0, 0.0]).then_some(vk))
            .unwrap();
        assert_eq!(tri.number_of_incident_edges(base_vertex_key), 4);

        let index = tri.build_adjacency_index().unwrap();
        assert_eq!(tri.number_of_edges_with_index(&index), 9);

        // A base vertex is incident to both cells.
        assert_eq!(tri.adjacent_cells(base_vertex_key).count(), 2);
        assert_eq!(
            tri.adjacent_cells_with_index(&index, base_vertex_key)
                .count(),
            2
        );
        assert_eq!(
            tri.number_of_adjacent_cells_with_index(&index, base_vertex_key),
            2
        );

        // A base vertex has degree 4: two base edges + two apex edges.
        assert_eq!(tri.number_of_incident_edges(base_vertex_key), 4);
        assert_eq!(
            tri.incident_edges_with_index(&index, base_vertex_key)
                .count(),
            4
        );
        assert_eq!(
            tri.number_of_incident_edges_with_index(&index, base_vertex_key),
            4
        );

        // An apex has degree 3: connected to all three base vertices.
        let apex_vertex_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [1.0, 0.7, 1.5]).then_some(vk))
            .unwrap();
        assert_eq!(tri.number_of_incident_edges(apex_vertex_key), 3);
        assert_eq!(
            tri.adjacent_cells_with_index(&index, apex_vertex_key)
                .count(),
            1
        );
        assert_eq!(
            tri.number_of_adjacent_cells_with_index(&index, apex_vertex_key),
            1
        );

        // Each cell has exactly one neighbor in the index.
        let cell_keys: Vec<_> = tri.cells().map(|(ck, _)| ck).collect();
        for &ck in &cell_keys {
            assert_eq!(tri.cell_neighbors_with_index(&index, ck).count(), 1);
            assert_eq!(tri.number_of_cell_neighbors_with_index(&index, ck), 1);
        }
    }

    #[test]
    fn test_topology_queries_missing_keys_are_empty_or_none() {
        // Use a "null" SlotMap key, which should never be present in a valid triangulation.
        let vertices_a = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt_a: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices_a).unwrap();
        let tri_a = dt_a.as_triangulation();

        let index = tri_a.build_adjacency_index().unwrap();

        let missing_vertex_key = VertexKey::default();
        assert_eq!(tri_a.adjacent_cells(missing_vertex_key).count(), 0);
        assert_eq!(
            tri_a
                .adjacent_cells_with_index(&index, missing_vertex_key)
                .count(),
            0
        );
        assert_eq!(
            tri_a.number_of_adjacent_cells_with_index(&index, missing_vertex_key),
            0
        );

        assert_eq!(tri_a.incident_edges(missing_vertex_key).count(), 0);
        assert_eq!(
            tri_a
                .incident_edges_with_index(&index, missing_vertex_key)
                .count(),
            0
        );
        assert_eq!(tri_a.number_of_incident_edges(missing_vertex_key), 0);
        assert_eq!(
            tri_a.number_of_incident_edges_with_index(&index, missing_vertex_key),
            0
        );
        assert!(tri_a.vertex_coords(missing_vertex_key).is_none());

        let missing_cell_key = CellKey::default();
        assert_eq!(tri_a.cell_neighbors(missing_cell_key).count(), 0);
        assert_eq!(
            tri_a
                .cell_neighbors_with_index(&index, missing_cell_key)
                .count(),
            0
        );
        assert_eq!(
            tri_a.number_of_cell_neighbors_with_index(&index, missing_cell_key),
            0
        );
        assert!(tri_a.cell_vertices(missing_cell_key).is_none());
    }

    #[test]
    fn test_topology_geometry_accessors_roundtrip() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let v_key = tri
            .vertices()
            .find_map(|(vk, _)| (tri.vertex_coords(vk)? == [1.0, 0.0]).then_some(vk))
            .unwrap();
        assert_eq!(tri.vertex_coords(v_key).unwrap(), [1.0, 0.0]);

        let cell_key = tri.cells().next().unwrap().0;
        let cell_vertices = tri.cell_vertices(cell_key).unwrap();
        assert_eq!(cell_vertices.len(), 3);
        assert!(cell_vertices.contains(&v_key));
    }

    #[test]
    fn test_build_adjacency_index_basic_invariants() {
        // Two tetrahedra sharing a triangular facet.
        let vertices: Vec<_> = vec![
            // Shared triangle
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([1.0, 2.0, 0.0]),
            // Two apices
            vertex!([1.0, 0.7, 1.5]),
            vertex!([1.0, 0.7, -1.5]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.as_triangulation();

        let index = tri.build_adjacency_index().unwrap();

        // Each cell has exactly one neighbor.
        let cell_keys: Vec<_> = tri.cells().map(|(ck, _)| ck).collect();
        assert_eq!(cell_keys.len(), 2);
        for &ck in &cell_keys {
            let neighbors = index.cell_to_neighbors.get(&ck).unwrap();
            assert_eq!(neighbors.len(), 1);
            assert!(cell_keys.contains(&neighbors[0]));
            assert_ne!(neighbors[0], ck);
        }

        // For every vertex, edges/cells lists exist and are consistent.
        for (vk, _) in tri.vertices() {
            let cells = index.vertex_to_cells.get(&vk).unwrap();
            assert!(!cells.is_empty());

            let edges = index.vertex_to_edges.get(&vk).unwrap();
            assert!(!edges.is_empty());
            assert!(
                edges
                    .iter()
                    .all(|e| matches!(e.endpoints(), (a, b) if a == vk || b == vk))
            );
        }
    }

    #[test]
    fn test_build_adjacency_index_empty_triangulation_is_empty() {
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        let index = tri.build_adjacency_index().unwrap();
        assert!(index.vertex_to_cells.is_empty());
        assert!(index.cell_to_neighbors.is_empty());
        assert!(index.vertex_to_edges.is_empty());
    }

    // =============================================================================
    // Triangulation insert_with_statistics tests (internal API)
    // =============================================================================

    #[test]
    fn triangulation_insert_with_statistics_basic_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Insert first vertex
        let (outcome, stats) = tri
            .insert_with_statistics(vertex!([0.0, 0.0]), None, None)
            .expect("insertion should succeed");

        assert!(matches!(
            outcome,
            InsertionOutcome::Inserted { hint: None, .. }
        ));
        assert_eq!(stats.attempts, 1);
        assert!(!stats.used_perturbation());
        assert!(!stats.skipped());
        assert!(stats.success());
        assert_eq!(tri.number_of_vertices(), 1);
    }

    #[test]
    fn triangulation_insert_with_statistics_bootstrap_3d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        // Insert D+1 vertices to create initial simplex
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        for (i, v) in vertices.into_iter().enumerate() {
            let (outcome, stats) = tri.insert_with_statistics(v, None, None).unwrap();

            assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
            assert_eq!(stats.attempts, 1);

            if i < 3 {
                // Bootstrap phase - no hint yet
                assert!(matches!(
                    outcome,
                    InsertionOutcome::Inserted { hint: None, .. }
                ));
            } else {
                // After D+1 vertices, hint should be available
                assert!(matches!(
                    outcome,
                    InsertionOutcome::Inserted { hint: Some(_), .. }
                ));
            }
        }

        assert_eq!(tri.number_of_vertices(), 4);
        assert_eq!(tri.number_of_cells(), 1);
    }

    #[test]
    fn triangulation_insert_with_statistics_hint_usage_4d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 4> =
            Triangulation::new_empty(FastKernel::new());

        // Build initial simplex
        for i in 0..5 {
            let mut coords = [0.0; 4];
            if i > 0 {
                coords[i - 1] = 1.0;
            }
            tri.insert_with_statistics(vertex!(coords), None, None)
                .unwrap();
        }

        // Insert with explicit hint
        let hint_cell = tri.cells().next().map(|(key, _)| key);
        let (outcome, stats) = tri
            .insert_with_statistics(vertex!([0.2, 0.2, 0.2, 0.2]), None, hint_cell)
            .unwrap();

        assert!(matches!(
            outcome,
            InsertionOutcome::Inserted { hint: Some(_), .. }
        ));
        assert_eq!(stats.attempts, 1);
        assert!(stats.success());
    }

    #[test]
    fn triangulation_insert_with_statistics_duplicate_coordinates_3d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        // Insert first vertex
        tri.insert_with_statistics(vertex!([1.0, 2.0, 3.0]), None, None)
            .unwrap();

        // Try duplicate - should be skipped
        let result = tri.insert_with_statistics(vertex!([1.0, 2.0, 3.0]), None, None);

        assert!(matches!(
            result,
            Ok((
                InsertionOutcome::Skipped {
                    error: InsertionError::DuplicateCoordinates { .. }
                },
                _
            ))
        ));
    }

    #[test]
    fn triangulation_insert_with_statistics_multiple_insertions_2d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        let points = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 1.0]),
            vertex!([0.3, 0.3]),
            vertex!([0.7, 0.3]),
        ];

        let mut all_succeeded = true;
        let mut max_attempts = 0;

        for point in points {
            match tri.insert_with_statistics(point, None, None) {
                Ok((InsertionOutcome::Inserted { .. }, stats)) => {
                    max_attempts = max_attempts.max(stats.attempts);
                    assert!(stats.success());
                }
                Ok((InsertionOutcome::Skipped { .. }, _)) | Err(_) => {
                    all_succeeded = false;
                }
            }
        }

        assert!(all_succeeded, "all insertions should succeed");
        assert!(max_attempts >= 1);
        assert_eq!(tri.number_of_vertices(), 5);
    }

    #[test]
    fn triangulation_insert_with_statistics_outcome_types() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Test Inserted variant
        let (outcome, _) = tri
            .insert_with_statistics(vertex!([0.0, 0.0]), None, None)
            .unwrap();

        match outcome {
            InsertionOutcome::Inserted { vertex_key, hint } => {
                // Verify we can access the fields
                assert!(tri.vertices().any(|(k, _)| k == vertex_key));
                assert_eq!(hint, None); // No hint during bootstrap
            }
            InsertionOutcome::Skipped { .. } => panic!("expected Inserted, got Skipped"),
        }
    }

    #[test]
    fn triangulation_insert_with_statistics_sequential_5d() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 5> =
            Triangulation::new_empty(FastKernel::new());

        // Insert 6 vertices to form initial simplex
        for i in 0..6 {
            let mut coords = [0.0; 5];
            if i > 0 {
                coords[i - 1] = 1.0;
            }

            let (outcome, stats) = tri
                .insert_with_statistics(vertex!(coords), None, None)
                .unwrap();

            assert!(matches!(outcome, InsertionOutcome::Inserted { .. }));
            assert_eq!(stats.attempts, 1);
            assert!(stats.success());
        }

        assert_eq!(tri.number_of_vertices(), 6);
        assert_eq!(tri.number_of_cells(), 1);
    }

    #[test]
    fn statistics_cells_removed_during_repair() {
        let mut tri: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        // Build simplex
        tri.insert_with_statistics(vertex!([0.0, 0.0]), None, None)
            .unwrap();
        tri.insert_with_statistics(vertex!([1.0, 0.0]), None, None)
            .unwrap();
        tri.insert_with_statistics(vertex!([0.5, 1.0]), None, None)
            .unwrap();

        let cells_before = tri.number_of_cells();

        // Insert interior point - might trigger repair
        let (_outcome, stats) = tri
            .insert_with_statistics(vertex!([0.5, 0.3]), None, None)
            .unwrap();

        let cells_after = tri.number_of_cells();

        // Basic sanity: repair can't remove more cells than existed before insertion.
        assert!(
            stats.cells_removed_during_repair <= cells_before,
            "cells_removed_during_repair ({}) should not exceed cell count before insertion ({}); cells after insertion: {}",
            stats.cells_removed_during_repair,
            cells_before,
            cells_after
        );
    }
}

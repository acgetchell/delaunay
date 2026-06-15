//! Incremental Delaunay insertion using cavity-based algorithm.
//!
//! This module implements efficient incremental insertion following CGAL's approach:
//! 1. Locate the simplex containing the new point (facet walking)
//! 2. Find conflict region (BFS with in_sphere tests)
//! 3. Extract cavity boundary facets
//! 4. Remove conflict simplices
//! 5. Fill cavity (create new simplices connecting boundary to new vertex)
//! 6. Wire neighbors locally (no global assign_neighbors call)
//!
//! ## Hull Extension and Visibility
//!
//! When inserting a vertex outside the current convex hull, the algorithm finds
//! *visible* boundary facets using orientation tests:
//! - A facet is **strictly visible** if the new point and the opposite vertex
//!   have opposite orientations relative to the facet's supporting hyperplane.
//! - **Coplanar cases** for the *query point* (orientation == 0) are treated as
//!   **weakly visible** to avoid missing horizon facets when the point lies on the
//!   hull plane. In **2D**, collinear cases are handled explicitly:
//!   - on-segment points trigger a boundary-edge split
//!   - off-segment collinearity is **not** treated as visible (avoids degenerate triangles)
//!   This still treats degeneracies of the hull facet itself (orientation_with_opposite == 0)
//!   as non-visible.
//! - For numerically robust weak visibility beyond coplanar cases, a threshold-based
//!   approach would be needed (not currently implemented).

#![forbid(unsafe_code)]

use crate::core::algorithms::flips::DelaunayRepairError;
use crate::core::algorithms::locate::{
    ConflictError, LocateError, LocateResult, extract_cavity_boundary,
};
use crate::core::collections::spatial_hash_grid::HashGridIndexError;
use crate::core::collections::{
    FastHashMap, FastHashSet, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE, SimplexKeyBuffer,
    SmallBuffer, VertexKeyBuffer,
};
use crate::core::construction::TriangulationConstructionError;
use crate::core::facet::{FacetError, FacetHandle};
use crate::core::simplex::{NeighborSlot, Simplex, SimplexValidationError};
use crate::core::tds::{
    DelaunayValidationErrorKind, EntityKind, GeometricError, InvariantErrorSummary,
    InvariantErrorSummaryDetail, NeighborValidationError, SimplexKey, Tds, TdsConstructionError,
    TdsError, TdsErrorKind, TriangulationValidationErrorKind, VertexKey,
};
use crate::core::traits::boundary_analysis::BoundaryAnalysis;
use crate::core::traits::data_type::DataType;
use crate::core::validation::TriangulationValidationError;
use crate::core::vertex::VertexValidationError;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::predicates::Orientation;
use crate::geometry::robust_predicates::robust_orientation;
use crate::geometry::traits::coordinate::{
    CoordinateConversionError, CoordinateConversionValue, CoordinateValidationError,
    CoordinateValues, DEFAULT_TOLERANCE_F64, InvalidCoordinateValue,
};
use crate::validation::DelaunayTriangulationValidationError;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Reason for hull extension failure.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::insertion::HullExtensionReason;
///
/// let reason = HullExtensionReason::NoVisibleFacets;
/// std::assert_matches!(reason, HullExtensionReason::NoVisibleFacets);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum HullExtensionReason {
    /// No visible boundary facets (coplanar with hull surface).
    NoVisibleFacets,
    /// Visible facets form an invalid patch.
    InvalidPatch {
        /// Details about why the patch was invalid.
        details: String,
    },
    /// Geometric predicate (orientation / in-sphere) failed.
    ///
    /// Preserves the structured [`CoordinateConversionError`] from the kernel or
    /// robust-predicate evaluation rather than collapsing it into a string.
    PredicateFailed(CoordinateConversionError),
    /// Lower-layer TDS error encountered during hull extension.
    ///
    /// Preserves the structured [`TdsError`] (e.g. from boundary-facet retrieval)
    /// rather than collapsing it into a string.
    Tds(TdsError),
    /// Other failure.
    Other {
        /// Underlying error message.
        message: String,
    },
}

impl fmt::Display for HullExtensionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoVisibleFacets => f.write_str(
                "No visible boundary facets found for exterior vertex (may be coplanar with hull surface)",
            ),
            Self::InvalidPatch { details } => write!(
                f,
                "Visible boundary facets are not a valid patch: {details}"
            ),
            Self::PredicateFailed(source) => {
                write!(f, "Geometric predicate failed: {source}")
            }
            Self::Tds(source) => write!(f, "TDS error: {source}"),
            Self::Other { message } => f.write_str(message),
        }
    }
}

/// Compact, typed summary of a [`TdsError`] used inside insertion-stage errors.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum TdsValidationFailure {
    /// The triangulation contains an invalid vertex.
    #[error("invalid vertex {vertex_id}: {source}")]
    InvalidVertex {
        /// UUID of the invalid vertex.
        vertex_id: uuid::Uuid,
        /// Underlying vertex validation error.
        #[source]
        source: VertexValidationError,
    },

    /// The triangulation contains an invalid simplex.
    #[error("invalid simplex {simplex_id}: {source}")]
    InvalidSimplex {
        /// UUID of the invalid simplex.
        simplex_id: uuid::Uuid,
        /// Underlying simplex validation error.
        #[source]
        source: SimplexValidationError,
    },

    /// Neighbor relationships are invalid.
    #[error("invalid neighbor relationships: {reason}")]
    InvalidNeighbors {
        /// Neighbor validation failure detail.
        #[source]
        reason: NeighborValidationError,
    },

    /// Coherent orientation was violated between adjacent simplices.
    #[error(
        "orientation invariant violated between simplices {simplex1_uuid} and {simplex2_uuid} \
         (facet indices {simplex1_facet_index}/{simplex2_facet_index} counts \
         {facet_vertex_count}/{simplex2_facet_vertex_count}, observed odd permutation \
         {observed_odd_permutation}, expected {expected_odd_permutation})"
    )]
    OrientationViolation {
        /// Key of the first simplex.
        simplex1_key: SimplexKey,
        /// UUID of the first simplex.
        simplex1_uuid: uuid::Uuid,
        /// Key of the second simplex.
        simplex2_key: SimplexKey,
        /// UUID of the second simplex.
        simplex2_uuid: uuid::Uuid,
        /// Facet index in the first simplex.
        simplex1_facet_index: usize,
        /// Facet index in the second simplex.
        simplex2_facet_index: usize,
        /// Number of vertices in the first facet ordering.
        facet_vertex_count: usize,
        /// Number of vertices in the second facet ordering.
        simplex2_facet_vertex_count: usize,
        /// Observed permutation parity.
        observed_odd_permutation: bool,
        /// Expected permutation parity.
        expected_odd_permutation: bool,
    },

    /// Duplicate simplices were detected.
    #[error("duplicate simplices detected: {message}")]
    DuplicateSimplices {
        /// Duplicate-simplex detail.
        message: String,
    },

    /// A simplex insertion or validation pass found a facet incident to too many simplices.
    ///
    /// During insertion preflight, the `candidate_*` fields identify the simplex that
    /// would exceed the PL-manifold facet multiplicity. During post-hoc
    /// validation, they identify one offending incident simplex from the over-shared
    /// facet.
    #[error(
        "facet {facet_key} exceeds incident-simplex limit: observed {attempted_incident_count} incident simplices, max {max_incident_count}; candidate/offending simplex {candidate_simplex_uuid} facet {candidate_facet_index}; other incident simplices {existing_incident_count}"
    )]
    FacetSharingViolation {
        /// Canonical key of the over-shared facet.
        facet_key: u64,
        /// Number of other/pre-existing simplices already incident to the facet.
        existing_incident_count: usize,
        /// Number of incident simplices observed, or that would exist after candidate insertion.
        attempted_incident_count: usize,
        /// Maximum allowed number of incident simplices for a PL-manifold facet.
        max_incident_count: usize,
        /// UUID of the candidate or offending simplex.
        candidate_simplex_uuid: uuid::Uuid,
        /// Facet index on the candidate or offending simplex.
        candidate_facet_index: usize,
    },

    /// Simplex creation failed inside TDS validation.
    #[error("failed to create simplex: {message}")]
    FailedToCreateSimplex {
        /// Simplex creation failure detail.
        message: String,
    },

    /// Simplices were not neighbors as expected.
    #[error("simplices {simplex1} and {simplex2} are not neighbors")]
    NotNeighbors {
        /// First simplex UUID.
        simplex1: uuid::Uuid,
        /// Second simplex UUID.
        simplex2: uuid::Uuid,
    },

    /// Entity mapping became inconsistent.
    #[error("{entity:?} mapping inconsistency: {message}")]
    MappingInconsistency {
        /// Entity with an inconsistent mapping.
        entity: EntityKind,
        /// Mapping inconsistency detail.
        message: String,
    },

    /// Vertex-key retrieval failed for a simplex.
    #[error("failed to retrieve vertex keys for simplex {simplex_id}: {message}")]
    VertexKeyRetrievalFailed {
        /// Simplex UUID.
        simplex_id: uuid::Uuid,
        /// Retrieval failure detail.
        message: String,
    },

    /// A simplex key was missing from storage.
    #[error("simplex key {simplex_key:?} not found: {context}")]
    SimplexNotFound {
        /// Missing simplex key.
        simplex_key: SimplexKey,
        /// Lookup context.
        context: String,
    },

    /// A vertex key was missing from storage.
    #[error("vertex key {vertex_key:?} not found: {context}")]
    VertexNotFound {
        /// Missing vertex key.
        vertex_key: VertexKey,
        /// Lookup context.
        context: String,
    },

    /// A dimensional invariant was violated.
    #[error("dimension mismatch: expected {expected}, got {actual}: {context}")]
    DimensionMismatch {
        /// Expected count.
        expected: usize,
        /// Observed count.
        actual: usize,
        /// Validation context.
        context: String,
    },

    /// An index exceeded the valid range.
    #[error("index out of bounds: index {index}, bound {bound}: {context}")]
    IndexOutOfBounds {
        /// Invalid index.
        index: usize,
        /// Exclusive upper bound.
        bound: usize,
        /// Access context.
        context: String,
    },

    /// Internal TDS consistency failed.
    #[error("internal data structure inconsistency: {message}")]
    InconsistentDataStructure {
        /// Inconsistency detail.
        message: String,
    },

    /// Geometric orientation or predicate validation failed.
    #[error("geometric validation failed: {source}")]
    Geometric {
        /// Underlying geometric validation error.
        #[source]
        source: GeometricError,
    },

    /// Facet validation failed.
    #[error("facet validation failed: {source}")]
    Facet {
        /// Structured facet failure.
        #[source]
        source: FacetError,
    },

    /// A simplex contains duplicate coordinates.
    #[error("duplicate coordinates in simplex {simplex_id}: {message}")]
    DuplicateCoordinatesInSimplex {
        /// UUID of the simplex containing duplicates.
        simplex_id: uuid::Uuid,
        /// Duplicate-coordinate detail.
        message: String,
    },
}

impl From<TdsError> for TdsValidationFailure {
    #[expect(
        clippy::too_many_lines,
        reason = "structured error mapping is intentionally exhaustive; simplex naming pushes it slightly over the lint limit"
    )]
    fn from(source: TdsError) -> Self {
        match source {
            TdsError::InvalidVertex { vertex_id, source } => {
                Self::InvalidVertex { vertex_id, source }
            }
            TdsError::InvalidSimplex { simplex_id, source } => {
                Self::InvalidSimplex { simplex_id, source }
            }
            TdsError::InvalidNeighbors { reason } => Self::InvalidNeighbors { reason },
            TdsError::OrientationViolation {
                simplex1_key,
                simplex1_uuid,
                simplex2_key,
                simplex2_uuid,
                simplex1_facet_index,
                simplex2_facet_index,
                facet_vertices,
                simplex2_facet_vertices,
                observed_odd_permutation,
                expected_odd_permutation,
            } => Self::OrientationViolation {
                simplex1_key,
                simplex1_uuid,
                simplex2_key,
                simplex2_uuid,
                simplex1_facet_index,
                simplex2_facet_index,
                facet_vertex_count: facet_vertices.len(),
                simplex2_facet_vertex_count: simplex2_facet_vertices.len(),
                observed_odd_permutation,
                expected_odd_permutation,
            },
            TdsError::DuplicateSimplices { message } => Self::DuplicateSimplices { message },
            TdsError::FacetSharingViolation {
                facet_key,
                existing_incident_count,
                attempted_incident_count,
                max_incident_count,
                candidate_simplex_uuid,
                candidate_facet_index,
            } => Self::FacetSharingViolation {
                facet_key,
                existing_incident_count,
                attempted_incident_count,
                max_incident_count,
                candidate_simplex_uuid,
                candidate_facet_index,
            },
            TdsError::FailedToCreateSimplex { message } => Self::FailedToCreateSimplex { message },
            TdsError::NotNeighbors { simplex1, simplex2 } => {
                Self::NotNeighbors { simplex1, simplex2 }
            }
            TdsError::MappingInconsistency { entity, message } => {
                Self::MappingInconsistency { entity, message }
            }
            TdsError::VertexKeyRetrievalFailed {
                simplex_id,
                message,
            } => Self::VertexKeyRetrievalFailed {
                simplex_id,
                message,
            },
            TdsError::SimplexNotFound {
                simplex_key,
                context,
            } => Self::SimplexNotFound {
                simplex_key,
                context,
            },
            TdsError::VertexNotFound {
                vertex_key,
                context,
            } => Self::VertexNotFound {
                vertex_key,
                context,
            },
            TdsError::DimensionMismatch {
                expected,
                actual,
                context,
            } => Self::DimensionMismatch {
                expected,
                actual,
                context,
            },
            TdsError::IndexOutOfBounds {
                index,
                bound,
                context,
            } => Self::IndexOutOfBounds {
                index,
                bound,
                context,
            },
            TdsError::InconsistentDataStructure { message } => {
                Self::InconsistentDataStructure { message }
            }
            TdsError::Geometric(source) => Self::Geometric { source },
            TdsError::FacetError(source) => Self::Facet { source },
            TdsError::DuplicateCoordinatesInSimplex {
                simplex_id,
                message,
            } => Self::DuplicateCoordinatesInSimplex {
                simplex_id,
                message,
            },
        }
    }
}

/// Compact, typed summary of a [`TdsConstructionError`].
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum TdsConstructionFailure {
    /// TDS validation failed during construction.
    #[error("TDS validation failed during construction: {reason}")]
    Validation {
        /// Structured validation failure.
        #[source]
        reason: TdsValidationFailure,
    },

    /// A duplicate UUID was inserted.
    #[error("duplicate UUID during construction: {entity:?} {uuid}")]
    DuplicateUuid {
        /// Entity kind.
        entity: EntityKind,
        /// Duplicated UUID.
        uuid: uuid::Uuid,
    },
}

impl From<TdsConstructionError> for TdsConstructionFailure {
    fn from(source: TdsConstructionError) -> Self {
        match source {
            TdsConstructionError::ValidationError(source) => Self::Validation {
                reason: source.into(),
            },
            TdsConstructionError::DuplicateUuid { entity, uuid } => {
                Self::DuplicateUuid { entity, uuid }
            }
        }
    }
}

/// Typed insertion-stage failure that should not occur while bootstrapping an initial simplex.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum InitialSimplexUnexpectedInsertionStage {
    /// Cavity filling escaped initial-simplex construction.
    #[error("cavity filling failed during insertion: {source}")]
    CavityFilling {
        /// Underlying cavity filling error.
        #[source]
        source: Box<CavityFillingError>,
    },

    /// Conflict-region extraction escaped initial-simplex construction.
    #[error("conflict region failed during insertion: {source}")]
    ConflictRegion {
        /// Underlying conflict-region error.
        #[source]
        source: ConflictError,
    },

    /// Point location escaped initial-simplex construction.
    #[error("point location failed during insertion: {source}")]
    Location {
        /// Underlying point-location error.
        #[source]
        source: LocateError,
    },

    /// Insertion detected non-manifold topology before incremental insertion began.
    #[error("facet {facet_hash:#x} shared by {simplex_count} simplices during initial simplex")]
    NonManifoldTopology {
        /// Hash of the over-shared facet.
        facet_hash: u64,
        /// Number of incident simplices sharing the facet.
        simplex_count: usize,
    },

    /// Hull extension escaped initial-simplex construction.
    #[error("hull extension failed during insertion: {reason}")]
    HullExtension {
        /// Structured hull-extension failure reason.
        reason: HullExtensionReason,
    },

    /// Delaunay validation escaped initial-simplex construction.
    #[error("Delaunay validation failed during insertion: {source}")]
    DelaunayValidation {
        /// Underlying Delaunay validation error.
        #[source]
        source: DelaunayTriangulationValidationError,
    },

    /// Topology validation escaped initial-simplex construction.
    #[error("topology validation failed during insertion: {source}")]
    TopologyValidation {
        /// Underlying topology validation error.
        #[source]
        source: Box<TriangulationValidationError>,
    },

    /// Final topology validation escaped initial-simplex construction.
    #[error("final topology validation failed after construction: {source}")]
    FinalTopologyValidation {
        /// Underlying final topology validation summary.
        #[source]
        source: InvariantErrorSummary,
    },

    /// Spatial index construction escaped initial-simplex construction.
    #[error("spatial index construction failed during insertion: {reason}")]
    SpatialIndexConstruction {
        /// Structured spatial-index construction failure.
        #[source]
        reason: SpatialIndexConstructionFailure,
    },
}

/// Structured reason why initial-simplex construction failed during insertion.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum InitialSimplexConstructionError {
    /// TDS validation failed while assembling the bootstrap simplex.
    #[error("TDS validation failed while building initial simplex: {source}")]
    TdsValidation {
        /// Underlying TDS validation error.
        #[source]
        source: TdsValidationFailure,
    },

    /// The bootstrap simplex attempted to insert a duplicate UUID.
    #[error("duplicate UUID while building initial simplex: {entity:?} {uuid}")]
    DuplicateUuid {
        /// Duplicated entity kind.
        entity: EntityKind,
        /// Duplicated UUID.
        uuid: uuid::Uuid,
    },

    /// Bootstrap simplex creation failed.
    #[error("failed to create bootstrap simplex: {message}")]
    FailedToCreateSimplex {
        /// Simplex creation failure detail.
        message: String,
    },

    /// Not enough vertices were available for the bootstrap simplex.
    #[error("insufficient vertices for {dimension}D initial simplex: {source}")]
    InsufficientVertices {
        /// Attempted dimension.
        dimension: usize,
        /// Underlying simplex validation error.
        #[source]
        source: SimplexValidationError,
    },

    /// Geometric degeneracy prevented bootstrap construction.
    #[error("geometric degeneracy while building initial simplex: {message}")]
    GeometricDegeneracy {
        /// Degeneracy detail.
        message: String,
    },

    /// Internal construction invariant failed.
    #[error("internal inconsistency while building initial simplex: {message}")]
    InternalInconsistency {
        /// Internal inconsistency detail.
        message: String,
    },

    /// Duplicate coordinates were detected in the bootstrap simplex.
    #[error("duplicate coordinates while building initial simplex: {coordinates}")]
    DuplicateCoordinates {
        /// Duplicate coordinate tuple stored as typed coordinate payloads.
        coordinates: CoordinateValues,
    },

    /// Local repair would remove more simplices than the active budget allowed.
    #[error(
        "local repair removal budget exceeded while building initial simplex: attempted {attempted}, max {max_simplices_removed}"
    )]
    LocalRepairBudgetExceeded {
        /// Maximum number of simplices this repair stage was allowed to remove.
        max_simplices_removed: usize,
        /// Number of simplices the repair stage attempted to remove.
        attempted: usize,
    },

    /// Periodic quotient construction is not release-validated for this dimension.
    #[error(
        "periodic image-point construction is release-validated only up to {max_validated_dimension}D; {dimension}D scalable quotient construction is tracked by issue #{tracking_issue}"
    )]
    UnsupportedPeriodicDimension {
        /// Requested triangulation dimension.
        dimension: usize,
        /// Highest dimension with release-validated periodic quotient construction.
        max_validated_dimension: usize,
        /// Tracking issue for extending periodic quotient support.
        tracking_issue: u32,
    },

    /// An insertion-stage-only construction error escaped initial-simplex construction.
    #[error(
        "unexpected insertion-stage construction error while building initial simplex: {reason}"
    )]
    UnexpectedInsertionStage {
        /// Structured insertion-stage failure that escaped bootstrap construction.
        #[source]
        reason: Box<InitialSimplexUnexpectedInsertionStage>,
    },
}

impl From<TdsConstructionError> for InitialSimplexConstructionError {
    fn from(source: TdsConstructionError) -> Self {
        match source {
            TdsConstructionError::ValidationError(source) => Self::TdsValidation {
                source: source.into(),
            },
            TdsConstructionError::DuplicateUuid { entity, uuid } => {
                Self::DuplicateUuid { entity, uuid }
            }
        }
    }
}

impl From<TriangulationConstructionError> for InitialSimplexConstructionError {
    fn from(source: TriangulationConstructionError) -> Self {
        match source {
            TriangulationConstructionError::Tds(source) => source.into(),
            TriangulationConstructionError::FailedToCreateSimplex { message } => {
                Self::FailedToCreateSimplex { message }
            }
            TriangulationConstructionError::InsertionCavityFilling { source } => {
                Self::UnexpectedInsertionStage {
                    reason: Box::new(InitialSimplexUnexpectedInsertionStage::CavityFilling {
                        source: Box::new(source),
                    }),
                }
            }
            TriangulationConstructionError::InsufficientVertices { dimension, source } => {
                Self::InsufficientVertices { dimension, source }
            }
            TriangulationConstructionError::GeometricDegeneracy { message } => {
                Self::GeometricDegeneracy { message }
            }
            TriangulationConstructionError::UnsupportedPeriodicDimension {
                dimension,
                max_validated_dimension,
                tracking_issue,
            } => Self::UnsupportedPeriodicDimension {
                dimension,
                max_validated_dimension,
                tracking_issue,
            },
            TriangulationConstructionError::InternalInconsistency { message } => {
                Self::InternalInconsistency { message }
            }
            TriangulationConstructionError::DuplicateCoordinates { coordinates } => {
                Self::DuplicateCoordinates { coordinates }
            }
            TriangulationConstructionError::SpatialIndexConstruction { reason } => {
                Self::UnexpectedInsertionStage {
                    reason: Box::new(
                        InitialSimplexUnexpectedInsertionStage::SpatialIndexConstruction { reason },
                    ),
                }
            }
            TriangulationConstructionError::InsertionConflictRegion { source } => {
                Self::UnexpectedInsertionStage {
                    reason: Box::new(InitialSimplexUnexpectedInsertionStage::ConflictRegion {
                        source,
                    }),
                }
            }
            TriangulationConstructionError::InsertionLocation { source } => {
                Self::UnexpectedInsertionStage {
                    reason: Box::new(InitialSimplexUnexpectedInsertionStage::Location { source }),
                }
            }
            TriangulationConstructionError::InsertionNonManifoldTopology {
                facet_hash,
                simplex_count,
            } => Self::UnexpectedInsertionStage {
                reason: Box::new(
                    InitialSimplexUnexpectedInsertionStage::NonManifoldTopology {
                        facet_hash,
                        simplex_count,
                    },
                ),
            },
            TriangulationConstructionError::InsertionHullExtension { reason } => {
                Self::UnexpectedInsertionStage {
                    reason: Box::new(InitialSimplexUnexpectedInsertionStage::HullExtension {
                        reason,
                    }),
                }
            }
            TriangulationConstructionError::InsertionDelaunayValidation { source } => {
                Self::UnexpectedInsertionStage {
                    reason: Box::new(InitialSimplexUnexpectedInsertionStage::DelaunayValidation {
                        source,
                    }),
                }
            }
            TriangulationConstructionError::InsertionTopologyValidation { source, .. } => {
                Self::UnexpectedInsertionStage {
                    reason: Box::new(InitialSimplexUnexpectedInsertionStage::TopologyValidation {
                        source: Box::new(source),
                    }),
                }
            }
            TriangulationConstructionError::LocalRepairBudgetExceeded {
                max_simplices_removed,
                attempted,
            } => Self::LocalRepairBudgetExceeded {
                max_simplices_removed,
                attempted,
            },
            TriangulationConstructionError::FinalTopologyValidation { source, .. } => {
                Self::UnexpectedInsertionStage {
                    reason: Box::new(
                        InitialSimplexUnexpectedInsertionStage::FinalTopologyValidation { source },
                    ),
                }
            }
        }
    }
}

/// Flip-repair failure category used by compact error summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairErrorKind {
    /// Repair exhausted its flip budget.
    NonConvergent,
    /// Repair completed but left a Delaunay violation.
    PostconditionFailed,
    /// Repair verification failed.
    VerificationFailed,
    /// Orientation canonicalization failed after repair.
    OrientationCanonicalizationFailed,
    /// Repair was not admissible under the topology guarantee.
    InvalidTopology,
    /// Heuristic topology rebuild failed during advanced repair.
    HeuristicRebuildFailed,
    /// A lower-level flip operation failed.
    Flip,
}

impl From<&DelaunayRepairError> for DelaunayRepairErrorKind {
    fn from(source: &DelaunayRepairError) -> Self {
        match source {
            DelaunayRepairError::NonConvergent { .. } => Self::NonConvergent,
            DelaunayRepairError::PostconditionFailed { .. } => Self::PostconditionFailed,
            DelaunayRepairError::VerificationFailed { .. } => Self::VerificationFailed,
            DelaunayRepairError::OrientationCanonicalizationFailed { .. } => {
                Self::OrientationCanonicalizationFailed
            }
            DelaunayRepairError::InvalidTopology { .. } => Self::InvalidTopology,
            DelaunayRepairError::HeuristicRebuildFailed { .. } => Self::HeuristicRebuildFailed,
            DelaunayRepairError::Flip { .. } => Self::Flip,
        }
    }
}

/// Compact summary of a [`DelaunayRepairError`] for small by-value error payloads.
///
/// The conversion preserves the top-level [`DelaunayRepairErrorKind`] and the
/// rendered diagnostic text. It intentionally drops bulky typed payloads, source
/// chains, and repair diagnostics; keep the original [`DelaunayRepairError`]
/// when callers need to inspect that data.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::{
///     DelaunayRepairError, DelaunayRepairErrorKind, DelaunayRepairErrorSummary,
/// };
///
/// let source = DelaunayRepairError::PostconditionFailed {
///     message: "remaining non-Delaunay facet".to_string(),
/// };
/// let summary = DelaunayRepairErrorSummary::from(&source);
///
/// assert_eq!(summary.kind, DelaunayRepairErrorKind::PostconditionFailed);
/// assert!(summary.message.contains("remaining non-Delaunay facet"));
/// ```
#[must_use]
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct DelaunayRepairErrorSummary {
    /// Structured repair failure category.
    pub kind: DelaunayRepairErrorKind,
    /// Full diagnostic text from the original repair error.
    pub message: String,
}

impl PartialEq for DelaunayRepairErrorSummary {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Eq for DelaunayRepairErrorSummary {}

impl From<&DelaunayRepairError> for DelaunayRepairErrorSummary {
    fn from(source: &DelaunayRepairError) -> Self {
        Self {
            kind: source.into(),
            message: source.to_string(),
        }
    }
}

impl From<DelaunayRepairError> for DelaunayRepairErrorSummary {
    fn from(source: DelaunayRepairError) -> Self {
        Self::from(&source)
    }
}

/// Insertion failure category used by compact error summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertionErrorKind {
    /// Conflict-region search failed.
    ConflictRegion,
    /// Point location failed.
    Location,
    /// Cavity filling failed.
    CavityFilling,
    /// Neighbor wiring failed.
    NeighborWiring,
    /// Non-manifold topology was detected.
    NonManifoldTopology,
    /// Hull extension failed.
    HullExtension,
    /// Delaunay validation failed after insertion.
    DelaunayValidationFailed,
    /// Flip-based Delaunay repair failed.
    DelaunayRepairFailed,
    /// Duplicate coordinates were supplied.
    DuplicateCoordinates,
    /// Duplicate UUID was supplied.
    DuplicateUuid,
    /// TDS topology validation failed.
    TopologyValidation,
    /// Triangulation-layer topology validation failed.
    TopologyValidationFailed,
    /// Local repair would exceed its simplex-removal budget.
    MaxSimplicesRemovedExceeded,
    /// Spatial index construction failed.
    SpatialIndexConstruction,
    /// A perturbation retry produced invalid coordinates.
    PerturbedCoordinateInvalid,
}

/// Nested discriminant preserved by an [`InsertionErrorSummary`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertionErrorSourceKind {
    /// TDS-layer topology validation failed.
    Tds(TdsErrorKind),
    /// Triangulation-layer topology validation failed.
    Triangulation(TriangulationValidationErrorKind),
    /// Level 4 Delaunay validation failed.
    Delaunay(DelaunayValidationErrorKind),
    /// Flip repair failed.
    DelaunayRepair(DelaunayRepairErrorKind),
}

/// Compact summary of an [`InsertionError`] for small by-value error payloads.
///
/// The conversion preserves the top-level [`InsertionErrorKind`], an optional
/// nested [`InsertionErrorSourceKind`] for wrapped validation or repair errors,
/// the original retryability decision, and the rendered diagnostic text. It
/// intentionally drops bulky typed payloads and source chains; keep the original
/// [`InsertionError`] when callers need the full structured context.
///
/// Equality compares the structured kind, nested source kind, and retryability
/// flag while ignoring [`Self::message`], so summary comparisons remain stable
/// across display-text changes.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::insertion::{
///     DelaunayRepairErrorKind, DelaunayRepairFailureContext, HullExtensionReason,
///     InsertionError, InsertionErrorKind, InsertionErrorSourceKind,
///     InsertionErrorSummary,
/// };
/// use delaunay::prelude::repair::DelaunayRepairError;
///
/// let source = InsertionError::DelaunayRepairFailed {
///     source: Box::new(DelaunayRepairError::PostconditionFailed {
///         message: "remaining non-Delaunay facet".to_string(),
///     }),
///     context: DelaunayRepairFailureContext::LocalRepair,
/// };
/// let summary = InsertionErrorSummary::from(source);
///
/// assert_eq!(summary.kind, InsertionErrorKind::DelaunayRepairFailed);
/// assert_eq!(
///     summary.source_kind,
///     Some(InsertionErrorSourceKind::DelaunayRepair(
///         DelaunayRepairErrorKind::PostconditionFailed,
///     )),
/// );
/// assert!(!summary.retryable);
///
/// let retryable = InsertionErrorSummary::from(InsertionError::HullExtension {
///     reason: HullExtensionReason::NoVisibleFacets,
/// });
/// assert!(retryable.retryable);
/// ```
#[must_use]
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct InsertionErrorSummary {
    /// Structured insertion failure category.
    pub kind: InsertionErrorKind,
    /// Nested structured source kind when the insertion error wraps another layer.
    pub source_kind: Option<InsertionErrorSourceKind>,
    /// Whether the original insertion error was retryable via perturbation.
    pub retryable: bool,
    /// Full diagnostic text from the original insertion error.
    pub message: String,
}

impl PartialEq for InsertionErrorSummary {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.source_kind == other.source_kind
            && self.retryable == other.retryable
    }
}

impl Eq for InsertionErrorSummary {}

impl InsertionErrorSummary {
    /// Returns true when the original insertion failure was retryable via perturbation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::tds::TriangulationValidationErrorKind;
    /// use delaunay::prelude::insertion::{
    ///     InsertionErrorKind, InsertionErrorSourceKind, InsertionErrorSummary,
    /// };
    ///
    /// let retryable = InsertionErrorSummary {
    ///     kind: InsertionErrorKind::TopologyValidationFailed,
    ///     source_kind: Some(InsertionErrorSourceKind::Triangulation(
    ///         TriangulationValidationErrorKind::ManifoldFacetMultiplicity,
    ///     )),
    ///     retryable: true,
    ///     message: "facet shared by too many simplices".to_string(),
    /// };
    /// assert!(retryable.is_retryable());
    ///
    /// let structural = InsertionErrorSummary {
    ///     kind: InsertionErrorKind::TopologyValidationFailed,
    ///     source_kind: Some(InsertionErrorSourceKind::Triangulation(
    ///         TriangulationValidationErrorKind::Disconnected,
    ///     )),
    ///     retryable: false,
    ///     message: "simplex graph disconnected".to_string(),
    /// };
    /// assert!(!structural.is_retryable());
    /// ```
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        self.retryable
    }
}

impl From<InsertionError> for InsertionErrorSummary {
    fn from(source: InsertionError) -> Self {
        let retryable = source.is_retryable();
        let kind = match &source {
            InsertionError::ConflictRegion(_) => InsertionErrorKind::ConflictRegion,
            InsertionError::Location(_) => InsertionErrorKind::Location,
            InsertionError::CavityFilling { .. } => InsertionErrorKind::CavityFilling,
            InsertionError::NeighborWiring { .. } => InsertionErrorKind::NeighborWiring,
            InsertionError::NonManifoldTopology { .. } => InsertionErrorKind::NonManifoldTopology,
            InsertionError::HullExtension { .. } => InsertionErrorKind::HullExtension,
            InsertionError::DelaunayValidationFailed { .. } => {
                InsertionErrorKind::DelaunayValidationFailed
            }
            InsertionError::DelaunayRepairFailed { .. } => InsertionErrorKind::DelaunayRepairFailed,
            InsertionError::DuplicateCoordinates { .. } => InsertionErrorKind::DuplicateCoordinates,
            InsertionError::DuplicateUuid { .. } => InsertionErrorKind::DuplicateUuid,
            InsertionError::TopologyValidation(_) => InsertionErrorKind::TopologyValidation,
            InsertionError::TopologyValidationFailed { .. } => {
                InsertionErrorKind::TopologyValidationFailed
            }
            InsertionError::MaxSimplicesRemovedExceeded { .. } => {
                InsertionErrorKind::MaxSimplicesRemovedExceeded
            }
            InsertionError::SpatialIndexConstruction { .. } => {
                InsertionErrorKind::SpatialIndexConstruction
            }
            InsertionError::PerturbedCoordinateInvalid { .. } => {
                InsertionErrorKind::PerturbedCoordinateInvalid
            }
        };
        let source_kind = match &source {
            InsertionError::DelaunayValidationFailed { source } => {
                Some(InsertionErrorSourceKind::Delaunay(source.into()))
            }
            InsertionError::DelaunayRepairFailed { source, .. } => Some(
                InsertionErrorSourceKind::DelaunayRepair(source.as_ref().into()),
            ),
            InsertionError::TopologyValidation(source) => {
                Some(InsertionErrorSourceKind::Tds(source.into()))
            }
            InsertionError::TopologyValidationFailed { source, .. } => {
                Some(InsertionErrorSourceKind::Triangulation(source.into()))
            }
            _ => None,
        };
        Self {
            kind,
            source_kind,
            retryable,
            message: source.to_string(),
        }
    }
}

/// Insertion-stage context for flip-based Delaunay repair failures.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairFailureContext {
    /// Local repair failed and the robust-kernel fallback also failed.
    LocalRepairRobustFallback {
        /// Original local repair failure that triggered the robust fallback.
        initial: DelaunayRepairErrorSummary,
    },
    /// Local repair failed with a non-recoverable repair error.
    LocalRepairNonRecoverable,
    /// Repair failed while canonicalizing orientation after insertion.
    OrientationCanonicalization,
    /// General local repair path.
    LocalRepair,
    /// Post-insertion repair path.
    PostInsertionRepair,
}

impl fmt::Display for DelaunayRepairFailureContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalRepairRobustFallback { initial } => {
                f.write_str("local repair failed (")?;
                initial.fmt(f)?;
                f.write_str("); robust fallback also failed")
            }
            Self::LocalRepairNonRecoverable => f.write_str("local repair failed (non-recoverable)"),
            Self::OrientationCanonicalization => f.write_str("orientation canonicalization"),
            Self::LocalRepair => f.write_str("local repair"),
            Self::PostInsertionRepair => f.write_str("post-insertion repair"),
        }
    }
}

/// Structured reason why neighbor repair failed after cavity repair.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::SimplexKey;
/// use delaunay::prelude::insertion::{
///     NeighborRebuildError, NeighborWiringError,
/// };
/// use slotmap::KeyData;
///
/// let simplex_key = SimplexKey::from(KeyData::from_ffi(11));
/// let err = NeighborRebuildError::Wiring {
///     reason: NeighborWiringError::MissingSimplex { simplex_key },
/// };
/// std::assert_matches!(err, NeighborRebuildError::Wiring { .. });
/// ```
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum NeighborRebuildError {
    /// Neighbor wiring failed while repairing neighbor pointers.
    #[error("neighbor wiring failed while repairing neighbors: {reason}")]
    Wiring {
        /// Structured wiring failure.
        #[source]
        reason: NeighborWiringError,
    },

    /// The repaired facet incidence is non-manifold.
    #[error(
        "non-manifold topology while repairing neighbors: facet {facet_hash:#x} shared by {simplex_count} simplices"
    )]
    NonManifoldTopology {
        /// Hash of the over-shared facet.
        facet_hash: u64,
        /// Number of incident simplices.
        simplex_count: usize,
    },

    /// TDS validation failed while repairing neighbor pointers.
    #[error("TDS validation failed while repairing neighbors: {reason}")]
    TopologyValidation {
        /// Underlying TDS validation error.
        #[source]
        reason: TdsValidationFailure,
    },

    /// Another insertion error escaped the neighbor repair helper.
    #[error("unexpected neighbor repair error: {source}")]
    Unexpected {
        /// Insertion-layer error that did not map to a neighbor repair category.
        #[source]
        source: InsertionErrorSummary,
    },
}

/// Structured reason why cavity filling failed.
///
/// The high-level [`InsertionError::CavityFilling`] bucket identifies the
/// insertion stage; this type carries the recoverable, pattern-matchable reason
/// within that stage.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::SimplexKey;
/// use delaunay::prelude::insertion::CavityFillingError;
/// use slotmap::KeyData;
///
/// let simplex_key = SimplexKey::from(KeyData::from_ffi(7));
/// let err = CavityFillingError::InvalidFacetIndex {
///     simplex_key,
///     facet_index: 4,
///     vertex_count: 3,
/// };
/// std::assert_matches!(err, CavityFillingError::InvalidFacetIndex { .. });
/// ```
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum CavityFillingError {
    /// A boundary facet references a simplex that is no longer present.
    #[error("boundary facet simplex {simplex_key:?} not found")]
    MissingBoundarySimplex {
        /// Missing boundary simplex key.
        simplex_key: SimplexKey,
    },

    /// The vertex being inserted is not present in the TDS.
    #[error("inserted vertex {vertex_key:?} not found in TDS")]
    MissingInsertedVertex {
        /// Missing inserted vertex key.
        vertex_key: VertexKey,
    },

    /// A boundary simplex has the wrong number of vertices for the dimension.
    #[error("boundary simplex {simplex_key:?} has {actual} vertices, expected {expected}")]
    WrongSimplexArity {
        /// Simplex with the wrong arity.
        simplex_key: SimplexKey,
        /// Observed vertex count.
        actual: usize,
        /// Expected vertex count.
        expected: usize,
    },

    /// A facet index is outside the referenced simplex's vertex range.
    #[error(
        "facet index {facet_index} out of range for boundary simplex {simplex_key:?} with {vertex_count} vertices"
    )]
    InvalidFacetIndex {
        /// Simplex referenced by the facet handle.
        simplex_key: SimplexKey,
        /// Invalid facet index.
        facet_index: usize,
        /// Number of vertices in the referenced simplex.
        vertex_count: usize,
    },

    /// Creating a replacement simplex failed validation.
    #[error("failed to create replacement simplex: {source}")]
    SimplexCreation {
        /// Underlying simplex validation error.
        #[from]
        source: SimplexValidationError,
    },

    /// Inserting a replacement simplex into the TDS failed.
    #[error("failed to insert replacement simplex: {reason}")]
    SimplexInsertion {
        /// Underlying TDS construction error.
        #[source]
        reason: TdsConstructionFailure,
    },

    /// Initial simplex bootstrap failed.
    #[error("failed to build initial simplex: {reason}")]
    InitialSimplexConstruction {
        /// Underlying triangulation construction error.
        #[source]
        reason: InitialSimplexConstructionError,
    },

    /// A rebuilt TDS did not preserve the just-inserted vertex UUID.
    #[error("inserted vertex with UUID {uuid} not found in rebuilt TDS")]
    RebuiltVertexMissing {
        /// UUID that should have been present after rebuilding.
        uuid: uuid::Uuid,
    },

    /// The conflict region was empty and no fallback simplex was available.
    #[error(
        "empty conflict region for exterior insertion (fallback simplex: {fallback_simplex:?})"
    )]
    EmptyConflictRegion {
        /// Optional fallback simplex that was available to split.
        fallback_simplex: Option<SimplexKey>,
    },

    /// The extracted cavity boundary was empty and no fallback simplex was available.
    #[error(
        "empty cavity boundary for exterior insertion (fallback simplex: {fallback_simplex:?})"
    )]
    EmptyBoundary {
        /// Optional fallback simplex that was available to split.
        fallback_simplex: Option<SimplexKey>,
    },

    /// Facet sharing remained invalid after local insertion repair.
    #[error("facet sharing invalid after insertion repairs during {stage}")]
    InvalidFacetSharingAfterRepair {
        /// Repair stage that observed invalid facet sharing.
        stage: CavityRepairStage,
    },

    /// Repairing neighbor pointers after cavity repair failed.
    #[error("failed to repair neighbors after insertion repairs: {reason}")]
    NeighborRebuild {
        /// Underlying insertion-layer error from the neighbor rebuild.
        #[source]
        reason: NeighborRebuildError,
    },

    /// Bootstrap perturbation scale could not be represented in the scalar type.
    #[error("failed to convert perturbation scale {value} into scalar type")]
    PerturbationScaleConversion {
        /// Requested perturbation scale stored as a typed coordinate-conversion payload.
        value: CoordinateConversionValue,
    },

    /// The locate result lies on a lower-dimensional feature that insertion does not support yet.
    #[error("unsupported degenerate insertion location: {location:?}")]
    UnsupportedDegenerateLocation {
        /// Degenerate location returned by point location.
        location: LocateResult,
    },

    /// Fan filling produced no replacement simplices.
    #[error("fan triangulation produced no simplices")]
    EmptyFanTriangulation,
}

/// Stage where cavity repair detected invalid facet sharing.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::insertion::CavityRepairStage;
///
/// assert_eq!(
///     CavityRepairStage::PrimaryInsertion.to_string(),
///     "primary insertion"
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CavityRepairStage {
    /// Primary cavity insertion path.
    PrimaryInsertion,
    /// Fan triangulation fallback path.
    FanTriangulation,
}

impl fmt::Display for CavityRepairStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrimaryInsertion => f.write_str("primary insertion"),
            Self::FanTriangulation => f.write_str("fan triangulation"),
        }
    }
}

/// Structured reason why neighbor wiring failed.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::SimplexKey;
/// use delaunay::prelude::insertion::NeighborWiringError;
/// use slotmap::KeyData;
///
/// let simplex_key = SimplexKey::from(KeyData::from_ffi(11));
/// let err = NeighborWiringError::MissingSimplex { simplex_key };
/// std::assert_matches!(err, NeighborWiringError::MissingSimplex { .. });
/// ```
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum NeighborWiringError {
    /// A simplex required for neighbor wiring was not found.
    #[error("simplex {simplex_key:?} not found during neighbor wiring")]
    MissingSimplex {
        /// Missing simplex key.
        simplex_key: SimplexKey,
    },

    /// A facet index is outside the referenced simplex's vertex range.
    #[error(
        "facet index {facet_index} out of range for simplex {simplex_key:?} with {vertex_count} vertices"
    )]
    InvalidFacetIndex {
        /// Referenced simplex.
        simplex_key: SimplexKey,
        /// Invalid facet index.
        facet_index: usize,
        /// Number of vertices in the referenced simplex.
        vertex_count: usize,
    },

    /// A facet index cannot fit in the compact facet-index storage type.
    #[error("facet index {facet_index} exceeds compact facet-index maximum {max}")]
    FacetIndexOverflow {
        /// Facet index that overflowed.
        facet_index: usize,
        /// Maximum storable facet index.
        max: u8,
    },

    /// A simplex has the wrong number of vertices for the triangulation dimension.
    #[error(
        "simplex {simplex_key:?} has {found} vertices; expected {expected} for neighbor wiring"
    )]
    WrongSimplexArity {
        /// Simplex with mismatched arity.
        simplex_key: SimplexKey,
        /// Expected number of vertices.
        expected: usize,
        /// Observed number of vertices.
        found: usize,
    },

    /// An external facet did not match any new-simplex boundary facet.
    #[error(
        "external facet {facet_index} on simplex {simplex_key:?} with hash {facet_hash:#x} did not match any new-simplex facet"
    )]
    ExternalFacetNotFound {
        /// External simplex being glued.
        simplex_key: SimplexKey,
        /// Facet index on the external simplex.
        facet_index: u8,
        /// Hash of the external facet vertex set.
        facet_hash: u64,
    },

    /// An external facet matched a facet already shared by multiple new simplices.
    #[error(
        "external facet {facet_index} on simplex {simplex_key:?} with hash {facet_hash:#x} matched {existing_incidents} new-simplex incidents"
    )]
    ExternalFacetAlreadyShared {
        /// External simplex being glued.
        simplex_key: SimplexKey,
        /// Facet index on the external simplex.
        facet_index: u8,
        /// Hash of the external facet vertex set.
        facet_hash: u64,
        /// Number of existing new-simplex incidents for the facet.
        existing_incidents: usize,
    },

    /// A simplex points to itself as a neighbor.
    #[error("simplex {simplex_key:?} has a self-neighbor pointer")]
    SelfNeighbor {
        /// Simplex containing the self-neighbor pointer.
        simplex_key: SimplexKey,
    },

    /// A neighbor pointer references a missing simplex.
    #[error("simplex {simplex_key:?} has neighbor pointer to missing simplex {neighbor_key:?}")]
    MissingNeighborTarget {
        /// Simplex containing the stale neighbor pointer.
        simplex_key: SimplexKey,
        /// Missing neighbor key.
        neighbor_key: SimplexKey,
    },

    /// A neighbor pointer does not point back through the matching mirror facet.
    #[error(
        "simplex {simplex_key:?} facet {facet_index} points to {neighbor_key:?}, \
         but the mirror facet {mirror_facet_index:?} points back to {neighbor_back:?}"
    )]
    AsymmetricNeighbor {
        /// Simplex containing the neighbor pointer.
        simplex_key: SimplexKey,
        /// Facet index in `simplex_key`.
        facet_index: usize,
        /// Neighbor referenced by `simplex_key`.
        neighbor_key: SimplexKey,
        /// Matching facet index in `neighbor_key`, if one could be found.
        mirror_facet_index: Option<usize>,
        /// Neighbor's pointer back through the mirror facet.
        neighbor_back: Option<SimplexKey>,
    },

    /// Neighbor traversal discovered more simplices than the TDS contains.
    #[error("neighbor walk visited {visited} unique simplices but triangulation contains {total}")]
    NeighborWalkExceededSimplexCount {
        /// Number of unique simplices visited.
        visited: usize,
        /// Number of simplices in the TDS.
        total: usize,
    },
}

fn ensure_neighbor_wiring_simplex_arity<const D: usize>(
    simplex_key: SimplexKey,
    vertex_count: usize,
) -> Result<(), NeighborWiringError> {
    if vertex_count == D + 1 {
        return Ok(());
    }

    Err(NeighborWiringError::WrongSimplexArity {
        simplex_key,
        expected: D + 1,
        found: vertex_count,
    })
}

/// Typed reason a spatial insertion index could not be constructed.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     CoordinateConversionValue, FiniteCoordinateValue, InvalidCoordinateValue,
/// };
/// use delaunay::prelude::insertion::SpatialIndexConstructionFailure;
///
/// let non_finite = SpatialIndexConstructionFailure::NonFiniteCellSize {
///     value: InvalidCoordinateValue::Nan,
/// };
/// std::assert_matches!(
///     non_finite,
///     SpatialIndexConstructionFailure::NonFiniteCellSize { .. }
/// );
///
/// let finite = FiniteCoordinateValue::try_new(0.0)?;
/// let non_positive = SpatialIndexConstructionFailure::NonPositiveCellSize {
///     value: CoordinateConversionValue::Scalar(finite),
/// };
/// std::assert_matches!(
///     non_positive,
///     SpatialIndexConstructionFailure::NonPositiveCellSize { .. }
/// );
/// # Ok::<(), InvalidCoordinateValue>(())
/// ```
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum SpatialIndexConstructionFailure {
    /// Hash-grid cell size was non-finite.
    #[error("hash-grid cell size is non-finite: {value}")]
    NonFiniteCellSize {
        /// Non-finite cell-size category.
        value: InvalidCoordinateValue,
    },

    /// Hash-grid cell size was finite but non-positive.
    #[error("hash-grid cell size must be positive, got {value}")]
    NonPositiveCellSize {
        /// Rejected cell-size value.
        value: CoordinateConversionValue,
    },
}

impl From<HashGridIndexError> for SpatialIndexConstructionFailure {
    fn from(source: HashGridIndexError) -> Self {
        match source {
            HashGridIndexError::NonFiniteCellSize { value } => Self::NonFiniteCellSize { value },
            HashGridIndexError::NonPositiveCellSize { value } => Self::NonPositiveCellSize {
                value: CoordinateConversionValue::from_f64(value),
            },
        }
    }
}

/// Error during incremental insertion.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::insertion::InsertionError;
///
/// use delaunay::prelude::geometry::CoordinateValues;
///
/// let err = InsertionError::DuplicateCoordinates {
///     coordinates: CoordinateValues::from([0.0, 0.0, 0.0]),
/// };
/// std::assert_matches!(err, InsertionError::DuplicateCoordinates { .. });
/// ```
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum InsertionError {
    /// Conflict region finding failed
    #[error("Conflict region error: {0}")]
    ConflictRegion(#[from] ConflictError),

    /// Point location failed
    #[error("Location error: {0}")]
    Location(#[from] LocateError),

    /// Cavity filling failed.
    #[error("Cavity filling failed: {reason}")]
    CavityFilling {
        /// Structured reason for the cavity-filling failure.
        #[source]
        reason: CavityFillingError,
    },

    /// Neighbor wiring failed.
    #[error("Neighbor wiring failed: {reason}")]
    NeighborWiring {
        /// Structured reason for the neighbor-wiring failure.
        #[source]
        reason: NeighborWiringError,
    },

    /// Non-manifold topology detected during neighbor wiring.
    ///
    /// This occurs when a facet is shared by more than 2 simplices, violating
    /// the manifold property. This is typically caused by geometric degeneracy
    /// and can often be resolved via coordinate perturbation.
    #[error(
        "Non-manifold topology: facet {facet_hash:#x} shared by {simplex_count} simplices (expected ≤2)"
    )]
    NonManifoldTopology {
        /// Hash of the facet vertices
        facet_hash: u64,
        /// Number of simplices sharing this facet
        simplex_count: usize,
    },

    /// Hull extension failed (finding visible boundary facets).
    #[error("Hull extension failed: {reason}")]
    HullExtension {
        /// Structured reason for failure.
        reason: HullExtensionReason,
    },

    /// Global Delaunay validation failed after insertion.
    ///
    /// This indicates the triangulation is structurally valid but violates the
    /// empty-circumsphere property (Level 4).
    #[error("Delaunay validation failed: {source}")]
    DelaunayValidationFailed {
        /// The structured Level 4 validation error.
        #[source]
        source: DelaunayTriangulationValidationError,
    },

    /// Flip-based Delaunay repair failed.
    ///
    /// This variant is used when a Delaunay repair pass (local or fallback)
    /// cannot converge or otherwise fails. It preserves the structured
    /// [`DelaunayRepairError`]
    /// rather than collapsing it into a string.
    #[error("Delaunay repair failed ({context}): {source}")]
    DelaunayRepairFailed {
        /// The underlying repair error.
        #[source]
        source: Box<DelaunayRepairError>,
        /// Operational context describing the repair path that failed.
        context: DelaunayRepairFailureContext,
    },

    /// Attempted to insert a vertex with coordinates that already exist.
    #[error(
        "Duplicate coordinates: vertex with coordinates {coordinates} already exists in the triangulation"
    )]
    DuplicateCoordinates {
        /// Duplicate coordinate tuple stored as typed coordinate payloads.
        coordinates: CoordinateValues,
    },

    /// Attempted to insert an entity with a UUID that already exists.
    #[error("Duplicate UUID: {entity:?} with UUID {uuid} already exists")]
    DuplicateUuid {
        /// The type of entity.
        entity: EntityKind,
        /// The UUID that was duplicated.
        uuid: uuid::Uuid,
    },

    /// Topology validation or repair failed.
    #[error("Topology validation error: {0}")]
    TopologyValidation(#[from] TdsError),

    /// Level 3 topology validation failed (Triangulation layer).
    ///
    /// This preserves the structured [`TriangulationValidationError`] without wrapping it into a
    /// [`TdsError`],
    /// avoiding lower-layer (`Tds`) errors depending on higher-layer (`Triangulation`) errors.
    #[error("{message}: {source}")]
    TopologyValidationFailed {
        /// High-level context for when the topology validation failed.
        message: String,

        /// The underlying Level 3 validation error.
        #[source]
        source: TriangulationValidationError,
    },

    /// Local facet repair would remove more simplices than the caller allowed.
    ///
    /// This is emitted by
    /// [`Triangulation::repair_local_facet_issues`](crate::Triangulation::repair_local_facet_issues)
    /// before neighbor repair or validation runs, so callers can retry with a
    /// larger budget without committing a partial topology edit.
    #[error(
        "Local facet repair removal budget exceeded: would remove {attempted} simplices, maximum is {max_simplices_removed}"
    )]
    MaxSimplicesRemovedExceeded {
        /// Maximum simplices the caller allowed this repair to remove.
        max_simplices_removed: usize,
        /// Number of simplices selected for removal.
        attempted: usize,
    },

    /// Spatial index construction failed before insertion.
    #[error("Spatial index construction failed: {reason}")]
    SpatialIndexConstruction {
        /// Structured spatial-index construction failure.
        #[source]
        reason: SpatialIndexConstructionFailure,
    },

    /// A perturbation retry produced non-finite coordinates.
    #[error("Perturbation retry produced invalid coordinates: {source}")]
    PerturbedCoordinateInvalid {
        /// Structured coordinate validation failure for the perturbed point.
        #[source]
        source: CoordinateValidationError,
    },
}

impl From<HashGridIndexError> for InsertionError {
    fn from(source: HashGridIndexError) -> Self {
        Self::SpatialIndexConstruction {
            reason: source.into(),
        }
    }
}

impl From<CavityFillingError> for InsertionError {
    fn from(reason: CavityFillingError) -> Self {
        Self::CavityFilling { reason }
    }
}

impl From<TdsConstructionError> for CavityFillingError {
    fn from(source: TdsConstructionError) -> Self {
        Self::SimplexInsertion {
            reason: source.into(),
        }
    }
}

impl From<TdsConstructionError> for InsertionError {
    fn from(source: TdsConstructionError) -> Self {
        match source {
            TdsConstructionError::ValidationError(source) => Self::TopologyValidation(source),
            TdsConstructionError::DuplicateUuid { entity, uuid } => {
                Self::DuplicateUuid { entity, uuid }
            }
        }
    }
}

impl From<NeighborWiringError> for InsertionError {
    fn from(reason: NeighborWiringError) -> Self {
        Self::NeighborWiring { reason }
    }
}

impl From<InsertionError> for NeighborRebuildError {
    fn from(source: InsertionError) -> Self {
        match source {
            InsertionError::NeighborWiring { reason } => Self::Wiring { reason },
            InsertionError::NonManifoldTopology {
                facet_hash,
                simplex_count,
            } => Self::NonManifoldTopology {
                facet_hash,
                simplex_count,
            },
            InsertionError::TopologyValidation(source) => Self::TopologyValidation {
                reason: source.into(),
            },
            other => Self::Unexpected {
                source: other.into(),
            },
        }
    }
}

impl InsertionError {
    /// Returns true if this error is retryable via coordinate perturbation.
    ///
    /// Retryable errors are geometric degeneracies that may be resolved by
    /// slightly perturbing the vertex coordinates:
    /// - Non-manifold topology (facets shared by >2 simplices, ridge fans)
    /// - Topology validation failures during repair
    /// - Conflict-region boundary degeneracies (disconnected/open boundaries)
    ///
    /// Non-retryable errors are structural failures that won't be fixed by perturbation:
    /// - Duplicate UUIDs
    /// - Duplicate coordinates
    /// - Generic construction or wiring failures
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::insertion::{HullExtensionReason, InsertionError};
    ///
    /// let retryable = InsertionError::NonManifoldTopology {
    ///     facet_hash: 1,
    ///     simplex_count: 3,
    /// };
    /// assert!(retryable.is_retryable());
    ///
    /// use delaunay::prelude::geometry::CoordinateValues;
    ///
    /// let not_retryable = InsertionError::DuplicateCoordinates {
    ///     coordinates: CoordinateValues::from([0.0, 0.0, 0.0]),
    /// };
    /// assert!(!not_retryable.is_retryable());
    ///
    /// let hull = InsertionError::HullExtension {
    ///     reason: HullExtensionReason::NoVisibleFacets,
    /// };
    /// assert!(hull.is_retryable());
    /// ```
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        match self {
            // Non-manifold topology detected during wiring is retryable (geometry degeneracy).
            Self::NonManifoldTopology { .. } => true,
            // Level 3 topology validation: retryability depends on the inner error.
            // Geometry-related topology violations (manifold facet multiplicity, link
            // conditions) may be resolved by perturbation; structural invariant failures
            // (Euler mismatch, inconsistent data structure) cannot.
            Self::TopologyValidationFailed { source, .. } => {
                Self::is_level3_error_retryable(source)
            }
            // TDS-level topology errors: perturbation-retryable sub-variants are
            // geometric, orientation, or facet-multiplicity degeneracies.
            // Structural errors (missing simplices, broken invariants) won't be fixed by perturbation.
            Self::TopologyValidation(tds_err) => Self::is_tds_error_retryable(tds_err),
            // Conflict region errors: only geometry-degeneracy variants are retryable.
            // Structural variants (InvalidStartSimplex, PredicateError, SimplexDataAccessFailed,
            // InvalidSimplexArity, MissingSimplexVertex, InternalInconsistency — regardless of which typed
            // `InternalInconsistencySite` carries the failure context) represent caller
            // or implementation errors that perturbation cannot fix, and so fall
            // through to non-retryable by omission.
            Self::ConflictRegion(ce) => {
                matches!(
                    ce,
                    ConflictError::NonManifoldFacet { .. }
                        | ConflictError::RidgeFan { .. }
                        | ConflictError::DisconnectedBoundary { .. }
                        | ConflictError::OpenBoundary { .. }
                )
            }
            // All other errors are not retryable, except for the specific hull-extension
            // degeneracy handled below.
            //
            // Location errors are treated as non-retryable: `locate()` falls back to a scan when
            // facet-walking fails to make progress (cycle / step limit). Remaining location errors
            // are structural (invalid simplex references) or predicate failures.
            Self::HullExtension { reason } => {
                // Hull extension can fail when the query point is nearly coplanar with the hull
                // surface (no *strictly* visible facets). This is a geometric degeneracy that may
                // be resolved by a perturbation retry.
                matches!(
                    reason,
                    HullExtensionReason::NoVisibleFacets | HullExtensionReason::InvalidPatch { .. }
                )
            }
            Self::CavityFilling { reason } => Self::is_cavity_filling_error_retryable(reason),
            // Neighbor wiring errors are structural failures (missing simplices, index
            // overflow, etc.). Non-manifold topology detection uses the dedicated
            // `NonManifoldTopology` variant.
            Self::NeighborWiring { .. }
            | Self::Location(_)
            | Self::DelaunayValidationFailed { .. }
            | Self::DelaunayRepairFailed { .. }
            | Self::DuplicateCoordinates { .. }
            | Self::DuplicateUuid { .. }
            | Self::MaxSimplicesRemovedExceeded { .. }
            | Self::SpatialIndexConstruction { .. }
            | Self::PerturbedCoordinateInvalid { .. } => false,
        }
    }

    /// Check whether a TDS-level validation error is perturbation-retryable.
    ///
    /// Geometric predicate failures, coherent-orientation violations, and
    /// facet-sharing violations can all be caused by near-degenerate cavity
    /// boundaries that may change after coordinate perturbation. All other
    /// [`TdsError`] variants represent structural bugs that perturbation cannot fix.
    const fn is_tds_error_retryable(tds_err: &TdsError) -> bool {
        matches!(
            tds_err,
            TdsError::Geometric(_)
                | TdsError::OrientationViolation { .. }
                | TdsError::FacetSharingViolation { .. }
        )
    }

    /// Check whether a compact TDS validation summary is perturbation-retryable.
    const fn is_tds_validation_failure_retryable(err: &TdsValidationFailure) -> bool {
        matches!(
            err,
            TdsValidationFailure::Geometric { .. }
                | TdsValidationFailure::OrientationViolation { .. }
                | TdsValidationFailure::FacetSharingViolation { .. }
        )
    }

    /// Check whether a cavity-filling failure is safe to retry after rollback.
    const fn is_cavity_filling_error_retryable(err: &CavityFillingError) -> bool {
        match err {
            CavityFillingError::InvalidFacetSharingAfterRepair { .. } => true,
            CavityFillingError::SimplexInsertion { reason } => match reason {
                TdsConstructionFailure::Validation { reason } => {
                    Self::is_tds_validation_failure_retryable(reason)
                }
                TdsConstructionFailure::DuplicateUuid { .. } => false,
            },
            CavityFillingError::InitialSimplexConstruction { reason } => match reason {
                InitialSimplexConstructionError::TdsValidation { source } => {
                    Self::is_tds_validation_failure_retryable(source)
                }
                InitialSimplexConstructionError::GeometricDegeneracy { .. } => true,
                InitialSimplexConstructionError::DuplicateUuid { .. }
                | InitialSimplexConstructionError::FailedToCreateSimplex { .. }
                | InitialSimplexConstructionError::InsufficientVertices { .. }
                | InitialSimplexConstructionError::InternalInconsistency { .. }
                | InitialSimplexConstructionError::DuplicateCoordinates { .. }
                | InitialSimplexConstructionError::LocalRepairBudgetExceeded { .. }
                | InitialSimplexConstructionError::UnsupportedPeriodicDimension { .. } => false,
                InitialSimplexConstructionError::UnexpectedInsertionStage { reason } => {
                    Self::is_unexpected_initial_simplex_stage_retryable(reason)
                }
            },
            CavityFillingError::NeighborRebuild { reason } => match reason {
                NeighborRebuildError::NonManifoldTopology { .. } => true,
                NeighborRebuildError::TopologyValidation { reason } => {
                    Self::is_tds_validation_failure_retryable(reason)
                }
                NeighborRebuildError::Unexpected { source } => source.is_retryable(),
                NeighborRebuildError::Wiring { .. } => false,
            },
            CavityFillingError::MissingBoundarySimplex { .. }
            | CavityFillingError::MissingInsertedVertex { .. }
            | CavityFillingError::WrongSimplexArity { .. }
            | CavityFillingError::InvalidFacetIndex { .. }
            | CavityFillingError::SimplexCreation { .. }
            | CavityFillingError::RebuiltVertexMissing { .. }
            | CavityFillingError::EmptyConflictRegion { .. }
            | CavityFillingError::EmptyBoundary { .. }
            | CavityFillingError::PerturbationScaleConversion { .. }
            | CavityFillingError::UnsupportedDegenerateLocation { .. }
            | CavityFillingError::EmptyFanTriangulation => false,
        }
    }

    /// Check whether an insertion-stage error that escaped bootstrap construction is retryable.
    const fn is_unexpected_initial_simplex_stage_retryable(
        err: &InitialSimplexUnexpectedInsertionStage,
    ) -> bool {
        match err {
            InitialSimplexUnexpectedInsertionStage::CavityFilling { source } => {
                Self::is_cavity_filling_error_retryable(source)
            }
            InitialSimplexUnexpectedInsertionStage::ConflictRegion { source } => {
                matches!(
                    source,
                    ConflictError::NonManifoldFacet { .. }
                        | ConflictError::RidgeFan { .. }
                        | ConflictError::DisconnectedBoundary { .. }
                        | ConflictError::OpenBoundary { .. }
                )
            }
            InitialSimplexUnexpectedInsertionStage::NonManifoldTopology { .. } => true,
            InitialSimplexUnexpectedInsertionStage::HullExtension { reason } => {
                matches!(
                    reason,
                    HullExtensionReason::NoVisibleFacets | HullExtensionReason::InvalidPatch { .. }
                )
            }
            InitialSimplexUnexpectedInsertionStage::TopologyValidation { source } => {
                Self::is_level3_error_retryable(source)
            }
            InitialSimplexUnexpectedInsertionStage::FinalTopologyValidation { source } => {
                Self::is_invariant_error_summary_retryable(source)
            }
            InitialSimplexUnexpectedInsertionStage::Location { .. }
            | InitialSimplexUnexpectedInsertionStage::DelaunayValidation { .. }
            | InitialSimplexUnexpectedInsertionStage::SpatialIndexConstruction { .. } => false,
        }
    }

    /// Check whether a compact final validation summary is perturbation-retryable.
    const fn is_invariant_error_summary_retryable(err: &InvariantErrorSummary) -> bool {
        match err.detail {
            InvariantErrorSummaryDetail::Tds(kind) => matches!(
                kind,
                TdsErrorKind::Geometric
                    | TdsErrorKind::OrientationViolation
                    | TdsErrorKind::FacetSharingViolation
            ),
            InvariantErrorSummaryDetail::Triangulation(kind) => matches!(
                kind,
                TriangulationValidationErrorKind::ManifoldFacetMultiplicity
                    | TriangulationValidationErrorKind::BoundaryRidgeMultiplicity
                    | TriangulationValidationErrorKind::RidgeLinkNotManifold
                    | TriangulationValidationErrorKind::VertexLinkNotManifold
                    | TriangulationValidationErrorKind::OrientationPromotionNonConvergence
                    | TriangulationValidationErrorKind::IsolatedVertex
            ),
            InvariantErrorSummaryDetail::Delaunay(_) => false,
        }
    }

    /// Check whether a Level 3 (Triangulation) validation error is geometry-related (retryable).
    const fn is_level3_error_retryable(err: &TriangulationValidationError) -> bool {
        match err {
            // Geometry-related topology violations: a near-degenerate insertion can create
            // non-manifold facets, broken links, or boundary violations that perturbation
            // may resolve.
            //
            // `IsolatedVertex` is retryable because a geometrically-sensitive conflict
            // region can leave a pre-existing vertex with no incident simplices; perturbing
            // coordinates changes the conflict region and can avoid stranding the vertex.
            TriangulationValidationError::ManifoldFacetMultiplicity { .. }
            | TriangulationValidationError::BoundaryRidgeMultiplicity { .. }
            | TriangulationValidationError::RidgeLinkNotManifold { .. }
            | TriangulationValidationError::VertexLinkNotManifold { .. }
            | TriangulationValidationError::OrientationPromotionNonConvergence { .. }
            | TriangulationValidationError::IsolatedVertex { .. } => true,
            // All other variants (structural invariant violations, future additions)
            // are conservatively treated as non-retryable.
            _ => false,
        }
    }
}

/// Fill cavity by creating new simplices connecting boundary facets to new vertex.
///
/// Each boundary facet becomes the base of a new (D+1)-simplex with the new vertex as apex.
///
/// # Arguments
/// - `tds` - Mutable triangulation data structure
/// - `new_vertex_key` - Key of the newly inserted vertex
/// - `boundary_facets` - Facets forming the cavity boundary
///
/// # Returns
/// Buffer of newly created simplex keys
///
/// # Errors
///
/// Returns [`InsertionError`] if the cavity cannot be expanded into valid new
/// simplices. Recoverable causes include:
/// - `new_vertex_key` does not identify a vertex in `tds`.
/// - A boundary [`FacetHandle`] references a missing simplex or an invalid facet
///   index.
/// - A boundary simplex has the wrong vertex count for dimension `D`.
/// - Boundary facets imply duplicate or non-manifold replacement simplices.
/// - Geometric orientation checks fail while canonicalizing the new simplices.
/// - Simplex insertion into the underlying [`Tds`] fails.
///
/// # Partial Mutation on Error
///
/// **IMPORTANT**: If this function returns an error, the TDS may be left in a
/// partially updated state with some new simplices already inserted. This function
/// does NOT rollback simplices created before the error occurred.
///
/// This behavior is intentional for performance reasons (avoiding double validation)
/// and is safe because:
/// - This function is only called during vertex insertion operations
/// - If insertion fails, the entire triangulation is typically discarded
/// - Callers should not attempt to use or recover a triangulation after
///   receiving an error from this function
///
/// If you need transactional semantics (all-or-nothing insertion), you must
/// implement rollback logic at a higher level by cloning the TDS before calling
/// this function.
///
/// **Note (Debug Builds)**: In debug builds, this function checks for duplicate
/// boundary facets and logs warnings if found. Duplicate facets will create
/// overlapping simplices, which will be detected and repaired by subsequent topology
/// validation passes (see `detect_local_facet_issues` / `repair_local_facet_issues`).
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::insertion::fill_cavity;
/// use delaunay::prelude::tds::FacetHandle;
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Insertion(#[from] delaunay::prelude::insertion::InsertionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 3> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let mut tds = dt.tds().clone();
/// let Some(vkey) = tds.vertex_keys().next() else { return Ok(()); };
/// let boundary_facets: Vec<FacetHandle> = Vec::new();
///
/// let new_simplices = fill_cavity(&mut tds, vkey, &boundary_facets)?;
/// assert!(new_simplices.is_empty());
/// # Ok(())
/// # }
/// ```
pub fn fill_cavity<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    new_vertex_key: VertexKey,
    boundary_facets: &[FacetHandle],
) -> Result<SimplexKeyBuffer, InsertionError>
where
    U: DataType,
    V: DataType,
{
    fill_cavity_impl(
        tds,
        new_vertex_key,
        boundary_facets,
        CavityInsertionTopology::Checked,
    )
}

/// Fills a replacement cavity using the transactional replacement insertion path.
pub(crate) fn fill_cavity_replacing_simplices<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    new_vertex_key: VertexKey,
    boundary_facets: &[FacetHandle],
) -> Result<SimplexKeyBuffer, InsertionError>
where
    U: DataType,
    V: DataType,
{
    fill_cavity_impl(
        tds,
        new_vertex_key,
        boundary_facets,
        CavityInsertionTopology::Prechecked,
    )
}

/// Fills a caller-validated cavity without per-simplex global insertion scans.
fn fill_cavity_with_prechecked_topology<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    new_vertex_key: VertexKey,
    boundary_facets: &[FacetHandle],
) -> Result<SimplexKeyBuffer, InsertionError>
where
    U: DataType,
    V: DataType,
{
    fill_cavity_impl(
        tds,
        new_vertex_key,
        boundary_facets,
        CavityInsertionTopology::Prechecked,
    )
}

/// Topology-check mode used while filling new cavity simplices.
#[derive(Clone, Copy)]
enum CavityInsertionTopology {
    /// Run the standard TDS insertion checks against every existing simplex.
    Checked,
    /// Skip global insertion scans because the caller validated the local boundary.
    Prechecked,
}

/// Shared cavity-fill implementation for checked, replacement, and prechecked insertions.
#[expect(
    clippy::too_many_lines,
    reason = "Cavity filling includes detailed debug instrumentation and error handling"
)]
fn fill_cavity_impl<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    new_vertex_key: VertexKey,
    boundary_facets: &[FacetHandle],
    insertion_topology: CavityInsertionTopology,
) -> Result<SimplexKeyBuffer, InsertionError>
where
    U: DataType,
    V: DataType,
{
    if !tds.contains_vertex_key(new_vertex_key) {
        return Err(CavityFillingError::MissingInsertedVertex {
            vertex_key: new_vertex_key,
        }
        .into());
    }

    #[cfg(debug_assertions)]
    {
        let log_enabled = std::env::var_os("DELAUNAY_DEBUG_CAVITY").is_some();
        // Check for duplicate boundary facets
        let mut seen_facets: FastHashMap<u64, Vec<FacetHandle>> = FastHashMap::default();
        for facet_handle in boundary_facets {
            if let Some(boundary_simplex) = tds.simplex(facet_handle.simplex_key()) {
                let facet_idx = usize::from(facet_handle.facet_index());
                let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (i, &vertex_key) in boundary_simplex.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vkeys.push(vertex_key);
                    }
                }
                facet_vkeys.sort_unstable();
                let facet_hash = facet_hash_from_sorted_vertices(&facet_vkeys);
                seen_facets
                    .entry(facet_hash)
                    .or_default()
                    .push(*facet_handle);
            }
        }
        let duplicates: Vec<_> = seen_facets
            .iter()
            .filter(|(_, handles)| handles.len() > 1)
            .collect();
        if !duplicates.is_empty() {
            tracing::warn!(
                duplicate_facets = duplicates.len(),
                "fill_cavity: duplicate boundary facets will create overlapping simplices"
            );
            for (hash, handles) in &duplicates {
                tracing::warn!(
                    facet_hash = *hash,
                    instances = handles.len(),
                    handles = ?handles,
                    "fill_cavity: duplicate boundary facet"
                );
            }
        } else if log_enabled {
            tracing::debug!(
                boundary_facets = boundary_facets.len(),
                "fill_cavity: no duplicate boundary facet hashes"
            );
        }

        if log_enabled {
            let mut ridge_counts: FastHashMap<u64, usize> = FastHashMap::default();
            let mut ridge_vertices_map: FastHashMap<u64, VertexKeyBuffer> = FastHashMap::default();

            for facet_handle in boundary_facets {
                let Some(boundary_simplex) = tds.simplex(facet_handle.simplex_key()) else {
                    tracing::warn!(
                        simplex_key = ?facet_handle.simplex_key(),
                        "fill_cavity: missing boundary simplex while building ridge incidence"
                    );
                    continue;
                };
                let facet_idx = usize::from(facet_handle.facet_index());
                let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (i, &vertex_key) in boundary_simplex.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vkeys.push(vertex_key);
                    }
                }

                if facet_vkeys.len() < 2 {
                    continue;
                }

                for omit in 0..facet_vkeys.len() {
                    let mut ridge_vertices =
                        SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                    for (j, &vkey) in facet_vkeys.iter().enumerate() {
                        if j != omit {
                            ridge_vertices.push(vkey);
                        }
                    }
                    ridge_vertices.sort_unstable();
                    let ridge_hash = facet_hash_from_sorted_vertices(&ridge_vertices);
                    *ridge_counts.entry(ridge_hash).or_insert(0) += 1;
                    ridge_vertices_map
                        .entry(ridge_hash)
                        .or_insert_with(|| ridge_vertices.iter().copied().collect());
                }
            }

            let mut ridge_boundary = 0usize;
            let mut ridge_internal = 0usize;
            let mut ridge_over_shared = 0usize;
            for count in ridge_counts.values() {
                match *count {
                    1 => ridge_boundary += 1,
                    2 => ridge_internal += 1,
                    _ => ridge_over_shared += 1,
                }
            }

            tracing::debug!(
                boundary_facets = boundary_facets.len(),
                ridge_boundary,
                ridge_internal,
                ridge_over_shared,
                "fill_cavity: boundary ridge incidence summary"
            );

            if ridge_over_shared > 0 {
                let mut logged = 0usize;
                for (&ridge_hash, &count) in &ridge_counts {
                    if count <= 2 {
                        continue;
                    }
                    if logged >= 10 {
                        break;
                    }
                    tracing::debug!(
                        ridge_hash,
                        ridge_count = count,
                        ridge_vertices = ?ridge_vertices_map.get(&ridge_hash),
                        "fill_cavity: ridge shared by >2 boundary facets"
                    );
                    logged += 1;
                }
            }
        }
    }

    let mut new_simplices = SimplexKeyBuffer::new();

    for facet_handle in boundary_facets {
        let boundary_simplex = tds.simplex(facet_handle.simplex_key()).ok_or_else(|| {
            CavityFillingError::MissingBoundarySimplex {
                simplex_key: facet_handle.simplex_key(),
            }
        })?;

        // Validate boundary simplex has correct dimensionality (D+1 vertices)
        if boundary_simplex.number_of_vertices() != D + 1 {
            return Err(CavityFillingError::WrongSimplexArity {
                simplex_key: facet_handle.simplex_key(),
                actual: boundary_simplex.number_of_vertices(),
                expected: D + 1,
            }
            .into());
        }

        let facet_idx = usize::from(facet_handle.facet_index());
        if facet_idx >= boundary_simplex.number_of_vertices() {
            return Err(CavityFillingError::InvalidFacetIndex {
                simplex_key: facet_handle.simplex_key(),
                facet_index: facet_idx,
                vertex_count: boundary_simplex.number_of_vertices(),
            }
            .into());
        }
        let mut new_simplex_vertices =
            SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();

        // Get vertices of the facet (all except the opposite vertex)
        for (i, &vertex_key) in boundary_simplex.vertices().iter().enumerate() {
            if i != facet_idx {
                new_simplex_vertices.push(vertex_key);
            }
        }

        // Add the new vertex as the apex
        new_simplex_vertices.push(new_vertex_key);
        // The facet order copied above matches the boundary-simplex facet order.
        // For coherent orientation across that shared facet, odd permutation is required
        // exactly when (facet_idx + apex_idx) is even (apex_idx = D).
        let expected_odd_permutation = (facet_idx + D).is_multiple_of(2);
        if expected_odd_permutation && D >= 2 {
            new_simplex_vertices.swap(0, 1);
        }

        // Create and insert the new simplex
        let new_simplex =
            Simplex::try_new(new_simplex_vertices).map_err(CavityFillingError::from)?;
        let simplex_key = match insertion_topology {
            CavityInsertionTopology::Checked => {
                tds.insert_simplex_with_mapping_trusted_vertices(new_simplex)
            }
            CavityInsertionTopology::Prechecked => {
                tds.insert_simplex_with_mapping_prechecked_topology(new_simplex)
            }
        }
        .map_err(CavityFillingError::from)?;

        // Simplex creation provenance: log each newly created simplex with its
        // vertex ordering, geometric orientation, and source boundary facet.
        // Helps trace which insertion step produces negative-orientation simplices.
        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_CAVITY").is_some()
            && let Some(created_simplex) = tds.simplex(simplex_key)
        {
            let simplex_points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
                created_simplex
                    .vertices()
                    .iter()
                    .filter_map(|&vk| tds.vertex(vk).map(|v| *v.point()))
                    .collect();
            let orientation: Option<i32> = if simplex_points.len() == D + 1 {
                match robust_orientation(&simplex_points) {
                    Ok(Orientation::POSITIVE) => Some(1),
                    Ok(Orientation::NEGATIVE) => Some(-1),
                    Ok(Orientation::DEGENERATE) => Some(0),
                    Err(ref e) => {
                        tracing::warn!(
                            simplex_key = ?simplex_key,
                            vertex_keys = ?created_simplex.vertices(),
                            error = %e,
                            "fill_cavity: robust_orientation failed for created simplex"
                        );
                        None
                    }
                }
            } else {
                tracing::warn!(
                    simplex_key = ?simplex_key,
                    vertex_keys = ?created_simplex.vertices(),
                    actual_len = simplex_points.len(),
                    expected_len = D + 1,
                    "fill_cavity: incomplete vertex data for orientation (missing vertices)"
                );
                None
            };
            tracing::debug!(
                simplex_key = ?simplex_key,
                vertex_keys = ?created_simplex.vertices(),
                orientation = ?orientation,
                source_boundary_simplex = ?facet_handle.simplex_key(),
                source_facet_index = usize::from(facet_handle.facet_index()),
                "fill_cavity: created simplex provenance"
            );
        }

        new_simplices.push(simplex_key);
    }

    // Defensive check: 1:1 correspondence is guaranteed by construction
    // (one iteration per boundary facet, one simplex push per iteration)
    debug_assert_eq!(
        boundary_facets.len(),
        new_simplices.len(),
        "Created {} simplices for {} boundary facets (should be 1:1)",
        new_simplices.len(),
        boundary_facets.len()
    );

    Ok(new_simplices)
}

/// Wire neighbor relationships for newly created cavity simplices.
///
/// This function wires:
/// - **Internal facets** between newly created simplices (new↔new)
/// - **Boundary facets** between a new simplex and an existing simplex, using caller-supplied
///   external facet handles (new↔existing)
///
/// The design goal is to keep wiring **local**: callers provide the small set of
/// existing facets that bound the cavity/horizon, avoiding an O(#simplices) global scan.
///
/// The algorithm:
/// 1. Index all facets of `new_simplices` by a canonical facet hash (sorted vertex keys)
/// 2. For each `external_facet`, add it to the facet hash entry *only* if the entry
///    currently has exactly 1 incident simplex (i.e., a new-simplex boundary facet)
/// 3. Wire mutual neighbor relationships for facet-hash entries with exactly 2 incidents
///
/// # Arguments
/// - `tds` - Mutable triangulation data structure
/// - `new_simplices` - Keys of newly created simplices
/// - `external_facets` - Facets on existing simplices that should be glued to the new simplices
/// - `conflict_simplices` - Optional set of simplices being removed (for debug classification only)
///
/// # Returns
/// Ok(()) if wiring succeeds
///
/// # Errors
/// Returns error if neighbor wiring fails or simplices cannot be found.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::insertion::{InsertionError, wire_cavity_neighbors};
/// use delaunay::prelude::collections::SimplexKeyBuffer;
/// use delaunay::prelude::tds::Tds;
///
/// # fn main() -> Result<(), InsertionError> {
/// let mut tds: Tds<(), (), 3> = Tds::empty();
/// let new_simplices = SimplexKeyBuffer::new();
///
/// wire_cavity_neighbors(&mut tds, &new_simplices, [], None)?;
/// # Ok(())
/// # }
/// ```
#[expect(
    clippy::too_many_lines,
    reason = "Neighbor wiring keeps cohesive logic and debug accounting together"
)]
pub fn wire_cavity_neighbors<U, V, const D: usize, I>(
    tds: &mut Tds<U, V, D>,
    new_simplices: &SimplexKeyBuffer,
    external_facets: I,
    conflict_simplices: Option<&SimplexKeyBuffer>,
) -> Result<(), InsertionError>
where
    U: DataType,
    V: DataType,
    I: IntoIterator<Item = FacetHandle>,
{
    type FacetIncidents = SmallBuffer<(SimplexKey, u8), 2>;
    type FacetMap = FastHashMap<u64, FacetIncidents>;
    let mut facet_map: FacetMap = FastHashMap::default();

    // `conflict_simplices` is used only for debug instrumentation, but CI also compiles in
    // release mode with `-D warnings`.
    #[cfg(not(debug_assertions))]
    let _conflict_simplices = conflict_simplices;

    #[cfg(debug_assertions)]
    let log_enabled = std::env::var_os("DELAUNAY_DEBUG_CAVITY").is_some();
    #[cfg(debug_assertions)]
    let ridge_link_debug = std::env::var_os("DELAUNAY_DEBUG_RIDGE_LINK").is_some();
    // Index all facets of new simplices.
    for &simplex_key in new_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;
        let vertex_count = simplex.number_of_vertices();
        ensure_neighbor_wiring_simplex_arity::<D>(simplex_key, vertex_count)?;

        for facet_idx in 0..vertex_count {
            let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in simplex.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vkeys.push(vkey);
                }
            }

            facet_vkeys.sort_unstable();
            let facet_key = facet_hash_from_sorted_vertices(&facet_vkeys);

            let facet_idx_u8 =
                u8::try_from(facet_idx).map_err(|_| NeighborWiringError::FacetIndexOverflow {
                    facet_index: facet_idx,
                    max: u8::MAX,
                })?;

            facet_map
                .entry(facet_key)
                .or_default()
                .push((simplex_key, facet_idx_u8));
        }
    }

    // Index caller-supplied external facets (existing simplices) that should glue to
    // new-simplex boundary facets.
    for external in external_facets {
        let simplex_key = external.simplex_key();
        let facet_idx = usize::from(external.facet_index());

        let simplex = tds
            .simplex(simplex_key)
            .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;

        if facet_idx >= simplex.number_of_vertices() {
            return Err(NeighborWiringError::InvalidFacetIndex {
                simplex_key,
                facet_index: facet_idx,
                vertex_count: simplex.number_of_vertices(),
            }
            .into());
        }

        let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        for (i, &vkey) in simplex.vertices().iter().enumerate() {
            if i != facet_idx {
                facet_vkeys.push(vkey);
            }
        }
        facet_vkeys.sort_unstable();
        let facet_key = facet_hash_from_sorted_vertices(&facet_vkeys);

        let Some(incidents) = facet_map.get_mut(&facet_key) else {
            return Err(NeighborWiringError::ExternalFacetNotFound {
                simplex_key,
                facet_index: external.facet_index(),
                facet_hash: facet_key,
            }
            .into());
        };

        // Only glue to boundary facets (len == 1). If len >= 2, the facet is already
        // shared by multiple new simplices (internal). Adding an external simplex would create
        // a non-manifold facet shared by 3+ simplices.
        if incidents.len() == 1 {
            incidents.push((simplex_key, external.facet_index()));
        } else {
            return Err(NeighborWiringError::ExternalFacetAlreadyShared {
                simplex_key,
                facet_index: external.facet_index(),
                facet_hash: facet_key,
                existing_incidents: incidents.len(),
            }
            .into());
        }
    }

    #[cfg(debug_assertions)]
    let conflict_set: FastHashSet<SimplexKey> = conflict_simplices
        .map(|simplices| simplices.iter().copied().collect())
        .unwrap_or_default();
    #[cfg(debug_assertions)]
    let new_simplices_set: FastHashSet<SimplexKey> = new_simplices.iter().copied().collect();

    // Wire all matching facets (both internal and external).
    // Two simplices share a facet if they have the same facet key.
    for (facet_key, simplices) in &facet_map {
        if simplices.len() == 2 {
            let (c1, idx1) = simplices[0];
            let (c2, idx2) = simplices[1];

            set_neighbor(tds, c1, idx1, Some(c2))?;
            set_neighbor(tds, c2, idx2, Some(c1))?;
        } else if simplices.len() > 2 {
            #[cfg(debug_assertions)]
            {
                let simplex_types: Vec<String> = simplices
                    .iter()
                    .map(|(ck, _)| {
                        if new_simplices_set.contains(ck) {
                            format!("NEW:{ck:?}")
                        } else if conflict_set.contains(ck) {
                            format!("CONFLICT:{ck:?}")
                        } else {
                            format!("EXISTING:{ck:?}")
                        }
                    })
                    .collect();
                tracing::warn!(
                    facet_hash = *facet_key,
                    simplex_count = simplices.len(),
                    simplex_types = ?simplex_types,
                    "wire_cavity_neighbors: non-manifold facet shared by >2 simplices"
                );
            }
            return Err(InsertionError::NonManifoldTopology {
                facet_hash: *facet_key,
                simplex_count: simplices.len(),
            });
        }
        // simplices.len() == 1 means it's a boundary facet (no neighbor)
    }

    #[cfg(debug_assertions)]
    if log_enabled {
        let mut boundary_facets = 0usize;
        let mut internal_facets = 0usize;
        let mut over_shared_facets = 0usize;
        for simplices in facet_map.values() {
            match simplices.len() {
                1 => boundary_facets += 1,
                2 => internal_facets += 1,
                _ => over_shared_facets += 1,
            }
        }
        tracing::debug!(
            new_simplices = new_simplices.len(),
            conflict_simplices = conflict_simplices.map_or(0, SimplexKeyBuffer::len),
            internal_facets,
            boundary_facets,
            over_shared_facets,
            "wire_cavity_neighbors: facet summary"
        );

        if over_shared_facets > 0 {
            let mut logged = 0usize;
            for (facet_key, simplices) in &facet_map {
                if simplices.len() <= 2 {
                    continue;
                }
                if logged >= 10 {
                    break;
                }
                tracing::debug!(
                    facet_hash = *facet_key,
                    simplex_count = simplices.len(),
                    simplices = ?simplices,
                    "wire_cavity_neighbors: facet shared by >2 simplices in map"
                );
                logged += 1;
            }
        }
    }

    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_NEIGHBORS").is_some() {
        let mut mismatches = 0usize;
        for &simplex_key in new_simplices {
            let Some(simplex) = tds.simplex(simplex_key) else {
                continue;
            };
            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };
            for (facet_idx, neighbor_opt) in neighbors.enumerate() {
                let Some(neighbor_key) = neighbor_opt else {
                    continue;
                };
                let Some(neighbor_simplex) = tds.simplex(neighbor_key) else {
                    continue;
                };
                let Some(mirror_idx) = simplex.mirror_facet_index(facet_idx, neighbor_simplex)
                else {
                    mismatches += 1;
                    tracing::warn!(
                        simplex = ?simplex_key,
                        facet_idx,
                        neighbor = ?neighbor_key,
                        "wire_cavity_neighbors: missing mirror facet index"
                    );
                    continue;
                };
                let neighbor_back = neighbor_simplex.neighbor_key(mirror_idx).flatten();
                if neighbor_back != Some(simplex_key) {
                    mismatches += 1;
                    tracing::warn!(
                        simplex = ?simplex_key,
                        facet_idx,
                        neighbor = ?neighbor_key,
                        mirror_idx,
                        neighbor_back = ?neighbor_back,
                        "wire_cavity_neighbors: asymmetric neighbor pointer"
                    );
                }
            }
        }

        tracing::debug!(
            mismatches,
            "wire_cavity_neighbors: neighbor symmetry check complete"
        );
    }

    #[cfg(debug_assertions)]
    if ridge_link_debug {
        let mut total_slots = 0usize;
        let mut neighbor_new = 0usize;
        let mut neighbor_existing = 0usize;
        let mut neighbor_conflict = 0usize;
        let mut neighbor_missing = 0usize;
        let mut neighbor_none = 0usize;
        let mut anomaly_samples: Vec<(SimplexKey, usize, Option<SimplexKey>, String)> = Vec::new();

        for &simplex_key in new_simplices {
            let Some(simplex) = tds.simplex(simplex_key) else {
                continue;
            };

            let vertex_count = simplex.number_of_vertices();
            total_slots = total_slots.saturating_add(vertex_count);

            let Some(neighbors) = simplex.neighbor_keys() else {
                neighbor_none = neighbor_none.saturating_add(vertex_count);
                continue;
            };

            for (facet_idx, neighbor_opt) in neighbors.enumerate() {
                match neighbor_opt {
                    None => {
                        neighbor_none = neighbor_none.saturating_add(1);
                    }
                    Some(neighbor_key) => {
                        if new_simplices_set.contains(&neighbor_key) {
                            neighbor_new = neighbor_new.saturating_add(1);
                        } else if conflict_set.contains(&neighbor_key) {
                            neighbor_conflict = neighbor_conflict.saturating_add(1);
                            if anomaly_samples.len() < 10 {
                                anomaly_samples.push((
                                    simplex_key,
                                    facet_idx,
                                    Some(neighbor_key),
                                    "CONFLICT".to_string(),
                                ));
                            }
                        } else if tds.contains_simplex(neighbor_key) {
                            neighbor_existing = neighbor_existing.saturating_add(1);
                        } else {
                            neighbor_missing = neighbor_missing.saturating_add(1);
                            if anomaly_samples.len() < 10 {
                                anomaly_samples.push((
                                    simplex_key,
                                    facet_idx,
                                    Some(neighbor_key),
                                    "MISSING".to_string(),
                                ));
                            }
                        }
                    }
                }
            }
        }

        tracing::debug!(
            new_simplices = new_simplices.len(),
            total_slots,
            neighbor_new,
            neighbor_existing,
            neighbor_conflict,
            neighbor_missing,
            neighbor_none,
            "wire_cavity_neighbors: new-simplex neighbor classification summary"
        );

        if neighbor_conflict > 0 || neighbor_missing > 0 {
            tracing::warn!(
                neighbor_conflict,
                neighbor_missing,
                anomaly_samples = ?anomaly_samples,
                "wire_cavity_neighbors: unexpected neighbor classifications for new simplices"
            );
        }
    }

    Ok(())
}

/// Collect facets on existing (non-internal) simplices that share a facet with the internal boundary.
///
/// Given:
/// - `internal_simplices`: a set of simplices that will be removed/replaced
/// - `boundary_facets`: facet handles on *internal* simplices that lie on the boundary of that set
///
/// This returns facet handles on *external* simplices (neighbors of `internal_simplices`) whose facet
/// vertex sets match one of the boundary facets.
///
/// This is used to wire new simplices to the pre-existing triangulation without performing a global
/// scan over all simplices.
pub(crate) fn external_facets_for_boundary<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    internal_simplices: &SimplexKeyBuffer,
    boundary_facets: &[FacetHandle],
) -> Result<SmallBuffer<FacetHandle, 64>, InsertionError>
where
    U: DataType,
    V: DataType,
{
    if internal_simplices.is_empty() || boundary_facets.is_empty() {
        return Ok(SmallBuffer::new());
    }

    let internal_set: FastHashSet<SimplexKey> = internal_simplices.iter().copied().collect();

    // Hashes of boundary facets on internal simplices.
    let mut boundary_hashes: FastHashSet<u64> = FastHashSet::default();
    for &facet in boundary_facets {
        let simplex_key = facet.simplex_key();
        let facet_idx = usize::from(facet.facet_index());

        let simplex = tds
            .simplex(simplex_key)
            .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;

        if facet_idx >= simplex.number_of_vertices() {
            return Err(NeighborWiringError::InvalidFacetIndex {
                simplex_key,
                facet_index: facet_idx,
                vertex_count: simplex.number_of_vertices(),
            }
            .into());
        }

        let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        for (i, &vkey) in simplex.vertices().iter().enumerate() {
            if i != facet_idx {
                facet_vkeys.push(vkey);
            }
        }
        facet_vkeys.sort_unstable();
        boundary_hashes.insert(facet_hash_from_sorted_vertices(&facet_vkeys));
    }

    // Candidate external simplices are those reachable via neighbor pointers from the internal set.
    let mut candidate_simplices: FastHashSet<SimplexKey> = FastHashSet::default();
    for &simplex_key in internal_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;
        let Some(neighbors) = simplex.neighbor_keys() else {
            continue;
        };

        for neighbor_opt in neighbors {
            let Some(neighbor_key) = neighbor_opt else {
                continue;
            };
            if !internal_set.contains(&neighbor_key) {
                candidate_simplices.insert(neighbor_key);
            }
        }
    }

    let mut external_facets: SmallBuffer<FacetHandle, 64> = SmallBuffer::new();

    for &simplex_key in &candidate_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;

        for facet_idx in 0..simplex.number_of_vertices() {
            let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in simplex.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vkeys.push(vkey);
                }
            }
            facet_vkeys.sort_unstable();
            let facet_hash = facet_hash_from_sorted_vertices(&facet_vkeys);

            if !boundary_hashes.contains(&facet_hash) {
                continue;
            }

            let facet_idx_u8 =
                u8::try_from(facet_idx).map_err(|_| NeighborWiringError::FacetIndexOverflow {
                    facet_index: facet_idx,
                    max: u8::MAX,
                })?;
            external_facets.push(FacetHandle::from_validated(simplex_key, facet_idx_u8));
        }
    }

    Ok(external_facets)
}

/// Helper: Set a single neighbor relationship
fn set_neighbor<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    simplex_key: SimplexKey,
    facet_idx: u8,
    neighbor: Option<SimplexKey>,
) -> Result<(), InsertionError>
where
    U: DataType,
    V: DataType,
{
    let simplex = tds
        .simplex_mut(simplex_key)
        .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;
    let facet_idx = usize::from(facet_idx);
    if facet_idx >= simplex.number_of_vertices() {
        return Err(NeighborWiringError::InvalidFacetIndex {
            simplex_key,
            facet_index: facet_idx,
            vertex_count: simplex.number_of_vertices(),
        }
        .into());
    }

    if simplex.neighbor_slots().is_none() {
        set_simplex_neighbors_from_keys(simplex, (0..=D).map(|_| None))?;
    }
    let neighbors = simplex.ensure_neighbors_buffer_mut();
    neighbors[facet_idx] = NeighborSlot::from_neighbor_key(neighbor);

    Ok(())
}

/// Installs a fully assigned neighbor buffer and preserves typed simplex context on arity errors.
fn set_simplex_neighbors_from_keys<V, const D: usize>(
    simplex: &mut Simplex<V, D>,
    neighbors: impl IntoIterator<Item = Option<SimplexKey>>,
) -> Result<(), InsertionError> {
    let simplex_id = simplex.uuid();
    simplex
        .set_neighbors_from_keys(neighbors)
        .map_err(|source| TdsError::InvalidSimplex { simplex_id, source }.into())
}

/// Hash a facet from sorted vertex keys.
///
/// Uses [`FastHasher`] for deterministic hashing consistent with other
/// internal collections ([`FastHashMap`], [`FastHashSet`]).
fn facet_hash_from_sorted_vertices(sorted_vkeys: &[VertexKey]) -> u64 {
    debug_assert!(
        sorted_vkeys.windows(2).all(|pair| pair[0] <= pair[1]),
        "facet_hash_from_sorted_vertices: input must be sorted"
    );

    let mut hasher = FastHasher::default();
    for &vkey in sorted_vkeys {
        vkey.hash(&mut hasher);
    }
    hasher.finish()
}

/// Compute a canonical hash for one simplex facet so local repair can match
/// newly exposed facets without scanning the full triangulation.
fn facet_hash_for_simplex<V, const D: usize>(simplex: &Simplex<V, D>, facet_idx: usize) -> u64 {
    let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
    for (i, &vkey) in simplex.vertices().iter().enumerate() {
        if i != facet_idx {
            facet_vkeys.push(vkey);
        }
    }
    facet_vkeys.sort_unstable();
    facet_hash_from_sorted_vertices(&facet_vkeys)
}

/// Return whether `neighbor_key` is the simplex incident across `simplex`'s `facet_idx`.
fn neighbor_slot_points_across_facet<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex: &Simplex<V, D>,
    facet_idx: usize,
    neighbor_key: SimplexKey,
) -> bool
where
    U: DataType,
    V: DataType,
{
    tds.simplex(neighbor_key).is_some_and(|neighbor_simplex| {
        simplex
            .mirror_facet_index(facet_idx, neighbor_simplex)
            .is_some()
    })
}

/// Repair neighbor pointers for a locally affected simplex set.
///
/// This is the cold-path counterpart to [`repair_neighbor_pointers`]. It avoids a
/// triangulation-wide facet scan after a small repair removes simplices by:
/// - seeding the affected set from `seeds` and `optional_external_simplices`,
/// - adding one-hop neighbors reachable through current pointers,
/// - building facet incidence only for that affected set, and
/// - filling only empty, dangling, or facet-incompatible neighbor slots that can be matched locally.
///
/// Existing valid neighbor pointers are preserved, including pointers into simplices outside
/// the affected set.
///
/// This helper is intentionally scoped: it only considers `seeds`,
/// `optional_external_simplices`, and one-hop neighbors reachable from those simplices.
/// Damage outside that local region is ignored. Use [`repair_neighbor_pointers`]
/// when callers need a full triangulation-wide rebuild, or run validation after
/// local repair when the affected region is not known precisely.
///
/// # Arguments
///
/// - `tds` - triangulation data structure whose neighbor slots may be repaired.
/// - `seeds` - simplices that mark the local region touched by cavity repair.
/// - `optional_external_simplices` - outside simplices that share facets with the repaired region.
///
/// # Returns
///
/// `Ok(n)` where `n` is the number of neighbor-pointer slots whose values changed.
///
/// # Errors
///
/// Returns [`InsertionError::NeighborWiring`] if an affected simplex is malformed, a facet index
/// cannot fit in `u8`, or debug-only local validation finds a dangling or asymmetric neighbor
/// pointer. Returns [`InsertionError::NonManifoldTopology`] if local facet incidence has more
/// than two incident simplices for one facet.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::{DelaunayTriangulation, DelaunayTriangulationBuilder};
/// use delaunay::prelude::DelaunayTriangulationConstructionError;
/// use delaunay::prelude::insertion::{
///     InsertionError, TdsMutationError, repair_neighbor_pointers_local,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Mutation(#[from] TdsMutationError),
/// #     #[error(transparent)]
/// #     Insertion(#[from] InsertionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = [
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 1.1]).expect("finite vertex coordinates"),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 2> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let mut tds = dt.tds().clone();
///
/// let Some((simplex_key, facet_idx, neighbor_key)) = tds
///     .simplices()
///     .find_map(|(simplex_key, simplex)| {
///         simplex.neighbors()?.enumerate().find_map(|(facet_idx, neighbor)| {
///             neighbor.map(|neighbor_key| (simplex_key, facet_idx, neighbor_key))
///         })
///     })
/// else {
///     return Ok(());
/// };
///
/// tds.clear_all_neighbors();
///
/// let repaired = repair_neighbor_pointers_local(&mut tds, &[simplex_key], Some(&[neighbor_key]))?;
/// assert!(repaired >= 2);
/// assert_eq!(
///     tds.simplex(simplex_key)
///         .and_then(|simplex| simplex.neighbor_key(facet_idx).flatten()),
///     Some(neighbor_key)
/// );
/// # Ok(())
/// # }
/// ```
#[expect(
    clippy::too_many_lines,
    reason = "Local neighbor repair keeps affected-set construction, facet indexing, and slot application together"
)]
pub fn repair_neighbor_pointers_local<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    seeds: &[SimplexKey],
    optional_external_simplices: Option<&[SimplexKey]>,
) -> Result<usize, InsertionError>
where
    U: DataType,
    V: DataType,
{
    type FacetIncidents = SmallBuffer<(SimplexKey, u8), 2>;

    let mut affected_simplices: FastHashSet<SimplexKey> = FastHashSet::default();
    for &simplex_key in seeds {
        if tds.contains_simplex(simplex_key) {
            affected_simplices.insert(simplex_key);
        }
    }
    if let Some(external_simplices) = optional_external_simplices {
        for &simplex_key in external_simplices {
            if tds.contains_simplex(simplex_key) {
                affected_simplices.insert(simplex_key);
            }
        }
    }

    if affected_simplices.is_empty() {
        return Ok(0);
    }

    // Expand once through existing pointers.  This keeps the repair local while
    // including the surviving simplices whose facets may now match a repaired seed.
    let seed_snapshot: SimplexKeyBuffer = affected_simplices.iter().copied().collect();
    for simplex_key in seed_snapshot {
        let Some(simplex) = tds.simplex(simplex_key) else {
            continue;
        };
        let Some(neighbors) = simplex.neighbor_keys() else {
            continue;
        };
        for neighbor_key in neighbors.flatten() {
            if tds.contains_simplex(neighbor_key) {
                affected_simplices.insert(neighbor_key);
            }
        }
    }

    let mut facet_map: FastHashMap<u64, FacetIncidents> = FastHashMap::default();

    for &simplex_key in &affected_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;

        let vertex_count = simplex.number_of_vertices();
        ensure_neighbor_wiring_simplex_arity::<D>(simplex_key, vertex_count)?;

        for facet_idx in 0..vertex_count {
            let facet_hash = facet_hash_for_simplex(simplex, facet_idx);
            let facet_idx_u8 =
                u8::try_from(facet_idx).map_err(|_| NeighborWiringError::FacetIndexOverflow {
                    facet_index: facet_idx,
                    max: u8::MAX,
                })?;

            let entry = facet_map.entry(facet_hash).or_default();
            entry.push((simplex_key, facet_idx_u8));

            if entry.len() > 2 {
                return Err(InsertionError::NonManifoldTopology {
                    facet_hash,
                    simplex_count: entry.len(),
                });
            }
        }
    }

    let mut total_neighbor_slots_fixed = 0usize;

    for &simplex_key in &affected_simplices {
        let (vertex_count, old_neighbors, replacement_by_facet, current_usable_by_facet) = {
            let simplex = tds
                .simplex(simplex_key)
                .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;
            let vertex_count = simplex.number_of_vertices();
            let old_neighbors: SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE> =
                simplex.neighbor_keys().map_or_else(
                    || SmallBuffer::from_elem(None, vertex_count),
                    |neighbors| {
                        let mut old_neighbors = SmallBuffer::new();
                        old_neighbors.extend(neighbors);
                        old_neighbors.resize(vertex_count, None);
                        old_neighbors
                    },
                );

            let mut replacement_by_facet =
                SmallBuffer::<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            replacement_by_facet.resize(vertex_count, None);

            let mut current_usable_by_facet =
                SmallBuffer::<bool, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            current_usable_by_facet.resize(vertex_count, false);

            for facet_idx in 0..vertex_count {
                let current_neighbor = old_neighbors.get(facet_idx).copied().flatten();
                let current_usable = current_neighbor.is_some_and(|neighbor_key| {
                    neighbor_slot_points_across_facet(tds, simplex, facet_idx, neighbor_key)
                });
                current_usable_by_facet[facet_idx] = current_usable;
                if current_usable {
                    continue;
                }

                let facet_hash = facet_hash_for_simplex(simplex, facet_idx);
                let Some(incidents) = facet_map.get(&facet_hash) else {
                    continue;
                };
                let [(c1, _), (c2, _)] = incidents.as_slice() else {
                    continue;
                };

                replacement_by_facet[facet_idx] = if *c1 == simplex_key {
                    Some(*c2)
                } else if *c2 == simplex_key {
                    Some(*c1)
                } else {
                    None
                };
            }

            (
                vertex_count,
                old_neighbors,
                replacement_by_facet,
                current_usable_by_facet,
            )
        };

        let mut rebuilt_neighbors = old_neighbors;
        let mut changed = false;
        for facet_idx in 0..vertex_count {
            let current_neighbor = rebuilt_neighbors.get(facet_idx).copied().flatten();
            if current_usable_by_facet[facet_idx] {
                continue;
            }

            let replacement = replacement_by_facet[facet_idx];
            if current_neighbor != replacement {
                rebuilt_neighbors[facet_idx] = replacement;
                total_neighbor_slots_fixed = total_neighbor_slots_fixed.saturating_add(1);
                changed = true;
            }
        }

        if !changed {
            continue;
        }

        let simplex = tds
            .simplex_mut(simplex_key)
            .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;
        set_simplex_neighbors_from_keys(simplex, rebuilt_neighbors)?;
    }

    #[cfg(debug_assertions)]
    {
        validate_no_neighbor_cycles_for_simplices(tds, &affected_simplices)?;
        validate_neighbor_symmetry_for_simplices(tds, &affected_simplices)?;
    }

    Ok(total_neighbor_slots_fixed)
}

/// Repair neighbor pointers using a global facet-incidence rebuild.
///
/// This performs a **global** reconstruction of the simplex-neighbor graph:
/// - Build a facet → incident-simplices map from vertex keys (purely combinatorial)
/// - Wire mutual neighbors for facets shared by exactly 2 simplices
/// - Clear all other neighbor slots (boundary facets)
///
/// Unlike the original "scan for missing neighbors" approach, this avoids an
/// O(n²) facet-matching loop and runs in O(n·D) time.
///
/// # Returns
/// `Ok(n)` where `n` is the number of neighbor-pointer slots whose values changed.
///
/// # Errors
/// Returns [`InsertionError::NonManifoldTopology`] if any facet is shared by more than
/// 2 simplices, since neighbor pointers are not well-defined in that case.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayTriangulation, DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
/// };
/// use delaunay::prelude::insertion::{InsertionError, repair_neighbor_pointers};
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Insertion(#[from] InsertionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0]).expect("finite vertex coordinates"),
///     delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0]).expect("finite vertex coordinates"),
/// ];
/// let dt: DelaunayTriangulation<_, (), (), 2> =
///     DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let mut tds = dt.tds().clone();
///
/// let _changed_slots = repair_neighbor_pointers(&mut tds)?;
/// assert!(tds.is_valid().is_ok());
/// # Ok(())
/// # }
/// ```
#[expect(
    clippy::too_many_lines,
    reason = "Neighbor rebuild keeps facet indexing + wiring + application cohesive; prefer correctness and debuggability"
)]
pub fn repair_neighbor_pointers<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
) -> Result<usize, InsertionError>
where
    U: DataType,
    V: DataType,
{
    type FacetIncidents = SmallBuffer<(SimplexKey, u8), 2>;

    let simplex_keys: SimplexKeyBuffer = tds.simplices().map(|(key, _)| key).collect();

    #[cfg(debug_assertions)]
    tracing::trace!(
        simplices = simplex_keys.len(),
        "repair_neighbor_pointers: rebuilding neighbor pointers"
    );

    if simplex_keys.is_empty() {
        return Ok(0);
    }

    // facet_hash -> [(simplex_key, facet_index_opposite_to_facet)]
    let mut facet_map: FastHashMap<u64, FacetIncidents> = FastHashMap::default();

    // simplex_key -> neighbor buffer (len = #vertices in the simplex)
    let mut neighbors_by_simplex: FastHashMap<
        SimplexKey,
        SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE>,
    > = FastHashMap::default();

    for &simplex_key in &simplex_keys {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;

        let vertex_count = simplex.number_of_vertices();
        ensure_neighbor_wiring_simplex_arity::<D>(simplex_key, vertex_count)?;

        let mut neighbors = SmallBuffer::with_capacity(vertex_count);
        neighbors.resize(vertex_count, None);
        neighbors_by_simplex.insert(simplex_key, neighbors);

        let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();

        for facet_idx in 0..vertex_count {
            facet_vkeys.clear();
            for (i, &vkey) in simplex.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vkeys.push(vkey);
                }
            }
            facet_vkeys.sort_unstable();
            let facet_hash = facet_hash_from_sorted_vertices(&facet_vkeys);

            let facet_idx_u8 =
                u8::try_from(facet_idx).map_err(|_| NeighborWiringError::FacetIndexOverflow {
                    facet_index: facet_idx,
                    max: u8::MAX,
                })?;

            let entry = facet_map.entry(facet_hash).or_default();
            entry.push((simplex_key, facet_idx_u8));

            if entry.len() > 2 {
                return Err(InsertionError::NonManifoldTopology {
                    facet_hash,
                    simplex_count: entry.len(),
                });
            }
        }
    }

    // Wire mutual neighbors for facets shared by exactly 2 simplices.
    for (facet_hash, incidents) in facet_map {
        match incidents.as_slice() {
            [(c1, i1), (c2, i2)] => {
                {
                    let n1 = neighbors_by_simplex
                        .get_mut(c1)
                        .ok_or(NeighborWiringError::MissingSimplex { simplex_key: *c1 })?;
                    n1[usize::from(*i1)] = Some(*c2);
                }
                {
                    let n2 = neighbors_by_simplex
                        .get_mut(c2)
                        .ok_or(NeighborWiringError::MissingSimplex { simplex_key: *c2 })?;
                    n2[usize::from(*i2)] = Some(*c1);
                }
            }
            [_] | [] => {
                // Boundary facet => leave as None.
            }
            many => {
                return Err(InsertionError::NonManifoldTopology {
                    facet_hash,
                    simplex_count: many.len(),
                });
            }
        }
    }

    // Apply rebuilt neighbors and count changed slots.
    let mut total_neighbor_slots_fixed: usize = 0;
    for (simplex_key, rebuilt) in neighbors_by_simplex {
        let old_neighbors: SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE> = {
            let simplex = tds
                .simplex(simplex_key)
                .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;

            simplex.neighbor_keys().map_or_else(
                || SmallBuffer::from_elem(None, rebuilt.len()),
                Iterator::collect,
            )
        };

        total_neighbor_slots_fixed = total_neighbor_slots_fixed.saturating_add(
            old_neighbors
                .iter()
                .zip(rebuilt.iter())
                .filter(|(a, b)| a != b)
                .count(),
        );

        let simplex = tds
            .simplex_mut(simplex_key)
            .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;

        set_simplex_neighbors_from_keys(simplex, rebuilt)?;
    }

    #[cfg(debug_assertions)]
    tracing::trace!(
        neighbor_pointers_updated = total_neighbor_slots_fixed,
        "repair_neighbor_pointers: neighbor rebuild complete"
    );
    // Ensure rebuilt topology is also coherently oriented (used by fixtures/tests that
    // construct simplices manually and rely on this utility to produce a fully valid TDS).
    tds.normalize_coherent_orientation()?;

    // Validate no cycles were introduced (debug mode only)
    #[cfg(debug_assertions)]
    validate_no_neighbor_cycles(tds)?;

    Ok(total_neighbor_slots_fixed)
}

/// Debug-only sanity check for neighbor pointers.
///
/// This does **not** attempt to prove the neighbor graph is acyclic (triangulations
/// naturally contain cycles). Instead, it ensures that walking neighbor pointers from a
/// few sample simplices:
/// - terminates (by visiting each discovered simplex at most once), and
/// - does not encounter pointers to missing simplex keys.
///
/// **Performance**: O(n·D) in the worst case for each sampled start simplex.
///
/// # Errors
/// Returns `NeighborWiring` if a neighbor pointer references a missing simplex key.
#[cfg(debug_assertions)]
fn validate_no_neighbor_cycles<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Result<(), InsertionError>
where
    U: DataType,
    V: DataType,
{
    // Sample a few simplices and try walking through their neighbor graph.
    let sample_simplices: SimplexKeyBuffer = tds.simplices().map(|(key, _)| key).take(10).collect();
    let max_simplices = tds.number_of_simplices();

    for &start_simplex in &sample_simplices {
        let mut visited: FastHashSet<SimplexKey> = FastHashSet::default();
        let mut to_visit = vec![start_simplex];
        visited.insert(start_simplex);

        while let Some(current) = to_visit.pop() {
            let simplex = tds
                .simplex(current)
                .ok_or(NeighborWiringError::MissingSimplex {
                    simplex_key: current,
                })?;

            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };

            for neighbor_opt in neighbors {
                let Some(neighbor_key) = neighbor_opt else {
                    continue;
                };

                if neighbor_key == current {
                    return Err(NeighborWiringError::SelfNeighbor {
                        simplex_key: current,
                    }
                    .into());
                }

                if !tds.contains_simplex(neighbor_key) {
                    return Err(NeighborWiringError::MissingNeighborTarget {
                        simplex_key: current,
                        neighbor_key,
                    }
                    .into());
                }

                if visited.insert(neighbor_key) {
                    to_visit.push(neighbor_key);
                    if visited.len() > max_simplices {
                        return Err(NeighborWiringError::NeighborWalkExceededSimplexCount {
                            visited: visited.len(),
                            total: max_simplices,
                        }
                        .into());
                    }
                }
            }
        }
    }

    tracing::trace!("validate_no_neighbor_cycles: neighbor walk terminated");
    Ok(())
}

/// Validate neighbor walks from a local affected set after partial pointer repair.
#[cfg(debug_assertions)]
fn validate_no_neighbor_cycles_for_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    sample_simplices: &FastHashSet<SimplexKey>,
) -> Result<(), InsertionError>
where
    U: DataType,
    V: DataType,
{
    let sample_simplices: SimplexKeyBuffer = sample_simplices.iter().copied().take(10).collect();
    let max_simplices = tds.number_of_simplices();

    for &start_simplex in &sample_simplices {
        let mut visited: FastHashSet<SimplexKey> = FastHashSet::default();
        let mut to_visit = vec![start_simplex];
        visited.insert(start_simplex);

        while let Some(current) = to_visit.pop() {
            let simplex = tds
                .simplex(current)
                .ok_or(NeighborWiringError::MissingSimplex {
                    simplex_key: current,
                })?;

            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };

            for neighbor_opt in neighbors {
                let Some(neighbor_key) = neighbor_opt else {
                    continue;
                };

                if neighbor_key == current {
                    return Err(NeighborWiringError::SelfNeighbor {
                        simplex_key: current,
                    }
                    .into());
                }

                if !tds.contains_simplex(neighbor_key) {
                    return Err(NeighborWiringError::MissingNeighborTarget {
                        simplex_key: current,
                        neighbor_key,
                    }
                    .into());
                }

                if visited.insert(neighbor_key) {
                    to_visit.push(neighbor_key);
                    if visited.len() > max_simplices {
                        return Err(NeighborWiringError::NeighborWalkExceededSimplexCount {
                            visited: visited.len(),
                            total: max_simplices,
                        }
                        .into());
                    }
                }
            }
        }
    }

    Ok(())
}

/// Check mirror-facet symmetry for neighbor slots touched by local repair.
#[cfg(debug_assertions)]
fn validate_neighbor_symmetry_for_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    affected_simplices: &FastHashSet<SimplexKey>,
) -> Result<(), InsertionError>
where
    U: DataType,
    V: DataType,
{
    for &simplex_key in affected_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(NeighborWiringError::MissingSimplex { simplex_key })?;
        let Some(neighbors) = simplex.neighbor_keys() else {
            continue;
        };

        for (facet_idx, neighbor_opt) in neighbors.enumerate() {
            let Some(neighbor_key) = neighbor_opt else {
                continue;
            };

            if neighbor_key == simplex_key {
                return Err(NeighborWiringError::SelfNeighbor { simplex_key }.into());
            }

            let Some(neighbor_simplex) = tds.simplex(neighbor_key) else {
                return Err(NeighborWiringError::MissingNeighborTarget {
                    simplex_key,
                    neighbor_key,
                }
                .into());
            };

            let mirror_facet_index = simplex.mirror_facet_index(facet_idx, neighbor_simplex);
            let neighbor_back = mirror_facet_index
                .and_then(|mirror_idx| neighbor_simplex.neighbor_key(mirror_idx).flatten());

            if neighbor_back != Some(simplex_key) {
                return Err(NeighborWiringError::AsymmetricNeighbor {
                    simplex_key,
                    facet_index: facet_idx,
                    neighbor_key,
                    mirror_facet_index,
                    neighbor_back,
                }
                .into());
            }
        }
    }

    Ok(())
}

/// Extend the convex hull by connecting an exterior vertex to visible boundary facets.
///
/// This function is used when a vertex is outside the current convex hull.
/// It finds all visible boundary facets and creates new simplices connecting them to the new vertex.
///
/// # Arguments
/// - `tds` - Mutable triangulation data structure
/// - `kernel` - Geometric kernel for orientation tests
/// - `new_vertex_key` - Key of the newly inserted vertex
/// - `point` - Coordinates of the new vertex
///
/// # Returns
/// Buffer of newly created simplex keys
///
/// # Errors
/// Returns error if:
/// - Finding visible facets fails
/// - Cavity filling or neighbor wiring fails
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::insertion::extend_hull;
/// use delaunay::prelude::tds::Tds;
/// use delaunay::prelude::tds::VertexKey;
/// use delaunay::prelude::geometry::FastKernel;
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
/// use slotmap::Key;
///
/// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
/// let mut tds: Tds<(), (), 3> = Tds::empty();
/// let vkey = VertexKey::null();
/// let kernel = FastKernel::<f64>::new();
/// let point = Point::try_from([2.0, 2.0, 2.0])?;
///
/// let result = extend_hull(&mut tds, &kernel, vkey, &point);
/// assert!(result.is_err());
/// # Ok(())
/// # }
/// ```
pub fn extend_hull<K, U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    kernel: &K,
    new_vertex_key: VertexKey,
    point: &Point<D>,
) -> Result<SimplexKeyBuffer, InsertionError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    // 2D special-case: if the point is collinear with a boundary edge and lies on
    // that edge segment, split the edge instead of building new hull triangles.
    if D == 2
        && let Some(edge_facet) = find_boundary_edge_split_facet(tds, point)?
    {
        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
            tracing::debug!(
                point = ?point,
                simplex_key = ?edge_facet.simplex_key(),
                facet_index = usize::from(edge_facet.facet_index()),
                "extend_hull: 2D boundary-edge split"
            );
        }

        let mut conflict_simplices = SimplexKeyBuffer::new();
        conflict_simplices.push(edge_facet.simplex_key());

        let mut boundary_facets = extract_cavity_boundary(tds, &conflict_simplices)
            .map_err(InsertionError::ConflictRegion)?;
        boundary_facets.retain(|facet| {
            facet.simplex_key() != edge_facet.simplex_key()
                || facet.facet_index() != edge_facet.facet_index()
        });

        validate_boundary_edge_split_facet_count(boundary_facets.len())?;

        let external_facets =
            external_facets_for_boundary(tds, &conflict_simplices, &boundary_facets)?;

        let new_simplices = fill_cavity_replacing_simplices(tds, new_vertex_key, &boundary_facets)?;
        wire_cavity_neighbors(
            tds,
            &new_simplices,
            external_facets.iter().copied(),
            Some(&conflict_simplices),
        )?;
        let _ = tds.remove_simplices_by_keys(&conflict_simplices);

        return Ok(new_simplices);
    }

    // Find visible boundary facets
    let visible_facets = match find_visible_boundary_facets(tds, kernel, point) {
        Ok(facets) => facets,
        Err(err) => {
            #[cfg(debug_assertions)]
            if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
                tracing::debug!(error = ?err, "extend_hull: visibility computation failed");
            }
            return Err(err);
        }
    };

    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_HULL").is_some() {
        let total_boundary = tds
            .boundary_facets()
            .map_err(|e| InsertionError::HullExtension {
                reason: HullExtensionReason::Tds(e),
            })
            .and_then(|mut facets| {
                facets
                    .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                    .map_err(boundary_facet_iteration_error)
            })?;
        tracing::debug!(
            point = ?point,
            visible_facets = visible_facets.len(),
            total_boundary,
            "extend_hull: visibility summary"
        );
    }

    if visible_facets.is_empty() {
        return Err(InsertionError::HullExtension {
            reason: HullExtensionReason::NoVisibleFacets,
        });
    }

    // Visible hull facets are boundary facets. The new apex cannot already
    // appear in existing simplices, and internal facets are checked when wiring
    // the new simplices below, so avoid a global insertion scan per hull facet.
    let new_simplices = fill_cavity_with_prechecked_topology(tds, new_vertex_key, &visible_facets)?;

    // Wire neighbors using comprehensive facet matching
    // For hull extension, no conflict simplices (nothing is removed)
    wire_cavity_neighbors(tds, &new_simplices, visible_facets.iter().copied(), None)?;

    Ok(new_simplices)
}

fn hull_extension_tds_error(source: impl Into<TdsError>) -> InsertionError {
    InsertionError::HullExtension {
        reason: HullExtensionReason::Tds(source.into()),
    }
}

fn boundary_facet_iteration_error(source: FacetError) -> InsertionError {
    hull_extension_tds_error(source)
}

fn missing_boundary_simplex(simplex_key: SimplexKey, context: &'static str) -> InsertionError {
    hull_extension_tds_error(TdsError::SimplexNotFound {
        simplex_key,
        context: context.to_string(),
    })
}

fn missing_boundary_vertex(
    vertex_key: VertexKey,
    simplex_key: SimplexKey,
    context: &'static str,
) -> InsertionError {
    hull_extension_tds_error(TdsError::VertexNotFound {
        vertex_key,
        context: format!("{context} {simplex_key:?}"),
    })
}

fn invalid_boundary_facet_index(facet_index: u8, facet_count: usize) -> InsertionError {
    hull_extension_tds_error(FacetError::InvalidFacetIndex {
        index: facet_index,
        facet_count,
    })
}

fn validate_boundary_edge_split_facet_count(facet_count: usize) -> Result<(), InsertionError> {
    if facet_count == 2 {
        return Ok(());
    }

    Err(InsertionError::HullExtension {
        reason: HullExtensionReason::InvalidPatch {
            details: format!("2D boundary edge split expected 2 facets, got {facet_count}"),
        },
    })
}

fn find_boundary_edge_split_facet<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    point: &Point<D>,
) -> Result<Option<FacetHandle>, InsertionError>
where
    U: DataType,
    V: DataType,
{
    if D != 2 {
        return Ok(None);
    }

    let mut match_facet: Option<FacetHandle> = None;
    let tol = DEFAULT_TOLERANCE_F64;

    let boundary_facets = tds
        .boundary_facets()
        .map_err(|e| InsertionError::HullExtension {
            reason: HullExtensionReason::Tds(e),
        })?;

    for facet_view in boundary_facets {
        let facet_view = facet_view.map_err(boundary_facet_iteration_error)?;
        let simplex_key = facet_view.simplex_key();
        let facet_index = facet_view.facet_index();
        let simplex = tds.simplex(simplex_key).ok_or_else(|| {
            missing_boundary_simplex(simplex_key, "2D boundary edge split facet lookup")
        })?;

        let mut edge_points = SmallBuffer::<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        let mut opposite_point: Option<Point<D>> = None;

        for (i, &vkey) in simplex.vertices().iter().enumerate() {
            let vertex = tds.vertex(vkey).ok_or_else(|| {
                missing_boundary_vertex(vkey, simplex_key, "2D boundary edge split facet")
            })?;
            if i == usize::from(facet_index) {
                opposite_point = Some(*vertex.point());
            } else {
                edge_points.push(*vertex.point());
            }
        }

        if edge_points.len() != 2 {
            continue;
        }

        let opposite_point = opposite_point
            .ok_or_else(|| invalid_boundary_facet_index(facet_index, simplex.vertices().len()))?;

        let mut simplex_points = SmallBuffer::<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        simplex_points.extend(edge_points.iter().copied());
        simplex_points.push(opposite_point);

        // Use exact orientation (not kernel SoS) to detect true degeneracy.
        // SoS-based kernels never return 0, which would mask the geometric
        // property we need here: is the opposite simplex truly degenerate?
        let opposite_degenerate = matches!(
            robust_orientation(&simplex_points).map_err(|e| InsertionError::HullExtension {
                reason: HullExtensionReason::PredicateFailed(e),
            })?,
            Orientation::DEGENERATE
        );

        if opposite_degenerate {
            continue;
        }

        let mut edge_line = SmallBuffer::<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        edge_line.extend(edge_points.iter().copied());
        edge_line.push(*point);

        // Use exact orientation (not kernel SoS) to detect true collinearity.
        // SoS-based kernels never return 0, which would mask the geometric
        // property we need here: is the point truly on the line through the edge?
        let is_collinear = matches!(
            robust_orientation(&edge_line).map_err(|e| InsertionError::HullExtension {
                reason: HullExtensionReason::PredicateFailed(e),
            })?,
            Orientation::DEGENERATE
        );

        if !is_collinear {
            continue;
        }

        let p0 = edge_points[0].coords();
        let p1 = edge_points[1].coords();
        let p = point.coords();
        let (min_x, max_x) = if p0[0] <= p1[0] {
            (p0[0], p1[0])
        } else {
            (p1[0], p0[0])
        };
        let (min_y, max_y) = if p0[1] <= p1[1] {
            (p0[1], p1[1])
        } else {
            (p1[1], p0[1])
        };
        let on_segment = p[0] >= min_x - tol
            && p[0] <= max_x + tol
            && p[1] >= min_y - tol
            && p[1] <= max_y + tol;

        if on_segment {
            if match_facet.is_some() {
                return Err(InsertionError::HullExtension {
                    reason: HullExtensionReason::InvalidPatch {
                        details: "2D boundary edge split matched multiple facets".to_string(),
                    },
                });
            }
            match_facet = Some(FacetHandle::from_validated(simplex_key, facet_index));
        }
    }

    Ok(match_facet)
}

/// Find all boundary facets visible from a point.
///
/// A boundary facet is visible from a point if the point is on the positive side
/// of the facet's supporting hyperplane (determined by orientation test).
///
/// **Visibility criterion:**
/// - **Strictly visible**: Opposite orientations (orientation signs differ)
/// - **Coplanar** (query orientation == 0): Treated as **weakly visible** to avoid
///   missing horizon facets when the point lies on the hull plane.
/// - **Facet degeneracy** (opposite orientation == 0): Treated as non-visible.
/// - For numerically robust weak visibility beyond coplanar cases, the orientation
///   test logic would need an epsilon-based threshold (not currently implemented).
///
/// # Arguments
/// - `tds` - The triangulation data structure
/// - `kernel` - Geometric kernel for orientation tests
/// - `point` - The query point
///
/// # Returns
/// Vector of facet handles representing visible boundary facets
///
/// # Errors
/// Returns `HullExtension` error if:
/// - Boundary facets cannot be retrieved
/// - Orientation tests fail
/// - Simplex/vertex lookups fail
#[expect(
    clippy::too_many_lines,
    reason = "Visibility checks and diagnostic summaries are kept in a single routine"
)]
fn find_visible_boundary_facets<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    point: &Point<D>,
) -> Result<Vec<FacetHandle>, InsertionError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let mut visible_facets = Vec::new();

    #[cfg(debug_assertions)]
    let log_enabled = std::env::var_os("DELAUNAY_DEBUG_HULL").is_some();
    #[cfg(debug_assertions)]
    let detail_enabled = std::env::var_os("DELAUNAY_DEBUG_HULL_DETAIL").is_some();
    #[cfg(debug_assertions)]
    let track_orientations = detail_enabled || log_enabled;
    #[cfg(debug_assertions)]
    let mut boundary_facets_count = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_opposite_positive = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_opposite_negative = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_opposite_zero = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_point_positive = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_point_negative = 0usize;
    #[cfg(debug_assertions)]
    let mut orientation_point_zero = 0usize;
    #[cfg(debug_assertions)]
    let mut visible_facets_strict = 0usize;
    #[cfg(debug_assertions)]
    let mut visible_facets_weak = 0usize;
    #[cfg(debug_assertions)]
    let mut degenerate_facets: Vec<FacetHandle> = Vec::new();

    // Get all boundary facets
    let boundary_facets = tds
        .boundary_facets()
        .map_err(|e| InsertionError::HullExtension {
            reason: HullExtensionReason::Tds(e),
        })?;

    // Test each boundary facet for visibility
    for facet_view in boundary_facets {
        let facet_view = facet_view.map_err(boundary_facet_iteration_error)?;
        #[cfg(debug_assertions)]
        if track_orientations {
            boundary_facets_count += 1;
        }
        let simplex_key = facet_view.simplex_key();
        let facet_index = facet_view.facet_index();

        // Get the simplex and its vertices
        let simplex = tds.simplex(simplex_key).ok_or_else(|| {
            missing_boundary_simplex(simplex_key, "visible boundary facet lookup")
        })?;

        // Collect points for the simplex in canonical order: facet vertices + opposite vertex.
        let mut simplex_points = SmallBuffer::<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        let mut opposite_point: Option<Point<D>> = None;

        for (i, &vkey) in simplex.vertices().iter().enumerate() {
            let vertex = tds.vertex(vkey).ok_or_else(|| {
                missing_boundary_vertex(vkey, simplex_key, "visible boundary facet")
            })?;
            if i == usize::from(facet_index) {
                opposite_point = Some(*vertex.point());
            } else {
                simplex_points.push(*vertex.point());
            }
        }

        let opposite_point = opposite_point
            .ok_or_else(|| invalid_boundary_facet_index(facet_index, simplex.vertices().len()))?;

        // Append opposite vertex in canonical order.
        simplex_points.push(opposite_point);

        // Test orientation: if point is on same side as inside of hull, facet is visible.
        // For a boundary facet, we want to know if the new point is on the "outside" side
        // relative to the opposite vertex.
        let orientation_with_opposite =
            kernel
                .orientation(&simplex_points)
                .map_err(|e| InsertionError::HullExtension {
                    reason: HullExtensionReason::PredicateFailed(e),
                })?;

        // Replace opposite vertex with query point (last entry in canonical order).
        let last_index = simplex_points.len() - 1;
        simplex_points[last_index] = *point;
        let orientation_with_point =
            kernel
                .orientation(&simplex_points)
                .map_err(|e| InsertionError::HullExtension {
                    reason: HullExtensionReason::PredicateFailed(e),
                })?;

        #[cfg(debug_assertions)]
        if log_enabled && D == 2 && orientation_with_point == 0 {
            let p0 = simplex_points[0].coords();
            let p1 = simplex_points[1].coords();
            let p = point.coords();
            let tol = DEFAULT_TOLERANCE_F64;
            let (min_x, max_x) = if p0[0] <= p1[0] {
                (p0[0], p1[0])
            } else {
                (p1[0], p0[0])
            };
            let (min_y, max_y) = if p0[1] <= p1[1] {
                (p0[1], p1[1])
            } else {
                (p1[1], p0[1])
            };
            let min_x = min_x - tol;
            let max_x = max_x + tol;
            let min_y = min_y - tol;
            let max_y = max_y + tol;
            let on_segment = p[0] >= min_x && p[0] <= max_x && p[1] >= min_y && p[1] <= max_y;
            tracing::debug!(
                simplex_key = ?simplex_key,
                facet_index = usize::from(facet_index),
                point = ?point,
                edge_start = ?p0,
                edge_end = ?p1,
                on_segment,
                "find_visible_boundary_facets: query point coplanar with boundary edge"
            );
        }

        // Facet is visible if the point lies on the opposite side of the facet
        // relative to the opposite vertex. This avoids assuming consistent facet
        // orientations across the boundary.
        //
        // Weak visibility: treat query-point coplanarity as visible (orientation == 0).
        // Hull-facet degeneracy (orientation_with_opposite == 0) is still treated as
        // non-visible to avoid propagating degenerate facets.
        //
        // In 2D, collinear cases are handled via explicit boundary-edge splitting instead;
        // do not treat coplanar edges as visible here to avoid zero-area triangles.
        let is_strict_visible = orientation_with_opposite != 0
            && orientation_with_point != 0
            && orientation_with_opposite.signum() != orientation_with_point.signum();
        let is_weak_visible = if D == 2 {
            false
        } else {
            orientation_with_opposite != 0 && orientation_with_point == 0
        };
        let is_visible = is_strict_visible || is_weak_visible;

        #[cfg(debug_assertions)]
        if track_orientations {
            match orientation_with_opposite.cmp(&0) {
                std::cmp::Ordering::Greater => orientation_opposite_positive += 1,
                std::cmp::Ordering::Less => orientation_opposite_negative += 1,
                std::cmp::Ordering::Equal => orientation_opposite_zero += 1,
            }
            match orientation_with_point.cmp(&0) {
                std::cmp::Ordering::Greater => orientation_point_positive += 1,
                std::cmp::Ordering::Less => orientation_point_negative += 1,
                std::cmp::Ordering::Equal => orientation_point_zero += 1,
            }
            if is_strict_visible {
                visible_facets_strict += 1;
            }
            if is_weak_visible {
                visible_facets_weak += 1;
            }
        }
        #[cfg(debug_assertions)]
        if log_enabled && orientation_with_opposite == 0 && degenerate_facets.len() < 10 {
            degenerate_facets.push(FacetHandle::from_validated(simplex_key, facet_index));
        }
        #[cfg(debug_assertions)]
        if detail_enabled {
            tracing::trace!(
                simplex_key = ?simplex_key,
                facet_index = usize::from(facet_index),
                orientation_with_opposite,
                orientation_with_point,
                is_strict_visible,
                is_weak_visible,
                "find_visible_boundary_facets: facet orientation"
            );
        }

        if is_visible {
            visible_facets.push(FacetHandle::from_validated(simplex_key, facet_index));
        }
    }

    #[cfg(debug_assertions)]
    if log_enabled && visible_facets.is_empty() {
        tracing::debug!(
            point = ?point,
            boundary_facets = boundary_facets_count,
            visible_facets = visible_facets.len(),
            visible_facets_strict,
            visible_facets_weak,
            orientation_opposite_positive,
            orientation_opposite_negative,
            orientation_opposite_zero,
            orientation_point_positive,
            orientation_point_negative,
            orientation_point_zero,
            degenerate_facets = ?degenerate_facets,
            "find_visible_boundary_facets: no visible facets"
        );
    }

    if D >= 2 && !visible_facets.is_empty() {
        let mut ridge_to_facets: FastHashMap<u64, Vec<usize>> = FastHashMap::default();
        let mut ridge_counts: FastHashMap<u64, usize> = FastHashMap::default();
        let mut ridge_vertices_map: FastHashMap<u64, VertexKeyBuffer> = FastHashMap::default();

        for (facet_idx, facet_handle) in visible_facets.iter().enumerate() {
            let Some(simplex) = tds.simplex(facet_handle.simplex_key()) else {
                #[cfg(debug_assertions)]
                tracing::warn!(
                    simplex_key = ?facet_handle.simplex_key(),
                    "find_visible_boundary_facets: missing simplex while summarizing ridges"
                );
                continue;
            };
            let facet_index = usize::from(facet_handle.facet_index());
            let mut facet_vertices = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in simplex.vertices().iter().enumerate() {
                if i != facet_index {
                    facet_vertices.push(vkey);
                }
            }

            if facet_vertices.len() < 2 {
                continue;
            }

            for omit in 0..facet_vertices.len() {
                let mut ridge_vertices =
                    SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (j, &vkey) in facet_vertices.iter().enumerate() {
                    if j != omit {
                        ridge_vertices.push(vkey);
                    }
                }
                ridge_vertices.sort_unstable();
                let ridge_hash = facet_hash_from_sorted_vertices(&ridge_vertices);
                ridge_to_facets
                    .entry(ridge_hash)
                    .or_default()
                    .push(facet_idx);
                *ridge_counts.entry(ridge_hash).or_insert(0) += 1;
                ridge_vertices_map
                    .entry(ridge_hash)
                    .or_insert_with(|| ridge_vertices.clone());
            }
        }

        let mut boundary_ridges = 0usize;
        #[cfg(debug_assertions)]
        let mut internal_ridges = 0usize;
        let mut over_shared_ridges = 0usize;
        for count in ridge_counts.values() {
            match *count {
                1 => boundary_ridges += 1,
                2 => {
                    #[cfg(debug_assertions)]
                    {
                        internal_ridges += 1;
                    }
                }
                _ => over_shared_ridges += 1,
            }
        }

        let mut boundary_components = 0usize;
        let mut boundary_subface_nonmanifold = 0usize;
        #[cfg(debug_assertions)]
        let mut boundary_component_sizes: Vec<usize> = Vec::new();
        #[cfg(debug_assertions)]
        let mut boundary_degree_min: Option<usize> = None;
        #[cfg(debug_assertions)]
        let mut boundary_degree_max: Option<usize> = None;
        #[cfg(debug_assertions)]
        let mut boundary_degree_zero = 0usize;
        #[cfg(debug_assertions)]
        let mut boundary_degree_one = 0usize;
        #[cfg(debug_assertions)]
        let mut boundary_degree_two = 0usize;
        #[cfg(debug_assertions)]
        let mut boundary_degree_over = 0usize;
        #[cfg(debug_assertions)]
        let want_subface_samples = detail_enabled || log_enabled;
        #[cfg(debug_assertions)]
        let mut subface_samples: Vec<(u64, usize, Option<VertexKeyBuffer>)> = Vec::new();
        if D >= 3 && boundary_ridges > 0 {
            let mut boundary_ridge_keys: Vec<u64> = Vec::with_capacity(boundary_ridges);
            for (ridge_hash, count) in &ridge_counts {
                if *count == 1 {
                    boundary_ridge_keys.push(*ridge_hash);
                }
            }

            let mut face_to_ridges: FastHashMap<u64, Vec<usize>> = FastHashMap::default();
            let mut subface_vertices_map: FastHashMap<u64, VertexKeyBuffer> =
                FastHashMap::default();
            let mut subface_vertices: VertexKeyBuffer = VertexKeyBuffer::new();

            for (idx, ridge_hash) in boundary_ridge_keys.iter().enumerate() {
                let Some(ridge_vertices) = ridge_vertices_map.get(ridge_hash) else {
                    continue;
                };
                if ridge_vertices.len() < 2 {
                    continue;
                }
                for omit in 0..ridge_vertices.len() {
                    subface_vertices.clear();
                    for (j, &vk) in ridge_vertices.iter().enumerate() {
                        if j != omit {
                            subface_vertices.push(vk);
                        }
                    }
                    subface_vertices.sort_unstable();
                    let subface_hash = facet_hash_from_sorted_vertices(&subface_vertices);
                    face_to_ridges.entry(subface_hash).or_default().push(idx);
                    subface_vertices_map
                        .entry(subface_hash)
                        .or_insert_with(|| subface_vertices.iter().copied().collect());
                }
            }

            let mut ridge_adjacency: Vec<Vec<usize>> = vec![Vec::new(); boundary_ridge_keys.len()];
            for (subface_hash, ridges) in &face_to_ridges {
                if ridges.len() != 2 {
                    boundary_subface_nonmanifold += 1;
                    #[cfg(debug_assertions)]
                    if want_subface_samples && subface_samples.len() < 10 {
                        subface_samples.push((
                            *subface_hash,
                            ridges.len(),
                            subface_vertices_map.get(subface_hash).cloned(),
                        ));
                    }
                }
                #[cfg(not(debug_assertions))]
                let _subface_hash = subface_hash;
                if ridges.len() < 2 {
                    continue;
                }
                for i in 0..ridges.len() {
                    for j in (i + 1)..ridges.len() {
                        let a = ridges[i];
                        let b = ridges[j];
                        ridge_adjacency[a].push(b);
                        ridge_adjacency[b].push(a);
                    }
                }
            }

            #[cfg(debug_assertions)]
            for adj in &ridge_adjacency {
                let degree = adj.len();
                match degree {
                    0 => boundary_degree_zero += 1,
                    1 => boundary_degree_one += 1,
                    2 => boundary_degree_two += 1,
                    _ => boundary_degree_over += 1,
                }
                boundary_degree_min =
                    Some(boundary_degree_min.map_or(degree, |min| min.min(degree)));
                boundary_degree_max =
                    Some(boundary_degree_max.map_or(degree, |max| max.max(degree)));
            }

            let mut visited = vec![false; boundary_ridge_keys.len()];
            for start in 0..boundary_ridge_keys.len() {
                if visited[start] {
                    continue;
                }
                boundary_components += 1;
                let mut stack = vec![start];
                visited[start] = true;
                #[cfg(debug_assertions)]
                let mut component_size = 0usize;
                while let Some(r) = stack.pop() {
                    #[cfg(debug_assertions)]
                    {
                        component_size += 1;
                    }
                    for &n in &ridge_adjacency[r] {
                        if !visited[n] {
                            visited[n] = true;
                            stack.push(n);
                        }
                    }
                }
                #[cfg(debug_assertions)]
                boundary_component_sizes.push(component_size);
            }
        }

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); visible_facets.len()];
        for facets in ridge_to_facets.values() {
            if facets.len() < 2 {
                continue;
            }
            for i in 0..facets.len() {
                for j in (i + 1)..facets.len() {
                    let a = facets[i];
                    let b = facets[j];
                    adjacency[a].push(b);
                    adjacency[b].push(a);
                }
            }
        }

        let mut visited = vec![false; visible_facets.len()];
        let mut components = 0usize;
        #[cfg(debug_assertions)]
        let mut component_sizes: Vec<usize> = Vec::new();
        for start in 0..visible_facets.len() {
            if visited[start] {
                continue;
            }
            components += 1;
            let mut stack = vec![start];
            visited[start] = true;
            #[cfg(debug_assertions)]
            let mut component_size = 0usize;
            while let Some(f) = stack.pop() {
                #[cfg(debug_assertions)]
                {
                    component_size += 1;
                }
                for &n in &adjacency[f] {
                    if !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
            }
            #[cfg(debug_assertions)]
            component_sizes.push(component_size);
        }

        if over_shared_ridges > 0
            || components > 1
            || boundary_ridges == 0
            || (D >= 3 && boundary_components > 1)
            || (D >= 3 && boundary_subface_nonmanifold > 0)
        {
            #[cfg(debug_assertions)]
            if detail_enabled || log_enabled {
                let visible_sample = &visible_facets[..visible_facets.len().min(10)];
                tracing::debug!(
                    point = ?point,
                    visible_facets = visible_facets.len(),
                    visible_facets_sample = ?visible_sample,
                    ridge_boundary = boundary_ridges,
                    ridge_internal = internal_ridges,
                    ridge_over_shared = over_shared_ridges,
                    components,
                    component_sizes = ?component_sizes,
                    boundary_components,
                    boundary_component_sizes = ?boundary_component_sizes,
                    boundary_subface_nonmanifold,
                    boundary_degree_min,
                    boundary_degree_max,
                    boundary_degree_zero,
                    boundary_degree_one,
                    boundary_degree_two,
                    boundary_degree_over,
                    orientation_opposite_zero,
                    orientation_point_zero,
                    degenerate_facets = ?degenerate_facets,
                    subface_samples = ?subface_samples,
                    "find_visible_boundary_facets: invalid patch"
                );
            }
            return Err(InsertionError::HullExtension {
                reason: HullExtensionReason::InvalidPatch {
                    details: format!(
                        "boundary_ridges={boundary_ridges}, ridge_fans={over_shared_ridges}, components={components}, boundary_components={boundary_components}, boundary_subface_nonmanifold={boundary_subface_nonmanifold}",
                    ),
                },
            });
        }

        #[cfg(debug_assertions)]
        if detail_enabled {
            tracing::debug!(
                visible_facets = visible_facets.len(),
                ridge_boundary = boundary_ridges,
                ridge_internal = internal_ridges,
                ridge_over_shared = over_shared_ridges,
                components,
                component_sizes = ?component_sizes,
                boundary_components,
                boundary_component_sizes = ?boundary_component_sizes,
                boundary_subface_nonmanifold,
                boundary_degree_min,
                boundary_degree_max,
                boundary_degree_zero,
                boundary_degree_one,
                boundary_degree_two,
                boundary_degree_over,
                orientation_opposite_zero,
                orientation_point_zero,
                "find_visible_boundary_facets: ridge connectivity summary"
            );
        }
    }

    #[cfg(debug_assertions)]
    if detail_enabled {
        let mixed_orientation =
            orientation_opposite_positive > 0 && orientation_opposite_negative > 0;
        tracing::debug!(
            boundary_facets = boundary_facets_count,
            visible_facets = visible_facets.len(),
            visible_facets_strict,
            visible_facets_weak,
            orientation_opposite_positive,
            orientation_opposite_negative,
            orientation_opposite_zero,
            orientation_point_positive,
            orientation_point_negative,
            orientation_point_zero,
            mixed_orientation,
            "find_visible_boundary_facets: boundary facet orientation consistency"
        );
        tracing::debug!(
            boundary_facets = boundary_facets_count,
            visible_facets = visible_facets.len(),
            visible_facets_strict,
            visible_facets_weak,
            orientation_opposite_positive,
            orientation_opposite_negative,
            orientation_opposite_zero,
            orientation_point_positive,
            orientation_point_negative,
            orientation_point_zero,
            "find_visible_boundary_facets: orientation summary"
        );
    }

    Ok(visible_facets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayTriangulation;
    use crate::core::algorithms::flips::{
        DelaunayRepairDiagnostics, DelaunayRepairVerificationContext, FlipError, RepairQueueOrder,
    };
    use crate::core::algorithms::locate::InternalInconsistencySite;
    use crate::core::collections::SimplexKeyBuffer;
    use crate::core::tds::GeometricError;
    use crate::core::validation::TopologyGuarantee;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::traits::coordinate::{
        CoordinateConversionError, CoordinateConversionValue, CoordinateValidationError,
        InvalidCoordinateValue,
    };
    use crate::topology::characteristics::euler::TopologyClassification;
    use slotmap::KeyData;
    use std::assert_matches;

    /// Return one mutual neighbor pair from a test TDS.
    fn first_neighbor_pair<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
    ) -> Option<(SimplexKey, u8, SimplexKey, u8)>
    where
        U: DataType,
        V: DataType,
    {
        for (simplex_key, simplex) in tds.simplices() {
            let Some(neighbors) = simplex.neighbor_keys() else {
                continue;
            };
            for (facet_idx, neighbor_opt) in neighbors.enumerate() {
                let Some(neighbor_key) = neighbor_opt else {
                    continue;
                };
                let Some(neighbor_simplex) = tds.simplex(neighbor_key) else {
                    continue;
                };
                let mirror_idx = simplex.mirror_facet_index(facet_idx, neighbor_simplex)?;
                let facet_idx = u8::try_from(facet_idx).ok()?;
                let mirror_idx = u8::try_from(mirror_idx).ok()?;
                return Some((simplex_key, facet_idx, neighbor_key, mirror_idx));
            }
        }
        None
    }

    /// Macro to generate cavity filling tests for different dimensions
    macro_rules! test_fill_cavity {
        ($dim:literal, $initial_vertices:expr, $new_vertex:expr, $expected_facets:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_fill_cavity_ $dim d>]() {
                    // Create initial simplex
                    let vertices = $initial_vertices;
                    let mut dt = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices).unwrap();
                    let tds = dt.tds_mut();

                    // Insert new vertex
                    let new_vertex = $new_vertex;
                    let new_vkey = tds.insert_vertex_with_mapping(new_vertex).unwrap();

                    // Find the single simplex and create boundary facets (one per face)
                    let simplex_key = tds.simplex_keys().next().unwrap();
                    let boundary_facets: Vec<FacetHandle> = (0..=$dim)
                        .map(|i| FacetHandle::from_validated(simplex_key, i))
                        .collect();

                    // Verify expected number of facets
                    assert_eq!(boundary_facets.len(), $expected_facets);

                    // Fill cavity
                    let new_simplices = fill_cavity(tds, new_vkey, &boundary_facets).unwrap();

                    // Should create one simplex per boundary facet
                    assert_eq!(new_simplices.len(), $expected_facets);

                    // Wire neighbors (glue new simplices to the original simplex's facets)
                    wire_cavity_neighbors(
                        tds,
                        &new_simplices,
                        boundary_facets.iter().copied(),
                        None,
                    )
                    .unwrap();

                    // Validate neighbor consistency
                    assert!(
                        tds.is_valid().is_ok(),
                        "TDS should be valid after cavity filling and neighbor wiring: {:?}",
                        tds.is_valid().err()
                    );

                    // Verify all new simplices have correct vertex count
                    for &simplex_key in &new_simplices {
                        let simplex = tds.simplex(simplex_key).unwrap();
                        assert_eq!(
                            simplex.number_of_vertices(),
                            $dim + 1,
                            "New simplex should have D+1 vertices"
                        );
                    }
                }
            }
        };
    }

    // Generate tests for dimensions 2-5
    test_fill_cavity!(
        2,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ],
        crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5]).unwrap(),
        3 // D+1 facets for a 2-simplex
    );

    test_fill_cavity!(
        3,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ],
        crate::core::vertex::Vertex::<(), _>::try_new([0.25, 0.25, 0.25]).unwrap(),
        4 // D+1 facets for a 3-simplex
    );

    test_fill_cavity!(
        4,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
        crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2]).unwrap(),
        5 // D+1 facets for a 4-simplex
    );

    test_fill_cavity!(
        5,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
        crate::core::vertex::Vertex::<(), _>::try_new([0.15, 0.15, 0.15, 0.15, 0.15]).unwrap(),
        6 // D+1 facets for a 5-simplex
    );

    // Error case tests

    #[test]
    fn test_fill_cavity_with_invalid_vertex_key() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));
        let simplex_key = tds.simplex_keys().next().unwrap();
        let boundary_facets: Vec<FacetHandle> = (0..=2)
            .map(|i| FacetHandle::from_validated(simplex_key, i))
            .collect();

        let result = fill_cavity(tds, invalid_vkey, &boundary_facets);
        assert!(
            matches!(
                result,
                Err(InsertionError::CavityFilling {
                    reason: CavityFillingError::MissingInsertedVertex { .. },
                })
            ),
            "Expected CavityFilling error, got: {result:?}"
        );
    }

    #[test]
    fn test_fill_cavity_with_invalid_facet_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let new_vkey = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5]).unwrap(),
            )
            .unwrap();
        let invalid_simplex_key = SimplexKey::from(KeyData::from_ffi(u64::MAX));
        let invalid_boundary_facets: Vec<FacetHandle> = (0..=2)
            .map(|i| FacetHandle::from_validated(invalid_simplex_key, i))
            .collect();

        let result = fill_cavity(tds, new_vkey, &invalid_boundary_facets);
        assert!(result.is_err());
        assert_matches!(
            result,
            Err(InsertionError::CavityFilling {
                reason: CavityFillingError::MissingBoundarySimplex { .. },
            })
        );
    }

    #[test]
    fn test_fill_cavity_with_invalid_facet_index() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let new_vkey = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5]).unwrap(),
            )
            .unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        let original_simplex_count = tds.number_of_simplices();
        let invalid_boundary_facets = vec![FacetHandle::from_validated(simplex_key, 3)];

        let result = fill_cavity(tds, new_vkey, &invalid_boundary_facets);

        assert_matches!(
            result,
            Err(InsertionError::CavityFilling {
                reason: CavityFillingError::InvalidFacetIndex { .. },
            })
        );
        assert_eq!(
            tds.number_of_simplices(),
            original_simplex_count,
            "invalid facet index must fail before inserting replacement simplices"
        );
    }

    #[test]
    fn test_wire_cavity_neighbors_with_invalid_simplices() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let mut invalid_simplices = SimplexKeyBuffer::new();
        invalid_simplices.push(SimplexKey::from(KeyData::from_ffi(u64::MAX)));
        invalid_simplices.push(SimplexKey::from(KeyData::from_ffi(u64::MAX - 1)));

        let result = wire_cavity_neighbors(tds, &invalid_simplices, [], None);
        assert!(result.is_err());
        assert_matches!(
            result,
            Err(InsertionError::NeighborWiring {
                reason: NeighborWiringError::MissingSimplex { .. },
            })
        );
    }

    #[test]
    fn test_wire_cavity_neighbors_errors_on_unmatched_external_facet() {
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
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0]).unwrap(),
            )
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([3.0, 2.0]).unwrap(),
            )
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 3.0]).unwrap(),
            )
            .unwrap();

        let new_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let external_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v3, v4, v5], None).unwrap(),
            )
            .unwrap();

        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.push(new_simplex);

        let err = wire_cavity_neighbors(
            &mut tds,
            &new_simplices,
            [FacetHandle::from_validated(external_simplex, 0)],
            None,
        )
        .unwrap_err();

        assert_matches!(
            err,
            InsertionError::NeighborWiring {
                reason: NeighborWiringError::ExternalFacetNotFound {
                    simplex_key,
                    facet_index: 0,
                    ..
                },
            } if simplex_key == external_simplex
        );
    }

    #[test]
    fn test_wire_cavity_neighbors_errors_on_already_shared_external_facet() {
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
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0]).unwrap(),
            )
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0]).unwrap(),
            )
            .unwrap();

        let new_simplex_a = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let new_simplex_b = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v0, v3], None).unwrap(),
            )
            .unwrap();
        let external_simplex = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![v0, v1, v4], None).unwrap(),
            )
            .unwrap();

        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.push(new_simplex_a);
        new_simplices.push(new_simplex_b);

        let err = wire_cavity_neighbors(
            &mut tds,
            &new_simplices,
            [FacetHandle::from_validated(external_simplex, 2)],
            None,
        )
        .unwrap_err();

        assert_matches!(
            err,
            InsertionError::NeighborWiring {
                reason: NeighborWiringError::ExternalFacetAlreadyShared {
                    simplex_key,
                    facet_index: 2,
                    existing_incidents: 2,
                    ..
                },
            } if simplex_key == external_simplex
        );
    }

    #[test]
    fn test_external_facets_for_boundary_errors_on_missing_internal_simplex() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let missing_simplex = SimplexKey::from(KeyData::from_ffi(u64::MAX));
        let mut internal_simplices = SimplexKeyBuffer::new();
        internal_simplices.push(missing_simplex);
        let boundary_facets = [FacetHandle::from_validated(missing_simplex, 0)];

        let err =
            external_facets_for_boundary(&tds, &internal_simplices, &boundary_facets).unwrap_err();

        assert_matches!(
            err,
            InsertionError::NeighborWiring {
                reason: NeighborWiringError::MissingSimplex { simplex_key },
            } if simplex_key == missing_simplex
        );
    }

    #[test]
    fn test_external_facets_for_boundary_errors_on_missing_neighbor_simplex() {
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
        let missing_neighbor = SimplexKey::from(KeyData::from_ffi(u64::MAX - 1));
        tds.simplex_mut(simplex_key)
            .unwrap()
            .set_neighbors_from_keys(vec![Some(missing_neighbor), None, None])
            .unwrap();

        let mut internal_simplices = SimplexKeyBuffer::new();
        internal_simplices.push(simplex_key);
        let boundary_facets = [FacetHandle::from_validated(simplex_key, 0)];

        let err =
            external_facets_for_boundary(&tds, &internal_simplices, &boundary_facets).unwrap_err();

        assert_matches!(
            err,
            InsertionError::NeighborWiring {
                reason: NeighborWiringError::MissingSimplex { simplex_key },
            } if simplex_key == missing_neighbor
        );
    }

    #[test]
    fn test_external_facets_for_boundary_finds_shared_edge_only() {
        // Two triangles share one edge; only that edge should be returned as an external facet.
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

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v0, v3], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let mut internal_simplices = SimplexKeyBuffer::new();
        internal_simplices.push(c1);

        // Internal set has a single simplex, so all its facets are boundary facets.
        let boundary_facets: Vec<FacetHandle> = (0..=2)
            .map(|i| FacetHandle::from_validated(c1, i))
            .collect();

        let external_facets =
            external_facets_for_boundary(&tds, &internal_simplices, &boundary_facets).unwrap();
        assert_eq!(external_facets.len(), 1);

        let external = external_facets[0];
        assert_eq!(external.simplex_key(), c2);

        let simplex = tds.simplex(external.simplex_key()).unwrap();
        let facet_idx = usize::from(external.facet_index());

        let mut edge: SmallBuffer<VertexKey, 2> = SmallBuffer::new();
        for (i, &vkey) in simplex.vertices().iter().enumerate() {
            if i != facet_idx {
                edge.push(vkey);
            }
        }
        assert_eq!(edge.len(), 2);
        assert!(edge.contains(&v0));
        assert!(edge.contains(&v1));
    }

    #[test]
    fn test_fill_cavity_with_empty_boundary_facets() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let new_vkey = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5]).unwrap(),
            )
            .unwrap();
        let empty_facets: Vec<FacetHandle> = vec![];
        let result = fill_cavity(tds, new_vkey, &empty_facets);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_fill_cavity_errors_on_boundary_simplex_wrong_vertex_count() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        // Insert a new vertex (apex)
        let new_vkey = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5]).unwrap(),
            )
            .unwrap();

        // Corrupt the single boundary simplex by adding one extra vertex key.
        let simplex_key = tds.simplex_keys().next().unwrap();
        let extra_vkey = tds.simplex(simplex_key).unwrap().vertices()[0];
        tds.simplex_mut(simplex_key)
            .unwrap()
            .push_vertex_key(extra_vkey);

        let boundary_facets = vec![FacetHandle::from_validated(simplex_key, 0)];
        let err = fill_cavity(tds, new_vkey, &boundary_facets).unwrap_err();

        assert_matches!(
            err,
            InsertionError::CavityFilling {
                reason: CavityFillingError::WrongSimplexArity { .. },
            }
        );
    }

    #[test]
    fn test_wire_cavity_neighbors_reports_non_manifold_topology() {
        // Three triangles sharing the same edge (v_a,v_b) is non-manifold in 2D.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0]).unwrap(),
            )
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();
        let c3 = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![v_a, v_b, v_e], None).unwrap(),
            )
            .unwrap();

        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.push(c1);
        new_simplices.push(c2);
        new_simplices.push(c3);

        let err = wire_cavity_neighbors(&mut tds, &new_simplices, [], None).unwrap_err();
        assert_matches!(
            err,
            InsertionError::NonManifoldTopology {
                simplex_count: 3,
                ..
            }
        );
    }

    #[test]
    fn test_wire_cavity_neighbors_errors_on_wrong_simplex_arity() {
        // Force a 2D simplex away from triangle arity so wiring reports the invariant directly.
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let simplex_key = tds.simplex_keys().next().unwrap();
        let vkey0 = tds.simplex(simplex_key).unwrap().vertices()[0];

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.push_vertex_key(vkey0);
        }

        let mut new_simplices = SimplexKeyBuffer::new();
        new_simplices.push(simplex_key);

        let err = wire_cavity_neighbors(tds, &new_simplices, [], None).unwrap_err();
        assert_matches!(
            err,
            InsertionError::NeighborWiring {
                reason: NeighborWiringError::WrongSimplexArity {
                    simplex_key: key,
                    expected: 3,
                    found: 4,
                },
            } if key == simplex_key
        );
    }

    #[test]
    fn test_neighbor_wiring_error_facet_index_overflow_variant_remains_distinct() {
        let err = NeighborWiringError::FacetIndexOverflow {
            facet_index: usize::from(u8::MAX) + 1,
            max: u8::MAX,
        };

        assert_matches!(
            err,
            NeighborWiringError::FacetIndexOverflow {
                facet_index,
                max: u8::MAX,
            } if facet_index == usize::from(u8::MAX) + 1
        );
    }

    #[test]
    fn test_neighbor_wiring_error_wrong_simplex_arity_reports_expected_and_found() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(42));
        let err = NeighborWiringError::WrongSimplexArity {
            simplex_key,
            expected: 3,
            found: 2,
        };

        assert_matches!(
            &err,
            NeighborWiringError::WrongSimplexArity {
                simplex_key: key,
                expected: 3,
                found: 2,
            } if *key == simplex_key
        );
        assert!(err.to_string().contains("expected 3"));
        assert!(err.to_string().contains("has 2 vertices"));
    }

    #[test]
    fn test_neighbor_rebuild_error_unexpected_preserves_source_summary() {
        let source = InsertionError::DuplicateCoordinates {
            coordinates: CoordinateValues::from([0.0, 0.0]),
        };

        let err = NeighborRebuildError::from(source.clone());

        match err {
            NeighborRebuildError::Unexpected { source: summary } => {
                assert_eq!(summary.kind, InsertionErrorKind::DuplicateCoordinates);
                assert_eq!(summary.source_kind, None);
                assert_eq!(summary.message, source.to_string());
            }
            other => panic!("expected unexpected neighbor repair error, got {other:?}"),
        }
    }

    #[test]
    fn spatial_index_construction_error_preserves_typed_reason() {
        let err = InsertionError::from(HashGridIndexError::NonPositiveCellSize { value: 0.0 });

        assert_matches!(
            &err,
            InsertionError::SpatialIndexConstruction {
                reason: SpatialIndexConstructionFailure::NonPositiveCellSize {
                    value: CoordinateConversionValue::Scalar(value),
                },
            } if value.get() == 0.0
        );

        let summary = InsertionErrorSummary::from(err);
        assert_eq!(summary.kind, InsertionErrorKind::SpatialIndexConstruction);
        assert!(!summary.is_retryable());
    }

    #[test]
    fn perturbed_coordinate_error_summary_preserves_kind_and_retryability() {
        let source = CoordinateValidationError::InvalidCoordinate {
            coordinate_index: 0,
            coordinate_value: InvalidCoordinateValue::Nan,
            dimension: 2,
        };
        let err = InsertionError::PerturbedCoordinateInvalid { source };

        let summary = InsertionErrorSummary::from(err.clone());

        assert_eq!(summary.kind, InsertionErrorKind::PerturbedCoordinateInvalid);
        assert_eq!(summary.source_kind, None);
        assert_eq!(summary.message, err.to_string());
        assert!(!summary.is_retryable());
        assert!(!err.is_retryable());
    }

    #[test]
    fn initial_simplex_unexpected_stage_preserves_retryability() {
        let reason = CavityFillingError::InitialSimplexConstruction {
            reason: InitialSimplexConstructionError::UnexpectedInsertionStage {
                reason: Box::new(InitialSimplexUnexpectedInsertionStage::ConflictRegion {
                    source: ConflictError::NonManifoldFacet {
                        facet_hash: 0xabc,
                        simplex_count: 3,
                    },
                }),
            },
        };
        assert!(InsertionError::CavityFilling { reason }.is_retryable());

        let reason = CavityFillingError::InitialSimplexConstruction {
            reason: InitialSimplexConstructionError::UnexpectedInsertionStage {
                reason: Box::new(InitialSimplexUnexpectedInsertionStage::Location {
                    source: LocateError::InvalidSimplex {
                        simplex_key: SimplexKey::from(KeyData::from_ffi(7)),
                    },
                }),
            },
        };
        assert!(!InsertionError::CavityFilling { reason }.is_retryable());
    }

    fn sample_delaunay_repair_diagnostics_for_summary() -> DelaunayRepairDiagnostics {
        DelaunayRepairDiagnostics {
            facets_checked: 7,
            flips_performed: 3,
            max_queue_len: 5,
            ambiguous_predicates: 2,
            ambiguous_predicate_samples: vec![11, 13],
            predicate_failures: 1,
            cycle_detections: 4,
            cycle_signature_samples: vec![17, 19],
            attempt: 2,
            queue_order: RepairQueueOrder::Lifo,
        }
    }

    #[test]
    fn test_delaunay_repair_error_summary_covers_all_kinds() {
        let cases = [
            (
                DelaunayRepairError::NonConvergent {
                    max_flips: 42,
                    diagnostics: Box::new(sample_delaunay_repair_diagnostics_for_summary()),
                },
                DelaunayRepairErrorKind::NonConvergent,
            ),
            (
                DelaunayRepairError::PostconditionFailed {
                    message: "remaining non-Delaunay facet".to_string(),
                },
                DelaunayRepairErrorKind::PostconditionFailed,
            ),
            (
                DelaunayRepairError::VerificationFailed {
                    context: DelaunayRepairVerificationContext::StrictValidation,
                    source: Box::new(FlipError::DegenerateSimplex),
                },
                DelaunayRepairErrorKind::VerificationFailed,
            ),
            (
                DelaunayRepairError::OrientationCanonicalizationFailed {
                    message: "orientation pass failed".to_string(),
                },
                DelaunayRepairErrorKind::OrientationCanonicalizationFailed,
            ),
            (
                DelaunayRepairError::InvalidTopology {
                    required: TopologyGuarantee::PLManifold,
                    found: TopologyGuarantee::Pseudomanifold,
                    message: "repair requires PL topology",
                },
                DelaunayRepairErrorKind::InvalidTopology,
            ),
            (
                DelaunayRepairError::HeuristicRebuildFailed {
                    message: "fallback rebuild failed".to_string(),
                },
                DelaunayRepairErrorKind::HeuristicRebuildFailed,
            ),
            (
                DelaunayRepairError::from(FlipError::DegenerateSimplex),
                DelaunayRepairErrorKind::Flip,
            ),
        ];

        for (source, expected_kind) in cases {
            let summary = DelaunayRepairErrorSummary::from(&source);
            assert_eq!(summary.kind, expected_kind);
            assert_eq!(summary.message, source.to_string());
        }
    }

    #[test]
    fn test_insertion_error_summary_preserves_nested_source_kind() {
        let cases = [
            (
                InsertionError::TopologyValidation(TdsError::InconsistentDataStructure {
                    message: "dangling neighbor".to_string(),
                }),
                InsertionErrorKind::TopologyValidation,
                Some(InsertionErrorSourceKind::Tds(
                    TdsErrorKind::InconsistentDataStructure,
                )),
            ),
            (
                InsertionError::TopologyValidationFailed {
                    message: "post-insertion topology validation failed".to_string(),
                    source: TriangulationValidationError::Disconnected { simplex_count: 2 },
                },
                InsertionErrorKind::TopologyValidationFailed,
                Some(InsertionErrorSourceKind::Triangulation(
                    TriangulationValidationErrorKind::Disconnected,
                )),
            ),
            (
                InsertionError::DelaunayValidationFailed {
                    source: DelaunayTriangulationValidationError::VerificationFailed {
                        message: "non-Delaunay facet".to_string(),
                    },
                },
                InsertionErrorKind::DelaunayValidationFailed,
                Some(InsertionErrorSourceKind::Delaunay(
                    DelaunayValidationErrorKind::VerificationFailed,
                )),
            ),
            (
                InsertionError::DelaunayRepairFailed {
                    source: Box::new(DelaunayRepairError::HeuristicRebuildFailed {
                        message: "rebuild could not restore topology".to_string(),
                    }),
                    context: DelaunayRepairFailureContext::LocalRepair,
                },
                InsertionErrorKind::DelaunayRepairFailed,
                Some(InsertionErrorSourceKind::DelaunayRepair(
                    DelaunayRepairErrorKind::HeuristicRebuildFailed,
                )),
            ),
        ];

        for (source, expected_kind, expected_source_kind) in cases {
            let summary = InsertionErrorSummary::from(source.clone());
            assert_eq!(summary.kind, expected_kind);
            assert_eq!(summary.source_kind, expected_source_kind);
            assert_eq!(summary.message, source.to_string());
        }
    }

    #[test]
    fn test_insertion_error_summary_preserves_repair_budget_error() {
        let source = InsertionError::MaxSimplicesRemovedExceeded {
            max_simplices_removed: 2,
            attempted: 3,
        };
        let summary = InsertionErrorSummary::from(source.clone());

        assert_eq!(
            summary.kind,
            InsertionErrorKind::MaxSimplicesRemovedExceeded
        );
        assert_eq!(summary.source_kind, None);
        assert_eq!(summary.message, source.to_string());
        assert!(!summary.retryable);
        assert!(!source.is_retryable());
    }

    #[test]
    fn test_insertion_error_summary_retryability_covers_tds_source_kinds() {
        let geometric = InsertionErrorSummary {
            kind: InsertionErrorKind::TopologyValidation,
            source_kind: Some(InsertionErrorSourceKind::Tds(TdsErrorKind::Geometric)),
            retryable: true,
            message: String::new(),
        };
        let orientation = InsertionErrorSummary {
            kind: InsertionErrorKind::TopologyValidation,
            source_kind: Some(InsertionErrorSourceKind::Tds(
                TdsErrorKind::OrientationViolation,
            )),
            retryable: true,
            message: String::new(),
        };
        let structural = InsertionErrorSummary {
            kind: InsertionErrorKind::TopologyValidation,
            source_kind: Some(InsertionErrorSourceKind::Tds(
                TdsErrorKind::InconsistentDataStructure,
            )),
            retryable: false,
            message: String::new(),
        };

        assert!(geometric.is_retryable());
        assert!(orientation.is_retryable());
        assert!(!structural.is_retryable());
    }

    #[test]
    fn test_robust_fallback_context_preserves_initial_repair_summary() {
        let initial = DelaunayRepairError::PostconditionFailed {
            message: "local predicate violation".to_string(),
        };
        let context = DelaunayRepairFailureContext::LocalRepairRobustFallback {
            initial: DelaunayRepairErrorSummary::from(&initial),
        };

        let msg = context.to_string();
        assert!(msg.contains("local repair failed"));
        assert!(msg.contains("local predicate violation"));
        assert!(msg.contains("robust fallback also failed"));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Covers retryability for each structured cavity-filling payload"
    )]
    fn test_cavity_filling_retryability_inspects_construction_payloads() {
        let geometry_failure = TdsValidationFailure::Geometric {
            source: GeometricError::DegenerateOrientation {
                message: "near-degenerate".to_string(),
            },
        };
        let structural_failure = TdsValidationFailure::InconsistentDataStructure {
            message: "dangling link".to_string(),
        };
        let facet_failure = TdsValidationFailure::FacetSharingViolation {
            facet_key: 0x1234,
            existing_incident_count: 2,
            attempted_incident_count: 3,
            max_incident_count: 2,
            candidate_simplex_uuid: uuid::Uuid::nil(),
            candidate_facet_index: 1,
        };
        let facet_message = facet_failure.to_string();
        assert!(facet_message.contains("exceeds incident-simplex limit"));
        assert!(!facet_message.contains("after inserting candidate simplex"));

        assert!(
            InsertionError::CavityFilling {
                reason: CavityFillingError::SimplexInsertion {
                    reason: TdsConstructionFailure::Validation {
                        reason: geometry_failure.clone(),
                    },
                },
            }
            .is_retryable()
        );
        assert!(
            InsertionError::CavityFilling {
                reason: CavityFillingError::SimplexInsertion {
                    reason: TdsConstructionFailure::Validation {
                        reason: facet_failure,
                    },
                },
            }
            .is_retryable()
        );
        assert!(
            !InsertionError::CavityFilling {
                reason: CavityFillingError::SimplexInsertion {
                    reason: TdsConstructionFailure::Validation {
                        reason: structural_failure,
                    },
                },
            }
            .is_retryable()
        );
        assert!(
            InsertionError::CavityFilling {
                reason: CavityFillingError::InitialSimplexConstruction {
                    reason: InitialSimplexConstructionError::TdsValidation {
                        source: geometry_failure,
                    },
                },
            }
            .is_retryable()
        );
        assert!(
            InsertionError::CavityFilling {
                reason: CavityFillingError::InitialSimplexConstruction {
                    reason: InitialSimplexConstructionError::GeometricDegeneracy {
                        message: "coplanar bootstrap".to_string(),
                    },
                },
            }
            .is_retryable()
        );
        assert!(
            !InsertionError::CavityFilling {
                reason: CavityFillingError::InitialSimplexConstruction {
                    reason: InitialSimplexConstructionError::LocalRepairBudgetExceeded {
                        max_simplices_removed: 2,
                        attempted: 3,
                    },
                },
            }
            .is_retryable()
        );
        assert!(
            InsertionError::CavityFilling {
                reason: CavityFillingError::NeighborRebuild {
                    reason: NeighborRebuildError::from(InsertionError::TopologyValidationFailed {
                        message: "local topology validation failed".to_string(),
                        source: TriangulationValidationError::ManifoldFacetMultiplicity {
                            facet_key: 0x1234,
                            simplex_count: 3,
                        },
                    }),
                },
            }
            .is_retryable()
        );
        assert!(
            !InsertionError::CavityFilling {
                reason: CavityFillingError::NeighborRebuild {
                    reason: NeighborRebuildError::from(InsertionError::TopologyValidationFailed {
                        message: "structural topology validation failed".to_string(),
                        source: TriangulationValidationError::Disconnected { simplex_count: 2 },
                    }),
                },
            }
            .is_retryable()
        );
    }

    #[test]
    fn test_initial_simplex_construction_error_preserves_repair_budget() {
        let source = TriangulationConstructionError::LocalRepairBudgetExceeded {
            max_simplices_removed: 2,
            attempted: 3,
        };

        let converted = InitialSimplexConstructionError::from(source);

        assert_eq!(
            converted,
            InitialSimplexConstructionError::LocalRepairBudgetExceeded {
                max_simplices_removed: 2,
                attempted: 3,
            }
        );
    }

    #[test]
    fn test_initial_simplex_construction_error_preserves_unsupported_periodic_dimension() {
        let source = TriangulationConstructionError::UnsupportedPeriodicDimension {
            dimension: 4,
            max_validated_dimension: 3,
            tracking_issue: 416,
        };

        let converted = InitialSimplexConstructionError::from(source);

        assert_eq!(
            converted,
            InitialSimplexConstructionError::UnsupportedPeriodicDimension {
                dimension: 4,
                max_validated_dimension: 3,
                tracking_issue: 416,
            }
        );
    }

    // InsertionError::is_retryable() tests

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Exhaustive coverage of all InsertionError retryability classifications"
    )]
    fn test_insertion_error_retryable() {
        // Retryable errors
        assert!(
            InsertionError::NonManifoldTopology {
                facet_hash: 0x12345,
                simplex_count: 3
            }
            .is_retryable()
        );

        // InconsistentDataStructure is now non-retryable (structural bug, not geometry).
        assert!(
            !InsertionError::TopologyValidation(TdsError::InconsistentDataStructure {
                message: "test".to_string()
            })
            .is_retryable()
        );
        // Geometry-related variants are still retryable.
        assert!(
            InsertionError::TopologyValidation(TdsError::Geometric(
                GeometricError::DegenerateOrientation {
                    message: "test".to_string()
                }
            ))
            .is_retryable()
        );
        assert!(
            InsertionError::TopologyValidation(TdsError::Geometric(
                GeometricError::NegativeOrientation {
                    message: "test".to_string()
                }
            ))
            .is_retryable()
        );
        assert!(
            InsertionError::TopologyValidation(TdsError::OrientationViolation {
                simplex1_key: SimplexKey::from(KeyData::from_ffi(1)),
                simplex1_uuid: uuid::Uuid::nil(),
                simplex2_key: SimplexKey::from(KeyData::from_ffi(2)),
                simplex2_uuid: uuid::Uuid::nil(),
                simplex1_facet_index: 0,
                simplex2_facet_index: 1,
                facet_vertices: vec![],
                simplex2_facet_vertices: vec![],
                observed_odd_permutation: true,
                expected_odd_permutation: false,
            })
            .is_retryable()
        );
        assert!(
            InsertionError::TopologyValidation(TdsError::FacetSharingViolation {
                facet_key: 0x1234,
                existing_incident_count: 2,
                attempted_incident_count: 3,
                max_incident_count: 2,
                candidate_simplex_uuid: uuid::Uuid::nil(),
                candidate_facet_index: 1,
            })
            .is_retryable()
        );
        // IsolatedVertex is retryable: during insertion, a geometrically-sensitive
        // conflict region can leave a pre-existing vertex with no incident simplices;
        // perturbing coordinates changes the conflict region.
        assert!(
            InsertionError::TopologyValidationFailed {
                message: "test".to_string(),
                source: TriangulationValidationError::IsolatedVertex {
                    vertex_key: VertexKey::from(KeyData::from_ffi(1)),
                    vertex_uuid: uuid::Uuid::nil(),
                },
            }
            .is_retryable()
        );

        // TopologyValidationFailed wrapping a structural error is non-retryable.
        assert!(
            !InsertionError::TopologyValidationFailed {
                message: "test".to_string(),
                source: TriangulationValidationError::EulerCharacteristicMismatch {
                    computed: 3,
                    expected: 2,
                    classification: TopologyClassification::Ball(3),
                },
            }
            .is_retryable()
        );

        // TopologyValidationFailed wrapping a geometry-related error is retryable.
        let geometry_l3 = TriangulationValidationError::ManifoldFacetMultiplicity {
            facet_key: 0x12345,
            simplex_count: 3,
        };
        assert!(
            InsertionError::TopologyValidationFailed {
                message: "test".to_string(),
                source: geometry_l3,
            }
            .is_retryable()
        );

        // TopologyValidationFailed wrapping BoundaryRidgeMultiplicity is retryable.
        assert!(
            InsertionError::TopologyValidationFailed {
                message: "test".to_string(),
                source: TriangulationValidationError::BoundaryRidgeMultiplicity {
                    ridge_key: 0xab,
                    boundary_facet_count: 3,
                },
            }
            .is_retryable()
        );
        // TopologyValidationFailed wrapping RidgeLinkNotManifold is retryable.
        assert!(
            InsertionError::TopologyValidationFailed {
                message: "test".to_string(),
                source: TriangulationValidationError::RidgeLinkNotManifold {
                    ridge_key: 0xcd,
                    link_vertex_count: 4,
                    link_edge_count: 5,
                    max_degree: 3,
                    degree_one_vertices: 1,
                    connected: false,
                },
            }
            .is_retryable()
        );
        // TopologyValidationFailed wrapping VertexLinkNotManifold is retryable.
        assert!(
            InsertionError::TopologyValidationFailed {
                message: "test".to_string(),
                source: TriangulationValidationError::VertexLinkNotManifold {
                    vertex_key: VertexKey::from(KeyData::from_ffi(1)),
                    link_vertex_count: 3,
                    link_simplex_count: 4,
                    boundary_facet_count: 1,
                    max_degree: 2,
                    connected: false,
                    interior_vertex: true,
                },
            }
            .is_retryable()
        );
        // TopologyValidationFailed wrapping EulerCharacteristicMismatch is NOT retryable
        // (wildcard fallback).
        assert!(
            !InsertionError::TopologyValidationFailed {
                message: "test".to_string(),
                source: TriangulationValidationError::EulerCharacteristicMismatch {
                    computed: 3,
                    expected: 2,
                    classification: TopologyClassification::Ball(3),
                },
            }
            .is_retryable()
        );

        // NeighborWiring is unconditionally non-retryable.
        assert!(
            !InsertionError::NeighborWiring {
                reason: NeighborWiringError::MissingSimplex {
                    simplex_key: SimplexKey::from(KeyData::from_ffi(u64::MAX)),
                }
            }
            .is_retryable()
        );

        // Conflict-region errors
        assert!(
            InsertionError::ConflictRegion(ConflictError::NonManifoldFacet {
                facet_hash: 0x12345_u64,
                simplex_count: 3,
            })
            .is_retryable()
        );
        assert!(
            InsertionError::ConflictRegion(ConflictError::RidgeFan {
                facet_count: 3,
                ridge_vertex_count: 2,
                extra_simplices: vec![],
            })
            .is_retryable()
        );
        // extra_simplices contents do not affect retryability — a non-empty vec is also retryable.
        assert!(
            InsertionError::ConflictRegion(ConflictError::RidgeFan {
                facet_count: 3,
                ridge_vertex_count: 2,
                extra_simplices: vec![SimplexKey::from(KeyData::from_ffi(1))],
            })
            .is_retryable()
        );
        assert!(
            !InsertionError::ConflictRegion(ConflictError::InvalidStartSimplex {
                simplex_key: SimplexKey::from(KeyData::from_ffi(u64::MAX)),
            })
            .is_retryable()
        );
        assert!(
            !InsertionError::ConflictRegion(ConflictError::PredicateError {
                source: CoordinateConversionError::ConversionFailed {
                    coordinate_index: 0,
                    coordinate_value: CoordinateConversionValue::Other("test".to_string()),
                    from_type: "f64",
                    to_type: "f64",
                },
            })
            .is_retryable()
        );
        assert!(
            !InsertionError::ConflictRegion(ConflictError::SimplexDataAccessFailed {
                simplex_key: SimplexKey::from(KeyData::from_ffi(1)),
                message: "test".to_string(),
            })
            .is_retryable()
        );
        assert!(
            !InsertionError::ConflictRegion(ConflictError::InvalidSimplexArity {
                simplex_key: SimplexKey::from(KeyData::from_ffi(1)),
                expected: 3,
                found: 2,
            })
            .is_retryable()
        );
        assert!(
            !InsertionError::ConflictRegion(ConflictError::MissingSimplexVertex {
                simplex_key: SimplexKey::from(KeyData::from_ffi(1)),
                vertex_key: VertexKey::from(KeyData::from_ffi(999)),
            })
            .is_retryable()
        );
        // InternalInconsistency is not retryable regardless of the typed site:
        // perturbation cannot fix logic errors.
        assert!(
            !InsertionError::ConflictRegion(ConflictError::InternalInconsistency {
                site: InternalInconsistencySite::RidgeFanExtraFacetOutOfBounds {
                    index: 7,
                    boundary_facets_len: 5,
                    extra_facets_len: 3,
                },
            })
            .is_retryable()
        );
        assert!(
            !InsertionError::ConflictRegion(ConflictError::InternalInconsistency {
                site: InternalInconsistencySite::OpenBoundaryMissingFirstFacet {
                    first_facet: 4,
                    boundary_facets_len: 2,
                    facet_count: 1,
                    ridge_vertex_count: 2,
                },
            })
            .is_retryable()
        );
        assert!(
            !InsertionError::ConflictRegion(ConflictError::InternalInconsistency {
                site: InternalInconsistencySite::RidgeInfoMissingSecondFacet {
                    first_facet: 0,
                    boundary_facets_len: 2,
                    ridge_vertex_count: 2,
                },
            })
            .is_retryable()
        );
        assert!(
            InsertionError::ConflictRegion(ConflictError::DisconnectedBoundary {
                visited: 1,
                total: 3,
                disconnected_simplices: vec![],
            })
            .is_retryable()
        );
        assert!(
            InsertionError::ConflictRegion(ConflictError::OpenBoundary {
                facet_count: 1,
                ridge_vertex_count: 2,
                open_simplex: SimplexKey::from(KeyData::from_ffi(1)),
            })
            .is_retryable()
        );

        // Non-retryable errors
        assert!(
            !InsertionError::DuplicateUuid {
                entity: EntityKind::Vertex,
                uuid: uuid::Uuid::new_v4(),
            }
            .is_retryable()
        );

        assert!(
            !InsertionError::DuplicateCoordinates {
                coordinates: CoordinateValues::from([0.0, 0.0, 0.0])
            }
            .is_retryable()
        );

        assert!(
            !InsertionError::CavityFilling {
                reason: CavityFillingError::EmptyFanTriangulation,
            }
            .is_retryable()
        );
        assert!(
            InsertionError::CavityFilling {
                reason: CavityFillingError::InvalidFacetSharingAfterRepair {
                    stage: CavityRepairStage::PrimaryInsertion,
                },
            }
            .is_retryable()
        );
        assert!(
            InsertionError::CavityFilling {
                reason: CavityFillingError::NeighborRebuild {
                    reason: NeighborRebuildError::NonManifoldTopology {
                        facet_hash: 0x123,
                        simplex_count: 3,
                    },
                },
            }
            .is_retryable()
        );

        assert!(
            InsertionError::HullExtension {
                reason: HullExtensionReason::NoVisibleFacets
            }
            .is_retryable()
        );

        assert!(
            InsertionError::HullExtension {
                reason: HullExtensionReason::InvalidPatch {
                    details: "test".to_string(),
                }
            }
            .is_retryable()
        );

        assert!(
            !InsertionError::HullExtension {
                reason: HullExtensionReason::Other {
                    message: "Failed to get boundary facets: test".to_string()
                }
            }
            .is_retryable()
        );

        assert!(
            !InsertionError::HullExtension {
                reason: HullExtensionReason::PredicateFailed(
                    CoordinateConversionError::ConversionFailed {
                        coordinate_index: 0,
                        coordinate_value: CoordinateConversionValue::Other("test".to_string()),
                        from_type: "f64",
                        to_type: "f64",
                    }
                )
            }
            .is_retryable()
        );

        assert!(
            !InsertionError::HullExtension {
                reason: HullExtensionReason::Tds(TdsError::InconsistentDataStructure {
                    message: "test".to_string(),
                })
            }
            .is_retryable()
        );
    }

    #[test]
    fn hull_extension_error_helpers_preserve_typed_sources() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let vertex_key = VertexKey::from(KeyData::from_ffi(2));

        assert_matches!(
            missing_boundary_simplex(simplex_key, "facet lookup"),
            InsertionError::HullExtension {
                reason: HullExtensionReason::Tds(TdsError::SimplexNotFound {
                    simplex_key: found_simplex_key,
                    context,
                }),
            } if found_simplex_key == simplex_key && context == "facet lookup"
        );

        assert_matches!(
            missing_boundary_vertex(vertex_key, simplex_key, "visible boundary facet"),
            InsertionError::HullExtension {
                reason: HullExtensionReason::Tds(TdsError::VertexNotFound {
                    vertex_key: found_vertex_key,
                    context,
                }),
            } if found_vertex_key == vertex_key
                && context.starts_with("visible boundary facet")
                && context.contains(&format!("{simplex_key:?}"))
        );

        assert_matches!(
            invalid_boundary_facet_index(3, 2),
            InsertionError::HullExtension {
                reason: HullExtensionReason::Tds(TdsError::FacetError(
                    FacetError::InvalidFacetIndex {
                        index: 3,
                        facet_count: 2,
                    }
                )),
            }
        );

        assert_matches!(
            boundary_facet_iteration_error(FacetError::InsideVertexNotFound),
            InsertionError::HullExtension {
                reason: HullExtensionReason::Tds(TdsError::FacetError(
                    FacetError::InsideVertexNotFound
                )),
            }
        );
    }

    // repair_neighbor_pointers tests

    /// Macro to generate `repair_neighbor_pointers` tests for different dimensions
    macro_rules! test_repair_neighbors {
        ($dim:literal, $initial_vertices:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_repair_neighbor_pointers_ $dim d>]() {
                    let vertices = $initial_vertices;
                    let mut dt = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices).unwrap();
                    let tds = dt.tds_mut();

                    // Verify all neighbor pointers are initially valid
                    for (_, simplex) in tds.simplices() {
                        if let Some(neighbors) = simplex.neighbors() {
                            for neighbor_opt in neighbors {
                                if let Some(neighbor_key) = neighbor_opt {
                                    assert!(tds.contains_simplex(neighbor_key), "Neighbor should exist");
                                }
                            }
                        }
                    }

                    // Repair should succeed (no-op since pointers are valid)
                    let fixed = repair_neighbor_pointers(tds).unwrap();
                    assert_eq!(fixed, 0);

                    // Verify all pointers still valid after repair
                    for (_, simplex) in tds.simplices() {
                        if let Some(neighbors) = simplex.neighbors() {
                            for neighbor_opt in neighbors {
                                if let Some(neighbor_key) = neighbor_opt {
                                    assert!(tds.contains_simplex(neighbor_key), "Neighbor should still exist after repair");
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    test_repair_neighbors!(
        2,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.5]).unwrap(),
        ]
    );

    test_repair_neighbors!(
        3,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.25, 0.25, 0.25]).unwrap(),
        ]
    );

    test_repair_neighbors!(
        4,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2]).unwrap(),
        ]
    );

    test_repair_neighbors!(
        5,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.15, 0.15, 0.15, 0.15, 0.15]).unwrap(),
        ]
    );

    #[test]
    fn test_repair_neighbor_pointers_reconstructs_missing_neighbors() {
        // Create a simple 2D triangulation with two triangles.
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.1]).unwrap(), // break cocircular symmetry
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        // Remove all neighbor pointers.
        tds.clear_all_neighbors();
        assert!(tds.simplices().all(|(_, c)| c.neighbors().is_none()));

        // Repair should rebuild internal adjacencies.
        let fixed = repair_neighbor_pointers(tds).unwrap();
        assert!(
            fixed > 0,
            "Expected at least one neighbor pointer to be repaired"
        );

        let any_internal_neighbor = tds.simplices().any(|(_, c)| {
            c.neighbors()
                .is_some_and(|mut n| n.any(|neighbor| neighbor.is_some()))
        });
        assert!(
            any_internal_neighbor,
            "Expected at least one internal neighbor after repair"
        );

        assert!(tds.is_valid().is_ok());
    }

    macro_rules! test_repair_neighbor_pointers_local_dimensions {
        (
            $dim:literal,
            $initial_vertices:expr,
            $shared_facet_vertices:expr,
            $opposite_vertices:expr
        ) => {
            pastey::paste! {
                #[test]
                fn [<test_repair_neighbor_pointers_local_reconstructs_missing_slot_ $dim d>]() {
                    let vertices = $initial_vertices;
                    let mut dt = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices).unwrap();
                    let tds = dt.tds_mut();
                    let (simplex_key, facet_idx, neighbor_key, _) =
                        first_neighbor_pair(tds).expect("test triangulation should have adjacent simplices");

                    set_neighbor(tds, simplex_key, facet_idx, None).unwrap();
                    let repaired = repair_neighbor_pointers_local(tds, &[simplex_key], Some(&[neighbor_key]))
                        .expect("local repair should reconstruct the missing slot");

                    assert_eq!(repaired, 1);
                    let simplex = tds.simplex(simplex_key).unwrap();
                    assert_eq!(
                        simplex.neighbor_key(usize::from(facet_idx)).flatten(),
                        Some(neighbor_key)
                    );
                    assert!(tds.is_valid().is_ok());
                }

                #[test]
                fn [<test_repair_neighbor_pointers_local_reports_non_manifold_incidence_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let shared_vertices = $shared_facet_vertices;
                    let opposite_vertices = $opposite_vertices;

                    let mut shared_keys = Vec::new();
                    for vertex in shared_vertices {
                        shared_keys.push(tds.insert_vertex_with_mapping(vertex).unwrap());
                    }

                    let mut simplex_keys = Vec::new();
                    for (idx, vertex) in opposite_vertices.into_iter().enumerate() {
                        let opposite_key = tds.insert_vertex_with_mapping(vertex).unwrap();
                        let mut vertices = shared_keys.clone();
                        vertices.push(opposite_key);
                        let simplex = Simplex::try_new_with_data(vertices, None).unwrap();
                        let simplex_key = if idx < 2 {
                            tds.insert_simplex_with_mapping(simplex)
                        } else {
                            tds.insert_simplex_bypassing_topology_checks_for_test(simplex)
                        }
                        .unwrap();
                        simplex_keys.push(simplex_key);
                    }

                    let err = repair_neighbor_pointers_local(&mut tds, &simplex_keys, None).unwrap_err();

                    assert_matches!(
                        err,
                        InsertionError::NonManifoldTopology { simplex_count: 3, .. }
                    );
                }
            }
        };
    }

    test_repair_neighbor_pointers_local_dimensions!(
        2,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.1]).unwrap(),
        ],
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap()
        ],
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0]).unwrap(),
        ]
    );

    test_repair_neighbor_pointers_local_dimensions!(
        3,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.25, 0.25, 0.25]).unwrap(),
        ],
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
        ],
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, -1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0, 1.0]).unwrap(),
        ]
    );

    test_repair_neighbor_pointers_local_dimensions!(
        4,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.2, 0.2, 0.2, 0.2]).unwrap(),
        ],
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0]).unwrap(),
        ],
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, -1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0, 1.0, 1.0]).unwrap(),
        ]
    );

    test_repair_neighbor_pointers_local_dimensions!(
        5,
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.15, 0.15, 0.15, 0.15, 0.15]).unwrap(),
        ],
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
        ],
        vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0, 0.0, -1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
        ]
    );

    #[test]
    fn test_repair_neighbor_pointers_local_replaces_stale_neighbor_slot() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.1]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();
        let (simplex_key, facet_idx, neighbor_key, _) =
            first_neighbor_pair(tds).expect("test triangulation should have adjacent simplices");
        let stale_neighbor = SimplexKey::from(KeyData::from_ffi(u64::MAX - 7));
        assert!(!tds.contains_simplex(stale_neighbor));

        set_neighbor(tds, simplex_key, facet_idx, Some(stale_neighbor)).unwrap();
        let repaired = repair_neighbor_pointers_local(tds, &[simplex_key], Some(&[neighbor_key]))
            .expect("local repair should replace a stale neighbor slot");

        assert_eq!(repaired, 1);
        let simplex = tds.simplex(simplex_key).unwrap();
        assert_eq!(
            simplex.neighbor_key(usize::from(facet_idx)).flatten(),
            Some(neighbor_key)
        );
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_repair_neighbor_pointers_local_replaces_wrong_live_neighbor_slot() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_a = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, -1.0]).unwrap(),
            )
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v_f = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 1.0]).unwrap(),
            )
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();
        let wrong_live_neighbor = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_e, v_f], None).unwrap(),
            )
            .unwrap();

        let shared_facet_idx = 2usize;
        let shared_facet_idx_u8 = u8::try_from(shared_facet_idx).unwrap();
        assert!(
            tds.simplex(c1)
                .unwrap()
                .mirror_facet_index(shared_facet_idx, tds.simplex(wrong_live_neighbor).unwrap())
                .is_none()
        );

        set_neighbor(&mut tds, c1, shared_facet_idx_u8, Some(wrong_live_neighbor)).unwrap();
        set_neighbor(&mut tds, c2, shared_facet_idx_u8, Some(c1)).unwrap();

        let repaired = repair_neighbor_pointers_local(&mut tds, &[c1], Some(&[c2]))
            .expect("local repair should replace a live neighbor across the wrong facet");

        assert_eq!(repaired, 1);
        let simplex = tds.simplex(c1).unwrap();
        assert_eq!(simplex.neighbor_key(shared_facet_idx).flatten(), Some(c2));
    }

    #[test]
    fn test_repair_neighbor_pointers_local_does_not_scan_unseeded_simplices() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.1]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.5, 0.35]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();
        let (simplex_key, facet_idx, _neighbor_key, _) =
            first_neighbor_pair(tds).expect("test triangulation should have adjacent simplices");

        set_neighbor(tds, simplex_key, facet_idx, None).unwrap();
        let repaired = repair_neighbor_pointers_local(tds, &[], None)
            .expect("local repair should ignore unseeded damage");

        assert_eq!(repaired, 0);
        let simplex = tds.simplex(simplex_key).unwrap();
        assert_eq!(simplex.neighbor_key(usize::from(facet_idx)).flatten(), None);

        // The global repair still sees the missing slot, proving the local path was scoped.
        assert!(repair_neighbor_pointers(tds).unwrap() > 0);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_extend_hull_adds_simplices_for_exterior_vertex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let kernel = FastKernel::<f64>::new();
        let p = Point::from_validated_coords([2.0, 2.0]);
        let new_vkey = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([2.0, 2.0]).unwrap(),
            )
            .unwrap();

        let new_simplices = extend_hull(tds, &kernel, new_vkey, &p).unwrap();
        assert!(!new_simplices.is_empty());
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_extend_hull_errors_when_no_visible_facets() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let tds = dt.tds_mut();

        let kernel = FastKernel::<f64>::new();
        let p = Point::from_validated_coords([0.25, 0.25]); // inside
        let new_vkey = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.25, 0.25]).unwrap(),
            )
            .unwrap();

        let err = extend_hull(tds, &kernel, new_vkey, &p).unwrap_err();
        assert_matches!(
            err,
            InsertionError::HullExtension {
                reason: HullExtensionReason::NoVisibleFacets
            }
        );
    }

    #[test]
    fn test_boundary_edge_split_invalid_boundary_count_is_retryable_invalid_patch() {
        let err = validate_boundary_edge_split_facet_count(1).unwrap_err();

        assert!(err.is_retryable());
        assert_matches!(
            err,
            InsertionError::HullExtension {
                reason: HullExtensionReason::InvalidPatch { details },
            } if details == "2D boundary edge split expected 2 facets, got 1"
        );
    }

    #[test]
    fn test_find_boundary_edge_split_facet_on_segment_2d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let point = Point::from_validated_coords([0.5, 0.0]); // on boundary edge

        let facet = find_boundary_edge_split_facet(dt.tds(), &point).unwrap();
        assert!(facet.is_some());
    }

    #[test]
    fn test_find_boundary_edge_split_facet_hull_vertex_is_retryable_invalid_patch() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let point = Point::from_validated_coords([0.0, 0.0]);

        let err = find_boundary_edge_split_facet(dt.tds(), &point).unwrap_err();

        assert!(err.is_retryable());
        assert_matches!(
            err,
            InsertionError::HullExtension {
                reason: HullExtensionReason::InvalidPatch { details },
            } if details == "2D boundary edge split matched multiple facets"
        );
    }

    #[test]
    fn test_find_boundary_edge_split_facet_off_segment_returns_none_2d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let point = Point::from_validated_coords([2.0, 0.0]); // collinear with an edge line, outside segment

        let facet = find_boundary_edge_split_facet(dt.tds(), &point).unwrap();
        assert!(facet.is_none());
    }

    #[test]
    fn test_find_visible_boundary_facets_inside_returns_empty_2d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([0.2, 0.2]); // inside simplex

        let visible = find_visible_boundary_facets(dt.tds(), &kernel, &point).unwrap();
        assert!(visible.is_empty());
    }

    #[test]
    fn test_find_visible_boundary_facets_outside_returns_non_empty_2d() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let dt = DelaunayTriangulation::<_, (), (), 2>::new(&vertices).unwrap();
        let kernel = FastKernel::<f64>::new();
        let point = Point::from_validated_coords([3.0, 3.0]); // clearly outside

        let visible = find_visible_boundary_facets(dt.tds(), &kernel, &point).unwrap();
        assert!(!visible.is_empty());
        assert!(visible.len() <= 3);
    }

    #[test]
    fn test_set_neighbor_errors_on_missing_simplex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let missing = SimplexKey::from(KeyData::from_ffi(u64::MAX));

        let err = set_neighbor(&mut tds, missing, 0, None).unwrap_err();
        assert_matches!(
            err,
            InsertionError::NeighborWiring {
                reason: NeighborWiringError::MissingSimplex { .. },
            }
        );
    }

    #[test]
    fn test_set_neighbor_errors_on_invalid_facet_index() {
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

        let err = set_neighbor(&mut tds, simplex_key, 3, None).unwrap_err();
        assert_matches!(
            err,
            InsertionError::NeighborWiring {
                reason: NeighborWiringError::InvalidFacetIndex {
                    facet_index: 3,
                    vertex_count: 3,
                    ..
                },
            }
        );
    }

    #[test]
    fn test_external_facets_for_boundary_empty_inputs_returns_empty() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let internal_simplices = SimplexKeyBuffer::new();
        let boundary_facets: Vec<FacetHandle> = Vec::new();

        let external =
            external_facets_for_boundary(&tds, &internal_simplices, &boundary_facets).unwrap();
        assert!(external.is_empty());
    }

    #[test]
    fn test_repair_neighbor_pointers_empty_tds_returns_zero() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let fixed = repair_neighbor_pointers(&mut tds).unwrap();
        assert_eq!(fixed, 0);
    }
}

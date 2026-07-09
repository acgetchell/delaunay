//! Generic triangulation construction helpers.
//!
//! This module owns the generic construction vocabulary for
//! [`Triangulation`](crate::prelude::triangulation::Triangulation)
//! and the initial-simplex bootstrap used before incremental insertion takes
//! over. Mutation-heavy insertion and repair orchestration remain implemented
//! with the triangulation type until they can be split into narrower modules.

use crate::core::algorithms::flips::DelaunayRepairError;
use crate::core::algorithms::incremental_insertion::{
    CavityFillingError, DelaunayRepairFailureContext, HullExtensionReason, InsertionError,
    InsertionTopologyValidationContext, NeighborWiringError, SpatialIndexConstructionFailure,
};
use crate::core::algorithms::locate::{ConflictError, LocateError};
use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::core::realization::TriangulationRealizationValidationError;
use crate::core::simplex::{Simplex, SimplexValidationError};
use crate::core::tds::{
    InvariantError, SimplexKey, Tds, TdsConstructionError, TdsError, VertexKey,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::util::PeriodicFacetKeyDerivationError;
use crate::core::validation::TriangulationValidationError;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::predicates::Orientation;
use crate::geometry::robust_predicates::robust_orientation;
use crate::geometry::traits::coordinate::{CoordinateValidationError, CoordinateValues};
use crate::topology::traits::topological_space::TopologyKind;
use crate::validation::DelaunayTriangulationValidationError;
use thiserror::Error;

/// Classifies the construction phase that failed final Levels 1–3 validation.
///
/// This context is carried by
/// [`TriangulationConstructionError::FinalTopologyValidation`] so callers can
/// distinguish ordinary construction finalization from periodic-quotient and
/// random-generation validation failures without parsing display text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinalTopologyValidationContext {
    /// Standard final validation after Euclidean construction.
    ConstructionFinalize,
    /// Final Levels 1-3 topology validation for a periodic quotient.
    PeriodicQuotientTopology,
    /// Final Levels 1-3 topology validation for a generated random triangulation.
    RandomGeneration,
}

impl std::fmt::Display for FinalTopologyValidationContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstructionFinalize => {
                f.write_str("topology validation failed after construction")
            }
            Self::PeriodicQuotientTopology => {
                f.write_str("periodic quotient failed final Levels 1-3 topology validation")
            }
            Self::RandomGeneration => {
                f.write_str("random triangulation failed final Levels 1-3 topology validation")
            }
        }
    }
}

/// Classifies the construction phase that failed final Level 5 Delaunay validation.
///
/// This context is carried by
/// [`TriangulationConstructionError::FinalDelaunayValidation`] so callers can
/// distinguish ordinary construction finalization from periodic-quotient
/// Delaunay checks without collapsing the typed source error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinalDelaunayValidationContext {
    /// Standard final Level 5 Delaunay validation after construction.
    ConstructionFinalize,
    /// Final Level 5 Delaunay validation for a periodic quotient.
    PeriodicQuotientDelaunay,
}

impl std::fmt::Display for FinalDelaunayValidationContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstructionFinalize => {
                f.write_str("Delaunay validation failed after construction")
            }
            Self::PeriodicQuotientDelaunay => {
                f.write_str("periodic quotient failed final Level 5 Delaunay validation")
            }
        }
    }
}

/// Structured reason periodic quotient facet-key derivation failed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum PeriodicQuotientFacetKeyDerivationFailure {
    /// The lifted simplex does not have the expected simplex arity (`D + 1` vertices).
    #[error("invalid lifted simplex arity: expected {expected} vertices, got {actual}")]
    InvalidLiftedSimplexArity {
        /// Expected number of lifted vertices (`D + 1`).
        expected: usize,
        /// Actual lifted vertex count provided by the caller.
        actual: usize,
    },

    /// The requested facet index exceeds the lifted-vertex count.
    #[error("facet index {facet_index} out of bounds for lifted vertex count {vertex_count}")]
    FacetIndexOutOfBounds {
        /// Requested facet index.
        facet_index: usize,
        /// Number of lifted vertices available.
        vertex_count: usize,
    },

    /// Relative periodic offset component is outside the encodable byte-delta range.
    #[error(
        "periodic offset component {component} (axis {axis}) is out of encodable range 0..=255"
    )]
    RelativeOffsetOutOfRange {
        /// Axis whose shifted component failed.
        axis: usize,
        /// Shifted component value.
        component: i16,
    },
}

impl From<PeriodicFacetKeyDerivationError> for PeriodicQuotientFacetKeyDerivationFailure {
    fn from(source: PeriodicFacetKeyDerivationError) -> Self {
        match source {
            PeriodicFacetKeyDerivationError::InvalidLiftedSimplexArity { expected, actual } => {
                Self::InvalidLiftedSimplexArity { expected, actual }
            }
            PeriodicFacetKeyDerivationError::FacetIndexOutOfBounds {
                facet_index,
                vertex_count,
            } => Self::FacetIndexOutOfBounds {
                facet_index,
                vertex_count,
            },
            PeriodicFacetKeyDerivationError::RelativeOffsetOutOfRange { axis, component } => {
                Self::RelativeOffsetOutOfRange { axis, component }
            }
        }
    }
}

/// Errors that can occur during triangulation construction.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::{
///     FastKernel, Triangulation, TriangulationConstructionError,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] TriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0]?,
/// ];
/// let result: Result<_, TriangulationConstructionError> =
///     Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);
/// assert!(result.is_ok());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TriangulationConstructionError {
    /// Lower-layer construction error in the TDS.
    #[error(transparent)]
    Tds(#[from] TdsConstructionError),

    /// Failed to create a simplex during triangulation construction.
    #[error("Failed to create simplex during construction: {message}")]
    FailedToCreateSimplex {
        /// Description of the simplex creation failure.
        message: String,
    },

    /// Failed to create a simplex while reconstructing a periodic quotient.
    #[error("Failed to create periodic quotient simplex during construction: {source}")]
    PeriodicQuotientSimplexCreation {
        /// Underlying simplex validation error.
        #[source]
        source: SimplexValidationError,
    },

    /// Periodic quotient facet-key derivation failed.
    #[error("Periodic quotient facet-key derivation failed for facet {facet_index}: {reason}")]
    PeriodicQuotientFacetKeyDerivation {
        /// Requested facet index.
        facet_index: usize,
        /// Structured derivation failure.
        #[source]
        reason: PeriodicQuotientFacetKeyDerivationFailure,
    },

    /// Cavity filling failed during incremental construction.
    #[error("Cavity filling failed during insertion: {source}")]
    InsertionCavityFilling {
        /// Underlying cavity-filling error.
        #[source]
        source: CavityFillingError,
    },

    /// Neighbor wiring failed during incremental construction.
    #[error("Neighbor wiring failed during insertion: {source}")]
    InsertionNeighborWiring {
        /// Underlying neighbor-wiring error.
        #[source]
        source: NeighborWiringError,
    },

    /// Flip-based Delaunay repair failed during incremental construction.
    #[error("Delaunay repair failed during insertion ({context}): {source}")]
    InsertionDelaunayRepair {
        /// Operational context describing the repair path that failed.
        context: DelaunayRepairFailureContext,
        /// Underlying typed repair failure.
        #[source]
        source: Box<DelaunayRepairError>,
    },

    /// Perturbation retry generated coordinates that violate point invariants.
    #[error("Perturbation retry produced invalid coordinates during insertion: {source}")]
    InsertionPerturbedCoordinateInvalid {
        /// Underlying coordinate validation failure.
        #[source]
        source: CoordinateValidationError,
    },

    /// Post-construction orientation canonicalization failed due to input geometry.
    #[error("Geometric orientation canonicalization failed after construction: {source}")]
    OrientationCanonicalizationGeometric {
        /// Typed insertion-layer source that failed during orientation canonicalization.
        #[source]
        source: Box<InsertionError>,
    },

    /// Post-construction orientation canonicalization failed due to an internal invariant.
    #[error("Internal orientation canonicalization failed after construction: {source}")]
    OrientationCanonicalizationInternal {
        /// Typed insertion-layer source that failed during orientation canonicalization.
        #[source]
        source: Box<InsertionError>,
    },

    /// Insufficient vertices to create a triangulation.
    #[error("Insufficient vertices for {dimension}D triangulation: {source}")]
    InsufficientVertices {
        /// The dimension that was attempted.
        dimension: usize,
        /// The underlying simplex validation error.
        #[source]
        source: SimplexValidationError,
    },

    /// Geometric degeneracy prevents triangulation construction.
    #[error("Geometric degeneracy encountered during construction: {message}")]
    GeometricDegeneracy {
        /// Description of the degeneracy issue.
        message: String,
    },

    /// Periodic quotient construction is not release-validated for this dimension.
    #[error(
        "Periodic image-point construction is release-validated only up to {max_validated_dimension}D; {dimension}D scalable quotient construction is tracked by issue #{tracking_issue}"
    )]
    UnsupportedPeriodicDimension {
        /// Requested triangulation dimension.
        dimension: usize,
        /// Highest dimension with release-validated periodic quotient construction.
        max_validated_dimension: usize,
        /// Tracking issue for extending periodic quotient support.
        tracking_issue: u32,
    },

    /// Periodic image-point construction was requested for an unsupported topology.
    #[error(
        "Periodic image-point construction requires periodic facet signatures, but {topology:?} topology does not support them"
    )]
    PeriodicImageUnsupportedTopology {
        /// Topology kind that does not support periodic facet signatures.
        topology: TopologyKind,
    },

    /// Periodic image-point construction could not obtain a periodic domain.
    #[error(
        "Periodic image-point construction requires a periodic domain, but {topology:?} topology does not expose one"
    )]
    PeriodicImageMissingDomain {
        /// Topology kind that did not expose a periodic domain.
        topology: TopologyKind,
    },

    /// Periodic image-point construction received too few canonical vertices.
    #[error(
        "Periodic {dimension}D triangulation requires at least {minimum_vertex_count} points, got {actual_vertex_count}"
    )]
    PeriodicImageInsufficientVertices {
        /// Requested periodic dimension.
        dimension: usize,
        /// Minimum canonical vertex count required by the construction.
        minimum_vertex_count: usize,
        /// Actual canonical vertex count provided by the caller.
        actual_vertex_count: usize,
    },

    /// Periodic image generation produced coordinates that violate point invariants.
    #[error(
        "Periodic image coordinates for canonical vertex {canonical_vertex_index} image {image_index} violated point invariants: {source}"
    )]
    PeriodicImageCoordinateValidation {
        /// Zero-based canonical vertex index.
        canonical_vertex_index: usize,
        /// Zero-based periodic image index.
        image_index: usize,
        /// Underlying coordinate validation failure.
        #[source]
        source: CoordinateValidationError,
    },

    /// Periodic image construction lost at least one canonical vertex in the expanded DT.
    #[error(
        "Periodic expanded DT is missing at least one canonical vertex out of {canonical_vertex_count}"
    )]
    PeriodicImageMissingCanonicalVertices {
        /// Number of canonical vertices that should be present in the expanded DT.
        canonical_vertex_count: usize,
    },

    /// Periodic image construction failed while canonicalizing simplex orientation.
    #[error("Periodic image construction failed to canonicalize orientation after build: {source}")]
    PeriodicImageOrientationCanonicalization {
        /// Underlying insertion/orientation failure.
        #[source]
        source: Box<InsertionError>,
    },

    /// Periodic image construction failed geometric simplex-orientation validation.
    #[error(
        "Periodic image construction failed geometric orientation validation after build: {source}"
    )]
    PeriodicImageGeometricOrientationValidation {
        /// Underlying TDS orientation validation failure.
        #[source]
        source: Box<TdsError>,
    },

    /// Periodic quotient reconstruction produced no representative simplices.
    #[error("Periodic quotient reconstruction produced no surviving representative simplices")]
    PeriodicQuotientEmptyReconstruction,

    /// Periodic quotient candidate extraction found no usable image simplices.
    #[error(
        "Periodic quotient candidate extraction found no usable image simplices among {full_simplex_count} full-image simplices for {canonical_vertex_count} canonical vertices"
    )]
    PeriodicQuotientNoCandidates {
        /// Number of simplices in the full image triangulation.
        full_simplex_count: usize,
        /// Number of canonical vertices that needed quotient coverage.
        canonical_vertex_count: usize,
    },

    /// Periodic quotient selection failed to select any candidate simplex.
    #[error(
        "Periodic quotient selection chose no candidate simplices from {candidate_count} candidates after {search_attempts} attempts"
    )]
    PeriodicQuotientSelectionEmpty {
        /// Number of candidate quotient simplices.
        candidate_count: usize,
        /// Number of deterministic search attempts.
        search_attempts: usize,
    },

    /// Periodic quotient selection left boundary facets in a 2D quotient.
    #[error(
        "Periodic quotient selection left {boundary_facet_count} boundary facets after {search_attempts} attempts"
    )]
    PeriodicQuotientSelectionBoundaryFacets {
        /// Number of unmatched boundary facets.
        boundary_facet_count: usize,
        /// Number of deterministic search attempts.
        search_attempts: usize,
        /// Number of vertices in the full image triangulation.
        full_vertex_count: usize,
        /// Number of simplices in the full image triangulation.
        full_simplex_count: usize,
        /// Number of canonical vertices that needed quotient coverage.
        canonical_vertex_count: usize,
        /// Number of candidate quotient simplices.
        candidate_count: usize,
        /// Number of candidate simplices selected by the best attempt.
        selected_simplex_count: usize,
    },

    /// Periodic quotient selection did not reach χ = 0 in 2D.
    #[error(
        "Periodic quotient selection could not reach χ = 0 in 2D; best |χ|={best_abs_chi} after {search_attempts} attempts"
    )]
    PeriodicQuotientSelectionEulerCharacteristic {
        /// Best absolute Euler-characteristic residual observed.
        best_abs_chi: i64,
        /// Number of deterministic search attempts.
        search_attempts: usize,
    },

    /// Periodic quotient selection did not cover every canonical vertex.
    #[error(
        "Periodic quotient selection covered only {covered_vertex_count} of {canonical_vertex_count} canonical vertices in {dimension}D"
    )]
    PeriodicQuotientSelectionIncompleteCoverage {
        /// Requested quotient dimension.
        dimension: usize,
        /// Number of canonical vertices covered by selected simplices.
        covered_vertex_count: usize,
        /// Number of canonical vertices that needed quotient coverage.
        canonical_vertex_count: usize,
    },

    /// Periodic quotient reconstruction over-shared one or more facets.
    #[error(
        "Periodic quotient reconstruction over-shared {overloaded_facet_count} facets across {selected_simplex_count} selected simplices"
    )]
    PeriodicQuotientOverloadedFacets {
        /// Number of periodic facet signatures with multiplicity greater than two.
        overloaded_facet_count: usize,
        /// Number of selected representative quotient simplices.
        selected_simplex_count: usize,
    },

    /// Periodic quotient reconstruction found an invalid facet multiplicity.
    #[error(
        "Periodic quotient facet signature has {occurrence_count} occurrences, expected 1 or 2"
    )]
    PeriodicQuotientFacetMultiplicity {
        /// Number of simplices/facets sharing the periodic facet signature.
        occurrence_count: usize,
    },

    /// Periodic quotient reconstruction left neighbor slots unmatched.
    #[error(
        "Periodic quotient reconstruction left {unmatched_neighbor_slots} unmatched neighbor slots"
    )]
    PeriodicQuotientUnmatchedNeighbors {
        /// Number of neighbor slots that were not paired by symbolic facet signatures.
        unmatched_neighbor_slots: usize,
    },

    /// Periodic quotient reconstruction lost a temporary neighbor-update buffer.
    #[error("Missing neighbor vector for periodic quotient simplex {simplex_key:?}")]
    PeriodicQuotientMissingNeighborVector {
        /// Quotient simplex whose neighbor vector was missing.
        simplex_key: SimplexKey,
    },

    /// Conflict-region extraction failed during incremental construction.
    #[error("Conflict region failed during insertion: {source}")]
    InsertionConflictRegion {
        /// Underlying conflict-region error.
        #[source]
        source: ConflictError,
    },

    /// Point location failed during incremental construction.
    #[error("Point location failed during insertion: {source}")]
    InsertionLocation {
        /// Underlying point-location error.
        #[source]
        source: LocateError,
    },

    /// Incremental insertion detected non-manifold topology.
    #[error(
        "Non-manifold topology during insertion: facet {facet_hash:#x} shared by {simplex_count} simplices"
    )]
    InsertionNonManifoldTopology {
        /// Hash of the over-shared facet.
        facet_hash: u64,
        /// Number of simplices sharing the facet.
        simplex_count: usize,
    },

    /// Hull extension failed during incremental construction.
    #[error("Hull extension failed during insertion: {reason}")]
    InsertionHullExtension {
        /// Structured hull-extension failure reason.
        #[source]
        reason: HullExtensionReason,
    },

    /// Level 5 Delaunay validation failed during incremental construction.
    #[error("Delaunay validation failed during insertion: {source}")]
    InsertionDelaunayValidation {
        /// Underlying Delaunay validation error.
        #[source]
        source: DelaunayTriangulationValidationError,
    },

    /// Level 4 realization validation failed during incremental construction.
    #[error("Realization validation failed during insertion: {source}")]
    InsertionRealizationValidation {
        /// Underlying realization validation error.
        #[source]
        source: TriangulationRealizationValidationError,
    },

    /// Level 3 topology validation failed during incremental construction.
    #[error("{context}: {source}")]
    InsertionTopologyValidation {
        /// High-level insertion context.
        context: InsertionTopologyValidationContext,
        /// Underlying topology validation error.
        #[source]
        source: TriangulationValidationError,
    },

    /// Local facet repair would remove more simplices than the active budget allowed.
    #[error(
        "Local facet repair removal budget exceeded during construction: would remove {attempted} simplices, maximum is {max_simplices_removed}"
    )]
    LocalRepairBudgetExceeded {
        /// Maximum simplices the repair budget allowed for removal.
        max_simplices_removed: usize,
        /// Number of simplices selected for removal.
        attempted: usize,
    },

    /// Final cumulative topology validation failed after construction.
    ///
    /// Mirrors [`InsertionTopologyValidation`](Self::InsertionTopologyValidation)
    /// for post-build checks that run after the incremental insertion phase.
    #[error("{context}: {source}")]
    FinalTopologyValidation {
        /// Finalization phase that produced the validation failure.
        context: FinalTopologyValidationContext,
        /// Underlying validation error.
        #[source]
        source: Box<InvariantError>,
    },

    /// Final Delaunay validation failed after construction.
    #[error("{context}: {source}")]
    FinalDelaunayValidation {
        /// Finalization phase that produced the validation failure.
        context: FinalDelaunayValidationContext,
        /// Underlying Delaunay validation error.
        #[source]
        source: DelaunayTriangulationValidationError,
    },

    /// Attempted to insert a vertex with coordinates that already exist.
    #[error(
        "Duplicate coordinates: vertex with coordinates {coordinates} already exists in the triangulation"
    )]
    DuplicateCoordinates {
        /// Duplicate coordinate tuple stored as typed coordinate payloads.
        coordinates: CoordinateValues,
    },

    /// Spatial index construction failed during triangulation construction.
    #[error("Spatial index construction failed during construction: {reason}")]
    SpatialIndexConstruction {
        /// Structured spatial-index construction failure.
        #[source]
        reason: SpatialIndexConstructionFailure,
    },

    /// Internal bookkeeping state became inconsistent during construction.
    ///
    /// This indicates a bug in the construction algorithm rather than invalid
    /// input or geometric degeneracy.
    #[error("Internal inconsistency during construction: {message}")]
    InternalInconsistency {
        /// Description of the inconsistency.
        message: String,
    },
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Build initial D-simplex from D+1 vertices with degeneracy validation.
    ///
    /// This creates a TDS with a single simplex containing all D+1 vertices,
    /// with explicit boundary neighbor slots. The simplex is validated to
    /// ensure it is non-degenerate (vertices span full D-dimensional space).
    ///
    /// **Design Note**: This method uses [`robust_orientation`] directly \[1]
    /// (Shewchuk robust predicates; see `REFERENCES.md`) for
    /// the non-degeneracy check, bypassing the kernel. This avoids `SoS`
    /// tie-breaking that would mask truly degenerate input and keeps the
    /// method independent of kernel state.
    ///
    /// # Arguments
    ///
    /// - `vertices`: Exactly D+1 vertices to form the initial simplex
    ///
    /// # Returns
    ///
    /// A TDS containing one D-simplex with all vertices, ready for incremental insertion.
    ///
    /// # Errors
    ///
    /// Returns an error if the vertex count is not exactly D+1, if the
    /// vertices are geometrically degenerate, if vertex/simplex insertion
    /// fails, or if duplicate UUIDs are detected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::{FastKernel, Triangulation};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::prelude::triangulation::TriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let tds = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices)?;
    /// assert_eq!(tds.number_of_vertices(), 3);
    /// assert_eq!(tds.number_of_simplices(), 1);
    /// assert_eq!(tds.dim(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build_initial_simplex(
        vertices: &[Vertex<U, D>],
    ) -> Result<Tds<U, V, D>, TriangulationConstructionError> {
        if vertices.len() != D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: SimplexValidationError::InsufficientVertices {
                    actual: vertices.len(),
                    expected: D + 1,
                    dimension: D,
                },
            });
        }

        for vertex in vertices {
            vertex.is_valid().map_err(|source| {
                TriangulationConstructionError::Tds(TdsConstructionError::ValidationError(
                    TdsError::InvalidVertex {
                        vertex_id: vertex.uuid(),
                        source,
                    },
                ))
            })?;
        }

        let points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            vertices.iter().map(|v| *v.point()).collect();

        let exact_orientation = robust_orientation(&points[..]).map_err(|e| {
            TriangulationConstructionError::FailedToCreateSimplex {
                message: format!("Exact orientation test failed: {e}"),
            }
        })?;

        if matches!(exact_orientation, Orientation::DEGENERATE) {
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

        let orientation = match exact_orientation {
            Orientation::POSITIVE => 1,
            Orientation::NEGATIVE => -1,
            Orientation::DEGENERATE => {
                return Err(TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Degenerate initial simplex in {D}D (unreachable)"),
                });
            }
        };

        let mut tds = Tds::empty();
        let mut vertex_keys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        for vertex in vertices {
            let vkey = tds.insert_vertex_with_mapping(*vertex)?;
            vertex_keys.push(vkey);
        }

        if orientation < 0 {
            if vertex_keys.len() >= 2 {
                vertex_keys.swap(0, 1);
            } else {
                return Err(TriangulationConstructionError::FailedToCreateSimplex {
                    message: format!(
                        "Cannot canonicalize orientation for {}D simplex with {} vertex key(s)",
                        D,
                        vertex_keys.len(),
                    ),
                });
            }
        }

        let simplex = Simplex::try_new(vertex_keys).map_err(|e| {
            TriangulationConstructionError::FailedToCreateSimplex {
                message: format!("Failed to create initial simplex: {e}"),
            }
        })?;

        let _simplex_key = tds.insert_simplex_with_mapping(simplex)?;

        tds.assign_neighbors()
            .map_err(TdsConstructionError::ValidationError)?;
        tds.assign_incident_simplices()
            .map_err(|e| TdsConstructionError::ValidationError(e.into()))?;

        Ok(tds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::simplex::NeighborSlot;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::traits::coordinate::{CoordinateValidationError, InvalidCoordinateValue};
    use crate::vertex;
    use std::assert_matches;

    #[test]
    fn internal_inconsistency_display() {
        let err = TriangulationConstructionError::InternalInconsistency {
            message: "missing vertex in lookup table".to_string(),
        };

        assert_eq!(
            err.to_string(),
            "Internal inconsistency during construction: missing vertex in lookup table"
        );
    }

    #[test]
    fn final_topology_validation_context_display() {
        let cases = [
            (
                FinalTopologyValidationContext::ConstructionFinalize,
                "topology validation failed after construction",
            ),
            (
                FinalTopologyValidationContext::PeriodicQuotientTopology,
                "periodic quotient failed final Levels 1-3 topology validation",
            ),
            (
                FinalTopologyValidationContext::RandomGeneration,
                "random triangulation failed final Levels 1-3 topology validation",
            ),
        ];

        for (context, expected) in cases {
            assert_eq!(context.to_string(), expected);
        }
    }

    #[test]
    fn final_delaunay_validation_context_display() {
        assert_eq!(
            FinalDelaunayValidationContext::ConstructionFinalize.to_string(),
            "Delaunay validation failed after construction"
        );
        assert_eq!(
            FinalDelaunayValidationContext::PeriodicQuotientDelaunay.to_string(),
            "periodic quotient failed final Level 5 Delaunay validation"
        );
    }

    #[test]
    fn insertion_hull_extension_exposes_typed_source() {
        let reason = HullExtensionReason::Tds(TdsError::InconsistentDataStructure {
            message: "missing boundary facet".to_string(),
        });
        let error = TriangulationConstructionError::InsertionHullExtension { reason };
        let source = std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<HullExtensionReason>());
        assert_matches!(source, Some(HullExtensionReason::Tds(_)));
    }

    #[test]
    fn insufficient_vertices_exposes_typed_source() {
        let error = TriangulationConstructionError::InsufficientVertices {
            dimension: 3,
            source: SimplexValidationError::InsufficientVertices {
                actual: 3,
                expected: 4,
                dimension: 3,
            },
        };

        let source = std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<SimplexValidationError>());
        assert_matches!(
            source,
            Some(SimplexValidationError::InsufficientVertices {
                actual: 3,
                expected: 4,
                dimension: 3,
            })
        );
    }

    macro_rules! test_build_initial_simplex {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<build_initial_simplex_ $dim d>]() {
                    let vertices: Vec<Vertex<(), $dim>> = vec![
                        $(vertex!($simplex_coords).unwrap()),+
                    ];

                    let expected_vertices = vertices.len();
                    assert_eq!(expected_vertices, $dim + 1);

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();

                    assert_eq!(tds.number_of_vertices(), expected_vertices);
                    assert_eq!(tds.number_of_simplices(), 1);
                    assert_eq!(tds.dim(), $dim as i32);
                    assert_eq!(tds.vertices().count(), expected_vertices);

                    let (_, simplex) = tds.simplices().next()
                        .expect("initial simplex should exist");
                    assert_eq!(simplex.number_of_vertices(), expected_vertices);

                    for (_, vertex) in tds.vertices() {
                        assert!(vertex.incident_simplex().is_some());
                    }

                    let neighbors = simplex
                        .neighbor_slots()
                        .expect("initial simplex should assign boundary neighbor slots");
                    assert!(neighbors.iter().all(|slot| *slot == NeighborSlot::Boundary));
                }
            }
        };
    }

    test_build_initial_simplex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    test_build_initial_simplex!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
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

    #[test]
    fn build_initial_simplex_insufficient_vertices() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert_matches!(
            result,
            Err(TriangulationConstructionError::InsufficientVertices { dimension: 3, .. })
        );
    }

    #[test]
    fn build_initial_simplex_too_many_vertices() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5]).unwrap(),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert_matches!(
            result,
            Err(TriangulationConstructionError::InsufficientVertices { .. })
        );
    }

    macro_rules! test_build_initial_simplex_rejects_non_finite_vertex_coordinate_dimensions {
        ($($dim:expr),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<build_initial_simplex_rejects_non_finite_vertex_coordinate_ $dim d>]() {
                        let mut invalid_coords = [0.0_f64; $dim];
                        invalid_coords[0] = 1.0;
                        invalid_coords[1] = f64::NAN;

                        assert_matches!(
                            Point::<$dim>::try_new(invalid_coords),
                            Err(CoordinateValidationError::InvalidCoordinate {
                                coordinate_index: 1,
                                coordinate_value: InvalidCoordinateValue::Nan,
                                dimension: $dim,
                            })
                        );
                    }
                )+
            }
        };
    }

    test_build_initial_simplex_rejects_non_finite_vertex_coordinate_dimensions!(2, 3, 4, 5);

    #[test]
    fn build_initial_simplex_with_user_data() {
        let v1 = vertex!([0.0, 0.0]; data = 42_usize).unwrap();
        let v2 = vertex!([1.0, 0.0]; data = 43_usize).unwrap();
        let v3 = vertex!([0.0, 1.0]; data = 44_usize).unwrap();

        let vertices = vec![v1, v2, v3];
        let tds = Triangulation::<FastKernel<f64>, usize, (), 2>::build_initial_simplex(&vertices)
            .unwrap();

        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.number_of_simplices(), 1);

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

    #[test]
    fn build_initial_simplex_rejects_collinear_2d() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([2.0, 0.0]).unwrap(),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert_matches!(
            result,
            Err(TriangulationConstructionError::GeometricDegeneracy { .. })
        );
    }

    #[test]
    fn build_initial_simplex_rejects_coplanar_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.5, 0.5, 0.0]).unwrap(),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert_matches!(
            result,
            Err(TriangulationConstructionError::GeometricDegeneracy { .. })
        );
    }
}

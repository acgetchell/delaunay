//! TDS error and validation-report vocabulary.
//!
//! This module keeps the public error vocabulary separate from the storage and
//! mutation implementation.

use super::{SimplexKey, VertexKey};
use crate::core::algorithms::flips::{FlipError, FlipNeighborWiringError};
use crate::core::facet::FacetError;
use crate::core::realization::TriangulationRealizationValidationError;
use crate::core::simplex::SimplexValidationError;
use crate::core::validation::TriangulationValidationError;
use crate::core::vertex::VertexValidationError;
use crate::validation::DelaunayTriangulationValidationError;
use thiserror::Error;
use uuid::Uuid;

// =============================================================================
// CONSTRUCTION STATE TYPES
// =============================================================================

/// Represents the construction state of a triangulation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::TriangulationConstructionState;
///
/// let state = TriangulationConstructionState::Incomplete(2);
/// std::assert_matches!(state, TriangulationConstructionState::Incomplete(2));
///
/// let default_state = TriangulationConstructionState::default();
/// std::assert_matches!(
///     default_state,
///     TriangulationConstructionState::Incomplete(0)
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriangulationConstructionState {
    /// The triangulation has insufficient vertices to form a complete D-dimensional triangulation.
    /// Contains the number of vertices currently stored.
    Incomplete(usize),
    /// The triangulation is complete and valid with at least D+1 vertices and proper simplex structure.
    Constructed,
}

impl Default for TriangulationConstructionState {
    fn default() -> Self {
        Self::Incomplete(0)
    }
}

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur during TDS construction operations.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::{EntityKind, TdsConstructionError};
/// use uuid::Uuid;
///
/// let err = TdsConstructionError::DuplicateUuid {
///     entity: EntityKind::Vertex,
///     uuid: Uuid::nil(),
/// };
/// std::assert_matches!(err, TdsConstructionError::DuplicateUuid { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TdsConstructionError {
    /// Validation error during construction.
    #[error("Validation error during construction: {0}")]
    ValidationError(#[from] TdsError),
    /// Attempted to insert an entity with a UUID that already exists.
    #[error("Duplicate UUID: {entity:?} with UUID {uuid} already exists")]
    DuplicateUuid {
        /// The type of entity.
        entity: EntityKind,
        /// The UUID that was duplicated.
        uuid: Uuid,
    },
}

/// Represents the type of entity in the triangulation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::EntityKind;
///
/// let kind = EntityKind::Simplex;
/// assert_eq!(kind, EntityKind::Simplex);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityKind {
    /// A vertex entity.
    Vertex,
    /// A simplex entity.
    Simplex,
}

/// Geometric orientation/predicate errors.
///
/// These errors indicate floating-point or geometric degeneracy issues
/// (e.g., nearly coplanar input producing a zero or negative determinant)
/// rather than internal data structure bugs. They are retryable via
/// coordinate perturbation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::GeometricError;
///
/// let err = GeometricError::DegenerateOrientation {
///     message: "det=0".to_string(),
/// };
/// std::assert_matches!(err, GeometricError::DegenerateOrientation { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeometricError {
    /// Geometric orientation degeneracy detected during orientation canonicalization.
    ///
    /// This indicates a geometry-related issue (e.g., nearly coplanar input points
    /// producing a zero determinant, or a kernel predicate evaluation failure)
    /// rather than an internal data structure bug.
    #[error("Degenerate geometric orientation: {message}")]
    DegenerateOrientation {
        /// Description of the degeneracy.
        message: String,
    },
    /// Negative geometric orientation detected after canonicalization.
    ///
    /// A simplex has `det < 0` even after orientation canonicalization passes.  This
    /// typically indicates floating-point sign instability for near-degenerate input
    /// (the fast kernel gives inconsistent sign results across calls) rather than a
    /// data-structure corruption bug.
    #[error("Negative geometric orientation: {message}")]
    NegativeOrientation {
        /// Description of the negative-orientation condition.
        message: String,
    },
}

/// Example usage
///
/// ```
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Insertion(#[from] delaunay::prelude::insertion::InsertionError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)]
/// #     TdsConstruction(#[from] delaunay::prelude::tds::TdsConstructionError),
/// #     #[error(transparent)]
/// #     Invariant(#[from] delaunay::prelude::tds::InvariantError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     Simplex(#[from] delaunay::prelude::tds::SimplexValidationError),
/// #     #[error(transparent)]
/// #     Validation(#[from] delaunay::DelaunayTriangulationValidationError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Build a simple 3D triangulation
/// let vertices = [
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
/// assert_eq!(dt.number_of_vertices(), 4);
/// assert!(dt.is_valid_delaunay().is_ok());
/// # Ok(())
/// # }
/// ```
///
/// Which side of a neighbor relationship is missing a shared-facet vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SharedFacetMismatchSide {
    /// The source simplex's facet vertex is missing from the neighbor.
    SourceFacet,
    /// The neighbor simplex's facet vertex is missing from the source simplex.
    NeighborFacet,
}

/// Structured reason why neighbor relationships failed validation.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum NeighborValidationError {
    /// A neighbor buffer has the wrong arity for the triangulation dimension.
    #[error("Neighbor vector length {actual} != expected {expected} during {context}")]
    LengthMismatch {
        /// Observed number of neighbor slots.
        actual: usize,
        /// Expected number of neighbor slots.
        expected: usize,
        /// Validation context.
        context: String,
    },
    /// A neighbor buffer contains an unassigned facet slot.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) has unassigned neighbor slot at facet {facet_index} during {context}"
    )]
    UnassignedNeighborSlot {
        /// Simplex containing the unassigned slot.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the unassigned slot.
        simplex_uuid: Uuid,
        /// Facet slot that has not been assigned as boundary or neighbor.
        facet_index: usize,
        /// Validation context.
        context: String,
    },
    /// A non-periodic simplex points to itself as a neighbor.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) has non-periodic self-neighbor at facet {facet_index}"
    )]
    NonPeriodicSelfNeighbor {
        /// Simplex whose neighbor pointer references itself.
        simplex_key: SimplexKey,
        /// UUID of the simplex whose neighbor pointer references itself.
        simplex_uuid: Uuid,
        /// Facet slot containing the self-neighbor.
        facet_index: usize,
    },
    /// A neighbor pointer references a missing simplex key.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} references missing neighbor {neighbor_key:?} during {context}"
    )]
    MissingNeighborSimplex {
        /// Simplex containing the stale neighbor pointer.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the stale neighbor pointer.
        simplex_uuid: Uuid,
        /// Facet slot containing the stale neighbor pointer.
        facet_index: usize,
        /// Missing neighbor key.
        neighbor_key: SimplexKey,
        /// Validation context.
        context: String,
    },
    /// A neighbor pointer references a simplex removed by the current local edit.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} references removed neighbor {neighbor_key:?}"
    )]
    ReferencedRemovedNeighbor {
        /// Simplex containing the stale local-edit neighbor pointer.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the stale local-edit neighbor pointer.
        simplex_uuid: Uuid,
        /// Facet slot containing the removed neighbor pointer.
        facet_index: usize,
        /// Removed neighbor key.
        neighbor_key: SimplexKey,
    },
    /// A neighbor pair does not share exactly the facet opposite the slot.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} shares {shared_count} vertices with neighbor, expected {expected}"
    )]
    SharedVertexCountMismatch {
        /// Simplex containing the invalid neighbor pointer.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the invalid neighbor pointer.
        simplex_uuid: Uuid,
        /// Facet slot being checked.
        facet_index: usize,
        /// Number of shared vertices observed.
        shared_count: usize,
        /// Expected number of shared vertices.
        expected: usize,
    },
    /// A neighbor is opposite a different vertex slot than the pointer position.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) neighbor at facet {facet_index} is opposite {observed_opposite:?}, expected {expected_opposite}"
    )]
    OppositeVertexMismatch {
        /// Simplex containing the invalid neighbor pointer.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the invalid neighbor pointer.
        simplex_uuid: Uuid,
        /// Facet slot being checked.
        facet_index: usize,
        /// Observed opposite vertex slot.
        observed_opposite: Option<usize>,
        /// Expected opposite vertex slot.
        expected_opposite: usize,
    },
    /// The facet incidence map is missing a facet implied by a simplex.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} key {facet_key} is missing from facet incidence"
    )]
    FacetIncidenceMissing {
        /// Simplex whose facet was missing from the incidence map.
        simplex_key: SimplexKey,
        /// UUID of the simplex whose facet was missing from the incidence map.
        simplex_uuid: Uuid,
        /// Facet index.
        facet_index: usize,
        /// Canonical facet key.
        facet_key: u64,
    },
    /// A facet incidence entry exists but does not include the edited simplex/facet.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} key {facet_key} does not reference the edited facet"
    )]
    FacetIncidenceDoesNotReferenceSimplex {
        /// Simplex being edited.
        simplex_key: SimplexKey,
        /// UUID of the simplex being edited.
        simplex_uuid: Uuid,
        /// Facet index being edited.
        facet_index: usize,
        /// Canonical facet key.
        facet_key: u64,
    },
    /// A facet incidence entry has non-manifold multiplicity.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} key {facet_key} is shared by {simplex_count} simplices"
    )]
    FacetIncidenceMultiplicity {
        /// Simplex being edited.
        simplex_key: SimplexKey,
        /// UUID of the simplex being edited.
        simplex_uuid: Uuid,
        /// Facet index being edited.
        facet_index: usize,
        /// Canonical facet key.
        facet_key: u64,
        /// Number of simplices incident to the facet.
        simplex_count: usize,
    },
    /// A proposed neighbor pointer disagrees with facet incidence.
    #[error(
        "Simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index} proposed neighbor {proposed_neighbor:?} does not match expected {expected_neighbor:?}"
    )]
    NeighborIncidenceMismatch {
        /// Simplex being edited.
        simplex_key: SimplexKey,
        /// UUID of the simplex being edited.
        simplex_uuid: Uuid,
        /// Facet index being edited.
        facet_index: usize,
        /// Proposed neighbor pointer.
        proposed_neighbor: Option<SimplexKey>,
        /// Neighbor expected from facet incidence.
        expected_neighbor: Option<SimplexKey>,
    },
    /// A facet slot index is outside the neighbor buffer.
    #[error("Neighbor facet index {facet_index} out of bounds for {slot_count} neighbor slots")]
    NeighborSlotOutOfBounds {
        /// Invalid facet index.
        facet_index: usize,
        /// Number of available neighbor slots.
        slot_count: usize,
    },
    /// The mirror facet could not be found between adjacent simplices.
    #[error(
        "Could not determine mirror facet during {context}: simplex {simplex_uuid}[{facet_index}] -> neighbor {neighbor_uuid}"
    )]
    MirrorFacetMissing {
        /// UUID of the source simplex.
        simplex_uuid: Uuid,
        /// Facet index in the source simplex.
        facet_index: usize,
        /// UUID of the neighbor simplex.
        neighbor_uuid: Uuid,
        /// Validation context.
        context: String,
    },
    /// Shared-vertex analysis found more than one possible mirror facet.
    #[error(
        "Mirror facet is ambiguous: simplex {simplex_uuid} and neighbor {neighbor_uuid} differ by more than one vertex"
    )]
    MirrorFacetAmbiguous {
        /// UUID of the source simplex.
        simplex_uuid: Uuid,
        /// UUID of the neighbor simplex.
        neighbor_uuid: Uuid,
    },
    /// Shared-vertex analysis found that two simplices share every vertex.
    #[error(
        "Mirror facet could not be determined: simplex {simplex_uuid} and neighbor {neighbor_uuid} share all vertices"
    )]
    MirrorFacetDuplicateSimplices {
        /// UUID of the source simplex.
        simplex_uuid: Uuid,
        /// UUID of the neighbor simplex.
        neighbor_uuid: Uuid,
    },
    /// A computed mirror facet disagrees with shared-vertex analysis.
    #[error(
        "Mirror facet index mismatch: simplex {simplex_uuid}[{facet_index}] -> neighbor {neighbor_uuid}; observed {observed_mirror_index}, expected {expected_mirror_index}"
    )]
    MirrorFacetIndexMismatch {
        /// UUID of the source simplex.
        simplex_uuid: Uuid,
        /// Facet index in the source simplex.
        facet_index: usize,
        /// UUID of the neighbor simplex.
        neighbor_uuid: Uuid,
        /// Mirror index returned by simplex logic.
        observed_mirror_index: usize,
        /// Mirror index implied by shared-vertex analysis.
        expected_mirror_index: usize,
    },
    /// A shared facet is missing a vertex on one side of a neighbor pair.
    #[error(
        "Shared facet mismatch ({side:?}): simplex {simplex_uuid}[{facet_index}] and neighbor {neighbor_uuid}[{mirror_index}] are missing vertex {missing_vertex:?}"
    )]
    SharedFacetMissingVertex {
        /// Which side exposed the missing vertex.
        side: SharedFacetMismatchSide,
        /// UUID of the source simplex.
        simplex_uuid: Uuid,
        /// Facet index in the source simplex.
        facet_index: usize,
        /// UUID of the neighbor simplex.
        neighbor_uuid: Uuid,
        /// Mirror facet index in the neighbor simplex.
        mirror_index: usize,
        /// Missing vertex key.
        missing_vertex: VertexKey,
    },
    /// A neighbor does not carry the required reciprocal pointer.
    #[error(
        "Neighbor back-reference mismatch during {context}: simplex {simplex_uuid}[{facet_index}] -> {neighbor_key:?} should be mirrored by {neighbor_uuid}[{mirror_index}] -> {simplex_key:?}, found {observed:?}"
    )]
    BackReferenceMismatch {
        /// Source simplex key.
        simplex_key: SimplexKey,
        /// Source simplex UUID.
        simplex_uuid: Uuid,
        /// Source facet index.
        facet_index: usize,
        /// Neighbor simplex key.
        neighbor_key: SimplexKey,
        /// Neighbor simplex UUID.
        neighbor_uuid: Uuid,
        /// Mirror facet index in the neighbor.
        mirror_index: usize,
        /// Observed back-reference, or `None` if absent.
        observed: Option<SimplexKey>,
        /// Validation context.
        context: String,
    },
    /// A reciprocal update would overwrite another back-reference.
    #[error(
        "Neighbor simplex {neighbor_uuid}[{mirror_index}] already references {existing_back_ref:?}; refusing to overwrite with {requested_back_ref:?}"
    )]
    ExistingBackReferenceConflict {
        /// Neighbor UUID.
        neighbor_uuid: Uuid,
        /// Mirror facet index in the neighbor.
        mirror_index: usize,
        /// Existing back-reference.
        existing_back_ref: SimplexKey,
        /// Requested back-reference.
        requested_back_ref: SimplexKey,
    },
    /// A boundary facet has a neighbor pointer.
    #[error(
        "Boundary facet {facet_key} unexpectedly has neighbor {neighbor_key:?} across simplex {simplex_uuid}[{facet_index}]"
    )]
    BoundaryFacetHasNeighbor {
        /// Boundary facet key.
        facet_key: u64,
        /// Simplex containing the boundary facet.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the boundary facet.
        simplex_uuid: Uuid,
        /// Facet index in the simplex.
        facet_index: usize,
        /// Unexpected neighbor key.
        neighbor_key: SimplexKey,
    },
    /// A boundary facet has inadmissible self-adjacency.
    #[error(
        "Boundary facet {facet_key} has non-periodic self-neighbor across simplex {simplex_uuid}[{facet_index}]"
    )]
    BoundaryFacetHasNonPeriodicSelfNeighbor {
        /// Boundary facet key.
        facet_key: u64,
        /// Simplex containing the boundary facet.
        simplex_key: SimplexKey,
        /// UUID of the simplex containing the boundary facet.
        simplex_uuid: Uuid,
        /// Facet index in the simplex.
        facet_index: usize,
    },
    /// An interior facet's two incident simplices do not point to each other.
    #[error(
        "Interior facet {facet_key} has inconsistent neighbor pointers: {first_simplex_uuid}[{first_facet_index}] -> {first_neighbor:?}, {second_simplex_uuid}[{second_facet_index}] -> {second_neighbor:?}"
    )]
    InteriorFacetNeighborMismatch {
        /// Interior facet key.
        facet_key: u64,
        /// First incident simplex key.
        first_simplex_key: SimplexKey,
        /// First incident simplex UUID.
        first_simplex_uuid: Uuid,
        /// Facet index in the first simplex.
        first_facet_index: usize,
        /// Neighbor pointer observed in the first simplex.
        first_neighbor: Option<SimplexKey>,
        /// Second incident simplex key.
        second_simplex_key: SimplexKey,
        /// Second incident simplex UUID.
        second_simplex_uuid: Uuid,
        /// Facet index in the second simplex.
        second_facet_index: usize,
        /// Neighbor pointer observed in the second simplex.
        second_neighbor: Option<SimplexKey>,
    },
    /// A facet's vertex order could not be built for neighbor validation.
    #[error(
        "Could not build facet order during {context}: simplex {simplex_uuid} (key {simplex_key:?}) facet {facet_index}: {source}"
    )]
    FacetOrderUnavailable {
        /// Simplex whose facet order could not be built.
        simplex_key: SimplexKey,
        /// UUID of the simplex whose facet order could not be built.
        simplex_uuid: Uuid,
        /// Facet index whose order could not be built.
        facet_index: usize,
        /// Validation context.
        context: String,
        /// Underlying [`FlipError`] raised while deriving the facet order.
        #[source]
        source: Box<FlipError>,
    },
    /// Bistellar flip neighbor wiring failed while preserving TDS invariants.
    #[error("Flip neighbor wiring failed: {reason}")]
    FlipNeighborWiring {
        /// Structured flip wiring failure.
        #[source]
        reason: Box<FlipNeighborWiringError>,
    },
}

/// Errors that can occur during triangulation validation (post-construction).
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TdsError {
    /// The triangulation contains an invalid vertex.
    #[error("Invalid vertex {vertex_id}: {source}")]
    InvalidVertex {
        /// The UUID of the invalid vertex.
        vertex_id: Uuid,
        /// The underlying vertex validation error.
        source: VertexValidationError,
    },
    /// The triangulation contains an invalid simplex.
    #[error("Invalid simplex {simplex_id}: {source}")]
    InvalidSimplex {
        /// The UUID of the invalid simplex.
        simplex_id: Uuid,
        /// The underlying simplex validation error.
        source: SimplexValidationError,
    },
    /// Neighbor relationships are invalid.
    #[error("Invalid neighbor relationships: {reason}")]
    InvalidNeighbors {
        /// Structured neighbor validation failure.
        #[source]
        reason: NeighborValidationError,
    },
    /// Coherent orientation invariant violated between adjacent simplices.
    #[error(
        "Orientation invariant violated between simplices {simplex1_uuid} and {simplex2_uuid}; shared facet orderings {facet_vertices:?} vs {simplex2_facet_vertices:?} (simplex1 facet index {simplex1_facet_index}, simplex2 facet index {simplex2_facet_index}, observed_odd_permutation={observed_odd_permutation}, expected_odd_permutation={expected_odd_permutation})"
    )]
    OrientationViolation {
        /// Key of the first simplex.
        simplex1_key: SimplexKey,
        /// UUID of the first simplex.
        simplex1_uuid: Uuid,
        /// Key of the second simplex.
        simplex2_key: SimplexKey,
        /// UUID of the second simplex.
        simplex2_uuid: Uuid,
        /// Facet index in the first simplex.
        simplex1_facet_index: usize,
        /// Facet index in the second simplex.
        simplex2_facet_index: usize,
        /// Vertex keys of the shared facet in `simplex1` ordering (excluding `simplex1_facet_index`).
        facet_vertices: Vec<VertexKey>,
        /// Vertex keys of the shared facet in `simplex2` ordering (excluding `simplex2_facet_index`).
        simplex2_facet_vertices: Vec<VertexKey>,
        /// Observed parity of the permutation from `facet_vertices` to `simplex2_facet_vertices`.
        observed_odd_permutation: bool,
        /// Expected odd-permutation parity under the coherent boundary-orientation convention.
        expected_odd_permutation: bool,
    },
    /// The triangulation contains duplicate simplices.
    #[error("Duplicate simplices detected: {message}")]
    DuplicateSimplices {
        /// Description of the duplicate simplex validation failure.
        message: String,
    },
    /// Explicit input contains duplicate maximal simplices.
    #[error(
        "Duplicate explicit simplices at input indices {existing_simplex_index} and {duplicate_simplex_index} with input vertex indices {vertex_indices:?}"
    )]
    #[non_exhaustive]
    DuplicateExplicitSimplices {
        /// Index of the first simplex specification with this canonical vertex set.
        existing_simplex_index: usize,
        /// Index of the duplicate simplex specification.
        duplicate_simplex_index: usize,
        /// Canonical sorted input vertex-index identity shared by both simplex specifications.
        vertex_indices: Vec<usize>,
    },
    /// A simplex insertion or validation pass found a facet incident to too many simplices.
    ///
    /// During insertion preflight, the `candidate_*` fields identify the simplex that
    /// would exceed the PL-manifold facet multiplicity. During post-hoc
    /// validation, they identify one offending incident simplex from the over-shared
    /// facet.
    #[error(
        "Facet {facet_key} exceeds incident-simplex limit: observed {attempted_incident_count} incident simplices, max {max_incident_count}; candidate/offending simplex {candidate_simplex_uuid} facet {candidate_facet_index}; other incident simplices {existing_incident_count}"
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
        candidate_simplex_uuid: Uuid,
        /// Facet index on the candidate or offending simplex.
        candidate_facet_index: usize,
    },
    /// Explicit input contains a facet incident to too many maximal simplices.
    #[error(
        "Explicit facet {facet_key} with input vertex indices {facet_vertex_indices:?} exceeds incident-simplex limit: input simplex {candidate_simplex_index} would make {attempted_incident_count} incident simplices, max {max_incident_count}; candidate facet {candidate_facet_index}; existing incident simplices {existing_incident_count}"
    )]
    #[non_exhaustive]
    ExplicitFacetSharingViolation {
        /// Canonical key of the over-shared facet.
        facet_key: u64,
        /// Canonical sorted input vertex indices defining the over-shared facet.
        facet_vertex_indices: Vec<usize>,
        /// Number of pre-existing explicit simplices already incident to the facet.
        existing_incident_count: usize,
        /// Number of incident simplices that would exist after the candidate input simplex.
        attempted_incident_count: usize,
        /// Maximum allowed number of incident simplices for a PL-manifold facet.
        max_incident_count: usize,
        /// Input index of the simplex that would exceed facet multiplicity.
        candidate_simplex_index: usize,
        /// Facet index on the candidate input simplex.
        candidate_facet_index: usize,
    },
    /// Failed to create a simplex during triangulation.
    #[error("Failed to create simplex: {message}")]
    FailedToCreateSimplex {
        /// Description of the simplex creation failure.
        message: String,
    },
    /// Simplices are not neighbors as expected
    #[error("Simplices {simplex1:?} and {simplex2:?} are not neighbors")]
    NotNeighbors {
        /// The first simplex UUID.
        simplex1: Uuid,
        /// The second simplex UUID.
        simplex2: Uuid,
    },
    /// Entity mapping inconsistency (vertex or simplex).
    #[error("{entity:?} mapping inconsistency: {message}")]
    MappingInconsistency {
        /// The type of entity with the mapping issue.
        entity: EntityKind,
        /// Description of the mapping inconsistency.
        message: String,
    },
    /// Failed to retrieve vertex keys for a simplex during neighbor assignment.
    #[error("Failed to retrieve vertex keys for simplex {simplex_id}: {message}")]
    VertexKeyRetrievalFailed {
        /// The UUID of the simplex that failed.
        simplex_id: Uuid,
        /// Description of the failure.
        message: String,
    },
    /// A simplex key was expected in storage but not found.
    ///
    /// This typically indicates a dangling simplex reference or stale key
    /// after topology mutations (simplex removal, cavity filling, etc.).
    #[error("Simplex key {simplex_key:?} not found: {context}")]
    SimplexNotFound {
        /// The simplex key that was not found in storage.
        simplex_key: SimplexKey,
        /// Description of the context where the lookup failed.
        context: String,
    },
    /// A vertex key was expected in storage but not found.
    ///
    /// This typically indicates a dangling vertex reference or stale key
    /// after topology mutations.
    #[error("Vertex key {vertex_key:?} not found: {context}")]
    VertexNotFound {
        /// The vertex key that was not found in storage.
        vertex_key: VertexKey,
        /// Description of the context where the lookup failed.
        context: String,
    },
    /// A dimensional invariant was violated (wrong vertex count, offset count, etc.).
    ///
    /// A simplex, facet, ridge, or link has a different number of elements than
    /// expected for the triangulation dimension `D`.
    #[error("Dimension mismatch: expected {expected}, got {actual} — {context}")]
    DimensionMismatch {
        /// The expected count.
        expected: usize,
        /// The actual count observed.
        actual: usize,
        /// Description of what was being checked.
        context: String,
    },
    /// An index exceeded the valid range for the target structure.
    #[error("Index out of bounds: index {index}, bound {bound} — {context}")]
    IndexOutOfBounds {
        /// The index that was out of bounds.
        index: usize,
        /// The exclusive upper bound.
        bound: usize,
        /// Description of what was being accessed.
        context: String,
    },
    /// Canonical vertex incidence still references a simplex that this mutation removed.
    #[error(
        "Vertex-to-simplices index still lists removed simplex {simplex_key:?} for vertex {vertex_key:?} after incidence removal"
    )]
    RemovedSimplexStillIncident {
        /// Vertex whose canonical incidence retained the removed simplex.
        vertex_key: VertexKey,
        /// Removed simplex still present in the vertex incidence set.
        simplex_key: SimplexKey,
    },
    /// Canonical vertex incidence disagrees with simplex vertex membership.
    #[error(
        "Vertex-to-simplices index lists simplex {simplex_key:?} for vertex {vertex_key:?}, but the simplex does not contain the vertex"
    )]
    VertexIncidenceMismatch {
        /// Vertex listed in the canonical incidence relation.
        vertex_key: VertexKey,
        /// Simplex listed for the vertex but missing that vertex in storage.
        simplex_key: SimplexKey,
    },
    /// Internal data structure inconsistency.
    ///
    /// This is the fallback for structural invariant violations that carry
    /// open-ended diagnostic context and do not fit a more specific variant.
    /// Prefer [`SimplexNotFound`], [`VertexNotFound`], [`DimensionMismatch`],
    /// [`IndexOutOfBounds`], [`RemovedSimplexStillIncident`], or
    /// [`VertexIncidenceMismatch`] when applicable.
    ///
    /// [`SimplexNotFound`]: TdsError::SimplexNotFound
    /// [`VertexNotFound`]: TdsError::VertexNotFound
    /// [`DimensionMismatch`]: TdsError::DimensionMismatch
    /// [`IndexOutOfBounds`]: TdsError::IndexOutOfBounds
    /// [`RemovedSimplexStillIncident`]: TdsError::RemovedSimplexStillIncident
    /// [`VertexIncidenceMismatch`]: TdsError::VertexIncidenceMismatch
    #[error("Internal data structure inconsistency: {message}")]
    InconsistentDataStructure {
        /// Description of the inconsistency.
        message: String,
    },
    /// Geometric orientation/predicate error (e.g., degenerate or negative orientation).
    ///
    /// This wraps a [`GeometricError`] and indicates a floating-point or geometric
    /// degeneracy issue rather than an internal data structure bug.
    #[error(transparent)]
    Geometric(#[from] GeometricError),

    /// Facet operation failed during validation.
    #[error("Facet operation failed: {0}")]
    FacetError(#[from] FacetError),
    /// A simplex contains two or more vertices with identical coordinates.
    ///
    /// This is distinct from [`SimplexValidationError::DuplicateVertices`] which checks
    /// for duplicate vertex *keys*. This variant detects the case where different
    /// vertex keys reference geometrically identical points — producing a zero-volume
    /// simplex that is catastrophic for `SoS` and Pachner moves.
    #[error("Duplicate coordinates in simplex {simplex_id}: {message}")]
    DuplicateCoordinatesInSimplex {
        /// UUID of the simplex containing duplicate-coordinate vertices.
        simplex_id: Uuid,
        /// Description of which vertices share coordinates.
        message: String,
    },
}

/// Discriminant for compact [`TdsError`] summaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TdsErrorKind {
    /// A vertex failed validation.
    InvalidVertex,
    /// A simplex failed validation.
    InvalidSimplex,
    /// Neighbor relationships were invalid.
    InvalidNeighbors,
    /// Adjacent simplices violated coherent orientation.
    OrientationViolation,
    /// Duplicate maximal simplices were detected.
    DuplicateSimplices,
    /// A facet exceeded, or would exceed, the allowed incident-simplex count.
    FacetSharingViolation,
    /// Simplex creation failed.
    FailedToCreateSimplex,
    /// Expected neighbor relation was absent.
    NotNeighbors,
    /// UUID-to-key mapping was inconsistent.
    MappingInconsistency,
    /// Simplex vertex-key lookup failed.
    VertexKeyRetrievalFailed,
    /// A referenced simplex key was missing.
    SimplexNotFound,
    /// A referenced vertex key was missing.
    VertexNotFound,
    /// A dimension/count invariant was violated.
    DimensionMismatch,
    /// An index exceeded its valid bound.
    IndexOutOfBounds,
    /// Canonical incidence still referenced a removed simplex.
    RemovedSimplexStillIncident,
    /// Canonical incidence disagreed with simplex vertex membership.
    VertexIncidenceMismatch,
    /// Internal TDS state was inconsistent.
    InconsistentDataStructure,
    /// A geometric validation failure occurred.
    Geometric,
    /// A facet operation failed.
    FacetError,
    /// A simplex contained duplicate coordinates.
    DuplicateCoordinatesInSimplex,
}

impl From<&TdsError> for TdsErrorKind {
    fn from(source: &TdsError) -> Self {
        match source {
            TdsError::InvalidVertex { .. } => Self::InvalidVertex,
            TdsError::InvalidSimplex { .. } => Self::InvalidSimplex,
            TdsError::InvalidNeighbors { .. } => Self::InvalidNeighbors,
            TdsError::OrientationViolation { .. } => Self::OrientationViolation,
            TdsError::DuplicateSimplices { .. } | TdsError::DuplicateExplicitSimplices { .. } => {
                Self::DuplicateSimplices
            }
            TdsError::FacetSharingViolation { .. }
            | TdsError::ExplicitFacetSharingViolation { .. } => Self::FacetSharingViolation,
            TdsError::FailedToCreateSimplex { .. } => Self::FailedToCreateSimplex,
            TdsError::NotNeighbors { .. } => Self::NotNeighbors,
            TdsError::MappingInconsistency { .. } => Self::MappingInconsistency,
            TdsError::VertexKeyRetrievalFailed { .. } => Self::VertexKeyRetrievalFailed,
            TdsError::SimplexNotFound { .. } => Self::SimplexNotFound,
            TdsError::VertexNotFound { .. } => Self::VertexNotFound,
            TdsError::DimensionMismatch { .. } => Self::DimensionMismatch,
            TdsError::IndexOutOfBounds { .. } => Self::IndexOutOfBounds,
            TdsError::RemovedSimplexStillIncident { .. } => Self::RemovedSimplexStillIncident,
            TdsError::VertexIncidenceMismatch { .. } => Self::VertexIncidenceMismatch,
            TdsError::InconsistentDataStructure { .. } => Self::InconsistentDataStructure,
            TdsError::Geometric(_) => Self::Geometric,
            TdsError::FacetError(_) => Self::FacetError,
            TdsError::DuplicateCoordinatesInSimplex { .. } => Self::DuplicateCoordinatesInSimplex,
        }
    }
}

/// Errors that can occur during TDS mutation operations.
///
/// This error is a thin wrapper around [`TdsError`]. Mutation operations can fail
/// for the same reasons as validation (i.e., because an invariant would be violated or a
/// consistency check fails while attempting to perform the mutation).
///
/// The wrapper exists to make call sites and API docs semantically explicit, while also
/// allowing this error to evolve into a richer, dedicated type in a future release without
/// breaking the public API surface.
///
/// # Stability / conversion contract
///
/// `TdsMutationError` currently supports lossless conversion to and from [`TdsError`]
/// via the provided `From`/`Into` impls. If this wrapper evolves to include mutation-specific
/// context (additional fields/variants), converting `TdsMutationError` into [`TdsError`]
/// may become lossy.
///
/// Callers that want to preserve potential future mutation-specific details should avoid
/// converting back to [`TdsError`] and instead propagate/handle `TdsMutationError`
/// directly.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::{TdsError, TdsMutationError};
///
/// let err = TdsError::InconsistentDataStructure {
///     message: "bad neighbors".to_string(),
/// };
/// let mutation: TdsMutationError = err.clone().into();
/// let round_trip: TdsError = mutation.clone().into();
/// assert_eq!(round_trip, err);
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[error(transparent)]
#[must_use]
pub struct TdsMutationError(TdsError);

impl TdsMutationError {
    /// Returns a reference to the underlying [`TdsError`].
    #[must_use]
    pub const fn as_tds_error(&self) -> &TdsError {
        &self.0
    }

    /// Consumes this wrapper and returns the underlying [`TdsError`].
    #[must_use]
    pub fn into_inner(self) -> TdsError {
        self.0
    }
}

impl From<TdsError> for TdsMutationError {
    fn from(err: TdsError) -> Self {
        Self(err)
    }
}

impl From<TdsMutationError> for TdsError {
    fn from(err: TdsMutationError) -> Self {
        err.0
    }
}

/// Classifies the kind of triangulation invariant that failed during validation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::InvariantKind;
///
/// let kind = InvariantKind::Topology;
/// assert_eq!(kind, InvariantKind::Topology);
/// ```
///
/// This is used by [`TriangulationValidationReport`] to group related errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum InvariantKind {
    /// Per-vertex validity (finite coordinates, non-nil UUID, etc.).
    VertexValidity,
    /// Per-simplex validity (vertex count, duplicate vertices, nil UUID, etc.).
    SimplexValidity,
    /// No simplices contain vertices with identical coordinates (geometric uniqueness).
    SimplexCoordinateUniqueness,
    /// Vertex UUID↔key mapping invariants.
    VertexMappings,
    /// Simplex UUID↔key mapping invariants.
    SimplexMappings,
    /// Simplices reference only valid vertex keys (no stale/missing vertex keys).
    SimplexVertexKeys,
    /// Vertex incidence invariants (`Vertex::incident_simplex` pointers are non-dangling + consistent).
    VertexIncidence,
    /// Maintained vertex→simplices incidence index is complete and consistent.
    VertexToSimplicesIndex,
    /// No duplicate maximal simplices with identical vertex sets.
    DuplicateSimplices,
    /// Facet sharing invariants (each facet shared by at most 2 simplices).
    FacetSharing,
    /// Neighbor topology and mutual-consistency invariants.
    NeighborConsistency,
    /// Coherent combinatorial orientation (adjacent simplices induce opposite facet orientations).
    CoherentOrientation,
    /// Simplex neighbor graph connectivity (single connected component).
    Connectedness,
    /// Triangulation/topology invariants (manifold-with-boundary, Euler characteristic).
    Topology,
    /// Realized geometry (nondegenerate simplices, no illegal overlap).
    Realization,
    /// Delaunay empty-circumsphere property.
    DelaunayProperty,
}

/// A union error type that can represent any layer's validation failure.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::{InvariantError, TdsError};
///
/// let err = InvariantError::Tds(TdsError::InconsistentDataStructure {
///     message: "bad neighbors".to_string(),
/// });
/// std::assert_matches!(err, InvariantError::Tds(_));
/// ```
///
/// This is used by [`TriangulationValidationReport`] so that diagnostic reporting can
/// preserve structured errors from each layer (TDS / topology / Delaunay) without
/// stringification.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum InvariantError {
    /// Level 1–2 (elements + TDS structure).
    #[error(transparent)]
    Tds(#[from] TdsError),

    /// Level 3 (topology).
    #[error(transparent)]
    Triangulation(#[from] TriangulationValidationError),

    /// Level 4 (realized geometry).
    #[error(transparent)]
    Realization(#[from] TriangulationRealizationValidationError),

    /// Level 5 (Delaunay property).
    #[error(transparent)]
    Delaunay(#[from] DelaunayTriangulationValidationError),
}

/// Discriminant for compact Level 3 topology-validation summaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TriangulationValidationErrorKind {
    /// A facet had invalid manifold multiplicity.
    ManifoldFacetMultiplicity,
    /// A boundary ridge had invalid boundary-facet multiplicity.
    BoundaryRidgeMultiplicity,
    /// A closed topology contained an open boundary facet.
    BoundaryFacetInClosedTopology,
    /// A non-periodic topology contained periodic self-identification metadata.
    PeriodicIdentificationInNonPeriodicTopology,
    /// A requested ridge candidate was not present in any simplex.
    RidgeNotFound,
    /// A ridge link failed PL-manifold validation.
    RidgeLinkNotManifold,
    /// A vertex link failed PL-manifold validation.
    VertexLinkNotManifold,
    /// The intrinsic simplex-orientation constraints are contradictory.
    NonOrientable,
    /// Euler characteristic did not match the expected classification.
    EulerCharacteristicMismatch,
    /// A vertex was not incident to any simplex.
    IsolatedVertex,
    /// The simplex-neighbor graph was disconnected.
    Disconnected,
    /// Positive-orientation promotion did not converge.
    OrientationPromotionNonConvergence,
}

impl From<&TriangulationValidationError> for TriangulationValidationErrorKind {
    fn from(source: &TriangulationValidationError) -> Self {
        match source {
            TriangulationValidationError::ManifoldFacetMultiplicity { .. } => {
                Self::ManifoldFacetMultiplicity
            }
            TriangulationValidationError::BoundaryRidgeMultiplicity { .. } => {
                Self::BoundaryRidgeMultiplicity
            }
            TriangulationValidationError::BoundaryFacetInClosedTopology { .. } => {
                Self::BoundaryFacetInClosedTopology
            }
            TriangulationValidationError::PeriodicIdentificationInNonPeriodicTopology {
                ..
            } => Self::PeriodicIdentificationInNonPeriodicTopology,
            TriangulationValidationError::RidgeNotFound { .. } => Self::RidgeNotFound,
            TriangulationValidationError::RidgeLinkNotManifold { .. } => Self::RidgeLinkNotManifold,
            TriangulationValidationError::VertexLinkNotManifold { .. } => {
                Self::VertexLinkNotManifold
            }
            TriangulationValidationError::NonOrientable { .. } => Self::NonOrientable,
            TriangulationValidationError::EulerCharacteristicMismatch { .. } => {
                Self::EulerCharacteristicMismatch
            }
            TriangulationValidationError::IsolatedVertex { .. } => Self::IsolatedVertex,
            TriangulationValidationError::Disconnected { .. } => Self::Disconnected,
            TriangulationValidationError::OrientationPromotionNonConvergence { .. } => {
                Self::OrientationPromotionNonConvergence
            }
        }
    }
}

/// Discriminant for compact Level 5 Delaunay-validation summaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DelaunayValidationErrorKind {
    /// Lower-layer TDS validation failed.
    Tds,
    /// Lower-layer topology validation failed.
    Triangulation,
    /// Lower-layer realized-geometry validation failed.
    Realization,
    /// Delaunay verification failed.
    VerificationFailed,
    /// Typed repair validation failed.
    RepairOperationFailed,
}

impl From<&DelaunayTriangulationValidationError> for DelaunayValidationErrorKind {
    fn from(source: &DelaunayTriangulationValidationError) -> Self {
        match source {
            DelaunayTriangulationValidationError::Tds(_) => Self::Tds,
            DelaunayTriangulationValidationError::Triangulation(_) => Self::Triangulation,
            DelaunayTriangulationValidationError::Realization(_) => Self::Realization,
            DelaunayTriangulationValidationError::VerificationFailed { .. } => {
                Self::VerificationFailed
            }
            DelaunayTriangulationValidationError::RepairOperationFailed { .. } => {
                Self::RepairOperationFailed
            }
        }
    }
}

/// A single invariant violation recorded during validation diagnostics.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::{
///     InvariantError, InvariantKind, InvariantViolation, TdsError,
/// };
///
/// let violation = InvariantViolation {
///     kind: InvariantKind::Topology,
///     error: InvariantError::Tds(TdsError::InconsistentDataStructure {
///         message: "bad neighbors".to_string(),
///     }),
/// };
/// assert_eq!(violation.kind, InvariantKind::Topology);
/// ```
#[derive(Clone, Debug)]
pub struct InvariantViolation {
    /// The kind of invariant that failed.
    pub kind: InvariantKind,
    /// The detailed validation error explaining the failure.
    pub error: InvariantError,
}

/// Aggregate report of one or more validation failures.
///
/// This is returned by
/// [`DelaunayTriangulation::validation_report()`]
/// to surface all failed invariants at once for debugging and test diagnostics.
///
/// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::TriangulationValidationReport;
///
/// let report = TriangulationValidationReport { violations: Vec::new() };
/// assert!(report.is_empty());
/// ```
#[derive(Clone, Debug)]
pub struct TriangulationValidationReport {
    /// The ordered list of invariant violations that occurred.
    pub violations: Vec<InvariantViolation>,
}

impl TriangulationValidationReport {
    /// Returns `true` if no violations were recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::flips::{DelaunayRepairError, DelaunayRepairPostconditionFailure};
    use crate::core::facet::FacetError;
    use crate::core::simplex::SimplexValidationError;
    use crate::core::util::uuid::UuidValidationError;
    use crate::core::validation::TriangulationValidationError;
    use crate::core::vertex::VertexValidationError;
    use crate::repair::DelaunayRepairOperation;
    use crate::topology::characteristics::euler::TopologyClassification;
    use crate::topology::traits::topological_space::TopologyKind;
    use crate::validation::{DelaunayTriangulationValidationError, DelaunayVerificationError};
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

    fn assert_tds_error_kind(source: &TdsError, expected: TdsErrorKind) {
        assert_eq!(TdsErrorKind::from(source), expected);
    }

    #[test]
    fn triangulation_construction_state_default_is_empty_incomplete() {
        assert_eq!(
            TriangulationConstructionState::default(),
            TriangulationConstructionState::Incomplete(0)
        );
    }

    #[test]
    fn tds_error_kind_from_error_preserves_validation_variants() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let other_simplex_key = SimplexKey::from(KeyData::from_ffi(2));
        let vertex_key = VertexKey::from(KeyData::from_ffi(3));
        let uuid = Uuid::new_v4();

        assert_tds_error_kind(
            &TdsError::InvalidVertex {
                vertex_id: uuid,
                source: VertexValidationError::InvalidUuid {
                    source: UuidValidationError::NilUuid,
                },
            },
            TdsErrorKind::InvalidVertex,
        );
        assert_tds_error_kind(
            &TdsError::InvalidSimplex {
                simplex_id: uuid,
                source: SimplexValidationError::DuplicateVertices,
            },
            TdsErrorKind::InvalidSimplex,
        );
        assert_tds_error_kind(
            &TdsError::InvalidNeighbors {
                reason: NeighborValidationError::NonPeriodicSelfNeighbor {
                    simplex_key,
                    simplex_uuid: uuid,
                    facet_index: 0,
                },
            },
            TdsErrorKind::InvalidNeighbors,
        );
        assert_tds_error_kind(
            &TdsError::OrientationViolation {
                simplex1_key: simplex_key,
                simplex1_uuid: uuid,
                simplex2_key: other_simplex_key,
                simplex2_uuid: Uuid::new_v4(),
                simplex1_facet_index: 0,
                simplex2_facet_index: 1,
                facet_vertices: vec![vertex_key],
                simplex2_facet_vertices: vec![vertex_key],
                observed_odd_permutation: false,
                expected_odd_permutation: true,
            },
            TdsErrorKind::OrientationViolation,
        );
        assert_tds_error_kind(
            &TdsError::Geometric(GeometricError::DegenerateOrientation {
                message: "zero determinant".to_string(),
            }),
            TdsErrorKind::Geometric,
        );
        assert_tds_error_kind(
            &TdsError::FacetError(FacetError::InvalidFacetIndex {
                index: 4,
                facet_count: 4,
            }),
            TdsErrorKind::FacetError,
        );
        assert_tds_error_kind(
            &TdsError::DuplicateCoordinatesInSimplex {
                simplex_id: uuid,
                message: "two vertices share coordinates".to_string(),
            },
            TdsErrorKind::DuplicateCoordinatesInSimplex,
        );
    }

    #[test]
    fn tds_error_kind_from_error_preserves_lookup_and_operation_variants() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let vertex_key = VertexKey::from(KeyData::from_ffi(3));
        let uuid = Uuid::new_v4();

        assert_tds_error_kind(
            &TdsError::DuplicateSimplices {
                message: "duplicate simplex vertex set".to_string(),
            },
            TdsErrorKind::DuplicateSimplices,
        );
        assert_tds_error_kind(
            &TdsError::FacetSharingViolation {
                facet_key: 42,
                existing_incident_count: 2,
                attempted_incident_count: 3,
                max_incident_count: 2,
                candidate_simplex_uuid: uuid,
                candidate_facet_index: 1,
            },
            TdsErrorKind::FacetSharingViolation,
        );
        assert_tds_error_kind(
            &TdsError::FailedToCreateSimplex {
                message: "simplex validation failed".to_string(),
            },
            TdsErrorKind::FailedToCreateSimplex,
        );
        assert_tds_error_kind(
            &TdsError::NotNeighbors {
                simplex1: uuid,
                simplex2: Uuid::new_v4(),
            },
            TdsErrorKind::NotNeighbors,
        );
        assert_tds_error_kind(
            &TdsError::MappingInconsistency {
                entity: EntityKind::Simplex,
                message: "uuid mapping was stale".to_string(),
            },
            TdsErrorKind::MappingInconsistency,
        );
        assert_tds_error_kind(
            &TdsError::VertexKeyRetrievalFailed {
                simplex_id: uuid,
                message: "simplex vertices unavailable".to_string(),
            },
            TdsErrorKind::VertexKeyRetrievalFailed,
        );
        assert_tds_error_kind(
            &TdsError::SimplexNotFound {
                simplex_key,
                context: "simplex lookup".to_string(),
            },
            TdsErrorKind::SimplexNotFound,
        );
        assert_tds_error_kind(
            &TdsError::VertexNotFound {
                vertex_key,
                context: "vertex lookup".to_string(),
            },
            TdsErrorKind::VertexNotFound,
        );
        assert_tds_error_kind(
            &TdsError::DimensionMismatch {
                expected: 4,
                actual: 3,
                context: "simplex arity".to_string(),
            },
            TdsErrorKind::DimensionMismatch,
        );
        assert_tds_error_kind(
            &TdsError::IndexOutOfBounds {
                index: 4,
                bound: 4,
                context: "facet index".to_string(),
            },
            TdsErrorKind::IndexOutOfBounds,
        );
        assert_tds_error_kind(
            &TdsError::InconsistentDataStructure {
                message: "dangling neighbor".to_string(),
            },
            TdsErrorKind::InconsistentDataStructure,
        );
    }

    #[test]
    fn tds_error_kind_from_error_preserves_explicit_operation_variants() {
        assert_tds_error_kind(
            &TdsError::DuplicateExplicitSimplices {
                existing_simplex_index: 0,
                duplicate_simplex_index: 1,
                vertex_indices: vec![3],
            },
            TdsErrorKind::DuplicateSimplices,
        );
        assert_tds_error_kind(
            &TdsError::ExplicitFacetSharingViolation {
                facet_key: 42,
                facet_vertex_indices: vec![0, 1],
                existing_incident_count: 2,
                attempted_incident_count: 3,
                max_incident_count: 2,
                candidate_simplex_index: 2,
                candidate_facet_index: 1,
            },
            TdsErrorKind::FacetSharingViolation,
        );
    }

    #[test]
    fn triangulation_validation_error_kind_from_error_preserves_all_variants() {
        let vertex_key = VertexKey::from(KeyData::from_ffi(3));
        let cases = [
            (
                TriangulationValidationError::ManifoldFacetMultiplicity {
                    facet_key: 0xabc,
                    simplex_count: 3,
                },
                TriangulationValidationErrorKind::ManifoldFacetMultiplicity,
            ),
            (
                TriangulationValidationError::BoundaryRidgeMultiplicity {
                    ridge_key: 0xdef,
                    boundary_facet_count: 3,
                },
                TriangulationValidationErrorKind::BoundaryRidgeMultiplicity,
            ),
            (
                TriangulationValidationError::BoundaryFacetInClosedTopology {
                    topology: TopologyKind::Spherical,
                    facet_key: 0x111,
                    simplex_key: SimplexKey::from(KeyData::from_ffi(5)),
                    simplex_uuid: Uuid::new_v4(),
                    facet_index: 1,
                },
                TriangulationValidationErrorKind::BoundaryFacetInClosedTopology,
            ),
            (
                TriangulationValidationError::PeriodicIdentificationInNonPeriodicTopology {
                    topology: TopologyKind::Euclidean,
                    facet_key: 0x222,
                    simplex_key: SimplexKey::from(KeyData::from_ffi(6)),
                    simplex_uuid: Uuid::new_v4(),
                    facet_index: 2,
                },
                TriangulationValidationErrorKind::PeriodicIdentificationInNonPeriodicTopology,
            ),
            (
                TriangulationValidationError::RidgeNotFound {
                    ridge_vertices: iter::once(vertex_key).collect(),
                },
                TriangulationValidationErrorKind::RidgeNotFound,
            ),
            (
                TriangulationValidationError::RidgeLinkNotManifold {
                    ridge_key: 0x123,
                    link_vertex_count: 4,
                    link_edge_count: 2,
                    max_degree: 3,
                    degree_one_vertices: 1,
                    connected: false,
                },
                TriangulationValidationErrorKind::RidgeLinkNotManifold,
            ),
            (
                TriangulationValidationError::VertexLinkNotManifold {
                    vertex_key,
                    link_vertex_count: 4,
                    link_simplex_count: 2,
                    boundary_facet_count: 1,
                    max_degree: 3,
                    connected: false,
                    interior_vertex: true,
                },
                TriangulationValidationErrorKind::VertexLinkNotManifold,
            ),
            (
                TriangulationValidationError::EulerCharacteristicMismatch {
                    computed: 0,
                    expected: 1,
                    classification: TopologyClassification::Ball(3),
                },
                TriangulationValidationErrorKind::EulerCharacteristicMismatch,
            ),
            (
                TriangulationValidationError::IsolatedVertex {
                    vertex_key,
                    vertex_uuid: Uuid::new_v4(),
                },
                TriangulationValidationErrorKind::IsolatedVertex,
            ),
            (
                TriangulationValidationError::Disconnected { simplex_count: 2 },
                TriangulationValidationErrorKind::Disconnected,
            ),
            (
                TriangulationValidationError::OrientationPromotionNonConvergence {
                    residual_count: 1,
                    sampled: vec![SimplexKey::from(KeyData::from_ffi(4))],
                },
                TriangulationValidationErrorKind::OrientationPromotionNonConvergence,
            ),
        ];

        for (source, expected) in cases {
            assert_eq!(TriangulationValidationErrorKind::from(&source), expected);
        }
    }

    #[test]
    fn delaunay_validation_error_kind_from_error_preserves_all_variants() {
        let cases = [
            (
                DelaunayTriangulationValidationError::from(TdsError::InconsistentDataStructure {
                    message: "dangling simplex".to_string(),
                }),
                DelaunayValidationErrorKind::Tds,
            ),
            (
                DelaunayTriangulationValidationError::from(
                    TriangulationValidationError::Disconnected { simplex_count: 2 },
                ),
                DelaunayValidationErrorKind::Triangulation,
            ),
            (
                synthetic_delaunay_verification_error("non-Delaunay facet"),
                DelaunayValidationErrorKind::VerificationFailed,
            ),
            (
                DelaunayTriangulationValidationError::RepairOperationFailed {
                    operation: DelaunayRepairOperation::VertexRemoval,
                    source: Box::new(DelaunayRepairError::PostconditionFailed {
                        reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected {
                            simplex_count: 1,
                        }),
                    }),
                },
                DelaunayValidationErrorKind::RepairOperationFailed,
            ),
        ];

        for (source, expected) in cases {
            assert_eq!(DelaunayValidationErrorKind::from(&source), expected);
        }
    }

    // ---- Error variant Display / construction coverage ----

    #[test]
    fn test_geometric_error_display() {
        let deg = GeometricError::DegenerateOrientation {
            message: "det=0".to_string(),
        };
        assert!(deg.to_string().contains("det=0"));

        let neg = GeometricError::NegativeOrientation {
            message: "det<0".to_string(),
        };
        assert!(neg.to_string().contains("det<0"));
    }

    #[test]
    fn test_tds_error_new_variant_display() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let vertex_key = VertexKey::from(KeyData::from_ffi(2));

        let err = TdsError::SimplexNotFound {
            simplex_key,
            context: "test lookup".to_string(),
        };
        assert!(err.to_string().contains("not found"));
        assert!(err.to_string().contains("test lookup"));

        let err = TdsError::VertexNotFound {
            vertex_key,
            context: "test vertex".to_string(),
        };
        assert!(err.to_string().contains("not found"));
        assert!(err.to_string().contains("test vertex"));

        let err = TdsError::DimensionMismatch {
            expected: 4,
            actual: 3,
            context: "simplex check".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains('4') && msg.contains('3') && msg.contains("simplex check"));

        let err = TdsError::IndexOutOfBounds {
            index: 10,
            bound: 5,
            context: "facet index".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("10") && msg.contains('5') && msg.contains("facet index"));
    }

    #[test]
    fn test_tds_error_geometric_variant_wraps_geometric_error() {
        let inner = GeometricError::DegenerateOrientation {
            message: "test".to_string(),
        };
        let err = TdsError::Geometric(inner.clone());
        assert!(err.to_string().contains("test"));
        assert_eq!(TdsError::from(inner.clone()), TdsError::Geometric(inner));
    }

    #[test]
    fn test_tds_mutation_error_accessors() {
        let inner = TdsError::InconsistentDataStructure {
            message: "test".to_string(),
        };
        let mutation = TdsMutationError::from(inner.clone());

        // as_tds_error returns a reference to the inner error.
        assert_eq!(mutation.as_tds_error(), &inner);

        // into_inner consumes the wrapper and returns the inner error.
        let recovered: TdsError = mutation.into_inner();
        assert_eq!(recovered, inner);
    }

    #[test]
    fn test_invariant_error_from_tds_and_triangulation() {
        let tds_err = TdsError::InconsistentDataStructure {
            message: "test".to_string(),
        };
        let inv = InvariantError::from(tds_err);
        assert_matches!(inv, InvariantError::Tds(_));

        let tri_err = TriangulationValidationError::EulerCharacteristicMismatch {
            computed: 1,
            expected: 2,
            classification: TopologyClassification::Ball(3),
        };
        let inv = InvariantError::from(tri_err);
        assert_matches!(inv, InvariantError::Triangulation(_));
    }

    #[test]
    fn test_invariant_error_from_delaunay_validation_error() {
        let dt_err = synthetic_delaunay_verification_error("test");
        let inv = InvariantError::from(dt_err);
        assert_matches!(inv, InvariantError::Delaunay(_));
    }

    #[test]
    fn test_invariant_kind_all_variants_are_distinct() {
        let kinds = [
            InvariantKind::VertexValidity,
            InvariantKind::SimplexValidity,
            InvariantKind::SimplexCoordinateUniqueness,
            InvariantKind::VertexMappings,
            InvariantKind::SimplexMappings,
            InvariantKind::SimplexVertexKeys,
            InvariantKind::VertexIncidence,
            InvariantKind::VertexToSimplicesIndex,
            InvariantKind::DuplicateSimplices,
            InvariantKind::FacetSharing,
            InvariantKind::NeighborConsistency,
            InvariantKind::CoherentOrientation,
            InvariantKind::Connectedness,
            InvariantKind::Topology,
            InvariantKind::DelaunayProperty,
        ];
        // All variants must be copyable and comparable.
        for (i, &a) in kinds.iter().enumerate() {
            assert_eq!(a, a);
            for &b in &kinds[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn test_invariant_violation_stores_kind_and_error() {
        let violation = InvariantViolation {
            kind: InvariantKind::NeighborConsistency,
            error: InvariantError::Tds(TdsError::InconsistentDataStructure {
                message: "test".to_string(),
            }),
        };
        assert_eq!(violation.kind, InvariantKind::NeighborConsistency);
        assert_matches!(violation.error, InvariantError::Tds(_));
    }

    #[test]
    fn test_entity_kind_debug_output() {
        assert_eq!(format!("{:?}", EntityKind::Vertex), "Vertex");
        assert_eq!(format!("{:?}", EntityKind::Simplex), "Simplex");
        assert_ne!(EntityKind::Vertex, EntityKind::Simplex);
    }

    #[test]
    fn test_tds_mutation_error_from_round_trips() {
        // Test the full round-trip: TdsError -> TdsMutationError -> TdsError
        let original = TdsError::SimplexNotFound {
            simplex_key: SimplexKey::from(KeyData::from_ffi(42)),
            context: "round trip".to_string(),
        };
        let mutation = TdsMutationError::from(original.clone());
        assert_eq!(mutation.to_string(), original.to_string());
        let round_tripped: TdsError = mutation.into();
        assert_eq!(round_tripped, original);
    }

    #[test]
    fn test_geometric_error_from_into_tds_error() {
        let geo = GeometricError::NegativeOrientation {
            message: "det<0".to_string(),
        };
        let tds_err: TdsError = geo.into();
        assert_matches!(
            tds_err,
            TdsError::Geometric(GeometricError::NegativeOrientation { .. })
        );
        // Display propagates via #[error(transparent)].
        assert!(tds_err.to_string().contains("det<0"));
    }
}

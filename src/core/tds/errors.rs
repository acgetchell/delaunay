//! TDS error and validation-report vocabulary.
//!
//! This module keeps the public error vocabulary separate from the storage and
//! mutation implementation.

use super::{SimplexKey, VertexKey};
use crate::core::algorithms::flips::{FlipError, FlipNeighborWiringError};
use crate::core::facet::FacetError;
use crate::core::simplex::SimplexValidationError;
use crate::core::vertex::VertexValidationError;
use crate::geometry::traits::coordinate::CoordinateValues;
use thiserror::Error;
use uuid::Uuid;

// =============================================================================
// CONSTRUCTION STATE TYPES
// =============================================================================

/// Crate-internal publication state carried by draft-owned TDS storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TriangulationConstructionState {
    /// The triangulation has insufficient vertices to form a complete D-dimensional triangulation.
    /// Contains the number of vertices currently stored.
    Incomplete(usize),
    /// The TDS has crossed its Levels 1–2 publication boundary; this includes
    /// the verified empty complex.
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
    #[error("Validation error during construction: {source}")]
    ValidationError {
        /// Typed TDS validation source error.
        #[from]
        source: TdsError,
    },
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
/// use delaunay::prelude::{construction::*, tds::*};
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
/// #     Invariant(#[from] delaunay::prelude::validation::InvariantError),
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
        /// UUID pairs ordered as existing simplex, then duplicate or rejected candidate.
        simplex_pairs: Vec<[Uuid; 2]>,
        /// Supplemental diagnostic detail, not a machine-readable contract.
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
    /// Construction completion was requested before a full-dimensional simplex existed.
    #[error(
        "Cannot complete {dimension}D TDS construction with {vertex_count} vertices and {simplex_count} simplices"
    )]
    IncompleteConstruction {
        /// Compile-time ambient dimension of the TDS.
        dimension: usize,
        /// Number of vertices currently stored.
        vertex_count: usize,
        /// Number of maximal simplices currently stored.
        simplex_count: usize,
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
    /// Canonical vertex incidence has an unexpected number of entries.
    #[error("Vertex-to-simplices index has {actual} simplex incidences, expected {expected}")]
    VertexIncidenceCountMismatch {
        /// Observed incidence count.
        actual: usize,
        /// Count implied by simplex vertex membership.
        expected: usize,
    },
    /// A connected component cannot satisfy every coherent-orientation constraint.
    #[error(
        "Contradictory orientation constraints between simplices {simplex1_uuid} and {simplex2_uuid}"
    )]
    ContradictoryOrientationConstraints {
        /// UUID of the first simplex in the contradictory constraint.
        simplex1_uuid: Uuid,
        /// UUID of the second simplex in the contradictory constraint.
        simplex2_uuid: Uuid,
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
    Geometric {
        /// Typed geometric source error.
        #[from]
        source: GeometricError,
    },

    /// Facet operation failed during validation.
    #[error("Facet operation failed: {source}")]
    FacetError {
        /// Typed facet source error.
        #[from]
        source: FacetError,
    },
    /// A simplex contains two or more vertices with identical coordinates.
    ///
    /// This is distinct from [`SimplexValidationError::DuplicateVertices`] which checks
    /// for duplicate vertex *keys*. This variant detects the case where different
    /// vertex keys reference geometrically identical points — producing a zero-volume
    /// simplex that is catastrophic for `SoS` and Pachner moves.
    #[error(
        "Duplicate coordinates in simplex {simplex_id}: vertices {vertex_uuids:?} (keys {vertex_keys:?}) have identical coordinates {coordinates}"
    )]
    DuplicateCoordinatesInSimplex {
        /// UUID of the simplex containing duplicate-coordinate vertices.
        simplex_id: Uuid,
        /// Keys of the two distinct vertices with identical coordinates.
        vertex_keys: [VertexKey; 2],
        /// UUIDs corresponding to the two vertex keys, in the same order.
        vertex_uuids: [Uuid; 2],
        /// Shared coordinate tuple, preserved without display-string parsing.
        coordinates: CoordinateValues,
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
    /// Construction completion was requested before the TDS was ready.
    IncompleteConstruction,
    /// An index exceeded its valid bound.
    IndexOutOfBounds,
    /// Canonical incidence still referenced a removed simplex.
    RemovedSimplexStillIncident,
    /// Canonical incidence disagreed with simplex vertex membership.
    VertexIncidenceMismatch,
    /// Canonical incidence had an unexpected number of entries.
    VertexIncidenceCountMismatch,
    /// Coherent-orientation constraints contradicted one another.
    ContradictoryOrientationConstraints,
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
            TdsError::IncompleteConstruction { .. } => Self::IncompleteConstruction,
            TdsError::IndexOutOfBounds { .. } => Self::IndexOutOfBounds,
            TdsError::RemovedSimplexStillIncident { .. } => Self::RemovedSimplexStillIncident,
            TdsError::VertexIncidenceMismatch { .. } => Self::VertexIncidenceMismatch,
            TdsError::VertexIncidenceCountMismatch { .. } => Self::VertexIncidenceCountMismatch,
            TdsError::ContradictoryOrientationConstraints { .. } => {
                Self::ContradictoryOrientationConstraints
            }
            TdsError::InconsistentDataStructure { .. } => Self::InconsistentDataStructure,
            TdsError::Geometric { source: _ } => Self::Geometric,
            TdsError::FacetError { source: _ } => Self::FacetError,
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

/// Classifies the kind of Levels 1–2 invariant that failed during validation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::TdsInvariantKind;
///
/// let kind = TdsInvariantKind::NeighborConsistency;
/// assert_eq!(kind, TdsInvariantKind::NeighborConsistency);
/// ```
///
/// This is used by [`TdsValidationReport`] to group related errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TdsInvariantKind {
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
}

/// A single Levels 1–2 invariant violation.
///
/// The source is always a typed [`TdsError`]; higher-layer diagnostics are
/// represented by [`crate::validation::InvariantViolation`].
#[derive(Clone, Debug)]
pub struct TdsInvariantViolation {
    /// The local invariant group that failed.
    pub kind: TdsInvariantKind,
    /// The detailed element or structural validation error.
    pub error: TdsError,
}

/// Aggregate report of checkable Levels 1–2 validation failures.
///
/// [`Tds::structure_report`](super::Tds::structure_report) checks Level 2;
/// [`Tds::validation_report`](super::Tds::validation_report) also checks Level 1.
/// Reports retain check order and the typed source of every failure.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::{Tds, TdsValidationReport};
///
/// let tds = Tds::<(), (), 2>::empty();
/// let result: Result<(), TdsValidationReport> = tds.validation_report();
/// assert!(result.is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct TdsValidationReport {
    /// The ordered list of element and structural violations.
    pub violations: Vec<TdsInvariantViolation>,
}

impl TdsValidationReport {
    /// Returns `true` if no violations were recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::facet::FacetError;
    use crate::core::simplex::SimplexValidationError;
    use crate::core::util::uuid::UuidValidationError;
    use crate::core::vertex::VertexValidationError;
    use slotmap::KeyData;
    use std::assert_matches;

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
            &TdsError::Geometric {
                source: GeometricError::DegenerateOrientation {
                    message: "zero determinant".to_string(),
                },
            },
            TdsErrorKind::Geometric,
        );
        assert_tds_error_kind(
            &TdsError::FacetError {
                source: FacetError::InvalidFacetIndex {
                    index: 4,
                    facet_count: 4,
                },
            },
            TdsErrorKind::FacetError,
        );
        assert_tds_error_kind(
            &TdsError::DuplicateCoordinatesInSimplex {
                simplex_id: uuid,
                vertex_keys: [vertex_key, VertexKey::from(KeyData::from_ffi(4))],
                vertex_uuids: [uuid, Uuid::from_u128(2)],
                coordinates: CoordinateValues::from([0.0, 0.0]),
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
                simplex_pairs: vec![[uuid, Uuid::from_u128(2)]],
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
            &TdsError::IncompleteConstruction {
                dimension: 3,
                vertex_count: 3,
                simplex_count: 0,
            },
            TdsErrorKind::IncompleteConstruction,
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
        let err = TdsError::Geometric {
            source: inner.clone(),
        };
        assert!(err.to_string().contains("test"));
        assert_eq!(
            TdsError::from(inner.clone()),
            TdsError::Geometric { source: inner }
        );
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
            TdsError::Geometric {
                source: GeometricError::NegativeOrientation { .. }
            }
        );
        // Display propagates via #[error(transparent)].
        assert!(tds_err.to_string().contains("det<0"));
    }
}

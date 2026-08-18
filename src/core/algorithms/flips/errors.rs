//! Typed flip and Delaunay-repair failure taxonomy.

#![forbid(unsafe_code)]

use super::{
    CavityFillingError, ConflictError, CoordinateConversionError, CoordinateValidationError,
    CoordinateValues, DelaunayTriangulationValidationError, EdgeKey, EntityKind, Error, FacetError,
    FacetHandle, FlipDirection, FlipOrientationCheckStage, GlobalTopologyModelError,
    HullExtensionReason, InsertionError, InsertionErrorKind, InsertionTopologyValidationContext,
    LocateError, MAX_PRACTICAL_DIMENSION_SIZE, NeighborWiringError, RidgeHandle, SimplexKey,
    SimplexValidationError, SmallBuffer, SpatialIndexConstructionFailure, TdsConstructionFailure,
    TdsMutationError, TdsValidationFailure, TopologyGuarantee, TopologyOwnerId, TriangleHandle,
    TriangulationRealizationValidationError, TriangulationValidationError, VertexKey, fmt,
};

/// Predicate operation being evaluated by flip logic.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipPredicateOperation {
    /// Replacement-simplex orientation check while applying a flip.
    #[error("replacement-simplex orientation")]
    ReplacementSimplexOrientation,
    /// Replacement-simplex orientation postcondition during Delaunay repair.
    #[error("Delaunay-repair replacement-simplex orientation")]
    DelaunayRepairReplacementOrientation,
    /// Degenerate-simplex precheck before applying a flip.
    #[error("degenerate-simplex precheck")]
    DegenerateSimplexPrecheck,
    /// First k=2 insphere predicate.
    #[error("k=2 simplex-A insphere")]
    K2SimplexAInSphere,
    /// Second k=2 insphere predicate.
    #[error("k=2 simplex-B insphere")]
    K2SimplexBInSphere,
    /// k=3 insphere predicate.
    #[error("k=3 simplex insphere")]
    K3SimplexInSphere,
}

/// Structured reason a geometric predicate failed during a flip.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum FlipPredicateError {
    /// A coordinate-conversion or exact-predicate helper failed.
    #[error("{operation} predicate failed: {source}")]
    CoordinateConversion {
        /// Predicate operation being evaluated.
        operation: FlipPredicateOperation,
        /// Underlying coordinate conversion failure.
        #[source]
        source: CoordinateConversionError,
    },
    /// A topology model failed to lift a periodic vertex for predicate evaluation.
    #[error("failed to lift vertex {vertex_key:?} for periodic predicate: {source}")]
    PeriodicVertexLift {
        /// Vertex being lifted.
        vertex_key: VertexKey,
        /// Underlying topology-model error.
        #[source]
        source: GlobalTopologyModelError,
    },
    /// A lifted periodic vertex produced invalid point coordinates.
    #[error("lifted periodic vertex {vertex_key:?} produced invalid coordinates: {source}")]
    PeriodicLiftedPointValidation {
        /// Vertex whose lifted coordinates were invalid.
        vertex_key: VertexKey,
        /// Coordinate validation failure for the lifted point.
        #[source]
        source: CoordinateValidationError,
    },
    /// A topology model failed to lift a proposed k=1 vertex into the selected simplex frame.
    #[error("failed to lift proposed k=1 vertex into simplex {simplex_key:?} frame: {source}")]
    K1InsertedVertexLift {
        /// Simplex whose local realization frame was requested.
        simplex_key: SimplexKey,
        /// Underlying topology-model error.
        #[source]
        source: GlobalTopologyModelError,
    },
    /// A lifted proposed k=1 vertex produced invalid point coordinates.
    #[error(
        "proposed k=1 vertex lifted into simplex {simplex_key:?} frame produced invalid coordinates: {source}"
    )]
    K1InsertedVertexPointValidation {
        /// Simplex whose local realization frame was requested.
        simplex_key: SimplexKey,
        /// Coordinate validation failure for the lifted point.
        #[source]
        source: CoordinateValidationError,
    },
}

impl FlipPredicateError {
    pub(super) const fn coordinate_conversion(
        operation: FlipPredicateOperation,
        source: CoordinateConversionError,
    ) -> Self {
        Self::CoordinateConversion { operation, source }
    }
}

/// Structured reason a flip context is invalid before mutation.
///
/// These reasons are wrapped by [`FlipError::InvalidFlipContext`] so callers can
/// distinguish shape errors, replacement-orientation conflicts, and periodic
/// frame-alignment failures before any TDS mutation is committed.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{FlipContextError, FlipError};
///
/// let reason = FlipContextError::ReplacementPeriodicOffsetCountMismatch {
///     simplex_count: 2,
///     offset_count: 1,
/// };
/// let err: FlipError = reason.into();
/// std::assert_matches!(err, FlipError::InvalidFlipContext { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipContextError {
    /// The requested move size is outside `1..=D+1`.
    #[error("k must be in 1..=D+1 (k={k_move}, D={dimension})")]
    InvalidMoveSize {
        /// Requested k-move.
        k_move: usize,
        /// Triangulation dimension.
        dimension: usize,
    },
    /// Removed face has the wrong arity for the k-move.
    #[error("removed-face must have {expected} vertices, got {found}")]
    WrongRemovedFaceArity {
        /// Expected removed-face vertex count.
        expected: usize,
        /// Observed removed-face vertex count.
        found: usize,
    },
    /// Inserted face has the wrong arity for the k-move.
    #[error("k={k_move} inserted-face must have {expected} vertices, got {found}")]
    WrongInsertedFaceArity {
        /// Requested k-move.
        k_move: usize,
        /// Expected inserted-face vertex count.
        expected: usize,
        /// Observed inserted-face vertex count.
        found: usize,
    },
    /// The number of simplices selected for removal does not match the k-move.
    #[error("removed_simplices must have {expected} entries, got {found}")]
    WrongRemovedSimplexCount {
        /// Expected number of removed simplices.
        expected: usize,
        /// Observed number of removed simplices.
        found: usize,
    },
    /// Removed and inserted faces are not disjoint.
    #[error("removed-face and inserted-face must be disjoint")]
    OverlappingFaces,
    /// The TDS coherent-orientation invariant failed during debug/test flip validation.
    ///
    /// This diagnostic is reserved for validation builds because the check scans
    /// global TDS orientation state. Production callers should use explicit TDS
    /// or triangulation validation when they need this invariant checked.
    #[error(
        "TDS coherent orientation invariant violated during {stage:?} for k={k_move}, direction={direction:?}"
    )]
    CoherentOrientationViolation {
        /// Stage where the invariant was checked.
        stage: FlipOrientationCheckStage,
        /// k for the attempted move.
        k_move: usize,
        /// Direction of the attempted move.
        direction: FlipDirection,
    },
    /// Replacement-simplex offset sidecar length does not match the replacement simplices.
    #[error(
        "replacement periodic offset count {offset_count} does not match replacement simplex count {simplex_count}"
    )]
    ReplacementPeriodicOffsetCountMismatch {
        /// Number of replacement simplices.
        simplex_count: usize,
        /// Number of periodic-offset entries.
        offset_count: usize,
    },
    /// A periodic parity constraint referenced a replacement simplex without offsets.
    #[error(
        "replacement simplex {simplex_index} is missing periodic offsets for periodic facet parity"
    )]
    MissingReplacementPeriodicOffsets {
        /// Local replacement-simplex index.
        simplex_index: usize,
    },
    /// Replacement-simplex periodic offsets are not aligned with its vertex slots.
    #[error(
        "replacement simplex {simplex_index} periodic offset count {offset_count} does not match vertex count {vertex_count}"
    )]
    ReplacementPeriodicOffsetLengthMismatch {
        /// Local replacement-simplex index.
        simplex_index: usize,
        /// Number of periodic offsets.
        offset_count: usize,
        /// Number of replacement-simplex vertices.
        vertex_count: usize,
    },
    /// Replacement-simplex orientation constraints disagree.
    #[error(
        "conflicting replacement-simplex orientation constraints between local simplices {source_simplex_index} and {target_simplex_index}"
    )]
    ConflictingReplacementOrientationBetweenSimplices {
        /// First local replacement-simplex index.
        source_simplex_index: usize,
        /// Second local replacement-simplex index.
        target_simplex_index: usize,
    },
    /// Replacement-simplex orientation cannot be flipped because the simplex is too small.
    #[error("replacement simplex needs at least two vertices to flip orientation")]
    ReplacementSimplexTooSmallForOrientationFlip,
    /// Replacement orientation assignment referenced a missing local simplex.
    #[error("replacement orientation index {simplex_index} out of range")]
    ReplacementOrientationIndexOutOfRange {
        /// Local replacement-simplex index.
        simplex_index: usize,
    },
    /// Two parity constraints disagree for the same replacement simplex.
    #[error(
        "conflicting replacement-simplex orientation constraints for local simplex {simplex_index}"
    )]
    ConflictingReplacementOrientationForSimplex {
        /// Local replacement-simplex index.
        simplex_index: usize,
    },
    /// The facet-order permutation parity could not be derived.
    #[error("could not derive replacement facet-order permutation parity")]
    FacetOrderParityUnavailable,
    /// A facet index is outside the replacement simplex's vertex range.
    #[error(
        "facet index {facet_index} out of range for replacement simplex with {vertex_count} vertices"
    )]
    ReplacementFacetIndexOutOfRange {
        /// Invalid facet index.
        facet_index: usize,
        /// Replacement-simplex vertex count.
        vertex_count: usize,
    },
    /// A k=2 facet predicate received the wrong number of facet vertices.
    #[error("k=2 facet must have {expected} vertices, got {found}")]
    K2FacetArity {
        /// Expected facet vertex count.
        expected: usize,
        /// Observed facet vertex count.
        found: usize,
    },
    /// k=2 opposite vertices are not a valid complementary face.
    #[error("k=2 opposites must be distinct and not in the facet")]
    InvalidK2Opposites,
    /// A k=3 predicate received the wrong number of ridge vertices.
    #[error("k=3 ridge must have {expected} vertices, got {found}")]
    K3RidgeArity {
        /// Expected ridge vertex count.
        expected: usize,
        /// Observed ridge vertex count.
        found: usize,
    },
    /// Repeated slots for one periodic vertex disagree on its lifted representative.
    #[error(
        "conflicting periodic offsets for vertex {vertex_key:?} in simplex {simplex_key:?}: expected {expected_offset:?}, got {found_offset:?}"
    )]
    ConflictingPeriodicVertexOffset {
        /// Simplex containing the repeated vertex slots.
        simplex_key: SimplexKey,
        /// Repeated vertex whose slots disagree.
        vertex_key: VertexKey,
        /// First offset observed for the vertex.
        expected_offset: Vec<i8>,
        /// Conflicting offset observed in a later slot.
        found_offset: Vec<i8>,
    },
    /// Periodic frame alignment found contradictory translations.
    #[error(
        "conflicting periodic frame translations while aligning vertex {vertex_key:?} from simplex {source_simplex_key:?} into frame {target_simplex_key:?}: expected {expected_offset:?}, got {found_offset:?}"
    )]
    ConflictingPeriodicFrameTranslation {
        /// Vertex being aligned.
        vertex_key: VertexKey,
        /// Source simplex used for alignment.
        source_simplex_key: SimplexKey,
        /// Target simplex frame.
        target_simplex_key: SimplexKey,
        /// Previously derived offset.
        expected_offset: Vec<i8>,
        /// Conflicting candidate offset.
        found_offset: Vec<i8>,
    },
    /// Periodic frame alignment disagreed for an external-to-replacement facet.
    #[error(
        "conflicting periodic frame translations while aligning vertex {vertex_key:?} from external simplex {source_simplex_key:?} into replacement simplex {target_simplex_index}: expected {expected_offset:?}, got {found_offset:?}"
    )]
    ConflictingReplacementPeriodicFrameTranslation {
        /// Vertex being aligned.
        vertex_key: VertexKey,
        /// External source simplex used for alignment.
        source_simplex_key: SimplexKey,
        /// Target replacement-simplex index.
        target_simplex_index: usize,
        /// Previously derived offset.
        expected_offset: Vec<i8>,
        /// Conflicting candidate offset.
        found_offset: Vec<i8>,
    },
    /// No source simplex could align a periodic vertex into the target frame.
    #[error("cannot align periodic vertex {vertex_key:?} into frame {target_simplex_key:?}")]
    PeriodicVertexAlignmentFailed {
        /// Vertex being aligned.
        vertex_key: VertexKey,
        /// Target simplex frame.
        target_simplex_key: SimplexKey,
    },
    /// Periodic offset count does not match the simplex's vertex count.
    #[error(
        "simplex {simplex_key:?} periodic offset count {offset_count} does not match vertex count {vertex_count}"
    )]
    PeriodicOffsetCountMismatch {
        /// Simplex with malformed offsets.
        simplex_key: SimplexKey,
        /// Number of stored offsets.
        offset_count: usize,
        /// Number of simplex vertices.
        vertex_count: usize,
    },
    /// Periodic offset subtraction overflowed on an axis.
    #[error("periodic offset subtraction overflow on axis {axis}")]
    PeriodicOffsetSubtractionOverflow {
        /// Coordinate axis.
        axis: usize,
    },
    /// Periodic offset addition overflowed on an axis.
    #[error("periodic offset addition overflow on axis {axis}")]
    PeriodicOffsetAdditionOverflow {
        /// Coordinate axis.
        axis: usize,
    },
    /// Inverse predicate evaluation had no removed-simplex frame.
    #[error("inverse flip predicate requires at least one removed simplex frame")]
    MissingRemovedSimplexFrame,
}

/// Non-recursive summary of a flip error that reached another flip error path.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipFailureKind {
    /// [`PachnerProposal`](crate::pachner::PachnerProposal) belongs to a
    /// different topology owner.
    #[error("wrong topology owner")]
    WrongTopologyOwner,
    /// [`PachnerProposal`](crate::pachner::PachnerProposal) was minted before
    /// the current topology generation.
    #[error("stale topology proposal")]
    StaleTopologyProposal,
    /// Flips are not supported for this dimension.
    #[error("unsupported dimension")]
    UnsupportedDimension,
    /// The target lacks the topology proof required by the flip class.
    #[error("flip topology not admissible")]
    FlipTopologyNotAdmissible,
    /// Boundary facet.
    #[error("boundary facet")]
    BoundaryFacet,
    /// Missing simplex.
    #[error("missing simplex")]
    MissingSimplex,
    /// Dangling vertex-to-simplex incidence reference.
    #[error("dangling vertex incidence")]
    DanglingVertexIncidence,
    /// Missing vertex.
    #[error("missing vertex")]
    MissingVertex,
    /// Missing neighbor.
    #[error("missing neighbor")]
    MissingNeighbor,
    /// Dangling ridge-neighbor reference.
    #[error("dangling ridge neighbor")]
    DanglingRidgeNeighbor,
    /// Invalid facet adjacency.
    #[error("invalid facet adjacency")]
    InvalidFacetAdjacency,
    /// Invalid facet index.
    #[error("invalid facet index")]
    InvalidFacetIndex,
    /// Invalid ridge index.
    #[error("invalid ridge index")]
    InvalidRidgeIndex,
    /// Invalid ridge adjacency.
    #[error("invalid ridge adjacency")]
    InvalidRidgeAdjacency,
    /// Invalid ridge multiplicity.
    #[error("invalid ridge multiplicity")]
    InvalidRidgeMultiplicity,
    /// Invalid edge multiplicity.
    #[error("invalid edge multiplicity")]
    InvalidEdgeMultiplicity,
    /// Invalid triangle multiplicity.
    #[error("invalid triangle multiplicity")]
    InvalidTriangleMultiplicity,
    /// Invalid edge adjacency.
    #[error("invalid edge adjacency")]
    InvalidEdgeAdjacency,
    /// Invalid triangle adjacency.
    #[error("invalid triangle adjacency")]
    InvalidTriangleAdjacency,
    /// Invalid vertex multiplicity.
    #[error("invalid vertex multiplicity")]
    InvalidVertexMultiplicity,
    /// Invalid vertex adjacency.
    #[error("invalid vertex adjacency")]
    InvalidVertexAdjacency,
    /// Invalid flip context.
    #[error("invalid flip context")]
    InvalidFlipContext,
    /// Predicate failure.
    #[error("predicate failure")]
    PredicateFailure,
    /// Degenerate simplex.
    #[error("degenerate simplex")]
    DegenerateSimplex,
    /// Forward k=1 insertion point lies outside the selected simplex.
    #[error("k=1 insertion outside simplex")]
    K1InsertionOutsideSimplex,
    /// Negative orientation.
    #[error("negative orientation")]
    NegativeOrientation,
    /// Duplicate simplex.
    #[error("duplicate simplex")]
    DuplicateSimplex,
    /// Non-manifold facet.
    #[error("non-manifold facet")]
    NonManifoldFacet,
    /// Inserted simplex already exists.
    #[error("inserted simplex already exists")]
    InsertedSimplexAlreadyExists,
    /// Facet iteration failed.
    #[error("facet iteration")]
    FacetIteration,
    /// Simplex creation failed.
    #[error("simplex creation")]
    SimplexCreation,
    /// Flip transaction could not repair post-mutation orientation invariants.
    #[error("postcondition orientation repair")]
    PostconditionRepair,
    /// Flip transaction failed realized-geometry validation after mutation.
    #[error("realization validation")]
    RealizationValidation,
    /// Neighbor wiring failed.
    #[error("neighbor wiring")]
    NeighborWiring,
    /// Trial TDS validation failed before committing a flip.
    #[error("trial validation")]
    TrialValidation,
    /// Neighbor wiring reached a validation failure.
    #[error("wiring validation")]
    WiringValidation,
    /// Neighbor wiring reached a Delaunay repair failure.
    #[error("Delaunay repair failed")]
    DelaunayRepairFailed,
    /// TDS mutation failed.
    #[error("TDS mutation")]
    TdsMutation,
}

/// Non-recursive summary of a cavity-filling error at the flip wiring boundary.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipNeighborCavityFailureKind {
    /// Boundary simplex was missing.
    #[error("missing boundary simplex")]
    MissingBoundarySimplex,
    /// Inserted vertex was missing.
    #[error("missing inserted vertex")]
    MissingInsertedVertex,
    /// Boundary simplex had the wrong arity.
    #[error("wrong simplex arity")]
    WrongSimplexArity,
    /// Facet index was invalid.
    #[error("invalid facet index")]
    InvalidFacetIndex,
    /// Replacement simplex creation failed.
    #[error("simplex creation")]
    SimplexCreation,
    /// Replacement simplex insertion failed.
    #[error("simplex insertion")]
    SimplexInsertion,
    /// Initial simplex construction failed.
    #[error("initial simplex construction")]
    InitialSimplexConstruction,
    /// Rebuilt TDS lost the inserted vertex.
    #[error("rebuilt vertex missing")]
    RebuiltVertexMissing,
    /// Conflict region was empty.
    #[error("empty conflict region")]
    EmptyConflictRegion,
    /// Cavity boundary was empty.
    #[error("empty boundary")]
    EmptyBoundary,
    /// Facet sharing remained invalid after repair.
    #[error("invalid facet sharing after repair")]
    InvalidFacetSharingAfterRepair,
    /// Cavity filling created the wrong number of replacement simplices.
    #[error("boundary simplex count mismatch")]
    BoundarySimplexCountMismatch,
    /// Neighbor rebuild failed.
    #[error("neighbor rebuild")]
    NeighborRebuild,
    /// Perturbation scale conversion failed.
    #[error("perturbation scale conversion")]
    PerturbationScaleConversion,
    /// Degenerate insertion location is unsupported.
    #[error("unsupported degenerate location")]
    UnsupportedDegenerateLocation,
    /// Fan filling produced no simplices.
    #[error("empty fan triangulation")]
    EmptyFanTriangulation,
}

impl From<&CavityFillingError> for FlipNeighborCavityFailureKind {
    fn from(source: &CavityFillingError) -> Self {
        match source {
            CavityFillingError::MissingBoundarySimplex { .. } => Self::MissingBoundarySimplex,
            CavityFillingError::MissingInsertedVertex { .. } => Self::MissingInsertedVertex,
            CavityFillingError::WrongSimplexArity { .. } => Self::WrongSimplexArity,
            CavityFillingError::InvalidFacetIndex { .. } => Self::InvalidFacetIndex,
            CavityFillingError::SimplexCreation { .. } => Self::SimplexCreation,
            CavityFillingError::SimplexInsertion { .. } => Self::SimplexInsertion,
            CavityFillingError::InitialSimplexConstruction { .. } => {
                Self::InitialSimplexConstruction
            }
            CavityFillingError::RebuiltVertexMissing { .. } => Self::RebuiltVertexMissing,
            CavityFillingError::EmptyConflictRegion { .. } => Self::EmptyConflictRegion,
            CavityFillingError::EmptyBoundary { .. } => Self::EmptyBoundary,
            CavityFillingError::InvalidFacetSharingAfterRepair { .. } => {
                Self::InvalidFacetSharingAfterRepair
            }
            CavityFillingError::BoundarySimplexCountMismatch { .. } => {
                Self::BoundarySimplexCountMismatch
            }
            CavityFillingError::NeighborRebuild { .. } => Self::NeighborRebuild,
            CavityFillingError::PerturbationScaleConversion { .. } => {
                Self::PerturbationScaleConversion
            }
            CavityFillingError::UnsupportedDegenerateLocation { .. } => {
                Self::UnsupportedDegenerateLocation
            }
            CavityFillingError::EmptyFanTriangulation => Self::EmptyFanTriangulation,
        }
    }
}

impl From<CavityFillingError> for FlipNeighborCavityFailureKind {
    fn from(source: CavityFillingError) -> Self {
        Self::from(&source)
    }
}

/// Non-recursive summary of a hull-extension error at the flip wiring boundary.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipNeighborHullExtensionFailureKind {
    /// No visible facets were found.
    #[error("no visible facets")]
    NoVisibleFacets,
    /// Boundary-edge split matched the wrong number of facets.
    #[error("boundary edge split facet count")]
    BoundaryEdgeSplitFacetCount,
    /// Boundary-edge split matched more than one candidate facet.
    #[error("multiple boundary edge split facets")]
    MultipleBoundaryEdgeSplitFacets,
    /// Visible facets formed a disconnected or non-manifold patch.
    #[error("disconnected visible patch")]
    DisconnectedVisiblePatch,
    /// Geometric predicate failed.
    #[error("predicate failed")]
    PredicateFailed,
    /// Lower-layer TDS error.
    #[error("TDS")]
    Tds,
}

impl From<&HullExtensionReason> for FlipNeighborHullExtensionFailureKind {
    fn from(source: &HullExtensionReason) -> Self {
        match source {
            HullExtensionReason::NoVisibleFacets => Self::NoVisibleFacets,
            HullExtensionReason::BoundaryEdgeSplitFacetCount { .. } => {
                Self::BoundaryEdgeSplitFacetCount
            }
            HullExtensionReason::MultipleBoundaryEdgeSplitFacets { .. } => {
                Self::MultipleBoundaryEdgeSplitFacets
            }
            HullExtensionReason::DisconnectedVisiblePatch { .. } => Self::DisconnectedVisiblePatch,
            HullExtensionReason::PredicateFailed { source: _ } => Self::PredicateFailed,
            HullExtensionReason::Tds { source: _ } => Self::Tds,
        }
    }
}

impl From<HullExtensionReason> for FlipNeighborHullExtensionFailureKind {
    fn from(source: HullExtensionReason) -> Self {
        Self::from(&source)
    }
}

/// Non-recursive summary of a Delaunay validation error at the flip wiring boundary.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipNeighborDelaunayValidationFailureKind {
    /// Lower-layer TDS validation failed.
    #[error("TDS")]
    Tds,
    /// Lower-layer topology validation failed.
    #[error("triangulation")]
    Triangulation,
    /// Realized-geometry validation failed.
    #[error("realization")]
    Realization,
    /// Delaunay verification failed.
    #[error("verification failed")]
    VerificationFailed,
    /// Repair operation validation failed.
    #[error("repair operation failed")]
    RepairOperationFailed,
}

impl From<&DelaunayTriangulationValidationError> for FlipNeighborDelaunayValidationFailureKind {
    fn from(source: &DelaunayTriangulationValidationError) -> Self {
        match source {
            DelaunayTriangulationValidationError::Tds { source: _ } => Self::Tds,
            DelaunayTriangulationValidationError::Triangulation { source: _ } => {
                Self::Triangulation
            }
            DelaunayTriangulationValidationError::Realization { source: _ } => Self::Realization,
            DelaunayTriangulationValidationError::VerificationFailed { .. } => {
                Self::VerificationFailed
            }
            DelaunayTriangulationValidationError::RepairOperationFailed { .. } => {
                Self::RepairOperationFailed
            }
        }
    }
}

impl From<DelaunayTriangulationValidationError> for FlipNeighborDelaunayValidationFailureKind {
    fn from(source: DelaunayTriangulationValidationError) -> Self {
        Self::from(&source)
    }
}

/// Compact repair diagnostics preserved when realization repair failures in flip wiring errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlipNeighborRepairDiagnostics {
    /// Number of queued items checked.
    pub facets_checked: usize,
    /// Number of flips performed.
    pub flips_performed: usize,
    /// Maximum queue length observed.
    pub max_queue_len: usize,
    /// Count of ambiguous predicate evaluations.
    pub ambiguous_predicates: usize,
    /// Count of predicate failures.
    pub predicate_failures: usize,
    /// Count of detected flip cycles.
    pub cycle_detections: usize,
    /// Attempt number.
    pub attempt: usize,
    /// Queue ordering policy used for this attempt.
    pub queue_order: RepairQueueOrder,
}

impl fmt::Display for FlipNeighborRepairDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "checked={}, flips={}, max_queue={}, ambiguous={}, predicate_failures={}, cycles={}, attempt={}, order={:?}",
            self.facets_checked,
            self.flips_performed,
            self.max_queue_len,
            self.ambiguous_predicates,
            self.predicate_failures,
            self.cycle_detections,
            self.attempt,
            self.queue_order
        )
    }
}

impl From<DelaunayRepairDiagnostics> for FlipNeighborRepairDiagnostics {
    fn from(source: DelaunayRepairDiagnostics) -> Self {
        Self {
            facets_checked: source.facets_checked,
            flips_performed: source.flips_performed,
            max_queue_len: source.max_queue_len,
            ambiguous_predicates: source.ambiguous_predicates,
            predicate_failures: source.predicate_failures,
            cycle_detections: source.cycle_detections,
            attempt: source.attempt,
            queue_order: source.queue_order,
        }
    }
}

/// Non-recursive reason Delaunay repair reached flip neighbor wiring.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipNeighborRepairFailure {
    /// Repair did not converge within the flip budget.
    #[error("repair did not converge after {max_flips} flips ({diagnostics})")]
    NonConvergent {
        /// Maximum flips allowed.
        max_flips: usize,
        /// Diagnostics captured during the failed attempt.
        diagnostics: FlipNeighborRepairDiagnostics,
    },
    /// Repair completed but left a Delaunay violation.
    #[error("repair postcondition failed: {reason}")]
    PostconditionFailed {
        /// Structured postcondition failure reason.
        #[source]
        reason: DelaunayRepairPostconditionFailure,
    },
    /// Post-repair verification could not evaluate a local flip predicate.
    #[error("repair verification failed during {context}: {source_kind}")]
    VerificationFailed {
        /// Verification phase that failed.
        context: DelaunayRepairVerificationContext,
        /// Non-recursive class of the underlying flip error.
        source_kind: FlipFailureKind,
    },
    /// Repair completed but orientation canonicalization failed.
    #[error("repair orientation canonicalization failed: {reason}")]
    OrientationCanonicalizationFailed {
        /// Structured canonicalization failure reason.
        reason: DelaunayRepairOrientationCanonicalizationFailureKind,
    },
    /// Flip-based repair is not admissible under the current topology guarantee.
    #[error("repair requires {required:?} topology, found {found:?}: {message}")]
    InvalidTopology {
        /// Required topology guarantee.
        required: TopologyGuarantee,
        /// Actual topology guarantee.
        found: TopologyGuarantee,
        /// Additional context for the mismatch.
        message: &'static str,
    },
    /// Heuristic rebuild failed during advanced repair.
    #[error("heuristic rebuild failed: {reason}")]
    HeuristicRebuildFailed {
        /// Structured rebuild failure category.
        reason: DelaunayRepairHeuristicRebuildFailureKind,
    },
    /// Underlying flip error.
    #[error("flip error: {source_kind}")]
    Flip {
        /// Non-recursive class of the underlying flip error.
        source_kind: FlipFailureKind,
    },
}

/// Structured reason neighbor wiring failed during flip application.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum FlipNeighborWiringError {
    /// Boundary extraction failed before replacement simplices were created.
    #[error("flip boundary extraction failed: {source}")]
    BoundaryExtraction {
        /// Underlying conflict-region error.
        #[source]
        source: ConflictError,
    },
    /// Neighbor wiring failed with a structured insertion-layer reason.
    #[error("neighbor wiring failed: {source}")]
    NeighborWiring {
        /// Underlying neighbor-wiring error.
        #[source]
        source: NeighborWiringError,
    },
    /// The replacement simplices would create a non-manifold facet.
    #[error("non-manifold topology: facet {facet_hash:#x} shared by {simplex_count} simplices")]
    NonManifoldTopology {
        /// Over-shared facet hash.
        facet_hash: u64,
        /// Number of incident simplices.
        simplex_count: usize,
    },
    /// TDS topology validation failed while wiring neighbors.
    #[error("topology validation failed during neighbor wiring: {source}")]
    TopologyValidation {
        /// Underlying TDS validation error.
        #[source]
        source: TdsValidationFailure,
    },
    /// Conflict-region extraction reached flip neighbor wiring.
    #[error("conflict-region error reached flip neighbor wiring: {source}")]
    ConflictRegion {
        /// Underlying conflict-region error.
        #[source]
        source: ConflictError,
    },
    /// Point-location failure reached flip neighbor wiring.
    #[error("point-location error reached flip neighbor wiring: {source}")]
    Location {
        /// Underlying point-location error.
        #[source]
        source: LocateError,
    },
    /// Cavity filling failed while preparing flip neighbor wiring.
    #[error("cavity filling error reached flip neighbor wiring: {reason}")]
    CavityFilling {
        /// Structured cavity-filling reason.
        reason: FlipNeighborCavityFailureKind,
    },
    /// Hull extension failed while preparing flip neighbor wiring.
    #[error("hull extension error reached flip neighbor wiring: {reason}")]
    HullExtension {
        /// Structured hull-extension reason.
        reason: FlipNeighborHullExtensionFailureKind,
    },
    /// Delaunay validation failed while preparing flip neighbor wiring.
    #[error("Delaunay validation error reached flip neighbor wiring: {reason}")]
    DelaunayValidation {
        /// Structured validation reason.
        reason: FlipNeighborDelaunayValidationFailureKind,
    },
    /// Realization validation failed while preparing flip neighbor wiring.
    #[error("realization validation error reached flip neighbor wiring: {source}")]
    RealizationValidation {
        /// Underlying realization validation error, preserving simplex/pair witness context.
        #[source]
        source: TriangulationRealizationValidationError,
    },
    /// Delaunay repair failed while preparing flip neighbor wiring.
    #[error("Delaunay repair error reached flip neighbor wiring: {reason}")]
    DelaunayRepair {
        /// Structured non-recursive repair reason.
        #[source]
        reason: FlipNeighborRepairFailure,
    },
    /// Duplicate coordinates reached flip neighbor wiring.
    #[error("duplicate coordinates reached flip neighbor wiring: {coordinates}")]
    DuplicateCoordinates {
        /// Duplicate coordinate tuple stored as typed coordinate payloads.
        coordinates: CoordinateValues,
    },
    /// Duplicate UUID reached flip neighbor wiring.
    #[error("duplicate UUID reached flip neighbor wiring: {entity:?} {uuid}")]
    DuplicateUuid {
        /// Entity kind.
        entity: EntityKind,
        /// Duplicated UUID.
        uuid: uuid::Uuid,
    },
    /// Level 3 topology validation failed while preparing flip neighbor wiring.
    #[error("topology validation error reached flip neighbor wiring: {context}: {source}")]
    TopologyValidationFailed {
        /// High-level insertion context.
        context: InsertionTopologyValidationContext,
        /// Underlying topology validation error.
        #[source]
        source: TriangulationValidationError,
    },
    /// Local repair would exceed its simplex-removal budget.
    #[error(
        "local repair removal budget reached flip neighbor wiring: attempted {attempted}, max {max_simplices_removed}"
    )]
    MaxSimplicesRemovedExceeded {
        /// Maximum simplices allowed for removal.
        max_simplices_removed: usize,
        /// Number of simplices selected for removal.
        attempted: usize,
    },
    /// Spatial index construction failed before insertion.
    #[error("spatial index construction reached flip neighbor wiring: {reason}")]
    SpatialIndexConstruction {
        /// Structured spatial-index construction failure.
        #[source]
        reason: SpatialIndexConstructionFailure,
    },
    /// Perturbation retry produced invalid coordinates.
    #[error(
        "perturbation retry produced invalid coordinates before flip neighbor wiring: {source}"
    )]
    PerturbedCoordinateInvalid {
        /// Structured coordinate validation failure for the perturbed point.
        #[source]
        source: CoordinateValidationError,
    },
}

impl From<InsertionError> for FlipNeighborWiringError {
    fn from(source: InsertionError) -> Self {
        match source {
            InsertionError::NeighborWiring { reason } => Self::NeighborWiring { source: reason },
            InsertionError::NonManifoldTopology {
                facet_hash,
                simplex_count,
            } => Self::NonManifoldTopology {
                facet_hash,
                simplex_count,
            },
            InsertionError::TopologyValidation { source } => Self::TopologyValidation {
                source: source.into(),
            },
            InsertionError::ConflictRegion { source } => Self::ConflictRegion { source },
            InsertionError::Location { source } => Self::Location { source },
            InsertionError::CavityFilling { reason } => Self::CavityFilling {
                reason: reason.into(),
            },
            InsertionError::HullExtension { reason } => Self::HullExtension {
                reason: reason.into(),
            },
            InsertionError::DelaunayValidationFailed { source } => Self::DelaunayValidation {
                reason: source.into(),
            },
            InsertionError::RealizationValidationFailed { source } => {
                Self::RealizationValidation { source }
            }
            InsertionError::DelaunayRepairFailed { source, context: _ } => Self::DelaunayRepair {
                reason: FlipNeighborRepairFailure::from(*source),
            },
            InsertionError::DuplicateCoordinates { coordinates } => {
                Self::DuplicateCoordinates { coordinates }
            }
            InsertionError::DuplicateUuid { entity, uuid } => Self::DuplicateUuid { entity, uuid },
            InsertionError::TopologyValidationFailed { context, source } => {
                Self::TopologyValidationFailed { context, source }
            }
            InsertionError::MaxSimplicesRemovedExceeded {
                max_simplices_removed,
                attempted,
            } => Self::MaxSimplicesRemovedExceeded {
                max_simplices_removed,
                attempted,
            },
            InsertionError::SpatialIndexConstruction { reason } => {
                Self::SpatialIndexConstruction { reason }
            }
            InsertionError::PerturbedCoordinateInvalid { source } => {
                Self::PerturbedCoordinateInvalid { source }
            }
        }
    }
}

/// Structured reason a TDS mutation failed while applying a flip.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum FlipMutationError {
    /// Vertex insertion failed before a k=1 flip.
    #[error("vertex insertion failed: {source}")]
    VertexInsertion {
        /// Underlying TDS construction error.
        #[source]
        source: TdsConstructionFailure,
    },
    /// Replacement-simplex insertion failed.
    #[error("simplex insertion failed: {source}")]
    SimplexInsertion {
        /// Underlying TDS construction error.
        #[source]
        source: TdsConstructionFailure,
    },
    /// Removed-simplex deletion failed.
    #[error("simplex removal failed: {source}")]
    SimplexRemoval {
        /// Underlying TDS mutation error.
        #[source]
        source: TdsMutationError,
    },
    /// Transactional TDS validation failed before committing a flip.
    #[error(
        "transactional TDS validation failed after bistellar flip (k={k_move}, direction={direction:?}): {source}"
    )]
    TrialValidation {
        /// k for the attempted move.
        k_move: usize,
        /// Direction of the attempted move.
        direction: FlipDirection,
        /// Underlying TDS validation error.
        #[source]
        source: TdsValidationFailure,
    },
    /// Transactional TDS coherent-orientation validation failed before committing a flip.
    ///
    /// This diagnostic is debug/test-only in the flip hot path because it scans
    /// global TDS orientation state. Release-mode callers should use explicit
    /// validation boundaries when they need this invariant checked.
    #[error(
        "transactional TDS coherent orientation invariant violated during {stage:?} (k={k_move}, direction={direction:?})"
    )]
    CoherentOrientationViolation {
        /// Stage where the invariant was checked.
        stage: FlipOrientationCheckStage,
        /// k for the attempted move.
        k_move: usize,
        /// Direction of the attempted move.
        direction: FlipDirection,
    },
}

/// Structured reason inverse k=2 edge adjacency is inconsistent.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipEdgeAdjacencyError {
    /// Edge endpoints are identical.
    #[error("edge endpoints must be distinct ({vertex_key:?})")]
    DuplicateEndpoints {
        /// Repeated endpoint key.
        vertex_key: VertexKey,
    },
    /// Incident simplex does not contain both edge endpoints.
    #[error("simplex {simplex_key:?} does not contain edge vertices {v0:?} and {v1:?}")]
    SimplexMissingEdgeVertices {
        /// Simplex expected to contain the edge.
        simplex_key: SimplexKey,
        /// First edge endpoint.
        v0: VertexKey,
        /// Second edge endpoint.
        v1: VertexKey,
    },
    /// Stored simplex data contains the edge, but the edge is missing from the maintained incidence index.
    #[error("vertex incidence index does not list any simplex containing edge {v0:?}-{v1:?}")]
    MissingEdgeIncidence {
        /// First edge endpoint.
        v0: VertexKey,
        /// Second edge endpoint.
        v1: VertexKey,
    },
    /// A simplex contains an edge endpoint, but that endpoint's incidence index does not list it.
    #[error("vertex incidence index for {vertex_key:?} is missing simplex {simplex_key:?}")]
    MissingVertexIncidence {
        /// Vertex whose incidence list is missing the simplex key.
        vertex_key: VertexKey,
        /// Simplex expected in the vertex's incidence list.
        simplex_key: SimplexKey,
    },
    /// A vertex incidence entry points to a simplex that does not contain that vertex.
    #[error(
        "vertex incidence index for {vertex_key:?} incorrectly references simplex {simplex_key:?}"
    )]
    VertexIncidenceMismatch {
        /// Vertex whose incidence list contains the inconsistent simplex key.
        vertex_key: VertexKey,
        /// Simplex expected to contain the vertex.
        simplex_key: SimplexKey,
    },
    /// Edge star has the wrong opposite-vertex incidence pattern.
    #[error(
        "edge star must have {expected_vertices} distinct opposite vertices each appearing {expected_occurrences} times, found {found_vertices} distinct vertices"
    )]
    InvalidOppositeVertexIncidence {
        /// Expected number of distinct opposite vertices.
        expected_vertices: usize,
        /// Observed number of distinct opposite vertices.
        found_vertices: usize,
        /// Expected occurrence count for each opposite vertex.
        expected_occurrences: usize,
    },
}

/// Structured reason inverse k=3 triangle adjacency is inconsistent.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipTriangleAdjacencyError {
    /// Incident simplex does not contain all triangle vertices.
    #[error("simplex {simplex_key:?} does not contain triangle vertices {a:?}, {b:?}, and {c:?}")]
    SimplexMissingTriangleVertices {
        /// Simplex expected to contain the triangle.
        simplex_key: SimplexKey,
        /// First triangle vertex.
        a: VertexKey,
        /// Second triangle vertex.
        b: VertexKey,
        /// Third triangle vertex.
        c: VertexKey,
    },
    /// Triangle star has the wrong ridge-vertex incidence pattern.
    #[error(
        "triangle star must have {expected_vertices} ridge vertices each appearing {expected_occurrences} times, found {found_vertices} distinct vertices"
    )]
    InvalidRidgeVertexIncidence {
        /// Expected number of distinct ridge vertices.
        expected_vertices: usize,
        /// Observed number of distinct ridge vertices.
        found_vertices: usize,
        /// Expected occurrence count for each ridge vertex.
        expected_occurrences: usize,
    },
}

/// Error returned when constructing a [`TriangleHandle`] from invalid vertices.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TriangleHandleError {
    /// At least two triangle vertices refer to the same vertex.
    #[error("triangle vertices must be distinct, got {vertices:?}")]
    DuplicateVertices {
        /// The supplied triangle vertices.
        vertices: [VertexKey; 3],
    },
}

/// Structured reason inverse k=1 vertex-star adjacency is inconsistent.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlipVertexAdjacencyError {
    /// Incident simplex does not contain the removed vertex.
    #[error("simplex {simplex_key:?} does not contain vertex {vertex_key:?}")]
    SimplexMissingVertex {
        /// Simplex expected to contain the vertex.
        simplex_key: SimplexKey,
        /// Removed vertex.
        vertex_key: VertexKey,
    },
    /// Vertex star has the wrong link-vertex incidence pattern.
    #[error(
        "vertex star must have {expected_vertices} link vertices each appearing {expected_occurrences} times, found {found_vertices} distinct vertices"
    )]
    InvalidLinkVertexIncidence {
        /// Expected number of distinct link vertices.
        expected_vertices: usize,
        /// Observed number of distinct link vertices.
        found_vertices: usize,
        /// Expected occurrence count for each link vertex.
        expected_occurrences: usize,
    },
}

/// Errors that can occur during bistellar flips or repair.
///
/// The enum keeps small scalar, key, and short [`Vec`] diagnostics inline, but
/// boxes nested typed error payloads and exposes them as `#[source]` values.
/// Constructors and pattern matches for those variants use [`Box`], while typed
/// inspection remains available through `reason.as_ref()`, `source.as_ref()`,
/// [`Error::source`](std::error::Error::source), or the boxed [`SmallBuffer`]
/// witness directly without string parsing.
///
/// # Examples
///
/// ```rust
/// use delaunay::flips::{FlipContextError, FlipError};
///
/// let err = FlipError::UnsupportedDimension { dimension: 1 };
/// std::assert_matches!(err, FlipError::UnsupportedDimension { .. });
///
/// let err = FlipError::InvalidFlipContext {
///     reason: Box::new(FlipContextError::OverlappingFaces),
/// };
/// std::assert_matches!(
///     err,
///     FlipError::InvalidFlipContext { reason }
///         if matches!(reason.as_ref(), FlipContextError::OverlappingFaces)
/// );
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum FlipError {
    /// A [`PachnerProposal`](crate::pachner::PachnerProposal) was minted from a
    /// different owner.
    #[error("Topology proposal owner mismatch: expected {expected:?}, found {found:?}")]
    WrongTopologyOwner {
        /// Owner identity of the triangulation receiving the proposal.
        expected: TopologyOwnerId,
        /// Owner identity stored in the detached proposal.
        found: TopologyOwnerId,
    },
    /// A [`PachnerProposal`](crate::pachner::PachnerProposal) was minted from an
    /// older structural generation.
    #[error(
        "Topology proposal generation {proposal_generation} is stale for current generation {current_generation}"
    )]
    StaleTopologyProposal {
        /// Structural generation stored in the detached proposal.
        proposal_generation: u64,
        /// Current structural generation of the target triangulation.
        current_generation: u64,
    },
    /// Flips are not supported for this dimension.
    #[error("Bistellar flip not supported for D={dimension}")]
    UnsupportedDimension {
        /// Dimension of the triangulation.
        dimension: usize,
    },
    /// The requested flip lacks the PL-manifold proof required for mutation.
    #[error("Bistellar flip requires {required:?} topology, found {found:?}")]
    FlipTopologyNotAdmissible {
        /// Minimum topology proof required by the flip class.
        required: TopologyGuarantee,
        /// Topology proof carried by the target triangulation.
        found: TopologyGuarantee,
    },
    /// The facet is on the boundary (no adjacent simplex).
    #[error("Facet {facet:?} is on the boundary (no neighbor)")]
    BoundaryFacet {
        /// Facet handle.
        facet: FacetHandle,
    },
    /// The referenced simplex was not found.
    #[error("Simplex not found: {simplex_key:?}")]
    MissingSimplex {
        /// Missing simplex key.
        simplex_key: SimplexKey,
    },
    /// The vertex-to-simplices incidence index references a missing simplex.
    #[error("Vertex incidence index for {vertex_key:?} references missing simplex {simplex_key:?}")]
    DanglingVertexIncidence {
        /// Vertex whose incidence list contains the dangling simplex key.
        vertex_key: VertexKey,
        /// Missing simplex key referenced by the incidence index.
        simplex_key: SimplexKey,
    },
    /// The referenced vertex was not found.
    #[error("Vertex not found: {vertex_key:?}")]
    MissingVertex {
        /// Missing vertex key.
        vertex_key: VertexKey,
    },
    /// The neighbor simplex across the facet is missing.
    #[error("Neighbor simplex {neighbor_key:?} not found for facet {facet:?}")]
    MissingNeighbor {
        /// Facet handle.
        facet: FacetHandle,
        /// Missing neighbor key.
        neighbor_key: SimplexKey,
    },
    /// Ridge adjacency references a neighbor simplex key that is no longer live.
    #[error(
        "Ridge adjacency from simplex {simplex_key:?} references missing neighbor {neighbor_key:?}"
    )]
    DanglingRidgeNeighbor {
        /// Simplex whose neighbor table contains the dangling key.
        simplex_key: SimplexKey,
        /// Missing neighbor simplex key.
        neighbor_key: SimplexKey,
    },
    /// Facet adjacency information is inconsistent.
    #[error(
        "Facet adjacency mismatch between simplex {simplex_key:?} and neighbor {neighbor_key:?}"
    )]
    InvalidFacetAdjacency {
        /// Simplex key.
        simplex_key: SimplexKey,
        /// Neighbor simplex key.
        neighbor_key: SimplexKey,
    },
    /// The facet index is out of bounds for the simplex.
    #[error(
        "Facet index {facet_index} out of bounds for simplex {simplex_key:?} with {vertex_count} vertices"
    )]
    InvalidFacetIndex {
        /// Simplex key.
        simplex_key: SimplexKey,
        /// Facet index.
        facet_index: u8,
        /// Vertex count for the simplex.
        vertex_count: usize,
    },
    /// Ridge indices are invalid for the simplex.
    #[error(
        "Ridge indices ({omit_a}, {omit_b}) out of bounds for simplex {simplex_key:?} with {vertex_count} vertices"
    )]
    InvalidRidgeIndex {
        /// Simplex key.
        simplex_key: SimplexKey,
        /// First omitted index.
        omit_a: u8,
        /// Second omitted index.
        omit_b: u8,
        /// Vertex count for the simplex.
        vertex_count: usize,
    },
    /// Ridge adjacency information is inconsistent.
    #[error("Ridge adjacency mismatch for simplex {simplex_key:?}")]
    InvalidRidgeAdjacency {
        /// Simplex key.
        simplex_key: SimplexKey,
    },
    /// Ridge has an invalid multiplicity for k=3 flips.
    #[error("Ridge has invalid multiplicity {found}, expected 3")]
    InvalidRidgeMultiplicity {
        /// Number of incident simplices found.
        found: usize,
    },
    /// Edge has an invalid multiplicity for inverse k=2 flips.
    #[error("Edge has invalid multiplicity {found}, expected {expected}")]
    InvalidEdgeMultiplicity {
        /// Number of incident simplices found.
        found: usize,
        /// Expected multiplicity for the dimension.
        expected: usize,
    },
    /// Triangle has an invalid multiplicity for inverse k=3 flips.
    #[error("Triangle has invalid multiplicity {found}, expected {expected}")]
    InvalidTriangleMultiplicity {
        /// Number of incident simplices found.
        found: usize,
        /// Expected multiplicity for the dimension.
        expected: usize,
    },
    /// Edge adjacency information is inconsistent.
    #[error("Edge adjacency mismatch: {reason}")]
    InvalidEdgeAdjacency {
        /// Structured edge-adjacency reason.
        #[source]
        reason: Box<FlipEdgeAdjacencyError>,
    },
    /// Triangle adjacency information is inconsistent.
    #[error("Triangle adjacency mismatch: {reason}")]
    InvalidTriangleAdjacency {
        /// Structured triangle-adjacency reason.
        #[source]
        reason: Box<FlipTriangleAdjacencyError>,
    },
    /// Vertex star has an invalid multiplicity for inverse k=1 flips.
    #[error("Vertex star has invalid multiplicity {found}, expected {expected}")]
    InvalidVertexMultiplicity {
        /// Number of incident simplices found.
        found: usize,
        /// Expected multiplicity for the dimension.
        expected: usize,
    },
    /// Vertex adjacency information is inconsistent.
    #[error("Vertex adjacency mismatch: {reason}")]
    InvalidVertexAdjacency {
        /// Structured vertex-adjacency reason.
        #[source]
        reason: Box<FlipVertexAdjacencyError>,
    },
    /// Flip context is inconsistent with the requested move.
    #[error("Flip context invalid: {reason}")]
    InvalidFlipContext {
        /// Structured invalid-context reason.
        #[source]
        reason: Box<FlipContextError>,
    },
    /// Geometric predicate failed.
    #[error("Geometric predicate failed: {reason}")]
    PredicateFailure {
        /// Structured predicate failure.
        #[source]
        reason: Box<FlipPredicateError>,
    },
    /// Flip would create a degenerate simplex (zero orientation).
    #[error("Flip would create a degenerate simplex (zero orientation)")]
    DegenerateSimplex,
    /// A forward k=1 insertion point is outside the selected simplex in its active chart.
    #[error(
        "Proposed k=1 vertex lies outside simplex {simplex_key:?} across facet slot {opposite_vertex_index} opposite {opposite_vertex:?}"
    )]
    K1InsertionOutsideSimplex {
        /// Simplex selected for subdivision.
        simplex_key: SimplexKey,
        /// Vertex opposite a facet across which the proposed point lies outside.
        opposite_vertex: VertexKey,
        /// Vertex slot whose opposite facet the proposed point crosses.
        opposite_vertex_index: usize,
    },
    /// Delaunay repair would create a negative-orientation replacement simplex.
    #[error(
        "Delaunay repair would create a negative-orientation replacement simplex {simplex_vertices:?}"
    )]
    NegativeOrientation {
        /// Replacement simplex vertices in the rejected order.
        simplex_vertices: Vec<VertexKey>,
    },
    /// Flip would create a duplicate simplex.
    #[error("Flip would create a duplicate simplex")]
    DuplicateSimplex,
    /// Flip would create a non-manifold facet.
    #[error("Flip would create a non-manifold facet")]
    NonManifoldFacet,
    /// Flip would insert a simplex that already exists in the triangulation.
    ///
    /// This violates the bistellar move link condition and can create non-manifold
    /// codimension>1 singularities (e.g., disconnected ridge links).
    #[error(
        "Flip would insert simplex that already exists (k={k_move}, simplex={simplex_vertices:?}, existing_simplex={existing_simplex:?})"
    )]
    InsertedSimplexAlreadyExists {
        /// k for the attempted move.
        k_move: usize,
        /// Vertex keys of the inserted simplex.
        simplex_vertices: Box<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
        /// A witness simplex key that already contains the inserted simplex.
        existing_simplex: SimplexKey,
    },
    /// Facet iteration failed while seeding flip or repair work.
    #[error("Facet iteration failed: {source}")]
    FacetIteration {
        /// Structured facet iteration failure.
        #[source]
        source: Box<FacetError>,
    },
    /// Simplex creation failed.
    #[error(transparent)]
    SimplexCreation {
        /// Boxed simplex-validation source error.
        #[from]
        source: Box<SimplexValidationError>,
    },
    /// Flip transaction could not repair post-mutation orientation invariants.
    #[error("Flip postcondition orientation repair failed: {source}")]
    PostconditionRepair {
        /// Structured orientation-repair failure.
        #[source]
        source: Box<InsertionError>,
    },
    /// Flip transaction failed realized-geometry validation after mutation.
    #[error("Flip postcondition realization validation failed: {source}")]
    RealizationValidation {
        /// Structured Level 4 realization validation error.
        #[source]
        source: Box<TriangulationRealizationValidationError>,
    },
    /// Neighbor wiring failed during flip application.
    #[error("Neighbor wiring failed: {reason}")]
    NeighborWiring {
        /// Structured neighbor-wiring failure.
        #[source]
        reason: Box<FlipNeighborWiringError>,
    },
    /// TDS mutation failed.
    #[error("TDS mutation failed: {reason}")]
    TdsMutation {
        /// Structured TDS mutation failure.
        #[source]
        reason: Box<FlipMutationError>,
    },
}

impl From<FlipContextError> for FlipError {
    fn from(reason: FlipContextError) -> Self {
        Self::InvalidFlipContext {
            reason: Box::new(reason),
        }
    }
}

impl From<FlipPredicateError> for FlipError {
    fn from(reason: FlipPredicateError) -> Self {
        Self::PredicateFailure {
            reason: Box::new(reason),
        }
    }
}

impl From<FlipEdgeAdjacencyError> for FlipError {
    fn from(reason: FlipEdgeAdjacencyError) -> Self {
        Self::InvalidEdgeAdjacency {
            reason: Box::new(reason),
        }
    }
}

impl From<FlipTriangleAdjacencyError> for FlipError {
    fn from(reason: FlipTriangleAdjacencyError) -> Self {
        Self::InvalidTriangleAdjacency {
            reason: Box::new(reason),
        }
    }
}

impl From<FlipVertexAdjacencyError> for FlipError {
    fn from(reason: FlipVertexAdjacencyError) -> Self {
        Self::InvalidVertexAdjacency {
            reason: Box::new(reason),
        }
    }
}

impl From<SimplexValidationError> for FlipError {
    fn from(source: SimplexValidationError) -> Self {
        Self::SimplexCreation {
            source: Box::new(source),
        }
    }
}

impl From<FacetError> for FlipError {
    fn from(source: FacetError) -> Self {
        Self::FacetIteration {
            source: Box::new(source),
        }
    }
}

impl From<FlipNeighborWiringError> for FlipError {
    fn from(reason: FlipNeighborWiringError) -> Self {
        Self::NeighborWiring {
            reason: Box::new(reason),
        }
    }
}

impl From<FlipMutationError> for FlipError {
    fn from(reason: FlipMutationError) -> Self {
        Self::TdsMutation {
            reason: Box::new(reason),
        }
    }
}

impl From<&FlipError> for FlipFailureKind {
    fn from(source: &FlipError) -> Self {
        match source {
            FlipError::WrongTopologyOwner { .. } => Self::WrongTopologyOwner,
            FlipError::StaleTopologyProposal { .. } => Self::StaleTopologyProposal,
            FlipError::UnsupportedDimension { .. } => Self::UnsupportedDimension,
            FlipError::FlipTopologyNotAdmissible { .. } => Self::FlipTopologyNotAdmissible,
            FlipError::BoundaryFacet { .. } => Self::BoundaryFacet,
            FlipError::MissingSimplex { .. } => Self::MissingSimplex,
            FlipError::DanglingVertexIncidence { .. } => Self::DanglingVertexIncidence,
            FlipError::MissingVertex { .. } => Self::MissingVertex,
            FlipError::MissingNeighbor { .. } => Self::MissingNeighbor,
            FlipError::DanglingRidgeNeighbor { .. } => Self::DanglingRidgeNeighbor,
            FlipError::InvalidFacetAdjacency { .. } => Self::InvalidFacetAdjacency,
            FlipError::InvalidFacetIndex { .. } => Self::InvalidFacetIndex,
            FlipError::InvalidRidgeIndex { .. } => Self::InvalidRidgeIndex,
            FlipError::InvalidRidgeAdjacency { .. } => Self::InvalidRidgeAdjacency,
            FlipError::InvalidRidgeMultiplicity { .. } => Self::InvalidRidgeMultiplicity,
            FlipError::InvalidEdgeMultiplicity { .. } => Self::InvalidEdgeMultiplicity,
            FlipError::InvalidTriangleMultiplicity { .. } => Self::InvalidTriangleMultiplicity,
            FlipError::InvalidEdgeAdjacency { .. } => Self::InvalidEdgeAdjacency,
            FlipError::InvalidTriangleAdjacency { .. } => Self::InvalidTriangleAdjacency,
            FlipError::InvalidVertexMultiplicity { .. } => Self::InvalidVertexMultiplicity,
            FlipError::InvalidVertexAdjacency { .. } => Self::InvalidVertexAdjacency,
            FlipError::InvalidFlipContext { .. } => Self::InvalidFlipContext,
            FlipError::PredicateFailure { .. } => Self::PredicateFailure,
            FlipError::DegenerateSimplex => Self::DegenerateSimplex,
            FlipError::K1InsertionOutsideSimplex { .. } => Self::K1InsertionOutsideSimplex,
            FlipError::NegativeOrientation { .. } => Self::NegativeOrientation,
            FlipError::DuplicateSimplex => Self::DuplicateSimplex,
            FlipError::NonManifoldFacet => Self::NonManifoldFacet,
            FlipError::InsertedSimplexAlreadyExists { .. } => Self::InsertedSimplexAlreadyExists,
            FlipError::FacetIteration { .. } => Self::FacetIteration,
            FlipError::SimplexCreation { source: _ } => Self::SimplexCreation,
            FlipError::PostconditionRepair { .. } => Self::PostconditionRepair,
            FlipError::RealizationValidation { .. } => Self::RealizationValidation,
            FlipError::NeighborWiring { reason } => match reason.as_ref() {
                FlipNeighborWiringError::TopologyValidation { .. }
                | FlipNeighborWiringError::DelaunayValidation { .. }
                | FlipNeighborWiringError::RealizationValidation { .. }
                | FlipNeighborWiringError::TopologyValidationFailed { .. } => {
                    Self::WiringValidation
                }
                FlipNeighborWiringError::DelaunayRepair { .. } => Self::DelaunayRepairFailed,
                _ => Self::NeighborWiring,
            },
            FlipError::TdsMutation { reason }
                if matches!(
                    reason.as_ref(),
                    FlipMutationError::TrialValidation { .. }
                        | FlipMutationError::CoherentOrientationViolation { .. }
                ) =>
            {
                Self::TrialValidation
            }
            FlipError::TdsMutation { .. } => Self::TdsMutation,
        }
    }
}

impl From<FlipError> for FlipFailureKind {
    fn from(source: FlipError) -> Self {
        Self::from(&source)
    }
}
/// Queue ordering policy for flip repair attempts.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::RepairQueueOrder;
///
/// let order = RepairQueueOrder::Fifo;
/// assert_eq!(order, RepairQueueOrder::Fifo);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairQueueOrder {
    /// FIFO (breadth-like) ordering.
    Fifo,
    /// LIFO (depth-like) ordering.
    Lifo,
}

/// Diagnostics captured when flip-based repair fails to converge.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::{
///     DelaunayRepairDiagnostics, RepairQueueOrder,
/// };
///
/// let diagnostics = DelaunayRepairDiagnostics {
///     facets_checked: 0,
///     flips_performed: 0,
///     max_queue_len: 0,
///     ambiguous_predicates: 0,
///     ambiguous_predicate_samples: Vec::new(),
///     predicate_failures: 0,
///     cycle_detections: 0,
///     cycle_signature_samples: Vec::new(),
///     attempt: 1,
///     queue_order: RepairQueueOrder::Fifo,
/// };
/// assert!(diagnostics.to_string().contains("checked"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelaunayRepairDiagnostics {
    /// Number of queued items checked.
    pub facets_checked: usize,
    /// Number of flips performed.
    pub flips_performed: usize,
    /// Maximum queue length observed.
    pub max_queue_len: usize,
    /// Count of ambiguous predicate evaluations (boundary classifications).
    pub ambiguous_predicates: usize,
    /// Sample of ambiguous predicate site hashes (deterministic, truncated).
    pub ambiguous_predicate_samples: Vec<u64>,
    /// Count of predicate failures (conversion/robust fallback errors).
    pub predicate_failures: usize,
    /// Count of detected flip cycles (repeat flip signatures within a sliding window).
    pub cycle_detections: usize,
    /// Sample of repeated flip-context signature hashes (deterministic, truncated).
    pub cycle_signature_samples: Vec<u64>,
    /// Attempt number (1-based).
    pub attempt: usize,
    /// Queue ordering policy used for this attempt.
    pub queue_order: RepairQueueOrder,
}

impl fmt::Display for DelaunayRepairDiagnostics {
    /// Format a concise diagnostics summary.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "checked {} facets, ambiguous={}, max_queue={}, flips={}, attempt={}, order={:?}, predicate_failures={}, cycles={}, cycle_samples={:?}",
            self.facets_checked,
            self.ambiguous_predicates,
            self.max_queue_len,
            self.flips_performed,
            self.attempt,
            self.queue_order,
            self.predicate_failures,
            self.cycle_detections,
            self.cycle_signature_samples
        )
    }
}

/// Verification phase that failed during flip-based Delaunay repair.
///
/// This context is carried by [`DelaunayRepairError::VerificationFailed`] so
/// callers can distinguish generic post-repair validation from local k=2/k=3
/// postcondition checks without parsing the display message.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::{
///     DelaunayRepairError, DelaunayRepairVerificationContext, FlipError,
/// };
///
/// let err = DelaunayRepairError::VerificationFailed {
///     context: DelaunayRepairVerificationContext::StrictValidation,
///     source: Box::new(FlipError::DegenerateSimplex),
/// };
///
/// std::assert_matches!(
///     err,
///     DelaunayRepairError::VerificationFailed {
///         context: DelaunayRepairVerificationContext::StrictValidation,
///         ..
///     },
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairVerificationContext {
    /// Generic post-repair verification.
    PostRepairVerification,
    /// Strict validation pass.
    StrictValidation,
    /// Local k=2 degeneracy verification.
    LocalK2DegeneracyVerification,
    /// Local k=2 postcondition verification.
    LocalK2PostconditionVerification,
    /// Local k=3 degeneracy verification.
    LocalK3DegeneracyVerification,
    /// Local k=3 postcondition verification.
    LocalK3PostconditionVerification,
    /// Local inverse k=2 postcondition verification.
    LocalInverseK2PostconditionVerification,
    /// Local inverse k=3 postcondition verification.
    LocalInverseK3PostconditionVerification,
}

impl fmt::Display for DelaunayRepairVerificationContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PostRepairVerification => f.write_str("post-repair verification"),
            Self::StrictValidation => f.write_str("strict validation"),
            Self::LocalK2DegeneracyVerification => f.write_str("local k=2 degeneracy verification"),
            Self::LocalK2PostconditionVerification => {
                f.write_str("local k=2 postcondition verification")
            }
            Self::LocalK3DegeneracyVerification => f.write_str("local k=3 degeneracy verification"),
            Self::LocalK3PostconditionVerification => {
                f.write_str("local k=3 postcondition verification")
            }
            Self::LocalInverseK2PostconditionVerification => {
                f.write_str("local inverse k=2 postcondition verification")
            }
            Self::LocalInverseK3PostconditionVerification => {
                f.write_str("local inverse k=3 postcondition verification")
            }
        }
    }
}

/// Structured reason a repair pass failed its postcondition.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairPostconditionFailure {
    /// Repair disconnected the triangulation neighbor graph.
    Disconnected {
        /// Number of simplices remaining when the disconnected graph was detected.
        simplex_count: usize,
    },
    /// A local k=2 facet flip opportunity remained after repair.
    LocalK2Violation {
        /// Facet whose flip predicate still reports a violation.
        facet: FacetHandle,
        /// Optional opt-in diagnostic details captured under repair debug flags.
        debug_details: Option<String>,
    },
    /// A local k=3 ridge flip opportunity remained after repair.
    LocalK3Violation {
        /// Ridge whose flip predicate still reports a violation.
        ridge: RidgeHandle,
    },
    /// A local inverse k=2 edge-collapse opportunity remained after repair.
    LocalInverseK2Violation {
        /// Edge whose inverse flip predicate still reports a violation.
        edge: EdgeKey,
    },
    /// A local inverse k=3 triangle-collapse opportunity remained after repair.
    LocalInverseK3Violation {
        /// Triangle whose inverse flip predicate still reports a violation.
        triangle: TriangleHandle,
    },
}

impl fmt::Display for DelaunayRepairPostconditionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disconnected { simplex_count } => write!(
                f,
                "repair pass disconnected the triangulation ({simplex_count} simplices remain); neighbor wiring is incomplete"
            ),
            Self::LocalK2Violation {
                facet,
                debug_details,
            } => {
                write!(
                    f,
                    "local k=2 violation remains after repair (facet={facet:?})"
                )?;
                if let Some(details) = debug_details {
                    write!(f, "; {details}")?;
                }
                Ok(())
            }
            Self::LocalK3Violation { ridge } => {
                write!(
                    f,
                    "local k=3 violation remains after repair (ridge={ridge:?})"
                )
            }
            Self::LocalInverseK2Violation { edge } => {
                write!(
                    f,
                    "local inverse k=2 flip remains applicable after repair (edge={edge:?})"
                )
            }
            Self::LocalInverseK3Violation { triangle } => write!(
                f,
                "local inverse k=3 flip remains applicable after repair (triangle={triangle:?})"
            ),
        }
    }
}

impl std::error::Error for DelaunayRepairPostconditionFailure {}

/// Structured reason orientation canonicalization failed after repair.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayRepairOrientationCanonicalizationFailure {
    /// Positive-orientation promotion failed after a flip-repair pass.
    #[error("after flip repair: {source}")]
    AfterFlipRepair {
        /// Insertion-layer failure produced by orientation promotion.
        #[source]
        source: Box<InsertionError>,
    },
}

/// Compact orientation-canonicalization failure category for non-recursive summaries.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairOrientationCanonicalizationFailureKind {
    /// Positive-orientation promotion failed after a flip-repair pass.
    #[error("after flip repair: {source_kind:?}")]
    AfterFlipRepair {
        /// Category of the insertion-layer failure.
        source_kind: InsertionErrorKind,
    },
}

impl From<&DelaunayRepairOrientationCanonicalizationFailure>
    for DelaunayRepairOrientationCanonicalizationFailureKind
{
    fn from(source: &DelaunayRepairOrientationCanonicalizationFailure) -> Self {
        match source {
            DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair { source } => {
                Self::AfterFlipRepair {
                    source_kind: insertion_error_kind(source),
                }
            }
        }
    }
}

/// Passive vertex context reported with heuristic rebuild failures.
///
/// The fields identify the vertex position, UUID, and coordinates that were
/// being replayed when rebuild failed. This is diagnostic context only; repair
/// algorithms do not accept it back as proof of a valid vertex.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct DelaunayRepairHeuristicVertexContext {
    /// Position of the vertex in the shuffled rebuild order.
    pub index: usize,
    /// Stable vertex UUID.
    pub vertex_uuid: uuid::Uuid,
    /// Vertex coordinates at the rebuild boundary.
    pub coordinates: CoordinateValues,
}

impl fmt::Display for DelaunayRepairHeuristicVertexContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "idx={} uuid={} coords={}",
            self.index, self.vertex_uuid, self.coordinates
        )
    }
}

/// Structured reason heuristic rebuild failed during advanced repair.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayRepairHeuristicRebuildFailure {
    /// Heuristic rebuild recursion exceeded its guard depth.
    #[error("heuristic rebuild recursion depth exceeded {max_depth}")]
    RecursionDepthExceeded {
        /// Maximum permitted nested heuristic rebuild depth.
        max_depth: usize,
    },
    /// Primary repair, robust fallback, and heuristic rebuild all failed.
    #[error("primary repair failed ({primary}); robust fallback failed ({robust}); {heuristic}")]
    FallbackChainFailed {
        /// Primary flip-repair failure.
        #[source]
        primary: Box<DelaunayRepairError>,
        /// Robust-kernel fallback failure.
        robust: Box<DelaunayRepairError>,
        /// Heuristic rebuild failure.
        heuristic: Box<Self>,
    },
    /// A non-heuristic repair error escaped the heuristic rebuild path.
    #[error("heuristic rebuild failed with unexpected repair error: {source}")]
    UnexpectedRepairFailure {
        /// Repair error returned by the heuristic path.
        #[source]
        source: Box<DelaunayRepairError>,
    },
    /// The attempt loop exited without recording a rebuild attempt.
    #[error("heuristic rebuild made no attempts")]
    NoAttempts,
    /// Vertex insertion failed during heuristic rebuild.
    #[error("heuristic rebuild insertion failed at {vertex}: {source}")]
    InsertionFailed {
        /// Vertex being inserted.
        vertex: DelaunayRepairHeuristicVertexContext,
        /// Insertion failure.
        #[source]
        source: Box<InsertionError>,
    },
    /// Local repair failed after a heuristic rebuild insertion.
    #[error("heuristic rebuild repair failed at {vertex}: {source}")]
    RepairFailed {
        /// Vertex whose insertion triggered repair.
        vertex: DelaunayRepairHeuristicVertexContext,
        /// Insertion-layer repair failure.
        #[source]
        source: Box<InsertionError>,
    },
    /// Delaunay check failed after a heuristic rebuild insertion.
    #[error("heuristic rebuild Delaunay check failed at {vertex}: {source}")]
    DelaunayCheckFailed {
        /// Vertex whose insertion triggered the check.
        vertex: DelaunayRepairHeuristicVertexContext,
        /// Insertion-layer check failure.
        #[source]
        source: Box<InsertionError>,
    },
    /// A vertex was skipped during heuristic rebuild.
    #[error("heuristic rebuild skipped vertex at {vertex}: {source}")]
    SkippedVertex {
        /// Skipped vertex.
        vertex: DelaunayRepairHeuristicVertexContext,
        /// Insertion-layer skip reason.
        #[source]
        source: Box<InsertionError>,
    },
    /// One deterministic rebuild attempt failed.
    #[error(
        "attempt {attempt}/{max_attempts} (shuffle_seed={shuffle_seed} perturbation_seed={perturbation_seed}): {source}"
    )]
    AttemptFailed {
        /// 1-based attempt number.
        attempt: usize,
        /// Maximum number of attempts.
        max_attempts: usize,
        /// Shuffle seed used for this attempt.
        shuffle_seed: u64,
        /// Perturbation seed used for this attempt.
        perturbation_seed: u64,
        /// Attempt failure.
        #[source]
        source: Box<DelaunayRepairError>,
    },
    /// Every deterministic heuristic rebuild attempt failed.
    #[error("heuristic rebuild failed after {attempts} attempts: {last_failure}")]
    ExhaustedAttempts {
        /// Number of attempts tried.
        attempts: usize,
        /// Last observed attempt failure.
        #[source]
        last_failure: Box<Self>,
    },
}

/// Compact heuristic-rebuild failure category for non-recursive summaries.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelaunayRepairHeuristicRebuildFailureKind {
    /// Heuristic rebuild recursion exceeded its guard depth.
    #[error("recursion depth exceeded")]
    RecursionDepthExceeded,
    /// Primary repair, robust fallback, and heuristic rebuild all failed.
    #[error("fallback chain failed")]
    FallbackChainFailed,
    /// A non-heuristic repair error escaped the heuristic rebuild path.
    #[error("unexpected repair failure")]
    UnexpectedRepairFailure,
    /// The attempt loop exited without recording a rebuild attempt.
    #[error("no attempts")]
    NoAttempts,
    /// Vertex insertion failed during heuristic rebuild.
    #[error("insertion failed")]
    InsertionFailed,
    /// Local repair failed after a heuristic rebuild insertion.
    #[error("repair failed")]
    RepairFailed,
    /// Delaunay check failed after a heuristic rebuild insertion.
    #[error("Delaunay check failed")]
    DelaunayCheckFailed,
    /// A vertex was skipped during heuristic rebuild.
    #[error("skipped vertex")]
    SkippedVertex,
    /// One deterministic rebuild attempt failed.
    #[error("attempt failed")]
    AttemptFailed,
    /// Every deterministic heuristic rebuild attempt failed.
    #[error("attempts exhausted")]
    ExhaustedAttempts,
}

impl From<&DelaunayRepairHeuristicRebuildFailure> for DelaunayRepairHeuristicRebuildFailureKind {
    fn from(source: &DelaunayRepairHeuristicRebuildFailure) -> Self {
        match source {
            DelaunayRepairHeuristicRebuildFailure::RecursionDepthExceeded { .. } => {
                Self::RecursionDepthExceeded
            }
            DelaunayRepairHeuristicRebuildFailure::FallbackChainFailed { .. } => {
                Self::FallbackChainFailed
            }
            DelaunayRepairHeuristicRebuildFailure::UnexpectedRepairFailure { .. } => {
                Self::UnexpectedRepairFailure
            }
            DelaunayRepairHeuristicRebuildFailure::NoAttempts => Self::NoAttempts,
            DelaunayRepairHeuristicRebuildFailure::InsertionFailed { .. } => Self::InsertionFailed,
            DelaunayRepairHeuristicRebuildFailure::RepairFailed { .. } => Self::RepairFailed,
            DelaunayRepairHeuristicRebuildFailure::DelaunayCheckFailed { .. } => {
                Self::DelaunayCheckFailed
            }
            DelaunayRepairHeuristicRebuildFailure::SkippedVertex { .. } => Self::SkippedVertex,
            DelaunayRepairHeuristicRebuildFailure::AttemptFailed { .. } => Self::AttemptFailed,
            DelaunayRepairHeuristicRebuildFailure::ExhaustedAttempts { .. } => {
                Self::ExhaustedAttempts
            }
        }
    }
}

pub(super) const fn insertion_error_kind(source: &InsertionError) -> InsertionErrorKind {
    match source {
        InsertionError::ConflictRegion { source: _ } => InsertionErrorKind::ConflictRegion,
        InsertionError::Location { source: _ } => InsertionErrorKind::Location,
        InsertionError::CavityFilling { .. } => InsertionErrorKind::CavityFilling,
        InsertionError::NeighborWiring { .. } => InsertionErrorKind::NeighborWiring,
        InsertionError::NonManifoldTopology { .. } => InsertionErrorKind::NonManifoldTopology,
        InsertionError::HullExtension { .. } => InsertionErrorKind::HullExtension,
        InsertionError::DelaunayValidationFailed { .. } => {
            InsertionErrorKind::DelaunayValidationFailed
        }
        InsertionError::RealizationValidationFailed { .. } => {
            InsertionErrorKind::RealizationValidationFailed
        }
        InsertionError::DelaunayRepairFailed { .. } => InsertionErrorKind::DelaunayRepairFailed,
        InsertionError::DuplicateCoordinates { .. } => InsertionErrorKind::DuplicateCoordinates,
        InsertionError::DuplicateUuid { .. } => InsertionErrorKind::DuplicateUuid,
        InsertionError::TopologyValidation { source: _ } => InsertionErrorKind::TopologyValidation,
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
    }
}

/// Errors that can occur during flip-based Delaunay repair.
///
/// Large typed payloads are boxed to keep the public enum small and cheap to
/// move, while scalar fields and short diagnostic strings stay inline. Boxed
/// variants still preserve their concrete source type so callers can inspect
/// or pattern-match the full error chain when they need repair diagnostics.
/// For example, [`DelaunayRepairError::NonConvergent`] boxes
/// [`DelaunayRepairDiagnostics`], and [`DelaunayRepairError::Flip`] boxes the
/// underlying [`FlipError`].
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::{DelaunayRepairError, FlipError, TopologyGuarantee};
///
/// let err = DelaunayRepairError::InvalidTopology {
///     required: TopologyGuarantee::PLManifold,
///     found: TopologyGuarantee::Pseudomanifold,
///     message: "requires manifold",
/// };
/// std::assert_matches!(err, DelaunayRepairError::InvalidTopology { .. });
///
/// let flip_err = DelaunayRepairError::from(FlipError::DegenerateSimplex);
/// std::assert_matches!(
///     flip_err,
///     DelaunayRepairError::Flip { source }
///         if matches!(source.as_ref(), FlipError::DegenerateSimplex)
/// );
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayRepairError {
    /// Repair did not converge within the flip budget.
    #[error("Delaunay repair failed to converge after {max_flips} flips ({diagnostics})")]
    NonConvergent {
        /// Maximum flips allowed.
        max_flips: usize,
        /// Diagnostics captured during the failed attempt (boxed to keep the
        /// error enum small on the stack).
        diagnostics: Box<DelaunayRepairDiagnostics>,
    },
    /// Repair completed but left a Delaunay violation.
    #[error("Delaunay repair postcondition failed: {reason}")]
    PostconditionFailed {
        /// Structured postcondition failure reason.
        #[source]
        reason: Box<DelaunayRepairPostconditionFailure>,
    },
    /// Post-repair verification could not evaluate a local flip predicate.
    #[error("Delaunay repair verification failed during {context}: {source}")]
    VerificationFailed {
        /// Verification phase that failed.
        context: DelaunayRepairVerificationContext,
        /// Underlying flip or predicate error.
        #[source]
        source: Box<FlipError>,
    },
    /// Repair completed but orientation canonicalization failed.
    #[error("Delaunay repair orientation canonicalization failed: {reason}")]
    OrientationCanonicalizationFailed {
        /// Structured canonicalization failure reason.
        #[source]
        reason: Box<DelaunayRepairOrientationCanonicalizationFailure>,
    },
    /// Flip-based repair is not admissible under the current topology guarantee.
    #[error("Delaunay repair requires {required:?} topology, found {found:?}: {message}")]
    InvalidTopology {
        /// Required topology guarantee.
        required: TopologyGuarantee,
        /// Actual topology guarantee.
        found: TopologyGuarantee,
        /// Additional context for the mismatch.
        message: &'static str,
    },
    /// Heuristic rebuild failed during advanced repair.
    #[error("Heuristic rebuild failed: {reason}")]
    HeuristicRebuildFailed {
        /// Structured rebuild failure reason.
        #[source]
        reason: Box<DelaunayRepairHeuristicRebuildFailure>,
    },
    /// A lower-level [`FlipError`] stopped repair.
    ///
    /// The source is boxed to keep [`DelaunayRepairError`] compact while
    /// preserving the concrete flip failure for callers that need to inspect it.
    #[error("flip error: {source}")]
    Flip {
        /// Typed flip failure that stopped repair.
        #[source]
        source: Box<FlipError>,
    },
}

impl From<FlipError> for DelaunayRepairError {
    fn from(source: FlipError) -> Self {
        Self::Flip {
            source: Box::new(source),
        }
    }
}

impl From<FacetError> for DelaunayRepairError {
    fn from(source: FacetError) -> Self {
        Self::from(FlipError::from(source))
    }
}

impl From<DelaunayRepairError> for FlipNeighborRepairFailure {
    fn from(source: DelaunayRepairError) -> Self {
        match source {
            DelaunayRepairError::NonConvergent {
                max_flips,
                diagnostics,
            } => Self::NonConvergent {
                max_flips,
                diagnostics: (*diagnostics).into(),
            },
            DelaunayRepairError::PostconditionFailed { reason } => {
                Self::PostconditionFailed { reason: *reason }
            }
            DelaunayRepairError::VerificationFailed { context, source } => {
                Self::VerificationFailed {
                    context,
                    source_kind: FlipFailureKind::from(source.as_ref()),
                }
            }
            DelaunayRepairError::OrientationCanonicalizationFailed { reason } => {
                Self::OrientationCanonicalizationFailed {
                    reason: reason.as_ref().into(),
                }
            }
            DelaunayRepairError::InvalidTopology {
                required,
                found,
                message,
            } => Self::InvalidTopology {
                required,
                found,
                message,
            },
            DelaunayRepairError::HeuristicRebuildFailed { reason } => {
                Self::HeuristicRebuildFailed {
                    reason: reason.as_ref().into(),
                }
            }
            DelaunayRepairError::Flip { source } => Self::Flip {
                source_kind: FlipFailureKind::from(source.as_ref()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::core::algorithms::incremental_insertion::DelaunayRepairFailureContext;
    use crate::core::algorithms::locate::LocateResult;
    use crate::core::collections::{SimplexVertexKeyBuffer, SimplexVertexUuidBuffer, Uuid};
    use crate::core::realization::TriangulationRealizationSimplexDetail;
    use crate::core::tds::TdsError;
    use crate::core::validation::TopologyGuarantee;
    use crate::geometry::traits::coordinate::CoordinateConversionValue;
    use crate::repair::DelaunayRepairOperation;
    use crate::topology::traits::topological_space::TopologyKind;
    use slotmap::KeyData;
    use std::assert_matches;
    use std::{
        error::Error as _,
        mem::{align_of, size_of},
    };

    fn sample_heuristic_vertex_context() -> DelaunayRepairHeuristicVertexContext {
        DelaunayRepairHeuristicVertexContext {
            index: 3,
            vertex_uuid: Uuid::nil(),
            coordinates: CoordinateValues::from([1.0, 2.0]),
        }
    }

    #[test]
    fn test_flip_error_partial_eq() {
        let unsupported_1 = FlipError::UnsupportedDimension { dimension: 1 };
        let unsupported_1_copy = FlipError::UnsupportedDimension { dimension: 1 };
        let unsupported_2 = FlipError::UnsupportedDimension { dimension: 2 };
        assert_eq!(unsupported_1, unsupported_1_copy);
        assert_ne!(unsupported_1, unsupported_2);

        assert_ne!(FlipError::DegenerateSimplex, FlipError::DuplicateSimplex);
        assert_eq!(FlipError::NonManifoldFacet, FlipError::NonManifoldFacet);

        let ridge_4 = FlipError::InvalidRidgeMultiplicity { found: 4 };
        let ridge_4_copy = FlipError::InvalidRidgeMultiplicity { found: 4 };
        let ridge_5 = FlipError::InvalidRidgeMultiplicity { found: 5 };
        assert_eq!(ridge_4, ridge_4_copy);
        assert_ne!(ridge_4, ridge_5);
    }

    fn sample_tds_validation_failure() -> TdsValidationFailure {
        TdsValidationFailure::InconsistentDataStructure {
            message: "synthetic neighbor mismatch".to_string(),
        }
    }

    fn sample_repair_diagnostics() -> DelaunayRepairDiagnostics {
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
    fn test_flip_failure_kind_preserves_nested_validation_and_repair_reasons() {
        let trial_error = FlipError::from(FlipMutationError::TrialValidation {
            k_move: 2,
            direction: FlipDirection::Forward,
            source: sample_tds_validation_failure(),
        });
        assert_eq!(
            FlipFailureKind::from(&trial_error),
            FlipFailureKind::TrialValidation
        );

        let wiring_validation = FlipError::from(FlipNeighborWiringError::TopologyValidation {
            source: sample_tds_validation_failure(),
        });
        assert_eq!(
            FlipFailureKind::from(&wiring_validation),
            FlipFailureKind::WiringValidation
        );

        let repair_reason =
            FlipNeighborRepairFailure::from(DelaunayRepairError::VerificationFailed {
                context: DelaunayRepairVerificationContext::PostRepairVerification,
                source: Box::new(trial_error),
            });
        match &repair_reason {
            FlipNeighborRepairFailure::VerificationFailed {
                context,
                source_kind,
            } => {
                assert_eq!(
                    *context,
                    DelaunayRepairVerificationContext::PostRepairVerification
                );
                assert_eq!(*source_kind, FlipFailureKind::TrialValidation);
            }
            other => panic!("expected verification failure, got {other:?}"),
        }

        let flip_reason =
            FlipNeighborRepairFailure::from(DelaunayRepairError::from(FlipError::DuplicateSimplex));
        assert_eq!(
            flip_reason,
            FlipNeighborRepairFailure::Flip {
                source_kind: FlipFailureKind::DuplicateSimplex,
            }
        );

        let wiring_repair = FlipError::from(FlipNeighborWiringError::DelaunayRepair {
            reason: repair_reason,
        });
        assert_eq!(
            FlipFailureKind::from(&wiring_repair),
            FlipFailureKind::DelaunayRepairFailed
        );

        let dangling_ridge_neighbor = FlipError::DanglingRidgeNeighbor {
            simplex_key: SimplexKey::from(KeyData::from_ffi(1)),
            neighbor_key: SimplexKey::from(KeyData::from_ffi(2)),
        };
        assert_eq!(
            FlipFailureKind::from(&dangling_ridge_neighbor),
            FlipFailureKind::DanglingRidgeNeighbor
        );

        let dangling_vertex_incidence = FlipError::DanglingVertexIncidence {
            vertex_key: VertexKey::from(KeyData::from_ffi(1)),
            simplex_key: SimplexKey::from(KeyData::from_ffi(2)),
        };
        assert_eq!(
            FlipFailureKind::from(&dangling_vertex_incidence),
            FlipFailureKind::DanglingVertexIncidence
        );

        let simplex_creation = FlipError::from(SimplexValidationError::DuplicateVertices);
        assert_eq!(
            FlipFailureKind::from(&simplex_creation),
            FlipFailureKind::SimplexCreation
        );
    }

    fn assert_hull_extension_failure_kind(
        source: &HullExtensionReason,
        expected: FlipNeighborHullExtensionFailureKind,
        expected_display: &str,
    ) {
        let hull_kind = FlipNeighborHullExtensionFailureKind::from(source);
        assert_eq!(hull_kind, expected);
        assert_eq!(hull_kind.to_string(), expected_display);
    }

    #[test]
    fn test_flip_neighbor_hull_extension_failure_kind_conversions() {
        assert_hull_extension_failure_kind(
            &HullExtensionReason::BoundaryEdgeSplitFacetCount {
                expected: 2,
                actual: 1,
            },
            FlipNeighborHullExtensionFailureKind::BoundaryEdgeSplitFacetCount,
            "boundary edge split facet count",
        );

        assert_hull_extension_failure_kind(
            &HullExtensionReason::MultipleBoundaryEdgeSplitFacets {
                first: FacetHandle::from_validated(SimplexKey::default(), 0),
                second: FacetHandle::from_validated(SimplexKey::default(), 1),
            },
            FlipNeighborHullExtensionFailureKind::MultipleBoundaryEdgeSplitFacets,
            "multiple boundary edge split facets",
        );

        assert_hull_extension_failure_kind(
            &HullExtensionReason::DisconnectedVisiblePatch {
                boundary_ridges: 1,
                ridge_fans: 0,
                components: 2,
                boundary_components: 2,
                boundary_subface_nonmanifold: 0,
            },
            FlipNeighborHullExtensionFailureKind::DisconnectedVisiblePatch,
            "disconnected visible patch",
        );

        assert_hull_extension_failure_kind(
            &HullExtensionReason::PredicateFailed {
                source: CoordinateConversionError::InvalidSimplexPointCount {
                    actual: 2,
                    expected: 3,
                    dimension: 2,
                },
            },
            FlipNeighborHullExtensionFailureKind::PredicateFailed,
            "predicate failed",
        );

        assert_hull_extension_failure_kind(
            &HullExtensionReason::Tds {
                source: TdsError::InconsistentDataStructure {
                    message: "missing boundary facet".to_string(),
                },
            },
            FlipNeighborHullExtensionFailureKind::Tds,
            "TDS",
        );
    }

    #[expect(
        clippy::too_many_lines,
        reason = "test keeps the insertion suberror conversion matrix together"
    )]
    #[test]
    fn test_flip_neighbor_conversion_kinds_cover_insertion_suberrors() {
        let cavity_kind = FlipNeighborCavityFailureKind::from(
            &CavityFillingError::BoundarySimplexCountMismatch {
                boundary_facet_count: 3,
                new_simplex_count: 2,
            },
        );
        assert_eq!(
            cavity_kind,
            FlipNeighborCavityFailureKind::BoundarySimplexCountMismatch
        );
        assert_eq!(cavity_kind.to_string(), "boundary simplex count mismatch");

        let cavity_kind = FlipNeighborCavityFailureKind::from(
            &CavityFillingError::UnsupportedDegenerateLocation {
                location: LocateResult::Outside,
            },
        );
        assert_eq!(
            cavity_kind,
            FlipNeighborCavityFailureKind::UnsupportedDegenerateLocation
        );
        assert_eq!(cavity_kind.to_string(), "unsupported degenerate location");

        let validation_kind = FlipNeighborDelaunayValidationFailureKind::from(
            &DelaunayTriangulationValidationError::RepairOperationFailed {
                operation: DelaunayRepairOperation::VertexRemoval,
                source: Box::new(DelaunayRepairError::InvalidTopology {
                    required: TopologyGuarantee::PLManifold,
                    found: TopologyGuarantee::Pseudomanifold,
                    message: "repair requires PL topology",
                }),
            },
        );
        assert_eq!(
            validation_kind,
            FlipNeighborDelaunayValidationFailureKind::RepairOperationFailed
        );
        assert_eq!(validation_kind.to_string(), "repair operation failed");

        let validation_cases = [
            (
                DelaunayTriangulationValidationError::from(TdsError::InconsistentDataStructure {
                    message: "dangling simplex".to_string(),
                }),
                FlipNeighborDelaunayValidationFailureKind::Tds,
            ),
            (
                DelaunayTriangulationValidationError::from(
                    TriangulationValidationError::Disconnected { simplex_count: 2 },
                ),
                FlipNeighborDelaunayValidationFailureKind::Triangulation,
            ),
            (
                DelaunayTriangulationValidationError::Realization {
                    source: Box::new(
                        TriangulationRealizationValidationError::UnsupportedTopology {
                            topology: TopologyKind::Hyperbolic,
                            dimension: 2,
                        },
                    ),
                },
                FlipNeighborDelaunayValidationFailureKind::Realization,
            ),
        ];
        for (source, expected) in validation_cases {
            assert_eq!(
                FlipNeighborDelaunayValidationFailureKind::from(&source),
                expected
            );
        }

        let repair_wiring = FlipNeighborWiringError::from(InsertionError::DelaunayRepairFailed {
            source: Box::new(DelaunayRepairError::InvalidTopology {
                required: TopologyGuarantee::PLManifold,
                found: TopologyGuarantee::Pseudomanifold,
                message: "repair requires PL topology",
            }),
            context: DelaunayRepairFailureContext::PostInsertionRepair,
        });
        match repair_wiring {
            FlipNeighborWiringError::DelaunayRepair {
                reason:
                    FlipNeighborRepairFailure::InvalidTopology {
                        required,
                        found,
                        message,
                    },
            } => {
                assert_eq!(required, TopologyGuarantee::PLManifold);
                assert_eq!(found, TopologyGuarantee::Pseudomanifold);
                assert_eq!(message, "repair requires PL topology");
            }
            other => panic!("expected preserved Delaunay repair reason, got {other:?}"),
        }

        let budget_wiring =
            FlipNeighborWiringError::from(InsertionError::MaxSimplicesRemovedExceeded {
                max_simplices_removed: 2,
                attempted: 3,
            });
        assert_eq!(
            budget_wiring,
            FlipNeighborWiringError::MaxSimplicesRemovedExceeded {
                max_simplices_removed: 2,
                attempted: 3,
            }
        );

        let spatial_index_wiring =
            FlipNeighborWiringError::from(InsertionError::SpatialIndexConstruction {
                reason: SpatialIndexConstructionFailure::NonPositiveCellSize {
                    value: CoordinateConversionValue::from_f64(0.0),
                },
            });
        assert_eq!(
            spatial_index_wiring,
            FlipNeighborWiringError::SpatialIndexConstruction {
                reason: SpatialIndexConstructionFailure::NonPositiveCellSize {
                    value: CoordinateConversionValue::from_f64(0.0),
                },
            }
        );
    }

    #[test]
    fn flip_neighbor_wiring_classifies_and_preserves_boundary_sources() {
        let topology_error = InsertionError::TopologyValidation {
            source: TdsError::InconsistentDataStructure {
                message: "broken topology".to_string(),
            },
        };
        assert_eq!(
            insertion_error_kind(&topology_error),
            InsertionErrorKind::TopologyValidation
        );
        assert_eq!(
            FlipNeighborWiringError::from(topology_error),
            FlipNeighborWiringError::TopologyValidation {
                source: TdsValidationFailure::InconsistentDataStructure {
                    message: "broken topology".to_string(),
                },
            }
        );

        let simplex_key = SimplexKey::from(KeyData::from_ffi(9_001));
        let conflict_source = ConflictError::InvalidStartSimplex { simplex_key };
        let conflict_error = InsertionError::ConflictRegion {
            source: conflict_source.clone(),
        };
        assert_eq!(
            insertion_error_kind(&conflict_error),
            InsertionErrorKind::ConflictRegion
        );
        assert_eq!(
            FlipNeighborWiringError::from(conflict_error),
            FlipNeighborWiringError::ConflictRegion {
                source: conflict_source,
            }
        );

        let location_source = LocateError::InvalidSimplex { simplex_key };
        let location_error = InsertionError::Location {
            source: location_source.clone(),
        };
        assert_eq!(
            insertion_error_kind(&location_error),
            InsertionErrorKind::Location
        );
        assert_eq!(
            FlipNeighborWiringError::from(location_error),
            FlipNeighborWiringError::Location {
                source: location_source,
            }
        );
    }

    #[test]
    fn flip_neighbor_wiring_preserves_realization_validation_source() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(9_101));
        let simplex_uuid = Uuid::from_u128(0x9101);
        let vertices: SimplexVertexKeyBuffer = [
            VertexKey::from(KeyData::from_ffi(9_201)),
            VertexKey::from(KeyData::from_ffi(9_202)),
            VertexKey::from(KeyData::from_ffi(9_203)),
        ]
        .into_iter()
        .collect();
        let vertex_uuids: SimplexVertexUuidBuffer = [
            Uuid::from_u128(0x9201),
            Uuid::from_u128(0x9202),
            Uuid::from_u128(0x9203),
        ]
        .into_iter()
        .collect();

        let realization_source = TriangulationRealizationValidationError::DegenerateSimplex {
            simplex_key,
            simplex_uuid,
            detail: Box::new(TriangulationRealizationSimplexDetail {
                key: simplex_key,
                uuid: simplex_uuid,
                vertices,
                vertex_uuids,
            }),
            dimension: 2,
        };

        let realization_wiring =
            FlipNeighborWiringError::from(InsertionError::RealizationValidationFailed {
                source: realization_source.clone(),
            });
        let FlipNeighborWiringError::RealizationValidation { source } = &realization_wiring else {
            panic!("expected preserved realization validation source, got {realization_wiring:?}");
        };
        assert_eq!(source, &realization_source);
        let error_source = realization_wiring
            .source()
            .and_then(|source| source.downcast_ref::<TriangulationRealizationValidationError>())
            .expect("realization validation should remain the typed error source");
        assert_eq!(error_source, &realization_source);
        assert_eq!(
            FlipFailureKind::from(&FlipError::from(realization_wiring)),
            FlipFailureKind::WiringValidation
        );

        let transaction_realization = FlipError::RealizationValidation {
            source: Box::new(realization_source.clone()),
        };
        assert_eq!(
            FlipFailureKind::from(&transaction_realization),
            FlipFailureKind::RealizationValidation
        );

        let transaction_repair = FlipError::PostconditionRepair {
            source: Box::new(InsertionError::RealizationValidationFailed {
                source: realization_source,
            }),
        };
        assert_eq!(
            FlipFailureKind::from(&transaction_repair),
            FlipFailureKind::PostconditionRepair
        );
    }

    #[test]
    fn test_flip_neighbor_repair_diagnostics_preserve_summary_fields() {
        let diagnostics = sample_repair_diagnostics();
        let summary = FlipNeighborRepairDiagnostics::from(diagnostics.clone());

        assert_eq!(summary.facets_checked, diagnostics.facets_checked);
        assert_eq!(summary.flips_performed, diagnostics.flips_performed);
        assert_eq!(summary.max_queue_len, diagnostics.max_queue_len);
        assert_eq!(
            summary.ambiguous_predicates,
            diagnostics.ambiguous_predicates
        );
        assert_eq!(summary.predicate_failures, diagnostics.predicate_failures);
        assert_eq!(summary.cycle_detections, diagnostics.cycle_detections);
        assert_eq!(summary.attempt, diagnostics.attempt);
        assert_eq!(summary.queue_order, diagnostics.queue_order);
        assert_eq!(
            summary.to_string(),
            "checked=7, flips=3, max_queue=5, ambiguous=2, predicate_failures=1, cycles=4, attempt=2, order=Lifo"
        );

        let non_convergent = FlipNeighborRepairFailure::from(DelaunayRepairError::NonConvergent {
            max_flips: 42,
            diagnostics: Box::new(diagnostics),
        });
        match non_convergent {
            FlipNeighborRepairFailure::NonConvergent {
                max_flips,
                diagnostics,
            } => {
                assert_eq!(max_flips, 42);
                assert_eq!(diagnostics.flips_performed, 3);
            }
            other => panic!("expected non-convergent repair summary, got {other:?}"),
        }
    }

    #[test]
    fn test_delaunay_repair_error_partial_eq() {
        let post_test = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        };
        let post_test_copy = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        };
        let post_other = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 2 }),
        };
        assert_eq!(post_test, post_test_copy);
        assert_ne!(post_test, post_other);

        let verification_err = DelaunayRepairError::VerificationFailed {
            context: DelaunayRepairVerificationContext::StrictValidation,
            source: Box::new(FlipError::DegenerateSimplex),
        };
        let verification_err_copy = DelaunayRepairError::VerificationFailed {
            context: DelaunayRepairVerificationContext::StrictValidation,
            source: Box::new(FlipError::DegenerateSimplex),
        };
        let verification_other = DelaunayRepairError::VerificationFailed {
            context: DelaunayRepairVerificationContext::StrictValidation,
            source: Box::new(FlipError::DuplicateSimplex),
        };
        assert_eq!(verification_err, verification_err_copy);
        assert_ne!(verification_err, verification_other);

        let flip_err = DelaunayRepairError::from(FlipError::DegenerateSimplex);
        let flip_err_copy = DelaunayRepairError::from(FlipError::DegenerateSimplex);
        let flip_other = DelaunayRepairError::from(FlipError::DuplicateSimplex);
        assert_eq!(flip_err, flip_err_copy);
        assert_ne!(flip_err, flip_other);

        let canonicalization_err = DelaunayRepairError::OrientationCanonicalizationFailed {
            reason: Box::new(
                DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair {
                    source: Box::new(InsertionError::DuplicateCoordinates {
                        coordinates: CoordinateValues::from([0.0, 0.0]),
                    }),
                },
            ),
        };
        let canonicalization_err_copy = DelaunayRepairError::OrientationCanonicalizationFailed {
            reason: Box::new(
                DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair {
                    source: Box::new(InsertionError::DuplicateCoordinates {
                        coordinates: CoordinateValues::from([0.0, 0.0]),
                    }),
                },
            ),
        };
        let canonicalization_other = DelaunayRepairError::OrientationCanonicalizationFailed {
            reason: Box::new(
                DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair {
                    source: Box::new(InsertionError::DuplicateCoordinates {
                        coordinates: CoordinateValues::from([1.0, 1.0]),
                    }),
                },
            ),
        };
        assert_eq!(canonicalization_err, canonicalization_err_copy);
        assert_ne!(canonicalization_err, canonicalization_other);

        let topo_err = DelaunayRepairError::InvalidTopology {
            required: TopologyGuarantee::PLManifold,
            found: TopologyGuarantee::Pseudomanifold,
            message: "test",
        };
        let topo_err_copy = DelaunayRepairError::InvalidTopology {
            required: TopologyGuarantee::PLManifold,
            found: TopologyGuarantee::Pseudomanifold,
            message: "test",
        };
        assert_eq!(topo_err, topo_err_copy);

        // Different variants are never equal.
        assert_ne!(post_test, topo_err);
        assert_ne!(post_test, verification_err);
        assert_ne!(post_test, canonicalization_err);
    }

    #[test]
    fn test_postcondition_failure_display_covers_variants() {
        let simplex = SimplexKey::from(KeyData::from_ffi(91));
        let v0 = VertexKey::from(KeyData::from_ffi(101));
        let v1 = VertexKey::from(KeyData::from_ffi(102));
        let v2 = VertexKey::from(KeyData::from_ffi(103));
        let facet = FacetHandle::from_validated(simplex, 0);
        let ridge = RidgeHandle::from_validated(simplex, 0, 1);
        let edge = EdgeKey::from_validated_endpoints(v0, v1);
        let triangle = TriangleHandle::try_new(v0, v1, v2).unwrap();

        assert_eq!(
            DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 2 }.to_string(),
            "repair pass disconnected the triangulation (2 simplices remain); neighbor wiring is incomplete"
        );

        let k2 = DelaunayRepairPostconditionFailure::LocalK2Violation {
            facet,
            debug_details: Some("debug facet snapshot".to_string()),
        }
        .to_string();
        assert!(k2.contains("local k=2 violation remains after repair"));
        assert!(k2.contains("debug facet snapshot"));

        let k3 = DelaunayRepairPostconditionFailure::LocalK3Violation { ridge }.to_string();
        assert!(k3.contains("local k=3 violation remains after repair"));

        let inverse_k2 =
            DelaunayRepairPostconditionFailure::LocalInverseK2Violation { edge }.to_string();
        assert!(inverse_k2.contains("local inverse k=2 flip remains applicable after repair"));

        let inverse_k3 =
            DelaunayRepairPostconditionFailure::LocalInverseK3Violation { triangle }.to_string();
        assert!(inverse_k3.contains("local inverse k=3 flip remains applicable after repair"));
    }

    #[test]
    fn test_postcondition_failure_exposes_source() {
        let reason = DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 };
        let repair = DelaunayRepairError::PostconditionFailed {
            reason: Box::new(reason.clone()),
        };
        let source = repair
            .source()
            .and_then(|source| source.downcast_ref::<Box<DelaunayRepairPostconditionFailure>>());
        assert_eq!(source.map(Box::as_ref), Some(&reason));

        let neighbor_repair = FlipNeighborRepairFailure::PostconditionFailed { reason };
        assert_matches!(
            std::error::Error::source(&neighbor_repair)
                .and_then(|source| source.downcast_ref::<DelaunayRepairPostconditionFailure>()),
            Some(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 })
        );
    }

    #[test]
    fn test_heuristic_vertex_context_display() {
        let context = sample_heuristic_vertex_context().to_string();

        assert!(context.contains("idx=3"));
        assert!(context.contains("uuid=00000000-0000-0000-0000-000000000000"));
        assert!(context.contains("coords=[1.0, 2.0]"));
    }

    #[test]
    fn test_orientation_failure_kind_conversion() {
        let orientation_failure =
            DelaunayRepairOrientationCanonicalizationFailure::AfterFlipRepair {
                source: Box::new(InsertionError::DuplicateCoordinates {
                    coordinates: CoordinateValues::from([0.0, 0.0]),
                }),
            };

        assert_eq!(
            DelaunayRepairOrientationCanonicalizationFailureKind::from(&orientation_failure),
            DelaunayRepairOrientationCanonicalizationFailureKind::AfterFlipRepair {
                source_kind: InsertionErrorKind::DuplicateCoordinates,
            },
        );

        let orientation_repair = DelaunayRepairError::OrientationCanonicalizationFailed {
            reason: Box::new(orientation_failure),
        };
        assert_matches!(
            FlipNeighborRepairFailure::from(orientation_repair),
            FlipNeighborRepairFailure::OrientationCanonicalizationFailed {
                reason: DelaunayRepairOrientationCanonicalizationFailureKind::AfterFlipRepair {
                    source_kind: InsertionErrorKind::DuplicateCoordinates
                }
            }
        );
    }

    #[test]
    fn test_heuristic_rebuild_failure_kind_conversion() {
        let insertion_failure = InsertionError::DuplicateCoordinates {
            coordinates: CoordinateValues::from([0.0, 0.0]),
        };
        let repair_source = || DelaunayRepairError::from(FlipError::DegenerateSimplex);
        let vertex = sample_heuristic_vertex_context();
        let heuristic_cases = [
            (
                DelaunayRepairHeuristicRebuildFailure::RecursionDepthExceeded { max_depth: 1 },
                DelaunayRepairHeuristicRebuildFailureKind::RecursionDepthExceeded,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::FallbackChainFailed {
                    primary: Box::new(repair_source()),
                    robust: Box::new(repair_source()),
                    heuristic: Box::new(DelaunayRepairHeuristicRebuildFailure::NoAttempts),
                },
                DelaunayRepairHeuristicRebuildFailureKind::FallbackChainFailed,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::UnexpectedRepairFailure {
                    source: Box::new(repair_source()),
                },
                DelaunayRepairHeuristicRebuildFailureKind::UnexpectedRepairFailure,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::NoAttempts,
                DelaunayRepairHeuristicRebuildFailureKind::NoAttempts,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::InsertionFailed {
                    vertex: vertex.clone(),
                    source: Box::new(insertion_failure.clone()),
                },
                DelaunayRepairHeuristicRebuildFailureKind::InsertionFailed,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::RepairFailed {
                    vertex: vertex.clone(),
                    source: Box::new(insertion_failure.clone()),
                },
                DelaunayRepairHeuristicRebuildFailureKind::RepairFailed,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::DelaunayCheckFailed {
                    vertex: vertex.clone(),
                    source: Box::new(insertion_failure.clone()),
                },
                DelaunayRepairHeuristicRebuildFailureKind::DelaunayCheckFailed,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::SkippedVertex {
                    vertex,
                    source: Box::new(insertion_failure),
                },
                DelaunayRepairHeuristicRebuildFailureKind::SkippedVertex,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::AttemptFailed {
                    attempt: 1,
                    max_attempts: 2,
                    shuffle_seed: 3,
                    perturbation_seed: 4,
                    source: Box::new(repair_source()),
                },
                DelaunayRepairHeuristicRebuildFailureKind::AttemptFailed,
            ),
            (
                DelaunayRepairHeuristicRebuildFailure::ExhaustedAttempts {
                    attempts: 2,
                    last_failure: Box::new(DelaunayRepairHeuristicRebuildFailure::NoAttempts),
                },
                DelaunayRepairHeuristicRebuildFailureKind::ExhaustedAttempts,
            ),
        ];

        for (failure, expected_kind) in heuristic_cases {
            assert_eq!(
                DelaunayRepairHeuristicRebuildFailureKind::from(&failure),
                expected_kind,
            );
        }
    }

    #[test]
    fn test_flip_neighbor_repair_failure_conversion() {
        let heuristic_repair = DelaunayRepairError::HeuristicRebuildFailed {
            reason: Box::new(DelaunayRepairHeuristicRebuildFailure::NoAttempts),
        };
        assert_matches!(
            FlipNeighborRepairFailure::from(heuristic_repair),
            FlipNeighborRepairFailure::HeuristicRebuildFailed {
                reason: DelaunayRepairHeuristicRebuildFailureKind::NoAttempts
            }
        );
    }

    #[test]
    fn test_delaunay_repair_error_boxes_large_flip_sources() {
        assert!(
            std::mem::size_of::<DelaunayRepairError>() <= std::mem::size_of::<FlipError>(),
            "DelaunayRepairError should box FlipError payloads without exceeding FlipError size"
        );

        let err = DelaunayRepairError::from(FlipError::DegenerateSimplex);
        let source = err.source().expect("boxed flip source should be exposed");
        let source = source
            .downcast_ref::<Box<FlipError>>()
            .expect("source should remain a typed boxed FlipError");
        assert_matches!(source.as_ref(), FlipError::DegenerateSimplex);

        let DelaunayRepairError::Flip { source } = err else {
            panic!("expected boxed flip source");
        };
        assert_matches!(source.as_ref(), FlipError::DegenerateSimplex);
    }

    #[test]
    fn test_heuristic_exhausted_attempts_exposes_last_failure_source() {
        let exhausted = DelaunayRepairHeuristicRebuildFailure::ExhaustedAttempts {
            attempts: 6,
            last_failure: Box::new(DelaunayRepairHeuristicRebuildFailure::NoAttempts),
        };

        let source = exhausted
            .source()
            .expect("exhausted attempts should expose the last failure source")
            .downcast_ref::<Box<DelaunayRepairHeuristicRebuildFailure>>()
            .expect("source should remain a typed boxed heuristic failure");
        assert_matches!(
            source.as_ref(),
            DelaunayRepairHeuristicRebuildFailure::NoAttempts
        );
    }

    #[test]
    fn test_flip_error_boxes_nested_typed_payloads() {
        let max_nested_payload_size = [
            size_of::<FlipContextError>(),
            size_of::<FlipPredicateError>(),
            size_of::<FlipEdgeAdjacencyError>(),
            size_of::<FlipTriangleAdjacencyError>(),
            size_of::<FlipVertexAdjacencyError>(),
            size_of::<SimplexValidationError>(),
            size_of::<FlipNeighborWiringError>(),
            size_of::<FlipMutationError>(),
            size_of::<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        assert!(
            size_of::<FlipError>() < max_nested_payload_size,
            "boxed FlipError should stay smaller than its largest nested payload"
        );
        assert_eq!(align_of::<Result<(), FlipError>>(), align_of::<FlipError>());
        assert!(
            size_of::<Result<(), FlipError>>() <= size_of::<FlipError>() + size_of::<usize>(),
            "Result<(), FlipError> should remain within one machine word of FlipError"
        );

        let mutation = FlipError::from(FlipMutationError::TrialValidation {
            k_move: 2,
            direction: FlipDirection::Forward,
            source: sample_tds_validation_failure(),
        });
        let source = mutation
            .source()
            .expect("boxed mutation source should be exposed")
            .downcast_ref::<Box<FlipMutationError>>()
            .expect("source should remain a typed boxed FlipMutationError");
        assert_matches!(
            source.as_ref(),
            FlipMutationError::TrialValidation {
                k_move: 2,
                direction: FlipDirection::Forward,
                ..
            }
        );

        let FlipError::TdsMutation { reason } = mutation else {
            panic!("expected boxed TDS mutation reason");
        };
        assert_matches!(
            reason.as_ref(),
            FlipMutationError::TrialValidation {
                k_move: 2,
                direction: FlipDirection::Forward,
                ..
            }
        );

        let mut simplex_vertices = SmallBuffer::new();
        simplex_vertices.push(VertexKey::from(KeyData::from_ffi(1)));
        simplex_vertices.push(VertexKey::from(KeyData::from_ffi(2)));
        let duplicate = FlipError::InsertedSimplexAlreadyExists {
            k_move: 2,
            simplex_vertices: Box::new(simplex_vertices),
            existing_simplex: SimplexKey::from(KeyData::from_ffi(3)),
        };

        let FlipError::InsertedSimplexAlreadyExists {
            simplex_vertices, ..
        } = duplicate
        else {
            panic!("expected boxed simplex witness");
        };
        assert_eq!(simplex_vertices.as_ref().len(), 2);
    }

    #[test]
    fn test_flip_error_boxes_adjacency_payload_sources() {
        let edge_vertex = VertexKey::from(KeyData::from_ffi(4));
        let edge = FlipError::from(FlipEdgeAdjacencyError::DuplicateEndpoints {
            vertex_key: edge_vertex,
        });
        let source = edge
            .source()
            .expect("boxed edge-adjacency source should be exposed")
            .downcast_ref::<Box<FlipEdgeAdjacencyError>>()
            .expect("source should remain a typed boxed FlipEdgeAdjacencyError");
        assert_matches!(
            source.as_ref(),
            FlipEdgeAdjacencyError::DuplicateEndpoints { vertex_key }
                if *vertex_key == edge_vertex
        );

        let FlipError::InvalidEdgeAdjacency { reason } = edge else {
            panic!("expected boxed edge-adjacency reason");
        };
        assert_matches!(
            reason.as_ref(),
            FlipEdgeAdjacencyError::DuplicateEndpoints { vertex_key }
                if *vertex_key == edge_vertex
        );

        let simplex_key = SimplexKey::from(KeyData::from_ffi(5));
        let triangle_a = VertexKey::from(KeyData::from_ffi(6));
        let triangle_b = VertexKey::from(KeyData::from_ffi(7));
        let triangle_c = VertexKey::from(KeyData::from_ffi(8));
        let triangle =
            FlipError::from(FlipTriangleAdjacencyError::SimplexMissingTriangleVertices {
                simplex_key,
                a: triangle_a,
                b: triangle_b,
                c: triangle_c,
            });
        let source = triangle
            .source()
            .expect("boxed triangle-adjacency source should be exposed")
            .downcast_ref::<Box<FlipTriangleAdjacencyError>>()
            .expect("source should remain a typed boxed FlipTriangleAdjacencyError");
        assert_matches!(
            source.as_ref(),
            FlipTriangleAdjacencyError::SimplexMissingTriangleVertices { a, b, c, .. }
                if *a == triangle_a && *b == triangle_b && *c == triangle_c
        );

        let FlipError::InvalidTriangleAdjacency { reason } = triangle else {
            panic!("expected boxed triangle-adjacency reason");
        };
        assert_matches!(
            reason.as_ref(),
            FlipTriangleAdjacencyError::SimplexMissingTriangleVertices { a, b, c, .. }
                if *a == triangle_a && *b == triangle_b && *c == triangle_c
        );

        let vertex = FlipError::from(FlipVertexAdjacencyError::SimplexMissingVertex {
            simplex_key,
            vertex_key: edge_vertex,
        });
        let source = vertex
            .source()
            .expect("boxed vertex-adjacency source should be exposed")
            .downcast_ref::<Box<FlipVertexAdjacencyError>>()
            .expect("source should remain a typed boxed FlipVertexAdjacencyError");
        assert_matches!(
            source.as_ref(),
            FlipVertexAdjacencyError::SimplexMissingVertex { vertex_key, .. }
                if *vertex_key == edge_vertex
        );

        let FlipError::InvalidVertexAdjacency { reason } = vertex else {
            panic!("expected boxed vertex-adjacency reason");
        };
        assert_matches!(
            reason.as_ref(),
            FlipVertexAdjacencyError::SimplexMissingVertex { vertex_key, .. }
                if *vertex_key == edge_vertex
        );
    }

    #[test]
    fn test_delaunay_repair_verification_context_display_covers_all_variants() {
        let cases = [
            (
                DelaunayRepairVerificationContext::PostRepairVerification,
                "post-repair verification",
            ),
            (
                DelaunayRepairVerificationContext::StrictValidation,
                "strict validation",
            ),
            (
                DelaunayRepairVerificationContext::LocalK2DegeneracyVerification,
                "local k=2 degeneracy verification",
            ),
            (
                DelaunayRepairVerificationContext::LocalK2PostconditionVerification,
                "local k=2 postcondition verification",
            ),
            (
                DelaunayRepairVerificationContext::LocalK3DegeneracyVerification,
                "local k=3 degeneracy verification",
            ),
            (
                DelaunayRepairVerificationContext::LocalK3PostconditionVerification,
                "local k=3 postcondition verification",
            ),
            (
                DelaunayRepairVerificationContext::LocalInverseK2PostconditionVerification,
                "local inverse k=2 postcondition verification",
            ),
            (
                DelaunayRepairVerificationContext::LocalInverseK3PostconditionVerification,
                "local inverse k=3 postcondition verification",
            ),
        ];

        for (context, expected_display) in cases {
            assert_eq!(context.to_string(), expected_display);
            let err = DelaunayRepairError::VerificationFailed {
                context,
                source: Box::new(FlipError::DegenerateSimplex),
            };

            match err {
                DelaunayRepairError::VerificationFailed {
                    context: observed, ..
                } => assert_eq!(observed, context),
                other => panic!("expected verification failure, got {other:?}"),
            }
        }
    }
}

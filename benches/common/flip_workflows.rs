//! Shared public flip workflows for benchmark fixture validation.
//!
//! This module is included directly by benchmark and integration-test targets.
//! It centralizes candidate selection, flip execution, and exact topology
//! roundtrip diagnostics so tests certify the same public flip workflows that
//! Criterion measures.

use std::{collections::HashSet, fmt, num::TryFromIntError};

use delaunay::flips::{
    BistellarFlips, EdgeKey, FacetHandle, FlipError, RidgeHandle, SimplexKey, TriangleHandle,
    TriangleHandleError,
};
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationConstructionError,
    InsertionOrderStrategy, TopologyGuarantee, Vertex, vertex,
};
use delaunay::prelude::geometry::{CoordinateConversionError, Point, RobustKernel, simplex_volume};
use delaunay::prelude::query::{JaccardComputationError, format_jaccard_report};
use delaunay::prelude::tds::{EdgeKeyError, FacetError, InvariantError, TdsError, VertexKey};
use delaunay::prelude::topology::validation::{
    ManifoldError, RidgeCandidate, RidgeCandidateError, ridge_star_simplices,
};
use delaunay::prelude::validation::DelaunayTriangulationValidationError;
use thiserror::Error;
use uuid::Uuid;

/// Robust-kernel triangulation type used by explicit public flip fixtures.
pub type FlipTriangulation<const D: usize> = DelaunayTriangulation<RobustKernel<f64>, (), (), D>;

/// Result type for benchmark/test workflow helpers that preserves
/// [`FlipWorkflowError`] variants for setup diagnostics.
pub type FlipWorkflowResult<T> = Result<T, FlipWorkflowError>;

/// Errors that can occur while preparing or verifying benchmark flip workflows.
///
/// These errors are crate-private to the benchmark/test harness, but they keep
/// fixture setup and n=1 ergodicity failures structured so tests can assert the
/// failing contract without parsing display text.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum FlipWorkflowError {
    /// Fixture triangulation construction failed.
    #[error("failed to build {dimension}D flip fixture triangulation: {source}")]
    Construction {
        /// Fixture dimension.
        dimension: usize,
        /// Underlying triangulation construction failure.
        #[source]
        source: Box<DelaunayTriangulationConstructionError>,
    },

    /// Vertex coordinate conversion failed during fixture preparation.
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),

    /// A simplex key referenced by the workflow was not live.
    #[error("simplex key {simplex_key:?} does not exist")]
    MissingSimplex {
        /// Missing simplex key.
        simplex_key: SimplexKey,
    },

    /// A vertex key referenced by the workflow was not live.
    #[error("vertex key {vertex_key:?} does not exist")]
    MissingVertex {
        /// Missing vertex key.
        vertex_key: VertexKey,
    },

    /// Facet handle construction failed before candidate inspection.
    #[error("failed to construct facet handle {simplex_key:?}:{facet_index}: {source}")]
    FacetHandleConstruction {
        /// Candidate simplex key.
        simplex_key: SimplexKey,
        /// Candidate facet index.
        facet_index: u8,
        /// Underlying facet-handle construction failure.
        #[source]
        source: FacetError,
    },

    /// Ridge handle construction failed before candidate inspection.
    #[error("failed to construct ridge handle {simplex_key:?}:({omit_a}, {omit_b}): {source}")]
    RidgeHandleConstruction {
        /// Candidate simplex key.
        simplex_key: SimplexKey,
        /// First omitted index.
        omit_a: u8,
        /// Second omitted index.
        omit_b: u8,
        /// Underlying ridge-handle construction failure.
        #[source]
        source: Box<FlipError>,
    },

    /// Ridge vertices could not be parsed into a valid ridge candidate.
    #[error("invalid ridge candidate for {ridge:?}: {source}")]
    InvalidRidgeCandidate {
        /// Ridge handle being inspected.
        ridge: RidgeHandle,
        /// Underlying ridge candidate parsing failure.
        #[source]
        source: RidgeCandidateError,
    },

    /// Ridge-star support collection failed.
    #[error("failed to collect ridge star for {ridge:?}: {source}")]
    RidgeStar {
        /// Ridge handle being inspected.
        ridge: RidgeHandle,
        /// Underlying manifold helper failure.
        #[source]
        source: Box<ManifoldError>,
    },

    /// Snapshot collection found a dangling simplex-to-vertex incidence.
    #[error("simplex references missing vertex key {vertex_key:?}")]
    DanglingSnapshotVertex {
        /// Missing vertex key.
        vertex_key: VertexKey,
    },

    /// The k=1 inserted vertex could not be found after insertion.
    #[error("inserted k=1 vertex {uuid} is missing after insert")]
    MissingInsertedVertex {
        /// UUID assigned to the inserted vertex.
        uuid: Uuid,
    },

    /// Simplex vertex count could not be converted for centroid averaging.
    #[error("simplex vertex count does not fit in u32: {source}")]
    SimplexVertexCountTooLarge {
        /// Integer conversion failure.
        #[source]
        source: TryFromIntError,
    },

    /// Jaccard diagnostic formatting failed.
    #[error("failed to format {report} Jaccard report: {source}")]
    JaccardReportFormatting {
        /// Report being formatted.
        report: JaccardReportKind,
        /// Underlying Jaccard computation failure.
        #[source]
        source: JaccardComputationError,
    },

    /// An exact n=1 roundtrip produced a different topology.
    #[error(
        "{context} failed to recover the same triangulation\n\
         {vertex_report}\n\
         {simplex_report}\n\
         expected topology: {expected:#?}\n\
         actual topology: {actual:#?}"
    )]
    TopologyMismatch {
        /// Roundtrip context label.
        context: String,
        /// Jaccard diagnostics for vertex UUIDs.
        vertex_report: String,
        /// Jaccard diagnostics for simplex incidence.
        simplex_report: String,
        /// Expected [`TopologySnapshot`].
        expected: Box<TopologySnapshot>,
        /// Actual [`TopologySnapshot`].
        actual: Box<TopologySnapshot>,
    },

    /// No non-degenerate simplex satisfied the selection policy.
    #[error("flip benchmark triangulation has no non-degenerate {filter:?} simplex")]
    NoNondegenerateSimplex {
        /// Fixture dimension.
        dimension: usize,
        /// Candidate selection policy.
        filter: CandidateFilter,
    },

    /// No k=2 facet candidate satisfied the workflow requirements.
    #[error(
        "no flippable interior facet found for {dimension}D {filter:?} k=2 benchmark (last error: {last_error:?})"
    )]
    NoFlippableFacet {
        /// Fixture dimension.
        dimension: usize,
        /// Candidate selection policy.
        filter: CandidateFilter,
        /// Last rejected candidate detail, if one was reached.
        last_error: Option<Box<FlipCandidateError>>,
    },

    /// No k=3 ridge candidate satisfied the workflow requirements.
    #[error(
        "no flippable ridge found for {dimension}D {filter:?} k=3 benchmark (last error: {last_error:?})"
    )]
    NoFlippableRidge {
        /// Fixture dimension.
        dimension: usize,
        /// Candidate selection policy.
        filter: CandidateFilter,
        /// Last rejected candidate detail, if one was reached.
        last_error: Option<Box<FlipCandidateError>>,
    },

    /// A selected facet has no interior neighbor across its omitted vertex.
    #[error("facet {facet:?} does not have an interior neighbor")]
    FacetWithoutInteriorNeighbor {
        /// Facet handle.
        facet: FacetHandle,
    },

    /// A support-inspection facet index is out of bounds for its simplex.
    #[error(
        "facet index {facet_index} out of bounds for support simplex {simplex_key:?} with {vertex_count} vertices"
    )]
    InvalidFacetSupportIndex {
        /// Facet handle.
        facet: FacetHandle,
        /// Out-of-bounds facet index.
        facet_index: u8,
        /// Number of vertices in the support simplex.
        vertex_count: usize,
        /// Support simplex key.
        simplex_key: SimplexKey,
    },

    /// Support-inspection ridge indices are invalid for their simplex.
    #[error(
        "ridge indices ({omit_a}, {omit_b}) out of bounds for support simplex {simplex_key:?} with {vertex_count} vertices"
    )]
    InvalidRidgeSupportIndex {
        /// Ridge handle.
        ridge: RidgeHandle,
        /// First omitted index.
        omit_a: u8,
        /// Second omitted index.
        omit_b: u8,
        /// Number of vertices in the support simplex.
        vertex_count: usize,
        /// Support simplex key.
        simplex_key: SimplexKey,
    },

    /// Support-inspection ridge indices repeat the same omitted simplex vertex.
    #[error(
        "ridge indices ({omit_a}, {omit_b}) must be distinct for support simplex {simplex_key:?} with {vertex_count} vertices"
    )]
    DuplicateRidgeSupportIndex {
        /// Ridge handle.
        ridge: RidgeHandle,
        /// First omitted index.
        omit_a: u8,
        /// Second omitted index.
        omit_b: u8,
        /// Number of vertices in the support simplex.
        vertex_count: usize,
        /// Support simplex key.
        simplex_key: SimplexKey,
    },

    /// A public flip failed.
    #[error("{move_kind} flip failed for selected {dimension}D benchmark support: {source}")]
    FlipFailed {
        /// Fixture dimension.
        dimension: usize,
        /// Flip move kind.
        move_kind: FlipMoveKind,
        /// Underlying flip failure.
        #[source]
        source: Box<FlipError>,
    },

    /// A public inverse flip failed.
    #[error("{move_kind} inverse failed after selected {dimension}D flip: {source}")]
    InverseFlipFailed {
        /// Fixture dimension.
        dimension: usize,
        /// Flip move kind.
        move_kind: FlipMoveKind,
        /// Underlying flip failure.
        #[source]
        source: Box<FlipError>,
    },

    /// A flip reported the wrong inserted-face arity for its move kind.
    #[error("{move_kind} flip inserted {observed} vertices instead of {expected}")]
    UnexpectedInsertedFaceVertexCount {
        /// Flip move kind.
        move_kind: FlipMoveKind,
        /// Observed inserted vertex count.
        observed: usize,
        /// Expected inserted vertex count.
        expected: usize,
    },

    /// A flip reported two inserted edge endpoints that do not form a real edge.
    #[error("{move_kind} flip reported an invalid inserted edge: {source}")]
    InvalidInsertedEdge {
        /// Flip move kind.
        move_kind: FlipMoveKind,
        /// Underlying edge-key parsing failure.
        #[source]
        source: EdgeKeyError,
    },

    /// A flip reported three inserted triangle vertices that do not form a real triangle.
    #[cfg_attr(
        not(feature = "slow-tests"),
        allow(
            dead_code,
            reason = "k=3 inverse roundtrip diagnostics are exercised by slow 4D/5D fixture tests"
        )
    )]
    #[error("{move_kind} flip reported an invalid inserted triangle: {source}")]
    InvalidInsertedTriangle {
        /// Flip move kind.
        move_kind: FlipMoveKind,
        /// Underlying triangle-handle parsing failure.
        #[source]
        source: TriangleHandleError,
    },

    /// A roundtrip produced a triangulation that failed validation.
    #[error("{context} produced invalid triangulation: {source}")]
    InvalidAfterRoundtrip {
        /// Roundtrip context label.
        context: String,
        /// Underlying validation failure.
        #[source]
        source: DelaunayTriangulationValidationError,
    },
}

/// Jaccard report category used by
/// [`FlipWorkflowError::JaccardReportFormatting`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JaccardReportKind {
    /// Vertex UUID set report.
    VertexUuids,
    /// Simplex-to-vertex UUID incidence report.
    SimplexIncidence,
}

impl fmt::Display for JaccardReportKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexUuids => f.write_str("vertex UUID"),
            Self::SimplexIncidence => f.write_str("simplex-incidence"),
        }
    }
}

/// Public flip move kind used by [`FlipWorkflowError`] and
/// [`FlipCandidateError`] diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlipMoveKind {
    /// k=1 insertion/removal move.
    K1,
    /// k=2 facet/edge move.
    K2,
    /// k=3 ridge/triangle move.
    K3,
}

impl fmt::Display for FlipMoveKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::K1 => f.write_str("k=1"),
            Self::K2 => f.write_str("k=2"),
            Self::K3 => f.write_str("k=3"),
        }
    }
}

/// Rejected candidate detail retained while searching for a benchmark support.
///
/// Candidate errors are nested inside [`FlipWorkflowError::NoFlippableFacet`]
/// and [`FlipWorkflowError::NoFlippableRidge`] so failed fixture setup reports
/// the last concrete reason without treating one rejected support as the whole
/// selector failure.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum FlipCandidateError {
    /// A candidate flip failed.
    #[error("{move_kind} candidate flip failed: {source}")]
    FlipFailed {
        /// Candidate move kind.
        move_kind: FlipMoveKind,
        /// Underlying flip failure.
        #[source]
        source: Box<FlipError>,
    },
    /// A candidate inverse flip failed.
    #[error("{move_kind} candidate inverse failed: {source}")]
    InverseFlipFailed {
        /// Candidate move kind.
        move_kind: FlipMoveKind,
        /// Underlying flip failure.
        #[source]
        source: Box<FlipError>,
    },
    /// A candidate flip reported the wrong inserted-face arity.
    #[error("{move_kind} candidate inserted {observed} vertices instead of {expected}")]
    UnexpectedInsertedFaceVertexCount {
        /// Candidate move kind.
        move_kind: FlipMoveKind,
        /// Observed inserted vertex count.
        observed: usize,
        /// Expected inserted vertex count.
        expected: usize,
    },
    /// A candidate flip reported two inserted edge endpoints that do not form a real edge.
    #[error("{move_kind} candidate reported an invalid inserted edge: {source}")]
    InvalidInsertedEdge {
        /// Candidate move kind.
        move_kind: FlipMoveKind,
        /// Underlying edge-key parsing failure.
        #[source]
        source: EdgeKeyError,
    },
    /// A candidate flip reported three inserted triangle vertices that do not form a real triangle.
    #[error("{move_kind} candidate reported an invalid inserted triangle: {source}")]
    InvalidInsertedTriangle {
        /// Candidate move kind.
        move_kind: FlipMoveKind,
        /// Underlying triangle-handle parsing failure.
        #[source]
        source: TriangleHandleError,
    },
    /// Candidate validation failed after the forward flip.
    #[error("{move_kind} candidate produced invalid triangulation: {source}")]
    InvalidAfterForwardFlip {
        /// Candidate move kind.
        move_kind: FlipMoveKind,
        /// Underlying validation failure.
        #[source]
        source: Box<InvariantError>,
    },
}

/// Candidate selection policy for benchmark flip supports.
///
/// This lets the same public flip workflow helpers exercise ordinary fixtures
/// and adversarial fixtures while keeping the benchmark registration code small.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateFilter {
    /// Accept any valid public flip candidate.
    Any,
    /// Accept only candidates touching a configured adversarial fixture feature.
    TouchesAdversarialFeature,
}

impl CandidateFilter {
    /// Returns whether the candidate support points satisfy this selection policy.
    fn accepts<const D: usize>(self, points: &[Point<D>]) -> bool {
        match self {
            Self::Any => true,
            Self::TouchesAdversarialFeature => points
                .iter()
                .any(|point| has_adversarial_coordinate_feature(point.coords())),
        }
    }
}

/// Canonical topology fingerprint for exact n=1 flip roundtrip checks.
///
/// The snapshot deliberately uses stable vertex UUIDs and sorted simplex
/// incidence rather than slot-map keys so it detects topological changes while
/// ignoring incidental key allocation order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopologySnapshot {
    vertex_uuids: Vec<Uuid>,
    simplex_vertex_uuids: Vec<Vec<Uuid>>,
}

/// Builds a deterministic PL-manifold triangulation for public flip fixtures.
///
/// The input order is preserved so benchmark candidate selection remains stable
/// across runs.
///
/// # Errors
///
/// Returns [`FlipWorkflowError::Construction`] when fixture construction fails
/// or the requested [`TopologyGuarantee::PLManifold`] cannot be satisfied.
pub fn build_flip_dt<const D: usize>(
    points: &[[f64; D]],
) -> FlipWorkflowResult<FlipTriangulation<D>> {
    let vertices = points
        .iter()
        .map(|coords| vertex!(*coords))
        .collect::<Result<Vec<Vertex<(), D>>, _>>()?;
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);

    DelaunayTriangulation::try_with_topology_guarantee_and_options(
        &RobustKernel::new(),
        &vertices,
        TopologyGuarantee::PLManifold,
        options,
    )
    .map_err(|source| FlipWorkflowError::Construction {
        dimension: D,
        source: Box::new(source),
    })
}

/// Captures vertex identity and simplex incidence for exact roundtrip checks.
///
/// # Errors
///
/// Returns [`FlipWorkflowError::DanglingSnapshotVertex`] if a simplex
/// references a vertex key that is missing from the triangulation data
/// structure.
pub fn snapshot_topology<const D: usize>(
    dt: &FlipTriangulation<D>,
) -> FlipWorkflowResult<TopologySnapshot> {
    let tds = dt.tds();
    let mut vertex_uuids = tds
        .vertices()
        .map(|(_, vertex)| vertex.uuid())
        .collect::<Vec<_>>();
    vertex_uuids.sort();

    let mut simplex_vertex_uuids = tds
        .simplices()
        .map(|(_, simplex)| {
            simplex
                .vertices()
                .iter()
                .map(|&vertex_key| {
                    tds.vertex(vertex_key)
                        .map(Vertex::uuid)
                        .ok_or(FlipWorkflowError::DanglingSnapshotVertex { vertex_key })
                })
                .collect::<FlipWorkflowResult<Vec<_>>>()
                .map(|mut uuids| {
                    uuids.sort();
                    uuids
                })
        })
        .collect::<FlipWorkflowResult<Vec<_>>>()?;
    simplex_vertex_uuids.sort();

    Ok(TopologySnapshot {
        vertex_uuids,
        simplex_vertex_uuids,
    })
}

/// Verifies that a triangulation still matches a previously captured topology.
///
/// Exact equality is the pass condition. When equality fails, the returned
/// error includes Jaccard similarity reports for vertex UUIDs and simplex
/// incidence so near misses are diagnosable without weakening the assertion.
///
/// # Errors
///
/// Returns snapshot errors from [`snapshot_topology`],
/// [`FlipWorkflowError::JaccardReportFormatting`] when report generation
/// fails, or [`FlipWorkflowError::TopologyMismatch`] when the actual topology
/// differs from `expected`.
pub fn assert_same_topology<const D: usize>(
    actual_dt: &FlipTriangulation<D>,
    expected: &TopologySnapshot,
    context: &str,
) -> FlipWorkflowResult<()> {
    let actual = snapshot_topology(actual_dt)?;
    if actual == *expected {
        return Ok(());
    }

    let expected_vertices = expected
        .vertex_uuids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let actual_vertices = actual.vertex_uuids.iter().copied().collect::<HashSet<_>>();
    let expected_simplices = expected
        .simplex_vertex_uuids
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let actual_simplices = actual
        .simplex_vertex_uuids
        .iter()
        .cloned()
        .collect::<HashSet<_>>();

    let vertex_report = format_jaccard_report(
        &expected_vertices,
        &actual_vertices,
        "expected vertex UUIDs",
        "actual vertex UUIDs",
    )
    .map_err(|source| FlipWorkflowError::JaccardReportFormatting {
        report: JaccardReportKind::VertexUuids,
        source,
    })?;
    let simplex_report = format_jaccard_report(
        &expected_simplices,
        &actual_simplices,
        "expected simplex UUID incidence",
        "actual simplex UUID incidence",
    )
    .map_err(|source| FlipWorkflowError::JaccardReportFormatting {
        report: JaccardReportKind::SimplexIncidence,
        source,
    })?;

    Err(FlipWorkflowError::TopologyMismatch {
        context: context.to_string(),
        vertex_report,
        simplex_report,
        expected: Box::new(expected.clone()),
        actual: Box::new(actual),
    })
}

/// Selects the largest-volume simplex that satisfies `filter`.
///
/// This keeps k=1 benchmark setup deterministic while avoiding degenerate
/// insertion supports.
///
/// # Errors
///
/// Returns [`FlipWorkflowError::NoNondegenerateSimplex`] when no
/// non-degenerate simplex satisfies the requested [`CandidateFilter`].
pub fn largest_volume_simplex<const D: usize>(
    dt: &FlipTriangulation<D>,
    filter: CandidateFilter,
) -> FlipWorkflowResult<SimplexKey> {
    let mut largest = None;
    for (simplex_key, _) in dt.simplices() {
        let points = simplex_points(dt, simplex_key)?;
        if !filter.accepts(&points) {
            continue;
        }
        let Ok(volume) = simplex_volume(&points) else {
            continue;
        };
        if largest
            .as_ref()
            .is_none_or(|(_, largest_volume)| volume.total_cmp(largest_volume).is_gt())
        {
            largest = Some((simplex_key, volume));
        }
    }

    largest
        .map(|(simplex_key, _)| simplex_key)
        .ok_or(FlipWorkflowError::NoNondegenerateSimplex {
            dimension: D,
            filter,
        })
}

/// Selects a deterministic k=2 facet candidate that satisfies `filter`.
///
/// When `require_inverse` is true, candidate discovery also proves that the
/// public inverse edge move succeeds after the forward k=2 flip.
///
/// # Errors
///
/// Returns support-inspection errors directly. Returns
/// [`FlipWorkflowError::NoFlippableFacet`] when every inspected candidate is
/// rejected by the filter, forward flip, post-flip validation, or optional
/// inverse check.
pub fn flippable_k2_facet<const D: usize>(
    dt: &FlipTriangulation<D>,
    require_inverse: bool,
    filter: CandidateFilter,
) -> FlipWorkflowResult<FacetHandle> {
    let mut last_error = None;
    for (simplex_key, simplex) in dt.simplices() {
        let Some(neighbors) = simplex.neighbors() else {
            continue;
        };

        for (facet_index, neighbor) in neighbors.enumerate() {
            if neighbor.is_none() {
                continue;
            }
            let Ok(facet_index) = u8::try_from(facet_index) else {
                continue;
            };
            let facet =
                FacetHandle::try_new(dt.tds(), simplex_key, facet_index).map_err(|source| {
                    FlipWorkflowError::FacetHandleConstruction {
                        simplex_key,
                        facet_index,
                        source,
                    }
                })?;
            let support = facet_support_points(dt, facet)?;
            if !filter.accepts(&support) {
                continue;
            }

            let mut trial = dt.clone();
            match trial.flip_k2(facet) {
                Ok(info) => {
                    if info.inserted_face_vertices.len() != 2 {
                        last_error = Some(Box::new(
                            FlipCandidateError::UnexpectedInsertedFaceVertexCount {
                                move_kind: FlipMoveKind::K2,
                                observed: info.inserted_face_vertices.len(),
                                expected: 2,
                            },
                        ));
                        continue;
                    }
                    let edge = match EdgeKey::try_new(
                        trial.tds(),
                        info.inserted_face_vertices[0],
                        info.inserted_face_vertices[1],
                    ) {
                        Ok(edge) => edge,
                        Err(source) => {
                            last_error = Some(Box::new(FlipCandidateError::InvalidInsertedEdge {
                                move_kind: FlipMoveKind::K2,
                                source,
                            }));
                            continue;
                        }
                    };
                    if let Err(source) = trial.as_triangulation().validate() {
                        last_error = Some(Box::new(FlipCandidateError::InvalidAfterForwardFlip {
                            move_kind: FlipMoveKind::K2,
                            source: Box::new(source),
                        }));
                        continue;
                    }
                    if require_inverse && let Err(source) = trial.flip_k2_inverse_from_edge(edge) {
                        last_error = Some(Box::new(FlipCandidateError::InverseFlipFailed {
                            move_kind: FlipMoveKind::K2,
                            source: Box::new(source),
                        }));
                        continue;
                    }
                    return Ok(facet);
                }
                Err(source) => {
                    last_error = Some(Box::new(FlipCandidateError::FlipFailed {
                        move_kind: FlipMoveKind::K2,
                        source: Box::new(source),
                    }));
                }
            }
        }
    }

    Err(FlipWorkflowError::NoFlippableFacet {
        dimension: D,
        filter,
        last_error,
    })
}

/// Selects a deterministic k=3 ridge candidate that satisfies `filter`.
///
/// When `require_inverse` is true, candidate discovery also proves that the
/// public inverse triangle move succeeds after the forward k=3 flip.
///
/// # Errors
///
/// Returns support-inspection errors directly. Returns
/// [`FlipWorkflowError::NoFlippableRidge`] when every inspected candidate is
/// rejected by the filter, forward flip, post-flip validation, or optional
/// inverse check.
pub fn flippable_k3_ridge<const D: usize>(
    dt: &FlipTriangulation<D>,
    require_inverse: bool,
    filter: CandidateFilter,
) -> FlipWorkflowResult<RidgeHandle> {
    let mut last_error = None;
    for (simplex_key, simplex) in dt.simplices() {
        let vertex_count = simplex.number_of_vertices();
        for i in 0..vertex_count {
            for j in (i + 1)..vertex_count {
                let Ok(omit_a) = u8::try_from(i) else {
                    continue;
                };
                let Ok(omit_b) = u8::try_from(j) else {
                    continue;
                };
                let ridge = RidgeHandle::try_new(dt.tds(), simplex_key, omit_a, omit_b).map_err(
                    |source| FlipWorkflowError::RidgeHandleConstruction {
                        simplex_key,
                        omit_a,
                        omit_b,
                        source: Box::new(source),
                    },
                )?;
                let support = ridge_support_points(dt, ridge)?;
                if !filter.accepts(&support) {
                    continue;
                }

                let mut trial = dt.clone();
                match trial.flip_k3(ridge) {
                    Ok(info) => {
                        if info.inserted_face_vertices.len() != 3 {
                            last_error = Some(Box::new(
                                FlipCandidateError::UnexpectedInsertedFaceVertexCount {
                                    move_kind: FlipMoveKind::K3,
                                    observed: info.inserted_face_vertices.len(),
                                    expected: 3,
                                },
                            ));
                            continue;
                        }
                        let triangle = match TriangleHandle::try_new(
                            info.inserted_face_vertices[0],
                            info.inserted_face_vertices[1],
                            info.inserted_face_vertices[2],
                        ) {
                            Ok(triangle) => triangle,
                            Err(source) => {
                                last_error =
                                    Some(Box::new(FlipCandidateError::InvalidInsertedTriangle {
                                        move_kind: FlipMoveKind::K3,
                                        source,
                                    }));
                                continue;
                            }
                        };
                        if let Err(source) = trial.as_triangulation().validate() {
                            last_error =
                                Some(Box::new(FlipCandidateError::InvalidAfterForwardFlip {
                                    move_kind: FlipMoveKind::K3,
                                    source: Box::new(source),
                                }));
                            continue;
                        }
                        if require_inverse
                            && let Err(source) = trial.flip_k3_inverse_from_triangle(triangle)
                        {
                            last_error = Some(Box::new(FlipCandidateError::InverseFlipFailed {
                                move_kind: FlipMoveKind::K3,
                                source: Box::new(source),
                            }));
                            continue;
                        }
                        return Ok(ridge);
                    }
                    Err(source) => {
                        last_error = Some(Box::new(FlipCandidateError::FlipFailed {
                            move_kind: FlipMoveKind::K3,
                            source: Box::new(source),
                        }));
                    }
                }
            }
        }
    }

    Err(FlipWorkflowError::NoFlippableRidge {
        dimension: D,
        filter,
        last_error,
    })
}

/// Executes a selected public k=2 flip without its inverse.
///
/// # Errors
///
/// Returns [`FlipWorkflowError::FlipFailed`] when the forward flip fails, or
/// [`FlipWorkflowError::UnexpectedInsertedFaceVertexCount`] when it does not
/// report the expected inserted edge.
pub fn forward_k2<const D: usize>(
    dt: &mut FlipTriangulation<D>,
    facet: FacetHandle,
) -> FlipWorkflowResult<()> {
    let info = dt
        .flip_k2(facet)
        .map_err(|source| FlipWorkflowError::FlipFailed {
            dimension: D,
            move_kind: FlipMoveKind::K2,
            source: Box::new(source),
        })?;
    if info.inserted_face_vertices.len() != 2 {
        return Err(FlipWorkflowError::UnexpectedInsertedFaceVertexCount {
            move_kind: FlipMoveKind::K2,
            observed: info.inserted_face_vertices.len(),
            expected: 2,
        });
    }
    Ok(())
}

/// Executes a public k=1 insert followed by its inverse remove.
///
/// # Errors
///
/// Returns [`FlipWorkflowError::MissingSimplex`],
/// [`FlipWorkflowError::MissingVertex`], or
/// [`FlipWorkflowError::SimplexVertexCountTooLarge`] while computing the
/// centroid. Returns [`FlipWorkflowError::FlipFailed`] for insertion failure,
/// [`FlipWorkflowError::MissingInsertedVertex`] if the inserted UUID cannot be
/// found, or [`FlipWorkflowError::InverseFlipFailed`] for removal failure.
pub fn roundtrip_k1<const D: usize>(
    dt: &mut FlipTriangulation<D>,
    simplex_key: SimplexKey,
) -> FlipWorkflowResult<()> {
    let new_vertex = vertex!(simplex_centroid(dt, simplex_key)?)?;
    let new_uuid = new_vertex.uuid();
    dt.flip_k1_insert(simplex_key, new_vertex)
        .map_err(|source| FlipWorkflowError::FlipFailed {
            dimension: D,
            move_kind: FlipMoveKind::K1,
            source: Box::new(source),
        })?;

    let new_key = dt
        .tds()
        .vertex_key_from_uuid(&new_uuid)
        .ok_or(FlipWorkflowError::MissingInsertedVertex { uuid: new_uuid })?;
    dt.flip_k1_remove(new_key)
        .map_err(|source| FlipWorkflowError::InverseFlipFailed {
            dimension: D,
            move_kind: FlipMoveKind::K1,
            source: Box::new(source),
        })
        .map(|_| ())
}

/// Executes a public k=2 flip followed by its inverse edge move.
///
/// # Errors
///
/// Returns [`FlipWorkflowError::FlipFailed`] when the forward flip fails,
/// [`FlipWorkflowError::UnexpectedInsertedFaceVertexCount`] when the forward
/// flip does not report an inserted edge, or
/// [`FlipWorkflowError::InverseFlipFailed`] when the inverse edge move fails.
pub fn roundtrip_k2<const D: usize>(
    dt: &mut FlipTriangulation<D>,
    facet: FacetHandle,
) -> FlipWorkflowResult<()> {
    let info = dt
        .flip_k2(facet)
        .map_err(|source| FlipWorkflowError::FlipFailed {
            dimension: D,
            move_kind: FlipMoveKind::K2,
            source: Box::new(source),
        })?;
    if info.inserted_face_vertices.len() != 2 {
        return Err(FlipWorkflowError::UnexpectedInsertedFaceVertexCount {
            move_kind: FlipMoveKind::K2,
            observed: info.inserted_face_vertices.len(),
            expected: 2,
        });
    }
    let edge = EdgeKey::try_new(
        dt.tds(),
        info.inserted_face_vertices[0],
        info.inserted_face_vertices[1],
    )
    .map_err(|source| FlipWorkflowError::InvalidInsertedEdge {
        move_kind: FlipMoveKind::K2,
        source,
    })?;
    dt.flip_k2_inverse_from_edge(edge)
        .map_err(|source| FlipWorkflowError::InverseFlipFailed {
            dimension: D,
            move_kind: FlipMoveKind::K2,
            source: Box::new(source),
        })
        .map(|_| ())
}

/// Executes a selected public k=3 flip without its inverse.
///
/// # Errors
///
/// Returns [`FlipWorkflowError::FlipFailed`] when the forward flip fails, or
/// [`FlipWorkflowError::UnexpectedInsertedFaceVertexCount`] when it does not
/// report the expected inserted triangle.
pub fn forward_k3<const D: usize>(
    dt: &mut FlipTriangulation<D>,
    ridge: RidgeHandle,
) -> FlipWorkflowResult<()> {
    let info = dt
        .flip_k3(ridge)
        .map_err(|source| FlipWorkflowError::FlipFailed {
            dimension: D,
            move_kind: FlipMoveKind::K3,
            source: Box::new(source),
        })?;
    if info.inserted_face_vertices.len() != 3 {
        return Err(FlipWorkflowError::UnexpectedInsertedFaceVertexCount {
            move_kind: FlipMoveKind::K3,
            observed: info.inserted_face_vertices.len(),
            expected: 3,
        });
    }
    Ok(())
}

/// Executes a public k=3 flip followed by its inverse triangle move.
///
/// # Errors
///
/// Returns [`FlipWorkflowError::FlipFailed`] when the forward flip fails,
/// [`FlipWorkflowError::UnexpectedInsertedFaceVertexCount`] when the forward
/// flip does not report an inserted triangle, or
/// [`FlipWorkflowError::InverseFlipFailed`] when the inverse triangle move
/// fails.
#[cfg_attr(
    not(feature = "slow-tests"),
    allow(
        dead_code,
        reason = "k=3 inverse roundtrips are exercised by slow 4D/5D fixture tests"
    )
)]
pub fn roundtrip_k3<const D: usize>(
    dt: &mut FlipTriangulation<D>,
    ridge: RidgeHandle,
) -> FlipWorkflowResult<()> {
    let info = dt
        .flip_k3(ridge)
        .map_err(|source| FlipWorkflowError::FlipFailed {
            dimension: D,
            move_kind: FlipMoveKind::K3,
            source: Box::new(source),
        })?;
    if info.inserted_face_vertices.len() != 3 {
        return Err(FlipWorkflowError::UnexpectedInsertedFaceVertexCount {
            move_kind: FlipMoveKind::K3,
            observed: info.inserted_face_vertices.len(),
            expected: 3,
        });
    }
    let triangle = TriangleHandle::try_new(
        info.inserted_face_vertices[0],
        info.inserted_face_vertices[1],
        info.inserted_face_vertices[2],
    )
    .map_err(|source| FlipWorkflowError::InvalidInsertedTriangle {
        move_kind: FlipMoveKind::K3,
        source,
    })?;
    dt.flip_k3_inverse_from_triangle(triangle)
        .map_err(|source| FlipWorkflowError::InverseFlipFailed {
            dimension: D,
            move_kind: FlipMoveKind::K3,
            source: Box::new(source),
        })
        .map(|_| ())
}

/// Verifies exact topology recovery for a selected k=1 roundtrip.
///
/// This is the n=1 ergodicity assertion used before benchmark timings are
/// emitted.
///
/// # Errors
///
/// Returns an error when snapshotting, the k=1 roundtrip,
/// [`DelaunayTriangulation::validate`] validation, or exact topology comparison
/// fails.
pub fn verify_k1_roundtrip<const D: usize>(
    base_dt: &FlipTriangulation<D>,
    simplex_key: SimplexKey,
    context: &str,
) -> FlipWorkflowResult<()> {
    let before = snapshot_topology(base_dt)?;
    let mut trial = base_dt.clone();
    roundtrip_k1(&mut trial, simplex_key)?;
    validate_topology_and_delaunay(&trial, context)?;
    assert_same_topology(&trial, &before, context)
}

/// Verifies exact topology recovery for a selected k=2 roundtrip.
///
/// This is the n=1 ergodicity assertion used before benchmark timings are
/// emitted.
///
/// # Errors
///
/// Returns an error when snapshotting, the k=2 roundtrip,
/// [`DelaunayTriangulation::validate`] validation, or exact topology comparison
/// fails.
pub fn verify_k2_roundtrip<const D: usize>(
    base_dt: &FlipTriangulation<D>,
    facet: FacetHandle,
    context: &str,
) -> FlipWorkflowResult<()> {
    let before = snapshot_topology(base_dt)?;
    let mut trial = base_dt.clone();
    roundtrip_k2(&mut trial, facet)?;
    validate_topology_and_delaunay(&trial, context)?;
    assert_same_topology(&trial, &before, context)
}

/// Verifies exact topology recovery for a selected k=3 roundtrip.
///
/// This is the n=1 ergodicity assertion used before benchmark timings are
/// emitted.
///
/// # Errors
///
/// Returns an error when snapshotting, the k=3 roundtrip,
/// [`DelaunayTriangulation::validate`] validation, or exact topology comparison
/// fails.
#[cfg_attr(
    not(feature = "slow-tests"),
    allow(
        dead_code,
        reason = "k=3 inverse roundtrips are exercised by slow 4D/5D fixture tests"
    )
)]
pub fn verify_k3_roundtrip<const D: usize>(
    base_dt: &FlipTriangulation<D>,
    ridge: RidgeHandle,
    context: &str,
) -> FlipWorkflowResult<()> {
    let before = snapshot_topology(base_dt)?;
    let mut trial = base_dt.clone();
    roundtrip_k3(&mut trial, ridge)?;
    validate_topology_and_delaunay(&trial, context)?;
    assert_same_topology(&trial, &before, context)
}

fn validate_topology_and_delaunay<const D: usize>(
    dt: &FlipTriangulation<D>,
    context: &str,
) -> FlipWorkflowResult<()> {
    dt.validate()
        .map_err(|source| FlipWorkflowError::InvalidAfterRoundtrip {
            context: context.to_string(),
            source,
        })
}

/// Reports whether a k=2 facet support touches an adversarial fixture feature.
///
/// # Errors
///
/// Returns support-inspection errors from `facet_support_points`.
pub fn facet_support_touches_adversarial_feature<const D: usize>(
    dt: &FlipTriangulation<D>,
    facet: FacetHandle,
) -> FlipWorkflowResult<bool> {
    facet_support_points(dt, facet)
        .map(|points| CandidateFilter::TouchesAdversarialFeature.accepts(&points))
}

/// Reports whether a k=3 ridge support touches an adversarial fixture feature.
///
/// # Errors
///
/// Returns support-inspection errors from `ridge_support_points`.
pub fn ridge_support_touches_adversarial_feature<const D: usize>(
    dt: &FlipTriangulation<D>,
    ridge: RidgeHandle,
) -> FlipWorkflowResult<bool> {
    ridge_support_points(dt, ridge)
        .map(|points| CandidateFilter::TouchesAdversarialFeature.accepts(&points))
}

/// Reports whether a k=1 simplex support touches an adversarial fixture feature.
///
/// # Errors
///
/// Returns support-inspection errors from [`simplex_points`].
pub fn simplex_touches_adversarial_feature<const D: usize>(
    dt: &FlipTriangulation<D>,
    simplex_key: SimplexKey,
) -> FlipWorkflowResult<bool> {
    simplex_points(dt, simplex_key)
        .map(|points| CandidateFilter::TouchesAdversarialFeature.accepts(&points))
}

/// Computes the centroid used as the inserted vertex for k=1 roundtrips.
///
/// # Errors
///
/// Returns an error when the simplex or one of its vertices is missing, or when
/// the simplex vertex count cannot be converted for averaging.
fn simplex_centroid<const D: usize>(
    dt: &FlipTriangulation<D>,
    simplex_key: SimplexKey,
) -> FlipWorkflowResult<[f64; D]> {
    let simplex = dt
        .tds()
        .simplex(simplex_key)
        .ok_or(FlipWorkflowError::MissingSimplex { simplex_key })?;

    let mut coords = [0.0_f64; D];
    for &vkey in simplex.vertices() {
        let vertex = dt
            .tds()
            .vertex(vkey)
            .ok_or(FlipWorkflowError::MissingVertex { vertex_key: vkey })?;
        let vcoords = vertex.point().coords();
        for (coord, vertex_coord) in coords.iter_mut().zip(vcoords) {
            *coord += *vertex_coord;
        }
    }

    let vertex_count = u32::try_from(simplex.vertices().len())
        .map(f64::from)
        .map_err(|source| FlipWorkflowError::SimplexVertexCountTooLarge { source })?;
    for coord in &mut coords {
        *coord /= vertex_count;
    }
    Ok(coords)
}

/// Collects the points belonging to a simplex.
///
/// # Errors
///
/// Returns an error when the simplex or one of its vertices is missing.
fn simplex_points<const D: usize>(
    dt: &FlipTriangulation<D>,
    simplex_key: SimplexKey,
) -> FlipWorkflowResult<Vec<Point<D>>> {
    let simplex = dt
        .tds()
        .simplex(simplex_key)
        .ok_or(FlipWorkflowError::MissingSimplex { simplex_key })?;
    vertex_points(dt, simplex.vertices())
}

/// Collects the union of vertices involved in a k=2 facet support.
///
/// # Errors
///
/// Returns an error when the facet simplex is missing, the facet index is out
/// of bounds, the facet is not interior, the neighboring simplex is missing, or
/// a support vertex is missing.
fn facet_support_points<const D: usize>(
    dt: &FlipTriangulation<D>,
    facet: FacetHandle,
) -> FlipWorkflowResult<Vec<Point<D>>> {
    let simplex =
        dt.tds()
            .simplex(facet.simplex_key())
            .ok_or(FlipWorkflowError::MissingSimplex {
                simplex_key: facet.simplex_key(),
            })?;
    let vertex_count = simplex.number_of_vertices();
    let facet_index = usize::from(facet.facet_index());
    if facet_index >= vertex_count {
        return Err(FlipWorkflowError::InvalidFacetSupportIndex {
            facet,
            facet_index: facet.facet_index(),
            vertex_count,
            simplex_key: facet.simplex_key(),
        });
    }
    let neighbor_key = simplex
        .neighbors()
        .and_then(|mut neighbors| neighbors.nth(facet_index).flatten())
        .ok_or(FlipWorkflowError::FacetWithoutInteriorNeighbor { facet })?;

    let mut keys = simplex.vertices().to_vec();
    let neighbor = dt
        .tds()
        .simplex(neighbor_key)
        .ok_or(FlipWorkflowError::MissingSimplex {
            simplex_key: neighbor_key,
        })?;
    keys.extend(neighbor.vertices());
    keys.sort_unstable();
    keys.dedup();
    vertex_points(dt, &keys)
}

/// Collects the union of simplex vertices across the full k=3 ridge star.
///
/// # Errors
///
/// Returns an error when the ridge simplex is missing, the ridge indices are
/// invalid for that simplex, ridge-star discovery fails, or one of its vertices
/// is missing.
fn ridge_support_points<const D: usize>(
    dt: &FlipTriangulation<D>,
    ridge: RidgeHandle,
) -> FlipWorkflowResult<Vec<Point<D>>> {
    let simplex =
        dt.tds()
            .simplex(ridge.simplex_key())
            .ok_or(FlipWorkflowError::MissingSimplex {
                simplex_key: ridge.simplex_key(),
            })?;
    let vertex_count = simplex.number_of_vertices();
    let omit_a = usize::from(ridge.omit_a());
    let omit_b = usize::from(ridge.omit_b());
    if omit_a >= vertex_count || omit_b >= vertex_count {
        return Err(FlipWorkflowError::InvalidRidgeSupportIndex {
            ridge,
            omit_a: ridge.omit_a(),
            omit_b: ridge.omit_b(),
            vertex_count,
            simplex_key: ridge.simplex_key(),
        });
    }
    if omit_a == omit_b {
        return Err(FlipWorkflowError::DuplicateRidgeSupportIndex {
            ridge,
            omit_a: ridge.omit_a(),
            omit_b: ridge.omit_b(),
            vertex_count,
            simplex_key: ridge.simplex_key(),
        });
    }

    let ridge_candidate = RidgeCandidate::<D>::try_from_vertices(
        simplex
            .vertices()
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != omit_a && *index != omit_b)
            .map(|(_, vertex_key)| *vertex_key),
    )
    .map_err(|source| FlipWorkflowError::InvalidRidgeCandidate { ridge, source })?;
    let star_simplices = ridge_star_simplices(dt.tds(), &ridge_candidate)
        .map_err(|source| ridge_star_error(ridge, source))?;

    let mut keys = Vec::new();
    for simplex_key in star_simplices {
        let star_simplex = dt
            .tds()
            .simplex(simplex_key)
            .ok_or(FlipWorkflowError::MissingSimplex { simplex_key })?;
        keys.extend(star_simplex.vertices());
    }
    keys.sort_unstable();
    keys.dedup();
    vertex_points(dt, &keys)
}

/// Preserves specific missing-simplex and missing-vertex support errors while
/// keeping other ridge-star failures attached to the inspected ridge.
fn ridge_star_error(ridge: RidgeHandle, source: ManifoldError) -> FlipWorkflowError {
    match source {
        ManifoldError::Tds(TdsError::SimplexNotFound { simplex_key, .. }) => {
            FlipWorkflowError::MissingSimplex { simplex_key }
        }
        ManifoldError::Tds(TdsError::VertexNotFound { vertex_key, .. }) => {
            FlipWorkflowError::MissingVertex { vertex_key }
        }
        source => FlipWorkflowError::RidgeStar {
            ridge,
            source: Box::new(source),
        },
    }
}

/// Resolves vertex keys to points.
///
/// # Errors
///
/// Returns an error when any requested vertex key is missing.
fn vertex_points<const D: usize>(
    dt: &FlipTriangulation<D>,
    keys: &[VertexKey],
) -> FlipWorkflowResult<Vec<Point<D>>> {
    keys.iter()
        .map(|vertex_key| {
            dt.tds()
                .vertex(*vertex_key)
                .map(|vertex| *vertex.point())
                .ok_or(FlipWorkflowError::MissingVertex {
                    vertex_key: *vertex_key,
                })
        })
        .collect()
}

/// Classifies whether coordinates match one of the adversarial fixture features.
///
/// This mirrors the stable/adversarial fixture construction contract so
/// selector tests can prove adversarial benchmark IDs actually touch
/// near-boundary, near-degenerate, or large-coordinate data.
fn has_adversarial_coordinate_feature<const D: usize>(coords: &[f64; D]) -> bool {
    let has_large_coordinate = coords.iter().any(|coord| coord.abs() >= 1.0e5);
    let has_near_boundary_nonzero = coords
        .iter()
        .any(|coord| *coord != 0.0 && coord.abs() <= 1.0e-8);
    let near_degenerate_center = match D {
        2 => all_close_to(coords, 0.5, 1.0e-8),
        3 => all_close_to(coords, 0.25, 1.0e-8),
        4 => all_close_to(coords, 0.20, 1.0e-8),
        5 => all_close_to(coords, 0.16, 1.0e-8),
        _ => false,
    };

    has_large_coordinate || has_near_boundary_nonzero || near_degenerate_center
}

/// Returns whether every coordinate is within `tolerance` of `target`.
fn all_close_to<const D: usize>(coords: &[f64; D], target: f64, tolerance: f64) -> bool {
    coords
        .iter()
        .all(|coord| (*coord - target).abs() <= tolerance)
}

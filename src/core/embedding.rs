//! Embedded-geometry validation for generic triangulations.
//!
//! This module owns Level 4 validation for generic [`Triangulation`](crate::Triangulation):
//! after the TDS and topology layers have certified a valid oriented simplicial
//! complex, the embedding layer verifies that maximal simplices are nondegenerate
//! and intersect only in their shared faces in the topology's active affine chart.

#![forbid(unsafe_code)]

use core::ops::ControlFlow;

use crate::core::collections::{
    FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SimplexVertexKeyBuffer, SimplexVertexUuidBuffer,
    SmallBuffer,
};
use crate::core::simplex::Simplex;
use crate::core::tds::{InvariantError, InvariantKind, SimplexKey, Tds, TdsError, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::validation::TriangulationValidationError;
use crate::geometry::embedding::{
    LabeledSimplexEmbedding, LabeledSimplexEmbeddingError, PeriodicSimplexSpanError,
    SimplexIntersectionFailure, axis_aligned_bounding_boxes_overlap, coordinate_range_for_axis,
    try_periodic_simplex_span, validate_simplex_embeddings_intersect_only_in_shared_faces,
};
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::predicates::Orientation;
use crate::geometry::robust_predicates::robust_orientation;
use crate::geometry::traits::coordinate::{
    CoordinateConversionError, CoordinateValidationError, InvalidCoordinateValue,
};
use crate::topology::traits::global_topology_model::{
    GlobalTopologyModel, GlobalTopologyModelError,
};
use crate::topology::traits::topological_space::TopologyKind;
use num_traits::ToPrimitive;
use thiserror::Error;
use uuid::Uuid;

/// Key- and UUID-based snapshot of one embedded simplex.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TriangulationEmbeddingSimplexDetail {
    /// Simplex key at validation time.
    pub key: SimplexKey,
    /// Simplex UUID at validation time.
    pub uuid: Uuid,
    /// Vertex keys stored by the simplex at validation time.
    pub vertices: SimplexVertexKeyBuffer,
    /// Vertex UUIDs stored by the simplex at validation time.
    pub vertex_uuids: SimplexVertexUuidBuffer,
}

/// Key- and UUID-based snapshot of one embedded simplex pair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TriangulationEmbeddingSimplexPairDetail {
    /// First simplex in the pair.
    pub first_simplex: TriangulationEmbeddingSimplexDetail,
    /// Second simplex in the pair.
    pub second_simplex: TriangulationEmbeddingSimplexDetail,
}

/// Detailed witness for an illegal embedded-simplex intersection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TriangulationEmbeddingIntersectionDetail {
    /// First simplex in the violating pair.
    pub first_simplex: TriangulationEmbeddingSimplexDetail,
    /// Second simplex in the violating pair.
    pub second_simplex: TriangulationEmbeddingSimplexDetail,
    /// Vertices shared by both simplices.
    pub shared_vertices: SimplexVertexKeyBuffer,
    /// UUIDs of vertices shared by both simplices.
    pub shared_vertex_uuids: SimplexVertexUuidBuffer,
    /// First-simplex vertices with positive barycentric weight at the witness.
    pub first_only_witness_vertices: SimplexVertexKeyBuffer,
    /// UUIDs of first-simplex vertices with positive barycentric weight at the witness.
    pub first_only_witness_vertex_uuids: SimplexVertexUuidBuffer,
    /// Second-simplex vertices with positive barycentric weight at the witness.
    pub second_only_witness_vertices: SimplexVertexKeyBuffer,
    /// UUIDs of second-simplex vertices with positive barycentric weight at the witness.
    pub second_only_witness_vertex_uuids: SimplexVertexUuidBuffer,
}

/// Invalid periodic-domain period observed during Level 4 embedding validation.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum PeriodicDomainPeriodError {
    /// A period was NaN or infinite.
    #[error("non-finite periodic domain period at axis {axis}: {period}")]
    NonFinitePeriod {
        /// Periodic axis with the invalid period.
        axis: usize,
        /// Classified invalid period value.
        period: InvalidCoordinateValue,
    },
    /// A finite period was zero or negative.
    #[error("non-positive periodic domain period at axis {axis}: {period}")]
    NonPositivePeriod {
        /// Periodic axis with the invalid period.
        axis: usize,
        /// Raw finite non-positive period.
        period: f64,
    },
}

impl From<PeriodicSimplexSpanError> for PeriodicDomainPeriodError {
    fn from(source: PeriodicSimplexSpanError) -> Self {
        match source {
            PeriodicSimplexSpanError::NonFinitePeriod { axis, period } => {
                Self::NonFinitePeriod { axis, period }
            }
            PeriodicSimplexSpanError::NonPositivePeriod { axis, period } => {
                Self::NonPositivePeriod { axis, period }
            }
        }
    }
}

/// Errors returned by embedded-geometry validation (Level 4).
///
/// This error type is independent of the Delaunay empty-circumsphere predicate:
/// it certifies that the generic triangulation is faithfully embedded in the
/// topology's supported affine chart before any Delaunay-specific predicate is
/// evaluated.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TriangulationEmbeddingValidationError {
    /// Lower-layer element or TDS structural validation failed (Levels 1-2).
    #[error(transparent)]
    Tds(Box<TdsError>),

    /// Lower-layer topology validation failed (Level 3).
    #[error(transparent)]
    Triangulation(Box<TriangulationValidationError>),

    /// Embedded-overlap validation is not yet defined for this topology model.
    #[error(
        "embedded validation is unsupported for {topology:?} topology in dimension {dimension}"
    )]
    UnsupportedTopology {
        /// Topology kind configured on the triangulation.
        topology: TopologyKind,
        /// Const-generic coordinate dimension.
        dimension: usize,
    },

    /// Topology-specific coordinate lifting failed while preparing an embedded simplex.
    #[error(
        "topology-specific lifting failed for simplex {simplex_uuid} (key {simplex_key:?}), vertex {vertex_key:?}: {source}"
    )]
    TopologyLifting {
        /// Simplex whose coordinates were being lifted.
        simplex_key: SimplexKey,
        /// UUID of the simplex whose coordinates were being lifted.
        simplex_uuid: Uuid,
        /// Vertex whose point triggered the lifting failure.
        vertex_key: VertexKey,
        /// UUID of the vertex whose point triggered the lifting failure.
        vertex_uuid: Uuid,
        /// Underlying topology model failure.
        #[source]
        source: GlobalTopologyModelError,
    },

    /// A simplex embedding reused a vertex label.
    #[error(
        "simplex {simplex_uuid} (key {simplex_key:?}) has duplicate embedding label {vertex_key:?} ({vertex_uuid}) at indices {first_index} and {duplicate_index}"
    )]
    DuplicateSimplexEmbeddingLabel {
        /// Key of the simplex with duplicate labels.
        simplex_key: SimplexKey,
        /// UUID of the simplex with duplicate labels.
        simplex_uuid: Uuid,
        /// Vertex-level diagnostic details for the malformed simplex.
        detail: Box<TriangulationEmbeddingSimplexDetail>,
        /// Duplicated vertex key.
        vertex_key: VertexKey,
        /// UUID of the duplicated vertex.
        vertex_uuid: Uuid,
        /// First embedding slot containing the label.
        first_index: usize,
        /// Later embedding slot containing the same label.
        duplicate_index: usize,
    },

    /// A simplex has exactly zero orientation and therefore zero D-volume.
    #[error("simplex {simplex_uuid} (key {simplex_key:?}) is degenerate in dimension {dimension}")]
    DegenerateSimplex {
        /// Key of the degenerate simplex.
        simplex_key: SimplexKey,
        /// UUID of the degenerate simplex.
        simplex_uuid: Uuid,
        /// Vertex-level diagnostic details for the degenerate simplex.
        detail: Box<TriangulationEmbeddingSimplexDetail>,
        /// Const-generic coordinate dimension.
        dimension: usize,
    },

    /// Coordinate validation failed while preparing an exact predicate input.
    #[error(
        "coordinate validation failed for simplex {simplex_uuid} (key {simplex_key:?}), vertex {vertex_key:?}: {source}"
    )]
    CoordinateValidation {
        /// Simplex whose coordinates were being validated.
        simplex_key: SimplexKey,
        /// UUID of the simplex whose coordinates were being validated.
        simplex_uuid: Uuid,
        /// Vertex whose point triggered the validation failure.
        vertex_key: VertexKey,
        /// UUID of the vertex whose point triggered the validation failure.
        vertex_uuid: Uuid,
        /// Underlying coordinate validation failure.
        #[source]
        source: CoordinateValidationError,
    },

    /// The exact orientation predicate failed for a simplex.
    #[error(
        "orientation predicate failed for simplex {simplex_uuid} (key {simplex_key:?}): {source}"
    )]
    PredicateFailed {
        /// Simplex whose orientation predicate failed.
        simplex_key: SimplexKey,
        /// UUID of the simplex whose orientation predicate failed.
        simplex_uuid: Uuid,
        /// Vertex-level diagnostic details for the simplex.
        detail: Box<TriangulationEmbeddingSimplexDetail>,
        /// Underlying coordinate conversion failure from the predicate boundary.
        #[source]
        source: CoordinateConversionError,
    },

    /// Exact rational barycentric construction found a singular simplex basis.
    #[error(
        "simplex {simplex_uuid} (key {simplex_key:?}) has a singular barycentric basis in dimension {dimension}"
    )]
    SingularBarycentricBasis {
        /// Simplex whose basis was singular.
        simplex_key: SimplexKey,
        /// UUID of the simplex whose basis was singular.
        simplex_uuid: Uuid,
        /// Vertex-level diagnostic details for the singular simplex.
        detail: Box<TriangulationEmbeddingSimplexDetail>,
        /// Const-generic coordinate dimension.
        dimension: usize,
    },

    /// Two maximal simplices intersect beyond the face spanned by their shared vertices.
    #[error(
        "simplices {first_simplex_uuid} (key {first_simplex_key:?}) and {second_simplex_uuid} (key {second_simplex_key:?}) intersect outside their shared face"
    )]
    SimplexIntersectionOutsideSharedFace {
        /// Key of the first offending simplex.
        first_simplex_key: SimplexKey,
        /// UUID of the first offending simplex.
        first_simplex_uuid: Uuid,
        /// Key of the second offending simplex.
        second_simplex_key: SimplexKey,
        /// UUID of the second offending simplex.
        second_simplex_uuid: Uuid,
        /// Vertex-level diagnostic details for the illegal intersection.
        detail: Box<TriangulationEmbeddingIntersectionDetail>,
    },

    /// A lifted periodic simplex spans at least one full period along an axis.
    ///
    /// Such a simplex cannot be certified as injective in one affine covering
    /// chart, so the quotient embedding is invalid before pairwise overlap
    /// checks run.
    #[error(
        "simplex {simplex_uuid} (key {simplex_key:?}) spans {span} along periodic axis {axis}, but the period is {period}"
    )]
    PeriodicSimplexSpansDomain {
        /// Key of the offending simplex.
        simplex_key: SimplexKey,
        /// UUID of the offending simplex.
        simplex_uuid: Uuid,
        /// Vertex-level diagnostic details for the offending simplex.
        detail: Box<TriangulationEmbeddingSimplexDetail>,
        /// Periodic axis whose lifted span is too wide.
        axis: usize,
        /// Lifted coordinate span along `axis`.
        span: f64,
        /// Fundamental-domain period along `axis`.
        period: f64,
    },

    /// A periodic domain period was invalid while checking embedded geometry.
    #[error(
        "invalid periodic domain period while validating simplex {simplex_uuid} (key {simplex_key:?}): {source}"
    )]
    InvalidPeriodicDomainPeriod {
        /// Key of the simplex being checked.
        simplex_key: SimplexKey,
        /// UUID of the simplex being checked.
        simplex_uuid: Uuid,
        /// Vertex-level diagnostic details for the simplex being checked.
        detail: Box<TriangulationEmbeddingSimplexDetail>,
        /// Underlying invalid-period error.
        #[source]
        source: PeriodicDomainPeriodError,
    },

    /// Periodic translate enumeration would require shifts outside the supported range.
    #[error(
        "periodic translate range for simplices {first_simplex_uuid} (key {first_simplex_key:?}) and {second_simplex_uuid} (key {second_simplex_key:?}) on axis {axis} exceeds i32 shift bounds: lower {lower_bound}, upper {upper_bound}"
    )]
    PeriodicTranslateRangeOverflow {
        /// Key of the first simplex in the pair.
        first_simplex_key: SimplexKey,
        /// UUID of the first simplex in the pair.
        first_simplex_uuid: Uuid,
        /// Key of the second simplex in the pair.
        second_simplex_key: SimplexKey,
        /// UUID of the second simplex in the pair.
        second_simplex_uuid: Uuid,
        /// Vertex-level diagnostic details for the pair.
        detail: Box<TriangulationEmbeddingSimplexPairDetail>,
        /// Periodic axis whose shift range overflowed.
        axis: usize,
        /// Lower floating-point shift bound before integer conversion.
        lower_bound: f64,
        /// Upper floating-point shift bound before integer conversion.
        upper_bound: f64,
    },

    /// A higher validation layer unexpectedly surfaced while running Level 4 validation.
    #[error("unexpected {kind:?} validation error while validating Level 4 embedding: {source}")]
    UnexpectedValidationLayer {
        /// Validation layer that leaked into the embedding boundary.
        kind: InvariantKind,
        /// Original typed validation error.
        #[source]
        source: Box<InvariantError>,
    },
}

impl From<TdsError> for TriangulationEmbeddingValidationError {
    fn from(source: TdsError) -> Self {
        Self::Tds(Box::new(source))
    }
}

impl From<TriangulationValidationError> for TriangulationEmbeddingValidationError {
    fn from(source: TriangulationValidationError) -> Self {
        Self::Triangulation(Box::new(source))
    }
}

/// Discriminant for compact Level 4 embedded-geometry validation summaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TriangulationEmbeddingValidationErrorKind {
    /// Lower-layer TDS validation failed.
    Tds,
    /// Lower-layer topology validation failed.
    Triangulation,
    /// The topology is not currently supported by embedded validation.
    UnsupportedTopology,
    /// Topology-specific coordinate lifting failed.
    TopologyLifting,
    /// A simplex embedding reused a vertex label.
    DuplicateSimplexEmbeddingLabel,
    /// A simplex has zero D-volume.
    DegenerateSimplex,
    /// Coordinate validation failed at the predicate boundary.
    CoordinateValidation,
    /// The robust orientation predicate failed.
    PredicateFailed,
    /// Exact barycentric coordinates could not be computed.
    SingularBarycentricBasis,
    /// Two simplices overlap outside their shared face.
    SimplexIntersectionOutsideSharedFace,
    /// A periodic simplex spans at least one full domain period.
    PeriodicSimplexSpansDomain,
    /// A periodic domain period was invalid.
    InvalidPeriodicDomainPeriod,
    /// Periodic translate enumeration exceeded supported shift bounds.
    PeriodicTranslateRangeOverflow,
    /// A higher validation layer unexpectedly surfaced during embedding validation.
    UnexpectedValidationLayer,
}

impl From<&TriangulationEmbeddingValidationError> for TriangulationEmbeddingValidationErrorKind {
    fn from(source: &TriangulationEmbeddingValidationError) -> Self {
        match source {
            TriangulationEmbeddingValidationError::Tds(_) => Self::Tds,
            TriangulationEmbeddingValidationError::Triangulation(_) => Self::Triangulation,
            TriangulationEmbeddingValidationError::UnsupportedTopology { .. } => {
                Self::UnsupportedTopology
            }
            TriangulationEmbeddingValidationError::TopologyLifting { .. } => Self::TopologyLifting,
            TriangulationEmbeddingValidationError::DuplicateSimplexEmbeddingLabel { .. } => {
                Self::DuplicateSimplexEmbeddingLabel
            }
            TriangulationEmbeddingValidationError::DegenerateSimplex { .. } => {
                Self::DegenerateSimplex
            }
            TriangulationEmbeddingValidationError::CoordinateValidation { .. } => {
                Self::CoordinateValidation
            }
            TriangulationEmbeddingValidationError::PredicateFailed { .. } => Self::PredicateFailed,
            TriangulationEmbeddingValidationError::SingularBarycentricBasis { .. } => {
                Self::SingularBarycentricBasis
            }
            TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace {
                ..
            } => Self::SimplexIntersectionOutsideSharedFace,
            TriangulationEmbeddingValidationError::PeriodicSimplexSpansDomain { .. } => {
                Self::PeriodicSimplexSpansDomain
            }
            TriangulationEmbeddingValidationError::InvalidPeriodicDomainPeriod { .. } => {
                Self::InvalidPeriodicDomainPeriod
            }
            TriangulationEmbeddingValidationError::PeriodicTranslateRangeOverflow { .. } => {
                Self::PeriodicTranslateRangeOverflow
            }
            TriangulationEmbeddingValidationError::UnexpectedValidationLayer { .. } => {
                Self::UnexpectedValidationLayer
            }
        }
    }
}

/// Structured Level 4 embedding validation report.
///
/// This report is the diagnostic counterpart to
/// [`Triangulation::is_valid_embedding`]. The fast-fail method returns the
/// first invalid embedding condition, while this report records every
/// simplex-level failure and every pairwise overlap failure that can be checked
/// after invalid simplices are excluded from pairwise intersection work.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct TriangulationEmbeddingValidationReport {
    /// Number of vertices in the triangulation when the report was generated.
    pub number_of_vertices: usize,
    /// Number of simplices in the triangulation when the report was generated.
    pub number_of_simplices: usize,
    /// Number of simplex embeddings prepared for Level 4 validation.
    pub checked_simplices: usize,
    /// Number of candidate simplex pairs examined by the overlap broad phase.
    ///
    /// For Euclidean charts this counts pairs whose bounding boxes overlap
    /// after the sweep-and-prune broad phase; for periodic charts it counts all
    /// non-degenerate pairs (exhaustive enumeration).
    pub checked_simplex_pairs: usize,
    /// Ordered list of Level 4 embedding violations.
    pub violations: Vec<TriangulationEmbeddingValidationError>,
}

impl TriangulationEmbeddingValidationReport {
    /// Returns `true` when no Level 4 embedding violations were found.
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
    /// std::assert_matches!(
    ///     dt.as_triangulation().embedding_report(),
    ///     Ok(report) if report.is_valid()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }
}

#[derive(Debug)]
struct EmbeddedSimplex<const D: usize> {
    key: SimplexKey,
    uuid: Uuid,
    vertex_keys: SimplexVertexKeyBuffer,
    vertex_uuids: SimplexVertexUuidBuffer,
    embedding: LabeledSimplexEmbedding<VertexKey, D>,
}

type PeriodicShiftRangeBuffer = SmallBuffer<(i32, i32), MAX_PRACTICAL_DIMENSION_SIZE>;

impl<const D: usize> EmbeddedSimplex<D> {
    /// Builds the lifted, labeled embedding for one TDS simplex while preserving
    /// simplex and vertex identities for later diagnostics.
    fn try_from_simplex<U, V>(
        tds: &Tds<U, V, D>,
        topology_model: &impl GlobalTopologyModel<D>,
        simplex_key: SimplexKey,
        simplex: &Simplex<V, D>,
    ) -> Result<Self, TriangulationEmbeddingValidationError> {
        let mut vertices = SimplexVertexKeyBuffer::with_capacity(simplex.number_of_vertices());
        let mut vertex_uuids = SimplexVertexUuidBuffer::with_capacity(simplex.number_of_vertices());
        let mut coords = SmallBuffer::<[f64; D], MAX_PRACTICAL_DIMENSION_SIZE>::with_capacity(
            simplex.number_of_vertices(),
        );

        let periodic_offsets = simplex.periodic_vertex_offsets();
        if let Some(offsets) = periodic_offsets
            && offsets.len() != simplex.number_of_vertices()
        {
            return Err(TdsError::DimensionMismatch {
                expected: simplex.number_of_vertices(),
                actual: offsets.len(),
                context: format!(
                    "simplex {:?} (key {simplex_key:?}) periodic offset count vs vertex count during embedding validation",
                    simplex.uuid(),
                ),
            }
            .into());
        }

        for (vertex_index, &vertex_key) in simplex.vertices().iter().enumerate() {
            let vertex = tds
                .vertex(vertex_key)
                .ok_or_else(|| TdsError::VertexNotFound {
                    vertex_key,
                    context: format!(
                        "embedded validation for simplex {:?} (key {simplex_key:?})",
                        simplex.uuid()
                    ),
                })?;
            vertices.push(vertex_key);
            vertex_uuids.push(vertex.uuid());
            let periodic_offset = periodic_offsets.map(|offsets| offsets[vertex_index]);
            let lifted_coords = topology_model
                .lift_for_orientation(*vertex.point().coords(), periodic_offset)
                .map_err(
                    |source| TriangulationEmbeddingValidationError::TopologyLifting {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        vertex_key,
                        vertex_uuid: vertex.uuid(),
                        source,
                    },
                )?;
            coords.push(lifted_coords);
        }

        let embedding =
            LabeledSimplexEmbedding::try_new(vertices.iter().copied(), coords.iter().copied())
                .map_err(|source| {
                    labeled_simplex_error_to_embedding_error(
                        source,
                        simplex_key,
                        simplex,
                        &vertices,
                        &vertex_uuids,
                    )
                })?;

        Ok(Self {
            key: simplex_key,
            uuid: simplex.uuid(),
            vertex_keys: vertices,
            vertex_uuids,
            embedding,
        })
    }

    /// Rehydrates one embedded vertex coordinate as a validated point for exact predicates.
    fn point_at(
        &self,
        vertex_index: usize,
    ) -> Result<Point<D>, TriangulationEmbeddingValidationError> {
        self.embedding.point_at(vertex_index).ok_or_else(|| {
            TdsError::DimensionMismatch {
                expected: self.embedding.labels().len(),
                actual: vertex_index.saturating_add(1),
                context: format!(
                    "embedded simplex {:?} (key {:?}) point index during Level 4 validation",
                    self.uuid, self.key,
                ),
            }
            .into()
        })
    }

    /// Finds a labeled vertex in this embedded simplex and validates its point coordinates.
    ///
    /// The full-facet shortcut uses keys rather than coordinate indices so its
    /// orientation predicates stay tied to the same vertex identities reported
    /// in Level 4 diagnostics.
    fn point_for_key(
        &self,
        vertex_key: VertexKey,
    ) -> Result<Point<D>, TriangulationEmbeddingValidationError> {
        let vertex_index = self
            .embedding
            .labels()
            .iter()
            .position(|candidate| *candidate == vertex_key)
            .ok_or_else(|| TdsError::VertexNotFound {
                vertex_key,
                context: format!(
                    "embedded simplex {:?} (key {:?}) facet-side validation",
                    self.uuid, self.key,
                ),
            })?;
        self.point_at(vertex_index)
    }

    /// Maps witness vertex keys back to UUIDs from this simplex snapshot.
    fn vertex_uuids_for_keys(&self, vertex_keys: &[VertexKey]) -> SimplexVertexUuidBuffer {
        let mut uuids = SimplexVertexUuidBuffer::with_capacity(vertex_keys.len());
        uuids.extend(vertex_keys.iter().filter_map(|vertex_key| {
            self.vertex_keys
                .iter()
                .zip(&self.vertex_uuids)
                .find_map(|(candidate, &uuid)| (candidate == vertex_key).then_some(uuid))
        }));
        uuids
    }

    /// Builds the public simplex detail payload reused by Level 4 error variants.
    fn detail(&self) -> TriangulationEmbeddingSimplexDetail {
        TriangulationEmbeddingSimplexDetail {
            key: self.key,
            uuid: self.uuid,
            vertices: self.vertex_keys.clone(),
            vertex_uuids: self.vertex_uuids.clone(),
        }
    }
}

/// Converts labeled simplex construction failures into Level 4 diagnostics
/// that preserve the owning simplex and vertex identities callers need for
/// repair planning.
fn labeled_simplex_error_to_embedding_error<V, const D: usize>(
    source: LabeledSimplexEmbeddingError,
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    vertex_keys: &SimplexVertexKeyBuffer,
    vertex_uuids: &SimplexVertexUuidBuffer,
) -> TriangulationEmbeddingValidationError {
    let (expected, actual) = match source {
        LabeledSimplexEmbeddingError::LabelCoordinateLengthMismatch {
            label_count,
            coordinate_count,
        } => (label_count, coordinate_count),
        LabeledSimplexEmbeddingError::InvalidArity { expected, actual } => (expected, actual),
        LabeledSimplexEmbeddingError::DuplicateLabel {
            first_index,
            duplicate_index,
        } => {
            return duplicate_simplex_embedding_label_error(
                simplex_key,
                simplex.uuid(),
                vertex_keys,
                vertex_uuids,
                first_index,
                duplicate_index,
                "duplicate embedding label during embedding validation",
            );
        }
        LabeledSimplexEmbeddingError::NonFiniteCoordinate {
            vertex_index,
            coordinate_index,
            coordinate_value,
        } => {
            let Some(&vertex_key) = vertex_keys.get(vertex_index) else {
                return TdsError::DimensionMismatch {
                    expected: vertex_keys.len(),
                    actual: vertex_index.saturating_add(1),
                    context: format!(
                        "simplex {:?} (key {simplex_key:?}) finite-coordinate diagnostic vertex index during embedding validation",
                        simplex.uuid(),
                    ),
                }
                .into();
            };
            let Some(&vertex_uuid) = vertex_uuids.get(vertex_index) else {
                return TdsError::DimensionMismatch {
                    expected: vertex_uuids.len(),
                    actual: vertex_index.saturating_add(1),
                    context: format!(
                        "simplex {:?} (key {simplex_key:?}) finite-coordinate diagnostic vertex UUID index during embedding validation",
                        simplex.uuid(),
                    ),
                }
                .into();
            };
            return TriangulationEmbeddingValidationError::CoordinateValidation {
                simplex_key,
                simplex_uuid: simplex.uuid(),
                vertex_key,
                vertex_uuid,
                source: CoordinateValidationError::InvalidCoordinate {
                    coordinate_index,
                    coordinate_value,
                    dimension: D,
                },
            };
        }
        LabeledSimplexEmbeddingError::InvalidPeriodicDomainPeriod { source } => {
            return TriangulationEmbeddingValidationError::InvalidPeriodicDomainPeriod {
                simplex_key,
                simplex_uuid: simplex.uuid(),
                detail: Box::new(TriangulationEmbeddingSimplexDetail {
                    key: simplex_key,
                    uuid: simplex.uuid(),
                    vertices: vertex_keys.clone(),
                    vertex_uuids: vertex_uuids.clone(),
                }),
                source: source.into(),
            };
        }
    };

    TdsError::DimensionMismatch {
        expected,
        actual,
        context: format!(
            "simplex {:?} (key {simplex_key:?}) arity during embedding validation",
            simplex.uuid(),
        ),
    }
    .into()
}

/// Preserves duplicate embedding labels as structured Level 4 diagnostics.
fn duplicate_simplex_embedding_label_error(
    simplex_key: SimplexKey,
    simplex_uuid: Uuid,
    vertex_keys: &SimplexVertexKeyBuffer,
    vertex_uuids: &SimplexVertexUuidBuffer,
    first_index: usize,
    duplicate_index: usize,
    context: &'static str,
) -> TriangulationEmbeddingValidationError {
    let Some(&vertex_key) = vertex_keys.get(first_index) else {
        return TdsError::DimensionMismatch {
            expected: vertex_keys.len(),
            actual: first_index.saturating_add(1),
            context: format!("{context} for simplex {simplex_uuid} (key {simplex_key:?})"),
        }
        .into();
    };
    let Some(&vertex_uuid) = vertex_uuids.get(first_index) else {
        return TdsError::DimensionMismatch {
            expected: vertex_uuids.len(),
            actual: first_index.saturating_add(1),
            context: format!(
                "{context} vertex UUID for simplex {simplex_uuid} (key {simplex_key:?})"
            ),
        }
        .into();
    };

    TriangulationEmbeddingValidationError::DuplicateSimplexEmbeddingLabel {
        simplex_key,
        simplex_uuid,
        detail: Box::new(TriangulationEmbeddingSimplexDetail {
            key: simplex_key,
            uuid: simplex_uuid,
            vertices: vertex_keys.clone(),
            vertex_uuids: vertex_uuids.clone(),
        }),
        vertex_key,
        vertex_uuid,
        first_index,
        duplicate_index,
    }
}

/// Converts translated embedded-simplex construction failures into the same
/// key- and UUID-rich public diagnostics as the primary embedding path.
fn labeled_simplex_error_to_embedded_simplex_error<const D: usize>(
    source: LabeledSimplexEmbeddingError,
    simplex: &EmbeddedSimplex<D>,
) -> TriangulationEmbeddingValidationError {
    let (expected, actual) = match source {
        LabeledSimplexEmbeddingError::LabelCoordinateLengthMismatch {
            label_count,
            coordinate_count,
        } => (label_count, coordinate_count),
        LabeledSimplexEmbeddingError::InvalidArity { expected, actual } => (expected, actual),
        LabeledSimplexEmbeddingError::DuplicateLabel {
            first_index,
            duplicate_index,
        } => {
            return duplicate_simplex_embedding_label_error(
                simplex.key,
                simplex.uuid,
                &simplex.vertex_keys,
                &simplex.vertex_uuids,
                first_index,
                duplicate_index,
                "duplicate translated embedding label during embedding validation",
            );
        }
        LabeledSimplexEmbeddingError::NonFiniteCoordinate {
            vertex_index,
            coordinate_index,
            coordinate_value,
        } => {
            let Some(&vertex_key) = simplex.vertex_keys.get(vertex_index) else {
                return TdsError::DimensionMismatch {
                    expected: simplex.vertex_keys.len(),
                    actual: vertex_index.saturating_add(1),
                    context: format!(
                        "simplex {:?} (key {:?}) finite-coordinate translated diagnostic vertex index during embedding validation",
                        simplex.uuid, simplex.key,
                    ),
                }
                .into();
            };
            let Some(&vertex_uuid) = simplex.vertex_uuids.get(vertex_index) else {
                return TdsError::DimensionMismatch {
                    expected: simplex.vertex_uuids.len(),
                    actual: vertex_index.saturating_add(1),
                    context: format!(
                        "simplex {:?} (key {:?}) finite-coordinate translated diagnostic vertex UUID index during embedding validation",
                        simplex.uuid, simplex.key,
                    ),
                }
                .into();
            };
            return TriangulationEmbeddingValidationError::CoordinateValidation {
                simplex_key: simplex.key,
                simplex_uuid: simplex.uuid,
                vertex_key,
                vertex_uuid,
                source: CoordinateValidationError::InvalidCoordinate {
                    coordinate_index,
                    coordinate_value,
                    dimension: D,
                },
            };
        }
        LabeledSimplexEmbeddingError::InvalidPeriodicDomainPeriod { source } => {
            return TriangulationEmbeddingValidationError::InvalidPeriodicDomainPeriod {
                simplex_key: simplex.key,
                simplex_uuid: simplex.uuid,
                detail: Box::new(simplex.detail()),
                source: source.into(),
            };
        }
    };

    TdsError::DimensionMismatch {
        expected,
        actual,
        context: format!(
            "simplex {:?} (key {:?}) arity during translated embedding validation",
            simplex.uuid, simplex.key,
        ),
    }
    .into()
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D> {
    /// Validates embedded geometry only (Level 4).
    ///
    /// This method assumes lower layers have already passed validation. Use
    /// [`validate_embedding`](Self::validate_embedding) for cumulative Levels
    /// 1-4 validation.
    ///
    /// Euclidean topology is validated in its ordinary affine chart. Toroidal
    /// topology is validated in the stored periodic covering-space charts and
    /// across periodic translates. Spherical and hyperbolic topology currently
    /// return [`TriangulationEmbeddingValidationError::UnsupportedTopology`]
    /// until their model-specific affine/projective chart validators are added.
    ///
    /// # Errors
    ///
    /// Returns [`TriangulationEmbeddingValidationError`] if the topology model is
    /// unsupported, a simplex is geometrically degenerate, a periodic simplex is
    /// not contained in a single covering chart, or two maximal simplices
    /// intersect outside their shared face.
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
    /// assert!(dt.as_triangulation().is_valid_embedding().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_valid_embedding(&self) -> Result<(), TriangulationEmbeddingValidationError> {
        if let Some(first_violation) = self.embedding_diagnostic()? {
            return Err(first_violation);
        }
        Ok(())
    }

    /// Returns the first actionable Level 4 embedding diagnostic, if any.
    ///
    /// This is the repair/retry-oriented counterpart to
    /// [`is_valid_embedding`](Self::is_valid_embedding). It returns at most one
    /// Level 4 violation with simplex keys, simplex UUIDs, and offending vertex
    /// keys/UUIDs where applicable.
    ///
    /// # Errors
    ///
    /// Returns [`TriangulationEmbeddingValidationError`] when simplex embedding
    /// preparation cannot continue because lower-layer TDS data are missing or
    /// malformed.
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
    /// std::assert_matches!(dt.as_triangulation().embedding_diagnostic(), Ok(None));
    /// # Ok(())
    /// # }
    /// ```
    pub fn embedding_diagnostic(
        &self,
    ) -> Result<Option<TriangulationEmbeddingValidationError>, TriangulationEmbeddingValidationError>
    {
        self.first_embedding_violation()
    }

    /// Builds a Level 4 embedding report with key- and UUID-based violation details.
    ///
    /// This method checks embedded geometry only. It does not run lower-layer
    /// TDS/topology validation and does not evaluate the Level 5 Delaunay
    /// property. Use [`validate_embedding`](Self::validate_embedding) for
    /// cumulative Levels 1-4 validation when pass/fail behavior is enough.
    ///
    /// # Errors
    ///
    /// Returns [`TriangulationEmbeddingValidationError`] when simplex embedding
    /// preparation cannot continue because lower-layer TDS data are missing or
    /// malformed. Ordinary Level 4 violations are returned inside the report.
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
    /// std::assert_matches!(
    ///     dt.as_triangulation().embedding_report(),
    ///     Ok(report) if report.is_valid()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn embedding_report(
        &self,
    ) -> Result<TriangulationEmbeddingValidationReport, TriangulationEmbeddingValidationError> {
        let topology_model = self.global_topology.model();
        let mut report = TriangulationEmbeddingValidationReport {
            number_of_vertices: self.tds.number_of_vertices(),
            number_of_simplices: self.tds.number_of_simplices(),
            checked_simplices: 0,
            checked_simplex_pairs: 0,
            violations: Vec::new(),
        };

        if !topology_model.supports_affine_embedding_validation() {
            report
                .violations
                .push(TriangulationEmbeddingValidationError::UnsupportedTopology {
                    topology: self.global_topology.kind(),
                    dimension: D,
                });
            return Ok(report);
        }

        let simplices = self.collect_embedded_simplices()?;
        report.checked_simplices = simplices.len();
        let periodic_domain = topology_model.periodic_domain();
        let periodic_periods = periodic_domain.map(|domain| *domain.periods());
        let mut invalid_simplex_keys = FastHashSet::default();

        for simplex in &simplices {
            if let Err(error) = validate_simplex_nondegenerate(simplex) {
                invalid_simplex_keys.insert(simplex.key);
                report.violations.push(error);
            }
            if let Some(domain) = periodic_domain
                && let Err(error) = validate_periodic_simplex_chart(simplex, domain.periods())
            {
                invalid_simplex_keys.insert(simplex.key);
                report.violations.push(error);
            }
        }

        let (checked_simplex_pairs, _) = for_each_candidate_simplex_pair::<D, ()>(
            &simplices,
            &invalid_simplex_keys,
            periodic_periods,
            |first, second| {
                if let Err(error) =
                    validate_topology_aware_simplex_pair(first, second, periodic_periods)
                {
                    report.violations.push(error);
                }
                ControlFlow::Continue(())
            },
        );
        report.checked_simplex_pairs = checked_simplex_pairs;

        Ok(report)
    }

    /// Performs cumulative validation for Levels 1-4.
    ///
    /// This validates:
    /// - **Levels 1-3** via [`Triangulation::validate`](Self::validate)
    /// - **Level 4** via [`Triangulation::is_valid_embedding`](Self::is_valid_embedding)
    ///
    /// # Errors
    ///
    /// Returns [`TriangulationEmbeddingValidationError`] if lower-layer
    /// validation fails, the topology cannot currently be embedded-validated,
    /// or embedded Euclidean geometry is invalid.
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
    /// assert!(dt.as_triangulation().validate_embedding().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate_embedding(&self) -> Result<(), TriangulationEmbeddingValidationError>
    where
        K: Kernel<D, Scalar = f64>,
        U: DataType,
        V: DataType,
    {
        self.validate().map_err(|error| match error {
            InvariantError::Tds(source) => source.into(),
            InvariantError::Triangulation(source) => source.into(),
            InvariantError::Embedding(source) => source,
            source @ InvariantError::Delaunay(_) => {
                TriangulationEmbeddingValidationError::UnexpectedValidationLayer {
                    kind: InvariantKind::DelaunayProperty,
                    source: Box::new(source),
                }
            }
        })?;
        self.is_valid_embedding()
    }

    /// Validates the Level 4 nondegeneracy invariant for a local simplex set.
    ///
    /// This intentionally does not perform pairwise overlap checks; insertion
    /// uses it as a cheap mutation-time guard so zero-volume simplices fail
    /// inside the existing rollback transaction. Full embedding validation
    /// remains the responsibility of [`is_valid_embedding`](Self::is_valid_embedding).
    pub(crate) fn validate_local_embedding_nondegeneracy(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<(), TriangulationEmbeddingValidationError> {
        let topology_model = self.global_topology.model();
        if !topology_model.supports_affine_embedding_validation() {
            return Ok(());
        }

        let periodic_domain = topology_model.periodic_domain();
        for &simplex_key in simplices {
            let simplex =
                self.tds
                    .simplex(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "local embedding nondegeneracy validation".to_string(),
                    })?;
            let embedded = EmbeddedSimplex::try_from_simplex(
                &self.tds,
                &topology_model,
                simplex_key,
                simplex,
            )?;
            validate_simplex_nondegenerate(&embedded)?;
            if let Some(domain) = periodic_domain {
                validate_periodic_simplex_chart(&embedded, domain.periods())?;
            }
        }

        Ok(())
    }

    /// Validates the Level 4 embedding invariant for a changed simplex scope.
    ///
    /// Insertion and repair already assume the pre-existing triangulation was
    /// embedding-valid before the local mutation. Under that precondition, only
    /// the changed simplices can introduce a new nondegenerate-simplex or
    /// pairwise-intersection violation, so this checks each scoped simplex
    /// against every candidate it can intersect instead of rescanning all old
    /// simplex pairs.
    pub(crate) fn validate_embedding_for_simplices(
        &self,
        local_simplices: &[SimplexKey],
    ) -> Result<(), TriangulationEmbeddingValidationError> {
        if local_simplices.is_empty() {
            return Ok(());
        }

        let topology_model = self.global_topology.model();
        if !topology_model.supports_affine_embedding_validation() {
            return Err(TriangulationEmbeddingValidationError::UnsupportedTopology {
                topology: self.global_topology.kind(),
                dimension: D,
            });
        }

        let mut local_simplex_keys = FastHashSet::default();
        local_simplex_keys.reserve(local_simplices.len());
        for &simplex_key in local_simplices {
            if !self.tds.contains_simplex(simplex_key) {
                return Err(TdsError::SimplexNotFound {
                    simplex_key,
                    context: "scoped embedding validation".to_string(),
                }
                .into());
            }
            local_simplex_keys.insert(simplex_key);
        }

        let simplices = self.collect_embedded_simplices()?;
        let periodic_domain = topology_model.periodic_domain();
        let periodic_periods = periodic_domain.map(|domain| *domain.periods());

        for simplex in &simplices {
            if !local_simplex_keys.contains(&simplex.key) {
                continue;
            }
            validate_simplex_nondegenerate(simplex)?;
            if let Some(domain) = periodic_domain {
                validate_periodic_simplex_chart(simplex, domain.periods())?;
            }
        }

        let empty_skip = FastHashSet::default();
        let (_, violation) =
            for_each_scoped_candidate_simplex_pair::<D, TriangulationEmbeddingValidationError>(
                &simplices,
                &empty_skip,
                &local_simplex_keys,
                periodic_periods,
                |first, second| match validate_topology_aware_simplex_pair(
                    first,
                    second,
                    periodic_periods,
                ) {
                    Ok(()) => ControlFlow::Continue(()),
                    Err(error) => ControlFlow::Break(error),
                },
            );

        if let Some(error) = violation {
            return Err(error);
        }

        Ok(())
    }

    /// Collects all simplex embeddings after applying the topology model's active chart.
    fn collect_embedded_simplices(
        &self,
    ) -> Result<Vec<EmbeddedSimplex<D>>, TriangulationEmbeddingValidationError> {
        let topology_model = self.global_topology.model();
        self.tds
            .simplices()
            .map(|(simplex_key, simplex)| {
                EmbeddedSimplex::try_from_simplex(&self.tds, &topology_model, simplex_key, simplex)
            })
            .collect()
    }

    fn first_embedding_violation(
        &self,
    ) -> Result<Option<TriangulationEmbeddingValidationError>, TriangulationEmbeddingValidationError>
    {
        let topology_model = self.global_topology.model();
        if !topology_model.supports_affine_embedding_validation() {
            return Ok(Some(
                TriangulationEmbeddingValidationError::UnsupportedTopology {
                    topology: self.global_topology.kind(),
                    dimension: D,
                },
            ));
        }

        let periodic_domain = topology_model.periodic_domain();
        let periodic_periods = periodic_domain.map(|domain| *domain.periods());
        let mut simplices = Vec::with_capacity(self.tds.number_of_simplices());
        for (simplex_key, simplex) in self.tds.simplices() {
            let embedded = EmbeddedSimplex::try_from_simplex(
                &self.tds,
                &topology_model,
                simplex_key,
                simplex,
            )?;
            if let Err(error) = validate_simplex_nondegenerate(&embedded) {
                return Ok(Some(error));
            }
            if let Some(domain) = periodic_domain
                && let Err(error) = validate_periodic_simplex_chart(&embedded, domain.periods())
            {
                return Ok(Some(error));
            }
            simplices.push(embedded);
        }

        let empty_skip: FastHashSet<SimplexKey> = FastHashSet::default();
        let (_, violation) =
            for_each_candidate_simplex_pair::<D, TriangulationEmbeddingValidationError>(
                &simplices,
                &empty_skip,
                periodic_periods,
                |first, second| match validate_topology_aware_simplex_pair(
                    first,
                    second,
                    periodic_periods,
                ) {
                    Ok(()) => ControlFlow::Continue(()),
                    Err(error) => ControlFlow::Break(error),
                },
            );

        Ok(violation)
    }
}

/// Dispatches pairwise overlap validation through Euclidean or periodic chart logic.
fn validate_topology_aware_simplex_pair<const D: usize>(
    first: &EmbeddedSimplex<D>,
    second: &EmbeddedSimplex<D>,
    periodic_periods: Option<[f64; D]>,
) -> Result<(), TriangulationEmbeddingValidationError> {
    let Some(periods) = periodic_periods else {
        if bounding_boxes_overlap(first, second) {
            if try_validate_full_facet_pair(first, second)? {
                return Ok(());
            }
            validate_simplex_pair_intersection(first, second)?;
        }
        return Ok(());
    };

    let shift_ranges = periodic_shift_ranges(first, second, &periods)?;
    let mut shift = [0_i32; D];
    validate_periodic_translates(first, second, &periods, &shift_ranges, 0, &mut shift)
}

/// Uses an exact side-of-facet test for adjacent simplices sharing a full facet.
///
/// When two nondegenerate D-simplices share D vertices, their intersection is
/// exactly the shared facet iff the two opposite vertices lie on opposite sides
/// of the shared facet. This avoids the more expensive barycentric intersection
/// solver for the common adjacent-pair case while preserving the same Level 4
/// error shape for invalid same-side embeddings.
fn try_validate_full_facet_pair<const D: usize>(
    first: &EmbeddedSimplex<D>,
    second: &EmbeddedSimplex<D>,
) -> Result<bool, TriangulationEmbeddingValidationError> {
    let mut shared = SimplexVertexKeyBuffer::new();
    let mut first_only = SimplexVertexKeyBuffer::new();
    let mut second_only = SimplexVertexKeyBuffer::new();

    for &vertex_key in &first.vertex_keys {
        if second.vertex_keys.contains(&vertex_key) {
            shared.push(vertex_key);
        } else {
            first_only.push(vertex_key);
        }
    }
    for &vertex_key in &second.vertex_keys {
        if !first.vertex_keys.contains(&vertex_key) {
            second_only.push(vertex_key);
        }
    }

    if shared.len() != D || first_only.len() != 1 || second_only.len() != 1 {
        return Ok(false);
    }

    let first_orientation = orientation_against_shared_facet(first, &shared, first_only[0])?;
    let second_orientation = orientation_against_shared_facet(second, &shared, second_only[0])?;
    match (first_orientation, second_orientation) {
        (Orientation::POSITIVE, Orientation::NEGATIVE)
        | (Orientation::NEGATIVE, Orientation::POSITIVE) => Ok(true),
        (
            Orientation::POSITIVE | Orientation::NEGATIVE,
            Orientation::POSITIVE | Orientation::NEGATIVE,
        ) => Err(shared_facet_same_side_intersection(
            first,
            second,
            shared,
            first_only,
            second_only,
        )),
        (Orientation::DEGENERATE, _) => {
            Err(TriangulationEmbeddingValidationError::DegenerateSimplex {
                simplex_key: first.key,
                simplex_uuid: first.uuid,
                detail: Box::new(first.detail()),
                dimension: D,
            })
        }
        (_, Orientation::DEGENERATE) => {
            Err(TriangulationEmbeddingValidationError::DegenerateSimplex {
                simplex_key: second.key,
                simplex_uuid: second.uuid,
                detail: Box::new(second.detail()),
                dimension: D,
            })
        }
    }
}

/// Computes which side of a shared facet the opposite vertex occupies.
///
/// The point order is the shared facet vertices followed by one opposite
/// vertex, so the sign can be compared between adjacent simplices without
/// constructing a barycentric intersection system.
fn orientation_against_shared_facet<const D: usize>(
    simplex: &EmbeddedSimplex<D>,
    shared: &SimplexVertexKeyBuffer,
    opposite: VertexKey,
) -> Result<Orientation, TriangulationEmbeddingValidationError> {
    let mut points = SmallBuffer::<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>::with_capacity(D + 1);
    for &vertex_key in shared {
        points.push(simplex.point_for_key(vertex_key)?);
    }
    points.push(simplex.point_for_key(opposite)?);

    robust_orientation(&points).map_err(|source| {
        TriangulationEmbeddingValidationError::PredicateFailed {
            simplex_key: simplex.key,
            simplex_uuid: simplex.uuid,
            detail: Box::new(simplex.detail()),
            source,
        }
    })
}

/// Builds the standard Level 4 overlap diagnostic for a failed facet-side test.
///
/// Keeping the same [`TriangulationEmbeddingValidationError`] variant as the
/// barycentric path lets repair/report callers consume one error contract
/// regardless of which validator found the illegal intersection.
fn shared_facet_same_side_intersection<const D: usize>(
    first: &EmbeddedSimplex<D>,
    second: &EmbeddedSimplex<D>,
    shared_vertices: SimplexVertexKeyBuffer,
    first_only_witness_vertices: SimplexVertexKeyBuffer,
    second_only_witness_vertices: SimplexVertexKeyBuffer,
) -> TriangulationEmbeddingValidationError {
    let shared_vertex_uuids = first.vertex_uuids_for_keys(&shared_vertices);
    let first_only_witness_vertex_uuids = first.vertex_uuids_for_keys(&first_only_witness_vertices);
    let second_only_witness_vertex_uuids =
        second.vertex_uuids_for_keys(&second_only_witness_vertices);

    TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace {
        first_simplex_key: first.key,
        first_simplex_uuid: first.uuid,
        second_simplex_key: second.key,
        second_simplex_uuid: second.uuid,
        detail: Box::new(TriangulationEmbeddingIntersectionDetail {
            first_simplex: first.detail(),
            second_simplex: second.detail(),
            shared_vertices,
            shared_vertex_uuids,
            first_only_witness_vertices,
            first_only_witness_vertex_uuids,
            second_only_witness_vertices,
            second_only_witness_vertex_uuids,
        }),
    }
}

/// Recursively checks every periodic translate that can overlap two simplex boxes.
fn validate_periodic_translates<const D: usize>(
    first: &EmbeddedSimplex<D>,
    second: &EmbeddedSimplex<D>,
    periods: &[f64; D],
    shift_ranges: &[(i32, i32)],
    axis: usize,
    shift: &mut [i32; D],
) -> Result<(), TriangulationEmbeddingValidationError> {
    if axis == D {
        let translated = translated_simplex(second, periods, shift)?;
        if bounding_boxes_overlap(first, &translated) {
            validate_simplex_pair_intersection(first, &translated)?;
        }
        return Ok(());
    }

    let (start, end) = shift_ranges[axis];
    for value in start..=end {
        shift[axis] = value;
        validate_periodic_translates(first, second, periods, shift_ranges, axis + 1, shift)?;
    }
    Ok(())
}

/// Computes the finite integer shift range needed to test possible periodic overlaps.
fn periodic_shift_ranges<const D: usize>(
    first: &EmbeddedSimplex<D>,
    second: &EmbeddedSimplex<D>,
    periods: &[f64; D],
) -> Result<PeriodicShiftRangeBuffer, TriangulationEmbeddingValidationError> {
    (0..D)
        .map(|axis| {
            let (first_min, first_max) = coordinate_range_for_axis(&first.embedding, axis)
                .expect("axis generated from 0..D must be valid");
            let (second_min, second_max) = coordinate_range_for_axis(&second.embedding, axis)
                .expect("axis generated from 0..D must be valid");
            let period = periods[axis];
            let lower_bound = ((first_min - second_max) / period).floor();
            let upper_bound = ((first_max - second_min) / period).ceil();
            let Some(start) = lower_bound.to_i32() else {
                return Err(periodic_translate_range_overflow(
                    first,
                    second,
                    axis,
                    lower_bound,
                    upper_bound,
                ));
            };
            let Some(end) = upper_bound.to_i32() else {
                return Err(periodic_translate_range_overflow(
                    first,
                    second,
                    axis,
                    lower_bound,
                    upper_bound,
                ));
            };
            Ok((start, end))
        })
        .collect()
}

/// Builds the shared diagnostic for periodic shift bounds that cannot fit in `i32`.
fn periodic_translate_range_overflow<const D: usize>(
    first: &EmbeddedSimplex<D>,
    second: &EmbeddedSimplex<D>,
    axis: usize,
    lower_bound: f64,
    upper_bound: f64,
) -> TriangulationEmbeddingValidationError {
    TriangulationEmbeddingValidationError::PeriodicTranslateRangeOverflow {
        first_simplex_key: first.key,
        first_simplex_uuid: first.uuid,
        second_simplex_key: second.key,
        second_simplex_uuid: second.uuid,
        detail: Box::new(TriangulationEmbeddingSimplexPairDetail {
            first_simplex: first.detail(),
            second_simplex: second.detail(),
        }),
        axis,
        lower_bound,
        upper_bound,
    }
}

/// Translates one embedded simplex into a neighboring periodic chart.
fn translated_simplex<const D: usize>(
    simplex: &EmbeddedSimplex<D>,
    periods: &[f64; D],
    shift: &[i32; D],
) -> Result<EmbeddedSimplex<D>, TriangulationEmbeddingValidationError> {
    let embedding = simplex
        .embedding
        .try_translated(periods, shift)
        .map_err(|source| labeled_simplex_error_to_embedded_simplex_error(source, simplex))?;
    Ok(EmbeddedSimplex {
        key: simplex.key,
        uuid: simplex.uuid,
        vertex_keys: simplex.vertex_keys.clone(),
        vertex_uuids: simplex.vertex_uuids.clone(),
        embedding,
    })
}

/// Rejects a periodic simplex whose lifted vertices cannot fit in one chart.
fn validate_periodic_simplex_chart<const D: usize>(
    simplex: &EmbeddedSimplex<D>,
    periods: &[f64; D],
) -> Result<(), TriangulationEmbeddingValidationError> {
    let span = try_periodic_simplex_span(&simplex.embedding, periods).map_err(|source| {
        TriangulationEmbeddingValidationError::InvalidPeriodicDomainPeriod {
            simplex_key: simplex.key,
            simplex_uuid: simplex.uuid,
            detail: Box::new(simplex.detail()),
            source: source.into(),
        }
    })?;
    if let Some(span) = span {
        return Err(
            TriangulationEmbeddingValidationError::PeriodicSimplexSpansDomain {
                simplex_key: simplex.key,
                simplex_uuid: simplex.uuid,
                detail: Box::new(simplex.detail()),
                axis: span.axis,
                span: span.span,
                period: span.period,
            },
        );
    }
    Ok(())
}

/// Rejects zero-volume simplices before pairwise overlap validation runs.
fn validate_simplex_nondegenerate<const D: usize>(
    simplex: &EmbeddedSimplex<D>,
) -> Result<(), TriangulationEmbeddingValidationError> {
    let points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
        (0..simplex.embedding.labels().len())
            .map(|index| simplex.point_at(index))
            .collect::<Result<_, _>>()?;

    match robust_orientation(&points) {
        Ok(Orientation::POSITIVE | Orientation::NEGATIVE) => Ok(()),
        Ok(Orientation::DEGENERATE) => {
            Err(TriangulationEmbeddingValidationError::DegenerateSimplex {
                simplex_key: simplex.key,
                simplex_uuid: simplex.uuid,
                detail: Box::new(simplex.detail()),
                dimension: D,
            })
        }
        Err(source) => Err(TriangulationEmbeddingValidationError::PredicateFailed {
            simplex_key: simplex.key,
            simplex_uuid: simplex.uuid,
            detail: Box::new(simplex.detail()),
            source,
        }),
    }
}

/// Applies the cheap bounding-box prefilter before exact intersection work.
fn bounding_boxes_overlap<const D: usize>(
    first: &EmbeddedSimplex<D>,
    second: &EmbeddedSimplex<D>,
) -> bool {
    axis_aligned_bounding_boxes_overlap(&first.embedding, &second.embedding)
}

/// Converts pure simplex-intersection failures into triangulation-level diagnostics.
fn validate_simplex_pair_intersection<const D: usize>(
    first: &EmbeddedSimplex<D>,
    second: &EmbeddedSimplex<D>,
) -> Result<(), TriangulationEmbeddingValidationError> {
    match validate_simplex_embeddings_intersect_only_in_shared_faces(
        &first.embedding,
        &second.embedding,
    ) {
        Ok(()) => Ok(()),
        Err(SimplexIntersectionFailure::SingularBarycentricBasis) => Err(
            TriangulationEmbeddingValidationError::SingularBarycentricBasis {
                simplex_key: first.key,
                simplex_uuid: first.uuid,
                detail: Box::new(first.detail()),
                dimension: D,
            },
        ),
        Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { witness, .. }) => {
            let shared_vertex_uuids = first.vertex_uuids_for_keys(&witness.shared);
            let first_only_witness_vertex_uuids =
                first.vertex_uuids_for_keys(&witness.first_only_witness);
            let second_only_witness_vertex_uuids =
                second.vertex_uuids_for_keys(&witness.second_only_witness);
            Err(
                TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace {
                    first_simplex_key: first.key,
                    first_simplex_uuid: first.uuid,
                    second_simplex_key: second.key,
                    second_simplex_uuid: second.uuid,
                    detail: Box::new(TriangulationEmbeddingIntersectionDetail {
                        first_simplex: first.detail(),
                        second_simplex: second.detail(),
                        shared_vertices: witness.shared,
                        shared_vertex_uuids,
                        first_only_witness_vertices: witness.first_only_witness,
                        first_only_witness_vertex_uuids,
                        second_only_witness_vertices: witness.second_only_witness,
                        second_only_witness_vertex_uuids,
                    }),
                },
            )
        }
    }
}

/// Axis-aligned bounding box for one embedded simplex, tagged with its index
/// in the validated simplex list.
#[derive(Clone, Copy, Debug)]
struct SimplexBoundingBox<const D: usize> {
    /// Index of the owning simplex in the embedded-simplex slice.
    simplex_index: usize,
    /// Per-axis lower bounds of the simplex vertices.
    min: [f64; D],
    /// Per-axis upper bounds of the simplex vertices.
    max: [f64; D],
}

impl<const D: usize> SimplexBoundingBox<D> {
    /// Computes the bounding box of an embedded simplex from its lifted coordinates.
    fn from_embedded(simplex_index: usize, simplex: &EmbeddedSimplex<D>) -> Self {
        let mut min = [f64::INFINITY; D];
        let mut max = [f64::NEG_INFINITY; D];
        for coords in simplex.embedding.coordinates() {
            for (axis, &value) in coords.iter().enumerate() {
                min[axis] = min[axis].min(value);
                max[axis] = max[axis].max(value);
            }
        }
        Self {
            simplex_index,
            min,
            max,
        }
    }

    /// Returns whether two boxes overlap on every axis.
    ///
    /// Two axis-aligned boxes intersect if and only if their projections
    /// overlap on every coordinate axis (the separating-axis test for AABBs;
    /// see Ericson, *Real-Time Collision Detection*, ch. 4-5).
    fn overlaps(&self, other: &Self) -> bool {
        (0..D).all(|axis| self.max[axis] >= other.min[axis] && other.max[axis] >= self.min[axis])
    }
}

/// Returns the axis with the largest global coordinate extent across all boxes.
///
/// Sweeping along the widest axis keeps the active set small, which is what
/// makes sweep-and-prune near-linear in practice.
fn widest_extent_axis<const D: usize>(boxes: &[SimplexBoundingBox<D>]) -> usize {
    let mut global_min = [f64::INFINITY; D];
    let mut global_max = [f64::NEG_INFINITY; D];
    for bounding_box in boxes {
        for (axis, (&min, &max)) in bounding_box.min.iter().zip(&bounding_box.max).enumerate() {
            global_min[axis] = global_min[axis].min(min);
            global_max[axis] = global_max[axis].max(max);
        }
    }
    (0..D)
        .map(|axis| (axis, global_max[axis] - global_min[axis]))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(axis, _)| axis)
}

/// Visits candidate overlapping simplex pairs for Level 4 embedding validation.
///
/// The all-pairs intersection test is `O(S^2)` in the number of simplices,
/// which dominates validation on large triangulations. For the Euclidean
/// affine chart this routine uses a **sweep-and-prune** broad phase over
/// axis-aligned bounding boxes (AABBs) to enumerate only pairs whose boxes
/// overlap, then hands each candidate to `on_pair` for the exact intersection
/// test. Returning [`ControlFlow::Break`] stops early (used by fast-fail
/// validation); the returned tuple reports the number of candidate pairs
/// examined and the break payload, if any.
///
/// # Soundness ("provably misses nothing")
///
/// Two AABBs intersect if and only if their projections overlap on every
/// coordinate axis (the separating-axis test for boxes). Sweep-and-prune sorts
/// boxes by their lower endpoint on one axis and, when processing a box `b`,
/// retires only active boxes whose upper endpoint precedes `b`'s lower endpoint
/// on that axis. Every still-active box therefore overlaps `b` on the sweep
/// axis, so the examined pairs are a superset of all pairs that overlap on
/// *every* axis. No intersecting simplex pair can be skipped, so replacing the
/// quadratic scan with this broad phase preserves Level 4 correctness while
/// only pruning pairs that provably cannot intersect.
///
/// Periodic (toroidal) charts are excluded: a simplex near one boundary can
/// overlap another near the opposite boundary through a wrap-around translate
/// whose lifted-chart AABB is far away, so a lifted-coordinate sweep is not
/// sound. Those charts (and the degenerate `D == 0` chart, which has no sweep
/// axis) retain exhaustive pairwise enumeration until a periodic-aware broad
/// phase is added.
///
/// # Complexity
///
/// Euclidean: about `O(S log S)` for triangulations with bounded local overlap
/// (sorting dominates); worst case `O(S^2)` when many boxes overlap on the
/// sweep axis. Periodic: `O(S^2)`.
///
/// # References
///
/// - Cohen, Lin, Manocha, and Ponamgi, "I-COLLIDE" (1995): sweep-and-prune.
/// - Baraff, "Dynamic Simulation of Non-Penetrating Rigid Bodies" (1992):
///   coordinate sort-and-sweep.
/// - Ericson, *Real-Time Collision Detection* (2005), ch. 7 (sweep-and-prune)
///   and ch. 4-5 (AABB separating-axis test).
///
/// See `REFERENCES.md`, "Embedded-Geometry Overlap Detection (Level 4 Validation)".
fn for_each_candidate_simplex_pair<const D: usize, B>(
    simplices: &[EmbeddedSimplex<D>],
    skip: &FastHashSet<SimplexKey>,
    periodic_periods: Option<[f64; D]>,
    on_pair: impl FnMut(&EmbeddedSimplex<D>, &EmbeddedSimplex<D>) -> ControlFlow<B>,
) -> (usize, Option<B>) {
    // Lifted-chart AABBs cannot express wrap-around overlaps, and a degenerate
    // 0-dimensional chart has no sweep axis, so both fall back to exhaustive
    // pairwise enumeration.
    if periodic_periods.is_some() || D == 0 {
        return exhaustive_candidate_simplex_pairs(simplices, skip, on_pair);
    }
    sweep_and_prune_candidate_simplex_pairs(simplices, skip, on_pair)
}

/// Visits candidate pairs where at least one simplex belongs to a changed scope.
fn for_each_scoped_candidate_simplex_pair<const D: usize, B>(
    simplices: &[EmbeddedSimplex<D>],
    skip: &FastHashSet<SimplexKey>,
    scope: &FastHashSet<SimplexKey>,
    periodic_periods: Option<[f64; D]>,
    mut on_pair: impl FnMut(&EmbeddedSimplex<D>, &EmbeddedSimplex<D>) -> ControlFlow<B>,
) -> (usize, Option<B>) {
    if scope.is_empty() {
        return for_each_candidate_simplex_pair(simplices, skip, periodic_periods, on_pair);
    }
    if periodic_periods.is_some() || D == 0 {
        return scoped_exhaustive_candidate_simplex_pairs(simplices, skip, scope, on_pair);
    }
    sweep_and_prune_candidate_simplex_pairs(simplices, skip, |first, second| {
        if scope.contains(&first.key) || scope.contains(&second.key) {
            on_pair(first, second)
        } else {
            ControlFlow::Continue(())
        }
    })
}

/// Exhaustive `O(S^2)` pairwise enumeration over non-skipped simplices.
fn exhaustive_candidate_simplex_pairs<const D: usize, B>(
    simplices: &[EmbeddedSimplex<D>],
    skip: &FastHashSet<SimplexKey>,
    mut on_pair: impl FnMut(&EmbeddedSimplex<D>, &EmbeddedSimplex<D>) -> ControlFlow<B>,
) -> (usize, Option<B>) {
    let mut examined = 0_usize;
    for (first_index, first_simplex) in simplices.iter().enumerate() {
        if skip.contains(&first_simplex.key) {
            continue;
        }

        for second_simplex in &simplices[first_index + 1..] {
            if skip.contains(&second_simplex.key) {
                continue;
            }

            examined += 1;
            if let ControlFlow::Break(value) = on_pair(first_simplex, second_simplex) {
                return (examined, Some(value));
            }
        }
    }
    (examined, None)
}

/// Exhaustive scoped pair enumeration for periodic charts.
///
/// The periodic path cannot use lifted-coordinate sweep-and-prune, but a local
/// mutation only needs changed-vs-all pairs. This keeps automatic insertion
/// validation proportional to the changed scope instead of all old pairs.
fn scoped_exhaustive_candidate_simplex_pairs<const D: usize, B>(
    simplices: &[EmbeddedSimplex<D>],
    skip: &FastHashSet<SimplexKey>,
    scope: &FastHashSet<SimplexKey>,
    mut on_pair: impl FnMut(&EmbeddedSimplex<D>, &EmbeddedSimplex<D>) -> ControlFlow<B>,
) -> (usize, Option<B>) {
    let mut examined = 0_usize;
    for (local_index, local_simplex) in simplices.iter().enumerate() {
        if !scope.contains(&local_simplex.key) || skip.contains(&local_simplex.key) {
            continue;
        }

        for (other_index, other_simplex) in simplices.iter().enumerate() {
            if other_index == local_index || skip.contains(&other_simplex.key) {
                continue;
            }
            if scope.contains(&other_simplex.key) && other_index < local_index {
                continue;
            }

            examined += 1;
            let first_index = local_index.min(other_index);
            let second_index = local_index.max(other_index);
            if let ControlFlow::Break(value) =
                on_pair(&simplices[first_index], &simplices[second_index])
            {
                return (examined, Some(value));
            }
        }
    }
    (examined, None)
}

/// Sweep-and-prune broad phase over Euclidean simplex bounding boxes.
///
/// See [`for_each_candidate_simplex_pair`] for the completeness argument and
/// references.
fn sweep_and_prune_candidate_simplex_pairs<const D: usize, B>(
    simplices: &[EmbeddedSimplex<D>],
    skip: &FastHashSet<SimplexKey>,
    mut on_pair: impl FnMut(&EmbeddedSimplex<D>, &EmbeddedSimplex<D>) -> ControlFlow<B>,
) -> (usize, Option<B>) {
    let mut boxes: Vec<SimplexBoundingBox<D>> = simplices
        .iter()
        .enumerate()
        .filter(|(_, simplex)| !skip.contains(&simplex.key))
        .map(|(index, simplex)| SimplexBoundingBox::from_embedded(index, simplex))
        .collect();
    if boxes.len() < 2 {
        return (0, None);
    }

    let sweep_axis = widest_extent_axis(&boxes);
    boxes.sort_unstable_by(|left, right| left.min[sweep_axis].total_cmp(&right.min[sweep_axis]));

    let mut active: Vec<usize> = Vec::new();
    let mut examined = 0_usize;
    for current in 0..boxes.len() {
        let current_min = boxes[current].min[sweep_axis];
        // Retire boxes that end before the current box begins on the sweep
        // axis; they cannot overlap the current box or any later one.
        active.retain(|&candidate| boxes[candidate].max[sweep_axis] >= current_min);
        for &candidate in &active {
            if !boxes[candidate].overlaps(&boxes[current]) {
                continue;
            }
            examined += 1;
            let first_index = boxes[candidate]
                .simplex_index
                .min(boxes[current].simplex_index);
            let second_index = boxes[candidate]
                .simplex_index
                .max(boxes[current].simplex_index);
            if let ControlFlow::Break(value) =
                on_pair(&simplices[first_index], &simplices[second_index])
            {
                return (examined, Some(value));
            }
        }
        active.push(current);
    }
    (examined, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::DelaunayTriangulationBuilder;
    use crate::core::tds::{Tds, TriangulationConstructionState};
    use crate::core::triangulation::Triangulation;
    use crate::core::vertex::Vertex;
    use crate::delaunay_property_validation::DelaunayValidationError;
    use crate::geometry::kernel::FastKernel;
    use crate::topology::traits::topological_space::{GlobalTopology, ToroidalConstructionMode};
    use crate::validation::{DelaunayTriangulationValidationError, DelaunayVerificationError};
    use crate::vertex;
    use approx::assert_abs_diff_eq;
    use std::assert_matches;

    fn test_vertex<const D: usize>(coords: [f64; D]) -> Vertex<(), D> {
        vertex!(coords).unwrap()
    }

    fn tds_from_vertices_and_simplices<const D: usize>(
        coords: &[[f64; D]],
        simplices: &[Vec<usize>],
    ) -> Tds<(), (), D> {
        tds_from_vertices_and_simplices_with_keys(coords, simplices).0
    }

    fn tds_from_vertices_and_simplices_with_keys<const D: usize>(
        coords: &[[f64; D]],
        simplices: &[Vec<usize>],
    ) -> (Tds<(), (), D>, Vec<SimplexKey>) {
        let mut tds = Tds::empty();
        let vertex_keys: Vec<_> = coords
            .iter()
            .map(|coords| {
                tds.insert_vertex_with_mapping(test_vertex(*coords))
                    .unwrap()
            })
            .collect();

        let mut simplex_keys = Vec::with_capacity(simplices.len());
        for simplex_vertices in simplices {
            let vertices: Vec<_> = simplex_vertices
                .iter()
                .map(|&index| vertex_keys[index])
                .collect();
            let simplex_key = tds
                .insert_simplex_with_mapping(Simplex::try_new_with_data(vertices, None).unwrap())
                .unwrap();
            simplex_keys.push(simplex_key);
        }

        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();
        (tds, simplex_keys)
    }

    fn tri_from_tds<const D: usize>(
        tds: Tds<(), (), D>,
    ) -> Triangulation<FastKernel<f64>, (), (), D> {
        Triangulation::new_with_tds(FastKernel::new(), tds)
    }

    fn tri_from_tds_with_topology<const D: usize>(
        tds: Tds<(), (), D>,
        global_topology: GlobalTopology<D>,
    ) -> Triangulation<FastKernel<f64>, (), (), D> {
        let mut tri = tri_from_tds(tds);
        tri.global_topology = global_topology;
        tri
    }

    fn assert_single_simplex_embeds<const D: usize>() {
        let mut coords = Vec::with_capacity(D + 1);
        coords.push([0.0; D]);
        for axis in 0..D {
            let mut point = [0.0; D];
            point[axis] = 1.0;
            coords.push(point);
        }
        let simplex = (0..=D).collect();
        let tri = tri_from_tds(tds_from_vertices_and_simplices(&coords, &[simplex]));
        assert!(tri.is_valid_embedding().is_ok());
    }

    #[test]
    fn is_valid_embedding_accepts_single_simplex_dimensions_two_through_five() {
        assert_single_simplex_embeds::<2>();
        assert_single_simplex_embeds::<3>();
        assert_single_simplex_embeds::<4>();
        assert_single_simplex_embeds::<5>();
    }

    #[test]
    fn validate_embedding_accepts_builder_constructed_triangulation() {
        let vertices = vec![
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
            test_vertex([0.25, 0.25, 0.25]),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build::<()>()
            .unwrap();

        assert!(dt.as_triangulation().validate_embedding().is_ok());
    }

    #[test]
    fn is_valid_embedding_accepts_two_tetrahedra_sharing_a_facet() {
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let tds = tds_from_vertices_and_simplices(&coords, &[vec![0, 1, 2, 3], vec![0, 2, 1, 4]]);
        let tri = tri_from_tds(tds);

        assert!(tri.is_valid_embedding().is_ok());
    }

    #[test]
    fn is_valid_embedding_rejects_full_facet_same_side_overlap() {
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.5],
        ];
        let tds = tds_from_vertices_and_simplices(&coords, &[vec![0, 1, 2, 3], vec![0, 2, 1, 4]]);
        let tri = tri_from_tds(tds);

        let err = tri.is_valid_embedding().unwrap_err();

        assert_matches!(
            err,
            TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace {
                detail,
                ..
            } if detail.shared_vertices.len() == 3
                && detail.shared_vertex_uuids.len() == 3
                && detail.first_only_witness_vertices.len() == 1
                && detail.first_only_witness_vertex_uuids.len() == 1
                && detail.second_only_witness_vertices.len() == 1
                && detail.second_only_witness_vertex_uuids.len() == 1
        );
    }

    #[test]
    fn validate_embedding_rejects_degenerate_simplex() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let tds = tds_from_vertices_and_simplices(&coords, &[vec![0, 1, 2]]);
        let tri = tri_from_tds(tds);

        let diagnostic = tri
            .embedding_diagnostic()
            .unwrap()
            .expect("degenerate simplex should produce a diagnostic");
        let report_first = tri
            .embedding_report()
            .unwrap()
            .violations
            .into_iter()
            .next()
            .expect("degenerate simplex should be the first report violation");
        assert_eq!(diagnostic, report_first);

        let err = tri.is_valid_embedding().unwrap_err();
        assert_eq!(err, diagnostic);
        assert_matches!(
            err,
            TriangulationEmbeddingValidationError::DegenerateSimplex { dimension: 2, .. }
        );
    }

    #[test]
    fn is_valid_embedding_preserves_duplicate_label_detail() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let (mut tds, simplex_keys) =
            tds_from_vertices_and_simplices_with_keys(&coords, &[vec![0, 1, 2]]);
        let simplex_key = simplex_keys[0];
        let (duplicate_key, middle_key, duplicate_uuid) = {
            let simplex = tds
                .simplex(simplex_key)
                .expect("fixture simplex should exist");
            let duplicate_key = simplex.vertices()[0];
            let middle_key = simplex.vertices()[1];
            let duplicate_uuid = tds
                .vertex(duplicate_key)
                .expect("duplicate fixture vertex should exist")
                .uuid();
            (duplicate_key, middle_key, duplicate_uuid)
        };
        {
            let simplex = tds
                .simplex_mut(simplex_key)
                .expect("fixture simplex should be mutable");
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(duplicate_key);
            simplex.push_vertex_key(middle_key);
            simplex.push_vertex_key(duplicate_key);
        }
        let tri = tri_from_tds(tds);

        let err = tri.is_valid_embedding().unwrap_err();

        assert_matches!(
            err,
            TriangulationEmbeddingValidationError::DuplicateSimplexEmbeddingLabel {
                simplex_key: observed_simplex_key,
                vertex_key,
                vertex_uuid,
                first_index: 0,
                duplicate_index: 2,
                detail,
                ..
            } if observed_simplex_key == simplex_key
                && vertex_key == duplicate_key
                && vertex_uuid == duplicate_uuid
                && detail.vertices.len() == 3
                && detail.vertex_uuids.len() == 3
        );
    }

    #[test]
    fn embedding_report_includes_degenerate_simplex_vertices() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let tds = tds_from_vertices_and_simplices(&coords, &[vec![0, 1, 2]]);
        let tri = tri_from_tds(tds);

        let report = tri
            .embedding_report()
            .expect("embedding report should be generated");
        assert!(!report.is_valid());
        assert_eq!(report.checked_simplices, 1);
        assert_matches!(
            &report.violations[..],
            [TriangulationEmbeddingValidationError::DegenerateSimplex {
                detail,
                dimension: 2,
                ..
            }] if detail.vertices.len() == 3 && detail.vertex_uuids.len() == 3
        );
    }

    #[test]
    fn is_valid_embedding_rejects_nonadjacent_edge_crossing() {
        let coords = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, -1.0]];
        let tds = tds_from_vertices_and_simplices(
            &coords,
            &[vec![0, 1, 2], vec![2, 1, 3], vec![3, 2, 4]],
        );
        let tri = tri_from_tds(tds);

        let err = tri.is_valid_embedding().unwrap_err();
        assert_matches!(
            err,
            TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace { .. }
        );
    }

    #[test]
    fn is_valid_embedding_sweep_and_prune_detects_interposed_overlap() {
        // Regression guard for the sweep-and-prune broad phase: triangles A and
        // B genuinely overlap (no shared vertices), but triangle C sits between
        // them in the sweep-axis ordering while overlapping neither. A naive
        // "compare only neighbors in sorted order" prune would drop the A/B
        // pair; sweep-and-prune keeps A active across C and still reports the
        // overlap, so the broad phase must not introduce a false negative.
        let coords = [
            [0.0, 0.0],  // 0  A
            [10.0, 0.0], // 1  A
            [0.0, 2.0],  // 2  A
            [3.0, -1.0], // 3  C (x between A and B, disjoint in y)
            [4.0, -1.0], // 4  C
            [3.5, -0.5], // 5  C
            [4.5, -1.0], // 6  B (overlaps A)
            [5.5, -1.0], // 7  B
            [4.5, 2.0],  // 8  B
        ];
        let tds = tds_from_vertices_and_simplices(
            &coords,
            &[vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]],
        );
        let tri = tri_from_tds(tds);

        let err = tri.is_valid_embedding().unwrap_err();
        assert_matches!(
            err,
            TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace { .. }
        );
    }

    #[test]
    fn embedding_report_includes_intersection_witness_vertices() {
        let coords = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, -1.0]];
        let tds = tds_from_vertices_and_simplices(
            &coords,
            &[vec![0, 1, 2], vec![2, 1, 3], vec![3, 2, 4]],
        );
        let tri = tri_from_tds(tds);

        let report = tri
            .embedding_report()
            .expect("embedding report should be generated");
        let intersection =
            report
                .violations
                .iter()
                .find(|violation| {
                    matches!(
                    violation,
                    TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace {
                        ..
                    }
                )
                })
                .expect("report should include an illegal simplex intersection");

        assert_matches!(
            intersection,
            TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace {
                detail,
                ..
            } if detail.first_simplex.vertices.len() == 3
                && detail.first_simplex.vertex_uuids.len() == 3
                && detail.second_simplex.vertices.len() == 3
                && detail.second_simplex.vertex_uuids.len() == 3
                && !detail.first_only_witness_vertices.is_empty()
                && detail.first_only_witness_vertices.len()
                    == detail.first_only_witness_vertex_uuids.len()
                && !detail.second_only_witness_vertices.is_empty()
                && detail.second_only_witness_vertices.len()
                    == detail.second_only_witness_vertex_uuids.len()
        );
    }

    #[test]
    fn is_valid_embedding_accepts_lifted_toroidal_simplex_chart() {
        let coords = [[0.9, 0.1], [0.1, 0.1], [0.9, 0.3]];
        let (mut tds, simplex_keys) =
            tds_from_vertices_and_simplices_with_keys(&coords, &[vec![0, 1, 2]]);
        tds.simplex_mut(simplex_keys[0])
            .unwrap()
            .set_periodic_vertex_offsets(vec![[0, 0], [1, 0], [0, 0]])
            .unwrap();
        let tri = tri_from_tds_with_topology(
            tds,
            GlobalTopology::try_toroidal([1.0, 1.0], ToroidalConstructionMode::PeriodicImagePoint)
                .unwrap(),
        );

        assert!(tri.is_valid_embedding().is_ok());
    }

    #[test]
    fn is_valid_embedding_rejects_unsupported_spherical_topology() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let tds = tds_from_vertices_and_simplices(&coords, &[vec![0, 1, 2]]);
        let tri = tri_from_tds_with_topology(tds, GlobalTopology::Spherical);

        let err = tri.is_valid_embedding().unwrap_err();
        assert_matches!(
            err,
            TriangulationEmbeddingValidationError::UnsupportedTopology {
                topology: TopologyKind::Spherical,
                dimension: 2,
            }
        );
    }

    #[test]
    fn is_valid_embedding_rejects_periodic_simplex_spanning_domain() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.25]];
        let tds = tds_from_vertices_and_simplices(&coords, &[vec![0, 1, 2]]);
        let tri = tri_from_tds_with_topology(
            tds,
            GlobalTopology::try_toroidal([1.0, 1.0], ToroidalConstructionMode::PeriodicImagePoint)
                .unwrap(),
        );

        let err = tri.is_valid_embedding().unwrap_err();
        let (span, period) = match err {
            TriangulationEmbeddingValidationError::PeriodicSimplexSpansDomain {
                axis: 0,
                span,
                period,
                ..
            } => (span, period),
            other => panic!("expected periodic simplex span violation, got {other:?}"),
        };
        assert_abs_diff_eq!(span, 1.0, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(period, 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn is_valid_embedding_rejects_periodic_translate_overlap() {
        let coords = [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.0, 0.8],
            [0.95, 0.1],
            [0.15, 0.1],
            [0.95, 0.3],
        ];
        let (mut tds, simplex_keys) =
            tds_from_vertices_and_simplices_with_keys(&coords, &[vec![0, 1, 2], vec![3, 4, 5]]);
        tds.simplex_mut(simplex_keys[1])
            .unwrap()
            .set_periodic_vertex_offsets(vec![[0, 0], [1, 0], [0, 0]])
            .unwrap();
        let tri = tri_from_tds_with_topology(
            tds,
            GlobalTopology::try_toroidal([1.0, 1.0], ToroidalConstructionMode::PeriodicImagePoint)
                .unwrap(),
        );

        let err = tri.is_valid_embedding().unwrap_err();
        assert_matches!(
            err,
            TriangulationEmbeddingValidationError::SimplexIntersectionOutsideSharedFace { .. }
        );
    }

    #[test]
    fn embedding_error_kind_covers_variants() {
        let source = TriangulationEmbeddingValidationError::DegenerateSimplex {
            simplex_key: SimplexKey::default(),
            simplex_uuid: Uuid::nil(),
            detail: Box::new(TriangulationEmbeddingSimplexDetail {
                key: SimplexKey::default(),
                uuid: Uuid::nil(),
                vertices: SimplexVertexKeyBuffer::new(),
                vertex_uuids: SimplexVertexUuidBuffer::new(),
            }),
            dimension: 2,
        };

        assert_eq!(
            TriangulationEmbeddingValidationErrorKind::from(&source),
            TriangulationEmbeddingValidationErrorKind::DegenerateSimplex,
        );

        let duplicate_label_source =
            TriangulationEmbeddingValidationError::DuplicateSimplexEmbeddingLabel {
                simplex_key: SimplexKey::default(),
                simplex_uuid: Uuid::nil(),
                detail: Box::new(TriangulationEmbeddingSimplexDetail {
                    key: SimplexKey::default(),
                    uuid: Uuid::nil(),
                    vertices: SimplexVertexKeyBuffer::new(),
                    vertex_uuids: SimplexVertexUuidBuffer::new(),
                }),
                vertex_key: VertexKey::default(),
                vertex_uuid: Uuid::nil(),
                first_index: 0,
                duplicate_index: 2,
            };

        assert_eq!(
            TriangulationEmbeddingValidationErrorKind::from(&duplicate_label_source),
            TriangulationEmbeddingValidationErrorKind::DuplicateSimplexEmbeddingLabel,
        );

        let invalid_period_source =
            TriangulationEmbeddingValidationError::InvalidPeriodicDomainPeriod {
                simplex_key: SimplexKey::default(),
                simplex_uuid: Uuid::nil(),
                detail: Box::new(TriangulationEmbeddingSimplexDetail {
                    key: SimplexKey::default(),
                    uuid: Uuid::nil(),
                    vertices: SimplexVertexKeyBuffer::new(),
                    vertex_uuids: SimplexVertexUuidBuffer::new(),
                }),
                source: PeriodicDomainPeriodError::NonPositivePeriod {
                    axis: 0,
                    period: 0.0,
                },
            };

        assert_eq!(
            TriangulationEmbeddingValidationErrorKind::from(&invalid_period_source),
            TriangulationEmbeddingValidationErrorKind::InvalidPeriodicDomainPeriod,
        );

        let unexpected_source = TriangulationEmbeddingValidationError::UnexpectedValidationLayer {
            kind: InvariantKind::DelaunayProperty,
            source: Box::new(InvariantError::Delaunay(
                DelaunayTriangulationValidationError::VerificationFailed {
                    source: Box::new(DelaunayVerificationError::from(
                        DelaunayValidationError::TriangulationState {
                            source: TdsError::InconsistentDataStructure {
                                message: "synthetic higher-layer failure".to_string(),
                            },
                        },
                    )),
                },
            )),
        };

        assert_eq!(
            TriangulationEmbeddingValidationErrorKind::from(&unexpected_source),
            TriangulationEmbeddingValidationErrorKind::UnexpectedValidationLayer,
        );
    }
}

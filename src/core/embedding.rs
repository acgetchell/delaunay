//! Embedded-geometry validation for generic triangulations.
//!
//! This module owns Level 4 validation for generic [`Triangulation`](crate::Triangulation):
//! after the TDS and topology layers have certified a valid oriented simplicial
//! complex, the embedding layer verifies that maximal simplices are nondegenerate
//! and intersect only in their shared faces in the topology's active affine chart.

#![forbid(unsafe_code)]

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
    /// Number of simplex pairs considered for overlap validation.
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
        let vertex_key = self.embedding.labels()[vertex_index];
        let vertex_uuid = self.vertex_uuids[vertex_index];
        Point::try_new(self.embedding.coordinates()[vertex_index]).map_err(|source| {
            TriangulationEmbeddingValidationError::CoordinateValidation {
                simplex_key: self.key,
                simplex_uuid: self.uuid,
                vertex_key,
                vertex_uuid,
                source,
            }
        })
    }

    /// Maps witness vertex keys back to UUIDs from this simplex snapshot.
    fn vertex_uuids_for_keys(&self, vertex_keys: &[VertexKey]) -> SimplexVertexUuidBuffer {
        let mut uuids = SimplexVertexUuidBuffer::with_capacity(vertex_keys.len());
        uuids.extend(vertex_keys.iter().filter_map(|vertex_key| {
            self.vertex_keys
                .iter()
                .position(|candidate| candidate == vertex_key)
                .map(|index| self.vertex_uuids[index])
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

        for (first_index, first) in simplices.iter().enumerate() {
            for second in &simplices[first_index + 1..] {
                if invalid_simplex_keys.contains(&first.key)
                    || invalid_simplex_keys.contains(&second.key)
                {
                    continue;
                }
                report.checked_simplex_pairs += 1;
                if let Err(error) =
                    validate_topology_aware_simplex_pair(first, second, periodic_periods)
                {
                    report.violations.push(error);
                }
            }
        }

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

        for (first_index, first) in simplices.iter().enumerate() {
            for second in &simplices[first_index + 1..] {
                if let Err(error) =
                    validate_topology_aware_simplex_pair(first, second, periodic_periods)
                {
                    return Ok(Some(error));
                }
            }
        }

        Ok(None)
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
            validate_simplex_pair_intersection(first, second)?;
        }
        return Ok(());
    };

    let shift_ranges = periodic_shift_ranges(first, second, &periods)?;
    let mut shift = [0_i32; D];
    validate_periodic_translates(first, second, &periods, &shift_ranges, 0, &mut shift)
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
        Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace(witness)) => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::DelaunayTriangulationBuilder;
    use crate::core::tds::{Tds, TriangulationConstructionState};
    use crate::core::triangulation::Triangulation;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::FastKernel;
    use crate::topology::traits::topological_space::{GlobalTopology, ToroidalConstructionMode};
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

        let unexpected_source =
            TriangulationEmbeddingValidationError::UnexpectedValidationLayer {
                kind: InvariantKind::DelaunayProperty,
                source: Box::new(InvariantError::Delaunay(
                    crate::validation::DelaunayTriangulationValidationError::VerificationFailed {
                        source: Box::new(crate::validation::DelaunayVerificationError::from(
                            crate::delaunay_property_validation::DelaunayValidationError::TriangulationState {
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

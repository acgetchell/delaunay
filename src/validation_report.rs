//! Whole-stack validation diagnostics and promotion of layer-local reports.

#![forbid(unsafe_code)]

use crate::core::tds::{TdsError, TdsInvariantKind, TdsInvariantViolation, TdsValidationReport};
use crate::triangulation::realization::TriangulationRealizationValidationError;
use crate::triangulation::validation::TriangulationValidationError;
use crate::validation::DelaunayTriangulationValidationError;
use thiserror::Error;

/// Classifies the kind of triangulation invariant that failed during validation.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::validation::InvariantKind;
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
/// use delaunay::prelude::tds::TdsError;
/// use delaunay::prelude::validation::InvariantError;
///
/// let err = InvariantError::Tds {
///     source: TdsError::InconsistentDataStructure {
///         message: "bad neighbors".to_string(),
///     },
/// };
/// std::assert_matches!(err, InvariantError::Tds { .. });
/// ```
///
/// This is used by [`TriangulationValidationReport`] so that diagnostic reporting can
/// preserve structured errors from each layer (TDS / topology / realization / Delaunay) without
/// stringification.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum InvariantError {
    /// Level 1–2 (elements + TDS structure).
    #[error(transparent)]
    Tds {
        /// Typed TDS source error.
        #[from]
        source: TdsError,
    },

    /// Level 3 (topology).
    #[error(transparent)]
    Triangulation {
        /// Typed topology-validation source error.
        #[from]
        source: TriangulationValidationError,
    },

    /// Level 4 (realized geometry).
    #[error(transparent)]
    Realization {
        /// Typed realization-validation source error.
        #[from]
        source: TriangulationRealizationValidationError,
    },

    /// Level 5 (Delaunay property).
    #[error(transparent)]
    Delaunay {
        /// Typed Delaunay-validation source error.
        #[from]
        source: DelaunayTriangulationValidationError,
    },
}

/// A single invariant violation recorded during validation diagnostics.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::tds::TdsError;
/// use delaunay::prelude::validation::{InvariantError, InvariantKind, InvariantViolation};
///
/// let violation = InvariantViolation {
///     kind: InvariantKind::NeighborConsistency,
///     error: InvariantError::Tds {
///         source: TdsError::InconsistentDataStructure {
///             message: "bad neighbors".to_string(),
///         },
///     },
/// };
/// assert_eq!(violation.kind, InvariantKind::NeighborConsistency);
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
/// [`Triangulation::validation_report()`] and [`DelaunayTriangulation::validation_report()`]
/// to surface all failed invariants at once for debugging and test diagnostics.
///
/// [`DelaunayTriangulation::validation_report()`]: crate::DelaunayTriangulation::validation_report
/// [`Triangulation::validation_report()`]: crate::Triangulation::validation_report
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::validation::TriangulationValidationReport;
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

impl From<TdsInvariantKind> for InvariantKind {
    fn from(kind: TdsInvariantKind) -> Self {
        match kind {
            TdsInvariantKind::VertexValidity => Self::VertexValidity,
            TdsInvariantKind::SimplexValidity => Self::SimplexValidity,
            TdsInvariantKind::SimplexCoordinateUniqueness => Self::SimplexCoordinateUniqueness,
            TdsInvariantKind::VertexMappings => Self::VertexMappings,
            TdsInvariantKind::SimplexMappings => Self::SimplexMappings,
            TdsInvariantKind::SimplexVertexKeys => Self::SimplexVertexKeys,
            TdsInvariantKind::VertexIncidence => Self::VertexIncidence,
            TdsInvariantKind::VertexToSimplicesIndex => Self::VertexToSimplicesIndex,
            TdsInvariantKind::DuplicateSimplices => Self::DuplicateSimplices,
            TdsInvariantKind::FacetSharing => Self::FacetSharing,
            TdsInvariantKind::NeighborConsistency => Self::NeighborConsistency,
            TdsInvariantKind::CoherentOrientation => Self::CoherentOrientation,
        }
    }
}

impl From<TdsInvariantViolation> for InvariantViolation {
    fn from(violation: TdsInvariantViolation) -> Self {
        Self {
            kind: violation.kind.into(),
            error: violation.error.into(),
        }
    }
}

impl From<TdsValidationReport> for TriangulationValidationReport {
    fn from(report: TdsValidationReport) -> Self {
        Self {
            violations: report.violations.into_iter().map(Into::into).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::characteristics::euler::TopologyClassification;
    use crate::topology::traits::topological_space::TopologyKind;

    #[test]
    fn tds_report_promotion_preserves_order_categories_and_typed_sources() {
        let cases = [
            (
                TdsInvariantKind::VertexValidity,
                InvariantKind::VertexValidity,
            ),
            (
                TdsInvariantKind::SimplexValidity,
                InvariantKind::SimplexValidity,
            ),
            (
                TdsInvariantKind::SimplexCoordinateUniqueness,
                InvariantKind::SimplexCoordinateUniqueness,
            ),
            (
                TdsInvariantKind::VertexMappings,
                InvariantKind::VertexMappings,
            ),
            (
                TdsInvariantKind::SimplexMappings,
                InvariantKind::SimplexMappings,
            ),
            (
                TdsInvariantKind::SimplexVertexKeys,
                InvariantKind::SimplexVertexKeys,
            ),
            (
                TdsInvariantKind::VertexIncidence,
                InvariantKind::VertexIncidence,
            ),
            (
                TdsInvariantKind::VertexToSimplicesIndex,
                InvariantKind::VertexToSimplicesIndex,
            ),
            (
                TdsInvariantKind::DuplicateSimplices,
                InvariantKind::DuplicateSimplices,
            ),
            (TdsInvariantKind::FacetSharing, InvariantKind::FacetSharing),
            (
                TdsInvariantKind::NeighborConsistency,
                InvariantKind::NeighborConsistency,
            ),
            (
                TdsInvariantKind::CoherentOrientation,
                InvariantKind::CoherentOrientation,
            ),
        ];
        let original = TdsValidationReport {
            violations: cases
                .iter()
                .enumerate()
                .map(|(index, &(kind, _))| TdsInvariantViolation {
                    kind,
                    error: TdsError::IndexOutOfBounds {
                        index,
                        bound: 0,
                        context: "report promotion".to_string(),
                    },
                })
                .collect(),
        };
        let promoted = TriangulationValidationReport::from(original.clone());
        assert_eq!(promoted.violations.len(), original.violations.len());
        for ((violation, original), (_, kind)) in promoted
            .violations
            .into_iter()
            .zip(original.violations)
            .zip(cases)
        {
            assert_eq!(violation.kind, kind);
            assert_eq!(
                violation.error,
                InvariantError::Tds {
                    source: original.error
                }
            );
        }
        assert!(
            TriangulationValidationReport::from(TdsValidationReport {
                violations: Vec::new(),
            })
            .is_empty()
        );
    }

    #[test]
    fn higher_layer_promotions_preserve_typed_sources() {
        let topology = TriangulationValidationError::EulerCharacteristicMismatch {
            computed: 1,
            expected: 2,
            classification: TopologyClassification::Ball(3),
        };
        assert_eq!(
            InvariantError::from(topology.clone()),
            InvariantError::Triangulation { source: topology },
        );
        let realization = TriangulationRealizationValidationError::UnsupportedTopology {
            topology: TopologyKind::Hyperbolic,
            dimension: 2,
        };
        assert_eq!(
            InvariantError::from(realization.clone()),
            InvariantError::Realization {
                source: realization
            },
        );
        let delaunay = DelaunayTriangulationValidationError::Tds {
            source: Box::new(TdsError::InconsistentDataStructure {
                message: "typed source".to_string(),
            }),
        };
        assert_eq!(
            InvariantError::from(delaunay.clone()),
            InvariantError::Delaunay { source: delaunay },
        );
    }
}

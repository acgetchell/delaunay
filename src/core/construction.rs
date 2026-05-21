//! Generic triangulation construction helpers.
//!
//! This module owns the generic construction vocabulary for
//! [`Triangulation`](crate::core::triangulation::Triangulation)
//! and the initial-simplex bootstrap used before incremental insertion takes
//! over. Mutation-heavy insertion and repair orchestration remain implemented
//! with the triangulation type until they can be split into narrower modules.

use crate::core::algorithms::incremental_insertion::{CavityFillingError, HullExtensionReason};
use crate::core::algorithms::locate::{ConflictError, LocateError};
use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::core::simplex::{Simplex, SimplexValidationError};
use crate::core::tds::{InvariantErrorSummary, Tds, TdsConstructionError, TdsError, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::validation::TriangulationValidationError;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::predicates::Orientation;
use crate::geometry::robust_predicates::robust_orientation;
use crate::validation::DelaunayTriangulationValidationError;
use num_traits::NumCast;
use thiserror::Error;

/// Errors that can occur during triangulation construction.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::triangulation::{
///     FastKernel, Triangulation, TriangulationConstructionError, vertex,
/// };
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

    /// Failed to create a simplex during triangulation construction.
    #[error("Failed to create simplex during construction: {message}")]
    FailedToCreateSimplex {
        /// Description of the simplex creation failure.
        message: String,
    },

    /// Cavity filling failed during incremental construction.
    #[error("Cavity filling failed during insertion: {source}")]
    InsertionCavityFilling {
        /// Underlying cavity-filling error.
        #[source]
        source: CavityFillingError,
    },

    /// Insufficient vertices to create a triangulation.
    #[error("Insufficient vertices for {dimension}D triangulation: {source}")]
    InsufficientVertices {
        /// The dimension that was attempted.
        dimension: usize,
        /// The underlying simplex validation error.
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
        reason: HullExtensionReason,
    },

    /// Level 4 Delaunay validation failed during incremental construction.
    #[error("Delaunay validation failed during insertion: {source}")]
    InsertionDelaunayValidation {
        /// Underlying Delaunay validation error.
        #[source]
        source: DelaunayTriangulationValidationError,
    },

    /// Level 3 topology validation failed during incremental construction.
    #[error("{message}: {source}")]
    InsertionTopologyValidation {
        /// High-level insertion context.
        message: String,
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
    #[error("{message}: {source}")]
    FinalTopologyValidation {
        /// High-level finalization context.
        message: String,
        /// Underlying validation error.
        #[source]
        source: InvariantErrorSummary,
    },

    /// Attempted to insert a vertex with coordinates that already exist.
    #[error(
        "Duplicate coordinates: vertex with coordinates {coordinates} already exists in the triangulation"
    )]
    DuplicateCoordinates {
        /// String representation of the duplicate coordinates.
        coordinates: String,
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
    K: Kernel<D>,
    K::Scalar: NumCast,
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
    /// use delaunay::prelude::triangulation::{FastKernel, Triangulation, vertex};
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices)?;
    /// assert_eq!(tds.number_of_vertices(), 3);
    /// assert_eq!(tds.number_of_simplices(), 1);
    /// assert_eq!(tds.dim(), 2);
    /// # Ok::<(), delaunay::prelude::triangulation::TriangulationConstructionError>(())
    /// ```
    pub fn build_initial_simplex(
        vertices: &[Vertex<K::Scalar, U, D>],
    ) -> Result<Tds<K::Scalar, U, V, D>, TriangulationConstructionError> {
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

        let points: SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
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

        let simplex = Simplex::new(vertex_keys, None).map_err(|e| {
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
    use crate::core::vertex::VertexBuilder;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;
    use uuid::Uuid;

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

    macro_rules! test_build_initial_simplex {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<build_initial_simplex_ $dim d>]() {
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
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
        let vertices = vec![vertex!([0.0, 0.0, 0.0]), vertex!([1.0, 0.0, 0.0])];

        let result = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert!(matches!(
            result,
            Err(TriangulationConstructionError::InsufficientVertices { dimension: 3, .. })
        ));
    }

    #[test]
    fn build_initial_simplex_too_many_vertices() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert!(matches!(
            result,
            Err(TriangulationConstructionError::InsufficientVertices { .. })
        ));
    }

    fn invalid_initial_simplex_vertices<const D: usize>() -> Vec<Vertex<f64, (), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(vertex!([0.0_f64; D]));

        let mut invalid_coords = [0.0_f64; D];
        invalid_coords[0] = 1.0;
        invalid_coords[1] = f64::NAN;
        vertices.push(Vertex::new_with_uuid(
            Point::new(invalid_coords),
            Uuid::new_v4(),
            None,
        ));

        for axis in 1..D {
            let mut coords = [0.0_f64; D];
            coords[axis] = 1.0;
            vertices.push(vertex!(coords));
        }

        vertices
    }

    macro_rules! test_build_initial_simplex_rejects_invalid_vertex_dimensions {
        ($($dim:expr),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<build_initial_simplex_rejects_invalid_vertex_ $dim d>]() {
                        let vertices = invalid_initial_simplex_vertices::<$dim>();

                        let result = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices);

                        assert!(matches!(
                            result,
                            Err(TriangulationConstructionError::Tds(
                                TdsConstructionError::ValidationError(TdsError::InvalidVertex { .. })
                            ))
                        ));
                    }
                )+
            }
        };
    }

    test_build_initial_simplex_rejects_invalid_vertex_dimensions!(2, 3, 4, 5);

    #[test]
    fn build_initial_simplex_with_user_data() {
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
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, 0.0]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert!(matches!(
            result,
            Err(TriangulationConstructionError::GeometricDegeneracy { .. })
        ));
    }

    #[test]
    fn build_initial_simplex_rejects_coplanar_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.5, 0.5, 0.0]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert!(matches!(
            result,
            Err(TriangulationConstructionError::GeometricDegeneracy { .. })
        ));
    }
}

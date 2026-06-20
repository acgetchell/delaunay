//! Benchmark-only fixture builders for topology states public constructors reject.
//!
//! This module is compiled only with the crate's `bench` feature. Its helpers
//! let Criterion harnesses measure repair paths that require deliberately
//! invalid but structurally coherent TDS states, without exposing those mutation
//! capabilities in the normal library API.

#![forbid(unsafe_code)]

/// Fixtures for PL-manifold repair benchmarks.
pub mod pl_manifold {
    use crate::core::algorithms::pl_manifold_repair::{
        PlManifoldRepairConfig, PlManifoldRepairError, PlManifoldRepairStats,
        repair_facet_oversharing,
    };
    use crate::core::collections::SimplexVertexKeyBuffer;
    use crate::core::simplex::{Simplex, SimplexValidationError};
    use crate::core::tds::{Tds, TdsConstructionError, TdsError, VertexKey};
    use crate::core::vertex::Vertex;
    use crate::geometry::traits::coordinate::CoordinateConversionError;
    use crate::geometry::util::safe_usize_to_scalar;
    use crate::topology::manifold::validate_facet_degree;
    use thiserror::Error;

    /// Fixture for benchmarking facet over-sharing repair plus orphan cleanup in 3D.
    #[derive(Clone, Debug)]
    #[must_use]
    pub struct OversharedFacetOrphanCleanupFixture3d {
        /// Structurally coherent TDS with one over-shared facet per cluster.
        pub tds: Tds<(), (), 3>,
        /// Number of independent over-shared clusters in the fixture.
        pub cluster_count: usize,
    }

    /// Errors returned while building PL-manifold repair benchmark fixtures.
    #[derive(Clone, Debug, Error, PartialEq)]
    #[non_exhaustive]
    pub enum PlManifoldRepairFixtureError {
        /// The requested fixture contains no repair work.
        #[error("PL-manifold repair fixture requires at least one cluster")]
        EmptyClusterCount,

        /// The cluster index could not be converted to a finite coordinate offset.
        #[error("failed to convert cluster index {cluster_index} to a coordinate offset: {source}")]
        ClusterIndexConversion {
            /// Cluster index being converted.
            cluster_index: usize,
            /// Underlying coordinate conversion error.
            source: CoordinateConversionError,
        },

        /// A fixture vertex could not be constructed.
        #[error("failed to create vertex for cluster {cluster_index}: {source}")]
        Vertex {
            /// Cluster that was being inserted.
            cluster_index: usize,
            /// Underlying coordinate conversion error.
            source: CoordinateConversionError,
        },

        /// A fixture vertex could not be inserted into the TDS.
        #[error("failed to insert vertex for cluster {cluster_index}: {source}")]
        VertexInsert {
            /// Cluster that was being inserted.
            cluster_index: usize,
            /// Underlying typed TDS construction error.
            source: Box<TdsConstructionError>,
        },

        /// A fixture simplex could not be constructed.
        #[error("failed to create simplex for cluster {cluster_index}: {source}")]
        Simplex {
            /// Cluster that was being inserted.
            cluster_index: usize,
            /// Underlying simplex validation error.
            source: SimplexValidationError,
        },

        /// A fixture simplex could not be inserted into the TDS.
        #[error("failed to insert simplex for cluster {cluster_index}: {source}")]
        SimplexInsert {
            /// Cluster that was being inserted.
            cluster_index: usize,
            /// Underlying typed TDS construction error.
            source: Box<TdsConstructionError>,
        },

        /// Structural TDS validation failed for the deliberately invalid fixture.
        #[error("PL-manifold repair fixture has invalid structural TDS state: {source}")]
        StructuralValidation {
            /// Underlying TDS validation error.
            source: TdsError,
        },

        /// The fixture did not produce the intended facet-degree violation.
        #[error(
            "PL-manifold repair fixture with {cluster_count} clusters did not over-share facets"
        )]
        MissingOversharedFacet {
            /// Number of clusters requested.
            cluster_count: usize,
        },

        /// Repair validation of the fixture failed.
        #[error("PL-manifold repair fixture validation failed: {source}")]
        Repair {
            /// Underlying repair error.
            source: PlManifoldRepairError,
        },

        /// Repair succeeded but did not remove the expected topology.
        #[error(
            "PL-manifold repair fixture removed {actual_simplices} simplices and {actual_vertices} vertices; expected {expected_simplices} simplices and {expected_vertices} vertices"
        )]
        UnexpectedRepairStats {
            /// Expected number of removed simplices.
            expected_simplices: usize,
            /// Actual number of removed simplices.
            actual_simplices: usize,
            /// Expected number of removed orphan vertices.
            expected_vertices: usize,
            /// Actual number of removed orphan vertices.
            actual_vertices: usize,
        },
    }

    /// Builds a 3D fixture with one over-shared facet and one repair-created
    /// orphan vertex per cluster.
    ///
    /// Each cluster contains two valid tetrahedra sharing one triangular facet
    /// plus a deliberately skinny third tetrahedron sharing that same facet. The
    /// skinny simplex owns a unique apex, so removing it creates an isolated
    /// vertex for the repair pass to clean up.
    ///
    /// # Errors
    ///
    /// Returns [`PlManifoldRepairFixtureError`] if the requested cluster count
    /// is empty, coordinate conversion fails, TDS construction fails, or the
    /// resulting fixture does not contain facet over-sharing.
    fn overshared_facet_orphan_cleanup_3d(
        cluster_count: usize,
    ) -> Result<OversharedFacetOrphanCleanupFixture3d, PlManifoldRepairFixtureError> {
        if cluster_count == 0 {
            return Err(PlManifoldRepairFixtureError::EmptyClusterCount);
        }

        let mut tds = Tds::empty();
        for cluster_index in 0..cluster_count {
            insert_overshared_cluster(&mut tds, cluster_index)?;
        }

        validate_structural_fixture_state(&tds)?;
        let facet_map = tds
            .build_facet_to_simplices_map()
            .map_err(|source| PlManifoldRepairFixtureError::StructuralValidation { source })?;
        if validate_facet_degree(&facet_map).is_ok() {
            return Err(PlManifoldRepairFixtureError::MissingOversharedFacet { cluster_count });
        }

        Ok(OversharedFacetOrphanCleanupFixture3d { tds, cluster_count })
    }

    /// Builds the fixture and verifies the repair pass removes the intended
    /// simplices and orphan vertices.
    ///
    /// # Errors
    ///
    /// Returns [`PlManifoldRepairFixtureError`] if fixture construction fails,
    /// repair fails, or the repair stats differ from the fixture contract.
    pub fn validated_overshared_facet_orphan_cleanup_3d(
        cluster_count: usize,
    ) -> Result<OversharedFacetOrphanCleanupFixture3d, PlManifoldRepairFixtureError> {
        let fixture = overshared_facet_orphan_cleanup_3d(cluster_count)?;
        let mut repaired = fixture.tds.clone();
        let stats = repair_overshared_facet_orphan_cleanup_3d(&mut repaired)
            .map_err(|source| PlManifoldRepairFixtureError::Repair { source })?;

        if stats.simplices_removed != fixture.cluster_count
            || stats.removed_vertices.len() != fixture.cluster_count
        {
            return Err(PlManifoldRepairFixtureError::UnexpectedRepairStats {
                expected_simplices: fixture.cluster_count,
                actual_simplices: stats.simplices_removed,
                expected_vertices: fixture.cluster_count,
                actual_vertices: stats.removed_vertices.len(),
            });
        }

        Ok(fixture)
    }

    /// Repairs a benchmark fixture using the default PL-manifold repair budget.
    ///
    /// This helper keeps the raw repair primitive crate-internal while giving
    /// benchmark harnesses a stable feature-gated operation to measure.
    ///
    /// # Errors
    ///
    /// Returns [`PlManifoldRepairError`] if the fixture is structurally invalid
    /// or the default repair budget is exhausted.
    pub fn repair_overshared_facet_orphan_cleanup_3d(
        tds: &mut Tds<(), (), 3>,
    ) -> Result<PlManifoldRepairStats<(), (), 3>, PlManifoldRepairError> {
        repair_facet_oversharing(tds, &PlManifoldRepairConfig::default())
    }

    /// Inserts one independent over-shared-facet cluster for the benchmark fixture.
    ///
    /// The cluster intentionally creates exactly one third tetrahedron sharing a
    /// triangular facet, so validated public fixture construction can assert one
    /// removed simplex and one orphan cleanup per cluster.
    fn insert_overshared_cluster(
        tds: &mut Tds<(), (), 3>,
        cluster_index: usize,
    ) -> Result<(), PlManifoldRepairFixtureError> {
        let offset = safe_usize_to_scalar(cluster_index).map_err(|source| {
            PlManifoldRepairFixtureError::ClusterIndexConversion {
                cluster_index,
                source,
            }
        })? * 4.0;

        let vertices = [
            insert_vertex(tds, cluster_index, [offset, 0.0, 0.0])?,
            insert_vertex(tds, cluster_index, [offset + 1.0, 0.0, 0.0])?,
            insert_vertex(tds, cluster_index, [offset, 1.0, 0.0])?,
            insert_vertex(tds, cluster_index, [offset, 0.0, 1.0])?,
            insert_vertex(tds, cluster_index, [offset, 0.0, -1.0])?,
            insert_vertex(tds, cluster_index, [offset + 1.0e-6, 1.0e-6, 1.0e-6])?,
        ];

        insert_checked_simplex(
            tds,
            cluster_index,
            [vertices[0], vertices[1], vertices[2], vertices[3]],
        )?;
        insert_checked_simplex(
            tds,
            cluster_index,
            [vertices[0], vertices[1], vertices[2], vertices[4]],
        )?;
        insert_prechecked_simplex(
            tds,
            cluster_index,
            [vertices[0], vertices[1], vertices[2], vertices[5]],
        )?;
        Ok(())
    }

    /// Inserts a finite fixture vertex while preserving typed construction errors.
    fn insert_vertex(
        tds: &mut Tds<(), (), 3>,
        cluster_index: usize,
        coords: [f64; 3],
    ) -> Result<VertexKey, PlManifoldRepairFixtureError> {
        let vertex =
            Vertex::try_new(coords).map_err(|source| PlManifoldRepairFixtureError::Vertex {
                cluster_index,
                source,
            })?;
        tds.insert_vertex_with_mapping(vertex).map_err(|source| {
            PlManifoldRepairFixtureError::VertexInsert {
                cluster_index,
                source: Box::new(source),
            }
        })
    }

    /// Inserts a manifold-admissible fixture simplex through the normal TDS path.
    fn insert_checked_simplex(
        tds: &mut Tds<(), (), 3>,
        cluster_index: usize,
        vertices: [VertexKey; 4],
    ) -> Result<(), PlManifoldRepairFixtureError> {
        let simplex = Simplex::try_new_with_data(
            vertices.into_iter().collect::<SimplexVertexKeyBuffer>(),
            None,
        )
        .map_err(|source| PlManifoldRepairFixtureError::Simplex {
            cluster_index,
            source,
        })?;
        tds.insert_simplex_with_mapping(simplex)
            .map(|_| ())
            .map_err(|source| PlManifoldRepairFixtureError::SimplexInsert {
                cluster_index,
                source: Box::new(source),
            })
    }

    /// Inserts the deliberate over-shared simplex after local fixture assembly
    /// has already proved its vertices exist and its arity is correct.
    fn insert_prechecked_simplex(
        tds: &mut Tds<(), (), 3>,
        cluster_index: usize,
        vertices: [VertexKey; 4],
    ) -> Result<(), PlManifoldRepairFixtureError> {
        let simplex = Simplex::try_new_with_data(
            vertices.into_iter().collect::<SimplexVertexKeyBuffer>(),
            None,
        )
        .map_err(|source| PlManifoldRepairFixtureError::Simplex {
            cluster_index,
            source,
        })?;
        tds.insert_simplex_with_mapping_prechecked_topology(simplex)
            .map(|_| ())
            .map_err(|source| PlManifoldRepairFixtureError::SimplexInsert {
                cluster_index,
                source: Box::new(source),
            })
    }

    /// Validates Level 1-2 TDS structure before intentionally violating the
    /// PL-manifold facet-degree invariant measured by the benchmark.
    fn validate_structural_fixture_state(
        tds: &Tds<(), (), 3>,
    ) -> Result<(), PlManifoldRepairFixtureError> {
        tds.validate_vertex_mappings()
            .map_err(|source| PlManifoldRepairFixtureError::StructuralValidation { source })?;
        tds.validate_simplex_mappings()
            .map_err(|source| PlManifoldRepairFixtureError::StructuralValidation { source })?;
        tds.validate_simplex_vertex_keys()
            .map_err(|source| PlManifoldRepairFixtureError::StructuralValidation { source })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn overshared_fixture_repairs_expected_orphans() {
            let fixture = validated_overshared_facet_orphan_cleanup_3d(3)
                .expect("fixture should repair deterministically");

            assert_eq!(fixture.cluster_count, 3);
        }
    }
}

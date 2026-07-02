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
        PlManifoldRepairConfig, PlManifoldRepairError, PlManifoldRepairStage,
        PlManifoldRepairStats, manifold_error_matches_repair_stage, repair_facet_oversharing,
        repair_pl_manifold_topology,
    };
    use crate::core::collections::SimplexVertexKeyBuffer;
    use crate::core::simplex::{Simplex, SimplexValidationError};
    use crate::core::tds::{Tds, TdsConstructionError, TdsError, VertexKey};
    use crate::core::vertex::Vertex;
    use crate::geometry::traits::coordinate::CoordinateConversionError;
    use crate::geometry::util::safe_usize_to_scalar;
    use crate::topology::manifold::{
        ManifoldError, ValidatedFacetDegreeMap, validate_closed_boundary_from_validated_facet_map,
        validate_ridge_links, validate_vertex_links_from_validated_facet_map,
    };
    use crate::topology::traits::topological_space::GlobalTopology;
    use thiserror::Error;

    /// Fixture for benchmarking facet over-sharing repair plus orphan cleanup in 3D.
    #[derive(Clone, Debug)]
    #[must_use]
    pub struct OversharedFacetOrphanCleanupFixture3d {
        /// Structurally coherent TDS with one over-shared facet per cluster.
        tds: Tds<(), (), 3>,
        /// Number of independent over-shared clusters in the fixture.
        cluster_count: usize,
    }

    impl OversharedFacetOrphanCleanupFixture3d {
        /// Returns the structurally coherent benchmark TDS.
        #[must_use]
        pub const fn tds(&self) -> &Tds<(), (), 3> {
            &self.tds
        }

        /// Returns the number of independent over-shared clusters.
        #[must_use]
        pub const fn cluster_count(&self) -> usize {
            self.cluster_count
        }
    }

    /// Fixture for benchmarking targeted PL-manifold repair stages.
    #[derive(Clone, Debug)]
    #[must_use]
    pub struct TargetedTopologyRepairFixture<const D: usize> {
        /// Structurally coherent TDS with targeted PL-manifold violations.
        tds: Tds<(), (), D>,
        /// Number of independent targeted-violation clusters in the fixture.
        cluster_count: usize,
        /// Targeted repair stage exercised by this fixture.
        stage: PlManifoldRepairStage,
    }

    impl<const D: usize> TargetedTopologyRepairFixture<D> {
        /// Returns the structurally coherent benchmark TDS.
        #[must_use]
        pub const fn tds(&self) -> &Tds<(), (), D> {
            &self.tds
        }

        /// Returns the number of independent targeted-violation clusters.
        #[must_use]
        pub const fn cluster_count(&self) -> usize {
            self.cluster_count
        }

        /// Returns the targeted repair stage validated for this fixture.
        #[must_use]
        pub const fn stage(&self) -> PlManifoldRepairStage {
            self.stage
        }
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

        /// A local grid index could not be converted to a finite coordinate offset.
        #[error(
            "failed to convert grid index {grid_index} for cluster {cluster_index} to a coordinate offset: {source}"
        )]
        GridIndexConversion {
            /// Cluster that was being inserted.
            cluster_index: usize,
            /// Local grid index being converted.
            grid_index: usize,
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

        /// The fixture did not produce the intended targeted topology violation.
        #[error(
            "PL-manifold repair fixture with {cluster_count} clusters did not produce a {stage} violation"
        )]
        MissingTargetedViolation {
            /// Targeted repair stage expected from the fixture.
            stage: PlManifoldRepairStage,
            /// Number of clusters requested.
            cluster_count: usize,
        },

        /// The fixture produced a different topology violation than the one being benchmarked.
        #[error("expected {stage} fixture violation, but validation reported: {source}")]
        UnexpectedTargetedViolation {
            /// Targeted repair stage expected from the fixture.
            stage: PlManifoldRepairStage,
            /// Typed validation error that was observed instead.
            source: ManifoldError,
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

        /// Targeted repair succeeded but removed less topology than the fixture contains.
        #[error(
            "{stage} fixture removed {actual_simplices} simplices; expected at least {expected_simplices_at_least}"
        )]
        UnexpectedTargetedRepairStats {
            /// Targeted repair stage validated by the fixture.
            stage: PlManifoldRepairStage,
            /// Expected lower bound for removed simplices.
            expected_simplices_at_least: usize,
            /// Actual number of removed simplices.
            actual_simplices: usize,
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
        if ValidatedFacetDegreeMap::try_from_facet_map(&facet_map).is_ok() {
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

    /// Builds and validates a 3D boundary-ridge multiplicity repair benchmark fixture.
    ///
    /// # Errors
    ///
    /// Returns [`PlManifoldRepairFixtureError`] if construction fails, the
    /// initial topology does not fail with the boundary-ridge stage, or repair
    /// cannot restore targeted PL-manifold invariants.
    pub fn validated_boundary_ridge_multiplicity_repair_3d(
        cluster_count: usize,
    ) -> Result<TargetedTopologyRepairFixture<3>, PlManifoldRepairFixtureError> {
        validate_targeted_fixture(targeted_topology_fixture(
            cluster_count,
            PlManifoldRepairStage::BoundaryRidgeMultiplicity,
            insert_boundary_ridge_multiplicity_cluster,
        )?)
    }

    /// Builds and validates a 2D ridge-link repair benchmark fixture.
    ///
    /// # Errors
    ///
    /// Returns [`PlManifoldRepairFixtureError`] if construction fails, the
    /// initial topology does not fail with the ridge-link stage, or repair
    /// cannot restore targeted PL-manifold invariants.
    pub fn validated_ridge_link_repair_2d(
        cluster_count: usize,
    ) -> Result<TargetedTopologyRepairFixture<2>, PlManifoldRepairFixtureError> {
        validate_targeted_fixture(targeted_topology_fixture(
            cluster_count,
            PlManifoldRepairStage::RidgeLink,
            insert_ridge_link_cluster,
        )?)
    }

    /// Builds and validates a 3D vertex-link repair benchmark fixture.
    ///
    /// # Errors
    ///
    /// Returns [`PlManifoldRepairFixtureError`] if construction fails, the
    /// initial topology does not fail with the vertex-link stage, or repair
    /// cannot restore targeted PL-manifold invariants.
    pub fn validated_vertex_link_repair_3d(
        cluster_count: usize,
    ) -> Result<TargetedTopologyRepairFixture<3>, PlManifoldRepairFixtureError> {
        validate_targeted_fixture(targeted_topology_fixture(
            cluster_count,
            PlManifoldRepairStage::VertexLink,
            insert_vertex_link_cluster,
        )?)
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

    /// Repairs a targeted topology benchmark fixture using the default PL-manifold repair budget.
    ///
    /// # Errors
    ///
    /// Returns [`PlManifoldRepairError`] if the fixture is structurally invalid,
    /// targeted repair exhausts its default budget, or the postcondition
    /// validation fails.
    pub fn repair_targeted_pl_manifold_topology<const D: usize>(
        tds: &mut Tds<(), (), D>,
    ) -> Result<PlManifoldRepairStats<(), (), D>, PlManifoldRepairError> {
        repair_pl_manifold_topology(
            tds,
            GlobalTopology::Euclidean,
            &PlManifoldRepairConfig::default(),
        )
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
        let offset = cluster_offset(cluster_index, 4.0)?;

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

    /// Builds a targeted topology fixture from independent invalid clusters.
    fn targeted_topology_fixture<const D: usize>(
        cluster_count: usize,
        stage: PlManifoldRepairStage,
        mut insert_cluster: impl FnMut(
            &mut Tds<(), (), D>,
            usize,
        ) -> Result<(), PlManifoldRepairFixtureError>,
    ) -> Result<TargetedTopologyRepairFixture<D>, PlManifoldRepairFixtureError> {
        if cluster_count == 0 {
            return Err(PlManifoldRepairFixtureError::EmptyClusterCount);
        }

        let mut tds = Tds::empty();
        for cluster_index in 0..cluster_count {
            insert_cluster(&mut tds, cluster_index)?;
        }

        validate_structural_fixture_state(&tds)?;
        validate_expected_targeted_violation(&tds, stage, cluster_count)?;

        Ok(TargetedTopologyRepairFixture {
            tds,
            cluster_count,
            stage,
        })
    }

    /// Validates that a targeted fixture repairs successfully before benchmarking.
    fn validate_targeted_fixture<const D: usize>(
        fixture: TargetedTopologyRepairFixture<D>,
    ) -> Result<TargetedTopologyRepairFixture<D>, PlManifoldRepairFixtureError> {
        let mut repaired = fixture.tds.clone();
        let stats = repair_targeted_pl_manifold_topology(&mut repaired)
            .map_err(|source| PlManifoldRepairFixtureError::Repair { source })?;

        if !stats.succeeded || stats.simplices_removed < fixture.cluster_count {
            return Err(
                PlManifoldRepairFixtureError::UnexpectedTargetedRepairStats {
                    stage: fixture.stage,
                    expected_simplices_at_least: fixture.cluster_count,
                    actual_simplices: stats.simplices_removed,
                },
            );
        }

        Ok(fixture)
    }

    /// Confirms the initial fixture fails with the targeted validation stage being measured.
    fn validate_expected_targeted_violation<const D: usize>(
        tds: &Tds<(), (), D>,
        stage: PlManifoldRepairStage,
        cluster_count: usize,
    ) -> Result<(), PlManifoldRepairFixtureError> {
        match validate_targeted_stage(tds, stage) {
            Ok(()) => Err(PlManifoldRepairFixtureError::MissingTargetedViolation {
                stage,
                cluster_count,
            }),
            Err(source) if manifold_error_matches_repair_stage(&source, stage) => Ok(()),
            Err(source) => {
                Err(PlManifoldRepairFixtureError::UnexpectedTargetedViolation { stage, source })
            }
        }
    }

    /// Runs the targeted validator corresponding to a benchmark fixture stage.
    fn validate_targeted_stage<const D: usize>(
        tds: &Tds<(), (), D>,
        stage: PlManifoldRepairStage,
    ) -> Result<(), ManifoldError> {
        match stage {
            PlManifoldRepairStage::BoundaryRidgeMultiplicity => {
                validate_boundary_ridge_multiplicity_stage(tds)
            }
            PlManifoldRepairStage::RidgeLink => validate_ridge_links(tds),
            PlManifoldRepairStage::VertexLink => validate_vertex_link_stage(tds),
        }
    }

    /// Validates the boundary-ridge stage through the same parsed facet-degree proof.
    fn validate_boundary_ridge_multiplicity_stage<const D: usize>(
        tds: &Tds<(), (), D>,
    ) -> Result<(), ManifoldError> {
        let facet_to_simplices = tds.build_facet_to_simplices_map()?;
        let facet_to_simplices = ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices)?;
        validate_closed_boundary_from_validated_facet_map(
            tds,
            facet_to_simplices,
            GlobalTopology::Euclidean,
        )
    }

    /// Validates the vertex-link stage through the same parsed facet-degree proof.
    fn validate_vertex_link_stage<const D: usize>(
        tds: &Tds<(), (), D>,
    ) -> Result<(), ManifoldError> {
        let facet_to_simplices = tds.build_facet_to_simplices_map()?;
        let facet_to_simplices = ValidatedFacetDegreeMap::try_from_facet_map(&facet_to_simplices)?;
        validate_vertex_links_from_validated_facet_map(
            tds,
            facet_to_simplices,
            GlobalTopology::Euclidean,
        )
    }

    /// Inserts one independent cluster with two tetrahedra sharing an edge but no facet.
    fn insert_boundary_ridge_multiplicity_cluster(
        tds: &mut Tds<(), (), 3>,
        cluster_index: usize,
    ) -> Result<(), PlManifoldRepairFixtureError> {
        let offset = cluster_offset(cluster_index, 6.0)?;
        let shared_v0 = insert_vertex(tds, cluster_index, [offset, 0.0, 0.0])?;
        let shared_v1 = insert_vertex(tds, cluster_index, [offset + 2.0, 0.0, 0.0])?;
        let tet1_v2 = insert_vertex(tds, cluster_index, [offset + 0.1, 1.0, 0.2])?;
        let tet1_v3 = insert_vertex(tds, cluster_index, [offset + 0.2, 0.3, 1.3])?;
        let tet2_v2 = insert_vertex(tds, cluster_index, [offset + 0.4, -1.1, 0.7])?;
        let tet2_v3 = insert_vertex(tds, cluster_index, [offset + 0.6, 0.2, -1.4])?;

        for vertices in [
            [shared_v0, shared_v1, tet1_v2, tet1_v3],
            [shared_v0, shared_v1, tet2_v2, tet2_v3],
        ] {
            insert_checked_simplex(tds, cluster_index, vertices)?;
        }

        Ok(())
    }

    /// Inserts one 2D cluster made from two closed triangle-sphere complexes sharing one vertex.
    fn insert_ridge_link_cluster(
        tds: &mut Tds<(), (), 2>,
        cluster_index: usize,
    ) -> Result<(), PlManifoldRepairFixtureError> {
        let offset = cluster_offset(cluster_index, 16.0)?;
        let v0 = insert_vertex(tds, cluster_index, [offset, 0.0])?;
        let v1 = insert_vertex(tds, cluster_index, [offset + 1.0, 0.0])?;
        let v2 = insert_vertex(tds, cluster_index, [offset, 1.0])?;
        let v3 = insert_vertex(tds, cluster_index, [offset + 1.0, 1.0])?;
        let v4 = insert_vertex(tds, cluster_index, [offset + 8.0, 8.0])?;
        let v5 = insert_vertex(tds, cluster_index, [offset + 9.0, 8.0])?;
        let v6 = insert_vertex(tds, cluster_index, [offset + 8.0, 9.0])?;

        for vertices in [
            [v0, v1, v2],
            [v0, v1, v3],
            [v0, v2, v3],
            [v1, v2, v3],
            [v0, v4, v5],
            [v0, v4, v6],
            [v0, v5, v6],
            [v4, v5, v6],
        ] {
            insert_checked_simplex(tds, cluster_index, vertices)?;
        }

        Ok(())
    }

    /// Inserts one 3D cluster whose apex has a torus link instead of a sphere or ball link.
    fn insert_vertex_link_cluster(
        tds: &mut Tds<(), (), 3>,
        cluster_index: usize,
    ) -> Result<(), PlManifoldRepairFixtureError> {
        const N: usize = 3;
        const M: usize = 3;

        let offset = cluster_offset(cluster_index, 8.0)?;
        let mut grid: Vec<Vec<VertexKey>> = Vec::with_capacity(N);
        for i in 0..N {
            let mut row = Vec::with_capacity(M);
            for j in 0..M {
                let x = offset + grid_coordinate(cluster_index, i)?;
                let y = grid_coordinate(cluster_index, j)?;
                row.push(insert_vertex(tds, cluster_index, [x, y, 0.0])?);
            }
            grid.push(row);
        }
        let apex = insert_vertex(tds, cluster_index, [offset + 0.5, 0.5, 1.0])?;

        for i in 0..N {
            for j in 0..M {
                let i1 = (i + 1) % N;
                let j1 = (j + 1) % M;
                let v00 = grid[i][j];
                let v10 = grid[i1][j];
                let v01 = grid[i][j1];
                let v11 = grid[i1][j1];
                for vertices in [[v00, v10, v01, apex], [v10, v11, v01, apex]] {
                    insert_checked_simplex(tds, cluster_index, vertices)?;
                }
            }
        }

        Ok(())
    }

    /// Converts a cluster index to a separated coordinate offset.
    fn cluster_offset(
        cluster_index: usize,
        spacing: f64,
    ) -> Result<f64, PlManifoldRepairFixtureError> {
        Ok(safe_usize_to_scalar(cluster_index).map_err(|source| {
            PlManifoldRepairFixtureError::ClusterIndexConversion {
                cluster_index,
                source,
            }
        })? * spacing)
    }

    /// Converts a local grid index to a finite coordinate.
    fn grid_coordinate(
        cluster_index: usize,
        grid_index: usize,
    ) -> Result<f64, PlManifoldRepairFixtureError> {
        safe_usize_to_scalar(grid_index).map_err(|source| {
            PlManifoldRepairFixtureError::GridIndexConversion {
                cluster_index,
                grid_index,
                source,
            }
        })
    }

    /// Inserts a finite fixture vertex while preserving typed construction errors.
    fn insert_vertex<const D: usize>(
        tds: &mut Tds<(), (), D>,
        cluster_index: usize,
        coords: [f64; D],
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
    fn insert_checked_simplex<const D: usize, const N: usize>(
        tds: &mut Tds<(), (), D>,
        cluster_index: usize,
        vertices: [VertexKey; N],
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
    fn insert_prechecked_simplex<const D: usize, const N: usize>(
        tds: &mut Tds<(), (), D>,
        cluster_index: usize,
        vertices: [VertexKey; N],
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
    fn validate_structural_fixture_state<const D: usize>(
        tds: &Tds<(), (), D>,
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

            assert_eq!(fixture.cluster_count(), 3);
        }

        #[test]
        fn targeted_fixtures_repair_expected_stages() {
            let boundary = validated_boundary_ridge_multiplicity_repair_3d(2)
                .expect("boundary-ridge fixture should repair deterministically");
            let ridge = validated_ridge_link_repair_2d(2)
                .expect("ridge-link fixture should repair deterministically");
            let vertex = validated_vertex_link_repair_3d(2)
                .expect("vertex-link fixture should repair deterministically");

            assert_eq!(
                boundary.stage(),
                PlManifoldRepairStage::BoundaryRidgeMultiplicity
            );
            assert_eq!(ridge.stage(), PlManifoldRepairStage::RidgeLink);
            assert_eq!(vertex.stage(), PlManifoldRepairStage::VertexLink);
        }
    }
}

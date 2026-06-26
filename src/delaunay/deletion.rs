//! Vertex-deletion operations for Delaunay triangulations.
//!
//! This module owns post-construction vertex deletion, rollback, cache
//! invalidation, and repair-after-deletion error mapping. Vertex insertion
//! remains in `delaunay::insertion` because the insertion path has its own
//! cavity construction and post-insertion repair workflow.

#![forbid(unsafe_code)]

use crate::construction::local_repair_flip_budget;
#[cfg(test)]
use crate::construction::test_hooks;
use crate::core::algorithms::flips::{
    FlipError, apply_bistellar_flip_k1_inverse, repair_delaunay_with_flips_k2_k3,
};
use crate::core::collections::SimplexKeyBuffer;
use crate::core::tds::{InvariantError, NeighborValidationError, TdsError, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::validation::insertion_error_to_invariant_error;
use crate::delaunay_rollback::{DelaunayRollbackTransaction, DelaunaySpatialIndexRollback};
use crate::geometry::kernel::Kernel;
use crate::repair::DelaunayRepairOperation;
use crate::triangulation::DelaunayTriangulation;
use crate::validation::DelaunayTriangulationValidationError;
use thiserror::Error;

/// Errors returned by [`DelaunayTriangulation::delete_vertex`].
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DeleteVertexError {
    /// The requested vertex key is not live in this triangulation.
    #[error("Vertex key {vertex_key:?} is not present in this triangulation")]
    VertexNotFound {
        /// Missing or stale vertex key supplied by the caller.
        vertex_key: VertexKey,
    },

    /// Deleting the vertex would violate structural, topological, or Delaunay invariants.
    #[error("Vertex deletion failed to preserve triangulation invariants: {source}")]
    InvariantViolation {
        /// Structured invariant failure produced by the deletion attempt.
        #[from]
        source: InvariantError,
    },
}

// =============================================================================
// VERTEX DELETION (Requires Numeric Scalar Bounds)
// =============================================================================

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Deletes a vertex and retriangulates the resulting cavity using fan triangulation.
    ///
    /// This operation delegates to the core triangulation layer, which:
    /// 1. Finds all simplices containing the vertex
    /// 2. Deletes those simplices (creating a cavity)
    /// 3. Fills the cavity with fan triangulation
    /// 4. Wires neighbors and rebuilds vertex-simplex incidence
    /// 5. Deletes the vertex
    ///
    /// Fast-path: if the vertex star is a simplex (exactly D+1 incident simplices with
    /// consistent adjacency), this method collapses it via the **inverse k=1** bistellar
    /// flip. Otherwise it falls back to fan triangulation.
    ///
    /// This operation is topology-preserving on success: it returns `Ok` only after the
    /// post-deletion triangulation satisfies the required manifold and topology invariants. A
    /// candidate deletion that would collapse the mesh to a lower-dimensional remnant or isolate
    /// remaining vertices is rejected as a [`DeleteVertexError::InvariantViolation`] wrapping
    /// [`InvariantError::Triangulation`], and the pre-deletion state is restored. Both the inverse
    /// k=1 fast-path and fan triangulation may temporarily violate the Delaunay property in some
    /// cases. If the [`DelaunayRepairPolicy`](crate::DelaunayRepairPolicy) allows it, a flip-based
    /// repair pass is run automatically after deletion. Otherwise, Level 4 validation is run without
    /// mutating repair, and Delaunay violations roll back as invariant failures.
    ///
    /// The post-deletion repair and orientation canonicalization steps are
    /// transactional: if either step fails, this method restores the triangulation
    /// and insertion state to their pre-deletion state before returning the error.
    /// The spatial index is retained across rollback because its keys are
    /// validated against the live TDS before use. On successful deletion,
    /// topology-dependent locate hints are invalidated and the removed vertex key
    /// is pruned from the spatial index.
    ///
    /// **Future Enhancement**: Delaunay-aware cavity retriangulation will be added for
    /// deletions. For now, local retriangulation can still require post-deletion flip repair; if
    /// automatic repair is disabled and Level 4 validation detects a violation, deletion fails and
    /// rolls back.
    ///
    /// # Arguments
    ///
    /// * `vertex_key` - Key of the vertex to delete
    ///
    /// # Returns
    ///
    /// The number of simplices that were deleted along with the vertex.
    ///
    /// # Errors
    ///
    /// Returns [`DeleteVertexError`] if:
    /// - `vertex_key` does not refer to a live vertex in the triangulation
    ///   ([`DeleteVertexError::VertexNotFound`]).
    /// - The inverse k=1 flip encounters a neighbor-wiring failure
    ///   ([`DeleteVertexError::InvariantViolation`] wrapping [`InvariantError::Tds`]).
    /// - Fan retriangulation fails ([`DeleteVertexError::InvariantViolation`] wrapping
    ///   [`InvariantError::Tds`]).
    /// - Post-deletion topology validation fails, for example because deletion would leave
    ///   isolated vertices or a lower-dimensional remnant
    ///   ([`DeleteVertexError::InvariantViolation`] wrapping [`InvariantError::Triangulation`]).
    /// - Delaunay flip-based repair fails after deletion
    ///   ([`DeleteVertexError::InvariantViolation`] wrapping [`InvariantError::Delaunay`] wrapping
    ///   [`DelaunayTriangulationValidationError::RepairOperationFailed`]).
    /// - Level 4 Delaunay validation fails after deletion when automatic repair is disabled
    ///   ([`DeleteVertexError::InvariantViolation`] wrapping [`InvariantError::Delaunay`] wrapping
    ///   [`DelaunayTriangulationValidationError::VerificationFailed`]).
    /// - Orientation canonicalization fails after repair
    ///   ([`DeleteVertexError::InvariantViolation`] wrapping [`InvariantError::Tds`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     DeleteVertex(#[from] delaunay::DeleteVertexError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let interior = delaunay::vertex![0.3, 0.3]?;
    /// let interior_uuid = interior.uuid();
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    ///     interior,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    ///
    /// // Find the key of a known interior vertex.
    /// let Some((vertex_key, _)) = dt.vertices().find(|(_, v)| v.uuid() == interior_uuid) else {
    ///     return Ok(());
    /// };
    ///
    /// // Delete the vertex and all simplices containing it
    /// let simplices_removed = dt.delete_vertex(vertex_key)?;
    /// println!("Deleted {} simplices along with the vertex", simplices_removed);
    ///
    /// // Vertex deletion preserves topology; automatic repair is attempted when enabled.
    /// assert!(dt.as_triangulation().validate().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Deletions that would leave a non-manifold remnant fail and roll back:
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder};
    /// use delaunay::prelude::deletion::DeleteVertexError;
    /// use delaunay::prelude::tds::InvariantError;
    /// use delaunay::prelude::triangulation::TriangulationValidationError;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((vertex_key, _)) = dt.vertices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let err = dt
    ///     .delete_vertex(vertex_key)
    ///     .expect_err("deletion should leave an isolated vertex");
    /// std::assert_matches!(
    ///     err,
    ///     DeleteVertexError::InvariantViolation {
    ///         source: InvariantError::Triangulation(
    ///             TriangulationValidationError::IsolatedVertex { .. }
    ///         )
    ///     }
    /// );
    /// assert_eq!(dt.number_of_vertices(), 3);
    /// assert_eq!(dt.number_of_simplices(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete_vertex(&mut self, vertex_key: VertexKey) -> Result<usize, DeleteVertexError> {
        let Some(removed_vertex) = self.tri.tds.vertex(vertex_key) else {
            return Err(DeleteVertexError::VertexNotFound { vertex_key });
        };
        let removed_vertex_coords = *removed_vertex.point().coords();

        let mut transaction =
            DelaunayRollbackTransaction::begin(self, DelaunaySpatialIndexRollback::Restore);
        let result: Result<usize, InvariantError> = {
            let delaunay = transaction.delaunay_mut();
            (|| {
                // Fast path: inverse k=1 flip when the vertex star is a simplex.
                let mut seed_simplices: Option<SimplexKeyBuffer> = None;
                let simplices_removed =
                    match apply_bistellar_flip_k1_inverse(&mut delaunay.tri.tds, vertex_key) {
                        Ok(info) => {
                            seed_simplices = Some(info.new_simplices);
                            info.removed_simplices.len()
                        }
                        Err(FlipError::NeighborWiring { reason }) => {
                            return Err(TdsError::InvalidNeighbors {
                                reason: NeighborValidationError::FlipNeighborWiring { reason },
                            }
                            .into());
                        }
                        Err(_) => {
                            let outcome =
                                delaunay.tri.remove_vertex_with_repair_seeds(vertex_key)?;
                            if !outcome.repair_seed_simplices.is_empty() {
                                seed_simplices = Some(outcome.repair_seed_simplices);
                            }
                            outcome.simplices_removed
                        }
                    };

                let topology = delaunay.tri.topology_guarantee();
                if delaunay.should_run_delaunay_repair_after_mutation(topology) {
                    let seed_ref = seed_simplices.as_deref();
                    let repair_seed_count =
                        seed_ref.map_or_else(|| delaunay.tri.tds.number_of_simplices(), <[_]>::len);
                    let max_flips = local_repair_flip_budget::<D>(repair_seed_count);
                    let repair_result = {
                        delaunay.invalidate_locate_hint_cache();
                        let (tds, kernel) = (&mut delaunay.tri.tds, &delaunay.tri.kernel);
                        repair_delaunay_with_flips_k2_k3(
                            tds,
                            kernel,
                            seed_ref,
                            topology,
                            Some(max_flips),
                        )
                    };

                    #[cfg(test)]
                    let repair_result = if test_hooks::force_repair_nonconvergent_enabled() {
                        Err(test_hooks::synthetic_nonconvergent_error())
                    } else {
                        repair_result
                    };

                    repair_result.map_err(|source| {
                        InvariantError::Delaunay(
                            DelaunayTriangulationValidationError::RepairOperationFailed {
                                operation: DelaunayRepairOperation::VertexRemoval,
                                source: Box::new(source),
                            },
                        )
                    })?;

                    // Re-canonicalize geometric orientation (#258): flip repair may leave
                    // the global sign negative.
                    delaunay
                        .tri
                        .normalize_and_promote_positive_orientation()
                        .map_err(|e| {
                            insertion_error_to_invariant_error(
                                e,
                                "Orientation canonicalization failed after vertex deletion",
                            )
                        })?;
                } else {
                    delaunay.is_valid().map_err(InvariantError::Delaunay)?;
                }

                Ok(simplices_removed)
            })()
        };

        match result {
            Ok(simplices_removed) => {
                let delaunay = transaction.delaunay_mut();
                delaunay.insertion_state.last_inserted_simplex = None;
                if let Some(index) = delaunay.spatial_index.as_mut() {
                    index.remove_vertex(&vertex_key, &removed_vertex_coords);
                }
                transaction.commit();
                Ok(simplices_removed)
            }
            Err(err) => {
                transaction.rollback();
                Err(err.into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::flips::DelaunayRepairError;
    use crate::core::collections::spatial_hash_grid::HashGridIndex;
    use crate::core::validation::{TopologyGuarantee, TriangulationValidationError};
    use crate::core::vertex::Vertex;
    use crate::flips::BistellarFlips;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::geometry::util::safe_usize_to_scalar;
    use crate::repair::DelaunayRepairPolicy;
    use crate::vertex;
    use std::assert_matches;
    use std::sync::Once;
    use uuid::Uuid;

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer()
                .try_init();
        });
    }

    struct ForceRepairNonconvergentGuard {
        previous: bool,
    }

    impl ForceRepairNonconvergentGuard {
        fn enable() -> Self {
            Self {
                previous: test_hooks::set_force_repair_nonconvergent(true),
            }
        }
    }

    impl Drop for ForceRepairNonconvergentGuard {
        fn drop(&mut self) {
            let _ = test_hooks::set_force_repair_nonconvergent(self.previous);
        }
    }

    fn simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(vertex!([0.0; D]).unwrap());
        for axis in 0..D {
            let mut coords = [0.0; D];
            coords[axis] = 1.0;
            vertices.push(vertex!(coords).unwrap());
        }
        vertices
    }

    fn interior_vertex_for_k1_insert<const D: usize>() -> Vertex<(), D> {
        let denominator = safe_usize_to_scalar(D + 2)
            .expect("D + 2 should convert exactly for rollback test dimensions");
        let coord = 1.0 / denominator;
        vertex!([coord; D]).unwrap()
    }

    fn rollback_probe_vertex<const D: usize>(point_index: usize) -> Vertex<(), D> {
        let dimension = safe_usize_to_scalar(D).expect("test dimensions should convert exactly");
        let point_index_scalar =
            safe_usize_to_scalar(point_index).expect("point index should convert exactly");
        let mut coords = [0.2 / dimension; D];
        let axis = point_index % D;
        coords[axis] += point_index_scalar.mul_add(0.005, 0.02);
        vertex!(coords).unwrap()
    }

    fn incident_simplex_count<const D: usize>(
        dt: &DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
        vertex_key: VertexKey,
    ) -> usize {
        dt.simplices()
            .filter(|(_, simplex)| simplex.vertices().contains(&vertex_key))
            .count()
    }

    fn assert_forced_delete_vertex_rolls_back<const D: usize>(
        dt: &mut DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
        vertex_key: VertexKey,
        inserted_uuid: Uuid,
    ) {
        let vertex_count_before = dt.number_of_vertices();
        let simplex_count_before = dt.number_of_simplices();
        let hint_simplex_before = dt.simplices().next().map(|(key, _)| key);
        dt.insertion_state.last_inserted_simplex = hint_simplex_before;
        let mut spatial_index = HashGridIndex::<D>::try_new(1.0).unwrap();
        for (vertex_key, vertex) in dt.vertices() {
            spatial_index.insert_vertex(vertex_key, vertex.point().coords());
        }
        dt.spatial_index = Some(spatial_index);
        let last_inserted_simplex_before = dt.insertion_state.last_inserted_simplex;
        let spatial_index_before = dt
            .spatial_index
            .as_ref()
            .map(HashGridIndex::<D>::debug_snapshot);

        let _guard = ForceRepairNonconvergentGuard::enable();
        let result = dt.delete_vertex(vertex_key);
        let err = result.expect_err("forced repair failure should make deletion fail");
        match err {
            DeleteVertexError::InvariantViolation {
                source:
                    InvariantError::Delaunay(
                        DelaunayTriangulationValidationError::RepairOperationFailed {
                            operation: DelaunayRepairOperation::VertexRemoval,
                            source,
                        },
                    ),
            } if matches!(
                source.as_ref(),
                DelaunayRepairError::NonConvergent { max_flips: 0, .. }
            ) => {}
            DeleteVertexError::InvariantViolation {
                source:
                    InvariantError::Triangulation(
                        TriangulationValidationError::OrientationPromotionNonConvergence { .. },
                    )
                    | InvariantError::Tds(TdsError::FacetSharingViolation { .. }),
            } => {}
            other => panic!(
                "expected vertex-deletion rollback error from forced repair path, got {other:?}"
            ),
        }

        assert_eq!(dt.number_of_vertices(), vertex_count_before);
        assert_eq!(dt.number_of_simplices(), simplex_count_before);
        assert_eq!(
            dt.insertion_state.last_inserted_simplex, last_inserted_simplex_before,
            "delete_vertex rollback should restore last_inserted_simplex"
        );
        assert_eq!(
            dt.spatial_index
                .as_ref()
                .map(HashGridIndex::<D>::debug_snapshot),
            spatial_index_before,
            "delete_vertex rollback should restore spatial_index"
        );
        assert!(dt.vertices().any(|(_, v)| v.uuid() == inserted_uuid));
        assert!(dt.as_triangulation().validate().is_ok());
    }

    fn assert_delete_vertex_rollback<const D: usize>() {
        init_tracing();
        let vertices = simplex_vertices::<D>();

        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);

        let simplex_key = dt.simplices().next().unwrap().0;
        let inserted_vertex = interior_vertex_for_k1_insert::<D>();
        let inserted_uuid = inserted_vertex.uuid();
        dt.flip_k1_insert(simplex_key, inserted_vertex).unwrap();

        let vertex_key = dt
            .vertices()
            .find(|(_, v)| v.uuid() == inserted_uuid)
            .map(|(k, _)| k)
            .expect("Inserted vertex not found");

        assert_forced_delete_vertex_rolls_back(&mut dt, vertex_key, inserted_uuid);
    }

    fn assert_delete_vertex_fallback_rollback<const D: usize>() {
        init_tracing();
        let vertices = simplex_vertices::<D>();

        let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);

        let mut inserted_vertices = Vec::new();
        for point_index in 0..(D + 3) {
            let inserted_vertex = rollback_probe_vertex::<D>(point_index);
            let inserted_uuid = inserted_vertex.uuid();
            let vertex_key = dt
                .insert_vertex(inserted_vertex)
                .expect("rollback fallback fixture insertion should succeed");
            inserted_vertices.push((vertex_key, inserted_uuid));
        }

        let (vertex_key, inserted_uuid, incident_simplices) = inserted_vertices
            .iter()
            .find_map(|&(vertex_key, inserted_uuid)| {
                let incident_simplices = incident_simplex_count(&dt, vertex_key);
                (incident_simplices != D + 1).then_some((
                    vertex_key,
                    inserted_uuid,
                    incident_simplices,
                ))
            })
            .expect("expected at least one inserted vertex with a non-simplex star");
        assert_ne!(
            incident_simplices,
            D + 1,
            "fallback rollback fixture must avoid the inverse-k=1 simplex-star path"
        );

        assert_forced_delete_vertex_rolls_back(&mut dt, vertex_key, inserted_uuid);
    }

    macro_rules! gen_delete_vertex_rollback_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<delete_vertex_rollback_ $dim d>]() {
                    assert_delete_vertex_rollback::<$dim>();
                }

                #[test]
                fn [<delete_vertex_fallback_rollback_ $dim d>]() {
                    assert_delete_vertex_fallback_rollback::<$dim>();
                }
            }
        };
    }

    gen_delete_vertex_rollback_tests!(2);
    gen_delete_vertex_rollback_tests!(3);
    gen_delete_vertex_rollback_tests!(4);
    gen_delete_vertex_rollback_tests!(5);

    #[test]
    fn test_delete_vertex_fast_path_inverse_k1() {
        init_tracing();
        let vertices: Vec<Vertex<(), 3>> = vec![
            vertex![0.0, 0.0, 0.0].unwrap(),
            vertex![1.0, 0.0, 0.0].unwrap(),
            vertex![0.0, 1.0, 0.0].unwrap(),
            vertex![0.0, 0.0, 1.0].unwrap(),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);
        let original_vertex_count = dt.number_of_vertices();
        let original_simplex_count = dt.number_of_simplices();

        let simplex_key = dt.simplices().next().unwrap().0;
        let inserted_vertex = vertex![0.2, 0.2, 0.2].unwrap();
        let inserted_uuid = inserted_vertex.uuid();
        dt.flip_k1_insert(simplex_key, inserted_vertex).unwrap();

        assert_eq!(dt.number_of_vertices(), original_vertex_count + 1);
        assert_eq!(dt.number_of_simplices(), original_simplex_count + 3);

        let vertex_key = dt
            .vertices()
            .find(|(_, v)| v.uuid() == inserted_uuid)
            .map(|(k, _)| k)
            .expect("Inserted vertex not found");

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
        let removed_simplices = dt.delete_vertex(vertex_key).unwrap();

        assert_eq!(removed_simplices, 4);
        assert_eq!(dt.number_of_vertices(), original_vertex_count);
        assert_eq!(dt.number_of_simplices(), original_simplex_count);
        assert!(dt.as_triangulation().validate().is_ok());
        assert!(dt.vertices().all(|(_, v)| v.uuid() != inserted_uuid));
    }

    #[test]
    fn delete_vertex_invalidates_locate_hint_and_prunes_spatial_index() {
        init_tracing();
        let vertices: Vec<Vertex<(), 2>> = vec![
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let vertex_key = dt.insert_vertex(vertex![0.25, 0.25].unwrap()).unwrap();
        let hint_simplex = dt.simplices().next().map(|(key, _)| key);
        dt.insertion_state.last_inserted_simplex = hint_simplex;
        let mut spatial_index = HashGridIndex::<2>::try_new(1.0).unwrap();
        for (vertex_key, vertex) in dt.vertices() {
            spatial_index.insert_vertex(vertex_key, vertex.point().coords());
        }
        dt.spatial_index = Some(spatial_index);
        assert!(dt.insertion_state.last_inserted_simplex.is_some());
        assert!(dt.spatial_index.is_some());

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
        let removed_simplices = dt.delete_vertex(vertex_key).unwrap();

        assert!(removed_simplices > 0);
        assert!(dt.insertion_state.last_inserted_simplex.is_none());
        let spatial_index = dt
            .spatial_index
            .as_ref()
            .expect("successful vertex deletion should retain the spatial index");
        let mut found_removed_key = false;
        assert!(
            spatial_index.for_each_candidate_vertex_key(&[0.25, 0.25], |candidate| {
                found_removed_key |= candidate == vertex_key;
                true
            })
        );
        assert!(!found_removed_key);
        assert!(dt.as_triangulation().validate().is_ok());
    }

    #[test]
    fn delete_vertex_rejects_stale_key_without_mutating_topology() {
        init_tracing();
        let vertices = vec![
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let vertex_key = dt.insert_vertex(vertex![0.25, 0.25].unwrap()).unwrap();

        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);
        let removed_simplices = dt.delete_vertex(vertex_key).unwrap();
        assert!(removed_simplices > 0);

        let vertex_count_before_stale_attempt = dt.number_of_vertices();
        let simplex_count_before_stale_attempt = dt.number_of_simplices();
        let err = dt
            .delete_vertex(vertex_key)
            .expect_err("stale vertex key should be rejected");

        assert_matches!(
            err,
            DeleteVertexError::VertexNotFound { vertex_key: missing } if missing == vertex_key
        );
        assert_eq!(dt.number_of_vertices(), vertex_count_before_stale_attempt);
        assert_eq!(dt.number_of_simplices(), simplex_count_before_stale_attempt);
        assert!(dt.as_triangulation().validate().is_ok());
    }

    #[test]
    fn delete_vertex_without_repair_rolls_back_on_delaunay_violation() {
        init_tracing();
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
            vertex![1.0, 1.0].unwrap(),
            vertex![0.18, 0.42].unwrap(),
            vertex![0.52, 0.18].unwrap(),
            vertex![0.64, 0.72].unwrap(),
        ];
        let deleted_uuid = vertices[4].uuid();
        let mut dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        dt.is_valid().unwrap();
        dt.set_topology_guarantee(TopologyGuarantee::PLManifold);
        dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

        let vertex_key = dt
            .vertices()
            .find_map(|(key, vertex)| (vertex.uuid() == deleted_uuid).then_some(key))
            .expect("fixture vertex should be present");
        let vertex_count_before = dt.number_of_vertices();
        let simplex_count_before = dt.number_of_simplices();
        let hint_simplex_before = dt.simplices().next().map(|(key, _)| key);
        dt.insertion_state.last_inserted_simplex = hint_simplex_before;
        let mut spatial_index = HashGridIndex::<2>::try_new(1.0).unwrap();
        for (vertex_key, vertex) in dt.vertices() {
            spatial_index.insert_vertex(vertex_key, vertex.point().coords());
        }
        dt.spatial_index = Some(spatial_index);
        let spatial_index_before = dt
            .spatial_index
            .as_ref()
            .map(HashGridIndex::<2>::debug_snapshot);

        let err = dt
            .delete_vertex(vertex_key)
            .expect_err("disabled repair should roll back a Level 4 violation");

        assert_matches!(
            err,
            DeleteVertexError::InvariantViolation {
                source: InvariantError::Delaunay(
                    DelaunayTriangulationValidationError::VerificationFailed { .. },
                ),
            }
        );
        assert_eq!(dt.number_of_vertices(), vertex_count_before);
        assert_eq!(dt.number_of_simplices(), simplex_count_before);
        assert_eq!(
            dt.insertion_state.last_inserted_simplex,
            hint_simplex_before
        );
        assert_eq!(
            dt.spatial_index
                .as_ref()
                .map(HashGridIndex::<2>::debug_snapshot),
            spatial_index_before
        );
        assert!(dt.vertices().any(|(_, v)| v.uuid() == deleted_uuid));
        dt.is_valid().unwrap();
    }
}

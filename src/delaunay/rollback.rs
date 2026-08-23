//! Delaunay-level rollback guards for mutations with owner-coupled caches.

#![forbid(unsafe_code)]

use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::operations::DelaunayInsertionState;
use crate::core::tds::{Tds, TdsOwnerRollbackTransaction, TdsRollbackOwner, TdsRollbackWindow};
use crate::delaunay_model::{DelaunayTriangulation, EuclideanDelaunayReportDomain};
use crate::triangulation::validation::TopologyConstructionProvenance;
use crate::triangulation::{Triangulation, rollback::TriangulationRollbackWindow};

impl<K, U, V, const D: usize> TdsRollbackOwner<U, V, D> for DelaunayTriangulation<K, U, V, D> {
    fn rollback_tds(&self) -> &Tds<U, V, D> {
        &self.tri.tds
    }

    fn rollback_tds_mut(&mut self) -> &mut Tds<U, V, D> {
        &mut self.tri.tds
    }
}

/// Spatial-index policy for a Delaunay-level rollback transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DelaunaySpatialIndexRollback {
    /// Restore the pre-transaction spatial index exactly.
    Restore,
    /// Drop the index on rollback so it can be rebuilt lazily.
    Invalidate,
}

/// Scoped rollback guard for Delaunay-level mutations that must restore TDS
/// state together with owner-coupled insertion and cache state.
#[must_use = "rollback transactions restore on drop unless explicitly committed or rolled back"]
pub struct DelaunayRollbackTransaction<'dt, K, U, V, const D: usize>
where
    U: Clone,
    V: Clone,
{
    tds_transaction: TdsOwnerRollbackTransaction<'dt, DelaunayTriangulation<K, U, V, D>, U, V, D>,
    insertion_state_snapshot: DelaunayInsertionState,
    spatial_index_snapshot: Option<HashGridIndex<D>>,
    euclidean_report_domain_snapshot: EuclideanDelaunayReportDomain,
    topology_construction_provenance_snapshot: TopologyConstructionProvenance,
    spatial_index_rollback: DelaunaySpatialIndexRollback,
    finished: bool,
}

impl<'dt, K, U, V, const D: usize> DelaunayRollbackTransaction<'dt, K, U, V, D>
where
    U: Clone,
    V: Clone,
{
    /// Begins a Delaunay-level rollback window.
    pub(super) fn begin(
        owner: &'dt mut DelaunayTriangulation<K, U, V, D>,
        spatial_index_rollback: DelaunaySpatialIndexRollback,
    ) -> Self {
        let insertion_state_snapshot = owner.insertion_state;
        let euclidean_report_domain_snapshot = owner.euclidean_report_domain;
        let topology_construction_provenance_snapshot = owner.tri.topology_construction_provenance;
        let spatial_index_snapshot = match spatial_index_rollback {
            DelaunaySpatialIndexRollback::Restore => owner.spatial_index.clone(),
            DelaunaySpatialIndexRollback::Invalidate => None,
        };
        let tds_transaction = TdsOwnerRollbackTransaction::begin(owner);
        Self {
            tds_transaction,
            insertion_state_snapshot,
            spatial_index_snapshot,
            euclidean_report_domain_snapshot,
            topology_construction_provenance_snapshot,
            spatial_index_rollback,
            finished: false,
        }
    }

    /// Borrows the mutable owner for a mutation step inside the transaction.
    pub(super) const fn delaunay_mut(&mut self) -> &mut DelaunayTriangulation<K, U, V, D> {
        self.tds_transaction.owner_mut()
    }

    /// Restores all owner-coupled rollback state and keeps the transaction open.
    pub(super) fn restore(&mut self) {
        self.tds_transaction.restore();
        let owner = self.tds_transaction.owner_mut();
        owner.insertion_state = self.insertion_state_snapshot;
        owner.euclidean_report_domain = self.euclidean_report_domain_snapshot;
        owner.tri.topology_construction_provenance = self.topology_construction_provenance_snapshot;
        owner.spatial_index = match self.spatial_index_rollback {
            DelaunaySpatialIndexRollback::Restore => self.spatial_index_snapshot.clone(),
            DelaunaySpatialIndexRollback::Invalidate => None,
        };
    }

    /// Commits the mutation, preventing the drop guard from restoring the snapshot.
    pub(super) fn commit(mut self) {
        self.tds_transaction.commit_in_place();
        self.finished = true;
    }

    /// Restores the snapshot and closes the transaction.
    pub(super) fn rollback(mut self) {
        self.restore();
        self.tds_transaction.commit_in_place();
        self.finished = true;
    }
}

impl<K, U, V, const D: usize> Drop for DelaunayRollbackTransaction<'_, K, U, V, D>
where
    U: Clone,
    V: Clone,
{
    fn drop(&mut self) {
        if !self.finished {
            self.restore();
            self.tds_transaction.commit_in_place();
        }
    }
}

impl<K, U, V, const D: usize> TdsRollbackWindow<U, V, D>
    for DelaunayRollbackTransaction<'_, K, U, V, D>
where
    U: Clone,
    V: Clone,
{
    fn rollback_tds_mut(&mut self) -> &mut Tds<U, V, D> {
        &mut self.tds_transaction.owner_mut().tri.tds
    }

    fn restore_rollback_tds(&mut self) {
        self.restore();
    }
}

impl<K, U, V, const D: usize> TriangulationRollbackWindow<K, U, V, D>
    for DelaunayRollbackTransaction<'_, K, U, V, D>
where
    U: Clone,
    V: Clone,
{
    fn triangulation_mut(&mut self) -> &mut Triangulation<K, U, V, D> {
        &mut self.tds_transaction.owner_mut().tri
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::collections::spatial_hash_grid::test_support::HashGridIndexSnapshot;
    use crate::core::tds::VertexKey;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::vertex;

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

    fn test_triangulation<const D: usize>() -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>
    where
        AdaptiveKernel<f64>: crate::geometry::kernel::ExactPredicates<D>,
    {
        let vertices = simplex_vertices::<D>();
        DelaunayTriangulation::builder(&vertices).build().unwrap()
    }

    fn seed_spatial_index<const D: usize>(
        triangulation: &mut DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>,
    ) -> Option<HashGridIndexSnapshot> {
        let mut index = HashGridIndex::<D>::try_new(1.0).unwrap();
        for (vertex_key, vertex) in triangulation.vertices() {
            index.insert_vertex(vertex_key, vertex.point().coords());
        }
        triangulation.spatial_index = Some(index);
        triangulation
            .spatial_index
            .as_ref()
            .map(HashGridIndex::<D>::debug_snapshot)
    }

    fn insert_uncommitted_vertex<const D: usize>(
        transaction: &mut DelaunayRollbackTransaction<'_, AdaptiveKernel<f64>, (), (), D>,
    ) -> VertexKey {
        let vertex = vertex!([0.25; D]).unwrap();
        transaction
            .rollback_tds_mut()
            .insert_vertex_with_mapping(vertex)
            .unwrap()
    }

    macro_rules! assert_rollback_dimensions {
        ($case:ident) => {{
            $case::<2>();
            $case::<3>();
            $case::<4>();
            $case::<5>();
        }};
    }

    fn assert_restore_policy_drop_restores_auxiliary_state<const D: usize>()
    where
        AdaptiveKernel<f64>: crate::geometry::kernel::ExactPredicates<D>,
    {
        let mut triangulation = test_triangulation::<D>();
        let report_domain_before = triangulation.euclidean_report_domain;
        let provenance_before = triangulation.tri.topology_construction_provenance;
        let hint_before = triangulation.simplices().next().map(|(key, _)| key);
        triangulation.insertion_state.last_inserted_simplex = hint_before;
        let spatial_index_before = seed_spatial_index(&mut triangulation);
        let vertices_before = triangulation.number_of_vertices();
        let inserted_key;

        {
            let mut transaction = DelaunayRollbackTransaction::begin(
                &mut triangulation,
                DelaunaySpatialIndexRollback::Restore,
            );
            inserted_key = insert_uncommitted_vertex(&mut transaction);
            transaction
                .delaunay_mut()
                .insertion_state
                .last_inserted_simplex = None;
            transaction.delaunay_mut().spatial_index = None;
            transaction
                .delaunay_mut()
                .invalidate_euclidean_report_domain();
            transaction
                .delaunay_mut()
                .tri
                .topology_construction_provenance =
                crate::triangulation::validation::TopologyConstructionProvenance::Unproven;
        }

        assert_eq!(triangulation.number_of_vertices(), vertices_before);
        assert!(!triangulation.tri.tds.contains_vertex_key(inserted_key));
        assert_eq!(
            triangulation.insertion_state.last_inserted_simplex,
            hint_before
        );
        assert_eq!(
            triangulation
                .spatial_index
                .as_ref()
                .map(HashGridIndex::<D>::debug_snapshot),
            spatial_index_before
        );
        assert_eq!(triangulation.euclidean_report_domain, report_domain_before);
        assert_eq!(
            triangulation.tri.topology_construction_provenance,
            provenance_before
        );
    }

    fn assert_invalidate_policy_drop_drops_spatial_index<const D: usize>()
    where
        AdaptiveKernel<f64>: crate::geometry::kernel::ExactPredicates<D>,
    {
        let mut triangulation = test_triangulation::<D>();
        let hint_before = triangulation.simplices().next().map(|(key, _)| key);
        triangulation.insertion_state.last_inserted_simplex = hint_before;
        let _ = seed_spatial_index(&mut triangulation);
        let vertices_before = triangulation.number_of_vertices();
        let inserted_key;

        {
            let mut transaction = DelaunayRollbackTransaction::begin(
                &mut triangulation,
                DelaunaySpatialIndexRollback::Invalidate,
            );
            inserted_key = insert_uncommitted_vertex(&mut transaction);
            transaction
                .delaunay_mut()
                .insertion_state
                .last_inserted_simplex = None;
        }

        assert_eq!(triangulation.number_of_vertices(), vertices_before);
        assert!(!triangulation.tri.tds.contains_vertex_key(inserted_key));
        assert_eq!(
            triangulation.insertion_state.last_inserted_simplex,
            hint_before
        );
        assert!(triangulation.spatial_index.is_none());
    }

    fn assert_commit_keeps_mutations<const D: usize>()
    where
        AdaptiveKernel<f64>: crate::geometry::kernel::ExactPredicates<D>,
    {
        let mut triangulation = test_triangulation::<D>();
        let hint_before = triangulation.simplices().next().map(|(key, _)| key);
        triangulation.insertion_state.last_inserted_simplex = hint_before;
        let _ = seed_spatial_index(&mut triangulation);
        let vertices_before = triangulation.number_of_vertices();

        let mut transaction = DelaunayRollbackTransaction::begin(
            &mut triangulation,
            DelaunaySpatialIndexRollback::Restore,
        );
        let inserted_key = insert_uncommitted_vertex(&mut transaction);
        transaction
            .delaunay_mut()
            .insertion_state
            .last_inserted_simplex = None;
        transaction.delaunay_mut().spatial_index = None;
        transaction.commit();

        assert_eq!(triangulation.number_of_vertices(), vertices_before + 1);
        assert!(triangulation.tri.tds.contains_vertex_key(inserted_key));
        assert!(
            triangulation
                .insertion_state
                .last_inserted_simplex
                .is_none()
        );
        assert!(triangulation.spatial_index.is_none());
    }

    fn assert_explicit_rollback_restores_auxiliary_state<const D: usize>()
    where
        AdaptiveKernel<f64>: crate::geometry::kernel::ExactPredicates<D>,
    {
        let mut triangulation = test_triangulation::<D>();
        let report_domain_before = triangulation.euclidean_report_domain;
        let hint_before = triangulation.simplices().next().map(|(key, _)| key);
        triangulation.insertion_state.last_inserted_simplex = hint_before;
        let spatial_index_before = seed_spatial_index(&mut triangulation);
        let vertices_before = triangulation.number_of_vertices();

        let mut transaction = DelaunayRollbackTransaction::begin(
            &mut triangulation,
            DelaunaySpatialIndexRollback::Restore,
        );
        let inserted_key = insert_uncommitted_vertex(&mut transaction);
        transaction
            .delaunay_mut()
            .insertion_state
            .last_inserted_simplex = None;
        transaction.delaunay_mut().spatial_index = None;
        transaction
            .delaunay_mut()
            .invalidate_euclidean_report_domain();
        transaction.rollback();

        assert_eq!(triangulation.number_of_vertices(), vertices_before);
        assert!(!triangulation.tri.tds.contains_vertex_key(inserted_key));
        assert_eq!(
            triangulation.insertion_state.last_inserted_simplex,
            hint_before
        );
        assert_eq!(
            triangulation
                .spatial_index
                .as_ref()
                .map(HashGridIndex::<D>::debug_snapshot),
            spatial_index_before
        );
        assert_eq!(triangulation.euclidean_report_domain, report_domain_before);
    }

    #[test]
    fn delaunay_transaction_restore_policy_restores_auxiliary_state() {
        assert_rollback_dimensions!(assert_restore_policy_drop_restores_auxiliary_state);
    }

    #[test]
    fn delaunay_transaction_invalidate_policy_drops_spatial_index_on_restore() {
        assert_rollback_dimensions!(assert_invalidate_policy_drop_drops_spatial_index);
    }

    #[test]
    fn delaunay_transaction_commit_keeps_mutations() {
        assert_rollback_dimensions!(assert_commit_keeps_mutations);
    }

    #[test]
    fn delaunay_transaction_explicit_rollback_restores_auxiliary_state() {
        assert_rollback_dimensions!(assert_explicit_rollback_restores_auxiliary_state);
    }
}

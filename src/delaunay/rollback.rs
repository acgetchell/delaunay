//! Delaunay-level rollback guards for mutations with owner-coupled caches.

#![forbid(unsafe_code)]

use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::operations::DelaunayInsertionState;
use crate::core::tds::TdsRollbackSnapshot;
use crate::triangulation::DelaunayTriangulation;

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
    owner: &'dt mut DelaunayTriangulation<K, U, V, D>,
    tds_snapshot: TdsRollbackSnapshot<U, V, D>,
    insertion_state_snapshot: DelaunayInsertionState,
    spatial_index_snapshot: Option<HashGridIndex<D>>,
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
        let tds_snapshot = TdsRollbackSnapshot::capture(&owner.tri.tds);
        let insertion_state_snapshot = owner.insertion_state;
        let spatial_index_snapshot = match spatial_index_rollback {
            DelaunaySpatialIndexRollback::Restore => owner.spatial_index.clone(),
            DelaunaySpatialIndexRollback::Invalidate => None,
        };
        Self {
            owner,
            tds_snapshot,
            insertion_state_snapshot,
            spatial_index_snapshot,
            spatial_index_rollback,
            finished: false,
        }
    }

    /// Borrows the mutable owner for a mutation step inside the transaction.
    pub(super) const fn delaunay_mut(&mut self) -> &mut DelaunayTriangulation<K, U, V, D> {
        &mut *self.owner
    }

    /// Restores all owner-coupled rollback state and keeps the transaction open.
    pub(super) fn restore(&mut self) {
        self.tds_snapshot.restore_to(&mut self.owner.tri.tds);
        self.owner.insertion_state = self.insertion_state_snapshot;
        self.owner.spatial_index = match self.spatial_index_rollback {
            DelaunaySpatialIndexRollback::Restore => self.spatial_index_snapshot.clone(),
            DelaunaySpatialIndexRollback::Invalidate => None,
        };
    }

    /// Commits the mutation, preventing the drop guard from restoring the snapshot.
    pub(super) fn commit(mut self) {
        self.finished = true;
    }

    /// Restores the snapshot and closes the transaction.
    pub(super) fn rollback(mut self) {
        self.restore();
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::collections::spatial_hash_grid::HashGridIndexSnapshot;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::vertex;

    fn test_triangulation() -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        DelaunayTriangulation::try_new(&vertices).unwrap()
    }

    fn seed_spatial_index(
        triangulation: &mut DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2>,
    ) -> Option<HashGridIndexSnapshot> {
        let mut index = HashGridIndex::<2>::try_new(1.0).unwrap();
        for (vertex_key, vertex) in triangulation.vertices() {
            index.insert_vertex(vertex_key, vertex.point().coords());
        }
        triangulation.spatial_index = Some(index);
        triangulation
            .spatial_index
            .as_ref()
            .map(HashGridIndex::<2>::debug_snapshot)
    }

    fn insert_uncommitted_vertex(
        transaction: &mut DelaunayRollbackTransaction<'_, AdaptiveKernel<f64>, (), (), 2>,
    ) {
        let vertex = Vertex::<(), 2>::try_new([0.25, 0.25]).unwrap();
        transaction
            .delaunay_mut()
            .tri
            .tds
            .insert_vertex_with_mapping(vertex)
            .unwrap();
    }

    #[test]
    fn delaunay_transaction_restore_policy_restores_auxiliary_state() {
        let mut triangulation = test_triangulation();
        let hint_before = triangulation.simplices().next().map(|(key, _)| key);
        triangulation.insertion_state.last_inserted_simplex = hint_before;
        let spatial_index_before = seed_spatial_index(&mut triangulation);
        let vertices_before = triangulation.number_of_vertices();

        {
            let mut transaction = DelaunayRollbackTransaction::begin(
                &mut triangulation,
                DelaunaySpatialIndexRollback::Restore,
            );
            insert_uncommitted_vertex(&mut transaction);
            transaction
                .delaunay_mut()
                .insertion_state
                .last_inserted_simplex = None;
            transaction.delaunay_mut().spatial_index = None;
        }

        assert_eq!(triangulation.number_of_vertices(), vertices_before);
        assert_eq!(
            triangulation.insertion_state.last_inserted_simplex,
            hint_before
        );
        assert_eq!(
            triangulation
                .spatial_index
                .as_ref()
                .map(HashGridIndex::<2>::debug_snapshot),
            spatial_index_before
        );
    }

    #[test]
    fn delaunay_transaction_invalidate_policy_drops_spatial_index_on_restore() {
        let mut triangulation = test_triangulation();
        let hint_before = triangulation.simplices().next().map(|(key, _)| key);
        triangulation.insertion_state.last_inserted_simplex = hint_before;
        let _ = seed_spatial_index(&mut triangulation);
        let vertices_before = triangulation.number_of_vertices();

        {
            let mut transaction = DelaunayRollbackTransaction::begin(
                &mut triangulation,
                DelaunaySpatialIndexRollback::Invalidate,
            );
            insert_uncommitted_vertex(&mut transaction);
            transaction
                .delaunay_mut()
                .insertion_state
                .last_inserted_simplex = None;
        }

        assert_eq!(triangulation.number_of_vertices(), vertices_before);
        assert_eq!(
            triangulation.insertion_state.last_inserted_simplex,
            hint_before
        );
        assert!(triangulation.spatial_index.is_none());
    }
}

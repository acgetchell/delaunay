//! Triangulation-level rollback guards for scoped topology mutation windows.

#![forbid(unsafe_code)]

use crate::core::tds::TdsRollbackSnapshot;
use crate::core::triangulation::Triangulation;

/// Scoped rollback guard for a `Triangulation` mutation that snapshots only
/// the owned TDS while allowing method-level mutation through the owner.
#[must_use = "rollback transactions restore on drop unless explicitly committed or rolled back"]
pub(crate) struct TriangulationRollbackTransaction<'tri, K, U, V, const D: usize>
where
    U: Clone,
    V: Clone,
{
    owner: &'tri mut Triangulation<K, U, V, D>,
    tds_snapshot: TdsRollbackSnapshot<U, V, D>,
    finished: bool,
}

impl<'tri, K, U, V, const D: usize> TriangulationRollbackTransaction<'tri, K, U, V, D>
where
    U: Clone,
    V: Clone,
{
    /// Begins a rollback window by snapshotting the canonical TDS owner.
    pub(crate) fn begin(owner: &'tri mut Triangulation<K, U, V, D>) -> Self {
        let tds_snapshot = TdsRollbackSnapshot::capture(&owner.tds);
        Self {
            owner,
            tds_snapshot,
            finished: false,
        }
    }

    /// Borrows the mutable owner for a mutation step inside the transaction.
    pub(crate) const fn triangulation_mut(&mut self) -> &mut Triangulation<K, U, V, D> {
        &mut *self.owner
    }

    /// Restores the owner TDS to the saved state while keeping the transaction
    /// open for another attempt.
    pub(crate) fn restore(&mut self) {
        self.tds_snapshot.restore_to(&mut self.owner.tds);
    }

    /// Commits the mutation, preventing the drop guard from restoring the snapshot.
    pub(crate) fn commit(mut self) {
        self.finished = true;
    }

    /// Restores the snapshot and closes the transaction.
    pub(crate) fn rollback(mut self) {
        self.restore();
        self.finished = true;
    }
}

impl<K, U, V, const D: usize> Drop for TriangulationRollbackTransaction<'_, K, U, V, D>
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
    use crate::core::tds::Tds;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::FastKernel;
    use std::sync::Arc;

    fn insert_test_vertex<const D: usize>(
        triangulation: &mut Triangulation<FastKernel<f64>, (), (), D>,
        coordinate: f64,
    ) {
        let vertex = Vertex::<(), D>::try_new([coordinate; D]).unwrap();
        triangulation
            .tds
            .insert_vertex_with_mapping(vertex)
            .unwrap();
    }

    #[test]
    fn triangulation_transaction_drop_restores_tds() {
        let mut triangulation: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());

        {
            let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);
            insert_test_vertex(transaction.triangulation_mut(), 1.0);
        }

        assert_eq!(triangulation.tds.number_of_vertices(), 0);
    }

    #[test]
    fn triangulation_transaction_restore_keeps_window_open() {
        let mut triangulation: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());
        let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);

        insert_test_vertex(transaction.triangulation_mut(), 1.0);
        transaction.restore();
        insert_test_vertex(transaction.triangulation_mut(), 2.0);
        transaction.commit();

        assert_eq!(triangulation.tds.number_of_vertices(), 1);
    }

    #[test]
    fn triangulation_transaction_restore_allows_tds_field_replacement() {
        let mut triangulation: Triangulation<FastKernel<f64>, (), (), 2> =
            Triangulation::new_empty(FastKernel::new());
        insert_test_vertex(&mut triangulation, 1.0);
        let original_identity = Arc::clone(triangulation.tds.identity());

        let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);
        {
            let owner = transaction.triangulation_mut();
            owner.tds = Tds::empty();
            assert_eq!(owner.tds.number_of_vertices(), 0);
            assert!(!Arc::ptr_eq(&original_identity, owner.tds.identity()));
        }

        transaction.restore();
        {
            let owner = transaction.triangulation_mut();
            assert_eq!(owner.tds.number_of_vertices(), 1);
            assert!(Arc::ptr_eq(&original_identity, owner.tds.identity()));
        }
        transaction.commit();
    }
}

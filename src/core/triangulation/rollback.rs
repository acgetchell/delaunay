//! Triangulation-level rollback guards for scoped topology mutation windows.

#![forbid(unsafe_code)]

use crate::core::tds::{Tds, TdsOwnerRollbackTransaction, TdsRollbackOwner};
use crate::core::triangulation::Triangulation;

impl<K, U, V, const D: usize> TdsRollbackOwner<U, V, D> for Triangulation<K, U, V, D> {
    fn rollback_tds(&self) -> &Tds<U, V, D> {
        &self.tds
    }

    fn rollback_tds_mut(&mut self) -> &mut Tds<U, V, D> {
        &mut self.tds
    }
}

/// Scoped rollback guard for a `Triangulation` mutation that snapshots only
/// the owned TDS while allowing method-level mutation through the owner.
#[must_use = "rollback transactions restore on drop unless explicitly committed or rolled back"]
pub(crate) struct TriangulationRollbackTransaction<'tri, K, U, V, const D: usize>
where
    U: Clone,
    V: Clone,
{
    inner: TdsOwnerRollbackTransaction<'tri, Triangulation<K, U, V, D>, U, V, D>,
}

impl<'tri, K, U, V, const D: usize> TriangulationRollbackTransaction<'tri, K, U, V, D>
where
    U: Clone,
    V: Clone,
{
    /// Begins a rollback window by snapshotting the canonical TDS owner.
    pub(crate) fn begin(owner: &'tri mut Triangulation<K, U, V, D>) -> Self {
        Self {
            inner: TdsOwnerRollbackTransaction::begin(owner),
        }
    }

    /// Borrows the mutable owner for a mutation step inside the transaction.
    pub(crate) const fn triangulation_mut(&mut self) -> &mut Triangulation<K, U, V, D> {
        self.inner.owner_mut()
    }

    /// Restores the owner TDS to the saved state while keeping the transaction
    /// open for another attempt.
    pub(crate) fn restore(&mut self) {
        self.inner.restore();
    }

    /// Commits the mutation, preventing the drop guard from restoring the snapshot.
    pub(crate) fn commit(self) {
        self.inner.commit();
    }

    /// Restores the snapshot and closes the transaction.
    pub(crate) fn rollback(self) {
        self.inner.rollback();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::Tds;
    use crate::geometry::kernel::FastKernel;
    use crate::vertex;
    use std::sync::Arc;

    fn insert_test_vertex<const D: usize>(
        triangulation: &mut Triangulation<FastKernel<f64>, (), (), D>,
        coordinate: f64,
    ) {
        let vertex = vertex!([coordinate; D]).unwrap();
        triangulation
            .tds
            .insert_vertex_with_mapping(vertex)
            .unwrap();
    }

    macro_rules! assert_rollback_dimensions {
        ($case:ident) => {{
            $case::<2>();
            $case::<3>();
            $case::<4>();
            $case::<5>();
        }};
    }

    fn assert_drop_restores_tds<const D: usize>() {
        let mut triangulation: Triangulation<FastKernel<f64>, (), (), D> =
            Triangulation::new_empty(FastKernel::new());

        {
            let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);
            insert_test_vertex(transaction.triangulation_mut(), 1.0);
        }

        assert_eq!(triangulation.tds.number_of_vertices(), 0);
    }

    fn assert_restore_keeps_window_open<const D: usize>() {
        let mut triangulation: Triangulation<FastKernel<f64>, (), (), D> =
            Triangulation::new_empty(FastKernel::new());
        let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);

        insert_test_vertex(transaction.triangulation_mut(), 1.0);
        transaction.restore();
        insert_test_vertex(transaction.triangulation_mut(), 2.0);
        transaction.commit();

        assert_eq!(triangulation.tds.number_of_vertices(), 1);
    }

    fn assert_restore_allows_tds_field_replacement<const D: usize>() {
        let mut triangulation: Triangulation<FastKernel<f64>, (), (), D> =
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

    #[test]
    fn triangulation_transaction_drop_restores_tds() {
        assert_rollback_dimensions!(assert_drop_restores_tds);
    }

    #[test]
    fn triangulation_transaction_restore_keeps_window_open() {
        assert_rollback_dimensions!(assert_restore_keeps_window_open);
    }

    #[test]
    fn triangulation_transaction_restore_allows_tds_field_replacement() {
        assert_rollback_dimensions!(assert_restore_allows_tds_field_replacement);
    }
}

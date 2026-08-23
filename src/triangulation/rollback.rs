//! Triangulation-level rollback guards for scoped topology mutation windows.

#![forbid(unsafe_code)]

use crate::core::tds::{Tds, TdsOwnerRollbackTransaction, TdsRollbackOwner, TdsRollbackWindow};
use crate::triangulation::Triangulation;
use crate::triangulation::validation::TopologyConstructionProvenance;

impl<K, U, V, const D: usize> TdsRollbackOwner<U, V, D> for Triangulation<K, U, V, D> {
    fn rollback_tds(&self) -> &Tds<U, V, D> {
        &self.tds
    }

    fn rollback_tds_mut(&mut self) -> &mut Tds<U, V, D> {
        &mut self.tds
    }
}

/// Shared mutation surface for algorithms that need a triangulation owner
/// inside an owner-selected TDS rollback transaction.
///
/// Higher proof owners implement this trait so Levels 3–4 algorithms can reuse
/// the higher owner's rollback snapshot instead of nesting another full TDS
/// snapshot. The higher owner remains responsible for commit versus rollback
/// and for restoring any state coupled to the TDS.
#[expect(
    clippy::redundant_pub_crate,
    reason = "explicit crate visibility documents sharing with higher owner layers"
)]
pub(crate) trait TriangulationRollbackWindow<K, U, V, const D: usize>:
    TdsRollbackWindow<U, V, D>
{
    /// Borrows the Levels 3–4 owner for one mutation or validation step.
    fn triangulation_mut(&mut self) -> &mut Triangulation<K, U, V, D>;
}

/// Scoped rollback guard for a `Triangulation` mutation that snapshots only
/// the owned TDS while allowing method-level mutation through the owner.
#[must_use = "rollback transactions restore on drop unless explicitly committed or rolled back"]
pub struct TriangulationRollbackTransaction<'tri, K, U, V, const D: usize>
where
    U: Clone,
    V: Clone,
{
    inner: TdsOwnerRollbackTransaction<'tri, Triangulation<K, U, V, D>, U, V, D>,
    topology_construction_provenance_snapshot: TopologyConstructionProvenance,
}

impl<'tri, K, U, V, const D: usize> TriangulationRollbackTransaction<'tri, K, U, V, D>
where
    U: Clone,
    V: Clone,
{
    /// Begins a rollback window by snapshotting the canonical TDS owner.
    pub(crate) fn begin(owner: &'tri mut Triangulation<K, U, V, D>) -> Self {
        let topology_construction_provenance_snapshot = owner.topology_construction_provenance;
        Self {
            inner: TdsOwnerRollbackTransaction::begin(owner),
            topology_construction_provenance_snapshot,
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
        self.inner.owner_mut().topology_construction_provenance =
            self.topology_construction_provenance_snapshot;
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

impl<K, U, V, const D: usize> TdsRollbackWindow<U, V, D>
    for TriangulationRollbackTransaction<'_, K, U, V, D>
where
    U: Clone,
    V: Clone,
{
    fn rollback_tds_mut(&mut self) -> &mut Tds<U, V, D> {
        &mut self.inner.owner_mut().tds
    }

    fn restore_rollback_tds(&mut self) {
        self.restore();
    }
}

impl<K, U, V, const D: usize> TriangulationRollbackWindow<K, U, V, D>
    for TriangulationRollbackTransaction<'_, K, U, V, D>
where
    U: Clone,
    V: Clone,
{
    fn triangulation_mut(&mut self) -> &mut Triangulation<K, U, V, D> {
        self.inner.owner_mut()
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

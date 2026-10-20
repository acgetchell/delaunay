//! Triangulation-level rollback guards for scoped topology mutation windows.

#![forbid(unsafe_code)]

use crate::core::tds::{Tds, TdsOwnerRollbackTransaction, TdsRollbackOwner, TdsRollbackWindow};
use crate::topology::traits::topological_space::GlobalTopology;
use crate::triangulation::Triangulation;
use crate::triangulation::validation::{
    TopologyConstructionProvenance, TopologyGuarantee, ValidationPolicy,
};

impl<K, U, V, const D: usize> TdsRollbackOwner<U, V, D> for Triangulation<K, U, V, D> {
    fn rollback_tds_mut(&mut self) -> &mut Tds<U, V, D> {
        &mut self.tds
    }
}

/// Shared mutation surface for algorithms that need a triangulation owner
/// inside an owner-selected TDS rollback transaction.
///
/// Higher proof owners implement this trait so Levels 3–4 algorithms can reuse
/// the higher owner's rollback journal instead of opening a separate savepoint.
/// The higher owner remains responsible for commit versus rollback
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

/// Scoped rollback guard for a `Triangulation` mutation that journals the owned
/// TDS and snapshots topology configuration and provenance while allowing
/// method-level mutation through the owner.
#[must_use = "rollback transactions restore on drop unless explicitly committed or rolled back"]
pub struct TriangulationRollbackTransaction<'tri, K, U, V, const D: usize> {
    inner: TdsOwnerRollbackTransaction<'tri, Triangulation<K, U, V, D>, U, V, D>,
    topology_construction_provenance_snapshot: TopologyConstructionProvenance,
    topology_guarantee_snapshot: TopologyGuarantee,
    global_topology_snapshot: GlobalTopology<D>,
    validation_policy_snapshot: ValidationPolicy,
    finished: bool,
}

impl<'tri, K, U, V, const D: usize> TriangulationRollbackTransaction<'tri, K, U, V, D> {
    /// Begins a touched-record rollback window on the canonical TDS owner.
    pub(crate) fn begin(owner: &'tri mut Triangulation<K, U, V, D>) -> Self {
        let topology_construction_provenance_snapshot = owner.topology_construction_provenance;
        let topology_guarantee_snapshot = owner.topology_guarantee;
        let global_topology_snapshot = owner.global_topology;
        let validation_policy_snapshot = owner.validation_policy;
        Self {
            inner: TdsOwnerRollbackTransaction::begin(owner),
            topology_construction_provenance_snapshot,
            topology_guarantee_snapshot,
            global_topology_snapshot,
            validation_policy_snapshot,
            finished: false,
        }
    }

    /// Borrows the mutable owner for a mutation step inside the transaction.
    pub(crate) const fn triangulation_mut(&mut self) -> &mut Triangulation<K, U, V, D> {
        self.inner.owner_mut()
    }

    /// Restores the TDS, configuration, and provenance while keeping the
    /// transaction open for another attempt.
    pub(crate) fn restore(&mut self) {
        self.inner.restore();
        let owner = self.inner.owner_mut();
        owner.topology_construction_provenance = self.topology_construction_provenance_snapshot;
        owner.topology_guarantee = self.topology_guarantee_snapshot;
        owner.global_topology = self.global_topology_snapshot;
        owner.validation_policy = self.validation_policy_snapshot;
    }

    /// Commits the mutation, preventing the drop guard from restoring the snapshot.
    pub(crate) fn commit(mut self) {
        self.inner.commit_in_place();
        self.finished = true;
    }

    /// Restores the snapshot and closes the transaction.
    pub(crate) fn rollback(mut self) {
        self.restore();
        self.inner.commit_in_place();
        self.finished = true;
    }
}

impl<K, U, V, const D: usize> Drop for TriangulationRollbackTransaction<'_, K, U, V, D> {
    fn drop(&mut self) {
        if !self.finished {
            self.restore();
            self.inner.commit_in_place();
        }
    }
}

impl<K, U, V, const D: usize> TdsRollbackWindow<U, V, D>
    for TriangulationRollbackTransaction<'_, K, U, V, D>
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
{
    fn triangulation_mut(&mut self) -> &mut Triangulation<K, U, V, D> {
        self.inner.owner_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::rollback::test_support::NonClonePayload;
    use crate::geometry::kernel::FastKernel;
    use crate::vertex;
    use std::sync::Arc;

    type TestTriangulation<const D: usize> =
        Triangulation<FastKernel<f64>, NonClonePayload, NonClonePayload, D>;

    fn configuration<const D: usize>(
        owner: &TestTriangulation<D>,
    ) -> (
        TopologyGuarantee,
        GlobalTopology<D>,
        ValidationPolicy,
        TopologyConstructionProvenance,
    ) {
        (
            owner.topology_guarantee,
            owner.global_topology,
            owner.validation_policy,
            owner.topology_construction_provenance,
        )
    }

    fn change_configuration<const D: usize>(owner: &mut TestTriangulation<D>) {
        owner.topology_guarantee = TopologyGuarantee::Pseudomanifold;
        owner.global_topology = GlobalTopology::Spherical;
        owner.validation_policy = ValidationPolicy::Never;
        owner.topology_construction_provenance = TopologyConstructionProvenance::Unproven;
    }

    fn insert_test_vertex<const D: usize>(
        triangulation: &mut TestTriangulation<D>,
        coordinate: f64,
    ) {
        let vertex = vertex!([coordinate; D]; data = NonClonePayload(90)).unwrap();
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

    fn assert_drop_restores_owner_state<const D: usize>() {
        let mut triangulation: TestTriangulation<D> = Triangulation::new_empty(FastKernel::new());
        triangulation.topology_construction_provenance =
            TopologyConstructionProvenance::EuclideanDelaunayInsertion;
        let configuration_before = configuration(&triangulation);

        {
            let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);
            insert_test_vertex(transaction.triangulation_mut(), 1.0);
            change_configuration(transaction.triangulation_mut());
        }

        assert_eq!(triangulation.tds.number_of_vertices(), 0);
        assert_eq!(configuration(&triangulation), configuration_before);
    }

    fn assert_restore_keeps_window_open<const D: usize>() {
        let mut triangulation: TestTriangulation<D> = Triangulation::new_empty(FastKernel::new());
        // A nondefault policy proves restore recovers the saved configuration.
        triangulation.validation_policy = ValidationPolicy::Always;
        let configuration_before = configuration(&triangulation);
        let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);

        insert_test_vertex(transaction.triangulation_mut(), 1.0);
        change_configuration(transaction.triangulation_mut());
        transaction.restore();
        assert_eq!(
            configuration(transaction.triangulation_mut()),
            configuration_before
        );
        insert_test_vertex(transaction.triangulation_mut(), 2.0);
        change_configuration(transaction.triangulation_mut());
        let committed_configuration = configuration(transaction.triangulation_mut());
        transaction.commit();

        assert_eq!(triangulation.tds.number_of_vertices(), 1);
        assert_eq!(
            triangulation.tds.vertices().next().unwrap().1.data(),
            Some(&NonClonePayload(90))
        );
        assert_eq!(configuration(&triangulation), committed_configuration);
    }

    fn assert_restore_preserves_tds_owner_identity<const D: usize>() {
        let mut triangulation: TestTriangulation<D> = Triangulation::new_empty(FastKernel::new());
        insert_test_vertex(&mut triangulation, 1.0);
        let original_identity = Arc::clone(triangulation.tds.identity());

        let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);
        insert_test_vertex(transaction.triangulation_mut(), 2.0);

        transaction.restore();
        {
            let owner = transaction.triangulation_mut();
            assert_eq!(owner.tds.number_of_vertices(), 1);
            assert!(Arc::ptr_eq(&original_identity, owner.tds.identity()));
        }
        transaction.commit();
    }

    fn assert_rollback_restores_configuration_and_provenance<const D: usize>() {
        let mut triangulation: TestTriangulation<D> = Triangulation::new_empty(FastKernel::new());
        triangulation.topology_construction_provenance =
            TopologyConstructionProvenance::EuclideanDelaunayInsertion;
        let configuration_before = configuration(&triangulation);

        let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);
        change_configuration(transaction.triangulation_mut());
        transaction.rollback();

        assert_eq!(configuration(&triangulation), configuration_before);
    }

    fn assert_commit_keeps_configuration_and_provenance<const D: usize>() {
        let mut triangulation: TestTriangulation<D> = Triangulation::new_empty(FastKernel::new());
        triangulation.topology_construction_provenance =
            TopologyConstructionProvenance::EuclideanDelaunayInsertion;

        let mut transaction = TriangulationRollbackTransaction::begin(&mut triangulation);
        change_configuration(transaction.triangulation_mut());
        let committed_configuration = configuration(transaction.triangulation_mut());
        transaction.commit();

        assert_eq!(configuration(&triangulation), committed_configuration);
    }

    #[test]
    fn triangulation_transaction_drop_restores_owner_state() {
        assert_rollback_dimensions!(assert_drop_restores_owner_state);
    }

    #[test]
    fn triangulation_transaction_restore_keeps_window_open() {
        assert_rollback_dimensions!(assert_restore_keeps_window_open);
    }

    #[test]
    fn triangulation_transaction_restore_preserves_tds_owner_identity() {
        assert_rollback_dimensions!(assert_restore_preserves_tds_owner_identity);
    }

    #[test]
    fn triangulation_transaction_rollback_restores_configuration_and_provenance() {
        assert_rollback_dimensions!(assert_rollback_restores_configuration_and_provenance);
    }

    #[test]
    fn triangulation_transaction_commit_keeps_configuration_and_provenance() {
        assert_rollback_dimensions!(assert_commit_keeps_configuration_and_provenance);
    }
}

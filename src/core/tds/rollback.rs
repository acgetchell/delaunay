//! Proportional TDS rollback journals and owner-bound transactions.

#![forbid(unsafe_code)]

use crate::core::simplex::{Simplex, SimplexTopologySnapshot};
use crate::core::tds::errors::TriangulationConstructionState;
use crate::core::tds::incidence::VertexIncidenceSnapshot;
use crate::core::tds::{SimplexKey, Tds, VertexKey};
use crate::core::vertex::Vertex;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use uuid::Uuid;

/// Move-safe handle for a journal that crosses an internal proof transition.
///
/// Unlike a scoped [`TdsOwnerRollbackTransaction`], this token may travel beside
/// a TDS while validation consumes and rewraps that exact owner. Callers must
/// close it on every success and failure branch; owner identity and nesting
/// depth checks prevent applying it to another TDS or out of order.
#[derive(Clone, Debug)]
pub(crate) struct TdsRollbackSavepoint {
    owner_identity: Arc<Uuid>,
    journal_depth: usize,
}

/// Owner abstraction for rollback guards over a canonical [`Tds`].
pub(crate) trait TdsRollbackOwner<U, V, const D: usize> {
    /// Returns the canonical [`Tds`] that a rollback transaction mutates.
    fn rollback_tds_mut(&mut self) -> &mut Tds<U, V, D>;
}

/// Shared mutation surface for repair algorithms that participate in an
/// owner-selected TDS rollback transaction.
pub(crate) trait TdsRollbackWindow<U, V, const D: usize> {
    /// Borrows canonical TDS storage for one mutation or validation step.
    fn rollback_tds_mut(&mut self) -> &mut Tds<U, V, D>;

    /// Restores the transaction before-image while keeping the window open.
    fn restore_rollback_tds(&mut self);
}

impl<U, V, const D: usize> TdsRollbackOwner<U, V, D> for Tds<U, V, D> {
    fn rollback_tds_mut(&mut self) -> &mut Self {
        self
    }
}

/// Touched-record rollback journal for one TDS mutation window.
///
/// Removed values are held beside tombstoned storage slots so rollback can
/// restore their exact generational keys. Payload-independent topology fields
/// are copied lazily on first write. Thus ordinary successful insertion and
/// multi-flip repair scale with the mutation frontier rather than total TDS
/// storage.
#[derive(Debug)]
pub(in crate::core::tds) struct TdsRollbackJournal<U, V, const D: usize> {
    initial_generation: u64,
    initial_construction_state: TriangulationConstructionState,
    owner_identity: Arc<Uuid>,
    inserted_vertices: Vec<(VertexKey, Uuid)>,
    inserted_simplices: Vec<(SimplexKey, Uuid)>,
    removed_vertices: Vec<(VertexKey, Vertex<U, D>)>,
    removed_simplices: Vec<(SimplexKey, Simplex<V, D>)>,
    vertex_before_images: Vec<(VertexKey, Option<SimplexKey>)>,
    simplex_before_images: Vec<(SimplexKey, SimplexTopologySnapshot<D>)>,
    incidence_before_images: Vec<VertexIncidenceSnapshot>,
}

impl<U, V, const D: usize> TdsRollbackJournal<U, V, D> {
    fn new(tds: &Tds<U, V, D>) -> Self {
        Self {
            initial_generation: tds.generation(),
            initial_construction_state: tds.construction_state.clone(),
            owner_identity: Arc::clone(&tds.identity),
            inserted_vertices: Vec::new(),
            inserted_simplices: Vec::new(),
            removed_vertices: Vec::new(),
            removed_simplices: Vec::new(),
            vertex_before_images: Vec::new(),
            simplex_before_images: Vec::new(),
            incidence_before_images: Vec::new(),
        }
    }

    fn contains_inserted_vertex(&self, key: VertexKey) -> bool {
        self.inserted_vertices
            .iter()
            .any(|(candidate, _)| *candidate == key)
    }

    fn contains_inserted_simplex(&self, key: SimplexKey) -> bool {
        self.inserted_simplices
            .iter()
            .any(|(candidate, _)| *candidate == key)
    }
}

impl<U, V, const D: usize> Tds<U, V, D> {
    /// Opens an empty touched-record journal on this canonical owner.
    fn begin_rollback_journal(&mut self) {
        self.rollback_journals.push(TdsRollbackJournal::new(self));
    }

    /// Opens a move-safe journal for an internal consuming proof transition.
    pub(crate) fn begin_rollback_savepoint(&mut self) -> TdsRollbackSavepoint {
        self.begin_rollback_journal();
        TdsRollbackSavepoint {
            owner_identity: Arc::clone(&self.identity),
            journal_depth: self.rollback_journals.len(),
        }
    }

    /// Restores and closes a move-safe journal after a rejected proof transition.
    pub(crate) fn rollback_savepoint(&mut self, savepoint: TdsRollbackSavepoint) {
        self.assert_active_savepoint(&savepoint);
        drop(savepoint);
        self.rollback_active_journal();
    }

    /// Commits and closes a move-safe journal after a successful proof transition.
    pub(crate) fn commit_savepoint(&mut self, savepoint: TdsRollbackSavepoint) {
        self.assert_active_savepoint(&savepoint);
        drop(savepoint);
        self.commit_active_journal();
    }

    fn assert_active_savepoint(&self, savepoint: &TdsRollbackSavepoint) {
        assert!(
            Arc::ptr_eq(&self.identity, &savepoint.owner_identity),
            "rollback savepoint must retain the canonical TDS owner identity"
        );
        assert_eq!(
            self.rollback_journals.len(),
            savepoint.journal_depth,
            "rollback savepoints must close in nesting order"
        );
    }

    /// Records a successful vertex insertion in the active journal.
    pub(super) fn journal_inserted_vertex(&mut self, key: VertexKey, uuid: Uuid) {
        if let Some(journal) = self.rollback_journals.last_mut() {
            journal.inserted_vertices.push((key, uuid));
        }
    }

    /// Records a successful simplex insertion in the active journal.
    pub(super) fn journal_inserted_simplex(&mut self, key: SimplexKey, uuid: Uuid) {
        if let Some(journal) = self.rollback_journals.last_mut() {
            journal.inserted_simplices.push((key, uuid));
        }
    }

    /// Captures one vertex before its first write in the active journal.
    pub(super) fn journal_vertex_before_write(&mut self, key: VertexKey) {
        let Some(journal) = self.rollback_journals.last() else {
            return;
        };
        if journal.contains_inserted_vertex(key)
            || journal
                .vertex_before_images
                .iter()
                .any(|(candidate, _)| *candidate == key)
        {
            return;
        }
        let Some(before_image) = self.vertices.get(key).map(Vertex::incident_simplex) else {
            return;
        };
        self.rollback_journals
            .last_mut()
            .expect("journal presence was checked")
            .vertex_before_images
            .push((key, before_image));
    }

    /// Captures one simplex before its first write in the active journal.
    pub(super) fn journal_simplex_before_write(&mut self, key: SimplexKey) {
        let Some(journal) = self.rollback_journals.last() else {
            return;
        };
        if journal.contains_inserted_simplex(key)
            || journal
                .simplex_before_images
                .iter()
                .any(|(candidate, _)| *candidate == key)
        {
            return;
        }
        let Some(before_image) = self.simplices.get(key).map(Simplex::topology_snapshot) else {
            return;
        };
        self.rollback_journals
            .last_mut()
            .expect("journal presence was checked")
            .simplex_before_images
            .push((key, before_image));
    }

    /// Captures canonical incidence buffers before their first mutation.
    pub(super) fn journal_incidence_before_write(
        &mut self,
        vertex_keys: impl IntoIterator<Item = VertexKey>,
    ) {
        let Some(journal) = self.rollback_journals.last() else {
            return;
        };
        let mut new_snapshots = Vec::new();
        for vertex_key in vertex_keys {
            if journal
                .incidence_before_images
                .iter()
                .any(|(candidate, _)| *candidate == vertex_key)
                || new_snapshots
                    .iter()
                    .any(|(candidate, _): &VertexIncidenceSnapshot| *candidate == vertex_key)
            {
                continue;
            }
            new_snapshots.push(self.vertex_to_simplices.snapshot_entry(vertex_key));
        }
        self.rollback_journals
            .last_mut()
            .expect("journal presence was checked")
            .incidence_before_images
            .extend(new_snapshots);
    }

    /// Removes one simplex while preserving its exact key for rollback.
    pub(super) fn remove_simplex_storage_transactionally(
        &mut self,
        key: SimplexKey,
    ) -> Option<Uuid> {
        let Some(journal) = self.rollback_journals.last() else {
            return self.simplices.remove(key).map(|simplex| simplex.uuid());
        };
        if journal.contains_inserted_simplex(key) {
            return self.simplices.remove(key).map(|simplex| simplex.uuid());
        }
        let removed = self.simplices.tombstone(key)?;
        let uuid = removed.uuid();
        self.rollback_journals
            .last_mut()
            .expect("journal presence was checked")
            .removed_simplices
            .push((key, removed));
        Some(uuid)
    }

    /// Removes one vertex while preserving its exact key for rollback.
    pub(super) fn remove_vertex_storage_transactionally(&mut self, key: VertexKey) -> Option<Uuid> {
        let Some(journal) = self.rollback_journals.last() else {
            return self.vertices.remove(key).map(|vertex| vertex.uuid());
        };
        if journal.contains_inserted_vertex(key) {
            return self.vertices.remove(key).map(|vertex| vertex.uuid());
        }
        let removed = self.vertices.tombstone(key)?;
        let uuid = removed.uuid();
        self.rollback_journals
            .last_mut()
            .expect("journal presence was checked")
            .removed_vertices
            .push((key, removed));
        Some(uuid)
    }

    /// Restores and closes the active journal.
    fn rollback_active_journal(&mut self) {
        let journal = self
            .rollback_journals
            .pop()
            .expect("rollback transaction lost its active TDS journal");
        assert!(
            Arc::ptr_eq(&self.identity, &journal.owner_identity),
            "rollback transaction must retain the canonical TDS owner identity"
        );

        for (key, uuid) in journal.inserted_simplices.iter().rev() {
            self.simplices.remove(*key);
            self.uuid_to_simplex_key.remove(uuid);
        }
        for (key, uuid) in journal.inserted_vertices.iter().rev() {
            self.vertices.remove(*key);
            self.uuid_to_vertex_key.remove(uuid);
        }

        for (key, simplex) in journal.removed_simplices {
            let uuid = simplex.uuid();
            let restored = self.simplices.restore_tombstone(key, simplex);
            assert!(
                restored.is_ok(),
                "private rollback journal must retain simplex tombstone {key:?}"
            );
            self.uuid_to_simplex_key.insert(uuid, key);
        }
        for (key, vertex) in journal.removed_vertices {
            let uuid = vertex.uuid();
            let restored = self.vertices.restore_tombstone(key, vertex);
            assert!(
                restored.is_ok(),
                "private rollback journal must retain vertex tombstone {key:?}"
            );
            self.uuid_to_vertex_key.insert(uuid, key);
        }

        for (key, before_image) in journal.simplex_before_images {
            let slot = self
                .simplices
                .get_mut(key)
                .expect("private rollback journal retains every simplex before-image target");
            slot.restore_topology(before_image);
        }
        for (key, before_image) in journal.vertex_before_images {
            let slot = self
                .vertices
                .get_mut(key)
                .expect("private rollback journal retains every vertex before-image target");
            slot.set_incident_simplex(before_image);
        }
        for snapshot in journal.incidence_before_images {
            self.vertex_to_simplices.restore_entry(snapshot);
        }

        self.construction_state = journal.initial_construction_state;
        self.generation
            .store(journal.initial_generation, Ordering::Relaxed);
    }

    /// Commits and closes the active journal.
    fn commit_active_journal(&mut self) {
        let mut journal = self
            .rollback_journals
            .pop()
            .expect("rollback transaction lost its active TDS journal");
        assert!(
            Arc::ptr_eq(&self.identity, &journal.owner_identity),
            "rollback transaction must retain the canonical TDS owner identity"
        );

        let Some(parent) = self.rollback_journals.last() else {
            for (key, _) in journal.removed_simplices {
                self.simplices.finalize_tombstone(key);
            }
            for (key, _) in journal.removed_vertices {
                self.vertices.finalize_tombstone(key);
            }
            return;
        };

        let removed_parent_inserted_simplices: Vec<_> = journal
            .removed_simplices
            .extract_if(.., |(key, _)| parent.contains_inserted_simplex(*key))
            .map(|(key, _)| key)
            .collect();
        let removed_parent_inserted_vertices: Vec<_> = journal
            .removed_vertices
            .extract_if(.., |(key, _)| parent.contains_inserted_vertex(*key))
            .map(|(key, _)| key)
            .collect();
        for key in removed_parent_inserted_simplices {
            self.simplices.finalize_tombstone(key);
        }
        for key in removed_parent_inserted_vertices {
            self.vertices.finalize_tombstone(key);
        }

        let parent = self
            .rollback_journals
            .last_mut()
            .expect("nested journal parent was checked");
        for inserted in journal.inserted_vertices {
            if !parent.contains_inserted_vertex(inserted.0) {
                parent.inserted_vertices.push(inserted);
            }
        }
        for inserted in journal.inserted_simplices {
            if !parent.contains_inserted_simplex(inserted.0) {
                parent.inserted_simplices.push(inserted);
            }
        }
        parent.removed_vertices.extend(journal.removed_vertices);
        parent.removed_simplices.extend(journal.removed_simplices);
        for before_image in journal.vertex_before_images {
            if !parent.contains_inserted_vertex(before_image.0)
                && !parent
                    .vertex_before_images
                    .iter()
                    .any(|(key, _)| *key == before_image.0)
            {
                parent.vertex_before_images.push(before_image);
            }
        }
        for before_image in journal.simplex_before_images {
            if !parent.contains_inserted_simplex(before_image.0)
                && !parent
                    .simplex_before_images
                    .iter()
                    .any(|(key, _)| *key == before_image.0)
            {
                parent.simplex_before_images.push(before_image);
            }
        }
        for before_image in journal.incidence_before_images {
            if !parent
                .incidence_before_images
                .iter()
                .any(|(key, _)| *key == before_image.0)
            {
                parent.incidence_before_images.push(before_image);
            }
        }
    }
}

/// Scoped rollback guard for a mutation that must either commit explicitly or
/// restore the original TDS state.
#[must_use = "rollback transactions restore on drop unless explicitly committed or rolled back"]
pub(crate) struct TdsOwnerRollbackTransaction<'owner, O, U, V, const D: usize>
where
    O: TdsRollbackOwner<U, V, D>,
    U: Clone,
    V: Clone,
{
    owner: &'owner mut O,
    finished: bool,
    _payload: std::marker::PhantomData<(U, V)>,
}

impl<'owner, O, U, V, const D: usize> TdsOwnerRollbackTransaction<'owner, O, U, V, D>
where
    O: TdsRollbackOwner<U, V, D>,
    U: Clone,
    V: Clone,
{
    /// Begins an empty touched-record rollback window.
    pub(crate) fn begin(owner: &'owner mut O) -> Self {
        owner.rollback_tds_mut().begin_rollback_journal();
        Self {
            owner,
            finished: false,
            _payload: std::marker::PhantomData,
        }
    }

    /// Borrows the mutable owner for a mutation step inside the transaction.
    pub(crate) const fn owner_mut(&mut self) -> &mut O {
        &mut *self.owner
    }

    /// Restores the owner while keeping the transaction open for another attempt.
    pub(crate) fn restore(&mut self) {
        self.owner.rollback_tds_mut().rollback_active_journal();
        self.owner.rollback_tds_mut().begin_rollback_journal();
    }

    /// Commits the mutation, preventing the drop guard from restoring it.
    pub(crate) fn commit(mut self) {
        self.owner.rollback_tds_mut().commit_active_journal();
        self.finished = true;
    }

    /// Restores the before-image and closes the transaction.
    pub(crate) fn rollback(mut self) {
        self.owner.rollback_tds_mut().rollback_active_journal();
        self.finished = true;
    }

    /// Commits for wrapper guards that own their own drop policy.
    pub(crate) fn commit_in_place(&mut self) {
        self.owner.rollback_tds_mut().commit_active_journal();
        self.finished = true;
    }
}

impl<O, U, V, const D: usize> TdsRollbackWindow<U, V, D>
    for TdsOwnerRollbackTransaction<'_, O, U, V, D>
where
    O: TdsRollbackOwner<U, V, D>,
    U: Clone,
    V: Clone,
{
    fn rollback_tds_mut(&mut self) -> &mut Tds<U, V, D> {
        self.owner.rollback_tds_mut()
    }

    fn restore_rollback_tds(&mut self) {
        self.restore();
    }
}

impl<O, U, V, const D: usize> Drop for TdsOwnerRollbackTransaction<'_, O, U, V, D>
where
    O: TdsRollbackOwner<U, V, D>,
    U: Clone,
    V: Clone,
{
    fn drop(&mut self) {
        if !self.finished {
            self.owner.rollback_tds_mut().rollback_active_journal();
        }
    }
}

/// Owner-bound rollback guard for functions that mutate a [`Tds`] directly.
pub(crate) type TdsRollbackTransaction<'tds, U, V, const D: usize> =
    TdsOwnerRollbackTransaction<'tds, Tds<U, V, D>, U, V, D>;

impl<U, V, const D: usize> TdsOwnerRollbackTransaction<'_, Tds<U, V, D>, U, V, D>
where
    U: Clone,
    V: Clone,
{
    /// Borrows the mutable TDS for a mutation step inside the transaction.
    pub(crate) const fn tds_mut(&mut self) -> &mut Tds<U, V, D> {
        self.owner_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertex;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    #[derive(Debug)]
    struct CloneTracked(Arc<AtomicUsize>);

    impl Clone for CloneTracked {
        fn clone(&self) -> Self {
            self.0.fetch_add(1, Ordering::Relaxed);
            Self(Arc::clone(&self.0))
        }
    }

    #[test]
    fn rollback_restores_exact_key_generation_and_owner_identity() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let identity = Arc::clone(tds.identity());
        let initial_generation = tds.generation();

        let inserted_key = {
            let mut transaction = TdsRollbackTransaction::begin(&mut tds);
            let key = transaction
                .tds_mut()
                .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
                .unwrap();
            transaction.rollback();
            key
        };

        assert!(tds.vertex(inserted_key).is_none());
        assert_eq!(tds.generation(), initial_generation);
        assert!(Arc::ptr_eq(&identity, tds.identity()));
    }

    fn assert_insertion_journal_does_not_clone_untouched_storage<const D: usize>() {
        let clone_count = Arc::new(AtomicUsize::new(0));
        let mut tds: Tds<CloneTracked, (), D> = Tds::empty();
        for coordinate in 0..128 {
            tds.insert_vertex_with_mapping(
                vertex![
                    [f64::from(coordinate); D];
                    data = CloneTracked(Arc::clone(&clone_count))
                ]
                .unwrap(),
            )
            .unwrap();
        }

        let mut transaction = TdsRollbackTransaction::begin(&mut tds);
        transaction
            .tds_mut()
            .insert_vertex_with_mapping(
                vertex![[256.0; D]; data = CloneTracked(Arc::clone(&clone_count))].unwrap(),
            )
            .unwrap();

        let topology_before_image_count = transaction
            .tds_mut()
            .rollback_journals
            .last()
            .map_or(0, |journal| {
                journal.vertex_before_images.len() + journal.simplex_before_images.len()
            });
        assert_eq!(topology_before_image_count, 0);
        assert_eq!(clone_count.load(Ordering::Relaxed), 0);
        transaction.rollback();
        assert_eq!(tds.number_of_vertices(), 128);
        assert_eq!(clone_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn insertion_journal_does_not_clone_untouched_storage_in_2d_through_5d() {
        assert_insertion_journal_does_not_clone_untouched_storage::<2>();
        assert_insertion_journal_does_not_clone_untouched_storage::<3>();
        assert_insertion_journal_does_not_clone_untouched_storage::<4>();
        assert_insertion_journal_does_not_clone_untouched_storage::<5>();
    }
}

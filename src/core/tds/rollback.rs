//! TDS rollback snapshots and transactions.

#![forbid(unsafe_code)]

use crate::core::tds::Tds;
use std::ptr;

/// Owner abstraction for rollback guards that snapshot an embedded canonical [`Tds`].
pub(crate) trait TdsRollbackOwner<U, V, const D: usize> {
    /// Returns the canonical [`Tds`] that a rollback transaction snapshots.
    fn rollback_tds(&self) -> &Tds<U, V, D>;

    /// Returns the canonical [`Tds`] that a rollback transaction restores.
    fn rollback_tds_mut(&mut self) -> &mut Tds<U, V, D>;
}

impl<U, V, const D: usize> TdsRollbackOwner<U, V, D> for Tds<U, V, D> {
    fn rollback_tds(&self) -> &Self {
        self
    }

    fn rollback_tds_mut(&mut self) -> &mut Self {
        self
    }
}

/// Owned TDS rollback snapshot kept private behind owner-bound guards.
struct TdsRollbackSnapshot<U, V, const D: usize> {
    owner: *const (),
    snapshot: Tds<U, V, D>,
}

impl<U, V, const D: usize> TdsRollbackSnapshot<U, V, D>
where
    U: Clone,
    V: Clone,
{
    /// Captures a rollback snapshot with rollback-preserving identity semantics.
    fn capture(tds: &Tds<U, V, D>) -> Self {
        Self {
            owner: ptr::from_ref(tds).cast::<()>(),
            snapshot: tds.clone_for_rollback(),
        }
    }

    /// Restores `tds` from this snapshot while preserving rollback identity.
    ///
    /// # Panics
    ///
    /// Panics if `tds` is not the canonical owner location this snapshot was
    /// captured from.
    fn restore_to(&self, tds: &mut Tds<U, V, D>) {
        let target_owner = ptr::from_ref(tds).cast::<()>();
        assert!(
            ptr::eq(self.owner, target_owner),
            "rollback snapshot must be restored to the TDS owner location it was captured from"
        );
        tds.clone_from_for_rollback(&self.snapshot);
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
    snapshot: TdsRollbackSnapshot<U, V, D>,
    finished: bool,
}

impl<'owner, O, U, V, const D: usize> TdsOwnerRollbackTransaction<'owner, O, U, V, D>
where
    O: TdsRollbackOwner<U, V, D>,
    U: Clone,
    V: Clone,
{
    /// Begins a rollback window by snapshotting the owner's canonical TDS while
    /// retaining runtime identity for cache provenance.
    pub(crate) fn begin(owner: &'owner mut O) -> Self {
        let snapshot = TdsRollbackSnapshot::capture(owner.rollback_tds());
        Self {
            owner,
            snapshot,
            finished: false,
        }
    }

    /// Borrows the mutable owner for a mutation step inside the transaction.
    pub(crate) const fn owner_mut(&mut self) -> &mut O {
        &mut *self.owner
    }

    /// Restores the owner to the saved state while keeping the transaction open
    /// for another attempt.
    pub(crate) fn restore(&mut self) {
        self.snapshot.restore_to(self.owner.rollback_tds_mut());
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

    /// Marks the transaction committed for wrapper guards that own their own drop policy.
    pub(crate) const fn commit_in_place(&mut self) {
        self.finished = true;
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
            self.restore();
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

    #[test]
    #[should_panic(
        expected = "rollback snapshot must be restored to the TDS owner location it was captured from"
    )]
    fn snapshot_restore_rejects_cross_owner_target() {
        let source: Tds<(), (), 2> = Tds::empty();
        let snapshot = TdsRollbackSnapshot::capture(&source);
        let mut target: Tds<(), (), 2> = Tds::empty();

        snapshot.restore_to(&mut target);
    }
}

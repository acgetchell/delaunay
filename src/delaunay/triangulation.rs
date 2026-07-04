//! Delaunay triangulation layer with incremental insertion.
//!
//! This layer adds Delaunay-specific operations on top of the generic
//! `Triangulation` struct, following CGAL's architecture.

#![forbid(unsafe_code)]

use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::operations::DelaunayInsertionState;
use crate::core::tds::{TopologyOwner, TopologyOwnerId};
use crate::core::triangulation::Triangulation;

/// Delaunay triangulation with incremental insertion support.
///
/// # Type Parameters
/// - `K`: Geometric kernel implementing predicates
/// - `U`: User data type for vertices
/// - `V`: User data type for simplices
/// - `D`: Dimension of the triangulation
///
/// # Delaunay Property Note
///
/// The triangulation satisfies **structural validity** (all TDS invariants) and
/// uses **flip-based repairs** to restore the local Delaunay property after insertion.
/// By default, k=2/k=3 bistellar flip queues run automatically after each successful
/// insertion (see [`DelaunayRepairPolicy`](crate::DelaunayRepairPolicy)).
///
/// For applications requiring explicit verification, you can still call
/// [`is_valid_delaunay`](Self::is_valid_delaunay) (Level 5) or [`validate`](Self::validate) (Levels 1–5).
/// If flip-based repair fails to converge, insertion returns an error and the
/// triangulation is left structurally valid but not guaranteed Delaunay.
///
/// See: [Issue #120 Investigation](https://github.com/acgetchell/delaunay/blob/main/docs/archive/issue_120_investigation.md)
///
/// # Implementation
///
/// Uses efficient incremental cavity-based insertion algorithm:
/// - ✅ Point location (facet walking) - [`locate`]
/// - ✅ Conflict region computation (local BFS) - [`find_conflict_region`]
/// - ✅ Cavity extraction and filling - [`extract_cavity_boundary`] plus internal cavity replacement
/// - ✅ Local neighbor wiring after cavity replacement
/// - ✅ Hull extension for outside points
/// - ✅ Flip-based Delaunay repair (k=2/k=3 bistellar flips)
///
/// [`locate`]: crate::algorithms::locate
/// [`find_conflict_region`]: crate::algorithms::find_conflict_region
/// [`extract_cavity_boundary`]: crate::algorithms::extract_cavity_boundary
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
///
/// # fn main() -> DelaunayResult<()> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// assert_eq!(dt.number_of_simplices(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct DelaunayTriangulation<K, U, V, const D: usize> {
    /// The underlying generic triangulation.
    pub(crate) tri: Triangulation<K, U, V, D>,
    /// Ephemeral insertion/repair state (hint caching + repair scheduling).
    pub(crate) insertion_state: DelaunayInsertionState,
    /// Optional spatial hash-grid index used to accelerate duplicate detection and locate-hint
    /// selection during incremental insertion.
    ///
    /// This is a performance-only cache and is not serialized; it may be rebuilt lazily.
    /// Query paths validate returned vertex keys against the live TDS, so the
    /// cache can survive transactional rollbacks even if they leave behind stale
    /// keys from an insertion that did not commit.
    pub(crate) spatial_index: Option<HashGridIndex<D>>,
}

impl<K, U, V, const D: usize> TopologyOwner for DelaunayTriangulation<K, U, V, D> {
    #[inline]
    fn topology_owner_id(&self) -> TopologyOwnerId {
        self.tri.topology_owner_id()
    }

    #[inline]
    fn topology_generation(&self) -> u64 {
        self.tri.topology_generation()
    }
}

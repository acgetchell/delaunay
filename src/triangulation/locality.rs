//! Locality helpers for triangulation construction and repair.
//!
//! These utilities sit at the boundary between spatial locality and topological
//! locality: callers may use Hilbert ordering or point-location hints to find a
//! nearby insertion site, then pass the concrete cell keys touched by the TDS
//! mutation here to build bounded repair frontiers.

#![forbid(unsafe_code)]

use crate::core::algorithms::locate::{ConflictError, find_conflict_region};
use crate::core::collections::{CellKeyBuffer, FastHashSet, fast_hash_set_with_capacity};
use crate::core::tds::{CellKey, Tds};
use crate::core::traits::data_type::DataType;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;

/// Local conflict-seed collection result for exterior insertion repair.
#[must_use]
pub struct LocalConflictSeedCells {
    /// Live cells that should seed local Delaunay repair.
    pub seed_cells: CellKeyBuffer,
    /// Number of cells returned by the local conflict-region search before any fallback seed.
    pub conflict_cells_found: usize,
}

/// Adds live, deduplicated candidate cells to a pending local repair frontier.
///
/// Returns the number of cells newly appended to `pending_seed_cells`.
pub fn accumulate_live_cell_seeds<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    candidate_seed_cells: &[CellKey],
    pending_seed_cells: &mut Vec<CellKey>,
    pending_seen: &mut FastHashSet<CellKey>,
) -> usize
where
    U: DataType,
    V: DataType,
{
    let mut added = 0usize;
    for &cell_key in candidate_seed_cells {
        if tds.contains_cell(cell_key) && pending_seen.insert(cell_key) {
            pending_seed_cells.push(cell_key);
            added = added.saturating_add(1);
        }
    }
    added
}

/// Adds live, deduplicated candidate cells to a compact repair seed buffer.
///
/// Returns the number of cells newly appended to `seed_cells`.
pub fn append_live_unique_cell_seeds<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    candidate_seed_cells: &[CellKey],
    seed_cells: &mut CellKeyBuffer,
) -> usize
where
    U: DataType,
    V: DataType,
{
    let mut seen: FastHashSet<CellKey> =
        fast_hash_set_with_capacity(seed_cells.len().saturating_add(candidate_seed_cells.len()));
    seen.extend(seed_cells.iter().copied());

    let mut added = 0usize;
    for &cell_key in candidate_seed_cells {
        if tds.contains_cell(cell_key) && seen.insert(cell_key) {
            seed_cells.push(cell_key);
            added = added.saturating_add(1);
        }
    }
    added
}

/// Retains only live, deduplicated cells in a pending local repair frontier.
pub fn retain_live_cell_seeds<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    seed_cells: &mut Vec<CellKey>,
    seen: &mut FastHashSet<CellKey>,
) where
    U: DataType,
    V: DataType,
{
    seen.clear();
    seed_cells.retain(|cell_key| tds.contains_cell(*cell_key) && seen.insert(*cell_key));
}

/// Clears a local repair frontier and its deduplication set together.
pub fn clear_cell_seed_set(seed_cells: &mut Vec<CellKey>, seen: &mut FastHashSet<CellKey>) {
    seed_cells.clear();
    seen.clear();
}

/// Retains conflict cells and records removed cells as local repair seeds.
pub fn retain_cells_and_record_removed(
    conflict_cells: &mut CellKeyBuffer,
    repair_seed_cells: &mut CellKeyBuffer,
    mut keep_cell: impl FnMut(CellKey) -> bool,
) {
    conflict_cells.retain(|cell_key| {
        let keep = keep_cell(*cell_key);
        if !keep {
            repair_seed_cells.push(*cell_key);
        }
        keep
    });
}

/// Replaces conflict cells and records cells missing from the replacement.
pub fn replace_cells_and_record_removed(
    conflict_cells: &mut CellKeyBuffer,
    repair_seed_cells: &mut CellKeyBuffer,
    replacement: CellKeyBuffer,
) {
    let replacement_set: FastHashSet<CellKey> = replacement.iter().copied().collect();
    for &cell_key in conflict_cells.iter() {
        if !replacement_set.contains(&cell_key) {
            repair_seed_cells.push(cell_key);
        }
    }
    *conflict_cells = replacement;
}

/// Collects local repair seeds for an exterior insertion from the terminal walk cell.
///
/// The terminal cell is adjacent to the hull facet crossed by point location, so a
/// BFS conflict search from it gives a bounded local frontier without scanning the
/// entire triangulation. If no circumsphere conflict is found, the terminal cell
/// itself is still a useful local seed.
pub fn collect_local_exterior_conflict_seed_cells<K, U, V, const D: usize>(
    tds: &Tds<K::Scalar, U, V, D>,
    kernel: &K,
    point: &Point<K::Scalar, D>,
    terminal_cell: CellKey,
) -> Result<LocalConflictSeedCells, ConflictError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    let mut seed_cells = CellKeyBuffer::new();
    if !tds.contains_cell(terminal_cell) {
        return Ok(LocalConflictSeedCells {
            seed_cells,
            conflict_cells_found: 0,
        });
    }

    let computed = find_conflict_region(tds, kernel, point, terminal_cell)?;
    let conflict_cells_found = computed.len();
    if computed.is_empty() {
        seed_cells.push(terminal_cell);
    } else {
        seed_cells = computed;
    }

    Ok(LocalConflictSeedCells {
        seed_cells,
        conflict_cells_found,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::triangulation::Triangulation;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::triangulation::delaunay::DelaunayTriangulation;
    use crate::vertex;
    use slotmap::KeyData;

    fn simplex_triangulation_3d() -> Triangulation<FastKernel<f64>, (), (), 3> {
        let vertices = [
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds)
    }

    #[test]
    fn accumulate_live_cell_seeds_dedupes_and_ignores_stale() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let all_cells: Vec<CellKey> = dt.cells().map(|(cell_key, _)| cell_key).collect();
        assert!(
            all_cells.len() >= 2,
            "fixture should produce multiple cells for seed accumulation"
        );

        let stale_cell = CellKey::from(KeyData::from_ffi(999_999));
        let mut pending_seed_cells = vec![all_cells[0]];
        let mut pending_seen: FastHashSet<CellKey> = pending_seed_cells.iter().copied().collect();
        let added = accumulate_live_cell_seeds(
            dt.tds(),
            &[all_cells[0], all_cells[1], all_cells[1], stale_cell],
            &mut pending_seed_cells,
            &mut pending_seen,
        );

        assert_eq!(added, 1);
        assert_eq!(pending_seed_cells, vec![all_cells[0], all_cells[1]]);
        assert!(!pending_seed_cells.contains(&stale_cell));

        let added_again = accumulate_live_cell_seeds(
            dt.tds(),
            &[all_cells[1]],
            &mut pending_seed_cells,
            &mut pending_seen,
        );
        assert_eq!(added_again, 0);
        assert_eq!(pending_seed_cells, vec![all_cells[0], all_cells[1]]);
    }

    #[test]
    fn append_live_unique_cell_seeds_dedupes_and_ignores_stale() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let all_cells: Vec<CellKey> = dt.cells().map(|(cell_key, _)| cell_key).collect();
        assert!(
            all_cells.len() >= 2,
            "fixture should produce multiple cells for compact seed accumulation"
        );

        let stale_cell = CellKey::from(KeyData::from_ffi(999_999));
        let mut seed_cells = CellKeyBuffer::new();
        seed_cells.push(all_cells[0]);
        let added = append_live_unique_cell_seeds(
            dt.tds(),
            &[all_cells[0], all_cells[1], stale_cell, all_cells[1]],
            &mut seed_cells,
        );

        assert_eq!(added, 1);
        assert_eq!(
            seed_cells.iter().copied().collect::<Vec<_>>(),
            vec![all_cells[0], all_cells[1],]
        );
    }

    #[test]
    fn retain_live_cell_seeds_filters_stale_and_dedupes() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([1.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::new(&vertices).unwrap();
        let all_cells: Vec<CellKey> = dt.cells().map(|(cell_key, _)| cell_key).collect();
        assert!(
            all_cells.len() >= 2,
            "fixture should produce multiple cells for seed retention"
        );

        let stale_cell = CellKey::from(KeyData::from_ffi(999_999));
        let mut seed_cells = vec![all_cells[0], stale_cell, all_cells[1], all_cells[0]];
        let mut seen = FastHashSet::default();
        retain_live_cell_seeds(dt.tds(), &mut seed_cells, &mut seen);

        assert_eq!(seed_cells, vec![all_cells[0], all_cells[1]]);
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn clear_cell_seed_set_clears_both_collections() {
        let stale_cell = CellKey::from(KeyData::from_ffi(999_999));
        let mut seed_cells = vec![stale_cell];
        let mut seen = FastHashSet::default();
        seen.insert(stale_cell);

        clear_cell_seed_set(&mut seed_cells, &mut seen);

        assert!(seed_cells.is_empty());
        assert!(seen.is_empty());
    }

    #[test]
    fn retain_and_replace_cells_record_removed_repair_seeds() {
        let a = CellKey::from(KeyData::from_ffi(31));
        let b = CellKey::from(KeyData::from_ffi(32));
        let c = CellKey::from(KeyData::from_ffi(33));
        let d = CellKey::from(KeyData::from_ffi(34));

        let mut conflict_cells: CellKeyBuffer = [a, b, c].into_iter().collect();
        let mut repair_seed_cells = CellKeyBuffer::new();
        retain_cells_and_record_removed(&mut conflict_cells, &mut repair_seed_cells, |ck| ck != b);
        assert_eq!(
            conflict_cells.iter().copied().collect::<Vec<_>>(),
            vec![a, c]
        );
        assert_eq!(
            repair_seed_cells.iter().copied().collect::<Vec<_>>(),
            vec![b]
        );

        let replacement: CellKeyBuffer = [c, d].into_iter().collect();
        replace_cells_and_record_removed(&mut conflict_cells, &mut repair_seed_cells, replacement);
        assert_eq!(
            conflict_cells.iter().copied().collect::<Vec<_>>(),
            vec![c, d]
        );
        assert_eq!(
            repair_seed_cells.iter().copied().collect::<Vec<_>>(),
            vec![b, a]
        );
    }

    #[test]
    fn collect_local_exterior_conflict_seed_cells_uses_terminal_seed_when_empty() {
        let tri = simplex_triangulation_3d();
        let terminal_cell = tri.tds.cell_keys().next().unwrap();
        let result = collect_local_exterior_conflict_seed_cells(
            &tri.tds,
            &FastKernel::new(),
            &Point::new([2.0, 2.0, 2.0]),
            terminal_cell,
        )
        .unwrap();

        assert_eq!(result.conflict_cells_found, 0);
        assert_eq!(
            result.seed_cells.iter().copied().collect::<Vec<_>>(),
            vec![terminal_cell]
        );
    }

    #[test]
    fn collect_local_exterior_conflict_seed_cells_returns_local_conflicts() {
        let tri = simplex_triangulation_3d();
        let terminal_cell = tri.tds.cell_keys().next().unwrap();
        let result = collect_local_exterior_conflict_seed_cells(
            &tri.tds,
            &FastKernel::new(),
            &Point::new([0.5, 0.5, 0.5]),
            terminal_cell,
        )
        .unwrap();

        assert_eq!(result.conflict_cells_found, 1);
        assert_eq!(
            result.seed_cells.iter().copied().collect::<Vec<_>>(),
            vec![terminal_cell]
        );
    }
}

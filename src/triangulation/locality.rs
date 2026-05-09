//! Locality helpers for triangulation construction and repair.
//!
//! These utilities sit at the boundary between spatial locality and topological
//! locality: callers may use Hilbert ordering or point-location hints to find a
//! nearby insertion site, then pass the concrete cell keys touched by the TDS
//! mutation here to build bounded repair frontiers.

#![forbid(unsafe_code)]

use crate::core::collections::FastHashSet;
use crate::core::tds::{CellKey, Tds};
use crate::core::traits::data_type::DataType;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangulation::delaunay::DelaunayTriangulation;
    use crate::vertex;
    use slotmap::KeyData;

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
}

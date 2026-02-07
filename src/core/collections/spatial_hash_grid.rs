//! Spatial hash-grid acceleration structures.
//!
//! This module provides a lightweight spatial index used to accelerate insertion.
//!
//! The primary use cases are:
//! - accelerate duplicate (near-duplicate) coordinate detection
//! - provide better starting hints for point location when no hint is available
//!
//! ## Rollback / transactional semantics
//!
//! The index can be used as ephemeral state (e.g. local to bulk insertion) so that
//! triangulation rollbacks do not need to snapshot/restore it.
//!
//! It can also be used as a persistent, performance-only cache for incremental
//! insertion. In that case, callers must ensure the index stays consistent with
//! the triangulation's vertex set (e.g. by snapshot/restore on outer rollbacks, or
//! by updating the index only after the full higher-level operation commits).

use super::{FastHashMap, SmallBuffer};
use crate::core::triangulation_data_structure::VertexKey;
use crate::geometry::traits::coordinate::CoordinateScalar;
use std::hash::{Hash, Hasher};

/// Maximum dimension supported by the hash-grid neighborhood walk.
///
/// The neighbor search enumerates the 3^D Moore neighborhood, which grows quickly.
const MAX_HASH_GRID_DIMENSION: usize = 5;

const BUCKET_INLINE_CAPACITY: usize = 8;

/// Hashable grid-cell key for a D-dimensional grid.
///
/// Internally, this stores integer-valued cell coordinates as the same scalar
/// type used for points. We avoid casting to an integer type to keep the index
/// generic over coordinate scalar types.
#[derive(Clone, Copy, Debug)]
struct GridKey<T, const D: usize>([T; D])
where
    T: CoordinateScalar;

impl<T, const D: usize> PartialEq for GridKey<T, D>
where
    T: CoordinateScalar,
{
    fn eq(&self, other: &Self) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| a.ordered_eq(b))
    }
}

impl<T, const D: usize> Eq for GridKey<T, D> where T: CoordinateScalar {}

impl<T, const D: usize> Hash for GridKey<T, D>
where
    T: CoordinateScalar,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for coord in &self.0 {
            coord.hash_scalar(state);
        }
    }
}

/// A simple spatial hash grid mapping grid cells to nearby keys.
///
/// The grid uses a fixed `cell_size` and indexes vertices by the floored cell
/// coordinates `floor(coord / cell_size)`.
#[derive(Clone, Debug)]
pub(in crate::core) struct HashGridIndex<T, const D: usize, K = VertexKey>
where
    T: CoordinateScalar,
{
    cell_size: T,
    usable: bool,
    cells: FastHashMap<GridKey<T, D>, SmallBuffer<K, BUCKET_INLINE_CAPACITY>>,
}

impl<T, const D: usize, K> HashGridIndex<T, D, K>
where
    T: CoordinateScalar,
    K: Copy,
{
    /// Create a new grid index with the given cell size.
    pub(in crate::core) fn new(cell_size: T) -> Self {
        let usable = D <= MAX_HASH_GRID_DIMENSION && cell_size.is_finite() && cell_size > T::zero();
        Self {
            cell_size,
            usable,
            cells: FastHashMap::default(),
        }
    }
    pub(in crate::core) const fn is_usable(&self) -> bool {
        self.usable
    }
    pub(in crate::core) const fn cell_size(&self) -> T
    where
        T: Copy,
    {
        self.cell_size
    }

    pub(in crate::core) fn clear(&mut self) {
        self.cells.clear();
    }

    const fn disable(&mut self) {
        self.usable = false;
    }
    pub(in crate::core) fn can_key_coords(&self, coords: &[T; D]) -> bool {
        self.key_for_coords(coords).is_some()
    }

    /// Insert a vertex into the appropriate grid cell.
    ///
    /// If the index is not usable, or the point cannot be keyed robustly, the
    /// index is disabled (so callers can fall back to linear scans).
    pub(in crate::core) fn insert_vertex(&mut self, vertex_key: K, coords: &[T; D]) {
        if !self.usable {
            return;
        }

        let Some(key) = self.key_for_coords(coords) else {
            self.disable();
            return;
        };

        self.cells.entry(key).or_default().push(vertex_key);
    }

    /// Visit all candidate vertex keys in the 3^D neighborhood around `coords`.
    ///
    /// Returns `true` if the index was used for the query (even if it yielded zero
    /// candidates). Returns `false` if the index was unusable for this query.
    pub(in crate::core) fn for_each_candidate_vertex_key<F>(
        &self,
        coords: &[T; D],
        mut f: F,
    ) -> bool
    where
        F: FnMut(K) -> bool,
    {
        if !self.usable {
            return false;
        }

        let Some(base_key) = self.key_for_coords(coords) else {
            return false;
        };

        let base = base_key.0;
        let mut current = base;

        Self::visit_neighbor_cells(0, &base, &mut current, &mut |neighbor| {
            if let Some(bucket) = self.cells.get(&neighbor) {
                for &vkey in bucket {
                    if !f(vkey) {
                        return false;
                    }
                }
            }
            true
        });

        true
    }

    fn key_for_coords(&self, coords: &[T; D]) -> Option<GridKey<T, D>> {
        if !self.usable {
            return None;
        }

        // Guard against non-finite or non-positive cell sizes.
        if !self.cell_size.is_finite() || self.cell_size <= T::zero() {
            return None;
        }

        let mut key = [T::zero(); D];
        let one = T::one();

        for (i, coord) in coords.iter().enumerate() {
            if !coord.is_finite() {
                return None;
            }

            let cell_coord = (*coord / self.cell_size).floor();
            if !cell_coord.is_finite() {
                return None;
            }

            // If the cell coordinate is too large to have unit resolution, neighbor
            // enumeration would be lossy (cell_coord + 1 == cell_coord). Disable.
            if (cell_coord + one).ordered_eq(&cell_coord) {
                return None;
            }

            key[i] = cell_coord;
        }

        Some(GridKey(key))
    }

    fn visit_neighbor_cells<F>(axis: usize, base: &[T; D], current: &mut [T; D], f: &mut F) -> bool
    where
        F: FnMut(GridKey<T, D>) -> bool,
    {
        if axis == D {
            return f(GridKey(*current));
        }

        let one = T::one();
        let offsets = [-one, T::zero(), one];

        for offset in offsets {
            current[axis] = base[axis] + offset;
            if !Self::visit_neighbor_cells(axis + 1, base, current, f) {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::collections::FastHashSet;
    use crate::core::triangulation_data_structure::VertexKey;
    use slotmap::SlotMap;

    #[test]
    fn test_hash_grid_index_keying_and_candidate_lookup_2d() {
        let mut slots: SlotMap<VertexKey, i32> = SlotMap::default();
        let v1 = slots.insert(1);
        let v2 = slots.insert(2);

        let mut grid: HashGridIndex<f64, 2> = HashGridIndex::new(1.0);

        grid.insert_vertex(v1, &[0.2, 0.2]);
        grid.insert_vertex(v2, &[-0.2, 0.2]);

        let mut found: FastHashSet<VertexKey> = FastHashSet::default();
        let used = grid.for_each_candidate_vertex_key(&[0.9, 0.1], |vkey| {
            found.insert(vkey);
            true
        });

        assert!(used);
        assert!(found.contains(&v1));
        assert!(found.contains(&v2));
    }

    #[test]
    fn test_hash_grid_index_candidate_visit_counts_2d() {
        let mut slots: SlotMap<VertexKey, i32> = SlotMap::default();
        let mut grid: HashGridIndex<f64, 2> = HashGridIndex::new(1.0);

        let mut expected: FastHashSet<VertexKey> = FastHashSet::default();

        for x in [-1.0, 0.0, 1.0] {
            for y in [-1.0, 0.0, 1.0] {
                let v = slots.insert(1);
                expected.insert(v);
                grid.insert_vertex(v, &[x + 0.25, y + 0.25]);
            }
        }

        let mut found: FastHashSet<VertexKey> = FastHashSet::default();
        let used = grid.for_each_candidate_vertex_key(&[0.25, 0.25], |vkey| {
            found.insert(vkey);
            true
        });

        assert!(used);
        assert_eq!(found.len(), expected.len());
        for v in expected {
            assert!(found.contains(&v));
        }
    }

    #[test]
    fn test_hash_grid_index_candidate_visit_counts_5d() {
        let mut slots: SlotMap<VertexKey, i32> = SlotMap::default();
        let mut grid: HashGridIndex<f64, 5> = HashGridIndex::new(1.0);

        let mut expected: FastHashSet<VertexKey> = FastHashSet::default();

        let offsets = [-1.0, 0.0, 1.0];
        for a in offsets {
            for b in offsets {
                for c in offsets {
                    for d in offsets {
                        for e in offsets {
                            let v = slots.insert(1);
                            expected.insert(v);
                            grid.insert_vertex(
                                v,
                                &[a + 0.25, b + 0.25, c + 0.25, d + 0.25, e + 0.25],
                            );
                        }
                    }
                }
            }
        }

        let mut found: FastHashSet<VertexKey> = FastHashSet::default();
        let used = grid.for_each_candidate_vertex_key(&[0.25, 0.25, 0.25, 0.25, 0.25], |vkey| {
            found.insert(vkey);
            true
        });

        assert!(used);
        assert_eq!(found.len(), 243);
        assert_eq!(found.len(), expected.len());
    }
}

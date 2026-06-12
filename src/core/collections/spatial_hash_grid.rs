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
//! insertion. In that case, callers should remove committed vertex deletions
//! explicitly. Query call sites that keep the grid across transactional rollback
//! must validate returned keys against the live triangulation because stale keys
//! from rolled-back insertions can remain in the cache.

use super::{SecureHashMap, SmallBuffer};
use crate::core::tds::VertexKey;
use crate::geometry::traits::coordinate::{CoordinateScalar, InvalidCoordinateValue};
use std::hash::{Hash, Hasher};

/// Maximum dimension supported by the hash-grid neighborhood walk.
///
/// The neighbor search enumerates the 3^D Moore neighborhood, which grows quickly.
const MAX_HASH_GRID_DIMENSION: usize = 5;

const BUCKET_INLINE_CAPACITY: usize = 8;

/// Errors that can occur while constructing a spatial hash-grid index.
///
/// Cell size is parsed at construction so [`HashGridIndex`] never stores a
/// non-finite or non-positive divisor. Unsupported dimensions are not
/// construction errors; they create an unusable index that callers can detect
/// with [`HashGridIndex::is_usable`] and treat as a linear-scan fallback.
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum HashGridIndexError<T = f64> {
    /// The cell size is non-finite.
    #[error("invalid hash-grid cell size: {value}; cell size must be finite and positive")]
    NonFiniteCellSize {
        /// The non-finite cell size category.
        value: InvalidCoordinateValue,
    },
    /// The finite cell size is zero or negative.
    #[error("invalid hash-grid cell size {value:?}; cell size must be positive")]
    NonPositiveCellSize {
        /// The finite non-positive cell size.
        value: T,
    },
}

/// Hashable grid-cell key for a D-dimensional grid.
///
/// Internally, this stores integer-valued cell coordinates as the same scalar
/// type used for points. We avoid casting to an integer type to keep the index
/// aligned with the crate's `CoordinateScalar` contract (`f64` today).
#[derive(Clone, Copy, Debug)]
struct GridKey<T, const D: usize>([T; D]);

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

/// Spatial hash grid mapping grid cells to nearby keys.
///
/// The grid uses a fixed `cell_size` and indexes vertices by the floored cell
/// coordinates `floor(coord / cell_size)`. [`HashGridIndex::try_new`] validates
/// that the stored cell size is finite and strictly positive, so lookup methods
/// only need to handle dimensional support and coordinate keyability.
#[derive(Clone, Debug)]
pub struct HashGridIndex<T, const D: usize, K = VertexKey> {
    cell_size: T,
    usable: bool,
    // Grid keys are derived directly from public coordinate input, so this map
    // uses randomized hashing instead of the crate's fast non-cryptographic
    // hasher.
    cells: SecureHashMap<GridKey<T, D>, SmallBuffer<K, BUCKET_INLINE_CAPACITY>>,
}

impl<T, const D: usize, K> HashGridIndex<T, D, K> {
    /// Returns whether the index is usable for accelerated lookup.
    ///
    /// This is `false` when `D` exceeds the bounded neighborhood walk supported
    /// by the grid, or after a coordinate could not be converted into a stable
    /// grid key. A `false` result tells callers to use the conservative
    /// linear-scan path.
    pub const fn is_usable(&self) -> bool {
        self.usable
    }

    /// Returns whether this const-generic dimension can use hash-grid indexing.
    pub const fn supports_dimension() -> bool {
        D <= MAX_HASH_GRID_DIMENSION
    }

    /// Returns the configured grid cell size.
    ///
    /// The returned value was parsed by [`HashGridIndex::try_new`] and is
    /// guaranteed to be finite and strictly positive for supported
    /// [`CoordinateScalar`] implementations.
    pub const fn cell_size(&self) -> T
    where
        T: Copy,
    {
        self.cell_size
    }

    /// Remove every indexed bucket without changing the configured cell size.
    pub fn clear(&mut self) {
        self.cells.clear();
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashGridIndexSnapshot {
    cell_size: String,
    usable: bool,
    cells: Vec<String>,
}

impl<T, const D: usize, K> HashGridIndex<T, D, K>
where
    T: CoordinateScalar,
{
    /// Creates a new grid index with a finite, positive cell size.
    ///
    /// Unsupported dimensions produce an unusable index, but invalid cell sizes
    /// are rejected before storage. This keeps all later grid-key arithmetic on
    /// validated scalar state.
    ///
    /// # Errors
    ///
    /// Returns [`HashGridIndexError::NonFiniteCellSize`] if `cell_size` is
    /// `NaN` or infinite, or [`HashGridIndexError::NonPositiveCellSize`] if it
    /// is finite and not strictly positive.
    pub fn try_new(cell_size: T) -> Result<Self, HashGridIndexError<T>> {
        if !cell_size.is_finite() {
            return Err(HashGridIndexError::NonFiniteCellSize {
                value: InvalidCoordinateValue::from_debug(&cell_size),
            });
        }

        if cell_size <= T::zero() {
            return Err(HashGridIndexError::NonPositiveCellSize { value: cell_size });
        }

        Ok(Self {
            cell_size,
            usable: D <= MAX_HASH_GRID_DIMENSION,
            cells: SecureHashMap::default(),
        })
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshot(&self) -> HashGridIndexSnapshot
    where
        T: std::fmt::Debug,
        K: std::fmt::Debug,
    {
        let mut cells = self
            .cells
            .iter()
            .map(|(key, bucket)| format!("{key:?}={bucket:?}"))
            .collect::<Vec<_>>();
        cells.sort();
        HashGridIndexSnapshot {
            cell_size: format!("{:?}", self.cell_size),
            usable: self.usable,
            cells,
        }
    }

    /// Marks the index unusable so callers fall back to a conservative scan.
    const fn disable(&mut self) {
        self.usable = false;
    }

    /// Returns whether `coords` can be mapped to a stable grid key.
    pub fn can_key_coords(&self, coords: &[T; D]) -> bool {
        self.key_for_coords(coords).is_some()
    }

    /// Insert a vertex into the appropriate grid cell.
    ///
    /// If the index is not usable, or the point cannot be keyed robustly, the
    /// index is disabled (so callers can fall back to linear scans).
    pub fn insert_vertex(&mut self, vertex_key: K, coords: &[T; D]) {
        if !self.usable {
            return;
        }

        let Some(key) = self.key_for_coords(coords) else {
            self.disable();
            return;
        };

        self.cells.entry(key).or_default().push(vertex_key);
    }

    /// Remove a vertex from the appropriate grid cell.
    ///
    /// If the coordinates cannot be keyed, the index is disabled so callers fall
    /// back to a full scan instead of querying a cache that may contain a stale
    /// entry we could not remove.
    pub fn remove_vertex(&mut self, vertex_key: &K, coords: &[T; D])
    where
        K: PartialEq,
    {
        if !self.usable {
            return;
        }

        let Some(key) = self.key_for_coords(coords) else {
            self.disable();
            return;
        };

        let Some(bucket) = self.cells.get_mut(&key) else {
            return;
        };
        bucket.retain(|candidate| candidate != vertex_key);

        if bucket.is_empty() {
            self.cells.remove(&key);
        }
    }

    /// Visit all candidate vertex keys in the 3^D neighborhood around `coords`.
    ///
    /// Returns `true` if the index was used for the query (even if it yielded zero
    /// candidates). Returns `false` if the index was unusable for this query.
    pub fn for_each_candidate_vertex_key<F>(&self, coords: &[T; D], mut f: F) -> bool
    where
        K: Copy,
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

        let _completed = Self::visit_neighbor_cells(0, &base, &mut current, &mut |neighbor| {
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

    /// Converts finite coordinates into a grid key when neighbor enumeration remains stable.
    fn key_for_coords(&self, coords: &[T; D]) -> Option<GridKey<T, D>> {
        if !self.usable {
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

    /// Recursively visits the Moore neighborhood around `base` and stops early on callback failure.
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
    use crate::core::tds::VertexKey;
    use crate::geometry::traits::coordinate::InvalidCoordinateValue;
    use approx::assert_abs_diff_eq;
    use slotmap::SlotMap;

    #[test]
    fn test_hash_grid_index_try_new_rejects_invalid_cell_size() {
        let grid: HashGridIndex<f64, 2> = HashGridIndex::try_new(1.0).unwrap();
        assert!(grid.is_usable());
        assert_abs_diff_eq!(grid.cell_size(), 1.0, epsilon = f64::EPSILON);

        assert!(matches!(
            HashGridIndex::<f64, 2>::try_new(0.0),
            Err(HashGridIndexError::NonPositiveCellSize { value: 0.0 })
        ));
        assert!(matches!(
            HashGridIndex::<f64, 2>::try_new(-1.0),
            Err(HashGridIndexError::NonPositiveCellSize { value: -1.0 })
        ));
        assert!(matches!(
            HashGridIndex::<f64, 2>::try_new(f64::NAN),
            Err(HashGridIndexError::NonFiniteCellSize {
                value: InvalidCoordinateValue::Nan
            })
        ));
        assert!(matches!(
            HashGridIndex::<f64, 2>::try_new(f64::INFINITY),
            Err(HashGridIndexError::NonFiniteCellSize {
                value: InvalidCoordinateValue::PositiveInfinity
            })
        ));
        assert!(matches!(
            HashGridIndex::<f64, 2>::try_new(f64::NEG_INFINITY),
            Err(HashGridIndexError::NonFiniteCellSize {
                value: InvalidCoordinateValue::NegativeInfinity
            })
        ));
    }

    #[test]
    fn test_hash_grid_index_keying_and_candidate_lookup_2d() {
        let mut slots: SlotMap<VertexKey, i32> = SlotMap::default();
        let v1 = slots.insert(1);
        let v2 = slots.insert(2);

        let mut grid: HashGridIndex<f64, 2> = HashGridIndex::try_new(1.0).unwrap();

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
        let mut grid: HashGridIndex<f64, 2> = HashGridIndex::try_new(1.0).unwrap();

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
    fn test_hash_grid_index_remove_vertex_prunes_bucket() {
        let mut slots: SlotMap<VertexKey, i32> = SlotMap::default();
        let v1 = slots.insert(1);
        let v2 = slots.insert(2);
        let mut grid: HashGridIndex<f64, 2> = HashGridIndex::try_new(1.0).unwrap();

        grid.insert_vertex(v1, &[0.25, 0.25]);
        grid.insert_vertex(v2, &[0.25, 0.25]);
        grid.remove_vertex(&v1, &[0.25, 0.25]);

        let mut found: FastHashSet<VertexKey> = FastHashSet::default();
        let used = grid.for_each_candidate_vertex_key(&[0.25, 0.25], |vkey| {
            found.insert(vkey);
            true
        });

        assert!(used);
        assert!(!found.contains(&v1));
        assert!(found.contains(&v2));

        grid.remove_vertex(&v2, &[0.25, 0.25]);
        let mut visited = false;
        assert!(grid.for_each_candidate_vertex_key(&[0.25, 0.25], |_| {
            visited = true;
            true
        }));
        assert!(!visited);
    }

    #[test]
    fn test_hash_grid_index_remove_vertex_noops_for_unusable_or_missing_bucket() {
        let mut slots: SlotMap<VertexKey, i32> = SlotMap::default();
        let v1 = slots.insert(1);
        let v2 = slots.insert(2);

        let mut unsupported: HashGridIndex<f64, 6> = HashGridIndex::try_new(1.0).unwrap();
        assert!(!unsupported.is_usable());
        unsupported.remove_vertex(&v1, &[0.0; 6]);
        assert!(!unsupported.is_usable());

        let mut grid: HashGridIndex<f64, 2> = HashGridIndex::try_new(1.0).unwrap();
        grid.insert_vertex(v1, &[0.25, 0.25]);
        grid.remove_vertex(&v2, &[20.25, 20.25]);

        let mut found = FastHashSet::default();
        assert!(grid.for_each_candidate_vertex_key(&[0.25, 0.25], |vkey| {
            found.insert(vkey);
            true
        }));
        assert!(found.contains(&v1));
        assert!(!found.contains(&v2));
    }

    #[test]
    fn test_hash_grid_index_remove_vertex_disables_on_unkeyable_coordinates() {
        let mut slots: SlotMap<VertexKey, i32> = SlotMap::default();
        let v = slots.insert(1);
        let mut grid: HashGridIndex<f64, 2> = HashGridIndex::try_new(1.0).unwrap();

        grid.insert_vertex(v, &[0.25, 0.25]);
        assert!(grid.is_usable());

        grid.remove_vertex(&v, &[f64::NAN, 0.25]);

        assert!(!grid.is_usable());
        assert!(!grid.for_each_candidate_vertex_key(&[0.25, 0.25], |_| true));
    }

    #[test]
    fn test_hash_grid_index_candidate_visit_counts_5d() {
        let mut slots: SlotMap<VertexKey, i32> = SlotMap::default();
        let mut grid: HashGridIndex<f64, 5> = HashGridIndex::try_new(1.0).unwrap();

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

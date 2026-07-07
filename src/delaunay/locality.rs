//! Locality helpers for triangulation construction and repair.
//!
//! These utilities sit at the boundary between spatial locality and topological
//! locality: callers may use Hilbert ordering or point-location hints to find a
//! nearby insertion site, then pass the concrete simplex keys touched by the TDS
//! mutation here to build bounded repair frontiers.

#![forbid(unsafe_code)]

use crate::core::algorithms::locate::{ConflictError, find_conflict_region};
use crate::core::collections::{FastHashSet, SimplexKeyBuffer, fast_hash_set_with_capacity};
use crate::core::tds::{SimplexKey, Tds};
use crate::core::traits::data_type::DataType;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;

/// Local conflict-seed collection result for exterior insertion repair.
#[must_use]
pub struct LocalConflictSeedSimplices {
    /// Live simplices that should seed local Delaunay repair.
    pub seed_simplices: SimplexKeyBuffer,
    /// Number of simplices returned by the local conflict-region search before any fallback seed.
    pub conflict_simplices_found: usize,
}

/// Adds live, deduplicated candidate simplices to a pending local repair frontier.
///
/// Returns the number of simplices newly appended to `pending_seed_simplices`.
pub fn accumulate_live_simplex_seeds<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    candidate_seed_simplices: &[SimplexKey],
    pending_seed_simplices: &mut SimplexKeyBuffer,
    pending_seen: &mut FastHashSet<SimplexKey>,
) -> usize {
    let mut added = 0usize;
    for &simplex_key in candidate_seed_simplices {
        if tds.contains_simplex(simplex_key) && pending_seen.insert(simplex_key) {
            pending_seed_simplices.push(simplex_key);
            added = added.saturating_add(1);
        }
    }
    added
}

/// Adds live, deduplicated candidate simplices to a compact repair seed buffer.
///
/// Returns the number of simplices newly appended to `seed_simplices`.
pub fn append_live_unique_simplex_seeds<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    candidate_seed_simplices: &[SimplexKey],
    seed_simplices: &mut SimplexKeyBuffer,
) -> usize {
    let mut seen: FastHashSet<SimplexKey> = fast_hash_set_with_capacity(
        seed_simplices
            .len()
            .saturating_add(candidate_seed_simplices.len()),
    );
    seen.extend(seed_simplices.iter().copied());

    let mut added = 0usize;
    for &simplex_key in candidate_seed_simplices {
        if tds.contains_simplex(simplex_key) && seen.insert(simplex_key) {
            seed_simplices.push(simplex_key);
            added = added.saturating_add(1);
        }
    }
    added
}

/// Retains only live, deduplicated simplices in a pending local repair frontier.
pub fn retain_live_simplex_seeds<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    seed_simplices: &mut SimplexKeyBuffer,
    seen: &mut FastHashSet<SimplexKey>,
) {
    seen.clear();
    seed_simplices
        .retain(|simplex_key| tds.contains_simplex(*simplex_key) && seen.insert(*simplex_key));
}

/// Clears a local repair frontier and its deduplication set together.
pub fn clear_simplex_seed_set(
    seed_simplices: &mut SimplexKeyBuffer,
    seen: &mut FastHashSet<SimplexKey>,
) {
    seed_simplices.clear();
    seen.clear();
}

/// Retains conflict simplices and records removed simplices as local repair seeds.
pub fn retain_simplices_and_record_removed(
    conflict_simplices: &mut SimplexKeyBuffer,
    repair_seed_simplices: &mut SimplexKeyBuffer,
    mut keep_simplex: impl FnMut(SimplexKey) -> bool,
) {
    conflict_simplices.retain(|simplex_key| {
        let keep = keep_simplex(*simplex_key);
        if !keep {
            repair_seed_simplices.push(*simplex_key);
        }
        keep
    });
}

/// Replaces conflict simplices and records simplices missing from the replacement.
pub fn replace_simplices_and_record_removed(
    conflict_simplices: &mut SimplexKeyBuffer,
    repair_seed_simplices: &mut SimplexKeyBuffer,
    replacement: SimplexKeyBuffer,
) {
    let replacement_set: FastHashSet<SimplexKey> = replacement.iter().copied().collect();
    for &simplex_key in conflict_simplices.iter() {
        if !replacement_set.contains(&simplex_key) {
            repair_seed_simplices.push(simplex_key);
        }
    }
    *conflict_simplices = replacement;
}

/// Collects local repair seeds for an exterior insertion from the terminal walk simplex.
///
/// The terminal simplex is adjacent to the hull facet crossed by point location, so a
/// BFS conflict search from it gives a bounded local frontier without scanning the
/// entire triangulation. If no circumsphere conflict is found, the terminal simplex
/// itself is still a useful local seed.
///
/// # Errors
///
/// Returns [`ConflictError`] when the terminal simplex is live but the bounded
/// conflict search cannot classify the local star, for example because a
/// simplex has invalid arity, references missing vertices, or a geometric
/// predicate cannot be evaluated.
pub fn collect_local_exterior_conflict_seed_simplices<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    point: &Point<D>,
    terminal_simplex: SimplexKey,
) -> Result<LocalConflictSeedSimplices, ConflictError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    let mut seed_simplices = SimplexKeyBuffer::new();
    if !tds.contains_simplex(terminal_simplex) {
        return Ok(LocalConflictSeedSimplices {
            seed_simplices,
            conflict_simplices_found: 0,
        });
    }

    let computed = find_conflict_region(tds, kernel, point, terminal_simplex)?;
    let conflict_simplices_found = computed.len();
    if computed.is_empty() {
        seed_simplices.push(terminal_simplex);
    } else {
        seed_simplices = computed;
    }

    Ok(LocalConflictSeedSimplices {
        seed_simplices,
        conflict_simplices_found,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::triangulation::Triangulation;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::point::Point;
    use crate::triangulation::DelaunayTriangulation;
    use crate::vertex;
    use slotmap::KeyData;

    fn simplex_triangulation_3d() -> Triangulation<FastKernel<f64>, (), (), 3> {
        let vertices = [
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds)
    }

    #[test]
    fn accumulate_live_simplex_seeds_dedupes_and_ignores_stale() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let all_simplices: Vec<SimplexKey> =
            dt.simplices().map(|(simplex_key, _)| simplex_key).collect();
        assert!(
            all_simplices.len() >= 2,
            "fixture should produce multiple simplices for seed accumulation"
        );

        let stale_simplex = SimplexKey::from(KeyData::from_ffi(999_999));
        let mut pending_seed_simplices = SimplexKeyBuffer::new();
        pending_seed_simplices.push(all_simplices[0]);
        let mut pending_seen: FastHashSet<SimplexKey> =
            pending_seed_simplices.iter().copied().collect();
        let added = accumulate_live_simplex_seeds(
            dt.tds(),
            &[
                all_simplices[0],
                all_simplices[1],
                all_simplices[1],
                stale_simplex,
            ],
            &mut pending_seed_simplices,
            &mut pending_seen,
        );

        assert_eq!(added, 1);
        assert_eq!(
            pending_seed_simplices.iter().copied().collect::<Vec<_>>(),
            vec![all_simplices[0], all_simplices[1]]
        );
        assert!(!pending_seed_simplices.contains(&stale_simplex));

        let added_again = accumulate_live_simplex_seeds(
            dt.tds(),
            &[all_simplices[1]],
            &mut pending_seed_simplices,
            &mut pending_seen,
        );
        assert_eq!(added_again, 0);
        assert_eq!(
            pending_seed_simplices.iter().copied().collect::<Vec<_>>(),
            vec![all_simplices[0], all_simplices[1]]
        );
    }

    #[test]
    fn append_live_unique_simplex_seeds_dedupes_and_ignores_stale() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let all_simplices: Vec<SimplexKey> =
            dt.simplices().map(|(simplex_key, _)| simplex_key).collect();
        assert!(
            all_simplices.len() >= 2,
            "fixture should produce multiple simplices for compact seed accumulation"
        );

        let stale_simplex = SimplexKey::from(KeyData::from_ffi(999_999));
        let mut seed_simplices = SimplexKeyBuffer::new();
        seed_simplices.push(all_simplices[0]);
        let added = append_live_unique_simplex_seeds(
            dt.tds(),
            &[
                all_simplices[0],
                all_simplices[1],
                stale_simplex,
                all_simplices[1],
            ],
            &mut seed_simplices,
        );

        assert_eq!(added, 1);
        assert_eq!(
            seed_simplices.iter().copied().collect::<Vec<_>>(),
            vec![all_simplices[0], all_simplices[1],]
        );
    }

    #[test]
    fn retain_live_simplex_seeds_filters_stale_and_dedupes() {
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
            vertex!([1.0, 1.0]).unwrap(),
            vertex!([0.5, 0.5]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let all_simplices: Vec<SimplexKey> =
            dt.simplices().map(|(simplex_key, _)| simplex_key).collect();
        assert!(
            all_simplices.len() >= 2,
            "fixture should produce multiple simplices for seed retention"
        );

        let stale_simplex = SimplexKey::from(KeyData::from_ffi(999_999));
        let mut seed_simplices = SimplexKeyBuffer::new();
        seed_simplices.extend([
            all_simplices[0],
            stale_simplex,
            all_simplices[1],
            all_simplices[0],
        ]);
        let mut seen = FastHashSet::default();
        retain_live_simplex_seeds(dt.tds(), &mut seed_simplices, &mut seen);

        assert_eq!(
            seed_simplices.iter().copied().collect::<Vec<_>>(),
            vec![all_simplices[0], all_simplices[1]]
        );
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn clear_simplex_seed_set_clears_both_collections() {
        let stale_simplex = SimplexKey::from(KeyData::from_ffi(999_999));
        let mut seed_simplices = SimplexKeyBuffer::new();
        seed_simplices.push(stale_simplex);
        let mut seen = FastHashSet::default();
        seen.insert(stale_simplex);

        clear_simplex_seed_set(&mut seed_simplices, &mut seen);

        assert!(seed_simplices.is_empty());
        assert!(seen.is_empty());
    }

    #[test]
    fn retain_and_replace_simplices_record_removed_repair_seeds() {
        let a = SimplexKey::from(KeyData::from_ffi(31));
        let b = SimplexKey::from(KeyData::from_ffi(32));
        let c = SimplexKey::from(KeyData::from_ffi(33));
        let d = SimplexKey::from(KeyData::from_ffi(34));

        let mut conflict_simplices: SimplexKeyBuffer = [a, b, c].into_iter().collect();
        let mut repair_seed_simplices = SimplexKeyBuffer::new();
        retain_simplices_and_record_removed(
            &mut conflict_simplices,
            &mut repair_seed_simplices,
            |ck| ck != b,
        );
        assert_eq!(
            conflict_simplices.iter().copied().collect::<Vec<_>>(),
            vec![a, c]
        );
        assert_eq!(
            repair_seed_simplices.iter().copied().collect::<Vec<_>>(),
            vec![b]
        );

        let replacement: SimplexKeyBuffer = [c, d].into_iter().collect();
        replace_simplices_and_record_removed(
            &mut conflict_simplices,
            &mut repair_seed_simplices,
            replacement,
        );
        assert_eq!(
            conflict_simplices.iter().copied().collect::<Vec<_>>(),
            vec![c, d]
        );
        assert_eq!(
            repair_seed_simplices.iter().copied().collect::<Vec<_>>(),
            vec![b, a]
        );
    }

    #[test]
    fn collect_local_exterior_conflict_seed_simplices_uses_terminal_seed_when_empty() {
        let tri = simplex_triangulation_3d();
        let terminal_simplex = tri.tds.simplex_keys().next().unwrap();
        let result = collect_local_exterior_conflict_seed_simplices(
            &tri.tds,
            &FastKernel::new(),
            &Point::try_new([2.0, 2.0, 2.0]).expect("finite point coordinates"),
            terminal_simplex,
        )
        .unwrap();

        assert_eq!(result.conflict_simplices_found, 0);
        assert_eq!(
            result.seed_simplices.iter().copied().collect::<Vec<_>>(),
            vec![terminal_simplex]
        );
    }

    #[test]
    fn collect_local_exterior_conflict_seed_simplices_returns_local_conflicts() {
        let tri = simplex_triangulation_3d();
        let terminal_simplex = tri.tds.simplex_keys().next().unwrap();
        let result = collect_local_exterior_conflict_seed_simplices(
            &tri.tds,
            &FastKernel::new(),
            &Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates"),
            terminal_simplex,
        )
        .unwrap();

        assert_eq!(result.conflict_simplices_found, 1);
        assert_eq!(
            result.seed_simplices.iter().copied().collect::<Vec<_>>(),
            vec![terminal_simplex]
        );
    }
}

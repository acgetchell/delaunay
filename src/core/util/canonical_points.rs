//! Canonical vertex-ordering helpers for geometric predicate call sites.
//!
//! These helpers ensure that kernel predicates always receive simplex vertices
//! sorted by [`VertexKey`], making `SoS` perturbation priority identity-based
//! rather than dependent on internal storage order.
//!
//! # Motivation
//!
//! The `SoS` implementation assigns perturbation priority by slice position.
//! If different call sites present the same vertex set in different orders,
//! `SoS` tie-breaking can be inconsistent.  By sorting vertices into a canonical
//! order (by `VertexKey` identity) before every kernel call, the existing
//! slice-position `SoS` becomes identity-based by construction.
//!
//! # Conventions
//!
//! - **Insphere**: sort all D+1 simplex vertices by key; test point is separate.
//! - **Orientation for comparison**: sort D facet vertices by key, extra vertex
//!   (opposite or query) always last.
//! - **Orientation for degeneracy check**: sort all D+1 vertices by key.

use crate::core::cell::Cell;
use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::core::traits::DataType;
use crate::core::triangulation_data_structure::{Tds, VertexKey};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::CoordinateScalar;
use slotmap::Key;

// =============================================================================
// CANONICAL POINT-COLLECTION HELPERS
// =============================================================================

/// Collect cell vertex points in canonical [`VertexKey`] order.
///
/// Sorts the cell's vertex keys by their stable identity
/// (`vk.data().as_ffi()`), then resolves each to its [`Point`].
///
/// Returns [`None`] if any vertex key cannot be resolved via the TDS.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure for vertex lookups
/// * `cell` - The cell whose vertices to collect
///
/// # Examples
///
/// ```rust,ignore
/// let points = sorted_cell_points(tds, cell)
///     .ok_or(SomeError::MissingVertex)?;
/// let sign = kernel.in_sphere(&points, &query_point)?;
/// ```
pub fn sorted_cell_points<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell: &Cell<T, U, V, D>,
) -> Option<SmallBuffer<Point<T, D>, MAX_PRACTICAL_DIMENSION_SIZE>>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        cell.vertices().iter().copied().collect();
    keys.sort_unstable_by_key(|vk| vk.data().as_ffi());

    let mut points = SmallBuffer::with_capacity(keys.len());
    for &vk in &keys {
        points.push(*tds.get_vertex_by_key(vk)?.point());
    }
    Some(points)
}

/// Collect facet vertex points in canonical [`VertexKey`] order, then append
/// `extra` at position D (last).
///
/// Sorts `facet_keys` by their stable identity (`vk.data().as_ffi()`),
/// resolves each to its [`Point`], and appends `extra` at the end.
///
/// Returns [`None`] if any vertex key cannot be resolved via the TDS.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure for vertex lookups
/// * `facet_keys` - Vertex keys forming the facet (will be sorted internally)
/// * `extra` - The extra point to append at position D (opposite vertex or query)
///
/// # Examples
///
/// ```rust,ignore
/// let points = sorted_facet_points_with_extra(tds, &facet_keys, opposite_point)
///     .ok_or(SomeError::MissingVertex)?;
/// let orient = kernel.orientation(&points)?;
/// ```
pub fn sorted_facet_points_with_extra<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    facet_keys: &[VertexKey],
    extra: Point<T, D>,
) -> Option<SmallBuffer<Point<T, D>, MAX_PRACTICAL_DIMENSION_SIZE>>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut sorted_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        facet_keys.iter().copied().collect();
    sorted_keys.sort_unstable_by_key(|vk| vk.data().as_ffi());

    let mut points = SmallBuffer::with_capacity(sorted_keys.len() + 1);
    for &vk in &sorted_keys {
        points.push(*tds.get_vertex_by_key(vk)?.point());
    }
    points.push(extra);
    Some(points)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::traits::coordinate::Coordinate;

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    /// Build a minimal TDS with the given points and return it along with the
    /// vertex keys in insertion order.
    fn build_tds_with_points<const D: usize>(
        coords: &[[f64; D]],
    ) -> (Tds<f64, (), (), D>, Vec<VertexKey>) {
        let mut tds = Tds::<f64, (), (), D>::empty();
        let mut keys = Vec::with_capacity(coords.len());
        for c in coords {
            let v = crate::core::vertex::VertexBuilder::<_, (), _>::default()
                .point(Point::new(*c))
                .build()
                .expect("vertex build should succeed");
            let vk = tds
                .insert_vertex_with_mapping(v)
                .expect("insert should succeed");
            keys.push(vk);
        }
        (tds, keys)
    }

    // =========================================================================
    // SORTED CELL POINTS TESTS
    // =========================================================================

    #[test]
    fn test_sorted_cell_points_produces_canonical_order() {
        // Build a TDS with 3 vertices and a cell referencing them
        let (mut tds, keys) = build_tds_with_points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

        // Create a cell with vertices in insertion order
        let cell = Cell::new(keys.clone(), None::<()>).expect("cell should be valid");
        let cell_key = tds
            .insert_cell_with_mapping(cell)
            .expect("insert should succeed");

        let cell_ref = tds.get_cell(cell_key).unwrap();
        let points = sorted_cell_points(&tds, cell_ref).expect("should resolve all vertices");

        // Verify points are in VertexKey canonical order
        let mut sorted_keys = keys;
        sorted_keys.sort_unstable_by_key(|vk| vk.data().as_ffi());

        for (i, &vk) in sorted_keys.iter().enumerate() {
            let expected = *tds.get_vertex_by_key(vk).unwrap().point();
            assert_eq!(points[i], expected);
        }
    }

    #[test]
    fn test_sorted_cell_points_permutation_invariant() {
        // Build a TDS with 3 vertices
        let (mut tds, keys) = build_tds_with_points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

        // Create two cells with vertices in different orders
        let cell_a =
            Cell::new(vec![keys[0], keys[1], keys[2]], None::<()>).expect("cell should be valid");
        let cell_b =
            Cell::new(vec![keys[2], keys[0], keys[1]], None::<()>).expect("cell should be valid");
        let key_a = tds
            .insert_cell_with_mapping(cell_a)
            .expect("insert should succeed");
        let key_b = tds
            .insert_cell_with_mapping(cell_b)
            .expect("insert should succeed");

        let points_a = sorted_cell_points(&tds, tds.get_cell(key_a).unwrap()).unwrap();
        let points_b = sorted_cell_points(&tds, tds.get_cell(key_b).unwrap()).unwrap();

        // Both should produce the same canonical ordering
        assert_eq!(points_a.as_slice(), points_b.as_slice());
    }

    // =========================================================================
    // SORTED FACET POINTS WITH EXTRA TESTS
    // =========================================================================

    #[test]
    fn test_sorted_facet_points_with_extra_appends_at_end() {
        let (tds, keys) = build_tds_with_points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

        let facet_keys = &[keys[0], keys[1]];
        let extra = Point::new([0.5, 0.5]);

        let points =
            sorted_facet_points_with_extra(&tds, facet_keys, extra).expect("should resolve");

        // Should have 3 points: 2 facet + 1 extra
        assert_eq!(points.len(), 3);
        // Extra point should be last
        assert_eq!(points[2], extra);
    }

    #[test]
    fn test_sorted_facet_points_with_extra_permutation_invariant() {
        let (tds, keys) = build_tds_with_points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

        let extra = Point::new([0.5, 0.5]);

        let points_a = sorted_facet_points_with_extra(&tds, &[keys[0], keys[1]], extra).unwrap();
        let points_b = sorted_facet_points_with_extra(&tds, &[keys[1], keys[0]], extra).unwrap();

        // Both orderings should produce the same result
        assert_eq!(points_a.as_slice(), points_b.as_slice());
    }

    // =========================================================================
    // SOS PERMUTATION-INVARIANCE TESTS (KERNEL INTEGRATION)
    // =========================================================================

    /// For a degenerate (co-spherical) configuration, verify that
    /// `AdaptiveKernel::in_sphere` produces the same sign regardless of
    /// cell vertex storage order, when canonical sorting is applied.
    #[test]
    fn test_canonical_insphere_permutation_invariant_2d() {
        use crate::geometry::kernel::{AdaptiveKernel, Kernel};

        // 3 points forming a right triangle + a cospherical test point.
        // The circumcircle of (0,0),(1,0),(0,1) passes through (1,1).
        let (mut tds, keys) = build_tds_with_points(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let test_point = Point::new([1.0, 1.0]);
        let kernel = AdaptiveKernel::<f64>::new();

        // All 6 permutations of the 3 vertices
        let permutations: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];

        let mut signs = Vec::new();
        for perm in &permutations {
            let cell = Cell::new(
                vec![keys[perm[0]], keys[perm[1]], keys[perm[2]]],
                None::<()>,
            )
            .expect("cell should be valid");
            let cell_key = tds
                .insert_cell_with_mapping(cell)
                .expect("insert should succeed");

            // Use canonical sorting to collect points
            let sorted = sorted_cell_points(&tds, tds.get_cell(cell_key).unwrap()).unwrap();
            let sign = kernel.in_sphere(&sorted, &test_point).unwrap();
            signs.push(sign);
        }

        // All permutations must produce the same sign
        assert!(
            signs.iter().all(|&s| s == signs[0]),
            "canonical sorting must make insphere permutation-invariant: {signs:?}"
        );
    }

    /// Same test for 3D: verify insphere sign is identical across **all 24**
    /// permutations of a co-spherical 3D simplex.
    #[test]
    fn test_canonical_insphere_permutation_invariant_3d() {
        use crate::geometry::kernel::{AdaptiveKernel, Kernel};

        // Standard 3D simplex + cospherical test point (1,1,1)
        let (mut tds, keys) = build_tds_with_points(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let test_point = Point::new([1.0, 1.0, 1.0]);
        let kernel = AdaptiveKernel::<f64>::new();

        // All 24 permutations of 4 vertices
        #[rustfmt::skip]
        let perms: [[usize; 4]; 24] = [
            [0,1,2,3], [0,1,3,2], [0,2,1,3], [0,2,3,1], [0,3,1,2], [0,3,2,1],
            [1,0,2,3], [1,0,3,2], [1,2,0,3], [1,2,3,0], [1,3,0,2], [1,3,2,0],
            [2,0,1,3], [2,0,3,1], [2,1,0,3], [2,1,3,0], [2,3,0,1], [2,3,1,0],
            [3,0,1,2], [3,0,2,1], [3,1,0,2], [3,1,2,0], [3,2,0,1], [3,2,1,0],
        ];

        let mut signs = Vec::new();
        for perm in &perms {
            let cell = Cell::new(
                vec![keys[perm[0]], keys[perm[1]], keys[perm[2]], keys[perm[3]]],
                None::<()>,
            )
            .expect("cell should be valid");
            let cell_key = tds
                .insert_cell_with_mapping(cell)
                .expect("insert should succeed");

            let sorted = sorted_cell_points(&tds, tds.get_cell(cell_key).unwrap()).unwrap();
            let sign = kernel.in_sphere(&sorted, &test_point).unwrap();
            signs.push(sign);
        }

        assert!(
            signs.iter().all(|&s| s == signs[0]),
            "canonical sorting must make 3D insphere permutation-invariant: {signs:?}"
        );
    }
}

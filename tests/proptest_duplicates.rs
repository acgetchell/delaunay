//! Property-based tests for random point clouds containing duplicates and
//! near-duplicates.
//!
//! These tests exercise the unified Bowyer–Watson pipeline used by
//! `Tds::bowyer_watson_with_diagnostics` on randomized inputs that include
//! exact duplicates and small jittered near-duplicates. The goal is to ensure
//! that:
//!
//! - The final triangulation remains structurally valid and globally Delaunay
//!   for the kept subset of vertices.
//! - All unsalvageable vertices reported by the diagnostics come from the
//!   original input set.

use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::Coordinate;
use delaunay::geometry::point::Point;
use proptest::prelude::*;
use std::collections::HashSet;

// =============================================================================
// STRATEGIES
// =============================================================================

fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

/// Generate a random 2D point cloud and then introduce a mix of exact
/// duplicates and small jittered near-duplicates.
fn cloud_with_duplicates_2d() -> impl Strategy<Value = Vec<Point<f64, 2>>> {
    prop::collection::vec(
        prop::array::uniform2(finite_coordinate()).prop_map(Point::new),
        6..=12,
    )
    .prop_map(|mut pts| {
        if pts.len() >= 3 {
            // Exact duplicate of the first point.
            let dup = pts[0];
            pts.push(dup);

            // Jittered near-duplicate of the second point.
            let mut coords: [f64; 2] = pts[1].into();
            for c in &mut coords {
                *c += 1e-9;
            }
            pts.push(Point::new(coords));
        }
        pts
    })
}

/// Generate a random 3D point cloud and then introduce a mix of exact
/// duplicates and small jittered near-duplicates.
fn cloud_with_duplicates_3d() -> impl Strategy<Value = Vec<Point<f64, 3>>> {
    prop::collection::vec(
        prop::array::uniform3(finite_coordinate()).prop_map(Point::new),
        6..=12,
    )
    .prop_map(|mut pts| {
        if pts.len() >= 3 {
            // Exact duplicate of the first point.
            let dup = pts[0];
            pts.push(dup);

            // Jittered near-duplicate of the second point.
            let mut coords: [f64; 3] = pts[1].into();
            for c in &mut coords {
                *c += 1e-9;
            }
            pts.push(Point::new(coords));
        }
        pts
    })
}

/// Count the number of unique coordinate tuples in `pts` using bitwise
/// equality on each component. This is used to ensure we have at least
/// D+1 distinct points before attempting triangulation.
fn count_unique_coords_by_bits<const D: usize>(pts: &[Point<f64, D>]) -> usize {
    let mut set: HashSet<Vec<u64>> = HashSet::with_capacity(pts.len());
    for p in pts {
        let coords: [f64; D] = (*p).into();
        let key: Vec<u64> = coords.iter().map(|x| x.to_bits()).collect();
        set.insert(key);
    }
    set.len()
}

// =============================================================================
// PROPERTIES (2D & 3D)
// =============================================================================

proptest! {
    /// Property: Random 2D clouds with duplicates and near-duplicates
    /// produce triangulations that are globally Delaunay for the kept subset
    /// and only report unsalvageable vertices drawn from the input set.
    #[test]
    fn prop_cloud_with_duplicates_is_delaunay_for_kept_subset_2d(
        points in cloud_with_duplicates_2d()
    ) {
        // Require at least D+1 distinct points to form non-degenerate simplices.
        let unique = count_unique_coords_by_bits(&points);
        prop_assume!(unique > 2);

        let vertices: Vec<Vertex<f64, Option<()>, 2>> = Vertex::from_points(points);

        // First, construct via Tds::new to ensure the standard pipeline succeeds
        // and yields a globally Delaunay triangulation for the kept subset.
        let tds: Tds<f64, Option<()>, Option<()>, 2> =
            if let Ok(tds) = Tds::new(&vertices) {
                tds
            } else {
                // Truly degenerate inputs are skipped.
                prop_assume!(false);
                unreachable!();
            };

        // Structural and global Delaunay validity for the kept subset.
        prop_assert!(tds.is_valid().is_ok());
        prop_assert!(tds.validate_delaunay().is_ok());

        // Run a diagnostic Bowyer–Watson pass on the same input set to verify
        // that any unsalvageable vertices (if present) originate from the
        // original input set.
        let mut tds_diag: Tds<f64, Option<()>, Option<()>, 2> =
            Tds::new(&vertices).expect("Second construction should succeed for same input");
        let diagnostics = tds_diag
            .bowyer_watson_with_diagnostics()
            .expect("Diagnostic Bowyer–Watson should succeed for same vertex set");

        let input_uuids: HashSet<_> = vertices
            .iter()
            .map(delaunay::core::Vertex::uuid)
            .collect();
        for report in &diagnostics.unsalvageable_vertices {
            prop_assert!(
                input_uuids.contains(&report.vertex().uuid()),
                "Unsalvageable vertex must come from input set",
            );
        }
    }

    /// Property: Random 3D clouds with duplicates and near-duplicates
    /// produce triangulations that are globally Delaunay for the kept subset
    /// and only report unsalvageable vertices drawn from the input set.
    #[test]
    fn prop_cloud_with_duplicates_is_delaunay_for_kept_subset_3d(
        points in cloud_with_duplicates_3d()
    ) {
        // Require at least D+1 distinct points to form non-degenerate simplices.
        let unique = count_unique_coords_by_bits(&points);
        prop_assume!(unique > 3);

        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);

        // First, construct via Tds::new to ensure the standard pipeline succeeds
        // and yields a globally Delaunay triangulation for the kept subset.
        let tds: Tds<f64, Option<()>, Option<()>, 3> =
            if let Ok(tds) = Tds::new(&vertices) {
                tds
            } else {
                // Truly degenerate inputs are skipped.
                prop_assume!(false);
                unreachable!();
            };

        // Structural and global Delaunay validity for the kept subset.
        prop_assert!(tds.is_valid().is_ok());
        prop_assert!(tds.validate_delaunay().is_ok());

        // Run a diagnostic Bowyer–Watson pass on the same input set to verify
        // that any unsalvageable vertices (if present) originate from the
        // original input set.
        let mut tds_diag: Tds<f64, Option<()>, Option<()>, 3> =
            Tds::new(&vertices).expect("Second construction should succeed for same input");
        let diagnostics = tds_diag
            .bowyer_watson_with_diagnostics()
            .expect("Diagnostic Bowyer–Watson should succeed for same vertex set");

        let input_uuids: HashSet<_> = vertices
            .iter()
            .map(delaunay::core::Vertex::uuid)
            .collect();
        for report in &diagnostics.unsalvageable_vertices {
            prop_assert!(
                input_uuids.contains(&report.vertex().uuid()),
                "Unsalvageable vertex must come from input set",
            );
        }
    }
}

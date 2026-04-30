//! Smoke test for conflict-region completeness verification.
//!
//! Exercises `verify_conflict_region_completeness` using the known-failing 3D
//! seed from #306 to confirm the diagnostic tooling works end-to-end.
//!
//! These tests are `#[ignore]` by default — run with:
//! ```bash
//! cargo test --test conflict_region_verification --features diagnostics -- --ignored --nocapture
//! ```

#![forbid(unsafe_code)]
#![cfg(feature = "diagnostics")]

use delaunay::core::algorithms::locate::{LocateResult, find_conflict_region, locate};
use delaunay::geometry::kernel::AdaptiveKernel;
use delaunay::geometry::util::generate_random_points_in_ball_seeded;
use delaunay::prelude::diagnostics::verify_conflict_region_completeness;
use delaunay::prelude::triangulation::*;

/// Verify that `verify_conflict_region_completeness` runs without panicking on
/// the known-failing 3D case (35 vertices, seed 0xE30C78582376677C, ball
/// radius 100) and reports any missed cells.
///
/// This test does NOT assert that the conflict region is complete — it's a
/// diagnostic smoke test that confirms the verification tooling is functional.
#[test]
#[ignore = "diagnostic test: exercises conflict-region verification tooling (#306)"]
fn verify_conflict_region_diagnostic_3d_35v() {
    // Same parameters as the #306 minimal reproducer.
    let seed: u64 = 0xE30C_7858_2376_677C;
    let n_points: usize = 35;
    let radius: f64 = 100.0;

    let points = generate_random_points_in_ball_seeded::<f64, 3>(n_points, radius, seed)
        .expect("point generation should succeed");

    let vertices: Vec<Vertex<f64, (), 3>> = points
        .iter()
        .map(|p| vertex!([p.coords()[0], p.coords()[1], p.coords()[2]]))
        .collect();

    // Build a triangulation from the first D+1 = 4 vertices, then insert the
    // rest one at a time while running conflict-region verification.
    let initial = &vertices[..4];
    let mut dt: DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3> =
        DelaunayTriangulation::new(initial).expect("initial simplex should succeed");

    let kernel = AdaptiveKernel::<f64>::new();
    let mut total_missed = 0_usize;
    let mut insertions_checked = 0_usize;
    let mut insert_errors: Vec<String> = Vec::new();

    for (i, v) in vertices[4..].iter().enumerate() {
        let point = *v.point();

        // Locate the point in the current triangulation.
        let Ok(location) = locate(dt.tds(), &kernel, &point, None) else {
            println!("[{i}] locate failed — skipping verification for this vertex");
            if let Err(e) = dt.insert(*v) {
                insert_errors.push(format!("[{i}] insert after locate failure: {e}"));
            }
            continue;
        };

        // Only verify interior insertions (the common case for #306).
        if let LocateResult::InsideCell(start_cell) = location {
            match find_conflict_region(dt.tds(), &kernel, &point, start_cell) {
                Ok(conflict_cells) => {
                    let missed = verify_conflict_region_completeness(
                        dt.tds(),
                        &kernel,
                        &point,
                        &conflict_cells,
                    );
                    if missed > 0 {
                        println!(
                            "[{i}] INCOMPLETE conflict region: {missed} cells missed \
                             (BFS found {}, brute-force found more)",
                            conflict_cells.len()
                        );
                    }
                    total_missed += missed;
                    insertions_checked += 1;
                }
                Err(e) => {
                    println!("[{i}] find_conflict_region failed: {e}");
                }
            }
        }

        // Perform the actual insertion; record failures for the summary.
        if let Err(e) = dt.insert(*v) {
            insert_errors.push(format!("[{i}] insert: {e}"));
        }
    }

    println!(
        "=== conflict-region verification summary ===\n\
         insertions checked: {insertions_checked}\n\
         total missed cells: {total_missed}\n\
         insertion errors:   {}\n\
         final vertices:     {}\n\
         final cells:        {}",
        insert_errors.len(),
        dt.number_of_vertices(),
        dt.number_of_cells(),
    );
    for err in &insert_errors {
        println!("  {err}");
    }

    // The verification path must have been exercised at least once.
    assert!(
        insertions_checked > 0,
        "no insertions checked — verification path not exercised"
    );
}

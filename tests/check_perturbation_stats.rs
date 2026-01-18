//! Temporary test to check if perturbation actually helps

use delaunay::core::vertex::VertexBuilder;
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::*;

#[test]
fn check_perturbation_effectiveness() {
    // Generate 50 random points with seed 123 (known problematic case)
    let points = generate_random_points_seeded::<f64, 3>(50, (-100.0, 100.0), 123).unwrap();

    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);
    // Ensure topology safety net runs after every insertion so invalid states are rolled back.
    dt.set_validation_policy(delaunay::core::triangulation::ValidationPolicy::Always);
    // Disable Delaunay repair to keep the test focused on perturbation and topology stability.
    dt.set_delaunay_repair_policy(
        delaunay::core::delaunay_triangulation::DelaunayRepairPolicy::Never,
    );

    let mut total_attempts_successful = 0usize;
    let mut first_try_success = 0usize;
    let mut perturbation_success = 0usize;
    let mut skipped = 0usize;
    let mut errored = 0usize;

    for point in points {
        let vertex = VertexBuilder::default().point(point).build().unwrap();

        match dt.insert_with_statistics(vertex) {
            Ok((InsertionOutcome::Inserted { .. }, stats)) => {
                total_attempts_successful += stats.attempts;
                if stats.attempts == 1 {
                    first_try_success += 1;
                } else if stats.used_perturbation() {
                    perturbation_success += 1;
                    println!(
                        "SUCCESS after {} attempts with perturbation",
                        stats.attempts
                    );
                }
            }
            Ok((InsertionOutcome::Skipped { error }, stats)) => {
                debug_assert!(stats.skipped());
                skipped += 1;
                println!("SKIPPED: {error:?}");
            }
            Err(e) => {
                errored += 1;
                println!("ERROR (non-retryable): {e:?}");
            }
        }
    }

    println!("\n=== Perturbation Effectiveness (seed 123, 50 points) ===");
    println!("First try success:                      {first_try_success}");
    println!("Perturbation success:                   {perturbation_success}");
    println!("Skipped:                                {skipped}");
    println!("Non-retryable errors:                   {errored}");
    println!("Total attempts (successful insertions): {total_attempts_successful}");

    let successful = first_try_success + perturbation_success;
    if successful > 0 {
        use num_traits::NumCast;
        let attempts_f64: f64 =
            NumCast::from(total_attempts_successful).expect("usize should always fit in f64");
        let successful_f64: f64 =
            NumCast::from(successful).expect("usize should always fit in f64");
        let avg_attempts = attempts_f64 / successful_f64;
        println!("Average attempts (per successful insertion): {avg_attempts:.2}");
    } else {
        println!("Average attempts (per successful insertion): N/A (no successful insertions)");
    }

    // Verify the triangulation is valid (Levels 1–3: elements + structure + topology)
    dt.as_triangulation().validate().unwrap();
    println!("Final vertex count:    {}", dt.number_of_vertices());
}

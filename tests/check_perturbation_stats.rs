//! Temporary test to check if perturbation actually helps

use delaunay::core::vertex::VertexBuilder;
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::*;

#[test]
fn check_perturbation_effectiveness() {
    // Generate 50 random points with seed 123 (known problematic case)
    let points = generate_random_points_seeded::<f64, 3>(50, (-100.0, 100.0), 123).unwrap();

    let mut dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();

    let mut total_attempts = 0usize;
    let mut first_try_success = 0usize;
    let mut perturbation_success = 0usize;
    let mut skipped = 0usize;

    for point in points {
        let vertex = VertexBuilder::default().point(point).build().unwrap();

        // Access statistics via triangulation_mut()
        match dt
            .triangulation_mut()
            .insert_with_statistics(vertex, None, None)
        {
            Ok((InsertionOutcome::Inserted { .. }, stats)) => {
                total_attempts += stats.attempts;
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
                skipped += 1;
                println!("ERROR (non-retryable): {e:?}");
            }
        }
    }

    println!("\n=== Perturbation Effectiveness (seed 123, 50 points) ===");
    println!("First try success:     {first_try_success}");
    println!("Perturbation success:  {perturbation_success}");
    println!("Skipped after retries: {skipped}");
    println!("Total attempts:        {total_attempts}");

    let successful = first_try_success + perturbation_success;
    if successful > 0 {
        use num_traits::NumCast;
        let attempts_f64: f64 =
            NumCast::from(total_attempts).expect("usize should always fit in f64");
        let successful_f64: f64 =
            NumCast::from(successful).expect("usize should always fit in f64");
        let avg_attempts = attempts_f64 / successful_f64;
        println!("Average attempts:      {avg_attempts:.2}");
    } else {
        println!("Average attempts:      N/A (no successful insertions)");
    }

    // Verify the triangulation is valid
    assert!(dt.is_valid().is_ok());
    println!("Final vertex count:    {}", dt.number_of_vertices());
}

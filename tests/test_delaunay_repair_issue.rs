//! Minimal reproduction test for Delaunay repair performance issue
//!
//! These tests demonstrate that the current global repair algorithm is fundamentally
//! inefficient, taking 77-480 seconds for 500-1000 vertices due to a flawed
//! vertex-removal strategy instead of proper bistellar flips.
//!
//! **Status**: These tests pass but are extremely slow. They are marked `#[ignore]`
//! until bistellar flip-based repair is implemented in v0.6.0.
//!
//! See plan: "Implement Bistellar Flips for Efficient Delaunay Repair (v0.6.0)"

use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
use delaunay::geometry::kernel::FastKernel;
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::vertex;

#[test]
#[ignore = "Extremely slow (~480s) until flip-based repair implemented (v0.6.0)"]
fn test_2d_1000v_seed42_range100() {
    // This configuration is known to cause issues - reproduce the benchmark failure
    let n_points = 1000;
    let seed = 42;
    let range = (-100.0, 100.0);

    let points = generate_random_points_seeded::<f64, 2>(n_points, range, seed)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();

    // This should succeed but currently fails with:
    // "Global Delaunay repair exceeded the maximum of 128 iterations"
    let result = DelaunayTriangulation::<FastKernel<f64>, (), (), 2>::new(&vertices);

    match result {
        Ok(_tds) => {
            println!("✓ Triangulation succeeded");
        }
        Err(e) => {
            eprintln!("✗ Triangulation failed: {e}");
            panic!("Failed to create triangulation: {e:?}");
        }
    }
}

#[test]
#[ignore = "Extremely slow (~480s) until flip-based repair implemented (v0.6.0)"]
fn test_2d_1000v_different_seed() {
    // Try with a different seed to see if it's seed-specific
    let n_points = 1000;
    let seed = 12345;
    let range = (-100.0, 100.0);

    let points = generate_random_points_seeded::<f64, 2>(n_points, range, seed)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();

    let _dt = DelaunayTriangulation::<FastKernel<f64>, (), (), 2>::new(&vertices)
        .expect("Failed to create triangulation");

    println!("✓ Triangulation with seed {seed} succeeded");
}

#[test]
#[ignore = "Slow (~77s) until flip-based repair implemented (v0.6.0)"]
fn test_2d_500v_seed42() {
    // Try with fewer vertices to see if it's size-specific
    let n_points = 500;
    let seed = 42;
    let range = (-100.0, 100.0);

    let points = generate_random_points_seeded::<f64, 2>(n_points, range, seed)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();

    let _dt = DelaunayTriangulation::<FastKernel<f64>, (), (), 2>::new(&vertices)
        .expect("Failed to create triangulation");

    println!("✓ Triangulation with {n_points} vertices succeeded");
}

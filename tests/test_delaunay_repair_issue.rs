//! Minimal reproduction test for Delaunay repair iteration limit issue
//!
//! This test reproduces the issue found in large_scale_performance benchmark
//! where 2D triangulation with 1000 vertices, seed 42, range (-100.0, 100.0)
//! fails with "Global Delaunay repair exceeded the maximum of 128 iterations"

use delaunay::core::triangulation_data_structure::Tds;
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::vertex;

#[test]
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
    let result = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices);

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
fn test_2d_1000v_different_seed() {
    // Try with a different seed to see if it's seed-specific
    let n_points = 1000;
    let seed = 12345;
    let range = (-100.0, 100.0);

    let points = generate_random_points_seeded::<f64, 2>(n_points, range, seed)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();

    let _tds = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices)
        .expect("Failed to create triangulation");
    
    println!("✓ Triangulation with seed {seed} succeeded");
}

#[test]
fn test_2d_500v_seed42() {
    // Try with fewer vertices to see if it's size-specific
    let n_points = 500;
    let seed = 42;
    let range = (-100.0, 100.0);

    let points = generate_random_points_seeded::<f64, 2>(n_points, range, seed)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();

    let _tds = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices)
        .expect("Failed to create triangulation");
    
    println!("✓ Triangulation with {n_points} vertices succeeded");
}

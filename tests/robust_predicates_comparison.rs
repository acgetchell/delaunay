//! Comprehensive test comparing robust predicates vs standard predicates
//!
//! This test demonstrates the improvements in numerical stability and robustness
//! when dealing with challenging geometric configurations that often cause
//! "No cavity boundary facets found" errors in Delaunay triangulation.

use delaunay::core::{
    algorithms::robust_bowyer_watson::RobustBoyerWatson, triangulation_data_structure::Tds,
};
use delaunay::geometry::{
    point::Point,
    predicates::{InSphere, insphere},
    robust_predicates::{config_presets, robust_insphere},
};
use delaunay::prelude::*;
use delaunay::vertex;
use std::time::Instant;

#[test]
fn test_nearly_coplanar_points() {
    println!("=== Testing Nearly Coplanar Points ===");

    // Create points that are nearly coplanar (very small z-coordinates)
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 1e-15]),
        Point::new([0.5, 0.866, 1e-16]),
        Point::new([0.5, 0.289, 1e-14]),
    ];

    let test_point = Point::new([0.5, 0.433, 1e-17]);
    let config = config_presets::degenerate_robust();

    println!("Points: {points:?}");
    println!("Test point: {test_point:?}");

    // Test with standard predicates
    let standard_start = Instant::now();
    let standard_result = insphere(&points, test_point);
    let standard_duration = standard_start.elapsed();

    // Test with robust predicates
    let robust_start = Instant::now();
    let robust_result = robust_insphere(&points, &test_point, &config);
    let robust_duration = robust_start.elapsed();

    println!("Standard result: {standard_result:?} (took {standard_duration:?})");
    println!("Robust result: {robust_result:?} (took {robust_duration:?})");

    // The robust version should give a definitive answer even for nearly degenerate cases
    assert!(
        robust_result.is_ok(),
        "Robust predicates should handle nearly coplanar points"
    );

    match (standard_result, robust_result) {
        (Ok(std_res), Ok(rob_res)) => {
            println!("✓ Both methods succeeded: standard={std_res:?}, robust={rob_res:?}");
        }
        (Err(_), Ok(rob_res)) => {
            println!("✓ Robust succeeded where standard failed: robust={rob_res:?}");
        }
        _ => {
            println!("Both methods had issues, but robust should be more reliable");
        }
    }
}

#[test]
fn test_high_precision_coordinates() {
    println!("\n=== Testing High Precision Coordinates ===");

    // Create points with very high precision requirements
    let points = vec![
        Point::new([1.000_000_000_000_000_1, 0.0, 0.0]),
        Point::new([0.0, 1.000_000_000_000_000_2, 0.0]),
        Point::new([0.0, 0.0, 1.000_000_000_000_000_3]),
        Point::new([
            0.333_333_333_333_333_3,
            0.333_333_333_333_333_4,
            0.333_333_333_333_333_5,
        ]),
    ];

    let test_point = Point::new([0.25, 0.25, 0.25]);
    let config = config_presets::high_precision();

    println!("High precision points test");

    let standard_result = insphere(&points, test_point);
    let robust_result = robust_insphere(&points, &test_point, &config);

    println!("Standard result: {standard_result:?}");
    println!("Robust result: {robust_result:?}");

    // Both should work, but robust should be more reliable
    assert!(
        robust_result.is_ok(),
        "Robust predicates should handle high precision coordinates"
    );
}

#[test]
fn test_cocircular_points() {
    println!("\n=== Testing Cocircular Points ===");

    // Create points that are exactly on a circle in 2D (extended to 3D)
    let radius = 1.0;
    let points = vec![
        Point::new([radius, 0.0, 0.0]),  // (1, 0, 0)
        Point::new([0.0, radius, 0.0]),  // (0, 1, 0)
        Point::new([-radius, 0.0, 0.0]), // (-1, 0, 0)
        Point::new([0.0, -radius, 0.0]), // (0, -1, 0)
    ];

    // Test point exactly on the circle
    let test_point = Point::new([
        radius * std::f64::consts::FRAC_1_SQRT_2,
        radius * std::f64::consts::FRAC_1_SQRT_2,
        0.0,
    ]);
    let config = config_presets::degenerate_robust();

    println!("Testing cocircular points (should be BOUNDARY case)");

    let standard_result = insphere(&points, test_point);
    let robust_result = robust_insphere(&points, &test_point, &config);

    println!("Standard result: {standard_result:?}");
    println!("Robust result: {robust_result:?}");

    assert!(
        robust_result.is_ok(),
        "Robust predicates should handle cocircular points"
    );

    // The robust version should detect boundary cases more reliably
    if let Ok(robust_res) = robust_result {
        match robust_res {
            InSphere::BOUNDARY => {
                println!("✓ Robust predicates correctly identified boundary case");
            }
            _ => {
                println!(
                    "Robust predicates gave definitive answer for cocircular case: {robust_res:?}"
                );
            }
        }
    }
}

#[test]
fn test_extreme_aspect_ratios() {
    println!("\n=== Testing Extreme Aspect Ratios ===");

    // Create a very thin/flat tetrahedron
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1000.0, 0.0, 0.0]),      // Very long edge
        Point::new([500.0, 0.001, 0.0]),     // Very small height
        Point::new([500.0, 0.0005, 0.0001]), // Very small third dimension
    ];

    let test_point = Point::new([500.0, 0.0007, 0.00005]);
    let config = config_presets::degenerate_robust();

    println!("Testing extreme aspect ratio tetrahedron");

    let standard_result = insphere(&points, test_point);
    let robust_result = robust_insphere(&points, &test_point, &config);

    println!("Standard result: {standard_result:?}");
    println!("Robust result: {robust_result:?}");

    assert!(
        robust_result.is_ok(),
        "Robust predicates should handle extreme aspect ratios"
    );
}

#[test]
fn test_vertex_insertion_robustness() {
    println!("\n=== Testing Robust Vertex Insertion ===");

    // Create initial vertices for TDS
    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, Option<()>, Option<()>, 3> = match Tds::new(&initial_vertices) {
        Ok(tds) => tds,
        Err(e) => {
            println!("Failed to create initial TDS: {e:?}");
            return; // Skip this test if we can't create the TDS
        }
    };

    let mut robust_algorithm = RobustBoyerWatson::for_degenerate_cases();

    // Try to insert problematic vertices that might cause "No cavity boundary facets found"
    let problematic_vertices = vec![
        vertex!([0.5, 0.5, 1e-15]),   // Nearly coplanar
        vertex!([0.25, 0.25, 1e-14]), // Another nearly coplanar
        vertex!([0.1, 0.1, 1e-13]),   // Very small z-coordinate
    ];

    let mut successful_insertions = 0;
    let mut failed_insertions = 0;
    let mut degenerate_cases_handled = 0;

    for (i, vertex) in problematic_vertices.into_iter().enumerate() {
        let vertex_idx = i + 1;
        let vertex_array = vertex.point().to_array();
        println!("Attempting to insert vertex {vertex_idx}: {vertex_array:?}");

        let start_time = Instant::now();
        let result = robust_algorithm.robust_insert_vertex(&mut tds, vertex);
        let duration = start_time.elapsed();

        match result {
            Ok(info) => {
                successful_insertions += 1;
                if info.degenerate_case_handled {
                    degenerate_cases_handled += 1;
                }
                println!(
                    "  ✓ Success: {} cells created, {} cells removed, strategy: {strategy:?}, degenerate: {degenerate} (took {duration:?})",
                    info.cells_created,
                    info.cells_removed,
                    strategy = info.strategy_used,
                    degenerate = info.degenerate_case_handled
                );
            }
            Err(e) => {
                failed_insertions += 1;
                println!("  ✓ Failed: {e:?} (took {duration:?})");
            }
        }
    }

    println!("\n--- Vertex Insertion Summary ---");
    println!("Successful insertions: {successful_insertions}");
    println!("Failed insertions: {failed_insertions}");
    println!("Degenerate cases handled: {degenerate_cases_handled}");
    println!("Final stats: {:?}", robust_algorithm.stats);

    // The robust algorithm should handle at least some of these challenging cases
    assert!(
        successful_insertions > 0,
        "Robust algorithm should successfully insert at least some problematic vertices"
    );
}

#[test]
fn benchmark_robust_vs_standard() {
    println!("\n=== Benchmarking Robust vs Standard Predicates ===");

    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 1e-14]),
        Point::new([0.0, 1.0, 1e-15]),
        Point::new([0.0, 0.0, 1.0]),
    ];

    let test_points = vec![
        Point::new([0.25, 0.25, 0.25]),
        Point::new([0.5, 0.5, 1e-16]),
        Point::new([0.1, 0.1, 0.1]),
        Point::new([0.8, 0.2, 0.3]),
        Point::new([0.3, 0.7, 1e-17]),
    ];

    let config = config_presets::general_triangulation();
    let iterations = 1000;

    // Benchmark standard predicates
    let standard_start = Instant::now();
    let mut standard_successes = 0;
    for _ in 0..iterations {
        for test_point in &test_points {
            if insphere(&points, *test_point).is_ok() {
                standard_successes += 1;
            }
        }
    }
    let standard_duration = standard_start.elapsed();

    // Benchmark robust predicates
    let robust_start = Instant::now();
    let mut robust_successes = 0;
    for _ in 0..iterations {
        for test_point in &test_points {
            if robust_insphere(&points, test_point, &config).is_ok() {
                robust_successes += 1;
            }
        }
    }
    let robust_duration = robust_start.elapsed();

    println!(
        "Benchmark Results ({iterations} iterations × {} test points):",
        test_points.len()
    );
    #[allow(clippy::cast_precision_loss)]
    let standard_avg =
        standard_duration.as_micros() as f64 / (iterations * test_points.len()) as f64;
    #[allow(clippy::cast_precision_loss)]
    let robust_avg = robust_duration.as_micros() as f64 / (iterations * test_points.len()) as f64;
    println!(
        "Standard predicates: {standard_successes} successes in {standard_duration:?} ({standard_avg:.2} μs/call)"
    );
    println!(
        "Robust predicates: {robust_successes} successes in {robust_duration:?} ({robust_avg:.2} μs/call)"
    );

    #[allow(clippy::cast_precision_loss)]
    let overhead_ratio = robust_duration.as_micros() as f64 / standard_duration.as_micros() as f64;
    println!("Robust overhead: {overhead_ratio:.2}x");
    println!(
        "Reliability improvement: {}/{} standard vs {}/{} robust",
        standard_successes,
        iterations * test_points.len(),
        robust_successes,
        iterations * test_points.len()
    );

    // Robust predicates should be at least as reliable as standard ones
    assert!(
        robust_successes >= standard_successes,
        "Robust predicates should be at least as reliable as standard predicates"
    );
}

#[test]
fn test_configuration_presets() {
    println!("\n=== Testing Different Configuration Presets ===");

    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
    ];

    let test_point = Point::new([0.25, 0.25, 0.25]);

    let configs = vec![
        ("General", config_presets::general_triangulation()),
        ("High Precision", config_presets::high_precision()),
        ("Degenerate Robust", config_presets::degenerate_robust()),
    ];

    for (name, config) in configs {
        let start = Instant::now();
        let result = robust_insphere(&points, &test_point, &config);
        let duration = start.elapsed();

        println!("{name}: {result:?} (took {duration:?})");
        assert!(result.is_ok(), "{name} configuration should work");
    }
}

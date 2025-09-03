//! Simple Memory Analysis Example
//!
//! This example demonstrates basic memory usage analysis for Delaunay triangulations
//! using the existing allocation counter infrastructure from the tests.

use delaunay::geometry::algorithms::ConvexHull;
use delaunay::prelude::*;
use delaunay::vertex;
use rand::Rng;
use std::time::Instant;

/// Create test helper that mimics the existing test infrastructure  
#[cfg(feature = "count-allocations")]
fn measure_with_result<F, R>(f: F) -> (R, allocation_counter::AllocationInfo)
where
    F: FnOnce() -> R,
{
    let mut result: Option<R> = None;
    let info = allocation_counter::measure(|| {
        result = Some(f());
    });
    (result.expect("Closure should have set result"), info)
}

#[cfg(not(feature = "count-allocations"))]
fn measure_with_result<F, R>(f: F) -> (R, ())
where
    F: FnOnce() -> R,
{
    (f(), ())
}

#[allow(unused_variables)] // tri_info and hull_info are used conditionally based on count-allocations feature
fn analyze_triangulation_memory(points: &[Point<f64, 3>]) {
    println!("  Analyzing 3D triangulation with {} points", points.len());

    let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();

    // Measure triangulation construction
    let start = Instant::now();
    let (tds, tri_info) = measure_with_result(|| Tds::<f64, (), (), 3>::new(&vertices).unwrap());
    let construction_time = start.elapsed();

    let num_vertices = tds.number_of_vertices();
    let num_cells = tds.number_of_cells();

    // Measure convex hull extraction
    let start = Instant::now();
    let (hull, hull_info) = measure_with_result(|| ConvexHull::from_triangulation(&tds).unwrap());
    let hull_time = start.elapsed();

    let hull_facets = hull.facet_count();

    // Print results
    println!("    Triangulation: {num_vertices} vertices, {num_cells} cells");
    println!("    Convex hull: {hull_facets} facets");
    println!("    Construction time: {construction_time:?}");
    println!("    Hull extraction time: {hull_time:?}");

    #[cfg(feature = "count-allocations")]
    {
        #[allow(clippy::cast_precision_loss)]
        {
            let tri_kb = tri_info.bytes_total as f64 / 1024.0;
            let hull_kb = hull_info.bytes_total as f64 / 1024.0;
            let bytes_per_vertex = tri_info.bytes_total as f64 / num_vertices as f64;
            let hull_ratio = hull_info.bytes_total as f64 / tri_info.bytes_total as f64 * 100.0;

            println!(
                "    Triangulation memory: {tri_kb:.1} KB ({bytes_per_vertex:.0} bytes/vertex)"
            );
            println!("    Hull memory: {hull_kb:.1} KB ({hull_ratio:.1}% of triangulation)");
        }
    }

    #[cfg(not(feature = "count-allocations"))]
    println!(
        "    (Memory tracking disabled - use --features count-allocations for detailed analysis)"
    );

    println!();
}

fn generate_random_points_3d(n: usize) -> Vec<Point<f64, 3>> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect()
}

fn main() {
    println!("Simple Memory Analysis for Delaunay Triangulations");
    println!("==================================================");

    #[cfg(feature = "count-allocations")]
    println!("✓ Allocation counter enabled - detailed memory tracking available");

    #[cfg(not(feature = "count-allocations"))]
    {
        println!("⚠ Allocation counter disabled - only basic metrics available");
        println!("  Run with --features count-allocations for detailed analysis");
    }

    println!();

    // Test different point counts to show scaling behavior
    let point_counts = [10, 20, 30];

    for &n_points in &point_counts {
        println!("=== Analysis for {n_points} Points ===");

        let points = generate_random_points_3d(n_points);
        analyze_triangulation_memory(&points);
    }

    println!("=== Key Insights ===");
    println!("• Memory usage scales roughly O(n log n) for 3D triangulations");
    println!("• Convex hulls typically use 10-30% of triangulation memory");
    println!("• Hull extraction is much faster than triangulation construction");
    println!("• Use --features count-allocations to see detailed allocation metrics");

    println!("\nFor comprehensive scaling analysis, run:");
    println!("  cargo bench --bench memory_scaling --features count-allocations");
    println!("  cargo bench --bench convex_hull_memory --features count-allocations");
}

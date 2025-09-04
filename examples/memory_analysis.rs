//! Simple Memory Analysis Example
//!
//! This example demonstrates basic memory usage analysis for Delaunay triangulations
//! using the existing allocation counter infrastructure from the tests.

use delaunay::geometry::algorithms::ConvexHull;
use delaunay::geometry::util::generate_random_points;
use delaunay::prelude::*;
use delaunay::vertex;
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

/// Macro to generate dimension-specific memory analysis functions
macro_rules! generate_memory_analysis {
    ($name:ident, $dim:literal) => {
        #[allow(unused_variables)] // tri_info and hull_info are used conditionally based on count-allocations feature
        fn $name(points: &[Point<f64, $dim>]) {
            println!("  Analyzing {}D triangulation with {} points", $dim, points.len());

            let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();

            // Measure triangulation construction
            let start = Instant::now();
            let (tds, tri_info) = measure_with_result(|| {
                Tds::<f64, (), (), $dim>::new(&vertices).expect("failed to build triangulation")
            });
            let construction_time = start.elapsed();

            let num_vertices = tds.number_of_vertices();
            let num_cells = tds.number_of_cells();

            // Measure convex hull extraction
            let start = Instant::now();
            let (hull, hull_info) = measure_with_result(|| {
                ConvexHull::from_triangulation(&tds)
                    .expect("failed to construct convex hull from triangulation")
            });
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
                    let tri_bytes = tri_info.bytes_total as f64;
                    let hull_bytes = hull_info.bytes_total as f64;
                    let tri_kb = tri_bytes / 1024.0;
                    let hull_kb = hull_bytes / 1024.0;
                    if tri_info.bytes_total > 0 && num_vertices > 0 {
                        let bytes_per_vertex = tri_bytes / num_vertices as f64;
                        let hull_ratio = (hull_bytes / tri_bytes) * 100.0;
                        println!("    Triangulation memory: {tri_kb:.1} KB ({bytes_per_vertex:.0} bytes/vertex)");
                        println!("    Hull memory: {hull_kb:.1} KB ({hull_ratio:.1}% of triangulation)");
                    } else {
                        println!("    Triangulation memory: {tri_kb:.1} KB");
                        println!("    Hull memory: {hull_kb:.1} KB");
                    }
                }
            }

            #[cfg(not(feature = "count-allocations"))]
            println!(
                "    (Memory tracking disabled - use --features count-allocations for detailed analysis)"
            );

            println!();
        }
    };
}

// Generate memory analysis functions for all dimensions
generate_memory_analysis!(analyze_triangulation_memory_2d, 2);
generate_memory_analysis!(analyze_triangulation_memory_3d, 3);
generate_memory_analysis!(analyze_triangulation_memory_4d, 4);
generate_memory_analysis!(analyze_triangulation_memory_5d, 5);

fn main() {
    println!("Memory Analysis for Delaunay Triangulations Across Dimensions");
    println!("=============================================================");

    #[cfg(feature = "count-allocations")]
    println!("✓ Allocation counter enabled - detailed memory tracking available");

    #[cfg(not(feature = "count-allocations"))]
    {
        println!("⚠ Allocation counter disabled - only basic metrics available");
        println!("  Run with --features count-allocations for detailed analysis");
    }

    println!();

    // Test with a moderate number of points across all dimensions
    let n_points = 25;
    let range = (-50.0, 50.0);

    println!("=== Memory Analysis with {n_points} Points ===");
    println!();

    // 2D Analysis
    println!("--- 2D Triangulation ---");
    let points_2d =
        generate_random_points::<f64, 2>(n_points, range).expect("Failed to generate 2D points");
    analyze_triangulation_memory_2d(&points_2d);

    // 3D Analysis
    println!("--- 3D Triangulation ---");
    let points_3d =
        generate_random_points::<f64, 3>(n_points, range).expect("Failed to generate 3D points");
    analyze_triangulation_memory_3d(&points_3d);

    // 4D Analysis
    println!("--- 4D Triangulation ---");
    let points_4d =
        generate_random_points::<f64, 4>(n_points, range).expect("Failed to generate 4D points");
    analyze_triangulation_memory_4d(&points_4d);

    // 5D Analysis
    println!("--- 5D Triangulation ---");
    let points_5d =
        generate_random_points::<f64, 5>(n_points, range).expect("Failed to generate 5D points");
    analyze_triangulation_memory_5d(&points_5d);

    println!("=== Key Insights (empirical) ===");
    println!(
        "• On random 3D inputs, memory tends to scale between O(n) and O(n log n), distribution-dependent"
    );
    println!(
        "• Convex hull memory is often a fraction of triangulation memory (ballpark 10–30%, varies)"
    );
    println!("• Hull extraction is typically faster than triangulation construction");
    println!("• Use --features count-allocations to see detailed allocation metrics");

    println!("\nFor comprehensive scaling analysis, run:");
    println!("  cargo bench --bench memory_scaling --features count-allocations");
    println!("  cargo bench --bench triangulation_vs_hull_memory --features count-allocations");
}

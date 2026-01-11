//! Simple Memory Analysis Example
//!
//! This example demonstrates basic memory usage analysis for Delaunay triangulations
//! using the existing allocation counter infrastructure from the tests.

use delaunay::prelude::query::*;
use std::time::Instant;

/// Bounds for random triangulation (min, max) - consistent with benchmarks
const BOUNDS: (f64, f64) = (-100.0, 100.0);

/// Macro to generate dimension-specific memory analysis functions
macro_rules! generate_memory_analysis {
    ($name:ident, $dim:literal) => {
        #[cfg_attr(
            not(feature = "count-allocations"),
            expect(
                unused_variables,
                reason = "tri_info and hull_info are only used when the count-allocations feature is enabled",
            )
        )]
        fn $name(n_points: usize, seed: u64) {
            println!("  Analyzing {}D triangulation with {} points", $dim, n_points);

            // Measure triangulation construction
            let start = Instant::now();
            let (dt_res, tri_info) = measure_with_result(|| {
                generate_random_triangulation::<f64, (), (), $dim>(
                    n_points,
                    BOUNDS,
                    None,
                    Some(seed),
                )
            });
            let dt = match dt_res {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("✗ Failed to build triangulation: {e}");
                    return;
                }
            };
            let construction_time = start.elapsed();

            let num_vertices = dt.tds().number_of_vertices();
            let num_cells = dt.tds().number_of_cells();

            // Measure convex hull extraction
            let start = Instant::now();
            let (hull_res, hull_info) = measure_with_result(|| {
                ConvexHull::from_triangulation(dt.as_triangulation())
            });
            let hull = match hull_res {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("✗ Failed to construct convex hull from triangulation: {e}");
                    return;
                }
            };
            let hull_time = start.elapsed();

            let hull_facets = hull.number_of_facets();

            // Print results
            println!("    Triangulation: {num_vertices} vertices, {num_cells} cells");
            println!("    Convex hull: {hull_facets} facets");
            println!("    Construction time: {construction_time:?}");
            println!("    Hull extraction time: {hull_time:?}");

            #[cfg(feature = "count-allocations")]
            {
                #[expect(clippy::cast_precision_loss, reason = "Converting byte counters to floating-point for human-friendly KiB/MiB output")]
                {
                    let tri_bytes = tri_info.bytes_total as f64;
                    let hull_bytes = hull_info.bytes_total as f64;
                    let tri_kb = tri_bytes / 1024.0;
                    let hull_kb = hull_bytes / 1024.0;
                    let tri_mib = tri_kb / 1024.0;
                    let hull_mib = hull_kb / 1024.0;
                    if tri_info.bytes_total > 0 && num_vertices > 0 {
                        let bytes_per_vertex = tri_bytes / num_vertices as f64;
                        let hull_ratio = (hull_bytes / tri_bytes) * 100.0;
                        if tri_kb >= 1024.0 {
                            println!("    Triangulation memory: {tri_kb:.1} KiB ({tri_mib:.2} MiB, {bytes_per_vertex:.0} bytes/vertex)");
                        } else {
                            println!("    Triangulation memory: {tri_kb:.1} KiB ({bytes_per_vertex:.0} bytes/vertex)");
                        }
                        if hull_kb >= 1024.0 {
                            println!("    Hull memory: {hull_kb:.1} KiB ({hull_mib:.2} MiB, {hull_ratio:.1}% of triangulation)");
                        } else {
                            println!("    Hull memory: {hull_kb:.1} KiB ({hull_ratio:.1}% of triangulation)");
                        }
                    } else {
                        if tri_kb >= 1024.0 {
                            println!("    Triangulation memory: {tri_kb:.1} KiB ({tri_mib:.2} MiB)");
                        } else {
                            println!("    Triangulation memory: {tri_kb:.1} KiB");
                        }
                        if hull_kb >= 1024.0 {
                            println!("    Hull memory: {hull_kb:.1} KiB ({hull_mib:.2} MiB)");
                        } else {
                            println!("    Hull memory: {hull_kb:.1} KiB");
                        }
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

    println!("=== Memory Analysis with {n_points} Points ===");
    println!();

    // 2D Analysis with seed for reproducibility
    println!("--- 2D Triangulation ---");
    analyze_triangulation_memory_2d(n_points, 12345);

    // 3D Analysis with different seed
    println!("--- 3D Triangulation ---");
    analyze_triangulation_memory_3d(n_points, 23456);

    // 4D Analysis with different seed
    println!("--- 4D Triangulation ---");
    analyze_triangulation_memory_4d(n_points, 34567);

    // 5D Analysis with different seed
    println!("--- 5D Triangulation ---");
    analyze_triangulation_memory_5d(n_points, 45678);

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
    println!(
        "  cargo bench --bench profiling_suite --features count-allocations -- memory_profiling"
    );
    println!("  cargo bench --bench triangulation_vs_hull_memory --features count-allocations");
}

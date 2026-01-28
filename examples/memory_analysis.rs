//! Simple Memory Analysis Example
//!
//! This example demonstrates basic memory usage analysis for Delaunay triangulations
//! using the existing allocation counter infrastructure from the tests.

use delaunay::prelude::query::*;
use std::any::Any;
use std::backtrace::Backtrace;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Once;
use std::time::Instant;
use tracing::{error, warn};

const SEED_CANDIDATES: &[u64] = &[1, 7, 11, 42, 99, 123, 666];
const SEED_CANDIDATES_4D: &[u64] = &[777];
const SEED_CANDIDATES_5D: &[u64] = &[888];
const POINT_COUNT_CANDIDATES_2D_3D: &[usize] = &[25, 20, 16, 12];
const POINT_COUNT_CANDIDATES_4D: &[usize] = &[12];
const POINT_COUNT_CANDIDATES_5D: &[usize] = &[10];

/// Bounds for random triangulation (min, max) - consistent with benchmarks
const BOUNDS: (f64, f64) = (-100.0, 100.0);

fn init_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    });
}

fn format_panic_payload(panic: &(dyn Any + Send)) -> String {
    panic.downcast_ref::<&str>().map_or_else(
        || {
            panic
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "unknown panic payload".to_string())
        },
        |message| (*message).to_string(),
    )
}

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
        #[expect(
            clippy::too_many_lines,
            reason = "Example keeps analysis flow in one function for readability"
        )]
        fn $name(point_counts: &[usize], seeds: &[u64]) {
            for &n_points in point_counts {
                println!("  Analyzing {}D triangulation with {} points", $dim, n_points);

                // Measure triangulation construction
                let start = Instant::now();
                let mut last_error: Option<String> = None;
                let mut used_seed: Option<u64> = None;
                let mut dt_res: Option<_> = None;
                let mut tri_info = None;
                for &candidate in seeds {
                    let candidate_result = catch_unwind(AssertUnwindSafe(|| {
                        measure_with_result(|| {
                            generate_random_triangulation::<f64, (), (), $dim>(
                                n_points,
                                BOUNDS,
                                None,
                                Some(candidate),
                            )
                        })
                    }));
                    match candidate_result {
                        Ok((candidate_res, candidate_info)) => match candidate_res {
                            Ok(dt) => {
                                dt_res = Some(dt);
                                tri_info = Some(candidate_info);
                                used_seed = Some(candidate);
                                break;
                            }
                            Err(e) => {
                                warn!(
                                    dim = $dim,
                                    points = n_points,
                                    seed = candidate,
                                    error = %e,
                                    "Seed failed to build triangulation"
                                );
                                last_error = Some(format!("{e}"));
                            }
                        },
                        Err(panic) => {
                            let payload = format_panic_payload(panic.as_ref());
                            let backtrace = Backtrace::force_capture();
                            error!(
                                dim = $dim,
                                points = n_points,
                                seed = candidate,
                                payload = %payload,
                                backtrace = %backtrace,
                                "Panic while building triangulation"
                            );
                            std::panic::resume_unwind(panic);
                        }
                    }
                }

                let (Some(dt), Some(tri_info)) = (dt_res, tri_info) else {
                    error!(
                        dim = $dim,
                        points = n_points,
                        seeds = ?seeds,
                        last_error = %last_error.unwrap_or_else(|| "unknown error".to_string()),
                        "Failed to build triangulation after trying all seeds"
                    );
                    println!();
                    continue;
                };
                let construction_time = start.elapsed();

            let num_vertices = dt.tds().number_of_vertices();
            let num_cells = dt.tds().number_of_cells();

            // Measure convex hull extraction
            let start = Instant::now();
            let hull_result = catch_unwind(AssertUnwindSafe(|| {
                measure_with_result(|| ConvexHull::from_triangulation(dt.as_triangulation()))
            }));
            let (hull_res, hull_info) = match hull_result {
                Ok(result) => result,
                Err(panic) => {
                    let payload = format_panic_payload(panic.as_ref());
                    let backtrace = Backtrace::force_capture();
                    error!(
                        dim = $dim,
                        points = n_points,
                        seed = used_seed,
                        payload = %payload,
                        backtrace = %backtrace,
                        "Panic while extracting convex hull"
                    );
                    std::panic::resume_unwind(panic);
                }
            };
            let hull = match hull_res {
                Ok(h) => h,
                Err(e) => {
                    error!(
                        dim = $dim,
                        points = n_points,
                        seed = used_seed,
                        error = %e,
                        "Failed to construct convex hull from triangulation"
                    );
                    return;
                }
            };
            let hull_time = start.elapsed();

            let hull_facets = hull.number_of_facets();

            // Print results
                println!("    Triangulation: {num_vertices} vertices, {num_cells} cells");
                println!("    Convex hull: {hull_facets} facets");
                if let Some(seed) = used_seed {
                    println!("    Seed: {seed}");
                }
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
                return;
            }

            eprintln!(
                "✗ Unable to build a {}D triangulation after trying point counts {point_counts:?}",
                $dim
            );
        }
    };
}

// Generate memory analysis functions for all dimensions
generate_memory_analysis!(analyze_triangulation_memory_2d, 2);
generate_memory_analysis!(analyze_triangulation_memory_3d, 3);
generate_memory_analysis!(analyze_triangulation_memory_4d, 4);
generate_memory_analysis!(analyze_triangulation_memory_5d, 5);

fn main() {
    init_tracing();
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

    let primary_points = POINT_COUNT_CANDIDATES_2D_3D.first().copied().unwrap_or(0);
    println!("=== Memory Analysis with {primary_points} Points ===");
    println!();

    // 2D Analysis with seed for reproducibility
    println!("--- 2D Triangulation ---");
    analyze_triangulation_memory_2d(POINT_COUNT_CANDIDATES_2D_3D, SEED_CANDIDATES);

    // 3D Analysis
    println!("--- 3D Triangulation ---");
    analyze_triangulation_memory_3d(POINT_COUNT_CANDIDATES_2D_3D, SEED_CANDIDATES);

    // 4D Analysis
    println!("--- 4D Triangulation ---");
    analyze_triangulation_memory_4d(POINT_COUNT_CANDIDATES_4D, SEED_CANDIDATES_4D);

    // 5D Analysis
    println!("--- 5D Triangulation ---");
    analyze_triangulation_memory_5d(POINT_COUNT_CANDIDATES_5D, SEED_CANDIDATES_5D);

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

//! Focused showcase demonstrating specific cases where robust predicates succeed
//! where standard predicates fail, directly addressing "No cavity boundary facets found" errors.

use delaunay::core::{
    algorithms::robust_bowyer_watson::RobustBowyerWatson, triangulation_data_structure::Tds,
};
use delaunay::geometry::{
    point::Point,
    predicates::insphere,
    robust_predicates::{config_presets, robust_insphere},
};
use delaunay::prelude::*;
use delaunay::vertex;
use num_traits::cast;
use std::time::Instant;

/// This test demonstrates the exact problem we're solving: standard predicates
/// failing on degenerate cases that would lead to triangulation failures.
#[test]
#[expect(clippy::too_many_lines)]
fn showcase_degenerate_failure_recovery() {
    println!("🎯 SHOWCASE: Robust Predicates Solving Real Failures");
    println!("{}", "=".repeat(50));

    // These are the exact kinds of configurations that cause
    // "No cavity boundary facets found" errors in Delaunay triangulation
    let problem_cases = vec![
        (
            "Nearly Coplanar Tetrahedron",
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 1e-16]),   // Almost zero z
                Point::new([0.5, 0.866, 1e-17]), // Almost zero z
                Point::new([0.5, 0.289, 1e-15]), // Almost zero z
            ],
            Point::new([0.5, 0.433, 1e-18]), // Test point almost coplanar
        ),
        (
            "Cocircular Points (Boundary Case)",
            vec![
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([-1.0, 0.0, 0.0]),
                Point::new([0.0, -1.0, 0.0]),
            ],
            Point::new([
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
            ]), // Exactly on circle
        ),
        (
            "High Precision Numerical Instability",
            vec![
                Point::new([1.000_000_000_000_000_0, 0.0, 0.0]),
                Point::new([0.000_000_000_000_000_1, 1.0, 0.0]),
                Point::new([0.0, 0.000_000_000_000_000_2, 1.0]),
                Point::new([
                    0.333_333_333_333_333_3,
                    0.333_333_333_333_333_3,
                    0.333_333_333_333_333_4,
                ]),
            ],
            Point::new([0.250_000_000_000_000_1, 0.249_999_999_999_999_9, 0.25]),
        ),
        (
            "Extreme Aspect Ratio (Needle Case)",
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1e6, 0.0, 0.0]),   // Very long edge
                Point::new([5e5, 1e-6, 0.0]),  // Very thin
                Point::new([5e5, 5e-7, 1e-9]), // Very thin in 3D
            ],
            Point::new([5e5, 7e-7, 5e-10]), // Inside the thin region
        ),
    ];

    let config = config_presets::degenerate_robust();
    let mut total_standard_failures = 0;
    let mut total_robust_successes = 0;

    for (case_name, points, test_point) in problem_cases {
        println!("\n📋 Testing: {case_name}");
        println!(
            "   Points: {:?}",
            points.iter().map(Point::to_array).collect::<Vec<_>>()
        );
        println!("   Test point: {:?}", test_point.to_array());

        // Test standard predicates
        let standard_start = Instant::now();
        let standard_result = insphere(&points, test_point);
        let standard_time = standard_start.elapsed();

        // Test robust predicates
        let robust_start = Instant::now();
        let robust_result = robust_insphere(&points, &test_point, &config);
        let robust_time = robust_start.elapsed();

        println!("   ⚡ Standard: {standard_result:?} (⏱️  {standard_time:?})");
        println!("   🛡️  Robust:   {robust_result:?} (⏱️  {robust_time:?})");

        match (&standard_result, &robust_result) {
            (Err(_), Ok(_)) => {
                println!("   ✅ PROBLEM SOLVED: Robust succeeded where standard failed!");
                total_standard_failures += 1;
                total_robust_successes += 1;
            }
            (Ok(std_res), Ok(rob_res)) => {
                if std_res == rob_res {
                    println!("   ✅ Both methods agreed: {std_res:?}");
                } else {
                    println!("   ⚠️  DIFFERENT RESULTS: Standard={std_res:?}, Robust={rob_res:?}");
                }
                total_robust_successes += 1;
            }
            (Ok(_), Err(_)) => {
                println!("   ❌ UNEXPECTED: Standard succeeded but robust failed");
            }
            (Err(_), Err(_)) => {
                println!("   ❌ Both methods failed");
                total_standard_failures += 1;
            }
        }
    }

    println!("\n{}", "=".repeat(50));
    println!("🏆 FINAL SCORE:");
    println!("   Standard failures: {total_standard_failures}/4");
    println!("   Robust successes:  {total_robust_successes}/4");

    if total_standard_failures > 0 && total_robust_successes == 4 {
        println!(
            "   🎉 ROBUST PREDICATES WIN! Solved {total_standard_failures} cases that would cause triangulation failures!"
        );
    }

    // The robust predicates should handle all cases
    assert_eq!(
        total_robust_successes, 4,
        "Robust predicates should handle all test cases"
    );
    assert!(
        total_standard_failures > 0,
        "We should have demonstrated at least one case where standard predicates fail"
    );
}

/// Stress test to find the breaking point and demonstrate robustness improvement
#[test]
fn stress_test_tolerance_limits() {
    println!("\n🔬 STRESS TEST: Finding Tolerance Breaking Points");
    println!("{}", "=".repeat(50));

    let base_points = [
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.5, 0.5, 0.0]),
    ];

    // Test with increasingly small perturbations
    let epsilons = vec![1e-10, 1e-12, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18];

    let config = config_presets::degenerate_robust();
    let mut standard_breaking_point = None;
    let mut robust_breaking_point = None;

    println!("Testing with progressively smaller perturbations...\n");

    for &eps in &epsilons {
        let points = vec![
            base_points[0],
            base_points[1],
            base_points[2],
            Point::new([0.5, 0.5, eps]), // Tiny z-coordinate
        ];

        let test_point = Point::new([0.25, 0.25, eps * 0.1]);

        let standard_result = insphere(&points, test_point);
        let robust_result = robust_insphere(&points, &test_point, &config);

        println!("ε = {eps:.0e}:");
        println!(
            "  Standard: {:?}",
            standard_result
                .as_ref()
                .map_or_else(|e| format!("FAILED: {e}"), |r| format!("{r:?}"))
        );
        println!(
            "  Robust:   {:?}",
            robust_result
                .as_ref()
                .map_or_else(|e| format!("FAILED: {e}"), |r| format!("{r:?}"))
        );

        if standard_result.is_err() && standard_breaking_point.is_none() {
            standard_breaking_point = Some(eps);
            println!("  🔥 STANDARD BREAKS HERE!");
        }

        if robust_result.is_err() && robust_breaking_point.is_none() {
            robust_breaking_point = Some(eps);
            println!("  ⚠️  ROBUST BREAKS HERE!");
        }

        println!();
    }

    match (standard_breaking_point, robust_breaking_point) {
        (Some(std_break), Some(rob_break)) => {
            println!("📊 BREAKING POINT ANALYSIS:");
            println!("   Standard predicates break at ε = {std_break:.0e}");
            println!("   Robust predicates break at   ε = {rob_break:.0e}");
            if rob_break < std_break {
                println!(
                    "   🏆 Robust predicates are {:.1e}x more tolerant!",
                    std_break / rob_break
                );
            }
        }
        (Some(std_break), None) => {
            println!(
                "📊 RESULT: Standard breaks at ε = {std_break:.0e}, but robust handles all cases!"
            );
            println!("   🏆 ROBUST PREDICATES WIN!");
        }
        (None, Some(rob_break)) => {
            println!(
                "📊 UNEXPECTED: Robust breaks at ε = {rob_break:.0e}, but standard handles all cases"
            );
        }
        (None, None) => {
            println!("📊 Both methods handled all test cases (need more extreme tests)");
        }
    }
}

/// Performance comparison showing the cost/benefit trade-off
#[test]
#[expect(clippy::too_many_lines)]
fn performance_cost_benefit_analysis() {
    println!("\n⚡ PERFORMANCE ANALYSIS: Cost vs Benefit");
    println!("{}", "=".repeat(50));

    // Regular case - both should work fine
    let regular_points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
    ];
    let regular_test = Point::new([0.25, 0.25, 0.25]);

    // Problematic case - standard likely to fail
    let problem_points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 1e-16]),
        Point::new([0.0, 1.0, 1e-17]),
        Point::new([0.5, 0.5, 1e-15]),
    ];
    let problem_test = Point::new([0.25, 0.25, 1e-18]);

    let config = config_presets::general_triangulation();
    let iterations = 10000;

    println!("Testing with {iterations} iterations each...\n");

    // Test regular case
    println!("📈 REGULAR CASE (both methods should work):");

    let mut std_regular_successes = 0;
    let mut std_regular_time = std::time::Duration::new(0, 0);

    for _ in 0..iterations {
        let start = Instant::now();
        if insphere(&regular_points, regular_test).is_ok() {
            std_regular_successes += 1;
        }
        std_regular_time += start.elapsed();
    }

    let mut rob_regular_successes = 0;
    let mut rob_regular_time = std::time::Duration::new(0, 0);

    for _ in 0..iterations {
        let start = Instant::now();
        if robust_insphere(&regular_points, &regular_test, &config).is_ok() {
            rob_regular_successes += 1;
        }
        rob_regular_time += start.elapsed();
    }

    #[expect(clippy::cast_precision_loss)]
    let std_avg = std_regular_time.as_nanos() as f64
        / <f64 as std::convert::From<_>>::from(iterations)
        / 1000.0;
    #[expect(clippy::cast_precision_loss)]
    let rob_avg = rob_regular_time.as_nanos() as f64
        / <f64 as std::convert::From<_>>::from(iterations)
        / 1000.0;
    println!("   Standard: {std_regular_successes}/{iterations} success, avg {std_avg:.2}μs");
    println!("   Robust:   {rob_regular_successes}/{iterations} success, avg {rob_avg:.2}μs");

    #[expect(clippy::cast_precision_loss)]
    let regular_overhead = rob_regular_time.as_nanos() as f64 / std_regular_time.as_nanos() as f64;
    println!("   Overhead: {regular_overhead:.2}x");

    // Test problematic case
    println!("\n🔥 PROBLEMATIC CASE (standard may fail):");

    let mut std_problem_successes = 0;
    let mut std_problem_time = std::time::Duration::new(0, 0);

    for _ in 0..iterations {
        let start = Instant::now();
        if insphere(&problem_points, problem_test).is_ok() {
            std_problem_successes += 1;
        }
        std_problem_time += start.elapsed();
    }

    let mut rob_problem_successes = 0;
    let mut rob_problem_time = std::time::Duration::new(0, 0);

    for _ in 0..iterations {
        let start = Instant::now();
        if robust_insphere(&problem_points, &problem_test, &config).is_ok() {
            rob_problem_successes += 1;
        }
        rob_problem_time += start.elapsed();
    }

    #[expect(clippy::cast_precision_loss)]
    let std_problem_avg = std_problem_time.as_nanos() as f64
        / <f64 as std::convert::From<_>>::from(iterations)
        / 1000.0;
    #[expect(clippy::cast_precision_loss)]
    let rob_problem_avg = rob_problem_time.as_nanos() as f64
        / <f64 as std::convert::From<_>>::from(iterations)
        / 1000.0;
    println!(
        "   Standard: {std_problem_successes}/{iterations} success, avg {std_problem_avg:.2}μs"
    );
    println!(
        "   Robust:   {rob_problem_successes}/{iterations} success, avg {rob_problem_avg:.2}μs"
    );

    let reliability_improvement = (<f64 as std::convert::From<_>>::from(rob_problem_successes)
        - <f64 as std::convert::From<_>>::from(std_problem_successes))
        / <f64 as std::convert::From<_>>::from(iterations)
        * 100.0;

    println!("\n🏆 COST-BENEFIT SUMMARY:");
    println!("   Regular case overhead: {regular_overhead:.2}x");
    println!(
        "   Reliability improvement: {reliability_improvement:.1}% more successes on hard cases"
    );

    if reliability_improvement > 0.0 {
        println!(
            "   💡 VERDICT: The {regular_overhead:.2}x overhead is worth it for {reliability_improvement:.1}% fewer failures!"
        );
    }

    // Robust should have equal or better success rate
    assert!(
        rob_problem_successes >= std_problem_successes,
        "Robust predicates should be at least as reliable as standard predicates"
    );
}

/// Show real triangulation scenario where this matters
#[test]
fn triangulation_scenario_demo() {
    println!("\n🏗️  REAL TRIANGULATION SCENARIO");
    println!("{}", "=".repeat(50));

    println!(
        "Simulating vertex insertion that commonly causes 'No cavity boundary facets found'..."
    );

    // Start with a basic tetrahedron
    let initial_vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let mut tds: Tds<f64, (), (), 3> =
        Tds::new(&initial_vertices).expect("Should be able to create basic tetrahedron");

    println!(
        "✅ Created initial tetrahedron with {} cells",
        tds.number_of_cells()
    );

    let mut robust_algorithm = RobustBowyerWatson::for_degenerate_cases();

    // These are the kinds of vertices that cause problems in real triangulations
    let challenging_insertions = vec![
        ("Nearly coplanar", vertex!([0.5, 0.5, 1e-15])),
        (
            "High precision",
            vertex!([
                0.333_333_333_333_333_3,
                0.333_333_333_333_333_4,
                0.333_333_333_333_333_5
            ]),
        ),
        ("Close to existing", vertex!([1e-14, 1e-14, 1e-14])),
        ("Boundary case", vertex!([0.5, 0.0, 0.5])),
        ("Another near-degenerate", vertex!([0.25, 0.25, 1e-16])),
    ];

    let total_cells_before = tds.number_of_cells();
    let mut successful_insertions = 0;

    for (description, vertex) in challenging_insertions {
        println!(
            "\n🔄 Inserting {} vertex: {:?}",
            description,
            vertex.point().to_array()
        );

        let cells_before = tds.number_of_cells();
        let start_time = Instant::now();

        match robust_algorithm.insert_vertex(&mut tds, vertex) {
            Ok(info) => {
                let duration = start_time.elapsed();
                successful_insertions += 1;

                println!("   ✅ SUCCESS in {duration:?}");
                println!("      Strategy: {:?}", info.strategy);
                let cells_created_i32 = cast(info.cells_created).unwrap_or(0i32);
                let cells_removed_i32 = cast(info.cells_removed).unwrap_or(0i32);
                let cell_change = cells_created_i32 - cells_removed_i32;
                println!(
                    "      Cells: {} → {} ({:+})",
                    cells_before,
                    cells_before + info.cells_created - info.cells_removed,
                    cell_change
                );

                if info.degenerate_case_handled {
                    println!("      🛡️  Handled degenerate case!");
                }
            }
            Err(e) => {
                let duration = start_time.elapsed();
                println!("   ❌ FAILED in {duration:?}: {e}");
            }
        }
    }

    let total_cells_after = tds.number_of_cells();

    println!("\n{}", "=".repeat(50));
    println!("🏁 TRIANGULATION COMPLETE:");
    println!("   Successful insertions: {successful_insertions}/5");
    let total_cells_after_i32 = cast(total_cells_after).unwrap_or(0i32);
    let total_cells_before_i32 = cast(total_cells_before).unwrap_or(0i32);
    let total_cell_change = total_cells_after_i32 - total_cells_before_i32;
    println!("   Total cells: {total_cells_before} → {total_cells_after} ({total_cell_change:+})");
    let (vertices_processed, cells_created, cells_removed) = robust_algorithm.get_statistics();
    println!(
        "   Algorithm stats: vertices_processed={vertices_processed}, cells_created={cells_created}, cells_removed={cells_removed}"
    );

    // This would commonly fail with standard Bowyer-Watson on such degenerate cases
    assert!(
        successful_insertions >= 3,
        "Should successfully insert most challenging vertices"
    );

    if successful_insertions == 5 {
        println!("   🎉 PERFECT SCORE! All challenging insertions succeeded!");
    } else {
        println!(
            "   ⚡ Still good - handled {successful_insertions}/5 challenging cases that often break standard algorithms"
        );
    }
}

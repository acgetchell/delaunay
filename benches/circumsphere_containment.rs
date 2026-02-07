//! Benchmarks for circumsphere containment algorithms.
//!
//! This benchmark suite compares the performance of three different algorithms for
//! determining whether a vertex is contained within the circumsphere of a simplex:
//!
//! 1. **insphere**: Standard determinant-based method (most numerically stable)
//! 2. **`insphere_distance`**: Distance-based method using explicit circumcenter calculation
//! 3. **`insphere_lifted`**: Matrix determinant method with lifted paraboloid approach
//!
//! The benchmarks include:
//! - Random query tests with multiple vertices
//! - Tests across different dimensions (2D, 3D, 4D, 5D)
//! - Edge case tests with boundary and distant vertices
//! - Numerical consistency validation between all three algorithms

use criterion::{Criterion, criterion_group, criterion_main};
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::query::*;
use std::hint::black_box;

/// Generate a standard D-dimensional simplex (D+1 vertices)
///
/// Creates a simplex with vertices at:
/// - Origin: [0, 0, ..., 0]
/// - Unit basis vectors: [1, 0, ..., 0], [0, 1, 0, ..., 0], ..., [0, 0, ..., 1]
fn standard_simplex<const D: usize>() -> Vec<Point<f64, D>> {
    let mut pts = Vec::with_capacity(D + 1);

    // Add origin
    pts.push(Point::new([0.0; D]));

    // Add unit basis vectors
    for i in 0..D {
        let mut coords = [0.0; D];
        coords[i] = 1.0;
        pts.push(Point::new(coords));
    }

    pts
}

/// Generate a random 3D simplex (tetrahedron) for benchmarking using seeded generation
fn generate_random_simplex_3d(seed: u64) -> Vec<Point<f64, 3>> {
    generate_random_points_seeded(4, (-10.0, 10.0), seed)
        .expect("Failed to generate random simplex points")
}

/// Generate a random 3D test point using seeded generation
fn generate_random_test_point_3d(seed: u64) -> Point<f64, 3> {
    generate_random_points_seeded(1, (-5.0, 5.0), seed)
        .expect("Failed to generate random test point")
        .into_iter()
        .next()
        .expect("Expected exactly one test point")
}

/// Benchmark with many random queries
fn benchmark_random_queries(c: &mut Criterion) {
    // Generate a fixed simplex for consistent benchmarking using seeded generation
    let simplex_points = generate_random_simplex_3d(42);

    // Generate many test points using seeded generation for reproducible results
    let test_points = generate_random_points_seeded(1000, (-5.0, 5.0), 123)
        .expect("Failed to generate random test points");

    c.bench_function("random/insphere_1000_queries", |b| {
        b.iter(|| {
            for test_point in &test_points {
                black_box(insphere(black_box(&simplex_points), black_box(*test_point)).unwrap());
            }
        });
    });

    c.bench_function("random/insphere_distance_1000_queries", |b| {
        b.iter(|| {
            for test_point in &test_points {
                black_box(
                    insphere_distance(black_box(&simplex_points), black_box(*test_point)).unwrap(),
                );
            }
        });
    });

    c.bench_function("random/insphere_lifted_1000_queries", |b| {
        b.iter(|| {
            for test_point in &test_points {
                black_box(
                    insphere_lifted(black_box(&simplex_points), black_box(*test_point)).unwrap(),
                );
            }
        });
    });
}

/// Macro to reduce duplication in benchmark functions
macro_rules! bench_simplex {
    ($c:ident, $dim:literal, $simplex:expr, $pt:expr) => {{
        $c.bench_function(concat!($dim, "d/insphere"), |b| {
            b.iter(|| black_box(insphere(black_box(&$simplex), black_box($pt)).unwrap()))
        });
        $c.bench_function(concat!($dim, "d/insphere_distance"), |b| {
            b.iter(|| black_box(insphere_distance(black_box(&$simplex), black_box($pt)).unwrap()))
        });
        $c.bench_function(concat!($dim, "d/insphere_lifted"), |b| {
            b.iter(|| black_box(insphere_lifted(black_box(&$simplex), black_box($pt)).unwrap()))
        });
    }};
}

/// Macro to reduce duplication in edge case benchmarks
macro_rules! bench_edge_case {
    ($c:ident, $dim:literal, $case:literal, $simplex:expr, $pt:expr) => {{
        $c.bench_function(
            concat!("edge_cases_", $dim, "d/", $case, "_insphere"),
            |b| b.iter(|| black_box(insphere(black_box(&$simplex), black_box($pt)).unwrap())),
        );
        $c.bench_function(
            concat!("edge_cases_", $dim, "d/", $case, "_distance"),
            |b| {
                b.iter(|| {
                    black_box(insphere_distance(black_box(&$simplex), black_box($pt)).unwrap())
                })
            },
        );
        $c.bench_function(concat!("edge_cases_", $dim, "d/", $case, "_lifted"), |b| {
            b.iter(|| black_box(insphere_lifted(black_box(&$simplex), black_box($pt)).unwrap()))
        });
    }};
}

/// Benchmark with different simplex sizes (2D, 3D, 4D, 5D)
fn benchmark_different_dimensions(c: &mut Criterion) {
    // 2D case - triangle in 2D space
    let simplex_2d = standard_simplex::<2>();
    let test_point_2d = Point::new([0.3, 0.3]);
    bench_simplex!(c, 2, simplex_2d, test_point_2d);

    // 3D case - tetrahedron in 3D space
    let simplex_3d = standard_simplex::<3>();
    let test_point_3d = Point::new([0.25, 0.25, 0.25]);
    bench_simplex!(c, 3, simplex_3d, test_point_3d);

    // 4D case - 4-simplex in 4D space
    let simplex_4d = standard_simplex::<4>();
    let test_point_4d = Point::new([0.2, 0.2, 0.2, 0.2]);
    bench_simplex!(c, 4, simplex_4d, test_point_4d);

    // 5D case - 5-simplex in 5D space
    let simplex_5d = standard_simplex::<5>();
    let test_point_5d = Point::new([0.16, 0.16, 0.16, 0.16, 0.16]);
    bench_simplex!(c, 5, simplex_5d, test_point_5d);
}

/// Benchmark edge cases (points on boundary, far away, near-boundary, etc.) across all dimensions
fn benchmark_edge_cases(c: &mut Criterion) {
    // 2D edge cases - triangle
    let simplex_2d = standard_simplex::<2>();
    let boundary_point_2d = simplex_2d[0]; // Point on boundary
    let far_point_2d = Point::new([1000.0, 1000.0]);
    // Near-boundary point (epsilon away from circumsphere)
    let eps = 1e-9;
    let near_boundary_2d = Point::new([eps, 0.0]);
    bench_edge_case!(c, 2, "boundary_point", simplex_2d, boundary_point_2d);
    bench_edge_case!(c, 2, "far_point", simplex_2d, far_point_2d);
    bench_edge_case!(c, 2, "near_boundary", simplex_2d, near_boundary_2d);

    // 3D edge cases - tetrahedron
    let simplex_3d = standard_simplex::<3>();
    let boundary_point_3d = simplex_3d[0]; // Point on boundary
    let far_point_3d = Point::new([1000.0, 1000.0, 1000.0]);
    // Near-boundary point (epsilon away from circumsphere)
    let near_boundary_3d = Point::new([eps, 0.0, 0.0]);
    bench_edge_case!(c, 3, "boundary_point", simplex_3d, boundary_point_3d);
    bench_edge_case!(c, 3, "far_point", simplex_3d, far_point_3d);
    bench_edge_case!(c, 3, "near_boundary", simplex_3d, near_boundary_3d);

    // 4D edge cases - 4-simplex
    let simplex_4d = standard_simplex::<4>();
    let boundary_point_4d = simplex_4d[0]; // Point on boundary
    let far_point_4d = Point::new([1000.0, 1000.0, 1000.0, 1000.0]);
    // Near-boundary point (epsilon away from circumsphere)
    let near_boundary_4d = Point::new([eps, 0.0, 0.0, 0.0]);
    bench_edge_case!(c, 4, "boundary_point", simplex_4d, boundary_point_4d);
    bench_edge_case!(c, 4, "far_point", simplex_4d, far_point_4d);
    bench_edge_case!(c, 4, "near_boundary", simplex_4d, near_boundary_4d);

    // 5D edge cases - 5-simplex
    let simplex_5d = standard_simplex::<5>();
    let boundary_point_5d = simplex_5d[0]; // Point on boundary
    let far_point_5d = Point::new([1000.0, 1000.0, 1000.0, 1000.0, 1000.0]);
    // Near-boundary point (epsilon away from circumsphere)
    let near_boundary_5d = Point::new([eps, 0.0, 0.0, 0.0, 0.0]);
    bench_edge_case!(c, 5, "boundary_point", simplex_5d, boundary_point_5d);
    bench_edge_case!(c, 5, "far_point", simplex_5d, far_point_5d);
    bench_edge_case!(c, 5, "near_boundary", simplex_5d, near_boundary_5d);
}

/// Numerical consistency test - compare results of all three methods
fn numerical_consistency_test() {
    println!("\n=== Numerical Consistency Test ===");
    let mut all_match = 0;
    let mut insphere_distance_matches = 0;
    let mut insphere_lifted_matches = 0;
    let mut distance_lifted_matches = 0;
    let mut total = 0;
    let mut disagreements = Vec::new();

    for i in 0..1000_u64 {
        let simplex_points = generate_random_simplex_3d(1000 + i);
        let test_point = generate_random_test_point_3d(2000 + i);

        let result_insphere = insphere(&simplex_points, test_point);
        let result_distance = insphere_distance(&simplex_points, test_point);
        let result_lifted = insphere_lifted(&simplex_points, test_point);

        if let (Ok(r1), Ok(r2), Ok(r3)) = (result_insphere, result_distance, result_lifted) {
            total += 1;

            // Check pairwise agreements
            if r1 == r2 {
                insphere_distance_matches += 1;
            }
            if r1 == r3 {
                insphere_lifted_matches += 1;
            }
            if r2 == r3 {
                distance_lifted_matches += 1;
            }

            // Check if all three agree
            if r1 == r2 && r2 == r3 {
                all_match += 1;
            } else {
                disagreements.push((simplex_points, test_point, r1, r2, r3));
            }
        }
    }

    if total == 0 {
        println!("Method Comparisons (0 total tests): no valid cases; skipping percentage report.");
        return;
    }
    println!("Method Comparisons ({total} total tests):");
    println!(
        "  insphere vs insphere_distance:  {}/{} ({:.2}%)",
        insphere_distance_matches,
        total,
        (f64::from(insphere_distance_matches) / f64::from(total)) * 100.0
    );
    println!(
        "  insphere vs insphere_lifted:    {}/{} ({:.2}%)",
        insphere_lifted_matches,
        total,
        (f64::from(insphere_lifted_matches) / f64::from(total)) * 100.0
    );
    println!(
        "  insphere_distance vs insphere_lifted: {}/{} ({:.2}%)",
        distance_lifted_matches,
        total,
        (f64::from(distance_lifted_matches) / f64::from(total)) * 100.0
    );
    println!(
        "  All three methods agree:        {}/{} ({:.2}%)",
        all_match,
        total,
        (f64::from(all_match) / f64::from(total)) * 100.0
    );

    if !disagreements.is_empty() {
        println!(
            "\nFound {} cases where methods disagree:",
            disagreements.len()
        );
        for (i, (simplex, test, r1, r2, r3)) in disagreements.iter().take(5).enumerate() {
            println!(
                "  Disagreement {}: insphere={}, distance={}, lifted={}",
                i + 1,
                r1,
                r2,
                r3
            );
            println!("    Test point: {:?}", test.coords());
            println!(
                "    Simplex: {:?}",
                simplex
                    .iter()
                    .map(delaunay::geometry::Point::coords)
                    .collect::<Vec<_>>()
            );
        }
    }
}

/// Main benchmark function that runs consistency test before benchmarks
fn benchmark_with_consistency_check(c: &mut Criterion) {
    // Run consistency test first
    numerical_consistency_test();

    // Then run benchmarks
    benchmark_random_queries(c);
    benchmark_different_dimensions(c);
    benchmark_edge_cases(c);
}

criterion_group!(benches, benchmark_with_consistency_check);
criterion_main!(benches);

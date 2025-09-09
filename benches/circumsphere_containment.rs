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

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use delaunay::prelude::*;
use rand::Rng;
use std::hint::black_box;

/// Generate a random simplex for benchmarking
fn generate_random_simplex_3d(rng: &mut impl Rng) -> Vec<Point<f64, 3>> {
    (0..4)
        .map(|_| {
            let x = rng.random::<f64>().mul_add(20.0, -10.0); // Range -10.0..10.0
            let y = rng.random::<f64>().mul_add(20.0, -10.0);
            let z = rng.random::<f64>().mul_add(20.0, -10.0);
            Point::new([x, y, z])
        })
        .collect()
}

/// Generate a random test point
fn generate_random_test_point_3d(rng: &mut impl Rng) -> Point<f64, 3> {
    let x = rng.random::<f64>().mul_add(10.0, -5.0); // Range -5.0..5.0
    let y = rng.random::<f64>().mul_add(10.0, -5.0);
    let z = rng.random::<f64>().mul_add(10.0, -5.0);
    Point::new([x, y, z])
}

/// Benchmark with many random queries
fn benchmark_random_queries(c: &mut Criterion) {
    let mut rng = rand::rng();

    // Generate a fixed simplex for consistent benchmarking
    let simplex_points = generate_random_simplex_3d(&mut rng);

    // Generate many test points
    let test_points: Vec<_> = (0..1000)
        .map(|_| generate_random_test_point_3d(&mut rng))
        .collect();

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

/// Benchmark with different simplex sizes (2D, 3D, 4D, 5D)
fn benchmark_different_dimensions(c: &mut Criterion) {
    let _rng = rand::rng();

    // 2D case - triangle in 2D space
    let simplex_2d = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([0.0, 1.0]),
    ];
    let test_point_2d = Point::new([0.3, 0.3]);

    c.bench_function("2d/insphere", |b| {
        b.iter(|| black_box(insphere(black_box(&simplex_2d), black_box(test_point_2d)).unwrap()));
    });

    c.bench_function("2d/insphere_distance", |b| {
        b.iter(|| {
            black_box(insphere_distance(black_box(&simplex_2d), black_box(test_point_2d)).unwrap())
        });
    });

    c.bench_function("2d/insphere_lifted", |b| {
        b.iter(|| {
            black_box(insphere_lifted(black_box(&simplex_2d), black_box(test_point_2d)).unwrap())
        });
    });

    // 3D case - tetrahedron in 3D space
    let simplex_3d = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
    ];
    let test_point_3d = Point::new([0.25, 0.25, 0.25]);

    c.bench_function("3d/insphere", |b| {
        b.iter(|| black_box(insphere(black_box(&simplex_3d), black_box(test_point_3d)).unwrap()));
    });

    c.bench_function("3d/insphere_distance", |b| {
        b.iter(|| {
            black_box(insphere_distance(black_box(&simplex_3d), black_box(test_point_3d)).unwrap())
        });
    });

    c.bench_function("3d/insphere_lifted", |b| {
        b.iter(|| {
            black_box(insphere_lifted(black_box(&simplex_3d), black_box(test_point_3d)).unwrap())
        });
    });

    // 4D case - 4-simplex in 4D space
    let simplex_4d = vec![
        Point::new([0.0, 0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0, 0.0]),
        Point::new([0.0, 0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 0.0, 1.0]),
    ];
    let test_point_4d = Point::new([0.2, 0.2, 0.2, 0.2]);

    c.bench_function("4d/insphere", |b| {
        b.iter(|| black_box(insphere(black_box(&simplex_4d), black_box(test_point_4d)).unwrap()));
    });

    c.bench_function("4d/insphere_distance", |b| {
        b.iter(|| {
            black_box(insphere_distance(black_box(&simplex_4d), black_box(test_point_4d)).unwrap())
        });
    });

    c.bench_function("4d/insphere_lifted", |b| {
        b.iter(|| {
            black_box(insphere_lifted(black_box(&simplex_4d), black_box(test_point_4d)).unwrap())
        });
    });

    // 5D case - 5-simplex in 5D space
    let simplex_5d = vec![
        Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
        Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
        Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
    ];
    let test_point_5d = Point::new([0.16, 0.16, 0.16, 0.16, 0.16]);

    c.bench_function("5d/insphere", |b| {
        b.iter(|| black_box(insphere(black_box(&simplex_5d), black_box(test_point_5d)).unwrap()));
    });

    c.bench_function("5d/insphere_distance", |b| {
        b.iter(|| {
            black_box(insphere_distance(black_box(&simplex_5d), black_box(test_point_5d)).unwrap())
        });
    });

    c.bench_function("5d/insphere_lifted", |b| {
        b.iter(|| {
            black_box(insphere_lifted(black_box(&simplex_5d), black_box(test_point_5d)).unwrap())
        });
    });
}

/// Benchmark edge cases (points on boundary, far away, etc.) across all dimensions
#[allow(clippy::too_many_lines)]
fn benchmark_edge_cases(c: &mut Criterion) {
    // 2D edge cases - triangle
    let simplex_2d = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([0.0, 1.0]),
    ];
    let boundary_point_2d = simplex_2d[0]; // Point on boundary
    let far_point_2d = Point::new([1000.0, 1000.0]);

    c.bench_function("edge_cases_2d/boundary_point_insphere", |b| {
        b.iter(|| {
            black_box(insphere(black_box(&simplex_2d), black_box(boundary_point_2d)).unwrap())
        });
    });

    c.bench_function("edge_cases_2d/boundary_point_distance", |b| {
        b.iter(|| {
            black_box(
                insphere_distance(black_box(&simplex_2d), black_box(boundary_point_2d)).unwrap(),
            )
        });
    });

    c.bench_function("edge_cases_2d/boundary_point_lifted", |b| {
        b.iter(|| {
            black_box(
                insphere_lifted(black_box(&simplex_2d), black_box(boundary_point_2d)).unwrap(),
            )
        });
    });

    c.bench_function("edge_cases_2d/far_point_insphere", |b| {
        b.iter(|| black_box(insphere(black_box(&simplex_2d), black_box(far_point_2d)).unwrap()));
    });

    c.bench_function("edge_cases_2d/far_point_distance", |b| {
        b.iter(|| {
            black_box(insphere_distance(black_box(&simplex_2d), black_box(far_point_2d)).unwrap())
        });
    });

    c.bench_function("edge_cases_2d/far_point_lifted", |b| {
        b.iter(|| {
            black_box(insphere_lifted(black_box(&simplex_2d), black_box(far_point_2d)).unwrap())
        });
    });

    // 3D edge cases - tetrahedron
    let simplex_3d = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
    ];
    let boundary_point_3d = simplex_3d[0]; // Point on boundary
    let far_point_3d = Point::new([1000.0, 1000.0, 1000.0]);

    c.bench_function("edge_cases_3d/boundary_point_insphere", |b| {
        b.iter(|| {
            black_box(insphere(black_box(&simplex_3d), black_box(boundary_point_3d)).unwrap())
        });
    });

    c.bench_function("edge_cases_3d/boundary_point_distance", |b| {
        b.iter(|| {
            black_box(
                insphere_distance(black_box(&simplex_3d), black_box(boundary_point_3d)).unwrap(),
            )
        });
    });

    c.bench_function("edge_cases_3d/boundary_point_lifted", |b| {
        b.iter(|| {
            black_box(
                insphere_lifted(black_box(&simplex_3d), black_box(boundary_point_3d)).unwrap(),
            )
        });
    });

    c.bench_function("edge_cases_3d/far_point_insphere", |b| {
        b.iter(|| black_box(insphere(black_box(&simplex_3d), black_box(far_point_3d)).unwrap()));
    });

    c.bench_function("edge_cases_3d/far_point_distance", |b| {
        b.iter(|| {
            black_box(insphere_distance(black_box(&simplex_3d), black_box(far_point_3d)).unwrap())
        });
    });

    c.bench_function("edge_cases_3d/far_point_lifted", |b| {
        b.iter(|| {
            black_box(insphere_lifted(black_box(&simplex_3d), black_box(far_point_3d)).unwrap())
        });
    });

    // 4D edge cases - 4-simplex
    let simplex_4d = vec![
        Point::new([0.0, 0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0, 0.0]),
        Point::new([0.0, 0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 0.0, 1.0]),
    ];
    let boundary_point_4d = simplex_4d[0]; // Point on boundary
    let far_point_4d = Point::new([1000.0, 1000.0, 1000.0, 1000.0]);

    c.bench_function("edge_cases_4d/boundary_point_insphere", |b| {
        b.iter(|| {
            black_box(insphere(black_box(&simplex_4d), black_box(boundary_point_4d)).unwrap())
        });
    });

    c.bench_function("edge_cases_4d/boundary_point_distance", |b| {
        b.iter(|| {
            black_box(
                insphere_distance(black_box(&simplex_4d), black_box(boundary_point_4d)).unwrap(),
            )
        });
    });

    c.bench_function("edge_cases_4d/boundary_point_lifted", |b| {
        b.iter(|| {
            black_box(
                insphere_lifted(black_box(&simplex_4d), black_box(boundary_point_4d)).unwrap(),
            )
        });
    });

    c.bench_function("edge_cases_4d/far_point_insphere", |b| {
        b.iter(|| black_box(insphere(black_box(&simplex_4d), black_box(far_point_4d)).unwrap()));
    });

    c.bench_function("edge_cases_4d/far_point_distance", |b| {
        b.iter(|| {
            black_box(insphere_distance(black_box(&simplex_4d), black_box(far_point_4d)).unwrap())
        });
    });

    c.bench_function("edge_cases_4d/far_point_lifted", |b| {
        b.iter(|| {
            black_box(insphere_lifted(black_box(&simplex_4d), black_box(far_point_4d)).unwrap())
        });
    });

    // 5D edge cases - 5-simplex
    let simplex_5d = vec![
        Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
        Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
        Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
    ];
    let boundary_point_5d = simplex_5d[0]; // Point on boundary
    let far_point_5d = Point::new([1000.0, 1000.0, 1000.0, 1000.0, 1000.0]);

    c.bench_function("edge_cases_5d/boundary_point_insphere", |b| {
        b.iter(|| {
            black_box(insphere(black_box(&simplex_5d), black_box(boundary_point_5d)).unwrap())
        });
    });

    c.bench_function("edge_cases_5d/boundary_point_distance", |b| {
        b.iter(|| {
            black_box(
                insphere_distance(black_box(&simplex_5d), black_box(boundary_point_5d)).unwrap(),
            )
        });
    });

    c.bench_function("edge_cases_5d/boundary_point_lifted", |b| {
        b.iter(|| {
            black_box(
                insphere_lifted(black_box(&simplex_5d), black_box(boundary_point_5d)).unwrap(),
            )
        });
    });

    c.bench_function("edge_cases_5d/far_point_insphere", |b| {
        b.iter(|| black_box(insphere(black_box(&simplex_5d), black_box(far_point_5d)).unwrap()));
    });

    c.bench_function("edge_cases_5d/far_point_distance", |b| {
        b.iter(|| {
            black_box(insphere_distance(black_box(&simplex_5d), black_box(far_point_5d)).unwrap())
        });
    });

    c.bench_function("edge_cases_5d/far_point_lifted", |b| {
        b.iter(|| {
            black_box(insphere_lifted(black_box(&simplex_5d), black_box(far_point_5d)).unwrap())
        });
    });
}

/// Numerical consistency test - compare results of all three methods
fn numerical_consistency_test() {
    println!("\n=== Numerical Consistency Test ===");
    let mut rng = rand::rng();
    let mut all_match = 0;
    let mut insphere_distance_matches = 0;
    let mut insphere_lifted_matches = 0;
    let mut distance_lifted_matches = 0;
    let mut total = 0;
    let mut disagreements = Vec::new();

    for _ in 0..1000 {
        let simplex_points = generate_random_simplex_3d(&mut rng);
        let test_point = generate_random_test_point_3d(&mut rng);

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
            println!("    Test point: {:?}", test.to_array());
            println!(
                "    Simplex: {:?}",
                simplex.iter().map(Coordinate::to_array).collect::<Vec<_>>()
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

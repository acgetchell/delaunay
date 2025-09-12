//! Benchmarks for d-dimensional Delaunay triangulation creation.
#![allow(missing_docs)] // Criterion macros generate undocumented functions
//!
//! This benchmark suite measures the performance of creating Delaunay triangulations
//! across different dimensions (2D, 3D, 4D, 5D) using 1,000 randomly generated vertices.
//!
//! # Benchmark Design
//!
//! - Uses seeded random point generation for reproducible results in CI
//! - Measures triangulation creation time and throughput (elements/second)
//! - Higher dimensions (≥5D) use reduced sample sizes to bound execution time
//! - Points are generated in the range (-100.0, 100.0) for each coordinate
//!
//! # Usage
//!
//! ```bash
//! # Run all triangulation creation benchmarks
//! cargo bench --bench triangulation_creation
//!
//! # Run specific dimension
//! cargo bench --bench triangulation_creation "2d_triangulation_creation"
//! ```
//!
//! # Performance Characteristics
//!
//! Triangulation creation complexity grows significantly with dimension:
//! - 2D: O(n log n) expected, very fast
//! - 3D: O(n log n) expected, moderate
//! - 4D: O(n²) worst case, slower
//! - 5D: O(n²) worst case, much slower

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::geometry::util::generate_random_triangulation;

// =============================================================================
// TRIANGULATION BENCHMARKS
// =============================================================================

/// Generic function to benchmark triangulation creation for a given dimension.
///
/// This function:
/// - Uses the `generate_random_triangulation` utility for consistent setup
/// - Measures triangulation creation time excluding Drop overhead via `iter_batched`
/// - Configures appropriate sample sizes based on dimension
///
/// # Parameters
/// - `D`: The dimension (const generic parameter)
/// - `c`: Criterion benchmark context
/// - `benchmark_name`: Name for the benchmark group
///
/// # Benchmark Structure
/// 1. Uses seeded random generation for reproducible results
/// 2. Creates 1,000 points in (-100.0, 100.0)ᴰ coordinate space
/// 3. Times triangulation creation excluding Drop via `iter_batched`
/// 4. Reports time and throughput metrics
fn bench_triangulation_creation_generic<const D: usize>(c: &mut Criterion, benchmark_name: &str)
where
    [f64; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
{
    let mut group = c.benchmark_group(benchmark_name);
    // Reduce sample size for higher dimensions to bound runtime
    if D >= 5 {
        group.sample_size(10);
    }
    group.throughput(Throughput::Elements(1_000u64));
    group.bench_function("triangulation", |b| {
        b.iter_batched(
            || (),
            |()| {
                generate_random_triangulation::<f64, (), (), D>(
                    1_000,
                    (-100.0, 100.0),
                    None,
                    Some(10_864 + D as u64),
                )
                .expect("Failed to generate triangulation")
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

// Benchmark functions using the generic implementation
// Each function benchmarks triangulation creation for a specific dimension

/// Benchmark 2D Delaunay triangulation creation with 1,000 vertices.
/// Expected to be very fast with O(n log n) complexity.
fn bench_triangulation_creation_2d(c: &mut Criterion) {
    bench_triangulation_creation_generic::<2>(c, "2d_triangulation_creation");
}

/// Benchmark 3D Delaunay triangulation creation with 1,000 vertices.
/// Moderate performance with O(n log n) expected complexity.
fn bench_triangulation_creation_3d(c: &mut Criterion) {
    bench_triangulation_creation_generic::<3>(c, "3d_triangulation_creation");
}

/// Benchmark 4D Delaunay triangulation creation with 1,000 vertices.
/// Slower performance with O(n²) worst-case complexity.
fn bench_triangulation_creation_4d(c: &mut Criterion) {
    bench_triangulation_creation_generic::<4>(c, "4d_triangulation_creation");
}

/// Benchmark 5D Delaunay triangulation creation with 1,000 vertices.
/// Much slower performance with O(n²) worst-case complexity.
/// Uses reduced sample size (10) to bound execution time.
fn bench_triangulation_creation_5d(c: &mut Criterion) {
    bench_triangulation_creation_generic::<5>(c, "5d_triangulation_creation");
}

// Criterion benchmark group containing all triangulation creation benchmarks.
// This group includes benchmarks for 2D, 3D, 4D, and 5D triangulation creation,
// each measuring the performance of creating a Delaunay triangulation from 1,000
// randomly generated vertices.
criterion_group!(
    benches,
    bench_triangulation_creation_2d,
    bench_triangulation_creation_3d,
    bench_triangulation_creation_4d,
    bench_triangulation_creation_5d
);

// Main entry point for the triangulation creation benchmark suite.
criterion_main!(benches);

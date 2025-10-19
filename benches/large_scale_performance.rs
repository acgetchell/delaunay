//! Large-scale triangulation performance benchmarks
//!
//! This benchmark suite measures performance characteristics of Delaunay triangulation
//! construction, validation, and queries for large point sets across multiple dimensions.
//!
//! # Benchmark Coverage
//!
//! - **Dimensions**: 2D, 3D, 4D (practical range)
//! - **Vertex counts**: 1,000, 5,000, 10,000 (scalability testing)
//! - **Measurements**:
//!   - Construction time (Bowyer-Watson algorithm)
//!   - Memory usage delta during construction (process RSS)
//!   - Validation time (topology checks)
//!   - Query performance (neighbor finding, iteration)
//!
//! # Phase 4 `SlotMap` Evaluation
//!
//! This benchmark is specifically designed to support Phase 4 `SlotMap` evaluation:
//! - Large-scale tests reveal `SlotMap` performance characteristics
//! - Query benchmarks measure lookup/iteration efficiency
//! - Memory measurements show allocation patterns
//! - Scalability tests (1K→5K→10K) reveal growth patterns
//!
//! # Reproducibility
//!
//! All benchmarks use seeded RNG (`StdRng::seed_from_u64`) for deterministic results.
//! Points are generated within bounded coordinates (-100.0, 100.0) to avoid degeneracies.
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Compile benchmarks
//! just bench-compile
//!
//! # Run all large-scale benchmarks
//! cargo bench --bench large_scale_performance
//!
//! # Run specific dimension
//! cargo bench --bench large_scale_performance -- "2D"
//!
//! # Run specific measurement type
//! cargo bench --bench large_scale_performance -- "construction"
//! cargo bench --bench large_scale_performance -- "queries"
//!
//! # Create baseline for comparison
//! just bench-baseline
//!
//! # Compare against baseline
//! just bench-compare
//! ```
//!
//! # Performance Notes
//!
//! - 2D/3D with 10,000 vertices: ~1-10 seconds construction
//! - 4D with 10,000 vertices: ~30-120 seconds construction (complexity increases rapidly)
//! - Memory usage scales with O(n^⌈d/2⌉) cells in d dimensions
//! - Query performance depends on `SlotMap` implementation efficiency
//!
//! # CI Considerations
//!
//! For CI runs with time constraints:
//! - Set `CRITERION_SAMPLE_SIZE=10` environment variable
//! - Run smaller configurations: `cargo bench -- "1000"`
//! - Consider skipping 4D/10000 for faster CI: `cargo bench -- "/2D|3D/"`

#![allow(missing_docs)]

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::vertex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::hint::black_box;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

/// Memory usage information for benchmarking
#[derive(Debug, Clone)]
struct MemoryInfo {
    before: u64,
    after: u64,
    delta: i64,
}

/// Get current process memory usage in KB
fn get_memory_usage() -> u64 {
    static SYS: OnceLock<Mutex<System>> = OnceLock::new();
    let pid = sysinfo::get_current_pid().expect("Failed to get current PID");
    let sys = SYS.get_or_init(|| {
        Mutex::new(System::new_with_specifics(
            RefreshKind::new().with_processes(ProcessRefreshKind::new().with_memory()),
        ))
    });
    let mut system = sys.lock().expect("lock System");
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::new().with_memory(),
    );
    system
        .process(pid)
        .map_or(0, |process| process.memory() / 1024) // bytes → KB
}

/// Measure memory delta during triangulation construction
fn measure_construction_with_memory<const D: usize>(n_points: usize, seed: u64) -> MemoryInfo
where
    [f64; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    let mem_before = get_memory_usage();

    // Generate points and construct triangulation
    let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), seed)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();

    let _tds: Tds<f64, Option<()>, Option<()>, D> =
        Tds::new(&vertices).expect("Failed to create triangulation");

    let mem_after = get_memory_usage();

    let delta_i128 = i128::from(mem_after) - i128::from(mem_before);
    #[allow(clippy::cast_possible_truncation)] // Clamped to i64 range, safe to cast
    let delta = delta_i128.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
    MemoryInfo {
        before: mem_before,
        after: mem_after,
        delta,
    }
}

// =============================================================================
// Construction Benchmarks
// =============================================================================

/// Benchmark: Triangulation construction time
fn bench_construction<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize)
where
    [f64; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    let bench_name = format!("construction/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.throughput(Throughput::Elements(n_points as u64));

    // Adjust sample size for 4D large cases to bound execution time
    if D == 4 && n_points >= 5000 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(120));
    }

    group.bench_function("construct", |b| {
        b.iter_batched(
            || {
                // Setup: Generate points (not measured)
                let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), 42)
                    .expect("Failed to generate points");
                points.into_iter().map(|p| vertex!(p)).collect::<Vec<_>>()
            },
            |vertices| {
                // Measured operation: Construct triangulation
                let tds: Tds<f64, Option<()>, Option<()>, D> =
                    Tds::new(black_box(&vertices)).expect("Failed to create triangulation");
                black_box(tds)
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

// =============================================================================
// Memory Benchmarks
// =============================================================================

/// Benchmark: Memory usage measurement (informational, not timing-focused)
fn bench_memory_usage<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize)
where
    [f64; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    let bench_name = format!("memory/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);

    // Single measurement for memory delta - reduce sample size
    group.sample_size(10);

    group.bench_function("construction_memory_delta", |b| {
        b.iter(|| {
            let mem_info = measure_construction_with_memory::<D>(n_points, 42);
            // Report memory usage to stderr (won't interfere with benchmark timing)
            if std::env::var_os("BENCH_PRINT_MEM").is_some() {
                eprintln!(
                    "Memory: before={} KB, after={} KB, delta={} KB",
                    mem_info.before, mem_info.after, mem_info.delta
                );
            }
            black_box(mem_info)
        });
    });

    group.finish();
}

// =============================================================================
// Validation Benchmarks
// =============================================================================

/// Benchmark: Topology validation time
fn bench_validation<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize)
where
    [f64; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    let bench_name = format!("validation/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.throughput(Throughput::Elements(n_points as u64));

    // Adjust sample size for large cases
    if n_points >= 5000 {
        group.sample_size(10);
    }

    // Pre-generate triangulation for validation benchmarks
    let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), 42)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();
    let tds: Tds<f64, Option<()>, Option<()>, D> =
        Tds::new(&vertices).expect("Failed to create triangulation");

    group.bench_function("validate_topology", |b| {
        b.iter(|| {
            // Measure validation time for all cells
            let mut all_valid = true;
            for cell_key in tds.cell_keys() {
                let neighbors = tds.find_neighbors_by_key(cell_key);
                if let Err(e) = tds.validate_neighbor_topology(cell_key, &neighbors) {
                    all_valid = false;
                    black_box(e);
                }
            }
            black_box(all_valid)
        });
    });

    group.finish();
}

// =============================================================================
// Query Performance Benchmarks
// =============================================================================

/// Benchmark: Neighbor query performance (critical for `SlotMap` evaluation)
fn bench_neighbor_queries<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize)
where
    [f64; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    let bench_name = format!("queries/neighbors/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);

    // Pre-generate triangulation
    let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), 42)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();
    let tds: Tds<f64, Option<()>, Option<()>, D> =
        Tds::new(&vertices).expect("Failed to create triangulation");

    // Collect cell keys for iteration
    let cell_keys: Vec<_> = tds.cell_keys().collect();
    let num_cells = cell_keys.len();

    group.throughput(Throughput::Elements(num_cells as u64));

    group.bench_function("find_neighbors_all_cells", |b| {
        b.iter(|| {
            // Query neighbors for all cells - measures `SlotMap` lookup performance
            for &cell_key in &cell_keys {
                let neighbors = tds.find_neighbors_by_key(cell_key);
                black_box(neighbors);
            }
        });
    });

    group.finish();
}

/// Benchmark: Vertex iteration performance (tests `SlotMap` iteration efficiency)
fn bench_vertex_iteration<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize)
where
    [f64; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    let bench_name = format!("queries/vertices/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.throughput(Throughput::Elements(n_points as u64));

    // Pre-generate triangulation
    let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), 42)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();
    let tds: Tds<f64, Option<()>, Option<()>, D> =
        Tds::new(&vertices).expect("Failed to create triangulation");

    group.bench_function("iterate_all_vertices", |b| {
        b.iter(|| {
            // Iterate through all vertices - measures `SlotMap` iteration performance
            let mut count = 0;
            for vertex in tds.vertex_iter() {
                black_box(vertex);
                count += 1;
            }
            black_box(count)
        });
    });

    group.finish();
}

/// Benchmark: Cell iteration performance (tests `SlotMap` cell iteration)
fn bench_cell_iteration<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize)
where
    [f64; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    let bench_name = format!("queries/cells/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);

    // Pre-generate triangulation
    let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), 42)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();
    let tds: Tds<f64, Option<()>, Option<()>, D> =
        Tds::new(&vertices).expect("Failed to create triangulation");

    let num_cells = tds.number_of_cells();
    group.throughput(Throughput::Elements(num_cells as u64));

    group.bench_function("iterate_all_cells", |b| {
        b.iter(|| {
            // Iterate through all cells - measures `SlotMap` cell iteration performance
            let mut count = 0;
            for cell_key in tds.cell_keys() {
                black_box(cell_key);
                count += 1;
            }
            black_box(count)
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark Suites by Dimension
// =============================================================================

fn bench_2d_suite(c: &mut Criterion) {
    for &n_points in &[1000, 5000, 10_000] {
        bench_construction::<2>(c, "2D", n_points);
        bench_memory_usage::<2>(c, "2D", n_points);
        bench_validation::<2>(c, "2D", n_points);
        bench_neighbor_queries::<2>(c, "2D", n_points);
        bench_vertex_iteration::<2>(c, "2D", n_points);
        bench_cell_iteration::<2>(c, "2D", n_points);
    }
}

fn bench_3d_suite(c: &mut Criterion) {
    for &n_points in &[1000, 5000, 10_000] {
        bench_construction::<3>(c, "3D", n_points);
        bench_memory_usage::<3>(c, "3D", n_points);
        bench_validation::<3>(c, "3D", n_points);
        bench_neighbor_queries::<3>(c, "3D", n_points);
        bench_vertex_iteration::<3>(c, "3D", n_points);
        bench_cell_iteration::<3>(c, "3D", n_points);
    }
}

fn bench_4d_suite(c: &mut Criterion) {
    // Note: 4D with 10,000 vertices can be very slow (30-120 seconds)
    // Consider reducing to [1000, 5000] for faster CI runs
    for &n_points in &[1000, 5000, 10_000] {
        bench_construction::<4>(c, "4D", n_points);
        bench_memory_usage::<4>(c, "4D", n_points);
        bench_validation::<4>(c, "4D", n_points);
        bench_neighbor_queries::<4>(c, "4D", n_points);
        bench_vertex_iteration::<4>(c, "4D", n_points);
        bench_cell_iteration::<4>(c, "4D", n_points);
    }
}

criterion_group!(
    name = large_scale_benches;
    config = Criterion::default();
    targets = bench_2d_suite, bench_3d_suite, bench_4d_suite
);

criterion_main!(large_scale_benches);

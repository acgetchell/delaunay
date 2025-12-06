//! Large-scale triangulation performance benchmarks
//!
//! This benchmark suite measures performance characteristics of Delaunay triangulation
//! construction, validation, and queries for large point sets across multiple dimensions.
//!
//! # Benchmark Coverage
//!
//! - **Dimensions**: 2D, 3D, 4D, 5D (complete coverage)
//! - **Vertex counts**: 1K-10K (2D/3D), 1K-3K (4D default), 500-1K (5D)
//! - **Measurements**:
//!   - Construction time (incremental insertion algorithm)
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
//! # Point Distribution
//!
//! - **Current**: Uniform random distribution (seeded for reproducibility)
//! - **Future**: Grid and Poisson disk distributions planned but not yet implemented
//!
//! # Reproducibility
//!
//! All benchmarks use seeded RNG for deterministic results via `generate_random_points_seeded()`.
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
//! # Performance Notes & Scaling Strategy
//!
//! **Timing Estimates (per dimension suite):**
//! - 2D: [1K, 5K, 10K] → ~15-30 minutes
//! - 3D: [1K, 5K, 10K] → ~20-40 minutes  
//! - 4D: [1K, 3K] (default) → ~30-60 minutes
//! - 4D: [1K, 5K, 10K] (large scale) → ~2-4 hours
//! - 5D: [500, 1K] → ~30-60 minutes
//!
//! **Total benchmark runtime:**
//! - Default (local): ~2-3 hours (2D/3D/4D/5D, recommended for development)
//! - Large scale: ~4-6 hours (includes 4D@10K, requires compute cluster)
//!
//! **Scaling for Large Runs:**
//! ```bash
//! # Enable 10K points for 4D (use on compute cluster)
//! BENCH_LARGE_SCALE=1 cargo bench --bench large_scale_performance
//! ```
//!
//! **Memory complexity:** O(n^⌈d/2⌉) cells in d dimensions
//! **Query performance:** Directly measures `SlotMap` iteration efficiency

#![allow(missing_docs)]

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::vertex;
use std::hint::black_box;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

/// Memory usage information for benchmarking (in KiB)
#[derive(Debug, Clone)]
struct MemoryInfo {
    before: u64,
    after: u64,
    delta: i64,
    tds_delta: i64,
}

/// Get current process memory usage in KiB
fn get_memory_usage() -> u64 {
    static SYS: OnceLock<Mutex<System>> = OnceLock::new();
    static UNIT_LOGGED: std::sync::Once = std::sync::Once::new();

    // Log memory unit on first call for clarity in all benchmark runs
    UNIT_LOGGED.call_once(|| {
        eprintln!("[INFO] Memory measurements in KiB (sysinfo::Process::memory() / 1024)");
    });

    let pid = sysinfo::get_current_pid().expect("Failed to get current PID");
    let sys = SYS.get_or_init(|| {
        Mutex::new(System::new_with_specifics(
            RefreshKind::nothing().with_processes(ProcessRefreshKind::nothing().with_memory()),
        ))
    });
    let mut system = sys.lock().expect("lock System");
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    system
        .process(pid)
        .map_or(0, |process| process.memory() / 1024) // bytes → KiB
}

/// Measure memory delta during triangulation construction
fn measure_construction_with_memory<const D: usize>(n_points: usize, seed: u64) -> MemoryInfo {
    let mem_before = get_memory_usage();

    // Generate points and vertices (setup overhead)
    let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), seed)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();

    // Measure memory before triangulation construction to isolate allocation
    let mem_before_tds = get_memory_usage();

    let dt = DelaunayTriangulation::new(&vertices).expect("Failed to create triangulation");

    let mem_after = get_memory_usage();

    // Keep dt alive for accurate memory measurement
    black_box(&dt);

    // Total delta includes setup + TDS
    let delta_i128 = i128::from(mem_after) - i128::from(mem_before);
    #[expect(clippy::cast_possible_truncation)] // Clamped to i64 range, safe to cast
    let delta = delta_i128.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;

    // TDS-only delta excludes setup overhead
    let tds_delta_i128 = i128::from(mem_after) - i128::from(mem_before_tds);
    #[expect(clippy::cast_possible_truncation)] // Clamped to i64 range, safe to cast
    let tds_delta = tds_delta_i128.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;

    MemoryInfo {
        before: mem_before,
        after: mem_after,
        delta,
        tds_delta,
    }
}

// =============================================================================
// Construction Benchmarks
// =============================================================================

/// Benchmark: Triangulation construction time
fn bench_construction<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize) {
    let bench_name = format!("construction/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.throughput(Throughput::Elements(n_points as u64));

    // Adjust sample size for heavy cases to bound execution time
    if (D == 4 && n_points >= 5000) || D == 5 {
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
                let dt = DelaunayTriangulation::new(black_box(&vertices))
                    .expect("Failed to create triangulation");
                black_box(dt)
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
fn bench_memory_usage<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize) {
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
                    "Memory: before={} KiB, after={} KiB, delta={} KiB (TDS-only: {} KiB)",
                    mem_info.before, mem_info.after, mem_info.delta, mem_info.tds_delta
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
fn bench_validation<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize) {
    let bench_name = format!("validation/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);

    // Adjust sample size for large cases and 5D
    if n_points >= 5000 || D == 5 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(120));
    }

    // Pre-generate triangulation for validation benchmarks
    let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), 42)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();
    let dt = DelaunayTriangulation::new(&vertices).expect("Failed to create triangulation");
    let tds = dt.tds();

    // Throughput in terms of cells we actually validate
    group.throughput(Throughput::Elements(tds.number_of_cells() as u64));

    // Collect cell keys once to avoid re-enumeration overhead in the hot loop
    let cell_keys: Vec<_> = tds.cell_keys().collect();

    group.bench_function("validate_topology", |b| {
        b.iter(|| {
            // Measure validation time for all cells
            let mut all_valid = true;
            for &cell_key in &cell_keys {
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
fn bench_neighbor_queries<const D: usize>(
    c: &mut Criterion,
    dimension_name: &str,
    n_points: usize,
) {
    let bench_name = format!("queries/neighbors/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);

    // Adjust sample size for very heavy cases (5D or large 4D)
    if D == 5 || (D == 4 && n_points >= 5000) {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(120));
    }

    // Pre-generate triangulation
    let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), 42)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();
    let dt = DelaunayTriangulation::new(&vertices).expect("Failed to create triangulation");
    let tds = dt.tds();

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
fn bench_vertex_iteration<const D: usize>(
    c: &mut Criterion,
    dimension_name: &str,
    n_points: usize,
) {
    let bench_name = format!("queries/vertices/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.throughput(Throughput::Elements(n_points as u64));

    // Adjust sample size for very heavy cases (5D or large 4D)
    if D == 5 || (D == 4 && n_points >= 5000) {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(120));
    }

    // Pre-generate triangulation
    let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), 42)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();
    let dt = DelaunayTriangulation::new(&vertices).expect("Failed to create triangulation");
    let tds = dt.tds();

    group.bench_function("iterate_all_vertices", |b| {
        b.iter(|| {
            // Iterate through all vertices - measures `SlotMap` iteration performance
            let mut count = 0;
            for (_, vertex) in tds.vertices() {
                black_box(vertex);
                count += 1;
            }
            black_box(count)
        });
    });

    group.finish();
}

/// Benchmark: Cell iteration performance (tests `SlotMap` cell iteration)
fn bench_cell_iteration<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize) {
    let bench_name = format!("queries/cells/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);

    // Adjust sample size for very heavy cases (5D or large 4D)
    if D == 5 || (D == 4 && n_points >= 5000) {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(120));
    }

    // Pre-generate triangulation
    let points = generate_random_points_seeded::<f64, D>(n_points, (-100.0, 100.0), 42)
        .expect("Failed to generate points");
    let vertices: Vec<_> = points.into_iter().map(|p| vertex!(p)).collect();
    let dt = DelaunayTriangulation::new(&vertices).expect("Failed to create triangulation");
    let tds = dt.tds();

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
    // 4D scaling: Default uses smaller sizes for reasonable runtime (~30-60 min total)
    // Set BENCH_LARGE_SCALE=1 (or any truthy value) for 10K points (requires compute cluster, ~2-4 hours)
    // Note: BENCH_LARGE_SCALE=0 or BENCH_LARGE_SCALE=false will disable large scale mode
    let large_scale_enabled = std::env::var("BENCH_LARGE_SCALE")
        .ok()
        .is_some_and(|v| !v.is_empty() && v != "0" && v != "false");

    let point_counts: &[usize] = if large_scale_enabled {
        &[1000, 5000, 10_000] // Full scale for cluster runs
    } else {
        &[1000, 3000] // Reduced scale for local development (<2 hours total)
    };

    for &n_points in point_counts {
        bench_construction::<4>(c, "4D", n_points);
        bench_memory_usage::<4>(c, "4D", n_points);
        bench_validation::<4>(c, "4D", n_points);
        bench_neighbor_queries::<4>(c, "4D", n_points);
        bench_vertex_iteration::<4>(c, "4D", n_points);
        bench_cell_iteration::<4>(c, "4D", n_points);
    }
}

fn bench_5d_suite(c: &mut Criterion) {
    // 5D scaling: Very small sizes due to extreme computational cost
    // Complexity grows as O(n^⌈5/2⌉) ≈ O(n³) for cells
    for &n_points in &[500, 1000] {
        bench_construction::<5>(c, "5D", n_points);
        bench_memory_usage::<5>(c, "5D", n_points);
        bench_validation::<5>(c, "5D", n_points);
        bench_neighbor_queries::<5>(c, "5D", n_points);
        bench_vertex_iteration::<5>(c, "5D", n_points);
        bench_cell_iteration::<5>(c, "5D", n_points);
    }
}

criterion_group!(
    name = large_scale_benches;
    config = {
        let sample_size = std::env::var("BENCH_SAMPLE_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let warm_up_secs = std::env::var("BENCH_WARMUP_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let measurement_secs = std::env::var("BENCH_MEASUREMENT_TIME")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);

        Criterion::default()
            .sample_size(sample_size)
            .warm_up_time(Duration::from_secs(warm_up_secs))
            .measurement_time(Duration::from_secs(measurement_secs))
    };
    targets = bench_2d_suite, bench_3d_suite, bench_4d_suite, bench_5d_suite
);

criterion_main!(large_scale_benches);

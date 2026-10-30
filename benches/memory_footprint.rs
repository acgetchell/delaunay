//! Process RSS and instrumented construction allocation diagnostics.
//!
//! Manual diagnostics extracted from `profiling_suite`; excluded from release signals.
//! Fixtures are initialized inside selected Criterion callbacks, before timing.

#[cfg(feature = "count-allocations")]
use allocation_counter::{AllocationInfo, measure};
use bench_utils::OrAbortWithContext;
#[cfg(feature = "count-allocations")]
use criterion::{BenchmarkGroup, BenchmarkId, measurement::WallTime};
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(feature = "count-allocations")]
use delaunay::prelude::collections::SmallBuffer;
#[cfg(feature = "count-allocations")]
use delaunay::prelude::construction::DelaunayTriangulationBuilder;
use delaunay::prelude::geometry::{AdaptiveKernel, ExactPredicates};
#[cfg(feature = "count-allocations")]
use std::time::{Duration, Instant};
use std::{
    hint::black_box,
    sync::{Mutex, OnceLock},
};
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System, get_current_pid};

#[path = "common/bench_utils.rs"]
pub mod bench_utils;
#[path = "common/profiling_fixtures.rs"]
mod profiling_fixtures;
use bench_utils::OrAbort;
#[cfg(feature = "count-allocations")]
use profiling_fixtures::{DEFAULT_SEED, PointDistribution, gen_points, is_dev_mode};
use profiling_fixtures::{
    bench_time, benchmark_vertices_from_generated_points, construct_triangulation,
    criterion_config, generated_points_in_range, scale_cases, seed_for_case, validated_fixture,
    wide_bounds,
};
#[cfg(any(feature = "count-allocations", feature = "bench-logging"))]
use std::env;

#[cfg(feature = "count-allocations")]
const BENCHMARK_ITERATION_BUFFER_SIZE: usize = 8;
/// Memory usage information for benchmarking, in KiB.
#[cfg_attr(
    not(feature = "bench-logging"),
    expect(
        dead_code,
        reason = "Memory fields are unpacked by optional bench diagnostics"
    )
)]
#[derive(Debug, Clone)]
struct MemoryInfo {
    before: u64,
    after: u64,
    delta: i64,
    tds_delta: i64,
}
/// Return current process memory usage in KiB.
fn memory_usage_kib() -> u64 {
    static SYS: OnceLock<Mutex<System>> = OnceLock::new();

    let pid = get_current_pid().or_abort();
    let sys = SYS.get_or_init(|| {
        Mutex::new(
            System::new_with_specifics(
                RefreshKind::nothing().with_processes(ProcessRefreshKind::nothing().with_memory()),
            )
            .or_abort(),
        )
    });
    let mut system = sys.lock().or_abort();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    system
        .process(pid)
        .or_abort("current process missing from RSS observation")
        .memory()
        / 1024
}

/// Measure memory delta during triangulation construction.
fn measure_construction_with_memory<const D: usize>(n_points: usize, seed: u64) -> MemoryInfo
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let mem_before = memory_usage_kib();
    let points = generated_points_in_range::<D>(n_points, wide_bounds(), seed);
    let vertices = benchmark_vertices_from_generated_points(&points);

    let mem_before_tds = memory_usage_kib();
    let dt = construct_triangulation::<D>(&vertices, seed);
    let mem_after = memory_usage_kib();
    black_box(&dt);

    let delta_i128 = i128::from(mem_after) - i128::from(mem_before);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to i64 range before casting"
    )]
    let delta = delta_i128.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;

    let tds_delta_i128 = i128::from(mem_after) - i128::from(mem_before_tds);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to i64 range before casting"
    )]
    let tds_delta = tds_delta_i128.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;

    MemoryInfo {
        before: mem_before,
        after: mem_after,
        delta,
        tds_delta,
    }
}

/// Benchmark process RSS deltas during construction.
fn bench_rss_memory_usage<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize)
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let bench_name = format!("memory/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.sample_size(10);
    group.measurement_time(bench_time(5));

    let seed = seed_for_case::<D>(n_points);

    let preflight = OnceLock::new();
    group.bench_function("construction_memory_delta", |b| {
        preflight.get_or_init(|| {
            black_box(validated_fixture::<D>(n_points, seed));
            #[cfg(feature = "bench-logging")]
            if env::var_os("BENCH_PRINT_MEM").is_some() {
                let mem_info = measure_construction_with_memory::<D>(n_points, seed);
                tracing::info!(
                    "Memory sample: before={} KiB, after={} KiB, delta={} KiB (TDS-only: {} KiB)",
                    mem_info.before,
                    mem_info.after,
                    mem_info.delta,
                    mem_info.tds_delta
                );
            }

            let _ = memory_usage_kib();
        });

        b.iter(|| {
            let mem_info = measure_construction_with_memory::<D>(n_points, seed);
            black_box(mem_info)
        });
    });

    group.finish();
}

/// Read the memory summary percentile from `BENCH_PERCENTILE` (default: 95).
#[cfg(feature = "count-allocations")]
fn configured_percentile() -> usize {
    env::var("BENCH_PERCENTILE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map_or(95, |p| p.clamp(1, 100))
}

/// Format a percentile as an ordinal label for the memory summary.
#[cfg(feature = "count-allocations")]
fn percentile_label(percentile: usize) -> String {
    let suffix = match percentile % 100 {
        11..=13 => "th",
        _ => match percentile % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        },
    };
    format!("{percentile}{suffix}")
}

/// Return percentile from a slice of values using nearest-rank method.
#[cfg(feature = "count-allocations")]
fn percentile_nearest_rank(values: &mut [u64], percentile: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }

    values.sort_unstable();
    let n = values.len();
    // nearest-rank: ceil(p/100 * n), clamped to [1, n]
    let rank = ((percentile.saturating_mul(n)).saturating_add(99))
        .saturating_div(100)
        .clamp(1, n);
    let index = rank - 1; // safe: rank in [1, n]
    values[index]
}

/// Print memory allocation summary
#[cfg(feature = "count-allocations")]
#[expect(
    clippy::cast_precision_loss,
    reason = "allocation summary reports byte ratios where f64 precision is sufficient"
)]
fn print_alloc_summary(
    info: &AllocationInfo,
    description: &str,
    actual_point_count: usize,
    percentile: usize,
    percentile_value: u64,
) {
    println!("\n=== Memory Allocation Summary for {description} ({actual_point_count} points) ===");
    println!("Total allocations: {}", info.count_total);
    println!("Current allocations: {}", info.count_current);
    println!("Max allocations: {}", info.count_max);
    println!("Total bytes allocated: {}", info.bytes_total);
    println!("Current bytes allocated: {}", info.bytes_current);
    println!(
        "Max bytes allocated: {} ({:.2} MB)",
        info.bytes_max,
        info.bytes_max as f64 / (1024.0 * 1024.0)
    );
    println!(
        "{} percentile bytes: {} ({:.2} MB)",
        percentile_label(percentile),
        percentile_value,
        percentile_value as f64 / (1024.0 * 1024.0)
    );
    if actual_point_count > 0 {
        println!(
            "Bytes per point (peak): {:.1}",
            info.bytes_max as f64 / actual_point_count as f64
        );
    } else {
        println!("Bytes per point (peak): n/a (0 points)");
    }
    println!("=====================================\n");
}

#[cfg(feature = "count-allocations")]
#[expect(
    clippy::cast_possible_wrap,
    reason = "sample counts fit signed indexing for percentile diagnostics"
)]
fn print_alloc_summary_from_samples<const D: usize>(
    allocation_infos: &SmallBuffer<AllocationInfo, BENCHMARK_ITERATION_BUFFER_SIZE>,
    actual_point_counts: &SmallBuffer<usize, BENCHMARK_ITERATION_BUFFER_SIZE>,
) {
    if allocation_infos.is_empty() {
        return;
    }

    // Safe cast for division: Criterion sample buffers here are small and non-empty.
    let divisor_unsigned = allocation_infos.len() as u64;
    let divisor_signed = allocation_infos.len() as i64;
    let avg_info = AllocationInfo {
        count_total: allocation_infos.iter().map(|i| i.count_total).sum::<u64>() / divisor_unsigned,
        count_current: allocation_infos
            .iter()
            .map(|i| i.count_current)
            .sum::<i64>()
            / divisor_signed,
        count_max: allocation_infos
            .iter()
            .map(|i| i.count_max)
            .max()
            .unwrap_or(0),
        bytes_total: allocation_infos.iter().map(|i| i.bytes_total).sum::<u64>() / divisor_unsigned,
        bytes_current: allocation_infos
            .iter()
            .map(|i| i.bytes_current)
            .sum::<i64>()
            / divisor_signed,
        bytes_max: allocation_infos
            .iter()
            .map(|i| i.bytes_max)
            .max()
            .unwrap_or(0),
    };
    let avg_actual_count = if actual_point_counts.is_empty() {
        0
    } else {
        actual_point_counts.iter().sum::<usize>() / actual_point_counts.len()
    };

    let mut bytes_max_values: Vec<u64> = allocation_infos.iter().map(|i| i.bytes_max).collect();
    let percentile = configured_percentile();
    let percentile_value = percentile_nearest_rank(&mut bytes_max_values, percentile);

    print_alloc_summary(
        &avg_info,
        &format!("{D}D Triangulation"),
        avg_actual_count,
        percentile,
        percentile_value,
    );
}

/// Generic helper to benchmark memory usage for a specific dimension D
#[cfg(feature = "count-allocations")]
fn bench_memory_usage<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    bench_id_prefix: &str,
    count: usize,
) where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let mut allocation_infos: SmallBuffer<AllocationInfo, BENCHMARK_ITERATION_BUFFER_SIZE> =
        SmallBuffer::new();

    let mut actual_point_counts: SmallBuffer<usize, BENCHMARK_ITERATION_BUFFER_SIZE> =
        SmallBuffer::new();
    let preflight = OnceLock::new();

    group.bench_with_input(
        BenchmarkId::new(bench_id_prefix, count),
        &count,
        |b, &count| {
            preflight.get_or_init(|| {
                black_box(validated_fixture::<D>(count, DEFAULT_SEED));
            });
            b.iter_custom(|iters| {
                let mut total_time = Duration::new(0, 0);

                for _ in 0..iters {
                    let points = gen_points::<D>(count, PointDistribution::Random, DEFAULT_SEED);
                    let pts_len = points.len();
                    let vertices = benchmark_vertices_from_generated_points(&points);
                    let start_time = Instant::now();

                    let alloc_info = measure(|| {
                        let dt = DelaunayTriangulationBuilder::new(&vertices)
                            .build()
                            .or_abort();
                        black_box(dt);
                    });

                    total_time += start_time.elapsed();

                    allocation_infos.push(alloc_info);
                    actual_point_counts.push(pts_len);
                }

                total_time
            });
        },
    );

    print_alloc_summary_from_samples::<D>(&allocation_infos, &actual_point_counts);
}

/// Register no allocation workloads without the instrumented allocator.
#[cfg(not(feature = "count-allocations"))]
const fn benchmark_memory_profiling(_c: &mut Criterion) {}

/// Memory usage profiling across different scales and dimensions using allocation counter
#[cfg(feature = "count-allocations")]
fn benchmark_memory_profiling(c: &mut Criterion) {
    let counts = if is_dev_mode() {
        &[1_000, 10_000][..]
    } else {
        &[1_000, 10_000, 100_000][..]
    };

    let mut group = c.benchmark_group("memory_profiling");
    group.measurement_time(bench_time(30));

    for &count in counts {
        // 2D Memory Profiling
        bench_memory_usage::<2>(&mut group, "memory_usage_2d", count);

        // 3D Memory Profiling (smaller counts due to complexity)
        if count <= 10_000 {
            bench_memory_usage::<3>(&mut group, "memory_usage_3d", count);
        }

        // 4D Memory Profiling (even smaller counts due to exponential complexity)
        if count <= 3_000 {
            bench_memory_usage::<4>(&mut group, "memory_usage_4d", count);
        }

        // 5D Memory Profiling (very small counts due to very high complexity)
        if count <= 1_000 {
            bench_memory_usage::<5>(&mut group, "memory_usage_5d", count);
        }
    }

    group.finish();
}

/// Registers RSS diagnostics without constructing allocation fixtures.
fn bench_rss(c: &mut Criterion) {
    scale_cases!(c, bench_rss_memory_usage);
}

criterion_group!(name = benches; config = criterion_config(); targets = bench_rss, benchmark_memory_profiling);
criterion_main!(benches);

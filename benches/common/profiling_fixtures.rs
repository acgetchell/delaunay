//! Deterministic fixtures and configuration shared by manual diagnostic targets.
#![allow(
    dead_code,
    reason = "Each focused target uses only its diagnostic family's helpers"
)]

use crate::bench_utils::OrAbort;
use criterion::Criterion;
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, RetryPolicy, Vertex,
};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange, ExactPredicates, Point};
use delaunay::try_vertices_from_points;
use std::{env, num::NonZeroUsize, sync::OnceLock, time::Duration};
#[cfg(feature = "bench-logging")]
use tracing_subscriber::EnvFilter;

pub const DEFAULT_SEED: u64 = 42;
/// Builds a benchmark point from hard-coded finite coordinates.
pub fn finite_point<const D: usize>(coords: [f64; D]) -> Point<D> {
    Point::try_new(coords).unwrap_or_else(|_| std::process::abort())
}

/// Parses a non-zero retry count used by fixed benchmark policies.
pub fn retry_attempts(value: usize) -> NonZeroUsize {
    let Some(attempts) = NonZeroUsize::new(value) else {
        unreachable!("hard-coded retry attempt count must be non-zero");
    };
    attempts
}

/// Parses benchmark coordinate bounds and aborts if the fixture is invalid.
pub fn coordinate_range(min: f64, max: f64) -> CoordinateRange<f64> {
    CoordinateRange::try_new(min, max).or_abort()
}

/// Returns broad bounds used by general random benchmark point clouds.
pub fn wide_bounds() -> CoordinateRange<f64> {
    coordinate_range(-100.0, 100.0)
}

/// Returns compact bounds used by adversarial benchmark point clouds.
pub fn adversarial_bounds() -> CoordinateRange<f64> {
    coordinate_range(-1.0, 1.0)
}

/// Parse `PROFILING_DEV_MODE` environment variable as boolean-like value
/// Returns true for: "1", "true", "TRUE", "yes", "on" (case-insensitive)
/// Returns false for anything else (including "0", "false", empty, or unset)
pub fn is_dev_mode() -> bool {
    let dev = env::var("PROFILING_DEV_MODE").ok();
    dev.as_deref().is_some_and(|s| {
        s == "1"
            || s.eq_ignore_ascii_case("true")
            || s.eq_ignore_ascii_case("yes")
            || s.eq_ignore_ascii_case("on")
    })
}

/// Parse `BENCH_LARGE_SCALE` as a boolean-like value.
pub fn is_large_scale_mode() -> bool {
    let large = env::var("BENCH_LARGE_SCALE").ok();
    large.as_deref().is_some_and(|s| {
        s == "1"
            || s.eq_ignore_ascii_case("true")
            || s.eq_ignore_ascii_case("yes")
            || s.eq_ignore_ascii_case("on")
    })
}

/// Return the deterministic base seed for random point generation.
pub fn benchmark_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        let seed = env::var("DELAUNAY_BENCH_SEED")
            .ok()
            .and_then(|s| {
                let s = s.trim();
                s.strip_prefix("0x")
                    .or_else(|| s.strip_prefix("0X"))
                    .map_or_else(|| s.parse().ok(), |hex| u64::from_str_radix(hex, 16).ok())
            })
            .unwrap_or(DEFAULT_SEED);

        if env::var("PRINT_BENCH_SEED").is_ok() {
            #[cfg(feature = "bench-logging")]
            tracing::info!("Benchmark seed: 0x{seed:X} ({seed})");
        }

        seed
    })
}

pub fn benchmark_retry_attempts() -> NonZeroUsize {
    static ATTEMPTS: OnceLock<NonZeroUsize> = OnceLock::new();
    *ATTEMPTS.get_or_init(|| {
        let attempts = env::var("DELAUNAY_BENCH_RETRY_ATTEMPTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6)
            .max(1);

        let Some(attempts) = NonZeroUsize::new(attempts) else {
            unreachable!("attempts clamped to >= 1");
        };
        attempts
    })
}

pub fn seed_for_case<const D: usize>(n_points: usize) -> u64 {
    const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

    let base_seed = benchmark_seed();
    base_seed
        .wrapping_add((n_points as u64).wrapping_mul(SEED_SALT))
        .wrapping_add((D as u64).wrapping_mul(SEED_SALT.rotate_left(17)))
}

pub fn construction_options(seed: u64) -> ConstructionOptions {
    ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts: benchmark_retry_attempts(),
        base_seed: Some(seed),
    })
}

pub fn construct_triangulation<const D: usize>(
    vertices: &[Vertex<(), D>],
    seed: u64,
) -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    DelaunayTriangulation::builder(vertices)
        .construction_options(construction_options(seed))
        .build()
        .or_abort()
}

/// Certifies immutable diagnostic fixtures before read-only timing begins.
pub fn validated_fixture<const D: usize>(
    count: usize,
    seed: u64,
) -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let points = generated_points_in_range::<D>(count, wide_bounds(), seed);
    let vertices = vertices_from_points(&points);
    let dt = construct_triangulation::<D>(&vertices, seed);
    assert_eq!(dt.number_of_vertices(), count);
    dt.validate().or_abort();
    dt
}

/// Converts generated point fixtures into benchmark vertices without attaching data.
pub fn vertices_from_points<const D: usize>(points: &[Point<D>]) -> Vec<Vertex<(), D>> {
    try_vertices_from_points(points).or_abort()
}

/// Generates deterministic benchmark points inside validated bounds.
pub fn generated_points_in_range<const D: usize>(
    count: usize,
    bounds: CoordinateRange<f64>,
    seed: u64,
) -> Vec<Point<D>> {
    generate_random_points_in_range_seeded::<D>(count, bounds, seed).or_abort()
}

/// Helper function to parse benchmark measurement time from environment
/// Guards against zero/invalid values by ensuring minimum of 1 second
pub fn bench_time(default_secs: u64) -> Duration {
    let secs = env::var("BENCH_MEASUREMENT_TIME")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map_or_else(|| default_secs.max(1), |parsed| parsed.max(1));
    Duration::from_secs(secs)
}

/// Point distribution types for comprehensive testing
#[derive(Debug, Clone, Copy)]
pub enum PointDistribution {
    Random,
    Adversarial,
}

/// Generate points according to the specified distribution
pub fn gen_points<const D: usize>(
    count: usize,
    distribution: PointDistribution,
    seed: u64,
) -> Vec<Point<D>> {
    match distribution {
        PointDistribution::Random => generated_points_in_range(count, wide_bounds(), seed),
        PointDistribution::Adversarial => generated_points_in_range::<D>(
            count,
            adversarial_bounds(),
            seed ^ 0xA5A5_A5A5_A5A5_A5A5,
        )
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let index = u32::try_from(index).or_abort();
            let mut coords = [0.0_f64; D];
            for (axis, coord) in coords.iter_mut().enumerate() {
                let axis_number = u32::try_from(axis + 1).or_abort();
                let base: f64 = point.coords()[axis];
                let cluster_offset = f64::from(index % 7) * 1.0e-3;
                let axis_offset = f64::from(axis_number) * 0.25;
                let perturbation = f64::from((index + axis_number) % 11) * 1.0e-6;
                *coord = base.mul_add(1.0e3, 1.0e9 + axis_offset + cluster_offset + perturbation);
            }
            finite_point(coords)
        })
        .collect(),
    }
}

/// Configures manual diagnostics without changing the curated release experiment.
pub fn criterion_config() -> Criterion {
    #[cfg(feature = "bench-logging")]
    {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    }
    let sample_size = env::var("BENCH_SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_usize)
        .max(10);
    let warm_up_secs = env::var("BENCH_WARMUP_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    Criterion::default()
        .sample_size(sample_size)
        .warm_up_time(Duration::from_secs(warm_up_secs))
        .measurement_time(bench_time(60))
}

/// Keeps the original large-scale input grid shared across diagnostic families.
#[allow(
    unused_macros,
    reason = "Query-only targets do not register the scaling grid"
)]
macro_rules! scale_cases {
    ($c:ident, $($bench:ident),+) => {{
        for &n in &[1000, 5000, 10_000] { $( $bench::<2>($c, "2D", n); $bench::<3>($c, "3D", n); )+ }
        let counts: &[usize] = if $crate::profiling_fixtures::is_large_scale_mode() { &[1000, 5000, 10_000] } else { &[1000, 3000] };
        for &n in counts { $( $bench::<4>($c, "4D", n); )+ }
        for &n in &[500, 1000] { $( $bench::<5>($c, "5D", n); )+ }
    }};
}
#[allow(
    unused_imports,
    reason = "Query-only targets do not register the scaling grid"
)]
pub(crate) use scale_cases;

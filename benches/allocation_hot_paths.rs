#![forbid(unsafe_code)]

//! Allocation-contract microbenchmarks for public hot paths.
//!
//! Run with:
//!
//! ```bash
//! cargo bench --profile perf --bench allocation_hot_paths --features count-allocations -- --noplot
//! ```
//!
//! Without `count-allocations`, this target compiles and reports a no-op
//! placeholder so workspace benchmark compile checks remain feature-neutral.

use criterion::{criterion_group, criterion_main};
#[cfg(feature = "count-allocations")]
#[path = "common/bench_utils.rs"]
mod bench_utils;

#[cfg(feature = "count-allocations")]
mod allocation_contracts {
    use allocation_counter::AllocationInfo;
    use approx::assert_relative_eq;
    use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
    use delaunay::prelude::algorithms::LocateResult;
    use delaunay::prelude::construction::{
        ConstructionOptions, DelaunayIncrementalBuilder, DelaunayTriangulation, RetryPolicy, Vertex,
    };
    use delaunay::prelude::generators::generate_random_points_in_range_seeded;
    use delaunay::prelude::geometry::{
        AdaptiveKernel, CoordinateRange, ExactPredicates, Point, simplex_volume,
    };
    use delaunay::prelude::query::measure_with_result;
    use delaunay::prelude::tds::{SimplexKey, TdsError, VertexKey, facet_key_from_vertices};
    use delaunay::{try_vertices_from_points, vertex};
    use std::assert_matches;
    use std::{hint::black_box, num::NonZeroUsize, time::Duration};
    use thiserror::Error;

    use super::bench_utils::{OrAbort, OrAbortWithContext};

    const CANARY_COUNT_2D: usize = 4_000;
    const CANARY_COUNT_3D: usize = 750;
    const CANARY_COUNT_4D: usize = 75;
    const CANARY_COUNT_5D: usize = 25;
    const CANARY_SEED_2D: u64 = 4_042;
    const CANARY_SEED_3D: u64 = 873;
    const CANARY_SEED_4D: u64 = 531;
    const CANARY_SEED_5D: u64 = 816;
    const SAMPLE_SIZE: usize = 32;
    type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

    #[derive(Debug, Error)]
    enum AllocationBenchError {
        #[error("{dimension}D fixture did not contain a simplex")]
        MissingSimplex { dimension: usize },

        #[error("{dimension}D fixture simplex has {actual} vertices, expected at least {required}")]
        SimplexTooSmall {
            dimension: usize,
            required: usize,
            actual: usize,
        },

        #[error("{dimension}D fixture vertex {vertex_key:?} was missing")]
        MissingVertex {
            dimension: usize,
            vertex_key: VertexKey,
        },

        #[error("TDS lookup failed: {source}")]
        Tds {
            #[from]
            source: TdsError,
        },
    }

    struct DimensionFixture<const D: usize> {
        dt: BenchTriangulation<D>,
        simplex_key: SimplexKey,
        facet_vertices: [VertexKey; D],
        query: Point<D>,
        simplex_count: usize,
        vertex_count: usize,
    }

    fn retry_attempts(value: usize) -> NonZeroUsize {
        let Some(attempts) = NonZeroUsize::new(value) else {
            unreachable!("hard-coded retry attempt count must be non-zero");
        };
        attempts
    }

    fn benchmark_bounds() -> CoordinateRange<f64> {
        CoordinateRange::try_new(-100.0_f64, 100.0).or_abort()
    }

    fn canary_vertices<const D: usize>(count: usize, seed: u64) -> Vec<Vertex<(), D>> {
        let points =
            generate_random_points_in_range_seeded::<D>(count, benchmark_bounds(), seed).or_abort();
        try_vertices_from_points(&points).or_abort()
    }

    fn first_simplex_key<const D: usize>(
        dt: &BenchTriangulation<D>,
    ) -> Result<SimplexKey, AllocationBenchError> {
        dt.simplices()
            .map(|(simplex_key, _)| simplex_key)
            .next()
            .ok_or(AllocationBenchError::MissingSimplex { dimension: D })
    }

    fn simplex_points<const D: usize>(
        dt: &BenchTriangulation<D>,
        simplex_key: SimplexKey,
    ) -> Result<Vec<Point<D>>, AllocationBenchError> {
        dt.simplex_vertices(simplex_key)?
            .iter()
            .copied()
            .map(|vertex_key| {
                dt.vertex(vertex_key).map(|vertex| *vertex.point()).ok_or(
                    AllocationBenchError::MissingVertex {
                        dimension: D,
                        vertex_key,
                    },
                )
            })
            .collect()
    }

    fn representative_simplex_key<const D: usize>(
        dt: &BenchTriangulation<D>,
    ) -> Result<SimplexKey, AllocationBenchError> {
        let mut best: Option<(SimplexKey, f64)> = None;

        for (simplex_key, _) in dt.simplices() {
            let points = simplex_points(dt, simplex_key)?;
            let Ok(volume) = simplex_volume(&points) else {
                continue;
            };
            let volume = volume.abs();
            if !volume.is_finite() || volume <= 0.0 {
                continue;
            }

            match best {
                Some((_, best_volume)) if best_volume >= volume => {}
                _ => best = Some((simplex_key, volume)),
            }
        }

        best.map_or_else(|| first_simplex_key(dt), |(simplex_key, _)| Ok(simplex_key))
    }

    fn first_facet_vertices<const D: usize>(
        dt: &BenchTriangulation<D>,
        simplex_key: SimplexKey,
    ) -> Result<[VertexKey; D], AllocationBenchError> {
        let vertices = dt.simplex_vertices(simplex_key)?;
        if vertices.len() < D {
            return Err(AllocationBenchError::SimplexTooSmall {
                dimension: D,
                required: D,
                actual: vertices.len(),
            });
        }

        let mut facet_vertices = [vertices[0]; D];
        facet_vertices.copy_from_slice(&vertices[..D]);
        Ok(facet_vertices)
    }

    fn prepare_fixture<const D: usize>(count: usize, seed: u64) -> DimensionFixture<D>
    where
        AdaptiveKernel<f64>: ExactPredicates<D>,
    {
        let vertices = canary_vertices::<D>(count, seed);
        let attempts = retry_attempts(6);
        let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
            attempts,
            base_seed: Some(seed),
        });
        let dt: BenchTriangulation<D> = DelaunayTriangulation::builder(&vertices)
            .construction_options(options)
            .build()
            .or_abort();
        let simplex_key = representative_simplex_key(&dt).or_abort();
        let facet_vertices = first_facet_vertices(&dt, simplex_key).or_abort();
        let query = dt.simplex_barycenter(simplex_key).or_abort();
        let simplex_count = dt.number_of_simplices();
        let vertex_count = dt.number_of_vertices();

        DimensionFixture {
            dt,
            simplex_key,
            facet_vertices,
            query,
            simplex_count,
            vertex_count,
        }
    }

    fn assert_zero_allocations(info: &AllocationInfo, operation: &str) {
        assert_eq!(
            info.count_total, 0,
            "{operation} should not allocate; allocation info: {info:?}"
        );
        assert_eq!(
            info.bytes_total, 0,
            "{operation} should allocate zero bytes; allocation info: {info:?}"
        );
        assert_eq!(
            info.count_current, 0,
            "{operation} should not retain allocations; allocation info: {info:?}"
        );
        assert_eq!(
            info.bytes_current, 0,
            "{operation} should retain zero bytes; allocation info: {info:?}"
        );
    }

    fn assert_allocation_budget(info: &AllocationInfo, operation: &str, max_allocations: u64) {
        assert!(
            info.count_total <= max_allocations,
            "{operation} exceeded allocation budget {max_allocations}; allocation info: {info:?}"
        );
        assert_eq!(
            info.count_current, 0,
            "{operation} should not retain allocations; allocation info: {info:?}"
        );
        assert_eq!(
            info.bytes_current, 0,
            "{operation} should retain zero bytes; allocation info: {info:?}"
        );
    }

    /// Keeps first-simplex publication bounded and verifies all temporary heap
    /// ownership is released with the published owner.
    fn assert_bootstrap_allocation_budget<const D: usize>(info: &AllocationInfo) {
        const MAX_BOOTSTRAP_ALLOCATIONS: u64 = 2_000;
        const MAX_BOOTSTRAP_BYTES: u64 = 8 * 1024 * 1024;
        assert!(
            info.count_total <= MAX_BOOTSTRAP_ALLOCATIONS,
            "{D}D bootstrap publication exceeded {MAX_BOOTSTRAP_ALLOCATIONS} allocations; allocation info: {info:?}"
        );
        assert!(
            info.bytes_total <= MAX_BOOTSTRAP_BYTES,
            "{D}D bootstrap publication exceeded {MAX_BOOTSTRAP_BYTES} allocated bytes; allocation info: {info:?}"
        );
        assert_eq!(
            info.count_current, 0,
            "{D}D bootstrap publication should retain no allocations after its owner is dropped; allocation info: {info:?}"
        );
        assert_eq!(
            info.bytes_current, 0,
            "{D}D bootstrap publication should retain no bytes after its owner is dropped; allocation info: {info:?}"
        );
    }

    /// Calibrates insertion against one owner clone while leaving headroom for
    /// dimension-dependent exact-predicate fallback allocations.
    fn assert_post_bootstrap_insertion_budget<const D: usize>(
        info: &AllocationInfo,
        single_owner_clone: &AllocationInfo,
    ) {
        let allocation_headroom = match D {
            2 | 3 => 10_000,
            4 => 100_000,
            5 => 5_000_000,
            _ => 10_000_000,
        };
        let byte_headroom = match D {
            2 | 3 => 2 * 1024 * 1024,
            4 => 8 * 1024 * 1024,
            5 => 128 * 1024 * 1024,
            _ => 256 * 1024 * 1024,
        };
        let max_allocations = single_owner_clone
            .count_total
            .saturating_mul(4)
            .saturating_add(allocation_headroom);
        let max_bytes = single_owner_clone
            .bytes_total
            .saturating_mul(2)
            .saturating_add(byte_headroom);
        assert!(
            info.count_total <= max_allocations,
            "{D}D post-bootstrap insertion exceeded the single-owner-clone calibrated allocation budget {max_allocations}; insertion={info:?}, clone={single_owner_clone:?}"
        );
        assert!(
            info.bytes_total <= max_bytes,
            "{D}D post-bootstrap insertion exceeded the single-owner-clone calibrated byte budget {max_bytes}; insertion={info:?}, clone={single_owner_clone:?}"
        );
        let current_allocations = u64::try_from(info.count_current).or_abort();
        let current_bytes = u64::try_from(info.bytes_current).or_abort();
        assert!(
            current_allocations <= info.count_total,
            "{D}D post-bootstrap insertion retained more allocations than it made; allocation info: {info:?}"
        );
        assert!(
            current_bytes <= info.bytes_total,
            "{D}D post-bootstrap insertion retained more bytes than it allocated; allocation info: {info:?}"
        );
    }

    const fn locate_fast_path_allocation_budget<const D: usize>() -> u64 {
        match D {
            2 | 3 => 1,
            4 => 2_000,
            5 => 4_000,
            _ => 10_000,
        }
    }

    /// Builds the canonical positively oriented axis simplex used at bootstrap.
    fn bootstrap_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(vertex!([0.0; D]).or_abort());
        for axis in 0..D {
            let mut coords = [0.0; D];
            coords[axis] = 1.0;
            vertices.push(vertex!(coords).or_abort());
        }
        vertices
    }

    /// Measures the complete public D+1 bootstrap and Levels 1–5 publication path.
    fn bench_bootstrap_publication<const D: usize>(group: &mut BenchmarkGroup<'_, WallTime>)
    where
        AdaptiveKernel<f64>: ExactPredicates<D>,
    {
        let vertices = bootstrap_vertices::<D>();

        group.bench_function(
            BenchmarkId::new(format!("bounded_alloc/bootstrap_publication_{D}d"), D + 1),
            |b| {
                b.iter(|| {
                    let (counts, info) = measure_with_result(|| {
                        let mut builder: DelaunayIncrementalBuilder<_, (), (), D> =
                            DelaunayIncrementalBuilder::new();
                        for vertex in vertices.iter().copied() {
                            builder.insert_vertex(vertex).or_abort();
                        }
                        let dt = builder.finish().or_abort();
                        let counts = (dt.number_of_vertices(), dt.number_of_simplices());
                        black_box(&dt);
                        drop(dt);
                        counts
                    });

                    assert_eq!(counts, (D + 1, 1));
                    assert_bootstrap_allocation_budget::<D>(&info);
                });
            },
        );
    }

    /// Measures one public insertion into an existing calibrated owner without
    /// counting the Criterion fixture clone.
    fn bench_post_bootstrap_insertion<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        fixture: &DimensionFixture<D>,
    ) {
        let candidate = vertex!(*fixture.query.coords()).or_abort();
        let ((), single_owner_clone) = measure_with_result(|| drop(fixture.dt.clone()));
        let vertex_count = fixture.vertex_count;

        group.bench_function(
            BenchmarkId::new(
                format!("bounded_alloc/post_bootstrap_insert_{D}d"),
                vertex_count,
            ),
            |b| {
                b.iter_batched(
                    || fixture.dt.clone(),
                    |mut dt| {
                        let (inserted, info) =
                            measure_with_result(|| dt.insert_vertex(candidate).map_err(Box::new));
                        let vertex_key = inserted.or_abort();
                        assert_eq!(dt.number_of_vertices(), vertex_count + 1);
                        assert!(dt.vertex(vertex_key).is_some());
                        assert_post_bootstrap_insertion_budget::<D>(&info, &single_owner_clone);
                        black_box(dt);
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    fn bench_public_iterators<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        fixture: &DimensionFixture<D>,
    ) {
        let tri = fixture.dt.as_triangulation();
        let simplex_count = fixture.simplex_count;
        let vertex_count = fixture.vertex_count;

        group.bench_function(
            BenchmarkId::new(format!("zero_alloc/public_iterators_{D}d"), vertex_count),
            |b| {
                b.iter(|| {
                    let (counts, info) = measure_with_result(|| {
                        black_box((
                            tri.simplices().count(),
                            tri.vertices().count(),
                            fixture.dt.simplices().count(),
                            fixture.dt.vertices().count(),
                        ))
                    });

                    assert_eq!(
                        counts,
                        (simplex_count, vertex_count, simplex_count, vertex_count,)
                    );
                    assert_zero_allocations(&info, "public simplices()/vertices() iterators");
                });
            },
        );
    }

    fn bench_simplex_vertices<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        fixture: &DimensionFixture<D>,
    ) {
        let simplex_key = fixture.simplex_key;

        group.bench_function(
            BenchmarkId::new(
                format!("zero_alloc/simplex_vertices_{D}d"),
                fixture.vertex_count,
            ),
            |b| {
                b.iter(|| {
                    let (vertex_count, info) = measure_with_result(|| {
                        fixture
                            .dt
                            .simplex_vertices(simplex_key)
                            .map(<[VertexKey]>::len)
                    });
                    assert_eq!(vertex_count.or_abort(), D + 1);
                    assert_zero_allocations(&info, "DelaunayTriangulation::simplex_vertices");
                });
            },
        );
    }

    fn bench_simplex_barycenter<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        fixture: &DimensionFixture<D>,
    ) {
        let simplex_key = fixture.simplex_key;

        group.bench_function(
            BenchmarkId::new(
                format!("zero_alloc/simplex_barycenter_{D}d"),
                fixture.vertex_count,
            ),
            |b| {
                b.iter(|| {
                    let (barycenter, info) = measure_with_result(|| {
                        black_box(fixture.dt.simplex_barycenter(simplex_key))
                    });
                    let barycenter = barycenter.or_abort();
                    assert_relative_eq!(
                        barycenter.coords().as_slice(),
                        fixture.query.coords().as_slice(),
                        epsilon = f64::EPSILON
                    );
                    assert_zero_allocations(&info, "DelaunayTriangulation::simplex_barycenter");
                    black_box(barycenter);
                });
            },
        );
    }

    fn bench_simplex_vertex_uuid_iter<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        fixture: &DimensionFixture<D>,
    ) {
        let simplex = fixture
            .dt
            .simplex(fixture.simplex_key)
            .or_abort(format!("{D}D benchmark simplex should exist"));

        group.bench_function(
            BenchmarkId::new(
                format!("zero_alloc/simplex_vertex_uuid_iter_{D}d"),
                fixture.vertex_count,
            ),
            |b| {
                b.iter(|| {
                    let (uuid_count, info) = measure_with_result(|| {
                        simplex
                            .vertices()
                            .iter()
                            .copied()
                            .filter(|&vertex_key| {
                                fixture.dt.vertex_uuid_from_key(vertex_key).is_some()
                            })
                            .count()
                    });
                    assert_eq!(uuid_count, D + 1);
                    assert_zero_allocations(&info, "DelaunayTriangulation::vertex_uuid_from_key");
                });
            },
        );
    }

    fn bench_facet_key_from_vertices<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        fixture: &DimensionFixture<D>,
    ) {
        group.bench_function(
            BenchmarkId::new(
                format!("zero_alloc/facet_key_from_vertices_{D}d"),
                fixture.vertex_count,
            ),
            |b| {
                b.iter(|| {
                    let (facet_key, info) = measure_with_result(|| {
                        black_box(facet_key_from_vertices(&fixture.facet_vertices))
                    });
                    assert_ne!(facet_key, 0);
                    assert_zero_allocations(&info, "facet_key_from_vertices");
                });
            },
        );
    }

    fn bench_locate_with_hint_fast_path<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        fixture: &DimensionFixture<D>,
    ) {
        let simplex_key = fixture.simplex_key;

        group.bench_function(
            BenchmarkId::new(
                format!("bounded_alloc/locate_with_hint_fast_kernel_{D}d"),
                fixture.vertex_count,
            ),
            |b| {
                b.iter(|| {
                    let (locate_result, info) = measure_with_result(|| {
                        fixture.dt.locate_with_stats(&fixture.query, Some(simplex_key))
                    });
                    let (location, stats) = locate_result.or_abort();

                    assert_matches!(location, LocateResult::InsideSimplex(found) if found == simplex_key);
                    assert!(stats.used_hint);
                    assert!(!stats.fell_back_to_scan());
                    assert_allocation_budget(
                        &info,
                        "hinted locate_with_stats fast path",
                        locate_fast_path_allocation_budget::<D>(),
                    );
                });
            },
        );
    }

    fn bench_dimension<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        count: usize,
        seed: u64,
    ) where
        AdaptiveKernel<f64>: ExactPredicates<D>,
    {
        let fixture = prepare_fixture::<D>(count, seed);

        bench_bootstrap_publication::<D>(group);
        bench_post_bootstrap_insertion(group, &fixture);
        bench_public_iterators(group, &fixture);
        bench_simplex_vertices(group, &fixture);
        bench_simplex_barycenter(group, &fixture);
        bench_simplex_vertex_uuid_iter(group, &fixture);
        bench_facet_key_from_vertices(group, &fixture);
        bench_locate_with_hint_fast_path(group, &fixture);
    }

    pub fn bench_allocation_hot_paths(c: &mut Criterion) {
        let mut group = c.benchmark_group("allocation_hot_paths");
        group.sample_size(SAMPLE_SIZE);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));

        bench_dimension::<2>(&mut group, CANARY_COUNT_2D, CANARY_SEED_2D);
        bench_dimension::<3>(&mut group, CANARY_COUNT_3D, CANARY_SEED_3D);
        bench_dimension::<4>(&mut group, CANARY_COUNT_4D, CANARY_SEED_4D);
        bench_dimension::<5>(&mut group, CANARY_COUNT_5D, CANARY_SEED_5D);
        group.finish();
    }
}

#[cfg(feature = "count-allocations")]
use allocation_contracts::bench_allocation_hot_paths;

#[cfg(not(feature = "count-allocations"))]
fn bench_allocation_hot_paths(c: &mut criterion::Criterion) {
    let mut group = c.benchmark_group("allocation_hot_paths");
    group.bench_function("count_allocations_feature_disabled", |b| b.iter(|| ()));
    group.finish();
}

criterion_group!(benches, bench_allocation_hot_paths);
criterion_main!(benches);

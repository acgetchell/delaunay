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
    use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
    use delaunay::prelude::algorithms::{LocateResult, locate_with_stats};
    use delaunay::prelude::construction::{
        ConstructionOptions, DelaunayTriangulation, RetryPolicy, Vertex,
    };
    use delaunay::prelude::generators::generate_random_points_in_range_seeded;
    use delaunay::prelude::geometry::{
        AdaptiveKernel, CoordinateRange, FastKernel, Point, simplex_volume,
    };
    use delaunay::prelude::query::measure_with_result;
    use delaunay::prelude::tds::{SimplexKey, TdsError, VertexKey, facet_key_from_vertices};
    use delaunay::try_vertices_from_points;
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

    fn finite_point<const D: usize>(coords: [f64; D]) -> Point<D> {
        Point::try_new(coords).unwrap_or_else(|_| std::process::abort())
    }

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
        dt.tds()
            .simplex_keys()
            .next()
            .ok_or(AllocationBenchError::MissingSimplex { dimension: D })
    }

    fn simplex_points<const D: usize>(
        dt: &BenchTriangulation<D>,
        simplex_key: SimplexKey,
    ) -> Result<Vec<Point<D>>, AllocationBenchError> {
        let tds = dt.tds();

        tds.simplex_vertices(simplex_key)?
            .iter()
            .copied()
            .map(|vertex_key| {
                tds.vertex(vertex_key).map(|vertex| *vertex.point()).ok_or(
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

        for simplex_key in dt.tds().simplex_keys() {
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
        let vertices = dt.tds().simplex_vertices(simplex_key)?;
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

    fn simplex_barycenter<const D: usize>(
        dt: &BenchTriangulation<D>,
        simplex_key: SimplexKey,
    ) -> Result<Point<D>, AllocationBenchError> {
        let points = simplex_points(dt, simplex_key)?;
        let mut coords = [0.0_f64; D];
        for point in &points {
            for (coord, vertex_coord) in coords.iter_mut().zip(point.coords()) {
                *coord += *vertex_coord;
            }
        }

        let vertex_count = u32::try_from(points.len()).or_abort();
        let inv_vertex_count = 1.0 / f64::from(vertex_count);
        for coord in &mut coords {
            *coord *= inv_vertex_count;
        }

        Ok(finite_point(coords))
    }

    fn prepare_fixture<const D: usize>(count: usize, seed: u64) -> DimensionFixture<D> {
        let vertices = canary_vertices::<D>(count, seed);
        let attempts = retry_attempts(6);
        let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
            attempts,
            base_seed: Some(seed),
        });
        let dt = BenchTriangulation::<D>::try_new_with_options(&vertices, options).or_abort();
        let simplex_key = representative_simplex_key(&dt).or_abort();
        let facet_vertices = first_facet_vertices(&dt, simplex_key).or_abort();
        let query = simplex_barycenter(&dt, simplex_key).or_abort();
        let simplex_count = dt.tds().simplices().count();
        let vertex_count = dt.tds().vertices().count();

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

    const fn locate_fast_path_allocation_budget<const D: usize>() -> u64 {
        match D {
            2 | 3 => 1,
            4 => 2_000,
            5 => 4_000,
            _ => 10_000,
        }
    }

    fn bench_public_iterators<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        fixture: &DimensionFixture<D>,
    ) {
        let tds = fixture.dt.tds();
        let tri = fixture.dt.as_triangulation();
        let simplex_count = fixture.simplex_count;
        let vertex_count = fixture.vertex_count;

        group.bench_function(
            BenchmarkId::new(format!("zero_alloc/public_iterators_{D}d"), vertex_count),
            |b| {
                b.iter(|| {
                    let (counts, info) = measure_with_result(|| {
                        black_box((
                            tds.simplices().count(),
                            tds.vertices().count(),
                            tds.simplex_keys().count(),
                            tds.vertex_keys().count(),
                            tri.simplices().count(),
                            tri.vertices().count(),
                            fixture.dt.simplices().count(),
                            fixture.dt.vertices().count(),
                        ))
                    });

                    assert_eq!(
                        counts,
                        (
                            simplex_count,
                            vertex_count,
                            simplex_count,
                            vertex_count,
                            simplex_count,
                            vertex_count,
                            simplex_count,
                            vertex_count,
                        )
                    );
                    assert_zero_allocations(
                        &info,
                        "TDS and public simplices()/vertices() iterators",
                    );
                });
            },
        );
    }

    fn bench_tds_simplex_vertices<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        fixture: &DimensionFixture<D>,
    ) {
        let tds = fixture.dt.tds();
        let simplex_key = fixture.simplex_key;

        group.bench_function(
            BenchmarkId::new(
                format!("zero_alloc/tds_simplex_vertices_{D}d"),
                fixture.vertex_count,
            ),
            |b| {
                b.iter(|| {
                    let (vertex_count, info) = measure_with_result(|| {
                        tds.simplex_vertices(simplex_key).map(|keys| keys.len())
                    });
                    assert_eq!(vertex_count.or_abort(), D + 1);
                    assert_zero_allocations(&info, "Tds::simplex_vertices");
                });
            },
        );
    }

    fn bench_simplex_vertex_uuid_iter<const D: usize>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        fixture: &DimensionFixture<D>,
    ) {
        let tds = fixture.dt.tds();
        let simplex = tds
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
                            .vertex_uuid_iter(tds)
                            .try_fold(0usize, |count, uuid| uuid.map(|_| count + 1))
                    });
                    assert_eq!(uuid_count.or_abort(), D + 1);
                    assert_zero_allocations(&info, "Simplex::vertex_uuid_iter");
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
        let kernel = FastKernel::<f64>::new();
        let simplex_key = fixture.simplex_key;

        group.bench_function(
            BenchmarkId::new(
                format!("bounded_alloc/locate_with_hint_fast_kernel_{D}d"),
                fixture.vertex_count,
            ),
            |b| {
                b.iter(|| {
                    let (locate_result, info) = measure_with_result(|| {
                        locate_with_stats(fixture.dt.tds(), &kernel, &fixture.query, Some(simplex_key))
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
    ) {
        let fixture = prepare_fixture::<D>(count, seed);

        bench_public_iterators(group, &fixture);
        bench_tds_simplex_vertices(group, &fixture);
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

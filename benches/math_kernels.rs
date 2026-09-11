//! Targeted math-kernel diagnostics, deliberately outside the release-signal suite.
//!
//! Analytical fixtures certify answers before sampling. Construction, fixture
//! validation, matrix-path checks, and reusable indexes stay outside timed work.
//! Every measured fallible operation aborts on failure rather than timing errors.

use approx::assert_relative_eq;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use delaunay::prelude::construction::DelaunayTriangulationBuilder;
use delaunay::prelude::geometry::{
    AdaptiveKernel, ExactPredicates, Matrix, Orientation, Point, circumradius, facet_measure,
    inradius, normalized_volume, radius_ratio, safe_usize_to_scalar, simplex_orientation,
    simplex_volume, surface_measure,
};
use delaunay::prelude::topology::validation::{
    count_boundary_simplices, count_simplices, euler_characteristic, validate_closed_boundary,
    validate_ridge_links, validate_vertex_links,
};
use delaunay::prelude::try_dedup_vertices_epsilon;
use delaunay::try_vertices_from_points;
use std::{hint::black_box, time::Duration};

/// Shared fatal-error adapters keep failed operations out of timing evidence.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, OrAbortWithContext};

/// Axis-aligned simplices have independent closed-form volume and radius oracles.
fn axis_simplex<const D: usize>(height: f64) -> Vec<Point<D>> {
    let mut points = vec![Point::try_new([0.0; D]).or_abort()];
    for axis in 0..D {
        let mut coordinates = [0.0; D];
        coordinates[axis] = if axis == D - 1 { height } else { 1.0 };
        points.push(Point::try_new(coordinates).or_abort());
    }
    points
}

/// Certifies the actual filter decision so a thin simplex is never merely
/// assumed to exercise exact fallback. `N` is the homogeneous matrix size.
fn bench_orientation<const D: usize, const N: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    assert_eq!(N, D + 1);
    let mut cancellation = axis_simplex::<D>(1.0);
    let mut first = [0.0; D];
    first[0] = 1.0;
    first[1] = 1.0;
    cancellation[1] = Point::try_new(first).or_abort();
    let mut second = first;
    second[1] += f64::EPSILON;
    cancellation[2] = Point::try_new(second).or_abort();
    let mut degenerate = cancellation.clone();
    degenerate[2] = Point::try_new(first.map(|coordinate| 2.0 * coordinate)).or_abort();

    // The homogeneous determinant of origin + basis rows has sign (-1)^D.
    // Replacing the leading 2x2 identity by [[1, 1], [1, 1 + eps]]
    // preserves that sign, while [[1, 1], [2, 2]] has determinant zero.
    let nonzero = if D.is_multiple_of(2) {
        Orientation::POSITIVE
    } else {
        Orientation::NEGATIVE
    };
    for (case, points, expected, expect_fast) in [
        ("well_conditioned", axis_simplex::<D>(1.0), nonzero, N <= 4),
        ("exact_nonzero", cancellation, nonzero, false),
        ("exact_zero", degenerate, Orientation::DEGENERATE, false),
    ] {
        let matrix = Matrix::<N>::try_from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                if column == D {
                    1.0
                } else {
                    points[row].coords()[column]
                }
            })
        }))
        .or_abort();
        let filter = matrix.det_direct_with_errbound().or_abort();
        let decisive = filter
            .is_some_and(|estimate| estimate.determinant().abs() > estimate.absolute_error_bound());
        assert_eq!(decisive, expect_fast, "{case}/{D}d filter path changed");
        assert_eq!(simplex_orientation(&points).or_abort(), expected);

        group.bench_function(BenchmarkId::new(case, format!("{D}d")), |b| {
            b.iter(|| black_box(simplex_orientation(black_box(&points)).or_abort()));
        });
    }
}

/// Measures geometric helpers on regular-scale and thin, still valid simplices.
#[expect(
    clippy::too_many_lines,
    reason = "Keep each geometry fixture, independent oracles, and timed cases together"
)]
fn bench_geometry<const D: usize>(group: &mut BenchmarkGroup<'_, WallTime>, height: f64)
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let dimension = safe_usize_to_scalar(D).or_abort();
    let factorial = (1..=D)
        .map(|value| safe_usize_to_scalar(value).or_abort())
        .product::<f64>();
    let points = axis_simplex::<D>(height);
    let vertices = try_vertices_from_points(&points).or_abort();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .build()
        .or_abort();
    dt.validate().or_abort();
    assert_eq!(dt.number_of_simplices(), 1);
    let simplex_key = dt.simplices().next().or_abort("missing axis simplex").0;
    let tri = dt.as_triangulation();
    let boundary_facets = dt
        .boundary_facets()
        .or_abort()
        .collect::<Result<Vec<_>, _>>()
        .or_abort();

    // For axis lengths (1, ..., 1, h), volume = h/D!, R = ||a||/2,
    // and r = 1 / (sum(1/a_i) + ||1/a||). Count the three edge families
    // analytically instead of using the production edge/volume helpers.
    let volume = height / factorial;
    let facet = &points[1..];
    let facet_volume = dimension * volume * (dimension - 1.0 + height.powi(-2)).sqrt();
    let surface = (dimension * volume).mul_add(dimension - 1.0 + height.recip(), facet_volume);
    let radius = height.mul_add(height, dimension - 1.0).sqrt() / 2.0;
    let inner_radius =
        1.0 / (dimension - 1.0 + 1.0 / height + (dimension - 1.0 + height.powi(-2)).sqrt());
    let edge_sum = [
        dimension - 1.0 + height,
        (dimension - 1.0) * (dimension - 2.0) / 2.0 * 2.0_f64.sqrt(),
        (dimension - 1.0) * height.mul_add(height, 1.0).sqrt(),
    ]
    .into_iter()
    .sum::<f64>();
    let mean_edge = edge_sum / (dimension * (dimension + 1.0) / 2.0);
    let normalized = volume / mean_edge.powi(i32::try_from(D).or_abort());
    assert_relative_eq!(
        simplex_volume(&points).or_abort(),
        volume,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        facet_measure(facet).or_abort(),
        facet_volume,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        surface_measure(&boundary_facets).or_abort(),
        surface,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        circumradius(&points).or_abort(),
        radius,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        inradius(&points).or_abort(),
        inner_radius,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        radius_ratio(tri, simplex_key).or_abort(),
        radius / inner_radius,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        normalized_volume(tri, simplex_key).or_abort(),
        normalized,
        max_relative = 1e-10
    );
    let barycenter = tri.simplex_barycenter(simplex_key).or_abort();
    for (axis, coordinate) in barycenter.coords().iter().enumerate() {
        let axis_length = if axis == D - 1 { height } else { 1.0 };
        assert_relative_eq!(
            *coordinate,
            axis_length / (dimension + 1.0),
            max_relative = 1e-12
        );
    }

    group.bench_function(BenchmarkId::new("volume", format!("{D}d")), |b| {
        b.iter(|| black_box(simplex_volume(black_box(&points)).or_abort()));
    });
    group.bench_function(BenchmarkId::new("facet_measure", format!("{D}d")), |b| {
        b.iter(|| black_box(facet_measure(black_box(facet)).or_abort()));
    });
    group.bench_function(BenchmarkId::new("surface_measure", format!("{D}d")), |b| {
        b.iter(|| black_box(surface_measure(black_box(&boundary_facets)).or_abort()));
    });
    group.bench_function(BenchmarkId::new("circumradius", format!("{D}d")), |b| {
        b.iter(|| black_box(circumradius(black_box(&points)).or_abort()));
    });
    group.bench_function(BenchmarkId::new("inradius", format!("{D}d")), |b| {
        b.iter(|| black_box(inradius(black_box(&points)).or_abort()));
    });
    group.bench_function(BenchmarkId::new("radius_ratio", format!("{D}d")), |b| {
        b.iter(|| black_box(radius_ratio(black_box(tri), black_box(simplex_key)).or_abort()));
    });
    group.bench_function(
        BenchmarkId::new("normalized_volume", format!("{D}d")),
        |b| {
            b.iter(|| {
                black_box(normalized_volume(black_box(tri), black_box(simplex_key)).or_abort())
            });
        },
    );
    group.bench_function(BenchmarkId::new("barycenter", format!("{D}d")), |b| {
        b.iter(|| {
            black_box(
                black_box(tri)
                    .simplex_barycenter(black_box(simplex_key))
                    .or_abort(),
            )
        });
    });
}

/// Small exact combinatorial counts provide a topology oracle independent of TDS traversal.
fn binomial(n: usize, k: usize) -> usize {
    (0..k).fold(1, |value, index| value * (n - index) / (index + 1))
}

/// A cone over the cross-polytope boundary has both boundary and interior links,
/// 2^D maximal simplices, and an analytical f-vector in every tested dimension.
fn bench_topology<const D: usize>(group: &mut BenchmarkGroup<'_, WallTime>)
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let mut points = vec![Point::try_new([0.0; D]).or_abort()];
    for axis in 0..D {
        for sign in [-1.0, 1.0] {
            let mut coordinates = [0.0; D];
            coordinates[axis] = sign;
            points.push(Point::try_new(coordinates).or_abort());
        }
    }
    let vertices = try_vertices_from_points(&points).or_abort();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .build()
        .or_abort();
    dt.validate().or_abort();
    let tri = dt.as_triangulation();
    // Public demotion supplies a separate Levels 1-2 owner for component scans;
    // retain the original proof-bearing owner for the cumulative Level 3 case.
    let storage = tri.clone().into_tds();
    let tds = &storage;
    let index = tds.build_facet_to_simplices_index().or_abort();
    let topology = tri.global_topology();
    let counts = count_simplices(tds).or_abort();
    let boundary = count_boundary_simplices(tds, topology).or_abort();
    for k in 0..=D {
        let boundary_count = if k < D {
            (1 << (k + 1)) * binomial(D, k + 1)
        } else {
            0
        };
        let cone_count = if k == 0 { 1 } else { (1 << k) * binomial(D, k) };
        assert_eq!(boundary.count(k), boundary_count, "{D}d boundary f_{k}");
        assert_eq!(
            counts.count(k),
            boundary_count + cone_count,
            "{D}d cone f_{k}"
        );
    }
    assert_eq!(euler_characteristic(&counts), 1);
    assert_eq!(dt.number_of_simplices(), 1 << D);
    let ridge_count = tri.ridges().map(OrAbort::or_abort).count();
    assert_eq!(ridge_count, counts.count(D - 2));
    validate_closed_boundary(&index, topology).or_abort();
    validate_ridge_links(tds).or_abort();
    // A demoted TDS has no construction provenance. The standalone checker
    // cannot certify 3D-or-higher links; retain owner-level validation for D>=4.
    if D <= 3 {
        validate_vertex_links(&index, topology).or_abort();
    }

    group.throughput(Throughput::Elements(
        u64::try_from(dt.number_of_simplices()).or_abort(),
    ));
    let size = format!("{D}d_{}simplices", dt.number_of_simplices());
    group.bench_function(BenchmarkId::new("euler_with_face_count", &size), |b| {
        b.iter(|| {
            let counts = count_simplices(black_box(tds)).or_abort();
            black_box(euler_characteristic(&counts))
        });
    });
    group.bench_function(BenchmarkId::new("ridges", &size), |b| {
        b.iter(|| black_box(black_box(tri).ridges().map(OrAbort::or_abort).count()));
    });
    group.bench_function(
        BenchmarkId::new("closed_boundary_prebuilt_index", &size),
        |b| {
            b.iter(|| validate_closed_boundary(black_box(&index), black_box(topology)).or_abort());
        },
    );
    group.bench_function(BenchmarkId::new("ridge_links", &size), |b| {
        b.iter(|| validate_ridge_links(black_box(tds)).or_abort());
    });
    if D <= 3 {
        group.bench_function(
            BenchmarkId::new("vertex_links_prebuilt_index", &size),
            |b| {
                b.iter(|| validate_vertex_links(black_box(&index), black_box(topology)).or_abort());
            },
        );
    }
    group.bench_function(BenchmarkId::new("is_valid_topology", &size), |b| {
        b.iter(|| black_box(tri).is_valid_topology().or_abort());
    });
}

/// Axis clusters have exact binary distances and an independently known survivor order.
fn bench_epsilon_deduplication<const D: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    for (case, offsets) in [
        ("separated", &[0.0][..]),
        ("near_duplicates", &[0.0, 0.25][..]),
    ] {
        let mut points = Vec::new();
        for index in 0..64 {
            let position = 2.0 * safe_usize_to_scalar(index).or_abort();
            for offset in offsets {
                let mut coordinates = [0.0; D];
                coordinates[0] = position + offset;
                points.push(Point::try_new(coordinates).or_abort());
            }
        }
        let vertices = try_vertices_from_points(&points).or_abort();
        let unique = try_dedup_vertices_epsilon(&vertices, 0.5).or_abort();
        assert_eq!(unique.len(), 64);
        for (index, vertex) in unique.iter().enumerate() {
            let mut expected = [0.0; D];
            expected[0] = 2.0 * safe_usize_to_scalar(index).or_abort();
            assert_eq!(
                vertex.point().coords().map(f64::to_bits),
                expected.map(f64::to_bits)
            );
        }
        group.bench_function(BenchmarkId::new(case, format!("{D}d")), |b| {
            b.iter(|| {
                black_box(
                    try_dedup_vertices_epsilon(black_box(&vertices), black_box(0.5)).or_abort(),
                )
            });
        });
    }
}

/// Registers the same kernel families across the supported routine dimensions.
fn bench_math_kernels(c: &mut Criterion) {
    // Each group must span every dimension: finishing another group with the
    // same name would overwrite its HTML summary with only the latest cases.
    {
        let mut orientation = c.benchmark_group("math/orientation");
        orientation.throughput(Throughput::Elements(1));
        bench_orientation::<2, 3>(&mut orientation);
        bench_orientation::<3, 4>(&mut orientation);
        bench_orientation::<4, 5>(&mut orientation);
        bench_orientation::<5, 6>(&mut orientation);
        orientation.finish();
    }

    for (case, height) in [("axis_simplex", 1.0), ("thin_axis_simplex", 1.0 / 1024.0)] {
        let mut geometry = c.benchmark_group(format!("math/geometry/{case}"));
        geometry.throughput(Throughput::Elements(1));
        bench_geometry::<2>(&mut geometry, height);
        bench_geometry::<3>(&mut geometry, height);
        bench_geometry::<4>(&mut geometry, height);
        bench_geometry::<5>(&mut geometry, height);
        bench_geometry::<6>(&mut geometry, height);
        geometry.finish();
    }

    {
        let mut deduplication = c.benchmark_group("math/epsilon_deduplication");
        bench_epsilon_deduplication::<2>(&mut deduplication);
        bench_epsilon_deduplication::<3>(&mut deduplication);
        bench_epsilon_deduplication::<4>(&mut deduplication);
        bench_epsilon_deduplication::<5>(&mut deduplication);
        deduplication.finish();
    }

    let mut topology = c.benchmark_group("math/topology/cross_polytope");
    bench_topology::<2>(&mut topology);
    bench_topology::<3>(&mut topology);
    bench_topology::<4>(&mut topology);
    bench_topology::<5>(&mut topology);
    topology.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = bench_math_kernels
}
criterion_main!(benches);

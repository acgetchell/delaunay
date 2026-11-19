//! Measure rollback after a 3D insertion's conflict-region predicate fails.

#![forbid(unsafe_code)]

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use delaunay::prelude::algorithms::ConflictError;
use delaunay::prelude::construction::{DelaunayTriangulation, Vertex};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{
    AdaptiveKernel, CoordinateConversionError, CoordinateRange, ExactPredicates, Kernel, Point,
};
use delaunay::prelude::insertion::InsertionError;
use delaunay::{try_vertices_from_points, vertex};

#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, abort_benchmark};

const REJECTED_COORDS: [f64; 3] = [0.375, 0.25, 0.125];
const INJECTED_ERROR: CoordinateConversionError =
    CoordinateConversionError::UnsupportedMatrixDimension {
        requested: 4,
        max: 3,
    };

#[derive(Clone)]
struct FailingQueryKernel;

impl Kernel<3> for FailingQueryKernel {
    type Scalar = f64;

    fn orientation(&self, points: &[Point<3>]) -> Result<i32, CoordinateConversionError> {
        AdaptiveKernel::new().orientation(points)
    }

    fn in_sphere(
        &self,
        points: &[Point<3>],
        query: &Point<3>,
    ) -> Result<i32, CoordinateConversionError> {
        if query.coords().map(f64::to_bits) == REJECTED_COORDS.map(f64::to_bits) {
            return Err(INJECTED_ERROR);
        }
        AdaptiveKernel::new().in_sphere(points, query)
    }
}

impl ExactPredicates<3> for FailingQueryKernel {}

type Triangulation3d = DelaunayTriangulation<FailingQueryKernel, (), (), 3>;

fn fixture(count: usize) -> Triangulation3d {
    let mut vertices = vec![
        vertex![0.0, 0.0, 0.0].or_abort(),
        vertex![1.0, 0.0, 0.0].or_abort(),
        vertex![0.0, 1.0, 0.0].or_abort(),
        vertex![0.0, 0.0, 1.0].or_abort(),
    ];
    let range = CoordinateRange::try_new(0.02, 0.25).or_abort();
    let points =
        generate_random_points_in_range_seeded::<3>(count - vertices.len(), range, 510).or_abort();
    vertices.extend(try_vertices_from_points(&points).or_abort());
    let triangulation = DelaunayTriangulation::builder(&vertices)
        .build_with_kernel(&FailingQueryKernel)
        .or_abort();
    assert_eq!(triangulation.number_of_vertices(), count);
    triangulation.validate().or_abort();
    triangulation
}

fn reject_insertion(triangulation: &mut Triangulation3d, vertex: Vertex<(), 3>) {
    match triangulation.insert_vertex(vertex) {
        Err(InsertionError::ConflictRegion {
            source: ConflictError::PredicateError { source },
        }) if source == INJECTED_ERROR => {}
        result => abort_benchmark(format!(
            "expected the injected predicate error, got {result:?}"
        )),
    }
}

fn benchmark_insertion_rollback(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("insertion_rollback/3d_conflict_predicate");
    for count in [16, 128] {
        let mut triangulation = fixture(count);
        let vertex = vertex![REJECTED_COORDS].or_abort();
        let generation = triangulation.topology_generation();
        let vertices: Vec<_> = triangulation.vertices().map(|(key, _)| key).collect();
        let simplices: Vec<_> = triangulation.simplices().map(|(key, _)| key).collect();

        reject_insertion(&mut triangulation, vertex);
        group.bench_function(BenchmarkId::from_parameter(count), |bencher| {
            bencher.iter_batched(
                || vertex,
                |candidate| reject_insertion(&mut triangulation, candidate),
                BatchSize::SmallInput,
            );
        });

        assert_eq!(triangulation.topology_generation(), generation);
        assert!(triangulation.vertices().map(|(key, _)| key).eq(vertices));
        assert!(triangulation.simplices().map(|(key, _)| key).eq(simplices));
        triangulation.validate().or_abort();
    }
    group.finish();
}

criterion_group!(benches, benchmark_insertion_rollback);
criterion_main!(benches);

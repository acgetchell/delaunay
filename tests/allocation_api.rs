//! Allocation-bounded tests for performance-sensitive public hot paths.
//!
//! These tests run only with `--features count-allocations` because the feature
//! installs the allocation-counting global allocator.

#![cfg(feature = "count-allocations")]

use allocation_counter::AllocationInfo;
use delaunay::prelude::algorithms::{LocateError, LocateResult, locate_with_stats};
use delaunay::prelude::geometry::{Coordinate, FastKernel, Point};
use delaunay::prelude::tds::{
    SimplexKey, SimplexValidationError, TdsError, VertexKey, facet_key_from_vertices,
    measure_with_result,
};
use delaunay::prelude::triangulation::construction::{
    DelaunayTriangulation, DelaunayTriangulationConstructionError, vertex,
};
use std::hint::black_box;
use thiserror::Error;

const SIMPLEX_5D_DIMENSION: usize = 5;
const SIMPLEX_VERTEX_COUNT: usize = SIMPLEX_5D_DIMENSION + 1;
const SIMPLEX_SIMPLEX_COUNT: usize = 1;
const LOCATE_FAST_PATH_ALLOCATION_BUDGET: u64 = 1;

type TestTriangulation2D = DelaunayTriangulation<FastKernel<f64>, (), (), 2>;
type TestTriangulation5D = DelaunayTriangulation<FastKernel<f64>, (), (), SIMPLEX_5D_DIMENSION>;

/// Typed failure modes for allocation-bounded public API checks.
#[derive(Debug, Error)]
enum AllocationTestError {
    #[error("triangulation construction failed: {source}")]
    Construction {
        #[from]
        source: DelaunayTriangulationConstructionError,
    },

    #[error("TDS lookup failed: {source}")]
    Tds {
        #[from]
        source: TdsError,
    },

    #[error("simplex validation failed: {source}")]
    Simplex {
        #[from]
        source: SimplexValidationError,
    },

    #[error("point location failed: {source}")]
    Locate {
        #[from]
        source: LocateError,
    },

    #[error("deterministic simplex fixture did not contain a simplex")]
    MissingSimplex,

    #[error("fixture simplex has {actual} vertices, expected at least {required}")]
    SimplexTooSmall { required: usize, actual: usize },
}

/// Build a minimal 2D simplex for the hinted locate fast-path allocation check.
fn deterministic_2d_simplex() -> Result<TestTriangulation2D, AllocationTestError> {
    let vertices = [
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];

    Ok(DelaunayTriangulation::with_kernel(
        &FastKernel::new(),
        &vertices,
    )?)
}

/// Build a 5D unit simplex fixture for stack-sized topology hot-path checks.
fn deterministic_5d_simplex() -> Result<TestTriangulation5D, AllocationTestError> {
    let vertices = [
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
    ];

    Ok(DelaunayTriangulation::with_kernel(
        &FastKernel::new(),
        &vertices,
    )?)
}

/// Return the only simplex in the deterministic 5D simplex fixture.
fn only_5d_simplex_key(dt: &TestTriangulation5D) -> Result<SimplexKey, AllocationTestError> {
    dt.tds()
        .simplex_keys()
        .next()
        .ok_or(AllocationTestError::MissingSimplex)
}

/// Extract one facet from the 5D simplex without allocating a temporary buffer.
fn first_facet_vertices(
    dt: &TestTriangulation5D,
) -> Result<[VertexKey; SIMPLEX_5D_DIMENSION], AllocationTestError> {
    let simplex_key = only_5d_simplex_key(dt)?;
    let simplex = dt
        .tds()
        .simplex(simplex_key)
        .ok_or(AllocationTestError::MissingSimplex)?;
    let vertices = simplex.vertices();
    if vertices.len() < SIMPLEX_5D_DIMENSION {
        return Err(AllocationTestError::SimplexTooSmall {
            required: SIMPLEX_5D_DIMENSION,
            actual: vertices.len(),
        });
    }

    let mut facet_vertices = [vertices[0]; SIMPLEX_5D_DIMENSION];
    facet_vertices.copy_from_slice(&vertices[..SIMPLEX_5D_DIMENSION]);
    Ok(facet_vertices)
}

/// Assert an operation performed no allocations and retained no allocator state.
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

/// Assert an operation stayed within a known allocation-count budget.
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

#[test]
fn tds_and_public_iterators_are_zero_allocation() -> Result<(), AllocationTestError> {
    let dt = deterministic_5d_simplex()?;
    let tds = dt.tds();
    let tri = dt.as_triangulation();

    let (counts, info) = measure_with_result(|| {
        black_box((
            tds.simplices().count(),
            tds.vertices().count(),
            tds.simplex_keys().count(),
            tds.vertex_keys().count(),
            tri.simplices().count(),
            tri.vertices().count(),
            dt.simplices().count(),
            dt.vertices().count(),
        ))
    });

    assert_eq!(
        counts,
        (
            SIMPLEX_SIMPLEX_COUNT,
            SIMPLEX_VERTEX_COUNT,
            SIMPLEX_SIMPLEX_COUNT,
            SIMPLEX_VERTEX_COUNT,
            SIMPLEX_SIMPLEX_COUNT,
            SIMPLEX_VERTEX_COUNT,
            SIMPLEX_SIMPLEX_COUNT,
            SIMPLEX_VERTEX_COUNT,
        )
    );
    assert_zero_allocations(&info, "TDS and public simplices()/vertices() iterators");
    Ok(())
}

#[test]
fn tds_simplex_vertices_is_zero_allocation() -> Result<(), AllocationTestError> {
    let dt = deterministic_5d_simplex()?;
    let simplex_key = only_5d_simplex_key(&dt)?;

    let (vertex_count, info) = measure_with_result(|| {
        dt.tds()
            .simplex_vertices(simplex_key)
            .map(|keys| keys.len())
    });
    assert_eq!(vertex_count?, SIMPLEX_VERTEX_COUNT);
    assert_zero_allocations(&info, "Tds::simplex_vertices");
    Ok(())
}

#[test]
fn simplex_vertex_uuid_iter_is_zero_allocation() -> Result<(), AllocationTestError> {
    let dt = deterministic_5d_simplex()?;
    let tds = dt.tds();
    let simplex_key = only_5d_simplex_key(&dt)?;
    let simplex = tds
        .simplex(simplex_key)
        .ok_or(AllocationTestError::MissingSimplex)?;
    let expected_uuids = simplex.vertex_uuids(tds)?;

    let (result, info) = measure_with_result(|| {
        let iter = simplex.vertex_uuid_iter(tds);
        let exact_size_len = iter.len();
        let mut matched = 0usize;

        for (index, uuid) in iter.enumerate() {
            let uuid = uuid?;
            if expected_uuids
                .get(index)
                .is_some_and(|expected| *expected == uuid)
            {
                matched += 1;
            }
        }

        Ok::<_, SimplexValidationError>((exact_size_len, matched))
    });
    let (exact_size_len, matched) = result?;

    assert_eq!(exact_size_len, SIMPLEX_VERTEX_COUNT);
    assert_eq!(matched, expected_uuids.len());
    assert_zero_allocations(&info, "Simplex::vertex_uuid_iter");
    Ok(())
}

#[test]
fn facet_key_from_vertices_is_zero_allocation() -> Result<(), AllocationTestError> {
    let facet_vertices = first_facet_vertices(&deterministic_5d_simplex()?)?;

    let (facet_key, info) =
        measure_with_result(|| black_box(facet_key_from_vertices(&facet_vertices)));
    assert_ne!(facet_key, 0);
    assert_zero_allocations(&info, "facet_key_from_vertices");
    Ok(())
}

#[test]
fn locate_with_hint_fast_path_stays_allocation_bounded() -> Result<(), AllocationTestError> {
    let dt = deterministic_2d_simplex()?;
    let simplex_key = dt
        .tds()
        .simplex_keys()
        .next()
        .ok_or(AllocationTestError::MissingSimplex)?;
    let kernel = FastKernel::<f64>::new();
    let query = Point::new([0.25, 0.25]);

    let (locate_result, info) =
        measure_with_result(|| locate_with_stats(dt.tds(), &kernel, &query, Some(simplex_key)));
    let (location, stats) = locate_result?;

    assert!(matches!(location, LocateResult::InsideSimplex(found) if found == simplex_key));
    assert!(stats.used_hint);
    assert!(!stats.fell_back_to_scan());
    assert_allocation_budget(
        &info,
        "hinted locate_with_stats fast path",
        LOCATE_FAST_PATH_ALLOCATION_BUDGET,
    );
    Ok(())
}

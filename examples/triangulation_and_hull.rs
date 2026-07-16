#![forbid(unsafe_code)]

//! # 3D/4D/5D Triangulation and Convex Hull Workflow
//!
//! This example builds seeded 3D through 5D Delaunay triangulations, traverses
//! basic topology, locates points, evaluates simplex quality, extracts convex
//! hulls, and runs hull queries.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --release --example triangulation_and_hull
//! ```

use std::num::NonZeroUsize;
use std::time::Instant;

use delaunay::prelude::algorithms::LocateError;
use delaunay::prelude::construction::{
    ConstructionOptions, DedupPolicy, DelaunayTriangulation, DelaunayTriangulationBuilder,
    DelaunayTriangulationConstructionError, RetryPolicy, try_vertices_from_points,
};
use delaunay::prelude::generators::{
    RandomPointGenerationError, generate_random_points_in_range_seeded,
};
use delaunay::prelude::geometry::{
    AdaptiveKernel, CoordinateConversionError, CoordinateRange, CoordinateRangeError,
    CoordinateValidationError, QualityError, normalized_volume, radius_ratio,
};
use delaunay::prelude::query::{
    ConvexHull, ConvexHullConstructionError, Point, QueryError, TopologyIndexBuildError,
};

type WorkflowTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

#[derive(Debug, thiserror::Error)]
enum WorkflowExampleError {
    #[error(transparent)]
    CoordinateRange(#[from] CoordinateRangeError),
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    TopologyIndex(#[from] TopologyIndexBuildError),
    #[error(transparent)]
    Query(#[from] QueryError),
    #[error(transparent)]
    Locate(#[from] LocateError),
    #[error(transparent)]
    Quality(#[from] QualityError),
    #[error("convex hull operation failed: {source}")]
    ConvexHull {
        #[source]
        source: Box<ConvexHullConstructionError>,
    },
    #[error("retry attempt count must be non-zero")]
    ZeroRetryAttempts,
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
    #[error(transparent)]
    CoordinateValidation(#[from] CoordinateValidationError),
    #[error(transparent)]
    PointGeneration(#[from] RandomPointGenerationError),
    #[error("point count {point_count} is too large for centroid normalization")]
    PointCountTooLarge { point_count: usize },
    #[error("{dimension}D triangulation unexpectedly contains no simplices")]
    EmptyTriangulation { dimension: usize },
}

impl From<ConvexHullConstructionError> for WorkflowExampleError {
    fn from(source: ConvexHullConstructionError) -> Self {
        Self::ConvexHull {
            source: Box::new(source),
        }
    }
}

fn main() -> Result<(), WorkflowExampleError> {
    let bounds = CoordinateRange::try_new(-100.0_f64, 100.0)?;
    run_case::<3>("3D", 750, 873, bounds)?;
    println!();
    run_case::<4>("4D", 75, 531, bounds)?;
    println!();
    run_case::<5>("5D", 18, 421, bounds)?;
    Ok(())
}

/// Runs the same construction, query, quality, and hull workflow in one dimension.
fn run_case<const D: usize>(
    label: &str,
    point_count: usize,
    seed: u64,
    bounds: CoordinateRange<f64>,
) -> Result<(), WorkflowExampleError> {
    let points = generate_random_points_in_range_seeded::<D>(point_count, bounds, seed)?;
    let vertices = try_vertices_from_points(&points)?;
    let options = ConstructionOptions::default()
        .with_dedup_policy(DedupPolicy::Exact)
        .with_retry_policy(RetryPolicy::Shuffled {
            attempts: retry_attempts()?,
            base_seed: Some(seed),
        });

    println!("{label} Delaunay triangulation ({point_count} seeded points)");
    let start = Instant::now();
    let dt: WorkflowTriangulation<D> = DelaunayTriangulationBuilder::new(&vertices)
        .construction_options(options)
        .build()?;
    println!("  construction: {:?}", start.elapsed());
    println!("  vertices:  {}", dt.number_of_vertices());
    println!("  simplices: {}", dt.number_of_simplices());

    let edge_index = dt.build_edge_index()?;
    println!("  edges:     {}", edge_index.number_of_edges());

    let boundary_facet_count = dt.boundary_facets()?.try_fold(0_usize, |count, facet| {
        facet
            .map(|_| count + 1)
            .map_err(|source| QueryError::TriangulationCorrupted {
                source: Box::new(source.into()),
            })
    })?;
    println!("  boundary facets: {boundary_facet_count}");

    let hull = ConvexHull::try_from_triangulation(dt.as_triangulation())?;
    println!("  hull facets: {}", hull.number_of_facets());

    let inside = centroid_point(&points)?;
    let outside = Point::try_new([bounds.max() * 2.5; D])?;

    let location = dt.locate(&inside, None)?;
    println!("  centroid location: {location:?}");

    let Some((simplex_key, _)) = dt.simplices().next() else {
        return Err(WorkflowExampleError::EmptyTriangulation { dimension: D });
    };
    println!(
        "  first-simplex quality: radius ratio {:.6}, normalized volume {:.6}",
        radius_ratio(dt.as_triangulation(), simplex_key)?,
        normalized_volume(dt.as_triangulation(), simplex_key)?
    );

    println!(
        "  hull query: centroid outside? {}",
        hull.is_point_outside(&inside, dt.as_triangulation())?
    );
    println!(
        "  hull query: exterior outside? {}",
        hull.is_point_outside(&outside, dt.as_triangulation())?
    );

    let visible_facets = hull.find_visible_facets(&outside, dt.as_triangulation())?;
    println!(
        "  facets visible from exterior point: {}",
        visible_facets.len()
    );

    if let Some(index) = hull.find_nearest_visible_facet(&outside, dt.as_triangulation())? {
        println!("  nearest visible facet index: {index}");
    }

    Ok(())
}

/// Returns the non-zero retry count used by every dimension in this example.
fn retry_attempts() -> Result<NonZeroUsize, WorkflowExampleError> {
    NonZeroUsize::new(6).ok_or(WorkflowExampleError::ZeroRetryAttempts)
}

/// Computes a finite centroid used for point-location and hull-containment queries.
fn centroid_point<const D: usize>(points: &[Point<D>]) -> Result<Point<D>, WorkflowExampleError> {
    let mut coords = [0.0; D];
    for point in points {
        for (coord, value) in coords.iter_mut().zip(point.coords()) {
            *coord += *value;
        }
    }

    let point_count =
        u32::try_from(points.len()).map_err(|_| WorkflowExampleError::PointCountTooLarge {
            point_count: points.len(),
        })?;
    let point_count = f64::from(point_count);
    for coord in &mut coords {
        *coord /= point_count;
    }

    Ok(Point::try_new(coords)?)
}

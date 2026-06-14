#![forbid(unsafe_code)]

//! # 3D/4D Triangulation and Convex Hull Workflow
//!
//! This example builds seeded 3D and 4D Delaunay triangulations, traverses
//! basic topology, extracts convex hulls, and runs hull queries.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --release --example triangulation_and_hull
//! ```

use std::num::NonZeroUsize;
use std::time::Instant;

use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationConstructionError, RetryPolicy,
};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{
    AdaptiveKernel, CoordinateConversionError, CoordinateRange, CoordinateRangeError,
    CoordinateValidationError,
};
use delaunay::prelude::query::{
    AdjacencyIndexBuildError, ConvexHull, ConvexHullConstructionError, Point, QueryError,
};

type WorkflowTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

#[derive(Debug, thiserror::Error)]
enum WorkflowExampleError {
    #[error(transparent)]
    CoordinateRange(#[from] CoordinateRangeError),
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    AdjacencyIndex(#[from] AdjacencyIndexBuildError),
    #[error(transparent)]
    Query(#[from] QueryError),
    #[error("convex hull operation failed: {source}")]
    ConvexHull {
        #[source]
        source: Box<ConvexHullConstructionError>,
    },
    #[error("retry attempt count must be non-zero")]
    ZeroRetryAttempts,
    #[error(transparent)]
    CoordinateValidation(#[from] CoordinateValidationError),
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
    #[error("point count {point_count} is too large for centroid normalization")]
    PointCountTooLarge { point_count: usize },
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
    Ok(())
}

fn run_case<const D: usize>(
    label: &str,
    point_count: usize,
    seed: u64,
    bounds: CoordinateRange<f64>,
) -> Result<(), WorkflowExampleError> {
    let points = generate_random_points_in_range_seeded::<D>(point_count, bounds, seed);
    let vertices = points
        .iter()
        .map(|point| delaunay::prelude::Vertex::<(), _>::try_new((*point).into()))
        .collect::<Result<Vec<_>, _>>()?;
    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts: retry_attempts()?,
        base_seed: Some(seed),
    });

    println!("{label} Delaunay triangulation ({point_count} seeded points)");
    let start = Instant::now();
    let dt: WorkflowTriangulation<D> = DelaunayTriangulation::new_with_options(&vertices, options)?;
    println!("  construction: {:?}", start.elapsed());
    println!("  vertices:  {}", dt.number_of_vertices());
    println!("  simplices: {}", dt.number_of_simplices());

    let index = dt.build_adjacency_index()?;
    println!("  edges:     {}", index.number_of_edges());

    println!("  boundary facets: {}", dt.boundary_facets()?.count());

    let hull = ConvexHull::from_triangulation(dt.as_triangulation())?;
    println!("  hull facets: {}", hull.number_of_facets());

    let inside = centroid_point(&points)?;
    let outside = Point::try_new([bounds.max() * 2.5; D])?;

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

fn retry_attempts() -> Result<NonZeroUsize, WorkflowExampleError> {
    NonZeroUsize::new(6).ok_or(WorkflowExampleError::ZeroRetryAttempts)
}

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

#![forbid(unsafe_code)]

//! Integration tests for workflows demonstrated by runnable examples.

use delaunay::prelude::construction::{
    DelaunayTriangulation, DelaunayTriangulationConstructionError,
};
use delaunay::prelude::geometry::{CoordinateConversionError, CoordinateValidationError};
use delaunay::prelude::query::{ConvexHull, Point, QueryError};

#[test]
fn triangulation_and_hull_workflow_remains_valid() -> Result<(), WorkflowTestError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.35, 0.25, 0.20])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.20, 0.60, 0.25])?,
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::try_new(&vertices)?;
    dt.validate()?;

    let edge_index = dt.build_edge_index()?;
    assert!(edge_index.number_of_edges() > 0);

    let boundary_facets: Vec<_> = dt
        .boundary_facets()?
        .map(|facet| {
            facet.map_err(|source| QueryError::TriangulationCorrupted {
                source: source.into(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert!(!boundary_facets.is_empty());

    let hull = ConvexHull::try_from_triangulation(dt.as_triangulation())?;
    hull.validate(dt.as_triangulation())?;
    assert_eq!(hull.number_of_facets(), boundary_facets.len());

    let inside = Point::try_new([0.25, 0.25, 0.25])?;
    let outside = Point::try_new([2.0, 2.0, 2.0])?;

    assert!(!hull.is_point_outside(&inside, dt.as_triangulation())?);
    assert!(hull.is_point_outside(&outside, dt.as_triangulation())?);
    assert!(
        !hull
            .find_visible_facets(&outside, dt.as_triangulation())?
            .is_empty()
    );
    assert!(
        hull.find_nearest_visible_facet(&outside, dt.as_triangulation())?
            .is_some()
    );

    Ok(())
}

#[derive(Debug, thiserror::Error)]
enum WorkflowTestError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
    #[error(transparent)]
    CoordinateValidation(#[from] CoordinateValidationError),
    #[error(transparent)]
    Validation(#[from] delaunay::prelude::validation::DelaunayTriangulationValidationError),
    #[error(transparent)]
    TopologyIndex(#[from] delaunay::prelude::query::TopologyIndexBuildError),
    #[error(transparent)]
    Query(#[from] QueryError),
    #[error("convex hull construction failed: {source}")]
    ConvexHullConstruction {
        #[source]
        source: Box<delaunay::prelude::query::ConvexHullConstructionError>,
    },
    #[error("convex hull validation failed: {source}")]
    ConvexHullValidation {
        #[source]
        source: Box<delaunay::prelude::query::ConvexHullValidationError>,
    },
}

impl From<delaunay::prelude::query::ConvexHullConstructionError> for WorkflowTestError {
    fn from(source: delaunay::prelude::query::ConvexHullConstructionError) -> Self {
        Self::ConvexHullConstruction {
            source: Box::new(source),
        }
    }
}

impl From<delaunay::prelude::query::ConvexHullValidationError> for WorkflowTestError {
    fn from(source: delaunay::prelude::query::ConvexHullValidationError) -> Self {
        Self::ConvexHullValidation {
            source: Box::new(source),
        }
    }
}

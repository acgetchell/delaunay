//! Prototype spherical Delaunay construction fixtures.

use std::{
    assert_matches,
    f64::consts::{FRAC_PI_2, PI},
};

use delaunay::prelude::construction::{
    ConstructionOptions, SphericalDelaunayBuilder, SphericalDelaunayConstructionError,
    SphericalDelaunayValidationError, SphericalSimplex, SphericalSimplexError,
};
use delaunay::prelude::topology::spaces::{SphericalMetric, SphericalPoint, SphericalPointError};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {actual:?} to be within {tolerance:?} of {expected:?}",
    );
}

#[test]
fn spherical_metric_normalizes_and_uses_geodesic_distance() {
    let metric = SphericalMetric::<2>::unit();
    let point = metric
        .canonicalize([3.0, 4.0, 0.0])
        .expect("finite nonzero coordinates should canonicalize");
    assert_eq!(point.intrinsic_dimension(), 2);
    assert_eq!(point.ambient_dimension(), 3);
    assert_close(point.squared_norm(), 1.0, 1.0e-12);
    assert_close(point.coords()[0], 0.6, 1.0e-12);
    assert_close(point.coords()[1], 0.8, 1.0e-12);

    let x = SphericalPoint::<2>::try_new([1.0, 0.0, 0.0])
        .expect("unit x-axis point should canonicalize");
    let y = SphericalPoint::<2>::try_new([0.0, 1.0, 0.0])
        .expect("unit y-axis point should canonicalize");
    assert_close(
        metric
            .try_distance(&x, &y)
            .expect("matching metric and point radii should produce a distance"),
        FRAC_PI_2,
        1.0e-12,
    );
}

#[test]
fn spherical_metric_slice_constructors_preserve_radius_metadata() {
    let metric =
        SphericalMetric::<2>::try_new(2.0).expect("positive finite radius should define a metric");
    assert_close(metric.radius(), 2.0, 0.0);
    assert_eq!(metric.ambient_dimension(), 3);

    let point = SphericalPoint::<2>::try_from_slice(&[3.0, 0.0, 4.0])
        .expect("unit point should canonicalize from a slice");
    assert_eq!(point.intrinsic_dimension(), 2);
    assert_close(point.radius(), 1.0, 0.0);
    assert_close(point.squared_norm(), 1.0, 1.0e-12);

    let radius_two = metric
        .canonicalize_slice(&[0.0, 6.0, 8.0])
        .expect("metric should canonicalize a raw slice onto its radius");
    assert_close(radius_two.radius(), 2.0, 0.0);
    assert_close(radius_two.squared_norm(), 4.0, 1.0e-12);
}

#[test]
fn spherical_metric_rejects_distance_between_mismatched_radii() {
    let metric = SphericalMetric::<2>::unit();
    let unit =
        SphericalPoint::<2>::try_new([1.0, 0.0, 0.0]).expect("unit point should canonicalize");
    let radius_two = SphericalPoint::<2>::try_new_with_radius([0.0, 1.0, 0.0], 2.0)
        .expect("radius-two point should canonicalize");

    assert_matches!(
        metric.try_distance(&unit, &radius_two),
        Err(SphericalPointError::MismatchedRadius { expected, actual })
            if expected.to_bits() == 1.0_f64.to_bits()
                && actual.to_bits() == 2.0_f64.to_bits()
    );
}

#[test]
fn spherical_s2_tetrahedron_hull_facets_are_triangles() {
    let points = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];

    let triangulation = SphericalDelaunayBuilder::<2>::try_new(points)
        .expect("tetrahedron points should canonicalize")
        .build()
        .expect("tetrahedron hull should construct spherical S^2 Delaunay simplices");

    assert_eq!(triangulation.dimension(), 2);
    assert_eq!(triangulation.ambient_dimension(), 3);
    assert_eq!(triangulation.number_of_vertices(), 4);
    assert_eq!(triangulation.number_of_simplices(), 4);
    assert_close(triangulation.radius(), 1.0, 0.0);
    assert_eq!(triangulation.points().len(), 4);
    for simplex in triangulation.simplices() {
        assert_eq!(simplex.dimension(), 2);
        assert_eq!(simplex.vertex_indices().len(), 3);
    }
    triangulation
        .validate_topology()
        .expect("tetrahedron boundary should satisfy Level 3 topology");
    triangulation
        .is_valid_topology()
        .expect("Level 3 wrapper should validate spherical topology");
    triangulation
        .validate_realization()
        .expect("tetrahedron facets should satisfy spherical Level 4 realization");
    triangulation
        .is_valid_realization()
        .expect("Level 4 wrapper should validate spherical realization");
    triangulation
        .validate_delaunay()
        .expect("tetrahedron facets should satisfy spherical Level 5 Delaunay");
    triangulation
        .is_valid_delaunay()
        .expect("Level 5 wrapper should validate spherical Delaunay");
}

#[test]
fn spherical_builder_from_points_accepts_options() {
    let points = vec![
        SphericalPoint::<2>::try_new([1.0, 1.0, 1.0]).expect("finite point"),
        SphericalPoint::<2>::try_new([1.0, -1.0, -1.0]).expect("finite point"),
        SphericalPoint::<2>::try_new([-1.0, 1.0, -1.0]).expect("finite point"),
        SphericalPoint::<2>::try_new([-1.0, -1.0, 1.0]).expect("finite point"),
    ];

    let triangulation = SphericalDelaunayBuilder::<2>::try_from_points(points)
        .expect("matching radii should create a spherical builder")
        .construction_options(ConstructionOptions::default())
        .build()
        .expect("tetrahedron boundary should build from normalized points");

    assert_eq!(triangulation.dimension(), 2);
    assert_eq!(triangulation.number_of_vertices(), 4);
    assert_eq!(triangulation.number_of_simplices(), 4);
    triangulation
        .validate()
        .expect("builder-from-points path should preserve spherical validation");
}

#[test]
fn spherical_s3_simplex_boundary_facets_are_tetrahedra() {
    let points = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
    ];

    let triangulation = SphericalDelaunayBuilder::<3>::try_new(points)
        .expect("4-simplex points should canonicalize")
        .build()
        .expect("4-simplex hull should construct spherical S^3 Delaunay simplices");

    assert_eq!(triangulation.dimension(), 3);
    assert_eq!(triangulation.ambient_dimension(), 4);
    assert_eq!(triangulation.number_of_vertices(), 5);
    assert_eq!(triangulation.number_of_simplices(), 5);
    for simplex in triangulation.simplices() {
        assert_eq!(simplex.vertex_indices().len(), 4);
    }
    triangulation
        .validate()
        .expect("validate should run through spherical Level 5");
}

#[test]
fn spherical_s3_near_antipodal_fixture_constructs() {
    let points = [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0e-9, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 1.0e-9, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, 1.0e-9],
        [0.0, 0.0, 0.0, 1.0],
        [1.0e-9, 0.0, 0.0, -1.0],
    ];

    let triangulation = SphericalDelaunayBuilder::<3>::try_new(points)
        .expect("near-antipodal S^3 points should canonicalize")
        .build()
        .expect("near-antipodal S^3 fixture should construct");

    assert_eq!(triangulation.dimension(), 3);
    assert_eq!(triangulation.ambient_dimension(), 4);
    assert!(triangulation.number_of_simplices() >= 5);
    assert!(
        triangulation
            .distance_between_vertices(0, 1)
            .expect("fixture vertices should exist")
            > PI - 1.0e-6
    );
    triangulation
        .validate_delaunay()
        .expect("near-antipodal S^3 fixture should satisfy spherical Level 5");
}

#[test]
fn spherical_s2_near_antipodal_fixture_constructs() {
    let points = [
        [1.0, 0.0, 0.0],
        [-1.0, 1.0e-9, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];

    let triangulation = SphericalDelaunayBuilder::<2>::try_new(points)
        .expect("near-antipodal points should canonicalize")
        .build()
        .expect("near-antipodal fixture should construct");

    assert_eq!(triangulation.dimension(), 2);
    assert_eq!(triangulation.ambient_dimension(), 3);
    assert!(triangulation.number_of_simplices() >= 4);
    triangulation
        .validate_delaunay()
        .expect("near-antipodal fixture should satisfy spherical Level 5");
}

#[test]
fn spherical_zero_vector_is_rejected() {
    let err = SphericalPoint::<2>::try_new([0.0, 0.0, 0.0])
        .expect_err("zero vectors cannot be projected onto a sphere");
    assert_matches!(err, SphericalPointError::ZeroNorm);
}

#[test]
fn spherical_point_rejects_invalid_coordinates_and_radius() {
    let wrong_arity = SphericalPoint::<2>::try_new([1.0, 0.0])
        .expect_err("S^2 points require three ambient coordinates");
    assert_matches!(
        wrong_arity,
        SphericalPointError::InvalidAmbientCoordinateCount {
            dimension: 2,
            expected: 3,
            actual: 2,
        }
    );

    let non_finite = SphericalPoint::<2>::try_new([1.0, f64::INFINITY, 0.0])
        .expect_err("non-finite coordinates must be rejected");
    assert_matches!(
        non_finite,
        SphericalPointError::NonFiniteCoordinate { axis: 1, value } if value.is_infinite()
    );

    let invalid_radius = SphericalMetric::<2>::try_new(0.0)
        .expect_err("zero radius cannot define a spherical metric");
    assert_matches!(
        invalid_radius,
        SphericalPointError::InvalidRadius { radius } if radius == 0.0
    );
}

#[test]
fn spherical_simplex_rejects_invalid_public_inputs() {
    let wrong_arity = SphericalSimplex::<2>::try_new(vec![0, 1], 3)
        .expect_err("S^2 simplices require three vertices");
    assert_matches!(
        wrong_arity,
        SphericalSimplexError::InvalidArity {
            dimension: 2,
            expected: 3,
            actual: 2,
        }
    );

    let out_of_bounds = SphericalSimplex::<2>::try_new(vec![0, 1, 3], 3)
        .expect_err("simplex vertices must reference existing points");
    assert_matches!(
        out_of_bounds,
        SphericalSimplexError::VertexIndexOutOfBounds {
            vertex_index: 3,
            vertex_count: 3,
        }
    );

    let duplicate = SphericalSimplex::<2>::try_new(vec![0, 1, 1], 3)
        .expect_err("simplex vertices must be unique");
    assert_matches!(
        duplicate,
        SphericalSimplexError::DuplicateVertex { vertex_index: 1 }
    );
}

#[test]
fn spherical_builder_reports_typed_boundary_errors() {
    let invalid_radius = SphericalDelaunayBuilder::<2>::try_new_with_radius([[1.0, 0.0, 0.0]], 0.0)
        .expect_err("builder radius must be finite and positive");
    assert_matches!(
        invalid_radius,
        SphericalDelaunayConstructionError::Metric {
            source: SphericalPointError::InvalidRadius { radius },
        } if radius == 0.0
    );

    let malformed = [[1.0, 0.0]];
    let err = SphericalDelaunayBuilder::<2>::try_new(malformed)
        .expect_err("malformed ambient coordinate arity should fail before build");
    assert_matches!(
        err,
        SphericalDelaunayConstructionError::Point {
            point_index: 0,
            source: SphericalPointError::InvalidAmbientCoordinateCount {
                dimension: 2,
                expected: 3,
                actual: 2,
            },
        }
    );

    let too_few_points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let err = SphericalDelaunayBuilder::<2>::try_new(too_few_points)
        .expect("valid points should canonicalize before hull cardinality checks")
        .build()
        .expect_err("S^2 hull construction requires at least four points");
    assert_matches!(
        err,
        SphericalDelaunayConstructionError::InsufficientVertices {
            dimension: 2,
            minimum: 4,
            actual: 3,
        }
    );

    let one_hemisphere = [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, -1.0, 2.0],
    ];
    let err = SphericalDelaunayBuilder::<2>::try_new(one_hemisphere)
        .expect("finite nonzero points should canonicalize before hull coverage checks")
        .build()
        .expect_err("points confined to one open hemisphere cannot enclose the origin");
    assert_matches!(
        err,
        SphericalDelaunayConstructionError::OriginOutsideConvexHull
    );
}

#[test]
fn spherical_builder_rejects_mismatched_point_radii() {
    let unit =
        SphericalPoint::<2>::try_new([1.0, 0.0, 0.0]).expect("unit point should canonicalize");
    let radius_two = SphericalPoint::<2>::try_new_with_radius([0.0, 1.0, 0.0], 2.0)
        .expect("radius-two point should canonicalize");

    let err = SphericalDelaunayBuilder::<2>::try_from_points(vec![unit, radius_two])
        .expect_err("builder points must share one spherical radius");
    assert_matches!(
        err,
        SphericalDelaunayConstructionError::Point {
            point_index: 1,
            source: SphericalPointError::MismatchedRadius { expected, actual },
        } if expected.to_bits() == 1.0_f64.to_bits()
            && actual.to_bits() == 2.0_f64.to_bits()
    );
}

#[test]
fn spherical_distance_between_vertices_reports_index_errors() {
    let points = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    let triangulation = SphericalDelaunayBuilder::<2>::try_new(points)
        .expect("tetrahedron points should canonicalize")
        .build()
        .expect("tetrahedron hull should construct");

    let err = triangulation
        .distance_between_vertices(0, 4)
        .expect_err("missing vertex indices should be reported as typed errors");
    assert_matches!(
        err,
        SphericalDelaunayValidationError::VertexIndexOutOfBounds {
            vertex_index: 4,
            vertex_count: 4,
        }
    );

    let err = triangulation
        .distance_between_vertices(4, 0)
        .expect_err("missing left vertex index should be reported before distance computation");
    assert_matches!(
        err,
        SphericalDelaunayValidationError::VertexIndexOutOfBounds {
            vertex_index: 4,
            vertex_count: 4,
        }
    );
}

#[test]
fn spherical_dimension_outside_prototype_is_typed() {
    let points = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0],
    ];

    let err = SphericalDelaunayBuilder::<4>::try_new(points)
        .expect("S^4 inputs should canonicalize before prototype dispatch")
        .build()
        .expect_err("S^4 is outside the bounded prototype");
    assert_matches!(
        err,
        SphericalDelaunayConstructionError::UnsupportedDimension {
            dimension: 4,
            max_validated_dimension: 3,
            ..
        }
    );
}

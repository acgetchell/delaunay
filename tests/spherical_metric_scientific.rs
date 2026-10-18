//! Independent analytical and metamorphic evidence for spherical arc lengths.

use std::f64::consts::PI;

use approx::assert_relative_eq;
use delaunay::prelude::topology::spaces::{SphericalMetric, SphericalPoint, SphericalPointError};

/// Embeds a known planar direction using the first and last ambient coordinates.
fn planar_point<const D: usize>(direction: [f64; 2], radius: f64) -> SphericalPoint<D> {
    let mut coords = vec![0.0; D + 1];
    coords[0] = direction[0];
    coords[D] = direction[1];
    SphericalPoint::try_from_slice_with_radius(&coords, radius)
        .expect("the finite nonzero fixture direction must remain representable")
}

#[test]
fn spherical_distance_resolves_nearly_coincident_points() {
    fn check<const D: usize>() {
        let metric = SphericalMetric::<D>::unit();
        let axis = planar_point::<D>([1.0, 0.0], 1.0);
        for slope in [
            f64::from_bits(0x3e10_0000_0000_0000), // 2^-30
            1.0e-100,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::from_bits(3),
        ] {
            let nearby = planar_point::<D>([1.0, slope], 1.0);
            let distance = metric.try_distance(&axis, &nearby).unwrap();
            // The exact angle is atan(slope). For these slopes the omitted
            // cubic term is smaller than half an ulp, so it rounds to slope.
            // Zero absolute tolerance prevents accepting a collapsed distance.
            assert_relative_eq!(
                distance,
                slope,
                epsilon = 0.0,
                max_relative = 8.0 * f64::EPSILON
            );
            assert_eq!(
                metric.try_distance(&nearby, &axis).unwrap().to_bits(),
                distance.to_bits()
            );
        }
    }
    check::<2>();
    check::<3>();
    check::<4>();
    check::<5>();
}

#[test]
fn spherical_distance_is_zero_for_identical_points() {
    fn check<const D: usize>() {
        for radius in [f64::MIN_POSITIVE / 2.0, 1.0, f64::MAX / 4.0, f64::MAX] {
            let metric = SphericalMetric::<D>::try_new(radius).unwrap();
            let diagonal = planar_point::<D>([1.0, 1.0], radius);
            assert_eq!(
                metric.try_distance(&diagonal, &diagonal).unwrap().to_bits(),
                0,
                "self-distance must be positive zero in intrinsic dimension {D}"
            );
        }
    }
    check::<2>();
    check::<3>();
    check::<4>();
    check::<5>();
}

#[test]
fn spherical_distance_resolves_near_antipodes() {
    fn check<const D: usize>() {
        let slope = f64::from_bits(0x3e10_0000_0000_0000); // 2^-30
        let metric = SphericalMetric::<D>::unit();
        let axis = planar_point::<D>([1.0, 0.0], 1.0);
        let near_antipode = planar_point::<D>([-1.0, slope], 1.0);
        let distance = metric.try_distance(&axis, &near_antipode).unwrap();
        // The exact angle is pi - atan(slope), not pi.
        assert!(distance < PI);
        assert_relative_eq!(distance, PI - slope, epsilon = 4.0 * f64::EPSILON);
    }
    check::<2>();
    check::<3>();
    check::<4>();
    check::<5>();
}

#[test]
fn spherical_distance_matches_independent_planar_angles() {
    fn check<const D: usize>() {
        let directions: [([i32; 2], [i32; 2]); 5] = [
            ([1, 0], [3, 4]),
            ([3, 4], [-4, 3]),
            ([3, 4], [-3, -4]),
            ([5, 12], [7, -24]),
            ([1, 0], [1, 0]),
        ];
        for (left, right) in directions {
            // Bounded integer cross and dot products are exact and do not
            // reuse the upstream compensated exterior-product construction.
            let cross = left[0] * right[1] - left[1] * right[0];
            let dot = left[0] * right[0] + left[1] * right[1];
            let angle = f64::from(cross.abs()).atan2(f64::from(dot));
            for radius in [f64::MIN_POSITIVE / 2.0, 0.5, 1.0, 2.0, f64::MAX / 4.0] {
                let metric = SphericalMetric::<D>::try_new(radius).unwrap();
                let a = planar_point::<D>(left.map(f64::from), radius);
                let b = planar_point::<D>(right.map(f64::from), radius);
                let distance = metric.try_distance(&a, &b).unwrap();
                assert_relative_eq!(
                    distance / radius,
                    angle,
                    epsilon = 8.0 * f64::EPSILON,
                    max_relative = 32.0 * f64::EPSILON
                );
                assert_eq!(
                    metric.try_distance(&b, &a).unwrap().to_bits(),
                    distance.to_bits()
                );
            }
        }
    }
    check::<2>();
    check::<3>();
    check::<4>();
    check::<5>();
}

#[test]
fn spherical_distance_matches_independent_dense_angles() -> Result<(), SphericalPointError> {
    fn check<const D: usize>() -> Result<(), SphericalPointError> {
        let ambient_length =
            f64::from(u32::try_from(D + 1).expect("fixture dimensions fit in u32"));
        // Every coordinate participates. Negating one coordinate changes the
        // exact dot product from n to n - 2; both squared norms remain n.
        // acos is well-conditioned for these angles and independent of the
        // upstream exterior-product atan2 construction.
        let angle = ((ambient_length - 2.0) / ambient_length).acos();
        let left = vec![1.0; D + 1];
        for axis in 0..=D {
            let mut right = left.clone();
            right[axis] = -1.0;
            for radius in [0.5, 1.0, 2.0] {
                let metric = SphericalMetric::<D>::try_new(radius)?;
                let a = SphericalPoint::try_from_slice_with_radius(&left, radius)?;
                let b = SphericalPoint::try_from_slice_with_radius(&right, radius)?;
                assert_relative_eq!(
                    metric.try_distance(&a, &b)?,
                    radius * angle,
                    epsilon = 0.0,
                    max_relative = 16.0 * f64::EPSILON
                );
            }
        }
        Ok(())
    }
    check::<2>()?;
    check::<3>()?;
    check::<4>()?;
    check::<5>()
}

#[test]
fn spherical_distance_preserves_small_arcs_at_maximum_radius() {
    fn check<const D: usize>() {
        let radius = f64::MAX;
        let metric = SphericalMetric::<D>::try_new(radius).unwrap();
        let axis = planar_point::<D>([1.0, 0.0], radius);
        for slope in [f64::from_bits(1), f64::from_bits(0x3e10_0000_0000_0000)] {
            let nearby = planar_point::<D>([1.0, slope], radius);
            let distance = metric.try_distance(&axis, &nearby).unwrap();
            assert_relative_eq!(
                distance,
                radius * slope,
                epsilon = 0.0,
                max_relative = 8.0 * f64::EPSILON
            );
        }
    }
    check::<2>();
    check::<3>();
    check::<4>();
    check::<5>();
}

#[test]
fn spherical_distance_has_no_matrix_dispatch_dimension_cap() -> Result<(), SphericalPointError> {
    // This metric accepts runtime ambient coordinates independently of the
    // bounded spherical triangulation prototype and fixed-matrix dispatch.
    let metric = SphericalMetric::<64>::unit();
    let axis = planar_point::<64>([1.0, 0.0], 1.0);
    let perpendicular = planar_point::<64>([0.0, 1.0], 1.0);
    assert_relative_eq!(
        metric.try_distance(&axis, &perpendicular)?,
        PI / 2.0,
        epsilon = f64::EPSILON
    );
    Ok(())
}

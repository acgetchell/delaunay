//! Property-based tests for toroidal coordinate and periodic-simplex invariants.
//!
//! Verifies two fundamental properties of [`canonicalize_point`] and exercises
//! transformed `T^2` periodic quotient constructions through the public builder:
//! - **In-domain**: the result always lies in `[0, L_i)` for every axis `i`.
//! - **Idempotent**: applying canonicalization twice gives the same result as once.
//! - **Periodic construction invariance**: axis reflection and input permutation
//!   preserve Levels 1-5 validity for the validated unit-domain fixture.
//! - **Periodic quality**: published simplices agree with independent Euclidean
//!   triangle formulas after applying their lattice offsets.

#[macro_use]
#[path = "common/proptest_config.rs"]
mod proptest_config;

use delaunay::prelude::construction::{DelaunayTriangulationBuilder, Vertex};
use delaunay::prelude::geometry::{RobustKernel, normalized_volume, radius_ratio};
use delaunay::prelude::topology::spaces::{TopologicalSpace, ToroidalSpace};
use delaunay::vertex;
use proptest::prelude::*;

// =============================================================================
// Strategies
// =============================================================================

/// Mixes ordinary positive periods with finite representability boundaries.
fn positive_period() -> impl Strategy<Value = f64> {
    prop_oneof![
        4 => 0.0001_f64..=100.0_f64,
        1 => Just(f64::from_bits(1)),
        1 => Just(f64::MIN_POSITIVE),
        1 => Just(f64::MAX),
    ]
}

/// Generates a `T^2` domain `[L_x, L_y]`.
fn domain_t2() -> impl Strategy<Value = [f64; 2]> {
    [positive_period(), positive_period()]
}

/// Generates an arbitrary (possibly out-of-domain) 2D coordinate pair.
fn coords_2d() -> impl Strategy<Value = [f64; 2]> {
    let c = prop_oneof![
        4 => -1000.0_f64..=1000.0_f64,
        1 => Just(-f64::from_bits(1)),
        1 => Just(-f64::MIN_POSITIVE),
        1 => Just(-0.0),
        1 => Just(f64::MIN),
        1 => Just(f64::MAX),
    ];
    [c.clone(), c]
}

fn assert_toroidal_roundoff_contract<const D: usize>() {
    let space = ToroidalSpace::<D>::try_new([3.0; D]).unwrap();
    for axis in 0..D {
        assert_eq!(
            space.wrap_coord(axis, -f64::EPSILON).map(f64::to_bits),
            Some(0.0_f64.to_bits())
        );
    }
    let mut coords = [-f64::EPSILON; D];
    space.canonicalize_point(&mut coords);
    assert_eq!(coords.map(f64::to_bits), [0.0_f64.to_bits(); D]);
    space.canonicalize_point(&mut coords);
    assert_eq!(coords.map(f64::to_bits), [0.0_f64.to_bits(); D]);

    // A representable remainder below the endpoint must not be snapped to zero.
    let space = ToroidalSpace::<D>::try_new([f64::MIN_POSITIVE; D]).unwrap();
    let mut coords = [-f64::from_bits(1); D];
    space.canonicalize_point(&mut coords);
    assert_eq!(
        coords.map(f64::to_bits),
        [f64::MIN_POSITIVE.next_down().to_bits(); D]
    );
}

/// Builds a proven non-degenerate fixture after axis reflection and input reordering.
fn transformed_periodic_vertices_t2(
    reflected_axes: [bool; 2],
    priorities: [u16; 7],
) -> Vec<Vertex<(), 2>> {
    let multipliers = [0.618_033_988_749_894_8, 0.414_213_562_373_095_03];
    let mut vertices: Vec<(usize, Vertex<(), 2>)> = (0..7)
        .map(|index| {
            let index_f64 = f64::from(u32::try_from(index).expect("fixture index fits in u32"));
            let coords = std::array::from_fn(|axis| {
                let axis_f64 = f64::from(u32::try_from(axis).expect("fixture axis fits in u32"));
                let stride = 0.037_f64.mul_add(axis_f64 + 1.0, multipliers[axis]);
                let base = 0.9_f64.mul_add(((index_f64 + 1.0) * stride).fract(), 0.05);
                if reflected_axes[axis] {
                    1.0 - base
                } else {
                    base
                }
            });
            (
                index,
                vertex!(coords).expect("transformed fixture coordinates are finite"),
            )
        })
        .collect();
    vertices.sort_by_key(|(index, _)| (priorities[*index], *index));
    vertices.into_iter().map(|(_, vertex)| vertex).collect()
}

// =============================================================================
// Tests
// =============================================================================

#[test]
fn canonicalize_roundoff_boundaries_2d_through_5d() {
    assert_toroidal_roundoff_contract::<2>();
    assert_toroidal_roundoff_contract::<3>();
    assert_toroidal_roundoff_contract::<4>();
    assert_toroidal_roundoff_contract::<5>();
}

repo_proptest! {
    /// Canonicalized coordinates always lie in `[0, L_i)` for every axis.
    #[test]
    fn prop_canonicalize_in_domain(domain in domain_t2(), mut coords in coords_2d()) {
        let space = ToroidalSpace::<2>::try_new(domain).unwrap();
        space.canonicalize_point(&mut coords);
        for (i, &period) in domain.iter().enumerate() {
            prop_assert!(
                coords[i] >= 0.0 && coords[i] < period,
                "axis {i}: coords[{i}] = {} not in [0, {})",
                coords[i], period
            );
        }
    }

    /// `canonicalize_point` is idempotent: applying it twice is the same as once.
    #[test]
    fn prop_canonicalize_idempotent(domain in domain_t2(), mut coords in coords_2d()) {
        let space = ToroidalSpace::<2>::try_new(domain).unwrap();
        space.canonicalize_point(&mut coords);
        let once = coords;
        space.canonicalize_point(&mut coords);
        prop_assert_eq!(
            once.map(f64::to_bits),
            coords.map(f64::to_bits),
            "canonicalization must be bit-exactly idempotent"
        );
    }
}

repo_proptest! {
    /// Periodic quotients retain validity and lifted quality under reflection and input permutation.
    #[test]
    fn prop_periodic_quotient_reflection_and_permutation_preserve_invariants(
        reflected_axes in prop::array::uniform2(any::<bool>()),
        priorities in prop::array::uniform7(any::<u16>()),
    ) {
        let vertices = transformed_periodic_vertices_t2(reflected_axes, priorities);
        let kernel = RobustKernel::new();
        let build_result = DelaunayTriangulationBuilder::new(&vertices)
            .try_toroidal([1.0_f64; 2])
            .expect("generated periods are finite and positive")
            .build_with_kernel(&kernel);
        prop_assert!(
            build_result.is_ok(),
            "transformed periodic construction failed: {:?}",
            build_result.as_ref().err(),
        );
        let triangulation = build_result.expect("successful result checked above");

        prop_assert_eq!(triangulation.number_of_vertices(), vertices.len());
        prop_assert!(triangulation.number_of_simplices() > 0);
        prop_assert!(triangulation.global_topology().is_periodic());
        let validation = triangulation.validate();
        prop_assert!(
            validation.is_ok(),
            "transformed periodic triangulation failed validation: {:?}",
            validation.as_ref().err(),
        );

        let mut saw_nonzero_offset = false;
        for (simplex_key, simplex) in triangulation.simplices() {
            let offsets = simplex.periodic_vertex_offsets();
            prop_assert!(offsets.is_some(), "periodic simplex omitted lifted offsets");
            let offsets = offsets.expect("presence checked above");
            prop_assert_eq!(offsets.len(), 3);
            saw_nonzero_offset |= offsets.iter().flatten().any(|component| *component != 0);

            // Independent triangle formulas in the explicit unit-domain lift.
            // This checks the published owner/offset wiring, complementing the
            // dimension-generic local chart fixtures in the quality module.
            let points: [[f64; 2]; 3] = std::array::from_fn(|slot| {
                let vertex = triangulation.vertex(simplex.vertices()[slot]).expect("validated vertex");
                std::array::from_fn(|axis| vertex.point().coords()[axis] + f64::from(offsets[slot][axis]))
            });
            let edges: [f64; 3] = std::array::from_fn(|i| {
                let next = (i + 1) % 3;
                (points[next][0] - points[i][0]).hypot(points[next][1] - points[i][1])
            });
            let area = (points[1][0] - points[0][0]).mul_add(
                points[2][1] - points[0][1],
                -(points[1][1] - points[0][1]) * (points[2][0] - points[0][0]),
            ).abs() / 2.0;
            prop_assert!(area > 0.0);
            let perimeter: f64 = edges.iter().sum();
            let expected_ratio = edges.iter().product::<f64>() * perimeter / (8.0 * area * area);
            let expected_volume = 9.0 * area / (perimeter * perimeter);
            let ratio = radius_ratio(triangulation.as_triangulation(), simplex_key)
                .map_err(|error| TestCaseError::fail(format!("periodic radius ratio failed: {error:?}")))?;
            let volume = normalized_volume(triangulation.as_triangulation(), simplex_key)
                .map_err(|error| TestCaseError::fail(format!("periodic normalized volume failed: {error:?}")))?;
            prop_assert!((ratio - expected_ratio).abs() <= 1e-10 * expected_ratio);
            prop_assert!((volume - expected_volume).abs() <= 1e-10 * expected_volume);
        }
        prop_assert!(saw_nonzero_offset, "fixture did not exercise a translated simplex image");
    }
}

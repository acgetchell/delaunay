//! Property-based tests for toroidal coordinate and periodic-simplex invariants.
//!
//! Verifies two fundamental properties of [`canonicalize_point`] and exercises
//! transformed 2D periodic quotient constructions through the public builder:
//! - **In-domain**: the result always lies in `[0, L_i)` for every axis `i`.
//! - **Idempotent**: applying canonicalization twice gives the same result as once.
//! - **Periodic construction invariance**: axis reflection and input permutation
//!   preserve Levels 1-5 validity for the validated unit-domain fixture.

#[cfg(feature = "slow-tests")]
use delaunay::prelude::construction::{DelaunayTriangulationBuilder, Vertex};
#[cfg(feature = "slow-tests")]
use delaunay::prelude::geometry::RobustKernel;
use delaunay::prelude::topology::spaces::{TopologicalSpace, ToroidalSpace};
#[cfg(feature = "slow-tests")]
use delaunay::vertex;
use proptest::prelude::*;

// =============================================================================
// Strategies
// =============================================================================

/// Generates a positive period in `(0.0001, 100.0]`.
fn positive_period() -> impl Strategy<Value = f64> {
    (0.0001_f64..=100.0_f64).prop_filter("must be positive finite", |p| p.is_finite() && *p > 0.0)
}

/// Generates a 2D toroidal domain `[L_x, L_y]`.
fn domain_2d() -> impl Strategy<Value = [f64; 2]> {
    [positive_period(), positive_period()]
}

/// Generates an arbitrary (possibly out-of-domain) 2D coordinate pair.
fn coords_2d() -> impl Strategy<Value = [f64; 2]> {
    let c = (-1000.0_f64..=1000.0_f64).prop_filter("finite", |x| x.is_finite());
    [c.clone(), c]
}

/// Builds a proven non-degenerate fixture after axis reflection and input reordering.
#[cfg(feature = "slow-tests")]
fn transformed_periodic_vertices_2d(
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

proptest! {
    /// Canonicalized coordinates always lie in `[0, L_i)` for every axis.
    #[test]
    fn prop_canonicalize_in_domain(domain in domain_2d(), mut coords in coords_2d()) {
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
    fn prop_canonicalize_idempotent(domain in domain_2d(), mut coords in coords_2d()) {
        let space = ToroidalSpace::<2>::try_new(domain).unwrap();
        space.canonicalize_point(&mut coords);
        let once = coords;
        space.canonicalize_point(&mut coords);
        let twice = coords;
        for i in 0..2 {
            prop_assert!(
                (once[i] - twice[i]).abs() < 1e-12,
                "axis {i}: once={} twice={} — canonicalize_point is not idempotent",
                once[i], twice[i]
            );
        }
    }
}

#[cfg(feature = "slow-tests")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Periodic quotient construction remains valid under axis reflections and input permutation.
    #[test]
    fn prop_periodic_quotient_reflection_and_permutation_preserve_invariants(
        reflected_axes in prop::array::uniform2(any::<bool>()),
        priorities in prop::array::uniform7(any::<u16>()),
    ) {
        let vertices = transformed_periodic_vertices_2d(reflected_axes, priorities);
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
        for (_, simplex) in triangulation.simplices() {
            let offsets = simplex.periodic_vertex_offsets();
            prop_assert!(offsets.is_some(), "periodic simplex omitted lifted offsets");
            let offsets = offsets.expect("presence checked above");
            prop_assert_eq!(offsets.len(), 3);
            saw_nonzero_offset |= offsets.iter().flatten().any(|component| *component != 0);
        }
        prop_assert!(saw_nonzero_offset, "fixture did not exercise a translated simplex image");
    }
}

//! Property-based tests for [`ToroidalSpace`] canonicalization.
//!
//! Verifies two fundamental properties of [`canonicalize_point`]:
//! - **In-domain**: the result always lies in `[0, L_i)` for every axis `i`.
//! - **Idempotent**: applying canonicalization twice gives the same result as once.

use delaunay::topology::spaces::ToroidalSpace;
use delaunay::topology::traits::topological_space::TopologicalSpace;
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

// =============================================================================
// Tests
// =============================================================================

proptest! {
    /// Canonicalized coordinates always lie in `[0, L_i)` for every axis.
    #[test]
    fn prop_canonicalize_in_domain(domain in domain_2d(), mut coords in coords_2d()) {
        let space = ToroidalSpace::<2>::new(domain);
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
        let space = ToroidalSpace::<2>::new(domain);
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

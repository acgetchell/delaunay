//! Finite check trait for validating coordinate values.
//!
//! This module provides the `FiniteCheck` trait which is used to validate
//! that coordinate values are finite (not NaN or infinite). This is essential
//! for geometric computations where infinite or NaN values would produce
//! invalid results.

/// Helper trait for checking finiteness of coordinates.
///
/// This trait provides a unified interface for checking whether a numeric value
/// is finite (not NaN or infinite). It's primarily used to validate coordinate
/// values in geometric types like points and vectors.
///
/// # Examples
///
/// ```
/// use delaunay::geometry::FiniteCheck;
///
/// let valid_value = 3.14f64;
/// assert!(valid_value.is_finite_generic());
///
/// let invalid_nan = f64::NAN;
/// assert!(!invalid_nan.is_finite_generic());
///
/// let invalid_inf = f64::INFINITY;
/// assert!(!invalid_inf.is_finite_generic());
/// ```
pub trait FiniteCheck {
    /// Returns true if the value is finite (not NaN or infinite).
    ///
    /// This method provides a consistent way to check finiteness across
    /// different numeric types, particularly floating-point types where
    /// NaN and infinity values are possible.
    ///
    /// # Returns
    ///
    /// - `true` if the value is finite
    /// - `false` if the value is NaN, positive infinity, or negative infinity
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::geometry::FiniteCheck;
    ///
    /// // Valid finite values
    /// assert!(1.0f64.is_finite_generic());
    /// assert!((-42.5f32).is_finite_generic());
    /// assert!(0.0f64.is_finite_generic());
    /// assert!(f64::MAX.is_finite_generic());
    /// assert!(f64::MIN.is_finite_generic());
    ///
    /// // Invalid non-finite values
    /// assert!(!f64::NAN.is_finite_generic());
    /// assert!(!f64::INFINITY.is_finite_generic());
    /// assert!(!f64::NEG_INFINITY.is_finite_generic());
    /// assert!(!f32::NAN.is_finite_generic());
    /// assert!(!f32::INFINITY.is_finite_generic());
    /// ```
    fn is_finite_generic(&self) -> bool;
}

// Unified macro for implementing FiniteCheck for floating-point types
macro_rules! impl_finite_check {
    (float: $($t:ty),*) => {
        $(
            impl FiniteCheck for $t {
                #[inline(always)]
                fn is_finite_generic(&self) -> bool {
                    self.is_finite()
                }
            }
        )*
    };
}

// Implement FiniteCheck for standard floating-point types
impl_finite_check!(float: f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_check_f64() {
        // Test valid finite f64 values
        assert!(1.0f64.is_finite_generic());
        assert!((-1.0f64).is_finite_generic());
        assert!(0.0f64.is_finite_generic());
        assert!((-0.0f64).is_finite_generic());
        assert!(f64::MAX.is_finite_generic());
        assert!(f64::MIN.is_finite_generic());
        assert!(f64::MIN_POSITIVE.is_finite_generic());
        assert!(1e308f64.is_finite_generic());
        assert!(1e-308f64.is_finite_generic());

        // Test invalid non-finite f64 values
        assert!(!f64::NAN.is_finite_generic());
        assert!(!f64::INFINITY.is_finite_generic());
        assert!(!f64::NEG_INFINITY.is_finite_generic());
    }

    #[test]
    fn finite_check_f32() {
        // Test valid finite f32 values
        assert!(1.0f32.is_finite_generic());
        assert!((-1.0f32).is_finite_generic());
        assert!(0.0f32.is_finite_generic());
        assert!((-0.0f32).is_finite_generic());
        assert!(f32::MAX.is_finite_generic());
        assert!(f32::MIN.is_finite_generic());
        assert!(f32::MIN_POSITIVE.is_finite_generic());
        assert!(1e38f32.is_finite_generic());
        assert!(1e-38f32.is_finite_generic());

        // Test invalid non-finite f32 values
        assert!(!f32::NAN.is_finite_generic());
        assert!(!f32::INFINITY.is_finite_generic());
        assert!(!f32::NEG_INFINITY.is_finite_generic());
    }

    #[test]
    fn finite_check_edge_cases() {
        // Test subnormal numbers (should be finite)
        let subnormal_f64 = f64::MIN_POSITIVE / 2.0;
        assert!(subnormal_f64.is_finite_generic());

        let subnormal_f32 = f32::MIN_POSITIVE / 2.0;
        assert!(subnormal_f32.is_finite_generic());

        // Test very large finite values
        let large_f64 = f64::MAX / 2.0;
        assert!(large_f64.is_finite_generic());

        let large_f32 = f32::MAX / 2.0;
        assert!(large_f32.is_finite_generic());

        // Test very small finite values
        let small_f64 = f64::MIN_POSITIVE * 2.0;
        assert!(small_f64.is_finite_generic());

        let small_f32 = f32::MIN_POSITIVE * 2.0;
        assert!(small_f32.is_finite_generic());
    }

    #[test]
    fn finite_check_different_nan_patterns() {
        // Test different ways to create NaN (all should be non-finite)
        let nan1 = f64::NAN;
        assert!(!nan1.is_finite_generic());

        #[allow(clippy::zero_divided_by_zero)]
        let nan2 = 0.0f64 / 0.0f64;
        assert!(!nan2.is_finite_generic());

        let nan3 = f64::INFINITY - f64::INFINITY;
        assert!(!nan3.is_finite_generic());

        // Same for f32
        let f32_nan1 = f32::NAN;
        assert!(!f32_nan1.is_finite_generic());

        #[allow(clippy::zero_divided_by_zero)]
        let f32_nan2 = 0.0f32 / 0.0f32;
        assert!(!f32_nan2.is_finite_generic());
    }

    #[test]
    fn finite_check_arithmetic_results() {
        // Test results of arithmetic operations
        let finite_result = 1.0f64 + 2.0f64;
        assert!(finite_result.is_finite_generic());

        let finite_product = 3.0f64 * 4.0f64;
        assert!(finite_product.is_finite_generic());

        let finite_quotient = 10.0f64 / 2.0f64;
        assert!(finite_quotient.is_finite_generic());

        // Operations that produce infinity
        let inf_result = f64::MAX * 2.0f64;
        assert!(!inf_result.is_finite_generic());

        let div_by_zero = 1.0f64 / 0.0f64;
        assert!(!div_by_zero.is_finite_generic());
    }

    #[test]
    fn finite_check_consistency() {
        // Verify that is_finite_generic is consistent with std::is_finite

        let test_values_f64 = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e308,
            -1e308,
        ];

        for &value in &test_values_f64 {
            assert_eq!(
                value.is_finite_generic(),
                value.is_finite(),
                "is_finite_generic should match is_finite for value: {value}"
            );
        }

        let test_values_f32 = [
            0.0f32,
            -0.0f32,
            1.0f32,
            -1.0f32,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e38f32,
            -1e38f32,
        ];

        for &value in &test_values_f32 {
            assert_eq!(
                value.is_finite_generic(),
                value.is_finite(),
                "is_finite_generic should match is_finite for value: {value}"
            );
        }
    }
}

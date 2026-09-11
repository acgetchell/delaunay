//! Validated periodic domains shared by geometry and topology.
//!
//! [`ToroidalDomain`](crate::geometry::periodic::ToroidalDomain) stores one finite,
//! strictly positive period per coordinate
//! axis, in the same units as the chart coordinates. Parse raw periods once
//! with [`ToroidalDomain::try_new`](crate::geometry::periodic::ToroidalDomain::try_new),
//! then reuse the domain for geometry and topology operations.
//!
//! For chart-local geometry, use
//! [`LabeledSimplexRealization::try_translated`](crate::geometry::realization::LabeledSimplexRealization::try_translated)
//! to shift a simplex by integer periods and
//! [`periodic_simplex_span`](crate::geometry::realization::periodic_simplex_span)
//! to find an axis spanning at least a full period. The domain proves period
//! validity; translations still check newly computed coordinate representability.

use thiserror::Error;

/// Errors that can occur while parsing a toroidal fundamental domain.
///
/// Toroidal domains require every period to be finite and strictly positive.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{ToroidalDomain, ToroidalDomainError};
///
/// std::assert_matches!(
///     ToroidalDomain::<2>::try_new([1.0, 0.0]),
///     Err(ToroidalDomainError::InvalidPeriod { axis: 1, period })
///         if period.abs() < f64::EPSILON
/// );
/// ```
#[derive(Clone, Copy, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ToroidalDomainError {
    /// A domain period was not finite and strictly positive.
    #[error("Invalid toroidal period {period:?} on axis {axis}; expected finite value > 0")]
    InvalidPeriod {
        /// Axis index containing the invalid period.
        axis: usize,
        /// Invalid period value.
        period: f64,
    },
}

/// Validated toroidal fundamental-domain periods.
///
/// This type carries the invariant that every period is finite and strictly
/// positive, so geometry and topology consumers can use the periods without
/// reparsing them.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::ToroidalDomain;
///
/// # fn main() -> Result<(), delaunay::prelude::geometry::ToroidalDomainError> {
/// let domain = ToroidalDomain::<2>::try_new([1.0, 2.0])?;
/// assert_eq!(domain.periods(), &[1.0, 2.0]);
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToroidalDomain<const D: usize> {
    periods: [f64; D],
}

impl<const D: usize> ToroidalDomain<D> {
    /// Creates a validated toroidal domain from raw periods.
    ///
    /// # Errors
    ///
    /// Returns [`ToroidalDomainError::InvalidPeriod`] when any period is
    /// non-finite, zero, or negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::{ToroidalDomain, ToroidalDomainError};
    ///
    /// # fn main() -> Result<(), ToroidalDomainError> {
    /// let domain = ToroidalDomain::<2>::try_new([1.0, 2.0])?;
    /// assert_eq!(domain.periods(), &[1.0, 2.0]);
    ///
    /// std::assert_matches!(
    ///     ToroidalDomain::<2>::try_new([0.0, 2.0]),
    ///     Err(ToroidalDomainError::InvalidPeriod { axis: 0, period })
    ///         if period.abs() < f64::EPSILON
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(periods: [f64; D]) -> Result<Self, ToroidalDomainError> {
        for (axis, period) in periods.iter().copied().enumerate() {
            if !period.is_finite() || period <= 0.0 {
                return Err(ToroidalDomainError::InvalidPeriod { axis, period });
            }
        }
        Ok(Self { periods })
    }

    /// Creates a unit toroidal domain with period `1.0` on every axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::ToroidalDomain;
    ///
    /// let domain = ToroidalDomain::<3>::unit();
    /// assert_eq!(domain.periods(), &[1.0, 1.0, 1.0]);
    /// ```
    pub const fn unit() -> Self {
        Self { periods: [1.0; D] }
    }

    /// Returns the validated periods.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::ToroidalDomain;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::geometry::ToroidalDomainError> {
    /// let domain = ToroidalDomain::<2>::try_new([2.0, 3.0])?;
    /// assert_eq!(domain.periods(), &[2.0, 3.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn periods(&self) -> &[f64; D] {
        &self.periods
    }

    /// Returns the period for one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::ToroidalDomain;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::geometry::ToroidalDomainError> {
    /// let domain = ToroidalDomain::<2>::try_new([2.0, 3.0])?;
    /// assert_eq!(domain.period(0), Some(2.0));
    /// assert_eq!(domain.period(2), None);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn period(&self, axis: usize) -> Option<f64> {
        self.periods.get(axis).copied()
    }

    /// Consumes the domain and returns the validated raw periods.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::ToroidalDomain;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::geometry::ToroidalDomainError> {
    /// let domain = ToroidalDomain::<2>::try_new([2.0, 3.0])?;
    /// assert_eq!(domain.into_periods(), [2.0, 3.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn into_periods(self) -> [f64; D] {
        self.periods
    }
}

impl<const D: usize> TryFrom<[f64; D]> for ToroidalDomain<D> {
    type Error = ToroidalDomainError;

    fn try_from(value: [f64; D]) -> Result<Self, Self::Error> {
        Self::try_new(value)
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use super::*;

    #[test]
    fn test_toroidal_domain_try_new_rejects_invalid_periods() {
        let zero = ToroidalDomain::<2>::try_new([1.0, 0.0]).unwrap_err();
        assert_matches!(
            zero,
            ToroidalDomainError::InvalidPeriod { axis: 1, period }
                if period.abs() < f64::EPSILON
        );

        let negative = ToroidalDomain::<2>::try_new([-1.0, 1.0]).unwrap_err();
        assert_matches!(
            negative,
            ToroidalDomainError::InvalidPeriod { axis: 0, period }
                if period < 0.0
        );

        let nan = ToroidalDomain::<2>::try_new([f64::NAN, 1.0]).unwrap_err();
        assert_matches!(
            nan,
            ToroidalDomainError::InvalidPeriod { axis: 0, period }
                if period.is_nan()
        );

        let infinite = ToroidalDomain::<2>::try_new([1.0, f64::INFINITY]).unwrap_err();
        assert_matches!(
            infinite,
            ToroidalDomainError::InvalidPeriod { axis: 1, period }
                if period.is_infinite()
        );
    }

    #[test]
    fn test_toroidal_domain_try_from_and_into_periods_preserve_validation() {
        let periods = [1.0, 2.0, 4.0];
        let domain = ToroidalDomain::<3>::try_from(periods).unwrap();
        let expected_bits = periods.map(f64::to_bits);
        assert_eq!(domain.periods().map(f64::to_bits), expected_bits);
        assert_eq!(domain.into_periods().map(f64::to_bits), expected_bits);

        let invalid = ToroidalDomain::<3>::try_from([1.0, f64::NEG_INFINITY, 4.0]).unwrap_err();
        assert_matches!(
            invalid,
            ToroidalDomainError::InvalidPeriod { axis: 1, period }
                if period.is_infinite() && period.is_sign_negative()
        );
    }
}

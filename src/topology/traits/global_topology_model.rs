//! Internal behavior models for [`GlobalTopology`] metadata.
//!
//! Public APIs continue to expose
//! [`GlobalTopology`]
//! as runtime metadata. This module provides scalar-generic behavior models used by
//! core triangulation/build paths to avoid ad-hoc topology branching.

#![forbid(unsafe_code)]

use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::topology::traits::topological_space::{GlobalTopology, TopologyKind};
use num_traits::NumCast;
use thiserror::Error;

/// Errors emitted by [`GlobalTopologyModel`] behavior calls.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum GlobalTopologyModelError {
    /// A toroidal period is invalid (must be finite and strictly positive).
    #[error("Invalid toroidal period {period:?} on axis {axis}; expected finite value > 0")]
    InvalidToroidalPeriod {
        /// Axis index containing the invalid period.
        axis: usize,
        /// Invalid period value.
        period: f64,
    },

    /// A coordinate could not be converted to/from `f64` during canonicalization/lifting.
    #[error("Failed scalar conversion while processing axis {axis} with value {value:?}")]
    ScalarConversion {
        /// Axis index where conversion failed.
        axis: usize,
        /// Source value that failed to convert.
        value: f64,
    },

    /// A coordinate is non-finite and cannot be canonicalized.
    #[error("Non-finite coordinate encountered while processing axis {axis}: {value:?}")]
    NonFiniteCoordinate {
        /// Axis index where non-finite coordinate was encountered.
        axis: usize,
        /// Non-finite coordinate value.
        value: f64,
    },

    /// Periodic offsets were requested for a non-periodic topology model.
    #[error("Periodic offsets are unsupported for {kind:?} topology")]
    PeriodicOffsetsUnsupported {
        /// Topology kind that rejected periodic offsets.
        kind: TopologyKind,
    },
}

/// Scalar-generic topology behavior abstraction used by core triangulation code.
pub trait GlobalTopologyModel<const D: usize> {
    /// Returns the topology kind represented by this model.
    fn kind(&self) -> TopologyKind;

    /// Returns whether this topology allows boundary facets.
    fn allows_boundary(&self) -> bool;

    /// Validates model configuration.
    ///
    /// Defaults to no-op for models without runtime parameters.
    ///
    /// # Errors
    ///
    /// Returns [`GlobalTopologyModelError::InvalidToroidalPeriod`] for invalid
    /// runtime configuration in models that require positive finite periods.
    fn validate_configuration(&self) -> Result<(), GlobalTopologyModelError> {
        Ok(())
    }

    /// Canonicalizes coordinates according to topology constraints.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`GlobalTopologyModelError::InvalidToroidalPeriod`] when toroidal
    ///   domain periods are invalid.
    /// - [`GlobalTopologyModelError::NonFiniteCoordinate`] when a coordinate is
    ///   `NaN` or infinite.
    /// - [`GlobalTopologyModelError::ScalarConversion`] when scalar
    ///   conversion to/from `f64` fails.
    fn canonicalize_point_in_place<T>(
        &self,
        coords: &mut [T; D],
    ) -> Result<(), GlobalTopologyModelError>
    where
        T: CoordinateScalar;

    /// Lifts coordinates for orientation predicates.
    ///
    /// For periodic models, `periodic_offset` applies lattice translation; for
    /// non-periodic models, passing an offset is an error.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`GlobalTopologyModelError::PeriodicOffsetsUnsupported`] when offsets
    ///   are supplied for non-periodic models.
    /// - [`GlobalTopologyModelError::InvalidToroidalPeriod`] when toroidal
    ///   domain periods are invalid.
    /// - [`GlobalTopologyModelError::ScalarConversion`] when scalar
    ///   conversion fails during lifting.
    fn lift_for_orientation<T>(
        &self,
        coords: [T; D],
        periodic_offset: Option<[i8; D]>,
    ) -> Result<[T; D], GlobalTopologyModelError>
    where
        T: CoordinateScalar;

    /// Returns the periodic domain when relevant.
    fn periodic_domain(&self) -> Option<&[f64; D]> {
        None
    }

    /// Optional hook indicating that periodic facet/signature behavior is available.
    fn supports_periodic_facet_signatures(&self) -> bool {
        false
    }
}

/// Euclidean behavior model (identity canonicalization/lifting).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EuclideanModel;

impl<const D: usize> GlobalTopologyModel<D> for EuclideanModel {
    fn kind(&self) -> TopologyKind {
        TopologyKind::Euclidean
    }

    fn allows_boundary(&self) -> bool {
        true
    }

    fn canonicalize_point_in_place<T>(
        &self,
        _coords: &mut [T; D],
    ) -> Result<(), GlobalTopologyModelError>
    where
        T: CoordinateScalar,
    {
        Ok(())
    }

    fn lift_for_orientation<T>(
        &self,
        coords: [T; D],
        periodic_offset: Option<[i8; D]>,
    ) -> Result<[T; D], GlobalTopologyModelError>
    where
        T: CoordinateScalar,
    {
        if periodic_offset.is_some() {
            return Err(GlobalTopologyModelError::PeriodicOffsetsUnsupported {
                kind: TopologyKind::Euclidean,
            });
        }
        Ok(coords)
    }
}

/// Toroidal behavior model (domain wrapping + lattice-offset lifting).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToroidalModel<const D: usize> {
    /// Fundamental-domain periods.
    pub domain: [f64; D],
}

impl<const D: usize> ToroidalModel<D> {
    /// Creates a toroidal model for the provided domain periods.
    #[must_use]
    pub const fn new(domain: [f64; D]) -> Self {
        Self { domain }
    }
}

impl<const D: usize> GlobalTopologyModel<D> for ToroidalModel<D> {
    fn kind(&self) -> TopologyKind {
        TopologyKind::Toroidal
    }

    fn allows_boundary(&self) -> bool {
        false
    }

    fn validate_configuration(&self) -> Result<(), GlobalTopologyModelError> {
        for (axis, period) in self.domain.iter().copied().enumerate().take(D) {
            if !period.is_finite() || period <= 0.0 {
                return Err(GlobalTopologyModelError::InvalidToroidalPeriod { axis, period });
            }
        }
        Ok(())
    }

    fn canonicalize_point_in_place<T>(
        &self,
        coords: &mut [T; D],
    ) -> Result<(), GlobalTopologyModelError>
    where
        T: CoordinateScalar,
    {
        self.validate_configuration()?;
        for (axis, coord_ref) in coords.iter_mut().enumerate().take(D) {
            let period = self.domain[axis];
            let Some(coord) = coord_ref.to_f64() else {
                let value = if coord_ref.is_nan() {
                    f64::NAN
                } else if coord_ref.is_infinite() {
                    if coord_ref.is_sign_negative() {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                } else {
                    f64::NAN
                };
                return Err(GlobalTopologyModelError::ScalarConversion { axis, value });
            };
            if !coord.is_finite() {
                return Err(GlobalTopologyModelError::NonFiniteCoordinate { axis, value: coord });
            }
            let wrapped = coord.rem_euclid(period);
            *coord_ref = <T as NumCast>::from(wrapped).ok_or(
                GlobalTopologyModelError::ScalarConversion {
                    axis,
                    value: wrapped,
                },
            )?;
        }
        Ok(())
    }

    fn lift_for_orientation<T>(
        &self,
        mut coords: [T; D],
        periodic_offset: Option<[i8; D]>,
    ) -> Result<[T; D], GlobalTopologyModelError>
    where
        T: CoordinateScalar,
    {
        let Some(offset) = periodic_offset else {
            return Ok(coords);
        };

        self.validate_configuration()?;
        for axis in 0..D {
            let period = self.domain[axis];
            let period_scalar =
                <T as NumCast>::from(period).ok_or(GlobalTopologyModelError::ScalarConversion {
                    axis,
                    value: period,
                })?;
            let offset_value = <f64 as From<i8>>::from(offset[axis]);
            let offset_scalar = <T as NumCast>::from(offset[axis]).ok_or(
                GlobalTopologyModelError::ScalarConversion {
                    axis,
                    value: offset_value,
                },
            )?;
            coords[axis] = coords[axis] + offset_scalar * period_scalar;
        }
        Ok(coords)
    }

    fn periodic_domain(&self) -> Option<&[f64; D]> {
        Some(&self.domain)
    }

    fn supports_periodic_facet_signatures(&self) -> bool {
        true
    }
}

/// Spherical behavior model scaffold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SphericalModel;

impl<const D: usize> GlobalTopologyModel<D> for SphericalModel {
    fn kind(&self) -> TopologyKind {
        TopologyKind::Spherical
    }

    fn allows_boundary(&self) -> bool {
        false
    }

    fn canonicalize_point_in_place<T>(
        &self,
        _coords: &mut [T; D],
    ) -> Result<(), GlobalTopologyModelError>
    where
        T: CoordinateScalar,
    {
        // Scaffold: full sphere-constrained projection will be expanded in #188.
        Ok(())
    }

    fn lift_for_orientation<T>(
        &self,
        coords: [T; D],
        periodic_offset: Option<[i8; D]>,
    ) -> Result<[T; D], GlobalTopologyModelError>
    where
        T: CoordinateScalar,
    {
        if periodic_offset.is_some() {
            return Err(GlobalTopologyModelError::PeriodicOffsetsUnsupported {
                kind: TopologyKind::Spherical,
            });
        }
        Ok(coords)
    }
}

/// Hyperbolic behavior model scaffold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HyperbolicModel;

impl<const D: usize> GlobalTopologyModel<D> for HyperbolicModel {
    fn kind(&self) -> TopologyKind {
        TopologyKind::Hyperbolic
    }

    fn allows_boundary(&self) -> bool {
        false
    }

    fn canonicalize_point_in_place<T>(
        &self,
        _coords: &mut [T; D],
    ) -> Result<(), GlobalTopologyModelError>
    where
        T: CoordinateScalar,
    {
        // Scaffold: full model-constrained projection is future work.
        Ok(())
    }

    fn lift_for_orientation<T>(
        &self,
        coords: [T; D],
        periodic_offset: Option<[i8; D]>,
    ) -> Result<[T; D], GlobalTopologyModelError>
    where
        T: CoordinateScalar,
    {
        if periodic_offset.is_some() {
            return Err(GlobalTopologyModelError::PeriodicOffsetsUnsupported {
                kind: TopologyKind::Hyperbolic,
            });
        }
        Ok(coords)
    }
}

/// Internal adapter from public [`GlobalTopology`] metadata to concrete behavior models.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GlobalTopologyModelAdapter<const D: usize> {
    /// Euclidean behavior.
    Euclidean(EuclideanModel),
    /// Toroidal behavior.
    Toroidal(ToroidalModel<D>),
    /// Spherical behavior.
    Spherical(SphericalModel),
    /// Hyperbolic behavior.
    Hyperbolic(HyperbolicModel),
}

impl<const D: usize> GlobalTopologyModelAdapter<D> {
    /// Builds a behavior adapter from public topology metadata.
    #[must_use]
    pub const fn from_global_topology(topology: GlobalTopology<D>) -> Self {
        match topology {
            GlobalTopology::Euclidean => Self::Euclidean(EuclideanModel),
            GlobalTopology::Toroidal { domain, .. } => Self::Toroidal(ToroidalModel::new(domain)),
            GlobalTopology::Spherical => Self::Spherical(SphericalModel),
            GlobalTopology::Hyperbolic => Self::Hyperbolic(HyperbolicModel),
        }
    }
}

impl<const D: usize> From<GlobalTopology<D>> for GlobalTopologyModelAdapter<D> {
    fn from(value: GlobalTopology<D>) -> Self {
        Self::from_global_topology(value)
    }
}

impl<const D: usize> GlobalTopology<D> {
    /// Returns the internal behavior adapter corresponding to this metadata.
    #[must_use]
    pub const fn model(self) -> GlobalTopologyModelAdapter<D> {
        GlobalTopologyModelAdapter::from_global_topology(self)
    }
}

impl<const D: usize> GlobalTopologyModel<D> for GlobalTopologyModelAdapter<D> {
    fn kind(&self) -> TopologyKind {
        match self {
            Self::Euclidean(..) => TopologyKind::Euclidean,
            Self::Toroidal(..) => TopologyKind::Toroidal,
            Self::Spherical(..) => TopologyKind::Spherical,
            Self::Hyperbolic(..) => TopologyKind::Hyperbolic,
        }
    }

    fn allows_boundary(&self) -> bool {
        match self {
            Self::Euclidean(..) => true,
            Self::Toroidal(..) | Self::Spherical(..) | Self::Hyperbolic(..) => false,
        }
    }

    fn validate_configuration(&self) -> Result<(), GlobalTopologyModelError> {
        match self {
            Self::Euclidean(model) => {
                <EuclideanModel as GlobalTopologyModel<D>>::validate_configuration(model)
            }
            Self::Toroidal(model) => {
                <ToroidalModel<D> as GlobalTopologyModel<D>>::validate_configuration(model)
            }
            Self::Spherical(model) => {
                <SphericalModel as GlobalTopologyModel<D>>::validate_configuration(model)
            }
            Self::Hyperbolic(model) => {
                <HyperbolicModel as GlobalTopologyModel<D>>::validate_configuration(model)
            }
        }
    }

    fn canonicalize_point_in_place<T>(
        &self,
        coords: &mut [T; D],
    ) -> Result<(), GlobalTopologyModelError>
    where
        T: CoordinateScalar,
    {
        match self {
            Self::Euclidean(model) => {
                <EuclideanModel as GlobalTopologyModel<D>>::canonicalize_point_in_place(
                    model, coords,
                )
            }
            Self::Toroidal(model) => {
                <ToroidalModel<D> as GlobalTopologyModel<D>>::canonicalize_point_in_place(
                    model, coords,
                )
            }
            Self::Spherical(model) => {
                <SphericalModel as GlobalTopologyModel<D>>::canonicalize_point_in_place(
                    model, coords,
                )
            }
            Self::Hyperbolic(model) => {
                <HyperbolicModel as GlobalTopologyModel<D>>::canonicalize_point_in_place(
                    model, coords,
                )
            }
        }
    }

    fn lift_for_orientation<T>(
        &self,
        coords: [T; D],
        periodic_offset: Option<[i8; D]>,
    ) -> Result<[T; D], GlobalTopologyModelError>
    where
        T: CoordinateScalar,
    {
        match self {
            Self::Euclidean(model) => {
                <EuclideanModel as GlobalTopologyModel<D>>::lift_for_orientation(
                    model,
                    coords,
                    periodic_offset,
                )
            }
            Self::Toroidal(model) => {
                <ToroidalModel<D> as GlobalTopologyModel<D>>::lift_for_orientation(
                    model,
                    coords,
                    periodic_offset,
                )
            }
            Self::Spherical(model) => {
                <SphericalModel as GlobalTopologyModel<D>>::lift_for_orientation(
                    model,
                    coords,
                    periodic_offset,
                )
            }
            Self::Hyperbolic(model) => {
                <HyperbolicModel as GlobalTopologyModel<D>>::lift_for_orientation(
                    model,
                    coords,
                    periodic_offset,
                )
            }
        }
    }

    fn periodic_domain(&self) -> Option<&[f64; D]> {
        match self {
            Self::Euclidean(model) => {
                <EuclideanModel as GlobalTopologyModel<D>>::periodic_domain(model)
            }
            Self::Toroidal(model) => {
                <ToroidalModel<D> as GlobalTopologyModel<D>>::periodic_domain(model)
            }
            Self::Spherical(model) => {
                <SphericalModel as GlobalTopologyModel<D>>::periodic_domain(model)
            }
            Self::Hyperbolic(model) => {
                <HyperbolicModel as GlobalTopologyModel<D>>::periodic_domain(model)
            }
        }
    }

    fn supports_periodic_facet_signatures(&self) -> bool {
        match self {
            Self::Euclidean(model) => {
                <EuclideanModel as GlobalTopologyModel<D>>::supports_periodic_facet_signatures(
                    model,
                )
            }
            Self::Toroidal(model) => {
                <ToroidalModel<D> as GlobalTopologyModel<D>>::supports_periodic_facet_signatures(
                    model,
                )
            }
            Self::Spherical(model) => {
                <SphericalModel as GlobalTopologyModel<D>>::supports_periodic_facet_signatures(
                    model,
                )
            }
            Self::Hyperbolic(model) => {
                <HyperbolicModel as GlobalTopologyModel<D>>::supports_periodic_facet_signatures(
                    model,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn model_adapter_dispatch_matches_global_topology_metadata() {
        let euclidean = GlobalTopology::<3>::Euclidean.model();
        assert_eq!(euclidean.kind(), TopologyKind::Euclidean);
        assert!(euclidean.allows_boundary());

        let toroidal = GlobalTopology::<2>::Toroidal {
            domain: [2.0, 3.0],
            mode:
                crate::topology::traits::topological_space::ToroidalConstructionMode::Canonicalized,
        }
        .model();
        assert_eq!(toroidal.kind(), TopologyKind::Toroidal);
        assert!(!toroidal.allows_boundary());
    }

    #[test]
    fn toroidal_model_canonicalization_wraps_coordinates_into_domain() {
        let model = ToroidalModel::<2>::new([2.0, 3.0]);
        let mut coords = [2.5_f64, -1.0_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert_relative_eq!(coords[0], 0.5);
        assert_relative_eq!(coords[1], 2.0);
    }

    #[test]
    fn toroidal_model_canonicalization_rejects_non_finite_coordinates() {
        let model = ToroidalModel::<2>::new([2.0, 3.0]);
        let mut coords = [f64::INFINITY, 1.0_f64];
        let err = model.canonicalize_point_in_place(&mut coords).unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::NonFiniteCoordinate { axis: 0, value }
                if value.is_infinite() && value.is_sign_positive()
        ));
    }

    #[test]
    fn toroidal_model_lift_applies_lattice_offset() {
        let model = ToroidalModel::<2>::new([2.0, 3.0]);
        let lifted = model
            .lift_for_orientation([0.5_f64, 0.25_f64], Some([1, -1]))
            .unwrap();
        assert_relative_eq!(lifted[0], 2.5);
        assert_relative_eq!(lifted[1], -2.75);
    }

    #[test]
    fn non_periodic_models_reject_periodic_offsets() {
        let model = EuclideanModel;
        let err = model
            .lift_for_orientation([0.5_f64, 0.25_f64], Some([1, 0]))
            .unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::PeriodicOffsetsUnsupported {
                kind: TopologyKind::Euclidean,
            }
        ));
    }
}

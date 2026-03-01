//! Internal behavior models for [`GlobalTopology`] metadata.
//!
//! Public APIs continue to expose [`GlobalTopology`] as runtime metadata. This module provides
//! scalar-generic behavior models used by core triangulation/build paths to avoid ad-hoc topology
//! branching.
//!
//! # Overview
//!
//! The [`GlobalTopologyModel`] trait abstracts topology-specific behavior for coordinate
//! canonicalization and orientation predicate lifting. Concrete implementations include:
//!
//! - [`EuclideanModel`]: Identity operations (no wrapping or lifting)
//! - [`ToroidalModel`]: Domain wrapping and lattice-offset lifting for periodic boundaries
//! - [`SphericalModel`]: Scaffold for future sphere-constrained projection
//! - [`HyperbolicModel`]: Scaffold for future hyperbolic projection
//!
//! The [`GlobalTopologyModelAdapter`] enum provides dynamic dispatch over these models and is
//! obtained via [`GlobalTopology::model()`].
//!

#![forbid(unsafe_code)]

use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::topology::traits::topological_space::{
    GlobalTopology, TopologyKind, ToroidalConstructionMode,
};
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
    ///
    /// For periodic topologies (e.g., toroidal), this returns the fundamental domain periods.
    /// For non-periodic topologies, returns `None`.
    fn periodic_domain(&self) -> Option<&[f64; D]> {
        None
    }

    /// Indicates whether periodic facet/signature behavior is available.
    ///
    /// Returns `true` for periodic topologies that support lattice-offset tracking on cells.
    /// This is used internally by the triangulation to determine whether periodic vertex offsets
    /// should be stored and processed.
    fn supports_periodic_facet_signatures(&self) -> bool {
        false
    }

    /// Indicates whether periodic offsets are supported for orientation lifting.
    ///
    /// This is checked when validating/using per-vertex periodic offsets while preparing
    /// points for orientation predicates.
    ///
    /// By default this mirrors [`supports_periodic_facet_signatures`](Self::supports_periodic_facet_signatures)
    /// to preserve current behavior.
    fn supports_periodic_orientation_offsets(&self) -> bool {
        self.supports_periodic_facet_signatures()
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
    /// Construction mode (Phase 1 canonicalized vs Phase 2 periodic image-point).
    pub mode: ToroidalConstructionMode,
}

impl<const D: usize> ToroidalModel<D> {
    /// Creates a toroidal model for the provided domain periods and construction mode.
    ///
    /// Note: `ToroidalModel` is internal; users should access via [`GlobalTopology::model()`].
    #[must_use]
    pub const fn new(domain: [f64; D], mode: ToroidalConstructionMode) -> Self {
        Self { domain, mode }
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
        // Validate finiteness before performing arithmetic
        for (axis, coord_ref) in coords.iter().enumerate().take(D) {
            if let Some(coord_f64) = coord_ref.to_f64()
                && !coord_f64.is_finite()
            {
                return Err(GlobalTopologyModelError::NonFiniteCoordinate {
                    axis,
                    value: coord_f64,
                });
            }
        }
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
        matches!(self.mode, ToroidalConstructionMode::PeriodicImagePoint)
    }

    fn supports_periodic_orientation_offsets(&self) -> bool {
        self.supports_periodic_facet_signatures()
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
    ///
    /// This constructor is used internally by [`GlobalTopology::model()`] to convert public
    /// topology metadata into an internal behavior model.
    #[must_use]
    pub const fn from_global_topology(topology: GlobalTopology<D>) -> Self {
        match topology {
            GlobalTopology::Euclidean => Self::Euclidean(EuclideanModel),
            GlobalTopology::Toroidal { domain, mode } => {
                Self::Toroidal(ToroidalModel::new(domain, mode))
            }
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
    ///
    /// This method converts the public [`GlobalTopology`] metadata into an internal behavior model
    /// that implements the `GlobalTopologyModel` trait. The returned adapter can be used internally
    /// to perform topology-specific operations like coordinate canonicalization and orientation lifting.
    #[must_use]
    pub const fn model(self) -> GlobalTopologyModelAdapter<D> {
        GlobalTopologyModelAdapter::from_global_topology(self)
    }
}

impl<const D: usize> GlobalTopologyModel<D> for GlobalTopologyModelAdapter<D> {
    fn kind(&self) -> TopologyKind {
        match self {
            Self::Euclidean(model) => <EuclideanModel as GlobalTopologyModel<D>>::kind(model),
            Self::Toroidal(model) => <ToroidalModel<D> as GlobalTopologyModel<D>>::kind(model),
            Self::Spherical(model) => <SphericalModel as GlobalTopologyModel<D>>::kind(model),
            Self::Hyperbolic(model) => <HyperbolicModel as GlobalTopologyModel<D>>::kind(model),
        }
    }

    fn allows_boundary(&self) -> bool {
        match self {
            Self::Euclidean(model) => {
                <EuclideanModel as GlobalTopologyModel<D>>::allows_boundary(model)
            }
            Self::Toroidal(model) => {
                <ToroidalModel<D> as GlobalTopologyModel<D>>::allows_boundary(model)
            }
            Self::Spherical(model) => {
                <SphericalModel as GlobalTopologyModel<D>>::allows_boundary(model)
            }
            Self::Hyperbolic(model) => {
                <HyperbolicModel as GlobalTopologyModel<D>>::allows_boundary(model)
            }
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

    fn supports_periodic_orientation_offsets(&self) -> bool {
        match self {
            Self::Euclidean(model) => {
                <EuclideanModel as GlobalTopologyModel<D>>::supports_periodic_orientation_offsets(
                    model,
                )
            }
            Self::Toroidal(model) => {
                <ToroidalModel<D> as GlobalTopologyModel<D>>::supports_periodic_orientation_offsets(
                    model,
                )
            }
            Self::Spherical(model) => {
                <SphericalModel as GlobalTopologyModel<D>>::supports_periodic_orientation_offsets(
                    model,
                )
            }
            Self::Hyperbolic(model) => {
                <HyperbolicModel as GlobalTopologyModel<D>>::supports_periodic_orientation_offsets(
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
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let mut coords = [2.5_f64, -1.0_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert_relative_eq!(coords[0], 0.5);
        assert_relative_eq!(coords[1], 2.0);
    }

    #[test]
    fn toroidal_model_canonicalization_rejects_non_finite_coordinates() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let mut coords = [f64::INFINITY, 1.0_f64];
        let err = model.canonicalize_point_in_place(&mut coords).unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::NonFiniteCoordinate { axis: 0, value }
                if value.is_infinite() && value.is_sign_positive()
        ));
    }

    #[test]
    fn toroidal_model_canonicalization_rejects_nan_coordinates() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let mut coords = [f64::NAN, 1.0_f64];
        let err = model.canonicalize_point_in_place(&mut coords).unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::NonFiniteCoordinate { axis: 0, value }
                if value.is_nan()
        ));
    }

    #[test]
    fn toroidal_model_lift_applies_lattice_offset() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let lifted = model
            .lift_for_orientation([0.5_f64, 0.25_f64], Some([1, -1]))
            .unwrap();
        assert_relative_eq!(lifted[0], 2.5);
        assert_relative_eq!(lifted[1], -2.75);
    }

    #[test]
    fn toroidal_model_lift_rejects_non_finite_coordinates() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let err = model
            .lift_for_orientation([f64::NAN, 0.5_f64], Some([1, 0]))
            .unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::NonFiniteCoordinate { axis: 0, value }
                if value.is_nan()
        ));
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

    // =========================================================================
    // EuclideanModel tests
    // =========================================================================

    #[test]
    fn euclidean_model_returns_correct_kind() {
        let model = EuclideanModel;
        assert_eq!(
            <EuclideanModel as GlobalTopologyModel<2>>::kind(&model),
            TopologyKind::Euclidean
        );
    }

    #[test]
    fn euclidean_model_allows_boundary() {
        let model = EuclideanModel;
        assert!(<EuclideanModel as GlobalTopologyModel<2>>::allows_boundary(
            &model
        ));
    }

    #[test]
    fn euclidean_model_validation_always_succeeds() {
        let model = EuclideanModel;
        assert!(<EuclideanModel as GlobalTopologyModel<2>>::validate_configuration(&model).is_ok());
    }

    #[test]
    fn euclidean_model_canonicalization_is_identity() {
        let model = EuclideanModel;
        let mut coords = [1.5_f64, 2.5_f64, 3.5_f64];
        <EuclideanModel as GlobalTopologyModel<3>>::canonicalize_point_in_place(
            &model,
            &mut coords,
        )
        .unwrap();
        assert_relative_eq!(coords[0], 1.5);
        assert_relative_eq!(coords[1], 2.5);
        assert_relative_eq!(coords[2], 3.5);
    }

    #[test]
    fn euclidean_model_lift_without_offset_is_identity() {
        let model = EuclideanModel;
        let coords = [1.5_f64, 2.5_f64];
        let lifted =
            <EuclideanModel as GlobalTopologyModel<2>>::lift_for_orientation(&model, coords, None)
                .unwrap();
        assert_relative_eq!(lifted[0], 1.5);
        assert_relative_eq!(lifted[1], 2.5);
    }

    #[test]
    fn euclidean_model_has_no_periodic_domain() {
        let model = EuclideanModel;
        assert_eq!(
            <EuclideanModel as GlobalTopologyModel<2>>::periodic_domain(&model),
            None
        );
    }

    #[test]
    fn euclidean_model_does_not_support_periodic_signatures() {
        let model = EuclideanModel;
        assert!(
            !<EuclideanModel as GlobalTopologyModel<2>>::supports_periodic_facet_signatures(&model)
        );
    }

    // =========================================================================
    // SphericalModel and HyperbolicModel tests
    // =========================================================================

    #[test]
    fn spherical_model_returns_correct_kind() {
        let model = SphericalModel;
        assert_eq!(
            <SphericalModel as GlobalTopologyModel<2>>::kind(&model),
            TopologyKind::Spherical
        );
    }

    #[test]
    fn spherical_model_does_not_allow_boundary() {
        let model = SphericalModel;
        assert!(!<SphericalModel as GlobalTopologyModel<2>>::allows_boundary(&model));
    }

    #[test]
    fn spherical_model_rejects_periodic_offsets() {
        let model = SphericalModel;
        let err = <SphericalModel as GlobalTopologyModel<2>>::lift_for_orientation(
            &model,
            [1.0_f64, 0.0_f64],
            Some([1, 0]),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::PeriodicOffsetsUnsupported {
                kind: TopologyKind::Spherical,
            }
        ));
    }

    #[test]
    fn spherical_model_has_no_periodic_domain() {
        let model = SphericalModel;
        assert_eq!(
            <SphericalModel as GlobalTopologyModel<2>>::periodic_domain(&model),
            None
        );
    }

    #[test]
    fn hyperbolic_model_returns_correct_kind() {
        let model = HyperbolicModel;
        assert_eq!(
            <HyperbolicModel as GlobalTopologyModel<2>>::kind(&model),
            TopologyKind::Hyperbolic
        );
    }

    #[test]
    fn hyperbolic_model_does_not_allow_boundary() {
        let model = HyperbolicModel;
        assert!(!<HyperbolicModel as GlobalTopologyModel<2>>::allows_boundary(&model));
    }

    #[test]
    fn hyperbolic_model_rejects_periodic_offsets() {
        let model = HyperbolicModel;
        let err = <HyperbolicModel as GlobalTopologyModel<2>>::lift_for_orientation(
            &model,
            [1.0_f64, 0.0_f64],
            Some([1, 0]),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::PeriodicOffsetsUnsupported {
                kind: TopologyKind::Hyperbolic,
            }
        ));
    }

    #[test]
    fn hyperbolic_model_has_no_periodic_domain() {
        let model = HyperbolicModel;
        assert_eq!(
            <HyperbolicModel as GlobalTopologyModel<2>>::periodic_domain(&model),
            None
        );
    }

    // =========================================================================
    // GlobalTopologyModelAdapter delegation tests
    // =========================================================================

    #[test]
    fn adapter_delegates_kind_to_euclidean_model() {
        let adapter = GlobalTopologyModelAdapter::<2>::Euclidean(EuclideanModel);
        assert_eq!(adapter.kind(), TopologyKind::Euclidean);
    }

    #[test]
    fn adapter_delegates_allows_boundary_to_toroidal_model() {
        let adapter = GlobalTopologyModelAdapter::<2>::Toroidal(ToroidalModel::new(
            [2.0, 3.0],
            ToroidalConstructionMode::Canonicalized,
        ));
        assert!(!adapter.allows_boundary());
    }

    #[test]
    fn adapter_delegates_validate_configuration_to_toroidal_model() {
        let adapter = GlobalTopologyModelAdapter::<2>::Toroidal(ToroidalModel::new(
            [2.0, 3.0],
            ToroidalConstructionMode::Canonicalized,
        ));
        assert!(adapter.validate_configuration().is_ok());

        let bad_adapter = GlobalTopologyModelAdapter::<2>::Toroidal(ToroidalModel::new(
            [0.0, 3.0],
            ToroidalConstructionMode::Canonicalized,
        ));
        assert!(bad_adapter.validate_configuration().is_err());
    }

    #[test]
    fn adapter_delegates_canonicalize_to_toroidal_model() {
        let adapter = GlobalTopologyModelAdapter::<2>::Toroidal(ToroidalModel::new(
            [2.0, 3.0],
            ToroidalConstructionMode::Canonicalized,
        ));
        let mut coords = [2.5_f64, -1.0_f64];
        adapter.canonicalize_point_in_place(&mut coords).unwrap();
        assert_relative_eq!(coords[0], 0.5);
        assert_relative_eq!(coords[1], 2.0);
    }

    #[test]
    fn adapter_delegates_lift_to_toroidal_model() {
        let adapter = GlobalTopologyModelAdapter::<2>::Toroidal(ToroidalModel::new(
            [2.0, 3.0],
            ToroidalConstructionMode::Canonicalized,
        ));
        let lifted = adapter
            .lift_for_orientation([0.5_f64, 0.25_f64], Some([1, -1]))
            .unwrap();
        assert_relative_eq!(lifted[0], 2.5);
        assert_relative_eq!(lifted[1], -2.75);
    }

    #[test]
    fn adapter_delegates_periodic_domain_to_models() {
        let euclidean = GlobalTopologyModelAdapter::<2>::Euclidean(EuclideanModel);
        assert_eq!(euclidean.periodic_domain(), None);

        let toroidal = GlobalTopologyModelAdapter::<2>::Toroidal(ToroidalModel::new(
            [2.0, 3.0],
            ToroidalConstructionMode::Canonicalized,
        ));
        assert_eq!(toroidal.periodic_domain(), Some(&[2.0, 3.0]));
    }

    #[test]
    fn adapter_delegates_periodic_signatures_to_models() {
        let euclidean = GlobalTopologyModelAdapter::<2>::Euclidean(EuclideanModel);
        assert!(!euclidean.supports_periodic_facet_signatures());

        // Canonicalized mode does not support periodic facet signatures
        let toroidal_canon = GlobalTopologyModelAdapter::<2>::Toroidal(ToroidalModel::new(
            [2.0, 3.0],
            ToroidalConstructionMode::Canonicalized,
        ));
        assert!(!toroidal_canon.supports_periodic_facet_signatures());

        // PeriodicImagePoint mode supports periodic facet signatures
        let toroidal_periodic = GlobalTopologyModelAdapter::<2>::Toroidal(ToroidalModel::new(
            [2.0, 3.0],
            ToroidalConstructionMode::PeriodicImagePoint,
        ));
        assert!(toroidal_periodic.supports_periodic_facet_signatures());
    }

    #[test]
    fn adapter_delegates_periodic_orientation_offsets_to_models() {
        let euclidean = GlobalTopologyModelAdapter::<2>::Euclidean(EuclideanModel);
        assert!(!euclidean.supports_periodic_orientation_offsets());

        // Canonicalized mode does not support periodic orientation offsets
        let toroidal_canon = GlobalTopologyModelAdapter::<2>::Toroidal(ToroidalModel::new(
            [2.0, 3.0],
            ToroidalConstructionMode::Canonicalized,
        ));
        assert!(!toroidal_canon.supports_periodic_orientation_offsets());

        // PeriodicImagePoint mode supports periodic orientation offsets
        let toroidal_periodic = GlobalTopologyModelAdapter::<2>::Toroidal(ToroidalModel::new(
            [2.0, 3.0],
            ToroidalConstructionMode::PeriodicImagePoint,
        ));
        assert!(toroidal_periodic.supports_periodic_orientation_offsets());
    }

    // =========================================================================
    // Error handling tests
    // =========================================================================

    #[test]
    fn toroidal_model_rejects_zero_period() {
        let model = ToroidalModel::<2>::new([0.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let err = model.validate_configuration().unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::InvalidToroidalPeriod { axis: 0, period }
                if period.abs() < f64::EPSILON
        ));
    }

    #[test]
    fn toroidal_model_rejects_negative_period() {
        let model = ToroidalModel::<2>::new([2.0, -1.0], ToroidalConstructionMode::Canonicalized);
        let err = model.validate_configuration().unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::InvalidToroidalPeriod { axis: 1, period }
                if period < 0.0
        ));
    }

    #[test]
    fn toroidal_model_rejects_infinite_period() {
        let model = ToroidalModel::<2>::new(
            [f64::INFINITY, 3.0],
            ToroidalConstructionMode::Canonicalized,
        );
        let err = model.validate_configuration().unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::InvalidToroidalPeriod { axis: 0, period }
                if period.is_infinite()
        ));
    }

    #[test]
    fn toroidal_model_rejects_nan_period() {
        let model =
            ToroidalModel::<2>::new([f64::NAN, 3.0], ToroidalConstructionMode::Canonicalized);
        let err = model.validate_configuration().unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::InvalidToroidalPeriod { axis: 0, period }
                if period.is_nan()
        ));
    }

    #[test]
    fn toroidal_model_canonicalization_rejects_negative_infinity() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let mut coords = [f64::NEG_INFINITY, 1.0_f64];
        let err = model.canonicalize_point_in_place(&mut coords).unwrap_err();
        assert!(matches!(
            err,
            GlobalTopologyModelError::NonFiniteCoordinate { axis: 0, value }
                if value.is_infinite() && value.is_sign_negative()
        ));
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn toroidal_model_canonicalization_handles_large_coordinates() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let mut coords = [1e10_f64, -1e10_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        // Should wrap into [0, 2.0) and [0, 3.0)
        assert!(coords[0] >= 0.0 && coords[0] < 2.0);
        assert!(coords[1] >= 0.0 && coords[1] < 3.0);
    }

    #[test]
    fn toroidal_model_canonicalization_handles_exact_period() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let mut coords = [2.0_f64, 3.0_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert_relative_eq!(coords[0], 0.0);
        assert_relative_eq!(coords[1], 0.0);
    }

    #[test]
    fn toroidal_model_lift_with_zero_offset_is_identity() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let coords = [0.5_f64, 1.5_f64];
        let lifted = model.lift_for_orientation(coords, Some([0, 0])).unwrap();
        assert_relative_eq!(lifted[0], 0.5);
        assert_relative_eq!(lifted[1], 1.5);
    }

    #[test]
    fn toroidal_model_lift_with_large_offset() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let coords = [0.5_f64, 0.25_f64];
        let lifted = model
            .lift_for_orientation(coords, Some([127, -128]))
            .unwrap();
        assert_relative_eq!(lifted[0], 127.0_f64.mul_add(2.0, 0.5));
        assert_relative_eq!(lifted[1], (-128.0_f64).mul_add(3.0, 0.25));
    }

    #[test]
    fn toroidal_model_works_with_f32() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let mut coords = [2.5_f32, -1.0_f32];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert!((coords[0] - 0.5).abs() < 1e-6);
        assert!((coords[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn toroidal_model_works_in_higher_dimensions() {
        let model = ToroidalModel::<5>::new(
            [2.0, 3.0, 4.0, 5.0, 6.0],
            ToroidalConstructionMode::Canonicalized,
        );
        let mut coords = [2.5_f64, -1.0_f64, 8.5_f64, 10.5_f64, 12.5_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert_relative_eq!(coords[0], 0.5);
        assert_relative_eq!(coords[1], 2.0);
        assert_relative_eq!(coords[2], 0.5);
        assert_relative_eq!(coords[3], 0.5);
        assert_relative_eq!(coords[4], 0.5);
    }

    #[test]
    fn global_topology_model_adapter_from_trait_conversion() {
        let topology = GlobalTopology::<2>::Euclidean;
        let adapter: GlobalTopologyModelAdapter<2> = topology.into();
        assert_eq!(adapter.kind(), TopologyKind::Euclidean);
    }

    // =========================================================================
    // ToroidalModel edge case tests
    // =========================================================================

    #[test]
    fn toroidal_model_handles_very_small_periods() {
        let model = ToroidalModel::<2>::new(
            [1e-6_f64, 1e-6_f64],
            ToroidalConstructionMode::Canonicalized,
        );
        assert!(model.validate_configuration().is_ok());

        let mut coords = [5e-7_f64, 1.5e-6_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert!(coords[0] >= 0.0 && coords[0] < 1e-6);
        assert!(coords[1] >= 0.0 && coords[1] < 1e-6);
    }

    #[test]
    fn toroidal_model_handles_boundary_coordinates() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);

        // Coordinate exactly at zero
        let mut coords = [0.0_f64, 0.0_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert_relative_eq!(coords[0], 0.0);
        assert_relative_eq!(coords[1], 0.0);

        // Coordinate just below period
        let epsilon = 1e-10_f64;
        let mut coords = [2.0 - epsilon, 3.0 - epsilon];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert!(coords[0] >= 0.0 && coords[0] < 2.0);
        assert!(coords[1] >= 0.0 && coords[1] < 3.0);
        assert_relative_eq!(coords[0], 2.0 - epsilon);
        assert_relative_eq!(coords[1], 3.0 - epsilon);
    }

    #[test]
    fn toroidal_model_handles_multi_wrap_coordinates() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);

        // Coordinate requiring multiple wraps
        let mut coords = [10.5_f64, -15.25_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert_relative_eq!(coords[0], 0.5);
        assert_relative_eq!(coords[1], 2.75);

        // Very large positive coordinate
        let mut coords = [1000.75_f64, 2000.5_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert!(coords[0] >= 0.0 && coords[0] < 2.0);
        assert!(coords[1] >= 0.0 && coords[1] < 3.0);

        // Very large negative coordinate
        let mut coords = [-1000.75_f64, -2000.5_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert!(coords[0] >= 0.0 && coords[0] < 2.0);
        assert!(coords[1] >= 0.0 && coords[1] < 3.0);
    }

    #[test]
    fn toroidal_model_handles_mixed_positive_negative_wrapping() {
        let model =
            ToroidalModel::<3>::new([2.0, 3.0, 4.0], ToroidalConstructionMode::Canonicalized);

        let mut coords = [5.5_f64, -1.0_f64, 0.5_f64];
        model.canonicalize_point_in_place(&mut coords).unwrap();
        assert_relative_eq!(coords[0], 1.5);
        assert_relative_eq!(coords[1], 2.0);
        assert_relative_eq!(coords[2], 0.5);
    }

    #[test]
    fn toroidal_model_lift_with_mixed_offsets() {
        let model =
            ToroidalModel::<3>::new([2.0, 3.0, 4.0], ToroidalConstructionMode::Canonicalized);

        let lifted = model
            .lift_for_orientation([0.5_f64, 1.0_f64, 2.0_f64], Some([1, 0, -1]))
            .unwrap();
        assert_relative_eq!(lifted[0], 2.5);
        assert_relative_eq!(lifted[1], 1.0);
        assert_relative_eq!(lifted[2], -2.0);
    }

    // =========================================================================
    // Error message quality tests
    // =========================================================================

    #[test]
    fn invalid_toroidal_period_error_includes_axis_and_value() {
        let model =
            ToroidalModel::<3>::new([2.0, -5.0, 3.0], ToroidalConstructionMode::Canonicalized);
        let err = model.validate_configuration().unwrap_err();

        let err_str = err.to_string();
        assert!(
            err_str.contains("axis 1"),
            "Error should mention axis: {err_str}"
        );
        assert!(
            err_str.contains("-5") || err_str.contains("5.0"),
            "Error should include period value: {err_str}"
        );
        assert!(
            err_str.contains("expected finite value > 0") || err_str.contains("> 0"),
            "Error should explain expected range: {err_str}"
        );
    }

    #[test]
    fn non_finite_coordinate_error_includes_axis_and_value() {
        let model = ToroidalModel::<4>::new(
            [2.0, 3.0, 4.0, 5.0],
            ToroidalConstructionMode::Canonicalized,
        );
        let mut coords = [1.0_f64, 2.0_f64, f64::INFINITY, 3.0_f64];
        let err = model.canonicalize_point_in_place(&mut coords).unwrap_err();

        let err_str = err.to_string();
        assert!(
            err_str.contains("axis 2"),
            "Error should mention axis: {err_str}"
        );
        assert!(
            err_str.contains("inf") || err_str.contains("Infinity"),
            "Error should indicate non-finite value: {err_str}"
        );
    }

    #[test]
    fn periodic_offsets_unsupported_error_includes_topology_kind() {
        let model = SphericalModel;
        let err = model
            .lift_for_orientation([1.0_f64, 2.0_f64, 3.0_f64], Some([1, 0, -1]))
            .unwrap_err();

        let err_str = err.to_string();
        assert!(
            err_str.contains("Spherical") || err_str.contains("spherical"),
            "Error should mention topology kind: {err_str}"
        );
        assert!(
            err_str.contains("unsupported") || err_str.contains("not supported"),
            "Error should indicate unsupported operation: {err_str}"
        );
    }

    #[test]
    fn scalar_conversion_error_includes_axis_context() {
        // This test verifies the error structure for scalar conversion failures.
        // In practice, conversion failures are rare with f64/f32, but the error
        // variant exists for completeness and custom scalar types.
        let err = GlobalTopologyModelError::ScalarConversion {
            axis: 2,
            value: 1.5e308_f64,
        };

        let err_str = err.to_string();
        assert!(
            err_str.contains("axis 2"),
            "Error should mention axis: {err_str}"
        );
        assert!(
            err_str.contains("1.5e308") || err_str.contains("value"),
            "Error should include problematic value: {err_str}"
        );
    }

    // =========================================================================
    // Trait bounds and scalar type tests
    // =========================================================================

    #[test]
    fn toroidal_model_works_with_different_float_types() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);

        // Test with f32
        let mut coords_f32 = [2.5_f32, -1.0_f32];
        model.canonicalize_point_in_place(&mut coords_f32).unwrap();
        assert!((coords_f32[0] - 0.5).abs() < 1e-6);
        assert!((coords_f32[1] - 2.0).abs() < 1e-6);

        // Test with f64
        let mut coords_f64 = [2.5_f64, -1.0_f64];
        model.canonicalize_point_in_place(&mut coords_f64).unwrap();
        assert_relative_eq!(coords_f64[0], 0.5);
        assert_relative_eq!(coords_f64[1], 2.0);
    }

    #[test]
    fn euclidean_model_works_with_different_float_types() {
        let model = EuclideanModel;

        // Test with f32
        let coords_f32 = [1.5_f32, 2.5_f32];
        let lifted = model.lift_for_orientation(coords_f32, None).unwrap();
        assert!((lifted[0] - 1.5).abs() < 1e-6);
        assert!((lifted[1] - 2.5).abs() < 1e-6);

        // Test with f64
        let coords_f64 = [1.5_f64, 2.5_f64];
        let lifted = model.lift_for_orientation(coords_f64, None).unwrap();
        assert_relative_eq!(lifted[0], 1.5);
        assert_relative_eq!(lifted[1], 2.5);
    }

    #[test]
    fn toroidal_model_lift_works_with_different_float_types() {
        let model = ToroidalModel::<2>::new([2.0, 3.0], ToroidalConstructionMode::Canonicalized);

        // Test with f32
        let lifted_f32 = model
            .lift_for_orientation([0.5_f32, 0.25_f32], Some([1, -1]))
            .unwrap();
        assert!((lifted_f32[0] - 2.5).abs() < 1e-6);
        assert!((lifted_f32[1] - (-2.75)).abs() < 1e-6);

        // Test with f64
        let lifted_f64 = model
            .lift_for_orientation([0.5_f64, 0.25_f64], Some([1, -1]))
            .unwrap();
        assert_relative_eq!(lifted_f64[0], 2.5);
        assert_relative_eq!(lifted_f64[1], -2.75);
    }
}

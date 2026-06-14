//! Core trait for topological spaces and related error types.
//!
//! This module defines the fundamental abstraction for different topological
//! spaces (planar, spherical, toroidal) that triangulations can inhabit.
//! `GlobalTopology` metadata from this module is used by triangulation/build paths.
//! Topology-specific behavior is delegated through the internal
//! `global_topology_model` adapter layer.

use crate::core::{facet::FacetError, tds::TdsError};
use thiserror::Error;

/// Errors that can occur during topology computation or validation.
///
/// These errors arise from simplex counting, classification, or
/// Euler characteristic validation failures.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::spaces::TopologyError;
/// use delaunay::prelude::tds::TdsError;
///
/// let error = TopologyError::FacetMapBuild {
///     source: TdsError::InconsistentDataStructure {
///         message: "facet map invariant failed".to_string(),
///     },
/// };
/// std::assert_matches!(error, TopologyError::FacetMapBuild { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TopologyError {
    /// Failed to build the facet-to-simplices incidence map.
    #[error("Failed to build facet incidence map during topology analysis: {source}")]
    FacetMapBuild {
        /// Underlying TDS failure.
        #[source]
        source: TdsError,
    },

    /// Failed to enumerate boundary facets.
    #[error("Failed to enumerate boundary facets during topology analysis: {source}")]
    BoundaryFacetEnumeration {
        /// Underlying TDS failure.
        #[source]
        source: TdsError,
    },

    /// Failed to access the simplex for a boundary facet.
    #[error("Failed to access boundary facet simplex during topology analysis: {source}")]
    BoundaryFacetSimplexAccess {
        /// Underlying facet failure.
        #[source]
        source: FacetError,
    },

    /// Failed to count boundary facets while classifying topology.
    #[error("Failed to count boundary facets during topology classification: {source}")]
    BoundaryFacetCount {
        /// Underlying TDS failure.
        #[source]
        source: TdsError,
    },

    /// Euler characteristic does not match expected value.
    ///
    /// NOTE: Currently unused - validation returns `TopologyCheckResult` with
    /// structured diagnostics instead. This variant is reserved for future use
    /// when more structured error reporting is needed at the error boundary.
    #[error(
        "Euler characteristic mismatch: computed χ={computed}, expected χ={expected} for {topology_type}"
    )]
    EulerMismatch {
        /// The computed Euler characteristic.
        computed: isize,
        /// The expected Euler characteristic.
        expected: isize,
        /// Human-readable topology type description.
        topology_type: String,
    },
}

/// Classification of topological spaces for triangulations.
///
/// This enum categorizes the fundamental geometry of the space in which
/// a triangulation is embedded. Different topologies have different
/// properties regarding boundary conditions and geometric constraints.
///
/// # Future Use
///
/// This is currently unused but provides the foundation for future support
/// of non-Euclidean triangulations (spherical, toroidal, hyperbolic).
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::spaces::TopologyKind;
///
/// let kind = TopologyKind::Euclidean;
/// assert_eq!(format!("{:?}", kind), "Euclidean");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyKind {
    /// Euclidean (flat) space with standard distance metric.
    ///
    /// This is the default for most triangulations. Allows boundary facets
    /// (convex hull) and has no periodic wrapping.
    Euclidean,

    /// Toroidal space with periodic boundary conditions.
    ///
    /// Points wrap around at domain boundaries. No true boundary facets exist
    /// as opposite edges are identified.
    Toroidal,

    /// Spherical space embedded on the surface of a sphere.
    ///
    /// All points lie on a sphere surface. No boundary facets as the space
    /// is closed and compact.
    Spherical,

    /// Hyperbolic space with negative curvature.
    ///
    /// Non-Euclidean geometry where parallel lines diverge. Distance and
    /// angle calculations differ from Euclidean space.
    Hyperbolic,
}

/// Construction mode metadata for toroidal triangulations.
///
/// This distinguishes between:
/// - canonicalized builds (`.try_canonicalized_toroidal(...)`) and
/// - true periodic quotient builds (`.try_toroidal(...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalConstructionMode {
    /// Canonicalized toroidal mode: coordinates are wrapped into the fundamental domain
    /// before Euclidean triangulation construction.
    Canonicalized,
    /// Periodic toroidal mode: 3^D image-point construction with periodic quotient
    /// neighbor rewiring.
    PeriodicImagePoint,
    /// Explicit simplex construction: the caller provided combinatorial connectivity
    /// directly and declared toroidal topology metadata for validation purposes.
    ///
    /// No coordinate canonicalization or image-point expansion is performed.
    Explicit,
}

/// Errors that can occur while parsing a toroidal fundamental domain.
///
/// Toroidal domains require every period to be finite and strictly positive.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::spaces::{ToroidalDomain, ToroidalDomainError};
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
/// positive, so stored topology metadata cannot represent invalid domains.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::spaces::ToroidalDomain;
///
/// # fn main() -> Result<(), delaunay::prelude::topology::spaces::ToroidalDomainError> {
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
    /// use delaunay::prelude::topology::spaces::{ToroidalDomain, ToroidalDomainError};
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
    /// use delaunay::prelude::topology::spaces::ToroidalDomain;
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
    /// use delaunay::prelude::topology::spaces::ToroidalDomain;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::topology::spaces::ToroidalDomainError> {
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
    /// use delaunay::prelude::topology::spaces::ToroidalDomain;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::topology::spaces::ToroidalDomainError> {
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
    /// use delaunay::prelude::topology::spaces::ToroidalDomain;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::topology::spaces::ToroidalDomainError> {
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

/// Runtime metadata describing the global topological space associated with a triangulation.
///
/// This enum is stored on triangulations so callers can query whether a result was
/// constructed in Euclidean or toroidal mode after construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GlobalTopology<const D: usize> {
    /// Euclidean (flat) space.
    Euclidean,
    /// Toroidal (periodic) space with explicit domain and construction mode.
    Toroidal {
        /// Validated fundamental-domain periods `[L_0, ..., L_{D-1}]`.
        domain: ToroidalDomain<D>,
        /// How the toroidal triangulation was constructed.
        mode: ToroidalConstructionMode,
    },
    /// Spherical space.
    Spherical,
    /// Hyperbolic space.
    Hyperbolic,
}

impl<const D: usize> Default for GlobalTopology<D> {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl<const D: usize> GlobalTopology<D> {
    /// Default global-topology metadata for triangulations.
    pub const DEFAULT: Self = Self::Euclidean;

    /// Creates toroidal global-topology metadata from raw domain periods.
    ///
    /// # Errors
    ///
    /// Returns [`ToroidalDomainError::InvalidPeriod`] when any period is
    /// non-finite, zero, or negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::{
    ///     GlobalTopology, ToroidalConstructionMode, ToroidalDomainError,
    /// };
    ///
    /// # fn main() -> Result<(), ToroidalDomainError> {
    /// let topology = GlobalTopology::<2>::try_toroidal(
    ///     [1.0, 2.0],
    ///     ToroidalConstructionMode::PeriodicImagePoint,
    /// )?;
    ///
    /// assert!(topology.is_toroidal());
    /// assert!(topology.is_periodic());
    /// assert!(!topology.allows_boundary());
    ///
    /// std::assert_matches!(
    ///     GlobalTopology::<2>::try_toroidal(
    ///         [1.0, 0.0],
    ///         ToroidalConstructionMode::PeriodicImagePoint,
    ///     ),
    ///     Err(ToroidalDomainError::InvalidPeriod { axis: 1, period })
    ///         if period.abs() < f64::EPSILON
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_toroidal(
        domain: [f64; D],
        mode: ToroidalConstructionMode,
    ) -> Result<Self, ToroidalDomainError> {
        Ok(Self::Toroidal {
            domain: ToroidalDomain::try_new(domain)?,
            mode,
        })
    }

    /// Returns the corresponding high-level topology kind.
    #[must_use]
    pub const fn kind(self) -> TopologyKind {
        match self {
            Self::Euclidean => TopologyKind::Euclidean,
            Self::Toroidal { .. } => TopologyKind::Toroidal,
            Self::Spherical => TopologyKind::Spherical,
            Self::Hyperbolic => TopologyKind::Hyperbolic,
        }
    }

    /// Returns whether boundary facets are allowed for this global topology.
    #[must_use]
    pub const fn allows_boundary(self) -> bool {
        match self {
            Self::Euclidean => true,
            Self::Toroidal { .. } | Self::Spherical | Self::Hyperbolic => false,
        }
    }

    /// Returns `true` for Euclidean global topology metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::GlobalTopology;
    ///
    /// let topo = GlobalTopology::<3>::Euclidean;
    /// assert!(topo.is_euclidean());
    /// assert!(!topo.is_toroidal());
    /// ```
    #[must_use]
    pub const fn is_euclidean(self) -> bool {
        matches!(self, Self::Euclidean)
    }

    /// Returns `true` for toroidal global topology metadata.
    #[must_use]
    pub const fn is_toroidal(self) -> bool {
        matches!(self, Self::Toroidal { .. })
    }

    /// Returns `true` when this represents a true periodic image-point toroidal build.
    #[must_use]
    pub const fn is_periodic(self) -> bool {
        matches!(
            self,
            Self::Toroidal {
                mode: ToroidalConstructionMode::PeriodicImagePoint,
                ..
            }
        )
    }
}

/// Trait for topological spaces that triangulations can inhabit.
///
/// This trait abstracts over different geometric spaces (Euclidean, spherical,
/// toroidal, hyperbolic) to enable topology-aware triangulation algorithms.
///
/// The dimension is specified via the associated constant `DIM`, which must
/// match the dimension of the associated `Tds<U, V, D>`. This ensures
/// type safety and prevents dimension mismatches.
///
/// # Future Use
///
/// This is currently unused but provides the interface for future support
/// of non-Euclidean triangulations. Implementations will handle topology-specific
/// operations like point canonicalization and boundary conditions.
///
/// When implemented, the topological space will be stored in
/// `Triangulation<K, U, V, D>` and its `DIM` must equal `D`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::spaces::{TopologicalSpace, TopologyKind};
///
/// // Future: EuclideanSpace will implement this trait
/// // let space = EuclideanSpace::<3>::new();
/// // assert_eq!(EuclideanSpace::<3>::DIM, 3);
/// // assert_eq!(space.kind(), TopologyKind::Euclidean);
/// // assert!(space.allows_boundary());
/// ```
pub trait TopologicalSpace {
    /// The dimension of this topological space.
    ///
    /// This must match the dimension `D` of the associated triangulation
    /// `Tds<U, V, D>` to ensure geometric consistency.
    const DIM: usize;

    /// Returns the kind of topological space.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::{TopologicalSpace, TopologyKind};
    ///
    /// struct DummySpace;
    ///
    /// impl TopologicalSpace for DummySpace {
    ///     const DIM: usize = 3;
    ///
    ///     fn kind(&self) -> TopologyKind {
    ///         TopologyKind::Euclidean
    ///     }
    ///
    ///     fn allows_boundary(&self) -> bool {
    ///         true
    ///     }
    ///
    ///     fn canonicalize_point(&self, _coords: &mut [f64]) {}
    ///
    ///     fn fundamental_domain(&self) -> Option<&[f64]> {
    ///         None
    ///     }
    /// }
    ///
    /// let space = DummySpace;
    /// assert_eq!(space.kind(), TopologyKind::Euclidean);
    /// ```
    fn kind(&self) -> TopologyKind;

    /// Returns whether this topology allows boundary facets.
    ///
    /// # Returns
    ///
    /// - `true` for Euclidean spaces (convex hull boundary allowed)
    /// - `false` for closed manifolds like spherical or toroidal spaces
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::{TopologicalSpace, TopologyKind};
    ///
    /// struct DummySpace {
    ///     allows: bool,
    /// }
    ///
    /// impl TopologicalSpace for DummySpace {
    ///     const DIM: usize = 2;
    ///
    ///     fn kind(&self) -> TopologyKind {
    ///         if self.allows {
    ///             TopologyKind::Euclidean
    ///         } else {
    ///             TopologyKind::Toroidal
    ///         }
    ///     }
    ///
    ///     fn allows_boundary(&self) -> bool {
    ///         self.allows
    ///     }
    ///
    ///     fn canonicalize_point(&self, _coords: &mut [f64]) {}
    ///
    ///     fn fundamental_domain(&self) -> Option<&[f64]> {
    ///         None
    ///     }
    /// }
    ///
    /// let euclidean = DummySpace { allows: true };
    /// let toroidal = DummySpace { allows: false };
    /// assert!(euclidean.allows_boundary());
    /// assert!(!toroidal.allows_boundary());
    /// ```
    fn allows_boundary(&self) -> bool;

    /// Canonicalizes a point to conform to the topology's constraints.
    ///
    /// Different topologies have different canonicalization rules:
    /// - **Euclidean**: No modification (identity operation)
    /// - **Toroidal**: Wraps coordinates into fundamental domain `[0, L)`
    /// - **Spherical**: Currently a scaffolded identity operation; unit-sphere
    ///   projection is tracked separately
    /// - **Hyperbolic**: Currently a scaffolded identity operation
    ///
    /// The coordinate slice length must match `Self::DIM`.
    ///
    /// # Arguments
    ///
    /// * `coords` - Mutable slice of point coordinates to canonicalize
    ///
    /// # Panics
    ///
    /// May panic if `coords.len() != Self::DIM` (implementation-defined).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::{TopologicalSpace, TopologyKind};
    ///
    /// struct ToroidalSpace {
    ///     domain: [f64; 2],
    /// }
    ///
    /// impl TopologicalSpace for ToroidalSpace {
    ///     const DIM: usize = 2;
    ///
    ///     fn kind(&self) -> TopologyKind {
    ///         TopologyKind::Toroidal
    ///     }
    ///
    ///     fn allows_boundary(&self) -> bool {
    ///         false
    ///     }
    ///
    ///     fn canonicalize_point(&self, coords: &mut [f64]) {
    ///         for (coord, domain) in coords.iter_mut().zip(self.domain) {
    ///             *coord = coord.rem_euclid(domain);
    ///         }
    ///     }
    ///
    ///     fn fundamental_domain(&self) -> Option<&[f64]> {
    ///         Some(&self.domain)
    ///     }
    /// }
    ///
    /// let space = ToroidalSpace { domain: [1.0, 1.0] };
    /// let mut point = [1.5, -0.3];
    /// space.canonicalize_point(&mut point);
    /// assert_eq!(point, [0.5, 0.7]); // Wrapped into [0, 1)
    /// ```
    fn canonicalize_point(&self, coords: &mut [f64]);

    /// Returns the fundamental domain for periodic topologies.
    ///
    /// For periodic spaces (toroidal), this returns a slice view of the fundamental
    /// domain. For non-periodic spaces, returns `None`.
    ///
    /// The returned slice length equals `Self::DIM`.
    ///
    /// # Returns
    ///
    /// - `Some(&[L₀, L₁, ..., L_D])` for periodic spaces (domain size per dimension)
    /// - `None` for non-periodic spaces (Euclidean, spherical, hyperbolic)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::{TopologicalSpace, TopologyKind};
    ///
    /// struct DummySpace {
    ///     domain: Option<[f64; 2]>,
    /// }
    ///
    /// impl TopologicalSpace for DummySpace {
    ///     const DIM: usize = 2;
    ///
    ///     fn kind(&self) -> TopologyKind {
    ///         if self.domain.is_some() {
    ///             TopologyKind::Toroidal
    ///         } else {
    ///             TopologyKind::Euclidean
    ///         }
    ///     }
    ///
    ///     fn allows_boundary(&self) -> bool {
    ///         self.domain.is_none()
    ///     }
    ///
    ///     fn canonicalize_point(&self, _coords: &mut [f64]) {}
    ///
    ///     fn fundamental_domain(&self) -> Option<&[f64]> {
    ///         self.domain.as_ref().map(|domain| &domain[..])
    ///     }
    /// }
    ///
    /// let toroidal = DummySpace {
    ///     domain: Some([2.0, 3.0]),
    /// };
    /// assert_eq!(toroidal.fundamental_domain(), Some(&[2.0, 3.0][..]));
    ///
    /// let euclidean = DummySpace { domain: None };
    /// assert_eq!(euclidean.fundamental_domain(), None);
    /// ```
    fn fundamental_domain(&self) -> Option<&[f64]>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::assert_matches;

    #[test]
    fn test_topology_error_display() {
        let counting = TopologyError::FacetMapBuild {
            source: TdsError::InconsistentDataStructure {
                message: "test message".to_string(),
            },
        };
        assert_eq!(
            counting.to_string(),
            "Failed to build facet incidence map during topology analysis: Internal data structure inconsistency: test message"
        );

        let classification = TopologyError::BoundaryFacetCount {
            source: TdsError::InconsistentDataStructure {
                message: "another test".to_string(),
            },
        };
        assert_eq!(
            classification.to_string(),
            "Failed to count boundary facets during topology classification: Internal data structure inconsistency: another test"
        );

        let euler = TopologyError::EulerMismatch {
            computed: 2,
            expected: 1,
            topology_type: "sphere".to_string(),
        };
        assert_eq!(
            euler.to_string(),
            "Euler characteristic mismatch: computed χ=2, expected χ=1 for sphere"
        );
    }

    #[test]
    fn test_topology_error_equality() {
        let err1 = TopologyError::FacetMapBuild {
            source: TdsError::InconsistentDataStructure {
                message: "msg".to_string(),
            },
        };
        let err2 = TopologyError::FacetMapBuild {
            source: TdsError::InconsistentDataStructure {
                message: "msg".to_string(),
            },
        };
        let err3 = TopologyError::FacetMapBuild {
            source: TdsError::InconsistentDataStructure {
                message: "different".to_string(),
            },
        };

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
        assert_ne!(
            err1,
            TopologyError::BoundaryFacetCount {
                source: TdsError::InconsistentDataStructure {
                    message: "msg".to_string(),
                },
            }
        );
    }

    #[test]
    fn test_topology_kind_debug() {
        assert_eq!(format!("{:?}", TopologyKind::Euclidean), "Euclidean");
        assert_eq!(format!("{:?}", TopologyKind::Toroidal), "Toroidal");
        assert_eq!(format!("{:?}", TopologyKind::Spherical), "Spherical");
        assert_eq!(format!("{:?}", TopologyKind::Hyperbolic), "Hyperbolic");
    }

    #[test]
    fn test_toroidal_construction_mode_debug() {
        assert_eq!(
            format!("{:?}", ToroidalConstructionMode::Canonicalized),
            "Canonicalized"
        );
        assert_eq!(
            format!("{:?}", ToroidalConstructionMode::PeriodicImagePoint),
            "PeriodicImagePoint"
        );
        assert_eq!(
            format!("{:?}", ToroidalConstructionMode::Explicit),
            "Explicit"
        );
    }

    #[test]
    fn test_global_topology_default() {
        let default_topo: GlobalTopology<3> = GlobalTopology::default();
        assert_eq!(default_topo, GlobalTopology::Euclidean);
        assert_eq!(GlobalTopology::<3>::DEFAULT, GlobalTopology::Euclidean);
    }

    #[test]
    fn test_global_topology_kind() {
        assert_eq!(
            GlobalTopology::<2>::Euclidean.kind(),
            TopologyKind::Euclidean
        );
        assert_eq!(
            GlobalTopology::<3>::Spherical.kind(),
            TopologyKind::Spherical
        );
        assert_eq!(
            GlobalTopology::<4>::Hyperbolic.kind(),
            TopologyKind::Hyperbolic
        );

        let toroidal = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.0, 2.0]).unwrap(),
            mode: ToroidalConstructionMode::Canonicalized,
        };
        assert_eq!(toroidal.kind(), TopologyKind::Toroidal);
    }

    #[test]
    fn test_global_topology_allows_boundary() {
        assert!(GlobalTopology::<3>::Euclidean.allows_boundary());
        assert!(!GlobalTopology::<3>::Spherical.allows_boundary());
        assert!(!GlobalTopology::<3>::Hyperbolic.allows_boundary());

        let toroidal = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.0, 1.0]).unwrap(),
            mode: ToroidalConstructionMode::Canonicalized,
        };
        assert!(!toroidal.allows_boundary());
    }

    #[test]
    fn test_global_topology_is_euclidean() {
        assert!(GlobalTopology::<3>::Euclidean.is_euclidean());
        assert!(!GlobalTopology::<3>::Spherical.is_euclidean());
        assert!(!GlobalTopology::<3>::Hyperbolic.is_euclidean());

        let toroidal = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.0, 1.0]).unwrap(),
            mode: ToroidalConstructionMode::Canonicalized,
        };
        assert!(!toroidal.is_euclidean());
    }

    #[test]
    fn test_global_topology_is_toroidal() {
        assert!(!GlobalTopology::<3>::Euclidean.is_toroidal());
        assert!(!GlobalTopology::<3>::Spherical.is_toroidal());
        assert!(!GlobalTopology::<3>::Hyperbolic.is_toroidal());

        let toroidal = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.0, 1.0]).unwrap(),
            mode: ToroidalConstructionMode::Canonicalized,
        };
        assert!(toroidal.is_toroidal());
    }

    #[test]
    fn test_global_topology_is_periodic() {
        assert!(!GlobalTopology::<3>::Euclidean.is_periodic());
        assert!(!GlobalTopology::<3>::Spherical.is_periodic());
        assert!(!GlobalTopology::<3>::Hyperbolic.is_periodic());

        // Test different toroidal modes
        let canonicalized = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.0, 1.0]).unwrap(),
            mode: ToroidalConstructionMode::Canonicalized,
        };
        assert!(!canonicalized.is_periodic());

        let periodic = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.0, 1.0]).unwrap(),
            mode: ToroidalConstructionMode::PeriodicImagePoint,
        };
        assert!(periodic.is_periodic());

        let explicit = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.0, 1.0]).unwrap(),
            mode: ToroidalConstructionMode::Explicit,
        };
        assert!(!explicit.is_periodic());
    }

    #[test]
    fn test_global_topology_equality() {
        let topo1 = GlobalTopology::<3>::Euclidean;
        let topo2 = GlobalTopology::<3>::Euclidean;
        let topo3 = GlobalTopology::<3>::Spherical;

        assert_eq!(topo1, topo2);
        assert_ne!(topo1, topo3);

        let toroidal1 = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.0, 2.0]).unwrap(),
            mode: ToroidalConstructionMode::Canonicalized,
        };
        let toroidal2 = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.0, 2.0]).unwrap(),
            mode: ToroidalConstructionMode::Canonicalized,
        };
        let toroidal3 = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.0, 2.0]).unwrap(),
            mode: ToroidalConstructionMode::PeriodicImagePoint,
        };

        assert_eq!(toroidal1, toroidal2);
        assert_ne!(toroidal1, toroidal3);
    }

    #[test]
    fn test_global_topology_debug() {
        assert_eq!(format!("{:?}", GlobalTopology::<3>::Euclidean), "Euclidean");

        let toroidal = GlobalTopology::<2>::Toroidal {
            domain: ToroidalDomain::try_new([1.5, 2.5]).unwrap(),
            mode: ToroidalConstructionMode::Canonicalized,
        };
        let debug_str = format!("{toroidal:?}");
        assert!(debug_str.contains("Toroidal"));
        assert!(debug_str.contains("domain"));
        assert!(debug_str.contains("mode"));
    }

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
        let domain = ToroidalDomain::<3>::try_from([1.0, 2.0, 4.0]).unwrap();
        assert_relative_eq!(domain.periods()[0], 1.0);
        assert_relative_eq!(domain.periods()[1], 2.0);
        assert_relative_eq!(domain.periods()[2], 4.0);

        let periods = domain.into_periods();
        assert_relative_eq!(periods[0], 1.0);
        assert_relative_eq!(periods[1], 2.0);
        assert_relative_eq!(periods[2], 4.0);

        let invalid = ToroidalDomain::<3>::try_from([1.0, f64::NEG_INFINITY, 4.0]).unwrap_err();
        assert_matches!(
            invalid,
            ToroidalDomainError::InvalidPeriod { axis: 1, period }
                if period.is_infinite() && period.is_sign_negative()
        );
    }

    #[test]
    fn test_global_topology_try_toroidal_parses_domain() {
        let topology =
            GlobalTopology::try_toroidal([1.0, 2.0], ToroidalConstructionMode::Canonicalized)
                .unwrap();
        assert_eq!(topology.kind(), TopologyKind::Toroidal);
        assert!(!topology.is_periodic());

        let err =
            GlobalTopology::<2>::try_toroidal([0.0, 2.0], ToroidalConstructionMode::Canonicalized)
                .unwrap_err();
        assert_matches!(
            err,
            ToroidalDomainError::InvalidPeriod { axis: 0, period }
                if period.abs() < f64::EPSILON
        );
    }
}

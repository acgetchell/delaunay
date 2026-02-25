//! Core trait for topological spaces and related error types.
//!
//! This module defines the fundamental abstraction for different topological
//! spaces (planar, spherical, toroidal) that triangulations can inhabit.
//! NOTE: As of now, Triangulation does NOT use this. It is future plumbing
//! for topological spaces (spherical, toroidal, and possibly hyperbolic).

use thiserror::Error;

/// Errors that can occur during topology computation or validation.
///
/// These errors arise from simplex counting, classification, or
/// Euler characteristic validation failures.
///
/// # Examples
///
/// ```rust
/// use delaunay::topology::traits::topological_space::TopologyError;
///
/// let error = TopologyError::Counting("Failed to enumerate edges".to_string());
/// assert_eq!(error.to_string(), "Failed to count simplices: Failed to enumerate edges");
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TopologyError {
    /// Failed to count simplices during topology analysis.
    #[error("Failed to count simplices: {0}")]
    Counting(String),

    /// Failed to classify the triangulation's topology.
    #[error("Failed to classify triangulation: {0}")]
    Classification(String),

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
/// use delaunay::topology::traits::topological_space::TopologyKind;
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
/// - Phase 1 canonicalized builds (`.toroidal(...)`) and
/// - Phase 2 true periodic quotient builds (`.toroidal_periodic(...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToroidalConstructionMode {
    /// Phase 1 toroidal mode: coordinates are wrapped into the fundamental domain
    /// before Euclidean triangulation construction.
    Canonicalized,
    /// Phase 2 toroidal mode: 3^D image-point construction with periodic quotient
    /// neighbor rewiring.
    PeriodicImagePoint,
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
        /// Fundamental domain periods `[L_0, ..., L_{D-1}]`.
        domain: [f64; D],
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
/// match the dimension of the associated `Tds<T, U, V, D>`. This ensures
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
/// use delaunay::topology::traits::topological_space::{TopologicalSpace, TopologyKind};
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
    /// `Tds<T, U, V, D>` to ensure geometric consistency.
    const DIM: usize;

    /// Returns the kind of topological space.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::topology::traits::topological_space::{TopologicalSpace, TopologyKind};
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
    /// use delaunay::topology::traits::topological_space::{TopologicalSpace, TopologyKind};
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
    /// - **Spherical**: Projects onto unit sphere surface
    /// - **Hyperbolic**: Projects into valid hyperbolic space region
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
    /// use delaunay::topology::traits::topological_space::{TopologicalSpace, TopologyKind};
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
    ///
    /// TODO: When implementing full topology support, consider making this generic:
    /// `fn canonicalize_point<T: CoordinateScalar>(&self, coords: &mut [T])`
    /// to match the triangulation's scalar type instead of hardcoding `f64`.
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
    /// use delaunay::topology::traits::topological_space::{TopologicalSpace, TopologyKind};
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
    ///
    /// TODO: When implementing full topology support, consider making this generic:
    /// `fn fundamental_domain<T: CoordinateScalar>(&self) -> Option<&[T]>`
    /// to match the triangulation's scalar type instead of hardcoding `f64`.
    fn fundamental_domain(&self) -> Option<&[f64]>;
}

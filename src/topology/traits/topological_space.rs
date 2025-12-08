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
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TopologyError {
    /// Failed to count simplices during topology analysis.
    #[error("Failed to count simplices: {0}")]
    Counting(String),

    /// Failed to classify the triangulation's topology.
    #[error("Failed to classify triangulation: {0}")]
    Classification(String),

    /// Euler characteristic does not match expected value.
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

/// Trait for topological spaces that triangulations can inhabit.
///
/// This trait abstracts over different geometric spaces (Euclidean, spherical,
/// toroidal, hyperbolic) to enable topology-aware triangulation algorithms.
///
/// # Future Use
///
/// This is currently unused but provides the interface for future support
/// of non-Euclidean triangulations. Implementations will handle topology-specific
/// operations like point canonicalization and boundary conditions.
///
/// # Examples
///
/// ```rust
/// use delaunay::topology::traits::topological_space::{TopologicalSpace, TopologyKind};
///
/// // Future: EuclideanSpace will implement this trait
/// // let space = EuclideanSpace::new();
/// // assert_eq!(space.kind(), TopologyKind::Euclidean);
/// // assert!(space.allows_boundary());
/// ```
pub trait TopologicalSpace {
    /// Returns the kind of topological space.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Future usage when trait is implemented
    /// let space = EuclideanSpace::new();
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
    /// ```rust,ignore
    /// // Future usage
    /// let euclidean = EuclideanSpace::new();
    /// assert!(euclidean.allows_boundary());
    ///
    /// let toroidal = ToroidalSpace::new([1.0, 1.0]);
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
    /// # Arguments
    ///
    /// * `coords` - Mutable reference to point coordinates to canonicalize
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Future usage for toroidal space
    /// let space = ToroidalSpace::new([1.0, 1.0]);
    /// let mut point = [1.5, -0.3];
    /// space.canonicalize_point(&mut point);
    /// assert_eq!(point, [0.5, 0.7]); // Wrapped into [0, 1)
    /// ```
    fn canonicalize_point<const D: usize>(&self, coords: &mut [f64; D]);

    /// Returns the fundamental domain for periodic topologies.
    ///
    /// For periodic spaces (toroidal), this returns the size of the fundamental
    /// domain in each dimension. For non-periodic spaces, returns `None`.
    ///
    /// # Returns
    ///
    /// - `Some([L₀, L₁, ..., L_D])` for periodic spaces (domain size per dimension)
    /// - `None` for non-periodic spaces (Euclidean, spherical, hyperbolic)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Future usage
    /// let toroidal = ToroidalSpace::new([2.0, 3.0]);
    /// assert_eq!(toroidal.fundamental_domain(), Some([2.0, 3.0]));
    ///
    /// let euclidean = EuclideanSpace::new();
    /// assert_eq!(euclidean.fundamental_domain(), None);
    /// ```
    fn fundamental_domain<const D: usize>(&self) -> Option<[f64; D]>;
}

//! Random triangulation generation functions.
//!
//! This module provides utilities for generating random Delaunay triangulations
//! with various topology guarantees.

#![forbid(unsafe_code)]

use super::point_generation::{
    RandomPointGenerationError, generate_random_points_in_range,
    generate_random_points_in_range_seeded,
};
use crate::construction::{
    ConstructionOptions, DelaunayConstructionFailure, DelaunayTriangulationConstructionError,
    InsertionOrderStrategy, RetryPolicy,
};
use crate::core::construction::TriangulationConstructionError;
use crate::core::simplex::SimplexValidationError;
use crate::core::traits::data_type::DataType;
use crate::core::validation::TopologyGuarantee;
use crate::core::vertex::Vertex;
use crate::geometry::coordinate_range::{CoordinateRange, CoordinateRangeError};
use crate::geometry::kernel::{AdaptiveKernel, Kernel};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{CoordinateScalar, FiniteCheck};
use crate::triangulation::DelaunayTriangulation;
use rand::SeedableRng;
use rand::distr::uniform::SampleUniform;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::{fmt::Debug, num::NonZeroUsize};

const RANDOM_TRIANGULATION_MAX_SHUFFLE_ATTEMPTS: usize = 6;
const RANDOM_TRIANGULATION_MAX_POINTSET_ATTEMPTS: usize = 6;
const RANDOM_TRIANGULATION_POINTSET_SEED_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Wraps random point generation failures in the construction error hierarchy.
///
/// Keeping this mapping centralized preserves the public contract that invalid
/// random-generation parameters remain distinguishable from geometric
/// degeneracy during triangulation construction.
const fn random_point_generation_error(
    source: RandomPointGenerationError<f64>,
) -> DelaunayTriangulationConstructionError {
    DelaunayTriangulationConstructionError::Triangulation(
        DelaunayConstructionFailure::RandomPointGeneration { source },
    )
}

/// Generates random points through the seeded or unseeded public generator path.
///
/// This helper keeps all random triangulation entry points on the same typed
/// error boundary: generation failures become
/// [`DelaunayConstructionFailure::RandomPointGeneration`] before construction
/// or validation can reinterpret them as geometric failures.
fn random_points_with_seed<T, const D: usize>(
    n_points: usize,
    bounds: CoordinateRange<T>,
    seed: Option<u64>,
) -> Vec<Point<T, D>>
where
    T: CoordinateScalar + SampleUniform,
{
    #[expect(
        clippy::option_if_let_else,
        reason = "explicit match keeps seeded and unseeded generator paths readable"
    )]
    match seed {
        Some(seed_value) => generate_random_points_in_range_seeded(n_points, bounds, seed_value),
        None => generate_random_points_in_range(n_points, bounds),
    }
}

/// Runs the topology validation required before a random triangulation is returned.
///
/// Random generation may retry with different point sets and shuffles, so this
/// helper keeps the acceptance criterion identical for every attempted
/// construction path.
fn validate_random_triangulation<K, U, V, const D: usize>(
    dt: DelaunayTriangulation<K, U, V, D>,
) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationConstructionError>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    dt.as_triangulation().validate().map_err(|e| {
        TriangulationConstructionError::FinalTopologyValidation {
            message: "random triangulation failed final Levels 1-3 topology validation".to_string(),
            source: e.into(),
        }
    })?;
    Ok(dt)
}

/// Checks whether a successfully validated random triangulation retained enough vertices.
///
/// Batch construction may skip duplicate or retry-exhausted degenerate inputs;
/// this predicate prevents severely depleted random point sets from being
/// accepted as successful fixtures.
fn random_triangulation_is_acceptable<K, U, V, const D: usize>(
    dt: &DelaunayTriangulation<K, U, V, D>,
    min_vertices: usize,
) -> bool
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    dt.number_of_vertices() >= min_vertices
}

/// Attempts one deterministic random-triangulation build from a fixed vertex order.
///
/// This isolates the single-attempt contract used by the shuffled fallback
/// loop: build with input order, validate topology, then enforce the minimum
/// retained-vertex threshold.
fn random_triangulation_try_build<K, T, U, V, const D: usize>(
    kernel: &K,
    vertices: &[Vertex<T, U, D>],
    min_vertices: usize,
    topology_guarantee: TopologyGuarantee,
) -> Result<Option<DelaunayTriangulation<K, U, V, D>>, DelaunayTriangulationConstructionError>
where
    K: Kernel<D, Scalar = T>,
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Important: use `Input` insertion order here so that the caller can apply
    // deterministic shuffles during the robust fallback loop.
    let options = ConstructionOptions::default()
        .with_insertion_order(InsertionOrderStrategy::Input)
        .with_retry_policy(RetryPolicy::Disabled);

    let dt = DelaunayTriangulation::with_topology_guarantee_and_options(
        kernel,
        vertices,
        topology_guarantee,
        options,
    )?;
    let dt = validate_random_triangulation(dt)?;

    Ok(random_triangulation_is_acceptable(&dt, min_vertices).then_some(dt))
}

/// Converts generated points into vertices while preserving optional shared vertex data.
///
/// Vertex construction from generated, already-validated points is infallible,
/// so random-triangulation errors remain attributable to point generation,
/// construction, or validation rather than wrapper allocation.
fn random_triangulation_build_vertices<T, U, const D: usize>(
    points: Vec<Point<T, D>>,
    vertex_data: Option<U>,
) -> Vec<Vertex<T, U, D>>
where
    U: Copy,
{
    match vertex_data {
        Some(data) => points
            .into_iter()
            .map(|point| Vertex::from_point_with_data(point, data))
            .collect(),
        None => points.into_iter().map(Vertex::from_point).collect(),
    }
}

/// Creates an [`AdaptiveKernel`] for triangulation construction.
///
/// All code paths that build a non-empty triangulation should use this factory
/// so they share the same kernel type.
const fn make_adaptive_kernel<T>() -> AdaptiveKernel<T> {
    AdaptiveKernel::new()
}

/// Tries the original vertex order and deterministic shuffled fallbacks.
///
/// The public random-triangulation APIs call this after point generation so
/// all retry attempts share the same [`AdaptiveKernel`] and topology guarantee.
fn random_triangulation_try_with_vertices<T, U, V, const D: usize>(
    vertices: &[Vertex<T, U, D>],
    min_vertices: usize,
    shuffle_seed: Option<u64>,
    topology_guarantee: TopologyGuarantee,
) -> Result<
    Option<DelaunayTriangulation<AdaptiveKernel<T>, U, V, D>>,
    DelaunayTriangulationConstructionError,
>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let adaptive_kernel = make_adaptive_kernel::<T>();
    let mut last_error = None;

    match random_triangulation_try_build(
        &adaptive_kernel,
        vertices,
        min_vertices,
        topology_guarantee,
    ) {
        Ok(Some(dt)) => return Ok(Some(dt)),
        Ok(None) => {}
        Err(error) => last_error = Some(error),
    }

    for attempt in 0..RANDOM_TRIANGULATION_MAX_SHUFFLE_ATTEMPTS {
        let mut shuffled = vertices.to_vec();
        if let Some(seed_value) = shuffle_seed {
            let mix = seed_value.wrapping_add(attempt as u64 + 1);
            let mut rng = StdRng::seed_from_u64(mix);
            shuffled.shuffle(&mut rng);
        } else {
            let mut rng = rand::rng();
            shuffled.shuffle(&mut rng);
        }

        match random_triangulation_try_build(
            &adaptive_kernel,
            &shuffled,
            min_vertices,
            topology_guarantee,
        ) {
            Ok(Some(dt)) => return Ok(Some(dt)),
            Ok(None) => {}
            Err(error) => last_error = Some(error),
        }
    }

    last_error.map_or_else(|| Ok(None), Err)
}

/// This utility function combines random point generation and triangulation creation.
///
/// It generates random points using either seeded or unseeded random generation,
/// converts them to vertices, and creates a Delaunay triangulation using the
/// incremental cavity-based insertion algorithm.
/// Raw tuple bounds are parsed into a [`CoordinateRange`] before any point
/// generation or construction work begins.
///
/// This function is particularly useful for testing, benchmarking, and creating
/// triangulations for analysis or visualization purposes.
///
/// # Type Parameters
///
/// * `U` - Vertex data type (must implement `DataType`)
/// * `V` - Simplex data type (must implement `DataType`)
/// * `D` - Dimensionality (const generic parameter)
///
/// # Arguments
///
/// * `n_points` - Non-zero number of random points to generate
/// * `bounds` - `f64` coordinate bounds as `(min, max)` tuple
/// * `vertex_data` - Optional data to attach to each generated vertex
/// * `seed` - Optional seed for reproducible results. If `None`, uses thread-local RNG
///
/// # Returns
///
/// A `Result` containing either:
/// - `Ok(DelaunayTriangulation<AdaptiveKernel<f64>, U, V, D>)` - The successfully created triangulation
/// - `Err(DelaunayTriangulationConstructionError)` - An error from point generation or triangulation construction
///
/// # Errors
///
/// Returns `DelaunayTriangulationConstructionError` with different variants depending on the failure:
///
/// **Random point generation** (`RandomPointGeneration`):
/// - Coordinate range parsing fails because a bound is non-finite, equal,
///   decreasing, or incomparable
///
/// **Insufficient vertices** (`InsufficientVertices`):
/// - When `n_points < D + 1` (need at least D+1 points for a D-dimensional simplex)
/// - Example: 1-2 points in 2D, 1-3 points in 3D
///
/// **Geometric degeneracy** (`GeometricDegeneracy`):
/// - Points form a degenerate configuration (all collinear, coplanar, etc.)
/// - Numerical instability during construction
///
/// **Final validation failures** (`FinalTopologyValidation`, `FinalDelaunayValidation`):
/// - Topology/Euler validation failure after construction attempts
/// - Delaunay-property validation failure after construction attempts
///
/// **Other construction failures** (various variants):
/// - Simplex construction errors
/// - Internal construction invariants
///
/// Vertex construction from generated points is infallible; failures are
/// returned from point generation, triangulation construction, or validation.
///
/// # Examples
///
/// ```no_run
/// use delaunay::prelude::construction::DelaunayTriangulationConstructionError;
/// use delaunay::prelude::generators::generate_random_triangulation;
/// use std::num::NonZeroUsize;
///
/// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
/// # let Some(fifty) = NonZeroUsize::new(50) else {
/// #     return Ok(());
/// # };
/// # let Some(thirty) = NonZeroUsize::new(30) else {
/// #     return Ok(());
/// # };
/// # let Some(twenty) = NonZeroUsize::new(20) else {
/// #     return Ok(());
/// # };
/// // Generate a 2D triangulation with 50 points, no seed (random each time)
/// let triangulation_2d = generate_random_triangulation::<(), (), 2>(
///     fifty,
///     (-10.0, 10.0),
///     None,
///     None
/// );
///
/// // Generate a 3D triangulation with 30 points, seeded for reproducibility  
/// let triangulation_3d = generate_random_triangulation::<(), (), 3>(
///     thirty,
///     (-5.0, 5.0),
///     None,
///     Some(42)
/// );
///
/// // Generate a 4D triangulation with custom vertex data
/// let triangulation_4d = generate_random_triangulation::<i32, (), 4>(
///     twenty,
///     (0.0, 1.0),
///     Some(123),
///     Some(456)
/// );
///
/// // For string-like data, use fixed-size character arrays (Copy types)
/// let triangulation_with_strings = generate_random_triangulation::<[char; 8], (), 2>(
///     twenty,
///     (0.0, 1.0),
///     Some(['v', 'e', 'r', 't', 'e', 'x', '_', 'A']),
///     Some(789)
/// );
///
/// // Access the underlying Tds if needed
/// let dt = triangulation_3d?;
/// let vertex_count = dt.tds().number_of_vertices();
/// # Ok(())
/// # }
/// ```
///
/// # Note on String Data
///
/// Due to the `DataType` trait requiring `Copy`, `String` and `&str` cannot be used directly
/// as vertex data. For string-like data, consider using:
/// - Fixed-size character arrays: `[char; N]`
/// - Small integer types that can be mapped to strings: `u32`, `u64`
/// - Custom Copy types that wrap string-like data
///
/// # Performance Notes
///
/// - Point generation is O(n) and typically fast
/// - Triangulation construction complexity varies by dimension:
///   - 2D, 3D: O(n log n) expected with incremental insertion
///   - 4D+: O(n²) worst case, significantly slower for large point sets
/// - Consider using smaller point counts for dimensions ≥ 4
///
/// # See Also
///
/// - [`generate_random_points`](crate::geometry::util::generate_random_points) - For generating points without triangulation
/// - [`generate_random_points_seeded`](crate::geometry::util::generate_random_points_seeded) - For seeded random point generation only
/// - [`DelaunayTriangulationBuilder`](crate::DelaunayTriangulationBuilder) - For creating triangulations from existing vertices
/// - [`RandomTriangulationBuilder`] - For more control over construction options
pub fn generate_random_triangulation<U, V, const D: usize>(
    n_points: NonZeroUsize,
    bounds: (f64, f64),
    vertex_data: Option<U>,
    seed: Option<u64>,
) -> Result<
    DelaunayTriangulation<AdaptiveKernel<f64>, U, V, D>,
    DelaunayTriangulationConstructionError,
>
where
    U: DataType,
    V: DataType,
{
    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_UNUSED_IMPORTS").is_some() {
        tracing::debug!(
            n_points = n_points.get(),
            dimension = D,
            seed = ?seed,
            "triangulation_generation::generate_random_triangulation called"
        );
    }
    let bounds = CoordinateRange::try_from(bounds)
        .map_err(RandomPointGenerationError::from)
        .map_err(random_point_generation_error)?;

    generate_random_triangulation_in_range_with_topology_guarantee(
        n_points,
        bounds,
        vertex_data,
        seed,
        TopologyGuarantee::DEFAULT,
    )
}

/// This utility function combines random point generation and triangulation creation.
///
/// This variant allows selecting the [`TopologyGuarantee`] used during construction.
/// Raw tuple bounds are parsed into an f64 [`CoordinateRange`] before point
/// generation starts, matching the crate's current f64 coordinate input policy.
///
/// # Errors
///
/// Returns `Err(DelaunayTriangulationConstructionError)` if:
/// - Coordinate range parsing fails for invalid bounds.
/// - Triangulation construction fails due to geometric degeneracy, insertion errors, or the
///   requested topology guarantee not being satisfiable.
/// - Validation fails after the robust fallback attempts.
/// - Internal point-set bookkeeping is inconsistent (initial point set unexpectedly consumed).
///
/// # Examples
///
/// ```no_run
/// use delaunay::prelude::construction::DelaunayTriangulationConstructionError;
/// use delaunay::prelude::generators::generate_random_triangulation_with_topology_guarantee;
/// use delaunay::prelude::TopologyGuarantee;
/// use std::num::NonZeroUsize;
///
/// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
/// # let Some(twenty) = NonZeroUsize::new(20) else {
/// #     return Ok(());
/// # };
/// let dt = generate_random_triangulation_with_topology_guarantee::<(), (), 3>(
///     twenty,
///     (-1.0, 1.0),
///     None,
///     Some(123),
///     TopologyGuarantee::Pseudomanifold,
/// )?;
/// assert_eq!(dt.dim(), 3);
/// # Ok(())
/// # }
/// ```
pub fn generate_random_triangulation_with_topology_guarantee<U, V, const D: usize>(
    n_points: NonZeroUsize,
    bounds: (f64, f64),
    vertex_data: Option<U>,
    seed: Option<u64>,
    topology_guarantee: TopologyGuarantee,
) -> Result<
    DelaunayTriangulation<AdaptiveKernel<f64>, U, V, D>,
    DelaunayTriangulationConstructionError,
>
where
    U: DataType,
    V: DataType,
{
    let bounds = CoordinateRange::try_from(bounds)
        .map_err(RandomPointGenerationError::from)
        .map_err(random_point_generation_error)?;

    generate_random_triangulation_in_range_with_topology_guarantee(
        n_points,
        bounds,
        vertex_data,
        seed,
        topology_guarantee,
    )
}

/// Generates a random triangulation from validated coordinate bounds.
///
/// Because `bounds` is already a [`CoordinateRange`], this function does not
/// perform raw tuple range parsing and cannot fail with an invalid-bounds error.
///
/// # Arguments
///
/// * `n_points` - Non-zero number of random points to generate.
/// * `bounds` - Validated coordinate bounds shared by every coordinate axis.
/// * `vertex_data` - Optional data attached to every generated vertex.
/// * `seed` - Optional seed for reproducible point generation.
///
/// # Returns
///
/// A validated random [`DelaunayTriangulation`] using the default topology
/// guarantee.
///
/// # Errors
///
/// Returns `Err(DelaunayTriangulationConstructionError)` if there are too few
/// requested points, triangulation construction fails, or validation fails
/// after robust fallback attempts. Invalid coordinate bounds cannot occur here
/// because callers provide a validated [`CoordinateRange`].
///
/// # Examples
///
/// ```no_run
/// use delaunay::prelude::construction::{
///     DelaunayConstructionFailure, DelaunayTriangulationConstructionError,
/// };
/// use delaunay::prelude::generators::{
///     CoordinateRange, CoordinateRangeError, generate_random_triangulation_in_range,
/// };
/// use std::num::NonZeroUsize;
///
/// # fn make_range() -> Result<CoordinateRange<f64>, CoordinateRangeError> {
/// #     CoordinateRange::try_new(-1.0_f64, 1.0)
/// # }
/// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
/// # let Some(twelve) = NonZeroUsize::new(12) else {
/// #     return Ok(());
/// # };
/// let range = make_range().map_err(|source| {
///     DelaunayTriangulationConstructionError::Triangulation(
///         DelaunayConstructionFailure::RandomPointGeneration { source: source.into() },
///     )
/// })?;
/// let dt = generate_random_triangulation_in_range::<f64, (), (), 3>(
///     twelve,
///     range,
///     None,
///     Some(123),
/// )?;
/// assert_eq!(dt.dim(), 3);
/// # Ok(())
/// # }
/// ```
pub fn generate_random_triangulation_in_range<T, U, V, const D: usize>(
    n_points: NonZeroUsize,
    bounds: CoordinateRange<T>,
    vertex_data: Option<U>,
    seed: Option<u64>,
) -> Result<DelaunayTriangulation<AdaptiveKernel<T>, U, V, D>, DelaunayTriangulationConstructionError>
where
    T: CoordinateScalar + SampleUniform,
    U: DataType,
    V: DataType,
{
    generate_random_triangulation_in_range_with_topology_guarantee(
        n_points,
        bounds,
        vertex_data,
        seed,
        TopologyGuarantee::DEFAULT,
    )
}

/// Generates a random triangulation from validated coordinate bounds and a topology guarantee.
///
/// Because `bounds` is already a [`CoordinateRange`], this function does not
/// perform raw tuple range parsing and cannot fail with an invalid-bounds error.
///
/// # Arguments
///
/// * `n_points` - Non-zero number of random points to generate.
/// * `bounds` - Validated coordinate bounds shared by every coordinate axis.
/// * `vertex_data` - Optional data attached to every generated vertex.
/// * `seed` - Optional seed for reproducible point generation.
/// * `topology_guarantee` - Topology guarantee required of the returned triangulation.
///
/// # Returns
///
/// A validated random [`DelaunayTriangulation`] satisfying `topology_guarantee`.
///
/// # Errors
///
/// Returns `Err(DelaunayTriangulationConstructionError)` if there are too few
/// requested points, triangulation construction fails, or validation fails
/// after robust fallback attempts. Invalid coordinate bounds cannot occur here
/// because callers provide a validated [`CoordinateRange`].
///
/// # Examples
///
/// ```no_run
/// use delaunay::prelude::construction::{
///     DelaunayConstructionFailure, DelaunayTriangulationConstructionError,
/// };
/// use delaunay::prelude::generators::{
///     CoordinateRange, CoordinateRangeError,
///     generate_random_triangulation_in_range_with_topology_guarantee,
/// };
/// use delaunay::prelude::TopologyGuarantee;
/// use std::num::NonZeroUsize;
///
/// # fn make_range() -> Result<CoordinateRange<f64>, CoordinateRangeError> {
/// #     CoordinateRange::try_new(-1.0_f64, 1.0)
/// # }
/// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
/// # let Some(twelve) = NonZeroUsize::new(12) else {
/// #     return Ok(());
/// # };
/// let range = make_range().map_err(|source| {
///     DelaunayTriangulationConstructionError::Triangulation(
///         DelaunayConstructionFailure::RandomPointGeneration { source: source.into() },
///     )
/// })?;
/// let dt = generate_random_triangulation_in_range_with_topology_guarantee::<f64, (), (), 3>(
///     twelve,
///     range,
///     None,
///     Some(123),
///     TopologyGuarantee::Pseudomanifold,
/// )?;
/// assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
/// # Ok(())
/// # }
/// ```
pub fn generate_random_triangulation_in_range_with_topology_guarantee<T, U, V, const D: usize>(
    n_points: NonZeroUsize,
    bounds: CoordinateRange<T>,
    vertex_data: Option<U>,
    seed: Option<u64>,
    topology_guarantee: TopologyGuarantee,
) -> Result<DelaunayTriangulation<AdaptiveKernel<T>, U, V, D>, DelaunayTriangulationConstructionError>
where
    T: CoordinateScalar + SampleUniform,
    U: DataType,
    V: DataType,
{
    let n_points = n_points.get();

    if n_points < D + 1 {
        return Err(TriangulationConstructionError::InsufficientVertices {
            dimension: D,
            source: SimplexValidationError::InsufficientVertices {
                actual: n_points,
                expected: D + 1,
                dimension: D,
            },
        }
        .into());
    }

    let points: Vec<Point<T, D>> = random_points_with_seed(n_points, bounds, seed);

    let min_vertices = (n_points / 6).max(D + 1);

    let mut initial_points = Some(points);
    let mut last_error = None;

    for attempt in 0..RANDOM_TRIANGULATION_MAX_POINTSET_ATTEMPTS {
        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_RANDOM_POINTSET_RETRIES").is_some() {
            tracing::debug!(
                attempt,
                max_attempts = RANDOM_TRIANGULATION_MAX_POINTSET_ATTEMPTS,
                "random_triangulation: pointset attempt"
            );
        }
        let point_seed = seed.map(|base| {
            if attempt > 0 {
                base ^ RANDOM_TRIANGULATION_POINTSET_SEED_MIX.wrapping_mul(attempt as u64)
            } else {
                base
            }
        });

        let points = if attempt == 0 {
            initial_points.take().ok_or_else(|| {
                DelaunayTriangulationConstructionError::from(
                    TriangulationConstructionError::InternalInconsistency {
                        message: "initial points already consumed".to_owned(),
                    },
                )
            })?
        } else {
            random_points_with_seed(n_points, bounds, point_seed)
        };

        let vertices = random_triangulation_build_vertices(points, vertex_data);
        match random_triangulation_try_with_vertices(
            &vertices,
            min_vertices,
            point_seed,
            topology_guarantee,
        ) {
            Ok(Some(dt)) => return Ok(dt),
            Ok(None) => {}
            Err(error) => last_error = Some(error),
        }
    }

    if let Some(error) = last_error {
        return Err(error);
    }

    Err(TriangulationConstructionError::GeometricDegeneracy {
        message: "Random triangulation failed validation after robust fallback".to_string(),
    }
    .into())
}

/// Builder for generating random Delaunay triangulations with flexible construction options.
///
/// This builder provides a fluent API for constructing random triangulations with control over:
/// - Insertion order strategy (`Input`, `Hilbert`)
/// - Topology guarantee (`None`, `PLManifold`)
/// - Construction options (deduplication, retry policy)
/// - Topology/Euler validation (the final triangulation must pass Level-3 checks)
///
/// # Examples
///
/// ```no_run
/// use delaunay::prelude::construction::{
///     DelaunayConstructionFailure, DelaunayTriangulationConstructionError,
/// };
/// use delaunay::prelude::generators::{
///     CoordinateRangeError, InsertionOrderStrategy, RandomTriangulationBuilder,
/// };
/// use delaunay::prelude::TopologyGuarantee;
/// use std::num::NonZeroUsize;
///
/// # fn random_generation_error(
/// #     source: CoordinateRangeError,
/// # ) -> DelaunayTriangulationConstructionError {
/// #     DelaunayTriangulationConstructionError::Triangulation(
/// #         DelaunayConstructionFailure::RandomPointGeneration {
/// #             source: source.into(),
/// #         },
/// #     )
/// # }
/// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
/// # let Some(twenty) = NonZeroUsize::new(20) else {
/// #     return Ok(());
/// # };
/// # let Some(one_hundred) = NonZeroUsize::new(100) else {
/// #     return Ok(());
/// # };
/// // Override the default `Hilbert` ordering with `Input` ordering.
/// let dt = RandomTriangulationBuilder::try_new(twenty, (-3.0, 3.0))
///     .map_err(random_generation_error)?
///     .seed(666)
///     .insertion_order(InsertionOrderStrategy::Input)
///     .build::<(), (), 3>()?;
///
/// // Build with PLManifold guarantee
/// let dt_manifold = RandomTriangulationBuilder::try_new(one_hundred, (-10.0, 10.0))
///     .map_err(random_generation_error)?
///     .seed(777)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build::<(), (), 4>()?;
/// # Ok(())
/// # }
/// ```
#[must_use]
pub struct RandomTriangulationBuilder<T> {
    n_points: NonZeroUsize,
    bounds: CoordinateRange<T>,
    seed: Option<u64>,
    topology_guarantee: TopologyGuarantee,
    construction_options: ConstructionOptions,
}

impl<T> RandomTriangulationBuilder<T> {
    /// Creates a new builder with the specified number of points and raw coordinate bounds.
    ///
    /// # Arguments
    ///
    /// * `n_points` - Non-zero number of random points to generate
    /// * `bounds` - Coordinate bounds as `(min, max)` tuple
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateRangeError::NonFiniteBound`] if either bound is
    /// non-finite, or [`CoordinateRangeError::NonIncreasing`] if the bounds are
    /// equal, decreasing, or incomparable.
    ///
    /// # Defaults
    ///
    /// - No seed (random)
    /// - Default topology guarantee ([`TopologyGuarantee::PLManifold`])
    /// - [`InsertionOrderStrategy::Hilbert`] insertion order (improves spatial locality during bulk insertion)
    /// - Default construction options ([`InitialSimplexStrategy::MaxVolume`] initial simplex,
    ///   shuffled retries, no explicit deduplication)
    ///
    /// [`InitialSimplexStrategy::MaxVolume`]: crate::construction::InitialSimplexStrategy::MaxVolume
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::prelude::generators::{
    ///     CoordinateRangeError, RandomTriangulationBuilder,
    /// };
    /// use std::num::NonZeroUsize;
    ///
    /// # fn main() -> Result<(), CoordinateRangeError> {
    /// # let Some(ten) = NonZeroUsize::new(10) else {
    /// #     return Ok(());
    /// # };
    /// let builder = RandomTriangulationBuilder::try_new(ten, (-1.0, 1.0))?.seed(42);
    /// let _ = builder;
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(n_points: NonZeroUsize, bounds: (T, T)) -> Result<Self, CoordinateRangeError<T>>
    where
        T: Debug + FiniteCheck + PartialOrd,
    {
        Ok(Self {
            n_points,
            bounds: CoordinateRange::try_from(bounds)?,
            seed: None,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            construction_options: ConstructionOptions::default(),
        })
    }

    /// Creates a new builder from already-validated coordinate bounds.
    ///
    /// Use this constructor when the caller has already parsed raw tuple bounds
    /// into a [`CoordinateRange`]. It is infallible because `bounds` already
    /// carries the finite, strictly-increasing range invariant.
    ///
    /// # Arguments
    ///
    /// * `n_points` - Non-zero number of random points to generate.
    /// * `bounds` - Validated coordinate bounds shared by every coordinate axis.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::prelude::generators::{
    ///     CoordinateRange, CoordinateRangeError, RandomTriangulationBuilder,
    /// };
    /// use std::num::NonZeroUsize;
    ///
    /// # fn main() -> Result<(), CoordinateRangeError> {
    /// # let Some(ten) = NonZeroUsize::new(10) else {
    /// #     return Ok(());
    /// # };
    /// let range = CoordinateRange::try_new(-1.0_f64, 1.0)?;
    /// let builder = RandomTriangulationBuilder::new_in_range(ten, range).seed(42);
    /// let _ = builder;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_in_range(n_points: NonZeroUsize, bounds: CoordinateRange<T>) -> Self {
        Self {
            n_points,
            bounds,
            seed: None,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            construction_options: ConstructionOptions::default(),
        }
    }

    /// Sets the random seed for reproducible triangulation generation.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::prelude::generators::{
    ///     CoordinateRangeError, RandomTriangulationBuilder,
    /// };
    /// use std::num::NonZeroUsize;
    ///
    /// # fn main() -> Result<(), CoordinateRangeError> {
    /// # let Some(ten) = NonZeroUsize::new(10) else {
    /// #     return Ok(());
    /// # };
    /// let builder = RandomTriangulationBuilder::try_new(ten, (-1.0, 1.0))?
    ///     .seed(42);
    /// let _ = builder;
    /// # Ok(())
    /// # }
    /// ```
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the topology guarantee for the triangulation.
    ///
    /// See [`TopologyGuarantee`] for available options.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::prelude::generators::{
    ///     CoordinateRangeError, RandomTriangulationBuilder,
    /// };
    /// use delaunay::prelude::TopologyGuarantee;
    /// use std::num::NonZeroUsize;
    ///
    /// # fn main() -> Result<(), CoordinateRangeError> {
    /// # let Some(ten) = NonZeroUsize::new(10) else {
    /// #     return Ok(());
    /// # };
    /// let builder = RandomTriangulationBuilder::try_new(ten, (-1.0, 1.0))?
    ///     .topology_guarantee(TopologyGuarantee::Pseudomanifold);
    /// let _ = builder;
    /// # Ok(())
    /// # }
    /// ```
    pub const fn topology_guarantee(mut self, topology_guarantee: TopologyGuarantee) -> Self {
        self.topology_guarantee = topology_guarantee;
        self
    }

    /// Sets the insertion order strategy.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::prelude::construction::{
    ///     DelaunayConstructionFailure, DelaunayTriangulationConstructionError,
    /// };
    /// use delaunay::prelude::generators::{
    ///     CoordinateRangeError, InsertionOrderStrategy, RandomTriangulationBuilder,
    /// };
    /// use std::num::NonZeroUsize;
    ///
    /// # fn random_generation_error(
    /// #     source: CoordinateRangeError,
    /// # ) -> DelaunayTriangulationConstructionError {
    /// #     DelaunayTriangulationConstructionError::Triangulation(
    /// #         DelaunayConstructionFailure::RandomPointGeneration {
    /// #             source: source.into(),
    /// #         },
    /// #     )
    /// # }
    /// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    /// # let Some(twenty) = NonZeroUsize::new(20) else {
    /// #     return Ok(());
    /// # };
    /// // Override the default `Hilbert` ordering with `Input` ordering.
    /// let dt = RandomTriangulationBuilder::try_new(twenty, (-3.0, 3.0))
    ///     .map_err(random_generation_error)?
    ///     .seed(666)
    ///     .insertion_order(InsertionOrderStrategy::Input)
    ///     .build::<(), (), 3>()?;
    /// # Ok(())
    /// # }
    /// ```
    pub const fn insertion_order(mut self, strategy: InsertionOrderStrategy) -> Self {
        self.construction_options = self.construction_options.with_insertion_order(strategy);
        self
    }

    /// Sets the full construction options.
    ///
    /// This provides access to advanced options like deduplication and retry policies.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::prelude::construction::ConstructionOptions;
    /// use delaunay::prelude::generators::{
    ///     CoordinateRangeError, RandomTriangulationBuilder,
    /// };
    /// use std::num::NonZeroUsize;
    ///
    /// # fn main() -> Result<(), CoordinateRangeError> {
    /// # let Some(ten) = NonZeroUsize::new(10) else {
    /// #     return Ok(());
    /// # };
    /// let builder = RandomTriangulationBuilder::try_new(ten, (-1.0, 1.0))?
    ///     .construction_options(ConstructionOptions::default());
    /// let _ = builder;
    /// # Ok(())
    /// # }
    /// ```
    pub const fn construction_options(mut self, options: ConstructionOptions) -> Self {
        self.construction_options = options;
        self
    }
}

impl<T> RandomTriangulationBuilder<T>
where
    T: CoordinateScalar + SampleUniform,
{
    /// Builds the random triangulation with the configured options.
    ///
    /// The builder stores a [`CoordinateRange`] after [`Self::try_new`] or
    /// [`Self::new_in_range`] has run, so this method consumes validated bounds
    /// and does not perform raw tuple range parsing.
    ///
    /// # Type Parameters
    ///
    /// * `U` - Vertex data type (must implement `DataType`)
    /// * `V` - Simplex data type (must implement `DataType`)
    /// * `D` - Dimensionality (const generic parameter)
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Insufficient vertices for the requested dimension (`n_points < D + 1`)
    /// - Triangulation construction fails (geometric degeneracy, etc.)
    /// - Construction fails after the configured retry policy (no robust-kernel fallback here)
    ///
    /// Invalid coordinate bounds cannot occur here because builder construction
    /// has already parsed them into a [`CoordinateRange`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::prelude::construction::{
    ///     DelaunayConstructionFailure, DelaunayTriangulationConstructionError,
    /// };
    /// use delaunay::prelude::generators::{CoordinateRangeError, RandomTriangulationBuilder};
    /// use std::num::NonZeroUsize;
    ///
    /// # fn random_generation_error(
    /// #     source: CoordinateRangeError,
    /// # ) -> DelaunayTriangulationConstructionError {
    /// #     DelaunayTriangulationConstructionError::Triangulation(
    /// #         DelaunayConstructionFailure::RandomPointGeneration {
    /// #             source: source.into(),
    /// #         },
    /// #     )
    /// # }
    /// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    /// # let Some(twelve) = NonZeroUsize::new(12) else {
    /// #     return Ok(());
    /// # };
    /// let dt = RandomTriangulationBuilder::try_new(twelve, (-2.0, 2.0))
    ///     .map_err(random_generation_error)?
    ///     .seed(7)
    ///     .build::<(), (), 3>()?;
    /// assert_eq!(dt.dim(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build<U, V, const D: usize>(
        self,
    ) -> Result<
        DelaunayTriangulation<AdaptiveKernel<T>, U, V, D>,
        DelaunayTriangulationConstructionError,
    >
    where
        U: DataType,
        V: DataType,
    {
        self.build_with_vertex_data(None)
    }

    /// Builds the random triangulation with vertex data attached to each vertex.
    ///
    /// The builder stores a [`CoordinateRange`] after [`Self::try_new`] or
    /// [`Self::new_in_range`] has run, so this method consumes validated bounds
    /// and does not perform raw tuple range parsing.
    ///
    /// # Arguments
    ///
    /// * `vertex_data` - Optional data to attach to each generated vertex
    ///
    /// # Errors
    ///
    /// Returns `Err` if construction fails (see [`build`](Self::build) for details).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::prelude::construction::{
    ///     DelaunayConstructionFailure, DelaunayTriangulationConstructionError,
    /// };
    /// use delaunay::prelude::generators::{CoordinateRangeError, RandomTriangulationBuilder};
    /// use std::num::NonZeroUsize;
    ///
    /// # fn random_generation_error(
    /// #     source: CoordinateRangeError,
    /// # ) -> DelaunayTriangulationConstructionError {
    /// #     DelaunayTriangulationConstructionError::Triangulation(
    /// #         DelaunayConstructionFailure::RandomPointGeneration {
    /// #             source: source.into(),
    /// #         },
    /// #     )
    /// # }
    /// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    /// # let Some(twelve) = NonZeroUsize::new(12) else {
    /// #     return Ok(());
    /// # };
    /// let dt = RandomTriangulationBuilder::try_new(twelve, (-2.0, 2.0))
    ///     .map_err(random_generation_error)?
    ///     .seed(7)
    ///     .build_with_vertex_data::<(), (), 3>(None)?;
    /// assert_eq!(dt.dim(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn build_with_vertex_data<U, V, const D: usize>(
        self,
        vertex_data: Option<U>,
    ) -> Result<
        DelaunayTriangulation<AdaptiveKernel<T>, U, V, D>,
        DelaunayTriangulationConstructionError,
    >
    where
        U: DataType,
        V: DataType,
    {
        let n_points = self.n_points.get();

        if n_points < D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: SimplexValidationError::InsufficientVertices {
                    actual: n_points,
                    expected: D + 1,
                    dimension: D,
                },
            }
            .into());
        }

        let points: Vec<Point<T, D>> = random_points_with_seed(n_points, self.bounds, self.seed);

        // Convert to vertices
        let vertices = random_triangulation_build_vertices(points, vertex_data);

        // Build triangulation with configured options
        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_RANDOM_BUILDER").is_some() {
            tracing::debug!(
                n_points,
                topology_guarantee = ?self.topology_guarantee,
                insertion_order = ?self.construction_options.insertion_order(),
                dedup_policy = ?self.construction_options.dedup_policy(),
                retry_policy = ?self.construction_options.retry_policy(),
                "random_triangulation_builder: single call to with_topology_guarantee_and_options"
            );
        }
        let dt = DelaunayTriangulation::with_topology_guarantee_and_options(
            &make_adaptive_kernel::<T>(),
            &vertices,
            self.topology_guarantee,
            self.construction_options,
        )?;
        validate_random_triangulation(dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::coordinate_range::{
        CoordinateRangeBound, CoordinateRangeOrdering, InvalidCoordinateValue,
    };
    use crate::vertex;
    use approx::assert_relative_eq;
    use std::assert_matches;

    /// Builds non-zero point-count literals for random triangulation tests.
    const fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test point count must be non-zero")
    }

    // =============================================================================
    // RANDOM TRIANGULATION GENERATION TESTS
    // =============================================================================

    #[test]
    fn test_generate_random_triangulation_basic() {
        // Test 2D triangulation creation
        let triangulation_2d =
            generate_random_triangulation::<(), (), 2>(nonzero(10), (-5.0, 5.0), None, Some(42))
                .unwrap();

        assert!(
            triangulation_2d.number_of_vertices() >= 3,
            "Expected at least 3 vertices in 2D triangulation, got {}",
            triangulation_2d.number_of_vertices()
        );
        assert_eq!(triangulation_2d.dim(), 2);
        triangulation_2d.is_valid().unwrap();

        // Test 3D triangulation creation with data
        let triangulation_3d = generate_random_triangulation::<i32, (), 3>(
            nonzero(8),
            (0.0, 1.0),
            Some(123),
            Some(456),
        )
        .unwrap();

        assert!(
            triangulation_3d.number_of_vertices() >= 4,
            "Expected at least 4 vertices in 3D triangulation, got {}",
            triangulation_3d.number_of_vertices()
        );
        assert_eq!(triangulation_3d.dim(), 3);
        triangulation_3d.is_valid().unwrap();

        // Exercise repeatable construction with two deterministic seeds.
        let triangulation_seeded =
            generate_random_triangulation::<(), (), 2>(nonzero(5), (-1.0, 1.0), None, Some(789))
                .unwrap();

        let triangulation_different_seed =
            generate_random_triangulation::<(), (), 2>(nonzero(5), (-1.0, 1.0), None, Some(790))
                .unwrap();

        triangulation_seeded.is_valid().unwrap();
        triangulation_different_seed.is_valid().unwrap();
        assert!(
            triangulation_seeded.number_of_vertices() >= 3,
            "Expected at least 3 vertices in seeded 2D triangulation, got {}",
            triangulation_seeded.number_of_vertices()
        );
        assert!(
            triangulation_different_seed.number_of_vertices() >= 3,
            "Expected at least 3 vertices in second seeded 2D triangulation, got {}",
            triangulation_different_seed.number_of_vertices()
        );
    }

    #[test]
    fn test_generate_random_triangulation_error_cases() {
        let result = generate_random_triangulation::<(), (), 2>(
            nonzero(10),
            (5.0, 1.0), // min > max
            None,
            Some(42),
        );
        let Err(DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::RandomPointGeneration { source },
        )) = result
        else {
            panic!("expected RandomPointGeneration error");
        };
        assert_eq!(
            source,
            RandomPointGenerationError::InvalidCoordinateRange {
                source: CoordinateRangeError::NonIncreasing {
                    ordering: CoordinateRangeOrdering::Decreasing,
                    min: 5.0,
                    max: 1.0,
                },
            }
        );

        // Zero points cannot cross the typed API boundary.
        assert_eq!(NonZeroUsize::new(0), None);
    }

    #[test]
    fn test_generate_random_triangulation_rejects_nonfinite_bounds_as_generation_error() {
        let nan_bounds = generate_random_triangulation::<(), (), 2>(
            nonzero(10),
            (f64::NAN, 1.0),
            None,
            Some(42),
        );

        assert_matches!(
            nan_bounds,
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::RandomPointGeneration {
                    source: RandomPointGenerationError::InvalidCoordinateRange {
                        source: CoordinateRangeError::NonFiniteBound { bound, value }
                    }
                }
            )) if bound == CoordinateRangeBound::Minimum && value == InvalidCoordinateValue::Nan
        );

        let Err(CoordinateRangeError::NonFiniteBound { bound, value }) =
            RandomTriangulationBuilder::<f64>::try_new(nonzero(10), (0.0, f64::INFINITY))
        else {
            panic!("expected invalid infinite bounds to fail");
        };
        assert_eq!(bound, CoordinateRangeBound::Maximum);
        assert_eq!(value, InvalidCoordinateValue::PositiveInfinity);
    }

    #[test]
    fn test_random_triangulation_range_apis_accept_validated_bounds() {
        let range = CoordinateRange::try_new(-1.0_f64, 1.0).unwrap();

        let triangulation = generate_random_triangulation_in_range::<f64, (), (), 2>(
            nonzero(10),
            range,
            None,
            Some(42),
        )
        .unwrap();
        assert_eq!(triangulation.dim(), 2);
        triangulation.is_valid().unwrap();

        let builder_triangulation = RandomTriangulationBuilder::new_in_range(nonzero(10), range)
            .seed(43)
            .build::<(), (), 2>()
            .unwrap();
        assert_eq!(builder_triangulation.dim(), 2);
        builder_triangulation.is_valid().unwrap();

        let guaranteed_triangulation =
            generate_random_triangulation_in_range_with_topology_guarantee::<f64, (), (), 2>(
                nonzero(10),
                range,
                None,
                Some(44),
                TopologyGuarantee::Pseudomanifold,
            )
            .unwrap();
        assert_eq!(
            guaranteed_triangulation.topology_guarantee(),
            TopologyGuarantee::Pseudomanifold
        );
        guaranteed_triangulation.is_valid().unwrap();
    }

    #[test]
    fn test_random_triangulation_builder_success_and_error_paths() {
        let triangulation = RandomTriangulationBuilder::<f64>::try_new(nonzero(10), (-5.0, 5.0))
            .unwrap()
            .seed(42)
            .build::<(), (), 2>()
            .unwrap();
        assert_eq!(triangulation.dim(), 2);
        assert!(triangulation.number_of_vertices() >= 3);
        triangulation.is_valid().unwrap();

        let triangulation_with_data =
            RandomTriangulationBuilder::<f64>::try_new(nonzero(10), (-5.0, 5.0))
                .unwrap()
                .seed(43)
                .build_with_vertex_data::<u32, (), 2>(Some(7))
                .unwrap();
        let vertex_data: Vec<_> = triangulation_with_data
            .tds()
            .vertices()
            .filter_map(|(_, vertex)| vertex.data().copied())
            .collect();
        assert_eq!(
            vertex_data.len(),
            triangulation_with_data.number_of_vertices()
        );
        assert!(vertex_data.iter().all(|&data| data == 7));
        triangulation_with_data.is_valid().unwrap();

        let too_few_vertices = RandomTriangulationBuilder::<f64>::try_new(nonzero(2), (-1.0, 1.0))
            .unwrap()
            .build::<(), (), 2>();
        let Err(DelaunayTriangulationConstructionError::Triangulation(
            DelaunayConstructionFailure::InsufficientVertices { dimension, source },
        )) = too_few_vertices
        else {
            panic!("expected InsufficientVertices error");
        };
        assert_eq!(dimension, 2);
        assert_eq!(
            source,
            SimplexValidationError::InsufficientVertices {
                actual: 2,
                expected: 3,
                dimension: 2,
            }
        );

        let invalid_bounds = RandomTriangulationBuilder::<f64>::try_new(nonzero(10), (5.0, 1.0));
        let Err(CoordinateRangeError::NonIncreasing { ordering, min, max }) = invalid_bounds else {
            panic!("expected invalid bounds to fail");
        };
        assert_eq!(ordering, CoordinateRangeOrdering::Decreasing);
        assert_relative_eq!(min, 5.0, epsilon = f64::EPSILON);
        assert_relative_eq!(max, 1.0, epsilon = f64::EPSILON);

        let seeded_invalid_bounds =
            RandomTriangulationBuilder::<f64>::try_new(nonzero(10), (5.0, 1.0));
        let Err(CoordinateRangeError::NonIncreasing { ordering, min, max }) = seeded_invalid_bounds
        else {
            panic!("expected seeded invalid bounds to fail");
        };
        assert_eq!(ordering, CoordinateRangeOrdering::Decreasing);
        assert_relative_eq!(min, 5.0, epsilon = f64::EPSILON);
        assert_relative_eq!(max, 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_generate_random_triangulation_reproducibility() {
        // Same seed should produce identical triangulations
        let triangulation1 =
            generate_random_triangulation::<(), (), 3>(nonzero(6), (-2.0, 2.0), None, Some(12345))
                .unwrap();

        let triangulation2 =
            generate_random_triangulation::<(), (), 3>(nonzero(6), (-2.0, 2.0), None, Some(12345))
                .unwrap();

        // Should have same structural properties
        assert_eq!(
            triangulation1.number_of_vertices(),
            triangulation2.number_of_vertices()
        );
        assert_eq!(
            triangulation1.number_of_simplices(),
            triangulation2.number_of_simplices()
        );
        assert_eq!(triangulation1.dim(), triangulation2.dim());
    }

    #[test]
    fn test_random_triangulation_try_with_vertices_exercises_fallbacks() {
        // Use a valid 2D simplex, but require more vertices than provided to force retries.
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let result = random_triangulation_try_with_vertices::<f64, (), (), 2>(
            &vertices,
            vertices.len() + 1,
            Some(7),
            TopologyGuarantee::PLManifold,
        );

        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_generate_random_triangulation_dimensions() {
        // Test different dimensional triangulations with parameter sets that are
        // also reused by examples. These (n_points, bounds, seed) triples have been
        // chosen to produce valid Delaunay triangulations without exhausting the
        // global Delaunay-repair limits in CI, while still exercising nontrivial
        // point sets in each dimension.

        // 2D with sufficient points for full triangulation
        let tri_2d =
            generate_random_triangulation::<(), (), 2>(nonzero(15), (0.0, 10.0), None, Some(555))
                .unwrap();
        assert_eq!(tri_2d.dim(), 2);
        assert!(tri_2d.number_of_simplices() > 0);

        // 3D with sufficient points for full triangulation
        let tri_3d =
            generate_random_triangulation::<(), (), 3>(nonzero(20), (-3.0, 3.0), None, Some(666))
                .unwrap();
        assert_eq!(tri_3d.dim(), 3);
        assert!(tri_3d.number_of_simplices() > 0);

        // 4D with sufficient points for full triangulation
        let tri_4d =
            generate_random_triangulation::<(), (), 4>(nonzero(12), (-1.0, 1.0), None, Some(777))
                .unwrap();
        assert_eq!(tri_4d.dim(), 4);
        assert!(tri_4d.number_of_simplices() > 0);

        // 5D with sufficient points for full triangulation
        let tri_5d =
            generate_random_triangulation::<(), (), 5>(nonzero(10), (0.0, 5.0), None, Some(888))
                .unwrap();
        assert_eq!(tri_5d.dim(), 5);
        assert!(tri_5d.number_of_simplices() > 0);
    }

    #[test]
    fn test_generate_random_triangulation_with_data() {
        // Test with different data types for vertices

        // Test with fixed-size character array (Copy type that can represent strings)
        // NOTE: This is a workaround for the DataType trait requiring Copy, which
        // prevents using String or &str directly due to lifetime/ownership constraints
        let tri_with_char_array = generate_random_triangulation::<[char; 8], (), 2>(
            nonzero(6),
            (-2.0, 2.0),
            Some(['v', 'e', 'r', 't', 'e', 'x', '_', 'd']),
            Some(888),
        )
        .unwrap();

        assert!(
            tri_with_char_array.number_of_vertices() >= 3,
            "Expected at least 3 vertices in 2D triangulation with data, got {}",
            tri_with_char_array.number_of_vertices()
        );
        tri_with_char_array.tds().is_valid().unwrap();

        // Convert the char array to a string to demonstrate string-like usage
        let char_array_data = ['v', 'e', 'r', 't', 'e', 'x', '_', 'd'];
        let string_representation: String = char_array_data.iter().collect();
        assert_eq!(string_representation, "vertex_d");

        // Test with integer data (try multiple deterministic seeds to avoid rare degeneracies)
        let seeds = [999_u64, 123, 456, 789, 2024];
        let mut tri_with_int_data: Option<DelaunayTriangulation<AdaptiveKernel<f64>, u32, (), 3>> =
            None;
        let mut last_err: Option<String> = None;
        for seed in seeds {
            match generate_random_triangulation::<u32, (), 3>(
                nonzero(8),
                (0.0, 5.0),
                Some(42u32),
                Some(seed),
            ) {
                Ok(tri) => {
                    tri_with_int_data = Some(tri);
                    break;
                }
                Err(e) => {
                    last_err = Some(format!("{e}"));
                }
            }
        }
        let tri_with_int_data = tri_with_int_data.unwrap_or_else(|| {
            panic!("All seeds failed to generate 3D triangulation with int data: {last_err:?}")
        });

        assert!(
            tri_with_int_data.number_of_vertices() >= 4,
            "Expected at least 4 vertices in 3D triangulation with data, got {}",
            tri_with_int_data.number_of_vertices()
        );
        tri_with_int_data.tds().is_valid().unwrap();

        // Test without data (None)
        let tri_no_data =
            generate_random_triangulation::<(), (), 2>(nonzero(5), (-1.0, 1.0), None, Some(111))
                .unwrap();

        assert!(
            tri_no_data.number_of_vertices() >= 3,
            "Expected at least 3 vertices in 2D triangulation without data, got {}",
            tri_no_data.number_of_vertices()
        );
        tri_no_data.tds().is_valid().unwrap();
    }
}

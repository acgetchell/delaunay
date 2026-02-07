//! Random triangulation generation functions.
//!
//! This module provides utilities for generating random Delaunay triangulations
//! with various topology guarantees.

#![forbid(unsafe_code)]

use super::point_generation::{generate_random_points, generate_random_points_seeded};
use crate::core::delaunay_triangulation::{
    ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationConstructionError,
    InsertionOrderStrategy, RetryPolicy,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::{TopologyGuarantee, TriangulationConstructionError};
use crate::core::vertex::{Vertex, VertexBuilder};
use crate::geometry::kernel::{FastKernel, RobustKernel};
use crate::geometry::point::Point;
use crate::geometry::robust_predicates::config_presets;
use crate::geometry::traits::coordinate::{CoordinateScalar, ScalarAccumulative};
use rand::SeedableRng;
use rand::distr::uniform::SampleUniform;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

const RANDOM_TRIANGULATION_MAX_SHUFFLE_ATTEMPTS: usize = 6;
const RANDOM_TRIANGULATION_MAX_POINTSET_ATTEMPTS: usize = 6;
const RANDOM_TRIANGULATION_POINTSET_SEED_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

fn validate_random_triangulation<K, U, V, const D: usize>(
    dt: DelaunayTriangulation<K, U, V, D>,
) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationConstructionError>
where
    K: crate::geometry::kernel::Kernel<D>,
    K::Scalar: ScalarAccumulative,
    U: DataType,
    V: DataType,
{
    dt.as_triangulation().validate().map_err(|e| {
        TriangulationConstructionError::GeometricDegeneracy {
            message: format!("Random triangulation failed topology/Euler validation: {e}"),
        }
    })?;
    Ok(dt)
}

fn random_triangulation_is_acceptable<K, U, V, const D: usize>(
    dt: &DelaunayTriangulation<K, U, V, D>,
    min_vertices: usize,
) -> bool
where
    K: crate::geometry::kernel::Kernel<D>,
    K::Scalar: ScalarAccumulative,
    U: DataType,
    V: DataType,
{
    dt.number_of_vertices() >= min_vertices
}

fn random_triangulation_try_build<K, T, U, V, const D: usize>(
    kernel: &K,
    vertices: &[Vertex<T, U, D>],
    min_vertices: usize,
    topology_guarantee: TopologyGuarantee,
) -> Option<DelaunayTriangulation<K, U, V, D>>
where
    K: crate::geometry::kernel::Kernel<D, Scalar = T>,
    T: ScalarAccumulative,
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
    )
    .ok()?;
    let dt = validate_random_triangulation(dt).ok()?;

    random_triangulation_is_acceptable(&dt, min_vertices).then_some(dt)
}

fn random_triangulation_build_vertices<T, U, const D: usize>(
    points: Vec<Point<T, D>>,
    vertex_data: Option<U>,
) -> Vec<Vertex<T, U, D>>
where
    T: CoordinateScalar,
    U: DataType,
{
    points
        .into_iter()
        .map(|point| {
            vertex_data.map_or_else(
                || {
                    VertexBuilder::default()
                        .point(point)
                        .build()
                        .expect("Failed to build vertex without data")
                },
                |data| {
                    VertexBuilder::default()
                        .point(point)
                        .data(data)
                        .build()
                        .expect("Failed to build vertex with data")
                },
            )
        })
        .collect()
}

fn random_triangulation_to_fast_kernel<T, U, V, const D: usize>(
    dt: &DelaunayTriangulation<RobustKernel<T>, U, V, D>,
    topology_guarantee: TopologyGuarantee,
) -> DelaunayTriangulation<FastKernel<T>, U, V, D>
where
    T: ScalarAccumulative,
    U: DataType,
    V: DataType,
{
    let tds = dt.tds().clone();
    DelaunayTriangulation::from_tds_with_topology_guarantee(
        tds,
        FastKernel::new(),
        topology_guarantee,
    )
}

fn random_triangulation_try_with_vertices<T, U, V, const D: usize>(
    vertices: &[Vertex<T, U, D>],
    min_vertices: usize,
    shuffle_seed: Option<u64>,
    topology_guarantee: TopologyGuarantee,
) -> Option<DelaunayTriangulation<FastKernel<T>, U, V, D>>
where
    T: ScalarAccumulative,
    U: DataType,
    V: DataType,
{
    if let Some(dt) = random_triangulation_try_build(
        &FastKernel::new(),
        vertices,
        min_vertices,
        topology_guarantee,
    ) {
        return Some(dt);
    }

    let robust_config = config_presets::degenerate_robust::<T>();
    let robust_kernel = RobustKernel::with_config(robust_config);

    if let Some(dt) =
        random_triangulation_try_build(&robust_kernel, vertices, min_vertices, topology_guarantee)
    {
        return Some(random_triangulation_to_fast_kernel(&dt, topology_guarantee));
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

        if let Some(dt) = random_triangulation_try_build(
            &robust_kernel,
            &shuffled,
            min_vertices,
            topology_guarantee,
        ) {
            return Some(random_triangulation_to_fast_kernel(&dt, topology_guarantee));
        }
    }

    None
}

/// This utility function combines random point generation and triangulation creation.
///
/// It generates random points using either seeded or unseeded random generation,
/// converts them to vertices, and creates a Delaunay triangulation using the
/// incremental cavity-based insertion algorithm.
///
/// This function is particularly useful for testing, benchmarking, and creating
/// triangulations for analysis or visualization purposes.
///
/// # Type Parameters
///
/// * `T` - Coordinate scalar type (must implement `ScalarAccumulative + SampleUniform`)
/// * `U` - Vertex data type (must implement `DataType`)
/// * `V` - Cell data type (must implement `DataType`)
/// * `D` - Dimensionality (const generic parameter)
///
/// # Arguments
///
/// * `n_points` - Number of random points to generate
/// * `bounds` - Coordinate bounds as `(min, max)` tuple
/// * `vertex_data` - Optional data to attach to each generated vertex
/// * `seed` - Optional seed for reproducible results. If `None`, uses thread-local RNG
///
/// # Returns
///
/// A `Result` containing either:
/// - `Ok(DelaunayTriangulation<FastKernel<T>, U, V, D>)` - The successfully created triangulation
/// - `Err(DelaunayTriangulationConstructionError)` - An error from point generation or triangulation construction
///
/// # Errors
///
/// Returns `DelaunayTriangulationConstructionError` with different variants depending on the failure:
///
/// **Invalid parameters** (mapped to `GeometricDegeneracy`):
/// - Point generation fails due to invalid bounds (e.g., `min > max`)
/// - Random number generator initialization fails
/// - **Note**: These are not strictly geometric degeneracy but are mapped to this
///   error variant for API simplicity. Consider validating bounds before calling
///   if you need to distinguish parameter errors from geometric issues.
///
/// **Empty triangulation** (special case):
/// - When `n_points = 0`, returns an empty triangulation with 0 vertices and `dim()` = -1
/// - This is not an error; use incremental insertion to add vertices later
///
/// **Insufficient vertices** (`InsufficientVertices`):
/// - When `0 < n_points < D + 1` (need at least D+1 points for a D-dimensional simplex)
/// - Example: 1-2 points in 2D, 1-3 points in 3D
///
/// **Geometric degeneracy** (`GeometricDegeneracy`):
/// - Points form a degenerate configuration (all collinear, coplanar, etc.)
/// - Numerical instability during construction
/// - Topology/Euler validation failure after robust fallback attempts
///
/// **Other construction failures** (various variants):
/// - Vertex/cell construction errors
/// - Triangulation validation failures
///
/// # Panics
///
/// This function can panic if:
/// - Vertex construction fails due to invalid data types or constraints
/// - This should not happen with valid inputs and supported data types
///
/// # Examples
///
/// ```no_run
/// use delaunay::geometry::util::generate_random_triangulation;
///
/// // Generate a 2D triangulation with 50 points, no seed (random each time)
/// let triangulation_2d = generate_random_triangulation::<f64, (), (), 2>(
///     50,
///     (-10.0, 10.0),
///     None,
///     None
/// );
///
/// // Generate a 3D triangulation with 30 points, seeded for reproducibility  
/// let triangulation_3d = generate_random_triangulation::<f64, (), (), 3>(
///     30,
///     (-5.0, 5.0),
///     None,
///     Some(42)
/// );
///
/// // Generate a 4D triangulation with custom vertex data
/// let triangulation_4d = generate_random_triangulation::<f64, i32, (), 4>(
///     20,
///     (0.0, 1.0),
///     Some(123),
///     Some(456)
/// );
///
/// // For string-like data, use fixed-size character arrays (Copy types)
/// let triangulation_with_strings = generate_random_triangulation::<f64, [char; 8], (), 2>(
///     20,
///     (0.0, 1.0),
///     Some(['v', 'e', 'r', 't', 'e', 'x', '_', 'A']),
///     Some(789)
/// );
///
/// // Access the underlying Tds if needed
/// let dt = triangulation_3d.unwrap();
/// let vertex_count = dt.tds().number_of_vertices();
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
/// - [`generate_random_points`] - For generating points without triangulation
/// - [`generate_random_points_seeded`] - For seeded random point generation only
/// - [`DelaunayTriangulation::new`] - For creating triangulations from existing vertices
/// - [`RandomTriangulationBuilder`] - For more control over construction options
pub fn generate_random_triangulation<T, U, V, const D: usize>(
    n_points: usize,
    bounds: (T, T),
    vertex_data: Option<U>,
    seed: Option<u64>,
) -> Result<DelaunayTriangulation<FastKernel<T>, U, V, D>, DelaunayTriangulationConstructionError>
where
    T: ScalarAccumulative + SampleUniform,
    U: DataType,
    V: DataType,
{
    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_UNUSED_IMPORTS").is_some() {
        eprintln!(
            "triangulation_generation::generate_random_triangulation called (n_points={n_points}, D={D}, seed={seed:?})"
        );
    }
    generate_random_triangulation_with_topology_guarantee(
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
///
/// # Errors
///
/// Returns `Err(DelaunayTriangulationConstructionError)` if:
/// - Random point generation fails (invalid bounds or RNG issues), mapped to
///   `TriangulationConstructionError::GeometricDegeneracy`.
/// - Triangulation construction fails due to geometric degeneracy, insertion errors, or the
///   requested topology guarantee not being satisfiable.
/// - Validation fails after the robust fallback attempts.
///
/// # Panics
///
/// Panics if internal point-set bookkeeping is inconsistent (this indicates a bug). In
/// particular, it will panic if the initial point set is unexpectedly consumed.
///
/// # Examples
///
/// ```no_run
/// use delaunay::geometry::util::generate_random_triangulation_with_topology_guarantee;
/// use delaunay::core::triangulation::TopologyGuarantee;
///
/// let dt = generate_random_triangulation_with_topology_guarantee::<f64, (), (), 3>(
///     20,
///     (-1.0, 1.0),
///     None,
///     Some(123),
///     TopologyGuarantee::Pseudomanifold,
/// )
/// .unwrap();
/// assert_eq!(dt.dim(), 3);
/// ```
#[allow(clippy::too_many_lines)]
pub fn generate_random_triangulation_with_topology_guarantee<T, U, V, const D: usize>(
    n_points: usize,
    bounds: (T, T),
    vertex_data: Option<U>,
    seed: Option<u64>,
    topology_guarantee: TopologyGuarantee,
) -> Result<DelaunayTriangulation<FastKernel<T>, U, V, D>, DelaunayTriangulationConstructionError>
where
    T: ScalarAccumulative + SampleUniform,
    U: DataType,
    V: DataType,
{
    // Handle empty triangulation case (0 points)
    if n_points == 0 {
        let kernel = FastKernel::new();
        return Ok(
            DelaunayTriangulation::with_empty_kernel_and_topology_guarantee(
                kernel,
                topology_guarantee,
            ),
        );
    }

    if n_points < D + 1 {
        return Err(TriangulationConstructionError::InsufficientVertices {
            dimension: D,
            source: crate::core::cell::CellValidationError::InsufficientVertices {
                actual: n_points,
                expected: D + 1,
                dimension: D,
            },
        }
        .into());
    }

    // Generate random points (seeded or unseeded)
    // Note: GeometricDegeneracy error wraps both point generation failures (invalid bounds, RNG issues)
    // and actual geometric degeneracy. This is a semantic approximation - point generation failures
    // are not strictly geometric degeneracy, but map to this error for API simplicity.
    let points: Vec<Point<T, D>> =
        match seed {
            Some(seed_value) => generate_random_points_seeded(n_points, bounds, seed_value)
                .map_err(|e| TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Random point generation failed: {e}"),
                })?,
            None => generate_random_points(n_points, bounds).map_err(|e| {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Random point generation failed: {e}"),
                }
            })?,
        };

    let min_vertices = (n_points / 6).max(D + 1);

    let mut initial_points = Some(points);

    for attempt in 0..RANDOM_TRIANGULATION_MAX_POINTSET_ATTEMPTS {
        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_RANDOM_POINTSET_RETRIES").is_some() {
            eprintln!(
                "random_triangulation: pointset attempt {attempt} of {RANDOM_TRIANGULATION_MAX_POINTSET_ATTEMPTS} (0-based)"
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
            initial_points
                .take()
                .expect("initial points already consumed")
        } else {
            match point_seed {
                Some(seed_value) => generate_random_points_seeded(n_points, bounds, seed_value)
                    .map_err(|e| TriangulationConstructionError::GeometricDegeneracy {
                        message: format!("Random point generation failed: {e}"),
                    })?,
                None => generate_random_points(n_points, bounds).map_err(|e| {
                    TriangulationConstructionError::GeometricDegeneracy {
                        message: format!("Random point generation failed: {e}"),
                    }
                })?,
            }
        };

        let vertices = random_triangulation_build_vertices(points, vertex_data);
        if let Some(dt) = random_triangulation_try_with_vertices(
            &vertices,
            min_vertices,
            point_seed,
            topology_guarantee,
        ) {
            return Ok(dt);
        }
    }

    Err(TriangulationConstructionError::GeometricDegeneracy {
        message: "Random triangulation failed validation after robust fallback".to_string(),
    }
    .into())
}

/// Builder for generating random Delaunay triangulations with flexible construction options.
///
/// This builder provides a fluent API for constructing random triangulations with control over:
/// - Insertion order strategy (`Input`, `Lexicographic`, `Morton`, `Hilbert`)
/// - Topology guarantee (`None`, `PLManifold`)
/// - Construction options (deduplication, retry policy)
/// - Topology/Euler validation (the final triangulation must pass Level-3 checks)
///
/// # Examples
///
/// ```no_run
/// use delaunay::geometry::util::RandomTriangulationBuilder;
/// use delaunay::core::InsertionOrderStrategy;
/// use delaunay::core::triangulation::TopologyGuarantee;
///
/// // Override the default `Hilbert` ordering with `Input` ordering.
/// let dt = RandomTriangulationBuilder::new(20, (-3.0, 3.0))
///     .seed(666)
///     .insertion_order(InsertionOrderStrategy::Input)
///     .build::<(), (), 3>()
///     .unwrap();
///
/// // Build with PLManifold guarantee
/// let dt_manifold = RandomTriangulationBuilder::new(100, (-10.0, 10.0))
///     .seed(777)
///     .topology_guarantee(TopologyGuarantee::PLManifold)
///     .build::<(), (), 4>()
///     .unwrap();
/// ```
pub struct RandomTriangulationBuilder<T> {
    n_points: usize,
    bounds: (T, T),
    seed: Option<u64>,
    topology_guarantee: TopologyGuarantee,
    construction_options: ConstructionOptions,
}

impl<T> RandomTriangulationBuilder<T>
where
    T: ScalarAccumulative + SampleUniform,
{
    /// Creates a new builder with the specified number of points and coordinate bounds.
    ///
    /// # Arguments
    ///
    /// * `n_points` - Number of random points to generate
    /// * `bounds` - Coordinate bounds as `(min, max)` tuple
    ///
    /// # Defaults
    ///
    /// - No seed (random)
    /// - Default topology guarantee (None)
    /// - `Hilbert` insertion order (improves spatial locality during bulk insertion)
    /// - Default construction options (no deduplication, debug-only retries)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::geometry::util::RandomTriangulationBuilder;
    ///
    /// let builder = RandomTriangulationBuilder::new(10, (-1.0, 1.0)).seed(42);
    /// let _ = builder;
    /// ```
    #[must_use]
    pub fn new(n_points: usize, bounds: (T, T)) -> Self {
        Self {
            n_points,
            bounds,
            seed: None,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            construction_options: ConstructionOptions::default(),
        }
    }

    /// Sets the random seed for reproducible triangulation generation.
    #[must_use]
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the topology guarantee for the triangulation.
    ///
    /// See [`TopologyGuarantee`] for available options.
    #[must_use]
    pub const fn topology_guarantee(mut self, topology_guarantee: TopologyGuarantee) -> Self {
        self.topology_guarantee = topology_guarantee;
        self
    }

    /// Sets the insertion order strategy.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::geometry::util::RandomTriangulationBuilder;
    /// use delaunay::core::InsertionOrderStrategy;
    ///
    /// // Override the default `Hilbert` ordering with `Input` ordering.
    /// let dt = RandomTriangulationBuilder::new(20, (-3.0, 3.0))
    ///     .seed(666)
    ///     .insertion_order(InsertionOrderStrategy::Input)
    ///     .build::<(), (), 3>()
    ///     .unwrap();
    /// ```
    #[must_use]
    pub const fn insertion_order(mut self, strategy: InsertionOrderStrategy) -> Self {
        self.construction_options = self.construction_options.with_insertion_order(strategy);
        self
    }

    /// Sets the full construction options.
    ///
    /// This provides access to advanced options like deduplication and retry policies.
    #[must_use]
    pub const fn construction_options(mut self, options: ConstructionOptions) -> Self {
        self.construction_options = options;
        self
    }

    /// Builds the random triangulation with the configured options.
    ///
    /// # Type Parameters
    ///
    /// * `U` - Vertex data type (must implement `DataType`)
    /// * `V` - Cell data type (must implement `DataType`)
    /// * `D` - Dimensionality (const generic parameter)
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Random point generation fails (invalid bounds, RNG issues)
    /// - Insufficient vertices for the requested dimension (`n_points < D + 1`)
    /// - Triangulation construction fails (geometric degeneracy, etc.)
    /// - Construction fails after the configured retry policy (no robust-kernel fallback here)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use delaunay::geometry::util::RandomTriangulationBuilder;
    ///
    /// let dt = RandomTriangulationBuilder::new(12, (-2.0, 2.0))
    ///     .seed(7)
    ///     .build::<(), (), 3>()
    ///     .unwrap();
    /// assert_eq!(dt.dim(), 3);
    /// ```
    pub fn build<U, V, const D: usize>(
        self,
    ) -> Result<DelaunayTriangulation<FastKernel<T>, U, V, D>, DelaunayTriangulationConstructionError>
    where
        U: DataType,
        V: DataType,
    {
        self.build_with_vertex_data(None)
    }

    /// Builds the random triangulation with vertex data attached to each vertex.
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
    /// use delaunay::geometry::util::RandomTriangulationBuilder;
    ///
    /// let dt = RandomTriangulationBuilder::new(12, (-2.0, 2.0))
    ///     .seed(7)
    ///     .build_with_vertex_data::<(), (), 3>(None)
    ///     .unwrap();
    /// assert_eq!(dt.dim(), 3);
    /// ```
    pub fn build_with_vertex_data<U, V, const D: usize>(
        self,
        vertex_data: Option<U>,
    ) -> Result<DelaunayTriangulation<FastKernel<T>, U, V, D>, DelaunayTriangulationConstructionError>
    where
        U: DataType,
        V: DataType,
    {
        // Handle empty triangulation case (0 points)
        if self.n_points == 0 {
            let kernel = FastKernel::new();
            return Ok(
                DelaunayTriangulation::with_empty_kernel_and_topology_guarantee(
                    kernel,
                    self.topology_guarantee,
                ),
            );
        }

        if self.n_points < D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: crate::core::cell::CellValidationError::InsufficientVertices {
                    actual: self.n_points,
                    expected: D + 1,
                    dimension: D,
                },
            }
            .into());
        }

        // Generate random points
        let points: Vec<Point<T, D>> = match self.seed {
            Some(seed_value) => {
                generate_random_points_seeded(self.n_points, self.bounds, seed_value).map_err(
                    |e| TriangulationConstructionError::GeometricDegeneracy {
                        message: format!("Random point generation failed: {e}"),
                    },
                )?
            }
            None => generate_random_points(self.n_points, self.bounds).map_err(|e| {
                TriangulationConstructionError::GeometricDegeneracy {
                    message: format!("Random point generation failed: {e}"),
                }
            })?,
        };

        // Convert to vertices
        let vertices = random_triangulation_build_vertices(points, vertex_data);

        // Build triangulation with configured options
        #[cfg(debug_assertions)]
        if std::env::var_os("DELAUNAY_DEBUG_RANDOM_BUILDER").is_some() {
            eprintln!(
                "random_triangulation_builder: single call to with_topology_guarantee_and_options with n_points={}, topology_guarantee={:?}, insertion_order={:?}, dedup_policy={:?}, retry_policy={:?}",
                self.n_points,
                self.topology_guarantee,
                self.construction_options.insertion_order(),
                self.construction_options.dedup_policy(),
                self.construction_options.retry_policy(),
            );
        }
        let dt = DelaunayTriangulation::with_topology_guarantee_and_options(
            &FastKernel::new(),
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
    use crate::vertex;
    // =============================================================================
    // RANDOM TRIANGULATION GENERATION TESTS
    // =============================================================================

    #[test]
    #[ignore = "Flaky: unseeded random generation occasionally fails with cavity filling errors - needs investigation"]
    fn test_generate_random_triangulation_basic() {
        // Test 2D triangulation creation
        let triangulation_2d =
            generate_random_triangulation::<f64, (), (), 2>(10, (-5.0, 5.0), None, Some(42))
                .unwrap();

        assert!(
            triangulation_2d.number_of_vertices() >= 3,
            "Expected at least 3 vertices in 2D triangulation, got {}",
            triangulation_2d.number_of_vertices()
        );
        assert_eq!(triangulation_2d.dim(), 2);
        let valid_2d = triangulation_2d.is_valid();
        if let Err(e) = &valid_2d {
            println!("test_generate_random_triangulation_basic (2D): TDS invalid: {e}");
        }
        assert!(valid_2d.is_ok());

        // Test 3D triangulation creation with data
        let triangulation_3d =
            generate_random_triangulation::<f64, i32, (), 3>(8, (0.0, 1.0), Some(123), Some(456))
                .unwrap();

        assert!(
            triangulation_3d.number_of_vertices() >= 4,
            "Expected at least 4 vertices in 3D triangulation, got {}",
            triangulation_3d.number_of_vertices()
        );
        assert_eq!(triangulation_3d.dim(), 3);
        let valid_3d = triangulation_3d.is_valid();
        if let Err(e) = &valid_3d {
            println!("test_generate_random_triangulation_basic (3D): TDS invalid: {e}");
        }
        assert!(valid_3d.is_ok());

        // Test seeded vs unseeded (should get different results)
        let triangulation_seeded =
            generate_random_triangulation::<f64, (), (), 2>(5, (-1.0, 1.0), None, Some(789))
                .unwrap();

        let triangulation_unseeded =
            generate_random_triangulation::<f64, (), (), 2>(5, (-1.0, 1.0), None, None).unwrap();

        // Both should be valid
        let valid_seeded = triangulation_seeded.is_valid();
        if let Err(e) = &valid_seeded {
            println!("test_generate_random_triangulation_basic (seeded 2D): TDS invalid: {e}");
        }
        assert!(valid_seeded.is_ok());

        let valid_unseeded = triangulation_unseeded.is_valid();
        if let Err(e) = &valid_unseeded {
            println!("test_generate_random_triangulation_basic (unseeded 2D): TDS invalid: {e}");
        }
        assert!(valid_unseeded.is_ok());
        assert!(
            triangulation_seeded.number_of_vertices() >= 3,
            "Expected at least 3 vertices in seeded 2D triangulation, got {}",
            triangulation_seeded.number_of_vertices()
        );
        assert!(
            triangulation_unseeded.number_of_vertices() >= 3,
            "Expected at least 3 vertices in unseeded 2D triangulation, got {}",
            triangulation_unseeded.number_of_vertices()
        );
    }

    #[test]
    fn test_generate_random_triangulation_error_cases() {
        // Test invalid bounds
        let result = generate_random_triangulation::<f64, (), (), 2>(
            10,
            (5.0, 1.0), // min > max
            None,
            Some(42),
        );
        assert!(result.is_err());

        // Test zero points - should succeed with empty triangulation
        let result =
            generate_random_triangulation::<f64, (), (), 2>(0, (-1.0, 1.0), None, Some(42));
        assert!(result.is_ok());
        let triangulation = result.unwrap();
        assert_eq!(triangulation.number_of_vertices(), 0);
        assert_eq!(triangulation.dim(), -1);
    }

    #[test]
    fn test_generate_random_triangulation_reproducibility() {
        // Same seed should produce identical triangulations
        let triangulation1 =
            generate_random_triangulation::<f64, (), (), 3>(6, (-2.0, 2.0), None, Some(12345))
                .unwrap();

        let triangulation2 =
            generate_random_triangulation::<f64, (), (), 3>(6, (-2.0, 2.0), None, Some(12345))
                .unwrap();

        // Should have same structural properties
        assert_eq!(
            triangulation1.number_of_vertices(),
            triangulation2.number_of_vertices()
        );
        assert_eq!(
            triangulation1.number_of_cells(),
            triangulation2.number_of_cells()
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

        assert!(result.is_none());
    }

    #[test]
    #[ignore = "High-dimensional random triangulations occasionally hit cavity filling errors in CI - needs robust predicate investigation"]
    fn test_generate_random_triangulation_dimensions() {
        // Test different dimensional triangulations with parameter sets that are
        // also reused by examples. These (n_points, bounds, seed) triples have been
        // chosen to produce valid Delaunay triangulations without exhausting the
        // global Delaunay-repair limits in CI, while still exercising nontrivial
        // point sets in each dimension.

        // 2D with sufficient points for full triangulation
        let tri_2d =
            generate_random_triangulation::<f64, (), (), 2>(15, (0.0, 10.0), None, Some(555))
                .unwrap();
        assert_eq!(tri_2d.dim(), 2);
        assert!(tri_2d.number_of_cells() > 0);

        // 3D with sufficient points for full triangulation
        let tri_3d =
            generate_random_triangulation::<f64, (), (), 3>(20, (-3.0, 3.0), None, Some(666))
                .unwrap();
        assert_eq!(tri_3d.dim(), 3);
        assert!(tri_3d.number_of_cells() > 0);

        // 4D with sufficient points for full triangulation
        let tri_4d =
            generate_random_triangulation::<f64, (), (), 4>(12, (-1.0, 1.0), None, Some(777))
                .unwrap();
        assert_eq!(tri_4d.dim(), 4);
        assert!(tri_4d.number_of_cells() > 0);

        // 5D with sufficient points for full triangulation
        let tri_5d =
            generate_random_triangulation::<f64, (), (), 5>(10, (0.0, 5.0), None, Some(888))
                .unwrap();
        assert_eq!(tri_5d.dim(), 5);
        assert!(tri_5d.number_of_cells() > 0);
    }

    #[test]
    fn test_generate_random_triangulation_with_data() {
        // Test with different data types for vertices

        // Test with fixed-size character array (Copy type that can represent strings)
        // NOTE: This is a workaround for the DataType trait requiring Copy, which
        // prevents using String or &str directly due to lifetime/ownership constraints
        let tri_with_char_array = generate_random_triangulation::<f64, [char; 8], (), 2>(
            6,
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
        let valid_char = tri_with_char_array.tds().is_valid();
        if let Err(e) = &valid_char {
            println!(
                "test_generate_random_triangulation_with_data (2D char data): TDS invalid: {e}"
            );
        }
        assert!(valid_char.is_ok());

        // Convert the char array to a string to demonstrate string-like usage
        let char_array_data = ['v', 'e', 'r', 't', 'e', 'x', '_', 'd'];
        let string_representation: String = char_array_data.iter().collect();
        assert_eq!(string_representation, "vertex_d");

        // Test with integer data (try multiple deterministic seeds to avoid rare degeneracies)
        let seeds = [999_u64, 123, 456, 789, 2024];
        let mut tri_with_int_data: Option<DelaunayTriangulation<FastKernel<f64>, u32, (), 3>> =
            None;
        let mut last_err: Option<String> = None;
        for seed in seeds {
            match generate_random_triangulation::<f64, u32, (), 3>(
                8,
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
        let valid_int = tri_with_int_data.tds().is_valid();
        if let Err(e) = &valid_int {
            println!(
                "test_generate_random_triangulation_with_data (3D int data): TDS invalid: {e}"
            );
        }
        assert!(valid_int.is_ok());

        // Test without data (None)
        let tri_no_data =
            generate_random_triangulation::<f64, (), (), 2>(5, (-1.0, 1.0), None, Some(111))
                .unwrap();

        assert!(
            tri_no_data.number_of_vertices() >= 3,
            "Expected at least 3 vertices in 2D triangulation without data, got {}",
            tri_no_data.number_of_vertices()
        );
        let valid_no_data = tri_no_data.tds().is_valid();
        if let Err(e) = &valid_no_data {
            println!("test_generate_random_triangulation_with_data (2D no data): TDS invalid: {e}");
        }
        assert!(valid_no_data.is_ok());
    }
}

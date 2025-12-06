//! Generic triangulation combining kernel and combinatorial data structure.
//!
//! Following CGAL's architecture, the `Triangulation` struct combines:
//! - A geometric `Kernel` for predicates
//! - A purely combinatorial `Tds` for topology
//!
//! This layer provides geometric operations while delegating topology to Tds.
//!
//! # Validation Hierarchy
//!
//! The library provides **four levels** of validation, each building on the previous:
//!
//! ## Level 1: Element Validity
//! - **Methods**: [`Cell::is_valid()`], [`Vertex::is_valid()`]
//! - **Checks**: Basic data integrity (coordinate validity, UUID presence, proper initialization)
//! - **Cost**: O(1) per element
//! - **Use**: Building blocks for higher-level validation
//!
//! ## Level 2: TDS Structural Validity
//! - **Method**: [`Tds::is_valid()`]
//! - **Checks**:
//!   - UUID ↔ Key mapping consistency
//!   - No duplicate cells (same vertex sets)
//!   - Facet sharing invariant (≤2 cells per facet)
//!   - Neighbor consistency (mutual relationships)
//!   - All cells valid (calls Level 1)
//! - **Cost**: O(N×D²) where N = cells, D = dimension
//! - **Use**: Verify combinatorial correctness after construction or mutation
//!
//! ## Level 3: Manifold Topology
//! - **Method**: [`Triangulation::validate_manifold()`](crate::core::triangulation::Triangulation::validate_manifold)
//! - **Checks**:
//!   - All TDS invariants (calls Level 2)
//!   - Strengthened facet property (exactly 1 or 2 cells per facet)
//!   - Euler characteristic (χ = V - E + F - C matches expected topology)
//! - **Cost**: O(N×D²) for simplex counting
//! - **Use**: Verify the triangulation forms a valid topological manifold
//!
//! ## Level 4: Delaunay Property
//! - **Method**: [`DelaunayTriangulation::validate_delaunay()`]
//! - **Checks**:
//!   - Empty circumsphere property (no vertex inside any cell's circumsphere)
//!   - Uses geometric predicates from kernel
//! - **Cost**: O(N×V) where N = cells, V = vertices
//! - **Use**: Verify geometric optimality of the triangulation
//!
//! ## Usage Guidelines
//!
//! ```rust
//! use delaunay::prelude::*;
//!
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let dt = DelaunayTriangulation::new(&vertices).unwrap();
//!
//! // Quick structural check (Level 2)
//! assert!(dt.is_valid().is_ok());
//!
//! // Thorough manifold check (Level 3, includes Level 2)
//! assert!(dt.triangulation().validate_manifold().is_ok());
//!
//! // Full geometric validation (Level 4, most expensive)
//! assert!(dt.validate_delaunay().is_ok());
//! ```
//!
//! **Performance**: Use Level 2 for most production validation. Reserve Level 3 for
//! tests/debug builds, and Level 4 for critical verification or debugging geometric issues.
//!
//! [`Cell::is_valid()`]: crate::core::cell::Cell::is_valid
//! [`Vertex::is_valid()`]: crate::core::vertex::Vertex::is_valid
//! [`Tds::is_valid()`]: crate::core::triangulation_data_structure::Tds::is_valid
//! [`DelaunayTriangulation::validate_delaunay()`]: crate::core::delaunay_triangulation::DelaunayTriangulation::validate_delaunay

use core::iter::Sum;
use core::ops::{AddAssign, Div, SubAssign};
use std::cmp::Ordering as CmpOrdering;

use num_traits::NumCast;
use uuid::Uuid;

use crate::core::algorithms::incremental_insertion::{
    InsertionError, InsertionStatistics, repair_neighbor_pointers,
};
use crate::core::cell::Cell;
use crate::core::collections::{
    CellKeyBuffer, CellKeySet, FacetIssuesMap, FastHasher, MAX_PRACTICAL_DIMENSION_SIZE,
    SmallBuffer, ValidCellsBuffer, VertexKeySet,
};
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{
    CellKey, Tds, TriangulationConstructionError, TriangulationValidationError, VertexKey,
};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::quality::radius_ratio;
use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::geometry::util::safe_scalar_to_f64;
use std::hash::{Hash, Hasher};

/// Maximum number of repair iterations for fixing non-manifold topology after insertion.
///
/// This limit prevents infinite loops in the rare case where repair cannot make progress.
/// In practice, most insertions require 0-2 iterations to restore manifold topology.
const MAX_REPAIR_ITERATIONS: usize = 10;

/// Generic triangulation combining kernel and data structure.
///
/// # Type Parameters
/// - `K`: Geometric kernel implementing predicates
/// - `U`: User data type for vertices
/// - `V`: User data type for cells
/// - `D`: Dimension of the triangulation
///
/// # Phase 2 TODO
/// Add geometric operations that use the kernel for predicates.
#[derive(Clone, Debug)]
pub struct Triangulation<K, U, V, const D: usize>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    /// The geometric kernel for predicates.
    pub(crate) kernel: K,
    /// The combinatorial triangulation data structure.
    pub(crate) tds: Tds<K::Scalar, U, V, D>,
}

// =============================================================================
// Basic Accessors (Minimal Bounds)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    /// Create an empty triangulation with the given kernel.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let tri: Triangulation<FastKernel<f64>, (), (), 3> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// assert_eq!(tri.number_of_vertices(), 0);
    /// assert_eq!(tri.number_of_cells(), 0);
    /// assert_eq!(tri.dim(), -1); // Empty triangulation has dimension -1
    /// ```
    #[must_use]
    pub fn new_empty(kernel: K) -> Self {
        Self {
            kernel,
            tds: Tds::empty(),
        }
    }

    /// Returns an iterator over all cells in the triangulation.
    ///
    /// Delegates to the underlying Tds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.triangulation();
    ///
    /// // Iterate over cells
    /// for (_cell_key, cell) in tri.cells() {
    ///     assert_eq!(cell.number_of_vertices(), 3); // 2D triangle
    /// }
    /// assert_eq!(tri.cells().count(), 1);
    /// ```
    pub fn cells(&self) -> impl Iterator<Item = (CellKey, &Cell<K::Scalar, U, V, D>)> {
        self.tds.cells()
    }

    /// Returns an iterator over all vertices in the triangulation.
    ///
    /// Delegates to the underlying Tds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// let tri = dt.triangulation();
    ///
    /// // Iterate over vertices
    /// for (_vertex_key, vertex) in tri.vertices() {
    ///     assert_eq!(vertex.dim(), 2); // 2D vertices
    /// }
    /// assert_eq!(tri.vertices().count(), 3);
    /// ```
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<K::Scalar, U, D>)> {
        self.tds.vertices()
    }

    /// Returns the number of vertices in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.triangulation().number_of_vertices(), 4);
    /// ```
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.tds.number_of_vertices()
    }

    /// Returns the number of cells in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.triangulation().number_of_cells(), 1); // Single tetrahedron
    /// ```
    #[must_use]
    pub fn number_of_cells(&self) -> usize {
        self.tds.number_of_cells()
    }

    /// Returns the dimension of the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // Empty triangulation has dimension -1
    /// let empty: Triangulation<FastKernel<f64>, (), (), 3> =
    ///     Triangulation::new_empty(FastKernel::new());
    /// assert_eq!(empty.dim(), -1);
    ///
    /// // 3D tetrahedron has dimension 3
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert_eq!(dt.triangulation().dim(), 3);
    /// ```
    #[must_use]
    pub fn dim(&self) -> i32 {
        self.tds.dim()
    }

    /// Returns an iterator over all facets in the triangulation.
    ///
    /// This provides efficient access to all facets without pre-allocating a vector.
    /// Each facet is represented as a lightweight `FacetView` that references the
    /// underlying triangulation data.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for all facets in the triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Iterate over all facets
    /// let facet_count = dt.triangulation().facets().count();
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 facets
    /// ```
    pub fn facets(&self) -> AllFacetsIter<'_, K::Scalar, U, V, D> {
        AllFacetsIter::new(&self.tds)
    }

    /// Returns an iterator over boundary (hull) facets in the triangulation.
    ///
    /// Boundary facets are those that belong to exactly one cell. This method
    /// computes the facet-to-cells map internally for convenience.
    ///
    /// # Returns
    ///
    /// An iterator yielding `FacetView` objects for boundary facets only.
    ///
    /// # Panics
    ///
    /// Panics if the triangulation data structure is corrupted (cells have invalid
    /// neighbor relationships or facet information). This indicates a bug in the
    /// library and should never happen with a properly constructed triangulation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// let boundary_count = dt.triangulation().boundary_facets().count();
    /// assert_eq!(boundary_count, 4); // All facets are on boundary
    /// ```
    pub fn boundary_facets(&self) -> BoundaryFacetsIter<'_, K::Scalar, U, V, D> {
        // build_facet_to_cells_map only fails if cells have invalid structure,
        // which should never happen in a valid triangulation
        let facet_map = self
            .tds
            .build_facet_to_cells_map()
            .expect("Failed to build facet map - triangulation structure is corrupted");
        BoundaryFacetsIter::new(&self.tds, facet_map)
    }

    /// Validates that the triangulation forms a valid manifold.
    ///
    /// This method validates **manifold topology** on top of the structural invariants checked by
    /// [`Tds::is_valid()`](crate::core::triangulation_data_structure::Tds::is_valid).
    ///
    /// # Validation Hierarchy
    ///
    /// - **Level 1: Element Validity** - [`Cell::is_valid()`](crate::core::cell::Cell::is_valid),
    ///   [`Vertex::is_valid()`](crate::core::vertex::Vertex::is_valid)
    /// - **Level 2: TDS Structural Validity** - [`Tds::is_valid()`](crate::core::triangulation_data_structure::Tds::is_valid)
    ///   (mappings, no duplicates, facet sharing ≤2, neighbor consistency)
    /// - **Level 3: Manifold Topology** - **This method** (manifold facet property, Euler characteristic)
    /// - **Level 4: Delaunay Property** - [`DelaunayTriangulation::validate_delaunay()`](crate::core::delaunay_triangulation::DelaunayTriangulation::validate_delaunay)
    ///
    /// # Manifold Requirements
    ///
    /// A valid manifold triangulation must satisfy:
    ///
    /// 1. **All TDS structural invariants** (validated first)
    /// 2. **Manifold facet property**: Each facet belongs to exactly 1 cell (boundary) or exactly 2 cells (interior)
    /// 3. **Euler characteristic**: χ matches the expected value for the topological space
    ///    - 2D closed: χ = V - E + F = 2 (sphere)
    ///    - 3D with boundary: χ = V - E + F - C = 1 (ball)
    ///    - General D-dimensional with boundary: χ = 1
    ///
    /// # Performance
    ///
    /// **Time Complexity**: O(N×D²) where N = number of cells, D = dimension
    /// - Facet map construction: O(N×D)
    /// - Manifold facet check: O(N×D)
    /// - Simplex counting for Euler: O(N×D²)
    ///
    /// For large triangulations, this is expensive. Use judiciously in tests or debug builds.
    ///
    /// # Errors
    ///
    /// Returns [`TriangulationValidationError`] if:
    /// - Any TDS structural invariant fails (mappings, duplicates, facet sharing, neighbors)
    /// - Any facet is shared by 0 or >2 cells (non-manifold)
    /// - Euler characteristic doesn't match expected value
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::*;
    ///
    /// // Valid 3D triangulation (single tetrahedron)
    /// let vertices = [
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let dt = DelaunayTriangulation::new(&vertices).unwrap();
    /// assert!(dt.triangulation().validate_manifold().is_ok());
    /// ```
    pub fn validate_manifold(&self) -> Result<(), TriangulationValidationError>
    where
        K::Scalar: AddAssign + SubAssign + Sum,
    {
        // 1. First validate all TDS structural invariants
        //    (mappings, no duplicates, facet sharing ≤2, neighbor consistency)
        self.tds.is_valid()?;

        // 2. Strengthen facet sharing to manifold property:
        //    - Boundary facets: exactly 1 cell
        //    - Interior facets: exactly 2 cells
        //    (No facets with 0 cells allowed in a manifold)
        self.validate_manifold_facets()?;

        // 3. Validate Euler characteristic for the topological space
        self.validate_euler_characteristic()?;

        Ok(())
    }

    /// Validates the manifold facet property.
    ///
    /// In a valid manifold, every facet must belong to exactly 1 cell (boundary facet)
    /// or exactly 2 cells (interior facet). This is stronger than the TDS invariant
    /// which only requires ≤2 cells per facet.
    fn validate_manifold_facets(&self) -> Result<(), TriangulationValidationError> {
        use crate::core::collections::FacetToCellsMap;

        // Build facet-to-cells map
        let facet_to_cells: FacetToCellsMap = self.tds.build_facet_to_cells_map()?;

        // Check that each facet has exactly 1 or 2 cells
        for (facet_key, cell_facet_pairs) in &facet_to_cells {
            let count = cell_facet_pairs.len();
            if count == 0 || count > 2 {
                return Err(TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Non-manifold facet: facet with key {facet_key} belongs to {count} cells (expected 1 or 2 for manifold)"
                    ),
                });
            }
        }

        Ok(())
    }

    /// Validates the Euler characteristic for the triangulation.
    ///
    /// Computes χ = Σ(-1)^d × `N_d` where `N_d` is the number of d-dimensional simplices.
    ///
    /// # Expected Values
    ///
    /// - **Empty triangulation**: χ = 0 (no simplices)
    /// - **D-dimensional manifold with boundary**: χ = 1
    /// - **Closed 2-manifold (sphere)**: χ = 2
    /// - **Closed 2-manifold with g handles (genus g)**: χ = 2 - 2g
    ///
    /// Currently assumes **manifold with boundary** (most common case for Delaunay triangulations).
    #[allow(clippy::cast_possible_wrap)] // Vertex/cell counts won't exceed i64 range in practice
    #[allow(clippy::too_many_lines)] // Comprehensive validation across dimensions
    fn validate_euler_characteristic(&self) -> Result<(), TriangulationValidationError>
    where
        K::Scalar: AddAssign + SubAssign + Sum,
    {
        use crate::core::util::{extract_edge_set, extract_facet_identifier_set};

        // Handle empty triangulation
        if self.number_of_vertices() == 0 {
            return Ok(()); // Empty triangulation is valid (χ = 0)
        }

        // Count simplices by dimension
        let n_vertices = self.number_of_vertices() as i64;
        let n_cells = self.number_of_cells() as i64;

        // Compute Euler characteristic based on dimension
        let dim = self.dim();

        match dim {
            -1 => Ok(()), // Empty triangulation
            0 => {
                // 0D: just isolated vertices, χ = V
                // For a valid triangulation, we expect χ = 1 (single vertex)
                let chi = n_vertices;
                if chi != 1 {
                    return Err(TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Euler characteristic mismatch for 0D: χ = {chi} (expected 1 for single vertex)"
                        ),
                    });
                }
                Ok(())
            }
            1 => {
                // 1D: χ = V - E
                // For a manifold with boundary (path): χ = 1
                let n_edges = n_cells; // In 1D, cells are edges
                let chi = n_vertices - n_edges;
                if chi != 1 {
                    return Err(TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Euler characteristic mismatch for 1D: χ = V - E = {n_vertices} - {n_edges} = {chi} (expected 1 for path)"
                        ),
                    });
                }
                Ok(())
            }
            2 => {
                // 2D: χ = V - E + F
                let n_edges = extract_edge_set(&self.tds)
                    .map_err(
                        |e| TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "Failed to extract edges for Euler characteristic: {e}"
                            ),
                        },
                    )?
                    .len() as i64;
                let n_faces = n_cells; // In 2D, cells are faces (triangles)

                let chi = n_vertices - n_edges + n_faces;

                // For a 2D manifold with boundary (disk): χ = 1
                // For a closed 2D manifold (sphere): χ = 2
                // We expect χ = 1 for typical Delaunay triangulations with convex hull boundary
                if chi != 1 && chi != 2 {
                    return Err(TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Euler characteristic mismatch for 2D: χ = V - E + F = {n_vertices} - {n_edges} + {n_faces} = {chi} (expected 1 for disk or 2 for sphere)"
                        ),
                    });
                }
                Ok(())
            }
            3 => {
                // 3D: χ = V - E + F - C
                let n_edges = extract_edge_set(&self.tds)
                    .map_err(
                        |e| TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "Failed to extract edges for Euler characteristic: {e}"
                            ),
                        },
                    )?
                    .len() as i64;
                let n_faces = extract_facet_identifier_set(&self.tds)
                    .map_err(
                        |e| TriangulationValidationError::InconsistentDataStructure {
                            message: format!(
                                "Failed to extract facets for Euler characteristic: {e}"
                            ),
                        },
                    )?
                    .len() as i64;

                let chi = n_vertices - n_edges + n_faces - n_cells;

                // For a 3D manifold with boundary (ball): χ = 1
                // For a closed 3D manifold (sphere): χ = 0 (but rare for Delaunay)
                if chi != 1 && chi != 0 {
                    return Err(TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Euler characteristic mismatch for 3D: χ = V - E + F - C = {n_vertices} - {n_edges} + {n_faces} - {n_cells} = {chi} (expected 1 for ball or 0 for sphere)"
                        ),
                    });
                }
                Ok(())
            }
            4..=7 => {
                // Higher dimensions: Use generalized Euler characteristic
                // For D-dimensional manifold with boundary: χ = 1
                // χ = Σ(-1)^d × N_d
                //
                // For now, we'll compute what we can and check if χ makes sense
                // Full implementation would require counting all d-simplices
                //
                // Simplified check: just ensure structure is consistent
                // TODO: Implement full d-simplex counting for dimensions 4-7
                Ok(())
            }
            _ => {
                // Unsupported dimension
                Err(TriangulationValidationError::InconsistentDataStructure {
                    message: format!(
                        "Euler characteristic validation not supported for dimension {dim}"
                    ),
                })
            }
        }
    }
}

// =============================================================================
// Geometric Operations (Requires Numeric Scalar Bounds)
// =============================================================================

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: AddAssign + SubAssign + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    /// Build initial D-simplex from D+1 vertices with degeneracy validation.
    ///
    /// This creates a Tds with a single cell containing all D+1 vertices,
    /// with no neighbor relationships (all boundary facets). The simplex is
    /// validated to ensure it is non-degenerate (vertices span full D-dimensional space).
    ///
    /// **Design Note**: This method uses `K::default()` to construct a kernel instance
    /// for the orientation test, relying on the design principle that kernels are stateless
    /// and reconstructible. If stateful kernels are introduced in the future, this method
    /// should accept an explicit kernel parameter instead.
    ///
    /// # Arguments
    /// - `vertices`: Exactly D+1 vertices to form the initial simplex
    ///
    /// # Returns
    /// A Tds containing one D-cell with all vertices, ready for incremental insertion.
    ///
    /// # Errors
    /// Returns error if:
    /// - Wrong number of vertices (must be exactly D+1)
    /// - Vertices are degenerate (collinear in 2D, coplanar in 3D, etc.)
    /// - Vertex or cell insertion fails
    /// - Duplicate UUIDs detected
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // Create a 2D triangle (initial simplex)
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
    /// assert_eq!(tds.number_of_vertices(), 3);
    /// assert_eq!(tds.number_of_cells(), 1);
    /// assert_eq!(tds.dim(), 2);
    ///
    /// // Error: wrong number of vertices (need exactly D+1)
    /// let bad_vertices = vec![vertex!([0.0, 0.0])];
    /// let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&bad_vertices);
    /// assert!(result.is_err());
    ///
    /// // Error: collinear points in 2D (degenerate simplex)
    /// let collinear = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([2.0, 0.0]),
    /// ];
    /// let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&collinear);
    /// assert!(result.is_err());
    /// ```
    pub fn build_initial_simplex(
        vertices: &[Vertex<K::Scalar, U, D>],
    ) -> Result<Tds<K::Scalar, U, V, D>, TriangulationConstructionError>
    where
        K::Scalar: CoordinateScalar,
    {
        if vertices.len() != D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: crate::core::cell::CellValidationError::InsufficientVertices {
                    actual: vertices.len(),
                    expected: D + 1,
                    dimension: D,
                },
            });
        }

        // Validate that the simplex is non-degenerate using orientation test
        // A degenerate simplex (collinear/coplanar) has zero orientation
        let kernel = K::default();

        // Collect points into stack-allocated buffer (at most 8 points for D ≤ 7)
        let points: SmallBuffer<Point<K::Scalar, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            vertices.iter().map(|v| *v.point()).collect();

        // Check orientation - zero (0) means degenerate
        // orientation() returns -1 (negative), 0 (degenerate), or +1 (positive)
        let orientation = kernel.orientation(&points[..]).map_err(|e| {
            TriangulationConstructionError::FailedToCreateCell {
                message: format!("Orientation test failed: {e}"),
            }
        })?;

        if orientation == 0 {
            return Err(TriangulationConstructionError::GeometricDegeneracy {
                message: format!(
                    "Degenerate initial simplex: vertices are collinear/coplanar in {}D space. \
                     The {} input vertices do not span a full {}-dimensional simplex. \
                     Provide non-degenerate vertices to create a valid triangulation.",
                    D,
                    D + 1,
                    D
                ),
            });
        }

        // Create empty Tds
        let mut tds = Tds::empty();

        // Insert all vertices and collect their keys
        let mut vertex_keys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
        for vertex in vertices {
            let vkey = tds.insert_vertex_with_mapping(*vertex)?;
            vertex_keys.push(vkey);
        }

        // Create single D-cell from all vertices
        // Note: Cell::new() handles vertex ordering/orientation internally
        let cell = Cell::new(vertex_keys, None).map_err(|e| {
            TriangulationConstructionError::FailedToCreateCell {
                message: format!("Failed to create initial simplex cell: {e}"),
            }
        })?;

        // Insert the cell
        let _cell_key = tds.insert_cell_with_mapping(cell)?;

        // Assign incident cells to vertices (each vertex points to this one cell)
        // This is required for proper Tds structure
        tds.assign_incident_cells()
            .map_err(TriangulationConstructionError::ValidationError)?;

        Ok(tds)
    }

    /// Insert a vertex into the triangulation using cavity-based algorithm.
    ///
    /// This is a generic insertion method that handles:
    /// - **Bootstrap (< D+1 vertices)**: Accumulates vertices without creating cells
    /// - **Initial simplex (D+1 vertices)**: Automatically builds the first D-cell
    /// - **Incremental (> D+1 vertices)**: Cavity-based insertion or hull extension
    ///
    /// # Arguments
    /// - `vertex`: The vertex to insert
    /// - `conflict_cells`: Optional conflict region (cells to be removed). Required for
    ///   interior points, not needed for exterior points (hull extension).
    /// - `hint`: Optional cell hint for point location (improves performance)
    ///
    /// # Algorithm
    /// 1. Insert vertex into Tds
    /// 2. Check vertex count:
    ///    - If < D+1: Return (bootstrap phase)
    ///    - If == D+1: Build initial simplex from all vertices
    ///    - If > D+1: Continue with steps 3-7
    /// 3. Locate cell containing the point
    /// 4. Handle location result:
    ///    - `InsideCell`: Use provided `conflict_cells` for cavity-based insertion
    ///    - `Outside`: Extend hull (no conflict cells needed)
    /// 5. Extract cavity boundary (if interior)
    /// 6. Fill cavity (create new cells)
    /// 7. Wire neighbors locally
    /// 8. Remove conflict cells (if interior)
    /// 9. Repair invalid facet sharing
    ///
    /// # Returns
    /// - `Ok(VertexKey)`: The key of the inserted vertex
    /// - New cell keys via the returned result (for hint caching at higher layers)
    ///
    /// # Errors
    /// Returns error if:
    /// - Duplicate UUID detected
    /// - Initial simplex construction fails
    /// - Point location fails
    /// - Interior point without `conflict_cells` parameter
    /// - Cavity operations fail
    /// - Degenerate location (`OnFacet`, `OnEdge`, `OnVertex`) - not yet implemented
    ///
    /// # Examples
    ///
    /// Bootstrap phase (first D+1 vertices build initial simplex automatically):
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// // Create empty 3D triangulation
    /// let mut tri: Triangulation<FastKernel<f64>, (), (), 3> =
    ///     Triangulation::new_empty(FastKernel::new());
    ///
    /// // Bootstrap phase: first 3 vertices accumulate without creating cells
    /// tri.insert(vertex!([0.0, 0.0, 0.0]), None, None).unwrap();
    /// assert_eq!(tri.number_of_vertices(), 1);
    /// assert_eq!(tri.number_of_cells(), 0); // No cells yet
    ///
    /// tri.insert(vertex!([1.0, 0.0, 0.0]), None, None).unwrap();
    /// assert_eq!(tri.number_of_vertices(), 2);
    /// assert_eq!(tri.number_of_cells(), 0); // Still no cells
    ///
    /// tri.insert(vertex!([0.0, 1.0, 0.0]), None, None).unwrap();
    /// assert_eq!(tri.number_of_vertices(), 3);
    /// assert_eq!(tri.number_of_cells(), 0); // Still no cells
    ///
    /// // 4th vertex triggers initial simplex creation
    /// let (_, hint) = tri.insert(vertex!([0.0, 0.0, 1.0]), None, None).unwrap();
    /// assert_eq!(tri.number_of_vertices(), 4);
    /// assert_eq!(tri.number_of_cells(), 1); // Initial simplex created!
    /// assert!(hint.is_some()); // Hint available for next insertion
    /// ```
    ///
    /// **Note**: For insertions beyond D+1 vertices, use `DelaunayTriangulation::insert()`
    /// instead, which handles conflict region computation automatically.
    pub fn insert(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_cells: Option<&CellKeyBuffer>,
        hint: Option<CellKey>,
    ) -> Result<(VertexKey, Option<CellKey>), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        // Use transactional insertion with perturbation retry, discard stats
        // 5 retry attempts: 1e-4, 1e-3, 1e-2, 2e-2, 5e-2 (up to 5% perturbation)
        let ((vkey, hint), _stats) = self.insert_transactional(vertex, conflict_cells, hint, 5)?;
        Ok((vkey, hint))
    }

    /// Insert a vertex and return statistics about the operation.
    ///
    /// This method returns detailed statistics about the insertion including:
    /// - Number of attempts (perturbation retries)
    /// - Whether the vertex was skipped
    /// - Number of cells removed during repair
    ///
    /// This is useful for testing, debugging, and understanding how the
    /// triangulation handles geometric degeneracies.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let (vkey, stats) = dt.insert_with_statistics(vertex, None, None)?;
    /// println!("Inserted with {} attempts, {} cells repaired",
    ///          stats.attempts, stats.cells_removed_during_repair);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails after all retry attempts.
    pub fn insert_with_statistics(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_cells: Option<&CellKeyBuffer>,
        hint: Option<CellKey>,
    ) -> Result<((VertexKey, Option<CellKey>), InsertionStatistics), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        // 5 retry attempts: 1e-4, 1e-3, 1e-2, 2e-2, 5e-2 (up to 5% perturbation)
        self.insert_transactional(vertex, conflict_cells, hint, 5)
    }

    /// Transactional insertion with automatic rollback and perturbation retry.
    ///
    /// This ensures the triangulation always remains in a valid state by:
    /// 1. Cloning TDS before each insertion attempt (snapshot)
    /// 2. Attempting insertion  
    /// 3. On failure: restore TDS from snapshot, perturb vertex, retry
    /// 4. If all attempts fail: restore TDS and return error
    ///
    /// This guarantees we transition from one valid manifold to another.
    #[allow(clippy::too_many_lines)]
    fn insert_transactional(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_cells: Option<&CellKeyBuffer>,
        hint: Option<CellKey>,
        max_perturbation_attempts: usize,
    ) -> Result<((VertexKey, Option<CellKey>), InsertionStatistics), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        use crate::core::vertex::VertexBuilder;
        use crate::geometry::point::Point;
        use crate::geometry::traits::coordinate::Coordinate;
        use num_traits::{Float, NumCast, One, Zero};

        let mut stats = InsertionStatistics::default();
        let original_coords = *vertex.point().coords();
        let mut current_vertex = vertex;

        for attempt in 0..=max_perturbation_attempts {
            stats.attempts = attempt + 1;

            // Apply perturbation for retry attempts
            if attempt > 0 {
                stats.used_perturbation = true;
                let mut perturbed_coords = original_coords;
                // Progressive perturbation schedule:
                // Attempt 1: 1e-4 (0.01%), Attempt 2: 1e-3 (0.1%), Attempt 3: 1e-2 (1%)
                // Attempt 4: 2e-2 (2%), Attempt 5: 5e-2 (5%)
                // This balances resolving degeneracies without introducing locate cycles
                let epsilon_value = match attempt {
                    1 => 1e-4,
                    2 => 1e-3,
                    3 => 1e-2,
                    4 => 2e-2,
                    _ => 5e-2, // 5% for attempt 5 and beyond
                };
                let epsilon = <K::Scalar as NumCast>::from(epsilon_value)
                    .expect("Failed to convert perturbation scale");

                for (idx, coord) in perturbed_coords.iter_mut().enumerate() {
                    let abs_coord = if *coord < K::Scalar::zero() {
                        -*coord
                    } else {
                        *coord
                    };
                    let perturbation_scale = epsilon * abs_coord.max(K::Scalar::one());
                    let perturbation = if (attempt + idx) % 2 == 0 {
                        perturbation_scale
                    } else {
                        -perturbation_scale
                    };
                    *coord += perturbation;
                }

                current_vertex = vertex.data.map_or_else(
                    || {
                        VertexBuilder::default()
                            .point(Point::new(perturbed_coords))
                            .build()
                            .expect("Failed to build perturbed vertex")
                    },
                    |data| {
                        VertexBuilder::default()
                            .point(Point::new(perturbed_coords))
                            .data(data)
                            .build()
                            .expect("Failed to build perturbed vertex")
                    },
                );
            }

            // Clone TDS for rollback (transactional semantics)
            let tds_snapshot = self.tds.clone();

            // Try insertion
            let result = self.try_insert_impl(current_vertex, conflict_cells, hint);

            match result {
                Ok((result, cells_removed)) => {
                    stats.cells_removed_during_repair = cells_removed;
                    stats.success = true;
                    if attempt > 0 {
                        eprintln!(
                            "Warning: Geometric degeneracy resolved via perturbation (attempt {attempt})"
                        );
                    }
                    return Ok((result, stats));
                }
                Err(e) => {
                    // Any error - rollback to snapshot
                    self.tds = tds_snapshot;

                    // Check if this is a retryable error (geometric degeneracy)
                    let is_retryable = e.is_retryable();

                    if is_retryable && attempt < max_perturbation_attempts {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "RETRYING: Attempt {} failed with: {e}. Applying perturbation...",
                            attempt + 1
                        );
                    } else if is_retryable {
                        stats.skipped = true;
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "SKIPPED: Could not insert vertex after {} attempts (perturbations up to {:.1}%). Last error: {e}. Vertex skipped to maintain manifold.",
                            max_perturbation_attempts + 1,
                            match max_perturbation_attempts {
                                0 => 0.0,
                                1 => 0.01,
                                2 => 0.1,
                                3 => 1.0,
                                4 => 2.0,
                                5 => 5.0,
                                _ => 10.0,
                            }
                        );
                        return Err(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        unreachable!("Loop should have returned in all cases");
    }

    /// Internal implementation of insert without retry logic.
    /// Returns the result and the number of cells removed during repair.
    ///
    /// Note: `conflict_cells` parameter is optional. If `None`, it will be computed automatically
    /// for interior points using `locate()` + `find_conflict_region()`.
    #[allow(clippy::too_many_lines)]
    fn try_insert_impl(
        &mut self,
        vertex: Vertex<K::Scalar, U, D>,
        conflict_cells: Option<&CellKeyBuffer>,
        hint: Option<CellKey>,
    ) -> Result<((VertexKey, Option<CellKey>), usize), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        use crate::core::algorithms::incremental_insertion::{
            extend_hull, fill_cavity, wire_cavity_neighbors,
        };
        use crate::core::algorithms::locate::{
            LocateResult, extract_cavity_boundary, find_conflict_region, locate,
        };

        // CRITICAL: Capture UUID and point BEFORE inserting into TDS
        // Rationale:
        // - inserted_uuid: Needed to remap v_key after TDS rebuild (lines 736-744)
        //   when building initial simplex. The rebuild replaces self.tds entirely,
        //   invalidating all previous VertexKeys.
        // - point: Needed for locate(), find_conflict_region(), and extend_hull() calls
        //   (lines 752, 760, 879, 895). After TDS rebuild, we cannot access the vertex
        //   via the old v_key, so we must have the point value captured.
        let inserted_uuid = vertex.uuid();
        let point = *vertex.point();

        // 1. Insert vertex into Tds
        let mut v_key = self.tds.insert_vertex_with_mapping(vertex)?;

        // 2. Check if we need to bootstrap the initial simplex
        let num_vertices = self.tds.number_of_vertices();

        if num_vertices < D + 1 {
            // Bootstrap phase: just accumulate vertices, no cells yet
            return Ok(((v_key, None), 0));
        } else if num_vertices == D + 1 {
            // Build initial simplex from all D+1 vertices
            let all_vertices: Vec<_> = self.tds.vertices().map(|(_, v)| *v).collect();
            let new_tds = Self::build_initial_simplex(&all_vertices).map_err(|e| {
                InsertionError::CavityFilling {
                    message: format!("Failed to build initial simplex: {e}"),
                }
            })?;

            // Replace empty TDS with simplex TDS (preserve kernel)
            self.tds = new_tds;

            // Re-map vertex key to the rebuilt TDS
            v_key = self
                .tds
                .vertex_key_from_uuid(&inserted_uuid)
                .ok_or_else(|| InsertionError::CavityFilling {
                    message: "Inserted vertex not found in rebuilt TDS".to_string(),
                })?;

            // Return first cell key for hint caching
            let first_cell = self.tds.cell_keys().next();
            return Ok(((v_key, first_cell), 0));
        }

        // 3. Locate containing cell (for vertex D+2 and beyond)
        let location = locate(&self.tds, &self.kernel, &point, hint)?;

        // 4. Compute conflict cells if not provided (for interior points)
        let conflict_cells_owned;
        let conflict_cells = match (location, conflict_cells) {
            (LocateResult::InsideCell(start_cell), None) => {
                // Interior point: compute conflict region automatically
                conflict_cells_owned =
                    find_conflict_region(&self.tds, &self.kernel, &point, start_cell)?;
                Some(&conflict_cells_owned)
            }
            (LocateResult::InsideCell(_), Some(cells)) => Some(cells), // Use provided
            (LocateResult::Outside, _) => None, // Hull extension doesn't need conflict region
            (location, _) => {
                // Degenerate locations (OnFacet, OnEdge, OnVertex)
                return Err(InsertionError::CavityFilling {
                    message: format!(
                        "Unhandled degenerate location: {location:?}. Point lies on facet/edge/vertex which is not yet supported."
                    ),
                });
            }
        };

        // 5. Handle different location results
        match location {
            LocateResult::InsideCell(_start_cell) => {
                // Interior vertex: use computed or provided conflict_cells
                let conflict_cells =
                    conflict_cells.expect("conflict_cells should be computed above");

                // 5. Extract cavity boundary
                let boundary_facets = extract_cavity_boundary(&self.tds, conflict_cells)?;

                // 6. Fill cavity BEFORE removing old cells
                let new_cells = fill_cavity(&mut self.tds, v_key, &boundary_facets)?;

                // 7. Wire neighbors (while both old and new cells exist)
                wire_cavity_neighbors(&mut self.tds, &new_cells, Some(conflict_cells))?;

                // 8. Remove conflict cells (now that new cells are wired up)
                let _removed_count = self.tds.remove_cells_by_keys(conflict_cells);

                // 9. Iteratively repair non-manifold topology until facet sharing is valid
                let mut total_removed = 0;
                #[allow(unused_variables)]
                for iteration in 0..MAX_REPAIR_ITERATIONS {
                    // Check for non-manifold issues in newly created cells (local scan)
                    // This keeps the repair O(k·D) where k is the cavity size, rather than O(N·D)
                    let cells_to_check: CellKeyBuffer = new_cells
                        .iter()
                        .copied()
                        .filter(|ck| self.tds.contains_cell(*ck))
                        .collect();

                    if let Some(issues) = self.detect_local_facet_issues(&cells_to_check)? {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "Repair iteration {}: {} over-shared facets detected, removing cells...",
                            iteration + 1,
                            issues.len()
                        );

                        let removed = self.repair_local_facet_issues(&issues)?;

                        // Early exit if repair made no progress
                        if removed == 0 {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "No cells removed in iteration {} - repair cannot make progress",
                                iteration + 1
                            );
                            return Err(InsertionError::TopologyValidation(
                                TriangulationValidationError::InconsistentDataStructure {
                                    message: format!(
                                        "Repair stalled: {} over-shared facets remain but no cells could be removed",
                                        issues.len()
                                    ),
                                },
                            ));
                        }

                        total_removed += removed;

                        #[cfg(debug_assertions)]
                        eprintln!("Removed {removed} cells (total: {total_removed})");

                        // Early exit if repair succeeded
                        if self.tds.validate_facet_sharing().is_ok() {
                            break;
                        }
                    } else {
                        // No more non-manifold issues - safe to rebuild neighbors
                        break;
                    }
                }

                // 10. Rebuild neighbor pointers now that topology is manifold
                #[cfg(debug_assertions)]
                eprintln!("After repair loop (interior): total_removed={total_removed}");

                if total_removed > 0 {
                    // Double-check that facet sharing is actually valid
                    let facet_valid = self.tds.validate_facet_sharing().is_ok();
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Before assign_neighbors (interior): facet_sharing_valid={facet_valid}, cells={}",
                        self.tds.number_of_cells()
                    );

                    if !facet_valid {
                        return Err(InsertionError::CavityFilling {
                            message: "Facet sharing still invalid after repairs - cannot safely rebuild neighbors".to_string(),
                        });
                    }

                    #[cfg(debug_assertions)]
                    {
                        // Check for duplicate cells
                        let cell_count = self.tds.number_of_cells();
                        let unique_cells: std::collections::HashSet<_> = self
                            .tds
                            .cells()
                            .map(|(_, c)| c.vertices().to_vec())
                            .collect();
                        if unique_cells.len() != cell_count {
                            eprintln!(
                                "WARNING: {} duplicate cells detected before assign_neighbors",
                                cell_count - unique_cells.len()
                            );
                        }
                    }

                    // Use repair_neighbor_pointers for surgical reconstruction
                    // This preserves existing correct pointers and only fixes broken ones
                    repair_neighbor_pointers(&mut self.tds).map_err(|e| {
                        InsertionError::CavityFilling {
                            message: format!("Failed to rebuild neighbors after repairs: {e}"),
                        }
                    })?;

                    #[cfg(debug_assertions)]
                    eprintln!(
                        "repair_neighbor_pointers completed (interior), assigning incident cells..."
                    );

                    self.tds.assign_incident_cells().map_err(|e| {
                        InsertionError::CavityFilling {
                            message: format!("Failed to assign incident cells after repairs: {e}"),
                        }
                    })?;

                    // Validate neighbor pointers by forcing a full facet walk (no hint).
                    // This uses locate's built-in cycle detection (tracks visited cells,
                    // errors after 10k steps). Cost: O(D·log(n)) for one locate call.
                    // If repair created cycles, there's a good chance they'll be encountered
                    // during the walk to locate the just-inserted point.
                    // Note: Debug builds also run validate_no_neighbor_cycles() in
                    // repair_neighbor_pointers() for comprehensive O(n·D) BFS validation.
                    let _ = locate(&self.tds, &self.kernel, &point, None)?;
                } else {
                    #[cfg(debug_assertions)]
                    eprintln!("No cells removed (interior), skipping neighbor rebuild");
                }

                // Return vertex key and hint for next insertion
                let hint = new_cells
                    .iter()
                    .copied()
                    .find(|ck| self.tds.contains_cell(*ck));
                Ok(((v_key, hint), total_removed))
            }
            LocateResult::Outside => {
                // Exterior vertex: extend convex hull
                let new_cells = extend_hull(&mut self.tds, &self.kernel, v_key, &point)?;

                // Iteratively repair non-manifold topology until facet sharing is valid
                let mut total_removed = 0;
                #[allow(unused_variables)]
                for iteration in 0..MAX_REPAIR_ITERATIONS {
                    // Check for non-manifold issues in newly created hull cells (local scan)
                    // This keeps the repair O(k·D) where k is the number of new hull cells, rather than O(N·D)
                    let cells_to_check: CellKeyBuffer = new_cells
                        .iter()
                        .copied()
                        .filter(|ck| self.tds.contains_cell(*ck))
                        .collect();

                    if let Some(issues) = self.detect_local_facet_issues(&cells_to_check)? {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "Hull extension repair iteration {}: {} over-shared facets detected, removing cells...",
                            iteration + 1,
                            issues.len()
                        );

                        let removed = self.repair_local_facet_issues(&issues)?;

                        // Early exit if repair made no progress
                        if removed == 0 {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "No cells removed in iteration {} - repair cannot make progress",
                                iteration + 1
                            );
                            return Err(InsertionError::TopologyValidation(
                                TriangulationValidationError::InconsistentDataStructure {
                                    message: format!(
                                        "Hull extension repair stalled: {} over-shared facets remain but no cells could be removed",
                                        issues.len()
                                    ),
                                },
                            ));
                        }

                        total_removed += removed;

                        #[cfg(debug_assertions)]
                        eprintln!("Removed {removed} cells (total: {total_removed})");

                        // Early exit if repair succeeded
                        if self.tds.validate_facet_sharing().is_ok() {
                            break;
                        }
                    } else {
                        // No more non-manifold issues - safe to rebuild neighbors
                        break;
                    }
                }

                // Rebuild neighbor pointers now that topology is manifold
                if total_removed > 0 {
                    // Double-check that facet sharing is actually valid
                    let facet_valid = self.tds.validate_facet_sharing().is_ok();
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Before repair_neighbor_pointers: facet_sharing_valid={facet_valid}, cells={}",
                        self.tds.number_of_cells()
                    );

                    if !facet_valid {
                        return Err(InsertionError::CavityFilling {
                            message: "Facet sharing still invalid after repairs - cannot safely rebuild neighbors".to_string(),
                        });
                    }

                    // Use repair_neighbor_pointers for surgical reconstruction
                    // This preserves existing correct pointers and only fixes broken ones
                    repair_neighbor_pointers(&mut self.tds).map_err(|e| {
                        InsertionError::CavityFilling {
                            message: format!("Failed to rebuild neighbors after repairs: {e}"),
                        }
                    })?;

                    #[cfg(debug_assertions)]
                    eprintln!("repair_neighbor_pointers completed, assigning incident cells...");

                    self.tds.assign_incident_cells().map_err(|e| {
                        InsertionError::CavityFilling {
                            message: format!("Failed to assign incident cells after repairs: {e}"),
                        }
                    })?;
                }

                // Return vertex key and hint for next insertion
                let hint = new_cells
                    .iter()
                    .copied()
                    .find(|ck| self.tds.contains_cell(*ck));
                Ok(((v_key, hint), total_removed))
            }
            LocateResult::OnFacet(_, _) | LocateResult::OnEdge(_) | LocateResult::OnVertex(_) => {
                // These degenerate cases are already handled at lines 772-779 above,
                // so this arm is unreachable. Included only for exhaustiveness.
                unreachable!("Degenerate locations should have been handled earlier")
            }
        }
    }

    /// Removes a vertex and retriangulates the resulting cavity using fan triangulation.
    ///
    /// This operation maintains topological consistency by:
    /// 1. Finding all cells containing the vertex
    /// 2. Removing those cells (creating a cavity)
    /// 3. Extracting the cavity boundary facets
    /// 4. Filling the cavity with a fan triangulation (pick apex, connect to all boundary facets)
    /// 5. Wiring neighbors to maintain consistency
    /// 6. Removing the vertex itself
    ///
    /// **Fan Triangulation**: The cavity is filled by picking one boundary vertex as an apex
    /// and connecting it to all boundary facets. This is fast and maintains all topological
    /// invariants, though it may create poorly-shaped cells in some cases.
    ///
    /// # Arguments
    ///
    /// * `vertex` - Reference to the vertex to remove
    ///
    /// # Returns
    ///
    /// The number of cells that were removed along with the vertex.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Cavity extraction fails
    /// - Fan triangulation fails
    /// - Neighbor wiring fails
    /// - Vertex removal fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    ///
    /// let vertices = [
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    ///     vertex!([1.0, 1.0]),
    /// ];
    /// let mut dt = DelaunayTriangulation::new(&vertices).unwrap();
    ///
    /// // Remove a vertex - cavity is automatically retriangulated
    /// let vertex_to_remove = dt.vertices().next().unwrap().1.clone();
    /// let cells_removed = dt.remove_vertex(&vertex_to_remove).unwrap();
    /// assert!(dt.is_valid().is_ok());
    /// ```
    pub fn remove_vertex(
        &mut self,
        vertex: &Vertex<K::Scalar, U, D>,
    ) -> Result<usize, crate::core::triangulation_data_structure::TriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        use crate::core::algorithms::incremental_insertion::wire_cavity_neighbors;
        use crate::core::algorithms::locate::extract_cavity_boundary;

        // Find the vertex key
        let Some(vertex_key) = self.tds.vertex_key_from_uuid(&vertex.uuid()) else {
            return Ok(0); // Vertex not found, nothing to remove
        };

        // Collect all cells containing this vertex by scanning all cells
        let cells_to_remove: CellKeyBuffer = self
            .tds
            .cells()
            .filter_map(|(cell_key, cell)| {
                if cell.vertices().contains(&vertex_key) {
                    Some(cell_key)
                } else {
                    None
                }
            })
            .collect();

        if cells_to_remove.is_empty() {
            // Vertex exists but has no incident cells - use Tds removal
            return self.tds.remove_vertex(vertex);
        }

        // Extract cavity boundary BEFORE removing cells
        let boundary_facets = extract_cavity_boundary(&self.tds, &cells_to_remove)
            .map_err(|e| crate::core::triangulation_data_structure::TriangulationValidationError::InconsistentDataStructure {
                message: format!("Failed to extract cavity boundary: {e}"),
            })?;

        // If boundary is empty, we're removing the entire triangulation
        if boundary_facets.is_empty() {
            // Use Tds removal for empty boundary case
            return self.tds.remove_vertex(vertex);
        }

        // Pick apex vertex for fan triangulation (first vertex of first boundary facet)
        let apex_vertex_key = self.pick_fan_apex(&boundary_facets)
            .ok_or_else(|| crate::core::triangulation_data_structure::TriangulationValidationError::InconsistentDataStructure {
                message: "Failed to find apex vertex for fan triangulation".to_string(),
            })?;

        // Fill cavity with fan triangulation BEFORE removing old cells
        // Use fan triangulation that skips boundary facets which already include the apex
        let new_cells = self
            .fan_fill_cavity(apex_vertex_key, &boundary_facets)
            .map_err(|e| crate::core::triangulation_data_structure::TriangulationValidationError::InconsistentDataStructure {
                message: format!("Fan triangulation failed: {e}"),
            })?;

        // Wire neighbors for the new cells (while both old and new cells exist)
        wire_cavity_neighbors(&mut self.tds, &new_cells, Some(&cells_to_remove)).map_err(|e| {
            TriangulationValidationError::InconsistentDataStructure {
                message: format!("Neighbor wiring failed: {e}"),
            }
        })?;

        // Remove the cells containing the vertex (now that new cells are wired up)
        // Note: remove_cells_by_keys() automatically clears neighbor pointers in surviving
        // cells that reference removed cells (sets them to None/boundary)
        let mut cells_removed = self.tds.remove_cells_by_keys(&cells_to_remove);

        // Validate facet topology for newly created cells (O(k*D) localized check)
        if let Some(issues) = self.detect_local_facet_issues(&new_cells)? {
            #[cfg(debug_assertions)]
            eprintln!(
                "Warning: {} over-shared facets detected after vertex removal, repairing...",
                issues.len()
            );
            let removed = self.repair_local_facet_issues(&issues)?;
            cells_removed += removed;
            #[cfg(debug_assertions)]
            eprintln!("Repaired by removing {removed} additional cells");

            // Repair neighbor pointers after removing additional cells
            // This ensures neighbor consistency after repair operations
            if removed > 0 {
                use crate::core::algorithms::incremental_insertion::repair_neighbor_pointers;
                repair_neighbor_pointers(&mut self.tds).map_err(|e| {
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!("Neighbor repair after facet issue repair failed: {e}"),
                    }
                })?;
            }
        }

        // Rebuild vertex-cell incidence for all vertices
        self.tds.assign_incident_cells()?;

        // Remove the vertex using Tds method (handles internal bookkeeping)
        self.tds.remove_vertex(vertex)?;

        Ok(cells_removed)
    }

    /// Pick an apex vertex for fan triangulation.
    ///
    /// Selects the first vertex from the first boundary facet as the apex.
    /// The fan will connect this apex to all boundary facets.
    ///
    /// # Arguments
    ///
    /// * `boundary_facets` - The cavity boundary facets
    ///
    /// # Returns
    ///
    /// The vertex key to use as apex, or None if no suitable vertex found.
    fn pick_fan_apex(
        &self,
        boundary_facets: &[crate::core::facet::FacetHandle],
    ) -> Option<VertexKey>
    where
        K::Scalar: CoordinateScalar,
    {
        // Get first boundary facet
        let first_facet = boundary_facets.first()?;
        let cell = self.tds.get_cell(first_facet.cell_key())?;

        // Get the first vertex from this facet (any vertex that's not the opposite one)
        let facet_idx = <usize as From<_>>::from(first_facet.facet_index());
        cell.vertices()
            .iter()
            .enumerate()
            .find(|(i, _)| *i != facet_idx)
            .map(|(_, &vkey)| vkey)
    }

    /// Fan-specific cavity fill: connect an existing apex vertex to boundary facets
    /// that do not already include the apex. This avoids creating degenerate cells
    /// with duplicate vertices when the apex lies on a boundary facet.
    fn fan_fill_cavity(
        &mut self,
        apex_vertex_key: VertexKey,
        boundary_facets: &[crate::core::facet::FacetHandle],
    ) -> Result<CellKeyBuffer, InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        let mut new_cells = CellKeyBuffer::new();

        for facet_handle in boundary_facets {
            let boundary_cell = self.tds.get_cell(facet_handle.cell_key()).ok_or_else(|| {
                InsertionError::CavityFilling {
                    message: format!(
                        "Boundary facet cell {:?} not found",
                        facet_handle.cell_key()
                    ),
                }
            })?;

            let facet_idx = <usize as From<_>>::from(facet_handle.facet_index());

            // Gather facet vertices (all except the opposite vertex)
            let mut facet_vertices = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
            for (i, &vkey) in boundary_cell.vertices().iter().enumerate() {
                if i != facet_idx {
                    facet_vertices.push(vkey);
                }
            }

            // Skip facets that already contain the apex to avoid duplicate vertices
            if facet_vertices.contains(&apex_vertex_key) {
                continue;
            }

            // Build new cell vertices = facet_vertices + apex
            let mut new_cell_vertices = facet_vertices;
            new_cell_vertices.push(apex_vertex_key);

            // Create and insert the new cell
            let new_cell =
                Cell::new(new_cell_vertices, None).map_err(|e| InsertionError::CavityFilling {
                    message: format!("Failed to create cell: {e}"),
                })?;
            let cell_key = self.tds.insert_cell_with_mapping(new_cell).map_err(|e| {
                InsertionError::CavityFilling {
                    message: format!("Failed to insert cell: {e}"),
                }
            })?;

            new_cells.push(cell_key);
        }

        if new_cells.is_empty() {
            return Err(InsertionError::CavityFilling {
                message: "Fan triangulation produced no cells (apex on all boundary facets?)"
                    .to_string(),
            });
        }

        Ok(new_cells)
    }

    // Phase 2 TODO: Add geometric operations using kernel predicates
    // - locate(point) - point location using facet walking

    /// Detects over-shared facets within a specific set of cells (localized check).
    ///
    /// This is an **O(k * D)** operation where k = number of cells to check,
    /// unlike global validation which is O(N * D) for the entire triangulation.
    ///
    /// # Performance
    ///
    /// - **Complexity**: O(k * D) where k = `cells.len()`, D = dimension
    /// - **Use case**: Detect issues in newly created cells after insertion/removal
    /// - **Comparison**: Global detection is O(N * D) where N = total cells
    ///
    /// # Arguments
    ///
    /// * `cells` - Keys of cells to check (typically newly created cells)
    ///
    /// # Returns
    ///
    /// `Ok(None)` if all facets are valid (≤2 cells per facet).
    /// `Ok(Some(issues))` if over-shared facets are detected, where issues is a map
    /// from facet hash to the cells sharing that facet.
    ///
    /// # Errors
    ///
    /// Returns error if cells cannot be accessed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // After inserting new cells via cavity filling, check for topology issues
    /// let new_cells = fill_cavity(&mut tri.tds, new_vertex_key, &boundary_facets)?;
    ///
    /// // Detect any over-shared facets (> 2 cells per facet)
    /// if let Some(issues) = tri.detect_local_facet_issues(&new_cells)? {
    ///     // Repair by removing lower-quality cells
    ///     let removed = tri.repair_local_facet_issues(&issues)?;
    ///     eprintln!("Warning: Repaired {} over-shared facets", issues.len());
    /// }
    /// ```
    pub fn detect_local_facet_issues(
        &self,
        cells: &[CellKey],
    ) -> Result<Option<FacetIssuesMap>, TriangulationValidationError>
    where
        K::Scalar: CoordinateScalar,
    {
        // Build facet map for ONLY the specified cells
        // This is O(k * D) instead of O(N * D)
        let mut facet_to_cells = FacetIssuesMap::default();

        // Index facets from the specified cells
        for &cell_key in cells {
            let Some(cell) = self.tds.get_cell(cell_key) else {
                continue; // Cell was removed, skip
            };

            // For each facet of this cell
            for facet_idx in 0..cell.number_of_vertices() {
                // Compute facet hash from sorted vertex keys
                let mut facet_vkeys = SmallBuffer::<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>::new();
                for (i, &vkey) in cell.vertices().iter().enumerate() {
                    if i != facet_idx {
                        facet_vkeys.push(vkey);
                    }
                }
                facet_vkeys.sort_unstable();

                // Hash the facet
                let mut hasher = FastHasher::default();
                for &vkey in &facet_vkeys {
                    vkey.hash(&mut hasher);
                }
                let facet_hash = hasher.finish();

                // Track this cell/facet pair
                let facet_idx_u8 = u8::try_from(facet_idx).map_err(|_| {
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Facet index {facet_idx} exceeds u8::MAX (dimension too high)"
                        ),
                    }
                })?;
                facet_to_cells
                    .entry(facet_hash)
                    .or_insert_with(SmallBuffer::new)
                    .push((cell_key, facet_idx_u8));
            }
        }

        // Filter to only over-shared facets (> 2 cells) in a single pass
        facet_to_cells.retain(|_, cell_facet_pairs| cell_facet_pairs.len() > 2);

        if facet_to_cells.is_empty() {
            Ok(None)
        } else {
            Ok(Some(facet_to_cells))
        }
    }

    /// Repairs over-shared facets by removing lower-quality cells.
    ///
    /// Uses geometric quality metrics (`radius_ratio`) to select which cells to keep
    /// when a facet is shared by more than 2 cells. UUID ordering is used as a tie-breaker
    /// when cells have equal quality. Errors if quality computation or conversion fails.
    ///
    /// # Performance
    ///
    /// - **Complexity**: O(m * q) where m = number of problematic facets, q = quality computation cost
    /// - **Localized**: Only processes cells involved in detected issues
    ///
    /// # Arguments
    ///
    /// * `issues` - Detected facet issues map from `detect_local_facet_issues()`
    ///
    /// # Returns
    ///
    /// Number of cells removed during repair.
    ///
    /// # Errors
    ///
    /// Returns error if quality evaluation or facet bookkeeping fails while
    /// selecting cells to remove. This function itself does not rebuild neighbors;
    /// callers are responsible for repairing or validating topology after removal
    /// (e.g., via `repair_neighbor_pointers` or a validation pass).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // After detecting issues, repair them locally
    /// if let Some(issues) = tri.detect_local_facet_issues(&new_cells)? {
    ///     let removed = tri.repair_local_facet_issues(&issues)?;
    ///     println!("Repaired {} over-shared facets by removing {} cells",
    ///              issues.len(), removed);
    /// }
    /// ```
    pub fn repair_local_facet_issues(
        &mut self,
        issues: &FacetIssuesMap,
    ) -> Result<usize, TriangulationValidationError>
    where
        K::Scalar: CoordinateScalar + Div<Output = K::Scalar>,
    {
        let mut cells_to_remove = CellKeySet::default();

        // For each over-shared facet, select cells to remove
        for cell_facet_pairs in issues.values() {
            let involved_cells: Vec<CellKey> = cell_facet_pairs.iter().map(|(ck, _)| *ck).collect();

            // Compute quality for each cell - propagate errors from quality evaluation
            let mut cell_qualities: Vec<(CellKey, f64, Uuid)> = Vec::new();
            for &cell_key in &involved_cells {
                let cell = self.tds.get_cell(cell_key).ok_or_else(|| {
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!("Cell {cell_key:?} not found during facet repair"),
                    }
                })?;
                let uuid = cell.uuid();

                // Propagate quality evaluation errors
                let ratio = radius_ratio(self, cell_key).map_err(|e| {
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!("Quality evaluation failed for cell {cell_key:?}: {e}"),
                    }
                })?;
                let ratio_f64 = safe_scalar_to_f64(ratio).map_err(|_| {
                    TriangulationValidationError::InconsistentDataStructure {
                        message: format!("Quality ratio conversion failed for cell {cell_key:?}"),
                    }
                })?;

                if ratio_f64.is_finite() {
                    cell_qualities.push((cell_key, ratio_f64, uuid));
                } else {
                    return Err(TriangulationValidationError::InconsistentDataStructure {
                        message: format!(
                            "Non-finite quality ratio {ratio_f64} for cell {cell_key:?}"
                        ),
                    });
                }
            }

            // Quality-based selection: keep 2 best, remove rest
            // Note: cell_qualities always has all involved_cells at this point since
            // any quality computation failure results in an early error return above
            cell_qualities.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(CmpOrdering::Equal)
                    .then_with(|| a.2.cmp(&b.2))
            });

            // Mark cells beyond the top 2 for removal
            for (cell_key, _, _) in cell_qualities.iter().skip(2) {
                if self.tds.contains_cell(*cell_key) {
                    cells_to_remove.insert(*cell_key);
                }
            }
        }

        // Remove the selected cells - do NOT rebuild neighbors here
        // Neighbor wiring should happen AFTER all non-manifold issues are resolved
        let to_remove: Vec<CellKey> = cells_to_remove.into_iter().collect();
        let removed_count = self.tds.remove_cells_by_keys(&to_remove);

        Ok(removed_count)
    }

    /// Attempts to fix invalid facet sharing by removing problematic cells using geometric quality metrics.
    ///
    /// Deprecated: This performs a global O(N·D) validation/repair pass. For incremental
    /// operations, use the localized O(k·D) approach instead:
    ///
    /// ```ignore
    /// // After insertion/removal, collect the cells you touched
    /// let affected_cells: Vec<CellKey> = /* cells that were just modified */;
    /// // Detect issues only in the affected region
    /// if let Some(issues) = triangulation.detect_local_facet_issues(&affected_cells) {
    ///     // Repair only that subset
    ///     triangulation.repair_local_facet_issues(&issues)?;
    /// }
    /// ```
    ///
    /// This is a **best-effort repair mechanism** that may not fully resolve all facet sharing
    /// violations in extreme cases. The method iterates up to 10 times, removing cells around
    /// over-shared facets using quality-based selection (`radius_ratio`) and UUID tie-breaking.
    ///
    /// This method belongs in the Triangulation layer (not Tds) because it uses geometric
    /// quality metrics to select which cells to keep when a facet is shared by more than 2 cells.
    ///
    /// # Returns
    ///
    /// Number of cells removed during the repair attempt.
    ///
    /// Important: `Ok(n)` does not guarantee that all facet sharing violations are resolved.
    /// This method may return successfully even if violations persist after reaching the
    /// iteration limit, making no progress, or when internal repair steps (neighbor assignment,
    /// duplicate removal) fail. Call `validate_facet_sharing()` if you require a post-check.
    ///
    /// # Errors
    ///
    /// Returns error if the facet map cannot be built (indicating structural corruption).
    ///
    /// Note: Some internal repair failures (duplicate removal, neighbor assignment) are
    /// logged in debug builds but do not cause this method to return an error.
    #[deprecated(
        since = "0.5.5",
        note = "Use detect_local_facet_issues() + repair_local_facet_issues() for localized O(k·D) repair."
    )]
    #[allow(clippy::too_many_lines)]
    pub fn fix_invalid_facet_sharing(&mut self) -> Result<usize, TriangulationValidationError>
    where
        K::Scalar: crate::geometry::traits::coordinate::CoordinateScalar + Div<Output = K::Scalar>,
    {
        // Safety limit for iteration count to prevent infinite loops
        const MAX_FIX_FACET_ITERATIONS: usize = 10;

        // First check if there are any facet sharing issues
        if self.tds.validate_facet_sharing().is_ok() {
            return Ok(0);
        }

        let mut total_removed = 0;

        for _iteration in 0..MAX_FIX_FACET_ITERATIONS {
            // Check if facet sharing is already valid
            if self.tds.validate_facet_sharing().is_ok() {
                return Ok(total_removed);
            }

            // Build facet map
            let facet_to_cells = self.tds.build_facet_to_cells_map()?;
            let mut cells_to_remove: CellKeySet = CellKeySet::default();

            // Find facets shared by more than 2 cells
            for (_facet_key, cell_facet_pairs) in facet_to_cells {
                if cell_facet_pairs.len() > 2 {
                    let first_cell_key = cell_facet_pairs[0].cell_key();
                    let first_facet_index = cell_facet_pairs[0].facet_index();

                    if self.tds.contains_cell(first_cell_key) {
                        let vertices = self.tds.get_cell_vertices(first_cell_key)?;
                        let mut facet_vertices = Vec::with_capacity(vertices.len() - 1);
                        let idx: usize = first_facet_index.into();
                        for (i, &key) in vertices.iter().enumerate() {
                            if i != idx {
                                facet_vertices.push(key);
                            }
                        }

                        let facet_vertices_set: VertexKeySet =
                            facet_vertices.iter().copied().collect();

                        let mut valid_cells = ValidCellsBuffer::new();
                        for facet_handle in &cell_facet_pairs {
                            let cell_key = facet_handle.cell_key();
                            if self.tds.contains_cell(cell_key) {
                                let cell_vertices_vec = self.tds.get_cell_vertices(cell_key)?;
                                let cell_vertices: VertexKeySet =
                                    cell_vertices_vec.iter().copied().collect();

                                if facet_vertices_set.is_subset(&cell_vertices) {
                                    valid_cells.push(cell_key);
                                } else {
                                    cells_to_remove.insert(cell_key);
                                }
                            }
                        }

                        // Quality-based selection when > 2 valid cells
                        if valid_cells.len() > 2 {
                            // Compute quality for each cell
                            let mut cell_qualities: Vec<(CellKey, f64, Uuid)> = valid_cells
                                .iter()
                                .filter_map(|&cell_key| {
                                    let quality_result = radius_ratio(self, cell_key);
                                    let uuid = self.tds.get_cell(cell_key)?.uuid();

                                    quality_result.ok().and_then(|ratio| {
                                        safe_scalar_to_f64(ratio)
                                            .ok()
                                            .filter(|r| r.is_finite())
                                            .map(|r| (cell_key, r, uuid))
                                    })
                                })
                                .collect();

                            // Use quality when available, fall back to UUID
                            if cell_qualities.len() == valid_cells.len()
                                && cell_qualities.len() >= 2
                            {
                                // Pure quality-based selection
                                cell_qualities.sort_unstable_by(|a, b| {
                                    a.1.partial_cmp(&b.1)
                                        .unwrap_or(CmpOrdering::Equal)
                                        .then_with(|| a.2.cmp(&b.2))
                                });

                                // Keep the two best quality cells
                                for (cell_key, _, _) in cell_qualities.iter().skip(2) {
                                    if self.tds.contains_cell(*cell_key) {
                                        cells_to_remove.insert(*cell_key);
                                    }
                                }
                            } else if !cell_qualities.is_empty() && cell_qualities.len() >= 2 {
                                // Hybrid: prefer scored cells
                                let scored_keys: CellKeySet =
                                    cell_qualities.iter().map(|(k, _, _)| *k).collect();

                                cell_qualities.sort_unstable_by(|a, b| {
                                    a.1.partial_cmp(&b.1)
                                        .unwrap_or(CmpOrdering::Equal)
                                        .then_with(|| a.2.cmp(&b.2))
                                });

                                let mut keep: Vec<CellKey> =
                                    cell_qualities.iter().take(2).map(|(k, _, _)| *k).collect();

                                // Fill with unscored if needed
                                if keep.len() < 2 {
                                    let mut unscored: Vec<CellKey> = valid_cells
                                        .iter()
                                        .copied()
                                        .filter(|k| !scored_keys.contains(k))
                                        .collect();
                                    unscored.sort_unstable_by(|a, b| {
                                        let uuid_a =
                                            self.tds.get_cell(*a).map(super::cell::Cell::uuid);
                                        let uuid_b =
                                            self.tds.get_cell(*b).map(super::cell::Cell::uuid);
                                        uuid_a.cmp(&uuid_b)
                                    });
                                    keep.extend(unscored.into_iter().take(2 - keep.len()));
                                }

                                for &cell_key in &valid_cells {
                                    if !keep.contains(&cell_key) && self.tds.contains_cell(cell_key)
                                    {
                                        cells_to_remove.insert(cell_key);
                                    }
                                }
                            } else {
                                // UUID fallback
                                valid_cells.sort_unstable_by(|a, b| {
                                    let uuid_a = self.tds.get_cell(*a).map(super::cell::Cell::uuid);
                                    let uuid_b = self.tds.get_cell(*b).map(super::cell::Cell::uuid);
                                    uuid_a.cmp(&uuid_b)
                                });
                                for &cell_key in valid_cells.iter().skip(2) {
                                    if self.tds.contains_cell(cell_key) {
                                        cells_to_remove.insert(cell_key);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let to_remove: Vec<CellKey> = cells_to_remove.into_iter().collect();

            // Remove cells
            let actually_removed = self.tds.remove_cells_by_keys(&to_remove);

            // Clean up duplicates
            let Ok(duplicate_cells_removed) = self.tds.remove_duplicate_cells() else {
                #[cfg(debug_assertions)]
                eprintln!(
                    "Warning: remove_duplicate_cells failed during facet repair (removed {actually_removed} cells)"
                );
                total_removed += actually_removed;
                continue;
            };

            // Skip neighbor repair during iterations - will do once at end

            let removed_this_iteration = actually_removed + duplicate_cells_removed;
            total_removed += removed_this_iteration;

            if removed_this_iteration == 0 || self.tds.validate_facet_sharing().is_ok() {
                break;
            }
        }

        // After all iterations complete, rebuild neighbor relationships once
        // Use repair_neighbor_pointers which tolerates non-manifold topology
        if total_removed > 0 {
            #[cfg(debug_assertions)]
            eprintln!("Repairs complete, rebuilding all neighbor pointers...");

            repair_neighbor_pointers(&mut self.tds).map_err(|e| {
                TriangulationValidationError::InconsistentDataStructure {
                    message: format!("repair_neighbor_pointers failed after repairs: {e}"),
                }
            })?;

            self.tds.assign_incident_cells()?;
        }

        Ok(total_removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vertex::VertexBuilder;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;

    /// Macro to generate `build_initial_simplex` tests across dimensions.
    ///
    /// This macro generates tests that verify `build_initial_simplex` by:
    /// 1. Creating D+1 affinely independent vertices
    /// 2. Calling `build_initial_simplex` directly
    /// 3. Verifying the Tds has correct structure (vertices, cells, dimension)
    ///
    /// # Usage
    /// ```ignore
    /// test_build_initial_simplex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    /// ```
    macro_rules! test_build_initial_simplex {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_build_initial_simplex_ $dim d>]() {
                    // Build initial simplex (D+1 vertices)
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let expected_vertices = vertices.len();
                    assert_eq!(expected_vertices, $dim + 1,
                        "Test must provide exactly D+1 vertices for {}D simplex", $dim);

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();

                    // Verify structure
                    assert_eq!(tds.number_of_vertices(), expected_vertices,
                        "{}D: Expected {} vertices", $dim, expected_vertices);
                    assert_eq!(tds.number_of_cells(), 1,
                        "{}D: Expected 1 cell", $dim);
                    assert_eq!(tds.dim(), $dim as i32,
                        "{}D: Expected dimension {}", $dim, $dim);

                    // Verify all vertices are present
                    assert_eq!(tds.vertices().count(), expected_vertices,
                        "{}D: All vertices should be in Tds", $dim);

                    // Verify the single cell has correct number of vertices
                    let (_, cell) = tds.cells().next()
                        .expect(&format!("{}D: Should have exactly one cell", $dim));
                    assert_eq!(cell.number_of_vertices(), expected_vertices,
                        "{}D: Cell should have {} vertices", $dim, expected_vertices);

                    // Verify incident cells are assigned
                    for (_, vertex) in tds.vertices() {
                        assert!(vertex.incident_cell.is_some(),
                            "{}D: All vertices should have incident cell assigned", $dim);
                    }

                    // Verify initial simplex has no neighbors (all boundary facets)
                    if let Some(neighbors) = cell.neighbors() {
                        assert!(neighbors.iter().all(|n| n.is_none()),
                            "{}D: Initial simplex should have no neighbors (all boundary)", $dim);
                    }
                }
            }
        };
    }

    // 2D: Triangle
    test_build_initial_simplex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

    // 3D: Tetrahedron
    test_build_initial_simplex!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );

    // 4D: 4-simplex
    test_build_initial_simplex!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );

    // 5D: 5-simplex
    test_build_initial_simplex!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    /// Macro to generate `validate_manifold` tests across dimensions.
    ///
    /// This macro generates tests that verify manifold validation by:
    /// 1. Creating a Delaunay triangulation from D+1 affinely independent vertices
    /// 2. Calling `validate_manifold()` on the triangulation
    /// 3. Verifying that the validation passes
    ///
    /// # Usage
    /// ```ignore
    /// test_validate_manifold!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    /// ```
    macro_rules! test_validate_manifold {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_validate_manifold_ $dim d>]() {
                    use crate::core::delaunay_triangulation::DelaunayTriangulation;

                    // Build triangulation from D+1 vertices (initial simplex)
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let expected_vertices = vertices.len();
                    assert_eq!(expected_vertices, $dim + 1,
                        "Test must provide exactly D+1 vertices for {}D simplex", $dim);

                    let dt = DelaunayTriangulation::new(&vertices)
                        .expect(&format!("Failed to create {}D triangulation", $dim));
                    let tri = dt.triangulation();

                    // Validate manifold properties
                    let result = tri.validate_manifold();
                    assert!(
                        result.is_ok(),
                        "{}D: Simple simplex should be a valid manifold. Error: {:?}",
                        $dim,
                        result.err()
                    );

                    // Also verify basic properties
                    assert_eq!(tri.number_of_vertices(), expected_vertices,
                        "{}D: Should have {} vertices", $dim, expected_vertices);
                    assert_eq!(tri.number_of_cells(), 1,
                        "{}D: Should have exactly 1 cell", $dim);
                }
            }
        };
    }

    // 2D: Triangle manifold
    test_validate_manifold!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

    // 3D: Tetrahedron manifold
    test_validate_manifold!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );

    // 4D: 4-simplex manifold
    test_validate_manifold!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );

    // 5D: 5-simplex manifold
    test_validate_manifold!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    #[test]
    fn test_validate_manifold_empty() {
        // Empty triangulation should pass manifold validation
        let tri: Triangulation<FastKernel<f64>, (), (), 3> =
            Triangulation::new_empty(FastKernel::new());

        assert!(
            tri.validate_manifold().is_ok(),
            "Empty triangulation should be a valid (empty) manifold"
        );
    }

    // NOTE: Tests for multiple-cell triangulations with interior points are omitted
    // because they may fail validation due to Issue #120 (rare Delaunay property violations
    // in near-degenerate configurations). The validate_manifold tests focus on simple
    // simplexes that reliably pass validation.

    #[test]
    fn test_validate_manifold_calls_tds_is_valid() {
        use crate::core::delaunay_triangulation::DelaunayTriangulation;

        // Verify that validate_manifold() checks TDS structural invariants first
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();
        let tri = dt.triangulation();

        // validate_manifold should pass if tds.is_valid() passes
        assert!(tri.tds.is_valid().is_ok(), "TDS should be valid");
        assert!(
            tri.validate_manifold().is_ok(),
            "Manifold validation should pass when TDS is valid"
        );
    }

    #[test]
    fn test_build_initial_simplex_insufficient_vertices() {
        // Try to build 3D simplex with only 2 vertices (need 4)
        let vertices = vec![vertex!([0.0, 0.0, 0.0]), vertex!([1.0, 0.0, 0.0])];

        let result = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert!(result.is_err());
        match result {
            Err(TriangulationConstructionError::InsufficientVertices { dimension, .. }) => {
                assert_eq!(dimension, 3);
            }
            _ => panic!("Expected InsufficientVertices error"),
        }
    }

    #[test]
    fn test_build_initial_simplex_too_many_vertices() {
        // Try to build 2D simplex with 4 vertices (need exactly 3)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
            vertex!([0.5, 0.5]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert!(result.is_err());
        match result {
            Err(TriangulationConstructionError::InsufficientVertices { .. }) => {}
            _ => panic!("Expected InsufficientVertices error for wrong count"),
        }
    }

    #[test]
    fn test_build_initial_simplex_with_user_data() {
        // Build vertices with user data
        let v1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .data(42_usize)
            .build()
            .unwrap();
        let v2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .data(43_usize)
            .build()
            .unwrap();
        let v3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .data(44_usize)
            .build()
            .unwrap();

        let vertices = vec![v1, v2, v3];
        let tds = Triangulation::<FastKernel<f64>, usize, (), 2>::build_initial_simplex(&vertices)
            .unwrap();

        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.number_of_cells(), 1);

        // Verify user data is preserved
        let data_values: Vec<_> = tds
            .vertices()
            .filter_map(|(_, v)| v.data.as_ref())
            .copied()
            .collect();
        assert_eq!(data_values.len(), 3);
        assert!(data_values.contains(&42));
        assert!(data_values.contains(&43));
        assert!(data_values.contains(&44));
    }

    // =============================================================================
    // Tests for build_initial_simplex degeneracy validation
    // =============================================================================

    #[test]
    fn test_build_initial_simplex_rejects_collinear_2d() {
        // Collinear points should be rejected by build_initial_simplex
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, 0.0]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices);

        assert!(result.is_err(), "Collinear points should be rejected");
        match result {
            Err(TriangulationConstructionError::GeometricDegeneracy { message }) => {
                assert!(
                    message.contains("Degenerate"),
                    "Error message should mention degeneracy"
                );
            }
            _ => panic!("Expected GeometricDegeneracy error for collinear points"),
        }
    }

    #[test]
    fn test_build_initial_simplex_rejects_coplanar_3d() {
        // Coplanar points should be rejected by build_initial_simplex
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.5, 0.5, 0.0]),
        ];

        let result = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices);

        assert!(result.is_err(), "Coplanar points should be rejected");
        match result {
            Err(TriangulationConstructionError::GeometricDegeneracy { message }) => {
                assert!(
                    message.contains("Degenerate") || message.contains("coplanar"),
                    "Error message should mention degeneracy or coplanarity"
                );
            }
            _ => panic!("Expected GeometricDegeneracy error for coplanar points"),
        }
    }

    /// Consolidated macro for facet validation tests across dimensions.
    ///
    /// Verifies the manifold topology invariant: each facet shared by at most 2 cells.
    /// Consolidates detection and repair tests into comprehensive suites.
    macro_rules! test_facet_validation {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_detect_local_facet_issues_ $dim d>]() {
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let tri = Triangulation::<FastKernel<f64>, (), (), $dim> { kernel: FastKernel::new(), tds };

                    // Valid simplex: should have no issues
                    let cell_keys: Vec<_> = tri.tds.cell_keys().collect();
                    assert_eq!(cell_keys.len(), 1);
                    let issues = tri.detect_local_facet_issues(&cell_keys).unwrap();
                    assert!(issues.is_none(), "{}D: Valid simplex should have no facet issues", $dim);

                    // Empty list: should return None
                    let issues = tri.detect_local_facet_issues(&[]).unwrap();
                    assert!(issues.is_none(), "{}D: Empty list should have no issues", $dim);

                    // Nonexistent cells: should be skipped gracefully
                    let fake_keys = vec![CellKey::default()];
                    let issues = tri.detect_local_facet_issues(&fake_keys).unwrap();
                    assert!(issues.is_none(), "{}D: Nonexistent cells should be skipped", $dim);

                    // Verify neighbors (all should be None for single cell)
                    let (_, cell) = tri.tds.cells().next().unwrap();
                    if let Some(neighbors) = cell.neighbors() {
                        assert!(neighbors.iter().all(|n| n.is_none()),
                            "{}D: Single cell should have no neighbors", $dim);
                    }
                }

                #[test]
                fn [<test_repair_local_facet_issues_ $dim d>]() {
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let mut tri = Triangulation::<FastKernel<f64>, (), (), $dim> { kernel: FastKernel::new(), tds };

                    // Empty issues map: should remove nothing
                    let empty_issues = FacetIssuesMap::default();
                    let removed = tri.repair_local_facet_issues(&empty_issues).unwrap();
                    assert_eq!(removed, 0, "{}D: Empty issues should remove 0 cells", $dim);
                    assert_eq!(tri.tds.number_of_cells(), 1, "{}D: Should still have 1 cell", $dim);
                }
            }
        };
    }

    /// Dimension-parametric `remove_vertex` tests.
    ///
    /// Verifies that vertex removal maintains neighbor pointer integrity and
    /// triangulation validity across dimensions.
    macro_rules! test_remove_vertex {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?], $interior_point:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_remove_vertex_neighbor_pointers_ $dim d>]() {
                    use crate::core::delaunay_triangulation::DelaunayTriangulation;

                    // Build triangulation with D+1 simplex vertices + 1 interior point
                    let vertices: Vec<Vertex<f64, (), $dim>> = {
                        let mut v = vec![$(vertex!($simplex_coords)),+];
                        v.push(vertex!($interior_point));
                        v
                    };

                    let mut dt = DelaunayTriangulation::new(&vertices)
                        .expect("Failed to create triangulation");

                    // Find and remove the interior vertex
                    let interior_vertex = dt
                        .vertices()
                        .find(|(_, v)| {
                            let coords = v.point().coords();
                            coords.iter()
                                .zip($interior_point.iter())
                                .all(|(a, b)| (a - b).abs() < 1e-10)
                        })
                        .map(|(_, v)| *v)
                        .expect("Interior vertex not found");

                    let initial_cell_count = dt.tds().number_of_cells();
                    dt.remove_vertex(&interior_vertex)
                        .expect("Failed to remove vertex");

                    // After removal, should have fewer cells (or same if just 1 simplex left)
                    assert!(dt.tds().number_of_cells() <= initial_cell_count,
                        "{}D: Cell count should not increase after removal", $dim);

                    // Verify neighbor pointer consistency:
                    // 1. No dangling pointers (all neighbor keys exist)
                    // 2. Neighbor relationships are symmetric
                    for (cell_key, cell) in dt.tds().cells() {
                        if let Some(neighbors) = cell.neighbors() {
                            for (facet_idx, neighbor_opt) in neighbors.iter().enumerate() {
                                if let Some(neighbor_key) = neighbor_opt {
                                    // Verify neighbor exists
                                    assert!(
                                        dt.tds().contains_cell(*neighbor_key),
                                        "{}D: Cell {cell_key:?} has neighbor pointer to non-existent cell {neighbor_key:?}",
                                        $dim
                                    );

                                    // Verify symmetry: neighbor should point back to us
                                    let neighbor_cell = dt
                                        .tds()
                                        .get_cell(*neighbor_key)
                                        .expect("Neighbor cell should exist");
                                    if let Some(neighbor_neighbors) = neighbor_cell.neighbors() {
                                        let points_back = neighbor_neighbors
                                            .iter()
                                            .any(|n| n.as_ref() == Some(&cell_key));
                                        assert!(
                                            points_back,
                                            "{}D: Cell {cell_key:?} has neighbor {neighbor_key:?} at facet {facet_idx}, but neighbor doesn't point back",
                                            $dim
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Verify triangulation is still valid
                    assert!(
                        dt.is_valid().is_ok(),
                        "{}D: Triangulation should be valid after vertex removal",
                        $dim
                    );
                }
            }
        };
    }

    /// Basic accessor tests across dimensions.
    macro_rules! test_basic_accessors {
        ($dim:expr, [$($simplex_coords:expr),+ $(,)?]) => {
            pastey::paste! {
                #[test]
                fn [<test_basic_accessors_ $dim d>]() {
                    // Empty triangulation
                    let empty: Triangulation<FastKernel<f64>, (), (), $dim> =
                        Triangulation::new_empty(FastKernel::new());
                    assert_eq!(empty.number_of_vertices(), 0);
                    assert_eq!(empty.number_of_cells(), 0);
                    assert_eq!(empty.dim(), -1);
                    assert_eq!(empty.cells().count(), 0);
                    assert_eq!(empty.vertices().count(), 0);
                    assert_eq!(empty.facets().count(), 0);
                    assert_eq!(empty.boundary_facets().count(), 0);

                    // Simplex triangulation
                    let vertices: Vec<Vertex<f64, (), $dim>> = vec![
                        $(vertex!($simplex_coords)),+
                    ];
                    let expected_vertex_count = vertices.len();

                    let tds = Triangulation::<FastKernel<f64>, (), (), $dim>::build_initial_simplex(&vertices)
                        .unwrap();
                    let tri = Triangulation::<FastKernel<f64>, (), (), $dim> { kernel: FastKernel::new(), tds };

                    assert_eq!(tri.number_of_vertices(), expected_vertex_count);
                    assert_eq!(tri.number_of_cells(), 1);
                    assert_eq!(tri.dim(), $dim as i32);
                    assert_eq!(tri.cells().count(), 1);
                    assert_eq!(tri.vertices().count(), expected_vertex_count);

                    // D-simplex has D+1 facets, all on boundary
                    let facet_count = tri.facets().count();
                    assert_eq!(facet_count, expected_vertex_count, "{}D: D-simplex should have D+1 facets", $dim);
                    let boundary_count = tri.boundary_facets().count();
                    assert_eq!(boundary_count, expected_vertex_count, "{}D: All facets should be on boundary", $dim);
                }
            }
        };
    }

    // Facet validation tests (2D - 5D)
    test_facet_validation!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    test_facet_validation!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
    test_facet_validation!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    test_facet_validation!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    // Basic accessor tests (2D - 5D)
    test_basic_accessors!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    test_basic_accessors!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
    test_basic_accessors!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    test_basic_accessors!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    );

    // Remove vertex tests (2D - 5D)
    test_remove_vertex!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [0.3, 0.3]);
    test_remove_vertex!(
        3,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [0.25, 0.25, 0.25]
    );
    test_remove_vertex!(
        4,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        [0.2, 0.2, 0.2, 0.2]
    );
    test_remove_vertex!(
        5,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        [0.16, 0.16, 0.16, 0.16, 0.16]
    );
}

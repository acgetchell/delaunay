//! Generic triangulation combining kernel and combinatorial data structure.
//!
//! Following CGAL's architecture, the `Triangulation` struct combines:
//! - A geometric `Kernel` for predicates
//! - A purely combinatorial `Tds` for topology
//!
//! This layer provides geometric operations while delegating topology to Tds.

use core::iter::Sum;
use core::ops::{AddAssign, SubAssign};

use num_traits::NumCast;

use crate::core::cell::Cell;
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, VertexKey};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;

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

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: AddAssign + SubAssign + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    /// Create an empty triangulation with the given kernel.
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
    pub fn cells(&self) -> impl Iterator<Item = (CellKey, &Cell<K::Scalar, U, V, D>)> {
        self.tds.cells()
    }

    /// Returns an iterator over all vertices in the triangulation.
    ///
    /// Delegates to the underlying Tds.
    pub fn vertices(&self) -> impl Iterator<Item = (VertexKey, &Vertex<K::Scalar, U, D>)> {
        self.tds.vertices()
    }

    /// Returns the number of vertices in the triangulation.
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.tds.number_of_vertices()
    }

    /// Returns the number of cells in the triangulation.
    #[must_use]
    pub fn number_of_cells(&self) -> usize {
        self.tds.number_of_cells()
    }

    /// Returns the dimension of the triangulation.
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
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::vertex;
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
    /// use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
    /// use delaunay::vertex;
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

    // Phase 2 TODO: Add geometric operations using kernel predicates
    // - locate(point) - point location using facet walking
    // - is_valid_geometric() - validate using kernel predicates
}

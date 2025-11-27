//! Delaunay triangulation layer with incremental insertion.
//!
//! This layer adds Delaunay-specific operations on top of the generic
//! `Triangulation` struct, following CGAL's architecture.

use core::iter::Sum;
use core::ops::{AddAssign, SubAssign};

use num_traits::NumCast;

use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::geometry::kernel::Kernel;

/// Delaunay triangulation with incremental insertion support.
///
/// # Type Parameters
/// - `K`: Geometric kernel implementing predicates
/// - `U`: User data type for vertices
/// - `V`: User data type for cells
/// - `D`: Dimension of the triangulation
///
/// # Phase 3 TODO
/// Implement incremental insertion with:
/// - Point location (facet walking)
/// - Conflict region computation (local BFS)
/// - Cavity extraction and filling
/// - Local neighbor wiring (no global `assign_neighbors`)
pub struct DelaunayTriangulation<K, U, V, const D: usize>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    /// The underlying generic triangulation.
    tri: Triangulation<K, U, V, D>,
}

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    K::Scalar: AddAssign + SubAssign + Sum + NumCast,
    U: DataType,
    V: DataType,
{
    /// Create an empty Delaunay triangulation with the given kernel.
    #[must_use]
    pub fn new_empty(kernel: K) -> Self {
        Self {
            tri: Triangulation::new_empty(kernel),
        }
    }

    /// Returns the number of vertices in the triangulation.
    #[must_use]
    pub fn number_of_vertices(&self) -> usize {
        self.tri.number_of_vertices()
    }

    /// Returns the number of cells in the triangulation.
    #[must_use]
    pub fn number_of_cells(&self) -> usize {
        self.tri.number_of_cells()
    }

    /// Returns the dimension of the triangulation.
    #[must_use]
    pub fn dim(&self) -> i32 {
        self.tri.dim()
    }

    // Phase 3 TODO: Implement incremental insertion
    // pub fn insert(&mut self, vertex: Vertex<K::Scalar, U, D>) -> Result<VertexKey, InsertionError>

    // Phase 3 TODO: Implement Delaunay validation
    // pub fn validate_delaunay(&self) -> Result<(), DelaunayValidationError>

    // Phase 3 TODO: Implement point location
    // pub fn locate(&self, point: &Point<K::Scalar, D>) -> Result<LocateResult, LocateError>
}

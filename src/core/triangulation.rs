//! Generic triangulation combining kernel and combinatorial data structure.
//!
//! Following CGAL's architecture, the `Triangulation` struct combines:
//! - A geometric `Kernel` for predicates
//! - A purely combinatorial `Tds` for topology
//!
//! This layer provides geometric operations while delegating topology to Tds.

use core::iter::Sum;
use core::ops::{AddAssign, Div, SubAssign};
use std::cmp::Ordering as CmpOrdering;

use num_traits::NumCast;
use uuid::Uuid;

use crate::core::algorithms::incremental_insertion::InsertionError;
use crate::core::cell::Cell;
use crate::core::collections::{
    CellKeyBuffer, CellKeySet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer, ValidCellsBuffer,
    VertexKeySet,
};
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{
    CellKey, Tds, TriangulationConstructionError, VertexKey,
};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::quality::radius_ratio;
use crate::geometry::traits::coordinate::CoordinateScalar;
use crate::geometry::util::safe_scalar_to_f64;

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
    /// Build initial D-simplex from D+1 vertices.
    ///
    /// This creates a Tds with a single cell containing all D+1 vertices,
    /// with no neighbor relationships (all boundary facets). This method
    /// does not require the Delaunay property - it only uses basic topology.
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
        conflict_cells: Option<CellKeyBuffer>,
        hint: Option<CellKey>,
    ) -> Result<(VertexKey, Option<CellKey>), InsertionError>
    where
        K::Scalar: CoordinateScalar,
    {
        use crate::core::algorithms::incremental_insertion::{
            extend_hull, fill_cavity, wire_cavity_neighbors,
        };
        use crate::core::algorithms::locate::{LocateResult, extract_cavity_boundary, locate};

        // 1. Insert vertex into Tds
        let v_key = self.tds.insert_vertex_with_mapping(vertex)?;

        // 2. Check if we need to bootstrap the initial simplex
        let num_vertices = self.tds.number_of_vertices();

        if num_vertices < D + 1 {
            // Bootstrap phase: just accumulate vertices, no cells yet
            return Ok((v_key, None));
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

            // Return first cell key for hint caching
            let first_cell = self.tds.cell_keys().next();
            return Ok((v_key, first_cell));
        }

        // 3. Locate containing cell (for vertex D+2 and beyond)
        let point = *self
            .tds
            .get_vertex_by_key(v_key)
            .ok_or_else(|| InsertionError::CavityFilling {
                message: "Vertex key invalid immediately after insertion".to_string(),
            })?
            .point();
        let location = locate(&self.tds, &self.kernel, &point, hint)?;

        // 4. Handle different location results
        match location {
            LocateResult::InsideCell(_start_cell) => {
                // Interior vertex: require conflict_cells parameter
                let conflict_cells =
                    conflict_cells.ok_or_else(|| InsertionError::CavityFilling {
                        message: "Interior point insertion requires conflict_cells parameter"
                            .to_string(),
                    })?;

                // 5. Extract cavity boundary
                let boundary_facets = extract_cavity_boundary(&self.tds, &conflict_cells)?;

                // 6. Fill cavity BEFORE removing old cells
                let new_cells = fill_cavity(&mut self.tds, v_key, &boundary_facets)?;

                // 7. Wire neighbors (while both old and new cells exist)
                wire_cavity_neighbors(&mut self.tds, &new_cells, Some(&conflict_cells))?;

                // 8. Remove conflict cells (now that new cells are wired up)
                let _removed_count = self.tds.remove_cells_by_keys(&conflict_cells);

                // 9. Repair any invalid facet sharing
                if let Err(e) = self.fix_invalid_facet_sharing() {
                    #[cfg(debug_assertions)]
                    eprintln!("Warning: facet sharing repair failed after insertion: {e}");
                    #[cfg(not(debug_assertions))]
                    let _ = e;
                }

                // Return vertex key and first new cell for hint caching
                Ok((v_key, new_cells.first().copied()))
            }
            LocateResult::Outside => {
                // Exterior vertex: extend convex hull
                let new_cells = extend_hull(&mut self.tds, &self.kernel, v_key, &point)?;

                // Return vertex key and first new cell for hint caching
                Ok((v_key, new_cells.first().copied()))
            }
            _ => {
                // TODO: Handle degenerate point locations (OnFacet, OnEdge, OnVertex)
                Err(InsertionError::CavityFilling {
                    message: format!(
                        "Unhandled degenerate location: {location:?}. Point lies on facet/edge/vertex which is not yet supported."
                    ),
                })
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
        wire_cavity_neighbors(&mut self.tds, &new_cells, Some(&cells_to_remove))
            .map_err(|e| crate::core::triangulation_data_structure::TriangulationValidationError::InconsistentDataStructure {
                message: format!("Neighbor wiring failed: {e}"),
            })?;

        // Remove the cells containing the vertex (now that new cells are wired up)
        let cells_removed = self.tds.remove_cells_by_keys(&cells_to_remove);

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

    /// Attempts to fix invalid facet sharing by removing problematic cells using geometric quality metrics.
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
    /// # Errors
    ///
    /// Returns error if the facet map cannot be built (indicating structural corruption).
    ///
    /// **Note**: Some internal repair failures (duplicate removal, neighbor assignment) are
    /// logged in debug builds but do not cause this method to return an error. The method
    /// may return `Ok(n)` even if some facet sharing violations remain after the repair attempt.
    #[allow(clippy::too_many_lines)]
    pub fn fix_invalid_facet_sharing(
        &mut self,
    ) -> Result<usize, crate::core::triangulation_data_structure::TriangulationValidationError>
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

            // Remove cells
            let to_remove: Vec<CellKey> = cells_to_remove.into_iter().collect();
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

            // Rebuild topology if needed
            if actually_removed > 0 && duplicate_cells_removed == 0 {
                if self.tds.assign_neighbors().is_err() {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Warning: assign_neighbors failed during facet repair (removed {actually_removed} cells)"
                    );
                    total_removed += actually_removed;
                    continue;
                }
                if self.tds.assign_incident_cells().is_err() {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Warning: assign_incident_cells failed during facet repair (removed {actually_removed} cells)"
                    );
                    total_removed += actually_removed;
                    continue;
                }
            }

            let removed_this_iteration = actually_removed + duplicate_cells_removed;
            total_removed += removed_this_iteration;

            if removed_this_iteration == 0 || self.tds.validate_facet_sharing().is_ok() {
                break;
            }
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
}

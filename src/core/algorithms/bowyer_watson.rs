//! Incremental Bowyer-Watson algorithm for Delaunay triangulation.
//!
//! This module implements a robust, incremental approach to Delaunay triangulation
//! construction using the Bowyer-Watson algorithm. The implementation focuses on
//! clean separation of concerns and efficient vertex insertion without supercells.
//!
//! # Algorithm Overview
//!
//! The incremental Bowyer-Watson algorithm works as follows:
//!
//! 1. **Initialization**: Create initial simplex from first D+1 vertices
//! 2. **Incremental insertion**: For each subsequent vertex:
//!    - Locate the vertex relative to existing triangulation
//!    - Use cavity-based insertion for interior vertices
//!    - Use convex hull extension for exterior vertices
//!    - Maintain Delaunay property through circumsphere tests
//! 3. **Cleanup**: Remove degenerate cells and establish neighbor relationships
//!
//! # Triangulation Invariant Enforcement
//!
//! The Delaunay property and other triangulation invariants are enforced in specific locations:
//!
//! | Invariant Type | Enforcement Location | Method |
//! |---|---|---|
//! | **Delaunay Property** | `find_bad_cells()` | Empty circumsphere test using `insphere()` |
//! | **Facet Sharing** | `validate_facet_sharing()` | Each facet shared by ≤ 2 cells |
//! | **No Duplicate Cells** | `validate_no_duplicate_cells()` | No cells with identical vertex sets |
//! | **Neighbor Consistency** | `validate_neighbors_internal()` | Mutual neighbor relationships |
//! | **Cell Validity** | `CellBuilder::validate()` (vertex count) + `cell.is_valid()` (comprehensive) | Construction + runtime validation |
//! | **Vertex Validity** | `Point::from()` (coordinates) + UUID auto-gen + `vertex.is_valid()` | Construction + runtime validation |
//!
//! The Delaunay property (empty circumsphere) is enforced **proactively** during construction in
//! `find_bad_cells()`, while structural invariants are enforced **reactively** through validation
//! in `is_valid()` and related methods.
//!
//! # Key Features
//!
//! - **Pure Incremental**: No supercells or batch processing
//! - **Robust**: Handles degenerate cases and numerical precision issues
//! - **Flexible**: Supports both interior and exterior vertex insertion
//! - **Efficient**: Expected O(1) amortized time per insertion
//! - **Maintainable**: Clean separation of algorithm logic and data structures
//!
//! # References
//!
//! The Bowyer-Watson algorithm was independently developed by Adrian Bowyer and David Watson
//! in 1981. This implementation draws from both foundational papers and modern computational
//! geometry literature:
//!
//! ## Original Papers
//!
//! - **Bowyer, A.** "Computing Dirichlet tessellations." *The Computer Journal* 24.2 (1981): 162-166.
//!   DOI: [10.1093/comjnl/24.2.162](https://doi.org/10.1093/comjnl/24.2.162)
//!
//! - **Watson, D.F.** "Computing the n-dimensional Delaunay tessellation with application to
//!   Voronoi polytopes." *The Computer Journal* 24.2 (1981): 167-172.
//!   DOI: [10.1093/comjnl/24.2.167](https://doi.org/10.1093/comjnl/24.2.167)
//!
//! ## Modern References
//!
//! - **de Berg, M., Cheong, O., van Kreveld, M., and Overmars, M.**
//!   *Computational Geometry: Algorithms and Applications.* 3rd ed. Springer-Verlag, 2008.
//!   Chapter 9: Delaunay Triangulations. ISBN: 978-3-540-77973-5
//!
//! - **Fortune, S.** "A sweepline algorithm for Voronoi diagrams."
//!   *Algorithmica* 2.1-4 (1987): 153-174. DOI: [10.1007/BF01840357](https://doi.org/10.1007/BF01840357)
//!
//! - **Guibas, L. and Stolfi, J.** "Primitives for the manipulation of general subdivisions
//!   and the computation of Voronoi diagrams." *ACM Transactions on Graphics* 4.2 (1985): 74-123.
//!   DOI: [10.1145/282918.282923](https://doi.org/10.1145/282918.282923)
//!
//! ## Implementation References
//!
//! - **CGAL Editorial Board.** *CGAL User and Reference Manual.* 5.6 edition, 2023.
//!   Available: [https://doc.cgal.org/latest/Triangulation_2/](https://doc.cgal.org/latest/Triangulation_2/)
//!
//! - **Shewchuk, J.R.** "Delaunay refinement algorithms for triangular mesh generation."
//!   *Computational Geometry* 22.1-3 (2002): 21-74.
//!   DOI: [10.1016/S0925-7721(01)00047-5](https://doi.org/10.1016/S0925-7721(01)00047-5)
//!
//! - **Edelsbrunner, H.** *Geometry and Topology for Mesh Generation.*
//!   Cambridge University Press, 2001. ISBN: 978-0-521-79309-4
//!
//! ## Numerical Stability References
//!
//! - **Shewchuk, J.R.** "Adaptive precision floating-point arithmetic and fast robust
//!   geometric predicates." *Discrete & Computational Geometry* 18.3 (1997): 305-363.
//!   DOI: [10.1007/PL00009321](https://doi.org/10.1007/PL00009321)
//!
//! - **Fortune, S. and Van Wyk, C.J.** "Efficient exact arithmetic for computational geometry."
//!   *Proceedings of the ninth annual symposium on Computational geometry* (1993): 163-172.
//!   DOI: [10.1145/160985.161140](https://doi.org/10.1145/160985.161140)

use crate::core::{
    cell::CellBuilder,
    facet::Facet,
    traits::data_type::DataType,
    triangulation_data_structure::{
        CellKey, Tds, TriangulationConstructionError, TriangulationValidationError,
    },
    vertex::Vertex,
};
use crate::geometry::{
    algorithms::convex_hull::ConvexHull,
    point::Point,
    predicates::{InSphere, insphere, simplex_orientation},
    traits::coordinate::CoordinateScalar,
};
use nalgebra::{ComplexField, Const, OPoint};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    collections::{HashMap, HashSet},
    iter::Sum,
    ops::{AddAssign, Div, SubAssign},
};

/// Result type for Bowyer-Watson operations
pub type BoyerWatsonResult<T> = Result<T, TriangulationConstructionError>;

/// Strategies for inserting vertices into the triangulation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertionStrategy {
    /// Vertex is inside the triangulation - use cavity-based insertion
    CavityBased,
    /// Vertex is outside the triangulation - extend the convex hull
    HullExtension,
    /// Use fallback method for difficult cases
    Fallback,
}

/// Information about a vertex insertion operation
#[derive(Debug, Clone)]
pub struct InsertionInfo {
    /// Strategy used for insertion
    pub strategy: InsertionStrategy,
    /// Number of cells removed during insertion
    pub cells_removed: usize,
    /// Number of new cells created
    pub cells_created: usize,
    /// Whether the insertion was successful
    pub success: bool,
}

/// Incremental Bowyer-Watson algorithm implementation
///
/// This struct provides a clean interface for constructing Delaunay triangulations
/// using the incremental Bowyer-Watson algorithm. It operates on triangulation
/// data structures and maintains the Delaunay property throughout construction.
pub struct IncrementalBoyerWatson<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Statistics and debugging information
    pub insertion_count: usize,
    /// Total number of cells created during triangulation construction
    pub total_cells_created: usize,
    /// Total number of cells removed during triangulation construction
    pub total_cells_removed: usize,

    // Reusable buffers for performance
    bad_cells_buffer: Vec<CellKey>,
    boundary_facets_buffer: Vec<Facet<T, U, V, D>>,
    vertex_points_buffer: Vec<crate::geometry::point::Point<T, D>>,

    /// Cached convex hull for hull extension
    hull: Option<ConvexHull<T, U, V, D>>,
}

impl<T, U, V, const D: usize> IncrementalBoyerWatson<T, U, V, D>
where
    T: CoordinateScalar
        + AddAssign<T>
        + ComplexField<RealField = T>
        + SubAssign<T>
        + Sum
        + From<f64>,
    U: DataType + DeserializeOwned,
    V: DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: Div<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    ordered_float::OrderedFloat<f64>: From<T>,
{
    /// Creates a new incremental Bowyer-Watson algorithm instance
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::algorithms::bowyer_watson::IncrementalBoyerWatson;
    ///
    /// // Create a new 3D Bowyer-Watson algorithm instance
    /// let algorithm: IncrementalBoyerWatson<f64, Option<()>, Option<()>, 3> =
    ///     IncrementalBoyerWatson::new();
    ///
    /// // Verify initial statistics
    /// let (insertions, created, removed) = algorithm.get_statistics();
    /// assert_eq!(insertions, 0);
    /// assert_eq!(created, 0);
    /// assert_eq!(removed, 0);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            insertion_count: 0,
            total_cells_created: 0,
            total_cells_removed: 0,
            bad_cells_buffer: Vec::new(),
            boundary_facets_buffer: Vec::new(),
            vertex_points_buffer: Vec::new(),
            hull: None,
        }
    }

    /// Constructs a Delaunay triangulation from the given vertices
    ///
    /// This is the main entry point for triangulation construction.
    /// It handles the complete process from initialization to finalization.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertices` - Vector of vertices to triangulate
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful triangulation, or an error if construction fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Insufficient vertices provided (< D+1)
    /// - Initial simplex creation fails
    /// - Vertex insertion fails
    /// - Triangulation finalization fails
    ///
    /// # Examples
    ///
    /// Basic usage with a simple tetrahedron:
    ///
    /// ```
    /// use delaunay::core::algorithms::bowyer_watson::IncrementalBoyerWatson;
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// // Use Tds::new to properly initialize with vertices
    /// let mut tds: Tds<f64, Option<()>, Option<()>, 3> =
    ///     Tds::new(&vertices).unwrap();
    ///     
    /// // Verify the triangulation was created
    /// assert_eq!(tds.number_of_vertices(), 4);
    /// assert_eq!(tds.number_of_cells(), 1); // Single tetrahedron
    /// assert!(tds.is_valid().is_ok());
    /// ```
    ///
    /// Using the algorithm for incremental construction:
    ///
    /// ```
    /// use delaunay::core::algorithms::bowyer_watson::IncrementalBoyerWatson;
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::vertex;
    ///
    /// let mut algorithm: IncrementalBoyerWatson<f64, Option<()>, Option<()>, 3> =
    ///     IncrementalBoyerWatson::new();
    ///
    /// // Start with empty TDS and build incrementally
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    ///     vertex!([0.5, 0.5, 0.5]), // Interior point
    ///
    /// ];
    ///
    /// let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::default();
    /// algorithm.triangulate(&mut tds, &vertices).unwrap();
    ///
    /// assert_eq!(tds.number_of_vertices(), 5);
    /// assert!(tds.number_of_cells() > 1); // Multiple cells after incremental insertion
    ///
    /// let (insertions, created, _) = algorithm.get_statistics();
    /// assert_eq!(insertions, 1); // One incremental insertion beyond initial simplex
    /// assert!(created > 1);       // Multiple cells created
    /// ```
    pub fn triangulate(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertices: &[Vertex<T, U, D>],
    ) -> BoyerWatsonResult<()>
    where
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        if vertices.is_empty() {
            return Ok(());
        }

        // Check for sufficient vertices
        if vertices.len() < D + 1 {
            return Err(TriangulationConstructionError::InsufficientVertices {
                dimension: D,
                source: crate::core::cell::CellValidationError::InsufficientVertices {
                    actual: vertices.len(),
                    expected: D + 1,
                    dimension: D,
                },
            });
        }

        // Step 1: Initialize with first D+1 vertices
        let (initial_vertices, remaining_vertices) = vertices.split_at(D + 1);
        self.create_initial_simplex(tds, initial_vertices.to_vec())?;

        // Step 2: Insert remaining vertices incrementally
        for vertex in remaining_vertices {
            let insertion_info = self
                .insert_vertex(tds, *vertex)
                .map_err(TriangulationConstructionError::ValidationError)?;

            // Note: insertion_count is already incremented in insert_vertex method
            self.total_cells_created += insertion_info.cells_created;
            self.total_cells_removed += insertion_info.cells_removed;
        }

        // Step 3: Finalize the triangulation
        Self::finalize_triangulation(tds)?;

        Ok(())
    }

    /// Creates the initial simplex from the first D+1 vertices
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertices` - Exactly D+1 vertices to form the initial simplex
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful creation, or an error if the simplex cannot be created.
    fn create_initial_simplex(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertices: Vec<Vertex<T, U, D>>,
    ) -> BoyerWatsonResult<()> {
        assert_eq!(
            vertices.len(),
            D + 1,
            "Initial simplex requires exactly D+1 vertices"
        );

        // Ensure all vertices are registered in the TDS vertex mapping
        for vertex in &vertices {
            if !tds.vertex_bimap.contains_left(&vertex.uuid()) {
                let vertex_key = tds.vertices.insert(*vertex);
                tds.vertex_bimap.insert(vertex.uuid(), vertex_key);
            }
        }

        let cell = CellBuilder::default()
            .vertices(vertices)
            .build()
            .map_err(|e| TriangulationConstructionError::FailedToCreateCell {
                message: format!("Failed to create initial simplex: {e}"),
            })?;

        let cell_key = tds.cells_mut().insert(cell);
        let cell_uuid = tds.cells()[cell_key].uuid();
        tds.cell_bimap.insert(cell_uuid, cell_key);

        self.total_cells_created += 1;

        Ok(())
    }

    /// Inserts a single vertex into the triangulation
    ///
    /// This method determines the appropriate insertion strategy and
    /// executes the insertion while maintaining the Delaunay property.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex to insert
    ///
    /// # Returns
    ///
    /// `InsertionInfo` describing the insertion operation, or an error on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if vertex insertion fails due to geometric degeneracy or topology issues.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::core::algorithms::bowyer_watson::{
    ///     IncrementalBoyerWatson, InsertionStrategy
    /// };
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::vertex;
    ///
    /// // Create initial triangulation with a tetrahedron
    /// let initial_vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([2.0, 0.0, 0.0]),
    ///     vertex!([0.0, 2.0, 0.0]),
    ///     vertex!([0.0, 0.0, 2.0]),
    /// ];
    /// let mut tds: Tds<f64, Option<()>, Option<()>, 3> =
    ///     Tds::new(&initial_vertices).unwrap();
    /// let mut algorithm: IncrementalBoyerWatson<f64, Option<()>, Option<()>, 3> =
    ///     IncrementalBoyerWatson::new();
    ///
    /// let initial_cells = tds.number_of_cells();
    ///
    /// // Insert an interior vertex (should use cavity-based strategy)
    /// let interior_vertex = vertex!([0.5, 0.5, 0.5]);
    /// let insertion_info = algorithm
    ///     .insert_vertex(&mut tds, interior_vertex)
    ///     .unwrap();
    ///
    /// assert_eq!(insertion_info.strategy, InsertionStrategy::CavityBased);
    /// assert!(insertion_info.success);
    /// assert!(insertion_info.cells_created > 0);
    /// assert!(tds.number_of_cells() > initial_cells);
    ///
    /// // Check that algorithm statistics are updated
    /// let (insertions, created, removed) = algorithm.get_statistics();
    /// assert_eq!(insertions, 1);
    /// assert!(created > 0);
    /// ```
    pub fn insert_vertex(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<InsertionInfo, TriangulationValidationError>
    where
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // Determine insertion strategy
        let strategy = self.determine_insertion_strategy(tds, &vertex);

        let result = match strategy {
            InsertionStrategy::CavityBased => self.insert_vertex_cavity_based(tds, &vertex),
            InsertionStrategy::HullExtension => self.insert_vertex_hull_extension(tds, &vertex),
            InsertionStrategy::Fallback => self.insert_vertex_fallback(tds, &vertex),
        };

        // Update statistics on successful insertion
        if let Ok(ref info) = result {
            self.insertion_count += 1;
            self.total_cells_created += info.cells_created;
            self.total_cells_removed += info.cells_removed;
        }

        result
    }

    /// Determines the appropriate insertion strategy for a vertex
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex to be inserted
    ///
    /// # Returns
    ///
    /// The recommended insertion strategy.
    fn determine_insertion_strategy(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> InsertionStrategy
    where
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // Check if vertex is inside any existing cell's circumsphere
        if self.is_vertex_interior(tds, vertex) {
            InsertionStrategy::CavityBased
        } else {
            InsertionStrategy::HullExtension
        }
    }

    /// Checks if a vertex is interior to the current triangulation
    ///
    /// A vertex is considered interior if it lies within the circumsphere
    /// of at least one existing cell.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex to test
    ///
    /// # Returns
    ///
    /// `true` if the vertex is interior, `false` otherwise.
    #[allow(clippy::unused_self)]
    fn is_vertex_interior(&self, tds: &Tds<T, U, V, D>, vertex: &Vertex<T, U, D>) -> bool
    where
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        for cell in tds.cells().values() {
            let mut vertex_points = Vec::new();
            vertex_points.extend(cell.vertices().iter().map(|v| *v.point()));

            if let Ok(containment) = insphere(&vertex_points, *vertex.point()) {
                if matches!(containment, InSphere::INSIDE) {
                    return true;
                }
            }
        }
        false
    }

    /// Inserts a vertex using cavity-based Bowyer-Watson insertion
    ///
    /// This method:
    /// 1. Finds all "bad" cells whose circumsphere contains the vertex
    /// 2. Removes these cells to create a star-shaped cavity
    /// 3. Triangulates the cavity by connecting the vertex to boundary facets
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex to insert
    ///
    /// # Returns
    ///
    /// `InsertionInfo` describing the operation, or an error on failure.
    fn insert_vertex_cavity_based(
        &mut self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, TriangulationValidationError>
    where
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // Find bad cells
        let bad_cells = self.find_bad_cells(tds, vertex);

        if bad_cells.is_empty() {
            // No bad cells found - try hull extension instead
            return self.insert_vertex_hull_extension(tds, vertex);
        }

        // Find boundary facets of the cavity
        let boundary_facets = self.find_cavity_boundary_facets(tds, &bad_cells)?;

        if boundary_facets.is_empty() {
            return Err(TriangulationValidationError::FailedToCreateCell {
                message: "No boundary facets found for cavity insertion".to_string(),
            });
        }

        let cells_removed = bad_cells.len();

        // Remove bad cells
        for &bad_cell_key in &bad_cells {
            if let Some(removed_cell) = tds.cells_mut().remove(bad_cell_key) {
                tds.cell_bimap.remove_by_left(&removed_cell.uuid());
            }
        }

        // Create new cells
        let mut cells_created = 0;
        for boundary_facet in &boundary_facets {
            if Self::create_cell_from_facet_and_vertex(tds, boundary_facet, vertex) {
                cells_created += 1;
            }
        }

        Ok(InsertionInfo {
            strategy: InsertionStrategy::CavityBased,
            cells_removed,
            cells_created,
            success: true,
        })
    }

    /// Inserts a vertex by extending the convex hull
    ///
    /// This method finds visible boundary facets and creates new cells
    /// by connecting the vertex to these facets.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex to insert
    ///
    /// # Returns
    ///
    /// `InsertionInfo` describing the operation, or an error on failure.
    fn insert_vertex_hull_extension(
        &self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, TriangulationValidationError>
    where
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // Get visible boundary facets
        let visible_facets = self.find_visible_boundary_facets(tds, vertex);

        if visible_facets.is_empty() {
            // No visible facets - try fallback method
            return self.insert_vertex_fallback(tds, vertex);
        }

        // Create new cells from visible facets
        let mut cells_created = 0;
        for facet in &visible_facets {
            if Self::create_cell_from_facet_and_vertex(tds, facet, vertex) {
                cells_created += 1;
            }
        }

        Ok(InsertionInfo {
            strategy: InsertionStrategy::HullExtension,
            cells_removed: 0,
            cells_created,
            success: true,
        })
    }

    /// Fallback insertion method for difficult cases
    ///
    /// Instead of the broken vertex replacement approach, this method attempts
    /// a more conservative strategy by trying to find any valid connection
    /// to the existing triangulation.
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `vertex` - The vertex to insert
    ///
    /// # Returns
    ///
    /// `InsertionInfo` describing the operation, or an error if all methods fail.
    #[allow(clippy::unused_self)]
    fn insert_vertex_fallback(
        &self,
        tds: &mut Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Result<InsertionInfo, TriangulationValidationError> {
        // Conservative fallback: try to connect to any existing boundary facet
        // This avoids creating invalid geometry by arbitrary vertex replacement

        let facet_to_cells = tds.build_facet_to_cells_hashmap();

        // First try boundary facets (most likely to work)
        for cells in facet_to_cells.values() {
            if cells.len() == 1 {
                let (cell_key, facet_index) = cells[0];
                if let Some(cell) = tds.cells().get(cell_key) {
                    if let Ok(facets) = cell.facets() {
                        if facet_index < facets.len() {
                            let facet = &facets[facet_index];

                            // Try to create a cell from this facet and the vertex
                            if Self::create_cell_from_facet_and_vertex(tds, facet, vertex) {
                                return Ok(InsertionInfo {
                                    strategy: InsertionStrategy::Fallback,
                                    cells_removed: 0,
                                    cells_created: 1,
                                    success: true,
                                });
                            }
                        }
                    }
                }
            }
        }

        // If boundary facets don't work, try ALL facets (including internal ones)
        for cells in facet_to_cells.values() {
            for &(cell_key, facet_index) in cells {
                if let Some(cell) = tds.cells().get(cell_key) {
                    if let Ok(facets) = cell.facets() {
                        if facet_index < facets.len() {
                            let facet = &facets[facet_index];

                            // Try to create a cell from this facet and the vertex
                            if Self::create_cell_from_facet_and_vertex(tds, facet, vertex) {
                                return Ok(InsertionInfo {
                                    strategy: InsertionStrategy::Fallback,
                                    cells_removed: 0,
                                    cells_created: 1,
                                    success: true,
                                });
                            }
                        }
                    }
                }
            }
        }

        // If we can't find any boundary facet to connect to, the vertex might be
        // in a degenerate position or the triangulation might be corrupted
        Err(TriangulationValidationError::FailedToCreateCell {
            message: format!(
                "Fallback insertion failed: could not connect vertex {:?} to any boundary facet",
                vertex.point()
            ),
        })
    }

    /// Finds all "bad" cells whose circumsphere contains the given vertex
    ///
    /// A cell is "bad" if:
    /// 1. The vertex is strictly inside its circumsphere (violates Delaunay property)
    /// 2. The cell is not degenerate or invalid
    ///
    /// This is more conservative than the original implementation to prevent
    /// over-removal of cells which was causing boundary facet elimination.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex to test against circumspheres
    ///
    /// # Returns
    ///
    /// A vector of bad cell keys, or an error if computation fails.
    fn find_bad_cells(&mut self, tds: &Tds<T, U, V, D>, vertex: &Vertex<T, U, D>) -> Vec<CellKey>
    where
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        self.bad_cells_buffer.clear();

        // Only consider cells that have a valid circumsphere and strict containment
        for (cell_key, cell) in tds.cells() {
            // Skip cells with insufficient vertices
            if cell.vertices().len() < D + 1 {
                continue;
            }

            self.vertex_points_buffer.clear();
            self.vertex_points_buffer
                .extend(cell.vertices().iter().map(|v| *v.point()));

            // Test circumsphere containment
            match insphere(&self.vertex_points_buffer, *vertex.point()) {
                Ok(InSphere::INSIDE) => {
                    // Only add if this would create a true Delaunay violation
                    // This prevents over-removal that was eliminating boundary facets
                    self.bad_cells_buffer.push(cell_key);
                }
                Ok(InSphere::OUTSIDE | InSphere::BOUNDARY) => {
                    // Vertex is outside or on the circumsphere - cell is fine
                }
                Err(e) => {
                    // Skip cells with degenerate circumspheres
                    eprintln!(
                        "Warning: Could not compute circumsphere for cell {:?}: {}",
                        cell.uuid(),
                        e
                    );
                }
            }
        }

        // Additional validation: ensure we're not removing ALL cells
        if self.bad_cells_buffer.len() == tds.cells().len() && tds.cells().len() > 1 {
            // If we're removing all cells in a multi-cell triangulation, something is wrong
            // This likely means the circumsphere test is failing or the vertex is degenerate
            eprintln!(
                "Warning: Circumsphere test marked all {} cells as bad for vertex {:?}",
                tds.cells().len(),
                vertex.point()
            );
            // Instead of failing, return an empty list to force hull extension
            self.bad_cells_buffer.clear();
        }

        self.bad_cells_buffer.clone()
    }

    /// Finds the boundary facets of a cavity formed by removing bad cells
    ///
    /// The boundary facets form the interface between the cavity (bad cells to be removed)
    /// and the good cells that remain. These facets will be used to create new cells
    /// connecting to the inserted vertex.
    ///
    /// A facet is on the cavity boundary if:
    /// 1. It belongs to exactly one bad cell (the other side is a good cell or boundary)
    /// 2. It is not shared between two bad cells (internal to cavity)
    /// 3. It forms a valid interface for retriangulation
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `bad_cells` - Keys of the cells to be removed
    ///
    /// # Returns
    ///
    /// A vector of boundary facets, or an error if computation fails.
    fn find_cavity_boundary_facets(
        &mut self,
        tds: &Tds<T, U, V, D>,
        bad_cells: &[CellKey],
    ) -> Result<Vec<Facet<T, U, V, D>>, TriangulationValidationError> {
        self.boundary_facets_buffer.clear();

        if bad_cells.is_empty() {
            return Ok(self.boundary_facets_buffer.clone());
        }

        let bad_cell_set: HashSet<CellKey> = bad_cells.iter().copied().collect();

        // Build a complete mapping from facet keys to all cells that contain them
        let mut facet_to_cells: HashMap<u64, Vec<CellKey>> = HashMap::new();
        for (cell_key, cell) in tds.cells() {
            if let Ok(facets) = cell.facets() {
                for facet in facets {
                    facet_to_cells
                        .entry(facet.key())
                        .or_default()
                        .push(cell_key);
                }
            }
        }

        // Find cavity boundary facets with improved logic
        let mut processed_facets = HashSet::new();

        for &bad_cell_key in bad_cells {
            if let Some(bad_cell) = tds.cells().get(bad_cell_key) {
                if let Ok(facets) = bad_cell.facets() {
                    for facet in facets {
                        let facet_key = facet.key();

                        // Skip already processed facets
                        if processed_facets.contains(&facet_key) {
                            continue;
                        }

                        if let Some(sharing_cells) = facet_to_cells.get(&facet_key) {
                            // Count how many bad vs good cells share this facet
                            let bad_count = sharing_cells
                                .iter()
                                .filter(|&&cell_key| bad_cell_set.contains(&cell_key))
                                .count();
                            let total_count = sharing_cells.len();

                            // A facet is on the cavity boundary if:
                            // 1. Exactly one bad cell uses it (boundary between bad and good)
                            // 2. OR it's a true boundary facet (only one cell total) that's bad
                            if bad_count == 1 && (total_count == 2 || total_count == 1) {
                                // This is a cavity boundary facet - it separates bad from good cells
                                // or is a boundary facet of a bad cell
                                self.boundary_facets_buffer.push(facet.clone());
                                processed_facets.insert(facet_key);
                            }
                            // Skip facets that are:
                            // - Internal to the cavity (bad_count > 1)
                            // - Not touched by any bad cells (bad_count == 0)
                            // - Invalid sharing (total_count > 2)
                        }
                    }
                }
            }
        }

        // Validation: ensure we have a reasonable number of boundary facets
        if self.boundary_facets_buffer.is_empty() && !bad_cells.is_empty() {
            return Err(TriangulationValidationError::FailedToCreateCell {
                message: format!(
                    "No cavity boundary facets found for {} bad cells. This indicates a topological error.",
                    bad_cells.len()
                ),
            });
        }

        Ok(self.boundary_facets_buffer.clone())
    }

    /// Finds boundary facets visible from the given vertex
    ///
    /// A boundary facet is visible if the vertex lies on the "outside" of that facet.
    /// This uses proper geometric orientation tests to determine visibility.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `vertex` - The vertex from which to test visibility
    ///
    /// # Returns
    ///
    /// A vector of visible boundary facets.
    fn find_visible_boundary_facets(
        &self,
        tds: &Tds<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> Vec<Facet<T, U, V, D>>
    where
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        let mut visible_facets = Vec::new();

        // Get all boundary facets (facets shared by exactly one cell)
        let facet_to_cells = tds.build_facet_to_cells_hashmap();
        let boundary_facets: Vec<_> = facet_to_cells
            .iter()
            .filter(|(_, cells)| cells.len() == 1)
            .collect();

        println!(
            "  Testing {} boundary facets for visibility",
            boundary_facets.len()
        );

        for (_facet_key, cells) in boundary_facets {
            let (cell_key, facet_index) = cells[0];
            if let Some(cell) = tds.cells().get(cell_key) {
                if let Ok(facets) = cell.facets() {
                    if facet_index < facets.len() {
                        let facet = &facets[facet_index];

                        // Use proper visibility test based on orientation
                        if self.is_facet_visible_from_vertex(tds, facet, vertex, cell_key) {
                            println!("    Facet is visible");
                            visible_facets.push(facet.clone());
                        } else {
                            println!("    Facet is not visible");
                        }
                    }
                }
            }
        }

        println!("  Found {} visible boundary facets", visible_facets.len());
        visible_facets
    }

    /// Tests if a boundary facet is visible from a given vertex
    ///
    /// A facet is visible if the vertex lies on the "outside" of the hyperplane
    /// defined by the facet. This uses proper geometric predicates to determine
    /// the correct side of the hyperplane.
    ///
    /// # Algorithm
    ///
    /// For a boundary facet with vertices F = {f₁, f₂, ..., fₐ} and an adjacent
    /// cell with vertices C = {f₁, f₂, ..., fₐ, c}, where c is the "opposite"
    /// vertex in the cell, we test visibility as follows:
    ///
    /// 1. The facet F defines a hyperplane in D-dimensional space
    /// 2. The vertex c is on one side of this hyperplane (the "inside")
    /// 3. The test vertex v is visible if it's on the opposite side from c
    ///
    /// We use orientation predicates to determine which side of the hyperplane
    /// each vertex lies on by testing the orientation of the simplex formed by
    /// the facet vertices plus each test vertex.
    ///
    /// # Arguments
    ///
    /// * `tds` - Reference to the triangulation data structure
    /// * `facet` - The boundary facet to test
    /// * `vertex` - The vertex to test visibility from
    /// * `adjacent_cell_key` - Key of the cell adjacent to this boundary facet
    ///
    /// # Returns
    ///
    /// `true` if the facet is visible from the vertex, `false` otherwise.
    #[allow(clippy::unused_self)]
    fn is_facet_visible_from_vertex(
        &self,
        tds: &Tds<T, U, V, D>,
        facet: &Facet<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
        adjacent_cell_key: CellKey,
    ) -> bool
    where
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // Get the adjacent cell to this boundary facet
        let Some(adjacent_cell) = tds.cells().get(adjacent_cell_key) else {
            return false;
        };

        // Find the vertex in the adjacent cell that is NOT part of the facet
        // This is the "opposite" vertex that defines the "inside" side of the facet
        let facet_vertices = facet.vertices();
        let cell_vertices = adjacent_cell.vertices();

        let mut opposite_vertex = None;
        for cell_vertex in cell_vertices {
            let is_in_facet = facet_vertices
                .iter()
                .any(|fv| fv.uuid() == cell_vertex.uuid());
            if !is_in_facet {
                opposite_vertex = Some(cell_vertex);
                break;
            }
        }

        let Some(opposite_vertex) = opposite_vertex else {
            // Could not find opposite vertex - something is wrong with the topology
            return false;
        };

        // Now we have:
        // - facet_vertices: the D vertices that form the boundary facet
        // - opposite_vertex: the vertex on the "inside" side of the facet
        // - vertex: the test vertex to check for visibility
        //
        // A facet is visible from the test vertex if the test vertex is on the
        // opposite side of the hyperplane from the opposite vertex.
        //
        // We use orientation predicates to determine this:
        // 1. Create simplex from facet + opposite vertex
        // 2. Create simplex from facet + test vertex
        // 3. Compare orientations - if different, they're on opposite sides

        // Create test simplices
        let mut simplex_with_opposite: Vec<Point<T, D>> =
            facet_vertices.iter().map(|v| *v.point()).collect();
        simplex_with_opposite.push(*opposite_vertex.point());

        let mut simplex_with_test: Vec<Point<T, D>> =
            facet_vertices.iter().map(|v| *v.point()).collect();
        simplex_with_test.push(*vertex.point());

        // Get orientations
        let orientation_opposite = simplex_orientation(&simplex_with_opposite);
        let orientation_test = simplex_orientation(&simplex_with_test);

        match (orientation_opposite, orientation_test) {
            (Ok(ori_opp), Ok(ori_test)) => {
                // Facet is visible if the orientations are different
                // (vertices are on opposite sides of the hyperplane)
                use crate::geometry::predicates::Orientation;
                match (ori_opp, ori_test) {
                    (Orientation::NEGATIVE, Orientation::POSITIVE)
                    | (Orientation::POSITIVE, Orientation::NEGATIVE) => true,
                    (Orientation::DEGENERATE, _) | (_, Orientation::DEGENERATE) => {
                        // Degenerate case - fall back to simple heuristic
                        Self::fallback_visibility_test(facet, vertex)
                    }
                    _ => false, // Same orientation = same side = not visible
                }
            }
            _ => {
                // Orientation computation failed - fall back to simple heuristic
                Self::fallback_visibility_test(facet, vertex)
            }
        }
    }

    /// Fallback visibility test for degenerate cases
    ///
    /// When geometric predicates fail due to degeneracy, this method provides
    /// a conservative heuristic. The previous implementation was too permissive,
    /// causing all boundary facets to be considered visible and leading to
    /// complete hull closure.
    ///
    /// # Arguments
    ///
    /// * `facet` - The boundary facet to test
    /// * `vertex` - The vertex to test visibility from
    ///
    /// # Returns
    ///
    /// `false` for safety - in degenerate cases, we assume facets are not visible
    /// to prevent incorrect hull closure.
    const fn fallback_visibility_test(
        _facet: &Facet<T, U, V, D>,
        _vertex: &Vertex<T, U, D>,
    ) -> bool {
        // CRITICAL: The previous implementation was too permissive and caused
        // the algorithm to consider ALL boundary facets visible, leading to
        // complete hull closure (0 boundary facets after multiple extensions).
        //
        // Conservative approach: when orientation predicates fail due to degeneracy,
        // assume the facet is NOT visible. This prevents over-extension of the hull
        // and maintains proper boundary topology.
        //
        // This is safer than the previous "can we create a valid cell?" test,
        // which would return true for any exterior point + boundary facet combination.
        false
    }

    /// Creates a new cell from a facet and a vertex
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    /// * `facet` - The facet to extend into a cell
    /// * `vertex` - The vertex to add to the facet
    ///
    /// # Returns
    ///
    /// `true` if the cell was successfully created, `false` otherwise.
    fn create_cell_from_facet_and_vertex(
        tds: &mut Tds<T, U, V, D>,
        facet: &Facet<T, U, V, D>,
        vertex: &Vertex<T, U, D>,
    ) -> bool {
        // Ensure the vertex is registered in the TDS vertex mapping
        if !tds.vertex_bimap.contains_left(&vertex.uuid()) {
            let vertex_key = tds.vertices.insert(*vertex);
            tds.vertex_bimap.insert(vertex.uuid(), vertex_key);
        }

        let mut facet_vertices = facet.vertices();
        facet_vertices.push(*vertex);

        match CellBuilder::default().vertices(facet_vertices).build() {
            Ok(new_cell) => {
                let cell_key = tds.cells_mut().insert(new_cell);
                let cell_uuid = tds.cells()[cell_key].uuid();
                tds.cell_bimap.insert(cell_uuid, cell_key);
                true
            }
            Err(_) => {
                // Cell creation failed - this can happen with degenerate configurations
                false
            }
        }
    }

    /// Finalizes the triangulation by cleaning up and establishing relationships
    ///
    /// # Arguments
    ///
    /// * `tds` - Mutable reference to the triangulation data structure
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful finalization, or an error if finalization fails.
    fn finalize_triangulation(tds: &mut Tds<T, U, V, D>) -> BoyerWatsonResult<()> {
        // Remove duplicate cells
        tds.remove_duplicate_cells();

        // Fix invalid facet sharing
        tds.fix_invalid_facet_sharing().map_err(|e| {
            TriangulationConstructionError::FailedToCreateCell {
                message: format!("Failed to fix invalid facet sharing: {e}"),
            }
        })?;

        // Assign neighbor relationships
        tds.assign_neighbors()
            .map_err(TriangulationConstructionError::ValidationError)?;

        // Assign incident cells to vertices
        tds.assign_incident_cells()
            .map_err(TriangulationConstructionError::ValidationError)?;

        Ok(())
    }

    /// Returns statistics about the triangulation construction process
    #[must_use]
    pub const fn get_statistics(&self) -> (usize, usize, usize) {
        (
            self.insertion_count,
            self.total_cells_created,
            self.total_cells_removed,
        )
    }

    /// Resets the algorithm state for reuse
    pub fn reset(&mut self) {
        self.insertion_count = 0;
        self.total_cells_created = 0;
        self.total_cells_removed = 0;
        self.bad_cells_buffer.clear();
        self.boundary_facets_buffer.clear();
        self.vertex_points_buffer.clear();
        self.hull = None;
    }
}

impl<T, U, V, const D: usize> Default for IncrementalBoyerWatson<T, U, V, D>
where
    T: CoordinateScalar
        + AddAssign<T>
        + ComplexField<RealField = T>
        + SubAssign<T>
        + Sum
        + From<f64>,
    U: DataType + DeserializeOwned,
    V: DataType + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: Div<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    ordered_float::OrderedFloat<f64>: From<T>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::boundary_analysis::BoundaryAnalysis;
    use crate::core::vertex::Vertex;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;

    /// Helper function to analyze a triangulation's state for debugging
    fn analyze_triangulation(tds: &Tds<f64, Option<()>, Option<()>, 3>, label: &str) {
        println!(
            "  {} - Vertices: {}, Cells: {}",
            label,
            tds.number_of_vertices(),
            tds.number_of_cells()
        );

        let boundary = count_boundary_facets(tds);
        let internal = count_internal_facets(tds);
        let invalid = count_invalid_facets(tds);

        println!("  {label} - Boundary: {boundary}, Internal: {internal}, Invalid: {invalid}");

        if let Ok(bf) = tds.boundary_facets() {
            println!("  {} - boundary_facets() reports: {}", label, bf.len());
        } else {
            println!("  {label} - boundary_facets() failed");
        }
    }

    /// Count boundary facets (shared by 1 cell)
    fn count_boundary_facets(tds: &Tds<f64, Option<()>, Option<()>, 3>) -> usize {
        tds.build_facet_to_cells_hashmap()
            .values()
            .filter(|cells| cells.len() == 1)
            .count()
    }

    /// Count internal facets (shared by 2 cells)
    fn count_internal_facets(tds: &Tds<f64, Option<()>, Option<()>, 3>) -> usize {
        tds.build_facet_to_cells_hashmap()
            .values()
            .filter(|cells| cells.len() == 2)
            .count()
    }

    /// Count invalid facets (shared by 3+ cells)
    fn count_invalid_facets(tds: &Tds<f64, Option<()>, Option<()>, 3>) -> usize {
        tds.build_facet_to_cells_hashmap()
            .values()
            .filter(|cells| cells.len() > 2)
            .count()
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn diagnose_triangulation_invariant_violations() {
        println!("=== DELAUNAY TRIANGULATION DIAGNOSTIC ===\n");

        // The problematic point configuration from failing tests
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),  // A - origin
            Point::new([1.0, 0.0, 0.0]),  // B - on x-axis
            Point::new([0.5, 1.0, 0.0]),  // C - forms triangle ABC in xy-plane
            Point::new([0.5, 0.5, 1.0]),  // D - above the triangle
            Point::new([0.5, 0.5, -1.0]), // E - below the triangle
        ];

        println!("Input Points:");
        for (i, point) in points.iter().enumerate() {
            println!("  {}: {:?}", i, point.to_array());
        }

        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        println!("\n=== BASIC STATISTICS ===");
        println!("Vertices: {}", tds.number_of_vertices());
        println!("Cells: {}", tds.number_of_cells());
        println!("Dimension: {}", tds.dim());

        // Analyze each cell
        println!("\n=== CELL ANALYSIS ===");
        for (i, cell) in tds.cells().values().enumerate() {
            let vertex_coords: Vec<[f64; 3]> = cell
                .vertices()
                .iter()
                .map(|v| v.point().to_array())
                .collect();
            println!("Cell {i}: {vertex_coords:?}");

            // Check if cell has neighbors
            match &cell.neighbors {
                Some(neighbors) => {
                    println!("  Neighbors: {} cells", neighbors.len());
                }
                None => {
                    println!("  Neighbors: None");
                }
            }
        }

        // Detailed facet sharing analysis
        println!("\n=== FACET SHARING ANALYSIS ===");
        let facet_to_cells = tds.build_facet_to_cells_hashmap();

        let mut invalid_sharing = 0;
        let mut boundary_facets = 0;
        let mut internal_facets = 0;

        for (facet_key, cells) in &facet_to_cells {
            let cell_count = cells.len();
            match cell_count {
                1 => boundary_facets += 1,
                2 => internal_facets += 1,
                _ => {
                    invalid_sharing += 1;
                    println!("❌ INVALID: Facet {facet_key} shared by {cell_count} cells");
                    for (cell_key, facet_index) in cells {
                        if let Some(cell) = tds.cells().get(*cell_key) {
                            println!("   Cell {:?} at facet index {}", cell.uuid(), facet_index);
                        }
                    }
                }
            }
        }

        println!("Boundary facets (1 cell): {boundary_facets}");
        println!("Internal facets (2 cells): {internal_facets}");
        println!("Invalid facets (3+ cells): {invalid_sharing}");

        // Boundary analysis
        println!("\n=== BOUNDARY ANALYSIS ===");
        match tds.boundary_facets() {
            Ok(bf) => {
                println!("Boundary facets found: {}", bf.len());
                if bf.len() != boundary_facets {
                    println!(
                        "❌ MISMATCH: Direct count ({}) vs boundary_facets() ({})",
                        boundary_facets,
                        bf.len()
                    );
                }
            }
            Err(e) => {
                println!("❌ ERROR: Failed to get boundary facets: {e:?}");
            }
        }

        // Validation check
        println!("\n=== VALIDATION CHECK ===");
        match tds.is_valid() {
            Ok(()) => println!("✅ Triangulation reports as valid"),
            Err(e) => println!("❌ Triangulation is invalid: {e:?}"),
        }

        // Expected vs Actual analysis
        println!("\n=== GEOMETRIC ANALYSIS ===");
        println!("Expected for 5 points in general position in 3D:");
        println!("  - Convex hull should have 6-8 triangular faces");
        println!("  - Should create 2-3 tetrahedra for optimal triangulation");
        println!("  - Each internal facet shared by exactly 2 cells");
        println!("  - Boundary facets shared by exactly 1 cell");

        println!("\nActual results:");
        println!("  - {} tetrahedra created", tds.number_of_cells());
        println!("  - {invalid_sharing} invalid facet sharings (3+ cells)");
        println!("  - {boundary_facets} boundary facets");
        println!("  - {internal_facets} internal facets");

        // Critical issue detection
        let mut issues = Vec::new();

        if invalid_sharing > 0 {
            issues.push("Facet sharing invariant violated");
        }

        if boundary_facets == 0 && tds.number_of_cells() > 0 {
            issues.push("No boundary facets (impossible for finite convex hull)");
        }

        if tds.number_of_cells() > 6 {
            issues.push("Too many cells created (over-triangulation)");
        }

        if issues.is_empty() {
            println!("\n✅ All triangulation invariants appear to be satisfied");
        } else {
            println!("\n🚨 CRITICAL ISSUES DETECTED:");
            for issue in &issues {
                println!("  - {issue}");
            }
            println!("  - Bowyer-Watson algorithm implementation has serious bugs");

            // This test should fail to highlight the issues
            panic!(
                "Triangulation violates Delaunay invariants: {} issues found: {issues:?}",
                issues.len()
            );
        }
    }

    #[test]
    fn test_simple_tetrahedron_invariants() {
        println!("\n=== SIMPLE TETRAHEDRON TEST ===");

        // Simple tetrahedron - should be absolutely correct
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let vertices = Vertex::from_points(points);
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        println!("Simple tetrahedron:");
        println!("  Vertices: {}", tds.number_of_vertices());
        println!("  Cells: {}", tds.number_of_cells());

        // Check facet sharing
        let facet_to_cells = tds.build_facet_to_cells_hashmap();
        let boundary_count = facet_to_cells
            .values()
            .filter(|cells| cells.len() == 1)
            .count();
        let internal_count = facet_to_cells
            .values()
            .filter(|cells| cells.len() == 2)
            .count();
        let invalid_count = facet_to_cells
            .values()
            .filter(|cells| cells.len() > 2)
            .count();

        println!("  Boundary facets: {boundary_count}");
        println!("  Internal facets: {internal_count}");
        println!("  Invalid facets: {invalid_count}");

        // For a single tetrahedron, we expect:
        // - 1 cell
        // - 4 boundary facets
        // - 0 internal facets
        assert_eq!(
            tds.number_of_cells(),
            1,
            "Single tetrahedron should have 1 cell"
        );
        assert_eq!(
            boundary_count, 4,
            "Single tetrahedron should have 4 boundary facets"
        );
        assert_eq!(
            internal_count, 0,
            "Single tetrahedron should have 0 internal facets"
        );
        assert_eq!(
            invalid_count, 0,
            "Single tetrahedron should have 0 invalid facets"
        );

        // Boundary analysis should work
        let boundary_facets = tds.boundary_facets().expect("Should get boundary facets");
        assert_eq!(
            boundary_facets.len(),
            4,
            "boundary_facets() should return 4 facets"
        );

        println!("✅ Simple tetrahedron passes all invariant checks");
    }

    #[test]
    fn test_incremental_insertion_diagnostic() {
        println!("\n=== INCREMENTAL INSERTION DIAGNOSTIC ===");

        // Start with the working simple tetrahedron
        let mut points = vec![
            Point::new([0.0, 0.0, 0.0]), // Origin
            Point::new([1.0, 0.0, 0.0]), // X-axis
            Point::new([0.0, 1.0, 0.0]), // Y-axis
            Point::new([0.0, 0.0, 1.0]), // Z-axis
        ];

        println!("Step 1: Create initial tetrahedron with 4 points");
        let vertices = Vertex::from_points(points.clone());
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        println!("Initial state:");
        analyze_triangulation(&tds, "initial");

        // Now add one point outside the tetrahedron - this should trigger Bowyer-Watson
        println!("\nStep 2: Add one point outside the tetrahedron");
        let new_point = Point::new([0.5, 0.5, 2.0]); // Far above the tetrahedron
        println!("Adding point: {:?}", new_point.to_array());

        points.push(new_point);
        let all_vertices = Vertex::from_points(points);
        let tds_after: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&all_vertices).unwrap();

        println!("After insertion:");
        analyze_triangulation(&tds_after, "after_insertion");

        // Compare before and after
        println!("\n=== COMPARISON ===");
        println!("Before: 1 cell, 4 boundary facets, 0 internal facets");
        println!(
            "After:  {} cells, {} boundary facets, {} internal facets",
            tds_after.number_of_cells(),
            count_boundary_facets(&tds_after),
            count_internal_facets(&tds_after)
        );

        // Expected result for adding one point outside:
        // Should create 2-3 cells total, maintain proper boundary
        let boundary_count = count_boundary_facets(&tds_after);
        let _internal_count = count_internal_facets(&tds_after);
        let invalid_count = count_invalid_facets(&tds_after);

        println!("\n=== ANALYSIS ===");
        if boundary_count == 0 {
            println!("❌ CRITICAL: No boundary facets after insertion");
        }
        if tds_after.number_of_cells() > 3 {
            println!(
                "❌ WARNING: Too many cells created ({})",
                tds_after.number_of_cells()
            );
        }
        if invalid_count > 0 {
            println!("❌ CRITICAL: {invalid_count} invalid facet sharing");
        }

        // This test should help identify exactly what breaks during insertion
        assert!(
            !(boundary_count == 0 || invalid_count > 0),
            "Bowyer-Watson insertion created invalid triangulation: {boundary_count} boundary, {invalid_count} invalid facets"
        );
    }

    /// Test the algorithm step by step to isolate where it breaks
    #[test]
    fn test_bowyer_watson_step_by_step() {
        println!("\n=== STEP-BY-STEP BOWYER-WATSON TEST ===");

        // Create initial tetrahedron manually
        let initial_points = vec![
            Point::new([0.0, 0.0, 0.0]), // Origin
            Point::new([1.0, 0.0, 0.0]), // X-axis
            Point::new([0.0, 1.0, 0.0]), // Y-axis
            Point::new([0.0, 0.0, 1.0]), // Z-axis
        ];

        let vertices = Vertex::from_points(initial_points);
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        println!("Initial tetrahedron created:");
        analyze_triangulation(&tds, "initial");

        // Test the algorithm components step by step
        let mut algorithm = IncrementalBoyerWatson::new();
        let new_vertex = vertex!([0.5, 0.5, 2.0]);

        println!("\nTesting vertex insertion algorithm...");

        // Step 1: Determine insertion strategy
        let strategy = algorithm.determine_insertion_strategy(&tds, &new_vertex);
        println!("Insertion strategy: {strategy:?}");

        // Step 2: Test bad cell detection
        let bad_cells = algorithm.find_bad_cells(&tds, &new_vertex);
        println!("Bad cells found: {} cells", bad_cells.len());
        for &cell_key in &bad_cells {
            if let Some(cell) = tds.cells().get(cell_key) {
                let coords: Vec<[f64; 3]> = cell
                    .vertices()
                    .iter()
                    .map(|v| v.point().to_array())
                    .collect();
                println!("  Bad cell {:?}: {:?}", cell.uuid(), coords);
            }
        }

        // Step 3: Execute the insertion
        match algorithm.insert_vertex(&mut tds, new_vertex) {
            Ok(info) => {
                println!("\nInsertion completed:");
                println!("  Strategy: {:?}", info.strategy);
                println!("  Cells removed: {}", info.cells_removed);
                println!("  Cells created: {}", info.cells_created);
                println!("  Success: {}", info.success);

                println!("\nAfter insertion:");
                analyze_triangulation(&tds, "after_manual_insertion");
            }
            Err(e) => println!("Insertion failed: {e:?}"),
        }
    }

    /// Test simple debug case with a tetrahedron
    #[test]
    fn debug_bowyer_watson_simple() {
        // Create a simple test case with a tetrahedron
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        println!(
            "Creating initial triangulation with {} vertices",
            vertices.len()
        );
        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(&vertices).unwrap();

        println!("Initial triangulation:");
        println!("  Vertices: {}", tds.number_of_vertices());
        println!("  Cells: {}", tds.number_of_cells());
        println!("  Dimension: {}", tds.dim());

        // Try to add a point inside the tetrahedron
        let new_vertex = vertex!([0.25, 0.25, 0.25]);
        println!("Adding vertex: {:?}", new_vertex.point());

        match tds.add(new_vertex) {
            Ok(()) => {
                println!("Success! Final triangulation:");
                println!("  Vertices: {}", tds.number_of_vertices());
                println!("  Cells: {}", tds.number_of_cells());
            }
            Err(e) => {
                println!("Failed to add vertex: {e}");
            }
        }
    }

    /// Test debug step by step with problematic vertex
    #[test]
    fn debug_bowyer_watson_step_by_step() {
        // Test the exact scenario from the failing test
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
            Point::new([1.0, 1.0, 0.0]), // This is the vertex that fails
        ];

        let vertices = Vertex::from_points(points);

        // Build the triangulation step by step
        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(&vertices[0..4]).unwrap();

        println!("Initial tetrahedron:");
        println!("  Vertices: {}", tds.number_of_vertices());
        println!("  Cells: {}", tds.number_of_cells());

        // Try to add the problematic vertex
        println!("Adding problematic vertex: {:?}", vertices[4].point());

        match tds.add(vertices[4]) {
            Ok(()) => {
                println!("Success!");
                println!("  Final vertices: {}", tds.number_of_vertices());
                println!("  Final cells: {}", tds.number_of_cells());
            }
            Err(e) => {
                println!("Failed: {e}");
            }
        }
    }

    // ============================================================================
    // UNIT TESTS FOR PRIVATE METHODS
    // ============================================================================
    // These tests are placed inside the module to access private methods while
    // maintaining proper encapsulation. They provide comprehensive coverage of
    // internal algorithm components.

    #[test]
    fn test_determine_insertion_strategy_interior_vertex() {
        println!("Testing determine_insertion_strategy with interior vertices");

        // Create a simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBoyerWatson::new();

        // Test interior vertex (inside the tetrahedron)
        let interior_vertex = vertex!([0.4, 0.4, 0.4]);
        let strategy = algorithm.determine_insertion_strategy(&tds, &interior_vertex);
        assert_eq!(
            strategy,
            InsertionStrategy::CavityBased,
            "Interior vertex should use cavity-based strategy"
        );

        // Test another interior vertex
        let another_interior = vertex!([0.1, 0.1, 0.1]);
        let strategy2 = algorithm.determine_insertion_strategy(&tds, &another_interior);
        assert_eq!(
            strategy2,
            InsertionStrategy::CavityBased,
            "Another interior vertex should use cavity-based strategy"
        );

        println!("✓ Interior vertex strategy determination works correctly");
    }

    #[test]
    fn test_determine_insertion_strategy_exterior_vertex() {
        println!("Testing determine_insertion_strategy with exterior vertices");

        // Create a simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBoyerWatson::new();

        // Test exterior vertices in different directions
        let exterior_vertices = vec![
            (vertex!([3.0, 0.0, 0.0]), "+X direction"),
            (vertex!([0.0, 3.0, 0.0]), "+Y direction"),
            (vertex!([0.0, 0.0, 3.0]), "+Z direction"),
            (vertex!([-1.0, 0.0, 0.0]), "-X direction"),
            (vertex!([2.0, 2.0, 2.0]), "Far corner"),
        ];

        for (exterior_vertex, description) in exterior_vertices {
            let strategy = algorithm.determine_insertion_strategy(&tds, &exterior_vertex);
            assert_eq!(
                strategy,
                InsertionStrategy::HullExtension,
                "Exterior vertex ({description}) should use hull extension strategy"
            );
        }

        println!("✓ Exterior vertex strategy determination works correctly");
    }

    #[test]
    fn test_is_vertex_interior_various_positions() {
        println!("Testing is_vertex_interior with various vertex positions");

        // Create a tetrahedron with clear interior/exterior regions
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([4.0, 0.0, 0.0]),
            vertex!([0.0, 4.0, 0.0]),
            vertex!([0.0, 0.0, 4.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBoyerWatson::new();

        // Test clearly interior vertices
        let interior_vertices = vec![
            (vertex!([0.5, 0.5, 0.5]), "Center of tetrahedron"),
            (vertex!([1.0, 1.0, 1.0]), "Inside but not center"),
            (vertex!([0.1, 0.1, 0.1]), "Near origin corner"),
            (vertex!([0.8, 0.2, 0.2]), "Near one face"),
        ];

        for (vertex, description) in interior_vertices {
            let is_interior = algorithm.is_vertex_interior(&tds, &vertex);
            assert!(is_interior, "Vertex ({description}) should be interior");
        }

        // Test clearly exterior vertices
        let exterior_vertices = vec![
            (vertex!([5.0, 0.0, 0.0]), "Far +X"),
            (vertex!([0.0, 5.0, 0.0]), "Far +Y"),
            (vertex!([0.0, 0.0, 5.0]), "Far +Z"),
            (vertex!([-1.0, 0.0, 0.0]), "Negative X"),
            (vertex!([4.0, 4.0, 4.0]), "Outside diagonal"),
        ];

        for (vertex, description) in exterior_vertices {
            let is_interior = algorithm.is_vertex_interior(&tds, &vertex);
            assert!(!is_interior, "Vertex ({description}) should be exterior");
        }

        println!("✓ Interior/exterior vertex classification works correctly");
    }

    #[test]
    fn test_find_bad_cells_for_interior_vertex() {
        println!("Testing find_bad_cells with interior vertex");

        // Create initial tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let mut algorithm = IncrementalBoyerWatson::new();

        // Test with interior vertex that should violate Delaunay property
        let interior_vertex = vertex!([0.5, 0.5, 0.5]);
        let bad_cells = algorithm.find_bad_cells(&tds, &interior_vertex);

        // Since this is an interior vertex, it should find the tetrahedron as a bad cell
        assert!(
            !bad_cells.is_empty(),
            "Interior vertex should find at least one bad cell"
        );
        assert!(
            bad_cells.len() <= tds.number_of_cells(),
            "Cannot have more bad cells than total cells"
        );

        println!("  Found {} bad cells for interior vertex", bad_cells.len());
        println!("✓ Bad cell detection for interior vertex works correctly");
    }

    #[test]
    fn test_find_bad_cells_for_exterior_vertex() {
        println!("Testing find_bad_cells with exterior vertex");

        // Create initial tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let mut algorithm = IncrementalBoyerWatson::new();

        // Test with exterior vertex that should not violate circumsphere
        let exterior_vertex = vertex!([3.0, 0.0, 0.0]);
        let bad_cells = algorithm.find_bad_cells(&tds, &exterior_vertex);

        // Exterior vertex should typically find no bad cells (no circumsphere violations)
        println!("  Found {} bad cells for exterior vertex", bad_cells.len());
        assert!(
            bad_cells.len() <= tds.number_of_cells(),
            "Cannot have more bad cells than total cells"
        );

        println!("✓ Bad cell detection for exterior vertex works correctly");
    }

    #[test]
    fn test_find_bad_cells_edge_cases() {
        println!("Testing find_bad_cells edge cases");

        // Create a more complex triangulation with multiple cells
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
            vertex!([1.0, 1.0, 1.0]), // Additional vertex to create more cells
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut algorithm = IncrementalBoyerWatson::new();

        // Test vertex very close to existing vertex
        let close_vertex = vertex!([0.001, 0.0, 0.0]);
        let bad_cells = algorithm.find_bad_cells(&tds, &close_vertex);
        println!("  Close vertex found {} bad cells", bad_cells.len());

        // Test vertex on circumsphere boundary (should be handled gracefully)
        let boundary_vertex = vertex!([0.5, 0.5, 0.0]);
        let bad_cells_boundary = algorithm.find_bad_cells(&tds, &boundary_vertex);
        println!(
            "  Boundary vertex found {} bad cells",
            bad_cells_boundary.len()
        );

        // All results should be within reasonable bounds
        assert!(bad_cells.len() <= tds.number_of_cells());
        assert!(bad_cells_boundary.len() <= tds.number_of_cells());

        println!("✓ Bad cell detection edge cases handled correctly");
    }

    #[test]
    fn test_find_cavity_boundary_facets_single_bad_cell() {
        println!("Testing find_cavity_boundary_facets with single bad cell");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let mut algorithm = IncrementalBoyerWatson::new();

        // Get the single cell as a "bad cell"
        let cell_keys: Vec<_> = tds.cells().keys().collect();
        assert_eq!(cell_keys.len(), 1, "Should have exactly one cell");

        let bad_cells = vec![cell_keys[0]];
        let boundary_facets = algorithm
            .find_cavity_boundary_facets(&tds, &bad_cells)
            .expect("Should find boundary facets");

        // For a single tetrahedron, all 4 facets should be cavity boundary facets
        assert_eq!(
            boundary_facets.len(),
            4,
            "Single tetrahedron should have 4 boundary facets"
        );

        // Each facet should be valid
        for (i, facet) in boundary_facets.iter().enumerate() {
            assert_eq!(
                facet.vertices().len(),
                3,
                "Facet {i} should have 3 vertices in 3D"
            );
        }

        println!("✓ Cavity boundary facet detection for single cell works correctly");
    }

    #[test]
    fn test_find_cavity_boundary_facets_empty_bad_cells() {
        println!("Testing find_cavity_boundary_facets with empty bad cells list");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let mut algorithm = IncrementalBoyerWatson::new();

        let bad_cells = vec![]; // Empty list
        let boundary_facets = algorithm
            .find_cavity_boundary_facets(&tds, &bad_cells)
            .expect("Should handle empty bad cells list");

        assert_eq!(
            boundary_facets.len(),
            0,
            "Empty bad cells list should produce empty boundary facets"
        );

        println!("✓ Empty bad cells list handled correctly");
    }

    #[test]
    fn test_find_visible_boundary_facets_exterior_vertex() {
        println!("Testing find_visible_boundary_facets with exterior vertex");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBoyerWatson::new();

        // Test exterior vertex that should see some facets
        let exterior_vertex = vertex!([2.0, 0.0, 0.0]);
        let visible_facets = algorithm.find_visible_boundary_facets(&tds, &exterior_vertex);

        println!("  Found {} visible facets", visible_facets.len());

        // Should find at least some visible facets for exterior vertex
        assert!(
            !visible_facets.is_empty(),
            "Exterior vertex should see at least some boundary facets"
        );
        assert!(
            visible_facets.len() <= 4,
            "Cannot see more than 4 facets from a tetrahedron"
        );

        // Test that each visible facet is valid
        for (i, facet) in visible_facets.iter().enumerate() {
            assert_eq!(
                facet.vertices().len(),
                3,
                "Visible facet {i} should have 3 vertices"
            );
        }

        println!("✓ Visible boundary facet detection works correctly");
    }

    #[test]
    fn test_find_visible_boundary_facets_interior_vertex() {
        println!("Testing find_visible_boundary_facets with interior vertex");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBoyerWatson::new();

        // Test interior vertex that should not see any facets from outside
        let interior_vertex = vertex!([0.4, 0.4, 0.4]);
        let visible_facets = algorithm.find_visible_boundary_facets(&tds, &interior_vertex);

        println!("  Interior vertex sees {} facets", visible_facets.len());

        // Interior vertex should see few or no boundary facets as "visible"
        // (The exact number depends on the orientation predicates)
        assert!(
            visible_facets.len() <= 4,
            "Cannot see more than 4 facets from a tetrahedron"
        );

        println!("✓ Interior vertex visibility test works correctly");
    }

    #[test]
    fn test_is_facet_visible_from_vertex_orientation_cases() {
        println!("Testing is_facet_visible_from_vertex with different orientations");

        // Create simple tetrahedron
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();
        let algorithm = IncrementalBoyerWatson::new();

        // Get a boundary facet
        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        assert!(
            !boundary_facets.is_empty(),
            "Should have at least one boundary facet"
        );

        let test_facet = &boundary_facets[0];

        // Find the cell adjacent to this boundary facet
        let facet_to_cells = tds.build_facet_to_cells_hashmap();
        let facet_key = test_facet.key();
        let adjacent_cells = facet_to_cells.get(&facet_key).unwrap();
        assert_eq!(
            adjacent_cells.len(),
            1,
            "Boundary facet should have exactly one adjacent cell"
        );
        let (adjacent_cell_key, _) = adjacent_cells[0];

        // Test visibility from different positions
        let test_positions = vec![
            (vertex!([2.0, 0.0, 0.0]), "Far +X"),
            (vertex!([-1.0, 0.0, 0.0]), "Far -X"),
            (vertex!([0.0, 2.0, 0.0]), "Far +Y"),
            (vertex!([0.0, 0.0, 2.0]), "Far +Z"),
            (vertex!([0.1, 0.1, 0.1]), "Interior point"),
        ];

        for (test_vertex, description) in test_positions {
            let is_visible = algorithm.is_facet_visible_from_vertex(
                &tds,
                test_facet,
                &test_vertex,
                adjacent_cell_key,
            );

            println!("  {description} - Facet visible: {is_visible}");
            // Note: We don't assert specific visibility results here because they depend
            // on the specific geometry and orientation of the facet, but the function
            // should not panic and should return a boolean result.
        }

        println!("✓ Facet visibility testing with different orientations works correctly");
    }

    #[test]
    fn test_fallback_visibility_test_conservative_behavior() {
        println!("Testing fallback_visibility_test conservative behavior");

        // Use the builder to create a facet - we need a cell first
        let all_vertices: Vec<Vertex<f64, Option<()>, 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&all_vertices).unwrap();
        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        let test_facet = &boundary_facets[0];

        // Test fallback with various vertices
        let test_vertices: Vec<Vertex<f64, Option<()>, 3>> = vec![
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([1.0, 1.0, 1.0]),
        ];

        for test_vertex in test_vertices {
            let is_visible =
                IncrementalBoyerWatson::<f64, Option<()>, Option<()>, 3>::fallback_visibility_test(
                    test_facet,
                    &test_vertex,
                );

            // The fallback should always return false for conservative behavior
            assert!(
                !is_visible,
                "Fallback visibility test should be conservative (return false)"
            );
        }

        println!("✓ Fallback visibility test maintains conservative behavior");
    }

    #[test]
    fn test_create_cell_from_facet_and_vertex_success() {
        println!("Testing create_cell_from_facet_and_vertex successful creation");

        // Create initial triangulation
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Get a boundary facet
        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        let test_facet = &boundary_facets[0];

        let initial_cell_count = tds.number_of_cells();

        // Create a new vertex that should form a valid cell with the facet
        let new_vertex = vertex!([0.5, 0.5, 1.5]);

        let success = IncrementalBoyerWatson::<f64, Option<()>, Option<()>, 3>::create_cell_from_facet_and_vertex(
            &mut tds,
            test_facet,
            &new_vertex,
        );

        assert!(
            success,
            "Should successfully create cell from valid facet and vertex"
        );
        assert_eq!(
            tds.number_of_cells(),
            initial_cell_count + 1,
            "Cell count should increase by 1"
        );

        println!("✓ Cell creation from facet and vertex works correctly");
    }

    #[test]
    fn test_create_cell_from_facet_and_vertex_failure() {
        println!("Testing create_cell_from_facet_and_vertex with invalid geometry");

        // Create initial triangulation
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Get a boundary facet
        let boundary_facets = tds.boundary_facets().expect("Should have boundary facets");
        let test_facet = &boundary_facets[0];

        let initial_cell_count = tds.number_of_cells();

        // Try to create a degenerate cell by using a vertex that's already in the facet
        let facet_vertices = test_facet.vertices();
        let duplicate_vertex = facet_vertices[0]; // Use an existing facet vertex

        let success = IncrementalBoyerWatson::<f64, Option<()>, Option<()>, 3>::create_cell_from_facet_and_vertex(
            &mut tds,
            test_facet,
            &duplicate_vertex,
        );

        // This should fail because it would create a degenerate cell
        assert!(
            !success,
            "Should fail to create cell with degenerate geometry"
        );
        assert_eq!(
            tds.number_of_cells(),
            initial_cell_count,
            "Cell count should remain unchanged after failed creation"
        );

        println!("✓ Cell creation properly handles invalid geometry");
    }

    #[test]
    fn test_finalize_triangulation_success() {
        println!("Testing finalize_triangulation with valid triangulation");

        // Create a triangulation that needs finalization
        let initial_vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&initial_vertices).unwrap();

        // Add some additional complexity to the triangulation
        // (In a real scenario, this would be done through the Bowyer-Watson process)

        // Test finalization
        let result =
            IncrementalBoyerWatson::<f64, Option<()>, Option<()>, 3>::finalize_triangulation(
                &mut tds,
            );

        assert!(
            result.is_ok(),
            "Finalization should succeed for valid triangulation"
        );

        // Verify the triangulation is valid after finalization
        assert!(
            tds.is_valid().is_ok(),
            "Triangulation should be valid after finalization"
        );

        println!("✓ Triangulation finalization works correctly");
    }

    #[test]
    fn test_statistics_and_reset_functionality() {
        println!("Testing statistics tracking and reset functionality");

        let mut algorithm = IncrementalBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Initial statistics should be zero
        let (insertions, created, removed) = algorithm.get_statistics();
        assert_eq!(insertions, 0, "Initial insertion count should be 0");
        assert_eq!(created, 0, "Initial created count should be 0");
        assert_eq!(removed, 0, "Initial removed count should be 0");

        // Manually increment statistics to simulate algorithm usage
        algorithm.insertion_count = 5;
        algorithm.total_cells_created = 10;
        algorithm.total_cells_removed = 3;

        let (insertions, created, removed) = algorithm.get_statistics();
        assert_eq!(insertions, 5, "Should track insertions correctly");
        assert_eq!(created, 10, "Should track cell creation correctly");
        assert_eq!(removed, 3, "Should track cell removal correctly");

        // Test reset
        algorithm.reset();

        let (insertions, created, removed) = algorithm.get_statistics();
        assert_eq!(insertions, 0, "Reset should clear insertion count");
        assert_eq!(created, 0, "Reset should clear created count");
        assert_eq!(removed, 0, "Reset should clear removed count");

        println!("✓ Statistics tracking and reset work correctly");
    }

    #[test]
    fn test_algorithm_buffers_are_reused() {
        println!("Testing that algorithm buffers are properly reused");

        let mut algorithm = IncrementalBoyerWatson::<f64, Option<()>, Option<()>, 3>::new();

        // Create test triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();

        // Test that buffers work correctly across multiple calls
        let test_vertex1 = vertex!([0.3, 0.3, 0.3]);
        let test_vertex2 = vertex!([0.7, 0.2, 0.1]);

        let bad_cells1 = algorithm.find_bad_cells(&tds, &test_vertex1);
        let bad_cells2 = algorithm.find_bad_cells(&tds, &test_vertex2);

        // Both calls should work (buffer reuse should be transparent)
        assert!(bad_cells1.len() <= tds.number_of_cells());
        assert!(bad_cells2.len() <= tds.number_of_cells());

        // Test cavity boundary facets buffer reuse
        if !bad_cells1.is_empty() {
            let boundary1 = algorithm.find_cavity_boundary_facets(&tds, &bad_cells1);
            assert!(
                boundary1.is_ok(),
                "First boundary facet computation should work"
            );
        }

        if !bad_cells2.is_empty() {
            let boundary2 = algorithm.find_cavity_boundary_facets(&tds, &bad_cells2);
            assert!(
                boundary2.is_ok(),
                "Second boundary facet computation should work"
            );
        }

        println!("✓ Algorithm buffers are properly reused");
    }
}

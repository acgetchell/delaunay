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

use crate::core::cell::Cell;
use crate::core::collections::{CellKeySet, ValidCellsBuffer, VertexKeySet};
use crate::core::facet::{AllFacetsIter, BoundaryFacetsIter};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, VertexKey};
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::geometry::quality::radius_ratio;
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

    /// Fixes invalid facet sharing by removing problematic cells using geometric quality metrics.
    ///
    /// This method belongs in the Triangulation layer (not Tds) because it uses geometric
    /// quality metrics (`radius_ratio`) to select which cells to keep when a facet is
    /// shared by more than 2 cells.
    ///
    /// # Returns
    ///
    /// Number of cells removed.
    ///
    /// # Errors
    ///
    /// Returns error if facet map cannot be built or topology repair fails.
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

        #[allow(unused_variables)] // iteration only used in debug_assertions
        for iteration in 0..MAX_FIX_FACET_ITERATIONS {
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
                total_removed += actually_removed;
                continue;
            };

            // Rebuild topology if needed
            if actually_removed > 0 && duplicate_cells_removed == 0 {
                if self.tds.assign_neighbors().is_err() {
                    total_removed += actually_removed;
                    continue;
                }
                if self.tds.assign_incident_cells().is_err() {
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

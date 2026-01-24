//! Delaunay empty-circumsphere property validation utilities.

use std::ops::{AddAssign, SubAssign};

use num_traits::cast::NumCast;
use smallvec::SmallVec;
use thiserror::Error;

use crate::core::cell::CellValidationError;
use crate::core::collections::ViolationBuffer;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, TdsValidationError, VertexKey};
use crate::geometry::point::Point;
use crate::geometry::predicates::InSphere;
use crate::geometry::robust_predicates::robust_insphere;
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};

/// Errors that can occur during Delaunay property validation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum DelaunayValidationError {
    /// A cell violates the Delaunay property (has an external vertex inside its circumsphere).
    #[error("Cell violates Delaunay property: cell contains vertex that is inside circumsphere")]
    DelaunayViolation {
        /// The key of the cell that violates the Delaunay property
        cell_key: CellKey,
    },
    /// TDS data structure corruption or other structural issues detected during validation.
    #[error("TDS corruption: {source}")]
    TriangulationState {
        /// The underlying TDS validation error (TDS-level invariants).
        #[source]
        source: TdsValidationError,
    },
    /// Invalid cell structure detected during validation.
    #[error("Invalid cell {cell_key:?}: {source}")]
    InvalidCell {
        /// The key of the invalid cell.
        cell_key: CellKey,
        /// The underlying cell error.
        #[source]
        source: CellValidationError,
    },
    /// Numeric predicate failure during Delaunay validation.
    #[error(
        "Numeric predicate failure while validating Delaunay property for cell {cell_key:?}, vertex {vertex_key:?}: {source}"
    )]
    NumericPredicateError {
        /// The key of the cell whose circumsphere was being tested.
        cell_key: CellKey,
        /// The key of the vertex being classified relative to the circumsphere.
        vertex_key: VertexKey,
        /// Underlying robust predicate error (e.g., conversion failure).
        #[source]
        source: CoordinateConversionError,
    },
}

// =============================================================================
// DELAUNAY PROPERTY VALIDATION
// =============================================================================

/// Internal helper: Check if a single cell violates the Delaunay property.
///
/// Returns `Ok(None)` if the cell satisfies the Delaunay property,
/// `Ok(Some(cell_key))` if it violates (has an external vertex inside its circumsphere),
/// or `Err(...)` if validation fails due to structural or numeric issues.
fn validate_cell_delaunay<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
    cell_vertex_points: &mut SmallVec<[Point<T, D>; 8]>,
    config: &crate::geometry::robust_predicates::RobustPredicateConfig<T>,
) -> Result<Option<CellKey>, DelaunayValidationError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    U: DataType,
    V: DataType,
{
    let Some(cell) = tds.get_cell(cell_key) else {
        // Cell doesn't exist (possibly removed), skip validation
        return Ok(None);
    };

    // Validate cell structure first
    cell.is_valid()
        .map_err(|source| DelaunayValidationError::InvalidCell { cell_key, source })?;

    // Get the cell's vertex set for exclusion
    let cell_vertex_keys: SmallVec<[VertexKey; 8]> = cell.vertices().iter().copied().collect();

    // Build the cell's circumsphere
    cell_vertex_points.clear();
    for &vkey in &cell_vertex_keys {
        let Some(v) = tds.get_vertex_by_key(vkey) else {
            return Err(DelaunayValidationError::TriangulationState {
                source: TdsValidationError::InconsistentDataStructure {
                    message: format!("Cell {cell_key:?} references non-existent vertex {vkey:?}"),
                },
            });
        };
        cell_vertex_points.push(*v.point());
    }

    // Check if any OTHER vertex is inside this cell's circumsphere
    for (test_vkey, test_vertex) in tds.vertices() {
        // Skip if this vertex is part of the cell
        if cell_vertex_keys.contains(&test_vkey) {
            continue;
        }

        // Test if this vertex is inside the cell's circumsphere using ROBUST predicates
        match robust_insphere(cell_vertex_points, test_vertex.point(), config) {
            Ok(InSphere::INSIDE) => {
                // Found a violation - this cell has an external vertex inside its circumsphere
                return Ok(Some(cell_key));
            }
            Ok(InSphere::BOUNDARY | InSphere::OUTSIDE) => {
                // Vertex is outside/on boundary; continue checking other vertices
            }
            Err(source) => {
                // Surface robust predicate failures as explicit validation errors
                return Err(DelaunayValidationError::NumericPredicateError {
                    cell_key,
                    vertex_key: test_vkey,
                    source,
                });
            }
        }
    }

    Ok(None)
}

/// Check if a triangulation satisfies the Delaunay property.
///
/// The Delaunay property states that no vertex should be inside the circumsphere
/// of any cell. This function checks all cells in the triangulation using robust
/// geometric predicates.
///
/// # ⚠️ Performance Warning
///
/// **This function is extremely expensive** - O(N×V) where N is the number of cells
/// and V is the number of vertices. For a triangulation with 10,000 cells and 5,000
/// vertices, this performs 50 million insphere tests. Use this primarily for:
/// - Debugging and testing
/// - Final validation after construction
/// - Verification of algorithm correctness
///
/// **Do NOT use this in production hot paths or for every vertex insertion.**
///
/// # Arguments
///
/// * `tds` - The triangulation data structure to validate
///
/// # Returns
///
/// `Ok(())` if all cells satisfy the Delaunay property, otherwise a [`DelaunayValidationError`]
/// describing the first violation found.
///
/// # Errors
///
/// Returns:
/// - [`DelaunayValidationError::DelaunayViolation`] if a cell has an external vertex inside its circumsphere
/// - [`DelaunayValidationError::TriangulationState`] if TDS corruption is detected
/// - [`DelaunayValidationError::InvalidCell`] if a cell has invalid structure
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::core::util::is_delaunay;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
///
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// // Check if triangulation is Delaunay
/// assert!(is_delaunay(tds).is_ok());
/// ```
#[deprecated(
    since = "0.6.1",
    note = "Use `DelaunayTriangulation::is_valid()` for Delaunay property validation (Level 4) or `DelaunayTriangulation::validate()` for layered validation (Levels 1-4). This will be removed in v0.7.0."
)]
pub fn is_delaunay<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<(), DelaunayValidationError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    U: DataType,
    V: DataType,
{
    // PERFORMANCE: O(N×V) - extremely expensive, use only for testing/validation
    // Check structural invariants first to distinguish "bad triangulation" from
    // "good triangulation but non-Delaunay"
    tds.is_valid()
        .map_err(|source| DelaunayValidationError::TriangulationState { source })?;

    is_delaunay_property_only(tds)
}

/// Internal helper: validate the Delaunay empty-circumsphere property only.
///
/// This performs the expensive geometric check but intentionally does **not** run
/// `tds.is_valid()` up front. Callers that want cumulative validation should run
/// lower-layer checks separately.
pub(crate) fn is_delaunay_property_only<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<(), DelaunayValidationError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    U: DataType,
    V: DataType,
{
    // Use robust predicates configuration for reliability
    let config = crate::geometry::robust_predicates::config_presets::general_triangulation::<T>();

    // Reusable buffer to minimize allocations
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

    // Check each cell using the shared validation helper
    for cell_key in tds.cell_keys() {
        if let Some(violating_cell) =
            validate_cell_delaunay(tds, cell_key, &mut cell_vertex_points, &config)?
        {
            return Err(DelaunayValidationError::DelaunayViolation {
                cell_key: violating_cell,
            });
        }
    }

    Ok(())
}

/// Find cells that violate the Delaunay property.
///
/// This is a variant of [`is_delaunay`] that returns ALL violating cells instead of
/// stopping at the first violation. This is useful for iterative cavity refinement
/// and debugging.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `cells_to_check` - Optional subset of cells to check. If `None`, checks all cells.
///   Missing cells (e.g., already removed during refinement) are silently skipped.
///
/// # Returns
///
/// A vector of `CellKey`s for cells that violate the Delaunay property.
///
/// # Errors
///
/// Returns [`DelaunayValidationError`] if:
/// - A cell references a non-existent vertex (TDS corruption)
/// - A cell has invalid structure (cell-level corruption)
/// - Robust geometric predicates fail (numerical issues)
///
/// Note: Missing cells in `cells_to_check` are silently skipped and do not cause errors,
/// as they may have been legitimately removed during iterative refinement.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::core::util::find_delaunay_violations;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
///
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// // Find all violating cells (should be empty for valid Delaunay triangulation)
/// let violations = find_delaunay_violations(tds, None).unwrap();
/// assert!(violations.is_empty());
/// ```
pub fn find_delaunay_violations<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cells_to_check: Option<&[CellKey]>,
) -> Result<ViolationBuffer, DelaunayValidationError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    U: DataType,
    V: DataType,
{
    let mut violating_cells = ViolationBuffer::new();
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

    // Use robust predicates configuration for reliability
    let config = crate::geometry::robust_predicates::config_presets::general_triangulation::<T>();

    #[cfg(any(test, debug_assertions))]
    match cells_to_check {
        Some(keys) => eprintln!(
            "[Delaunay debug] find_delaunay_violations: checking {} requested cells",
            keys.len()
        ),
        None => eprintln!("[Delaunay debug] find_delaunay_violations: checking all cells"),
    }

    #[cfg(any(test, debug_assertions))]
    let mut processed_cells = 0usize;

    // Helper closure to process cells
    let mut process_cell = |cell_key: CellKey| -> Result<(), DelaunayValidationError> {
        #[cfg(any(test, debug_assertions))]
        {
            processed_cells += 1;
        }

        if let Some(violating_cell) =
            validate_cell_delaunay(tds, cell_key, &mut cell_vertex_points, &config)?
        {
            violating_cells.push(violating_cell);
        }
        Ok(())
    };

    // Process cells based on input
    match cells_to_check {
        Some(keys) => {
            for &cell_key in keys {
                process_cell(cell_key)?;
            }
        }
        None => {
            for cell_key in tds.cell_keys() {
                process_cell(cell_key)?;
            }
        }
    }

    #[cfg(any(test, debug_assertions))]
    eprintln!(
        "[Delaunay debug] find_delaunay_violations: processed {} cells, found {} violating cells",
        processed_cells,
        violating_cells.len()
    );

    Ok(violating_cells)
}

/// Debug helper: print detailed information about the first detected Delaunay
/// violation (or all vertices if none are found) to aid in debugging.
///
/// This function is intended for use in tests and debug builds only. It uses the
/// same robust predicates as [`is_delaunay`] / [`find_delaunay_violations`] and
/// prints:
/// - A triangulation summary (vertex and cell counts)
/// - All vertices (keys, UUIDs, coordinates)
/// - All violating cells' vertices
/// - For the first violating cell:
///   - At least one offending external vertex (if found)
///   - Neighbor information for each facet
#[cfg(any(test, debug_assertions))]
#[expect(
    clippy::too_many_lines,
    reason = "Debug-only helper with intentionally verbose logging"
)]
pub fn debug_print_first_delaunay_violation<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cells_subset: Option<&[CellKey]>,
) where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum + NumCast,
    U: DataType,
    V: DataType,
{
    use crate::geometry::robust_predicates::config_presets;

    // First, find violating cells using the standard helper.
    let violations = match find_delaunay_violations(tds, cells_subset) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "[Delaunay debug] debug_print_first_delaunay_violation: error while finding violations: {e}"
            );
            return;
        }
    };

    eprintln!(
        "[Delaunay debug] Triangulation summary: {} vertices, {} cells",
        tds.number_of_vertices(),
        tds.number_of_cells()
    );

    // Dump all input vertices once for reproducibility.
    for (vkey, vertex) in tds.vertices() {
        eprintln!(
            "[Delaunay debug] Vertex {:?}: uuid={}, point={:?}",
            vkey,
            vertex.uuid(),
            vertex.point()
        );
    }

    if violations.is_empty() {
        eprintln!("[Delaunay debug] No Delaunay violations detected for requested cell subset");
        return;
    }

    eprintln!(
        "[Delaunay debug] Delaunay violations detected in {} cell(s):",
        violations.len()
    );

    // Reusable buffer for cell vertex points.
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

    // Dump each violating cell with its vertices.
    for cell_key in &violations {
        match tds.get_cell(*cell_key) {
            Some(cell) => {
                eprintln!(
                    "[Delaunay debug]  Cell {:?}: uuid={}, vertices:",
                    cell_key,
                    cell.uuid()
                );
                for &vkey in cell.vertices() {
                    match tds.get_vertex_by_key(vkey) {
                        Some(v) => {
                            eprintln!(
                                "[Delaunay debug]    vkey={:?}, uuid={}, point={:?}",
                                vkey,
                                v.uuid(),
                                v.point()
                            );
                        }
                        None => {
                            eprintln!("[Delaunay debug]    vkey={vkey:?} (missing in TDS)");
                        }
                    }
                }
            }
            None => {
                eprintln!(
                    "[Delaunay debug]  Cell {cell_key:?} not found in TDS during violation dump"
                );
            }
        }
    }

    // Focus on the first violating cell to identify at least one offending
    // external vertex and neighbor information.
    let first_cell_key = violations[0];
    let Some(cell) = tds.get_cell(first_cell_key) else {
        eprintln!("[Delaunay debug] First violating cell {first_cell_key:?} not found in TDS");
        return;
    };

    let cell_vertex_keys: SmallVec<[VertexKey; 8]> = cell.vertices().iter().copied().collect();

    cell_vertex_points.clear();
    for &vkey in &cell_vertex_keys {
        if let Some(v) = tds.get_vertex_by_key(vkey) {
            cell_vertex_points.push(*v.point());
        }
    }

    let config = config_presets::general_triangulation::<T>();
    let mut offending: Option<(VertexKey, Point<T, D>)> = None;

    for (test_vkey, test_vertex) in tds.vertices() {
        if cell_vertex_keys.contains(&test_vkey) {
            continue;
        }

        match robust_insphere(&cell_vertex_points, test_vertex.point(), &config) {
            Ok(InSphere::INSIDE) => {
                offending = Some((test_vkey, *test_vertex.point()));
                break;
            }
            Ok(InSphere::BOUNDARY | InSphere::OUTSIDE) => {}
            Err(e) => {
                eprintln!(
                    "[Delaunay debug] robust_insphere error while searching for offending vertex in cell {first_cell_key:?}: {e}",
                );
            }
        }
    }

    if let Some((off_vkey, off_point)) = offending {
        eprintln!(
            "[Delaunay debug]  Offending external vertex: vkey={off_vkey:?}, point={off_point:?}",
        );
    } else {
        eprintln!(
            "[Delaunay debug]  No offending external vertex found for first violating cell (possible degeneracy or removed vertices)"
        );
    }

    // Neighbor information for the first violating cell.
    if let Some(neighbors) = cell.neighbors() {
        for (facet_idx, neighbor_key_opt) in neighbors.iter().enumerate() {
            match neighbor_key_opt {
                Some(neighbor_key) => {
                    if let Some(neighbor_cell) = tds.get_cell(*neighbor_key) {
                        eprintln!(
                            "[Delaunay debug]  facet {facet_idx}: neighbor cell {neighbor_key:?}, uuid={}",
                            neighbor_cell.uuid()
                        );
                    } else {
                        eprintln!(
                            "[Delaunay debug]  facet {facet_idx}: neighbor cell {neighbor_key:?} missing from TDS",
                        );
                    }
                }
                None => {
                    eprintln!(
                        "[Delaunay debug]  facet {facet_idx}: no neighbor (hull facet or unassigned)"
                    );
                }
            }
        }
    } else {
        eprintln!(
            "[Delaunay debug]  First violating cell has no neighbors assigned (neighbors() == None)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::vertex;

    #[test]
    fn delaunay_validator_reports_no_violations_for_simple_tetrahedron() {
        println!("Testing Delaunay validator and debug helper on a simple 3D tetrahedron");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let dt =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let tds = &dt.as_triangulation().tds;

        // Basic Delaunay helpers should report no violations.
        assert!(
            is_delaunay_property_only(tds).is_ok(),
            "Simple tetrahedron should satisfy the Delaunay property"
        );
        let violations = find_delaunay_violations(tds, None).unwrap();
        assert!(
            violations.is_empty(),
            "find_delaunay_violations should report no violating cells for a tetrahedron"
        );

        // Smoke test for the debug helper: it should not panic and should print a
        // summary indicating that no violations were found.
        #[cfg(any(test, debug_assertions))]
        debug_print_first_delaunay_violation(tds, None);
    }
}

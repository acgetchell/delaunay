//! Delaunay empty-circumsphere property validation utilities.

#![forbid(unsafe_code)]

use crate::core::cell::CellValidationError;
use crate::core::collections::ViolationBuffer;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::{CellKey, Tds, TdsValidationError, VertexKey};
use crate::geometry::point::Point;
use crate::geometry::predicates::InSphere;
use crate::geometry::robust_predicates::robust_insphere;
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};
use smallvec::SmallVec;
use std::ops::{AddAssign, SubAssign};
use thiserror::Error;

/// Errors that can occur during Delaunay property validation.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::triangulation_data_structure::CellKey;
/// use delaunay::core::util::DelaunayValidationError;
/// use slotmap::KeyData;
///
/// let cell_key = CellKey::from(KeyData::from_ffi(1));
/// let err = DelaunayValidationError::DelaunayViolation { cell_key };
/// assert!(matches!(err, DelaunayValidationError::DelaunayViolation { .. }));
/// ```
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
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum,
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

/// Internal helper: validate the Delaunay empty-circumsphere property only.
///
/// This performs the expensive geometric check but intentionally does **not** run
/// `tds.is_valid()` up front. Callers that want cumulative validation should run
/// lower-layer checks separately.
pub(crate) fn is_delaunay_property_only<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<(), DelaunayValidationError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum,
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
/// This is a variant of the crate-private Delaunay-property-only check that returns ALL violating
/// cells instead of stopping at the first violation. This is useful for iterative cavity
/// refinement and debugging.
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
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum,
    U: DataType,
    V: DataType,
{
    let mut violating_cells = ViolationBuffer::new();
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

    // Use robust predicates configuration for reliability
    let config = crate::geometry::robust_predicates::config_presets::general_triangulation::<T>();

    #[cfg(any(test, debug_assertions))]
    if let Some(keys) = cells_to_check {
        tracing::debug!(
            "[Delaunay debug] find_delaunay_violations: checking {} requested cells",
            keys.len()
        );
    } else {
        tracing::debug!("[Delaunay debug] find_delaunay_violations: checking all cells");
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
    tracing::debug!(
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
/// same robust predicates as [`find_delaunay_violations`] (and the crate-private Delaunay-property-only check)
/// and prints:
/// - A triangulation summary (vertex and cell counts)
/// - All vertices (keys, UUIDs, coordinates)
/// - All violating cells' vertices
/// - For the first violating cell:
///   - At least one offending external vertex (if found)
///   - Neighbor information for each facet
///
/// # Examples
///
/// ```rust
/// use delaunay::core::triangulation_data_structure::Tds;
/// use delaunay::core::util::debug_print_first_delaunay_violation;
///
/// let tds: Tds<f64, (), (), 3> = Tds::empty();
/// debug_print_first_delaunay_violation(&tds, None);
/// ```
#[cfg(any(test, debug_assertions))]
#[expect(
    clippy::too_many_lines,
    reason = "Debug-only helper with intentionally verbose logging"
)]
pub fn debug_print_first_delaunay_violation<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cells_subset: Option<&[CellKey]>,
) where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + std::iter::Sum,
    U: DataType,
    V: DataType,
{
    use crate::geometry::robust_predicates::config_presets;

    // First, find violating cells using the standard helper.
    let violations = match find_delaunay_violations(tds, cells_subset) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                "[Delaunay debug] debug_print_first_delaunay_violation: error while finding violations: {e}"
            );
            return;
        }
    };
    tracing::debug!(
        "[Delaunay debug] Triangulation summary: {} vertices, {} cells",
        tds.number_of_vertices(),
        tds.number_of_cells()
    );

    // Dump all input vertices once for reproducibility.
    for (vkey, vertex) in tds.vertices() {
        tracing::debug!(
            "[Delaunay debug] Vertex {:?}: uuid={}, point={:?}",
            vkey,
            vertex.uuid(),
            vertex.point()
        );
    }

    if violations.is_empty() {
        tracing::debug!(
            "[Delaunay debug] No Delaunay violations detected for requested cell subset"
        );
        return;
    }
    tracing::debug!(
        "[Delaunay debug] Delaunay violations detected in {} cell(s):",
        violations.len()
    );

    // Reusable buffer for cell vertex points.
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

    // Dump each violating cell with its vertices.
    for cell_key in &violations {
        match tds.get_cell(*cell_key) {
            Some(cell) => {
                tracing::debug!(
                    "[Delaunay debug]  Cell {:?}: uuid={}, vertices:",
                    cell_key,
                    cell.uuid()
                );
                for &vkey in cell.vertices() {
                    match tds.get_vertex_by_key(vkey) {
                        Some(v) => {
                            tracing::debug!(
                                "[Delaunay debug]    vkey={:?}, uuid={}, point={:?}",
                                vkey,
                                v.uuid(),
                                v.point()
                            );
                        }
                        None => {
                            tracing::debug!("[Delaunay debug]    vkey={vkey:?} (missing in TDS)");
                        }
                    }
                }
            }
            None => {
                tracing::debug!(
                    "[Delaunay debug]  Cell {cell_key:?} not found in TDS during violation dump"
                );
            }
        }
    }

    // Focus on the first violating cell to identify at least one offending
    // external vertex and neighbor information.
    let first_cell_key = violations[0];
    let Some(cell) = tds.get_cell(first_cell_key) else {
        tracing::debug!(
            "[Delaunay debug] First violating cell {first_cell_key:?} not found in TDS"
        );
        return;
    };

    let cell_vertex_keys: SmallVec<[VertexKey; 8]> = cell.vertices().iter().copied().collect();

    cell_vertex_points.clear();
    let mut missing_vertex_keys: SmallVec<[VertexKey; 8]> = SmallVec::new();
    for &vkey in &cell_vertex_keys {
        match tds.get_vertex_by_key(vkey) {
            Some(v) => {
                cell_vertex_points.push(*v.point());
            }
            None => {
                missing_vertex_keys.push(vkey);
            }
        }
    }

    if cell_vertex_points.len() != cell_vertex_keys.len() {
        tracing::warn!(
            "[Delaunay debug] First violating cell {first_cell_key:?} references missing vertices; skipping robust_insphere search. Missing vertex keys: {missing_vertex_keys:?}",
        );
        return;
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
                tracing::warn!(
                    "[Delaunay debug] robust_insphere error while searching for offending vertex in cell {first_cell_key:?}: {e}",
                );
            }
        }
    }

    if let Some((off_vkey, off_point)) = offending {
        tracing::debug!(
            "[Delaunay debug]  Offending external vertex: vkey={off_vkey:?}, point={off_point:?}",
        );
    } else {
        tracing::debug!(
            "[Delaunay debug]  No offending external vertex found for first violating cell (possible degeneracy or removed vertices)"
        );
    }

    // Neighbor information for the first violating cell.
    if let Some(neighbors) = cell.neighbors() {
        for (facet_idx, neighbor_key_opt) in neighbors.iter().enumerate() {
            match neighbor_key_opt {
                Some(neighbor_key) => {
                    if let Some(neighbor_cell) = tds.get_cell(*neighbor_key) {
                        tracing::debug!(
                            "[Delaunay debug]  facet {facet_idx}: neighbor cell {neighbor_key:?}, uuid={}",
                            neighbor_cell.uuid()
                        );
                    } else {
                        tracing::debug!(
                            "[Delaunay debug]  facet {facet_idx}: neighbor cell {neighbor_key:?} missing from TDS",
                        );
                    }
                }
                None => {
                    tracing::debug!(
                        "[Delaunay debug]  facet {facet_idx}: no neighbor (hull facet or unassigned)"
                    );
                }
            }
        }
    } else {
        tracing::debug!(
            "[Delaunay debug]  First violating cell has no neighbors assigned (neighbors() == None)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cell::Cell;
    use crate::core::triangulation::Triangulation;
    use crate::core::util::make_uuid;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::{Coordinate, CoordinateConversionError};

    use crate::vertex;

    #[test]
    fn delaunay_validator_reports_no_violations_for_simple_tetrahedron() {
        init_tracing();
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

    fn init_tracing() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer()
                .try_init();
        });
    }

    fn build_non_delaunay_quad_2d() -> (Tds<f64, (), (), 2>, CellKey, CellKey) {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let d = tds.insert_vertex_with_mapping(vertex!([0.8, 0.8])).unwrap();

        let cell_1 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();
        let cell_2 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, c, d], None).unwrap())
            .unwrap();

        tds.assign_incident_cells().unwrap();

        (tds, cell_1, cell_2)
    }

    #[test]
    fn delaunay_validator_reports_violation_for_non_delaunay_quad_2d() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let d = tds.insert_vertex_with_mapping(vertex!([0.8, 0.8])).unwrap();

        let cell_1 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();
        let cell_2 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, c, d], None).unwrap())
            .unwrap();
        tds.assign_incident_cells().unwrap();

        match is_delaunay_property_only(&tds) {
            Err(DelaunayValidationError::DelaunayViolation { cell_key }) => {
                assert!(cell_key == cell_1 || cell_key == cell_2);
            }
            other => panic!("Expected DelaunayViolation, got {other:?}"),
        }
    }

    #[test]
    fn find_delaunay_violations_subset_skips_missing_cells() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let d = tds.insert_vertex_with_mapping(vertex!([0.8, 0.8])).unwrap();

        let cell_1 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();
        let _cell_2 = tds
            .insert_cell_with_mapping(Cell::new(vec![a, c, d], None).unwrap())
            .unwrap();
        tds.assign_incident_cells().unwrap();

        let violations =
            find_delaunay_violations(&tds, Some(&[cell_1, CellKey::default()])).unwrap();
        assert_eq!(violations.len(), 1);
        assert!(violations.contains(&cell_1));
    }

    #[test]
    fn delaunay_validation_handles_empty_tds() {
        init_tracing();
        let tds: Tds<f64, (), (), 2> = Tds::empty();

        assert!(is_delaunay_property_only(&tds).is_ok());
        let violations = find_delaunay_violations(&tds, None).unwrap();
        assert!(violations.is_empty());
    }

    #[test]
    fn find_delaunay_violations_subset_filters_non_violating_cell() {
        init_tracing();
        let (tds, cell_1, cell_2) = build_non_delaunay_quad_2d();

        let violations = find_delaunay_violations(&tds, None).unwrap();
        assert_eq!(violations.len(), 1);
        let violating_cell = violations[0];
        let non_violating_cell = if violating_cell == cell_1 {
            cell_2
        } else {
            cell_1
        };

        let subset = find_delaunay_violations(&tds, Some(&[non_violating_cell])).unwrap();
        assert!(subset.is_empty());
    }

    #[test]
    fn delaunay_property_only_handles_non_finite_vertex_without_error() {
        init_tracing();
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let a = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let b = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let c = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        tds.insert_cell_with_mapping(Cell::new(vec![a, b, c], None).unwrap())
            .unwrap();

        let invalid_uuid = make_uuid();
        tds.insert_vertex_with_mapping(Vertex::new_with_uuid(
            Point::new([f64::NAN, 0.0]),
            invalid_uuid,
            None,
        ))
        .unwrap();

        tds.remove_vertex(&Vertex::new_with_uuid(
            Point::new([f64::NAN, 0.0]),
            invalid_uuid,
            None,
        ))
        .unwrap();

        assert!(
            is_delaunay_property_only(&tds).is_ok(),
            "delaunay_property_only_handles_non_finite_vertex_without_error should skip non-finite vertices before validation"
        );
    }

    #[test]
    fn numeric_predicate_error_display_includes_context() {
        let cell_key = CellKey::from(slotmap::KeyData::from_ffi(1));
        let vertex_key = VertexKey::from(slotmap::KeyData::from_ffi(2));
        let source = CoordinateConversionError::NonFiniteValue {
            coordinate_index: 0,
            coordinate_value: "NaN".to_string(),
        };
        let err = DelaunayValidationError::NumericPredicateError {
            cell_key,
            vertex_key,
            source,
        };
        let message = err.to_string();

        assert!(message.contains("Numeric predicate failure"));
        assert!(message.contains("cell"));
        assert!(message.contains("vertex"));
        assert!(message.contains("Non-finite value"));
    }

    #[test]
    fn debug_print_first_delaunay_violation_handles_violations() {
        init_tracing();
        let (tds, _, _) = build_non_delaunay_quad_2d();

        #[cfg(any(test, debug_assertions))]
        debug_print_first_delaunay_violation(&tds, None);
    }

    #[test]
    fn delaunay_property_only_reports_triangulation_state_on_missing_vertex() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();
        let original_vertices = {
            let cell = tds.get_cell(cell_key).unwrap();
            cell.vertices().to_vec()
        };
        let invalid_vkey = VertexKey::from(slotmap::KeyData::from_ffi(u64::MAX));

        {
            let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
            cell.clear_vertex_keys();
            for (idx, &vkey) in original_vertices.iter().enumerate() {
                if idx == 0 {
                    cell.push_vertex_key(invalid_vkey);
                } else {
                    cell.push_vertex_key(vkey);
                }
            }
        }

        let err = is_delaunay_property_only(&tds).unwrap_err();
        assert!(matches!(
            err,
            DelaunayValidationError::TriangulationState { .. }
        ));
    }

    #[test]
    fn is_delaunay_property_only_reports_invalid_cell() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();
        let cell = tds.get_cell_by_key_mut(cell_key).unwrap();
        cell.neighbors = Some(vec![None, None].into()); // wrong length (expected 3)

        let err = is_delaunay_property_only(&tds).unwrap_err();
        assert!(matches!(err, DelaunayValidationError::InvalidCell { .. }));
    }
}

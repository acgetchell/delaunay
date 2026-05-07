//! Delaunay empty-circumsphere property validation utilities.

#![forbid(unsafe_code)]

use crate::core::cell::CellValidationError;
use crate::core::collections::ViolationBuffer;
#[cfg(any(test, feature = "diagnostics"))]
use crate::core::collections::{CellVertexBuffer, NeighborBuffer};
use crate::core::tds::{CellKey, Tds, TdsError, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::geometry::point::Point;
use crate::geometry::predicates::InSphere;
use crate::geometry::robust_predicates::robust_insphere;
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};
use smallvec::SmallVec;
use thiserror::Error;

/// Errors that can occur during Delaunay property validation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::CellKey;
/// use delaunay::prelude::triangulation::repair::DelaunayValidationError;
/// use slotmap::KeyData;
///
/// let cell_key = CellKey::from(KeyData::from_ffi(1));
/// let err = DelaunayValidationError::DelaunayViolation { cell_key };
/// assert!(matches!(err, DelaunayValidationError::DelaunayViolation { .. }));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
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
        source: TdsError,
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

/// Structured summary of Delaunay empty-circumsphere violations.
///
/// This diagnostic report is available with the `diagnostics` feature and is
/// intended for bug reports, regression tests, and local investigation. It
/// records stable TDS keys rather than copying all coordinates; callers can
/// look up coordinates, UUIDs, and cell data in the original [`Tds`].
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::diagnostics::delaunay_violation_report;
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let report = delaunay_violation_report(dt.tds(), None).unwrap();
/// assert!(report.is_valid());
/// ```
#[cfg(any(test, feature = "diagnostics"))]
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct DelaunayViolationReport {
    /// Number of vertices in the TDS when the report was generated.
    pub number_of_vertices: usize,
    /// Number of cells in the TDS when the report was generated.
    pub number_of_cells: usize,
    /// Number of requested cells considered by the report.
    ///
    /// When `cells_to_check` is `None`, this is the TDS cell count. When a
    /// subset is provided, this is the subset length; missing cell keys are
    /// still counted as requested work and are skipped by the violation scan.
    pub checked_cells: usize,
    /// Cells that failed the empty-circumsphere property.
    pub violating_cells: ViolationBuffer,
    /// Details for the first violating cell, if one is still present in the TDS.
    pub first_violation: Option<DelaunayViolationDetail>,
}

#[cfg(any(test, feature = "diagnostics"))]
impl DelaunayViolationReport {
    /// Returns `true` when no Delaunay violations were found.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::diagnostics::DelaunayViolationReport;
    ///
    /// let report = DelaunayViolationReport {
    ///     number_of_vertices: 0,
    ///     number_of_cells: 0,
    ///     checked_cells: 0,
    ///     violating_cells: Default::default(),
    ///     first_violation: None,
    /// };
    /// assert!(report.is_valid());
    /// ```
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.violating_cells.is_empty()
    }
}

/// Details for the first cell in a [`DelaunayViolationReport`].
///
/// The detail record keeps the report compact and key-oriented. Use
/// [`cell_key`](Self::cell_key), [`cell_vertices`](Self::cell_vertices), and
/// [`offending_vertex`](Self::offending_vertex) to recover full vertex or cell
/// records from the source [`Tds`].
#[cfg(any(test, feature = "diagnostics"))]
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct DelaunayViolationDetail {
    /// Violating cell key.
    pub cell_key: CellKey,
    /// Vertex keys stored by the violating cell at report time.
    pub cell_vertices: CellVertexBuffer,
    /// First external vertex found inside the cell circumsphere, if one could
    /// be identified.
    pub offending_vertex: Option<VertexKey>,
    /// Neighbor slots of the violating cell, preserving facet-index order.
    pub neighbor_cells: NeighborBuffer<Option<CellKey>>,
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
) -> Result<Option<CellKey>, DelaunayValidationError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    Ok(first_delaunay_violation_witness(tds, cell_key, cell_vertex_points)?.map(|_| cell_key))
}

/// Finds one external vertex that witnesses a cell's Delaunay violation.
fn first_delaunay_violation_witness<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
    cell_vertex_points: &mut SmallVec<[Point<T, D>; 8]>,
) -> Result<Option<VertexKey>, DelaunayValidationError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let Some(cell) = tds.cell(cell_key) else {
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
        let Some(v) = tds.vertex(vkey) else {
            return Err(DelaunayValidationError::TriangulationState {
                source: TdsError::InconsistentDataStructure {
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
        match robust_insphere(cell_vertex_points, test_vertex.point()) {
            Ok(InSphere::INSIDE) => {
                // For D≥4: check for the both_positive_artifact before reporting a violation.
                //
                // "Vertex j is always opposite neighbor j" means cell.neighbors()[k] is
                // the neighbor opposite cell.vertices()[k].  When robust_insphere gives
                // in_a > 0 AND in_b > 0 simultaneously for the two cells sharing a facet,
                // it is physically impossible by cofactor antisymmetry and indicates a
                // near-degenerate numerical artefact in the (D+2)×(D+2) insphere matrix.
                // The repair predicate already suppresses flips for such cases
                // (both_positive_artifact in delaunay_violation_k2_for_facet); validation
                // must be consistent and skip the same cases.
                //
                // To identify the neighbour B for which test_vkey is the apex:
                // - test_vkey is already confirmed not in cell A's vertex set.
                // - Since B shares a D-facet with A (D vertices in common), B has exactly
                //   one vertex not in A: its apex.  So test_vkey is B's apex ⇔
                //   test_vkey ∈ B.vertices().
                // - The corresponding apex of A w.r.t. B is A.vertices()[k] where
                //   A.neighbors()[k] = B  ("vertex j opposite neighbor j").
                if D >= 4 {
                    let is_artifact = 'artifact: {
                        let Some(a_neighbors) = cell.neighbors() else {
                            break 'artifact false;
                        };
                        let mut b_points: SmallVec<[Point<T, D>; 8]> =
                            SmallVec::with_capacity(D + 1);
                        for (k, neighbor_opt) in a_neighbors.iter().enumerate() {
                            let Some(b_key) = neighbor_opt else {
                                continue;
                            };
                            let Some(b_cell) = tds.cell(*b_key) else {
                                continue;
                            };
                            // Is test_vkey the apex of B w.r.t. A?
                            if !b_cell.vertices().contains(&test_vkey) {
                                continue;
                            }
                            // Found. A's apex w.r.t. B = cell.vertices()[k].
                            let apex_a_key = cell_vertex_keys[k];
                            let Some(apex_a_v) = tds.vertex(apex_a_key) else {
                                continue;
                            };
                            // Build B's circumsphere points.
                            b_points.clear();
                            let mut valid = true;
                            for &bv in b_cell.vertices() {
                                let Some(v) = tds.vertex(bv) else {
                                    valid = false;
                                    break;
                                };
                                b_points.push(*v.point());
                            }
                            if !valid {
                                continue;
                            }
                            // Symmetric check: is A's apex inside-or-on B's circumsphere?
                            //
                            // We suppress two co-degenerate artifact classes:
                            //  • Both-positive: both inspheres are > 0 simultaneously,
                            //    physically impossible by cofactor antisymmetry.
                            //  • Co-spherical: A sees V slightly inside (floating-point > 0)
                            //    but B sees A's apex exactly on the sphere (BOUNDARY == 0).
                            //    This happens with near co-spherical point sets in D≥4 where
                            //    the (D+2)×(D+2) determinant is near zero.  The repair
                            //    predicate would attempt a flip but every flip just cycles
                            //    the sign; suppressing here is consistent.
                            if matches!(
                                robust_insphere(&b_points, apex_a_v.point()),
                                Ok(InSphere::INSIDE | InSphere::BOUNDARY)
                            ) {
                                break 'artifact true;
                            }
                        }
                        false
                    };
                    if is_artifact {
                        continue;
                    }
                }
                // Genuine violation
                return Ok(Some(test_vkey));
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
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Reusable buffer to minimize allocations
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

    // Check each cell using the shared validation helper
    for cell_key in tds.cell_keys() {
        if let Some(violating_cell) =
            validate_cell_delaunay(tds, cell_key, &mut cell_vertex_points)?
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
/// use delaunay::prelude::query::*;
/// use delaunay::prelude::triangulation::repair::find_delaunay_violations;
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
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut violating_cells = ViolationBuffer::new();
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);

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
            validate_cell_delaunay(tds, cell_key, &mut cell_vertex_points)?
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

/// Build a structured Delaunay violation report.
///
/// This is the structured counterpart to
/// [`debug_print_first_delaunay_violation`]. It uses the same robust
/// empty-circumsphere scan as [`find_delaunay_violations`] and returns compact,
/// key-based diagnostics that can be attached to bug reports or inspected by
/// tests without relying on tracing output.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure to inspect.
/// * `cells_to_check` - Optional subset of cells to check. Missing cells are
///   skipped by the underlying scan but still counted in
///   [`DelaunayViolationReport::checked_cells`].
///
/// # Errors
///
/// Returns [`DelaunayValidationError`] if the underlying Delaunay scan finds
/// invalid cell structure, missing vertex references, or robust predicate
/// conversion failures.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::diagnostics::delaunay_violation_report;
/// use delaunay::prelude::triangulation::*;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
///
/// let report = delaunay_violation_report(dt.tds(), None).unwrap();
/// assert!(report.violating_cells.is_empty());
/// ```
#[cfg(any(test, feature = "diagnostics"))]
pub fn delaunay_violation_report<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cells_to_check: Option<&[CellKey]>,
) -> Result<DelaunayViolationReport, DelaunayValidationError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let violating_cells = find_delaunay_violations(tds, cells_to_check)?;
    let checked_cells = cells_to_check.map_or_else(|| tds.number_of_cells(), <[_]>::len);
    let first_violation = violating_cells
        .first()
        .and_then(|&cell_key| build_violation_detail(tds, cell_key));

    Ok(DelaunayViolationReport {
        number_of_vertices: tds.number_of_vertices(),
        number_of_cells: tds.number_of_cells(),
        checked_cells,
        violating_cells,
        first_violation,
    })
}

/// Builds the compact detail record for a violating cell that still exists in the TDS.
#[cfg(any(test, feature = "diagnostics"))]
fn build_violation_detail<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
) -> Option<DelaunayViolationDetail>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let cell = tds.cell(cell_key)?;
    let cell_vertices = cell.vertices().iter().copied().collect();
    let neighbor_cells = cell
        .neighbors()
        .map_or_else(NeighborBuffer::new, |neighbors| {
            neighbors.iter().copied().collect()
        });
    let offending_vertex = first_offending_vertex(tds, cell_key);

    Some(DelaunayViolationDetail {
        cell_key,
        cell_vertices,
        offending_vertex,
        neighbor_cells,
    })
}

/// Finds one external vertex that witnesses a cell's Delaunay violation, if available.
#[cfg(any(test, feature = "diagnostics"))]
fn first_offending_vertex<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
) -> Option<VertexKey>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let mut cell_vertex_points: SmallVec<[Point<T, D>; 8]> = SmallVec::with_capacity(D + 1);
    first_delaunay_violation_witness(tds, cell_key, &mut cell_vertex_points)
        .ok()
        .flatten()
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
/// use delaunay::prelude::tds::Tds;
/// use delaunay::prelude::diagnostics::debug_print_first_delaunay_violation;
///
/// let tds: Tds<f64, (), (), 3> = Tds::empty();
/// debug_print_first_delaunay_violation(&tds, None);
/// ```
#[cfg(any(test, feature = "diagnostics"))]
#[expect(
    clippy::too_many_lines,
    reason = "Debug-only helper with intentionally verbose logging"
)]
pub fn debug_print_first_delaunay_violation<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cells_subset: Option<&[CellKey]>,
) where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // First, build the structured report used by downstream diagnostics.
    let report = match delaunay_violation_report(tds, cells_subset) {
        Ok(report) => report,
        Err(e) => {
            tracing::warn!(
                "[Delaunay debug] debug_print_first_delaunay_violation: error while finding violations: {e}"
            );
            return;
        }
    };
    let violations = &report.violating_cells;
    tracing::debug!(
        "[Delaunay debug] Triangulation summary: {} vertices, {} cells, {} checked cells",
        report.number_of_vertices,
        report.number_of_cells,
        report.checked_cells,
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
    for cell_key in violations {
        match tds.cell(*cell_key) {
            Some(cell) => {
                tracing::debug!(
                    "[Delaunay debug]  Cell {:?}: uuid={}, vertices:",
                    cell_key,
                    cell.uuid()
                );
                for &vkey in cell.vertices() {
                    match tds.vertex(vkey) {
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
    let Some(cell) = tds.cell(first_cell_key) else {
        tracing::debug!(
            "[Delaunay debug] First violating cell {first_cell_key:?} not found in TDS"
        );
        return;
    };

    let cell_vertex_keys: SmallVec<[VertexKey; 8]> = cell.vertices().iter().copied().collect();

    cell_vertex_points.clear();
    let mut missing_vertex_keys: SmallVec<[VertexKey; 8]> = SmallVec::new();
    for &vkey in &cell_vertex_keys {
        match tds.vertex(vkey) {
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

    let mut offending: Option<(VertexKey, Point<T, D>)> = None;

    for (test_vkey, test_vertex) in tds.vertices() {
        if cell_vertex_keys.contains(&test_vkey) {
            continue;
        }

        match robust_insphere(&cell_vertex_points, test_vertex.point()) {
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
                    if let Some(neighbor_cell) = tds.cell(*neighbor_key) {
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
    use crate::core::algorithms::incremental_insertion::repair_neighbor_pointers;
    use crate::core::cell::Cell;
    use crate::core::triangulation::Triangulation;
    use crate::core::util::make_uuid;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::{Coordinate, CoordinateConversionError};
    use crate::triangulation::delaunay::DelaunayTriangulation;

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

        let dt = DelaunayTriangulation::new(&vertices).unwrap();
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
        #[cfg(any(test, feature = "diagnostics"))]
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
        repair_neighbor_pointers(&mut tds).unwrap();

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
        let invalid_vk = tds
            .insert_vertex_with_mapping(Vertex::new_with_uuid(
                Point::new([f64::NAN, 0.0]),
                invalid_uuid,
                None,
            ))
            .unwrap();

        tds.remove_vertex(invalid_vk).unwrap();

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

        #[cfg(any(test, feature = "diagnostics"))]
        debug_print_first_delaunay_violation(&tds, None);
    }

    #[test]
    fn delaunay_violation_report_summarizes_valid_tds() {
        init_tracing();
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::new(&vertices).unwrap();

        let report = delaunay_violation_report(dt.tds(), None).unwrap();

        assert!(report.is_valid());
        assert_eq!(report.number_of_vertices, 4);
        assert_eq!(report.number_of_cells, 1);
        assert_eq!(report.checked_cells, 1);
        assert!(report.first_violation.is_none());
    }

    #[test]
    fn delaunay_violation_report_includes_first_violation_detail() {
        init_tracing();
        let (tds, cell_1, cell_2) = build_non_delaunay_quad_2d();

        let report = delaunay_violation_report(&tds, None).unwrap();

        assert!(!report.is_valid());
        assert_eq!(report.violating_cells.len(), 1);
        let detail = report
            .first_violation
            .as_ref()
            .expect("violating report should include first violation details");
        assert!(detail.cell_key == cell_1 || detail.cell_key == cell_2);
        assert_eq!(detail.cell_vertices.len(), 3);
        assert_eq!(detail.neighbor_cells.len(), 3);
        assert!(detail.offending_vertex.is_some());
    }

    #[test]
    fn delaunay_violation_report_tracks_requested_subset_size() {
        init_tracing();
        let (tds, cell_1, _) = build_non_delaunay_quad_2d();

        let report = delaunay_violation_report(&tds, Some(&[cell_1, CellKey::default()])).unwrap();

        assert_eq!(report.checked_cells, 2);
        assert_eq!(report.violating_cells.as_slice(), &[cell_1]);
        assert_eq!(
            report
                .first_violation
                .as_ref()
                .map(|detail| detail.cell_key),
            Some(cell_1)
        );
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
            let cell = tds.cell(cell_key).unwrap();
            cell.vertices().to_vec()
        };
        let invalid_vkey = VertexKey::from(slotmap::KeyData::from_ffi(u64::MAX));

        {
            let cell = tds.cell_mut(cell_key).unwrap();
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
        let cell = tds.cell_mut(cell_key).unwrap();
        cell.neighbors = Some(vec![None, None].into()); // wrong length (expected 3)

        let err = is_delaunay_property_only(&tds).unwrap_err();
        assert!(matches!(err, DelaunayValidationError::InvalidCell { .. }));
    }
}

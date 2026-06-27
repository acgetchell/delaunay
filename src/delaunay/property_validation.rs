//! Delaunay empty-circumsphere property scans over bare TDS storage.
//!
//! This module is the reusable Level 5 property engine: it answers whether a
//! [`Tds`](crate::tds::Tds) violates the Delaunay empty-circumsphere condition and returns
//! repair-oriented keys for offending simplices, vertices, and neighbors. It
//! does not own wrapper-level validation policy, cumulative roll-up, or
//! construction proofs; those live in `validation`.

#![forbid(unsafe_code)]

use crate::core::collections::{NeighborBuffer, SimplexVertexKeyBuffer, ViolationBuffer};
use crate::core::simplex::{NeighborSlot, SimplexValidationError};
use crate::core::tds::{SimplexKey, Tds, TdsError, VertexKey};
use crate::geometry::point::Point;
use crate::geometry::predicates::InSphere;
use crate::geometry::robust_predicates::robust_insphere;
use crate::geometry::traits::coordinate::CoordinateConversionError;
use smallvec::SmallVec;
use thiserror::Error;

/// Errors that can occur during Delaunay property validation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::SimplexKey;
/// use delaunay::prelude::validation::DelaunayValidationError;
/// use slotmap::KeyData;
///
/// let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
/// let err = DelaunayValidationError::DelaunayViolation {
///     simplex_key,
///     simplex_vertices: Default::default(),
///     offending_vertex: None,
///     neighbor_simplices: Default::default(),
/// };
/// std::assert_matches!(err, DelaunayValidationError::DelaunayViolation { .. });
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayValidationError {
    /// A simplex violates the Delaunay property (has an external vertex inside its circumsphere).
    #[error(
        "Simplex {simplex_key:?} violates Delaunay property; offending vertex: {offending_vertex:?}"
    )]
    DelaunayViolation {
        /// The key of the simplex that violates the Delaunay property.
        simplex_key: SimplexKey,
        /// Vertex keys stored by the violating simplex at report time.
        ///
        /// Boxed to keep the error enum small while preserving typed repair
        /// context.
        simplex_vertices: Box<SimplexVertexKeyBuffer>,
        /// First external vertex found inside the simplex circumsphere, if one could
        /// be identified.
        offending_vertex: Option<VertexKey>,
        /// Neighbor slots of the violating simplex, preserving facet-index order.
        ///
        /// Boxed to keep the error enum small while preserving typed repair
        /// context.
        neighbor_simplices: Box<NeighborBuffer<NeighborSlot>>,
    },
    /// TDS data structure corruption or other structural issues detected during validation.
    #[error("TDS corruption: {source}")]
    TriangulationState {
        /// The underlying TDS validation error (TDS-level invariants).
        #[source]
        source: TdsError,
    },
    /// Invalid simplex structure detected during validation.
    #[error("Invalid simplex {simplex_key:?}: {source}")]
    InvalidSimplex {
        /// The key of the invalid simplex.
        simplex_key: SimplexKey,
        /// The underlying simplex error.
        #[source]
        source: SimplexValidationError,
    },
    /// Numeric predicate failure during Delaunay validation.
    #[error(
        "Numeric predicate failure while validating Delaunay property for simplex {simplex_key:?} against vertex {vertex_key:?}: {source}"
    )]
    NumericPredicateError {
        /// The key of the simplex whose circumsphere was being tested.
        simplex_key: SimplexKey,
        /// The key of the vertex being classified relative to the circumsphere.
        vertex_key: VertexKey,
        /// Underlying robust predicate error (e.g., conversion failure).
        #[source]
        source: CoordinateConversionError,
    },
}

/// Structured summary of Delaunay empty-circumsphere violations.
///
/// This diagnostic report is intended for repair planning, bug reports,
/// regression tests, and local investigation. It records stable TDS keys rather
/// than copying all coordinates; callers can look up coordinates, UUIDs, and
/// simplex data in the original [`Tds`].
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::validation::delaunay_violation_report;
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Validation(#[from] delaunay::DelaunayValidationError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let report = delaunay_violation_report(dt.tds(), None)?;
/// assert!(report.is_valid());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct DelaunayViolationReport {
    /// Number of vertices in the TDS when the report was generated.
    pub number_of_vertices: usize,
    /// Number of simplices in the TDS when the report was generated.
    pub number_of_simplices: usize,
    /// Number of requested simplices considered by the report.
    ///
    /// When `simplices_to_check` is `None`, this is the TDS simplex count. When a
    /// subset is provided, this is the subset length; missing simplex keys are
    /// still counted as requested work and are skipped by the violation scan.
    pub checked_simplices: usize,
    /// Simplices that failed the empty-circumsphere property.
    pub violating_simplices: ViolationBuffer,
    /// Details for each violating simplex that was still present in the TDS.
    pub violation_details: Vec<DelaunayViolationDetail>,
}

impl DelaunayViolationReport {
    /// Returns `true` when no Delaunay violations were found.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::validation::DelaunayViolationReport;
    ///
    /// let report = DelaunayViolationReport {
    ///     number_of_vertices: 0,
    ///     number_of_simplices: 0,
    ///     checked_simplices: 0,
    ///     violating_simplices: Default::default(),
    ///     violation_details: Vec::new(),
    /// };
    /// assert!(report.is_valid());
    /// ```
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.violating_simplices.is_empty() && self.violation_details.is_empty()
    }

    /// Returns the first violating-simplex detail, if one is present.
    ///
    /// This borrowed view is derived from
    /// [`violation_details`](Self::violation_details), avoiding a second owned
    /// copy that could diverge from the report's canonical detail list.
    #[must_use]
    pub fn first_violation(&self) -> Option<&DelaunayViolationDetail> {
        self.violation_details.first()
    }
}

/// Details for the first simplex in a [`DelaunayViolationReport`].
///
/// The detail record keeps the report compact and key-oriented. Use
/// [`simplex_key`](Self::simplex_key), [`simplex_vertices`](Self::simplex_vertices), and
/// [`offending_vertex`](Self::offending_vertex) to recover full vertex or simplex
/// records from the source [`Tds`].
///
/// [`neighbor_simplices`](Self::neighbor_simplices) preserves the violating simplex's raw
/// [`NeighborSlot`] state for each facet so diagnostics can distinguish
/// [`Boundary`](NeighborSlot::Boundary) hull facets,
/// [`Unassigned`](NeighborSlot::Unassigned) missing wiring, and
/// [`Neighbor`](NeighborSlot::Neighbor) simplex links.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct DelaunayViolationDetail {
    /// Violating simplex key.
    pub simplex_key: SimplexKey,
    /// Vertex keys stored by the violating simplex at report time.
    pub simplex_vertices: SimplexVertexKeyBuffer,
    /// First external vertex found inside the simplex circumsphere, if one could
    /// be identified.
    pub offending_vertex: Option<VertexKey>,
    /// Neighbor slots of the violating simplex, preserving facet-index order.
    pub neighbor_simplices: NeighborBuffer<NeighborSlot>,
}

// =============================================================================
// DELAUNAY PROPERTY VALIDATION
// =============================================================================

/// Internal helper: Check if a single simplex violates the Delaunay property.
///
/// Returns `Ok(None)` if the simplex satisfies the Delaunay property,
/// `Ok(Some(simplex_key))` if it violates (has an external vertex inside its circumsphere),
/// or `Err(...)` if validation fails due to structural or numeric issues.
fn validate_simplex_delaunay<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    simplex_vertex_points: &mut SmallVec<[Point<D>; 8]>,
) -> Result<Option<SimplexKey>, DelaunayValidationError> {
    Ok(
        first_delaunay_violation_witness(tds, simplex_key, simplex_vertex_points)?
            .map(|_| simplex_key),
    )
}

/// Finds one external vertex that witnesses a simplex's Delaunay violation.
fn first_delaunay_violation_witness<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    simplex_vertex_points: &mut SmallVec<[Point<D>; 8]>,
) -> Result<Option<VertexKey>, DelaunayValidationError> {
    let Some(simplex) = tds.simplex(simplex_key) else {
        // Simplex doesn't exist (possibly removed), skip validation
        return Ok(None);
    };

    // Validate simplex structure first
    simplex
        .is_valid()
        .map_err(|source| DelaunayValidationError::InvalidSimplex {
            simplex_key,
            source,
        })?;

    // Get the simplex's vertex set for exclusion
    let simplex_vertex_keys: SmallVec<[VertexKey; 8]> =
        simplex.vertices().iter().copied().collect();

    // Build the simplex's circumsphere
    simplex_vertex_points.clear();
    for &vkey in &simplex_vertex_keys {
        let Some(v) = tds.vertex(vkey) else {
            return Err(DelaunayValidationError::TriangulationState {
                source: TdsError::VertexNotFound {
                    vertex_key: vkey,
                    context: format!("Delaunay property validation for simplex {simplex_key:?}"),
                },
            });
        };
        simplex_vertex_points.push(*v.point());
    }

    // Check if any OTHER vertex is inside this simplex's circumsphere
    for (test_vkey, test_vertex) in tds.vertices() {
        // Skip if this vertex is part of the simplex
        if simplex_vertex_keys.contains(&test_vkey) {
            continue;
        }

        // Test if this vertex is inside the simplex's circumsphere using ROBUST predicates
        match robust_insphere(simplex_vertex_points, test_vertex.point()) {
            Ok(InSphere::INSIDE) => {
                // For D≥4: check for the both_positive_artifact before reporting a violation.
                //
                // "Vertex j is always opposite neighbor j" means simplex.neighbors()[k] is
                // the neighbor opposite simplex.vertices()[k].  When robust_insphere gives
                // in_a > 0 AND in_b > 0 simultaneously for the two simplices sharing a facet,
                // it is physically impossible by cofactor antisymmetry and indicates a
                // near-degenerate numerical artefact in the (D+2)×(D+2) insphere matrix.
                // The repair predicate already suppresses flips for such cases
                // (both_positive_artifact in delaunay_violation_k2_for_facet); validation
                // must be consistent and skip the same cases.
                //
                // To identify the neighbour B for which test_vkey is the apex:
                // - test_vkey is already confirmed not in simplex A's vertex set.
                // - Since B shares a D-facet with A (D vertices in common), B has exactly
                //   one vertex not in A: its apex.  So test_vkey is B's apex ⇔
                //   test_vkey ∈ B.vertices().
                // - The corresponding apex of A w.r.t. B is A.vertices()[k] where
                //   A.neighbors()[k] = B  ("vertex j opposite neighbor j").
                if D >= 4 {
                    let is_artifact = 'artifact: {
                        let Some(a_neighbors) = simplex.neighbor_keys() else {
                            break 'artifact false;
                        };
                        let mut b_points: SmallVec<[Point<D>; 8]> = SmallVec::with_capacity(D + 1);
                        for (k, neighbor_opt) in a_neighbors.enumerate() {
                            let Some(b_key) = neighbor_opt else {
                                continue;
                            };
                            let Some(b_simplex) = tds.simplex(b_key) else {
                                continue;
                            };
                            // Is test_vkey the apex of B w.r.t. A?
                            if !b_simplex.vertices().contains(&test_vkey) {
                                continue;
                            }
                            // Found. A's apex w.r.t. B = simplex.vertices()[k].
                            let apex_a_key = simplex_vertex_keys[k];
                            let Some(apex_a_v) = tds.vertex(apex_a_key) else {
                                continue;
                            };
                            // Build B's circumsphere points.
                            b_points.clear();
                            let mut valid = true;
                            for &bv in b_simplex.vertices() {
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
                    simplex_key,
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
pub fn is_delaunay_property_only<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Result<(), DelaunayValidationError> {
    // Reusable buffer to minimize allocations
    let mut simplex_vertex_points: SmallVec<[Point<D>; 8]> = SmallVec::with_capacity(D + 1);

    // Check each simplex using the shared validation helper
    for simplex_key in tds.simplex_keys() {
        if let Some(violating_simplex) =
            validate_simplex_delaunay(tds, simplex_key, &mut simplex_vertex_points)?
        {
            let detail = build_violation_detail(tds, violating_simplex).unwrap_or_else(|| {
                DelaunayViolationDetail {
                    simplex_key: violating_simplex,
                    simplex_vertices: SimplexVertexKeyBuffer::new(),
                    offending_vertex: None,
                    neighbor_simplices: NeighborBuffer::new(),
                }
            });
            return Err(detail.into());
        }
    }

    Ok(())
}

/// Find simplices that violate the Delaunay property.
///
/// This is a variant of the crate-private Delaunay-property-only check that returns ALL violating
/// simplices instead of stopping at the first violation. This is useful for iterative cavity
/// refinement and debugging.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure
/// * `simplices_to_check` - Optional subset of simplices to check. If `None`, checks all simplices.
///   Missing simplices (e.g., already removed during refinement) are silently skipped.
///
/// # Returns
///
/// A vector of `SimplexKey`s for simplices that violate the Delaunay property.
///
/// # Errors
///
/// Returns [`DelaunayValidationError`] if:
/// - A simplex references a non-existent vertex (TDS corruption)
/// - A simplex has invalid structure (simplex-level corruption)
/// - Robust geometric predicates fail (numerical issues)
///
/// Note: Missing simplices in `simplices_to_check` are silently skipped and do not cause errors,
/// as they may have been legitimately removed during iterative refinement.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::prelude::validation::find_delaunay_violations;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Validation(#[from] delaunay::DelaunayValidationError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
///
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let tds = dt.tds();
///
/// // Find all violating simplices (should be empty for valid Delaunay triangulation)
/// let violations = find_delaunay_violations(tds, None)?;
/// assert!(violations.is_empty());
/// # Ok(())
/// # }
/// ```
pub fn find_delaunay_violations<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices_to_check: Option<&[SimplexKey]>,
) -> Result<ViolationBuffer, DelaunayValidationError> {
    let mut violating_simplices = ViolationBuffer::new();
    let mut simplex_vertex_points: SmallVec<[Point<D>; 8]> = SmallVec::with_capacity(D + 1);

    #[cfg(any(test, debug_assertions))]
    if let Some(keys) = simplices_to_check {
        tracing::debug!(
            "[Delaunay debug] find_delaunay_violations: checking {} requested simplices",
            keys.len()
        );
    } else {
        tracing::debug!("[Delaunay debug] find_delaunay_violations: checking all simplices");
    }

    #[cfg(any(test, debug_assertions))]
    let mut processed_simplices = 0usize;

    // Helper closure to process simplices
    let mut process_simplex = |simplex_key: SimplexKey| -> Result<(), DelaunayValidationError> {
        #[cfg(any(test, debug_assertions))]
        {
            processed_simplices += 1;
        }

        if let Some(violating_simplex) =
            validate_simplex_delaunay(tds, simplex_key, &mut simplex_vertex_points)?
        {
            violating_simplices.push(violating_simplex);
        }
        Ok(())
    };

    // Process simplices based on input
    match simplices_to_check {
        Some(keys) => {
            for &simplex_key in keys {
                process_simplex(simplex_key)?;
            }
        }
        None => {
            for simplex_key in tds.simplex_keys() {
                process_simplex(simplex_key)?;
            }
        }
    }

    #[cfg(any(test, debug_assertions))]
    tracing::debug!(
        "[Delaunay debug] find_delaunay_violations: processed {} simplices, found {} violating simplices",
        processed_simplices,
        violating_simplices.len()
    );

    Ok(violating_simplices)
}

/// Build a structured Delaunay violation report.
///
/// This is the structured counterpart to
/// `debug_print_first_delaunay_violation`. It uses the same robust
/// empty-circumsphere scan as [`find_delaunay_violations`] and returns compact,
/// key-based diagnostics that can be attached to bug reports or inspected by
/// tests without relying on tracing output.
///
/// # Arguments
///
/// * `tds` - The triangulation data structure to inspect.
/// * `simplices_to_check` - Optional subset of simplices to check. Missing simplices are
///   skipped by the underlying scan but still counted in
///   [`DelaunayViolationReport::checked_simplices`].
///
/// # Errors
///
/// Returns [`DelaunayValidationError`] if the underlying Delaunay scan finds
/// invalid simplex structure, missing vertex references, or robust predicate
/// conversion failures.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::validation::delaunay_violation_report;
/// use delaunay::prelude::*;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Validation(#[from] delaunay::DelaunayValidationError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let report = delaunay_violation_report(dt.tds(), None)?;
/// assert!(report.violating_simplices.is_empty());
/// # Ok(())
/// # }
/// ```
pub fn delaunay_violation_report<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices_to_check: Option<&[SimplexKey]>,
) -> Result<DelaunayViolationReport, DelaunayValidationError> {
    let violating_simplices = find_delaunay_violations(tds, simplices_to_check)?;
    let checked_simplices =
        simplices_to_check.map_or_else(|| tds.number_of_simplices(), <[_]>::len);
    let violation_details: Vec<_> = violating_simplices
        .iter()
        .filter_map(|&simplex_key| build_violation_detail(tds, simplex_key))
        .collect();

    Ok(DelaunayViolationReport {
        number_of_vertices: tds.number_of_vertices(),
        number_of_simplices: tds.number_of_simplices(),
        checked_simplices,
        violating_simplices,
        violation_details,
    })
}

/// Builds the compact detail record for a violating simplex that still exists in the TDS.
fn build_violation_detail<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
) -> Option<DelaunayViolationDetail> {
    let simplex = tds.simplex(simplex_key)?;
    let simplex_vertices = simplex.vertices().iter().copied().collect();
    let neighbor_simplices = simplex
        .neighbor_slots()
        .map_or_else(NeighborBuffer::new, |slots| slots.iter().copied().collect());
    let offending_vertex = first_offending_vertex(tds, simplex_key);

    Some(DelaunayViolationDetail {
        simplex_key,
        simplex_vertices,
        offending_vertex,
        neighbor_simplices,
    })
}

/// Finds one external vertex that witnesses a simplex's Delaunay violation, if available.
fn first_offending_vertex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
) -> Option<VertexKey> {
    let mut simplex_vertex_points: SmallVec<[Point<D>; 8]> = SmallVec::with_capacity(D + 1);
    first_delaunay_violation_witness(tds, simplex_key, &mut simplex_vertex_points)
        .ok()
        .flatten()
}

impl From<DelaunayViolationDetail> for DelaunayValidationError {
    fn from(detail: DelaunayViolationDetail) -> Self {
        Self::DelaunayViolation {
            simplex_key: detail.simplex_key,
            simplex_vertices: Box::new(detail.simplex_vertices),
            offending_vertex: detail.offending_vertex,
            neighbor_simplices: Box::new(detail.neighbor_simplices),
        }
    }
}

/// Debug helper: print detailed information about the first detected Delaunay
/// violation (or all vertices if none are found) to aid in debugging.
///
/// This function is intended for use in tests and debug builds only. It uses the
/// same robust predicates as [`find_delaunay_violations`] (and the crate-private Delaunay-property-only check)
/// and prints:
/// - A triangulation summary (vertex and simplex counts)
/// - All vertices (keys, UUIDs, coordinates)
/// - All violating simplices' vertices
/// - For the first violating simplex:
///   - At least one offending external vertex (if found)
///   - Neighbor information for each facet
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::tds::Tds;
/// use delaunay::prelude::diagnostics::debug_print_first_delaunay_violation;
///
/// let tds: Tds<(), (), 3> = Tds::empty();
/// debug_print_first_delaunay_violation(&tds, None);
/// ```
#[cfg(any(test, feature = "diagnostics"))]
#[cfg_attr(docsrs, doc(cfg(feature = "diagnostics")))]
#[expect(
    clippy::too_many_lines,
    reason = "Debug-only helper with intentionally verbose logging"
)]
pub fn debug_print_first_delaunay_violation<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices_subset: Option<&[SimplexKey]>,
) {
    // First, build the structured report used by downstream diagnostics.
    let report = match delaunay_violation_report(tds, simplices_subset) {
        Ok(report) => report,
        Err(e) => {
            tracing::warn!(
                "[Delaunay debug] debug_print_first_delaunay_violation: error while finding violations: {e}"
            );
            return;
        }
    };
    let violations = &report.violating_simplices;
    tracing::debug!(
        "[Delaunay debug] Triangulation summary: {} vertices, {} simplices, {} checked simplices",
        report.number_of_vertices,
        report.number_of_simplices,
        report.checked_simplices,
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
            "[Delaunay debug] No Delaunay violations detected for requested simplex subset"
        );
        return;
    }
    tracing::debug!(
        "[Delaunay debug] Delaunay violations detected in {} simplex(s):",
        violations.len()
    );

    // Dump each violating simplex with its vertices.
    for simplex_key in violations {
        match tds.simplex(*simplex_key) {
            Some(simplex) => {
                tracing::debug!(
                    "[Delaunay debug]  Simplex {:?}: uuid={}, vertices:",
                    simplex_key,
                    simplex.uuid()
                );
                for &vkey in simplex.vertices() {
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
                    "[Delaunay debug]  Simplex {simplex_key:?} not found in TDS during violation dump"
                );
            }
        }
    }

    // Focus on the first violating simplex to identify at least one offending
    // external vertex and neighbor information.
    let first_simplex_key = violations[0];
    let Some(simplex) = tds.simplex(first_simplex_key) else {
        tracing::debug!(
            "[Delaunay debug] First violating simplex {first_simplex_key:?} not found in TDS"
        );
        return;
    };

    let offending = first_offending_vertex(tds, first_simplex_key).and_then(|vkey| {
        tds.vertex(vkey)
            .map(|vertex| (vkey, *vertex.point()))
            .or_else(|| {
                tracing::warn!(
                    "[Delaunay debug] First offending vertex {vkey:?} for simplex {first_simplex_key:?} was not found in TDS",
                );
                None
            })
    });

    if let Some((off_vkey, off_point)) = offending {
        tracing::debug!(
            "[Delaunay debug]  Offending external vertex: vkey={off_vkey:?}, point={off_point:?}",
        );
    } else {
        tracing::debug!(
            "[Delaunay debug]  No offending external vertex found for first violating simplex (possible degeneracy or removed vertices)"
        );
    }

    // Neighbor information for the first violating simplex.
    if let Some(neighbors) = simplex.neighbor_slots() {
        for (facet_idx, neighbor_slot) in neighbors.iter().copied().enumerate() {
            match neighbor_slot {
                NeighborSlot::Neighbor(neighbor_key) => {
                    if let Some(neighbor_simplex) = tds.simplex(neighbor_key) {
                        tracing::debug!(
                            "[Delaunay debug]  facet {facet_idx}: slot=Assigned, neighbor simplex {neighbor_key:?}, uuid={}",
                            neighbor_simplex.uuid()
                        );
                    } else {
                        tracing::debug!(
                            "[Delaunay debug]  facet {facet_idx}: slot=Assigned, neighbor simplex {neighbor_key:?} missing from TDS",
                        );
                    }
                }
                NeighborSlot::Boundary => {
                    tracing::debug!(
                        "[Delaunay debug]  facet {facet_idx}: slot=Boundary (hull facet)"
                    );
                }
                NeighborSlot::Unassigned => {
                    tracing::debug!(
                        "[Delaunay debug]  facet {facet_idx}: slot=Unassigned (missing neighbor wiring)"
                    );
                }
            }
        }
    } else {
        tracing::debug!(
            "[Delaunay debug]  First violating simplex has no neighbors assigned (neighbors() == None)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::incremental_insertion::repair_neighbor_pointers;
    use crate::core::simplex::{NeighborSlot, Simplex};
    use crate::core::triangulation::Triangulation;
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::FastKernel;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::{CoordinateConversionError, InvalidCoordinateValue};
    use crate::triangulation::DelaunayTriangulation;
    use crate::vertex;
    use slotmap::KeyData;
    use std::{assert_matches, sync::Once};

    fn test_vertex<const D: usize>(coords: [f64; D]) -> Vertex<(), D> {
        vertex!(coords).unwrap()
    }

    #[test]
    fn delaunay_validator_reports_no_violations_for_simple_tetrahedron() {
        init_tracing();
        tracing::debug!("Testing Delaunay validator and debug helper on a simple 3D tetrahedron");

        let vertices = vec![
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];

        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let tds = &dt.as_triangulation().tds;

        // Basic Delaunay helpers should report no violations.
        assert!(
            is_delaunay_property_only(tds).is_ok(),
            "Simple tetrahedron should satisfy the Delaunay property"
        );
        let violations = find_delaunay_violations(tds, None).unwrap();
        assert!(
            violations.is_empty(),
            "find_delaunay_violations should report no violating simplices for a tetrahedron"
        );

        // Smoke test for the debug helper: it should not panic and should print a
        // summary indicating that no violations were found.
        #[cfg(any(test, feature = "diagnostics"))]
        debug_print_first_delaunay_violation(tds, None);
    }

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer()
                .try_init();
        });
    }

    fn build_non_delaunay_quad_2d() -> (Tds<(), (), 2>, SimplexKey, SimplexKey) {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let a = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(test_vertex([0.8, 0.8]))
            .unwrap();

        let simplex_1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let simplex_2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, c, d], None).unwrap())
            .unwrap();

        tds.assign_incident_simplices().unwrap();
        repair_neighbor_pointers(&mut tds).unwrap();

        (tds, simplex_1, simplex_2)
    }

    #[test]
    fn delaunay_validator_reports_violation_for_non_delaunay_quad_2d() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let a = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(test_vertex([0.8, 0.8]))
            .unwrap();

        let simplex_1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let simplex_2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, c, d], None).unwrap())
            .unwrap();
        tds.assign_incident_simplices().unwrap();

        match is_delaunay_property_only(&tds) {
            Err(DelaunayValidationError::DelaunayViolation {
                simplex_key,
                offending_vertex,
                ..
            }) => {
                assert!(simplex_key == simplex_1 || simplex_key == simplex_2);
                assert!(offending_vertex.is_some());
            }
            other => panic!("Expected DelaunayViolation, got {other:?}"),
        }
    }

    #[test]
    fn find_delaunay_violations_subset_skips_missing_simplices() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let a = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(test_vertex([0.8, 0.8]))
            .unwrap();

        let simplex_1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _simplex_2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, c, d], None).unwrap())
            .unwrap();
        tds.assign_incident_simplices().unwrap();

        let violations =
            find_delaunay_violations(&tds, Some(&[simplex_1, SimplexKey::default()])).unwrap();
        assert_eq!(violations.len(), 1);
        assert!(violations.contains(&simplex_1));
    }

    #[test]
    fn delaunay_validation_handles_empty_tds() {
        init_tracing();
        let tds: Tds<(), (), 2> = Tds::empty();

        assert!(is_delaunay_property_only(&tds).is_ok());
        let violations = find_delaunay_violations(&tds, None).unwrap();
        assert!(violations.is_empty());
    }

    #[test]
    fn find_delaunay_violations_subset_filters_non_violating_simplex() {
        init_tracing();
        let (tds, simplex_1, simplex_2) = build_non_delaunay_quad_2d();

        let violations = find_delaunay_violations(&tds, None).unwrap();
        assert_eq!(violations.len(), 1);
        let violating_simplex = violations[0];
        let non_violating_simplex = if violating_simplex == simplex_1 {
            simplex_2
        } else {
            simplex_1
        };

        let subset = find_delaunay_violations(&tds, Some(&[non_violating_simplex])).unwrap();
        assert!(subset.is_empty());
    }

    #[test]
    fn delaunay_property_only_rejects_non_finite_vertex_at_point_boundary() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let a = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();

        tds.insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();

        assert!(Point::<2>::try_new([f64::NAN, 0.0]).is_err());

        assert!(
            is_delaunay_property_only(&tds).is_ok(),
            "delaunay_property_only should remain valid when non-finite vertices are rejected before insertion"
        );
    }

    #[test]
    fn numeric_predicate_error_display_includes_context() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(1));
        let vertex_key = VertexKey::from(KeyData::from_ffi(2));
        let source = CoordinateConversionError::NonFiniteValue {
            coordinate_index: 0,
            coordinate_value: InvalidCoordinateValue::Nan,
        };
        let err = DelaunayValidationError::NumericPredicateError {
            simplex_key,
            vertex_key,
            source,
        };
        let message = err.to_string();

        assert!(message.contains("Numeric predicate failure"));
        assert!(message.contains("simplex"));
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
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();

        let report = delaunay_violation_report(dt.tds(), None).unwrap();

        assert!(report.is_valid());
        assert_eq!(report.number_of_vertices, 4);
        assert_eq!(report.number_of_simplices, 1);
        assert_eq!(report.checked_simplices, 1);
        assert!(report.first_violation().is_none());
    }

    #[test]
    fn delaunay_violation_report_includes_first_violation_detail() {
        init_tracing();
        let (tds, simplex_1, simplex_2) = build_non_delaunay_quad_2d();

        let report = delaunay_violation_report(&tds, None).unwrap();

        assert!(!report.is_valid());
        assert_eq!(report.violating_simplices.len(), 1);
        let detail = report
            .first_violation()
            .expect("violating report should include first violation details");
        assert!(std::ptr::eq(detail, &report.violation_details[0]));
        assert!(detail.simplex_key == simplex_1 || detail.simplex_key == simplex_2);
        assert_eq!(detail.simplex_vertices.len(), 3);
        assert_eq!(detail.neighbor_simplices.len(), 3);
        let expected_neighbor = if detail.simplex_key == simplex_1 {
            simplex_2
        } else {
            simplex_1
        };
        assert!(
            detail.neighbor_simplices.iter().copied().any(
                |slot| matches!(slot, NeighborSlot::Neighbor(key) if key == expected_neighbor)
            ),
            "violation detail should preserve the assigned neighbor slot"
        );
        assert!(detail.offending_vertex.is_some());
    }

    #[test]
    fn delaunay_violation_detail_preserves_neighbor_slot_state() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let a = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        tds.assign_neighbors().unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.ensure_neighbors_buffer_mut()[1] = NeighborSlot::Unassigned;
        }

        let detail = build_violation_detail(&tds, simplex_key).unwrap();

        assert_eq!(
            detail.neighbor_simplices.as_slice(),
            &[
                NeighborSlot::Boundary,
                NeighborSlot::Unassigned,
                NeighborSlot::Boundary
            ]
        );
    }

    #[test]
    fn delaunay_violation_report_tracks_requested_subset_size() {
        init_tracing();
        let (tds, simplex_1, _) = build_non_delaunay_quad_2d();

        let report =
            delaunay_violation_report(&tds, Some(&[simplex_1, SimplexKey::default()])).unwrap();

        assert_eq!(report.checked_simplices, 2);
        assert_eq!(report.violating_simplices.as_slice(), &[simplex_1]);
        assert_eq!(
            report.first_violation().map(|detail| detail.simplex_key),
            Some(simplex_1)
        );
    }

    #[test]
    fn delaunay_property_only_reports_triangulation_state_on_missing_vertex() {
        init_tracing();
        let vertices = vec![
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        let original_vertices = {
            let simplex = tds.simplex(simplex_key).unwrap();
            simplex.vertices().to_vec()
        };
        let invalid_vkey = VertexKey::from(KeyData::from_ffi(u64::MAX));

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.clear_vertex_keys();
            for (idx, &vkey) in original_vertices.iter().enumerate() {
                if idx == 0 {
                    simplex.push_vertex_key(invalid_vkey);
                } else {
                    simplex.push_vertex_key(vkey);
                }
            }
        }

        let err = is_delaunay_property_only(&tds).unwrap_err();
        assert_matches!(
            err,
            DelaunayValidationError::TriangulationState {
                source: TdsError::VertexNotFound {
                    vertex_key,
                    ref context,
                },
            } if vertex_key == invalid_vkey
                && context.contains(&format!("{simplex_key:?}"))
        );
    }

    #[test]
    fn is_delaunay_property_only_reports_invalid_simplex() {
        init_tracing();
        let vertices = vec![
            test_vertex([0.0, 0.0]),
            test_vertex([1.0, 0.0]),
            test_vertex([0.0, 1.0]),
        ];

        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        let simplex = tds.simplex_mut(simplex_key).unwrap();
        simplex.ensure_neighbors_buffer_mut().truncate(2); // wrong length (expected 3)

        let err = is_delaunay_property_only(&tds).unwrap_err();
        assert_matches!(err, DelaunayValidationError::InvalidSimplex { .. });
    }
}

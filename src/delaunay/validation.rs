//! Delaunay-level validation APIs, proofs, and construction diagnostics.
//!
//! This module owns validation at the [`DelaunayTriangulation`](crate::DelaunayTriangulation)
//! boundary: Level 5 fast-fail checks, first diagnostics, aggregate reports,
//! cumulative validation roll-up, and construction-time validation proofs. The
//! lower-level empty-circumsphere scan over bare [`Tds`](crate::tds::Tds) storage lives in
//! `property_validation`.

#![forbid(unsafe_code)]

use crate::core::algorithms::flips::{
    DelaunayRepairError, verify_triangulation_via_flip_predicates,
};
use crate::core::algorithms::incremental_insertion::InsertionError;
use crate::core::operations::DelaunayInsertionState;
use crate::core::realization::TriangulationRealizationValidationError;
use crate::core::tds::{
    InvariantError, InvariantKind, InvariantViolation, SimplexKey, Tds, TdsError,
    TriangulationValidationReport,
};
use crate::core::traits::data_type::DataType;
use crate::core::triangulation::Triangulation;
use crate::core::validation::{TopologyGuarantee, TriangulationValidationError};
#[cfg(feature = "diagnostics")]
use crate::delaunay_property_validation::debug_print_first_delaunay_violation as debug_print_first_tds_delaunay_violation;
use crate::delaunay_property_validation::{
    DelaunayValidationError, DelaunayViolationReport,
    delaunay_violation_report as tds_delaunay_violation_report, is_delaunay_property_only,
};
use crate::geometry::kernel::Kernel;
use crate::repair::DelaunayRepairOperation;
use crate::topology::traits::topological_space::GlobalTopology;
use crate::triangulation::DelaunayTriangulation;
use std::num::NonZeroUsize;
use thiserror::Error;

/// Proof that a candidate's underlying TDS passed structural validation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TdsStructureValidationProof(());

/// Proof that a candidate passed the full validation boundary for a Delaunay wrapper.
///
/// The proof is minted only after Levels 1-4 triangulation validation and the
/// Level 5 Delaunay-property check succeed for the candidate's topology model.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DelaunayTriangulationValidationProof(());

/// Proof that a candidate passed Levels 1-4 through realized-geometry validation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TriangulationRealizationValidationProof(());

/// Internal assembly stage for a triangulation that has not crossed its validation boundary yet.
///
/// This keeps raw or freshly assembled [`Tds`] values out of the final
/// [`DelaunayTriangulation`] wrapper until the caller has proved the relevant
/// invariants for the construction path.
#[derive(Clone, Debug)]
pub(crate) struct DelaunayTriangulationCandidate<K, U, V, const D: usize> {
    candidate: DelaunayTriangulation<K, U, V, D>,
}

impl<K, U, V, const D: usize> DelaunayTriangulationCandidate<K, U, V, D> {
    /// Assembles a validation candidate with the topology context used for proof checks.
    ///
    /// The global topology is installed before any validation proof is minted so
    /// boundary classification and Euler checks use the construction path's
    /// intended topology rather than the Euclidean default.
    pub(crate) const fn assemble(
        tds: Tds<U, V, D>,
        kernel: K,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Self {
        let validation_policy = topology_guarantee.default_validation_policy();
        Self {
            candidate: DelaunayTriangulation {
                tri: Triangulation {
                    kernel,
                    tds,
                    global_topology,
                    validation_policy,
                    topology_guarantee,
                },
                insertion_state: DelaunayInsertionState::new(),
                spatial_index: None,
            },
        }
    }

    /// Validates Level 1–2 TDS structure and returns proof for structural-only assembly paths.
    pub(crate) fn validate_tds_structure(&self) -> Result<TdsStructureValidationProof, TdsError> {
        self.candidate.tri.tds.validate()?;
        Ok(TdsStructureValidationProof(()))
    }

    /// Converts a candidate using proof from [`Self::validate_delaunay_property`].
    pub(crate) fn into_validated_delaunay(
        self,
        _proof: DelaunayTriangulationValidationProof,
    ) -> DelaunayTriangulation<K, U, V, D> {
        self.candidate
    }

    /// Converts a candidate after the caller has proved structural validity.
    ///
    /// Periodic quotient construction uses this boundary after reconstructing
    /// closed neighbor and incidence relations. Exhaustive periodic realization
    /// validation remains available through the public cumulative validator.
    pub(crate) fn into_structurally_valid_delaunay(
        self,
        _proof: TdsStructureValidationProof,
    ) -> DelaunayTriangulation<K, U, V, D> {
        self.candidate
    }

    /// Converts a candidate after the caller has proved Levels 1-4 validity.
    pub(crate) fn into_realization_validated_delaunay(
        self,
        _proof: TriangulationRealizationValidationProof,
    ) -> DelaunayTriangulation<K, U, V, D> {
        self.candidate
    }

    #[cfg(test)]
    pub(crate) fn into_repairable_delaunay_for_test(self) -> DelaunayTriangulation<K, U, V, D> {
        self.candidate
    }
}

impl<K, U, V, const D: usize> DelaunayTriangulationCandidate<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
{
    /// Normalizes coherent orientation on the assembled candidate.
    pub(crate) fn normalize_and_promote_positive_orientation(
        &mut self,
    ) -> Result<(), InsertionError> {
        self.candidate
            .tri
            .normalize_and_promote_positive_orientation()
    }

    /// Validates Level 3 intrinsic topology.
    pub(crate) fn validate_topology(&self) -> Result<(), InvariantError> {
        self.candidate.tri.is_valid_topology()
    }

    /// Validates completion-time PL-manifold constraints.
    pub(crate) fn validate_at_completion(&self) -> Result<(), InvariantError> {
        self.candidate.tri.validate_at_completion()
    }

    /// Validates explicit geometric nondegeneracy constraints.
    pub(crate) fn validate_geometric_nondegeneracy(&self) -> Result<(), TdsError> {
        self.candidate.tri.validate_geometric_nondegeneracy()
    }
}

impl<K, U, V, const D: usize> DelaunayTriangulationCandidate<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Validates all invariants required before exposing a Delaunay wrapper.
    ///
    /// This preserves the public reconstruction contract for
    /// [`DelaunayTriangulation`]: a candidate cannot cross the boundary until
    /// its underlying [`Triangulation`] passes Levels 1-3 validation. Euclidean
    /// candidates additionally pass Level 4 realized-geometry validation before
    /// the Level 5 Delaunay property is checked with the topology-appropriate
    /// validator.
    pub(crate) fn validate_delaunay_property(
        &self,
    ) -> Result<DelaunayTriangulationValidationProof, DelaunayTriangulationValidationError> {
        self.candidate.tri.validate_realization()?;

        if self.candidate.global_topology().is_euclidean() {
            is_delaunay_property_only(&self.candidate.tri.tds).map_err(|source| {
                DelaunayTriangulationValidationError::VerificationFailed {
                    source: Box::new(DelaunayVerificationError::from(source)),
                }
            })?;
        } else {
            self.candidate.is_valid_delaunay()?;
        }

        Ok(DelaunayTriangulationValidationProof(()))
    }

    /// Validates Levels 1-4 without enforcing the Level 5 Delaunay property.
    pub(crate) fn validate_realization_only(
        &self,
    ) -> Result<TriangulationRealizationValidationProof, DelaunayTriangulationValidationError> {
        self.candidate.tri.validate_realization()?;
        Ok(TriangulationRealizationValidationProof(()))
    }
}

/// Typed source for Level 5 Delaunay verification failures.
///
/// Passive validation has two implementation paths:
/// - flip-predicate verification via
///   [`DelaunayTriangulation::verify_via_flip_predicates`](crate::DelaunayTriangulation::verify_via_flip_predicates),
///   used by [`DelaunayTriangulation::is_valid_delaunay`](crate::DelaunayTriangulation::is_valid_delaunay)
/// - empty-circumsphere validation via `is_delaunay_property_only`, used when
///   reconstructing Euclidean triangulations from raw [`Tds`]
///
/// This wrapper preserves which path failed and carries the original typed
/// error so callers can inspect predicate, topology, and simplex-key context
/// without parsing display text.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::{DelaunayRepairError, DelaunayRepairPostconditionFailure};
/// use delaunay::prelude::validation::{
///     DelaunayVerificationError, DelaunayVerificationErrorKind,
/// };
///
/// let source =
///     DelaunayVerificationError::from(DelaunayRepairError::PostconditionFailed {
///         reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
///     });
///
/// assert_eq!(
///     DelaunayVerificationErrorKind::from(&source),
///     DelaunayVerificationErrorKind::FlipPredicates,
/// );
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayVerificationError {
    /// Flip-predicate verification failed.
    #[error("flip-predicate verification failed: {source}")]
    FlipPredicates {
        /// Underlying flip verification error.
        #[source]
        source: Box<DelaunayRepairError>,
    },

    /// Empty-circumsphere validation failed.
    #[error("empty-circumsphere validation failed: {source}")]
    EmptyCircumsphere {
        /// Underlying Delaunay property validation error.
        #[source]
        source: Box<DelaunayValidationError>,
    },
}

impl From<DelaunayRepairError> for DelaunayVerificationError {
    fn from(source: DelaunayRepairError) -> Self {
        Self::FlipPredicates {
            source: Box::new(source),
        }
    }
}

impl From<DelaunayValidationError> for DelaunayVerificationError {
    fn from(source: DelaunayValidationError) -> Self {
        Self::EmptyCircumsphere {
            source: Box::new(source),
        }
    }
}

/// Discriminant for compact Level 5 verification-source summaries.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::repair::{DelaunayRepairError, DelaunayRepairPostconditionFailure};
/// use delaunay::prelude::validation::{
///     DelaunayVerificationError, DelaunayVerificationErrorKind,
/// };
///
/// let source =
///     DelaunayVerificationError::from(DelaunayRepairError::PostconditionFailed {
///         reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
///     });
/// let kind = DelaunayVerificationErrorKind::from(&source);
///
/// assert_eq!(kind, DelaunayVerificationErrorKind::FlipPredicates);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DelaunayVerificationErrorKind {
    /// Flip-predicate verification failed.
    FlipPredicates,
    /// Empty-circumsphere validation failed.
    EmptyCircumsphere,
}

impl From<&DelaunayVerificationError> for DelaunayVerificationErrorKind {
    fn from(source: &DelaunayVerificationError) -> Self {
        match source {
            DelaunayVerificationError::FlipPredicates { .. } => Self::FlipPredicates,
            DelaunayVerificationError::EmptyCircumsphere { .. } => Self::EmptyCircumsphere,
        }
    }
}

/// Errors that can occur during Delaunay triangulation validation and repair.
///
/// The first four variants are returned by [`DelaunayTriangulation::validate`](crate::DelaunayTriangulation::validate)
/// (validation Levels 1-5):
/// - [`Tds`](Self::Tds) — element or TDS structural errors (Levels 1–2).
/// - [`Triangulation`](Self::Triangulation) — topology errors (Level 3).
/// - [`Realization`](Self::Realization) — realized-geometry errors (Level 4).
/// - [`VerificationFailed`](Self::VerificationFailed) — Delaunay property violation (Level 5).
///
/// [`DelaunayTriangulation::is_valid_delaunay`](crate::DelaunayTriangulation::is_valid_delaunay) returns only the Level 5
/// [`VerificationFailed`](Self::VerificationFailed) variant.
///
/// The repair-failure variants are **not** returned by `validate()` or
/// `is_valid()`. They are produced by mutating operations that invoke
/// flip-based repair internally (e.g. [`DelaunayTriangulation::delete_vertex`](crate::DelaunayTriangulation::delete_vertex)).
///
/// When manually forwarding lower-layer validation errors, prefer
/// `DelaunayTriangulationValidationError::from(tds_error)` or `.into()` for
/// [`TdsError`] and [`TriangulationValidationError`]. The enum stores those
/// sources behind `Box` to keep `Result<_, DelaunayTriangulationValidationError>`
/// compact while preserving typed error inspection.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
/// use delaunay::prelude::validation::DelaunayTriangulationValidationError;
///
/// # fn main() -> DelaunayResult<()> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// let result: Result<(), DelaunayTriangulationValidationError> = dt.validate();
/// assert!(result.is_ok());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayTriangulationValidationError {
    /// Lower-layer element or TDS structural validation error (Levels 1–2).
    #[error(transparent)]
    Tds(Box<TdsError>),

    /// Lower-layer topology validation error (Level 3).
    #[error(transparent)]
    Triangulation(Box<TriangulationValidationError>),

    /// Lower-layer realized-geometry validation error (Level 4).
    #[error(transparent)]
    Realization(Box<TriangulationRealizationValidationError>),

    /// Flip-based Delaunay verification detected a violation.
    ///
    /// This is returned by [`DelaunayTriangulation::is_valid_delaunay`](crate::DelaunayTriangulation::is_valid_delaunay) when the fast
    /// O(simplices) flip-predicate scan finds a Delaunay violation.  The error is
    /// a Level 5 (Delaunay property) issue, not a Level 1–2 structural problem.
    /// The [`DelaunayVerificationError`] source distinguishes flip-predicate
    /// validation from empty-circumsphere reconstruction validation.
    #[error("Delaunay verification failed: {source}")]
    VerificationFailed {
        /// Typed verification failure source.
        #[source]
        source: Box<DelaunayVerificationError>,
    },

    /// Flip-based Delaunay repair failed during a specific mutating operation.
    ///
    /// This preserves the underlying [`DelaunayRepairError`] so callers can
    /// inspect budget exhaustion, topology errors, predicate failures, and other
    /// repair causes without parsing display text. Operations that report this
    /// variant are responsible for documenting whether failure is transactional;
    /// [`delete_vertex`](crate::DelaunayTriangulation::delete_vertex)
    /// restores the pre-deletion triangulation when post-deletion repair fails.
    ///
    /// **Not** returned by `validate()` or `is_valid()` — those use
    /// [`VerificationFailed`](Self::VerificationFailed) for passive checks.
    #[error("Delaunay repair failed during {operation}: {source}")]
    RepairOperationFailed {
        /// Mutating operation that invoked repair.
        operation: DelaunayRepairOperation,
        /// Underlying flip-repair failure.
        #[source]
        source: Box<DelaunayRepairError>,
    },
}

impl From<TdsError> for DelaunayTriangulationValidationError {
    fn from(source: TdsError) -> Self {
        Self::Tds(Box::new(source))
    }
}

impl From<TriangulationValidationError> for DelaunayTriangulationValidationError {
    fn from(source: TriangulationValidationError) -> Self {
        Self::Triangulation(Box::new(source))
    }
}

impl From<TriangulationRealizationValidationError> for DelaunayTriangulationValidationError {
    fn from(source: TriangulationRealizationValidationError) -> Self {
        match source {
            TriangulationRealizationValidationError::Tds(source) => Self::Tds(source),
            TriangulationRealizationValidationError::Triangulation(source) => {
                Self::Triangulation(source)
            }
            source => Self::Realization(Box::new(source)),
        }
    }
}

/// Cadence for explicit validation checkpoints during construction diagnostics.
///
/// This is separate from [`crate::ValidationPolicy`],
/// which controls automatic insertion-time validation inside
/// [`crate::Triangulation`]. Diagnostic
/// harnesses can use this cadence for explicit periodic
/// [`DelaunayTriangulation::is_valid_delaunay`](crate::DelaunayTriangulation::is_valid_delaunay)
/// checks without overloading repair policy or exposing raw `Option<usize>`
/// scheduling in logs.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::validation::ValidationCadence;
///
/// let cadence = ValidationCadence::from_optional_every(Some(128));
/// assert!(!cadence.should_validate(0));
/// assert!(!cadence.should_validate(127));
/// assert!(cadence.should_validate(128));
/// ```
#[must_use = "validation cadence values only affect diagnostics when they are used"]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationCadence {
    /// Disable explicit periodic validation checkpoints.
    Never,
    /// Run explicit validation every N successful insertion attempts.
    EveryN(NonZeroUsize),
}

impl ValidationCadence {
    /// Converts an optional integer cadence into a typed validation cadence.
    ///
    /// `None` and `Some(0)` disable periodic validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::validation::ValidationCadence;
    ///
    /// std::assert_matches!(
    ///     ValidationCadence::from_optional_every(Some(32)),
    ///     ValidationCadence::EveryN(every) if every.get() == 32,
    /// );
    /// assert_eq!(
    ///     ValidationCadence::from_optional_every(None),
    ///     ValidationCadence::Never,
    /// );
    /// ```
    pub const fn from_optional_every(validate_every: Option<usize>) -> Self {
        match validate_every {
            None | Some(0) => Self::Never,
            Some(every) => {
                if let Some(every) = NonZeroUsize::new(every) {
                    Self::EveryN(every)
                } else {
                    // Logically unreachable because `Some(0)` is matched above.
                    // Keep this branch so the function remains const without
                    // introducing unsafe code to construct `NonZeroUsize`.
                    Self::Never
                }
            }
        }
    }

    /// Returns true when validation should run for a one-based insertion count.
    ///
    /// A count of `0` never triggers validation because no insertion has
    /// completed yet.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::validation::ValidationCadence;
    ///
    /// let cadence = ValidationCadence::from_optional_every(Some(4));
    /// assert!(!cadence.should_validate(0));
    /// assert!(!cadence.should_validate(3));
    /// assert!(cadence.should_validate(4));
    /// ```
    #[must_use]
    pub const fn should_validate(self, insertion_count: usize) -> bool {
        match self {
            Self::Never => false,
            Self::EveryN(every) => {
                insertion_count != 0 && insertion_count.is_multiple_of(every.get())
            }
        }
    }
}

// =============================================================================
// VALIDATION (Minimal Bounds)
// =============================================================================

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    // -------------------------------------------------------------------------
    // VALIDATION
    // -------------------------------------------------------------------------

    /// Validates the Delaunay empty-circumsphere property (Level 5).
    ///
    /// This is the Delaunay layer's `is_valid`: it checks **only** the Delaunay property
    /// and intentionally does **not** run lower-layer validation.
    ///
    /// **Performance**: Uses fast O(simplices) flip-based verification instead of the naive
    /// O(simplices × vertices) brute-force check, providing ~40-100x speedup. This method is
    /// correct for all properly-constructed triangulations (which is the standard case).
    ///
    /// For cumulative validation across the whole hierarchy, use [`validate`](Self::validate).
    ///
    /// # Errors
    ///
    /// Returns a [`DelaunayTriangulationValidationError`] if Level 5 verification
    /// detects a Delaunay violation, or if the underlying triangulation state is
    /// inconsistent and prevents geometric predicates from being evaluated. The
    /// [`VerificationFailed`](DelaunayTriangulationValidationError::VerificationFailed)
    /// variant preserves the typed [`DelaunayVerificationError`] source.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    /// use delaunay::prelude::query::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices_4d = [
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices_4d).build()?;
    ///
    /// // Level 5: Delaunay property only
    /// assert!(dt.is_valid_delaunay().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_valid_delaunay(&self) -> Result<(), DelaunayTriangulationValidationError> {
        // Use fast flip-based verification (O(simplices) instead of O(simplices × vertices))
        self.verify_via_flip_predicates().map_err(|source| {
            DelaunayTriangulationValidationError::VerificationFailed {
                source: Box::new(DelaunayVerificationError::from(source)),
            }
        })
    }

    /// Returns the first actionable Level 5 Delaunay diagnostic, if any.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, vertex,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// assert!(dt.delaunay_diagnostic().is_none());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn delaunay_diagnostic(&self) -> Option<InvariantViolation> {
        self.delaunay_report()
            .err()
            .and_then(|report| report.violations.into_iter().next())
    }

    /// Builds a Level 5 Delaunay-property report.
    ///
    /// Euclidean triangulations use the all-violations empty-circumsphere scan.
    /// Non-Euclidean topologies currently use the topology-aware flip verifier
    /// and report the first violation it finds; this avoids applying an
    /// ordinary Euclidean circumsphere scan to periodic charts.
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationReport)` when one or more checkable
    /// Level 5 violations are found or when Delaunay predicates cannot be
    /// evaluated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, vertex,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// assert!(dt.delaunay_report().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn delaunay_report(&self) -> Result<(), TriangulationValidationReport> {
        if self.global_topology().is_euclidean() {
            return match tds_delaunay_violation_report(self.tds(), None) {
                Ok(report) if report.is_valid() => Ok(()),
                Ok(report) => Err(TriangulationValidationReport {
                    violations: report
                        .violation_details
                        .into_iter()
                        .map(|detail| InvariantViolation {
                            kind: InvariantKind::DelaunayProperty,
                            error: InvariantError::Delaunay(
                                DelaunayTriangulationValidationError::VerificationFailed {
                                    source: Box::new(DelaunayVerificationError::from(
                                        DelaunayValidationError::from(detail),
                                    )),
                                },
                            ),
                        })
                        .collect(),
                }),
                Err(source) => Err(TriangulationValidationReport {
                    violations: vec![InvariantViolation {
                        kind: InvariantKind::DelaunayProperty,
                        error: InvariantError::Delaunay(
                            DelaunayTriangulationValidationError::VerificationFailed {
                                source: Box::new(DelaunayVerificationError::from(source)),
                            },
                        ),
                    }],
                }),
            };
        }

        self.is_valid_delaunay()
            .map_err(|error| TriangulationValidationReport {
                violations: vec![InvariantViolation {
                    kind: InvariantKind::DelaunayProperty,
                    error: InvariantError::Delaunay(error),
                }],
            })
    }

    /// Builds a detailed Delaunay empty-circumsphere violation report.
    ///
    /// This is the high-level owner-bound counterpart to the TDS-level
    /// [`delaunay_violation_report`](crate::delaunay_violation_report) helper.
    /// It keeps callers on the `DelaunayTriangulation` API while returning the
    /// same typed, key-oriented diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayValidationError`] if the scan encounters invalid
    /// simplex structure, missing vertex references, or robust predicate
    /// conversion failures.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let report = dt.delaunay_violation_report(None)?;
    /// assert!(report.is_valid());
    /// # Ok(())
    /// # }
    /// ```
    pub fn delaunay_violation_report(
        &self,
        simplices_to_check: Option<&[SimplexKey]>,
    ) -> Result<DelaunayViolationReport, DelaunayValidationError> {
        tds_delaunay_violation_report(self.tds(), simplices_to_check)
    }

    /// Logs detailed information for the first Delaunay violation, when present.
    ///
    /// This diagnostics-only method keeps debug workflows on the high-level
    /// triangulation owner instead of requiring public TDS access.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// dt.debug_print_first_delaunay_violation(None);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "diagnostics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "diagnostics")))]
    pub fn debug_print_first_delaunay_violation(&self, simplices_subset: Option<&[SimplexKey]>) {
        debug_print_first_tds_delaunay_violation(self.tds(), simplices_subset);
    }

    /// Verify the Delaunay property via fast O(simplices) flip predicates.
    ///
    /// This checks the Delaunay property by testing all possible flip configurations
    /// (k=2 facets, k=3 ridges, and their inverses) instead of the naive O(simplices × vertices)
    /// brute-force check. This is ~40-100x faster while being equally correct.
    ///
    /// Ideal for property-based testing with many iterations.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayRepairError`] if any flip predicate detects a Delaunay violation.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    /// use delaunay::prelude::query::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    ///
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// // Fast O(N) verification
    /// assert!(dt.verify_via_flip_predicates().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn verify_via_flip_predicates(&self) -> Result<(), DelaunayRepairError> {
        verify_triangulation_via_flip_predicates(&self.tri)
    }

    /// Performs cumulative validation for Levels 1–5.
    ///
    /// This validates:
    /// - **Levels 1–4** via [`Triangulation::validate_realization`](crate::Triangulation::validate_realization)
    /// - **Level 5** via [`DelaunayTriangulation::is_valid_delaunay`](Self::is_valid_delaunay)
    ///
    /// # Errors
    ///
    /// Returns a [`DelaunayTriangulationValidationError`] if lower-layer validation fails, if
    /// Level 4 realized-geometry validation fails, or if the Delaunay property check (Level 5)
    /// fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    /// use delaunay::prelude::query::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices_4d = [
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices_4d).build()?;
    ///
    /// // Levels 1–5: elements + structure + topology + realization + Delaunay property
    /// assert!(dt.validate().is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate(&self) -> Result<(), DelaunayTriangulationValidationError> {
        self.tri.validate_realization()?;
        self.is_valid_delaunay()
    }

    /// Generate a comprehensive validation report for the full validation hierarchy.
    ///
    /// This is intended for debugging/telemetry (e.g. `insert_with_statistics`) where
    /// you want to see *all* violated invariants, not just the first one.
    ///
    /// # Notes
    /// - If UUID↔key mappings are inconsistent, this returns only mapping failures (other
    ///   checks may produce misleading secondary errors).
    /// - This report is **cumulative** across Levels 1–5.
    ///
    /// # Errors
    ///
    /// Returns `Err(TriangulationValidationReport)` containing all violated invariants.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    /// use delaunay::prelude::query::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// // Returns Ok(()) on success; otherwise returns a report listing all violations.
    /// let report = dt.validation_report();
    /// assert!(report.is_ok());
    /// # Ok(())
    /// # }
    /// ```
    pub fn validation_report(&self) -> Result<(), TriangulationValidationReport> {
        // Levels 1–3: reuse the Triangulation layer report.
        match self.tri.validation_report() {
            Ok(()) => {
                // Level 4 (realized geometry)
                let realization_report = self.tri.realization_report().map_err(|error| {
                    TriangulationValidationReport {
                        violations: vec![InvariantViolation {
                            kind: InvariantKind::Realization,
                            error: error.into(),
                        }],
                    }
                })?;
                let mut violations = Vec::new();
                if !realization_report.is_valid() {
                    violations.extend(realization_report.violations.into_iter().map(|error| {
                        InvariantViolation {
                            kind: InvariantKind::Realization,
                            error: error.into(),
                        }
                    }));
                }

                // Level 5 (Delaunay property)
                if let Err(delaunay_report) = self.delaunay_report() {
                    violations.extend(delaunay_report.violations);
                }

                if violations.is_empty() {
                    Ok(())
                } else {
                    Err(TriangulationValidationReport { violations })
                }
            }
            Err(mut report) => {
                // If mappings are inconsistent, return the lower-layer report unchanged.
                if report.violations.iter().any(|v| {
                    matches!(
                        v.kind,
                        InvariantKind::VertexMappings | InvariantKind::SimplexMappings
                    )
                }) {
                    return Err(report);
                }

                // Level 4 (realized geometry)
                match self.tri.realization_report() {
                    Ok(realization_report) => {
                        report
                            .violations
                            .extend(realization_report.violations.into_iter().map(|error| {
                                InvariantViolation {
                                    kind: InvariantKind::Realization,
                                    error: InvariantError::Realization(error),
                                }
                            }));
                    }
                    Err(source) => {
                        report.violations.push(InvariantViolation {
                            kind: InvariantKind::Realization,
                            error: InvariantError::Realization(source),
                        });
                    }
                }

                // Level 5 (Delaunay property)
                if let Err(delaunay_report) = self.delaunay_report() {
                    report.violations.extend(delaunay_report.violations);
                }

                if report.violations.is_empty() {
                    Ok(())
                } else {
                    Err(report)
                }
            }
        }
    }
    // -------------------------------------------------------------------------
    // PURE STRUCT ASSEMBLY
    // -------------------------------------------------------------------------
    /// Create a validated `DelaunayTriangulation` from a `Tds` with an explicit kernel.
    ///
    /// This is useful when you've serialized just the `Tds` and want to reconstruct
    /// the `DelaunayTriangulation` with a caller-supplied kernel. The `kernel`
    /// parameter provides the geometric predicates used during validation and later
    /// insertions.
    ///
    /// # Notes
    ///
    /// - The internal `insertion_state.last_inserted_simplex` "locate hint" is intentionally **not** persisted
    ///   across serialization boundaries. Reconstructing via `try_from_tds` (including the serde
    ///   `Deserialize` impl below) always resets it to `None`. This can make the first few
    ///   insertions after loading slightly slower, but is otherwise behaviorally irrelevant.
    /// - The internal spatial hash-grid index used to accelerate incremental insertion is also a
    ///   performance-only cache and is not serialized. Reconstructing via `try_from_tds` leaves it unset
    ///   so it can be rebuilt lazily on demand.
    /// - The topology guarantee ([`TopologyGuarantee`]) is also not serialized (this type serializes
    ///   only the `Tds`). Reconstructing via `try_from_tds` resets it to `TopologyGuarantee::DEFAULT`
    ///   (currently `PLManifold`). Call [`set_topology_guarantee`](Self::set_topology_guarantee)
    ///   after loading if you want to relax to `Pseudomanifold` for performance, or use
    ///   [`try_from_tds_with_topology_guarantee`](Self::try_from_tds_with_topology_guarantee) to set it
    ///   at construction time.
    /// - Runtime global topology metadata ([`GlobalTopology`]) is also not serialized. Reconstructing
    ///   via `try_from_tds` validates with [`GlobalTopology::Euclidean`]. Use
    ///   [`try_from_tds_with_topology_context`](Self::try_from_tds_with_topology_context) if you
    ///   need to validate toroidal or other non-default topology metadata during reconstruction.
    /// - Euclidean reconstruction validates Level 4 realized geometry, then
    ///   validates Level 5 with the crate's robust empty-circumsphere validator,
    ///   independent of the supplied runtime kernel. The supplied kernel is
    ///   stored for later queries and insertions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulation,
    /// };
    /// use delaunay::prelude::geometry::FastKernel;
    /// use delaunay::prelude::triangulation::Triangulation;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// // Reconstruct DelaunayTriangulation from imported low-level storage.
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 0.0, 1.0]?,
    /// ];
    /// let tds =
    ///     Triangulation::<FastKernel<f64>, (), (), 4>::build_initial_simplex(&vertices)?;
    /// let reconstructed = DelaunayTriangulation::try_from_tds(tds, FastKernel::new())?;
    /// assert_eq!(reconstructed.number_of_vertices(), 5);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationValidationError`] if the TDS violates
    /// structural, topological, realized-geometry, or Delaunay invariants.
    pub fn try_from_tds(
        tds: Tds<U, V, D>,
        kernel: K,
    ) -> Result<Self, DelaunayTriangulationValidationError> {
        Self::try_from_tds_with_topology_context(
            tds,
            kernel,
            TopologyGuarantee::DEFAULT,
            GlobalTopology::DEFAULT,
        )
    }

    /// Create a validated `DelaunayTriangulation` from a `Tds` with an explicit topology guarantee.
    ///
    /// The candidate is assembled with the requested guarantee, then validated
    /// at Levels 1-5 before being returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulation, TopologyGuarantee,
    /// };
    /// use delaunay::prelude::geometry::FastKernel;
    /// use delaunay::prelude::triangulation::Triangulation;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let tds =
    ///     Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices)?;
    /// let reconstructed = DelaunayTriangulation::try_from_tds_with_topology_guarantee(
    ///     tds,
    ///     FastKernel::new(),
    ///     TopologyGuarantee::PLManifoldStrict,
    /// )?;
    ///
    /// assert_eq!(
    ///     reconstructed.topology_guarantee(),
    ///     TopologyGuarantee::PLManifoldStrict
    /// );
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationValidationError`] if the TDS violates
    /// structural, topological, realized-geometry, or Delaunay invariants.
    pub fn try_from_tds_with_topology_guarantee(
        tds: Tds<U, V, D>,
        kernel: K,
        topology_guarantee: TopologyGuarantee,
    ) -> Result<Self, DelaunayTriangulationValidationError> {
        Self::try_from_tds_with_topology_context(
            tds,
            kernel,
            topology_guarantee,
            GlobalTopology::DEFAULT,
        )
    }

    /// Create a validated `DelaunayTriangulation` from a `Tds` with explicit topology context.
    ///
    /// This is the checked reconstruction path for serialized TDS data whose
    /// runtime [`TopologyGuarantee`] or [`GlobalTopology`] metadata must be
    /// restored before validation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulation, GlobalTopology, TopologyGuarantee,
    /// };
    /// use delaunay::prelude::geometry::FastKernel;
    /// use delaunay::prelude::triangulation::Triangulation;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let tds =
    ///     Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices)?;
    /// let reconstructed = DelaunayTriangulation::try_from_tds_with_topology_context(
    ///     tds,
    ///     FastKernel::new(),
    ///     TopologyGuarantee::PLManifoldStrict,
    ///     GlobalTopology::Euclidean,
    /// )?;
    ///
    /// assert_eq!(
    ///     reconstructed.topology_guarantee(),
    ///     TopologyGuarantee::PLManifoldStrict
    /// );
    /// assert_eq!(reconstructed.global_topology(), GlobalTopology::Euclidean);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTriangulationValidationError`] if the TDS violates
    /// structural, topological, realized-geometry, or Delaunay invariants under
    /// the supplied topology context.
    pub fn try_from_tds_with_topology_context(
        tds: Tds<U, V, D>,
        kernel: K,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Result<Self, DelaunayTriangulationValidationError> {
        let candidate = DelaunayTriangulationCandidate::assemble(
            tds,
            kernel,
            topology_guarantee,
            global_topology,
        );
        let proof = candidate.validate_delaunay_property()?;
        Ok(candidate.into_validated_delaunay(proof))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::flips::{
        DelaunayRepairDiagnostics, DelaunayRepairPostconditionFailure, RepairQueueOrder,
    };
    use crate::core::simplex::Simplex;
    use crate::core::tds::{SimplexKey, TriangulationConstructionState, VertexKey};
    use crate::core::vertex::Vertex;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::vertex;
    use slotmap::KeyData;
    use std::assert_matches;
    use std::{error::Error, sync::Once};
    use uuid::Uuid;

    fn test_vertex<const D: usize>(coords: [f64; D]) -> Vertex<(), D> {
        vertex!(coords).unwrap()
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

    fn non_delaunay_quad_tds() -> Tds<(), (), 2> {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([4.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([4.0, 2.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 2.0]))
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
        )
        .unwrap();
        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();
        tds
    }

    fn tds_from_2d_vertices_and_simplices(
        coords: &[[f64; 2]],
        simplices: &[Vec<usize>],
    ) -> Tds<(), (), 2> {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let vertex_keys: Vec<_> = coords
            .iter()
            .map(|coords| {
                tds.insert_vertex_with_mapping(test_vertex(*coords))
                    .unwrap()
            })
            .collect();

        for simplex_vertices in simplices {
            let vertices: Vec<_> = simplex_vertices
                .iter()
                .map(|&index| vertex_keys[index])
                .collect();
            tds.insert_simplex_with_mapping(Simplex::try_new_with_data(vertices, None).unwrap())
                .unwrap();
        }

        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();
        tds
    }

    fn unchecked_test_delaunay_from_tds<const D: usize>(
        tds: Tds<(), (), D>,
    ) -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> {
        DelaunayTriangulationCandidate::assemble(
            tds,
            AdaptiveKernel::new(),
            TopologyGuarantee::Pseudomanifold,
            GlobalTopology::Euclidean,
        )
        .into_repairable_delaunay_for_test()
    }

    fn synthetic_flip_verification_source(message: &str) -> DelaunayVerificationError {
        let _ = message;
        DelaunayVerificationError::from(DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        })
    }

    #[test]
    fn validation_cadence_maps_optional_every() {
        assert_eq!(
            ValidationCadence::from_optional_every(None),
            ValidationCadence::Never
        );
        assert_eq!(
            ValidationCadence::from_optional_every(Some(0)),
            ValidationCadence::Never
        );
        assert_eq!(
            ValidationCadence::from_optional_every(Some(128)),
            ValidationCadence::EveryN(NonZeroUsize::new(128).unwrap())
        );
    }

    #[test]
    fn validation_cadence_should_validate_on_multiples() {
        let cadence = ValidationCadence::EveryN(NonZeroUsize::new(64).unwrap());

        assert!(!cadence.should_validate(0));
        assert!(!cadence.should_validate(63));
        assert!(cadence.should_validate(64));
        assert!(!cadence.should_validate(65));
        assert!(cadence.should_validate(128));
        assert!(!ValidationCadence::Never.should_validate(64));
    }

    #[test]
    fn verification_failed_display_includes_context() {
        let err = DelaunayTriangulationValidationError::VerificationFailed {
            source: synthetic_flip_verification_source(
                "flip predicate detected non-Delaunay facet",
            )
            .into(),
        };
        let msg = err.to_string();

        assert!(
            msg.contains("Delaunay verification failed"),
            "Display should contain prefix: {msg}"
        );
        assert!(
            msg.contains("repair pass disconnected the triangulation"),
            "Display should contain inner message: {msg}"
        );
        let DelaunayTriangulationValidationError::VerificationFailed { source } = &err else {
            panic!("expected typed flip-predicate verification source, got {err:?}");
        };
        assert_matches!(
            source.as_ref(),
            DelaunayVerificationError::FlipPredicates { source }
                if matches!(
                    source.as_ref(),
                    DelaunayRepairError::PostconditionFailed { .. }
                )
        );
    }

    #[test]
    fn verification_error_kind_covers_empty_circumsphere_source() {
        let simplex_key = SimplexKey::default();
        let source = DelaunayVerificationError::from(DelaunayValidationError::DelaunayViolation {
            simplex_key,
            simplex_vertices: Box::default(),
            offending_vertex: None,
            neighbor_simplices: Box::default(),
        });

        assert_eq!(
            DelaunayVerificationErrorKind::from(&source),
            DelaunayVerificationErrorKind::EmptyCircumsphere,
        );
        assert!(source.to_string().contains("empty-circumsphere"));
        let DelaunayVerificationError::EmptyCircumsphere { source } = source else {
            panic!("expected empty-circumsphere source");
        };
        assert_matches!(
            source.as_ref(),
            DelaunayValidationError::DelaunayViolation {
                simplex_key: actual,
                ..
            } if *actual == simplex_key
        );
    }

    #[test]
    fn repair_operation_failed_preserves_source() {
        let source = DelaunayRepairError::NonConvergent {
            max_flips: 7,
            diagnostics: Box::new(DelaunayRepairDiagnostics {
                facets_checked: 3,
                flips_performed: 7,
                max_queue_len: 5,
                ambiguous_predicates: 0,
                ambiguous_predicate_samples: Vec::new(),
                predicate_failures: 0,
                cycle_detections: 0,
                cycle_signature_samples: Vec::new(),
                attempt: 1,
                queue_order: RepairQueueOrder::Fifo,
            }),
        };
        let err = DelaunayTriangulationValidationError::RepairOperationFailed {
            operation: DelaunayRepairOperation::VertexRemoval,
            source: Box::new(source),
        };

        let msg = err.to_string();
        assert!(msg.contains("vertex removal"));
        match &err {
            DelaunayTriangulationValidationError::RepairOperationFailed {
                operation: DelaunayRepairOperation::VertexRemoval,
                source,
            } if matches!(
                source.as_ref(),
                DelaunayRepairError::NonConvergent { max_flips: 7, .. }
            ) => {}
            other => panic!("expected typed vertex-removal repair source, got {other:?}"),
        }
        let chained = err
            .source()
            .expect("typed repair failure should expose source error")
            .to_string();
        assert!(chained.contains("failed to converge after 7 flips"));
    }

    #[test]
    fn tds_variant_display_delegates_to_source() {
        let inner = TdsError::InconsistentDataStructure {
            message: "broken link".to_string(),
        };
        let err = DelaunayTriangulationValidationError::from(inner);

        assert!(err.to_string().contains("broken link"));
    }

    #[test]
    fn triangulation_variant_display_delegates_to_source() {
        let inner = TriangulationValidationError::IsolatedVertex {
            vertex_key: VertexKey::from(KeyData::from_ffi(1)),
            vertex_uuid: Uuid::nil(),
        };
        let err = DelaunayTriangulationValidationError::from(inner);

        assert!(err.to_string().contains("Isolated vertex"));
    }

    #[test]
    fn try_from_tds_rejects_non_delaunay_connectivity() {
        init_tracing();
        let tds = non_delaunay_quad_tds();

        let err = DelaunayTriangulation::try_from_tds(tds, AdaptiveKernel::new())
            .expect_err("checked TDS reconstruction must reject non-Delaunay connectivity");

        assert!(
            matches!(
                err,
                DelaunayTriangulationValidationError::VerificationFailed { .. }
            ),
            "expected Level 5 validation failure, got {err:?}"
        );
    }

    #[test]
    fn try_from_tds_rejects_structural_validation_failure() {
        init_tracing();
        let vertices = [
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tds = dt.tds().clone();

        let vk = tds.vertex_keys().next().unwrap();
        let uuid = tds.vertex(vk).unwrap().uuid();
        tds.uuid_to_vertex_key.remove(&uuid);

        let err = DelaunayTriangulation::try_from_tds(tds, AdaptiveKernel::new())
            .expect_err("checked TDS reconstruction must reject broken UUID mappings");
        assert_matches!(
            err,
            DelaunayTriangulationValidationError::Tds(source)
                if matches!(source.as_ref(), TdsError::MappingInconsistency { .. })
        );
    }

    #[test]
    fn try_from_tds_rejects_topology_validation_failure() {
        init_tracing();
        let vertices = [
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tds = dt.tds().clone();

        let _ = tds
            .insert_vertex_with_mapping(test_vertex([0.5, 0.5, 0.5]))
            .unwrap();

        let err = DelaunayTriangulation::try_from_tds(tds, AdaptiveKernel::new())
            .expect_err("checked TDS reconstruction must reject isolated vertices");
        assert_matches!(
            err,
            DelaunayTriangulationValidationError::Triangulation(source)
                if matches!(
                    source.as_ref(),
                    TriangulationValidationError::IsolatedVertex { .. }
                )
        );
    }

    #[test]
    fn test_validation_report_ok_for_valid_triangulation() {
        init_tracing();
        let vertices = [
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        assert!(dt.validation_report().is_ok());
    }

    #[test]
    fn test_validation_report_returns_mapping_failures_only() {
        init_tracing();
        let vertices = [
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Break UUID↔key mappings: remove one vertex UUID entry.
        let uuid = dt.tri.tds.vertices().next().unwrap().1.uuid();
        dt.tri.tds.uuid_to_vertex_key.remove(&uuid);

        let report = dt.validation_report().unwrap_err();
        assert!(!report.violations.is_empty());
        assert!(report.violations.iter().all(|v| {
            matches!(
                v.kind,
                InvariantKind::VertexMappings | InvariantKind::SimplexMappings
            )
        }));

        // Early-return on mapping failures: do not add derived invariants.
        assert!(
            report
                .violations
                .iter()
                .all(|v| v.kind != InvariantKind::DelaunayProperty)
        );
    }

    #[test]
    fn test_validation_report_includes_vertex_incidence_violation() {
        init_tracing();
        let vertices = [
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];

        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Corrupt a `Vertex::incident_simplex` pointer.
        let vertex_key = dt.tri.tds.vertices().next().unwrap().0;
        dt.tri
            .tds
            .vertex_mut(vertex_key)
            .unwrap()
            .set_incident_simplex(Some(SimplexKey::default()));

        let report = dt.validation_report().unwrap_err();
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::VertexIncidence)
        );
    }

    #[test]
    fn validation_report_includes_delaunay_after_realization_violations() {
        init_tracing();
        let tds = tds_from_2d_vertices_and_simplices(
            &[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, -1.0]],
            &[vec![0, 1, 2], vec![2, 1, 3], vec![3, 2, 4]],
        );
        let dt = unchecked_test_delaunay_from_tds(tds);

        let report = dt.validation_report().unwrap_err();

        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::Realization),
            "expected realization violation in report: {report:?}"
        );
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.kind == InvariantKind::DelaunayProperty),
            "expected Delaunay violation in report: {report:?}"
        );
    }

    #[test]
    fn test_dt_validate_maps_tds_error_to_tds_variant() {
        init_tracing();
        let vertices = [
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Break vertex mapping so Level 2 structural validation fails.
        let vk = dt.tds().vertex_keys().next().unwrap();
        let uuid = dt.tds().vertex(vk).unwrap().uuid();
        dt.tds_mut_for_repair().uuid_to_vertex_key.remove(&uuid);

        match dt.validate() {
            Err(DelaunayTriangulationValidationError::Tds(source))
                if matches!(source.as_ref(), TdsError::MappingInconsistency { .. }) => {}
            other => panic!(
                "Expected DelaunayTriangulationValidationError::Tds(MappingInconsistency), got {other:?}"
            ),
        }
    }

    #[test]
    fn test_dt_validate_maps_topology_error_to_triangulation_variant() {
        init_tracing();
        let vertices = [
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];
        let mut dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Add an isolated vertex so Level 3 (topology) fails.
        let _ = dt
            .tds_mut_for_repair()
            .insert_vertex_with_mapping(test_vertex([0.5, 0.5, 0.5]))
            .unwrap();

        match dt.validate() {
            Err(DelaunayTriangulationValidationError::Triangulation(source))
                if matches!(
                    source.as_ref(),
                    TriangulationValidationError::IsolatedVertex { .. }
                ) => {}
            other => panic!(
                "Expected DelaunayTriangulationValidationError::Triangulation(IsolatedVertex), got {other:?}"
            ),
        }
    }
}

//! Delaunay-level validation APIs, consuming promotion boundaries, and construction diagnostics.
//!
//! This module owns validation at the [`DelaunayTriangulation`](crate::DelaunayTriangulation)
//! boundary: Level 5 fast-fail checks, first diagnostics, aggregate reports,
//! cumulative validation roll-up, and construction-time candidate promotion. The
//! lower-level empty-circumsphere scan over bare [`Tds`](crate::tds::Tds) storage lives in
//! `property_validation`.

#![forbid(unsafe_code)]

use crate::core::algorithms::flips::{
    DelaunayRepairError, verify_complete_euclidean_tds_via_robust_flip_predicates,
    verify_tds_via_flip_predicates_assuming_connected,
};
use crate::core::collections::ViolationBuffer;
use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::operations::DelaunayInsertionState;
use crate::core::tds::{
    InvariantError, InvariantKind, InvariantViolation, SimplexKey, Tds, TdsError, TopologyOwnerId,
    TriangulationValidationReport,
};
use crate::core::traits::data_type::DataType;
use crate::delaunay_model::{DelaunayTriangulation, EuclideanDelaunayReportDomain};
#[cfg(feature = "diagnostics")]
use crate::delaunay_property_validation::debug_print_first_delaunay_violation as debug_print_first_tds_delaunay_violation;
use crate::delaunay_property_validation::{
    DelaunayValidationError, DelaunayViolationReport,
    delaunay_violation_report as tds_delaunay_violation_report, is_delaunay_property_only,
};
use crate::geometry::kernel::Kernel;
use crate::refinement::RefinementError;
use crate::repair::DelaunayRepairOperation;
use crate::topology::traits::topological_space::GlobalTopology;
use crate::triangulation::Triangulation;
use crate::triangulation::builder::{TriangulationBuildFailure, TriangulationBuilder};
use crate::triangulation::realization::TriangulationRealizationValidationError;
use crate::triangulation::validation::{TopologyGuarantee, TriangulationValidationError};
use std::num::NonZeroUsize;
use thiserror::Error;

/// Level 5 evidence bound to one exact topology owner state.
///
/// The private fields prevent other modules from manufacturing a certificate
/// from a generation counter alone. Construction may retain this token across
/// non-topological bookkeeping, while any owner, generation, or topology-model
/// change makes it inapplicable and forces recertification at publication.
#[derive(Clone, Debug)]
pub(crate) struct DelaunayLevelFiveCertificate<const D: usize> {
    owner_id: TopologyOwnerId,
    generation: u64,
    global_topology: GlobalTopology<D>,
}

impl<const D: usize> DelaunayLevelFiveCertificate<D> {
    /// Returns whether this evidence describes the candidate's exact topology state.
    pub(crate) fn applies_to<K, U, V>(&self, triangulation: &Triangulation<K, U, V, D>) -> bool {
        self.owner_id == triangulation.tds.topology_owner_id()
            && self.generation == triangulation.tds.generation()
            && self.global_topology == triangulation.global_topology
    }
}

/// Runs the construction-time Level 5 predicate certificate and binds its proof
/// to the exact owner state that was checked.
pub(crate) fn certify_level_five_via_flip_predicates<K, U, V, const D: usize>(
    triangulation: &Triangulation<K, U, V, D>,
) -> Result<DelaunayLevelFiveCertificate<D>, DelaunayRepairError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    verify_tds_via_flip_predicates_assuming_connected(
        &triangulation.tds,
        &triangulation.kernel,
        triangulation.global_topology,
    )?;
    Ok(DelaunayLevelFiveCertificate {
        owner_id: triangulation.tds.topology_owner_id(),
        generation: triangulation.tds.generation(),
        global_topology: triangulation.global_topology,
    })
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D>,
{
    /// Builds a detailed Level 5 empty-circumsphere diagnostic without claiming
    /// that this Levels 1–4 owner is Delaunay.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayValidationError`] if a simplex is malformed, a vertex
    /// reference is stale, or an exact predicate cannot be evaluated.
    ///
    /// # Examples
    ///
    /// A Levels 1–4 owner can report a Level 5 violation without claiming the
    /// stronger [`DelaunayTriangulation`] contract:
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder,
    ///     DelaunayTriangulationConstructionError,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![4.0, 0.0]?,
    ///     delaunay::vertex![4.0, 2.0]?,
    ///     delaunay::vertex![1.0, 2.0]?,
    /// ];
    /// let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];
    /// let triangulation =
    ///     DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
    ///         .map_err(DelaunayTriangulationConstructionError::from)?
    ///         .build_triangulation()?;
    ///
    /// let report = triangulation.delaunay_violation_report(None)?;
    /// assert!(!report.is_valid());
    /// # Ok(())
    /// # }
    /// ```
    pub fn delaunay_violation_report(
        &self,
        simplices_to_check: Option<&[SimplexKey]>,
    ) -> Result<DelaunayViolationReport, DelaunayValidationError> {
        tds_delaunay_violation_report(self.tds(), simplices_to_check)
    }

    /// Logs the first Level 5 violation in this generic triangulation.
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
    /// let triangulation = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build_triangulation()?;
    /// triangulation.debug_print_first_delaunay_violation(None);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "diagnostics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "diagnostics")))]
    pub fn debug_print_first_delaunay_violation(&self, simplices_subset: Option<&[SimplexKey]>) {
        debug_print_first_tds_delaunay_violation(self.tds(), simplices_subset);
    }
}

/// Level 5 refinement workspace containing an already proven Levels 1–4 owner.
///
/// This candidate is distinct from incremental draft state: it never contains
/// partial connectivity or an unproven lower-layer owner. Its consuming
/// transition keeps the Level 5 check adjacent to construction of the final
/// [`DelaunayTriangulation`].
#[derive(Clone, Debug)]
pub(crate) struct DelaunayRefinementCandidate<K, U, V, const D: usize> {
    triangulation: Triangulation<K, U, V, D>,
    insertion_state: DelaunayInsertionState,
    spatial_index: Option<HashGridIndex<D>>,
    euclidean_report_domain: EuclideanDelaunayReportDomain,
}

impl<K, U, V, const D: usize> DelaunayRefinementCandidate<K, U, V, D> {
    /// Wraps a Levels 1–4 triangulation for Delaunay-specific certification.
    pub(crate) const fn from_triangulation(triangulation: Triangulation<K, U, V, D>) -> Self {
        Self {
            triangulation,
            insertion_state: DelaunayInsertionState::new(),
            spatial_index: None,
            euclidean_report_domain: EuclideanDelaunayReportDomain::Unproven,
        }
    }

    /// Retains construction-owned caches and provenance across final publication.
    pub(crate) const fn from_parts(
        triangulation: Triangulation<K, U, V, D>,
        insertion_state: DelaunayInsertionState,
        spatial_index: Option<HashGridIndex<D>>,
        euclidean_report_domain: EuclideanDelaunayReportDomain,
    ) -> Self {
        Self {
            triangulation,
            insertion_state,
            spatial_index,
            euclidean_report_domain,
        }
    }
}

impl<K, U, V, const D: usize> DelaunayRefinementCandidate<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Promotes the proof-bearing Levels 1–4 owner by checking only Level 5.
    ///
    /// `Triangulation` already represents the cumulative Levels 1–4 proof.
    /// Revalidating those layers here would weaken the type into an unchecked
    /// data bag and duplicate work at every refinement boundary.
    pub(crate) fn try_into_delaunay(
        self,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationRefinementError<K, U, V, D>>
    {
        if let Err(reason) = self.validate_level_five() {
            return Err(RefinementError::new(self.triangulation, reason));
        }

        Ok(self.into_delaunay_after_level_five_check())
    }

    /// Publishes using retained Level 5 evidence when it still matches exactly.
    ///
    /// Missing or stale evidence falls back to the ordinary checked transition,
    /// so this terminal cannot publish an owner from a detached generation token.
    pub(crate) fn try_into_delaunay_with_certificate(
        self,
        certificate: Option<&DelaunayLevelFiveCertificate<D>>,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationRefinementError<K, U, V, D>>
    {
        if certificate.is_some_and(|certificate| certificate.applies_to(&self.triangulation)) {
            return Ok(self.into_delaunay_after_level_five_check());
        }
        self.try_into_delaunay()
    }

    /// Checks only the Level 5 predicate while retaining candidate ownership.
    pub(crate) fn validate_level_five(&self) -> Result<(), DelaunayTriangulationValidationError> {
        if self.euclidean_report_domain.supports_local_certificate()
            && self.triangulation.global_topology().is_euclidean()
        {
            return verify_tds_via_flip_predicates_assuming_connected(
                &self.triangulation.tds,
                &self.triangulation.kernel,
                self.triangulation.global_topology,
            )
            .map_err(|source| {
                DelaunayTriangulationValidationError::VerificationFailed {
                    source: Box::new(DelaunayVerificationError::from(source)),
                }
            });
        }
        validate_level_five_for_refinement(&self.triangulation)
    }

    /// Publishes a candidate immediately after its attached Level 5 check succeeds.
    ///
    /// This trusted step is kept on the private candidate and must remain
    /// adjacent to the successful check or a transaction commit that performed
    /// the same check against the candidate's exact triangulation state.
    pub(crate) fn into_delaunay_after_level_five_check(self) -> DelaunayTriangulation<K, U, V, D> {
        DelaunayTriangulation {
            tri: self.triangulation,
            insertion_state: self.insertion_state,
            spatial_index: self.spatial_index,
            euclidean_report_domain: self.euclidean_report_domain,
        }
    }
}

/// Certifies only Level 5 for a proof-bearing Levels 1–4 owner.
///
/// Delaunayize uses this borrowed form before committing its outer rollback
/// transaction, ensuring that a failed final check can still restore and return
/// the original triangulation.
pub(crate) fn validate_level_five_for_refinement<K, U, V, const D: usize>(
    triangulation: &Triangulation<K, U, V, D>,
) -> Result<(), DelaunayTriangulationValidationError>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    if triangulation.global_topology().is_euclidean() {
        is_delaunay_property_only(&triangulation.tds).map_err(|source| {
            DelaunayTriangulationValidationError::VerificationFailed {
                source: Box::new(DelaunayVerificationError::from(source)),
            }
        })?;
    } else {
        verify_tds_via_flip_predicates_assuming_connected(
            &triangulation.tds,
            &triangulation.kernel,
            triangulation.global_topology,
        )
        .map_err(
            |source| DelaunayTriangulationValidationError::VerificationFailed {
                source: Box::new(DelaunayVerificationError::from(source)),
            },
        )?;
    }

    Ok(())
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
    Tds {
        /// Boxed TDS source error.
        source: Box<TdsError>,
    },

    /// Lower-layer topology validation error (Level 3).
    #[error(transparent)]
    Triangulation {
        /// Boxed topology-validation source error.
        source: Box<TriangulationValidationError>,
    },

    /// Lower-layer realized-geometry validation error (Level 4).
    #[error(transparent)]
    Realization {
        /// Boxed realization-validation source error.
        source: Box<TriangulationRealizationValidationError>,
    },

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
        Self::Tds {
            source: Box::new(source),
        }
    }
}

impl From<TriangulationValidationError> for DelaunayTriangulationValidationError {
    fn from(source: TriangulationValidationError) -> Self {
        Self::Triangulation {
            source: Box::new(source),
        }
    }
}

impl From<TriangulationRealizationValidationError> for DelaunayTriangulationValidationError {
    fn from(source: TriangulationRealizationValidationError) -> Self {
        match source {
            TriangulationRealizationValidationError::Tds { source } => Self::Tds { source },
            TriangulationRealizationValidationError::Triangulation { source } => {
                Self::Triangulation { source }
            }
            source => Self::Realization {
                source: Box::new(source),
            },
        }
    }
}

/// Recoverable failure to refine a Levels 1–4 [`Triangulation`] through Level 5.
pub type DelaunayTriangulationRefinementError<K, U, V, const D: usize> =
    RefinementError<Triangulation<K, U, V, D>, DelaunayTriangulationValidationError>;

/// Stage-specific failure while composing TDS restoration through Level 5.
///
/// The variant identifies the strongest proof-bearing owner reached before
/// rejection. A Levels 3–4 failure retains the original [`Tds`]; a Level 5
/// failure retains the successfully promoted [`Triangulation`].
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub(crate) enum DelaunayTdsRestorationError<K, U, V, const D: usize> {
    /// Levels 3–4 certification failed, leaving the Levels 1–2 owner intact.
    #[error("TDS-to-triangulation refinement failed: {failure}")]
    Triangulation {
        /// Recoverable TDS and typed Levels 3–4 diagnostic.
        #[source]
        failure: TriangulationBuildFailure<U, V, D>,
    },
    /// Level 5 certification failed after Levels 3–4 succeeded.
    #[error("triangulation-to-Delaunay refinement failed: {failure}")]
    Delaunay {
        /// Recoverable triangulation and typed Level 5 diagnostic.
        #[source]
        failure: DelaunayTriangulationRefinementError<K, U, V, D>,
    },
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
    /// Euclidean triangulations produced by complete point-set insertion first
    /// use robust local flip predicates as an O(simplices) validity certificate.
    /// A successful certificate avoids the all-vertices empty-circumsphere scan.
    /// Explicit, reconstructed, or otherwise unproven connectivity delegates
    /// directly to the global scan; a failed or inconclusive local certificate
    /// also falls back so invalid reports retain every violating simplex and the
    /// same typed details. Non-Euclidean topologies use the topology-aware flip
    /// verifier and report the first violation it finds, avoiding an ordinary
    /// Euclidean circumsphere scan in periodic charts.
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
            return match self.delaunay_violation_report(None) {
                Ok(report) if report.is_valid() => Ok(()),
                Ok(report) => Err(TriangulationValidationReport {
                    violations: report
                        .violation_details
                        .into_iter()
                        .map(|detail| InvariantViolation {
                            kind: InvariantKind::DelaunayProperty,
                            error: InvariantError::Delaunay {
                                source: DelaunayTriangulationValidationError::VerificationFailed {
                                    source: Box::new(DelaunayVerificationError::from(
                                        DelaunayValidationError::from(detail),
                                    )),
                                },
                            },
                        })
                        .collect(),
                }),
                Err(source) => Err(TriangulationValidationReport {
                    violations: vec![InvariantViolation {
                        kind: InvariantKind::DelaunayProperty,
                        error: InvariantError::Delaunay {
                            source: DelaunayTriangulationValidationError::VerificationFailed {
                                source: Box::new(DelaunayVerificationError::from(source)),
                            },
                        },
                    }],
                }),
            };
        }

        self.is_valid_delaunay()
            .map_err(|error| TriangulationValidationReport {
                violations: vec![InvariantViolation {
                    kind: InvariantKind::DelaunayProperty,
                    error: InvariantError::Delaunay { source: error },
                }],
            })
    }

    /// Builds a detailed Delaunay empty-circumsphere violation report.
    ///
    /// This is the high-level owner-bound counterpart to the TDS-level
    /// [`delaunay_violation_report`](crate::delaunay_violation_report) helper.
    /// For a full Euclidean report over connectivity proven to triangulate the
    /// complete point set, it first uses O(simplices) robust local flip predicates
    /// as a validity certificate. Explicit, reconstructed, and other unproven
    /// connectivity delegates directly to the TDS-level brute-force scan. A
    /// failed or inconclusive certificate also delegates so the returned
    /// violation set and typed diagnostics remain exact. Subset reports delegate
    /// directly because a global invalidity outside the subset says nothing
    /// about the requested simplices.
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
        if simplices_to_check.is_none()
            && self.global_topology().is_euclidean()
            && self.euclidean_report_domain.supports_local_certificate()
            && verify_complete_euclidean_tds_via_robust_flip_predicates(&self.tri.tds).is_ok()
        {
            return Ok(DelaunayViolationReport {
                number_of_vertices: self.number_of_vertices(),
                number_of_simplices: self.number_of_simplices(),
                checked_simplices: self.number_of_simplices(),
                violating_simplices: ViolationBuffer::new(),
                violation_details: Vec::new(),
            });
        }

        tds_delaunay_violation_report(&self.tri.tds, simplices_to_check)
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
        debug_print_first_tds_delaunay_violation(&self.tri.tds, simplices_subset);
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
        verify_tds_via_flip_predicates_assuming_connected(
            &self.tri.tds,
            &self.tri.kernel,
            self.tri.global_topology,
        )
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
                                    error: InvariantError::Realization { source: error },
                                }
                            }));
                    }
                    Err(source) => {
                        report.violations.push(InvariantViolation {
                            kind: InvariantKind::Realization,
                            error: InvariantError::Realization { source },
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
    /// Create a validated `DelaunayTriangulation` from a `Tds` with explicit topology context.
    ///
    /// This is the strict promotion path for validated serialized TDS data whose
    /// runtime [`TopologyGuarantee`] or [`GlobalTopology`] metadata must be
    /// restored. TDS deserialization supplies Levels 1–2; internally this
    /// composes the same strict Levels 3–4 certification used by
    /// [`TriangulationBuilder`] with an internal
    /// Level 5-only promotion. The result therefore carries
    /// the cumulative Levels 1–5 guarantee without rechecking an unchanged lower
    /// proof between boundaries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, GlobalTopology,
    ///     TopologyGuarantee,
    /// };
    /// use delaunay::prelude::geometry::FastKernel;
    /// use delaunay::prelude::triangulation::TriangulationBuilder;
    /// use delaunay::DelaunayRefinementBuilder;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let tds = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build_triangulation()?
    ///     .into_tds();
    /// let triangulation = TriangulationBuilder::new(tds, FastKernel::new())
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .global_topology(GlobalTopology::Euclidean)
    ///     .strict()
    ///     .build()
    ///     .expect("Levels 1-2 storage should satisfy Levels 3-4");
    /// let reconstructed = DelaunayRefinementBuilder::new(triangulation)
    ///     .build()
    ///     .expect("fixture should satisfy Level 5");
    ///
    /// assert_eq!(
    ///     reconstructed.topology_guarantee(),
    ///     TopologyGuarantee::PLManifold
    /// );
    /// assert_eq!(reconstructed.global_topology(), GlobalTopology::Euclidean);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayTdsRestorationError`] if construction is incomplete or
    /// if topology, realized geometry, or the Delaunay predicate fails under
    /// the supplied context. The error retains the strongest lower-layer proof
    /// owner established by the composed transition.
    pub(crate) fn try_restore_from_tds_with_topology_context(
        tds: Tds<U, V, D>,
        kernel: K,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Result<Self, DelaunayTdsRestorationError<K, U, V, D>> {
        let triangulation = TriangulationBuilder::new(tds, kernel)
            .topology_guarantee(topology_guarantee)
            .global_topology(global_topology)
            .strict()
            .build()
            .map_err(|failure| DelaunayTdsRestorationError::Triangulation { failure })?;
        DelaunayRefinementCandidate::from_triangulation(triangulation)
            .try_into_delaunay()
            .map_err(|failure| DelaunayTdsRestorationError::Delaunay { failure })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::DelaunayTriangulationBuilder;
    use crate::core::algorithms::flips::{
        DelaunayRepairDiagnostics, DelaunayRepairPostconditionFailure, RepairQueueOrder,
    };
    use crate::core::tds::{SimplexKey, TdsBuilder, VertexKey};
    use crate::core::vertex::Vertex;
    use crate::geometry::coordinate_range::CoordinateRange;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::CoordinateConversionError;
    use crate::geometry::util::generate_random_points_in_range_seeded;
    use crate::triangulation::builder::TriangulationBuilderError;
    use crate::vertex;
    use slotmap::KeyData;
    use std::assert_matches;
    use std::{error::Error, sync::Once};
    use uuid::Uuid;

    impl<K, U, V, const D: usize> DelaunayRefinementCandidate<K, U, V, D> {
        /// Builds an intentionally unproven test candidate from raw parts.
        pub(crate) const fn assemble_unchecked_for_test(
            tds: Tds<U, V, D>,
            kernel: K,
            topology_guarantee: TopologyGuarantee,
            global_topology: GlobalTopology<D>,
        ) -> Self {
            Self {
                triangulation: Triangulation {
                    kernel,
                    tds,
                    global_topology,
                    validation_policy: topology_guarantee.default_validation_policy(),
                    topology_guarantee,
                },
                insertion_state: DelaunayInsertionState::new(),
                spatial_index: None,
                euclidean_report_domain: EuclideanDelaunayReportDomain::Unproven,
            }
        }

        /// Deliberately bypasses Level 5 promotion for validation and internal
        /// repair failure tests.
        pub(crate) fn into_unproven_delaunay_for_test(self) -> DelaunayTriangulation<K, U, V, D> {
            DelaunayTriangulation {
                tri: self.triangulation,
                insertion_state: self.insertion_state,
                spatial_index: self.spatial_index,
                euclidean_report_domain: self.euclidean_report_domain,
            }
        }
    }

    #[derive(Clone, Debug)]
    struct PanickingKernel;

    impl<const D: usize> Kernel<D> for PanickingKernel {
        type Scalar = f64;

        fn orientation(&self, _points: &[Point<D>]) -> Result<i32, CoordinateConversionError> {
            panic!("the structured report must not call its owner's kernel")
        }

        fn in_sphere(
            &self,
            _simplex_points: &[Point<D>],
            _test_point: &Point<D>,
        ) -> Result<i32, CoordinateConversionError> {
            panic!("the structured report must not call its owner's kernel")
        }
    }

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
        let vertices = [
            test_vertex([0.0, 0.0]),
            test_vertex([4.0, 0.0]),
            test_vertex([4.0, 2.0]),
            test_vertex([1.0, 2.0]),
        ];
        let simplices = [vec![0, 1, 2], vec![0, 2, 3]];
        TdsBuilder::new(&vertices, &simplices).build().unwrap()
    }

    fn tds_from_2d_vertices_and_simplices(
        coords: &[[f64; 2]],
        simplices: &[Vec<usize>],
    ) -> Tds<(), (), 2> {
        let vertices: Vec<_> = coords.iter().copied().map(test_vertex).collect();
        TdsBuilder::new(&vertices, simplices).build().unwrap()
    }

    fn unchecked_test_delaunay_from_tds<const D: usize>(
        tds: Tds<(), (), D>,
    ) -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> {
        DelaunayRefinementCandidate::assemble_unchecked_for_test(
            tds,
            AdaptiveKernel::new(),
            TopologyGuarantee::Pseudomanifold,
            GlobalTopology::Euclidean,
        )
        .into_unproven_delaunay_for_test()
    }

    fn synthetic_flip_verification_source(message: &str) -> DelaunayVerificationError {
        let _ = message;
        DelaunayVerificationError::from(DelaunayRepairError::PostconditionFailed {
            reason: Box::new(DelaunayRepairPostconditionFailure::Disconnected { simplex_count: 1 }),
        })
    }

    fn randomized_delaunay<const D: usize>(
        base_seed: u64,
    ) -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> {
        let bounds = CoordinateRange::try_new(-10.0, 10.0).unwrap();
        for offset in 0..32 {
            let points = generate_random_points_in_range_seeded::<D>(
                D + 8,
                bounds,
                base_seed.wrapping_add(offset),
            )
            .unwrap();
            let vertices: Vec<_> = points
                .iter()
                .map(|point| test_vertex(*point.coords()))
                .collect();
            if let Ok(triangulation) = DelaunayTriangulationBuilder::new(&vertices).build() {
                return triangulation;
            }
        }

        panic!("failed to build seeded randomized {D}D Delaunay test fixture");
    }

    fn shared_facet_flip_adversary<const D: usize>() -> Triangulation<AdaptiveKernel<f64>, (), (), D>
    {
        let dim = u32::try_from(D).unwrap();
        let high_apex_coordinate = 1.1 / f64::from(dim);
        let mut vertices = Vec::with_capacity(D + 2);
        vertices.push(test_vertex([0.0; D]));
        for axis in 0..D {
            let mut coordinates = [0.0; D];
            coordinates[axis] = 1.0;
            vertices.push(test_vertex(coordinates));
        }
        vertices.push(test_vertex([high_apex_coordinate; D]));

        let low_simplex = (0..=D).collect();
        let mut high_simplex: Vec<usize> = (1..=D).collect();
        high_simplex.push(D + 1);
        let simplices = vec![low_simplex, high_simplex];

        DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
            .unwrap()
            .build_triangulation()
            .unwrap()
    }

    fn assert_randomized_full_report_matches_brute_force<const D: usize>() {
        for seed in [0x483, 0x483_0001, 0x483_0002] {
            let triangulation = randomized_delaunay::<D>(seed);
            assert!(triangulation.verify_via_flip_predicates().is_ok());

            let optimized = triangulation.delaunay_violation_report(None).unwrap();
            let tds = triangulation.into_triangulation().into_tds();
            let brute_force = tds_delaunay_violation_report(&tds, None).unwrap();
            assert_eq!(optimized, brute_force);
        }
    }

    fn assert_adversarial_full_report_matches_brute_force<const D: usize>() {
        let triangulation = shared_facet_flip_adversary::<D>();
        assert!(
            triangulation
                .delaunay_violation_report(None)
                .is_ok_and(|report| !report.is_valid())
        );

        let optimized = triangulation.delaunay_violation_report(None).unwrap();
        let brute_force = tds_delaunay_violation_report(triangulation.tds(), None).unwrap();
        assert_eq!(optimized, brute_force);
    }

    macro_rules! generate_full_report_agreement_tests {
        ($dimension:literal) => {
            pastey::paste! {
                #[test]
                fn [<randomized_full_report_matches_brute_force_ $dimension d>]() {
                    assert_randomized_full_report_matches_brute_force::<$dimension>();
                }

                #[test]
                fn [<adversarial_full_report_matches_brute_force_ $dimension d>]() {
                    assert_adversarial_full_report_matches_brute_force::<$dimension>();
                }
            }
        };
    }

    generate_full_report_agreement_tests!(2);
    generate_full_report_agreement_tests!(3);
    generate_full_report_agreement_tests!(4);
    generate_full_report_agreement_tests!(5);

    #[test]
    fn complete_point_set_report_uses_robust_predicates_instead_of_owner_kernel() {
        let source = randomized_delaunay::<2>(0x483_1001);
        assert_eq!(
            source.euclidean_report_domain,
            EuclideanDelaunayReportDomain::CompletePointSet
        );
        let topology_guarantee = source.topology_guarantee();
        let global_topology = source.global_topology();
        let source_tds = source.into_triangulation().into_tds();

        let mut triangulation = DelaunayRefinementCandidate::assemble_unchecked_for_test(
            source_tds,
            PanickingKernel,
            topology_guarantee,
            global_topology,
        )
        .into_unproven_delaunay_for_test();
        triangulation.euclidean_report_domain = EuclideanDelaunayReportDomain::CompletePointSet;

        let report = triangulation.delaunay_violation_report(None).unwrap();
        let tds = triangulation.into_triangulation().into_tds();
        let brute_force = tds_delaunay_violation_report(&tds, None).unwrap();
        assert_eq!(report, brute_force);
    }

    #[test]
    fn failed_complete_point_set_certificate_falls_back_to_global_report() {
        let triangulation = shared_facet_flip_adversary::<2>();
        let mut triangulation = DelaunayRefinementCandidate::from_triangulation(triangulation)
            .into_unproven_delaunay_for_test();
        triangulation.euclidean_report_domain = EuclideanDelaunayReportDomain::CompletePointSet;

        let report = triangulation.delaunay_violation_report(None).unwrap();
        let tds = triangulation.into_triangulation().into_tds();
        let brute_force = tds_delaunay_violation_report(&tds, None).unwrap();
        assert!(!report.is_valid());
        assert_eq!(report, brute_force);
    }

    #[test]
    fn unproven_connectivity_bypasses_local_certificate() {
        let tds = tds_from_2d_vertices_and_simplices(
            &[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.25, 0.25]],
            &[vec![0, 1, 2]],
        );
        assert!(verify_complete_euclidean_tds_via_robust_flip_predicates(&tds).is_ok());

        let triangulation = DelaunayRefinementCandidate::assemble_unchecked_for_test(
            tds,
            PanickingKernel,
            TopologyGuarantee::Pseudomanifold,
            GlobalTopology::Euclidean,
        )
        .into_unproven_delaunay_for_test();
        assert_eq!(
            triangulation.euclidean_report_domain,
            EuclideanDelaunayReportDomain::Unproven
        );

        let report = triangulation.delaunay_violation_report(None).unwrap();
        let tds = triangulation.into_triangulation().into_tds();
        let brute_force = tds_delaunay_violation_report(&tds, None).unwrap();
        assert!(!report.is_valid());
        assert_eq!(report, brute_force);
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

        let err = DelaunayTriangulation::try_restore_from_tds_with_topology_context(
            tds,
            AdaptiveKernel::new(),
            TopologyGuarantee::DEFAULT,
            GlobalTopology::DEFAULT,
        )
        .expect_err("checked TDS reconstruction must reject non-Delaunay connectivity");

        let failure = match err {
            DelaunayTdsRestorationError::Delaunay { failure } => failure,
            other => panic!("expected Level 5 refinement failure, got {other:?}"),
        };
        let (triangulation, reason) = failure.into_parts();
        assert!(matches!(
            reason,
            DelaunayTriangulationValidationError::VerificationFailed { .. }
        ));
        triangulation
            .validate_realization()
            .expect("failed Level 5 refinement must return the valid Levels 1-4 owner");
    }

    #[test]
    fn tds_audit_rejects_structural_corruption_before_proof_consumption() {
        init_tracing();
        let vertices = [
            test_vertex([0.0, 0.0, 0.0]),
            test_vertex([1.0, 0.0, 0.0]),
            test_vertex([0.0, 1.0, 0.0]),
            test_vertex([0.0, 0.0, 1.0]),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let mut tds = dt.into_triangulation().into_tds();

        let vk = tds.vertex_keys().next().unwrap();
        let uuid = tds.vertex(vk).unwrap().uuid();
        tds.remove_vertex_uuid_mapping_for_test(&uuid);

        let err = tds
            .validate()
            .expect_err("the Levels 1–2 owner audit must reject broken UUID mappings");
        assert_matches!(err, TdsError::MappingInconsistency { .. });
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
        let mut tds = dt.into_triangulation().into_tds();

        let _ = tds
            .insert_vertex_with_mapping(test_vertex([0.5, 0.5, 0.5]))
            .unwrap();

        let err = DelaunayTriangulation::try_restore_from_tds_with_topology_context(
            tds,
            AdaptiveKernel::new(),
            TopologyGuarantee::DEFAULT,
            GlobalTopology::DEFAULT,
        )
        .expect_err("checked TDS reconstruction must reject isolated vertices");
        let failure = match err {
            DelaunayTdsRestorationError::Triangulation { failure } => failure,
            other => panic!("expected Levels 3-4 refinement failure, got {other:?}"),
        };
        let (tds, reason) = failure.into_parts();
        assert_matches!(
            reason,
            TriangulationBuilderError::TopologyValidation { source }
                if matches!(
                    source.as_ref(),
                    InvariantError::Triangulation {
                        source: TriangulationValidationError::IsolatedVertex { .. }
                    }
                )
        );
        tds.validate()
            .expect("failed Levels 3-4 refinement must return the valid Levels 1-2 owner");
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
        dt.tri.tds.remove_vertex_uuid_mapping_for_test(&uuid);

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
            .vertex_mut_for_test(vertex_key)
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
        let vk = dt.vertices().next().unwrap().0;
        let uuid = dt.vertex(vk).unwrap().uuid();
        dt.tds_mut_for_repair()
            .remove_vertex_uuid_mapping_for_test(&uuid);

        match dt.validate() {
            Err(DelaunayTriangulationValidationError::Tds { source })
                if matches!(source.as_ref(), TdsError::MappingInconsistency { .. }) => {}
            other => panic!(
                "Expected DelaunayTriangulationValidationError::Tds with MappingInconsistency source, got {other:?}"
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
            Err(DelaunayTriangulationValidationError::Triangulation { source })
                if matches!(
                    source.as_ref(),
                    TriangulationValidationError::IsolatedVertex { .. }
                ) => {}
            other => panic!(
                "Expected DelaunayTriangulationValidationError::Triangulation with IsolatedVertex source, got {other:?}"
            ),
        }
    }
}

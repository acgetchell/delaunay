//! Fluent promotion from Levels 1–2 storage to a Levels 1–4 triangulation.

#![forbid(unsafe_code)]

use crate::core::algorithms::insertion::InsertionError;
use crate::core::tds::{InvariantError, Tds, TdsError};
use crate::core::traits::data_type::DataType;
use crate::geometry::kernel::Kernel;
use crate::refinement::RefinementError;
use crate::topology::traits::topological_space::GlobalTopology;
use crate::triangulation::Triangulation;
use crate::triangulation::draft::TriangulationDraft;
use crate::triangulation::realization::TriangulationRealizationValidationError;
use crate::triangulation::validation::{
    TopologyCertificationEvidence, TopologyConstructionProvenance, TopologyGuarantee,
    ValidationConfigurationError, ValidationPolicy,
};
use thiserror::Error;

/// Selects whether [`TriangulationBuilder`] may transform TDS storage.
///
/// The mode is explicit because it changes both successful representation and
/// construction cost. Both modes run the same final Levels 3–4 certification.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub(super) enum TriangulationBuildMode {
    /// Require the supplied TDS representation to satisfy Levels 3–4 unchanged.
    ///
    /// This mode performs no orientation normalization and allocates no rollback
    /// snapshot. Failure returns the unchanged TDS.
    #[default]
    Strict,
    /// Normalize orientation transactionally before Levels 3–4 certification.
    ///
    /// This explicit opt-in may change simplex ordering and structural
    /// generation on successful publication.
    Canonicalizing,
}

/// Typed failures while promoting a TDS through Levels 3–4.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TriangulationBuilderError {
    /// The requested topology guarantee and validation policy are incompatible.
    #[error("triangulation builder configuration is invalid: {source}")]
    ValidationConfiguration {
        /// Typed validation-policy configuration failure.
        #[source]
        source: ValidationConfigurationError,
    },
    /// Positive geometric orientation could not be established.
    #[error("orientation normalization failed during triangulation construction: {source}")]
    OrientationNormalization {
        /// Typed orientation transition failure.
        #[source]
        source: Box<InsertionError>,
    },
    /// Cumulative Levels 1–2 validation failed before higher-layer promotion.
    #[error("structural validation failed during triangulation construction: {source}")]
    StructuralValidation {
        /// Typed TDS validation failure.
        #[source]
        source: Box<TdsError>,
    },
    /// The requested Level 3 topology guarantee could not be proved.
    #[error("topology validation failed during triangulation construction: {source}")]
    TopologyValidation {
        /// Typed cumulative invariant failure.
        #[source]
        source: Box<InvariantError>,
    },
    /// At least one maximal simplex is geometrically degenerate.
    #[error("geometric nondegeneracy failed during triangulation construction: {source}")]
    GeometricNondegeneracy {
        /// Typed geometric validation failure.
        #[source]
        source: Box<TdsError>,
    },
    /// The complex is not a valid realization in its selected topology model.
    #[error("realization validation failed during triangulation construction: {source}")]
    RealizationValidation {
        /// Typed Level 4 validation failure.
        #[source]
        source: Box<TriangulationRealizationValidationError>,
    },
}

/// Recoverable failure to build a [`Triangulation`] from a Levels 1-2 [`Tds`].
///
/// The error retains the original TDS exactly as it was supplied to the
/// builder. Canonicalization may mutate a private candidate while publication
/// is attempted, but any failure restores owner identity, generation, and
/// storage before returning it for repair or retry.
pub type TriangulationBuildFailure<U, V, const D: usize> =
    RefinementError<Tds<U, V, D>, TriangulationBuilderError>;

/// Internal build result that retains metrics from the successful Level-3 pass.
pub(super) type TriangulationBuildWithTopologyEvidence<K, U, V, const D: usize> = Result<
    (Triangulation<K, U, V, D>, TopologyCertificationEvidence),
    TriangulationBuildFailure<U, V, D>,
>;

/// Fluent builder for a proof-bearing Levels 1–4 [`Triangulation`].
///
/// The builder consumes TDS storage, installs kernel and topology context, and
/// publishes a triangulation only after Levels 3–4 succeed. The default strict
/// mode certifies without transforming the supplied TDS. Select
/// [`Self::canonicalizing`] when publication may normalize positive geometric
/// orientation. Neither mode infers connectivity or repairs arbitrary topology
/// or realization failures.
///
/// # Transformation and cost
///
/// Strict construction performs no orientation mutation and allocates no
/// rollback snapshot. Explicit canonicalizing construction may change stored
/// simplex ordering and advance the TDS structural generation. It snapshots
/// the canonical TDS before normalization, adding storage and copy work linear
/// in the TDS representation. Either mode returns the exact input TDS in
/// [`TriangulationBuildFailure`] when publication fails.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateConversionError};
/// use delaunay::prelude::tds::{TdsBuilder, TdsBuilderError};
/// use delaunay::prelude::triangulation::{
///     TriangulationBuildFailure, TriangulationBuilder,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Coordinate(#[from] CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Tds(#[from] TdsBuilderError),
/// #     #[error(transparent)]
/// #     Triangulation(#[from] TriangulationBuildFailure<(), (), 2>),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = [
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0]?,
/// ];
/// let simplices = [vec![0, 1, 2]];
/// let tds = TdsBuilder::new(&vertices, &simplices).build()?;
///
/// let triangulation = TriangulationBuilder::new(tds, AdaptiveKernel::new()).build()?;
/// assert!(triangulation.validate_realization().is_ok());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct TriangulationBuilder<K, U, V, const D: usize> {
    tds: Tds<U, V, D>,
    kernel: K,
    topology_guarantee: TopologyGuarantee,
    global_topology: GlobalTopology<D>,
    validation_policy: Option<ValidationPolicy>,
    build_mode: TriangulationBuildMode,
    construction_provenance: TopologyConstructionProvenance,
}

impl<K, U, V, const D: usize> TriangulationBuilder<K, U, V, D> {
    /// Creates an inert Levels 3–4 promotion request from TDS storage and a kernel.
    ///
    /// Validation is deferred to [`build`](Self::build). Builder setters only
    /// record context and cannot publish an invalid domain value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::FastKernel;
    /// use delaunay::prelude::tds::Tds;
    /// use delaunay::prelude::triangulation::TriangulationBuilder;
    ///
    /// let tds: Tds<(), (), 2> = Tds::empty();
    /// let builder = TriangulationBuilder::new(tds, FastKernel::<f64>::new());
    /// assert_eq!(
    ///     builder.selected_topology_guarantee(),
    ///     delaunay::TopologyGuarantee::DEFAULT
    /// );
    /// ```
    #[must_use]
    pub const fn new(tds: Tds<U, V, D>, kernel: K) -> Self {
        Self {
            tds,
            kernel,
            topology_guarantee: TopologyGuarantee::DEFAULT,
            global_topology: GlobalTopology::DEFAULT,
            validation_policy: None,
            build_mode: TriangulationBuildMode::Strict,
            construction_provenance: TopologyConstructionProvenance::Unproven,
        }
    }

    /// Selects the Level 3 topology guarantee proved by [`build`](Self::build).
    #[must_use]
    pub const fn topology_guarantee(mut self, topology_guarantee: TopologyGuarantee) -> Self {
        self.topology_guarantee = topology_guarantee;
        self
    }

    /// Selects the global topology context used by Levels 3–4 validation.
    #[must_use]
    pub const fn global_topology(mut self, global_topology: GlobalTopology<D>) -> Self {
        self.global_topology = global_topology;
        self
    }

    /// Selects the audit cadence retained by the published triangulation.
    ///
    /// Compatibility with the selected [`TopologyGuarantee`] is checked by
    /// [`build`](Self::build), keeping this configuration step infallible.
    #[must_use]
    pub const fn validation_policy(mut self, validation_policy: ValidationPolicy) -> Self {
        self.validation_policy = Some(validation_policy);
        self
    }

    /// Selects transactional positive-orientation canonicalization before certification.
    #[must_use]
    pub const fn canonicalizing(mut self) -> Self {
        self.build_mode = TriangulationBuildMode::Canonicalizing;
        self
    }

    /// Attaches proof evidence from a crate-owned topology construction.
    #[must_use]
    pub(crate) const fn construction_provenance(
        mut self,
        provenance: TopologyConstructionProvenance,
    ) -> Self {
        self.construction_provenance = provenance;
        self
    }

    /// Returns the currently selected topology guarantee.
    #[must_use]
    pub const fn selected_topology_guarantee(&self) -> TopologyGuarantee {
        self.topology_guarantee
    }
}

impl<K, U, V, const D: usize> TriangulationBuilder<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Promotes the configured TDS into a proof-bearing Levels 1–4 triangulation.
    ///
    /// # Errors
    ///
    /// Returns [`TriangulationBuildFailure`] if configuration is incompatible,
    /// orientation cannot be normalized, or any cumulative validation layer
    /// through Level 4 rejects the candidate. The failure retains the original
    /// TDS so the caller can repair it or retry with different options. Success
    /// preserves the supplied TDS representation by default. Select
    /// [`Self::canonicalizing`] when success may normalize its orientation.
    pub fn build(self) -> Result<Triangulation<K, U, V, D>, TriangulationBuildFailure<U, V, D>> {
        self.build_with_topology_evidence()
            .map(|(triangulation, _evidence)| triangulation)
    }

    /// Publishes Levels 3–4 while retaining metrics from that exact proof pass.
    pub(crate) fn build_with_topology_evidence(
        self,
    ) -> TriangulationBuildWithTopologyEvidence<K, U, V, D> {
        let validation_policy = self
            .validation_policy
            .unwrap_or_else(|| self.topology_guarantee.default_validation_policy());

        TriangulationDraft::with_topology_context(
            self.tds,
            self.kernel,
            self.topology_guarantee,
            self.global_topology,
        )
        .construction_provenance(self.construction_provenance)
        .validation_policy(validation_policy)
        .finish_with_topology_evidence(self.build_mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::TdsBuilder;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::vertex;
    use std::assert_matches;

    #[test]
    fn build_mode_defaults_to_strict_certification() {
        assert_eq!(
            TriangulationBuildMode::default(),
            TriangulationBuildMode::Strict
        );
    }

    #[test]
    fn build_publishes_only_a_valid_realization() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let simplices = [vec![0, 1, 2]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();

        let triangulation = TriangulationBuilder::new(tds, AdaptiveKernel::new())
            .build()
            .unwrap();

        assert!(triangulation.validate_realization().is_ok());
    }

    #[test]
    fn retained_topology_evidence_is_bound_to_the_published_owner() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let simplices = [vec![0, 1, 2]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();

        let (triangulation, evidence) = TriangulationBuilder::new(tds, AdaptiveKernel::new())
            .build_with_topology_evidence()
            .unwrap();

        assert_eq!(
            evidence
                .simplex_counts(triangulation.tds())
                .map(|counts| counts.by_dim.as_slice()),
            Some([3, 3, 1].as_slice())
        );
        assert_eq!(evidence.euler_characteristic(triangulation.tds()), Some(1));

        let cloned_tds = triangulation.tds().clone();
        assert!(evidence.simplex_counts(&cloned_tds).is_none());
        assert!(evidence.euler_characteristic(&cloned_tds).is_none());

        let mut published_tds = triangulation.into_tds();
        assert!(evidence.simplex_counts(&published_tds).is_some());
        published_tds
            .insert_vertex_with_mapping(vertex![0.25, 0.25].unwrap())
            .unwrap();
        assert!(evidence.simplex_counts(&published_tds).is_none());
        assert!(evidence.euler_characteristic(&published_tds).is_none());
    }

    #[test]
    fn strict_build_preserves_valid_tds_representation() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let simplices = [vec![0, 1, 2]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();
        let expected_snapshot = serde_json::to_value(&tds).unwrap();
        let expected_owner = tds.topology_owner_id();
        let expected_generation = tds.generation();

        let triangulation = TriangulationBuilder::new(tds, AdaptiveKernel::new())
            .build()
            .unwrap();
        let restored = triangulation.into_tds();

        assert_eq!(restored.topology_owner_id(), expected_owner);
        assert_eq!(restored.generation(), expected_generation);
        assert_eq!(serde_json::to_value(&restored).unwrap(), expected_snapshot);
    }

    #[test]
    fn strict_build_rejects_orientation_that_canonicalizing_build_accepts() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let simplices = [vec![0, 2, 1]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();
        let strict_tds = tds.clone();
        let expected_snapshot = serde_json::to_value(&strict_tds).unwrap();
        let expected_owner = strict_tds.topology_owner_id();
        let expected_generation = strict_tds.generation();

        let failure = TriangulationBuilder::new(strict_tds, AdaptiveKernel::new())
            .build()
            .expect_err("strict publication must not rewrite negative orientation");
        assert_matches!(
            failure.reason(),
            TriangulationBuilderError::RealizationValidation { .. }
        );
        assert_eq!(failure.owner().topology_owner_id(), expected_owner);
        assert_eq!(failure.owner().generation(), expected_generation);
        assert_eq!(
            serde_json::to_value(failure.owner()).unwrap(),
            expected_snapshot
        );

        TriangulationBuilder::new(tds, AdaptiveKernel::new())
            .canonicalizing()
            .build()
            .expect("canonicalizing publication should normalize the same TDS");
    }

    #[test]
    fn build_rejects_incompatible_policy_before_publication() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let simplices = [vec![0, 1, 2]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();

        let result = TriangulationBuilder::new(tds, AdaptiveKernel::new())
            .topology_guarantee(TopologyGuarantee::PLManifold)
            .validation_policy(ValidationPolicy::Never)
            .build();

        assert_matches!(
            result.as_ref().map_err(RefinementError::reason),
            Err(TriangulationBuilderError::ValidationConfiguration { .. })
        );
    }

    #[test]
    fn explicit_high_dimensional_connectivity_cannot_forge_pl_link_proof() {
        let vertices = [
            vertex![0.0, 0.0, 0.0, 0.0].unwrap(),
            vertex![1.0, 0.0, 0.0, 0.0].unwrap(),
            vertex![0.0, 1.0, 0.0, 0.0].unwrap(),
            vertex![0.0, 0.0, 1.0, 0.0].unwrap(),
            vertex![0.0, 0.0, 0.0, 1.0].unwrap(),
            vertex![1.0, 1.0, 1.0, 1.0].unwrap(),
        ];
        let simplices = [vec![0, 1, 2, 3, 4], vec![1, 2, 3, 4, 5]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();

        let failure = TriangulationBuilder::new(tds, AdaptiveKernel::new())
            .canonicalizing()
            .build()
            .expect_err("raw connectivity cannot attest high-dimensional PL links");

        assert_matches!(
            failure.reason(),
            TriangulationBuilderError::TopologyValidation { source }
                if matches!(
                    source.as_ref(),
                    InvariantError::Triangulation {
                        source: crate::triangulation::validation::TriangulationValidationError::HighDimensionalVertexLinkUnproven { .. }
                    }
                )
        );
    }

    #[test]
    fn failed_publication_rolls_back_canonicalization_and_returns_original_tds() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let simplices = [vec![0, 2, 1]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();
        let expected_snapshot = serde_json::to_value(&tds).unwrap();
        let expected_owner = tds.topology_owner_id();
        let expected_generation = tds.generation();

        let mut canonicalized_probe = TriangulationDraft::with_topology_context(
            tds.clone(),
            AdaptiveKernel::new(),
            TopologyGuarantee::DEFAULT,
            GlobalTopology::DEFAULT,
        )
        .into_unpublished_triangulation();
        canonicalized_probe
            .normalize_and_promote_positive_orientation()
            .unwrap();
        assert_ne!(
            serde_json::to_value(&canonicalized_probe.tds).unwrap(),
            expected_snapshot,
            "fixture must mutate during orientation canonicalization"
        );
        assert_ne!(
            canonicalized_probe.tds.generation(),
            expected_generation,
            "fixture must advance the structural generation during canonicalization"
        );

        let failure = TriangulationBuilder::new(tds, AdaptiveKernel::new())
            .global_topology(GlobalTopology::Spherical)
            .canonicalizing()
            .build()
            .expect_err("an open simplex cannot represent a closed spherical topology");

        assert_matches!(
            failure.reason(),
            TriangulationBuilderError::TopologyValidation { .. }
        );
        assert_eq!(failure.owner().topology_owner_id(), expected_owner);
        assert_eq!(failure.owner().generation(), expected_generation);
        assert_eq!(
            serde_json::to_value(failure.owner()).unwrap(),
            expected_snapshot,
            "failed publication must return the exact pre-canonicalization TDS"
        );
        failure
            .owner()
            .validate()
            .expect("the recovered lower-layer owner must remain a valid TDS");
    }
}

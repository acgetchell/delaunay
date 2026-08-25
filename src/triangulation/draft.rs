//! Internal generic-triangulation publication state.

#![forbid(unsafe_code)]

use crate::core::tds::Tds;
use crate::core::traits::data_type::DataType;
use crate::geometry::kernel::Kernel;
use crate::refinement::RefinementError;
use crate::topology::traits::topological_space::GlobalTopology;
use crate::triangulation::Triangulation;
use crate::triangulation::builder::{
    TriangulationBuildFailure, TriangulationBuildMode, TriangulationBuildWithTopologyEvidence,
    TriangulationBuilderError,
};
use crate::triangulation::model::UnverifiedTriangulation;
use crate::triangulation::realization::{
    TriangulationCertificationError, TriangulationRealizationValidationError,
};
use crate::triangulation::validation::{
    TopologyCertificationEvidence, TopologyConstructionProvenance, TopologyGuarantee,
    ValidationPolicy,
};

/// Crate-internal unpublished Levels 3–4 candidate for a [Triangulation].
///
/// This type is implementation state shared by [`TriangulationBuilder`] and
/// higher-layer drafts; it is not a separate caller-facing construction API.
/// A proof-bearing [Tds] already owns complete explicit connectivity, so there
/// is no generic staged mutation to expose between the builder and publication.
/// Delaunay, constrained-Delaunay, and other geometry-specific algorithms own
/// any policy that infers or modifies connectivity.
///
/// [`TriangulationBuilder`]: crate::TriangulationBuilder
#[derive(Clone, Debug)]
pub struct TriangulationDraft<K, U, V, const D: usize> {
    triangulation: UnverifiedTriangulation<K, U, V, D>,
    selected_validation_policy: Option<ValidationPolicy>,
}

impl<K, U, V, const D: usize> TriangulationDraft<K, U, V, D> {
    /// Creates an unpublished candidate with explicit topology context.
    #[must_use]
    pub(crate) const fn with_topology_context(
        tds: Tds<U, V, D>,
        kernel: K,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Self {
        Self {
            triangulation: UnverifiedTriangulation::with_topology_context(
                tds,
                kernel,
                topology_guarantee,
                global_topology,
                TopologyConstructionProvenance::Unproven,
            ),
            selected_validation_policy: None,
        }
    }

    /// Selects the validation cadence retained by the published triangulation.
    ///
    /// Compatibility with the topology guarantee is checked by
    /// [finish](Self::finish).
    #[must_use]
    pub(crate) const fn validation_policy(mut self, validation_policy: ValidationPolicy) -> Self {
        self.selected_validation_policy = Some(validation_policy);
        self
    }

    /// Attaches construction evidence available only to crate-owned workflows.
    #[must_use]
    pub(crate) const fn construction_provenance(
        mut self,
        provenance: TopologyConstructionProvenance,
    ) -> Self {
        self.triangulation.storage.topology_construction_provenance = provenance;
        self
    }
}

impl<K, U, V, const D: usize> TriangulationDraft<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Publishes the staged workspace through canonicalizing Levels 3–4 validation.
    pub(crate) fn finish_canonicalizing(
        self,
    ) -> Result<Triangulation<K, U, V, D>, TriangulationBuildFailure<U, V, D>> {
        self.finish(TriangulationBuildMode::Canonicalizing)
    }

    /// Consumes the candidate and publishes it only after Levels 1–4 validation.
    ///
    /// # Errors
    ///
    /// Returns [`TriangulationBuildFailure`] if configuration is incompatible,
    /// orientation cannot be normalized, or any cumulative validation layer
    /// through Level 4 rejects the candidate. The failure retains the original
    /// TDS exactly as it was supplied to the draft.
    pub(super) fn finish(
        self,
        build_mode: TriangulationBuildMode,
    ) -> Result<Triangulation<K, U, V, D>, TriangulationBuildFailure<U, V, D>> {
        self.finish_with_topology_evidence(build_mode)
            .map(|(triangulation, _evidence)| triangulation)
    }

    /// Publishes Levels 3–4 and retains owner-bound topology metrics.
    pub(super) fn finish_with_topology_evidence(
        mut self,
        build_mode: TriangulationBuildMode,
    ) -> TriangulationBuildWithTopologyEvidence<K, U, V, D> {
        let validation_policy = self.selected_validation_policy.unwrap_or_else(|| {
            self.triangulation
                .topology_guarantee
                .default_validation_policy()
        });

        if let Err(source) = self
            .triangulation
            .try_set_validation_policy(validation_policy)
        {
            return Err(RefinementError::new(
                self.triangulation.into_tds(),
                TriangulationBuilderError::ValidationConfiguration { source },
            ));
        }

        if build_mode == TriangulationBuildMode::Strict {
            let topology_evidence = match self
                .triangulation
                .certify_levels_three_four()
                .map_err(map_certification_error)
            {
                Ok(evidence) => evidence,
                Err(source) => {
                    return Err(RefinementError::new(self.triangulation.into_tds(), source));
                }
            };
            return Ok((self.triangulation.into_verified(), topology_evidence));
        }

        let original_tds = self.triangulation.tds.clone_for_rollback();
        let publication: Result<TopologyCertificationEvidence, TriangulationBuilderError> =
            (|| {
                self.triangulation
                    .normalize_and_promote_positive_orientation()
                    .map_err(
                        |source| TriangulationBuilderError::OrientationNormalization {
                            source: Box::new(source),
                        },
                    )?;
                self.triangulation
                    .validate_geometric_nondegeneracy()
                    .map_err(|source| TriangulationBuilderError::GeometricNondegeneracy {
                        source: Box::new(source),
                    })?;
                let topology_evidence = self
                    .triangulation
                    .certify_levels_three_four()
                    .map_err(map_certification_error)?;
                Ok(topology_evidence)
            })();

        let topology_evidence =
            publication.map_err(|source| RefinementError::new(original_tds, source))?;

        Ok((self.triangulation.into_verified(), topology_evidence))
    }
}

fn map_certification_error(source: TriangulationCertificationError) -> TriangulationBuilderError {
    match source {
        TriangulationCertificationError::IncompleteConstruction { vertex_count } => {
            TriangulationBuilderError::RealizationValidation {
                source: Box::new(
                    TriangulationRealizationValidationError::IncompleteConstruction {
                        vertex_count,
                    },
                ),
            }
        }
        TriangulationCertificationError::Topology { source } => {
            TriangulationBuilderError::TopologyValidation {
                source: Box::new(source),
            }
        }
        TriangulationCertificationError::Realization { source } => {
            TriangulationBuilderError::RealizationValidation {
                source: Box::new(source),
            }
        }
    }
}

#[cfg(test)]
mod test_support {
    use super::*;

    impl<K, U, V, const D: usize> TriangulationDraft<K, U, V, D> {
        /// Consumes this layer's marker while a higher unpublished test fixture retains it.
        #[must_use]
        pub(in crate::triangulation) fn into_unpublished_triangulation(
            self,
        ) -> UnverifiedTriangulation<K, U, V, D> {
            self.triangulation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::TdsBuilder;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::vertex;

    #[test]
    fn explicit_connectivity_publishes_without_selecting_delaunay_connectivity() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let simplices = [vec![0, 1, 2]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();
        let draft = TriangulationDraft::with_topology_context(
            tds,
            AdaptiveKernel::new(),
            TopologyGuarantee::DEFAULT,
            GlobalTopology::DEFAULT,
        );

        let triangulation = draft
            .finish(TriangulationBuildMode::Canonicalizing)
            .unwrap();
        assert!(triangulation.validate_realization().is_ok());
    }

    #[test]
    fn empty_draft_publishes_as_the_verified_empty_triangulation() {
        let draft: TriangulationDraft<AdaptiveKernel<f64>, (), (), 2> =
            TriangulationDraft::with_topology_context(
                Tds::empty(),
                AdaptiveKernel::new(),
                TopologyGuarantee::DEFAULT,
                GlobalTopology::DEFAULT,
            );

        let triangulation = draft
            .finish(TriangulationBuildMode::Canonicalizing)
            .unwrap();
        assert_eq!(triangulation.number_of_vertices(), 0);
        assert!(triangulation.validate_realization().is_ok());
    }

    #[test]
    fn geometrically_degenerate_connectivity_cannot_publish() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![2.0, 0.0].unwrap(),
        ];
        let simplices = [vec![0, 1, 2]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();
        let draft = TriangulationDraft::with_topology_context(
            tds,
            AdaptiveKernel::new(),
            TopologyGuarantee::DEFAULT,
            GlobalTopology::DEFAULT,
        );

        assert!(
            draft
                .finish(TriangulationBuildMode::Canonicalizing)
                .is_err()
        );
    }
}

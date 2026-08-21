//! Internal generic-triangulation publication state.

#![forbid(unsafe_code)]

use crate::core::algorithms::incremental_insertion::InsertionError;
use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::tds::{SimplexKey, Tds, TdsError, VertexKey};
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::refinement::RefinementError;
use crate::topology::traits::topological_space::{GlobalTopology, TopologyKind};
use crate::triangulation::Triangulation;
use crate::triangulation::builder::{
    TriangulationBuildFailure, TriangulationBuildMode, TriangulationBuilderError,
};
use crate::triangulation::insertion::DetailedInsertionResult;
use crate::triangulation::realization::{
    TriangulationCertificationError, TriangulationRealizationValidationError,
};
use crate::triangulation::rollback::TriangulationRollbackTransaction;
use crate::triangulation::validation::{
    TopologyGuarantee, ValidationConfigurationError, ValidationPolicy,
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
    triangulation: Triangulation<K, U, V, D>,
    selected_validation_policy: Option<ValidationPolicy>,
}

impl<K, U, V, const D: usize> TriangulationDraft<K, U, V, D> {
    /// Creates an unpublished candidate with explicit topology context.
    #[must_use]
    pub const fn with_topology_context(
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
            selected_validation_policy: None,
        }
    }

    /// Selects the validation cadence retained by the published triangulation.
    ///
    /// Compatibility with the topology guarantee is checked by
    /// [finish](Self::finish).
    #[must_use]
    pub const fn validation_policy(mut self, validation_policy: ValidationPolicy) -> Self {
        self.selected_validation_policy = Some(validation_policy);
        self
    }

    /// Creates incomplete storage exclusively for a higher unpublished draft.
    #[must_use]
    pub(crate) fn empty_unpublished_with_topology_context(
        kernel: K,
        topology_guarantee: TopologyGuarantee,
        global_topology: GlobalTopology<D>,
    ) -> Self
    where
        K: Kernel<D>,
    {
        Self {
            triangulation: Triangulation::new_unpublished_with_topology_context(
                kernel,
                topology_guarantee,
                global_topology,
            ),
            selected_validation_policy: None,
        }
    }

    /// Clones an unpublished workspace for a failure-atomic publication attempt.
    #[must_use]
    pub(crate) fn clone_for_rollback(&self) -> Self
    where
        K: Clone,
        U: Clone,
        V: Clone,
    {
        Self {
            triangulation: Triangulation {
                kernel: self.triangulation.kernel.clone(),
                tds: self.triangulation.tds.clone_for_rollback(),
                global_topology: self.triangulation.global_topology,
                validation_policy: self.triangulation.validation_policy,
                topology_guarantee: self.triangulation.topology_guarantee,
            },
            selected_validation_policy: self.selected_validation_policy,
        }
    }

    /// Returns the number of vertices currently staged in this unpublished workspace.
    pub(crate) fn number_of_vertices(&self) -> usize {
        self.triangulation.number_of_vertices()
    }

    /// Returns the number of maximal simplices currently staged in this workspace.
    pub(crate) fn number_of_simplices(&self) -> usize {
        self.triangulation.number_of_simplices()
    }

    /// Returns the dimensionality reported by the unpublished storage.
    pub(crate) fn dim(&self) -> i32 {
        self.triangulation.dim()
    }

    /// Returns the topology guarantee selected for eventual publication.
    pub(crate) const fn topology_guarantee(&self) -> TopologyGuarantee {
        self.triangulation.topology_guarantee()
    }

    /// Returns the global topology selected for eventual publication.
    pub(crate) const fn global_topology(&self) -> GlobalTopology<D> {
        self.triangulation.global_topology()
    }

    /// Returns the high-level topology kind selected for eventual publication.
    pub(crate) const fn topology_kind(&self) -> TopologyKind {
        self.triangulation.topology_kind()
    }

    /// Returns the validation policy retained after publication.
    pub(crate) const fn retained_validation_policy(&self) -> ValidationPolicy {
        self.triangulation.validation_policy()
    }

    /// Updates a compatible validation policy without exposing the unpublished owner.
    pub(crate) fn try_set_validation_policy(
        &mut self,
        validation_policy: ValidationPolicy,
    ) -> Result<(), ValidationConfigurationError> {
        self.triangulation
            .try_set_validation_policy(validation_policy)?;
        self.selected_validation_policy = Some(validation_policy);
        Ok(())
    }

    /// Returns whether staged simplices currently have coherent orientation.
    pub(crate) fn is_coherently_oriented(&self) -> bool {
        self.triangulation.is_coherently_oriented()
    }

    /// Validates the staged Levels 1–2 structure without publishing it.
    pub(crate) fn validate_structure(&self) -> Result<(), TdsError> {
        self.triangulation.validate_structure()
    }
}

impl<K, U, V, const D: usize> TriangulationDraft<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Returns a staged vertex by key without exposing the unpublished owner.
    pub(crate) fn vertex(&self, key: VertexKey) -> Option<&Vertex<U, D>> {
        self.triangulation.vertex(key)
    }

    /// Performs one transactional insertion inside the unpublished workspace.
    pub(crate) fn insert_with_statistics_seeded_indexed_detailed(
        &mut self,
        vertex: Vertex<U, D>,
        hint: Option<SimplexKey>,
        perturbation_seed: u64,
        index: Option<&mut HashGridIndex<D>>,
    ) -> Result<DetailedInsertionResult, InsertionError> {
        self.triangulation
            .insert_with_statistics_seeded_indexed_detailed(
                vertex,
                None,
                hint,
                perturbation_seed,
                index,
                None,
            )
    }

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
        mut self,
        build_mode: TriangulationBuildMode,
    ) -> Result<Triangulation<K, U, V, D>, TriangulationBuildFailure<U, V, D>> {
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
                self.triangulation.tds,
                TriangulationBuilderError::ValidationConfiguration { source },
            ));
        }

        if build_mode == TriangulationBuildMode::Strict {
            if let Err(source) = self
                .triangulation
                .certify_levels_three_four()
                .map_err(map_certification_error)
            {
                return Err(RefinementError::new(self.triangulation.tds, source));
            }
            return Ok(self.triangulation);
        }

        let publication = {
            let mut transaction = TriangulationRollbackTransaction::begin(&mut self.triangulation);
            let result: Result<(), TriangulationBuilderError> = (|| {
                let triangulation = transaction.triangulation_mut();
                triangulation
                    .normalize_and_promote_positive_orientation()
                    .map_err(
                        |source| TriangulationBuilderError::OrientationNormalization {
                            source: Box::new(source),
                        },
                    )?;
                triangulation
                    .tds
                    .complete_construction()
                    .map_err(|source| TriangulationBuilderError::StructuralValidation {
                        source: Box::new(source),
                    })?;
                triangulation
                    .validate_geometric_nondegeneracy()
                    .map_err(|source| TriangulationBuilderError::GeometricNondegeneracy {
                        source: Box::new(source),
                    })?;
                triangulation
                    .certify_levels_three_four()
                    .map_err(map_certification_error)?;
                Ok(())
            })();

            match result {
                Ok(()) => {
                    transaction.commit();
                    Ok(())
                }
                Err(source) => {
                    transaction.rollback();
                    Err(source)
                }
            }
        };

        if let Err(source) = publication {
            return Err(RefinementError::new(self.triangulation.tds, source));
        }

        Ok(self.triangulation)
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
        pub(crate) fn into_unpublished_triangulation(self) -> Triangulation<K, U, V, D> {
            self.triangulation
        }

        /// Returns the unpublished storage identity for rollback regression tests.
        pub(crate) fn topology_owner_id(&self) -> crate::core::tds::TopologyOwnerId {
            self.triangulation.tds.topology_owner_id()
        }

        /// Returns the unpublished structural generation for rollback regression tests.
        pub(crate) fn topology_generation(&self) -> u64 {
            self.triangulation.tds.generation()
        }

        /// Selects an incompatible publication policy for rollback fault injection.
        pub(crate) const fn select_validation_policy_unchecked_for_test(
            &mut self,
            validation_policy: ValidationPolicy,
        ) {
            self.selected_validation_policy = Some(validation_policy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::{Tds, TdsBuilder};
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

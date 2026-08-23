//! Unpublished Level 5 proof state.

#![forbid(unsafe_code)]

use crate::core::collections::spatial_hash_grid::HashGridIndex;
use crate::core::operations::DelaunayInsertionState;
use crate::core::traits::data_type::DataType;
use crate::delaunay_model::{DelaunayTriangulation, EuclideanDelaunayReportDomain};
use crate::geometry::kernel::Kernel;
use crate::triangulation::Triangulation;
use crate::validation::{
    DelaunayLevelFiveCertificate, DelaunayTriangulationRefinementError,
    certify_level_five_for_refinement,
};

/// Unpublished Level 5 draft containing an already proven Levels 1–4 owner.
///
/// A draft contains exactly the immediately lower proof-bearing owner needed
/// for promotion. It is distinct from a construction workspace: it never
/// contains partial connectivity or an unproven lower-layer owner. Its
/// consuming transition keeps the Level 5 check adjacent to construction of
/// the final [`DelaunayTriangulation`].
#[derive(Clone, Debug)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "explicit crate visibility distinguishes unpublished state from the public owner"
)]
pub(crate) struct DelaunayTriangulationDraft<K, U, V, const D: usize> {
    triangulation: Triangulation<K, U, V, D>,
    insertion_state: DelaunayInsertionState,
    spatial_index: Option<HashGridIndex<D>>,
    euclidean_report_domain: EuclideanDelaunayReportDomain,
}

/// Private type state proving that this exact draft passed Level 5.
struct CertifiedDelaunayTriangulationDraft<K, U, V, const D: usize> {
    draft: DelaunayTriangulationDraft<K, U, V, D>,
    _certificate: DelaunayLevelFiveCertificate<D>,
}

impl<K, U, V, const D: usize> CertifiedDelaunayTriangulationDraft<K, U, V, D> {
    /// Publishes the owner after the only constructors of this type establish
    /// fresh or retained Level 5 evidence for the draft's exact topology state.
    fn publish(self) -> DelaunayTriangulation<K, U, V, D> {
        DelaunayTriangulation {
            tri: self.draft.triangulation,
            insertion_state: self.draft.insertion_state,
            spatial_index: self.draft.spatial_index,
            euclidean_report_domain: self.draft.euclidean_report_domain,
        }
    }
}

impl<K, U, V, const D: usize> DelaunayTriangulationDraft<K, U, V, D> {
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

impl<K, U, V, const D: usize> DelaunayTriangulationDraft<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Promotes the proof-bearing Levels 1–4 owner by checking only Level 5.
    ///
    /// `Triangulation` already represents the cumulative Levels 1–4 proof.
    /// Revalidating those layers here would weaken the type into an unchecked
    /// data bag and duplicate work at every promotion boundary.
    pub(crate) fn try_into_delaunay(
        self,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayTriangulationRefinementError<K, U, V, D>>
    {
        let certified = self.try_certify_level_five()?;
        Ok(certified.publish())
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
        if let Some(certificate) =
            certificate.filter(|certificate| certificate.applies_to(&self.triangulation))
        {
            return Ok(CertifiedDelaunayTriangulationDraft {
                draft: self,
                _certificate: certificate.clone(),
            }
            .publish());
        }
        self.try_into_delaunay()
    }

    /// Consumes the draft into a state carrying fresh Level 5 evidence for its exact owner.
    fn try_certify_level_five(
        self,
    ) -> Result<
        CertifiedDelaunayTriangulationDraft<K, U, V, D>,
        DelaunayTriangulationRefinementError<K, U, V, D>,
    > {
        let certificate = match certify_level_five_for_refinement(&self.triangulation) {
            Ok(certificate) => certificate,
            Err(reason) => {
                return Err(crate::refinement::RefinementError::new(
                    self.triangulation,
                    reason,
                ));
            }
        };
        Ok(CertifiedDelaunayTriangulationDraft {
            draft: self,
            _certificate: certificate,
        })
    }
}

#[cfg(test)]
mod test_support {
    use super::*;
    use crate::core::tds::Tds;
    use crate::topology::traits::topological_space::GlobalTopology;
    use crate::triangulation::validation::{TopologyConstructionProvenance, TopologyGuarantee};

    impl<K, U, V, const D: usize> DelaunayTriangulationDraft<K, U, V, D> {
        /// Builds an intentionally unproven test draft from raw parts.
        pub const fn assemble_unchecked_for_test(
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
                    topology_construction_provenance: TopologyConstructionProvenance::Unproven,
                },
                insertion_state: DelaunayInsertionState::new(),
                spatial_index: None,
                euclidean_report_domain: EuclideanDelaunayReportDomain::Unproven,
            }
        }

        /// Deliberately bypasses Level 5 promotion for validation and internal
        /// repair failure tests.
        pub fn into_unproven_delaunay_for_test(self) -> DelaunayTriangulation<K, U, V, D> {
            DelaunayTriangulation {
                tri: self.triangulation,
                insertion_state: self.insertion_state,
                spatial_index: self.spatial_index,
                euclidean_report_domain: self.euclidean_report_domain,
            }
        }

        /// Overrides report provenance for tests that exercise stale evidence.
        pub fn set_euclidean_report_domain_for_test(
            &mut self,
            euclidean_report_domain: EuclideanDelaunayReportDomain,
        ) {
            self.euclidean_report_domain = euclidean_report_domain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::DelaunayTriangulationBuilder;
    use crate::core::simplex::Simplex;
    use crate::validation::DelaunayTriangulationValidationError;
    use crate::vertex;

    #[test]
    fn foreign_level_five_certificate_cannot_publish_a_non_delaunay_draft() {
        let valid_vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let valid_triangulation = DelaunayTriangulationBuilder::new(&valid_vertices)
            .build_triangulation()
            .unwrap();
        let foreign_certificate = certify_level_five_for_refinement(&valid_triangulation).unwrap();

        let invalid_vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([4.0, 0.0]).unwrap(),
            vertex!([4.0, 2.0]).unwrap(),
            vertex!([1.0, 2.0]).unwrap(),
        ];
        let invalid_simplices = [vec![0, 1, 2], vec![0, 2, 3]];
        let invalid_triangulation = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(
            &invalid_vertices,
            &invalid_simplices,
        )
        .unwrap()
        .build_triangulation()
        .unwrap();

        let failure = DelaunayTriangulationDraft::from_triangulation(invalid_triangulation)
            .try_into_delaunay_with_certificate(Some(&foreign_certificate))
            .unwrap_err();

        assert_eq!(failure.owner().number_of_vertices(), invalid_vertices.len());
        assert!(matches!(
            failure.reason(),
            DelaunayTriangulationValidationError::VerificationFailed { .. }
        ));
    }

    #[test]
    fn stale_level_five_certificate_falls_back_to_full_certification() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([4.0, 0.0]).unwrap(),
            vertex!([4.0, 2.0]).unwrap(),
            vertex!([1.0, 2.0]).unwrap(),
        ];
        let mut triangulation = DelaunayTriangulationBuilder::new(&vertices)
            .build_triangulation()
            .unwrap();
        let certificate = certify_level_five_for_refinement(&triangulation).unwrap();
        let owner_id = triangulation.tds.topology_owner_id();
        let generation = triangulation.tds.generation();

        let vertex_keys: Vec<_> = vertices
            .iter()
            .map(|vertex| {
                triangulation
                    .tds
                    .vertex_key_from_uuid(&vertex.uuid())
                    .unwrap()
            })
            .collect();
        let simplex_keys: Vec<_> = triangulation.tds.simplex_keys().collect();
        triangulation
            .tds
            .remove_simplices_by_keys(&simplex_keys)
            .unwrap();
        for indices in [[0, 1, 2], [0, 2, 3]] {
            triangulation
                .tds
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(
                        indices.map(|index| vertex_keys[index]).to_vec(),
                        None,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        triangulation.tds.assign_neighbors().unwrap();
        triangulation.tds.assign_incident_simplices().unwrap();

        assert_eq!(triangulation.tds.topology_owner_id(), owner_id);
        assert_ne!(triangulation.tds.generation(), generation);
        let failure = DelaunayTriangulationDraft::from_triangulation(triangulation)
            .try_into_delaunay_with_certificate(Some(&certificate))
            .unwrap_err();
        assert!(matches!(
            failure.reason(),
            DelaunayTriangulationValidationError::VerificationFailed { .. }
        ));
    }
}

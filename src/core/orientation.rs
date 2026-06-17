//! Geometric orientation orchestration for generic [`Triangulation`](crate::Triangulation).
//!
//! This module owns triangulation-level orientation work: collecting lifted simplex
//! points from the TDS, validating geometric orientation, canonicalizing simplex
//! slot order, and normalizing coherent orientation after construction or edits.
//! Predicate implementations remain in the geometry layer.

use crate::core::algorithms::incremental_insertion::{
    InsertionError, InsertionTopologyValidationContext,
};
use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SimplexKeyBuffer, SmallBuffer};
use crate::core::simplex::Simplex;
use crate::core::tds::{GeometricError, SimplexKey, TdsError, VertexKey};
use crate::core::triangulation::Triangulation;
use crate::core::validation::TriangulationValidationError;
use crate::geometry::kernel::Kernel;
use crate::geometry::point::Point;
use crate::geometry::predicates::Orientation;
use crate::geometry::robust_predicates::robust_orientation;
use crate::topology::traits::global_topology_model::GlobalTopologyModel;

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
{
    /// Collect simplex points for orientation evaluation.
    ///
    /// For periodic simplices, this delegates per-vertex lattice-offset lifting to the active
    /// [`GlobalTopology`](crate::topology::traits::topological_space::GlobalTopology) behavior model.
    fn collect_simplex_points_for_orientation(
        &self,
        simplex_key: SimplexKey,
        simplex: &Simplex<V, D>,
        purpose: &str,
    ) -> Result<SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>, TdsError> {
        let topology_model = self.global_topology.model();
        let periodic_offsets = simplex.periodic_vertex_offsets();
        if let Some(offsets) = periodic_offsets {
            if offsets.len() != simplex.number_of_vertices() {
                return Err(TdsError::DimensionMismatch {
                    expected: simplex.number_of_vertices(),
                    actual: offsets.len(),
                    context: format!(
                        "simplex {:?} (key {simplex_key:?}) periodic offset count vs vertex count during {purpose}",
                        simplex.uuid(),
                    ),
                });
            }
            if !topology_model.supports_periodic_orientation_offsets() {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Simplex {:?} (key {simplex_key:?}) has periodic offsets (count {}) during {purpose}, but triangulation global topology is {:?} (kind {:?}, allows_boundary: {}, periodic_domain: {:?}); expected periodic-orientation-offset-capable topology",
                        simplex.uuid(),
                        offsets.len(),
                        self.global_topology,
                        topology_model.kind(),
                        topology_model.allows_boundary(),
                        topology_model.periodic_domain(),
                    ),
                });
            }
        }

        let mut points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(simplex.number_of_vertices());

        for (vertex_idx, &vertex_key) in simplex.vertices().iter().enumerate() {
            let vertex = self.tds.vertex(vertex_key).ok_or_else(|| {
                TdsError::VertexNotFound {
                    vertex_key,
                    context: format!(
                        "referenced by simplex {:?} (key {simplex_key:?}) at position {vertex_idx} during {purpose}",
                        simplex.uuid(),
                    ),
                }
            })?;
            let periodic_offset = periodic_offsets.map(|offsets| offsets[vertex_idx]);
            let lifted_coords = topology_model
                .lift_for_orientation(*vertex.point().coords(), periodic_offset)
                .map_err(|error| TdsError::InconsistentDataStructure {
                    message: format!(
                        "Failed to lift coordinates for vertex key {vertex_key:?} at slot {vertex_idx} in simplex {:?} (key {simplex_key:?}) during {purpose}: {error}",
                        simplex.uuid(),
                    ),
                })?;

            let lifted_point =
                Point::try_new(lifted_coords).map_err(|source| TdsError::InconsistentDataStructure {
                    message: format!(
                        "Lifted coordinates for vertex key {vertex_key:?} at slot {vertex_idx} in simplex {:?} (key {simplex_key:?}) during {purpose} violated point invariants: {source}",
                        simplex.uuid(),
                    ),
                })?;
            points.push(lifted_point);
        }

        Ok(points)
    }

    /// Evaluate a simplex's geometric orientation for a validation/canonicalization context.
    ///
    /// This helper uses [`robust_orientation`] directly, without `SoS`, so true
    /// degeneracy remains distinguishable from positive or negative orientation.
    pub(crate) fn evaluate_simplex_orientation_for_context(
        &self,
        simplex_key: SimplexKey,
        simplex: &Simplex<V, D>,
        purpose: &str,
        predicate_failure_prefix: &str,
    ) -> Result<i32, TdsError> {
        let points = self.collect_simplex_points_for_orientation(simplex_key, simplex, purpose)?;

        match robust_orientation(&points) {
            Ok(Orientation::POSITIVE) => Ok(1),
            Ok(Orientation::NEGATIVE) => Ok(-1),
            Ok(Orientation::DEGENERATE) => Ok(0),
            Err(error) => Err(TdsError::InconsistentDataStructure {
                message: format!(
                    "{predicate_failure_prefix} {:?} (key {simplex_key:?}): {error}",
                    simplex.uuid(),
                ),
            }),
        }
    }

    /// Validates geometric orientation sign for each stored simplex using exact arithmetic.
    ///
    /// Simplices are stored in canonical positive orientation order by construction and mutation
    /// paths; a negative sign indicates geometric/combinatorial mismatch.
    pub(crate) fn validate_geometric_simplex_orientation(&self) -> Result<(), TdsError> {
        for (simplex_key, simplex) in self.tds.simplices() {
            let orientation = self.evaluate_simplex_orientation_for_context(
                simplex_key,
                simplex,
                "geometric orientation validation",
                "Geometric orientation predicate failed for simplex",
            )?;
            if orientation < 0 {
                let vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    simplex.vertices().iter().copied().collect();
                let neighbor_keys: SmallBuffer<Option<SimplexKey>, MAX_PRACTICAL_DIMENSION_SIZE> =
                    simplex
                        .neighbor_keys()
                        .map(Iterator::collect)
                        .unwrap_or_default();
                tracing::debug!(
                    simplex_uuid = %simplex.uuid(),
                    ?simplex_key,
                    ?vertex_keys,
                    ?neighbor_keys,
                    orientation,
                    "negative geometric orientation detected during validation",
                );

                return Err(TdsError::Geometric(GeometricError::NegativeOrientation {
                    message: format!(
                        "Simplex {:?} (key {simplex_key:?}, vertices {vertex_keys:?}) has negative geometric orientation; expected positive canonical orientation",
                        simplex.uuid(),
                    ),
                }));
            }
        }

        Ok(())
    }

    /// Validates geometric orientation for a local set of simplices.
    pub(crate) fn validate_geometric_simplex_orientation_for_simplices(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<(), TdsError> {
        for &simplex_key in simplices {
            let simplex =
                self.tds
                    .simplex(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "local geometric orientation validation scope".to_string(),
                    })?;
            let orientation = self.evaluate_simplex_orientation_for_context(
                simplex_key,
                simplex,
                "local geometric orientation validation",
                "Geometric orientation predicate failed for local simplex",
            )?;
            if orientation < 0 {
                let vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                    simplex.vertices().iter().copied().collect();
                tracing::debug!(
                    simplex_uuid = %simplex.uuid(),
                    ?simplex_key,
                    ?vertex_keys,
                    orientation,
                    "negative geometric orientation detected during local validation",
                );

                return Err(TdsError::Geometric(GeometricError::NegativeOrientation {
                    message: format!(
                        "Simplex {:?} (key {simplex_key:?}, vertices {vertex_keys:?}) has negative geometric orientation; expected positive canonical orientation",
                        simplex.uuid(),
                    ),
                }));
            }
        }

        Ok(())
    }

    /// Validates local orientation invariants for simplices changed by insertion.
    pub(crate) fn validate_local_orientation_for_simplices(
        &self,
        simplices: &[SimplexKey],
    ) -> Result<(), InsertionError> {
        self.tds
            .validate_coherent_orientation_for_simplices(simplices)?;
        self.validate_geometric_simplex_orientation_for_simplices(simplices)?;
        Ok(())
    }

    /// Flip all negatively oriented simplices to positive orientation.
    fn promote_simplices_to_positive_orientation(&mut self) -> Result<bool, InsertionError> {
        let mut negative_simplices = SimplexKeyBuffer::new();

        for (simplex_key, simplex) in self.tds.simplices() {
            let orientation = self.evaluate_simplex_orientation_for_context(
                simplex_key,
                simplex,
                "positive-orientation promotion",
                "Geometric orientation predicate failed while promoting positive orientation for simplex",
            )?;
            if orientation == 0 {
                continue;
            }
            if orientation < 0 {
                negative_simplices.push(simplex_key);
            }
        }

        if negative_simplices.is_empty() {
            return Ok(false);
        }

        for simplex_key in negative_simplices {
            let simplex =
                self.tds
                    .simplex_mut(simplex_key)
                    .ok_or_else(|| TdsError::SimplexNotFound {
                        simplex_key,
                        context: "applying positive-orientation promotion".to_string(),
                    })?;
            if simplex.number_of_vertices() >= 2 {
                simplex.swap_vertex_slots(0, 1);
            }
        }

        self.tds.mark_topology_modified();
        Ok(true)
    }

    /// Check whether any simplex still requires positive-orientation promotion.
    fn simplices_require_positive_orientation_promotion(&self) -> Result<bool, InsertionError> {
        for (simplex_key, simplex) in self.tds.simplices() {
            let orientation = self.evaluate_simplex_orientation_for_context(
                simplex_key,
                simplex,
                "positive-orientation convergence check",
                "Geometric orientation predicate failed while checking positive-orientation convergence for simplex",
            )?;
            if orientation == 0 {
                continue;
            }
            if orientation < 0 {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// For connected non-periodic triangulations, canonicalize the coherent global sign.
    fn canonicalize_global_orientation_sign(&mut self) -> Result<(), InsertionError> {
        let representative_sign = {
            let mut sign = None;
            for (simplex_key, simplex) in self.tds.simplices() {
                let orientation = self.evaluate_simplex_orientation_for_context(
                    simplex_key,
                    simplex,
                    "global orientation-sign canonicalization",
                    "Geometric orientation predicate failed while canonicalizing global orientation sign for simplex",
                )?;
                if orientation != 0 {
                    sign = Some(orientation);
                    break;
                }
            }
            sign
        };

        if representative_sign != Some(-1) {
            return Ok(());
        }

        let simplex_keys: Vec<SimplexKey> = self.tds.simplex_keys().collect();
        let mut flipped_any = false;
        for simplex_key in simplex_keys {
            let Some(simplex) = self.tds.simplex_mut(simplex_key) else {
                continue;
            };
            if simplex.number_of_vertices() >= 2 {
                simplex.swap_vertex_slots(0, 1);
                flipped_any = true;
            }
        }

        if flipped_any {
            self.tds.mark_topology_modified();
        }

        Ok(())
    }

    /// Normalize coherent orientation and promote geometric orientation to the positive sign.
    pub(crate) fn normalize_and_promote_positive_orientation(
        &mut self,
    ) -> Result<(), InsertionError> {
        self.tds.normalize_coherent_orientation()?;
        self.canonicalize_global_orientation_sign()?;

        for _ in 0..3 {
            if !self.promote_simplices_to_positive_orientation()? {
                break;
            }
            self.tds.normalize_coherent_orientation()?;
        }

        if self.simplices_require_positive_orientation_promotion()? {
            let mut residual_count = 0_usize;
            let mut sample_keys: [Option<SimplexKey>; 5] = [None; 5];
            for (simplex_key, simplex) in self.tds.simplices() {
                let orientation = self.evaluate_simplex_orientation_for_context(
                    simplex_key,
                    simplex,
                    "residual negative-orientation sampling",
                    "Geometric orientation predicate failed while sampling residual negatives for simplex",
                )?;
                if orientation < 0 {
                    if residual_count < sample_keys.len() {
                        sample_keys[residual_count] = Some(simplex_key);
                    }
                    residual_count += 1;
                }
            }
            let sampled: Vec<SimplexKey> = sample_keys.into_iter().flatten().collect();
            return Err(InsertionError::TopologyValidationFailed {
                context: InsertionTopologyValidationContext::PositiveOrientationPromotion,
                source: TriangulationValidationError::OrientationPromotionNonConvergence {
                    residual_count,
                    sampled,
                },
            });
        }
        self.canonicalize_global_orientation_sign()?;
        Ok(())
    }

    /// Canonicalize a set of newly created simplices to positive geometric orientation.
    #[expect(
        clippy::too_many_lines,
        reason = "debug-only orientation diagnostics with dedup add conditional branches"
    )]
    pub(crate) fn canonicalize_positive_orientation_for_simplices(
        &mut self,
        simplices: &SimplexKeyBuffer,
    ) -> Result<(), InsertionError> {
        #[cfg(debug_assertions)]
        let debug_orientation = std::env::var_os("DELAUNAY_DEBUG_ORIENTATION").is_some();
        #[cfg(debug_assertions)]
        let mut orientation_warn_count = 0_usize;

        for &simplex_key in simplices {
            let orientation = {
                let simplex =
                    self.tds
                        .simplex(simplex_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key,
                            context: "canonicalizing insertion orientation".to_string(),
                        })?;
                self.evaluate_simplex_orientation_for_context(
                    simplex_key,
                    simplex,
                    "insertion orientation canonicalization",
                    "Geometric orientation predicate failed while canonicalizing simplex",
                )?
            };

            if orientation == 0 {
                continue;
            }

            if orientation < 0 {
                #[cfg(debug_assertions)]
                let pre_swap_vertices = if debug_orientation {
                    self.tds.simplex(simplex_key).map(|c| c.vertices().to_vec())
                } else {
                    None
                };

                let simplex =
                    self.tds
                        .simplex_mut(simplex_key)
                        .ok_or_else(|| TdsError::SimplexNotFound {
                            simplex_key,
                            context: "applying insertion orientation canonicalization".to_string(),
                        })?;
                if simplex.number_of_vertices() < 2 {
                    return Err(TdsError::DimensionMismatch {
                        expected: 2,
                        actual: simplex.number_of_vertices(),
                        context: format!(
                            "simplex {simplex_key:?} needs >= 2 vertices for orientation canonicalization"
                        ),
                    }
                    .into());
                }
                simplex.swap_vertex_slots(0, 1);

                #[cfg(debug_assertions)]
                if debug_orientation {
                    orientation_warn_count += 1;
                    if orientation_warn_count <= 3 {
                        let post_orientation = self.tds.simplex(simplex_key).map(|c| {
                            self.evaluate_simplex_orientation_for_context(
                                simplex_key,
                                c,
                                "orientation swap verification",
                                "orientation predicate failed during swap verification",
                            )
                        });
                        match post_orientation {
                            Some(Ok(post_o)) => {
                                tracing::warn!(
                                    simplex_key = ?simplex_key,
                                    pre_swap_vertices = ?pre_swap_vertices,
                                    pre_swap_orientation = orientation,
                                    post_swap_orientation = post_o,
                                    swap_fixed = post_o > 0,
                                    "canonicalize_positive_orientation: negative-orientation simplex swapped"
                                );
                            }
                            Some(Err(ref e)) => {
                                tracing::warn!(
                                    simplex_key = ?simplex_key,
                                    pre_swap_vertices = ?pre_swap_vertices,
                                    pre_swap_orientation = orientation,
                                    error = %e,
                                    "canonicalize_positive_orientation: post-swap verification failed"
                                );
                            }
                            None => {
                                tracing::warn!(
                                    simplex_key = ?simplex_key,
                                    pre_swap_vertices = ?pre_swap_vertices,
                                    pre_swap_orientation = orientation,
                                    "canonicalize_positive_orientation: simplex not found after swap"
                                );
                            }
                        }
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        if orientation_warn_count > 3 && debug_orientation {
            let suppressed = orientation_warn_count - 3;
            tracing::warn!(
                total_negative = orientation_warn_count,
                suppressed,
                "canonicalize_positive_orientation: suppressed {suppressed} additional negative-orientation warnings (see first 3 above)"
            );
        }

        Ok(())
    }

    /// Verifies that no simplex is geometrically degenerate.
    ///
    /// This is a sign-agnostic check: it flags simplices whose exact orientation
    /// determinant is zero regardless of the sign.
    pub(crate) fn validate_geometric_nondegeneracy(&self) -> Result<(), TdsError> {
        for (simplex_key, simplex) in self.tds.simplices() {
            let orientation = self.evaluate_simplex_orientation_for_context(
                simplex_key,
                simplex,
                "geometric nondegeneracy check",
                "Orientation predicate failed for simplex",
            )?;
            if orientation == 0 {
                return Err(TdsError::Geometric(GeometricError::DegenerateOrientation {
                    message: format!(
                        "Simplex {:?} (key {simplex_key:?}) is geometrically degenerate \
                         (zero-volume simplex from collinear/coplanar vertices)",
                        simplex.uuid(),
                    ),
                }));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::InvariantError;
    use crate::geometry::kernel::FastKernel;
    use crate::topology::traits::topological_space::{
        GlobalTopology, ToroidalConstructionMode, ToroidalDomainError,
    };
    use std::assert_matches;

    /// Regression test: a negatively oriented but topologically valid simplex
    /// passes topology-only validation while failing full validation.
    #[test]
    fn negative_oriented_simplex_topology_only() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        assert!(tri.is_valid().is_ok());
        assert!(tri.is_valid_topology_only().is_ok());

        let simplex_key = tri.tds.simplex_keys().next().unwrap();
        tri.tds
            .simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 1);

        assert!(tri.is_valid_topology_only().is_ok());
        assert!(tri.is_valid().is_err());
    }

    #[test]
    fn local_geometric_orientation_validation_errors_on_missing_scope_simplex() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 3>::new_with_tds(FastKernel::new(), tds);
        let simplex_key = tri.tds.simplex_keys().next().unwrap();
        assert_eq!(tri.tds.remove_simplices_by_keys(&[simplex_key]), 1);

        match tri.validate_geometric_simplex_orientation_for_simplices(&[simplex_key]) {
            Err(TdsError::SimplexNotFound {
                simplex_key: missing_key,
                ..
            }) => assert_eq!(missing_key, simplex_key),
            other => panic!("Expected SimplexNotFound, got {other:?}"),
        }
    }

    #[test]
    fn is_valid_rejects_negative_geometric_simplex_orientation() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();

        let simplex_key = tds.simplex_keys().next().unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 1);

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let err = tri.is_valid().unwrap_err();
        assert_matches!(
            err,
            InvariantError::Tds(TdsError::Geometric(GeometricError::NegativeOrientation { message }))
                if message.contains("negative geometric orientation")
        );
    }

    #[test]
    fn validate_geometric_simplex_orientation_returns_enriched_error_on_negative() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();

        let simplex_key = tds.simplex_keys().next().unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 1);

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let err = tri.validate_geometric_simplex_orientation().unwrap_err();
        assert!(
            matches!(
                &err,
                TdsError::Geometric(GeometricError::NegativeOrientation { message })
                    if message.contains("negative geometric orientation")
                       && message.contains("vertices")
            ),
            "Error should contain vertex keys: {err}"
        );
    }

    #[test]
    fn simplices_require_positive_orientation_promotion_detects_negative_without_mutating() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 1);

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let before: Vec<_> = tri.tds.simplex(simplex_key).unwrap().vertices().to_vec();

        assert!(
            tri.simplices_require_positive_orientation_promotion()
                .unwrap()
        );

        let after: Vec<_> = tri.tds.simplex(simplex_key).unwrap().vertices().to_vec();
        assert_eq!(before, after);
    }

    #[test]
    fn simplices_require_positive_orientation_promotion_false_for_positive_without_mutating() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
        ];
        let tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let before: Vec<_> = tri.tds.simplex(simplex_key).unwrap().vertices().to_vec();

        assert!(
            !tri.simplices_require_positive_orientation_promotion()
                .unwrap()
        );

        let after: Vec<_> = tri.tds.simplex(simplex_key).unwrap().vertices().to_vec();
        assert_eq!(before, after);
    }

    #[test]
    fn periodic_geometric_orientation_validation_uses_lifted_coordinates() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.8, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.8]).unwrap(),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [1, 0]])
            .unwrap();

        let mut tri =
            Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        tri.set_global_topology(
            GlobalTopology::try_toroidal([1.0, 1.0], ToroidalConstructionMode::PeriodicImagePoint)
                .unwrap(),
        );

        assert!(tri.validate_geometric_simplex_orientation().is_ok());

        tri.tds
            .simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 1);
        let err = tri.validate_geometric_simplex_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::Geometric(GeometricError::NegativeOrientation { message })
                if message.contains("negative geometric orientation")
        );
    }

    #[test]
    fn periodic_geometric_orientation_validation_requires_toroidal_metadata() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.8, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.8]).unwrap(),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [1, 0]])
            .unwrap();

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let err = tri.validate_geometric_simplex_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::InconsistentDataStructure { message }
                if message.contains("has periodic offsets")
                    && message.contains("expected periodic-orientation-offset-capable topology")
        );
    }

    #[test]
    fn periodic_geometric_orientation_validation_rejects_offset_count_mismatch() {
        let vertices = vec![
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.8, 0.0]).unwrap(),
            crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.8]).unwrap(),
        ];
        let mut tds =
            Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices).unwrap();
        let simplex_key = tds.simplex_keys().next().unwrap();
        tds.simplex_mut(simplex_key)
            .unwrap()
            .periodic_vertex_offsets = Some(vec![[0, 0], [1, 0]].into());

        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);
        let err = tri.validate_geometric_simplex_orientation().unwrap_err();
        assert_matches!(
            err,
            TdsError::DimensionMismatch {
                expected: 3,
                actual: 2,
                ..
            }
        );
    }

    #[test]
    fn periodic_geometric_orientation_rejects_invalid_domain_at_parse_boundary() {
        let err = GlobalTopology::<2>::try_toroidal(
            [0.0, 1.0],
            ToroidalConstructionMode::PeriodicImagePoint,
        )
        .unwrap_err();
        assert_matches!(
            err,
            ToroidalDomainError::InvalidPeriod { axis: 0, period }
                if period.abs() < f64::EPSILON
        );
    }
}

//! Failure-atomic bistellar flip validation and mutation primitives.

#![forbid(unsafe_code)]

#[cfg(debug_assertions)]
use super::FlipOrientationCheckStage;
use super::{
    AppliedFlip, BistellarFlipKind, DataType, EntityKind, FacetError, FacetHandle, FastHashMap,
    FastHashSet, FlipContext, FlipContextDyn, FlipContextError, FlipDirection, FlipError, FlipInfo,
    FlipMutationError, FlipNeighborWiringError, FlipPredicateError, FlipPredicateOperation,
    MAX_PRACTICAL_DIMENSION_SIZE, NeighborSlot, NeighborValidationError, Orientation, PreparedFlip,
    RemovedSimplexVertexSnapshot, ReplacementPeriodicOffsets, ReplacementSimplexVertices, Simplex,
    SimplexKey, SimplexKeyBuffer, SmallBuffer, Tds, TdsRollbackTransaction, TdsValidationFailure,
    VertexKey, VertexKeyList, build_flip_topology_index, env, external_facets_for_boundary,
    extract_cavity_boundary, facet_key_from_vertices, facet_order, facet_vertices_from_simplex,
    flip_would_create_nonmanifold_facets_any, flip_would_duplicate_simplex_any,
    normalized_facet_order_with_offsets, orient_replacement_simplices,
    periodic_offsets_or_zero_frame, permutation_odd, repair_trace_enabled,
    replacement_simplex_periodic_offsets, robust_orientation, validate_replacement_orientation,
    vertices_to_points, wire_cavity_neighbors,
};

pub(super) fn snapshot_removed_simplex_vertices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    removed_simplices: &SimplexKeyBuffer,
) -> Result<RemovedSimplexVertexSnapshot, FlipError>
where
    U: DataType,
    V: DataType,
{
    removed_simplices
        .iter()
        .copied()
        .map(|simplex_key| {
            let simplex = tds
                .simplex(simplex_key)
                .ok_or(FlipError::MissingSimplex { simplex_key })?;
            Ok(simplex.vertices().iter().copied().collect())
        })
        .collect()
}

/// Applies a bistellar flip using explicit k and vertex/simplex slices.
///
/// # Errors
///
/// Returns [`FlipError::DanglingVertexIncidence`] if the maintained incidence
/// index references a simplex that is no longer present, or another
/// [`FlipError`] when the move is invalid, geometrically degenerate,
/// non-manifold, or cannot be applied atomically.
#[expect(
    clippy::too_many_arguments,
    reason = "Flip mutation needs explicit move, cavity, policy, and validation inputs"
)]
pub(super) fn apply_bistellar_flip_with_k<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    k_move: usize,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
    removed_simplices: &SimplexKeyBuffer,
    direction: FlipDirection,
    orientation_policy: ReplacementOrientationPolicy,
    validation_scope: FlipValidationScope,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k_inner(
        tds,
        k_move,
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction,
        orientation_policy,
        validation_scope,
    )
}

/// Applies a bistellar flip without rollback.
///
/// The caller owns transaction rollback if this returns an error or if later
/// postconditions fail.
#[expect(
    clippy::too_many_arguments,
    reason = "Raw flip mutation needs explicit move, cavity, policy, and validation inputs"
)]
pub(super) fn apply_bistellar_flip_with_k_raw<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    k_move: usize,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
    removed_simplices: &SimplexKeyBuffer,
    direction: FlipDirection,
    orientation_policy: ReplacementOrientationPolicy,
    validation_scope: FlipValidationScope,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    let PreparedFlip {
        kind,
        direction,
        removed_simplices,
        removed_face_vertices,
        inserted_face_vertices,
        new_simplex_vertices,
        new_simplex_offsets,
        external_facets,
        removed_simplex_vertices,
    } = prepare_bistellar_flip(
        tds,
        k_move,
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction,
        orientation_policy,
    )?;

    let new_simplices = apply_prepared_flip_mutation(
        tds,
        new_simplex_vertices,
        new_simplex_offsets,
        &external_facets,
        &removed_simplices,
        k_move,
        direction,
        validation_scope,
    )?;

    Ok(AppliedFlip {
        info: FlipInfo {
            kind,
            direction,
            removed_simplices,
            new_simplices,
            removed_face_vertices,
            inserted_face_vertices,
        },
        removed_simplex_vertices,
    })
}

/// Applies a generic k-move without rollback.
///
/// The caller owns transaction rollback if this returns an error or if later
/// postconditions fail.
pub(crate) fn apply_bistellar_flip_raw<U, V, const D: usize, const K_MOVE: usize>(
    tds: &mut Tds<U, V, D>,
    context: &FlipContext<D, K_MOVE>,
) -> Result<FlipInfo<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    Ok(apply_bistellar_flip_with_k_raw(
        tds,
        K_MOVE,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::AllowSigned,
        FlipValidationScope::FullTds,
    )?
    .info)
}

/// Applies a runtime-k generic move without rollback.
///
/// The caller owns transaction rollback if this returns an error or if later
/// postconditions fail.
pub(crate) fn apply_bistellar_flip_dynamic_raw<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    k_move: usize,
    context: &FlipContextDyn<D>,
) -> Result<FlipInfo<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    Ok(apply_bistellar_flip_with_k_raw(
        tds,
        k_move,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::AllowSigned,
        FlipValidationScope::FullTds,
    )?
    .info)
}

/// Builds and validates the replacement side of a bistellar flip without mutating storage.
///
/// This shared preparation step is the source of truth for both immutable
/// feasibility checks and the mutating executor's deterministic pre-commit
/// checks.
#[expect(
    clippy::too_many_lines,
    reason = "Keep exact feasibility checks aligned with the mutating flip preflight"
)]
pub(super) fn prepare_bistellar_flip<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    k_move: usize,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
    removed_simplices: &SimplexKeyBuffer,
    direction: FlipDirection,
    orientation_policy: ReplacementOrientationPolicy,
) -> Result<PreparedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if k_move == 0 || k_move > D + 1 {
        return Err(FlipContextError::InvalidMoveSize {
            k_move,
            dimension: D,
        }
        .into());
    }

    let expected_removed_face = D + 2 - k_move;
    if removed_face_vertices.len() != expected_removed_face {
        return Err(FlipContextError::WrongRemovedFaceArity {
            expected: expected_removed_face,
            found: removed_face_vertices.len(),
        }
        .into());
    }
    if inserted_face_vertices.len() != k_move {
        return Err(FlipContextError::WrongInsertedFaceArity {
            k_move,
            expected: k_move,
            found: inserted_face_vertices.len(),
        }
        .into());
    }
    if removed_simplices.len() != k_move {
        return Err(FlipContextError::WrongRemovedSimplexCount {
            expected: k_move,
            found: removed_simplices.len(),
        }
        .into());
    }
    if removed_face_vertices
        .iter()
        .any(|v| inserted_face_vertices.contains(v))
    {
        return Err(FlipContextError::OverlappingFaces.into());
    }
    #[cfg(debug_assertions)]
    {
        // Coherent orientation is a validation-scale invariant. Keep the typed
        // diagnostic in debug/test builds, but do not scan the whole TDS inside
        // every release-mode flip on construction/repair hot paths.
        if !tds.is_coherently_oriented() {
            return Err(FlipContextError::CoherentOrientationViolation {
                stage: FlipOrientationCheckStage::BeforeMutation,
                k_move,
                direction,
            }
            .into());
        }
    }

    // Bistellar move legality: the inserted simplex must not already exist in the complex.
    //
    // If it does, applying the move can create non-manifold codimension>1 singularities
    // (e.g., disconnected ridge links in 3D when a k=2 flip inserts an already-existing edge).
    //
    // For facets (k==D) and full simplices (k==D+1), this is already covered by the existing
    // non-manifold facet / duplicate-simplex checks.
    if k_move >= 2
        && k_move < D
        && let Some(existing_simplex) =
            find_simplex_containing_simplex(tds, inserted_face_vertices, removed_simplices)?
    {
        if repair_trace_enabled() || env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
            tracing::debug!(
                "[repair] skip flip: inserted simplex already exists (k={k_move}, inserted_face={inserted_face_vertices:?}, existing_simplex={existing_simplex:?})"
            );
        }
        return Err(FlipError::InsertedSimplexAlreadyExists {
            k_move,
            simplex_vertices: Box::new(inserted_face_vertices.iter().copied().collect()),
            existing_simplex,
        });
    }

    let mut new_simplex_vertices =
        ReplacementSimplexVertices::with_capacity(removed_face_vertices.len());

    for &omit in removed_face_vertices {
        let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        vertices.extend_from_slice(inserted_face_vertices);
        for &v in removed_face_vertices {
            if v != omit {
                vertices.push(v);
            }
        }

        new_simplex_vertices.push(vertices);
    }

    let boundary_facets = extract_cavity_boundary(tds, removed_simplices).map_err(|source| {
        FlipError::from(FlipNeighborWiringError::BoundaryExtraction { source })
    })?;

    let external_facets = external_facets_for_boundary(tds, removed_simplices, &boundary_facets)
        .map_err(FlipNeighborWiringError::from)?;

    let topology_index = build_flip_topology_index(
        tds,
        &new_simplex_vertices,
        removed_simplices,
        inserted_face_vertices,
    );

    for vertices in &mut new_simplex_vertices {
        if flip_would_duplicate_simplex_any(tds, vertices, &topology_index) {
            return Err(FlipError::DuplicateSimplex);
        }
        if flip_would_create_nonmanifold_facets_any(vertices, &topology_index) {
            return Err(FlipError::NonManifoldFacet);
        }

        let points = vertices_to_points(tds, vertices)?;

        // Exact orientation: reject degenerate simplices and canonicalize to
        // positive orientation in one pass.  This function uses
        // robust_orientation (exact arithmetic, no SoS) rather than any
        // kernel predicate, so it is kernel-independent.
        match robust_orientation(&points) {
            Err(e) => {
                return Err(FlipPredicateError::coordinate_conversion(
                    FlipPredicateOperation::ReplacementSimplexOrientation,
                    e,
                )
                .into());
            }
            Ok(Orientation::DEGENERATE) => {
                if env::var_os("DELAUNAY_REPAIR_DEBUG_FACETS").is_some() {
                    tracing::debug!(
                        k_move,
                        direction = ?direction,
                        removed_face = ?removed_face_vertices,
                        inserted_face = ?inserted_face_vertices,
                        vertices = ?vertices,
                        "[repair] flip degenerate simplex (exact)"
                    );
                }
                return Err(FlipError::DegenerateSimplex);
            }
            Ok(Orientation::NEGATIVE) => {
                // Canonicalize to positive orientation by swapping two vertices.
                vertices.swap(0, 1);
            }
            Ok(Orientation::POSITIVE) => {}
        }
    }

    let newly_inserted_vertex = if k_move == 1 {
        inserted_face_vertices.first().copied()
    } else {
        None
    };
    let mut new_simplex_offsets = replacement_simplex_periodic_offsets(
        tds,
        &new_simplex_vertices,
        removed_simplices,
        &external_facets,
        newly_inserted_vertex,
    )?;

    orient_replacement_simplices(
        tds,
        &mut new_simplex_vertices,
        &mut new_simplex_offsets,
        &external_facets,
    )?;
    if matches!(
        orientation_policy,
        ReplacementOrientationPolicy::RequirePositive
    ) {
        validate_replacement_orientation(tds, &new_simplex_vertices)?;
    }

    // Snapshot the removed simplices' vertex lists before any TDS mutation so an
    // unexpected missing simplex aborts without leaving replacement simplices behind.
    // After `tds.remove_simplices_by_keys` runs, `tds.simplex(removed_key)` returns
    // `None`, which would strip the most useful context from predecessor-flip
    // traces (see #204 investigation).
    let removed_simplex_vertices = snapshot_removed_simplex_vertices(tds, removed_simplices)?;
    let inserted_face_vertex_list: VertexKeyList = inserted_face_vertices.iter().copied().collect();

    Ok(PreparedFlip {
        kind: BistellarFlipKind::from_validated(k_move, D),
        direction,
        removed_simplices: removed_simplices.iter().copied().collect(),
        removed_face_vertices: removed_face_vertices.iter().copied().collect(),
        inserted_face_vertices: inserted_face_vertex_list,
        new_simplex_vertices,
        new_simplex_offsets,
        external_facets,
        removed_simplex_vertices,
    })
}

/// Shared implementation for failure-atomic bistellar mutation.
///
/// The original TDS is mutated inside the shared rollback transaction and is
/// committed only after the replacement cavity has been fully rewired and
/// locally validated.
#[expect(
    clippy::too_many_arguments,
    reason = "Flip mutation needs explicit move, cavity, policy, and validation inputs"
)]
pub(super) fn apply_bistellar_flip_with_k_inner<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    k_move: usize,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
    removed_simplices: &SimplexKeyBuffer,
    direction: FlipDirection,
    orientation_policy: ReplacementOrientationPolicy,
    validation_scope: FlipValidationScope,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    let PreparedFlip {
        kind,
        direction,
        removed_simplices,
        removed_face_vertices,
        inserted_face_vertices,
        new_simplex_vertices,
        new_simplex_offsets,
        external_facets,
        removed_simplex_vertices,
    } = prepare_bistellar_flip(
        tds,
        k_move,
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction,
        orientation_policy,
    )?;

    let mut transaction = TdsRollbackTransaction::begin(tds);
    let new_simplices = apply_prepared_flip_mutation(
        transaction.tds_mut(),
        new_simplex_vertices,
        new_simplex_offsets,
        &external_facets,
        &removed_simplices,
        k_move,
        direction,
        validation_scope,
    )?;
    transaction.commit();

    Ok(AppliedFlip {
        info: FlipInfo {
            kind,
            direction,
            removed_simplices,
            new_simplices,
            removed_face_vertices,
            inserted_face_vertices,
        },
        removed_simplex_vertices,
    })
}

/// Mutates an already-prepared flip cavity inside the caller's rollback window.
///
/// This helper exists so public, transaction-backed flips and raw Pachner
/// primitives share the same insertion, neighbor wiring, removal, and
/// post-mutation validation sequence without each owning its own snapshot.
#[expect(
    clippy::too_many_arguments,
    reason = "Prepared flip mutation needs explicit replacement storage and validation policy"
)]
pub(super) fn apply_prepared_flip_mutation<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    new_simplex_vertices: ReplacementSimplexVertices,
    new_simplex_offsets: ReplacementPeriodicOffsets<D>,
    external_facets: &[FacetHandle],
    removed_simplices: &SimplexKeyBuffer,
    k_move: usize,
    direction: FlipDirection,
    validation_scope: FlipValidationScope,
) -> Result<SimplexKeyBuffer, FlipError>
where
    U: DataType,
    V: DataType,
{
    let mut new_simplices = SimplexKeyBuffer::new();

    for (vertices, periodic_offsets) in new_simplex_vertices.into_iter().zip(new_simplex_offsets) {
        let mut simplex = Simplex::try_new(vertices)?;
        if let Some(offsets) = periodic_offsets {
            simplex.set_periodic_vertex_offsets(offsets)?;
        }
        let simplex_key = tds
            .insert_simplex_with_mapping_prechecked_topology(simplex)
            .map_err(|source| FlipMutationError::SimplexInsertion {
                source: source.into(),
            })?;
        new_simplices.push(simplex_key);
    }

    wire_cavity_neighbors(
        tds,
        &new_simplices,
        external_facets.iter().copied(),
        Some(removed_simplices),
    )
    .map_err(FlipNeighborWiringError::from)?;

    tds.remove_simplices_by_keys(removed_simplices)
        .map_err(|source| FlipError::from(FlipMutationError::SimplexRemoval { source }))?;

    let validation_result = match validation_scope {
        FlipValidationScope::FullTds => tds.is_valid().map_err(TdsValidationFailure::from),
        FlipValidationScope::LocalCavity => {
            validate_flip_trial_cavity(tds, &new_simplices, external_facets, removed_simplices)
        }
    };
    validation_result.map_err(|source| {
        FlipError::from(FlipMutationError::TrialValidation {
            k_move,
            direction,
            source,
        })
    })?;

    #[cfg(debug_assertions)]
    {
        // This is intentionally debug/test-only for the same reason as the
        // pre-flip scan above: production validation already checks coherent
        // orientation at explicit validation boundaries.
        if !tds.is_coherently_oriented() {
            return Err(FlipError::from(
                FlipMutationError::CoherentOrientationViolation {
                    stage: FlipOrientationCheckStage::AfterTrialMutation,
                    k_move,
                    direction,
                },
            ));
        }
    }

    Ok(new_simplices)
}

/// Selects whether a flip is only topological or must preserve Delaunay geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReplacementOrientationPolicy {
    /// Allow coherent replacement simplices regardless of geometric sign.
    AllowSigned,
    /// Require replacement simplices to stay in positive canonical orientation.
    RequirePositive,
}

/// Selects the amount of TDS structure checked before committing a flip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FlipValidationScope {
    /// Validate the whole triangulation data structure.
    FullTds,
    /// Validate only the simplices whose adjacency can change during a cavity flip.
    LocalCavity,
}

/// Checks the flip cavity after mutation without rescanning the full TDS.
pub(super) fn validate_flip_trial_cavity<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    new_simplices: &[SimplexKey],
    external_facets: &[FacetHandle],
    removed_simplices: &[SimplexKey],
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    for &simplex_key in removed_simplices {
        if tds.contains_simplex(simplex_key) {
            return Err(TdsValidationFailure::InconsistentDataStructure {
                message: format!("flip trial still contains removed simplex {simplex_key:?}"),
            });
        }
        if tds.simplex_uuid_from_key(simplex_key).is_some() {
            return Err(TdsValidationFailure::MappingInconsistency {
                entity: EntityKind::Simplex,
                message: format!("flip trial still maps removed simplex key {simplex_key:?}"),
            });
        }
    }

    let mut affected_simplices = SimplexKeyBuffer::new();
    let mut affected_set = FastHashSet::default();
    for &simplex_key in new_simplices {
        push_unique_simplex_key(simplex_key, &mut affected_simplices, &mut affected_set);
    }
    for facet in external_facets {
        push_unique_simplex_key(
            facet.simplex_key(),
            &mut affected_simplices,
            &mut affected_set,
        );
    }

    validate_flip_trial_local_facet_sharing(tds, &affected_simplices)?;

    for &simplex_key in &affected_simplices {
        validate_flip_trial_simplex(tds, simplex_key, removed_simplices)?;
    }

    Ok(())
}

/// Adds a simplex to a small worklist while preserving first-seen order.
pub(super) fn push_unique_simplex_key(
    simplex_key: SimplexKey,
    simplices: &mut SimplexKeyBuffer,
    seen: &mut FastHashSet<SimplexKey>,
) {
    if seen.insert(simplex_key) {
        simplices.push(simplex_key);
    }
}

/// Ensures affected replacement simplices agree on shared facets and multiplicity.
pub(super) fn validate_flip_trial_local_facet_sharing<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    affected_simplices: &[SimplexKey],
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    type FacetIncidents = SmallBuffer<(SimplexKey, u8), 2>;
    let mut facet_to_simplices: FastHashMap<u64, FacetIncidents> = FastHashMap::default();

    for &simplex_key in affected_simplices {
        let simplex =
            tds.simplex(simplex_key)
                .ok_or_else(|| TdsValidationFailure::SimplexNotFound {
                    simplex_key,
                    context: "flip trial local facet sharing".to_string(),
                })?;
        if simplex.number_of_vertices() != D + 1 {
            return Err(TdsValidationFailure::DimensionMismatch {
                expected: D + 1,
                actual: simplex.number_of_vertices(),
                context: format!("flip trial simplex {simplex_key:?} arity"),
            });
        }

        for facet_idx in 0..simplex.number_of_vertices() {
            let facet_vertices = facet_vertices_from_simplex(simplex, facet_idx);
            let facet_idx_u8 =
                u8::try_from(facet_idx).map_err(|_| TdsValidationFailure::IndexOutOfBounds {
                    index: facet_idx,
                    bound: usize::from(u8::MAX),
                    context: "flip trial facet index".to_string(),
                })?;
            facet_to_simplices
                .entry(facet_key_from_vertices(&facet_vertices))
                .or_default()
                .push((simplex_key, facet_idx_u8));
        }
    }

    for (facet_key, incidents) in facet_to_simplices {
        match incidents.as_slice() {
            [_] => {}
            [(simplex_a, facet_a), (simplex_b, facet_b)] => {
                validate_flip_trial_mutual_facet_neighbors(
                    tds,
                    facet_key,
                    *simplex_a,
                    usize::from(*facet_a),
                    *simplex_b,
                    usize::from(*facet_b),
                )?;
            }
            _ => {
                return Err(TdsValidationFailure::Facet {
                    source: FacetError::InvalidFacetMultiplicity {
                        facet_key,
                        found: incidents.len(),
                    },
                });
            }
        }
    }

    Ok(())
}

/// Checks one affected simplex's local references after a flip mutation.
pub(super) fn validate_flip_trial_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    removed_simplices: &[SimplexKey],
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    let simplex =
        tds.simplex(simplex_key)
            .ok_or_else(|| TdsValidationFailure::SimplexNotFound {
                simplex_key,
                context: "flip trial local simplex validation".to_string(),
            })?;
    if tds.simplex_uuid_from_key(simplex_key) != Some(simplex.uuid()) {
        return Err(TdsValidationFailure::MappingInconsistency {
            entity: EntityKind::Simplex,
            message: format!(
                "missing or inconsistent UUID mapping for flip trial simplex {simplex_key:?}"
            ),
        });
    }

    if simplex.number_of_vertices() != D + 1 {
        return Err(TdsValidationFailure::DimensionMismatch {
            expected: D + 1,
            actual: simplex.number_of_vertices(),
            context: format!("flip trial simplex {simplex_key:?} arity"),
        });
    }

    validate_flip_trial_simplex_vertices(tds, simplex_key, simplex)?;
    validate_flip_trial_simplex_neighbors(tds, simplex_key, simplex, removed_simplices)
}

/// Verifies that affected simplices reference existing vertices with valid incidence.
pub(super) fn validate_flip_trial_simplex_vertices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    let mut seen_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(simplex.number_of_vertices());
    for &vertex_key in simplex.vertices() {
        if seen_vertices.contains(&vertex_key) {
            return Err(TdsValidationFailure::InconsistentDataStructure {
                message: format!(
                    "flip trial simplex {simplex_key:?} repeats vertex {vertex_key:?}"
                ),
            });
        }
        seen_vertices.push(vertex_key);

        let vertex =
            tds.vertex(vertex_key)
                .ok_or_else(|| TdsValidationFailure::VertexNotFound {
                    vertex_key,
                    context: format!("flip trial simplex {simplex_key:?} vertex reference"),
                })?;
        if tds.vertex_uuid_from_key(vertex_key) != Some(vertex.uuid()) {
            return Err(TdsValidationFailure::MappingInconsistency {
                entity: EntityKind::Vertex,
                message: format!(
                    "missing or inconsistent UUID mapping for flip trial vertex {vertex_key:?}"
                ),
            });
        }
        let Some(incident_simplex_key) = vertex.incident_simplex() else {
            continue;
        };
        let incident_simplex = tds.simplex(incident_simplex_key).ok_or_else(|| {
            TdsValidationFailure::SimplexNotFound {
                simplex_key: incident_simplex_key,
                context: format!("dangling incident_simplex pointer from vertex {vertex_key:?}"),
            }
        })?;
        if !incident_simplex.contains_vertex(vertex_key) {
            return Err(TdsValidationFailure::InconsistentDataStructure {
                message: format!(
                    "Vertex {vertex_key:?} incident_simplex {incident_simplex_key:?} does not contain the vertex"
                ),
            });
        }
    }

    Ok(())
}

/// Verifies affected-simplex neighbor links, mirror facets, and orientation parity.
pub(super) fn validate_flip_trial_simplex_neighbors<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    removed_simplices: &[SimplexKey],
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    let Some(neighbors) = simplex.neighbor_slots() else {
        return Ok(());
    };
    if neighbors.len() != D + 1 {
        return Err(TdsValidationFailure::InvalidNeighbors {
            reason: NeighborValidationError::LengthMismatch {
                actual: neighbors.len(),
                expected: D + 1,
                context: "flip trial neighbor validation".to_string(),
            },
        });
    }

    for (facet_idx, neighbor_slot) in neighbors.iter().copied().enumerate() {
        let neighbor_key = match neighbor_slot {
            NeighborSlot::Unassigned => {
                return Err(TdsValidationFailure::InvalidNeighbors {
                    reason: NeighborValidationError::UnassignedNeighborSlot {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        context: "flip trial neighbor validation".to_string(),
                    },
                });
            }
            NeighborSlot::Boundary => continue,
            NeighborSlot::Neighbor(neighbor_key) => neighbor_key,
        };
        if removed_simplices.contains(&neighbor_key) {
            return Err(TdsValidationFailure::InvalidNeighbors {
                reason: NeighborValidationError::ReferencedRemovedNeighbor {
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_key,
                },
            });
        }
        if neighbor_key == simplex_key {
            if simplex_allows_periodic_self_neighbor(simplex) {
                continue;
            }
            return Err(TdsValidationFailure::InvalidNeighbors {
                reason: NeighborValidationError::NonPeriodicSelfNeighbor {
                    simplex_key,
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                },
            });
        }

        let neighbor_simplex =
            tds.simplex(neighbor_key)
                .ok_or_else(|| TdsValidationFailure::InvalidNeighbors {
                    reason: NeighborValidationError::MissingNeighborSimplex {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        neighbor_key,
                        context: "flip trial neighbor validation".to_string(),
                    },
                })?;
        let mirror_idx = simplex
            .mirror_facet_index(facet_idx, neighbor_simplex)
            .ok_or_else(|| TdsValidationFailure::InvalidNeighbors {
                reason: NeighborValidationError::MirrorFacetMissing {
                    simplex_uuid: simplex.uuid(),
                    facet_index: facet_idx,
                    neighbor_uuid: neighbor_simplex.uuid(),
                    context: "flip trial neighbor validation".to_string(),
                },
            })?;
        validate_flip_trial_mutual_facet_neighbors(
            tds,
            facet_key_from_vertices(&facet_vertices_from_simplex(simplex, facet_idx)),
            simplex_key,
            facet_idx,
            neighbor_key,
            mirror_idx,
        )?;
        validate_flip_trial_neighbor_orientation(
            simplex_key,
            simplex,
            facet_idx,
            neighbor_key,
            neighbor_simplex,
            mirror_idx,
        )?;
    }

    Ok(())
}

/// Mirrors TDS validation's periodic self-neighbor allowance locally.
pub(super) fn simplex_allows_periodic_self_neighbor<V, const D: usize>(
    simplex: &Simplex<V, D>,
) -> bool {
    let Some(offsets) = simplex.periodic_vertex_offsets() else {
        return false;
    };
    !offsets.is_empty() && offsets.len() == simplex.number_of_vertices()
}

/// Requires two simplices sharing an affected facet to point back to each other.
pub(super) fn validate_flip_trial_mutual_facet_neighbors<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_key: u64,
    source_simplex_key: SimplexKey,
    source_facet: usize,
    target_simplex_key: SimplexKey,
    target_facet: usize,
) -> Result<(), TdsValidationFailure>
where
    U: DataType,
    V: DataType,
{
    let source_simplex =
        tds.simplex(source_simplex_key)
            .ok_or_else(|| TdsValidationFailure::SimplexNotFound {
                simplex_key: source_simplex_key,
                context: "flip trial mutual neighbor validation".to_string(),
            })?;
    let target_simplex =
        tds.simplex(target_simplex_key)
            .ok_or_else(|| TdsValidationFailure::SimplexNotFound {
                simplex_key: target_simplex_key,
                context: "flip trial mutual neighbor validation".to_string(),
            })?;

    let source_neighbor = source_simplex.neighbor_key(source_facet).flatten();
    let target_neighbor = target_simplex.neighbor_key(target_facet).flatten();

    if source_neighbor != Some(target_simplex_key) || target_neighbor != Some(source_simplex_key) {
        return Err(TdsValidationFailure::InvalidNeighbors {
            reason: NeighborValidationError::InteriorFacetNeighborMismatch {
                facet_key,
                first_simplex_key: source_simplex_key,
                first_simplex_uuid: source_simplex.uuid(),
                first_facet_index: source_facet,
                first_neighbor: source_neighbor,
                second_simplex_key: target_simplex_key,
                second_simplex_uuid: target_simplex.uuid(),
                second_facet_index: target_facet,
                second_neighbor: target_neighbor,
            },
        });
    }

    Ok(())
}

/// Checks coherent orientation across one locally affected neighbor pair.
pub(super) fn validate_flip_trial_neighbor_orientation<V, const D: usize>(
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    facet_idx: usize,
    neighbor_key: SimplexKey,
    neighbor_simplex: &Simplex<V, D>,
    mirror_idx: usize,
) -> Result<(), TdsValidationFailure> {
    let (observed_odd_permutation, expected_odd_permutation, facet_vertex_count, target_count) =
        match flip_trial_neighbor_orientation_parity(
            simplex_key,
            simplex,
            facet_idx,
            neighbor_key,
            neighbor_simplex,
            mirror_idx,
        ) {
            Ok(parity) => parity,
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::FacetOrderParityUnavailable
                ) =>
            {
                return Err(TdsValidationFailure::InconsistentDataStructure {
                    message: format!(
                        "Could not derive facet-order permutation parity between simplices {:?} and {:?}",
                        simplex.uuid(),
                        neighbor_simplex.uuid()
                    ),
                });
            }
            Err(err) => {
                return Err(TdsValidationFailure::InvalidNeighbors {
                    reason: NeighborValidationError::FacetOrderUnavailable {
                        simplex_key,
                        simplex_uuid: simplex.uuid(),
                        facet_index: facet_idx,
                        context: "facet parity in local flip validation".to_string(),
                        source: Box::new(err),
                    },
                });
            }
        };
    if observed_odd_permutation != expected_odd_permutation {
        return Err(TdsValidationFailure::OrientationViolation {
            simplex1_key: simplex_key,
            simplex1_uuid: simplex.uuid(),
            simplex2_key: neighbor_key,
            simplex2_uuid: neighbor_simplex.uuid(),
            simplex1_facet_index: facet_idx,
            simplex2_facet_index: mirror_idx,
            facet_vertex_count,
            simplex2_facet_vertex_count: target_count,
            observed_odd_permutation,
            expected_odd_permutation,
        });
    }

    Ok(())
}

/// Computes local neighbor-orientation parity, including periodic facet offsets.
pub(super) fn flip_trial_neighbor_orientation_parity<V, const D: usize>(
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    facet_idx: usize,
    neighbor_key: SimplexKey,
    neighbor_simplex: &Simplex<V, D>,
    mirror_idx: usize,
) -> Result<(bool, bool, usize, usize), FlipError> {
    let expected_odd_permutation = (facet_idx + mirror_idx).is_multiple_of(2);
    if simplex.periodic_vertex_offsets().is_some()
        || neighbor_simplex.periodic_vertex_offsets().is_some()
    {
        let source_offsets = periodic_offsets_or_zero_frame(simplex_key, simplex)?;
        let target_offsets = periodic_offsets_or_zero_frame(neighbor_key, neighbor_simplex)?;
        let source_order = normalized_facet_order_with_offsets(
            simplex_key,
            simplex.vertices(),
            source_offsets.as_ref(),
            facet_idx,
        )?;
        let target_order = normalized_facet_order_with_offsets(
            neighbor_key,
            neighbor_simplex.vertices(),
            target_offsets.as_ref(),
            mirror_idx,
        )?;
        let observed_odd_permutation = permutation_odd(&source_order, &target_order)
            .ok_or(FlipContextError::FacetOrderParityUnavailable)?;
        return Ok((
            observed_odd_permutation,
            expected_odd_permutation,
            source_order.len(),
            target_order.len(),
        ));
    }

    let source_order = facet_order(simplex.vertices(), facet_idx)?;
    let target_order = facet_order(neighbor_simplex.vertices(), mirror_idx)?;
    let observed_odd_permutation = permutation_odd(&source_order, &target_order)
        .ok_or(FlipContextError::FacetOrderParityUnavailable)?;
    Ok((
        observed_odd_permutation,
        expected_odd_permutation,
        source_order.len(),
        target_order.len(),
    ))
}

/// Detects replacement simplices that already exist outside the flip cavity.
///
/// This protects the bistellar link condition while also treating stale
/// incidence entries as structural corruption instead of silently ignoring
/// them.
///
/// # Errors
///
/// Returns [`FlipError::DanglingVertexIncidence`] if
/// [`Tds::simplex_keys_containing_vertex`] yields a simplex key that is no
/// longer present in storage.
pub(super) fn find_simplex_containing_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_vertices: &[VertexKey],
    removed_simplices: &[SimplexKey],
) -> Result<Option<SimplexKey>, FlipError> {
    let Some(&first) = simplex_vertices.first() else {
        return Ok(None);
    };

    for simplex_key in tds.simplex_keys_containing_vertex(first) {
        let Some(simplex) = tds.simplex(simplex_key) else {
            return Err(FlipError::DanglingVertexIncidence {
                vertex_key: first,
                simplex_key,
            });
        };

        if removed_simplices.contains(&simplex_key) {
            continue;
        }

        if simplex_vertices
            .iter()
            .copied()
            .all(|vk| simplex.contains_vertex(vk))
        {
            return Ok(Some(simplex_key));
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::init_tracing;
    use super::super::*;
    use super::*;
    use crate::core::algorithms::insertion::repair_neighbor_pointers;
    use crate::core::collections::Uuid;
    use crate::vertex;
    use proptest::prelude::*;
    use slotmap::KeyData;
    use std::assert_matches;
    use std::iter::once;
    /// Builds a simplex-basis vertex coordinate for dimension-generic flip tests.
    fn unit_vector<const D: usize>(index: usize) -> [f64; D] {
        let mut coords = [0.0; D];
        coords[index] = 1.0;
        coords
    }
    fn insert_standard_simplex_vertices<const D: usize>(
        tds: &mut Tds<(), (), D>,
    ) -> Vec<VertexKey> {
        let mut vertices = Vec::with_capacity(D + 1);
        vertices.push(
            tds.insert_vertex_with_mapping(vertex!([0.0; D]).unwrap())
                .unwrap(),
        );
        for axis in 0..D {
            vertices.push(
                tds.insert_vertex_with_mapping(vertex!(unit_vector::<D>(axis)).unwrap())
                    .unwrap(),
            );
        }
        vertices
    }
    /// Asserts exact vertex-to-periodic-offset slot pairing independent of simplex orientation.
    fn assert_simplex_offsets_by_vertex<const D: usize>(
        tds: &Tds<(), (), D>,
        simplex_key: SimplexKey,
        expected_offsets: &[(VertexKey, [i8; D])],
    ) {
        let simplex = tds.simplex(simplex_key).unwrap();
        let offsets = simplex
            .periodic_vertex_offsets()
            .expect("simplex should carry periodic offsets");
        assert_eq!(offsets.len(), simplex.number_of_vertices());
        assert_eq!(expected_offsets.len(), simplex.number_of_vertices());
        for &(vertex_key, expected_offset) in expected_offsets {
            let index = simplex
                .vertices()
                .iter()
                .position(|&candidate| candidate == vertex_key)
                .expect("expected vertex should be present in simplex");
            assert_eq!(
                offsets[index], expected_offset,
                "unexpected periodic offset for vertex {vertex_key:?} in simplex {simplex_key:?}"
            );
        }
    }

    /// Creates a non-axis-aligned point for high-dimensional roundtrip fixtures.
    fn skewed_point<const D: usize>() -> [f64; D] {
        let mut coords = [0.0; D];
        for (i, coord) in coords.iter_mut().enumerate().take(D) {
            let idx = f64::from(u32::try_from(i + 1).expect("index fits in u32"));
            *coord = 0.11 * idx;
        }
        coords
    }

    /// Builds a translated and scaled simplex-basis coordinate for proptests.
    fn translated_scaled_unit_vector<const D: usize>(
        index: usize,
        offset: f64,
        scale: f64,
    ) -> [f64; D] {
        let mut coords = [offset; D];
        coords[index] += scale;
        coords
    }

    /// Creates a translated non-axis-aligned point for k=3 flip proptests.
    fn translated_scaled_skewed_point<const D: usize>(offset: f64, scale: f64) -> [f64; D] {
        let mut coords = [offset; D];
        for (i, coord) in coords.iter_mut().enumerate().take(D) {
            let idx = f64::from(u32::try_from(i + 1).expect("index fits in u32"));
            *coord = (scale * 0.11).mul_add(idx, *coord);
        }
        coords
    }

    /// Returns inserted-face vertices after verifying the expected flip arity.
    fn inserted_face_vertices<const D: usize>(
        info: &FlipInfo<D>,
        expected: usize,
    ) -> Result<Vec<VertexKey>, TestCaseError> {
        let vertices: Vec<_> = info.inserted_face_vertices.iter().copied().collect();
        if vertices.len() != expected {
            return Err(TestCaseError::fail(format!(
                "flip reported {} inserted-face vertices, expected {expected}",
                vertices.len()
            )));
        }
        Ok(vertices)
    }
    macro_rules! gen_removed_simplex_snapshot_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_snapshot_removed_simplex_vertices_captures_vertices_and_reports_missing_simplex_ $dim d>]() {
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let vertices = insert_standard_simplex_vertices::<$dim>(&mut tds);
                    let simplex_key = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(vertices.clone(), None).unwrap())
                        .unwrap();

                    let removed_simplices: SimplexKeyBuffer = once(simplex_key).collect();
                    let snapshot = snapshot_removed_simplex_vertices(&tds, &removed_simplices).unwrap();
                    assert_eq!(snapshot.len(), 1);
                    assert_eq!(snapshot[0].iter().copied().collect::<Vec<_>>(), vertices);

                    let missing_simplex = SimplexKey::from(KeyData::from_ffi(999_999 + $dim));
                    let missing_simplices: SimplexKeyBuffer = once(missing_simplex).collect();
                    let err = snapshot_removed_simplex_vertices(&tds, &missing_simplices).unwrap_err();
                    assert_matches!(
                        err,
                        FlipError::MissingSimplex { simplex_key } if simplex_key == missing_simplex
                    );
                }

                #[test]
                fn [<test_last_applied_flip_preserves_removed_simplex_vertex_snapshots_ $dim d>]() {
                    let removed_simplex = SimplexKey::from(KeyData::from_ffi(101 + $dim));
                    let new_simplex = SimplexKey::from(KeyData::from_ffi(102 + $dim));
                    let v1 = VertexKey::from(KeyData::from_ffi(201 + $dim));
                    let v2 = VertexKey::from(KeyData::from_ffi(202 + $dim));
                    let v3 = VertexKey::from(KeyData::from_ffi(203 + $dim));
                    let v4 = VertexKey::from(KeyData::from_ffi(204 + $dim));

                    let mut removed_simplex_vertices = RemovedSimplexVertexSnapshot::new();
                    removed_simplex_vertices.push([v1, v2, v3].into_iter().collect::<VertexKeyList>());

                    let applied = AppliedFlip::<$dim> {
                        info: FlipInfo {
                            kind: BistellarFlipKind::from_validated(2, $dim),
                            direction: FlipDirection::Forward,
                            removed_simplices: once(removed_simplex).collect(),
                            new_simplices: once(new_simplex).collect(),
                            removed_face_vertices: [v3, v1].into_iter().collect(),
                            inserted_face_vertices: [v4, v2].into_iter().collect(),
                        },
                        removed_simplex_vertices,
                    };

                    let last = LastAppliedFlip::from_applied_flip(&applied);
                    assert_eq!(last.kind, BistellarFlipKind::from_validated(2, $dim));
                    assert_eq!(
                        last.removed_face_vertices
                            .iter()
                            .copied()
                            .collect::<Vec<_>>(),
                        vec![v1, v3]
                    );
                    assert_eq!(
                        last.inserted_face_vertices
                            .iter()
                            .copied()
                            .collect::<Vec<_>>(),
                        vec![v2, v4]
                    );
                    assert_eq!(
                        last.removed_simplices.iter().copied().collect::<Vec<_>>(),
                        vec![removed_simplex]
                    );
                    assert_eq!(
                        last.new_simplices.iter().copied().collect::<Vec<_>>(),
                        vec![new_simplex]
                    );

                    let lines = last.removed_simplex_vertex_lines();
                    assert_eq!(lines.len(), 1);
                    assert!(lines[0].contains(&format!("{removed_simplex:?}: vertices=")));
                    assert!(!lines[0].contains("missing-snapshot"));

                    let mut placeholder =
                        LastAppliedFlip::from_validated_flip_faces(
                            BistellarFlipKind::from_validated(2, $dim),
                            &[v1],
                            &[v2],
                        );
                    placeholder.removed_simplices.push(removed_simplex);
                    assert_eq!(
                        placeholder.removed_simplex_vertex_lines(),
                        vec![format!("{removed_simplex:?}: missing-snapshot")]
                    );
                }
            }
        };
    }

    gen_removed_simplex_snapshot_tests!(2);
    gen_removed_simplex_snapshot_tests!(3);
    gen_removed_simplex_snapshot_tests!(4);
    gen_removed_simplex_snapshot_tests!(5);
    fn facet_index_for_edge_2d(
        tds: &Tds<(), (), 2>,
        simplex_key: SimplexKey,
        edge_start: VertexKey,
        edge_end: VertexKey,
    ) -> u8 {
        let simplex = tds
            .simplex(simplex_key)
            .expect("simplex key missing in TDS");
        for facet_idx in 0..simplex.number_of_vertices() {
            let facet = facet_vertices_from_simplex(simplex, facet_idx);
            if facet.len() == 2 && facet.contains(&edge_start) && facet.contains(&edge_end) {
                return u8::try_from(facet_idx).expect("facet index fits in u8");
            }
        }

        panic!("edge ({edge_start:?}, {edge_end:?}) not found in simplex {simplex_key:?}");
    }
    #[test]
    fn test_resolve_facet_handle_for_key_remaps_after_slot_swap() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let stale_handle = FacetHandle::from_validated(simplex_key, 0);
        let stable_key = {
            let simplex = tds.simplex(simplex_key).unwrap();
            let facet_vertices =
                facet_vertices_from_simplex(simplex, usize::from(stale_handle.facet_index()));
            facet_key_from_vertices(&facet_vertices)
        };

        // Reorder slots so the original index no longer identifies the same facet.
        tds.simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 1);

        let resolved = resolve_facet_handle_for_key(&tds, stale_handle, stable_key)
            .expect("facet handle should be recoverable by stable key");
        assert_eq!(resolved.simplex_key(), simplex_key);
        assert_eq!(usize::from(resolved.facet_index()), 1);

        let resolved_key = {
            let simplex = tds.simplex(simplex_key).unwrap();
            let facet_vertices =
                facet_vertices_from_simplex(simplex, usize::from(resolved.facet_index()));
            facet_key_from_vertices(&facet_vertices)
        };
        assert_eq!(resolved_key, stable_key);
    }

    #[test]
    fn test_resolve_ridge_handle_for_key_remaps_after_slot_swap() {
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]).unwrap())
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let stale_handle = RidgeHandle::from_validated(simplex_key, 0, 1);
        let stable_key = {
            let simplex = tds.simplex(simplex_key).unwrap();
            let ridge_vertices = ridge_vertices_from_simplex(
                simplex,
                usize::from(stale_handle.omit_a()),
                usize::from(stale_handle.omit_b()),
            );
            facet_key_from_vertices(&ridge_vertices)
        };

        // Reorder slots so the original omit pair no longer identifies the same ridge.
        tds.simplex_mut(simplex_key)
            .unwrap()
            .swap_vertex_slots(0, 2);

        let resolved = resolve_ridge_handle_for_key(&tds, stale_handle, stable_key)
            .expect("ridge handle should be recoverable by stable key");
        assert_eq!(resolved.simplex_key(), simplex_key);
        assert_eq!((resolved.omit_a(), resolved.omit_b()), (1, 2));

        let resolved_key = {
            let simplex = tds.simplex(simplex_key).unwrap();
            let ridge_vertices = ridge_vertices_from_simplex(
                simplex,
                usize::from(resolved.omit_a()),
                usize::from(resolved.omit_b()),
            );
            facet_key_from_vertices(&ridge_vertices)
        };
        assert_eq!(resolved_key, stable_key);
    }

    #[test]
    fn test_k2_flip_rewires_external_neighbors_across_cavity_boundary() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_left_bottom = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v_right_bottom = tds
            .insert_vertex_with_mapping(vertex!([2.0, 0.0]).unwrap())
            .unwrap();
        let v_left_top = tds
            .insert_vertex_with_mapping(vertex!([0.0, 2.0]).unwrap())
            .unwrap();
        let v_right_top = tds
            .insert_vertex_with_mapping(vertex!([2.0, 2.0]).unwrap())
            .unwrap();
        let v_external = tds
            .insert_vertex_with_mapping(vertex!([-1.0, 1.0]).unwrap())
            .unwrap();

        // Flip cavity: two triangles sharing the bottom edge.
        let simplex_cavity_left = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_left_bottom, v_right_bottom, v_left_top], None)
                    .unwrap(),
            )
            .unwrap();
        let simplex_cavity_right = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_right_bottom, v_left_bottom, v_right_top], None)
                    .unwrap(),
            )
            .unwrap();

        // External simplex glued along the left edge of the cavity.
        let simplex_external_left = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_left_bottom, v_left_top, v_external], None)
                    .unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();
        assert!(tds.is_valid().is_ok());

        let facet_idx_flip_edge =
            facet_index_for_edge_2d(&tds, simplex_cavity_left, v_left_bottom, v_right_bottom);
        let ctx = build_k2_flip_context(
            &tds,
            FacetHandle::from_validated(simplex_cavity_left, facet_idx_flip_edge),
        )
        .unwrap();

        let info = apply_bistellar_flip_raw(&mut tds, &ctx).unwrap();

        assert!(!tds.contains_simplex(simplex_cavity_left));
        assert!(!tds.contains_simplex(simplex_cavity_right));
        assert!(tds.contains_simplex(simplex_external_left));

        // External simplex must be rewired from the removed simplex to a newly inserted simplex.
        let facet_idx_glue_edge =
            facet_index_for_edge_2d(&tds, simplex_external_left, v_left_bottom, v_left_top);
        let external_simplex = tds.simplex(simplex_external_left).unwrap();
        let neighbor_key_glue = external_simplex
            .neighbor_key(usize::from(facet_idx_glue_edge))
            .expect("external neighbors should exist")
            .expect("external simplex should have a neighbor across the glue edge after the flip");

        assert!(tds.contains_simplex(neighbor_key_glue));
        assert!(
            info.new_simplices
                .iter()
                .copied()
                .any(|k| k == neighbor_key_glue),
            "expected external neighbor across glue edge to be one of the flip-inserted simplices"
        );

        // Neighbor relation must be symmetric.
        let neighbor_simplex = tds.simplex(neighbor_key_glue).unwrap();
        let mirror_idx = external_simplex
            .mirror_facet_index(usize::from(facet_idx_glue_edge), neighbor_simplex)
            .expect("mirror facet index should exist");
        let neighbor_back = neighbor_simplex.neighbor_key(mirror_idx).flatten();
        assert_eq!(neighbor_back, Some(simplex_external_left));

        // Ensure flip did not leave any dangling neighbor pointers in the newly inserted simplices.
        for &simplex_key in &info.new_simplices {
            let simplex = tds.simplex(simplex_key).unwrap();
            if let Some(ns) = simplex.neighbors() {
                for neighbor_key in ns.flatten() {
                    assert!(
                        tds.contains_simplex(neighbor_key),
                        "dangling neighbor pointer from {simplex_key:?} to {neighbor_key:?}"
                    );
                }
            }
        }

        assert!(tds.is_valid().is_ok());
    }

    fn insert_periodic_simplex_with_offsets<const D: usize>(
        tds: &mut Tds<(), (), D>,
        vertices: Vec<VertexKey>,
        offsets: Vec<[i8; D]>,
    ) -> SimplexKey {
        let mut simplex = Simplex::try_new_with_data(vertices, None).unwrap();
        simplex.set_periodic_vertex_offsets(offsets).unwrap();
        tds.insert_simplex_with_mapping(simplex).unwrap()
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "regression test keeps the periodic flip fixture explicit"
    )]
    fn test_k2_flip_preserves_periodic_external_offsets() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v_left_bottom = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v_right_bottom = tds
            .insert_vertex_with_mapping(vertex!([2.0, 0.0]).unwrap())
            .unwrap();
        let v_left_top = tds
            .insert_vertex_with_mapping(vertex!([0.0, 2.0]).unwrap())
            .unwrap();
        let v_right_top = tds
            .insert_vertex_with_mapping(vertex!([2.0, 2.0]).unwrap())
            .unwrap();
        let v_external = tds
            .insert_vertex_with_mapping(vertex!([-1.0, 1.0]).unwrap())
            .unwrap();

        let offset_left_bottom = [0_i8, 0_i8];
        let offset_right_bottom = [1_i8, 0_i8];
        let offset_left_top = [0_i8, 1_i8];
        let offset_right_top = [1_i8, 1_i8];
        let offset_external = [0_i8, -1_i8];
        let simplex_cavity_left = insert_periodic_simplex_with_offsets(
            &mut tds,
            vec![v_left_bottom, v_right_bottom, v_left_top],
            vec![offset_left_bottom, offset_right_bottom, offset_left_top],
        );
        let simplex_cavity_right = insert_periodic_simplex_with_offsets(
            &mut tds,
            vec![v_right_bottom, v_left_bottom, v_right_top],
            vec![offset_right_bottom, offset_left_bottom, offset_right_top],
        );
        let simplex_external_left = insert_periodic_simplex_with_offsets(
            &mut tds,
            vec![v_left_bottom, v_left_top, v_external],
            vec![offset_left_bottom, offset_left_top, offset_external],
        );

        repair_neighbor_pointers(&mut tds).unwrap();
        assert!(tds.is_valid().is_ok());

        let facet_idx_flip_edge =
            facet_index_for_edge_2d(&tds, simplex_cavity_left, v_left_bottom, v_right_bottom);
        let ctx = build_k2_flip_context(
            &tds,
            FacetHandle::from_validated(simplex_cavity_left, facet_idx_flip_edge),
        )
        .unwrap();

        let info = apply_bistellar_flip_with_k(
            &mut tds,
            2,
            &ctx.removed_face_vertices,
            &ctx.inserted_face_vertices,
            &ctx.removed_simplices,
            ctx.direction,
            ReplacementOrientationPolicy::AllowSigned,
            FlipValidationScope::LocalCavity,
        )
        .unwrap()
        .info;

        assert!(!tds.contains_simplex(simplex_cavity_left));
        assert!(!tds.contains_simplex(simplex_cavity_right));
        assert!(tds.contains_simplex(simplex_external_left));
        let expected_left_replacement = [
            (v_left_bottom, offset_left_bottom),
            (v_left_top, offset_left_top),
            (v_right_top, offset_right_top),
        ];
        let expected_right_replacement = [
            (v_right_bottom, offset_right_bottom),
            (v_left_top, offset_left_top),
            (v_right_top, offset_right_top),
        ];
        for &simplex_key in &info.new_simplices {
            let simplex = tds.simplex(simplex_key).unwrap();
            let expected = if simplex.contains_vertex(v_left_bottom) {
                &expected_left_replacement
            } else {
                &expected_right_replacement
            };
            assert_simplex_offsets_by_vertex(&tds, simplex_key, expected);
        }

        let facet_idx_glue_edge =
            facet_index_for_edge_2d(&tds, simplex_external_left, v_left_bottom, v_left_top);
        let external_simplex = tds.simplex(simplex_external_left).unwrap();
        let neighbor_key_glue = external_simplex
            .neighbor_key(usize::from(facet_idx_glue_edge))
            .expect("external neighbors should exist")
            .expect("external simplex should have a replacement neighbor across the glue edge");
        assert!(
            info.new_simplices
                .iter()
                .copied()
                .any(|simplex_key| simplex_key == neighbor_key_glue),
            "expected periodic external facet to be wired to a flip replacement simplex"
        );
        assert_simplex_offsets_by_vertex(
            &tds,
            simplex_external_left,
            &[
                (v_left_bottom, offset_left_bottom),
                (v_left_top, offset_left_top),
                (v_external, offset_external),
            ],
        );
        assert_simplex_offsets_by_vertex(&tds, neighbor_key_glue, &expected_left_replacement);
        assert!(tds.is_valid().is_ok());
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TopologySnapshot {
        vertices: Vec<Uuid>,
        simplex_vertices: Vec<Vec<Uuid>>,
        simplex_neighbors: Vec<Vec<Option<Uuid>>>,
    }

    fn snapshot_topology<const D: usize>(tds: &Tds<(), (), D>) -> TopologySnapshot {
        let mut vertices: Vec<Uuid> = tds.vertices().map(|(_, vertex)| vertex.uuid()).collect();
        vertices.sort();

        let mut simplex_vertices: Vec<Vec<Uuid>> = tds
            .simplices()
            .map(|(_, simplex)| {
                let mut uuids: Vec<Uuid> = simplex
                    .vertices()
                    .iter()
                    .map(|&vkey| tds.vertex(vkey).expect("vertex key missing in TDS").uuid())
                    .collect();
                uuids.sort();
                uuids
            })
            .collect();
        simplex_vertices.sort();

        let simplex_neighbors = snapshot_neighbors(tds);

        TopologySnapshot {
            vertices,
            simplex_vertices,
            simplex_neighbors,
        }
    }

    fn snapshot_neighbors<const D: usize>(tds: &Tds<(), (), D>) -> Vec<Vec<Option<Uuid>>> {
        let mut simplex_neighbors: Vec<Vec<Option<Uuid>>> = tds
            .simplices()
            .map(|(_, simplex)| {
                let mut neighbors: Vec<Option<Uuid>> = simplex
                    .neighbors()
                    .map(|neighbor_keys| {
                        neighbor_keys
                            .map(|neighbor| {
                                neighbor.and_then(|neighbor_key| {
                                    tds.simplex(neighbor_key).map(Simplex::uuid)
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                neighbors.sort();
                neighbors
            })
            .collect();
        simplex_neighbors.sort();
        simplex_neighbors
    }

    fn assert_same_vertex_simplex_topology(actual: &TopologySnapshot, expected: &TopologySnapshot) {
        assert_eq!(actual.vertices, expected.vertices);
        assert_eq!(actual.simplex_vertices, expected.simplex_vertices);
    }

    #[test]
    fn test_flip_trial_validation_rejects_unassigned_neighbor_slot() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        tds.assign_neighbors().unwrap();

        {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            simplex.ensure_neighbors_buffer_mut()[0] = NeighborSlot::Unassigned;
        }

        let simplex = tds.simplex(simplex_key).unwrap();
        let err =
            validate_flip_trial_simplex_neighbors(&tds, simplex_key, simplex, &[]).unwrap_err();

        assert_matches!(
            err,
            TdsValidationFailure::InvalidNeighbors {
                reason: NeighborValidationError::UnassignedNeighborSlot { facet_index: 0, .. },
            }
        );
    }

    /// Checks that a k=2 flip and its inverse preserve topology in dimension `D`.
    #[expect(
        clippy::too_many_lines,
        reason = "The property fixture keeps k=2 setup, forward flip, inverse flip, and invariant checks together so failing cases shrink with full context."
    )]
    fn prop_bistellar_k2_roundtrip_for_dim<const D: usize>(
        offset: f64,
        scale: f64,
    ) -> Result<(), TestCaseError> {
        init_tracing();
        let mut tds: Tds<(), (), D> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(D);
        for i in 0..D {
            let vertex = tds
                .insert_vertex_with_mapping(
                    vertex!(translated_scaled_unit_vector::<D>(i, offset, scale)).unwrap(),
                )
                .map_err(|err| {
                    TestCaseError::fail(format!("shared vertex insertion failed: {err:?}"))
                })?;
            shared_vertices.push(vertex);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(vertex!([offset; D]).unwrap())
            .map_err(|err| TestCaseError::fail(format!("opposite A insertion failed: {err:?}")))?;
        let opposite_b = tds
            .insert_vertex_with_mapping(vertex!([offset + scale; D]).unwrap())
            .map_err(|err| TestCaseError::fail(format!("opposite B insertion failed: {err:?}")))?;

        let mut vertices_with_first_opposite = shared_vertices.clone();
        vertices_with_first_opposite.push(opposite_a);
        let simplex_a = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices_with_first_opposite, None).map_err(|err| {
                    TestCaseError::fail(format!("simplex A creation failed: {err:?}"))
                })?,
            )
            .map_err(|err| TestCaseError::fail(format!("simplex A insertion failed: {err:?}")))?;

        let mut vertices_with_second_opposite = shared_vertices.clone();
        vertices_with_second_opposite.push(opposite_b);
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vertices_with_second_opposite, None).map_err(|err| {
                TestCaseError::fail(format!("simplex B creation failed: {err:?}"))
            })?,
        )
        .map_err(|err| TestCaseError::fail(format!("simplex B insertion failed: {err:?}")))?;

        repair_neighbor_pointers(&mut tds)
            .map_err(|err| TestCaseError::fail(format!("neighbor repair failed: {err:?}")))?;

        let before = snapshot_topology(&tds);
        let facet = FacetHandle::from_validated(
            simplex_a,
            u8::try_from(D).map_err(|err| {
                TestCaseError::fail(format!("facet index conversion failed: {err:?}"))
            })?,
        );
        let context = build_k2_flip_context(&tds, facet)
            .map_err(|err| TestCaseError::fail(format!("k=2 context build failed: {err:?}")))?;
        let info = apply_bistellar_flip_raw(&mut tds, &context)
            .map_err(|err| TestCaseError::fail(format!("k=2 flip failed: {err:?}")))?;
        tds.is_valid()
            .map_err(|err| TestCaseError::fail(format!("post k=2 TDS invalid: {err:?}")))?;

        if D == 2 {
            let mut inverse_facet: Option<FacetHandle> = None;
            for &simplex_key in &info.new_simplices {
                let simplex = tds
                    .simplex(simplex_key)
                    .ok_or_else(|| TestCaseError::fail("new k=2 simplex missing"))?;
                if simplex.contains_vertex(opposite_a) && simplex.contains_vertex(opposite_b) {
                    let facet_index = simplex
                        .vertices()
                        .iter()
                        .position(|&vertex| vertex != opposite_a && vertex != opposite_b)
                        .ok_or_else(|| TestCaseError::fail("missing inverse k=2 facet vertex"))?;
                    inverse_facet = Some(FacetHandle::from_validated(
                        simplex_key,
                        u8::try_from(facet_index).map_err(|err| {
                            TestCaseError::fail(format!(
                                "inverse facet index conversion failed: {err:?}"
                            ))
                        })?,
                    ));
                    break;
                }
            }

            let facet =
                inverse_facet.ok_or_else(|| TestCaseError::fail("inverse k=2 facet not found"))?;
            let context_back = build_k2_flip_context(&tds, facet).map_err(|err| {
                TestCaseError::fail(format!("inverse k=2 context build failed: {err:?}"))
            })?;
            apply_bistellar_flip_raw(&mut tds, &context_back)
                .map_err(|err| TestCaseError::fail(format!("inverse k=2 flip failed: {err:?}")))?;
        } else {
            let inserted = inserted_face_vertices(&info, 2)?;
            let edge = match inserted.as_slice() {
                [a, b] => EdgeKey::from_validated_endpoints(*a, *b),
                _ => {
                    return Err(TestCaseError::fail(
                        "validated k=2 inserted-face arity changed",
                    ));
                }
            };
            let context_back = build_k2_flip_context_from_edge(&tds, edge).map_err(|err| {
                TestCaseError::fail(format!("inverse k=2 context build failed: {err:?}"))
            })?;
            apply_bistellar_flip_dynamic_raw(&mut tds, D, &context_back)
                .map_err(|err| TestCaseError::fail(format!("inverse k=2 flip failed: {err:?}")))?;
        }

        tds.is_valid()
            .map_err(|err| TestCaseError::fail(format!("post inverse k=2 TDS invalid: {err:?}")))?;
        let after = snapshot_topology(&tds);
        prop_assert_eq!(after.vertices, before.vertices);
        prop_assert_eq!(after.simplex_vertices, before.simplex_vertices);
        Ok(())
    }

    /// Checks that a k=3 flip and its inverse preserve topology in dimension `D`.
    #[expect(
        clippy::too_many_lines,
        reason = "The property fixture keeps k=3 setup, forward flip, inverse flip, and invariant checks together so failing cases shrink with full context."
    )]
    fn prop_bistellar_k3_roundtrip_for_dim<const D: usize>(
        offset: f64,
        scale: f64,
    ) -> Result<(), TestCaseError> {
        init_tracing();
        let ridge_vertex_count = D
            .checked_sub(1)
            .ok_or_else(|| TestCaseError::fail("k=3 fixture requires D >= 1"))?;
        let mut tds: Tds<(), (), D> = Tds::empty();
        let mut ridge_vertices = Vec::with_capacity(ridge_vertex_count);
        for i in 0..ridge_vertex_count {
            let vertex = tds
                .insert_vertex_with_mapping(
                    vertex!(translated_scaled_unit_vector::<D>(i, offset, scale)).unwrap(),
                )
                .map_err(|err| {
                    TestCaseError::fail(format!("ridge vertex insertion failed: {err:?}"))
                })?;
            ridge_vertices.push(vertex);
        }

        let a = tds
            .insert_vertex_with_mapping(vertex!([offset; D]).unwrap())
            .map_err(|err| TestCaseError::fail(format!("opposite A insertion failed: {err:?}")))?;
        let b = tds
            .insert_vertex_with_mapping(
                vertex!(translated_scaled_unit_vector::<D>(
                    ridge_vertex_count,
                    offset,
                    scale,
                ))
                .unwrap(),
            )
            .map_err(|err| TestCaseError::fail(format!("opposite B insertion failed: {err:?}")))?;
        let c = tds
            .insert_vertex_with_mapping(
                vertex!(translated_scaled_skewed_point::<D>(offset, scale)).unwrap(),
            )
            .map_err(|err| TestCaseError::fail(format!("opposite C insertion failed: {err:?}")))?;

        let mut first_vertices = ridge_vertices.clone();
        first_vertices.push(a);
        first_vertices.push(b);
        let first_simplex = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(first_vertices, None).map_err(
                |err| TestCaseError::fail(format!("simplex A creation failed: {err:?}")),
            )?)
            .map_err(|err| TestCaseError::fail(format!("simplex A insertion failed: {err:?}")))?;

        let mut second_vertices = ridge_vertices.clone();
        second_vertices.push(b);
        second_vertices.push(c);
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(second_vertices, None).map_err(|err| {
                TestCaseError::fail(format!("simplex B creation failed: {err:?}"))
            })?,
        )
        .map_err(|err| TestCaseError::fail(format!("simplex B insertion failed: {err:?}")))?;

        let mut third_vertices = ridge_vertices.clone();
        third_vertices.push(c);
        third_vertices.push(a);
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(third_vertices, None).map_err(|err| {
                TestCaseError::fail(format!("simplex C creation failed: {err:?}"))
            })?,
        )
        .map_err(|err| TestCaseError::fail(format!("simplex C insertion failed: {err:?}")))?;

        repair_neighbor_pointers(&mut tds)
            .map_err(|err| TestCaseError::fail(format!("neighbor repair failed: {err:?}")))?;

        let before = snapshot_topology(&tds);
        let ridge = RidgeHandle::from_validated(
            first_simplex,
            u8::try_from(ridge_vertex_count).map_err(|err| {
                TestCaseError::fail(format!("ridge index conversion failed: {err:?}"))
            })?,
            u8::try_from(D).map_err(|err| {
                TestCaseError::fail(format!("ridge index conversion failed: {err:?}"))
            })?,
        );
        let context = build_k3_flip_context(&tds, ridge)
            .map_err(|err| TestCaseError::fail(format!("k=3 context build failed: {err:?}")))?;
        let info = apply_bistellar_flip_raw(&mut tds, &context)
            .map_err(|err| TestCaseError::fail(format!("k=3 flip failed: {err:?}")))?;
        tds.is_valid()
            .map_err(|err| TestCaseError::fail(format!("post k=3 TDS invalid: {err:?}")))?;

        if D == 3 {
            let mut inverse_facet: Option<FacetHandle> = None;
            for &simplex_key in &info.new_simplices {
                let simplex = tds
                    .simplex(simplex_key)
                    .ok_or_else(|| TestCaseError::fail("new k=3 simplex missing"))?;
                if simplex.contains_vertex(a)
                    && simplex.contains_vertex(b)
                    && simplex.contains_vertex(c)
                {
                    let facet_index = simplex
                        .vertices()
                        .iter()
                        .position(|&vertex| vertex != a && vertex != b && vertex != c)
                        .ok_or_else(|| TestCaseError::fail("missing inverse k=3 facet vertex"))?;
                    inverse_facet = Some(FacetHandle::from_validated(
                        simplex_key,
                        u8::try_from(facet_index).map_err(|err| {
                            TestCaseError::fail(format!(
                                "inverse facet index conversion failed: {err:?}"
                            ))
                        })?,
                    ));
                    break;
                }
            }

            let facet =
                inverse_facet.ok_or_else(|| TestCaseError::fail("inverse k=3 facet not found"))?;
            let context_back = build_k2_flip_context(&tds, facet).map_err(|err| {
                TestCaseError::fail(format!("inverse k=3 context build failed: {err:?}"))
            })?;
            apply_bistellar_flip_raw(&mut tds, &context_back)
                .map_err(|err| TestCaseError::fail(format!("inverse k=3 flip failed: {err:?}")))?;
        } else {
            let inserted = inserted_face_vertices(&info, 3)?;
            let triangle = match inserted.as_slice() {
                [a, b, c] => TriangleHandle::try_new(*a, *b, *c).map_err(|err| {
                    TestCaseError::fail(format!("invalid inserted triangle: {err}"))
                })?,
                _ => {
                    return Err(TestCaseError::fail(
                        "validated k=3 inserted-face arity changed",
                    ));
                }
            };
            let context_back =
                build_k3_flip_context_from_triangle(&tds, triangle).map_err(|err| {
                    TestCaseError::fail(format!("inverse k=3 context build failed: {err:?}"))
                })?;
            apply_bistellar_flip_dynamic_raw(&mut tds, ridge_vertex_count, &context_back)
                .map_err(|err| TestCaseError::fail(format!("inverse k=3 flip failed: {err:?}")))?;
        }

        tds.is_valid()
            .map_err(|err| TestCaseError::fail(format!("post inverse k=3 TDS invalid: {err:?}")))?;
        let after = snapshot_topology(&tds);
        prop_assert_eq!(after.vertices, before.vertices);
        prop_assert_eq!(after.simplex_vertices, before.simplex_vertices);
        Ok(())
    }

    macro_rules! gen_bistellar_k2_roundtrip_properties {
        ($($dim:literal),+ $(,)?) => {
            pastey::paste! {
                $(
                    proptest! {
                        #![proptest_config(ProptestConfig::with_cases(16))]

                        #[test]
                        fn [<prop_bistellar_k2_roundtrip_ $dim d>](
                            offset in -2.0_f64..2.0,
                            scale in 0.5_f64..2.0,
                        ) {
                            prop_bistellar_k2_roundtrip_for_dim::<$dim>(offset, scale)?;
                        }
                    }
                )+
            }
        };
    }

    /// Exercises the 2D k=2 roundtrip fixture under non-proptest coverage runs.
    #[test]
    fn test_bistellar_k2_roundtrip_smoke_2d() {
        prop_bistellar_k2_roundtrip_for_dim::<2>(0.25, 1.0).unwrap();
    }

    /// Exercises the higher-dimensional k=2 inverse path under non-proptest coverage runs.
    #[test]
    fn test_bistellar_k2_roundtrip_smoke_4d() {
        prop_bistellar_k2_roundtrip_for_dim::<4>(-0.25, 1.25).unwrap();
    }

    macro_rules! gen_bistellar_k3_roundtrip_properties {
        ($($dim:literal),+ $(,)?) => {
            pastey::paste! {
                $(
                    proptest! {
                        #![proptest_config(ProptestConfig::with_cases(16))]

                        #[test]
                        fn [<prop_bistellar_k3_roundtrip_ $dim d>](
                            offset in -2.0_f64..2.0,
                            scale in 0.5_f64..2.0,
                        ) {
                            prop_bistellar_k3_roundtrip_for_dim::<$dim>(offset, scale)?;
                        }
                    }
                )+
            }
        };
    }

    /// Exercises the 3D k=3 roundtrip fixture under non-proptest coverage runs.
    #[test]
    fn test_bistellar_k3_roundtrip_smoke_3d() {
        prop_bistellar_k3_roundtrip_for_dim::<3>(0.25, 1.0).unwrap();
    }

    /// Exercises the higher-dimensional k=3 inverse path under non-proptest coverage runs.
    #[test]
    fn test_bistellar_k3_roundtrip_smoke_4d() {
        prop_bistellar_k3_roundtrip_for_dim::<4>(-0.25, 1.25).unwrap();
    }

    gen_bistellar_k2_roundtrip_properties!(2, 3, 4, 5);
    gen_bistellar_k3_roundtrip_properties!(3, 4, 5);

    macro_rules! test_bistellar_roundtrip_dimension {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_bistellar_k1_roundtrip_ $dim d>]() {
                    init_tracing();
                    let mut tds: Tds<(), (), $dim> = Tds::empty();

                    let origin = tds.insert_vertex_with_mapping(vertex!([0.0; $dim]).unwrap()).unwrap();
                    let mut vertices = Vec::with_capacity($dim + 1);
                    vertices.push(origin);
                    for i in 0..$dim {
                        let v = tds
                            .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>(i)).unwrap())
                            .unwrap();
                        vertices.push(v);
                    }

                    let simplex_key = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(vertices, None).unwrap())
                        .unwrap();

                    let before = snapshot_topology(&tds);

                    let new_vertex = vertex!([0.1; $dim]).unwrap();
                    let new_uuid = new_vertex.uuid();
                    let _info = apply_bistellar_flip_k1_raw(&mut tds, simplex_key, new_vertex)
                        .unwrap();
                    assert!(tds.is_valid().is_ok());

                    let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
                    let _info_back =
                        apply_bistellar_flip_k1_inverse_raw(&mut tds, new_key).unwrap();
                    assert!(tds.is_valid().is_ok());

                    assert_eq!(snapshot_topology(&tds), before);
                }

                #[test]
                fn [<test_bistellar_k2_roundtrip_ $dim d>]() {
                    init_tracing();
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let mut shared_vertices = Vec::with_capacity($dim);
                    for i in 0..$dim {
                        let v = tds
                            .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>(i)).unwrap())
                            .unwrap();
                        shared_vertices.push(v);
                    }

                    let opposite_a = tds
                        .insert_vertex_with_mapping(vertex!([0.0; $dim]).unwrap())
                        .unwrap();
                    let opposite_b = tds
                        .insert_vertex_with_mapping(vertex!([1.0; $dim]).unwrap())
                        .unwrap();

                    let mut vertices_with_first_opposite = shared_vertices.clone();
                    vertices_with_first_opposite.push(opposite_a);
                    let simplex_a = tds
                        .insert_simplex_with_mapping(
                            Simplex::try_new_with_data(vertices_with_first_opposite, None).unwrap(),
                        )
                        .unwrap();

                    let mut vertices_with_second_opposite = shared_vertices.clone();
                    vertices_with_second_opposite.push(opposite_b);
                    let _simplex_b = tds
                        .insert_simplex_with_mapping(
                            Simplex::try_new_with_data(vertices_with_second_opposite, None).unwrap(),
                        )
                        .unwrap();

                    repair_neighbor_pointers(&mut tds).unwrap();

                    let before = snapshot_topology(&tds);

                    let facet = FacetHandle::from_validated(simplex_a, u8::try_from($dim).unwrap());
                    let context = build_k2_flip_context(&tds, facet).unwrap();
                    let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();
                    assert!(tds.is_valid().is_ok());

                    if $dim == 2 {
                        let mut inverse_facet: Option<FacetHandle> = None;
                        for &simplex_key in &info.new_simplices {
                            let simplex = tds.simplex(simplex_key).unwrap();
                            if simplex.contains_vertex(opposite_a) && simplex.contains_vertex(opposite_b) {
                                let facet_index = simplex
                                    .vertices()
                                    .iter()
                                    .position(|&v| v != opposite_a && v != opposite_b)
                                    .expect("missing shared vertex for inverse k=2");
                                inverse_facet = Some(FacetHandle::from_validated(
                                    simplex_key,
                                    u8::try_from(facet_index).unwrap(),
                                ));
                                break;
                            }
                        }

                        let facet = inverse_facet.expect("inverse k=2 facet not found");
                        let context_back = build_k2_flip_context(&tds, facet).unwrap();
                        let _info_back =
                            apply_bistellar_flip_raw(&mut tds, &context_back).unwrap();
                    } else {
                        let edge = EdgeKey::from_validated_endpoints(opposite_a, opposite_b);
                        let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
                        let _info_back =
                            apply_bistellar_flip_dynamic_raw(&mut tds, $dim, &context_back)
                                .unwrap();
                    }

                    assert!(tds.is_valid().is_ok());
                    let after = snapshot_topology(&tds);
                    assert_same_vertex_simplex_topology(&after, &before);
                }
            }
        };
        ($dim:literal, k3) => {
            test_bistellar_roundtrip_dimension!($dim);
            pastey::paste! {
                #[test]
                fn [<test_bistellar_k3_roundtrip_ $dim d>]() {
                    init_tracing();
                    let mut tds: Tds<(), (), $dim> = Tds::empty();
                    let mut ridge_vertices = Vec::with_capacity($dim - 1);
                    for i in 0..($dim - 1) {
                        let v = tds
                            .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>(i)).unwrap())
                            .unwrap();
                        ridge_vertices.push(v);
                    }

                    let a = tds
                        .insert_vertex_with_mapping(vertex!([0.0; $dim]).unwrap())
                        .unwrap();
                    let b = tds
                        .insert_vertex_with_mapping(vertex!(unit_vector::<$dim>($dim - 1)).unwrap())
                        .unwrap();
                    let c = tds
                        .insert_vertex_with_mapping(vertex!(skewed_point::<$dim>()).unwrap())
                        .unwrap();

                    let mut c1_vertices = ridge_vertices.clone();
                    c1_vertices.push(a);
                    c1_vertices.push(b);
                    let c1 = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(c1_vertices, None).unwrap())
                        .unwrap();

                    let mut c2_vertices = ridge_vertices.clone();
                    c2_vertices.push(b);
                    c2_vertices.push(c);
                    let _c2 = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(c2_vertices, None).unwrap())
                        .unwrap();

                    let mut c3_vertices = ridge_vertices.clone();
                    c3_vertices.push(c);
                    c3_vertices.push(a);
                    let _c3 = tds
                        .insert_simplex_with_mapping(Simplex::try_new_with_data(c3_vertices, None).unwrap())
                        .unwrap();

                    repair_neighbor_pointers(&mut tds).unwrap();

                    let before = snapshot_topology(&tds);

                    let ridge = RidgeHandle::from_validated(
                        c1,
                        u8::try_from($dim - 1).unwrap(),
                        u8::try_from($dim).unwrap(),
                    );
                    let context = build_k3_flip_context(&tds, ridge).unwrap();
                    let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();
                    assert!(tds.is_valid().is_ok());

                    if $dim == 3 {
                        let mut inverse_facet: Option<FacetHandle> = None;
                        for &simplex_key in &info.new_simplices {
                            let simplex = tds.simplex(simplex_key).unwrap();
                            if simplex.contains_vertex(a)
                                && simplex.contains_vertex(b)
                                && simplex.contains_vertex(c)
                            {
                                let facet_index = simplex
                                    .vertices()
                                    .iter()
                                    .position(|&v| v != a && v != b && v != c)
                                    .expect("missing ridge vertex for inverse k=3");
                                inverse_facet = Some(FacetHandle::from_validated(
                                    simplex_key,
                                    u8::try_from(facet_index).unwrap(),
                                ));
                                break;
                            }
                        }

                        let facet = inverse_facet.expect("inverse k=3 facet not found");
                        let context_back = build_k2_flip_context(&tds, facet).unwrap();
                        let _info_back =
                            apply_bistellar_flip_raw(&mut tds, &context_back).unwrap();
                    } else {
                        let triangle = TriangleHandle::try_new(a, b, c).unwrap();
                        let context_back =
                            build_k3_flip_context_from_triangle(&tds, triangle).unwrap();
                        let _info_back = apply_bistellar_flip_dynamic_raw(
                            &mut tds,
                            $dim - 1,
                            &context_back,
                        )
                        .unwrap();
                    }

                    assert!(tds.is_valid().is_ok());
                    let after = snapshot_topology(&tds);
                    assert_same_vertex_simplex_topology(&after, &before);
                }
            }
        };
    }

    test_bistellar_roundtrip_dimension!(2);
    test_bistellar_roundtrip_dimension!(3, k3);
    test_bistellar_roundtrip_dimension!(4, k3);
    test_bistellar_roundtrip_dimension!(5, k3);
}

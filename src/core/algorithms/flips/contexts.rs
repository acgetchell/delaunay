//! Bistellar flip validation and context construction.

#![forbid(unsafe_code)]

use super::{
    BistellarFlipKind, DataType, EdgeKey, EdgeKeyError, EntityKind, EuclideanPointCache,
    ExternalFacetBuffer, FacetHandle, FastHashMap, FlipContext, FlipContextDyn, FlipContextError,
    FlipDirection, FlipEdgeAdjacencyError, FlipError, FlipFeasibility, FlipInfo, FlipMutationError,
    FlipPredicateError, FlipPredicateOperation, FlipTriangleAdjacencyError, FlipValidationScope,
    FlipVertexAdjacencyError, GlobalTopologyModel, GlobalTopologyModelAdapter, Kernel, Key,
    MAX_PRACTICAL_DIMENSION_SIZE, Orientation, Point, RemovedSimplexVertexSnapshot,
    RepairAttemptConfig, RepairDiagnostics, ReplacementOrientationPolicy,
    ReplacementPeriodicOffsets, ReplacementSimplexVertices, RidgeHandle, Simplex, SimplexKey,
    SimplexKeyBuffer, SmallBuffer, Tds, TdsConstructionFailure, TriangleHandle, Vertex, VertexKey,
    VertexKeyList, apply_bistellar_flip_dynamic_raw, apply_bistellar_flip_raw,
    apply_bistellar_flip_with_k_raw, collect_simplices_around_ridge, env,
    facet_vertices_from_simplex, k1_inserted_vertex_periodic_offset, lift_vertex_point,
    matching_source_simplex, missing_opposite_for_simplex, once, periodic_offsets_or_zero_frame,
    predicate_key_from_vertices, prepare_bistellar_flip, ridge_vertices_from_simplex,
    robust_orientation, simplex_extras_for_ridge, vertex_point, vertex_point_lifted_into_simplex,
    vertices_to_points, vertices_to_points_with_optional_lift,
};

/// Check whether a k=3 ridge violates the local Delaunay condition.
///
/// # Errors
///
/// Returns a [`FlipError`] if any referenced simplex/vertex is missing or a predicate
/// evaluation fails.
pub(super) fn is_delaunay_violation_k3<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    context: &FlipContext<D, 3>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D, Scalar = f64>,
{
    delaunay_violation_k3_for_ridge(
        tds,
        kernel,
        topology_model,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        None,
        config,
        diagnostics,
    )
}

/// Validate a generic k-move without mutating the TDS.
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would fail during deterministic
/// pre-mutation validation.
pub(crate) fn validate_bistellar_flip<U, V, const D: usize, const K_MOVE: usize>(
    tds: &Tds<U, V, D>,
    context: &FlipContext<D, K_MOVE>,
) -> Result<FlipFeasibility<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    Ok(prepare_bistellar_flip(
        tds,
        K_MOVE,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::AllowSigned,
    )?
    .into_feasibility())
}

/// Validate a runtime-k move without mutating the TDS.
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would fail during deterministic
/// pre-mutation validation.
pub(crate) fn validate_bistellar_flip_dynamic<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    k_move: usize,
    context: &FlipContextDyn<D>,
) -> Result<FlipFeasibility<D>, FlipError> {
    Ok(prepare_bistellar_flip(
        tds,
        k_move,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::AllowSigned,
    )?
    .into_feasibility())
}

/// Apply a k=2 Delaunay-repair move with positive replacement geometry.
pub(super) fn apply_delaunay_flip_k2<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    context: &FlipContext<D, 2>,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k_raw(
        tds,
        2,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::RequirePositive,
        FlipValidationScope::LocalCavity,
    )
}

/// Apply a k=3 Delaunay-repair move with positive replacement geometry.
///
/// This preserves positive replacement geometry inside the caller-owned repair
/// transaction.
pub(super) fn apply_delaunay_flip_k3<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    context: &FlipContext<D, 3>,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k_raw(
        tds,
        3,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::RequirePositive,
        FlipValidationScope::LocalCavity,
    )
}

/// Apply a dynamic-size Delaunay-repair move.
///
/// This variant is used when the repair search cannot statically name `k`; it
/// still routes through the same validated bistellar mutation path inside the
/// caller-owned repair transaction.
pub(super) fn apply_delaunay_flip_dynamic<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    k_move: usize,
    context: &FlipContextDyn<D>,
) -> Result<AppliedFlip<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    apply_bistellar_flip_with_k_raw(
        tds,
        k_move,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
        &context.removed_simplices,
        context.direction,
        ReplacementOrientationPolicy::RequirePositive,
        FlipValidationScope::LocalCavity,
    )
}

#[derive(Debug)]
pub(super) struct AppliedFlip<const D: usize> {
    pub(super) info: FlipInfo<D>,
    pub(super) removed_simplex_vertices: RemovedSimplexVertexSnapshot,
}

/// Fully validated local flip preflight shared by dry-run and mutating paths.
///
/// Keeping this state in one helper type makes [`FlipFeasibility`] report the
/// same deterministic pre-mutation decision that the executor uses before it
/// performs its failure-atomic trial mutation.
#[derive(Debug)]
pub(super) struct PreparedFlip<const D: usize> {
    pub(super) kind: BistellarFlipKind,
    pub(super) direction: FlipDirection,
    pub(super) removed_simplices: SimplexKeyBuffer,
    pub(super) removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    pub(super) inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>,
    pub(super) new_simplex_vertices: ReplacementSimplexVertices,
    pub(super) new_simplex_offsets: ReplacementPeriodicOffsets<D>,
    pub(super) external_facets: ExternalFacetBuffer,
    pub(super) removed_simplex_vertices: RemovedSimplexVertexSnapshot,
}

impl<const D: usize> PreparedFlip<D> {
    /// Convert a prepared immutable preflight into the public dry-run report.
    fn into_feasibility(self) -> FlipFeasibility<D> {
        FlipFeasibility {
            kind: self.kind,
            direction: self.direction,
            removed_simplices: self.removed_simplices,
            removed_face_vertices: self.removed_face_vertices,
            inserted_face_vertices: Some(self.inserted_face_vertices),
        }
    }
}
pub(crate) fn build_k2_flip_context<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet: FacetHandle,
) -> Result<FlipContext<D, 2>, FlipError> {
    if D < 2 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let simplex_a_key = facet.simplex_key();
    let simplex_a = tds
        .simplex(simplex_a_key)
        .ok_or(FlipError::MissingSimplex {
            simplex_key: simplex_a_key,
        })?;

    let facet_index_a = usize::from(facet.facet_index());
    let vertex_count = simplex_a.number_of_vertices();
    if facet_index_a >= vertex_count {
        return Err(FlipError::InvalidFacetIndex {
            simplex_key: simplex_a_key,
            facet_index: facet.facet_index(),
            vertex_count,
        });
    }

    let neighbor_key = simplex_a
        .neighbor_key(facet_index_a)
        .flatten()
        .ok_or(FlipError::BoundaryFacet { facet })?;

    let simplex_b = tds
        .simplex(neighbor_key)
        .ok_or(FlipError::MissingNeighbor {
            facet,
            neighbor_key,
        })?;

    let Some(facet_index_b) = simplex_a
        .mirror_facet_index(facet_index_a, simplex_b)
        .or_else(|| back_reference_facet_index(simplex_a_key, simplex_b))
    else {
        return Err(FlipError::InvalidFacetAdjacency {
            simplex_key: simplex_a_key,
            neighbor_key,
        });
    };

    let opposite_a = simplex_a.vertices()[facet_index_a];
    let opposite_b = simplex_b.vertices()[facet_index_b];

    let shared_facet = facet_vertices_from_simplex(simplex_a, facet_index_a);

    if shared_facet.len() != D {
        return Err(FlipError::InvalidFacetAdjacency {
            simplex_key: simplex_a_key,
            neighbor_key,
        });
    }

    if shared_facet.contains(&opposite_a)
        || shared_facet.contains(&opposite_b)
        || opposite_a == opposite_b
    {
        return Err(FlipError::InvalidFacetAdjacency {
            simplex_key: simplex_a_key,
            neighbor_key,
        });
    }

    for &v in &shared_facet {
        if !simplex_b.contains_vertex(v) {
            return Err(FlipError::InvalidFacetAdjacency {
                simplex_key: simplex_a_key,
                neighbor_key,
            });
        }
    }

    let removed_simplices: SimplexKeyBuffer = [simplex_a_key, neighbor_key].into_iter().collect();
    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(2);
    inserted_face_vertices.push(opposite_a);
    inserted_face_vertices.push(opposite_b);

    Ok(FlipContext {
        removed_face_vertices: shared_facet,
        inserted_face_vertices,
        removed_simplices,
        direction: FlipDirection::Forward,
    })
}

/// Finds the neighbor slot that points back to a source simplex when reciprocal
/// neighbor pointers are already available.
pub(super) fn back_reference_facet_index<V, const D: usize>(
    source_simplex: SimplexKey,
    neighbor_simplex: &Simplex<V, D>,
) -> Option<usize> {
    neighbor_simplex
        .neighbor_keys()?
        .position(|neighbor| neighbor == Some(source_simplex))
}

/// Increments a small vertex-incidence count buffer without allocating a hash map
/// for the tiny opposite-face sets used by inverse flip context builders.
pub(super) fn increment_vertex_count(
    counts: &mut SmallBuffer<(VertexKey, usize), MAX_PRACTICAL_DIMENSION_SIZE>,
    vertex_key: VertexKey,
) {
    if let Some((_vertex, count)) = counts
        .iter_mut()
        .find(|(existing_vertex, _count)| *existing_vertex == vertex_key)
    {
        *count += 1;
    } else {
        counts.push((vertex_key, 1));
    }
}

/// Resolves a simplex key that came from a vertex incidence list.
///
/// A miss here means the maintained vertex-to-simplices index is stale, not
/// that the caller supplied an arbitrary missing simplex key.
pub(super) fn simplex_from_vertex_incidence<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertex_key: VertexKey,
    simplex_key: SimplexKey,
) -> Result<&Simplex<V, D>, FlipError> {
    tds.simplex(simplex_key)
        .ok_or(FlipError::DanglingVertexIncidence {
            vertex_key,
            simplex_key,
        })
}
/// Converts borrowed edge-view validation errors into k=2 flip context errors.
///
/// [`build_k2_flip_context_from_edge`] exposes flip-specific error variants even
/// though it validates runtime edge handles through [`crate::core::edge::EdgeView`].
/// This mapping preserves caller-visible distinctions such as stale endpoints,
/// dangling vertex incidence, and invalid edge multiplicity.
pub(super) fn flip_error_from_edge_key_error<const D: usize>(error: EdgeKeyError) -> FlipError {
    match error {
        EdgeKeyError::DuplicateEndpoint { endpoint } => {
            FlipEdgeAdjacencyError::DuplicateEndpoints {
                vertex_key: endpoint,
            }
            .into()
        }
        EdgeKeyError::MissingEndpoint { endpoint } => FlipError::MissingVertex {
            vertex_key: endpoint,
        },
        EdgeKeyError::EdgeNotFound { .. } => FlipError::InvalidEdgeMultiplicity {
            found: 0,
            expected: D,
        },
        EdgeKeyError::MissingEdgeIncidence { v0, v1 } => {
            FlipEdgeAdjacencyError::MissingEdgeIncidence { v0, v1 }.into()
        }
        EdgeKeyError::MissingVertexIncidence {
            vertex_key,
            simplex_key,
        } => FlipEdgeAdjacencyError::MissingVertexIncidence {
            vertex_key,
            simplex_key,
        }
        .into(),
        EdgeKeyError::DanglingVertexIncidence {
            vertex_key,
            simplex_key,
        } => FlipError::DanglingVertexIncidence {
            vertex_key,
            simplex_key,
        },
        EdgeKeyError::VertexIncidenceMismatch {
            simplex_key,
            vertex_key,
        } => FlipEdgeAdjacencyError::VertexIncidenceMismatch {
            vertex_key,
            simplex_key,
        }
        .into(),
    }
}

/// Build inverse k=2 flip context from an edge and its incident simplices.
///
/// # Errors
///
/// Returns a [`FlipError`] if the edge is invalid, references missing vertices/simplices,
/// or the adjacency data is inconsistent.
pub(crate) fn build_k2_flip_context_from_edge<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    edge: EdgeKey,
) -> Result<FlipContextDyn<D>, FlipError> {
    if D < 3 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let edge_view = edge
        .view(tds)
        .map_err(flip_error_from_edge_key_error::<D>)?;
    let (v0, v1) = edge_view.endpoint_keys();
    let removed_simplices: SimplexKeyBuffer =
        edge_view.incident_simplices().iter().copied().collect();

    if removed_simplices.len() != D {
        return Err(FlipError::InvalidEdgeMultiplicity {
            found: removed_simplices.len(),
            expected: D,
        });
    }

    let mut counts: SmallBuffer<(VertexKey, usize), MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::new();
    for &simplex_key in &removed_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FlipError::MissingSimplex { simplex_key })?;
        if !simplex.contains_vertex(v0) || !simplex.contains_vertex(v1) {
            return Err(FlipEdgeAdjacencyError::SimplexMissingEdgeVertices {
                simplex_key,
                v0,
                v1,
            }
            .into());
        }
        for &vk in simplex.vertices() {
            if vk != v0 && vk != v1 {
                increment_vertex_count(&mut counts, vk);
            }
        }
    }

    if counts.len() != D || !counts.iter().all(|(_vertex, count)| *count == D - 1) {
        return Err(FlipEdgeAdjacencyError::InvalidOppositeVertexIncidence {
            expected_vertices: D,
            found_vertices: counts.len(),
            expected_occurrences: D - 1,
        }
        .into());
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        counts.iter().map(|(vertex, _count)| *vertex).collect();
    inserted_face_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

    let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(2);
    removed_face_vertices.push(v0);
    removed_face_vertices.push(v1);

    Ok(FlipContextDyn {
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction: FlipDirection::Inverse,
    })
}
/// Build a forward k=1 flip context from a simplex and inserted vertex.
pub(super) fn build_k1_forward_context_from_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    inserted_vertex: VertexKey,
) -> Result<FlipContext<D, 1>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let simplex = tds
        .simplex(simplex_key)
        .ok_or(FlipError::MissingSimplex { simplex_key })?;
    if tds.vertex(inserted_vertex).is_none() {
        return Err(FlipError::MissingVertex {
            vertex_key: inserted_vertex,
        });
    }

    let removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        simplex.vertices().iter().copied().collect();
    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(1);
    inserted_face_vertices.push(inserted_vertex);

    let removed_simplices: SimplexKeyBuffer = once(simplex_key).collect();

    Ok(FlipContext {
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction: FlipDirection::Forward,
    })
}

/// Build inverse k=1 flip context from a vertex and its incident simplices.
///
/// # Errors
///
/// Returns a [`FlipError`] if the vertex is missing, its incident simplex count is
/// not D+1, or the adjacency data is inconsistent.
pub(crate) fn build_k1_inverse_context<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertex_key: VertexKey,
) -> Result<FlipContextDyn<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    if tds.vertex(vertex_key).is_none() {
        return Err(FlipError::MissingVertex { vertex_key });
    }

    let removed_simplices: SimplexKeyBuffer =
        tds.simplex_keys_containing_vertex(vertex_key).collect();
    for &simplex_key in &removed_simplices {
        simplex_from_vertex_incidence(tds, vertex_key, simplex_key)?;
    }

    let expected = D + 1;
    if removed_simplices.len() != expected {
        return Err(FlipError::InvalidVertexMultiplicity {
            found: removed_simplices.len(),
            expected,
        });
    }

    let mut counts: FastHashMap<VertexKey, usize> = FastHashMap::default();
    let mut removed_simplices_buf: SimplexKeyBuffer = SimplexKeyBuffer::new();
    for &simplex_key in &removed_simplices {
        let simplex = simplex_from_vertex_incidence(tds, vertex_key, simplex_key)?;
        if !simplex.contains_vertex(vertex_key) {
            return Err(FlipVertexAdjacencyError::SimplexMissingVertex {
                simplex_key,
                vertex_key,
            }
            .into());
        }
        removed_simplices_buf.push(simplex_key);
        for &vk in simplex.vertices() {
            if vk != vertex_key {
                *counts.entry(vk).or_insert(0) += 1;
            }
        }
    }

    if counts.len() != expected || !counts.values().all(|&count| count == D) {
        return Err(FlipVertexAdjacencyError::InvalidLinkVertexIncidence {
            expected_vertices: expected,
            found_vertices: counts.len(),
            expected_occurrences: D,
        }
        .into());
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        counts.keys().copied().collect();
    inserted_face_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

    let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(1);
    removed_face_vertices.push(vertex_key);

    Ok(FlipContextDyn {
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices: removed_simplices_buf,
        direction: FlipDirection::Inverse,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "local predicate evaluation threads topology, source simplices, and diagnostics explicitly"
)]
/// Evaluate the k=2 facet flip predicate for a local Delaunay violation.
pub(super) fn delaunay_violation_k2_for_facet<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    facet_vertices: &[VertexKey],
    opposite_a: VertexKey,
    opposite_b: VertexKey,
    source_simplices: &[SimplexKey],
    frame_simplex: Option<SimplexKey>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D, Scalar = f64>,
{
    if facet_vertices.len() != D {
        return Err(FlipContextError::K2FacetArity {
            expected: D,
            found: facet_vertices.len(),
        }
        .into());
    }
    if facet_vertices.contains(&opposite_a)
        || facet_vertices.contains(&opposite_b)
        || opposite_a == opposite_b
    {
        return Err(FlipContextError::InvalidK2Opposites.into());
    }

    let mut simplex_vertices: [SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>; 2] = [
        SmallBuffer::with_capacity(D + 1),
        SmallBuffer::with_capacity(D + 1),
    ];
    for vertices in &mut simplex_vertices {
        vertices.extend_from_slice(facet_vertices);
    }
    simplex_vertices[0].push(opposite_a);
    simplex_vertices[1].push(opposite_b);

    // Sort by VertexKey for canonical SoS perturbation ordering
    simplex_vertices[0].sort_unstable_by_key(|v| v.data().as_ffi());
    simplex_vertices[1].sort_unstable_by_key(|v| v.data().as_ffi());

    // The two reciprocal in-sphere queries are algebraically equivalent before
    // perturbation, but evaluating both would give the positional SoS fallback
    // two different row assignments. Choose one query from the complete local
    // circuit by stable vertex identity so forward and inverse checks make the
    // same decision for cospherical input.
    let (simplex_index, test_vertex, operation) =
        if opposite_a.data().as_ffi() < opposite_b.data().as_ffi() {
            (0, opposite_b, FlipPredicateOperation::K2SimplexAInSphere)
        } else {
            (1, opposite_a, FlipPredicateOperation::K2SimplexBInSphere)
        };
    let predicate_vertices = &simplex_vertices[simplex_index];

    let (points, test_point) = if matches!(topology_model, GlobalTopologyModelAdapter::Euclidean(_))
    {
        let mut point_cache = EuclideanPointCache::new();
        (
            point_cache.points_for_vertices(tds, predicate_vertices)?,
            point_cache.point(tds, test_vertex)?,
        )
    } else {
        let source =
            matching_source_simplex(tds, predicate_vertices, source_simplices).or(frame_simplex);
        (
            vertices_to_points_with_optional_lift(
                tds,
                topology_model,
                predicate_vertices,
                source,
                source_simplices,
            )?,
            vertex_point_lifted_into_simplex(
                tds,
                topology_model,
                test_vertex,
                source,
                source_simplices,
            )?,
        )
    };
    let classification = match kernel.in_sphere(&points, &test_point) {
        Ok(value) => value,
        Err(error) => {
            diagnostics.record_predicate_failure();
            return Err(FlipPredicateError::coordinate_conversion(operation, error).into());
        }
    };

    // Record ambiguous sites when the predicate returns boundary/uncertain.
    if classification == 0 {
        let key = predicate_key_from_vertices(predicate_vertices, test_vertex);
        diagnostics.record_ambiguous(key);
    }

    let violates = classification > 0;
    if env::var_os("DELAUNAY_REPAIR_DEBUG_PREDICATES").is_some()
        && (violates || classification == 0)
    {
        tracing::debug!(
            facet_vertices = ?facet_vertices,
            opposite_a = ?opposite_a,
            opposite_b = ?opposite_b,
            predicate_simplex = ?predicate_vertices,
            test_vertex = ?test_vertex,
            classification,
            violates,
            attempt = config.attempt,
            "delaunay_violation_k2_for_facet: insphere classification"
        );
    }

    Ok(violates)
}
/// Check whether a flip would create a degenerate (zero-volume) simplex.
///
/// Builds the replacement simplices from the given removed/inserted face vertices
/// and checks each with [`robust_orientation`].  Returns `Ok(true)` if any
/// replacement simplex is degenerate.
pub(super) fn flip_would_create_degenerate_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    removed_face_vertices: &[VertexKey],
    inserted_face_vertices: &[VertexKey],
) -> Result<bool, FlipError> {
    for &omit in removed_face_vertices {
        let mut vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        vertices.extend_from_slice(inserted_face_vertices);
        for &v in removed_face_vertices {
            if v != omit {
                vertices.push(v);
            }
        }

        let points = vertices_to_points(tds, &vertices)?;
        // Use exact orientation (no SoS) so that truly degenerate simplices are
        // detected even when the kernel uses SoS.  Matches the pattern in
        // apply_bistellar_flip_with_k.
        match robust_orientation(&points) {
            Err(e) => {
                return Err(FlipPredicateError::coordinate_conversion(
                    FlipPredicateOperation::DegenerateSimplexPrecheck,
                    e,
                )
                .into());
            }
            Ok(Orientation::DEGENERATE) => return Ok(true),
            Ok(_) => {}
        }
    }

    Ok(false)
}

/// Check whether a k=2 flip would create a degenerate simplex.
pub(super) fn k2_flip_would_create_degenerate_simplex<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    context: &FlipContext<D, 2>,
) -> Result<bool, FlipError> {
    if context.inserted_face_vertices.len() != 2 {
        return Err(FlipContextError::WrongInsertedFaceArity {
            k_move: 2,
            expected: 2,
            found: context.inserted_face_vertices.len(),
        }
        .into());
    }

    flip_would_create_degenerate_simplex(
        tds,
        &context.removed_face_vertices,
        &context.inserted_face_vertices,
    )
}
/// Check whether a k=2 facet violates the local Delaunay condition.
///
/// # Errors
///
/// Returns a [`FlipError`] if any referenced simplex/vertex is missing or a predicate
/// evaluation fails.
pub(super) fn is_delaunay_violation_k2<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    context: &FlipContext<D, 2>,
    config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D, Scalar = f64>,
{
    if context.inserted_face_vertices.len() != 2 {
        return Err(FlipContextError::WrongInsertedFaceArity {
            k_move: 2,
            expected: 2,
            found: context.inserted_face_vertices.len(),
        }
        .into());
    }
    let opposite_a = context.inserted_face_vertices[0];
    let opposite_b = context.inserted_face_vertices[1];
    delaunay_violation_k2_for_facet(
        tds,
        kernel,
        topology_model,
        &context.removed_face_vertices,
        opposite_a,
        opposite_b,
        &context.removed_simplices,
        None,
        config,
        diagnostics,
    )
}

/// Validate a k=2 bistellar flip without mutating the TDS.
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would fail during deterministic
/// pre-mutation validation.
pub(crate) fn validate_bistellar_flip_k2<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    context: &FlipContext<D, 2>,
) -> Result<FlipFeasibility<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    validate_bistellar_flip::<U, V, D, 2>(tds, context)
}

/// Build flip context for a k=3 (ridge) flip.
///
/// # Errors
///
/// Returns a [`FlipError`] if the ridge is invalid, references missing simplices/vertices,
/// or the adjacency data is inconsistent.
pub(crate) fn build_k3_flip_context<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge: RidgeHandle,
) -> Result<FlipContext<D, 3>, FlipError> {
    build_k3_flip_context_with_star_limit(tds, ridge, None)
}

/// Builds k=3 repair context only for true three-simplex ridge stars.
pub(super) fn build_k3_flip_context_for_repair<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge: RidgeHandle,
) -> Result<FlipContext<D, 3>, FlipError> {
    build_k3_flip_context_with_star_limit(tds, ridge, Some(3))
}

/// Builds k=3 flip context while optionally rejecting ridge stars above a caller limit.
pub(super) fn build_k3_flip_context_with_star_limit<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge: RidgeHandle,
    max_simplices: Option<usize>,
) -> Result<FlipContext<D, 3>, FlipError> {
    if D < 3 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let simplex_key = ridge.simplex_key();
    let simplex = tds
        .simplex(simplex_key)
        .ok_or(FlipError::MissingSimplex { simplex_key })?;

    let vertex_count = simplex.number_of_vertices();
    let omit_a = usize::from(ridge.omit_a());
    let omit_b = usize::from(ridge.omit_b());
    if omit_a >= vertex_count || omit_b >= vertex_count || omit_a == omit_b {
        return Err(FlipError::InvalidRidgeIndex {
            simplex_key,
            omit_a: ridge.omit_a(),
            omit_b: ridge.omit_b(),
            vertex_count,
        });
    }

    let ridge_vertices = ridge_vertices_from_simplex(simplex, omit_a, omit_b);
    if ridge_vertices.len() != D - 1 {
        return Err(FlipError::InvalidRidgeAdjacency { simplex_key });
    }

    let simplices =
        collect_simplices_around_ridge(tds, simplex_key, &ridge_vertices, max_simplices)?;
    if simplices.len() != 3 {
        return Err(FlipError::InvalidRidgeMultiplicity {
            found: simplices.len(),
        });
    }

    // k=3 flip contexts are tiny (exactly 3 simplices, with 2 "extra" vertices per simplex).
    // Use flat buffers + linear counting to avoid HashMap/Vec overhead in this hot path.
    let mut opposite_counts: SmallBuffer<(VertexKey, u8), 3> = SmallBuffer::new();
    let mut extras_per_simplex: SmallBuffer<[VertexKey; 2], 3> = SmallBuffer::new();

    for &ck in &simplices {
        let simplex = tds
            .simplex(ck)
            .ok_or(FlipError::MissingSimplex { simplex_key: ck })?;
        let extras = simplex_extras_for_ridge(ck, simplex, &ridge_vertices)?;
        if extras.len() != 2 {
            return Err(FlipError::InvalidRidgeAdjacency { simplex_key: ck });
        }

        let extras_pair: [VertexKey; 2] = extras
            .as_slice()
            .try_into()
            .map_err(|_| FlipError::InvalidRidgeAdjacency { simplex_key: ck })?;

        for &v in &extras_pair {
            if let Some((_key, count)) = opposite_counts.iter_mut().find(|(key, _)| *key == v) {
                *count += 1;
            } else {
                opposite_counts.push((v, 1));
            }
        }

        extras_per_simplex.push(extras_pair);
    }

    if opposite_counts.len() != 3 || !opposite_counts.iter().all(|(_v, count)| *count == 2) {
        return Err(FlipError::InvalidRidgeAdjacency { simplex_key });
    }

    let mut opposite_vertices: SmallBuffer<VertexKey, 3> =
        opposite_counts.iter().map(|(v, _count)| *v).collect();
    opposite_vertices.sort_unstable();
    let opposite_vertices: [VertexKey; 3] = opposite_vertices
        .as_slice()
        .try_into()
        .map_err(|_| FlipError::InvalidRidgeAdjacency { simplex_key })?;

    for extras in &extras_per_simplex {
        let _missing = missing_opposite_for_simplex(extras, &opposite_vertices)
            .ok_or(FlipError::InvalidRidgeAdjacency { simplex_key })?;
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(3);
    inserted_face_vertices.extend(opposite_vertices);

    Ok(FlipContext {
        removed_face_vertices: ridge_vertices,
        inserted_face_vertices,
        removed_simplices: simplices,
        direction: FlipDirection::Forward,
    })
}

/// Build inverse k=3 flip context from a triangle and its incident simplices.
///
/// # Errors
///
/// Returns a [`FlipError`] if the triangle is invalid, references missing vertices/simplices,
/// or the adjacency data is inconsistent.
pub(crate) fn build_k3_flip_context_from_triangle<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    triangle: TriangleHandle,
) -> Result<FlipContextDyn<D>, FlipError> {
    if D < 4 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let [a, b, c] = triangle.vertices();
    if tds.vertex(a).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: a });
    }
    if tds.vertex(b).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: b });
    }
    if tds.vertex(c).is_none() {
        return Err(FlipError::MissingVertex { vertex_key: c });
    }

    let mut removed_simplices: SimplexKeyBuffer = SimplexKeyBuffer::new();
    for simplex_key in tds.simplex_keys_containing_vertex(a) {
        let simplex = simplex_from_vertex_incidence(tds, a, simplex_key)?;
        if simplex.contains_vertex(b) && simplex.contains_vertex(c) {
            removed_simplices.push(simplex_key);
        }
    }

    let expected = D - 1;
    if removed_simplices.len() != expected {
        return Err(FlipError::InvalidTriangleMultiplicity {
            found: removed_simplices.len(),
            expected,
        });
    }

    let mut counts: SmallBuffer<(VertexKey, usize), MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::new();
    for &simplex_key in &removed_simplices {
        let simplex = tds
            .simplex(simplex_key)
            .ok_or(FlipError::MissingSimplex { simplex_key })?;
        if !simplex.contains_vertex(a) || !simplex.contains_vertex(b) || !simplex.contains_vertex(c)
        {
            return Err(FlipTriangleAdjacencyError::SimplexMissingTriangleVertices {
                simplex_key,
                a,
                b,
                c,
            }
            .into());
        }
        for &vk in simplex.vertices() {
            if vk != a && vk != b && vk != c {
                increment_vertex_count(&mut counts, vk);
            }
        }
    }

    if counts.len() != expected || !counts.iter().all(|(_vertex, count)| *count == expected - 1) {
        return Err(FlipTriangleAdjacencyError::InvalidRidgeVertexIncidence {
            expected_vertices: expected,
            found_vertices: counts.len(),
            expected_occurrences: expected - 1,
        }
        .into());
    }

    let mut inserted_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        counts.iter().map(|(vertex, _count)| *vertex).collect();
    inserted_face_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

    let mut removed_face_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(3);
    removed_face_vertices.push(a);
    removed_face_vertices.push(b);
    removed_face_vertices.push(c);

    Ok(FlipContextDyn {
        removed_face_vertices,
        inserted_face_vertices,
        removed_simplices,
        direction: FlipDirection::Inverse,
    })
}
#[expect(
    clippy::too_many_arguments,
    reason = "Local predicate evaluation threads topology, source simplices, and diagnostics explicitly"
)]
/// Evaluate the k=3 ridge flip predicate for a local Delaunay violation.
pub(super) fn delaunay_violation_k3_for_ridge<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_model: &GlobalTopologyModelAdapter<D>,
    ridge_vertices: &[VertexKey],
    triangle_vertices: &[VertexKey],
    source_simplices: &[SimplexKey],
    frame_simplex: Option<SimplexKey>,
    _config: &RepairAttemptConfig,
    diagnostics: &mut RepairDiagnostics,
) -> Result<bool, FlipError>
where
    K: Kernel<D, Scalar = f64>,
{
    if triangle_vertices.len() != 3 {
        return Err(FlipContextError::WrongInsertedFaceArity {
            k_move: 3,
            expected: 3,
            found: triangle_vertices.len(),
        }
        .into());
    }
    if ridge_vertices.len() != D.saturating_sub(1) {
        return Err(FlipContextError::K3RidgeArity {
            expected: D.saturating_sub(1),
            found: ridge_vertices.len(),
        }
        .into());
    }

    let is_euclidean_topology = matches!(topology_model, GlobalTopologyModelAdapter::Euclidean(_));
    let mut euclidean_point_cache = EuclideanPointCache::new();

    for &missing in triangle_vertices {
        let mut simplex_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        simplex_vertices.extend_from_slice(ridge_vertices);
        for &v in triangle_vertices {
            if v != missing {
                simplex_vertices.push(v);
            }
        }

        // Sort by VertexKey for canonical SoS perturbation ordering
        simplex_vertices.sort_unstable_by_key(|v| v.data().as_ffi());

        let (points, missing_point) = if is_euclidean_topology {
            (
                euclidean_point_cache.points_for_vertices(tds, &simplex_vertices)?,
                euclidean_point_cache.point(tds, missing)?,
            )
        } else {
            let source_simplex =
                matching_source_simplex(tds, &simplex_vertices, source_simplices).or(frame_simplex);
            (
                vertices_to_points_with_optional_lift(
                    tds,
                    topology_model,
                    &simplex_vertices,
                    source_simplex,
                    source_simplices,
                )?,
                vertex_point_lifted_into_simplex(
                    tds,
                    topology_model,
                    missing,
                    source_simplex,
                    source_simplices,
                )?,
            )
        };

        let in_sphere_result = kernel.in_sphere(&points, &missing_point);
        let in_sphere = match in_sphere_result {
            Ok(value) => value,
            Err(e) => {
                diagnostics.record_predicate_failure();
                return Err(FlipPredicateError::coordinate_conversion(
                    FlipPredicateOperation::K3SimplexInSphere,
                    e,
                )
                .into());
            }
        };

        // Track ambiguous sites when the fast predicate returns boundary/uncertain.
        if in_sphere == 0 {
            let key = predicate_key_from_vertices(&simplex_vertices, missing);
            diagnostics.record_ambiguous(key);
        }

        if in_sphere > 0 {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Validate a k=3 bistellar flip without mutating the TDS.
///
/// # Errors
///
/// Returns a [`FlipError`] if the flip would fail during deterministic
/// pre-mutation validation.
pub(crate) fn validate_bistellar_flip_k3<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    context: &FlipContext<D, 3>,
) -> Result<FlipFeasibility<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    validate_bistellar_flip::<U, V, D, 3>(tds, context)
}

/// Lifts a proposed k=1 vertex into the same simplex-local chart used by mutation planning.
pub(super) fn k1_inserted_vertex_point_in_simplex_frame<U, V, const D: usize>(
    topology_model: &GlobalTopologyModelAdapter<D>,
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    vertex: &Vertex<U, D>,
) -> Result<Point<D>, FlipError> {
    let periodic_offset = if topology_model.supports_periodic_orientation_offsets() {
        Some(k1_inserted_vertex_periodic_offset(simplex_key, simplex)?)
    } else {
        None
    };
    let lifted_coords = topology_model
        .lift_for_orientation(*vertex.point().coords(), periodic_offset)
        .map_err(|source| FlipPredicateError::K1InsertedVertexLift {
            simplex_key,
            source,
        })?;
    Point::try_new(lifted_coords).map_err(|source| {
        FlipPredicateError::K1InsertedVertexPointValidation {
            simplex_key,
            source,
        }
        .into()
    })
}

/// Requires the proposed k=1 point to lie strictly inside the selected realization chart.
pub(super) fn validate_k1_insertion_realization<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    topology_model: &GlobalTopologyModelAdapter<D>,
    simplex_key: SimplexKey,
    simplex: &Simplex<V, D>,
    vertex: &Vertex<U, D>,
) -> Result<(), FlipError>
where
    U: DataType,
    V: DataType,
{
    let offsets = periodic_offsets_or_zero_frame(simplex_key, simplex)?;
    let use_offsets = topology_model.supports_periodic_orientation_offsets();
    let mut simplex_points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(simplex.number_of_vertices());
    for (vertex_index, &vertex_key) in simplex.vertices().iter().enumerate() {
        let periodic_offset = use_offsets.then_some(offsets[vertex_index]);
        simplex_points.push(lift_vertex_point(
            tds,
            topology_model,
            vertex_key,
            periodic_offset,
        )?);
    }

    let simplex_orientation = robust_orientation(&simplex_points).map_err(|source| {
        FlipPredicateError::coordinate_conversion(
            FlipPredicateOperation::ReplacementSimplexOrientation,
            source,
        )
    })?;
    if simplex_orientation == Orientation::DEGENERATE {
        return Err(FlipError::DegenerateSimplex);
    }

    let inserted_point =
        k1_inserted_vertex_point_in_simplex_frame(topology_model, simplex_key, simplex, vertex)?;
    for (omit_index, &opposite_vertex) in simplex.vertices().iter().enumerate() {
        let mut replacement_points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(D + 1);
        replacement_points.push(inserted_point);
        replacement_points.extend(
            simplex_points
                .iter()
                .enumerate()
                .filter_map(|(index, point)| (index != omit_index).then_some(*point)),
        );

        let replacement_orientation =
            robust_orientation(&replacement_points).map_err(|source| {
                FlipPredicateError::coordinate_conversion(
                    FlipPredicateOperation::ReplacementSimplexOrientation,
                    source,
                )
            })?;
        if replacement_orientation == Orientation::DEGENERATE {
            return Err(FlipError::DegenerateSimplex);
        }

        let expected_orientation = if omit_index.is_multiple_of(2) {
            simplex_orientation
        } else {
            match simplex_orientation {
                Orientation::POSITIVE => Orientation::NEGATIVE,
                Orientation::NEGATIVE => Orientation::POSITIVE,
                Orientation::DEGENERATE => return Err(FlipError::DegenerateSimplex),
            }
        };
        if replacement_orientation != expected_orientation {
            return Err(FlipError::K1InsertionOutsideSimplex {
                simplex_key,
                opposite_vertex,
                opposite_vertex_index: omit_index,
            });
        }
    }

    Ok(())
}

/// Validate a forward k=1 move (simplex split) without mutating the TDS.
///
/// # Errors
///
/// Returns a [`FlipError`] if the simplex is missing, the vertex UUID is already
/// present, the replacement simplices would be degenerate, or the inserted point
/// is outside the selected simplex in its active realization chart.
pub(crate) fn validate_bistellar_flip_k1_insert<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    topology_model: &GlobalTopologyModelAdapter<D>,
    simplex_key: SimplexKey,
    vertex: &Vertex<U, D>,
) -> Result<FlipFeasibility<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let vertex_uuid = vertex.uuid();
    if tds.vertex_key_from_uuid(&vertex_uuid).is_some() {
        return Err(FlipError::from(FlipMutationError::VertexInsertion {
            source: TdsConstructionFailure::DuplicateUuid {
                entity: EntityKind::Vertex,
                uuid: vertex_uuid,
            },
        }));
    }

    let simplex = tds
        .simplex(simplex_key)
        .ok_or(FlipError::MissingSimplex { simplex_key })?;
    let removed_face_vertices: VertexKeyList = simplex.vertices().iter().copied().collect();

    if !topology_model.supports_periodic_orientation_offsets() {
        for omit_index in 0..removed_face_vertices.len() {
            let mut points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::with_capacity(D + 1);
            points.push(*vertex.point());
            for (vertex_index, &vertex_key) in removed_face_vertices.iter().enumerate() {
                if vertex_index != omit_index {
                    points.push(vertex_point(tds, vertex_key)?);
                }
            }

            match robust_orientation(&points) {
                Ok(Orientation::POSITIVE | Orientation::NEGATIVE) => {}
                Ok(Orientation::DEGENERATE) => return Err(FlipError::DegenerateSimplex),
                Err(error) => {
                    return Err(FlipPredicateError::coordinate_conversion(
                        FlipPredicateOperation::ReplacementSimplexOrientation,
                        error,
                    )
                    .into());
                }
            }
        }
    }
    validate_k1_insertion_realization(tds, topology_model, simplex_key, simplex, vertex)?;

    Ok(FlipFeasibility {
        kind: BistellarFlipKind::from_validated(1, D),
        direction: FlipDirection::Forward,
        removed_simplices: once(simplex_key).collect(),
        removed_face_vertices,
        inserted_face_vertices: None,
    })
}

/// Apply a forward k=1 move without rollback.
///
/// The caller owns transaction rollback if this returns an error or if later
/// postconditions fail.
pub(crate) fn apply_bistellar_flip_k1_raw<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    simplex_key: SimplexKey,
    vertex: Vertex<U, D>,
) -> Result<FlipInfo<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let vertex_key = tds.insert_vertex_with_mapping(vertex).map_err(|source| {
        FlipMutationError::VertexInsertion {
            source: source.into(),
        }
    })?;
    let context = build_k1_forward_context_from_simplex(tds, simplex_key, vertex_key)?;
    apply_bistellar_flip_raw::<U, V, D, 1>(tds, &context)
}

/// Apply an inverse k=1 move without rollback.
///
/// The caller owns transaction rollback if this returns an error or if later
/// postconditions fail.
pub(crate) fn apply_bistellar_flip_k1_inverse_raw<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    vertex_key: VertexKey,
) -> Result<FlipInfo<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let context = build_k1_inverse_context(tds, vertex_key)?;
    let info = apply_bistellar_flip_dynamic_raw(tds, D + 1, &context)?;
    remove_k1_inverse_vertex(tds, vertex_key)?;

    Ok(info)
}

/// Removes the vertex collapsed by a successful inverse k=1 simplex rewrite.
fn remove_k1_inverse_vertex<U, V, const D: usize>(
    tds: &mut Tds<U, V, D>,
    vertex_key: VertexKey,
) -> Result<(), FlipError> {
    tds.remove_vertex(vertex_key)
        .map(|_| ())
        .map_err(|source| FlipMutationError::VertexRemoval { source }.into())
}

/// Validate an inverse k=1 move (vertex collapse) without mutating the TDS.
///
/// # Errors
///
/// Returns a [`FlipError`] if the vertex star is invalid or the replacement
/// simplex would be degenerate.
pub(crate) fn validate_bistellar_flip_k1_inverse<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    vertex_key: VertexKey,
) -> Result<FlipFeasibility<D>, FlipError>
where
    U: DataType,
    V: DataType,
{
    if D < 1 {
        return Err(FlipError::UnsupportedDimension { dimension: D });
    }

    let context = build_k1_inverse_context(tds, vertex_key)?;
    validate_bistellar_flip_dynamic(tds, D + 1, &context)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::init_tracing;
    use super::super::*;
    use super::*;
    use crate::core::algorithms::insertion::repair_neighbor_pointers;
    use crate::core::tds::TdsError;
    use crate::core::test_support::{assert_same_vertex_simplex_topology, snapshot_topology};
    use crate::geometry::traits::coordinate::CoordinateConversionError;
    use crate::topology::traits::GlobalTopology;
    use crate::vertex;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use slotmap::KeyData;
    use std::assert_matches;
    use std::iter::once;

    #[derive(Clone)]
    struct FailingInSphereKernel;

    impl<const D: usize> Kernel<D> for FailingInSphereKernel {
        type Scalar = f64;

        fn orientation(&self, _points: &[Point<D>]) -> Result<i32, CoordinateConversionError> {
            Ok(1)
        }

        fn in_sphere(
            &self,
            _simplex_points: &[Point<D>],
            _test_point: &Point<D>,
        ) -> Result<i32, CoordinateConversionError> {
            Err(CoordinateConversionError::UnsupportedMatrixDimension {
                requested: D + 2,
                max: D + 1,
            })
        }
    }

    #[test]
    fn k2_predicate_failure_preserves_operation_and_diagnostics() {
        let mut tds = Tds::<(), (), 2>::empty();
        let facet = [
            tds.insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
                .unwrap(),
            tds.insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
                .unwrap(),
        ];
        let opposite_a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();
        let topology_model = GlobalTopology::Euclidean.model();
        let config = RepairAttemptConfig {
            attempt: 1,
            queue_order: RepairQueueOrder::Fifo,
            max_flips_override: None,
        };

        for (first, second, expected_operation) in [
            (
                opposite_a,
                opposite_b,
                FlipPredicateOperation::K2SimplexAInSphere,
            ),
            (
                opposite_b,
                opposite_a,
                FlipPredicateOperation::K2SimplexBInSphere,
            ),
        ] {
            let mut diagnostics = RepairDiagnostics::default();
            let error = delaunay_violation_k2_for_facet(
                &tds,
                &FailingInSphereKernel,
                &topology_model,
                &facet,
                first,
                second,
                &[],
                None,
                &config,
                &mut diagnostics,
            )
            .unwrap_err();

            assert_matches!(
                error,
                FlipError::PredicateFailure { reason }
                    if matches!(
                        reason.as_ref(),
                        FlipPredicateError::CoordinateConversion {
                            operation,
                            source: CoordinateConversionError::UnsupportedMatrixDimension {
                                requested: 4,
                                max: 3,
                            },
                        } if *operation == expected_operation
                    )
            );
            assert_eq!(diagnostics.predicate_failures, 1);
        }
    }

    #[test]
    fn inverse_k1_vertex_cleanup_propagates_tds_failure() {
        let mut tds = Tds::<(), (), 2>::empty();
        let vertex_key = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let generation_before = tds.generation();
        let dangling_simplex = SimplexKey::from(KeyData::from_ffi(42));
        tds.add_simplex_to_vertex_incidence_for_test(vertex_key, dangling_simplex);

        let error = remove_k1_inverse_vertex(&mut tds, vertex_key).unwrap_err();

        let FlipError::TdsMutation { reason } = error else {
            panic!("expected typed TDS mutation error");
        };
        let FlipMutationError::VertexRemoval { source } = reason.as_ref() else {
            panic!("expected inverse-k1 vertex-removal error");
        };
        assert_matches!(
            source.as_tds_error(),
            TdsError::InconsistentDataStructure { .. }
        );
        assert!(tds.contains_vertex_key(vertex_key));
        assert_eq!(tds.generation(), generation_before);
    }

    /// Builds a simplex-basis vertex coordinate for dimension-generic flip tests.
    fn unit_vector<const D: usize>(index: usize) -> [f64; D] {
        let mut coords = [0.0; D];
        coords[index] = 1.0;
        coords
    }
    fn assert_context_has_nonzero_robust_orientation(
        tds: &Tds<(), (), 2>,
        context: &FlipContext<2, 2>,
    ) {
        for &omit in &context.removed_face_vertices {
            let mut verts: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::with_capacity(3);
            verts.extend_from_slice(&context.inserted_face_vertices);
            for &v in &context.removed_face_vertices {
                if v != omit {
                    verts.push(v);
                }
            }
            let points = vertices_to_points(tds, &verts).unwrap();
            match robust_orientation(&points) {
                Ok(Orientation::POSITIVE | Orientation::NEGATIVE) => {}
                other => panic!("robust_orientation must resolve to ±1, got {other:?}"),
            }
        }
    }

    fn synthetic_vertex_key(index: u64) -> VertexKey {
        VertexKey::from(KeyData::from_ffi(index))
    }

    fn synthetic_simplex_key(index: u64) -> SimplexKey {
        SimplexKey::from(KeyData::from_ffi(index))
    }

    fn to_dynamic<const D: usize, const K: usize>(context: FlipContext<D, K>) -> FlipContextDyn<D> {
        FlipContextDyn {
            removed_face_vertices: context.removed_face_vertices,
            inserted_face_vertices: context.inserted_face_vertices,
            removed_simplices: context.removed_simplices,
            direction: context.direction,
        }
    }

    fn dynamic_flip_rejects_bad_context_for_dimension<const D: usize>() {
        init_tracing();
        let mut tds: Tds<(), (), D> = Tds::empty();
        let vertices = (1..=D + 2)
            .map(|index| {
                synthetic_vertex_key(
                    u64::try_from(index).expect("test vertex key index should fit in u64"),
                )
            })
            .collect::<Vec<_>>();
        let c0 = synthetic_simplex_key(11);
        let c1 = synthetic_simplex_key(12);

        let valid_shape = FlipContextDyn {
            removed_face_vertices: vertices[..D].iter().copied().collect(),
            inserted_face_vertices: vertices[D..D + 2].iter().copied().collect(),
            removed_simplices: [c0, c1].into_iter().collect(),
            direction: FlipDirection::Forward,
        };

        assert_matches!(
            apply_bistellar_flip_dynamic_raw(&mut tds, 0, &valid_shape),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::InvalidMoveSize {
                        k_move: 0,
                        dimension,
                    } if *dimension == D
                )
        );
        assert_matches!(
            apply_bistellar_flip_dynamic_raw(&mut tds, D + 2, &valid_shape),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::InvalidMoveSize {
                        k_move,
                        dimension,
                    } if *k_move == D + 2 && *dimension == D
                )
        );

        let wrong_removed_face = FlipContextDyn {
            removed_face_vertices: vertices[..D - 1].iter().copied().collect(),
            ..valid_shape.clone()
        };
        assert_matches!(
            apply_bistellar_flip_dynamic_raw(&mut tds, 2, &wrong_removed_face),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::WrongRemovedFaceArity {
                        expected,
                        found,
                    } if *expected == D && *found == D - 1
                )
        );

        let wrong_inserted_face = FlipContextDyn {
            inserted_face_vertices: once(vertices[D]).collect(),
            ..valid_shape.clone()
        };
        assert_matches!(
            apply_bistellar_flip_dynamic_raw(&mut tds, 2, &wrong_inserted_face),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::WrongInsertedFaceArity {
                        k_move: 2,
                        expected: 2,
                        found: 1,
                    }
                )
        );

        let wrong_removed_simplices = FlipContextDyn {
            removed_simplices: once(c0).collect(),
            ..valid_shape.clone()
        };
        assert_matches!(
            apply_bistellar_flip_dynamic_raw(&mut tds, 2, &wrong_removed_simplices),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(
                    reason.as_ref(),
                    FlipContextError::WrongRemovedSimplexCount {
                        expected: 2,
                        found: 1,
                    }
                )
        );

        let overlapping_faces = FlipContextDyn {
            inserted_face_vertices: [vertices[D - 1], vertices[D]].into_iter().collect(),
            ..valid_shape
        };
        assert_matches!(
            apply_bistellar_flip_dynamic_raw(&mut tds, 2, &overlapping_faces),
            Err(FlipError::InvalidFlipContext { reason })
                if matches!(reason.as_ref(), FlipContextError::OverlappingFaces)
        );
        assert_eq!(tds.number_of_vertices(), 0);
        assert_eq!(tds.number_of_simplices(), 0);
    }

    macro_rules! gen_dynamic_flip_bad_context_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<dynamic_flip_rejects_bad_context_ $dim d>]() {
                    dynamic_flip_rejects_bad_context_for_dimension::<$dim>();
                }
            }
        };
    }

    gen_dynamic_flip_bad_context_tests!(2);
    gen_dynamic_flip_bad_context_tests!(3);
    gen_dynamic_flip_bad_context_tests!(4);
    gen_dynamic_flip_bad_context_tests!(5);

    #[test]
    fn test_flip_k2_2d_edge_flip() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.2]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2); // facet opposite vertex index 2 (edge AB)
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let feasibility = validate_bistellar_flip_k2(&tds, &context).unwrap();
        assert_eq!(feasibility.kind, BistellarFlipKind::from_validated(2, 2));
        assert_eq!(feasibility.direction, FlipDirection::Forward);
        assert_eq!(feasibility.removed_simplices.len(), 2);
        assert_eq!(
            feasibility
                .inserted_face_vertices
                .as_ref()
                .map(SmallBuffer::len),
            Some(2)
        );
        let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        assert_eq!(info.removed_simplices.len(), 2);
        assert_eq!(info.new_simplices.len(), 2);

        // After flip, we should have an edge between c and d in some simplex.
        let mut has_cd = false;
        for (_, simplex) in tds.simplices() {
            let verts = simplex.vertices();
            if verts.contains(&c) && verts.contains(&d) {
                has_cd = true;
            }
        }
        assert!(has_cd, "Expected flipped diagonal between c and d");

        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_rejects_duplicate_simplex() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.2]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, d], None).unwrap())
            .unwrap();

        // Pre-existing simplex that the flip would recreate (B,C,D)
        let _existing = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![b, c, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2); // facet opposite vertex index 2 (edge AB)
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let feasibility = validate_bistellar_flip_k2(&tds, &context);
        assert_matches!(feasibility, Err(FlipError::DuplicateSimplex));
        let result = apply_bistellar_flip_raw(&mut tds, &context);

        assert_matches!(result, Err(FlipError::DuplicateSimplex));
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_rejects_inserting_existing_edge_in_3d() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();

        // Opposite vertices across the shared face.
        let v_a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
            .unwrap();

        // Shared face vertices.
        let v_x = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let v_y = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let v_z = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 1.0]).unwrap())
            .unwrap();

        // Extra vertices for an existing tetrahedron containing the edge (v_a, v_b).
        let v_p = tds
            .insert_vertex_with_mapping(vertex!([2.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v_q = tds
            .insert_vertex_with_mapping(vertex!([2.0, 1.0, 0.0]).unwrap())
            .unwrap();

        // Two tetrahedra sharing face (v_x, v_y, v_z): a k=2 flip across that face would insert edge (v_a, v_b).
        let simplex_a = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_x, v_y, v_z], None).unwrap(),
            )
            .unwrap();
        let _simplex_b = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_b, v_x, v_y, v_z], None).unwrap(),
            )
            .unwrap();

        // Existing tetrahedron that already contains edge (v_a, v_b) but does not contain any of
        // the shared-face vertices (v_x, v_y, v_z).
        let _edge_witness = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_p, v_q], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();
        assert!(tds.is_valid().is_ok());

        // Face (v_x, v_y, v_z) is opposite v_a in `simplex_a` (index 0 by construction).
        let facet = FacetHandle::from_validated(simplex_a, 0);
        let ctx = build_k2_flip_context(&tds, facet).unwrap();

        let feasibility = validate_bistellar_flip_k2(&tds, &ctx);
        assert_matches!(
            feasibility,
            Err(FlipError::InsertedSimplexAlreadyExists { .. })
        );
        let result = apply_bistellar_flip_raw(&mut tds, &ctx);

        assert_matches!(result, Err(FlipError::InsertedSimplexAlreadyExists { .. }));
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_rejects_nonmanifold_internal_facet() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.2]).unwrap())
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(vertex!([2.0, 2.0]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_d], None).unwrap(),
            )
            .unwrap();

        // Existing simplex containing the would-be inserted diagonal (C,D).
        let _cd_external = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_c, v_d, v_e], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2); // facet opposite vertex index 2 (edge AB)
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let feasibility = validate_bistellar_flip_k2(&tds, &context);
        assert_matches!(feasibility, Err(FlipError::NonManifoldFacet));
        let result = apply_bistellar_flip_raw(&mut tds, &context);

        assert_matches!(result, Err(FlipError::NonManifoldFacet));
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_3d_two_to_three() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let v_a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v_b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let v_c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let v_d = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, 1.0]).unwrap())
            .unwrap();
        let v_e = tds
            .insert_vertex_with_mapping(vertex!([0.3, -0.1, -0.8]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_d], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_e], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 3); // facet opposite vertex d (ABC)
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        assert_eq!(info.new_simplices.len(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k3_3d_three_to_two() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, -1.0]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 2, 3);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::from_validated(3, 3));
        assert_eq!(info.removed_simplices.len(), 3);
        assert_eq!(info.new_simplices.len(), 2);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k3_4d_three_to_three() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, 0.2, 0.2]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 3, 4);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::from_validated(3, 4));
        assert_eq!(info.removed_simplices.len(), 3);
        assert_eq!(info.new_simplices.len(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k3_5d_three_to_four() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, 0.2, 0.2, 0.5]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 4, 5);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::from_validated(3, 5));
        assert_eq!(info.removed_simplices.len(), 3);
        assert_eq!(info.new_simplices.len(), 4);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k2_boundary_facet_error_2d() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let simplex = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();

        let before = snapshot_topology(&tds);
        let facet = FacetHandle::from_validated(simplex, 0);
        let err = build_k2_flip_context(&tds, facet).unwrap_err();
        assert_matches!(err, FlipError::BoundaryFacet { .. });
        assert_eq!(snapshot_topology(&tds), before);
    }

    #[test]
    fn test_flip_k3_invalid_ridge_multiplicity_3d() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![a, b, c, d], None).unwrap(),
            )
            .unwrap();

        let ridge = RidgeHandle::from_validated(simplex, 0, 1);
        let err = build_k3_flip_context(&tds, ridge).unwrap_err();
        assert_matches!(err, FlipError::InvalidRidgeMultiplicity { found: 1 });
    }

    #[test]
    fn test_flip_k3_reports_dangling_ridge_neighbor_3d() {
        init_tracing();
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let ridge_start = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let ridge_end = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let first_opposite = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let second_opposite = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let dangling_opposite = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0, 1.0]).unwrap())
            .unwrap();
        let simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(
                    vec![ridge_start, ridge_end, first_opposite, second_opposite],
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let dangling_neighbor = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(
                    vec![ridge_start, ridge_end, first_opposite, dangling_opposite],
                    None,
                )
                .unwrap(),
            )
            .unwrap();

        assert_eq!(
            tds.remove_simplices_by_keys(&[dangling_neighbor]).unwrap(),
            1
        );
        tds.simplex_mut(simplex)
            .expect("test simplex should exist")
            .set_neighbors_from_keys([Some(dangling_neighbor), None, None, None])
            .unwrap();

        let ridge = RidgeHandle::from_validated(simplex, 0, 1);
        let err = build_k3_flip_context(&tds, ridge).unwrap_err();
        assert_eq!(
            err,
            FlipError::DanglingRidgeNeighbor {
                simplex_key: simplex,
                neighbor_key: dangling_neighbor,
            }
        );
    }

    #[test]
    fn test_flip_k2_inverse_invalid_edge_multiplicity_4d() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(4);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<4>(i)).unwrap())
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(vertex!([0.0; 4]).unwrap())
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(vertex!([1.0; 4]).unwrap())
            .unwrap();

        let mut vertices_with_first_opposite = shared_vertices.clone();
        vertices_with_first_opposite.push(opposite_a);
        let _simplex_a = tds
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

        let edge = EdgeKey::from_validated_endpoints(opposite_a, opposite_b);
        let err = build_k2_flip_context_from_edge(&tds, edge).unwrap_err();
        assert_matches!(err, FlipError::InvalidEdgeMultiplicity { .. });
    }

    #[test]
    fn test_flip_k3_inverse_invalid_triangle_multiplicity_5d() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let origin = tds
            .insert_vertex_with_mapping(vertex!([0.0; 5]).unwrap())
            .unwrap();
        let mut vertices = Vec::with_capacity(6);
        vertices.push(origin);
        for i in 0..5 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<5>(i)).unwrap())
                .unwrap();
            vertices.push(v);
        }
        let _simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vertices.clone(), None).unwrap(),
            )
            .unwrap();

        let triangle = TriangleHandle::try_new(vertices[0], vertices[1], vertices[2]).unwrap();
        let err = build_k3_flip_context_from_triangle(&tds, triangle).unwrap_err();
        assert_matches!(
            err,
            FlipError::InvalidTriangleMultiplicity {
                found: 1,
                expected: 4,
            }
        );
    }

    #[test]
    fn test_dynamic_k2_forward_4d() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(4);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<4>(i)).unwrap())
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(vertex!([0.0; 4]).unwrap())
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(vertex!([1.0; 4]).unwrap())
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

        let facet = FacetHandle::from_validated(simplex_a, 4);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let context_dyn = to_dynamic(context);
        let info = apply_bistellar_flip_dynamic_raw(&mut tds, 2, &context_dyn).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::from_validated(2, 4));
        assert_eq!(info.removed_simplices.len(), 2);
        assert_eq!(info.new_simplices.len(), 4);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_dynamic_k3_forward_5d() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, 0.2, 0.2, 0.5]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 4, 5);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let context_dyn = to_dynamic(context);
        let info = apply_bistellar_flip_dynamic_raw(&mut tds, 3, &context_dyn).unwrap();

        assert_eq!(info.kind, BistellarFlipKind::from_validated(3, 5));
        assert_eq!(info.removed_simplices.len(), 3);
        assert_eq!(info.new_simplices.len(), 4);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_k2_roundtrip_randomized_3d() {
        init_tracing();
        let mut rng = StdRng::seed_from_u64(0x51f1_7a2b);

        for _ in 0..10 {
            let mut jitter = |v: [f64; 3]| {
                let mut out = v;
                for coord in &mut out {
                    *coord += rng.random_range(-0.03..0.03);
                }
                out
            };

            let mut tds: Tds<(), (), 3> = Tds::empty();
            let v_a = tds
                .insert_vertex_with_mapping(vertex!(jitter([0.0, 0.0, 0.0])).unwrap())
                .unwrap();
            let v_b = tds
                .insert_vertex_with_mapping(vertex!(jitter([1.0, 0.0, 0.0])).unwrap())
                .unwrap();
            let v_c = tds
                .insert_vertex_with_mapping(vertex!(jitter([0.0, 1.0, 0.0])).unwrap())
                .unwrap();
            let v_d = tds
                .insert_vertex_with_mapping(vertex!(jitter([0.2, 0.2, 1.0])).unwrap())
                .unwrap();
            let v_e = tds
                .insert_vertex_with_mapping(vertex!(jitter([0.3, -0.1, -0.8])).unwrap())
                .unwrap();

            let c1 = tds
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_d], None).unwrap(),
                )
                .unwrap();
            let _c2 = tds
                .insert_simplex_with_mapping(
                    Simplex::try_new_with_data(vec![v_a, v_b, v_c, v_e], None).unwrap(),
                )
                .unwrap();

            repair_neighbor_pointers(&mut tds).unwrap();

            let before = snapshot_topology(&tds);
            let facet = FacetHandle::from_validated(c1, 3);
            let context = build_k2_flip_context(&tds, facet).unwrap();
            let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();
            assert!(tds.is_valid().is_ok());

            let edge = EdgeKey::from_validated_endpoints(
                info.inserted_face_vertices[0],
                info.inserted_face_vertices[1],
            );
            let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
            let _info_back = apply_bistellar_flip_dynamic_raw(&mut tds, 3, &context_back).unwrap();

            assert!(tds.is_valid().is_ok());
            let after = snapshot_topology(&tds);
            assert_same_vertex_simplex_topology(&after, &before);
        }
    }

    #[test]
    fn test_flip_k2_robust_kernel_near_degenerate_2d() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1e-9]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let _info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        assert!(tds.is_valid().is_ok());
    }

    /// Verifies that `k2_flip_would_create_degenerate_simplex` detects a degenerate
    /// replacement simplex (collinear vertices in 2D).
    #[test]
    fn test_k2_flip_would_create_degenerate_simplex_degenerate() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        // a, c, d are collinear on the x-axis → replacement simplex {a,c,d} is degenerate
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([0.5, 0.0]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2);
        let context = build_k2_flip_context(&tds, facet).unwrap();

        let degenerate = k2_flip_would_create_degenerate_simplex(&tds, &context).unwrap();
        assert!(
            degenerate,
            "replacement simplices with collinear vertices should be degenerate"
        );
    }

    /// Verifies that `k2_flip_would_create_degenerate_simplex` returns false for
    /// non-degenerate simplices using `robust_orientation` (kernel-independent).
    #[test]
    fn test_k2_flip_would_create_degenerate_simplex_nondegenerate() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();
        let d = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, d], None).unwrap())
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let facet = FacetHandle::from_validated(c1, 2);
        let context = build_k2_flip_context(&tds, facet).unwrap();

        assert_context_has_nonzero_robust_orientation(&tds, &context);

        let degenerate = k2_flip_would_create_degenerate_simplex(&tds, &context).unwrap();
        assert!(!degenerate);
    }

    #[test]
    fn test_flip_k4_4d_four_to_two() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(4);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<4>(i)).unwrap())
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(vertex!([0.0; 4]).unwrap())
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(vertex!([1.0; 4]).unwrap())
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

        let facet = FacetHandle::from_validated(simplex_a, 4);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let _info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        let edge = EdgeKey::from_validated_endpoints(opposite_a, opposite_b);
        let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
        let info_back = apply_bistellar_flip_dynamic_raw(&mut tds, 4, &context_back).unwrap();

        assert_eq!(info_back.kind.k(), 4);
        assert_eq!(info_back.kind.d(), 4);
        assert_eq!(info_back.removed_simplices.len(), 4);
        assert_eq!(info_back.new_simplices.len(), 2);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k5_4d_five_to_one() {
        init_tracing();
        let mut tds: Tds<(), (), 4> = Tds::empty();
        let origin = tds
            .insert_vertex_with_mapping(vertex!([0.0; 4]).unwrap())
            .unwrap();
        let mut vertices = Vec::with_capacity(5);
        vertices.push(origin);
        for i in 0..4 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<4>(i)).unwrap())
                .unwrap();
            vertices.push(v);
        }

        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vertices, None).unwrap())
            .unwrap();

        let new_vertex = vertex!([0.1; 4]).unwrap();
        let new_uuid = new_vertex.uuid();
        let info = apply_bistellar_flip_k1_raw(&mut tds, simplex_key, new_vertex).unwrap();

        assert_eq!(info.kind.k(), 1);
        assert_eq!(info.new_simplices.len(), 5);

        let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
        let info_back = apply_bistellar_flip_k1_inverse_raw(&mut tds, new_key).unwrap();

        assert_eq!(info_back.kind.k(), 5);
        assert_eq!(info_back.kind.d(), 4);
        assert_eq!(info_back.removed_simplices.len(), 5);
        assert_eq!(info_back.new_simplices.len(), 1);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k4_5d_four_to_three() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let r0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let r3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.2, 0.2, 0.2, 0.2, 0.5]).unwrap())
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, a, b], None).unwrap(),
            )
            .unwrap();
        let _c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, b, c], None).unwrap(),
            )
            .unwrap();
        let _c3 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![r0, r1, r2, r3, c, a], None).unwrap(),
            )
            .unwrap();

        repair_neighbor_pointers(&mut tds).unwrap();

        let ridge = RidgeHandle::from_validated(c1, 4, 5);
        let context = build_k3_flip_context(&tds, ridge).unwrap();
        let info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        assert_eq!(info.kind.k(), 3);
        assert_eq!(info.inserted_face_vertices.len(), 3);

        let triangle = TriangleHandle::try_new(
            info.inserted_face_vertices[0],
            info.inserted_face_vertices[1],
            info.inserted_face_vertices[2],
        )
        .unwrap();
        let context_back = build_k3_flip_context_from_triangle(&tds, triangle).unwrap();
        let info_back = apply_bistellar_flip_dynamic_raw(&mut tds, 4, &context_back).unwrap();

        assert_eq!(info_back.kind.k(), 4);
        assert_eq!(info_back.kind.d(), 5);
        assert_eq!(info_back.removed_simplices.len(), 4);
        assert_eq!(info_back.new_simplices.len(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k5_5d_five_to_two() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let mut shared_vertices = Vec::with_capacity(5);
        for i in 0..5 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<5>(i)).unwrap())
                .unwrap();
            shared_vertices.push(v);
        }

        let opposite_a = tds
            .insert_vertex_with_mapping(vertex!([0.0; 5]).unwrap())
            .unwrap();
        let opposite_b = tds
            .insert_vertex_with_mapping(vertex!([1.0; 5]).unwrap())
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

        let facet = FacetHandle::from_validated(simplex_a, 5);
        let context = build_k2_flip_context(&tds, facet).unwrap();
        let _info = apply_bistellar_flip_raw(&mut tds, &context).unwrap();

        let edge = EdgeKey::from_validated_endpoints(opposite_a, opposite_b);
        let context_back = build_k2_flip_context_from_edge(&tds, edge).unwrap();
        let info_back = apply_bistellar_flip_dynamic_raw(&mut tds, 5, &context_back).unwrap();

        assert_eq!(info_back.kind.k(), 5);
        assert_eq!(info_back.kind.d(), 5);
        assert_eq!(info_back.removed_simplices.len(), 5);
        assert_eq!(info_back.new_simplices.len(), 2);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_flip_k6_5d_six_to_one() {
        init_tracing();
        let mut tds: Tds<(), (), 5> = Tds::empty();
        let origin = tds
            .insert_vertex_with_mapping(vertex!([0.0; 5]).unwrap())
            .unwrap();
        let mut vertices = Vec::with_capacity(6);
        vertices.push(origin);
        for i in 0..5 {
            let v = tds
                .insert_vertex_with_mapping(vertex!(unit_vector::<5>(i)).unwrap())
                .unwrap();
            vertices.push(v);
        }

        let simplex_key = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vertices, None).unwrap())
            .unwrap();

        let new_vertex = vertex!([0.1; 5]).unwrap();
        let new_uuid = new_vertex.uuid();
        let info = apply_bistellar_flip_k1_raw(&mut tds, simplex_key, new_vertex).unwrap();

        assert_eq!(info.kind.k(), 1);
        assert_eq!(info.new_simplices.len(), 6);

        let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
        let info_back = apply_bistellar_flip_k1_inverse_raw(&mut tds, new_key).unwrap();

        assert_eq!(info_back.kind.k(), 6);
        assert_eq!(info_back.kind.d(), 5);
        assert_eq!(info_back.removed_simplices.len(), 6);
        assert_eq!(info_back.new_simplices.len(), 1);
        assert!(tds.is_valid().is_ok());
    }
    #[test]
    fn test_flip_k1_2d_roundtrip() {
        init_tracing();
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let a = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let b = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let c = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
            .unwrap();

        let simplex = tds
            .insert_simplex_with_mapping(Simplex::try_new_with_data(vec![a, b, c], None).unwrap())
            .unwrap();

        let new_vertex = vertex!([0.2, 0.2]).unwrap();
        let new_uuid = new_vertex.uuid();
        let info = apply_bistellar_flip_k1_raw(&mut tds, simplex, new_vertex).unwrap();

        assert_eq!(info.kind.k(), 1);
        assert_eq!(info.kind.d(), 2);
        assert_eq!(tds.number_of_simplices(), 3);

        let new_key = tds.vertex_key_from_uuid(&new_uuid).unwrap();
        let info_back = apply_bistellar_flip_k1_inverse_raw(&mut tds, new_key).unwrap();

        assert_eq!(info_back.kind.k(), 3);
        assert_eq!(info_back.kind.d(), 2);
        assert_eq!(tds.number_of_simplices(), 1);
        assert_eq!(tds.number_of_vertices(), 3);
        assert!(tds.is_valid().is_ok());
    }

    #[test]
    fn test_coherent_orientation_violation_maps_to_invalid_flip_context() {
        let err: FlipError = FlipContextError::CoherentOrientationViolation {
            stage: FlipOrientationCheckStage::BeforeMutation,
            k_move: 2,
            direction: FlipDirection::Forward,
        }
        .into();

        assert_matches!(
            err,
            FlipError::InvalidFlipContext { reason }
                if matches!(
                    reason.as_ref(),
                    FlipContextError::CoherentOrientationViolation {
                        stage: FlipOrientationCheckStage::BeforeMutation,
                        k_move: 2,
                        direction: FlipDirection::Forward
                    }
                )
        );
    }

    #[test]
    fn test_coherent_orientation_violation_maps_to_tds_mutation() {
        let err: FlipError = FlipMutationError::CoherentOrientationViolation {
            stage: FlipOrientationCheckStage::AfterTrialMutation,
            k_move: 2,
            direction: FlipDirection::Forward,
        }
        .into();

        assert_eq!(
            FlipFailureKind::from(&err),
            FlipFailureKind::TrialValidation
        );
        assert_matches!(
            err,
            FlipError::TdsMutation { reason }
                if matches!(
                    reason.as_ref(),
                    FlipMutationError::CoherentOrientationViolation {
                        stage: FlipOrientationCheckStage::AfterTrialMutation,
                        k_move: 2,
                        direction: FlipDirection::Forward
                    }
                )
        );
    }
}

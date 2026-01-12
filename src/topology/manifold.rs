//! # PL-Manifold Validation (Topology Only)
//!
//! This module implements *topological* (combinatorial) invariants that are sufficient
//! to guarantee that a finite simplicial complex is a **piecewise-linear (PL) manifold
//! with boundary**, assuming the underlying triangulation data structure (TDS) is
//! structurally consistent.
//!
//! ## What is being validated
//!
//! Let `K` be a pure D-dimensional simplicial complex. Classical PL topology shows that
//! `K` is a PL-manifold with boundary if and only if the following local conditions hold:
//!
//! 1. **Codimension-1 (facet) condition**
//!    Every (D−1)-simplex is incident to exactly one or two D-simplices.
//!
//! 2. **Boundary codimension-2 condition**
//!    Every (D−2)-simplex lying on the boundary is incident to exactly two boundary facets
//!    (equivalently, ∂² = ∅).
//!
//! 3. **Codimension-2 link condition**
//!    The link of every (D−2)-simplex is a 1-dimensional PL-manifold:
//!    - a cycle (S¹) for interior ridges, or
//!    - a path (I) for boundary ridges.
//!
//! These conditions are enforced respectively by:
//!
//! - [`validate_facet_degree`]
//! - [`validate_closed_boundary`]
//! - [`validate_ridge_links`]
//!
//! Together, they are **sufficient** to guarantee that the link of *every* simplex
//! (including vertices) is a sphere or ball of the appropriate dimension, which is the
//! defining property of a PL-manifold.
//!
//! ## What is *not* checked here
//!
//! - Geometric predicates (e.g. Delaunay conditions)
//! - Metric properties
//! - Global topology classification (e.g. genus, Euler characteristic)
//! - TDS structural invariants (neighbor pointers, index validity, etc.)
//!
//! Those concerns are handled elsewhere:
//!
//! - TDS structural correctness is validated at **Level 2**
//! - Global invariants (Euler characteristic, classification) live in
//!   `topology::characteristics`
//!
//! ## Why ridge links instead of vertex links?
//!
//! Although PL-manifoldness is often defined in terms of **vertex links**, it is a
//! classical result that for simplicial complexes it is sufficient to verify
//! manifoldness of links in codimensions 1 and 2. In practice, validating **ridge links**
//! is both stronger and more efficient than attempting to classify vertex links directly,
//! and avoids false positives that can arise from scalar invariants such as Euler
//! characteristic alone.
//!
//! ## References
//!
//! - J. R. Munkres, *Elements of Algebraic Topology*, Addison–Wesley, 1984.
//!   (Chapter 9: Simplicial Manifolds and Links.)
//!
//! - C. P. Rourke & B. J. Sanderson, *Introduction to Piecewise-Linear Topology*,
//!   Springer, 1972.
//!
//! - A. Hatcher, *Algebraic Topology*, Cambridge University Press, 2002.
//!   (Appendix A: PL Manifolds and Links.)
//!
//! - H. Edelsbrunner & J. Harer, *Computational Topology*, AMS, 2010.
//!   (Sections on pseudomanifolds and manifold conditions in simplicial complexes.)
//!
//! These invariants are standard in computational geometry, Regge calculus, and
//! Causal Dynamical Triangulations (CDT), where curvature and local neighborhood
//! structure are only well-defined for simplicial PL-manifolds.

use thiserror::Error;

use crate::core::{
    collections::{
        CellKeySet, FacetToCellsMap, FastHashMap, FastHashSet, SmallBuffer, VertexKeyBuffer,
        fast_hash_map_with_capacity, fast_hash_set_with_capacity,
    },
    edge::EdgeKey,
    facet::facet_key_from_vertices,
    traits::DataType,
    triangulation_data_structure::{CellKey, Tds, TdsValidationError, VertexKey},
};
use crate::geometry::traits::coordinate::CoordinateScalar;

/// Errors that can occur during manifold (topology) validation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ManifoldError {
    /// The underlying triangulation data structure is internally inconsistent.
    #[error(transparent)]
    Tds(#[from] TdsValidationError),

    /// A facet belongs to an unexpected number of cells for a manifold-with-boundary.
    #[error(
        "Non-manifold facet: facet {facet_key:016x} belongs to {cell_count} cells (expected 1 or 2)"
    )]
    ManifoldFacetMultiplicity {
        /// The facet key with invalid multiplicity.
        facet_key: u64,
        /// The number of incident cells observed.
        cell_count: usize,
    },

    /// Boundary is not a closed (D-1)-manifold: a ridge on the boundary is incident to the
    /// wrong number of boundary facets.
    ///
    /// This detects "boundary of boundary" issues (codimension-2 manifoldness of the boundary).
    #[error(
        "Boundary is not closed: boundary ridge {ridge_key:016x} is incident to {boundary_facet_count} boundary facets (expected 2)"
    )]
    BoundaryRidgeMultiplicity {
        /// Canonical key for the (D-2)-simplex (ridge) on the boundary.
        ridge_key: u64,
        /// Number of incident boundary facets observed.
        boundary_facet_count: usize,
    },

    /// A ridge's link graph is not a 1-manifold (path or cycle).
    ///
    /// In a PL-manifold (with boundary), the link of every (D-2)-simplex is:
    /// - a cycle (S¹) for interior ridges, or
    /// - a path (I) for boundary ridges.
    #[error(
        "Ridge link is not a 1-manifold: ridge {ridge_key:016x} has link graph with {link_vertex_count} vertices, {link_edge_count} edges, max degree {max_degree}, degree-1 vertices {degree_one_vertices}, connected={connected} (expected connected cycle or path)"
    )]
    RidgeLinkNotManifold {
        /// Canonical key for the (D-2)-simplex (ridge).
        ridge_key: u64,
        /// Number of vertices in the ridge's link graph.
        link_vertex_count: usize,
        /// Number of edges in the ridge's link graph.
        link_edge_count: usize,
        /// Maximum vertex degree observed in the link graph.
        max_degree: usize,
        /// Number of vertices of degree 1 observed in the link graph.
        degree_one_vertices: usize,
        /// Whether the link graph is connected.
        connected: bool,
    },
}

/// Validates that each (D-1)-facet has degree 1 (boundary) or 2 (interior).
///
/// This enforces the codimension-1 pseudomanifold condition and is not sufficient by itself
/// to guarantee full PL-manifoldness.
///
/// # Errors
///
/// Returns [`ManifoldError::ManifoldFacetMultiplicity`] if any facet is incident
/// to a number of cells other than 1 or 2.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::topology::manifold::validate_facet_degree;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
/// let facet_to_cells = tds.build_facet_to_cells_map().unwrap();
///
/// validate_facet_degree(&facet_to_cells).unwrap();
/// ```
pub fn validate_facet_degree(facet_to_cells: &FacetToCellsMap) -> Result<(), ManifoldError> {
    for (facet_key, cell_facet_pairs) in facet_to_cells {
        match cell_facet_pairs.as_slice() {
            [_] | [_, _] => {}
            _ => {
                return Err(ManifoldError::ManifoldFacetMultiplicity {
                    facet_key: *facet_key,
                    cell_count: cell_facet_pairs.len(),
                });
            }
        }
    }

    Ok(())
}

/// Validates that the boundary (if present) is a closed (D-1)-manifold.
///
/// This is the codimension-2 pseudomanifold / manifold-with-boundary condition for
/// triangulations: every (D-2)-simplex (ridge) that lies on the boundary must be
/// incident to exactly 2 boundary facets.
/// This enforces the identity ∂² = ∅ (the boundary of a manifold has no boundary).
///
/// # Errors
///
/// Returns:
/// - [`ManifoldError::Tds`] if the underlying triangulation data structure is internally inconsistent.
/// - [`ManifoldError::BoundaryRidgeMultiplicity`] if a boundary ridge is incident to
///   a number of boundary facets other than 2.
///
/// Notes:
/// - Interior ridges can have arbitrary degree; this check only counts incidence among
///   boundary facets (facets with exactly 1 incident D-cell).
/// - If the triangulation has no boundary facets, this check is a no-op.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::topology::manifold::validate_closed_boundary;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
/// let facet_to_cells = tds.build_facet_to_cells_map().unwrap();
///
/// validate_closed_boundary(&tds, &facet_to_cells).unwrap();
/// ```
pub fn validate_closed_boundary<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    facet_to_cells: &FacetToCellsMap,
) -> Result<(), ManifoldError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // The boundary is a (D-1)-complex. Codimension-2 manifoldness is only meaningful for D>=2.
    if D < 2 {
        return Ok(());
    }

    // First count boundary facets so we can reserve reasonably.
    let boundary_facet_count = facet_to_cells
        .values()
        .filter(|handles| matches!(handles.as_slice(), [_]))
        .count();

    if boundary_facet_count == 0 {
        return Ok(());
    }

    // Each boundary facet contributes D ridges; each boundary ridge is shared by exactly 2
    // boundary facets in a closed boundary manifold.
    let estimated_boundary_ridges = boundary_facet_count
        .saturating_mul(D)
        .saturating_div(2)
        .max(1);

    let mut ridge_to_boundary_facet_count: FastHashMap<u64, usize> =
        fast_hash_map_with_capacity(estimated_boundary_ridges);

    let mut facet_vertices: VertexKeyBuffer = VertexKeyBuffer::with_capacity(D);
    let mut ridge_vertices: VertexKeyBuffer = VertexKeyBuffer::with_capacity(D.saturating_sub(1));

    for cell_facet_pairs in facet_to_cells.values() {
        // Only boundary facets (exactly one incident cell).
        let [handle] = cell_facet_pairs.as_slice() else {
            continue;
        };

        let cell_key = handle.cell_key();
        let facet_index = handle.facet_index() as usize;

        // Derive the facet's vertex keys from the owning cell.
        let cell_vertices = tds.get_cell_vertices(cell_key)?;
        if facet_index >= cell_vertices.len() {
            return Err(TdsValidationError::InconsistentDataStructure {
                message: format!(
                    "Boundary facet index {facet_index} out of bounds: cell {cell_key:?} has {} vertices",
                    cell_vertices.len()
                ),
            }
            .into());
        }

        facet_vertices.clear();
        for (i, &vk) in cell_vertices.iter().enumerate() {
            if i == facet_index {
                continue;
            }
            facet_vertices.push(vk);
        }

        if facet_vertices.len() != D {
            return Err(TdsValidationError::InconsistentDataStructure {
                message: format!(
                    "Boundary facet expected {D} vertices, got {} (cell_key={cell_key:?}, facet_index={facet_index})",
                    facet_vertices.len()
                ),
            }
            .into());
        }

        // Enumerate the (D-2)-faces (ridges) of this boundary facet by excluding each
        // facet vertex in turn.
        for omit in 0..facet_vertices.len() {
            ridge_vertices.clear();
            for (j, &vk) in facet_vertices.iter().enumerate() {
                if j == omit {
                    continue;
                }
                ridge_vertices.push(vk);
            }

            let ridge_key = facet_key_from_vertices(&ridge_vertices);
            *ridge_to_boundary_facet_count.entry(ridge_key).or_insert(0) += 1;
        }
    }

    for (ridge_key, boundary_facet_count) in ridge_to_boundary_facet_count {
        if boundary_facet_count != 2 {
            return Err(ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            });
        }
    }

    Ok(())
}

/// Computes the star of a ridge (a (D-2)-simplex) as the set of incident D-cells.
///
/// This is a local combinatorial query intended for reuse by topology validation and
/// (future) local topology mutations (e.g. bistellar flips).
///
/// This helper does **not** call `tds.is_valid()`; it performs lightweight checks and
/// returns [`ManifoldError::Tds`] if the underlying TDS is internally inconsistent.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Not used yet; intended for reuse by future local topology mutations (e.g. bistellar flips)"
    )
)]
pub(crate) fn ridge_star_cells<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    ridge_vertices: &[VertexKey],
) -> Result<SmallBuffer<CellKey, 8>, ManifoldError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Ridge stars are only meaningful for D>=2.
    if D < 2 {
        return Ok(SmallBuffer::new());
    }

    let expected_ridge_vertices = D.saturating_sub(1);
    if ridge_vertices.len() != expected_ridge_vertices {
        return Err(TdsValidationError::InconsistentDataStructure {
            message: format!(
                "Ridge expected {expected_ridge_vertices} vertices for {D}D, got {}",
                ridge_vertices.len(),
            ),
        }
        .into());
    }

    // Defensive: ensure all ridge vertices exist in the vertex store.
    //
    // Note: This is cheaper than `tds.is_valid()` and provides a clearer error when
    // callers use this helper on stale keys.
    for &vk in ridge_vertices {
        if !tds.contains_vertex_key(vk) {
            return Err(TdsValidationError::InconsistentDataStructure {
                message: format!("Ridge vertex {vk:?} not found in vertices storage map"),
            }
            .into());
        }
    }

    // Use the first ridge vertex to get a small candidate set (local star walk when possible).
    let candidates: CellKeySet = tds.find_cells_containing_vertex_by_key(ridge_vertices[0]);

    let mut star_cells: SmallBuffer<CellKey, 8> = SmallBuffer::with_capacity(candidates.len());

    for cell_key in candidates {
        let cell_vertices = tds.get_cell_vertices(cell_key)?;
        if ridge_vertices.iter().all(|&rv| cell_vertices.contains(&rv)) {
            star_cells.push(cell_key);
        }
    }

    Ok(star_cells)
}

pub(crate) fn ridge_link_edges_from_star<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    ridge_vertices: &[VertexKey],
    star_cells: &[CellKey],
) -> Result<SmallBuffer<(VertexKey, VertexKey), 8>, ManifoldError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Ridge links are only meaningful for D>=2.
    if D < 2 {
        return Ok(SmallBuffer::new());
    }

    let expected_ridge_vertices = D.saturating_sub(1);
    if ridge_vertices.len() != expected_ridge_vertices {
        return Err(TdsValidationError::InconsistentDataStructure {
            message: format!(
                "Ridge expected {expected_ridge_vertices} vertices for {D}D, got {}",
                ridge_vertices.len(),
            ),
        }
        .into());
    }

    let mut link_edges: SmallBuffer<(VertexKey, VertexKey), 8> =
        SmallBuffer::with_capacity(star_cells.len());

    let mut link_vertices: VertexKeyBuffer = VertexKeyBuffer::with_capacity(2);

    for &cell_key in star_cells {
        let cell_vertices = tds.get_cell_vertices(cell_key)?;

        link_vertices.clear();
        for &vk in &cell_vertices {
            if !ridge_vertices.contains(&vk) {
                link_vertices.push(vk);
            }
        }

        if link_vertices.len() != 2 {
            return Err(TdsValidationError::InconsistentDataStructure {
                message: format!(
                    "Ridge link expected 2 link vertices for {D}D, got {} (cell_key={cell_key:?})",
                    link_vertices.len(),
                ),
            }
            .into());
        }

        if link_vertices[0] == link_vertices[1] {
            return Err(TdsValidationError::InconsistentDataStructure {
                message: format!(
                    "Ridge link edge is a self-loop: link vertex {vk:?} repeated (cell_key={cell_key:?})",
                    vk = link_vertices[0],
                ),
            }
            .into());
        }

        link_edges.push((link_vertices[0], link_vertices[1]));
    }

    Ok(link_edges)
}

#[derive(Clone, Debug)]
struct RidgeStar {
    ridge_vertices: VertexKeyBuffer,
    star_cells: SmallBuffer<CellKey, 8>,
}

// Performance: This builds a ridge → star incidence map by visiting every cell and
// enumerating its ridges.
//
// In terms of D, each cell contributes C(D+1, 2) = O(D²) ridges, each with O(D) vertices.
// Therefore this pass is O(#cells × C(D+1,2) × D) time (i.e., O(#cells × D³) in D) and
// O(#cells × C(D+1,2)) additional memory for the incidence map.
//
// This is appropriate for Level 3 topology validation / debugging, but it can be expensive
// for extremely large triangulations (e.g., millions of cells) or higher-dimensional complexes.
fn build_ridge_star_map<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<FastHashMap<u64, RidgeStar>, ManifoldError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let cell_count = tds.number_of_cells();
    if cell_count == 0 {
        return Ok(FastHashMap::default());
    }

    // Each D-simplex has C(D+1, 2) ridges (omit two vertices).
    let ridges_per_cell = (D + 1).saturating_mul(D) / 2;

    // A crude-but-safe estimate: in a manifold, ridges are typically incident to ~2 cells, so the
    // number of unique ridges is often around half the total ridge incidences.
    let estimated_unique_ridges = cell_count
        .saturating_mul(ridges_per_cell)
        .saturating_div(2)
        .max(1);

    // Map ridge key -> ridge star (incident cells).
    let mut ridge_to_star: FastHashMap<u64, RidgeStar> =
        fast_hash_map_with_capacity(estimated_unique_ridges);

    let mut ridge_vertices: VertexKeyBuffer = VertexKeyBuffer::with_capacity(D.saturating_sub(1));

    for (cell_key, _cell) in tds.cells() {
        let cell_vertices = tds.get_cell_vertices(cell_key)?;

        if cell_vertices.len() != D + 1 {
            return Err(TdsValidationError::InconsistentDataStructure {
                message: format!(
                    "Cell {cell_key:?} expected {} vertices for {D}D, got {}",
                    D + 1,
                    cell_vertices.len(),
                ),
            }
            .into());
        }

        // Enumerate ridges in this cell by omitting two vertices.
        for omit_a in 0..cell_vertices.len() {
            for omit_b in (omit_a + 1)..cell_vertices.len() {
                ridge_vertices.clear();
                for (i, &vk) in cell_vertices.iter().enumerate() {
                    if i == omit_a || i == omit_b {
                        continue;
                    }
                    ridge_vertices.push(vk);
                }

                if ridge_vertices.len() != D.saturating_sub(1) {
                    return Err(TdsValidationError::InconsistentDataStructure {
                        message: format!(
                            "Ridge expected {} vertices for {D}D, got {} (cell_key={cell_key:?}, omit_a={omit_a}, omit_b={omit_b})",
                            D.saturating_sub(1),
                            ridge_vertices.len(),
                        ),
                    }
                    .into());
                }

                let ridge_key = facet_key_from_vertices(&ridge_vertices);
                let star = ridge_to_star.entry(ridge_key).or_insert_with(|| RidgeStar {
                    ridge_vertices: ridge_vertices.clone(),
                    star_cells: SmallBuffer::new(),
                });
                star.star_cells.push(cell_key);
            }
        }
    }

    Ok(ridge_to_star)
}

/// Validates the ridge-link condition for a PL-manifold (with boundary).
///
/// For a D-dimensional simplicial complex, the link of any (D-2)-simplex is a
/// 1-dimensional simplicial complex. In a PL-manifold (with boundary), this link must
/// be a 1-manifold:
/// - a **cycle** for interior ridges, or
/// - a **path** for boundary ridges.
///
/// This check rules out wedge and branching singularities that are not detected by
/// facet-degree or boundary-closure checks alone.
///
/// It is a strict refinement of codimension-1 manifoldness, and detects common
/// singularities like “two surface components glued at a single vertex” in 2D.
///
/// # Performance
///
/// This is intentionally more expensive than basic codimension-1 manifold validation
/// (e.g., [`validate_facet_degree`]) because it must inspect ridge stars/links across the
/// entire complex.
///
/// Roughly speaking, this requires a full pass over all cells to build a ridge → star
/// incidence map, which is O(#cells × C(D+1,2) × D) time (linear in #cells for fixed small D)
/// and uses additional memory proportional to total ridge incidences.
///
/// Prefer reserving this check for debug/test builds or on-demand validation in production,
/// especially for very large triangulations or higher-dimensional complexes.
///
/// # Errors
///
/// Returns:
/// - [`ManifoldError::Tds`] if the underlying triangulation data structure is internally inconsistent.
/// - [`ManifoldError::RidgeLinkNotManifold`] if any ridge has a link graph that is not a connected
///   cycle or path.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::topology::manifold::validate_ridge_links;
///
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let tds = Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
///
/// validate_ridge_links(&tds).unwrap();
/// ```
pub fn validate_ridge_links<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
) -> Result<(), ManifoldError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    // Ridge links are only meaningful for D>=2.
    if D < 2 {
        return Ok(());
    }

    if tds.number_of_cells() == 0 {
        return Ok(());
    }

    // Algorithm: ridge -> star (incident cells) -> link (edges) -> validate link graph.
    let ridge_to_star = build_ridge_star_map(tds)?;
    for (ridge_key, star) in ridge_to_star {
        let link_edges = ridge_link_edges_from_star(tds, &star.ridge_vertices, &star.star_cells)?;
        validate_ridge_link_graph(ridge_key, &link_edges)?;
    }

    Ok(())
}

fn validate_ridge_link_graph(
    ridge_key: u64,
    link_edges: &[(VertexKey, VertexKey)],
) -> Result<(), ManifoldError> {
    // De-duplicate parallel edges defensively: if the underlying TDS contains duplicate
    // cells/edges, the ridge link can contain repeated edges which would otherwise inflate
    // degrees and edge counts.
    let mut unique_edges: FastHashSet<EdgeKey> =
        fast_hash_set_with_capacity(link_edges.len().max(1));

    // Build adjacency lists for the (simple) link graph.
    let estimated_link_vertices = link_edges.len().saturating_mul(2).max(1);
    let mut adjacency: FastHashMap<VertexKey, SmallBuffer<VertexKey, 2>> =
        fast_hash_map_with_capacity(estimated_link_vertices);

    let mut max_degree = 0usize;
    let mut link_edge_count = 0usize;

    for &(a, b) in link_edges {
        let edge = EdgeKey::new(a, b);
        if !unique_edges.insert(edge) {
            continue;
        }
        link_edge_count += 1;

        let (a, b) = edge.endpoints();

        let a_neighbors = adjacency.entry(a).or_default();
        a_neighbors.push(b);
        max_degree = max_degree.max(a_neighbors.len());

        let b_neighbors = adjacency.entry(b).or_default();
        b_neighbors.push(a);
        max_degree = max_degree.max(b_neighbors.len());
    }

    let link_vertex_count = adjacency.len();

    let degree_one_vertices = adjacency.values().filter(|n| n.len() == 1).count();

    // Connectivity check: traverse the link graph.
    let connected = match adjacency.iter().next() {
        None => true,
        Some((&start, _)) => {
            let mut visited: FastHashSet<VertexKey> =
                fast_hash_set_with_capacity(link_vertex_count);
            let mut stack: VertexKeyBuffer = VertexKeyBuffer::with_capacity(link_vertex_count);
            stack.push(start);

            while let Some(v) = stack.pop() {
                if !visited.insert(v) {
                    continue;
                }

                let Some(neighbors) = adjacency.get(&v) else {
                    continue;
                };

                for &n in neighbors {
                    if !visited.contains(&n) {
                        stack.push(n);
                    }
                }
            }

            visited.len() == link_vertex_count
        }
    };

    // A 1-manifold graph is a connected union of cycles and paths.
    // For a connected graph with max degree <=2, that reduces to:
    // - degree_one_vertices == 0  => cycle
    // - degree_one_vertices == 2  => path
    let ok = connected && max_degree <= 2 && (degree_one_vertices == 0 || degree_one_vertices == 2);

    if ok {
        Ok(())
    } else {
        Err(ManifoldError::RidgeLinkNotManifold {
            ridge_key,
            link_vertex_count,
            link_edge_count,
            max_degree,
            degree_one_vertices,
            connected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::cell::Cell;
    use crate::core::triangulation::Triangulation;
    use crate::geometry::kernel::FastKernel;
    use crate::vertex;

    use slotmap::KeyData;

    #[test]
    fn test_validate_facet_degree_ok_for_single_tetrahedron() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let facet_to_cells = tds.build_facet_to_cells_map().unwrap();

        assert!(validate_facet_degree(&facet_to_cells).is_ok());
    }

    #[test]
    fn test_validate_facet_degree_ok_for_two_tetrahedra_sharing_facet() {
        // Two tetrahedra share a facet => that facet has degree 2, all others degree 1.
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        // Shared triangle.
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();

        // Apex points on opposite sides.
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, -1.0]))
            .unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2, v4], None).unwrap())
            .unwrap();

        let facet_to_cells = tds.build_facet_to_cells_map().unwrap();
        assert!(validate_facet_degree(&facet_to_cells).is_ok());
    }

    #[test]
    fn test_validate_facet_degree_errors_on_non_manifold_facet_multiplicity() {
        // Three tetrahedra share a single facet -> not a manifold-with-boundary.
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();

        let v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 2.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 3.0]))
            .unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2, v4], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2, v5], None).unwrap())
            .unwrap();

        let facet_to_cells = tds.build_facet_to_cells_map().unwrap();

        let expected_facet_key = facet_key_from_vertices(&[v0, v1, v2]);
        match validate_facet_degree(&facet_to_cells) {
            Err(ManifoldError::ManifoldFacetMultiplicity {
                facet_key,
                cell_count,
            }) => {
                assert_eq!(facet_key, expected_facet_key);
                assert_eq!(cell_count, 3);
            }
            other => panic!("Expected ManifoldFacetMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_closed_boundary_ok_for_single_tetrahedron() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let facet_to_cells = tds.build_facet_to_cells_map().unwrap();

        assert!(validate_closed_boundary(&tds, &facet_to_cells).is_ok());
    }

    #[test]
    fn test_validate_closed_boundary_errors_on_out_of_bounds_facet_index() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();
        let cell_key = tds.cells().next().unwrap().0;

        // Synthesize an invalid boundary facet handle: facet indices must be < D+1.
        let mut facet_to_cells: FacetToCellsMap = FacetToCellsMap::default();
        let mut handles: SmallBuffer<crate::core::facet::FacetHandle, 2> = SmallBuffer::new();
        handles.push(crate::core::facet::FacetHandle::new(cell_key, u8::MAX));
        facet_to_cells.insert(0_u64, handles);

        match validate_closed_boundary(&tds, &facet_to_cells) {
            Err(ManifoldError::Tds(TdsValidationError::InconsistentDataStructure { message })) => {
                assert!(
                    message.contains("out of bounds"),
                    "Unexpected message: {message}"
                );
            }
            other => panic!("Expected out-of-bounds facet index error, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_closed_boundary_noop_for_d_lt_2() {
        let vertices = vec![vertex!([0.0]), vertex!([1.0])];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 1>::build_initial_simplex(&vertices).unwrap();
        let facet_to_cells = tds.build_facet_to_cells_map().unwrap();

        // Codimension-2 boundary manifoldness is only meaningful for D>=2.
        assert!(validate_closed_boundary(&tds, &facet_to_cells).is_ok());
    }

    #[test]
    fn test_validate_closed_boundary_noop_for_closed_2d_surface() {
        // Build a closed 2D simplicial complex (topologically S²): 4 triangles on 4 vertices
        // where every edge is shared by exactly 2 triangles (manifold without boundary).
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v3 = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v2, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v1, v2, v3], None).unwrap())
            .unwrap();

        let facet_to_cells = tds.build_facet_to_cells_map().unwrap();

        // Sanity: no boundary facets (every edge has exactly 2 incident triangles).
        assert!(facet_to_cells.values().all(|handles| handles.len() == 2));

        assert!(validate_closed_boundary(&tds, &facet_to_cells).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_ok_for_closed_2d_surface() {
        // Same closed 2D surface as above, but exercises the ridge-link (PL) condition.
        //
        // In 2D, ridges are vertices and their links must be cycles for a closed surface.
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v3 = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v2, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v1, v2, v3], None).unwrap())
            .unwrap();

        // Sanity: pseudomanifold checks pass.
        let facet_to_cells = tds.build_facet_to_cells_map().unwrap();
        validate_facet_degree(&facet_to_cells).unwrap();
        validate_closed_boundary(&tds, &facet_to_cells).unwrap();

        assert!(validate_ridge_links(&tds).is_ok());
    }

    #[test]
    fn test_ridge_star_cells_returns_incident_cells_for_vertex_ridge_in_2d() {
        // In 2D, a ridge is a vertex and its star is the set of incident triangles.
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v3 = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let c012 = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let c013 = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v3], None).unwrap())
            .unwrap();
        let c023 = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v2, v3], None).unwrap())
            .unwrap();
        let _c123 = tds
            .insert_cell_with_mapping(Cell::new(vec![v1, v2, v3], None).unwrap())
            .unwrap();

        let star = ridge_star_cells(&tds, &[v0]).unwrap();
        let star_set: CellKeySet = star.iter().copied().collect();

        let expected: CellKeySet = [c012, c013, c023].into_iter().collect();
        assert_eq!(star_set, expected);
    }

    #[test]
    fn test_ridge_star_cells_errors_on_missing_vertex_key() {
        let tds: Tds<f64, (), (), 2> = Tds::empty();
        let missing = VertexKey::from(KeyData::from_ffi(u64::MAX));

        match ridge_star_cells(&tds, &[missing]) {
            Err(ManifoldError::Tds(TdsValidationError::InconsistentDataStructure { message })) => {
                assert!(message.contains("not found"));
            }
            other => panic!("Expected missing-vertex error, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_link_edges_from_star_rejects_self_loop_edge() {
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();

        // Corrupt the cell in-place: keep length == D+1 but introduce a duplicate link vertex.
        {
            let cell = tds
                .get_cell_by_key_mut(cell_key)
                .expect("cell key should be valid in test");
            cell.clear_vertex_keys();
            cell.push_vertex_key(v0);
            cell.push_vertex_key(v1);
            cell.push_vertex_key(v1);
        }

        // For ridge (vertex) v0, the link edge becomes (v1, v1), which is not a simplicial edge.
        match ridge_link_edges_from_star(&tds, &[v0], &[cell_key]) {
            Err(ManifoldError::Tds(TdsValidationError::InconsistentDataStructure { message })) => {
                assert!(
                    message.contains("self-loop"),
                    "Unexpected message: {message}"
                );
            }
            other => panic!("Expected self-loop edge error, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_ridge_link_graph_deduplicates_parallel_edges() {
        // Triangle cycle a-b-c-a, but with a duplicated edge.
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));

        let edges = vec![(a, b), (b, c), (c, a), (a, b)];
        assert!(validate_ridge_link_graph(0_u64, &edges).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_errors_on_corrupted_cell_with_duplicate_vertices() {
        // This is a defensive robustness test: a corrupted cell with duplicate vertices can
        // produce a malformed ridge link (wrong number of link vertices).
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();

        let cell_key = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();

        // Corrupt the cell in-place: keep length == D+1 but introduce a duplicate vertex.
        {
            let cell = tds
                .get_cell_by_key_mut(cell_key)
                .expect("cell key should be valid in test");
            cell.clear_vertex_keys();
            cell.push_vertex_key(v0);
            cell.push_vertex_key(v0);
            cell.push_vertex_key(v0);
        }

        match validate_ridge_links(&tds) {
            Err(ManifoldError::Tds(TdsValidationError::InconsistentDataStructure { message })) => {
                assert!(message.contains("Ridge link expected 2 link vertices"));
            }
            other => panic!("Expected ridge-link structural error, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_closed_boundary_errors_on_non_manifold_boundary_ridge() {
        // Two tetrahedra that share an edge but not a facet create a non-manifold boundary:
        // the shared edge is incident to 4 boundary triangles.
        let mut tds: Tds<f64, (), (), 3> = Tds::empty();

        // Shared edge
        let shared_edge_v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 0.0]))
            .unwrap();
        let shared_edge_v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0, 0.0]))
            .unwrap();

        // First tetrahedron
        let tet1_v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 1.0, 0.0]))
            .unwrap();
        let tet1_v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, 1.0]))
            .unwrap();

        // Second tetrahedron
        let tet2_v2 = tds
            .insert_vertex_with_mapping(vertex!([0.0, -1.0, 0.0]))
            .unwrap();
        let tet2_v3 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0, -1.0]))
            .unwrap();

        let _ = tds
            .insert_cell_with_mapping(
                Cell::new(vec![shared_edge_v0, shared_edge_v1, tet1_v2, tet1_v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(
                Cell::new(vec![shared_edge_v0, shared_edge_v1, tet2_v2, tet2_v3], None).unwrap(),
            )
            .unwrap();

        let facet_to_cells = tds.build_facet_to_cells_map().unwrap();

        // The shared edge should appear in 4 boundary facets.
        let expected_ridge_key = facet_key_from_vertices(&[shared_edge_v0, shared_edge_v1]);

        match validate_closed_boundary(&tds, &facet_to_cells) {
            Err(ManifoldError::BoundaryRidgeMultiplicity {
                ridge_key,
                boundary_facet_count,
            }) => {
                assert_eq!(ridge_key, expected_ridge_key);
                assert_eq!(boundary_facet_count, 4);
            }
            other => panic!("Expected BoundaryRidgeMultiplicity, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_ridge_links_ok_for_single_tetrahedron() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];

        let tds =
            Triangulation::<FastKernel<f64>, (), (), 3>::build_initial_simplex(&vertices).unwrap();

        assert!(validate_ridge_links(&tds).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_noop_for_empty_tds() {
        let tds: Tds<f64, (), (), 2> = Tds::empty();
        assert!(validate_ridge_links(&tds).is_ok());
    }

    #[test]
    fn test_validate_ridge_links_rejects_wedge_at_vertex_in_2d() {
        // Build two closed 2D spheres (boundaries of tetrahedra) that share a single vertex.
        // This is a pseudomanifold (every edge has degree 2), but it is not a PL 2-manifold:
        // the shared vertex has a disconnected link (two disjoint cycles).
        let mut tds: Tds<f64, (), (), 2> = Tds::empty();

        // Shared vertex.
        let v0 = tds.insert_vertex_with_mapping(vertex!([0.0, 0.0])).unwrap();

        // First tetrahedron boundary (4 triangles on 4 vertices).
        let v1 = tds.insert_vertex_with_mapping(vertex!([1.0, 0.0])).unwrap();
        let v2 = tds.insert_vertex_with_mapping(vertex!([0.0, 1.0])).unwrap();
        let v3 = tds.insert_vertex_with_mapping(vertex!([1.0, 1.0])).unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v2], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v1, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v2, v3], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v1, v2, v3], None).unwrap())
            .unwrap();

        // Second tetrahedron boundary (shares only v0).
        let v4 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 10.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(vertex!([11.0, 10.0]))
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(vertex!([10.0, 11.0]))
            .unwrap();

        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v4, v5], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v4, v6], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v0, v5, v6], None).unwrap())
            .unwrap();
        let _ = tds
            .insert_cell_with_mapping(Cell::new(vec![v4, v5, v6], None).unwrap())
            .unwrap();

        // Sanity: pseudomanifold-with-boundary checks pass (in fact, this complex is closed).
        let facet_to_cells = tds.build_facet_to_cells_map().unwrap();
        validate_facet_degree(&facet_to_cells).unwrap();
        validate_closed_boundary(&tds, &facet_to_cells).unwrap();

        let expected_ridge_key = facet_key_from_vertices(&[v0]);

        match validate_ridge_links(&tds) {
            Err(ManifoldError::RidgeLinkNotManifold {
                ridge_key,
                connected,
                degree_one_vertices,
                max_degree,
                ..
            }) => {
                assert_eq!(ridge_key, expected_ridge_key);
                assert!(!connected);
                assert_eq!(degree_one_vertices, 0);
                assert_eq!(max_degree, 2);
            }
            other => panic!("Expected RidgeLinkNotManifold, got {other:?}"),
        }
    }
}

//! Topology-only manifold validation utilities.
//!
//! This module contains invariants that depend only on the combinatorial structure
//! of the simplicial complex stored in a [`Tds`](crate::core::triangulation_data_structure::Tds).
//! These checks:
//! - are independent of geometry and Delaunay predicates,
//! - are not TDS structural invariants (Level 2), and
//! - are intended to back Level 3 (manifold topology) validation.

use thiserror::Error;

use crate::core::{
    collections::{FacetToCellsMap, FastHashMap, VertexKeyBuffer, fast_hash_map_with_capacity},
    facet::facet_key_from_vertices,
    traits::DataType,
    triangulation_data_structure::{Tds, TdsValidationError},
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
        "Non-manifold facet: facet {facet_key} belongs to {cell_count} cells (expected 1 or 2)"
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
}

/// Validates that each (D-1)-facet has degree 1 (boundary) or 2 (interior).
///
/// This is the codimension-1 pseudomanifold / manifold-with-boundary condition.
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::cell::Cell;
    use crate::core::triangulation::Triangulation;
    use crate::geometry::kernel::FastKernel;
    use crate::vertex;

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
    fn test_validate_closed_boundary_noop_for_closed_2d_surface() {
        // Build the boundary of a tetrahedron as a 2D simplicial complex (a closed S^2):
        // 4 triangles on 4 vertices, with every edge shared by exactly 2 triangles.
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
}

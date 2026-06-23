//! Euler characteristic computation for triangulated spaces.
//!
//! This module implements dimensional-generic Euler characteristic calculation
//! using the formula: χ = Σ(-1)^k · `f_k` where `f_k` is the number of `k`-simplices.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::*;
//! use delaunay::prelude::topology::validation::euler;
//!
//! # #[derive(Debug, thiserror::Error)]
//! # enum ExampleError {
//! #     #[error(transparent)]
//! #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
//! #     #[error(transparent)]
//! #     Topology(#[from] delaunay::topology::TopologyError),
//! #     #[error(transparent)]
//! #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
//! # }
//! # fn main() -> Result<(), ExampleError> {
//! let vertices = vec![
//!     delaunay::vertex![0.0, 0.0, 0.0]?,
//!     delaunay::vertex![1.0, 0.0, 0.0]?,
//!     delaunay::vertex![0.0, 1.0, 0.0]?,
//!     delaunay::vertex![0.0, 0.0, 1.0]?,
//! ];
//! let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! let counts = euler::count_simplices(dt.tds())?;
//! let chi = euler::euler_characteristic(&counts);
//! assert_eq!(chi, 1);  // Single tetrahedron has χ = 1
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use crate::core::{
    collections::{
        FacetToSimplicesMap, FastHashMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
        VertexKeyBuffer, fast_hash_map_with_capacity, fast_hash_set_with_capacity,
    },
    edge::EdgeKey,
    tds::{Tds, VertexKey},
};
use crate::topology::manifold::boundary_facet_keys_from_index;
use crate::topology::traits::topological_space::{GlobalTopology, TopologyError};

type LiftedCellVertex = (VertexKey, SmallBuffer<i16, MAX_PRACTICAL_DIMENSION_SIZE>);
type LiftedCellKey = SmallBuffer<LiftedCellVertex, MAX_PRACTICAL_DIMENSION_SIZE>;

/// Counts of k-simplices for all dimensions 0 ≤ k ≤ D.
///
/// Stores the f-vector (f₀, f₁, ..., `f_D`) where `f_k` is the number of
/// `k`-dimensional simplices:
/// - `f₀` = vertices (`0`-simplices)
/// - `f₁` = edges (`1`-simplices)
/// - `f₂` = triangular faces (`2`-simplices)
/// - `f₃` = tetrahedral simplices (`3`-simplices)
/// - `f_D` = `D`-dimensional simplices
///
/// In the topology literature this is commonly called the **f-vector**.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::validation::FVector;
///
/// // 2D triangle: 3 vertices, 3 edges, 1 face
/// let counts = FVector {
///     by_dim: vec![3, 3, 1],
/// };
///
/// assert_eq!(counts.count(0), 3);  // vertices
/// assert_eq!(counts.count(1), 3);  // edges
/// assert_eq!(counts.count(2), 1);  // faces
/// assert_eq!(counts.dimension(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FVector {
    /// `by_dim[k]` = `f_k` = number of `k`-simplices
    pub by_dim: Vec<usize>,
}

impl FVector {
    /// Get the number of `k`-simplices.
    ///
    /// Returns 0 if `k` is out of range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::validation::FVector;
    ///
    /// let counts = FVector {
    ///     by_dim: vec![4, 6, 4, 1],  // 3D tetrahedron
    /// };
    ///
    /// assert_eq!(counts.count(0), 4);  // 4 vertices
    /// assert_eq!(counts.count(1), 6);  // 6 edges
    /// assert_eq!(counts.count(2), 4);  // 4 faces
    /// assert_eq!(counts.count(3), 1);  // 1 simplex
    /// assert_eq!(counts.count(4), 0);  // out of range
    /// ```
    #[must_use]
    #[inline]
    pub fn count(&self, k: usize) -> usize {
        self.by_dim.get(k).copied().unwrap_or(0)
    }

    /// Get the dimension (maximum `k` where `f_k` > 0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::validation::FVector;
    ///
    /// let counts_3d = FVector {
    ///     by_dim: vec![4, 6, 4, 1],
    /// };
    /// assert_eq!(counts_3d.dimension(), 3);
    ///
    /// let counts_2d = FVector {
    ///     by_dim: vec![3, 3, 1],
    /// };
    /// assert_eq!(counts_2d.dimension(), 2);
    /// ```
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.by_dim.len().saturating_sub(1)
    }
}

/// Euler-check classification of a triangulation.
///
/// This is a coarse classification used to choose an expected Euler
/// characteristic after boundary facets have been interpreted under the
/// declared global topology. It is not a complete topological invariant:
/// different manifolds can share the same Euler characteristic.
///
/// # Variants
///
/// - `Empty`: No simplices (χ = 0)
/// - `SingleSimplex(D)`: One D-simplex (χ = 1)
/// - `Ball(D)`: D-ball with boundary (χ = 1)
/// - `ClosedSphere(D)`: Closed D-sphere without boundary (χ = 1 + (-1)^D)
/// - `Unknown`: Cannot determine topology
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::validation::TopologyClassification;
///
/// let ball = TopologyClassification::Ball(3);
/// assert_eq!(format!("{:?}", ball), "Ball(3)");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyClassification {
    /// Empty triangulation (no simplices).
    Empty,

    /// Single D-simplex (first simplex).
    ///
    /// The value represents the dimension D.
    SingleSimplex(usize),

    /// Topological D-ball (has boundary).
    ///
    /// Most finite Delaunay triangulations fall into this category.
    /// The value represents the dimension D.
    Ball(usize),

    /// Closed D-sphere (no boundary).
    ///
    /// Rare for finite triangulations; requires special construction
    /// (e.g., periodic boundary conditions).
    /// The value represents the dimension D.
    ClosedSphere(usize),

    /// Closed D-torus (no boundary).
    ///
    /// A toroidal mesh constructed with periodic boundary identification.
    /// The Euler characteristic of the D-torus T^D is always 0.
    /// The value represents the dimension D.
    ClosedToroid(usize),

    /// Cannot determine or doesn't fit known categories.
    Unknown,
}

/// Count all k-simplices in the triangulation.
///
/// Computes the complete f-vector (f₀, f₁, ..., `f_D`) where `f_k` is the
/// number of `k`-dimensional simplices.
///
/// # Algorithm
///
/// - `f₀` (vertices): Direct count from Tds - O(1)
/// - `f_D` (simplices): Direct count from Tds - O(1)
/// - `f_{D-1}` (facets): Use the internal facet-incidence map - O(N·D²)
/// - Intermediate `k`: Enumerate combinations from simplices - O(N · C(D+1, k+1))
///
/// For practical dimensions (D ≤ 5), this is efficient.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::euler;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Topology(#[from] delaunay::topology::TopologyError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.5, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let counts = euler::count_simplices(dt.tds())?;
/// assert_eq!(counts.count(0), 3);  // 3 vertices
/// assert_eq!(counts.count(1), 3);  // 3 edges
/// assert_eq!(counts.count(2), 1);  // 1 face
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`TopologyError::FacetMapBuild`] if simplex enumeration fails.
pub fn count_simplices<U, V, const D: usize>(tds: &Tds<U, V, D>) -> Result<FVector, TopologyError> {
    // Handle empty triangulation without building any facet map.
    let mut by_dim = vec![0usize; D + 1];
    by_dim[0] = tds.number_of_vertices();
    by_dim[D] = tds.number_of_simplices();
    if by_dim[D] == 0 {
        return Ok(FVector { by_dim });
    }

    // Build the facet map once, then compute counts from it.
    let facet_to_simplices = tds
        .build_facet_to_simplices_map()
        .map_err(|source| TopologyError::FacetMapBuild { source })?;

    Ok(count_simplices_with_facet_to_simplices_map(
        tds,
        &facet_to_simplices,
    ))
}

pub(crate) fn count_simplices_with_facet_to_simplices_map<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    facet_to_simplices: &FacetToSimplicesMap,
) -> FVector {
    let mut by_dim = vec![0usize; D + 1];

    // f₀: vertices (O(1))
    by_dim[0] = tds.number_of_vertices();

    // f_D: D-simplices (O(1))
    by_dim[D] = tds.number_of_simplices();

    // Handle empty triangulation
    if by_dim[D] == 0 {
        return FVector { by_dim };
    }

    // f_{D-1}: (D-1)-facets from precomputed map
    by_dim[D - 1] = facet_to_simplices.len();

    // Intermediate dimensions (1 ≤ k ≤ D-2): enumerate combinations.
    //
    // We keep a set per k and fill them in a single pass over simplices, which is faster than
    // re-iterating all simplices once per k.
    // Skip if D <= 2 (no intermediate dimensions)
    if D > 2 {
        let mut intermediate_simplex_sets: Vec<FastHashSet<LiftedCellKey>> =
            (0..(D - 2)).map(|_| FastHashSet::default()).collect();

        for (_simplex_key, simplex) in tds.simplices() {
            for simplex_dimension in 1..=D - 2 {
                let simplex_set =
                    &mut intermediate_simplex_sets[simplex_dimension.saturating_sub(1)];
                let simplex_size = simplex_dimension + 1; // k-simplex has k+1 vertices
                insert_lifted_simplices_of_size(
                    simplex.vertices(),
                    simplex.periodic_vertex_offsets(),
                    simplex_size,
                    simplex_set,
                );
            }
        }

        for simplex_dimension in 1..=D - 2 {
            by_dim[simplex_dimension] =
                intermediate_simplex_sets[simplex_dimension.saturating_sub(1)].len();
        }
    }

    FVector { by_dim }
}

fn insert_lifted_simplices_of_size<const D: usize>(
    vertex_keys: &[VertexKey],
    periodic_offsets: Option<&[[i8; D]]>,
    simplex_size: usize,
    simplex_set: &mut FastHashSet<LiftedCellKey>,
) {
    let n = vertex_keys.len();
    if n < simplex_size {
        return;
    }

    // Generate all C(n, simplex_size) combinations using the standard lexicographic algorithm.
    //
    // We maintain `indices[0..simplex_size]` as strictly increasing positions into `vertex_keys`.
    // To advance to the next combination:
    // 1) Find the rightmost index that can be incremented without running out of room (the pivot).
    // 2) Increment it.
    // 3) Reset all subsequent indices to consecutive values.
    let mut indices: SmallBuffer<usize, MAX_PRACTICAL_DIMENSION_SIZE> = (0..simplex_size).collect();

    'outer: loop {
        let mut simplex_vertices: LiftedCellKey = SmallBuffer::new();
        for &vertex_index in &indices {
            let offset: SmallBuffer<i16, MAX_PRACTICAL_DIMENSION_SIZE> = periodic_offsets
                .map_or_else(SmallBuffer::new, |offsets| {
                    offsets[vertex_index]
                        .iter()
                        .map(|&component| i16::from(component))
                        .collect()
                });
            simplex_vertices.push((vertex_keys[vertex_index], offset));
        }
        normalize_lifted_cell_key(&mut simplex_vertices);
        simplex_set.insert(simplex_vertices);

        // Generate next combination.
        // The maximum valid value at position `i` is `i + n - simplex_size`.
        //
        // Find the rightmost index that can be incremented.
        let mut pivot = simplex_size;
        loop {
            if pivot == 0 {
                break 'outer;
            }
            pivot -= 1;
            if indices[pivot] < pivot + n - simplex_size {
                break;
            }
        }

        indices[pivot] += 1;
        for position in (pivot + 1)..simplex_size {
            indices[position] = indices[position - 1] + 1;
        }
    }
}

fn normalize_lifted_cell_key(simplex_vertices: &mut LiftedCellKey) {
    simplex_vertices.sort_unstable_by(|(vertex_a, offset_a), (vertex_b, offset_b)| {
        vertex_a.cmp(vertex_b).then_with(|| offset_a.cmp(offset_b))
    });
    let anchor_offset = simplex_vertices
        .first()
        .map_or_else(SmallBuffer::new, |(_, offset)| offset.clone());
    let axes = simplex_vertices
        .iter()
        .map(|(_, offset)| offset.len())
        .max()
        .unwrap_or(0)
        .max(anchor_offset.len());
    for (_, offset) in simplex_vertices.iter_mut() {
        let mut normalized: SmallBuffer<i16, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(axes);
        for axis in 0..axes {
            let component = offset.get(axis).copied().unwrap_or(0)
                - anchor_offset.get(axis).copied().unwrap_or(0);
            normalized.push(component);
        }
        *offset = normalized;
    }
    simplex_vertices.sort_unstable_by(|(vertex_a, offset_a), (vertex_b, offset_b)| {
        vertex_a.cmp(vertex_b).then_with(|| offset_a.cmp(offset_b))
    });
}

fn insert_simplices_of_size(
    vertex_keys: &[VertexKey],
    simplex_size: usize,
    simplex_set: &mut FastHashSet<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>,
) {
    let n = vertex_keys.len();
    if n < simplex_size {
        return;
    }

    let mut indices: SmallBuffer<usize, MAX_PRACTICAL_DIMENSION_SIZE> = (0..simplex_size).collect();

    'outer: loop {
        let mut simplex_vertices: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();
        for &vertex_index in &indices {
            simplex_vertices.push(vertex_keys[vertex_index]);
        }
        simplex_set.insert(simplex_vertices);

        let mut pivot = simplex_size;
        loop {
            if pivot == 0 {
                break 'outer;
            }
            pivot -= 1;
            if indices[pivot] < pivot + n - simplex_size {
                break;
            }
        }

        indices[pivot] += 1;
        for position in (pivot + 1)..simplex_size {
            indices[position] = indices[position - 1] + 1;
        }
    }
}

/// Count simplices on the boundary (convex hull) only.
///
/// This computes simplex counts for just the boundary facets,
/// which form a (D-1)-dimensional simplicial complex.
///
/// # Algorithm
///
/// 1. Extract all boundary facets using `boundary_facets()`
/// 2. Collect unique vertices that appear on the boundary
/// 3. Count (D-1)-simplices (the boundary facets themselves)
/// 4. For intermediate dimensions k < D-1, enumerate k-simplices
///    from the boundary facets' vertex combinations
///
/// # Expected Euler Characteristics
///
/// The boundary forms a (D-1)-sphere S^(D-1):
/// - 2D triangulation: boundary is S¹ (circle) with χ = 0
/// - 3D triangulation: boundary is S² (sphere) with χ = 2
/// - 4D triangulation: boundary is S³ (3-sphere) with χ = 0
/// - 5D triangulation: boundary is S⁴ (4-sphere) with χ = 2
/// - Generally: χ(S^k) = 1 + (-1)^k
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::euler;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Topology(#[from] delaunay::topology::TopologyError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // 3D tetrahedron - boundary is S² (sphere)
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let boundary_counts = euler::count_boundary_simplices(dt.tds(), dt.global_topology())?;
/// let boundary_chi = euler::euler_characteristic(&boundary_counts);
/// assert_eq!(boundary_chi, 2);  // S² has χ = 2
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`TopologyError::BoundaryFacetEnumeration`] if the facet index cannot
/// be built, [`TopologyError::BoundaryClassification`] if the declared global
/// topology is incompatible with the observed facet incidences, or
/// [`TopologyError::BoundaryFacetSimplexAccess`] if a matching facet view cannot
/// be reconstructed from the TDS.
pub fn count_boundary_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<FVector, TopologyError> {
    // Get topology-approved boundary facets.
    let facet_index = tds
        .build_facet_to_simplices_index()
        .map_err(|source| TopologyError::BoundaryFacetEnumeration { source })?;
    let boundary_facet_keys = boundary_facet_keys_from_index(&facet_index, global_topology)
        .map_err(|source| TopologyError::BoundaryClassification {
            source: Box::new(source),
        })?;
    // Count boundary facets and unique boundary vertices in one pass. Keep the
    // facet-view construction in the pass so corrupt facet reconstruction still
    // surfaces as BoundaryFacetSimplexAccess.
    let mut num_boundary_facets = 0_usize;
    let mut boundary_vertices = FastHashSet::default();
    let mut intermediate_simplex_sets: Option<
        Vec<FastHashSet<SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>>>,
    > = (D > 2).then(|| (0..(D - 2)).map(|_| FastHashSet::default()).collect());

    for facet in tds.facets() {
        let facet = facet.map_err(|source| TopologyError::BoundaryFacetSimplexAccess { source })?;
        if !boundary_facet_keys.contains(&facet.key()) {
            continue;
        }

        num_boundary_facets += 1;
        let simplex = facet.simplex();
        let facet_index = usize::from(facet.facet_index());
        let mut facet_vertex_keys: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::new();

        for (vertex_position, &vertex_key) in simplex.vertices().iter().enumerate() {
            if vertex_position != facet_index {
                boundary_vertices.insert(vertex_key);
                facet_vertex_keys.push(vertex_key);
            }
        }

        let Some(intermediate_simplex_sets) = intermediate_simplex_sets.as_mut() else {
            continue;
        };

        // Sort once so every generated combination is already canonical,
        // avoiding per-combination sorting.
        facet_vertex_keys.sort();
        for simplex_dimension in 1..=D - 2 {
            let simplex_set = &mut intermediate_simplex_sets[simplex_dimension.saturating_sub(1)];
            let simplex_size = simplex_dimension + 1; // k-simplex has k+1 vertices
            insert_simplices_of_size(&facet_vertex_keys, simplex_size, simplex_set);
        }
    }

    if num_boundary_facets == 0 {
        // No boundary - return zero counts for (D-1)-dimensional complex
        return Ok(FVector { by_dim: vec![0; D] });
    }

    let num_boundary_vertices = boundary_vertices.len();

    // Initialize counts for (D-1)-dimensional complex
    // by_dim[0] = vertices, by_dim[1] = edges, ..., by_dim[D-1] = (D-1)-simplices
    let mut by_dim = vec![0; D];
    by_dim[0] = num_boundary_vertices;
    by_dim[D - 1] = num_boundary_facets;

    // Count intermediate k-simplices (1 ≤ k < D-1) by enumerating combinations
    // from boundary facets.
    if let Some(intermediate_simplex_sets) = intermediate_simplex_sets {
        for simplex_dimension in 1..=D - 2 {
            by_dim[simplex_dimension] =
                intermediate_simplex_sets[simplex_dimension.saturating_sub(1)].len();
        }
    }

    Ok(FVector { by_dim })
}

/// Compute Euler characteristic from simplex counts.
///
/// Uses the alternating sum formula: χ = Σ(-1)^k · `f_k`
///
/// # Formula
///
/// ```text
/// χ = f₀ - f₁ + f₂ - f₃ + ... ± f_D
///   = Σ(k=0 to D) (-1)^k · f_k
/// ```
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::validation::{FVector, euler_characteristic};
///
/// // 2D triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1
/// let counts = FVector {
///     by_dim: vec![3, 3, 1],
/// };
/// assert_eq!(euler_characteristic(&counts), 1);
///
/// // 3D tetrahedron: V=4, E=6, F=4, C=1 → χ = 4-6+4-1 = 1
/// let counts_3d = FVector {
///     by_dim: vec![4, 6, 4, 1],
/// };
/// assert_eq!(euler_characteristic(&counts_3d), 1);
/// ```
#[must_use]
#[expect(
    clippy::cast_possible_wrap,
    reason = "Simplex counts won't exceed isize::MAX in practice"
)]
pub fn euler_characteristic(counts: &FVector) -> isize {
    counts
        .by_dim
        .iter()
        .enumerate()
        .map(|(k, &f_k)| {
            let sign = if k % 2 == 0 { 1 } else { -1 };
            sign * (f_k as isize)
        })
        .sum()
}

pub(crate) fn triangulated_surface_euler_characteristic(
    triangles: &SmallBuffer<VertexKeyBuffer, 8>,
) -> isize {
    // For a triangulated 2D complex: χ = V - E + F.
    let mut vertices: FastHashSet<VertexKey> =
        fast_hash_set_with_capacity(triangles.len().saturating_mul(3).max(1));
    let mut edges: FastHashSet<EdgeKey> =
        fast_hash_set_with_capacity(triangles.len().saturating_mul(3).max(1));

    let mut face_count = 0usize;

    for tri in triangles {
        // Expect triangles.
        if tri.len() != 3 {
            continue;
        }
        face_count += 1;

        let a = tri[0];
        let b = tri[1];
        let c = tri[2];

        vertices.insert(a);
        vertices.insert(b);
        vertices.insert(c);

        edges.insert(EdgeKey::from_validated_endpoints(a, b));
        edges.insert(EdgeKey::from_validated_endpoints(b, c));
        edges.insert(EdgeKey::from_validated_endpoints(c, a));
    }

    let counts = FVector {
        by_dim: vec![vertices.len(), edges.len(), face_count],
    };

    euler_characteristic(&counts)
}

pub(crate) fn triangulated_surface_boundary_component_count(
    triangles: &SmallBuffer<VertexKeyBuffer, 8>,
) -> usize {
    // Count boundary edges (edges incident to exactly 1 triangle), then count connected components
    // in the boundary 1-skeleton.
    let mut edge_counts: FastHashMap<EdgeKey, usize> =
        fast_hash_map_with_capacity(triangles.len().saturating_mul(3).max(1));

    for tri in triangles {
        if tri.len() != 3 {
            continue;
        }

        let a = tri[0];
        let b = tri[1];
        let c = tri[2];

        for e in [
            EdgeKey::from_validated_endpoints(a, b),
            EdgeKey::from_validated_endpoints(b, c),
            EdgeKey::from_validated_endpoints(c, a),
        ] {
            *edge_counts.entry(e).or_insert(0) += 1;
        }
    }

    let mut boundary_adjacency: FastHashMap<VertexKey, SmallBuffer<VertexKey, 2>> =
        FastHashMap::default();

    for (e, count) in edge_counts {
        if count != 1 {
            continue;
        }

        let (u, v) = e.endpoints();
        boundary_adjacency.entry(u).or_default().push(v);
        boundary_adjacency.entry(v).or_default().push(u);
    }

    if boundary_adjacency.is_empty() {
        return 0;
    }

    let mut visited: FastHashSet<VertexKey> =
        fast_hash_set_with_capacity(boundary_adjacency.len().max(1));
    let mut components = 0usize;

    for &start in boundary_adjacency.keys() {
        if visited.contains(&start) {
            continue;
        }

        components += 1;

        let mut stack: VertexKeyBuffer = VertexKeyBuffer::with_capacity(16);
        stack.push(start);

        while let Some(v) = stack.pop() {
            if !visited.insert(v) {
                continue;
            }

            let Some(neigh) = boundary_adjacency.get(&v) else {
                continue;
            };

            for &n in neigh {
                if !visited.contains(&n) {
                    stack.push(n);
                }
            }
        }
    }

    components
}

/// Classifies a triangulation for Euler-characteristic compatibility checks.
///
/// This is a coarse classification, not a complete topology detector. Boundary
/// structure is interpreted through the supplied [`GlobalTopology`] so raw
/// one-sided incidence in a periodic quotient is not mistaken for boundary.
///
/// # Classification Logic
///
/// - No simplices → `Empty`
/// - One simplex with true boundary → `SingleSimplex(D)`
/// - True boundary → `Ball(D)`
/// - No boundary with toroidal metadata → `ClosedToroid(D)`
/// - No boundary otherwise → `ClosedSphere(D)` (rare)
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::{classify_triangulation, TopologyClassification};
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Topology(#[from] delaunay::topology::TopologyError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let classification = classify_triangulation(dt.tds(), dt.global_topology())?;
/// assert_eq!(classification, TopologyClassification::SingleSimplex(3));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`TopologyError::BoundaryFacetCount`] if the facet index cannot be
/// built, or [`TopologyError::BoundaryClassification`] if the declared global
/// topology is incompatible with the observed facet incidences. The latter
/// includes open one-sided facets in closed topology and periodic
/// self-identifications in non-periodic topology metadata.
pub fn classify_triangulation<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    global_topology: GlobalTopology<D>,
) -> Result<TopologyClassification, TopologyError> {
    let num_simplices = tds.number_of_simplices();

    // Empty triangulation
    if num_simplices == 0 {
        return Ok(TopologyClassification::Empty);
    }

    let facet_index = tds
        .build_facet_to_simplices_index()
        .map_err(|source| TopologyError::BoundaryFacetCount { source })?;
    let has_boundary = !boundary_facet_keys_from_index(&facet_index, global_topology)
        .map_err(|source| TopologyError::BoundaryClassification {
            source: Box::new(source),
        })?
        .is_empty();

    if num_simplices == 1 && has_boundary {
        Ok(TopologyClassification::SingleSimplex(D))
    } else if has_boundary {
        Ok(TopologyClassification::Ball(D))
    } else if global_topology.is_toroidal() {
        Ok(TopologyClassification::ClosedToroid(D))
    } else {
        // No boundary → closed manifold (assume sphere for now)
        Ok(TopologyClassification::ClosedSphere(D))
    }
}

/// Get expected χ for a topological classification.
///
/// # Expected Values
///
/// - `Empty`: χ = 0
/// - `SingleSimplex(_)`: χ = 1
/// - `Ball(_)`: χ = 1
/// - `ClosedSphere(d)`: χ = 1 + (-1)^d
/// - `Unknown`: None
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::validation::{TopologyClassification, expected_chi_for};
///
/// assert_eq!(expected_chi_for(&TopologyClassification::Empty), Some(0));
/// assert_eq!(expected_chi_for(&TopologyClassification::Ball(3)), Some(1));
/// assert_eq!(expected_chi_for(&TopologyClassification::ClosedSphere(2)), Some(2));
/// assert_eq!(expected_chi_for(&TopologyClassification::ClosedSphere(3)), Some(0));
/// assert_eq!(expected_chi_for(&TopologyClassification::Unknown), None);
/// ```
#[must_use]
pub fn expected_chi_for(classification: &TopologyClassification) -> Option<isize> {
    match classification {
        TopologyClassification::Empty => Some(0),
        TopologyClassification::SingleSimplex(_) | TopologyClassification::Ball(_) => Some(1),
        TopologyClassification::ClosedSphere(d) => {
            // χ(S^d) = 1 + (-1)^d
            Some(1 + if d % 2 == 0 { 1 } else { -1 })
        }
        TopologyClassification::ClosedToroid(_) => {
            // χ(T^d) = 0 for all d ≥ 1
            Some(0)
        }
        TopologyClassification::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::simplex::Simplex;
    use slotmap::{KeyData, SlotMap};

    #[test]
    fn test_simplex_counts() {
        let counts = FVector {
            by_dim: vec![3, 3, 1],
        };

        assert_eq!(counts.count(0), 3);
        assert_eq!(counts.count(1), 3);
        assert_eq!(counts.count(2), 1);
        assert_eq!(counts.count(3), 0); // out of range
        assert_eq!(counts.dimension(), 2);
    }

    #[test]
    fn test_insert_simplices_of_size() {
        let mut vertex_slots: SlotMap<VertexKey, ()> = SlotMap::default();
        let v0 = vertex_slots.insert(());
        let v1 = vertex_slots.insert(());
        let v2 = vertex_slots.insert(());
        let v3 = vertex_slots.insert(());

        // n < simplex_size => no combinations.
        let mut simplex_set = FastHashSet::default();
        insert_simplices_of_size(&[v0, v1], 3, &mut simplex_set);
        assert!(simplex_set.is_empty());

        // simplex_size == n => exactly one combination.
        let mut simplex_set = FastHashSet::default();
        let mut keys = vec![v2, v0, v1]; // deliberately unsorted
        keys.sort();
        insert_simplices_of_size(&keys, 3, &mut simplex_set);
        assert_eq!(simplex_set.len(), 1);

        let mut expected: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> = SmallBuffer::new();
        expected.push(v0);
        expected.push(v1);
        expected.push(v2);
        expected.sort();
        assert!(simplex_set.contains(&expected));

        // simplex_size == 1 => n singleton combinations.
        let mut simplex_set = FastHashSet::default();
        let mut keys = vec![v3, v1, v0]; // deliberately unsorted
        keys.sort();
        insert_simplices_of_size(&keys, 1, &mut simplex_set);
        assert_eq!(simplex_set.len(), keys.len());

        for &vk in &keys {
            let mut singleton: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::new();
            singleton.push(vk);
            assert!(simplex_set.contains(&singleton));
        }

        // C(3, 2) = 3 combinations.
        let mut simplex_set = FastHashSet::default();
        let mut keys = vec![v0, v2, v1]; // deliberately unsorted
        keys.sort();
        insert_simplices_of_size(&keys, 2, &mut simplex_set);
        assert_eq!(simplex_set.len(), 3);

        let expected_pairs = [(v0, v1), (v0, v2), (v1, v2)];
        for (a, b) in expected_pairs {
            let mut pair: SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE> = SmallBuffer::new();
            pair.push(a);
            pair.push(b);
            pair.sort();
            assert!(simplex_set.contains(&pair));
        }
    }

    #[test]
    fn test_euler_characteristic_2d() {
        // 2D triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1
        let counts = FVector {
            by_dim: vec![3, 3, 1],
        };
        assert_eq!(euler_characteristic(&counts), 1);
    }

    #[test]
    fn test_euler_characteristic_3d() {
        // 3D tetrahedron: V=4, E=6, F=4, C=1 → χ = 4-6+4-1 = 1
        let counts = FVector {
            by_dim: vec![4, 6, 4, 1],
        };
        assert_eq!(euler_characteristic(&counts), 1);
    }

    fn tri(a: VertexKey, b: VertexKey, c: VertexKey) -> VertexKeyBuffer {
        let mut t: VertexKeyBuffer = VertexKeyBuffer::with_capacity(3);
        t.push(a);
        t.push(b);
        t.push(c);
        t
    }

    #[test]
    fn test_triangulated_surface_euler_characteristic_single_triangle_is_one() {
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));

        let mut triangles: SmallBuffer<VertexKeyBuffer, 8> = SmallBuffer::new();
        triangles.push(tri(a, b, c));

        assert_eq!(triangulated_surface_euler_characteristic(&triangles), 1);
    }

    #[test]
    fn test_triangulated_surface_boundary_component_count_single_triangle_is_one() {
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));

        let mut triangles: SmallBuffer<VertexKeyBuffer, 8> = SmallBuffer::new();
        triangles.push(tri(a, b, c));

        assert_eq!(triangulated_surface_boundary_component_count(&triangles), 1);
    }

    #[test]
    fn test_triangulated_surface_two_triangles_sharing_edge_is_disk() {
        // Square split into two triangles (a,b,c) and (a,c,d).
        // V=4, E=5, F=2 => χ=1 (disk); boundary is one cycle.
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));
        let d = VertexKey::from(KeyData::from_ffi(4));

        let mut triangles: SmallBuffer<VertexKeyBuffer, 8> = SmallBuffer::new();
        triangles.push(tri(a, b, c));
        triangles.push(tri(a, c, d));

        assert_eq!(triangulated_surface_euler_characteristic(&triangles), 1);
        assert_eq!(triangulated_surface_boundary_component_count(&triangles), 1);
    }

    #[test]
    fn test_triangulated_surface_tetrahedron_boundary_is_sphere() {
        // Boundary of a tetrahedron: V=4, E=6, F=4 => χ=2 (S^2), no boundary.
        let a = VertexKey::from(KeyData::from_ffi(1));
        let b = VertexKey::from(KeyData::from_ffi(2));
        let c = VertexKey::from(KeyData::from_ffi(3));
        let d = VertexKey::from(KeyData::from_ffi(4));

        let mut triangles: SmallBuffer<VertexKeyBuffer, 8> = SmallBuffer::new();
        triangles.push(tri(a, b, c));
        triangles.push(tri(a, b, d));
        triangles.push(tri(a, c, d));
        triangles.push(tri(b, c, d));

        assert_eq!(triangulated_surface_euler_characteristic(&triangles), 2);
        assert_eq!(triangulated_surface_boundary_component_count(&triangles), 0);
    }

    #[test]
    fn test_triangulated_surface_two_disjoint_triangles_have_two_boundary_components() {
        // Disjoint union of two triangles:
        // χ = 1 + 1 = 2, and boundary has two connected components.
        let a0 = VertexKey::from(KeyData::from_ffi(1));
        let a1 = VertexKey::from(KeyData::from_ffi(2));
        let a2 = VertexKey::from(KeyData::from_ffi(3));
        let b0 = VertexKey::from(KeyData::from_ffi(4));
        let b1 = VertexKey::from(KeyData::from_ffi(5));
        let b2 = VertexKey::from(KeyData::from_ffi(6));

        let mut triangles: SmallBuffer<VertexKeyBuffer, 8> = SmallBuffer::new();
        triangles.push(tri(a0, a1, a2));
        triangles.push(tri(b0, b1, b2));

        assert_eq!(triangulated_surface_euler_characteristic(&triangles), 2);
        assert_eq!(triangulated_surface_boundary_component_count(&triangles), 2);
    }

    #[test]
    fn test_triangulated_surface_periodic_grid_torus_has_chi_zero_and_no_boundary() {
        // A small triangulated torus (T^2): χ(T^2) = 0 and it has no boundary.
        const N: usize = 3;
        const M: usize = 3;
        const BASE: u64 = 1_000;

        let v = |i: usize, j: usize| -> VertexKey {
            let idx = i * M + j;
            let idx_u64 = u64::try_from(idx).unwrap();
            VertexKey::from(KeyData::from_ffi(BASE + idx_u64))
        };

        let mut triangles: SmallBuffer<VertexKeyBuffer, 8> = SmallBuffer::new();
        for i in 0..N {
            for j in 0..M {
                let i1 = (i + 1) % N;
                let j1 = (j + 1) % M;

                let v00 = v(i, j);
                let v10 = v(i1, j);
                let v01 = v(i, j1);
                let v11 = v(i1, j1);

                triangles.push(tri(v00, v10, v01));
                triangles.push(tri(v10, v11, v01));
            }
        }

        assert_eq!(triangulated_surface_euler_characteristic(&triangles), 0);
        assert_eq!(triangulated_surface_boundary_component_count(&triangles), 0);
    }

    #[test]
    fn test_expected_chi_for() {
        assert_eq!(expected_chi_for(&TopologyClassification::Empty), Some(0));
        assert_eq!(
            expected_chi_for(&TopologyClassification::SingleSimplex(3)),
            Some(1)
        );
        assert_eq!(expected_chi_for(&TopologyClassification::Ball(3)), Some(1));
        assert_eq!(
            expected_chi_for(&TopologyClassification::ClosedSphere(2)),
            Some(2)
        );
        assert_eq!(
            expected_chi_for(&TopologyClassification::ClosedSphere(3)),
            Some(0)
        );
        assert_eq!(expected_chi_for(&TopologyClassification::Unknown), None);
        assert_eq!(
            expected_chi_for(&TopologyClassification::ClosedToroid(2)),
            Some(0)
        );
        assert_eq!(
            expected_chi_for(&TopologyClassification::ClosedToroid(3)),
            Some(0)
        );
    }

    fn build_closed_2d_surface_tds() -> Tds<(), (), 2> {
        // Build the boundary of a tetrahedron as a 2D simplicial complex (a closed S^2):
        // 4 triangles on 4 vertices, with every edge shared by exactly 2 triangles.
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            )
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(
                crate::core::vertex::Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
            )
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
            )
            .unwrap();
        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        tds
    }

    #[test]
    fn test_classify_triangulation_closed_sphere_2d_surface() {
        let tds = build_closed_2d_surface_tds();

        let classification = classify_triangulation(&tds, GlobalTopology::Euclidean).unwrap();
        assert_eq!(classification, TopologyClassification::ClosedSphere(2));

        let counts = count_simplices(&tds).unwrap();
        assert_eq!(counts.by_dim, vec![4, 6, 4]);
        assert_eq!(euler_characteristic(&counts), 2);
    }

    #[test]
    fn test_count_boundary_simplices_no_boundary_is_zero() {
        let tds = build_closed_2d_surface_tds();

        let boundary_counts = count_boundary_simplices(&tds, GlobalTopology::Euclidean).unwrap();
        assert_eq!(boundary_counts.by_dim, vec![0, 0]);
        assert_eq!(euler_characteristic(&boundary_counts), 0);
    }
}

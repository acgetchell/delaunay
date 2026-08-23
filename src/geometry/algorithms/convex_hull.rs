//! Owned convex-hull snapshots for d-dimensional triangulations.
//!
//! [`ConvexHull::try_from_triangulation`] is a parsing boundary: it copies the
//! boundary geometry and vertex payloads out of the source triangulation,
//! verifies that every stored facet is nondegenerate and supports every source
//! vertex, and publishes a [`ConvexHull`] only after those checks succeed. A
//! published hull is therefore self-contained; it neither stores runtime-local
//! TDS handles nor needs the source triangulation for later queries.
//!
//! # Example
//!
//! ```rust
//! use delaunay::prelude::*;
//! use delaunay::prelude::query::ConvexHull;
//!
//! # #[derive(Debug, thiserror::Error)]
//! # enum ExampleError {
//! #     #[error(transparent)]
//! #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
//! #     #[error(transparent)]
//! #     HullConstruction(#[from] delaunay::prelude::query::ConvexHullConstructionError),
//! #     #[error(transparent)]
//! #     HullQuery(#[from] delaunay::prelude::query::ConvexHullQueryError),
//! #     #[error(transparent)]
//! #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
//! # }
//! # fn main() -> Result<(), ExampleError> {
//! let vertices = [
//!     delaunay::vertex![0.0, 0.0, 0.0]?,
//!     delaunay::vertex![1.0, 0.0, 0.0]?,
//!     delaunay::vertex![0.0, 1.0, 0.0]?,
//!     delaunay::vertex![0.0, 0.0, 1.0]?,
//! ];
//! let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
//! let hull = ConvexHull::try_from_triangulation(dt.as_triangulation())?;
//! let outside = Point::try_from([2.0, 2.0, 2.0])?;
//!
//! assert!(hull.is_point_outside(&outside)?);
//! assert_eq!(hull.facets().count(), 4);
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use crate::core::collections::{
    FastHashMap, FastHashSet, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
};
use crate::core::facet::FacetError;
use crate::core::traits::data_type::DataType;
use crate::core::util::stable_facet_identifier_from_vertex_uuids;
use crate::core::vertex::Vertex;
use crate::geometry::point::Point;
use crate::geometry::predicates::{Orientation, simplex_orientation};
use crate::geometry::traits::coordinate::CoordinateConversionError;
use crate::geometry::util::{safe_usize_to_scalar, squared_norm};
use crate::triangulation::Triangulation;
use crate::triangulation::query::QueryError;
use thiserror::Error;
use uuid::Uuid;

/// Reasons that a triangulation cannot define a convex hull.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum ConvexHullInsufficientDataReason {
    /// The triangulation contains no vertices.
    #[error("triangulation contains no vertices")]
    NoVertices,
    /// The triangulation contains vertices but no simplices.
    #[error("triangulation contains no simplices")]
    NoSimplices,
    /// The triangulation has no boundary facets, as for a closed topology.
    #[error("triangulation contains no boundary facets")]
    NoBoundaryFacets,
}

/// Errors returned while parsing a triangulation boundary into a convex hull.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ConvexHullConstructionError {
    /// The topology-aware boundary query failed.
    #[error("failed to extract boundary facets from triangulation: {source}")]
    BoundaryFacetExtractionFailed {
        /// Underlying boundary-query error.
        #[source]
        source: Box<QueryError>,
    },
    /// The source does not contain enough structure to define a hull.
    #[error("insufficient data for convex hull construction: {reason}")]
    InsufficientData {
        /// Typed reason that construction cannot proceed.
        #[source]
        reason: ConvexHullInsufficientDataReason,
    },
    /// A boundary facet did not parse as a D-vertex facet.
    #[error("boundary facet {facet_index} is invalid: {source}")]
    InvalidFacet {
        /// Zero-based boundary-facet index.
        facet_index: usize,
        /// Underlying facet error.
        #[source]
        source: FacetError,
    },
    /// A boundary facet repeats one vertex identity.
    #[error("boundary facet {facet_index} repeats vertex {vertex_uuid}")]
    DuplicateFacetVertex {
        /// Zero-based boundary-facet index.
        facet_index: usize,
        /// Repeated stable vertex identity.
        vertex_uuid: Uuid,
    },
    /// Two boundary records describe the same canonical facet.
    #[error("boundary contains duplicate facet key {facet_key:016x}")]
    DuplicateFacet {
        /// Canonical facet identifier.
        facet_key: u64,
    },
    /// One stable vertex identity resolved to conflicting copied values.
    #[error("vertex {vertex_uuid} resolved to conflicting point or payload values")]
    ConflictingVertexIdentity {
        /// Conflicting stable vertex identity.
        vertex_uuid: Uuid,
    },
    /// Predicate setup failed while certifying a boundary facet.
    #[error("orientation failed while certifying boundary facet {facet_index}: {source}")]
    FacetOrientation {
        /// Zero-based boundary-facet index.
        facet_index: usize,
        /// Underlying coordinate conversion error.
        #[source]
        source: CoordinateConversionError,
    },
    /// A facet and its containing simplex have degenerate orientation.
    #[error("boundary facet {facet_index} has a degenerate inside witness")]
    DegenerateFacet {
        /// Zero-based boundary-facet index.
        facet_index: usize,
    },
    /// A source vertex lies outside one proposed supporting facet.
    #[error(
        "boundary facet {facet_index} is not supporting: vertex {vertex_uuid} lies on side \
         {vertex_side:?}, opposite the certified inside side {inside_side:?}"
    )]
    NonConvexBoundary {
        /// Zero-based boundary-facet index.
        facet_index: usize,
        /// Source vertex that disproves convexity.
        vertex_uuid: Uuid,
        /// Certified side containing the source simplex interior.
        inside_side: Orientation,
        /// Side containing `vertex_uuid`.
        vertex_side: Orientation,
    },
    /// A centroid divisor could not be represented safely.
    #[error("failed to compute boundary facet {facet_index} centroid: {source}")]
    FacetCentroid {
        /// Zero-based boundary-facet index.
        facet_index: usize,
        /// Underlying coordinate conversion error.
        #[source]
        source: CoordinateConversionError,
    },
}

/// Errors returned by geometric queries on an already certified hull.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ConvexHullQueryError {
    /// Predicate setup failed for a query point and hull facet.
    #[error("orientation failed while querying boundary facet {facet_index}: {source}")]
    FacetOrientation {
        /// Zero-based boundary-facet index.
        facet_index: usize,
        /// Underlying coordinate conversion error.
        #[source]
        source: CoordinateConversionError,
    },
}

/// An owned vertex in a convex-hull snapshot.
///
/// The snapshot deliberately omits the source TDS incident-simplex key because
/// that runtime-local handle would be meaningless after the source is mutated or
/// dropped.
#[derive(Clone, Copy, Debug)]
pub struct ConvexHullVertex<U, const D: usize> {
    point: Point<D>,
    uuid: Uuid,
    data: Option<U>,
}

impl<U, const D: usize> ConvexHullVertex<U, D> {
    /// Returns the stable identity copied from the source vertex.
    #[must_use]
    pub const fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// Returns the copied point.
    #[must_use]
    pub const fn point(&self) -> &Point<D> {
        &self.point
    }

    /// Returns the copied optional payload.
    #[must_use]
    pub const fn data(&self) -> Option<&U> {
        self.data.as_ref()
    }
}

impl<U: Copy, const D: usize> From<&Vertex<U, D>> for ConvexHullVertex<U, D> {
    fn from(vertex: &Vertex<U, D>) -> Self {
        Self {
            point: *vertex.point(),
            uuid: vertex.uuid(),
            data: vertex.data().copied(),
        }
    }
}

#[derive(Clone, Debug)]
struct StoredConvexHullFacet<const D: usize> {
    key: u64,
    vertex_indices: SmallBuffer<usize, MAX_PRACTICAL_DIMENSION_SIZE>,
    inside_side: Orientation,
    centroid: [f64; D],
}

#[derive(Debug)]
struct ConvexHullDraft<U, const D: usize> {
    vertices: Vec<ConvexHullVertex<U, D>>,
    facets: Vec<StoredConvexHullFacet<D>>,
}

/// A borrowed facet view over a self-contained [`ConvexHull`].
#[derive(Clone, Copy, Debug)]
pub struct ConvexHullFacetView<'hull, U, const D: usize> {
    index: usize,
    facet: &'hull StoredConvexHullFacet<D>,
    vertices: &'hull [ConvexHullVertex<U, D>],
}

impl<'hull, U, const D: usize> ConvexHullFacetView<'hull, U, D> {
    /// Returns this facet's zero-based index in the hull.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Returns this facet's canonical identifier.
    #[must_use]
    pub const fn key(&self) -> u64 {
        self.facet.key
    }

    /// Iterates over the facet's owned snapshot vertices.
    #[must_use]
    pub fn vertices(self) -> impl ExactSizeIterator<Item = &'hull ConvexHullVertex<U, D>> + 'hull {
        self.facet
            .vertex_indices
            .iter()
            .map(move |&vertex_index| &self.vertices[vertex_index])
    }

    /// Tests whether this facet is visible from `point`.
    ///
    /// A point coplanar with the facet is on the closed hull boundary and is not
    /// classified as visible.
    ///
    /// # Errors
    ///
    /// Returns [`ConvexHullQueryError::FacetOrientation`] if exact predicate
    /// setup cannot represent the supplied coordinates.
    pub fn is_visible_from_point(&self, point: &Point<D>) -> Result<bool, ConvexHullQueryError> {
        let mut points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> = self
            .facet
            .vertex_indices
            .iter()
            .map(|&vertex_index| *self.vertices[vertex_index].point())
            .collect();
        points.push(*point);
        let query_side = simplex_orientation(&points).map_err(|source| {
            ConvexHullQueryError::FacetOrientation {
                facet_index: self.index,
                source,
            }
        })?;

        Ok(matches!(
            (self.facet.inside_side, query_side),
            (Orientation::NEGATIVE, Orientation::POSITIVE)
                | (Orientation::POSITIVE, Orientation::NEGATIVE)
        ))
    }
}

/// A certified, immutable, and self-contained convex-hull snapshot.
///
/// Construction copies only hull vertices, their stable identities and payloads,
/// and the geometric evidence needed by visibility queries. It does not retain
/// source `SimplexKey`, `VertexKey`, or `FacetHandle` values. Consequently the
/// hull remains usable after the source triangulation is changed or dropped.
#[must_use]
#[derive(Clone, Debug)]
pub struct ConvexHull<U, const D: usize> {
    vertices: Vec<ConvexHullVertex<U, D>>,
    facets: Vec<StoredConvexHullFacet<D>>,
}

impl<U, const D: usize> ConvexHull<U, D> {
    /// Returns the number of unique vertices stored by this hull.
    #[must_use]
    pub const fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the number of facets stored by this hull.
    #[must_use]
    pub const fn number_of_facets(&self) -> usize {
        self.facets.len()
    }

    /// Returns the compile-time ambient dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        D
    }

    /// Returns the facet at `index` as an owner-bound view.
    #[must_use]
    pub fn facet(&self, index: usize) -> Option<ConvexHullFacetView<'_, U, D>> {
        self.facets.get(index).map(|facet| ConvexHullFacetView {
            index,
            facet,
            vertices: &self.vertices,
        })
    }

    /// Iterates over all facets as owner-bound views.
    #[must_use]
    pub fn facets(&self) -> impl ExactSizeIterator<Item = ConvexHullFacetView<'_, U, D>> + '_ {
        self.facets
            .iter()
            .enumerate()
            .map(|(index, facet)| ConvexHullFacetView {
                index,
                facet,
                vertices: &self.vertices,
            })
    }
}

impl<U, const D: usize> ConvexHull<U, D>
where
    U: DataType,
{
    /// Parses and certifies a convex hull from a triangulation boundary.
    ///
    /// The source may be any triangulation whose boundary is actually convex;
    /// construction does not assume that Level 3–4 triangulation validity proves
    /// convexity. Every source vertex is therefore checked against every proposed
    /// supporting facet before the owned hull is published.
    ///
    /// # Errors
    ///
    /// Returns [`ConvexHullConstructionError`] when boundary extraction fails,
    /// the input is empty or closed, a facet is malformed or degenerate, or a
    /// source vertex disproves boundary convexity.
    ///
    /// # Performance
    ///
    /// Construction copies the boundary and performs one orientation predicate
    /// per source-vertex/facet pair. Later visibility queries inspect only the
    /// owned hull and never rebuild a source-TDS lookup.
    #[expect(
        clippy::too_many_lines,
        reason = "the publication boundary keeps ordered parsing and proof checks visible together"
    )]
    pub fn try_from_triangulation<K, V>(
        tri: &Triangulation<K, U, V, D>,
    ) -> Result<Self, ConvexHullConstructionError>
    where
        V: DataType,
    {
        if tri.number_of_vertices() == 0 {
            return Err(ConvexHullConstructionError::InsufficientData {
                reason: ConvexHullInsufficientDataReason::NoVertices,
            });
        }
        if tri.number_of_simplices() == 0 {
            return Err(ConvexHullConstructionError::InsufficientData {
                reason: ConvexHullInsufficientDataReason::NoSimplices,
            });
        }

        let boundary = tri.boundary_facets().map_err(|source| {
            ConvexHullConstructionError::BoundaryFacetExtractionFailed {
                source: Box::new(source),
            }
        })?;
        let mut vertices: Vec<ConvexHullVertex<U, D>> = Vec::new();
        let mut vertex_indices = FastHashMap::<Uuid, usize>::default();
        let mut facets = Vec::new();
        let mut facet_keys = FastHashSet::default();

        for (facet_index, facet_result) in boundary.enumerate() {
            let facet = facet_result.map_err(|source| {
                ConvexHullConstructionError::BoundaryFacetExtractionFailed {
                    source: Box::new(QueryError::TriangulationCorrupted {
                        source: Box::new(source.into()),
                    }),
                }
            })?;
            let source_vertices: SmallBuffer<&Vertex<U, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
                facet.vertices().collect();
            if source_vertices.len() != D {
                return Err(ConvexHullConstructionError::InvalidFacet {
                    facet_index,
                    source: FacetError::InsufficientVertices {
                        expected: D,
                        actual: source_vertices.len(),
                        dimension: D,
                    },
                });
            }

            let mut facet_vertex_uuids = FastHashSet::default();
            let mut stored_indices: SmallBuffer<usize, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::with_capacity(D);
            for vertex in source_vertices {
                let vertex_uuid = vertex.uuid();
                if !facet_vertex_uuids.insert(vertex_uuid) {
                    return Err(ConvexHullConstructionError::DuplicateFacetVertex {
                        facet_index,
                        vertex_uuid,
                    });
                }

                let stored_index = if let Some(&stored_index) = vertex_indices.get(&vertex_uuid) {
                    let stored = &vertices[stored_index];
                    if stored.point() != vertex.point()
                        || stored.data().copied() != vertex.data().copied()
                    {
                        return Err(ConvexHullConstructionError::ConflictingVertexIdentity {
                            vertex_uuid,
                        });
                    }
                    stored_index
                } else {
                    let stored_index = vertices.len();
                    vertices.push(ConvexHullVertex::from(vertex));
                    vertex_indices.insert(vertex_uuid, stored_index);
                    stored_index
                };
                stored_indices.push(stored_index);
            }

            let facet_vertex_uuids: SmallBuffer<Uuid, MAX_PRACTICAL_DIMENSION_SIZE> =
                stored_indices
                    .iter()
                    .map(|&index| vertices[index].uuid())
                    .collect();
            let facet_key =
                stable_facet_identifier_from_vertex_uuids(facet_vertex_uuids.as_slice());
            if !facet_keys.insert(facet_key) {
                return Err(ConvexHullConstructionError::DuplicateFacet { facet_key });
            }

            let mut inside_points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
                stored_indices
                    .iter()
                    .map(|&index| *vertices[index].point())
                    .collect();
            inside_points.push(*facet.opposite_vertex().point());
            let inside_side = simplex_orientation(&inside_points).map_err(|source| {
                ConvexHullConstructionError::FacetOrientation {
                    facet_index,
                    source,
                }
            })?;
            if inside_side == Orientation::DEGENERATE {
                return Err(ConvexHullConstructionError::DegenerateFacet { facet_index });
            }

            let divisor = safe_usize_to_scalar(stored_indices.len()).map_err(|source| {
                ConvexHullConstructionError::FacetCentroid {
                    facet_index,
                    source,
                }
            })?;
            let mut centroid = [0.0; D];
            for &stored_index in &stored_indices {
                for (coordinate, value) in centroid
                    .iter_mut()
                    .zip(vertices[stored_index].point().coords())
                {
                    *coordinate += *value / divisor;
                }
            }

            facets.push(StoredConvexHullFacet {
                key: facet_key,
                vertex_indices: stored_indices,
                inside_side,
                centroid,
            });
        }

        if facets.is_empty() {
            return Err(ConvexHullConstructionError::InsufficientData {
                reason: ConvexHullInsufficientDataReason::NoBoundaryFacets,
            });
        }

        ConvexHullDraft { vertices, facets }.certify(tri)
    }

    /// Finds all facets visible from `point`.
    ///
    /// # Errors
    ///
    /// Returns [`ConvexHullQueryError`] if an orientation predicate cannot be
    /// evaluated for the query coordinates.
    pub fn find_visible_facets(
        &self,
        point: &Point<D>,
    ) -> Result<Vec<ConvexHullFacetView<'_, U, D>>, ConvexHullQueryError> {
        self.facets()
            .filter_map(|facet| match facet.is_visible_from_point(point) {
                Ok(true) => Some(Ok(facet)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            })
            .collect()
    }

    /// Finds the visible facet whose centroid is nearest to `point`.
    ///
    /// # Errors
    ///
    /// Returns [`ConvexHullQueryError`] if visibility classification fails.
    pub fn find_nearest_visible_facet(
        &self,
        point: &Point<D>,
    ) -> Result<Option<ConvexHullFacetView<'_, U, D>>, ConvexHullQueryError> {
        let mut nearest = None;
        for facet in self.facets() {
            if !facet.is_visible_from_point(point)? {
                continue;
            }

            let distance = squared_distance(point.coords(), &facet.facet.centroid);
            if nearest.as_ref().is_none_or(
                |(_, nearest_distance): &(ConvexHullFacetView<'_, U, D>, f64)| {
                    distance.total_cmp(nearest_distance).is_lt()
                },
            ) {
                nearest = Some((facet, distance));
            }
        }

        Ok(nearest.map(|(facet, _)| facet))
    }

    /// Returns whether `point` lies strictly outside the closed hull.
    ///
    /// # Errors
    ///
    /// Returns [`ConvexHullQueryError`] if an orientation predicate cannot be
    /// evaluated for the query coordinates.
    pub fn is_point_outside(&self, point: &Point<D>) -> Result<bool, ConvexHullQueryError> {
        for facet in self.facets() {
            if facet.is_visible_from_point(point)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl<U, const D: usize> ConvexHullDraft<U, D>
where
    U: DataType,
{
    fn certify<K, V>(
        self,
        tri: &Triangulation<K, U, V, D>,
    ) -> Result<ConvexHull<U, D>, ConvexHullConstructionError>
    where
        V: DataType,
    {
        for (facet_index, facet) in self.facets.iter().enumerate() {
            let facet_uuids: FastHashSet<Uuid> = facet
                .vertex_indices
                .iter()
                .map(|&index| self.vertices[index].uuid())
                .collect();
            let facet_points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> = facet
                .vertex_indices
                .iter()
                .map(|&index| *self.vertices[index].point())
                .collect();

            for (_, source_vertex) in tri.vertices() {
                if facet_uuids.contains(&source_vertex.uuid()) {
                    continue;
                }
                let mut points = facet_points.clone();
                points.push(*source_vertex.point());
                let vertex_side = simplex_orientation(&points).map_err(|source| {
                    ConvexHullConstructionError::FacetOrientation {
                        facet_index,
                        source,
                    }
                })?;
                if vertex_side != Orientation::DEGENERATE && vertex_side != facet.inside_side {
                    return Err(ConvexHullConstructionError::NonConvexBoundary {
                        facet_index,
                        vertex_uuid: source_vertex.uuid(),
                        inside_side: facet.inside_side,
                        vertex_side,
                    });
                }
            }
        }

        Ok(ConvexHull {
            vertices: self.vertices,
            facets: self.facets,
        })
    }
}

fn squared_distance<const D: usize>(left: &[f64; D], right: &[f64; D]) -> f64 {
    let mut difference = [0.0; D];
    for ((difference, left), right) in difference.iter_mut().zip(left).zip(right) {
        *difference = *left - *right;
    }
    squared_norm(&difference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::DelaunayTriangulationBuilder;
    use crate::core::tds::TdsBuilder;
    use crate::delaunay_model::DelaunayTriangulation;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::triangulation::builder::TriangulationBuilder;
    use crate::vertex;
    use std::assert_matches;
    use std::sync::{Arc, Barrier};

    macro_rules! simplex_hull_test {
        ($name:ident, $dimension:literal) => {
            #[test]
            fn $name() {
                let mut vertices = Vec::new();
                vertices.push(Vertex::<(), $dimension>::try_new([0.0; $dimension]).unwrap());
                for axis in 0..$dimension {
                    let mut point = [0.0; $dimension];
                    point[axis] = 1.0;
                    vertices.push(Vertex::try_new(point).unwrap());
                }
                let dt: DelaunayTriangulation<_, (), (), $dimension> =
                    DelaunayTriangulationBuilder::new(&vertices)
                        .build()
                        .unwrap();
                let hull = ConvexHull::try_from_triangulation(dt.as_triangulation()).unwrap();

                assert_eq!(hull.dimension(), $dimension);
                assert_eq!(hull.number_of_vertices(), $dimension + 1);
                assert_eq!(hull.number_of_facets(), $dimension + 1);
                assert!(
                    hull.facets()
                        .all(|facet| facet.vertices().len() == $dimension)
                );
            }
        };
    }

    simplex_hull_test!(simplex_hull_2d, 2);
    simplex_hull_test!(simplex_hull_3d, 3);
    simplex_hull_test!(simplex_hull_4d, 4);
    simplex_hull_test!(simplex_hull_5d, 5);

    fn tetrahedron()
    -> DelaunayTriangulation<crate::geometry::kernel::AdaptiveKernel<f64>, (), (), 3> {
        let vertices: [Vertex<(), 3>; 4] = [
            vertex![0.0, 0.0, 0.0].unwrap(),
            vertex![1.0, 0.0, 0.0].unwrap(),
            vertex![0.0, 1.0, 0.0].unwrap(),
            vertex![0.0, 0.0, 1.0].unwrap(),
        ];
        DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap()
    }

    #[test]
    fn snapshot_outlives_source_and_answers_queries() {
        let hull = {
            let dt = tetrahedron();
            ConvexHull::try_from_triangulation(dt.as_triangulation()).unwrap()
        };
        let inside = Point::try_new([0.2, 0.2, 0.2]).unwrap();
        let outside = Point::try_new([2.0, 2.0, 2.0]).unwrap();
        let surface = Point::try_new([0.2, 0.3, 0.0]).unwrap();

        assert!(!hull.is_point_outside(&inside).unwrap());
        assert!(hull.is_point_outside(&outside).unwrap());
        assert!(!hull.is_point_outside(&surface).unwrap());
        assert!(!hull.find_visible_facets(&outside).unwrap().is_empty());
        assert!(hull.find_nearest_visible_facet(&outside).unwrap().is_some());
    }

    #[test]
    fn snapshot_copies_vertex_payload_without_tds_handles() {
        let vertices: [Vertex<u8, 2>; 3] = [
            vertex![0.0, 0.0; data = 10_u8].unwrap(),
            vertex![1.0, 0.0; data = 11_u8].unwrap(),
            vertex![0.0, 1.0; data = 12_u8].unwrap(),
        ];
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();
        let hull = ConvexHull::try_from_triangulation(dt.as_triangulation()).unwrap();
        let payloads: FastHashSet<u8> = hull
            .facets()
            .flat_map(|facet| facet.vertices().filter_map(|vertex| vertex.data().copied()))
            .collect();

        assert_eq!(payloads, [10, 11, 12].into_iter().collect());
    }

    #[test]
    fn rejects_valid_triangulation_with_nonconvex_boundary() {
        let vertices: [Vertex<(), 2>; 4] = [
            vertex![0.0, 0.0].unwrap(),
            vertex![2.0, 0.0].unwrap(),
            vertex![1.0, 0.5].unwrap(),
            vertex![0.0, 2.0].unwrap(),
        ];
        let simplices = [vec![0, 1, 2], vec![0, 2, 3]];
        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();
        let triangulation = TriangulationBuilder::new(tds, AdaptiveKernel::new())
            .canonicalizing()
            .build()
            .unwrap();

        let error = ConvexHull::try_from_triangulation(&triangulation)
            .expect_err("a concave boundary must not publish as a convex hull");

        assert_matches!(error, ConvexHullConstructionError::NonConvexBoundary { .. });
    }

    #[test]
    fn owner_bound_facet_view_reports_visibility() {
        let dt = tetrahedron();
        let hull = ConvexHull::try_from_triangulation(dt.as_triangulation()).unwrap();
        let outside = Point::try_new([2.0, 2.0, 2.0]).unwrap();

        assert!(
            hull.facets()
                .any(|facet| facet.is_visible_from_point(&outside).unwrap())
        );
    }

    #[test]
    fn self_contained_queries_are_synchronized_by_immutable_borrows() {
        const WORKERS: usize = 4;
        let dt = tetrahedron();
        let hull = Arc::new(ConvexHull::try_from_triangulation(dt.as_triangulation()).unwrap());
        drop(dt);
        let start = Arc::new(Barrier::new(WORKERS));

        #[expect(
            clippy::needless_collect,
            reason = "all barrier participants must be spawned before any handle is joined"
        )]
        let handles: Vec<_> = (0..WORKERS)
            .map(|worker| {
                let hull = Arc::clone(&hull);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    let coordinate = 2.0 + safe_usize_to_scalar(worker).unwrap();
                    let point = Point::try_new([coordinate, 2.0, 2.0]).unwrap();
                    hull.is_point_outside(&point).unwrap()
                })
            })
            .collect();

        assert!(handles.into_iter().all(|handle| handle.join().unwrap()));
    }
}

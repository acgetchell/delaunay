//! General Euclidean edits on a proof-bearing Levels 1–4 owner.
//!
//! These operations accept non-Delaunay input and never certify Level 5.
//! For toroidal edits in a selected lifted chart, compose the transactional
//! [`PachnerMoves`](crate::pachner::PachnerMoves) proposal API instead.

use crate::core::algorithms::insertion::InsertionError;
use crate::core::tds::VertexKey;
use crate::core::traits::data_type::DataType;
use crate::core::vertex::Vertex;
use crate::geometry::kernel::Kernel;
use crate::topology::traits::topological_space::TopologyKind;
use crate::triangulation::Triangulation;
pub use crate::triangulation::repair::VertexRemovalError;

use thiserror::Error;

/// Structured rejection of a general Levels 1–4 vertex edit.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TriangulationEditError {
    /// General cavity editing requires Euclidean coordinates.
    #[error(
        "general vertex editing is unsupported for {topology:?}; use a chart-bound Pachner proposal"
    )]
    UnsupportedTopology {
        /// The owner's selected geometry.
        topology: TopologyKind,
    },
    /// Insertion or its cumulative postcondition failed.
    #[error("Level 4 insertion failed: {source}")]
    Insertion {
        /// The original insertion diagnostic.
        #[source]
        source: Box<InsertionError>,
    },
    /// Cavity deletion or its cumulative postcondition failed.
    #[error("Level 4 deletion failed: {source}")]
    Deletion {
        /// The original removal diagnostic.
        #[source]
        source: VertexRemovalError,
    },
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    /// Inserts an exact-coordinate vertex while preserving Levels 1–4.
    ///
    /// Interior insertion subdivides the containing simplex; exterior insertion
    /// extends the hull. Existing non-Delaunay state is accepted. The operation
    /// never perturbs the input coordinates or performs Level 5 repair. New
    /// simplices have no payload; surviving vertices and simplices retain theirs.
    /// A successful edit passes cumulative realization validation regardless of
    /// the configured validation cadence. Failure restores the complete owner,
    /// including generational keys and construction evidence.
    ///
    /// # Errors
    ///
    /// Returns [`TriangulationEditError`] for duplicates, unsupported degenerate
    /// locations, an empty owner requiring bootstrap construction, a non-Euclidean
    /// topology, or a failed geometric/topological postcondition. Toroidal callers
    /// can select a lifted simplex using [`PachnerMove::K1Insert`](crate::pachner::PachnerMove::K1Insert)
    /// and execute its checked proposal through [`PachnerMoves`](crate::pachner::PachnerMoves).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #   #[error(transparent)] Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #   #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #   #[error(transparent)] Edit(#[from] delaunay::TriangulationEditError),
    /// #   #[error(transparent)] Realization(#[from] delaunay::TriangulationRealizationValidationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let mut tri = DelaunayTriangulationBuilder::new(&vertices).build_triangulation()?;
    /// let key = tri.insert_vertex(vertex![0.2, 0.3]?)?;
    /// assert!(tri.vertex(key).is_some());
    /// tri.validate_realization()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert_vertex(
        &mut self,
        vertex: Vertex<U, D>,
    ) -> Result<VertexKey, TriangulationEditError> {
        self.require_euclidean_edit()?;
        self.insert_vertex_preserving_realization(vertex)
            .map_err(|source| TriangulationEditError::Insertion {
                source: Box::new(source),
            })
    }

    /// Deletes a vertex by retriangulating its cavity, preserving Levels 1–4.
    ///
    /// Returns the number of removed simplices, or zero for a missing key.
    /// Fan filling need not produce a Delaunay triangulation. New simplices have
    /// no payload, and surviving entities retain UUIDs and payloads. Failure
    /// restores the complete owner and retains the typed operation or invariant
    /// diagnostic. In high dimensions an arbitrary fan may lack a PL-link proof
    /// and is rejected; inverse stellar deletion is available through
    /// [`PachnerMoves`](crate::pachner::PachnerMoves).
    ///
    /// # Errors
    ///
    /// Returns [`TriangulationEditError`] when the topology is non-Euclidean,
    /// the cavity cannot be filled, or Levels 1–4 reject the candidate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #   #[error(transparent)] Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #   #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #   #[error(transparent)] Edit(#[from] delaunay::TriangulationEditError),
    /// #   #[error(transparent)] Realization(#[from] delaunay::TriangulationRealizationValidationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let mut tri = DelaunayTriangulationBuilder::new(&vertices).build_triangulation()?;
    /// let key = tri.insert_vertex(vertex![0.2, 0.3]?)?;
    /// assert_eq!(tri.delete_vertex(key)?, 3);
    /// assert!(tri.vertex(key).is_none());
    /// tri.validate_realization()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete_vertex(
        &mut self,
        vertex_key: VertexKey,
    ) -> Result<usize, TriangulationEditError> {
        self.require_euclidean_edit()?;
        self.remove_vertex_preserving_realization(vertex_key)
            .map_err(|source| TriangulationEditError::Deletion { source })
    }

    /// Prevents Euclidean cavity routines from interpreting periodic quotient coordinates.
    const fn require_euclidean_edit(&self) -> Result<(), TriangulationEditError> {
        match self.topology_kind() {
            TopologyKind::Euclidean => Ok(()),
            topology => Err(TriangulationEditError::UnsupportedTopology { topology }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayRefinementBuilder;
    use crate::core::tds::{Tds, TdsBuilder, TopologyOwner};
    use crate::geometry::kernel::RobustKernel;
    use crate::triangulation::builder::TriangulationBuilder;
    use crate::triangulation::validation::ValidationPolicy;
    use crate::{DelaunayTriangulationBuilder, vertex};

    /// Supplies a convex quadrilateral with the strictly non-Delaunay diagonal.
    fn non_delaunay_quad() -> Triangulation<RobustKernel<f64>, u32, u32, 2> {
        let vertices = [
            vertex!([0.0, 0.0]; data = 10).unwrap(),
            vertex!([2.0, 0.0]; data = 20).unwrap(),
            vertex!([2.0, 1.0]; data = 30).unwrap(),
            vertex!([0.0, 2.0]; data = 40).unwrap(),
        ];
        let tds = TdsBuilder::new(&vertices, &[vec![0, 1, 3], vec![1, 2, 3]])
            .simplex_data_type::<u32>()
            .build()
            .unwrap();
        TriangulationBuilder::new(tds, RobustKernel::new())
            .build()
            .unwrap()
    }

    /// Exercises exact-coordinate insertion and inverse cavity deletion in each dimension.
    fn round_trip<const D: usize>() {
        let mut vertices = vec![vertex!([0.0; D]; data = 7_u32).unwrap()];
        for axis in 0..D {
            let mut coordinates = [0.0; D];
            coordinates[axis] = 1.0;
            vertices.push(vertex!(coordinates; data = 9).unwrap());
        }
        let simplices = [(0..=D).collect::<Vec<_>>()];
        let tds = TdsBuilder::new(&vertices, &simplices)
            .simplex_data_type::<u32>()
            .build()
            .unwrap();
        let mut tri = TriangulationBuilder::new(tds, RobustKernel::new())
            .canonicalizing()
            .build()
            .unwrap();
        let original_vertices = tri
            .vertices()
            .map(|(key, vertex)| (key, vertex.uuid(), *vertex.data().unwrap()))
            .collect::<Vec<_>>();
        let inserted = vertex!([0.1; D]; data = 99).unwrap();
        let inserted_uuid = inserted.uuid();
        let key = tri.insert_vertex(inserted).unwrap();
        let actual = tri.vertex(key).unwrap();
        assert_eq!(actual.uuid(), inserted_uuid);
        assert_eq!(
            actual.point().coords().map(f64::to_bits),
            [0.1_f64.to_bits(); D]
        );
        assert_eq!(actual.data(), Some(&99));
        assert_eq!(tri.number_of_simplices(), D + 1);
        tri.validate_realization().unwrap();
        assert_eq!(tri.delete_vertex(key).unwrap(), D + 1);
        assert_eq!(tri.number_of_simplices(), 1);
        assert_eq!(tri.delete_vertex(key).unwrap(), 0);
        assert_eq!(
            tri.vertices()
                .map(|(key, vertex)| (key, vertex.uuid(), *vertex.data().unwrap()))
                .collect::<Vec<_>>(),
            original_vertices
        );
        tri.validate_realization().unwrap();
    }

    macro_rules! dimension_tests {
        ($($dimension:literal),+) => { $(pastey::paste! {
            #[test]
            fn [<insertion_deletion_round_trip_ $dimension d>]() {
                round_trip::<$dimension>();
            }
        })+ };
    }
    dimension_tests!(2, 3, 4, 5);

    #[test]
    fn non_delaunay_input_supports_interior_and_exterior_insertion() {
        let mut tri = non_delaunay_quad();
        assert!(DelaunayRefinementBuilder::new(tri.clone()).build().is_err());
        let key = tri
            .insert_vertex(vertex!([0.2, 0.3]; data = 50).unwrap())
            .unwrap();
        tri.validate_realization().unwrap();
        tri.delete_vertex(key).unwrap();
        tri.insert_vertex(vertex!([3.0, 3.0]; data = 60).unwrap())
            .unwrap();
        tri.validate_realization().unwrap();
    }

    #[test]
    fn failed_insertion_restores_owner_keys_generation_payloads_and_evidence() {
        let mut tri = non_delaunay_quad();
        tri.try_set_validation_policy(ValidationPolicy::ExplicitOnly)
            .unwrap();
        let original = tri.tds.clone_for_rollback();
        let serialized = serde_json::to_value(&tri.tds).unwrap();
        let handles = tri
            .vertices()
            .map(|(key, vertex)| (key, vertex.incident_simplex()))
            .collect::<Vec<_>>();
        let identity = tri.topology_owner_id();
        let generation = tri.topology_generation();
        let evidence = tri.topology_construction_provenance;
        // This point lies exactly on the existing interior diagonal. A stellar
        // subdivision cannot realize it without a degenerate simplex.
        assert!(matches!(
            tri.insert_vertex(vertex!([1.0, 1.0]; data = 99).unwrap()),
            Err(TriangulationEditError::Insertion { .. })
        ));
        assert_eq!(tri.tds, original);
        assert_eq!(serde_json::to_value(&tri.tds).unwrap(), serialized);
        assert_eq!(
            tri.vertices()
                .map(|(key, vertex)| (key, vertex.incident_simplex()))
                .collect::<Vec<_>>(),
            handles
        );
        assert_eq!(tri.topology_owner_id(), identity);
        assert_eq!(tri.topology_generation(), generation);
        assert_eq!(tri.topology_construction_provenance, evidence);
        assert_eq!(tri.validation_policy(), ValidationPolicy::ExplicitOnly);
        tri.validate_realization().unwrap();
    }

    #[test]
    fn failed_deletion_restores_complete_owner_and_typed_error() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];
        let mut tri = DelaunayTriangulationBuilder::new(&vertices)
            .build_triangulation()
            .unwrap();
        let key = tri.vertices().next().unwrap().0;
        let original = tri.tds.clone_for_rollback();
        let serialized = serde_json::to_value(&tri.tds).unwrap();
        let identity = tri.topology_owner_id();
        let generation = tri.topology_generation();
        assert!(matches!(
            tri.delete_vertex(key),
            Err(TriangulationEditError::Deletion { .. })
        ));
        assert_eq!(tri.tds, original);
        assert_eq!(serde_json::to_value(&tri.tds).unwrap(), serialized);
        assert_eq!(tri.topology_owner_id(), identity);
        assert_eq!(tri.topology_generation(), generation);
        tri.validate_realization().unwrap();
    }

    #[test]
    fn empty_owner_and_duplicate_insertion_fail_without_mutation() {
        let mut empty = TriangulationBuilder::new(Tds::<(), (), 2>::empty(), RobustKernel::new())
            .build()
            .unwrap();
        assert!(matches!(empty.insert_vertex(vertex![0.0, 0.0].unwrap()),
            Err(TriangulationEditError::Insertion { source })
                if matches!(*source, InsertionError::PublishedOwnerBootstrapRequiresBuilder { dimension: 2 })));
        assert_eq!(empty.number_of_vertices(), 0);
        let mut tri = non_delaunay_quad();
        let before = serde_json::to_value(&tri.tds).unwrap();
        assert!(
            matches!(tri.insert_vertex(vertex!([0.0, 0.0]; data = 99).unwrap()),
            Err(TriangulationEditError::Insertion { source })
                if matches!(*source, InsertionError::DuplicateCoordinates { .. }))
        );
        assert_eq!(serde_json::to_value(&tri.tds).unwrap(), before);
    }
}

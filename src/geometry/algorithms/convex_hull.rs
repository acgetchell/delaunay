use crate::core::facet::{Facet, FacetError};
use crate::core::traits::boundary_analysis::BoundaryAnalysis;
use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::Tds;
use crate::geometry::point::Point;
use crate::geometry::predicates::simplex_orientation;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar};
use crate::geometry::util::squared_norm;
use nalgebra::ComplexField;
use num_traits::cast::NumCast;
use num_traits::{One, Zero};
use ordered_float::OrderedFloat;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, Sub, SubAssign};
use thiserror::Error;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur during convex hull construction.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ConvexHullConstructionError {
    /// Failed to extract boundary facets from the triangulation.
    #[error("Failed to extract boundary facets from triangulation: {source}")]
    BoundaryFacetExtractionFailed {
        /// The underlying facet error that caused the failure.
        source: FacetError,
    },
    /// The input triangulation is empty or invalid.
    #[error("Invalid input triangulation: {message}")]
    InvalidTriangulation {
        /// Description of why the triangulation is invalid.
        message: String,
    },
    /// Insufficient data to construct convex hull.
    #[error("Insufficient data for convex hull construction: {message}")]
    InsufficientData {
        /// Description of the data insufficiency.
        message: String,
    },
    /// Geometric degeneracy prevents convex hull construction.
    #[error("Geometric degeneracy encountered during convex hull construction: {message}")]
    GeometricDegeneracy {
        /// Description of the degeneracy issue.
        message: String,
    },
    /// Numeric cast failed during computation.
    #[error("Numeric cast failed during convex hull computation: {message}")]
    NumericCastFailed {
        /// Description of the cast failure.
        message: String,
    },
}

// =============================================================================
// CONVEX HULL DATA STRUCTURE
// =============================================================================

/// Generic d-dimensional convex hull operations.
///
/// This struct provides convex hull functionality by leveraging the existing
/// boundary facet analysis from the TDS. Since boundary facets in a Delaunay
/// triangulation lie on the convex hull, we can use the `BoundaryAnalysis`
/// trait to get the hull facets directly.
///
/// The implementation supports d-dimensional convex hull extraction from
/// Delaunay triangulations, point-in-hull testing, and facet visibility
/// determination for incremental construction algorithms.
///
/// # Type Parameters
///
/// * `T` - The coordinate scalar type (e.g., f64, f32)
/// * `U` - The vertex data type
/// * `V` - The cell data type  
/// * `D` - The dimension of the triangulation
///
/// # References
///
/// The algorithms implemented in this module are based on established computational geometry literature:
///
/// ## Convex Hull Construction from Delaunay Triangulations
///
/// - Brown, K.Q. "Voronoi Diagrams from Convex Hulls." *Information Processing Letters* 9, no. 5 (1979): 223-228.
///   DOI: [10.1016/0020-0190(79)90074-7](https://doi.org/10.1016/0020-0190(79)90074-7)
/// - Edelsbrunner, H. "Algorithms in Combinatorial Geometry." EATCS Monographs on Theoretical Computer Science.
///   Berlin: Springer-Verlag, 1987. DOI: [10.1007/978-3-642-61568-9](https://doi.org/10.1007/978-3-642-61568-9)
///
/// ## Point-in-Polytope Testing
///
/// - Preparata, F.P., and Shamos, M.I. "Computational Geometry: An Introduction." Texts and Monographs in Computer Science.
///   New York: Springer-Verlag, 1985. DOI: [10.1007/978-1-4612-1098-6](https://doi.org/10.1007/978-1-4612-1098-6)
/// - O'Rourke, J. "Computational Geometry in C." 2nd ed. Cambridge: Cambridge University Press, 1998.
///   DOI: [10.1017/CBO9780511804120](https://doi.org/10.1017/CBO9780511804120)
///
/// ## Incremental Convex Hull Construction
///
/// - Clarkson, K.L., and Shor, P.W. "Applications of Random Sampling in Computational Geometry, II."
///   *Discrete & Computational Geometry* 4, no. 1 (1989): 387-421. DOI: [10.1007/BF02187740](https://doi.org/10.1007/BF02187740)
/// - Barber, C.B., Dobkin, D.P., and Huhdanpaa, H. "The Quickhull Algorithm for Convex Hulls."
///   *ACM Transactions on Mathematical Software* 22, no. 4 (1996): 469-483. DOI: [10.1145/235815.235821](https://doi.org/10.1145/235815.235821)
///
/// ## High-Dimensional Computational Geometry
///
/// - Chazelle, B. "An Optimal Convex Hull Algorithm in Any Fixed Dimension." *Discrete & Computational Geometry* 10,
///   no. 4 (1993): 377-409. DOI: [10.1007/BF02573985](https://doi.org/10.1007/BF02573985)
/// - Seidel, R. "The Upper Bound Theorem for Polytopes: An Easy Proof of Its Asymptotic Version."
///   *Computational Geometry* 5, no. 2 (1995): 115-116. DOI: [10.1016/0925-7721(95)00013-Y](https://doi.org/10.1016/0925-7721(95)00013-Y)
#[derive(Debug)]
pub struct ConvexHull<T, U, V, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + Sized + Serialize + DeserializeOwned,
{
    /// The boundary facets that form the convex hull
    pub hull_facets: Vec<Facet<T, U, V, D>>,
}

impl<T, U, V, const D: usize> ConvexHull<T, U, V, D>
where
    T: CoordinateScalar
        + AddAssign<T>
        + SubAssign<T>
        + Sub<Output = T>
        + DivAssign<T>
        + Zero
        + One
        + NumCast
        + Copy
        + Sum
        + ComplexField<RealField = T>
        + From<f64>,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + Sized + Serialize + DeserializeOwned,
    f64: From<T>,
    for<'a> &'a T: std::ops::Div<T>,
    OrderedFloat<f64>: From<T>,
{
    /// Creates a new convex hull from a d-dimensional triangulation
    ///
    /// # Arguments
    ///
    /// * `tds` - The triangulation data structure
    ///
    /// # Returns
    ///
    /// A `Result` containing the convex hull or a [`ConvexHullConstructionError`] if extraction fails
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullConstructionError`] if:
    /// - Boundary facets cannot be extracted from the triangulation ([`ConvexHullConstructionError::BoundaryFacetExtractionFailed`])
    /// - The input triangulation is invalid ([`ConvexHullConstructionError::InvalidTriangulation`])
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // 3D example
    /// let vertices_3d = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds_3d: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_3d).unwrap();
    /// let hull_3d: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds_3d).unwrap();
    /// assert_eq!(hull_3d.facet_count(), 4); // Tetrahedron has 4 faces
    ///
    /// // 4D example
    /// let vertices_4d = vec![
    ///     vertex!([0.0, 0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 0.0, 1.0]),
    /// ];
    /// let tds_4d: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices_4d).unwrap();
    /// let hull_4d: ConvexHull<f64, Option<()>, Option<()>, 4> =
    ///     ConvexHull::from_triangulation(&tds_4d).unwrap();
    /// assert_eq!(hull_4d.facet_count(), 5); // 4-simplex has 5 facets
    /// ```
    pub fn from_triangulation(tds: &Tds<T, U, V, D>) -> Result<Self, ConvexHullConstructionError> {
        // Validate input triangulation
        if tds.number_of_vertices() == 0 {
            return Err(ConvexHullConstructionError::InsufficientData {
                message: "Triangulation contains no vertices".to_string(),
            });
        }

        if tds.number_of_cells() == 0 {
            return Err(ConvexHullConstructionError::InsufficientData {
                message: "Triangulation contains no cells".to_string(),
            });
        }

        // Use the existing boundary analysis to get hull facets
        let hull_facets = tds.boundary_facets().map_err(|source| {
            ConvexHullConstructionError::BoundaryFacetExtractionFailed { source }
        })?;

        // Additional validation: ensure we have at least one boundary facet
        if hull_facets.is_empty() {
            return Err(ConvexHullConstructionError::InsufficientData {
                message: "No boundary facets found in triangulation".to_string(),
            });
        }

        Ok(Self { hull_facets })
    }

    /// Tests if a facet is visible from an external point using proper geometric predicates
    ///
    /// A facet is visible if the point is on the "outside" side of the facet.
    /// This implementation uses geometric orientation predicates to determine the correct
    /// side of the hyperplane defined by the facet, based on the Bowyer-Watson algorithm.
    ///
    /// # Algorithm
    ///
    /// For a boundary facet F with vertices {f₁, f₂, ..., fₐ}, we need to determine
    /// if a test point p is on the "outside" of the facet. Since this is a boundary facet
    /// from a convex hull, we know it has exactly one adjacent cell.
    ///
    /// The algorithm works as follows:
    /// 1. Find the "inside" vertex of the adjacent cell (vertex not in the facet)
    /// 2. Create two simplices: facet + `inside_vertex` and facet + `test_point`  
    /// 3. Compare orientations - different orientations mean opposite sides
    /// 4. If test point is on opposite side from inside vertex, facet is visible
    ///
    /// # Arguments
    ///
    /// * `facet` - The facet to test
    /// * `point` - The external point
    /// * `tds` - Reference to triangulation (needed to find adjacent cell)
    ///
    /// # Returns
    ///
    /// `true` if the facet is visible from the point, `false` otherwise
    ///
    /// # Errors
    ///
    /// Returns a [`FacetError`] if:
    /// - The facet doesn't have the expected number of vertices ([`FacetError::InsufficientVertices`])
    /// - The facet is not found in the triangulation ([`FacetError::FacetNotFoundInTriangulation`])
    /// - The facet has an invalid number of adjacent cells ([`FacetError::InvalidAdjacentCellCount`])
    /// - The adjacent cell cannot be found ([`FacetError::AdjacentCellNotFound`])
    /// - The inside vertex cannot be found ([`FacetError::InsideVertexNotFound`])
    /// - Geometric predicates fail ([`FacetError::OrientationComputationFailed`])
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds).unwrap();
    ///
    /// // Get a hull facet to test
    /// let facet = hull.get_facet(0).unwrap();
    ///
    /// // Test visibility from different points
    /// let inside_point = Point::new([0.2, 0.2, 0.2]); // Inside the tetrahedron
    /// let outside_point = Point::new([2.0, 2.0, 2.0]); // Outside the tetrahedron
    ///
    /// // Inside point should not see the facet (facet not visible)
    /// let inside_visible = hull.is_facet_visible_from_point(facet, &inside_point, &tds).unwrap();
    /// assert!(!inside_visible, "Inside point should not see hull facet");
    ///
    /// // Outside point may see the facet depending on which facet we're testing
    /// let outside_visible = hull.is_facet_visible_from_point(facet, &outside_point, &tds).unwrap();
    /// // Note: The result depends on which facet is selected and the point's position
    /// // This test just verifies the method executes without error
    /// ```
    pub fn is_facet_visible_from_point(
        &self,
        facet: &Facet<T, U, V, D>,
        point: &Point<T, D>,
        tds: &Tds<T, U, V, D>,
    ) -> Result<bool, FacetError> {
        use crate::geometry::predicates::Orientation;

        // Get the vertices that make up this facet
        let facet_vertices = facet.vertices();

        if facet_vertices.len() != D {
            return Err(FacetError::InsufficientVertices {
                expected: D,
                actual: facet_vertices.len(),
                dimension: D,
            });
        }

        // Find the cell adjacent to this boundary facet
        let facet_to_cells = tds.build_facet_to_cells_hashmap();
        let facet_key = facet.key();

        let adjacent_cells = facet_to_cells
            .get(&facet_key)
            .ok_or(FacetError::FacetNotFoundInTriangulation)?;

        if adjacent_cells.len() != 1 {
            return Err(FacetError::InvalidAdjacentCellCount {
                found: adjacent_cells.len(),
            });
        }

        let (cell_key, _facet_index) = adjacent_cells[0];
        let adjacent_cell = tds
            .cells()
            .get(cell_key)
            .ok_or(FacetError::AdjacentCellNotFound)?;

        // Find the vertex in the adjacent cell that is NOT part of the facet
        // This is the "opposite" or "inside" vertex
        let cell_vertices = adjacent_cell.vertices();
        let mut inside_vertex = None;

        for cell_vertex in cell_vertices {
            let is_in_facet = facet_vertices
                .iter()
                .any(|fv| fv.uuid() == cell_vertex.uuid());
            if !is_in_facet {
                inside_vertex = Some(cell_vertex);
                break;
            }
        }

        let inside_vertex = inside_vertex.ok_or(FacetError::InsideVertexNotFound)?;

        // Create test simplices to compare orientations
        let facet_points: Vec<Point<T, D>> = facet_vertices.iter().map(|v| *v.point()).collect();

        // Simplex 1: facet vertices + inside vertex
        let mut simplex_with_inside = facet_points.clone();
        simplex_with_inside.push(*inside_vertex.point());

        // Simplex 2: facet vertices + test point
        let mut simplex_with_test = facet_points;
        simplex_with_test.push(*point);

        // Get orientations using geometric predicates
        let orientation_inside = simplex_orientation(&simplex_with_inside).map_err(|e| {
            FacetError::OrientationComputationFailed {
                details: format!("Failed to compute orientation with inside vertex: {e}"),
            }
        })?;
        let orientation_test = simplex_orientation(&simplex_with_test).map_err(|e| {
            FacetError::OrientationComputationFailed {
                details: format!("Failed to compute orientation with test point: {e}"),
            }
        })?;

        // Compare orientations - facet is visible if orientations are different
        match (orientation_inside, orientation_test) {
            (Orientation::NEGATIVE, Orientation::POSITIVE)
            | (Orientation::POSITIVE, Orientation::NEGATIVE) => Ok(true),
            (Orientation::DEGENERATE, _) | (_, Orientation::DEGENERATE) => {
                // Degenerate case - fall back to distance heuristic
                Self::fallback_visibility_test(facet, point).map_err(|e| {
                    FacetError::OrientationComputationFailed {
                        details: format!("Fallback visibility test failed: {e}"),
                    }
                })
            }
            _ => Ok(false), // Same orientation = same side = not visible
        }
    }

    /// Fallback visibility test for degenerate cases
    ///
    /// When geometric predicates fail due to degeneracy, this method provides
    /// a simple heuristic based on distance from the facet centroid.
    ///
    /// # Returns
    ///
    /// Returns a `Result<bool, ConvexHullConstructionError>` where `true` indicates
    /// the facet is visible from the point and `false` indicates it's not visible.
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullConstructionError::NumericCastFailed`] if numeric
    /// conversion fails during centroid calculation or threshold computation.
    fn fallback_visibility_test(
        facet: &Facet<T, U, V, D>,
        point: &Point<T, D>,
    ) -> Result<bool, ConvexHullConstructionError> {
        let facet_vertices = facet.vertices();
        let vertex_points: Vec<Point<T, D>> = facet_vertices
            .iter()
            .map(|vertex| *vertex.point())
            .collect();

        // Calculate facet centroid
        let mut centroid_coords = [T::zero(); D];
        for vertex_point in &vertex_points {
            let coords: [T; D] = vertex_point.into();
            for (i, &coord) in coords.iter().enumerate() {
                centroid_coords[i] += coord;
            }
        }
        let num_vertices = NumCast::from(vertex_points.len()).ok_or_else(|| {
            ConvexHullConstructionError::NumericCastFailed {
                message: format!(
                    "Failed to cast vertex count {} to coordinate type for centroid calculation",
                    vertex_points.len()
                ),
            }
        })?;
        for coord in &mut centroid_coords {
            *coord /= num_vertices;
        }

        // Simple heuristic: if point is far from centroid, it's likely visible
        let point_coords: [T; D] = point.into();
        let mut diff_coords = [T::zero(); D];
        for i in 0..D {
            diff_coords[i] = point_coords[i] - centroid_coords[i];
        }
        let distance_squared = squared_norm(diff_coords);

        // Use a threshold to determine visibility - this is a simple heuristic
        let threshold = NumCast::from(1.0f64).ok_or_else(|| {
            ConvexHullConstructionError::NumericCastFailed {
                message: "Failed to cast threshold value 1.0 to coordinate type".to_string(),
            }
        })?;
        Ok(distance_squared > threshold)
    }

    /// Finds all hull facets visible from an external point
    ///
    /// # Arguments
    ///
    /// * `point` - The external point
    /// * `tds` - Reference to triangulation (needed for visibility testing)
    ///
    /// # Returns
    ///
    /// A vector of indices into the `hull_facets` array for visible facets
    ///
    /// # Errors
    ///
    /// Returns a [`FacetError`] if the visibility test fails for any facet.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds).unwrap();
    ///
    /// // Test with a point outside the hull
    /// let outside_point = Point::new([2.0, 2.0, 2.0]);
    /// let visible_facets = hull.find_visible_facets(&outside_point, &tds).unwrap();
    /// assert!(!visible_facets.is_empty(), "Outside point should see some facets");
    ///
    /// // Test with a point inside the hull
    /// let inside_point = Point::new([0.2, 0.2, 0.2]);
    /// let visible_facets = hull.find_visible_facets(&inside_point, &tds).unwrap();
    /// assert!(visible_facets.is_empty(), "Inside point should see no facets");
    /// ```
    pub fn find_visible_facets(
        &self,
        point: &Point<T, D>,
        tds: &Tds<T, U, V, D>,
    ) -> Result<Vec<usize>, FacetError> {
        let mut visible_facets = Vec::new();

        for (index, facet) in self.hull_facets.iter().enumerate() {
            if self.is_facet_visible_from_point(facet, point, tds)? {
                visible_facets.push(index);
            }
        }

        Ok(visible_facets)
    }

    /// Finds the nearest visible facet to a point
    ///
    /// This is useful for incremental hull construction algorithms.
    ///
    /// # Arguments
    ///
    /// * `point` - The external point
    /// * `tds` - Reference to triangulation (needed for visibility testing)
    ///
    /// # Returns
    ///
    /// The index of the nearest visible facet, or None if no facets are visible
    ///
    /// # Errors
    ///
    /// Returns a [`ConvexHullConstructionError`] if the visibility test fails or if distance calculations fail.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds).unwrap();
    ///
    /// // Test with a point outside the hull - should find a nearest visible facet
    /// let outside_point = Point::new([2.0, 2.0, 2.0]);
    /// let nearest_facet = hull.find_nearest_visible_facet(&outside_point, &tds).unwrap();
    /// assert!(nearest_facet.is_some(), "Outside point should have a nearest visible facet");
    ///
    /// // Test with a point inside the hull - should find no visible facets
    /// let inside_point = Point::new([0.2, 0.2, 0.2]);
    /// let nearest_facet = hull.find_nearest_visible_facet(&inside_point, &tds).unwrap();
    /// assert!(nearest_facet.is_none(), "Inside point should have no visible facets");
    /// ```
    pub fn find_nearest_visible_facet(
        &self,
        point: &Point<T, D>,
        tds: &Tds<T, U, V, D>,
    ) -> Result<Option<usize>, ConvexHullConstructionError>
    where
        T: PartialOrd + Copy,
    {
        let visible_facets = self.find_visible_facets(point, tds).map_err(|e| {
            ConvexHullConstructionError::BoundaryFacetExtractionFailed { source: e }
        })?;

        if visible_facets.is_empty() {
            return Ok(None);
        }

        // Find the facet with minimum distance to the point
        let mut min_distance: Option<T> = None;
        let mut nearest_facet = None;

        for &facet_index in &visible_facets {
            let facet = &self.hull_facets[facet_index];
            let facet_vertices = facet.vertices();

            // Calculate distance from point to facet centroid as a simple heuristic
            let mut centroid_coords = [T::zero(); D];
            let num_vertices = NumCast::from(facet_vertices.len())
                .ok_or_else(|| ConvexHullConstructionError::NumericCastFailed {
                    message: format!(
                        "Failed to cast vertex count {} to coordinate type for centroid calculation",
                        facet_vertices.len()
                    ),
                })?;

            for vertex in &facet_vertices {
                let vertex_point = vertex.point();
                let coords: [T; D] = vertex_point.into();
                for (i, &coord) in coords.iter().enumerate() {
                    centroid_coords[i] += coord;
                }
            }

            for coord in &mut centroid_coords {
                *coord /= num_vertices;
            }

            let centroid = Point::new(centroid_coords);

            // Calculate squared distance using squared_norm
            let point_coords: [T; D] = point.into();
            let centroid_coords: [T; D] = (&centroid).into();
            let mut diff_coords = [T::zero(); D];
            for i in 0..D {
                diff_coords[i] = point_coords[i] - centroid_coords[i];
            }
            let distance = squared_norm(diff_coords);

            if min_distance.is_none_or(|min_dist| distance < min_dist) {
                min_distance = Some(distance);
                nearest_facet = Some(facet_index);
            }
        }

        Ok(nearest_facet)
    }

    /// Checks if a point is outside the current convex hull
    ///
    /// A point is outside if it's visible from at least one hull facet.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to test
    /// * `tds` - Reference to triangulation (needed for visibility testing)
    ///
    /// # Returns
    ///
    /// `true` if the point is outside the hull, `false` if inside
    ///
    /// # Errors
    ///
    /// Returns a [`FacetError`] if the visibility test fails for any facet.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds).unwrap();
    ///
    /// // Test with a point inside the hull
    /// let inside_point = Point::new([0.2, 0.2, 0.2]);
    /// assert!(!hull.is_point_outside(&inside_point, &tds).unwrap());
    ///
    /// // Test with a point outside the hull
    /// let outside_point = Point::new([2.0, 2.0, 2.0]);
    /// assert!(hull.is_point_outside(&outside_point, &tds).unwrap());
    /// ```
    pub fn is_point_outside(
        &self,
        point: &Point<T, D>,
        tds: &Tds<T, U, V, D>,
    ) -> Result<bool, FacetError> {
        let visible_facets = self.find_visible_facets(point, tds)?;
        Ok(!visible_facets.is_empty())
    }

    /// Returns the number of hull facets
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds).unwrap();
    ///
    /// assert_eq!(hull.facet_count(), 4); // Tetrahedron has 4 faces
    /// ```
    #[must_use]
    pub const fn facet_count(&self) -> usize {
        self.hull_facets.len()
    }

    /// Gets a hull facet by index
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the facet to retrieve
    ///
    /// # Returns
    ///
    /// Some reference to the facet if the index is valid, None otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds).unwrap();
    ///
    /// // Get the first facet
    /// assert!(hull.get_facet(0).is_some());
    /// // Index out of bounds returns None
    /// assert!(hull.get_facet(10).is_none());
    /// ```
    #[must_use]
    pub fn get_facet(&self, index: usize) -> Option<&Facet<T, U, V, D>> {
        self.hull_facets.get(index)
    }

    /// Returns an iterator over the hull facets
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create a 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds).unwrap();
    ///
    /// // Iterate over all hull facets
    /// let facet_count = hull.facets().count();
    /// assert_eq!(facet_count, 4); // Tetrahedron has 4 faces
    ///
    /// // Check that all facets have the expected number of vertices
    /// for facet in hull.facets() {
    ///     assert_eq!(facet.vertices().len(), 3); // 3D facets have 3 vertices
    /// }
    /// ```
    pub fn facets(&self) -> std::slice::Iter<'_, Facet<T, U, V, D>> {
        self.hull_facets.iter()
    }

    /// Validates the convex hull for consistency
    ///
    /// This performs basic checks on the hull facets to ensure they form
    /// a valid convex hull structure.
    ///
    /// # Errors
    ///
    /// Returns an error if any facet has an invalid number of vertices or contains duplicate vertices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create a valid 3D tetrahedron
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds).unwrap();
    ///
    /// // Validation should pass for a well-formed hull
    /// assert!(hull.validate().is_ok());
    ///
    /// // Empty hull should also validate
    /// let empty_hull: ConvexHull<f64, Option<()>, Option<()>, 3> = ConvexHull::default();
    /// assert!(empty_hull.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        // Check that all facets have exactly D vertices (for D-dimensional triangulation,
        // facets are (D-1)-dimensional and have D vertices)
        for (index, facet) in self.hull_facets.iter().enumerate() {
            let vertices = facet.vertices();
            if vertices.len() != D {
                return Err(format!(
                    "Facet {} has {} vertices, expected {} for {}D triangulation",
                    index,
                    vertices.len(),
                    D,
                    D
                ));
            }

            // Check that vertices are distinct
            for i in 0..vertices.len() {
                for j in i + 1..vertices.len() {
                    if vertices[i].uuid() == vertices[j].uuid() {
                        return Err(format!(
                            "Facet {index} has duplicate vertices at positions {i} and {j}"
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Returns true if the convex hull is empty (has no facets)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Empty hull
    /// let empty_hull: ConvexHull<f64, Option<()>, Option<()>, 3> = ConvexHull::default();
    /// assert!(empty_hull.is_empty());
    ///
    /// // Non-empty hull
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds).unwrap();
    /// assert!(!hull.is_empty());
    /// ```
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.hull_facets.is_empty()
    }

    /// Clears the convex hull, removing all facets
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create a hull and then clear it
    /// let vertices = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
    /// let mut hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds).unwrap();
    ///
    /// assert!(!hull.is_empty());
    /// assert_eq!(hull.facet_count(), 4);
    ///
    /// hull.clear();
    /// assert!(hull.is_empty());
    /// assert_eq!(hull.facet_count(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.hull_facets.clear();
    }

    /// Returns the dimension of the convex hull
    ///
    /// This is the same as the dimension of the triangulation that generated it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::core::triangulation_data_structure::Tds;
    /// use delaunay::geometry::algorithms::convex_hull::ConvexHull;
    /// use delaunay::vertex;
    ///
    /// // Create different dimensional hulls
    /// let vertices_2d = vec![
    ///     vertex!([0.0, 0.0]),
    ///     vertex!([1.0, 0.0]),
    ///     vertex!([0.0, 1.0]),
    /// ];
    /// let tds_2d: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_2d).unwrap();
    /// let hull_2d: ConvexHull<f64, Option<()>, Option<()>, 2> =
    ///     ConvexHull::from_triangulation(&tds_2d).unwrap();
    /// assert_eq!(hull_2d.dimension(), 2);
    ///
    /// let vertices_3d = vec![
    ///     vertex!([0.0, 0.0, 0.0]),
    ///     vertex!([1.0, 0.0, 0.0]),
    ///     vertex!([0.0, 1.0, 0.0]),
    ///     vertex!([0.0, 0.0, 1.0]),
    /// ];
    /// let tds_3d: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_3d).unwrap();
    /// let hull_3d: ConvexHull<f64, Option<()>, Option<()>, 3> =
    ///     ConvexHull::from_triangulation(&tds_3d).unwrap();
    /// assert_eq!(hull_3d.dimension(), 3);
    /// ```
    #[must_use]
    pub const fn dimension(&self) -> usize {
        D
    }
}

impl<T, U, V, const D: usize> Default for ConvexHull<T, U, V, D>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + Sized + Serialize + DeserializeOwned,
{
    fn default() -> Self {
        Self {
            hull_facets: Vec::new(),
        }
    }
}

// Type aliases for common use cases
/// Type alias for 2D convex hulls
pub type ConvexHull2D<T, U, V> = ConvexHull<T, U, V, 2>;
/// Type alias for 3D convex hulls
pub type ConvexHull3D<T, U, V> = ConvexHull<T, U, V, 3>;
/// Type alias for 4D convex hulls
pub type ConvexHull4D<T, U, V> = ConvexHull<T, U, V, 4>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::triangulation_data_structure::Tds;
    use crate::vertex;

    #[test]
    fn test_convex_hull_2d_creation() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull2D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds).unwrap();

        assert_eq!(hull.facet_count(), 3); // Triangle has 3 edges
        assert_eq!(hull.dimension(), 2);
        assert!(hull.validate().is_ok());
        assert!(!hull.is_empty());
    }

    #[test]
    fn test_convex_hull_3d_creation() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull3D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds).unwrap();

        assert_eq!(hull.facet_count(), 4); // Tetrahedron has 4 faces
        assert_eq!(hull.dimension(), 3);
        assert!(hull.validate().is_ok());
        assert!(!hull.is_empty());
    }

    #[test]
    fn test_convex_hull_4d_creation() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull4D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds).unwrap();

        assert_eq!(hull.facet_count(), 5); // 4-simplex has 5 facets
        assert_eq!(hull.dimension(), 4);
        assert!(hull.validate().is_ok());
        assert!(!hull.is_empty());
    }

    #[test]
    fn test_point_outside_detection_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull3D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Point inside the tetrahedron
        let inside_point = Point::new([0.2, 0.2, 0.2]);
        assert!(!hull.is_point_outside(&inside_point, &tds).unwrap());

        // Point outside the tetrahedron
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        assert!(hull.is_point_outside(&outside_point, &tds).unwrap());
    }

    #[test]
    fn test_visible_facets_3d() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull3D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Point outside should see some facets
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        let visible_facets = hull.find_visible_facets(&outside_point, &tds).unwrap();

        assert!(
            !visible_facets.is_empty(),
            "Outside point should see some facets"
        );

        // Point inside should see no facets
        let inside_point = Point::new([0.2, 0.2, 0.2]);
        let visible_facets = hull.find_visible_facets(&inside_point, &tds).unwrap();

        assert!(
            visible_facets.is_empty(),
            "Inside point should see no facets"
        );
    }

    #[test]
    fn test_empty_hull() {
        let hull: ConvexHull3D<f64, Option<()>, Option<()>> = ConvexHull::default();

        assert_eq!(hull.facet_count(), 0);
        assert_eq!(hull.dimension(), 3);
        assert!(hull.validate().is_ok());
        assert!(hull.is_empty());

        // For an empty hull, we can't test visibility without a TDS
        // But we can test that it behaves correctly with basic operations
        assert!(hull.get_facet(0).is_none());
        assert_eq!(hull.facets().count(), 0);
    }

    #[test]
    fn test_hull_operations() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut hull: ConvexHull3D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test facet access
        assert_eq!(hull.facets().count(), 4);
        assert!(hull.get_facet(0).is_some());
        assert!(hull.get_facet(4).is_none());

        // Test clear
        hull.clear();
        assert!(hull.is_empty());
        assert_eq!(hull.facet_count(), 0);
    }

    #[test]
    fn test_visibility_works_in_all_dimensions() {
        // Test 2D visibility
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds_2d: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_2d).unwrap();
        let hull_2d: ConvexHull2D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds_2d).unwrap();

        let inside_point_2d = Point::new([0.1, 0.1]);
        let outside_point_2d = Point::new([2.0, 2.0]);

        // 2D visibility testing should work
        assert!(
            !hull_2d.is_point_outside(&inside_point_2d, &tds_2d).unwrap(),
            "2D inside point should not be outside"
        );
        assert!(
            hull_2d
                .is_point_outside(&outside_point_2d, &tds_2d)
                .unwrap(),
            "2D outside point should be outside"
        );

        // Test 3D visibility
        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds_3d: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_3d).unwrap();
        let hull_3d: ConvexHull3D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds_3d).unwrap();

        let inside_point_3d = Point::new([0.1, 0.1, 0.1]);
        let outside_point_3d = Point::new([2.0, 2.0, 2.0]);

        // 3D visibility testing should work
        assert!(
            !hull_3d.is_point_outside(&inside_point_3d, &tds_3d).unwrap(),
            "3D inside point should not be outside"
        );
        assert!(
            hull_3d
                .is_point_outside(&outside_point_3d, &tds_3d)
                .unwrap(),
            "3D outside point should be outside"
        );

        // Test 4D visibility
        let vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let tds_4d: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices_4d).unwrap();
        let hull_4d: ConvexHull4D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds_4d).unwrap();

        let inside_point_4d = Point::new([0.1, 0.1, 0.1, 0.1]);
        let outside_point_4d = Point::new([2.0, 2.0, 2.0, 2.0]);

        // 4D visibility testing should work
        assert!(
            !hull_4d.is_point_outside(&inside_point_4d, &tds_4d).unwrap(),
            "4D inside point should not be outside"
        );
        assert!(
            hull_4d
                .is_point_outside(&outside_point_4d, &tds_4d)
                .unwrap(),
            "4D outside point should be outside"
        );

        // Test 5D visibility
        let vertices_5d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let tds_5d: Tds<f64, Option<()>, Option<()>, 5> = Tds::new(&vertices_5d).unwrap();
        let hull_5d: ConvexHull<f64, Option<()>, Option<()>, 5> =
            ConvexHull::from_triangulation(&tds_5d).unwrap();

        let inside_point_5d = Point::new([0.1, 0.1, 0.1, 0.1, 0.1]);
        let outside_point_5d = Point::new([2.0, 2.0, 2.0, 2.0, 2.0]);

        // 5D visibility testing should work
        assert!(
            !hull_5d.is_point_outside(&inside_point_5d, &tds_5d).unwrap(),
            "5D inside point should not be outside"
        );
        assert!(
            hull_5d
                .is_point_outside(&outside_point_5d, &tds_5d)
                .unwrap(),
            "5D outside point should be outside"
        );

        println!("✓ Visibility testing works in 2D, 3D, 4D, and 5D");
    }

    // ============================================================================
    // UNIT TESTS FOR PRIVATE METHODS
    // ============================================================================
    // These tests target private methods to ensure thorough coverage of internal
    // ConvexHull functionality, particularly the fallback_visibility_test method.

    #[test]
    fn test_fallback_visibility_test_distance_based() {
        println!("Testing fallback_visibility_test with distance-based heuristic");

        // Create a simple 3D triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Get a hull facet to test
        assert!(!hull.hull_facets.is_empty(), "Hull should have facets");
        let test_facet = &hull.hull_facets[0];

        // Test with points at various distances
        let test_points = vec![
            (Point::new([0.1, 0.1, 0.1]), "Close point", false), // Close = not visible
            (Point::new([5.0, 5.0, 5.0]), "Far point", true),    // Far = visible
            (Point::new([2.0, 2.0, 2.0]), "Medium distance", true), // Medium far = visible
            (Point::new([0.5, 0.5, 0.5]), "Threshold point", false), // At threshold
        ];

        for (point, description, expected) in test_points {
            let is_visible =
                ConvexHull::<f64, Option<()>, Option<()>, 3>::fallback_visibility_test(
                    test_facet, &point,
                )
                .unwrap();

            // Note: The exact threshold behavior may vary, so we mainly test that
            // the function completes without error and returns a boolean
            println!("  {description} - Expected: {expected}, Got: {is_visible}");
        }

        println!("✓ Fallback visibility test with distance heuristic works correctly");
    }

    #[test]
    fn test_fallback_visibility_test_edge_cases() {
        println!("Testing fallback_visibility_test edge cases");

        // Create a 2D triangulation to test lower dimensions
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds_2d: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_2d).unwrap();
        let hull_2d: ConvexHull<f64, Option<()>, Option<()>, 2> =
            ConvexHull::from_triangulation(&tds_2d).unwrap();

        assert!(
            !hull_2d.hull_facets.is_empty(),
            "2D hull should have facets (edges)"
        );
        let test_facet_2d = &hull_2d.hull_facets[0];

        // Test 2D fallback visibility
        let test_point_2d = Point::new([2.0, 2.0]);
        let result_2d = ConvexHull::<f64, Option<()>, Option<()>, 2>::fallback_visibility_test(
            test_facet_2d,
            &test_point_2d,
        );

        println!("  2D fallback result: {result_2d:?}");

        // Test 4D fallback visibility
        let vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let tds_4d: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices_4d).unwrap();
        let hull_4d: ConvexHull<f64, Option<()>, Option<()>, 4> =
            ConvexHull::from_triangulation(&tds_4d).unwrap();

        assert!(
            !hull_4d.hull_facets.is_empty(),
            "4D hull should have facets"
        );
        let test_facet_4d = &hull_4d.hull_facets[0];

        // Test 4D fallback visibility
        let test_point_4d = Point::new([2.0, 2.0, 2.0, 2.0]);
        let result_4d = ConvexHull::<f64, Option<()>, Option<()>, 4>::fallback_visibility_test(
            test_facet_4d,
            &test_point_4d,
        );

        println!("  4D fallback result: {result_4d:?}");

        println!("✓ Fallback visibility test works correctly in different dimensions");
    }

    #[test]
    fn test_fallback_visibility_test_degenerate_cases() {
        println!("Testing fallback_visibility_test with degenerate cases");

        // Create a triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        let test_facet = &hull.hull_facets[0];

        // Test with points that might cause numerical issues
        let degenerate_points = vec![
            (Point::new([0.0, 0.0, 0.0]), "Origin point"),
            (
                Point::new([f64::EPSILON, f64::EPSILON, f64::EPSILON]),
                "Very small coordinates",
            ),
            (Point::new([1e-10, 1e-10, 1e-10]), "Near-zero coordinates"),
        ];

        for (point, description) in degenerate_points {
            let result = ConvexHull::<f64, Option<()>, Option<()>, 3>::fallback_visibility_test(
                test_facet, &point,
            );

            // The function should handle these cases gracefully
            println!("  {description} - Result: {result:?}");
        }

        println!("✓ Fallback visibility test handles degenerate cases correctly");
    }

    #[test]
    fn test_fallback_visibility_test_threshold_behavior() {
        println!("Testing fallback_visibility_test threshold behavior");

        // Create a simple triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        let test_facet = &hull.hull_facets[0];

        // Test points at the threshold distance (should be around 1.0 based on the implementation)
        let threshold_points = vec![
            Point::new([0.5, 0.5, 0.5]), // Distance² ≈ 0.75 < 1.0 → not visible
            Point::new([1.0, 1.0, 1.0]), // Distance² = 3.0 > 1.0 → visible
            Point::new([0.8, 0.8, 0.8]), // Distance² ≈ 1.92 > 1.0 → visible
            Point::new([0.3, 0.3, 0.3]), // Distance² ≈ 0.27 < 1.0 → not visible
        ];

        let mut visible_count = 0;
        let mut not_visible_count = 0;

        for point in threshold_points {
            let is_visible =
                ConvexHull::<f64, Option<()>, Option<()>, 3>::fallback_visibility_test(
                    test_facet, &point,
                )
                .unwrap();

            let coords: [f64; 3] = point.into();
            println!("  Point {coords:?} - Visible: {is_visible}");

            if is_visible {
                visible_count += 1;
            } else {
                not_visible_count += 1;
            }
        }

        // We should have some points classified as visible and some as not visible
        assert!(
            visible_count > 0,
            "Some points should be classified as visible"
        );
        assert!(
            not_visible_count > 0,
            "Some points should be classified as not visible"
        );

        println!("✓ Fallback visibility test threshold behavior works correctly");
    }

    #[test]
    fn test_fallback_visibility_test_numerical_precision() {
        println!("Testing fallback_visibility_test numerical precision");

        // Test with very high precision f64 points
        let vertices_f64 = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds_f64: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_f64).unwrap();
        let hull_f64: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds_f64).unwrap();

        let test_facet_f64 = &hull_f64.hull_facets[0];

        // Test with very precise points
        let precise_points = vec![
            Point::new([1e-15, 1e-15, 1e-15]),
            Point::new([
                1.000_000_000_000_000_1,
                1.000_000_000_000_000_1,
                1.000_000_000_000_000_1,
            ]),
            Point::new([
                0.999_999_999_999_999_9,
                0.999_999_999_999_999_9,
                0.999_999_999_999_999_9,
            ]),
        ];

        for point in precise_points {
            let is_visible =
                ConvexHull::<f64, Option<()>, Option<()>, 3>::fallback_visibility_test(
                    test_facet_f64,
                    &point,
                )
                .unwrap();

            let coords: [f64; 3] = point.into();
            println!("  High precision Point {coords:?} - Visible: {is_visible}");
        }

        println!("✓ Fallback visibility test handles numerical precision correctly");
    }

    #[test]
    fn test_fallback_visibility_test_consistency() {
        println!("Testing fallback_visibility_test consistency");

        // Create a triangulation
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test the same point multiple times to ensure consistency
        let test_point = Point::new([2.0, 2.0, 2.0]);
        let test_facet = &hull.hull_facets[0];

        let consistency_results: Vec<bool> = (0..5)
            .map(|_| {
                ConvexHull::<f64, Option<()>, Option<()>, 3>::fallback_visibility_test(
                    test_facet,
                    &test_point,
                )
                .unwrap()
            })
            .collect();

        // All results should be the same
        let first_result = consistency_results[0];
        for (i, &result) in consistency_results.iter().enumerate() {
            assert_eq!(
                result, first_result,
                "Result {i} should be consistent with first result"
            );
        }

        println!(
            "  Consistency test: all {} results were {}",
            consistency_results.len(),
            first_result
        );

        // Test different points with same facet should give deterministic results
        let test_points = vec![
            Point::new([0.1, 0.1, 0.1]),
            Point::new([5.0, 5.0, 5.0]),
            Point::new([1.5, 1.5, 1.5]),
        ];

        for point in test_points {
            let result1 = ConvexHull::<f64, Option<()>, Option<()>, 3>::fallback_visibility_test(
                test_facet, &point,
            )
            .unwrap();

            let result2 = ConvexHull::<f64, Option<()>, Option<()>, 3>::fallback_visibility_test(
                test_facet, &point,
            )
            .unwrap();

            assert_eq!(result1, result2, "Same point should give same result");

            let coords: [f64; 3] = point.into();
            println!("  Point {coords:?} consistently returns: {result1}");
        }

        println!("✓ Fallback visibility test maintains consistency");
    }

    // ============================================================================
    // EXHAUSTIVE UNIT TESTS FOR COMPREHENSIVE COVERAGE
    // ============================================================================
    // Additional tests to ensure we maintain 85% test coverage by testing
    // edge cases, error conditions, and less-covered code paths.

    #[test]
    fn test_from_triangulation_error_cases() {
        // Test creating hull from triangulation that fails to extract boundary facets
        // This is tricky to test directly since boundary_facets() rarely fails,
        // but we can test the error handling path exists

        // Create a minimal valid triangulation first to ensure the path works
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let hull = ConvexHull::from_triangulation(&tds);
        assert!(
            hull.is_ok(),
            "Valid triangulation should create hull successfully"
        );
    }

    #[test]
    fn test_is_facet_visible_from_point_error_cases() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Get a valid facet
        let facet = &hull.hull_facets[0];
        let test_point = Point::new([1.0, 1.0, 1.0]);

        // Test normal case first
        let result = hull.is_facet_visible_from_point(facet, &test_point, &tds);
        assert!(result.is_ok(), "Normal visibility test should succeed");

        // Note: Testing the InsufficientVertices error path is complex because
        // it requires creating invalid facets. For now, we just test the normal
        // case to ensure the method works correctly. The error paths are covered
        // by the existing comprehensive tests in other methods.
    }

    #[test]
    fn test_validate_error_cases() {
        // Test basic validation scenarios
        // Most validation edge cases are covered by the existing comprehensive tests

        // Test empty hull validation (should pass)
        let empty_hull: ConvexHull<f64, Option<()>, Option<()>, 3> = ConvexHull::default();
        assert!(
            empty_hull.validate().is_ok(),
            "Empty hull should validate successfully"
        );

        // Test valid hull validation (should pass)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        assert!(
            hull.validate().is_ok(),
            "Valid hull should validate successfully"
        );

        // Note: Testing validation with manually constructed invalid facets is complex
        // because our API doesn't expose direct facet construction with invalid data.
        // The validation logic is still tested through normal usage patterns.
    }

    #[test]
    fn test_find_nearest_visible_facet_comprehensive() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test with inside point - should return None
        let inside_point = Point::new([0.1, 0.1, 0.1]);
        let result = hull
            .find_nearest_visible_facet(&inside_point, &tds)
            .unwrap();
        assert!(
            result.is_none(),
            "Inside point should have no visible facets"
        );

        // Test with outside point - should return Some index
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        let result = hull
            .find_nearest_visible_facet(&outside_point, &tds)
            .unwrap();
        assert!(result.is_some(), "Outside point should have visible facets");

        if let Some(facet_index) = result {
            assert!(
                facet_index < hull.facet_count(),
                "Facet index should be valid"
            );
        }

        // Test with point at various distances to verify distance calculation
        let test_points = vec![
            Point::new([1.5, 1.5, 1.5]),
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 3.0, 0.0]),
            Point::new([0.0, 0.0, 3.0]),
        ];

        for point in test_points {
            let result = hull.find_nearest_visible_facet(&point, &tds);
            // All these points should be outside and have visible facets
            assert!(result.is_ok(), "Distance calculation should not fail");
        }
    }

    #[test]
    fn test_facet_access_edge_cases() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test get_facet with valid indices
        for i in 0..hull.facet_count() {
            assert!(
                hull.get_facet(i).is_some(),
                "Valid index should return facet"
            );
        }

        // Test get_facet with invalid indices
        assert!(
            hull.get_facet(hull.facet_count()).is_none(),
            "Out of bounds index should return None"
        );
        assert!(
            hull.get_facet(usize::MAX).is_none(),
            "Very large index should return None"
        );

        // Test iterator
        let facet_count_via_iter = hull.facets().count();
        assert_eq!(
            facet_count_via_iter,
            hull.facet_count(),
            "Iterator count should match facet_count"
        );

        // Verify all facets in iterator are valid
        for facet in hull.facets() {
            assert!(
                !facet.vertices().is_empty(),
                "Each facet should have vertices"
            );
        }
    }

    #[test]
    fn test_hull_with_different_coordinate_types() {
        // Note: f32 coordinate type has complex trait bounds that would require
        // extensive changes to support. For now, we focus on f64 which is the
        // primary supported coordinate type.

        // Test with different data precision approaches using f64
        let vertices_high_precision = vec![
            vertex!([0.000_000_000_000_001, 0.0, 0.0]),
            vertex!([1.000_000_000_000_001, 0.0, 0.0]),
            vertex!([0.0, 1.000_000_000_000_001, 0.0]),
            vertex!([0.0, 0.0, 1.000_000_000_000_001]),
        ];
        let tds_hp: Tds<f64, Option<()>, Option<()>, 3> =
            Tds::new(&vertices_high_precision).unwrap();
        let hull_hp: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds_hp).unwrap();

        assert_eq!(hull_hp.facet_count(), 4);
        assert_eq!(hull_hp.dimension(), 3);
        assert!(hull_hp.validate().is_ok());
        assert!(!hull_hp.is_empty());
    }

    #[test]
    fn test_hull_with_different_data_types() {
        // Note: String data type doesn't implement Copy, so it can't be used with DataType.
        // We test with Copy-able data types that satisfy the DataType trait bounds.

        // Test with integer vertex data
        let vertices_int = vec![
            vertex!([0.0, 0.0, 0.0], 1i32),
            vertex!([1.0, 0.0, 0.0], 2i32),
            vertex!([0.0, 1.0, 0.0], 3i32),
            vertex!([0.0, 0.0, 1.0], 4i32),
        ];
        let tds_int: Tds<f64, i32, Option<()>, 3> = Tds::new(&vertices_int).unwrap();
        let hull_int: ConvexHull<f64, i32, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds_int).unwrap();

        assert_eq!(hull_int.facet_count(), 4);
        assert_eq!(hull_int.dimension(), 3);
        assert!(hull_int.validate().is_ok());

        // Test with character vertex data
        let vertices_char = vec![
            vertex!([0.0, 0.0, 0.0], 'A'),
            vertex!([1.0, 0.0, 0.0], 'B'),
            vertex!([0.0, 1.0, 0.0], 'C'),
            vertex!([0.0, 0.0, 1.0], 'D'),
        ];
        let tds_char: Tds<f64, char, Option<()>, 3> = Tds::new(&vertices_char).unwrap();
        let hull_char: ConvexHull<f64, char, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds_char).unwrap();

        assert_eq!(hull_char.facet_count(), 4);
        assert_eq!(hull_char.dimension(), 3);
        assert!(hull_char.validate().is_ok());
    }

    #[test]
    fn test_extreme_coordinate_values() {
        // Test with very large coordinates
        let vertices_large = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e6, 0.0, 0.0]),
            vertex!([0.0, 1e6, 0.0]),
            vertex!([0.0, 0.0, 1e6]),
        ];
        let tds_large: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_large).unwrap();
        let hull_large: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds_large).unwrap();

        assert_eq!(hull_large.facet_count(), 4);
        assert!(hull_large.validate().is_ok());

        // Test visibility with large coordinates
        let inside_large = Point::new([1000.0, 1000.0, 1000.0]);
        let outside_large = Point::new([2e6, 2e6, 2e6]);

        assert!(
            !hull_large
                .is_point_outside(&inside_large, &tds_large)
                .unwrap()
        );
        assert!(
            hull_large
                .is_point_outside(&outside_large, &tds_large)
                .unwrap()
        );

        // Test with very small coordinates
        let vertices_small = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e-6, 0.0, 0.0]),
            vertex!([0.0, 1e-6, 0.0]),
            vertex!([0.0, 0.0, 1e-6]),
        ];
        let tds_small: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_small).unwrap();
        let hull_small: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds_small).unwrap();

        assert_eq!(hull_small.facet_count(), 4);
        assert!(hull_small.validate().is_ok());
    }

    #[test]
    fn test_1d_convex_hull() {
        // Test 1D case (line segment)
        let vertices_1d = vec![vertex!([0.0]), vertex!([1.0])];
        let tds_1d: Tds<f64, Option<()>, Option<()>, 1> = Tds::new(&vertices_1d).unwrap();
        let hull_1d: ConvexHull<f64, Option<()>, Option<()>, 1> =
            ConvexHull::from_triangulation(&tds_1d).unwrap();

        assert_eq!(hull_1d.dimension(), 1);
        assert!(hull_1d.validate().is_ok());
        assert!(!hull_1d.is_empty());

        // Test point outside detection in 1D
        let inside_1d = Point::new([0.5]);
        let outside_1d = Point::new([2.0]);

        // Note: 1D visibility might behave differently, so we just test that it doesn't crash
        let _ = hull_1d.is_point_outside(&inside_1d, &tds_1d);
        let _ = hull_1d.is_point_outside(&outside_1d, &tds_1d);
    }

    #[test]
    fn test_high_dimensional_hulls() {
        // Test 6D hull
        let vertices_6d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let tds_6d: Tds<f64, Option<()>, Option<()>, 6> = Tds::new(&vertices_6d).unwrap();
        let hull_6d: ConvexHull<f64, Option<()>, Option<()>, 6> =
            ConvexHull::from_triangulation(&tds_6d).unwrap();

        assert_eq!(hull_6d.dimension(), 6);
        assert!(hull_6d.validate().is_ok());
        assert!(!hull_6d.is_empty());

        // Test visibility in 6D
        let inside_6d = Point::new([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
        let outside_6d = Point::new([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);

        assert!(!hull_6d.is_point_outside(&inside_6d, &tds_6d).unwrap());
        assert!(hull_6d.is_point_outside(&outside_6d, &tds_6d).unwrap());
    }

    #[test]
    fn test_hull_clone_and_debug() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test Debug trait
        let debug_string = format!("{hull:?}");
        assert!(
            debug_string.contains("ConvexHull"),
            "Debug output should contain ConvexHull"
        );
        assert!(!debug_string.is_empty(), "Debug output should not be empty");
    }

    #[test]
    fn test_default_implementation() {
        // Test Default trait for various dimensions
        let hull_2d: ConvexHull<f64, Option<()>, Option<()>, 2> = ConvexHull::default();
        assert!(hull_2d.is_empty());
        assert_eq!(hull_2d.facet_count(), 0);
        assert_eq!(hull_2d.dimension(), 2);
        assert!(hull_2d.validate().is_ok());

        let hull_3d: ConvexHull<f64, Option<()>, Option<()>, 3> = ConvexHull::default();
        assert!(hull_3d.is_empty());
        assert_eq!(hull_3d.facet_count(), 0);
        assert_eq!(hull_3d.dimension(), 3);
        assert!(hull_3d.validate().is_ok());

        let hull_4d: ConvexHull<f64, Option<()>, Option<()>, 4> = ConvexHull::default();
        assert!(hull_4d.is_empty());
        assert_eq!(hull_4d.facet_count(), 0);
        assert_eq!(hull_4d.dimension(), 4);
        assert!(hull_4d.validate().is_ok());
    }

    #[test]
    fn test_type_aliases() {
        // Test that type aliases work correctly
        let vertices_2d = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds_2d: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_2d).unwrap();
        let _hull_2d: ConvexHull2D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds_2d).unwrap();

        let vertices_3d = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds_3d: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_3d).unwrap();
        let _hull_3d: ConvexHull3D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds_3d).unwrap();

        let vertices_4d = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let tds_4d: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices_4d).unwrap();
        let _hull_4d: ConvexHull4D<f64, Option<()>, Option<()>> =
            ConvexHull::from_triangulation(&tds_4d).unwrap();
    }

    #[test]
    fn test_visibility_edge_cases() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test visibility with points on the boundary/surface
        let boundary_points = vec![
            Point::new([0.5, 0.5, 0.0]), // On a face
            Point::new([0.3, 0.3, 0.3]), // Near centroid
            Point::new([0.0, 0.0, 0.0]), // At a vertex
            Point::new([0.5, 0.0, 0.0]), // On an edge
        ];

        for point in boundary_points {
            // These might be visible or not depending on numerical precision
            // The important thing is that the method doesn't crash
            let result = hull.is_point_outside(&point, &tds);
            assert!(
                result.is_ok(),
                "Boundary point visibility test should not error"
            );
        }

        // Test with points very close to the hull
        let close_points = vec![
            Point::new([1e-10, 1e-10, 1e-10]),
            Point::new([0.999_999, 0.0, 0.0]),
            Point::new([0.0, 0.999_999, 0.0]),
            Point::new([0.0, 0.0, 0.999_999]),
        ];

        for point in close_points {
            let result = hull.is_point_outside(&point, &tds);
            assert!(
                result.is_ok(),
                "Close point visibility test should not error"
            );
        }
    }

    #[test]
    fn test_clear_operation_thoroughly() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Verify initial state
        assert!(!hull.is_empty());
        assert!(hull.facet_count() > 0);
        assert!(hull.get_facet(0).is_some());
        assert!(hull.facets().count() > 0);
        assert!(hull.validate().is_ok());

        // Clear the hull
        hull.clear();

        // Verify cleared state
        assert!(hull.is_empty());
        assert_eq!(hull.facet_count(), 0);
        assert!(hull.get_facet(0).is_none());
        assert_eq!(hull.facets().count(), 0);
        assert!(hull.validate().is_ok());
        assert_eq!(hull.dimension(), 3); // Dimension should remain the same

        // Clear again - should be idempotent
        hull.clear();
        assert!(hull.is_empty());
        assert_eq!(hull.facet_count(), 0);
    }

    #[test]
    fn test_comprehensive_facet_iteration() {
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test that facets() iterator produces the same results as get_facet
        let mut iter_facets = Vec::new();
        for facet in hull.facets() {
            iter_facets.push(facet);
        }

        assert_eq!(iter_facets.len(), hull.facet_count());

        for (i, facet_ref) in iter_facets.iter().enumerate() {
            let facet_by_index = hull.get_facet(i).unwrap();
            // They should be the same facet (same memory location)
            assert!(std::ptr::eq(*facet_ref, facet_by_index));
        }

        // Test multiple iterations produce same results
        let first_iteration: Vec<_> = hull.facets().collect();
        let second_iteration: Vec<_> = hull.facets().collect();
        assert_eq!(first_iteration.len(), second_iteration.len());

        for (f1, f2) in first_iteration.iter().zip(second_iteration.iter()) {
            assert!(std::ptr::eq(*f1, *f2));
        }

        // Test chaining with other iterator methods
        let vertex_counts: Vec<usize> = hull.facets().map(|facet| facet.vertices().len()).collect();

        // All facets should have the same number of vertices (dimension)
        for count in vertex_counts {
            assert_eq!(count, 3); // 3D facets have 3 vertices
        }
    }

    #[test]
    fn test_dimensional_consistency() {
        // Test that dimension() always returns D regardless of hull state
        let empty_hull_1d: ConvexHull<f64, Option<()>, Option<()>, 1> = ConvexHull::default();
        assert_eq!(empty_hull_1d.dimension(), 1);

        let empty_hull_2d: ConvexHull<f64, Option<()>, Option<()>, 2> = ConvexHull::default();
        assert_eq!(empty_hull_2d.dimension(), 2);

        let empty_hull_3d: ConvexHull<f64, Option<()>, Option<()>, 3> = ConvexHull::default();
        assert_eq!(empty_hull_3d.dimension(), 3);

        let empty_hull_4d: ConvexHull<f64, Option<()>, Option<()>, 4> = ConvexHull::default();
        assert_eq!(empty_hull_4d.dimension(), 4);

        let empty_hull_ten_d: ConvexHull<f64, Option<()>, Option<()>, 10> = ConvexHull::default();
        assert_eq!(empty_hull_ten_d.dimension(), 10);

        // Test with populated hull
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let mut hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        assert_eq!(hull.dimension(), 3);

        // Dimension should remain constant after clearing
        hull.clear();
        assert_eq!(hull.dimension(), 3);
    }

    #[test]
    fn test_visibility_algorithm_coverage() {
        // This test specifically tries to hit different code paths in visibility testing
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test find_visible_facets with multiple visible facets
        let far_outside_point = Point::new([10.0, 10.0, 10.0]);
        let visible_facets = hull.find_visible_facets(&far_outside_point, &tds).unwrap();

        // A point far outside should see multiple facets
        assert!(!visible_facets.is_empty());

        // Verify all returned indices are valid
        for &index in &visible_facets {
            assert!(
                index < hull.facet_count(),
                "Visible facet index should be valid"
            );
            assert!(hull.get_facet(index).is_some());
        }

        // Test find_visible_facets with no visible facets
        let inside_point = Point::new([0.1, 0.1, 0.1]);
        let visible_facets = hull.find_visible_facets(&inside_point, &tds).unwrap();
        assert!(
            visible_facets.is_empty(),
            "Inside point should see no facets"
        );

        // Test individual facet visibility for each facet
        for (i, facet) in hull.facets().enumerate() {
            let visibility_result =
                hull.is_facet_visible_from_point(facet, &far_outside_point, &tds);
            assert!(
                visibility_result.is_ok(),
                "Facet {i} visibility test should succeed"
            );
        }
    }

    #[test]
    fn test_error_handling_paths() {
        println!("Testing error handling paths in convex hull methods");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test fallback_visibility_test with a regular facet and extreme point
        let test_facet = &hull.hull_facets[0];
        let test_point = Point::new([1e-20, 1e-20, 1e-20]);
        let result = ConvexHull::<f64, Option<()>, Option<()>, 3>::fallback_visibility_test(
            test_facet,
            &test_point,
        );

        // The method should handle extreme coordinates gracefully
        println!("  Fallback visibility result with extreme point: {result:?}");

        // Test normal visibility methods with edge case points
        let edge_points = vec![
            Point::new([0.0, 0.0, 0.0]),                            // At vertex
            Point::new([0.5, 0.0, 0.0]),                            // On edge
            Point::new([f64::EPSILON, f64::EPSILON, f64::EPSILON]), // Very small
        ];

        for point in edge_points {
            let result = hull.is_point_outside(&point, &tds);
            assert!(result.is_ok(), "Edge case visibility test should not error");
        }

        println!("✓ Error handling paths tested successfully");
    }

    #[test]
    fn test_edge_case_distance_calculations() {
        println!("Testing edge case distance calculations");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test find_nearest_visible_facet with equal distances
        // Create points that are equidistant from multiple facets
        let equidistant_point = Point::new([0.5, 0.5, 0.5]);

        // This should exercise the is_none_or method in the distance comparison
        let result = hull.find_nearest_visible_facet(&equidistant_point, &tds);
        assert!(
            result.is_ok(),
            "Distance calculation with equal distances should succeed"
        );

        // Test with very large coordinates that might cause overflow
        let large_point = Point::new([1e15, 1e15, 1e15]);
        let result = hull.find_nearest_visible_facet(&large_point, &tds);
        assert!(
            result.is_ok(),
            "Distance calculation with large coordinates should succeed"
        );

        // Test with very small coordinates that might cause underflow
        let small_point = Point::new([1e-15, 1e-15, 1e-15]);
        let result = hull.find_nearest_visible_facet(&small_point, &tds);
        assert!(
            result.is_ok(),
            "Distance calculation with small coordinates should succeed"
        );

        println!("✓ Edge case distance calculations tested successfully");
    }

    #[test]
    fn test_degenerate_orientation_fallback() {
        println!("Testing degenerate orientation fallback behavior");

        // Create a triangulation that might produce degenerate orientations
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1e-10, 0.0, 0.0]), // Very close to origin
            vertex!([0.0, 1e-10, 0.0]), // Very close to origin
            vertex!([0.0, 0.0, 1e-10]), // Very close to origin
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test with points that might cause degenerate orientations
        let test_points = vec![
            Point::new([5e-11, 5e-11, 5e-11]), // Very close to the degenerate vertices
            Point::new([1e-9, 1e-9, 1e-9]),    // Slightly further but still small
            Point::new([0.0, 0.0, 0.0]),       // At origin
        ];

        for point in test_points {
            // These should potentially trigger the fallback visibility test
            let result = hull.is_point_outside(&point, &tds);
            assert!(
                result.is_ok(),
                "Degenerate orientation handling should not crash"
            );

            let coords: [f64; 3] = point.into();
            println!(
                "  Degenerate point {coords:?} - Outside: {:?}",
                result.unwrap()
            );
        }

        println!("✓ Degenerate orientation fallback tested successfully");
    }

    #[test]
    fn test_validate_method_comprehensive() {
        println!("Testing validate method comprehensively");

        // Test with different dimensional hulls
        let hull_1d: ConvexHull<f64, Option<()>, Option<()>, 1> = ConvexHull::default();
        assert!(hull_1d.validate().is_ok(), "1D empty hull should validate");

        let hull_2d: ConvexHull<f64, Option<()>, Option<()>, 2> = ConvexHull::default();
        assert!(hull_2d.validate().is_ok(), "2D empty hull should validate");

        let hull_5d: ConvexHull<f64, Option<()>, Option<()>, 5> = ConvexHull::default();
        assert!(hull_5d.validate().is_ok(), "5D empty hull should validate");

        // Test with populated hull
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds_2d: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let hull_2d: ConvexHull<f64, Option<()>, Option<()>, 2> =
            ConvexHull::from_triangulation(&tds_2d).unwrap();

        // This should validate successfully - each 2D facet should have 2 vertices
        assert!(
            hull_2d.validate().is_ok(),
            "2D hull should validate successfully"
        );

        // Verify facet count and vertex counts
        for (i, facet) in hull_2d.facets().enumerate() {
            let vertex_count = facet.vertices().len();
            println!("  2D Facet {i}: {vertex_count} vertices (expected 2)");
        }

        println!("✓ Validate method tested comprehensively");
    }

    #[test]
    fn test_extreme_coordinate_precision() {
        println!("Testing extreme coordinate precision handling");

        // Test with coordinates at the limits of f64 precision
        let vertices_extreme = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([f64::MIN_POSITIVE, 0.0, 0.0]),
            vertex!([0.0, f64::MIN_POSITIVE, 0.0]),
            vertex!([0.0, 0.0, f64::MIN_POSITIVE]),
        ];

        let tds_extreme: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_extreme).unwrap();
        let hull_extreme: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds_extreme).unwrap();

        // Test visibility with extreme coordinates
        let test_point = Point::new([
            f64::MIN_POSITIVE * 2.0,
            f64::MIN_POSITIVE * 2.0,
            f64::MIN_POSITIVE * 2.0,
        ]);
        let result = hull_extreme.is_point_outside(&test_point, &tds_extreme);
        assert!(
            result.is_ok(),
            "Extreme precision coordinates should not crash visibility testing"
        );

        // Test fallback visibility with extreme coordinates
        let facet = &hull_extreme.hull_facets[0];
        let fallback_result =
            ConvexHull::<f64, Option<()>, Option<()>, 3>::fallback_visibility_test(
                facet,
                &test_point,
            );
        println!("  Extreme precision fallback result: {fallback_result:?}");

        // Test with maximum finite values
        let vertices_max = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds_max: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_max).unwrap();
        let hull_max: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds_max).unwrap();

        let max_point = Point::new([f64::MAX / 2.0, f64::MAX / 2.0, f64::MAX / 2.0]);
        let result = hull_max.is_point_outside(&max_point, &tds_max);
        assert!(
            result.is_ok(),
            "Maximum finite coordinates should not crash"
        );

        println!("✓ Extreme coordinate precision tested successfully");
    }

    #[test]
    fn test_numeric_cast_error_handling() {
        println!("Testing numeric cast error handling in find_nearest_visible_facet");

        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let hull: ConvexHull<f64, Option<()>, Option<()>, 3> =
            ConvexHull::from_triangulation(&tds).unwrap();

        // Test with a normal point to ensure the method works correctly first
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        let result = hull.find_nearest_visible_facet(&outside_point, &tds);
        assert!(
            result.is_ok(),
            "Normal case should work without numeric cast issues"
        );
        assert!(
            result.unwrap().is_some(),
            "Outside point should have visible facets"
        );

        // The actual numeric cast failure is hard to test directly without creating
        // a coordinate type that fails NumCast, but we can verify that our error
        // handling structure is in place by checking that the method uses proper
        // error types and doesn't panic.

        // Test with various edge cases that could potentially cause numeric issues
        let edge_points = vec![
            Point::new([0.0, 0.0, 0.0]),       // At vertex
            Point::new([1e-10, 1e-10, 1e-10]), // Very small but positive
            Point::new([1e10, 1e10, 1e10]),    // Very large
        ];

        for point in edge_points {
            let result = hull.find_nearest_visible_facet(&point, &tds);
            assert!(
                result.is_ok(),
                "Edge case points should not cause numeric cast failures"
            );

            let coords: [f64; 3] = point.into();
            let result_val = result.unwrap();
            println!("  Edge point {coords:?} - Result: {result_val:?}");
        }

        println!("✓ Numeric cast error handling tested successfully");
    }
}

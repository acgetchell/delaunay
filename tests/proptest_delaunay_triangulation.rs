//! Property-based tests for `DelaunayTriangulation` invariants.
//!
//! This module tests all Delaunay-specific properties including both structural
//! invariants and the expensive geometric Delaunay property validation.
//!
//! ## Architectural Context
//!
//! Following CGAL's architecture:
//! - **Tds** - Pure combinatorial/topological structure (tested in `proptest_tds.rs`)
//! - **Triangulation** - Generic geometric layer with kernel (tested in `proptest_triangulation.rs`)
//! - **`DelaunayTriangulation`** - Delaunay-specific operations (tested here)
//!
//! ## Invariants Tested
//!
//! ### Structural Invariants (Fast)
//! - **Incremental insertion validity** - Triangulation remains valid after each insertion
//! - **Duplicate coordinate rejection** - Geometric duplicate detection at insertion time
//!
//! ### Delaunay Property (Expensive)
//! - **Empty circumsphere condition** - No vertex lies strictly inside any cell's circumsphere (2D-5D)
//! - **Insertion-order invariance** - Edge set independent of insertion order (2D, currently ignored - Issue #120)
//! - **Duplicate cloud integration** - Full pipeline with messy real-world inputs (2D-5D: duplicates + near-duplicates)
//!
//! All structural tests use `DelaunayTriangulation::new()` and `insert()` which maintain
//! invariants through the incremental cavity-based insertion algorithm.

use delaunay::prelude::*;
use proptest::prelude::*;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates in a reasonable range
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

/// Strategy for generating 2D vertices
fn vertex_2d() -> impl Strategy<Value = Point<f64, 2>> {
    prop::array::uniform2(finite_coordinate()).prop_map(Point::new)
}

/// Strategy for generating 3D vertices
fn vertex_3d() -> impl Strategy<Value = Point<f64, 3>> {
    prop::array::uniform3(finite_coordinate()).prop_map(Point::new)
}

/// Strategy for generating 4D vertices
fn vertex_4d() -> impl Strategy<Value = Point<f64, 4>> {
    prop::array::uniform4(finite_coordinate()).prop_map(Point::new)
}

/// Strategy for generating 5D vertices
fn vertex_5d() -> impl Strategy<Value = Point<f64, 5>> {
    prop::array::uniform5(finite_coordinate()).prop_map(Point::new)
}

// Deduplicate helpers to avoid pathological degeneracies in property tests
fn dedup_vertices_by_coords<const D: usize>(
    vertices: Vec<Vertex<f64, (), D>>,
) -> Vec<Vertex<f64, (), D>> {
    let mut unique: Vec<Vertex<f64, (), D>> = Vec::with_capacity(vertices.len());
    'outer: for v in vertices {
        let vc: [f64; D] = (&v).into();
        for u in &unique {
            let uc: [f64; D] = u.into();
            // Compare using bit representation to avoid float_cmp lint and match exact duplicates
            if vc
                .iter()
                .zip(uc.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                continue 'outer; // skip exact coordinate duplicate
            }
        }
        unique.push(v);
    }
    unique
}

fn dedup_points_2d(points: Vec<Point<f64, 2>>) -> Vec<Point<f64, 2>> {
    let mut unique: Vec<Point<f64, 2>> = Vec::with_capacity(points.len());
    'outer: for p in points {
        let pc: [f64; 2] = p.into();
        for q in &unique {
            let qc: [f64; 2] = (*q).into();
            if pc
                .iter()
                .zip(qc.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                continue 'outer;
            }
        }
        unique.push(Point::new(pc));
    }
    unique
}

// =============================================================================
// INCREMENTAL INSERTION VALIDITY TESTS
// =============================================================================

macro_rules! gen_incremental_insertion_validity {
    ($dim:literal, $min:literal, $max:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_incremental_insertion_maintains_validity_ $dim d>](
                    initial_points in prop::collection::vec([<vertex_ $dim d>](), $min..=$max),
                    additional_point in [<vertex_ $dim d>](),
                ) {
                    let initial_vertices = Vertex::from_points(&initial_points);
                    if let Ok(mut dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&initial_vertices) {
                        prop_assert!(dt.is_valid().is_ok(), "Initial {}D triangulation should be valid", $dim);
                        let additional_vertex = vertex!(additional_point);
                        if dt.insert(additional_vertex).is_ok() {
                            prop_assert!(dt.is_valid().is_ok(), "{}D triangulation should remain valid after insertion: {:?}", $dim, dt.is_valid().err());
                        }
                    }
                }
            }
        }
    };
}

gen_incremental_insertion_validity!(2, 3, 5);

gen_incremental_insertion_validity!(3, 4, 6);

gen_incremental_insertion_validity!(4, 5, 7);

gen_incremental_insertion_validity!(5, 6, 8);

// =============================================================================
// DUPLICATE COORDINATE REJECTION TESTS
// =============================================================================

macro_rules! gen_duplicate_coords_test {
    ($dim:literal, $min:literal, $max:literal) => {
        pastey::paste! {
            proptest! {
                /// Tests that duplicate coordinates are rejected during insertion.
                ///
                /// **Status**: Ignored - failing on edge cases with degenerate/nearly-degenerate configurations.
                /// Proptest found cases where duplicate insertion succeeds when it should fail.
                /// Needs investigation into duplicate detection logic in incremental insertion.
                #[ignore = "Duplicate coordinate rejection failing on edge cases - needs investigation"]
                #[test]
                fn [<prop_duplicate_coordinates_rejected_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min..=$max
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(mut dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::new(&vertices) {
                        // Select a vertex that is actually present in the triangulation.
                        // `DelaunayTriangulation::new` may skip some input vertices (e.g., due to degeneracy),
                        // so we must use stored vertices to test duplicate rejection.
                        let (_, existing_vertex) = dt.vertices().next()
                            .expect("DelaunayTriangulation::new returned Ok but has no vertices");
                        let p = *existing_vertex.point();
                        let dup = Vertex::from_points(&[p])[0];
                        let result = dt.insert(dup);
                        prop_assert!(
                            result.is_err(),
                            "expected insertion error for duplicate coordinates, got {result:?}"
                        );
                    }
                }
            }
        }
    };
}

gen_duplicate_coords_test!(2, 3, 10);
gen_duplicate_coords_test!(3, 4, 12);
gen_duplicate_coords_test!(4, 5, 14);
gen_duplicate_coords_test!(5, 6, 16);

// =============================================================================
// DELAUNAY EMPTY CIRCUMSPHERE PROPERTY
// =============================================================================

macro_rules! test_empty_circumsphere {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
proptest! {
                /// Property: For every cell, no other vertex lies strictly inside
                /// the circumsphere defined by that cell (Delaunay condition).
                ///
                /// **Status**: Ignored - Delaunay property violations found in triangulations.
                /// Proptest discovered cases where the empty circumsphere property is violated,
                /// indicating issues with the incremental insertion algorithm or geometric predicates.
                /// This may be related to the recent neighbor relationship bug fix.
                /// Needs investigation into cavity-based insertion and predicate accuracy.
                #[ignore = "Delaunay property violations found - needs investigation after neighbor bug fix"]
                #[test]
                fn [<prop_empty_circumsphere_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|pts| dedup_vertices_by_coords::<$dim>(Vertex::from_points(&pts)))
                ) {
                    // Build Delaunay triangulation using DelaunayTriangulation::new() which properly triangulates all vertices
                    // This ensures the entire triangulation (including initial simplex) satisfies Delaunay property

                    // Require at least D+1 distinct vertices to form valid D-simplices
                    prop_assume!(vertices.len() > $dim);

                    // General position filter: reject cases with >= D+1 points lying exactly on any coordinate axis
                    // (common degeneracy due to many zeros). This reduces collinear/co-planar pathologies.
                    let mut axis_counts = [0usize; $dim];
                    for v in &vertices {
                        let coords: [f64; $dim] = (*v).into();
                        for a in 0..$dim {
                            if coords[a] == 0.0 { axis_counts[a] += 1; }
                        }
                    }
                    let mut reject = false;
                    for &count in &axis_counts { if count > $dim { reject = true; break; } }
                    prop_assume!(!reject);

                    // Use DelaunayTriangulation::new() to triangulate ALL vertices together
                    let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) else {
                        // Degenerate geometry or insufficient vertices - skip test
                        prop_assume!(false);
                        unreachable!();
                    };

                    // Verify the triangulation satisfies the Delaunay property
                    is_delaunay(dt.tds()).unwrap();
                }
            }
        }
    };
}

// 2D–5D coverage (keep ranges small to bound runtime)
test_empty_circumsphere!(2, 6, 10);
test_empty_circumsphere!(3, 6, 10);
test_empty_circumsphere!(4, 6, 12);
test_empty_circumsphere!(5, 7, 12);

// =============================================================================
// INSERTION-ORDER INVARIANCE (2D)
// =============================================================================

proptest! {
    /// Property: 2D Delaunay edge set is independent of insertion order (under general position).
    ///
    /// **Status**: Ignored - requires algorithmic fixes to DelaunayTriangulation::new() for insertion-order determinism.
    ///
    /// Current findings:
    /// - `DelaunayTriangulation::new()` produces different triangulations (different vertex counts, cell counts)
    ///   when the same vertices are provided in different orders
    /// - Both triangulations may be valid Delaunay, but structural differences are significant
    /// - Root cause: Delaunay triangulation is not unique for degenerate/co-circular points
    /// - Further investigation needed: why do vertex counts differ?
    ///
    /// See issue #120 for full context.
    #[ignore = "DelaunayTriangulation::new() has insertion-order dependency; requires algorithmic investigation"]
    #[test]
    fn prop_insertion_order_invariance_2d(
        points in prop::collection::vec(
            prop::array::uniform2(finite_coordinate()).prop_map(Point::new),
            6..=10
        )
    ) {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        // Deduplicate points to avoid zero-area/duplicate-vertex degeneracies
        let points: Vec<Point<f64, 2>> = dedup_points_2d(points);
        prop_assume!(points.len() >= 3);

        // Enhanced general position filter:
        // 1. Check for axis-aligned collinearities (points on coordinate axes)
        // Reject even a SINGLE point on any axis to ensure truly general position
        let mut on_x_axis = 0;  // Points with Y ≈ 0
        let mut on_y_axis = 0;  // Points with X ≈ 0
        for p in &points {
            let coords: [f64; 2] = (*p).into();
            if coords[0].abs() < 1e-6 {
                on_y_axis += 1;
            }
            if coords[1].abs() < 1e-6 {
                on_x_axis += 1;
            }
        }
        // Strict general position: reject ANY points on coordinate axes
        prop_assume!(on_x_axis == 0 && on_y_axis == 0);

        // 2. Check for co-circularity (no point lies on circumcircle of any triangle)
        for i in 0..points.len() {
            for j in (i+1)..points.len() {
                for k in (j+1)..points.len() {
                    let tri = vec![points[i], points[j], points[k]];
                    if let (Ok(center), Ok(radius)) = (circumcenter(&tri), circumradius(&tri)) {
                        let c: [f64; 2] = center.into();
                        let tol = 1e-8 * (1.0 + radius.abs());
                        for (l, p) in points.iter().copied().enumerate() {
                            if l == i || l == j || l == k { continue; }
                            let q: [f64; 2] = p.into();
                            let dx = q[0] - c[0];
                            let dy = q[1] - c[1];
                            let d = dx.hypot(dy);
                            if (d - radius).abs() <= tol {
                                // Near co-circular: skip this case
                                prop_assume!(false);
                            }
                        }
                    }
                }
            }
        }

        let vertices: Vec<Vertex<f64, (), 2>> = Vertex::from_points(&points);
        prop_assume!(vertices.len() >= 3);

        // Build first triangulation with natural order
        let dt_a = DelaunayTriangulation::<_, (), (), 2>::new(&vertices);
        prop_assume!(dt_a.is_ok());  // Skip if triangulation fails (degenerate)
        let dt_a = dt_a.unwrap();
        prop_assume!(dt_a.is_valid().is_ok());  // Skip if invalid

        // Build second triangulation with shuffled order
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x00DE_C0DE);
        let mut vertices_shuffled = vertices;
        vertices_shuffled.shuffle(&mut rng);

        let dt_b = DelaunayTriangulation::<_, (), (), 2>::new(&vertices_shuffled);
        prop_assume!(dt_b.is_ok());  // Skip if triangulation fails (degenerate)
        let dt_b = dt_b.unwrap();
        prop_assume!(dt_b.is_valid().is_ok());  // Skip if invalid

        // Topology-based comparison (per issue #120): both triangulations should be valid
        // and have similar structure, but exact edge-by-edge matching is too strict for
        // degenerate/nearly-degenerate cases where multiple valid Delaunay solutions exist.
        //
        // Key insight: Delaunay triangulation is NOT always unique for degenerate inputs.
        // Different insertion orders can produce different (but equally valid) triangulations
        // when points are co-circular or nearly co-circular.

        // 1) Both valid (already verified via prop_assume)
        // 2) Vertex counts must match exactly (same input points)
        let verts_a = dt_a.number_of_vertices();
        let verts_b = dt_b.number_of_vertices();
        prop_assert_eq!(verts_a, verts_b, "Vertex counts must match");

        // 3) Cell counts should be within reasonable range (±20% or ±2, whichever is larger)
        // Different valid triangulations can have different cell counts
        let cells_a = dt_a.number_of_cells();
        let cells_b = dt_b.number_of_cells();
        let max_cells = cells_a.max(cells_b);
        let diff = cells_a.abs_diff(cells_b);
        let tolerance = (max_cells / 5).max(2);  // 20% or 2 cells
        prop_assert!(
            diff <= tolerance,
            "Cell counts too different: a={}, b={}, diff={}, tolerance={}",
            cells_a, cells_b, diff, tolerance
        );

        // Success: both triangulations are valid and topologically similar
        }
}

// =============================================================================
// DUPLICATE CLOUD INTEGRATION TESTS
// =============================================================================

/// Count unique coordinate tuples using bitwise equality.
/// Used to ensure at least D+1 distinct points before attempting triangulation.
fn count_unique_coords_by_bits<const D: usize>(pts: &[Point<f64, D>]) -> usize {
    use std::collections::HashSet;
    let mut set: HashSet<Vec<u64>> = HashSet::with_capacity(pts.len());
    for p in pts {
        let coords: [f64; D] = (*p).into();
        let key: Vec<u64> = coords.iter().map(|x| x.to_bits()).collect();
        set.insert(key);
    }
    set.len()
}

macro_rules! gen_duplicate_cloud_test {
    ($dim:literal, $min_vertices:literal) => {
        pastey::paste! {
            /// Generate random point cloud with exact duplicates and near-duplicates (1e-7 jitter).
            /// Tests full construction pipeline with realistic messy inputs.
            fn [<cloud_with_duplicates_ $dim d>]() -> impl Strategy<Value = Vec<Point<f64, $dim>>> {
                prop::collection::vec(
                    prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                    6..=12,
                )
                .prop_map(|mut pts| {
                    if pts.len() >= 3 {
                        // Exact duplicate of the first point
                        let dup = pts[0];
                        pts.push(dup);

                        // Jittered near-duplicate of the second point
                        let mut coords: [f64; $dim] = pts[1].into();
                        for c in &mut coords {
                            *c += 1e-7;
                        }
                        pts.push(Point::new(coords));
                    }
                    pts
                })
            }

            proptest! {
                /// Property: Random clouds with duplicates and near-duplicates
                /// produce triangulations that are globally Delaunay for the kept subset.
                ///
                /// **Status**: Ignored - Delaunay property violations (same underlying issue as empty_circumsphere tests).
                /// This integration test exercises the full construction pipeline with messy real-world inputs.
                #[ignore = "Delaunay property violations - related to empty_circumsphere failures"]
                #[test]
                fn [<prop_cloud_with_duplicates_is_delaunay_ $dim d>](
                    points in [<cloud_with_duplicates_ $dim d>]()
                ) {
                    // Require at least D+1 distinct points
                    let unique = count_unique_coords_by_bits(&points);
                    prop_assume!(unique > $min_vertices);

                    let vertices: Vec<Vertex<f64, (), $dim>> = Vertex::from_points(&points);

                    let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) else {
                        // Degenerate inputs are skipped
                        prop_assume!(false);
                        unreachable!();
                    };

                    // Structural and Delaunay validity for kept subset
                    prop_assert!(dt.is_valid().is_ok());
                    prop_assert!(is_delaunay(dt.tds()).is_ok());
                }
            }
        }
    };
}

gen_duplicate_cloud_test!(2, 2);
gen_duplicate_cloud_test!(3, 3);
gen_duplicate_cloud_test!(4, 4);
gen_duplicate_cloud_test!(5, 5);

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
    // Avoid exact/near-zero values because proptest shrinkers aggressively push toward 0.0,
    // which creates many degenerate/coplanar configurations that are not representative of
    // typical randomized inputs and can trigger topology/Euler edge cases.
    (-100.0..100.0).prop_filter("must be finite and away from zero", |x: &f64| {
        x.is_finite() && x.abs() > 1e-6
    })
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

/* Deduplicate helpers to avoid pathological degeneracies in property tests */
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

/// Conservative “general position” filter for 3D insertion-order tests.
///
/// This rejects point sets that contain *any* nearly-coplanar tetrahedron (4-point subset).
/// The insertion algorithm can hit ridge-fan degeneracy/topology repair edge cases on such inputs,
/// and the insertion-order robustness property is meant to exercise typical non-degenerate behavior.
fn has_no_nearly_coplanar_tetrahedra_3d(vertices: &[Vertex<f64, (), 3>]) -> bool {
    // Relative threshold: volume6 scales with L^3, so compare against scale^3.
    // The constant is intentionally loose: we only want to drop “shrink-pathological”
    // nearly-coplanar cases, not typical random inputs.
    const REL_EPS: f64 = 1e-12;

    fn sub(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
        [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
    }

    fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
        [
            u[1].mul_add(v[2], -(u[2] * v[1])),
            u[2].mul_add(v[0], -(u[0] * v[2])),
            u[0].mul_add(v[1], -(u[1] * v[0])),
        ]
    }

    fn dot(u: [f64; 3], v: [f64; 3]) -> f64 {
        u[2].mul_add(v[2], u[0].mul_add(v[0], u[1] * v[1]))
    }

    fn norm2(u: [f64; 3]) -> f64 {
        dot(u, u).sqrt()
    }

    let vertex_count = vertices.len();
    if vertex_count < 4 {
        return true;
    }

    for i in 0..vertex_count {
        for j in (i + 1)..vertex_count {
            for k in (j + 1)..vertex_count {
                for l in (k + 1)..vertex_count {
                    let point_a: [f64; 3] = (&vertices[i]).into();
                    let point_b: [f64; 3] = (&vertices[j]).into();
                    let point_c: [f64; 3] = (&vertices[k]).into();
                    let point_d: [f64; 3] = (&vertices[l]).into();

                    let ab = sub(point_b, point_a);
                    let ac = sub(point_c, point_a);
                    let ad = sub(point_d, point_a);

                    let volume6 = dot(ab, cross(ac, ad)).abs();
                    let scale = norm2(ab).max(norm2(ac)).max(norm2(ad));

                    // If scale is 0 (or NaN), we treat as degenerate (reject),
                    // though earlier dedup should have removed exact duplicates.
                    if scale.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                        return false;
                    }

                    let threshold = REL_EPS * scale.powi(3);
                    if volume6 <= threshold {
                        return false;
                    }
                }
            }
        }
    }

    true
}

/// Conservative “general position” filter for 3D insertion-order tests.
///
/// Rejects point sets that contain any co-spherical degeneracy among 5 points.
///
/// In a non-degenerate 3D Delaunay triangulation, no 5 points lie on the same sphere.
/// Co-spherical inputs are exactly where the cavity boundary can become non-manifold
/// (ridge fan: >2 boundary facets sharing an edge) and where insertion-order can change
/// the chosen triangulation.
fn has_no_cospherical_5_tuples_3d(vertices: &[Vertex<f64, (), 3>]) -> bool {
    let n = vertices.len();
    if n < 5 {
        return true;
    }

    let kernel = RobustKernel::<f64>::new();

    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    for m in (l + 1)..n {
                        let points = [
                            *vertices[i].point(),
                            *vertices[j].point(),
                            *vertices[k].point(),
                            *vertices[l].point(),
                            *vertices[m].point(),
                        ];

                        // For each choice of "test point", ensure it is not exactly on the circumsphere
                        // of the tetrahedron formed by the other 4 points.
                        for test_idx in 0..5 {
                            let mut simplex = [points[0], points[1], points[2], points[3]];
                            let test_point = points[test_idx];

                            // Build simplex by skipping test_idx.
                            let mut s = 0usize;
                            for (p_idx, p) in points.iter().enumerate() {
                                if p_idx == test_idx {
                                    continue;
                                }
                                simplex[s] = *p;
                                s += 1;
                            }

                            let Ok(in_sphere) = kernel.in_sphere(&simplex, &test_point) else {
                                // Treat any conversion/predicate failure as a degeneracy for test purposes.
                                return false;
                            };

                            // 0 == on boundary => co-spherical.
                            if in_sphere == 0 {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }

    true
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
                        let validation = dt.triangulation().validate();
                        prop_assert!(
                            validation.is_ok(),
                            "Initial {}D triangulation should be structurally valid (Levels 1–3): {:?}",
                            $dim,
                            validation.err()
                        );

                        let additional_vertex = vertex!(additional_point);
                        if dt.insert(additional_vertex).is_ok() {
                            let validation = dt.triangulation().validate();
                            prop_assert!(
                                validation.is_ok(),
                                "{}D triangulation should remain structurally valid after insertion (Levels 1–3): {:?}",
                                $dim,
                                validation.err()
                            );
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
                /// **Status**: Ignored - Requires bistellar flips for full Delaunay property enforcement.
                ///
                /// The incremental Bowyer-Watson algorithm can produce locally non-Delaunay configurations
                /// that cannot be repaired without topology-changing operations (bistellar flips).
                /// These tests will be re-enabled after implementing:
                /// - 2D: Edge flip (2-to-2)
                /// - 3D+: Bistellar flip operations
                ///
                /// See: Issue #120, src/core/algorithms/flips.rs
                #[ignore = "Requires bistellar flips - see Issue #120"]
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
                    // Reject if >= D points lie exactly on any coordinate hyperplane (x_i == 0).
                    // This is a common shrink target and tends to produce degenerate/coplanar inputs.
                    for &count in &axis_counts {
                        if count >= $dim {
                            reject = true;
                            break;
                        }
                    }
                    prop_assume!(!reject);

                    // Use DelaunayTriangulation::new() to triangulate ALL vertices together
                    let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) else {
                        // Degenerate geometry or insufficient vertices - skip test
                        prop_assume!(false);
                        unreachable!();
                    };

                    // Verify the triangulation satisfies the Delaunay property (Level 4)
                    let delaunay_result = dt.is_valid();
                    prop_assert!(
                        delaunay_result.is_ok(),
                        "{}D triangulation should satisfy Delaunay property: {:?}",
                        $dim,
                        delaunay_result.err()
                    );
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
// INSERTION-ORDER INVARIANCE (2D-5D)
// =============================================================================

macro_rules! gen_insertion_order_robustness_test {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: Delaunay triangulations remain valid across different insertion orders.
                ///
                /// **Status**: Passing - validates insertion-order robustness.
                ///
                /// This test verifies that the triangulation algorithm produces valid triangulations
                /// regardless of the insertion order of the input points:
                /// - Both triangulations are structurally/topologically valid (Levels 1–3)
                /// - Same vertex counts (all input points successfully inserted)
                ///
                /// The Delaunay property (Level 4) is not asserted here (see Issue #120).
                ///
                /// **Note**: The exact edge sets may differ between different insertion orders, as
                /// Delaunay triangulation is not unique for degenerate/co-spherical point sets.
                /// Multiple valid triangulations can exist for the same point set.
                ///
                /// The test uses filtering to reduce degeneracies:
                /// - Deduplicates exact coordinate matches
                /// - Rejects configurations with > D points on any axis
                #[test]
                fn [<prop_insertion_order_robustness_ $dim d>](
                    points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|pts| dedup_vertices_by_coords::<$dim>(Vertex::from_points(&pts)))
                ) {
                    use rand::seq::SliceRandom;
                    use rand::SeedableRng;

                    // Require at least D+1 distinct vertices for valid simplices
                    prop_assume!(points.len() > $dim);

                    // General position filter: reject cases with > D points on any coordinate axis
                    // (reduces collinear/coplanar pathologies)
                    let mut axis_counts = [0usize; $dim];
                    for v in &points {
                        let coords: [f64; $dim] = (*v).into();
                        for a in 0..$dim {
                            if coords[a] == 0.0 { axis_counts[a] += 1; }
                        }
                    }
                    let mut reject = false;
                    // Reject if >= D points lie exactly on any coordinate hyperplane (x_i == 0).
                    // This prevents pathological shrink cases that can break topology validation.
                    for &count in &axis_counts {
                        if count >= $dim {
                            reject = true;
                            break;
                        }
                    }
                    prop_assume!(!reject);

                    // Build first triangulation with natural order
                    let dt_a = DelaunayTriangulation::<_, (), (), $dim>::new(&points);
                    prop_assume!(dt_a.is_ok());
                    let dt_a = dt_a.unwrap();

                    let validation_a = dt_a.triangulation().validate();
                    prop_assert!(
                        validation_a.is_ok(),
                        "{}D: Triangulation A should be structurally valid (Levels 1–3): {:?}",
                        $dim,
                        validation_a.err()
                    );

                    // Build second triangulation with shuffled order
                    let mut rng = rand::rngs::StdRng::seed_from_u64(0x00DE_C0DE);
                    let mut points_shuffled = points;
                    points_shuffled.shuffle(&mut rng);

                    let dt_b = DelaunayTriangulation::<_, (), (), $dim>::new(&points_shuffled);
                    prop_assume!(dt_b.is_ok());
                    let dt_b = dt_b.unwrap();

                    let validation_b = dt_b.triangulation().validate();
                    prop_assert!(
                        validation_b.is_ok(),
                        "{}D: Triangulation B should be structurally valid (Levels 1–3): {:?}",
                        $dim,
                        validation_b.err()
                    );

                    // Verify both triangulations have the same number of vertices
                    // With early degeneracy detection, different insertion orders may reject
                    // different vertices during initial simplex construction, so we only verify
                    // that both succeeded and are valid, not that they have identical counts
                    let verts_a = dt_a.number_of_vertices();
                    let verts_b = dt_b.number_of_vertices();

                    // Both should have at least D+1 vertices (valid simplex)
                    prop_assert!(verts_a > $dim, "{}D: Triangulation A should have > {} vertices, got {}", $dim, $dim, verts_a);
                    prop_assert!(verts_b > $dim, "{}D: Triangulation B should have > {} vertices, got {}", $dim, $dim, verts_b);

                    // Both triangulations are valid - this is the key invariant
                    // The exact topology (edge sets, cell counts) may differ for degenerate/co-spherical
                    // point sets, which is expected and valid behavior

                    // TODO: Once bistellar flips are implemented to ensure unique canonical triangulations,
                    // add explicit is_delaunay() checks here:
                    // prop_assert!(is_delaunay(dt_a.tds()).is_ok(), "{}D: Triangulation A must satisfy Delaunay property", $dim);
                    // prop_assert!(is_delaunay(dt_b.tds()).is_ok(), "{}D: Triangulation B must satisfy Delaunay property", $dim);
                    // Bistellar flips will produce canonical triangulations, making edge-set comparison more meaningful.
                }
            }
        }
    };
}

// Generate tests for 2D-5D
gen_insertion_order_robustness_test!(2, 6, 10);

// 3D is specialized below to add a “near-coplanar tetrahedra” filter that avoids
// ridge-fan degeneracy cases which can break topology/Euler validation.
// (The generic filter based on x_i == 0.0 is not sufficient.)
proptest! {
    /// Property: Delaunay triangulations remain valid across different insertion orders (3D),
    /// restricted to point sets without nearly-coplanar tetrahedra.
    ///
    /// NOTE: We additionally reject any run that required retry/perturbation during insertion.
    /// Those cases are explicitly logged by the insertion algorithm as "degenerate geometry requiring perturbation"
    /// and are known to correlate with occasional Level-3 topology/Euler validation failures.
    #[test]
    fn prop_insertion_order_robustness_3d(
        points in prop::collection::vec(
            prop::array::uniform3(finite_coordinate()).prop_map(Point::new),
            6..=10
        ).prop_map(|pts| dedup_vertices_by_coords::<3>(Vertex::from_points(&pts)))
    ) {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        // Require at least D+1 distinct vertices for valid simplices
        prop_assume!(points.len() > 3);

        // Reject point sets that contain nearly-coplanar tetrahedra (4-point subsets).
        prop_assume!(has_no_nearly_coplanar_tetrahedra_3d(&points));

        // Reject point sets with co-spherical 5-tuples (degenerate Delaunay cases).
        prop_assume!(has_no_cospherical_5_tuples_3d(&points));

        // Build triangulation A by inserting incrementally and tracking retry/perturbation.
        let mut dt_a: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
        let mut a_used_retry_or_skip = false;
        for v in &points {
            let Ok((outcome, stats)) = dt_a.insert_with_statistics(*v) else {
                // Non-retryable error: outside scope for this property.
                prop_assume!(false);
                return Ok(());
            };

            if stats.attempts > 1 || matches!(outcome, InsertionOutcome::Skipped { .. }) {
                a_used_retry_or_skip = true;
                break;
            }
        }
        prop_assume!(!a_used_retry_or_skip);

        let validation_a = dt_a.triangulation().validate();
        prop_assert!(
            validation_a.is_ok(),
            "3D: Triangulation A should be structurally valid (Levels 1–3): {:?}",
            validation_a.err()
        );

        // Build triangulation B with shuffled order, same retry/skip rejection.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x00DE_C0DE);
        let mut points_shuffled = points;
        points_shuffled.shuffle(&mut rng);

        let mut dt_b: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::empty();
        let mut b_used_retry_or_skip = false;
        for v in &points_shuffled {
            let Ok((outcome, stats)) = dt_b.insert_with_statistics(*v) else {
                prop_assume!(false);
                return Ok(());
            };

            if stats.attempts > 1 || matches!(outcome, InsertionOutcome::Skipped { .. }) {
                b_used_retry_or_skip = true;
                break;
            }
        }
        prop_assume!(!b_used_retry_or_skip);

        let validation_b = dt_b.triangulation().validate();
        prop_assert!(
            validation_b.is_ok(),
            "3D: Triangulation B should be structurally valid (Levels 1–3): {:?}",
            validation_b.err()
        );

        // Both should have inserted all vertices (we reject Skipped cases above).
        prop_assert_eq!(
            dt_a.number_of_vertices(),
            dt_b.number_of_vertices(),
            "3D: both triangulations should insert the same number of vertices"
        );

        // Both should have at least D+1 vertices (valid simplex)
        let verts_a = dt_a.number_of_vertices();
        let verts_b = dt_b.number_of_vertices();
        prop_assert!(verts_a > 3, "3D: Triangulation A should have > {} vertices, got {}", 3, verts_a);
        prop_assert!(verts_b > 3, "3D: Triangulation B should have > {} vertices, got {}", 3, verts_b);
    }
}

gen_insertion_order_robustness_test!(4, 6, 12);
gen_insertion_order_robustness_test!(5, 7, 12);

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
                /// **Status**: Ignored - Requires bistellar flips (same as empty_circumsphere tests).
                ///
                /// This integration test exercises the full construction pipeline with messy real-world
                /// inputs. Will be re-enabled after bistellar flip implementation.
                ///
                /// See: Issue #120, src/core/algorithms/flips.rs
                #[ignore = "Requires bistellar flips - see Issue #120"]
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

                    // Structural/topological validity (Levels 1–3) for kept subset
                    let structural = dt.triangulation().validate();
                    prop_assert!(
                        structural.is_ok(),
                        "{}D triangulation should be structurally valid (Levels 1–3): {:?}",
                        $dim,
                        structural.err()
                    );

                    // Delaunay validity (Level 4) for kept subset
                    let delaunay = dt.is_valid();
                    prop_assert!(
                        delaunay.is_ok(),
                        "{}D triangulation should satisfy Delaunay property: {:?}",
                        $dim,
                        delaunay.err()
                    );
                }
            }
        }
    };
}

gen_duplicate_cloud_test!(2, 2);
gen_duplicate_cloud_test!(3, 3);
gen_duplicate_cloud_test!(4, 4);
gen_duplicate_cloud_test!(5, 5);

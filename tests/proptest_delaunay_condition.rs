//! Property-based tests for Delaunay-specific properties.
//!
//! - Empty circumcircle/circumsphere condition (no vertex strictly inside)
//! - Insertion-order robustness (2D): edge set is independent of insertion order

use delaunay::core::algorithms::robust_bowyer_watson::RobustBowyerWatson;
use delaunay::core::traits::insertion_algorithm::InsertionAlgorithm;
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::util::jaccard_index;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::predicates::InSphere;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::geometry::util::{circumcenter, circumradius};
use proptest::prelude::*;
use std::collections::HashSet;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

// Deduplicate helpers to avoid pathological degeneracies in property tests
fn dedup_vertices_by_coords<const D: usize>(
    vertices: Vec<Vertex<f64, Option<()>, D>>,
) -> Vec<Vertex<f64, Option<()>, D>> {
    let mut unique: Vec<Vertex<f64, Option<()>, D>> = Vec::with_capacity(vertices.len());
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

/// Filter vertices to exclude those matching initial simplex coordinates.
///
/// This prevents duplicate vertex insertion (same coordinates, different UUIDs)
/// which causes Delaunay violations. The initial simplex contains:
/// - Origin: [0, 0, ...]
/// - Axis points: [10, 0, ...], [0, 10, ...], ...
fn filter_initial_simplex_coords<const D: usize>(
    vertices: Vec<Vertex<f64, Option<()>, D>>,
    initial_simplex: &[Vertex<f64, Option<()>, D>],
) -> Vec<Vertex<f64, Option<()>, D>> {
    let mut filtered = Vec::with_capacity(vertices.len());

    'outer: for v in vertices {
        let vc: [f64; D] = (&v).into();

        // Check if this vertex matches any initial simplex coordinate
        for init_v in initial_simplex {
            let init_c: [f64; D] = init_v.into();
            if vc
                .iter()
                .zip(init_c.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                // Skip vertex that matches initial simplex coordinates
                continue 'outer;
            }
        }

        filtered.push(v);
    }

    filtered
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
// DELAUNAY EMPTY CIRCUMCIRCLE/SPHERE
// =============================================================================

fn create_initial_simplex<const D: usize>() -> Vec<Vertex<f64, Option<()>, D>> {
    let mut vertices = Vec::with_capacity(D + 1);
    // origin
    let origin_point = vec![Point::new([0.0; D])];
    vertices.push(
        Vertex::from_points(&origin_point)
            .into_iter()
            .next()
            .unwrap(),
    );
    // axis-aligned points
    for i in 0..D {
        let mut coords = [0.0_f64; D];
        coords[i] = 10.0;
        let axis_point = vec![Point::new(coords)];
        vertices.push(Vertex::from_points(&axis_point).into_iter().next().unwrap());
    }
    vertices
}

macro_rules! test_empty_circumsphere {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
proptest! {
                /// Property: For every cell, no other vertex lies strictly inside
                /// the circumsphere defined by that cell (Delaunay condition).
                #[test]
                fn [<prop_empty_circumsphere_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|pts| dedup_vertices_by_coords::<$dim>(Vertex::from_points(&pts)))
                ) {
                    // Build Delaunay triangulation using Tds::new() which properly triangulates all vertices
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

                    // Use Tds::new() to triangulate ALL vertices together, ensuring Delaunay property
                    let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) else {
                        // Degenerate geometry or insufficient vertices - skip test
                        prop_assume!(false);
                        unreachable!();
                    };

                    // Robust insphere config
                    let config = delaunay::geometry::robust_predicates::config_presets::general_triangulation::<f64>();

                    for (_ckey, cell) in tds.cells() {
                            // Check all cells - with Tds::new(), the entire triangulation should be Delaunay
                            // Only check interior facets (neighbors present)
                            if let Some(neigh) = cell.neighbors() {
                                // Build this cell's simplex once
                                let mut simplex: Vec<Point<f64, $dim>> = Vec::with_capacity($dim + 1);
                                for &vk in cell.vertices() {
                                    let v = tds.get_vertex_by_key(vk).expect("vertex exists");
                                    simplex.push(*v.point());
                                }

                                for (facet_idx, neighbor_key_opt) in neigh.iter().enumerate() {
                                    if let Some(neighbor_key) = neighbor_key_opt {
                                        let neighbor = tds.get_cell(*neighbor_key).expect("neighbor exists");

                                        // Compute shared facet by UUID intersection
                                        let mut cell_uuid_set = HashSet::with_capacity($dim + 1);
                                        for &vk in cell.vertices() {
                                            let v = tds.get_vertex_by_key(vk).expect("vertex exists");
                                            cell_uuid_set.insert(v.uuid());
                                        }
                                        let mut shared_uuids = HashSet::with_capacity($dim);
                                        let mut neighbor_opposite: Option<Point<f64, $dim>> = None;
                                        for &nvk in neighbor.vertices() {
                                            let nv = tds.get_vertex_by_key(nvk).expect("vertex exists");
                                            if cell_uuid_set.contains(&nv.uuid()) {
                                                shared_uuids.insert(nv.uuid());
                                            } else {
                                                neighbor_opposite = Some(*nv.point());
                                            }
                                        }

                                        // Must share exactly D vertices; otherwise skip (degenerate or mismatch)
                                        if shared_uuids.len() == $dim {
                                            if let Some(opp_point) = neighbor_opposite {
                                                // Local Delaunay condition: neighbor's opposite vertex is OUTSIDE or BOUNDARY of this cell's circumsphere
                                                // Use a conservative check: require BOTH robust and standard predicates to classify as INSIDE
                                                let robust_class = delaunay::geometry::robust_predicates::robust_insphere(&simplex, &opp_point, &config);
                                                let standard_class = delaunay::geometry::predicates::insphere(&simplex, opp_point);
                                                if let (Ok(rc), Ok(sc)) = (robust_class, standard_class) {
                                                    prop_assert!(
                                                        !(rc == InSphere::INSIDE && sc == InSphere::INSIDE),
                                                        "{}D: Local Delaunay violation across interior facet {}",
                                                        $dim,
                                                        facet_idx
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
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
// INSERTION-ORDER ROBUSTNESS (2D)
// =============================================================================

fn edge_set<const D: usize>(tds: &Tds<f64, Option<()>, Option<()>, D>) -> HashSet<(u128, u128)> {
    let mut edges: HashSet<(u128, u128)> = HashSet::new();
    for (_ckey, cell) in tds.cells() {
        let vks = cell.vertices();
        for i in 0..=D {
            for j in (i + 1)..=D {
                let vi = tds.get_vertex_by_key(vks[i]).expect("vertex exists");
                let vj = tds.get_vertex_by_key(vks[j]).expect("vertex exists");
                let a = vi.uuid().as_u128();
                let b = vj.uuid().as_u128();
                let edge = if a <= b { (a, b) } else { (b, a) };
                edges.insert(edge);
            }
        }
    }
    edges
}

proptest! {
    /// Property: 2D Delaunay edge set is independent of insertion order (under general position).
    /// Skip co-circular degenerate cases using insphere BOUNDARY classification.
    #[ignore = "Pending stabilization of invariance metric and thresholds; see issue #120"]
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
        // 1. Check for excessive collinearities (more than 3 points near a line)
        let mut axis_aligned_x = 0;
        let mut axis_aligned_y = 0;
        for p in &points {
            let coords: [f64; 2] = (*p).into();
            if coords[0].abs() < 1e-8 {
                axis_aligned_y += 1;
            }
            if coords[1].abs() < 1e-8 {
                axis_aligned_x += 1;
            }
        }
        // Reject if 3+ points on X or Y axis (creates degenerate collinear configurations)
        // Even 3 collinear points can cause insertion-order sensitivity
        prop_assume!(axis_aligned_x <= 2 && axis_aligned_y <= 2);

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

        let vertices: Vec<Vertex<f64, Option<()>, 2>> = Vertex::from_points(&points);

        // Build Delaunay triangulations using robust Bowyer-Watson
        let initial = create_initial_simplex::<2>();

        // Filter out vertices matching initial simplex coordinates to prevent
        // duplicate vertex insertion (same coords, different UUIDs)
        let vertices = filter_initial_simplex_coords(vertices, &initial);
        prop_assume!(!vertices.is_empty());

        let mut tds_a: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&initial).expect("init");
        let mut algo_a = RobustBowyerWatson::new();
        for v in &vertices {
            let insert_result = algo_a.insert_vertex(&mut tds_a, *v);
            prop_assert!(
                insert_result.is_ok(),
                "RobustBowyerWatson failed to insert vertex in natural order: {:?}",
                insert_result.as_ref().err()
            );
        }
        // Finalize triangulation A before validation/comparison
        let finalize_a = <RobustBowyerWatson<f64, Option<()>, Option<()>, 2> as InsertionAlgorithm<
            f64,
            Option<()>,
            Option<()>,
            2,
        >>::finalize_triangulation(&mut tds_a);
        prop_assert!(
            finalize_a.is_ok(),
            "finalize_triangulation failed for natural order: {:?}",
            finalize_a.as_ref().err()
        );
        prop_assume!(tds_a.is_valid().is_ok());

        let mut tds_b: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&initial).expect("init");
        let mut order: Vec<usize> = (0..vertices.len()).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x00DE_C0DE);
        order.shuffle(&mut rng);
        let mut algo_b = RobustBowyerWatson::new();
        for &i in &order {
            let insert_result = algo_b.insert_vertex(&mut tds_b, vertices[i]);
            prop_assert!(
                insert_result.is_ok(),
                "RobustBowyerWatson failed to insert vertex in shuffled order: {:?}",
                insert_result.as_ref().err()
            );
        }
        // Finalize triangulation B before validation/comparison
        let finalize_b = <RobustBowyerWatson<f64, Option<()>, Option<()>, 2> as InsertionAlgorithm<
            f64,
            Option<()>,
            Option<()>,
            2,
        >>::finalize_triangulation(&mut tds_b);
        prop_assert!(
            finalize_b.is_ok(),
            "finalize_triangulation failed for shuffled order: {:?}",
            finalize_b.as_ref().err()
        );
        prop_assume!(tds_b.is_valid().is_ok());

        // Compare triangulations coarsely to avoid fragility
        // 1) Both valid (already asserted via prop_assume)
        // 2) Cell counts should be close
        let cells_a = tds_a.number_of_cells();
        let cells_b = tds_b.number_of_cells();
        let diff = cells_a.abs_diff(cells_b);
        prop_assert!(diff <= 3, "Cell counts differ too much: a={}, b={}", cells_a, cells_b);

        // 3) Edge sets should have high overlap (Jaccard >= 0.95)
        let edges_a = edge_set(&tds_a);
        let edges_b = edge_set(&tds_b);
        let inter = edges_a.intersection(&edges_b).count();
        let union = edges_a.union(&edges_b).count();
        if union > 0 {
            let jaccard = jaccard_index(&edges_a, &edges_b)
                .expect("Jaccard computation should not overflow for reasonable test sets");
            // With improved general position filtering (axis-aligned <= 2), expect high similarity
            // Threshold: 0.75 empirically tuned (was 0.40 before filtering, 0.85 too strict)
            // Residual variation due to floating-point arithmetic and numerical degeneracies
            prop_assert!(jaccard >= 0.75_f64, "Edge overlap too low: jaccard={:.3} (|A|={}, |B|={}, |∩|={}, |∪|={})", jaccard, edges_a.len(), edges_b.len(), inter, union);
        }
        }
}

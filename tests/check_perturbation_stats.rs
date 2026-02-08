//! Regression: in `TopologyGuarantee::PLManifold` mode, incremental insertion must
//! never commit a triangulation with invalid vertex links, independent of
//! `ValidationPolicy`.

use delaunay::core::vertex::VertexBuilder;
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::triangulation::*;

#[test]
fn pl_manifold_insertion_is_non_negotiable_under_validation_policy_never() {
    // Generate 50 random points with seed 123 (known problematic case)
    let points = generate_random_points_seeded::<f64, 3>(50, (-100.0, 100.0), 123).unwrap();

    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);

    // Default policy is OnSuspicion.
    assert_eq!(dt.validation_policy(), ValidationPolicy::OnSuspicion);

    // Even if the user disables global validation, PL-manifold insertion must still refuse to
    // commit invalid topology (vertex-link violations).
    dt.set_validation_policy(ValidationPolicy::Never);

    // Disable Delaunay repair to keep the test focused on topology guarantees.
    dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

    for point in points {
        let vertex = VertexBuilder::default().point(point).build().unwrap();

        match dt.insert_with_statistics(vertex) {
            Ok((InsertionOutcome::Inserted { .. }, _stats)) => {
                // Bootstrap phase inserts vertices without creating cells.
                // Level 3 topology validation is not meaningful until cells exist.
                if dt.number_of_cells() > 0 {
                    dt.as_triangulation().is_valid().unwrap();
                }
            }
            Ok((InsertionOutcome::Skipped { .. }, stats)) => {
                debug_assert!(stats.skipped());
            }
            Err(e) => panic!("unexpected non-retryable insertion error: {e}"),
        }
    }

    // Final verification: Levels 1–3 (elements + structure + topology).
    dt.as_triangulation().validate().unwrap();
}

#[test]
fn pl_manifold_insertion_never_commits_invalid_topology_after_bootstrap() {
    // Smaller/faster variant: once the initial simplex exists, every successful insertion
    // must leave Level 3 topology valid.
    let points = generate_random_points_seeded::<f64, 3>(25, (-100.0, 100.0), 123).unwrap();

    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifold);
    dt.set_validation_policy(ValidationPolicy::Never);
    dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

    for point in points {
        let vertex = VertexBuilder::default().point(point).build().unwrap();
        let Ok((outcome, _stats)) = dt.insert_with_statistics(vertex) else {
            panic!("unexpected non-retryable insertion error");
        };

        if dt.number_of_cells() == 0 {
            continue;
        }

        if matches!(outcome, InsertionOutcome::Inserted { .. }) {
            dt.as_triangulation().is_valid().unwrap();
        }
    }

    dt.as_triangulation().validate().unwrap();
}

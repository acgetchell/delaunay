//! Shared point fixtures for public bistellar flip benchmarks and tests.
//!
//! These fixtures are internal to the repository's benchmark/test targets. They
//! keep Criterion registration and automated fixture checks in sync without
//! adding crate-public API surface.
//!
//! The roundtrip-capable fixtures support n=1 ergodicity checks: a single
//! admissible Pachner/bistellar move followed by its inverse should recover the
//! same triangulation, including vertex identity and simplex incidence, matching
//! the local reversibility expected from the Pachner-move references in
//! `REFERENCES.md`.

/// Stable 2D PL-manifold configuration used for explicit bistellar flips.
///
/// The square hull plus well-separated interior vertices is the control case for
/// the 2D k=1 roundtrip and k=2 edge-flip benchmark workflows.
pub const STABLE_POINTS_2D: &[[f64; 2]] = &[
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.20, 0.35],
    [0.65, 0.25],
    [0.35, 0.80],
    [0.85, 0.70],
];

/// Stable 3D PL-manifold configuration used for explicit bistellar flips.
///
/// The unit tetrahedron hull plus well-separated interior vertices is the
/// control case for 3D k=1/k=2 roundtrips and the k=3 forward benchmark.
pub const STABLE_POINTS_3D: &[[f64; 3]] = &[
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.20, 0.20, 0.20],
    [0.75, 0.15, 0.30],
    [0.20, 0.70, 0.35],
    [0.30, 0.25, 0.80],
    [0.65, 0.60, 0.55],
];

/// Stable 4D PL-manifold configuration used for explicit bistellar flips.
///
/// The unit 4-simplex hull plus clustered but non-degenerate interior vertices
/// is the control case for 4D k=1/k=2/k=3 roundtrip benchmarks.
pub const STABLE_POINTS_4D: &[[f64; 4]] = &[
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.10, 0.10, 0.10, 0.10],
    [0.15, 0.10, 0.10, 0.10],
    [0.10, 0.15, 0.10, 0.10],
    [0.10, 0.10, 0.15, 0.10],
    [0.12, 0.12, 0.12, 0.12],
    [0.20, 0.15, 0.10, 0.05],
    [0.08, 0.18, 0.12, 0.14],
];

/// Stable 5D PL-manifold configuration used for explicit bistellar flips.
///
/// The unit 5-simplex hull plus clustered but non-degenerate interior vertices
/// is the control case for 5D k=1/k=2/k=3 roundtrip benchmarks.
pub const STABLE_POINTS_5D: &[[f64; 5]] = &[
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.10, 0.10, 0.10, 0.10, 0.10],
    [0.15, 0.10, 0.10, 0.10, 0.10],
    [0.10, 0.15, 0.10, 0.10, 0.10],
    [0.10, 0.10, 0.15, 0.10, 0.10],
    [0.10, 0.10, 0.10, 0.15, 0.10],
    [0.12, 0.12, 0.12, 0.12, 0.12],
    [0.20, 0.15, 0.10, 0.05, 0.12],
    [0.08, 0.18, 0.12, 0.14, 0.16],
];

/// Adversarial 2D PL-manifold configuration for explicit bistellar flips.
///
/// Includes cospherical square corners, near-boundary points close to hull
/// edges, paired near-duplicate interior points, and a large-coordinate hull
/// point.
pub const ADVERSARIAL_POINTS_2D: &[[f64; 2]] = &[
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [1.0e-9, 0.45],
    [0.45, 1.0e-9],
    [0.50, 0.50],
    [0.50, 0.50 + 1.0e-12],
    [1.0e6, -1.0e6],
];

/// Adversarial 3D PL-manifold configuration for explicit bistellar flips.
///
/// The origin, unit axes, and one extra hypercube corner give a D+2
/// cospherical set; the interior points sit close to coordinate boundary
/// facets, and the final vertex exercises a large-coordinate hull case.
pub const ADVERSARIAL_POINTS_3D: &[[f64; 3]] = &[
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0e-9, 0.25, 0.25],
    [0.25, 1.0e-9, 0.25],
    [0.25, 0.25, 1.0e-9],
    [0.25, 0.25, 0.25 + 1.0e-12],
    [1.0e6, -1.0e6, 1.0e6],
];

/// Adversarial 4D PL-manifold configuration for explicit bistellar flips.
///
/// Combines a D+2 cospherical set from the simplex vertices plus one extra
/// hypercube corner, near-boundary interior points, nearly degenerate interior
/// clustering, and a large-coordinate hull point.
pub const ADVERSARIAL_POINTS_4D: &[[f64; 4]] = &[
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
    [1.0e-9, 0.20, 0.20, 0.20],
    [0.20, 1.0e-9, 0.20, 0.20],
    [0.20, 0.20, 1.0e-9, 0.20],
    [0.20, 0.20, 0.20, 1.0e-9],
    [0.20, 0.20, 0.20, 0.20 + 1.0e-12],
    [0.200_000_001, 0.20, 0.20, 0.20],
    [1.0e6, -1.0e6, 1.0e6, -1.0e6],
];

/// Adversarial 5D PL-manifold configuration for explicit bistellar flips.
///
/// Combines a D+2 cospherical set from the simplex vertices plus one extra
/// hypercube corner, near-boundary interior points, nearly degenerate interior
/// clustering, and a large-coordinate hull point.
pub const ADVERSARIAL_POINTS_5D: &[[f64; 5]] = &[
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0e-9, 0.16, 0.16, 0.16, 0.16],
    [0.16, 1.0e-9, 0.16, 0.16, 0.16],
    [0.16, 0.16, 1.0e-9, 0.16, 0.16],
    [0.16, 0.16, 0.16, 1.0e-9, 0.16],
    [0.16, 0.16, 0.16, 0.16, 1.0e-9],
    [0.16, 0.16, 0.16, 0.16, 0.16 + 1.0e-12],
    [0.160_000_001, 0.16, 0.16, 0.16, 0.16],
    [1.0e6, -1.0e6, 1.0e6, -1.0e6, 1.0e6],
];

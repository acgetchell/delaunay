# Topological Space Notation Fixture

// ruleid: delaunay.docs.topological-space-notation
The 3D toroidal quotient is closed.

// ruleid: delaunay.docs.topological-space-notation
Periodic toroidal construction is validated in 2D.

// ruleid: delaunay.docs.topological-space-notation
A 2D spherical Delaunay triangulation is available.

// ruleid: delaunay.docs.topological-space-notation
S2 points require three ambient coordinates.

// ruleid: delaunay.docs.topological-space-notation
fn test_builder_toroidal_3d_compact_quotient() {}

// ruleid: delaunay.docs.topological-space-notation
fn test_spherical_2d_construction() {}

// ok: delaunay.docs.topological-space-notation
The `T^3` quotient is closed.

// ok: delaunay.docs.topological-space-notation
An `S^2` Delaunay triangulation is available.

// ok: delaunay.docs.topological-space-notation
A 3D Euclidean triangulation can have an `S^2` boundary.

// ok: delaunay.docs.topological-space-notation
A 3D simplicial complex can have a `T^2` vertex link.

// ok: delaunay.docs.topological-space-notation
fn test_builder_toroidal_t3_compact_quotient() {}

// ok: delaunay.docs.topological-space-notation
fn test_spherical_s2_construction() {}

// ok: delaunay.docs.topological-space-notation
fn test_robust_insphere_near_cospherical_3d_exact_sign() {}

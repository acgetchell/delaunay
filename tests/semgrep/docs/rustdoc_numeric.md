# Native prose and extracted Rust policy

<!-- ruleid: delaunay.docs.topological-space-notation -->
A 3D toroidal construction.
<!-- ok: delaunay.docs.topological-space-notation -->
A T^3 construction.

```rust
// ruleid: delaunay.rust.no-nonfinite-unwrap-defaults
let value = Some(1.0).unwrap_or(f64::NAN);
// ok: delaunay.rust.no-nonfinite-unwrap-defaults
let value = Some(1.0).unwrap_or(0.0);
```

```rust
// ruleid: delaunay.rust.no-algebraic-f64-operations
let value = f64::algebraic_add(1.0, 2.0);
// ok: delaunay.rust.no-algebraic-f64-operations
let value = 1.0_f64 + 2.0;
// ruleid: delaunay.rust.no-partial-cmp-ordering-defaults
let order = a.partial_cmp(&b).unwrap_or(Ordering::Equal);
// ok: delaunay.rust.no-partial-cmp-ordering-defaults
let order = a.partial_cmp(&b);
```

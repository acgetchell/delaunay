# Focused Prelude Reference

Use the narrowest prelude that matches the workflow. This keeps examples,
benchmarks, doctests, and integration tests clear about which part of the API
they exercise.

| Task | Import |
|---|---|
| Unified Pachner move workflow / local topology edits | `use delaunay::prelude::pachner::*` |
| Collection aliases and small buffers | `use delaunay::prelude::collections::*` |
| Construct/configure Euclidean, toroidal, or spherical Delaunay triangulations | `use delaunay::prelude::construction::*` |
| Construction telemetry diagnostics | `use delaunay::prelude::diagnostics::*` |
| Export stable simplicial-complex primitives | `use delaunay::prelude::export::*` |
| Validation policies, errors, reports, PL-manifold link errors, and Level 5 diagnostics | `use delaunay::prelude::validation::*` |
| Delaunay repair diagnostics and policies | `use delaunay::prelude::repair::*` |
| Delaunayize workflow | `use delaunay::prelude::delaunayize::*` |
| Hilbert ordering and quantization utilities | `use delaunay::prelude::ordering::*` |
| Incremental insertion diagnostics and result types | `use delaunay::prelude::insertion::*` |
| Post-construction vertex deletion errors and keys | `use delaunay::prelude::deletion::*` |
| Low-level TDS simplices, facets, keys, and validation reports | `use delaunay::prelude::tds::*` |
| Points, simplex embeddings, coordinate ranges, kernels, predicates, and geometric measures | `use delaunay::prelude::geometry::*` |
| Random points or triangulations for examples, tests, and benchmarks | `use delaunay::prelude::generators::*` |
| Read-only traversal, adjacency, ridge views, simplex barycenters, convex hulls, and comparison helpers | `use delaunay::prelude::query::*` |
| Topological space helpers, topology traits, spherical point/metric backends, and lifted toroidal IDs | `use delaunay::prelude::topology::spaces::*` |
| Low-level topology validation, Euler characteristic helpers, manifold validators, and ridge queries | `use delaunay::prelude::topology::validation::*` |

## Policy

- Focused preludes should remain small, orthogonal, and purpose-specific.
- Do not use one focused prelude as a compatibility bucket for adjacent
  workflows.
- If a focused prelude becomes too broad or ambiguous, fix the taxonomy rather
  than preserving accidental breadth.
- The root `delaunay::prelude::*` remains available for quick experiments and
  exploratory tests.
- Repository examples, benchmarks, and doctests should prefer focused preludes
  when one communicates the workflow clearly.
- Raw bistellar flip primitives remain available through `delaunay::flips` for
  expert/debug workflows, but they are intentionally not part of a prelude.
- User-facing local move workflows should import
  `delaunay::prelude::pachner::*`.

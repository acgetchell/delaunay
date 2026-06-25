# Code Organization Guide

This file is the required-reading architecture hub for the repository. It is
intentionally short: read it first, then follow the focused architecture links
only for the surface you are changing.

## Architecture Documents

The architecture directory is indexed in
[`architecture/README.md`](architecture/README.md). For common changes, use the
focused ownership map below.

| Need | Read |
|-----|-----|
| Repository tree, packaging shape, and top-level directories | [`architecture/project_structure.md`](architecture/project_structure.md) |
| `src/` module ownership, layer boundaries, and public namespace policy | [`architecture/module_map.md`](architecture/module_map.md) |
| Focused prelude taxonomy and import guidance | [`architecture/prelude_reference.md`](architecture/prelude_reference.md) |
| In-file Rust module layout and section-order conventions | [`architecture/module_patterns.md`](architecture/module_patterns.md) |

Development guidance is indexed in [`dev/README.md`](dev/README.md), with
commands in [`dev/commands.md`](dev/commands.md). Do not copy command matrices
into architecture docs; link to the command guide instead.

## Required Orientation

- `src/core/` is the internal TDS and algorithm layer. Public low-level access
  is exposed through curated root modules such as `delaunay::tds`,
  `delaunay::collections`, `delaunay::algorithms`, and `delaunay::query`, not
  through a broad public `delaunay::core` module.
- `src/delaunay/` owns Delaunay-facing construction, insertion, deletion,
  validation, repair, flips, Pachner moves, serialization, and query APIs.
- `src/geometry/` owns points, coordinate ranges, kernels, predicates,
  geometric quality measures, convex hull support, and coordinate conversion
  utilities.
- `src/io/` owns downstream-facing export data models for notebooks,
  visualization, analysis, and interchange. It does not own TDS hydration.
- `src/topology/` owns topology-space models, Euler characteristic helpers,
  manifold validation, ridge queries, and PL-manifold reasoning.
- `src/lib.rs` wires public modules, root re-exports, focused preludes, and the
  crate-level documentation map.
- `docs/dev/README.md` indexes the operational rules for agents. Keep
  architecture orientation here and detailed workflow/tooling instructions
  under `docs/dev/`.

## Layering Rules

- Dependency direction should stay `topology -> core`, not the reverse.
- `edge.rs` and `facet.rs` stay in `src/core/` because they are direct TDS
  traversal primitives. Ridge query/view types belong in `src/topology/`
  because ridge shape and link semantics depend on dimension and topology.
- Delaunay Level 4 validation belongs in `src/delaunay/validation.rs`; generic
  Level 1-3 validation belongs in the core/topology layers.
- Focused preludes should stay narrow and workflow-specific. Use
  `delaunay::prelude::pachner::*` for local move workflows, and import
  primitive bistellar flips directly from `delaunay::flips` only when testing,
  benchmarking, or documenting that primitive layer.

## Maintenance Notes

- When files move, update the focused architecture document that owns that
  surface rather than growing this hub.
- When command names, validation recipes, or tool policy change, start from
  [`dev/README.md`](dev/README.md), then update [`dev/commands.md`](dev/commands.md) or
  [`dev/tooling-alignment.md`](dev/tooling-alignment.md), not this file.
- Keep this hub small enough to load as part of every agent session.

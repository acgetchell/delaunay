# Validation Level Fixture

// ruleid: delaunay.docs.no-stale-four-level-validation-hierarchy
The crate provides a 4-level validation hierarchy.

// ruleid: delaunay.docs.no-stale-four-level-validation-hierarchy
The library provides four levels of validation.

// ruleid: delaunay.docs.no-stale-four-level-validation-hierarchy
Level 4: Delaunay Property

// ruleid: delaunay.docs.no-stale-four-level-validation-hierarchy
The Delaunay property (Level 4) is checked here.

// ruleid: delaunay.docs.no-stale-four-level-validation-hierarchy
Delaunay Level 4 validation belongs in the Delaunay module.

// ruleid: delaunay.docs.no-stale-four-level-validation-hierarchy
After using flips, optionally repair / verify the Delaunay property.

// ok: delaunay.docs.no-stale-four-level-validation-hierarchy
The crate provides a 5-level validation hierarchy.

// ok: delaunay.docs.no-stale-four-level-validation-hierarchy
Level 5: Geometric Predicates

// ruleid: delaunay.rust.no-tds-accessor-in-markdown-examples
let storage = dt.tds();

// ok: delaunay.rust.no-tds-accessor-in-markdown-examples
let index = dt.facet_incidence_index()?;

// ruleid: delaunay.rust.no-as-triangulation-storage-reach-through
let storage = dt.as_triangulation().tds;

// ruleid: delaunay.rust.no-as-triangulation-storage-reach-through, delaunay.rust.no-tds-accessor-in-markdown-examples
let storage = dt.as_triangulation().tds();

// ruleid: delaunay.rust.no-as-triangulation-storage-reach-through
let kernel = dt.as_triangulation().kernel;

// ruleid: delaunay.rust.no-as-triangulation-storage-reach-through
let kernel = dt.as_triangulation().kernel();

// ok: delaunay.rust.no-as-triangulation-storage-reach-through
let generation = dt.topology_generation();

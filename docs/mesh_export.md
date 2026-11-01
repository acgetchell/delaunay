# Visualization Export Schema

`delaunay` exposes a generic simplicial-complex export model for analysis,
visualization, notebooks, ML pipelines, and downstream crates. The API lives in
`delaunay::io::visualization` and is also available through
`delaunay::prelude::export`.

This format is distinct from the internal `Tds` serde snapshot. Use `Tds` /
`DelaunayTriangulation` serde when you need validated Rust round-trip
persistence. Use visualization export when another tool needs stable ids,
coordinates, connectivity, and adjacency without depending on private handle
formatting. A visualization export is an owned, detached interchange snapshot; regenerate it after
mutating the source triangulation.

## Rust API

Call `DelaunayTriangulation::to_visualization_data()` to create the default
schema.

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::export::{
    ValidatedVisualizationData, VisualizationData, VisualizationDataValidationError,
    VisualizationExportError,
};
use delaunay::prelude::geometry::CoordinateConversionError;

#[derive(Debug, thiserror::Error)]
enum ExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
    #[error(transparent)]
    Export(#[from] VisualizationExportError),
    #[error(transparent)]
    Validation(#[from] VisualizationDataValidationError),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

fn main() -> Result<(), ExampleError> {
    let vertices = vec![
        vertex![0.0, 0.0]?,
        vertex![1.0, 0.0]?,
        vertex![0.0, 1.0]?,
    ];
    let triangulation = DelaunayTriangulationBuilder::new(&vertices).build()?;
    let export = triangulation.to_visualization_data()?;
    let json = serde_json::to_string_pretty(&export)?;
    let decoded: VisualizationData<2> = serde_json::from_str(&json)?;
    let validated: ValidatedVisualizationData<2> = decoded.into_validated()?;

    assert!(json.contains("delaunay.simplicial_complex"));
    assert_eq!(validated.vertices().len(), 3);
    Ok(())
}
```

## JSON Shape

The v2 schema has four common top-level fields and an additional field for
toroidal realization data:

- `metadata`: schema name, schema version, producer, dimension, counts,
  topology kind, and topology guarantee. In Rust, the topology fields are
  typed schema enums that serde round-trips as the JSON strings shown below.
- `vertices`: stable vertex UUID and coordinate array.
- `simplices`: stable simplex UUID and ordered vertex UUID membership.
- `adjacency`: one record per simplex facet, with nullable neighbor simplex
  UUIDs for boundary facets.
- `periodic`: required for toroidal exports and absent otherwise. Contains
  `periods`, the positive finite domain lengths in dimension order, and
  `simplex_vertex_offsets`, an object keyed by simplex UUID whose value is an
  ordered list of integer lattice offsets.

Example, abbreviated:

```json
{
  "metadata": {
    "schema": "delaunay.simplicial_complex",
    "schema_version": 2,
    "producer": "delaunay",
    "dimension": 2,
    "vertex_count": 3,
    "simplex_count": 1,
    "topology_kind": "Euclidean",
    "topology_guarantee": "PLManifold"
  },
  "vertices": [
    {
      "id": "00000000-0000-0000-0000-000000000001",
      "coordinates": [0.0, 0.0]
    }
  ],
  "simplices": [
    {
      "id": "00000000-0000-0000-0000-000000000010",
      "vertex_ids": [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
        "00000000-0000-0000-0000-000000000003"
      ]
    }
  ],
  "adjacency": [
    {
      "simplex_id": "00000000-0000-0000-0000-000000000010",
      "facet_index": 0,
      "neighbor_simplex_id": null
    }
  ]
}
```

## Stability Contract

Entity ids are UUIDs stored on vertices and simplices. They are stable across
ordinary JSON serde round trips of the TDS snapshot and are safe for external
tools to store. Runtime `VertexKey` and `SimplexKey` values are intentionally
absent because they are storage-local handles.

The export sorts vertices and simplices by UUID for deterministic output.
Simplex `vertex_ids` preserve the simplex's stored vertex order because that
order carries orientation and facet-slot meaning.

Export copies the source `f64` coordinates unchanged. Bit-exact JSON coordinate
round trips depend on the reader; for `serde_json`, enable its `float_roundtrip`
feature when exact recovery is required. UUIDs and integer periodic offsets
round-trip exactly.

`adjacency[*].facet_index` is aligned with the simplex's vertex list: facet
`i` is opposite `simplices[*].vertex_ids[i]`. A `null` neighbor marks a
boundary facet.

For a toroidal simplex, slot `i` is realized at
`vertex.coordinates + periodic.simplex_vertex_offsets[simplex.id][i] * periodic.periods`,
component by component. Offsets are signed 8-bit integers, matching the source
triangulation, with one `D`-component offset for each of its `D + 1` vertex
slots. The same quotient vertex UUID can occur more than once in a simplex
when its offsets differ. Neighbor facets must have the same lifted vertex
identities after a common integer translation of their charts. A plotting
consumer must apply these offsets or explicitly reject toroidal data; plotting
the stored quotient coordinates alone loses the realized mesh.

An absent source offset buffer denotes a zero-offset chart. Export writes its
offsets explicitly as zero vectors, preserving the same realization.

Exports now use schema v2. Nonperiodic v1 JSON remains readable and valid.
Toroidal v1 JSON lacks the domain and offsets required to recover its
realization and is rejected by validation; regenerate it from the original
triangulation. Rust callers constructing `VisualizationData` literals must
supply the new `periodic` field (`None` for nonperiodic data).

After deserializing JSON from another process or file, call `into_validated()`
before handing records to topology-aware code. It parses the raw DTO into a
`ValidatedVisualizationData` proof-bearing wrapper; use `validate()` only when
a one-off check is enough. Validation checks schema metadata, topology category
strings, dimensional arity, counts, non-nil and duplicate ids, finite
coordinate values, connectivity references, one adjacency record per simplex
facet, and reciprocal non-boundary adjacency whose neighbor actually contains
the source facet vertices. Toroidal validation additionally checks domain and
offset arities, distinct lifted vertex identities, and exactly one reciprocal
facet with matching identities modulo lattice translation. Unsupported
dimensions fail before simplex-arity arithmetic, including empty snapshots.
This interchange validation does not certify geometric orientation, absence of
intersections, or the Delaunay property. It does not turn the records into canonical Rust
`Tds` storage, but the validated wrapper does canonicalize detached record order
by stable ids for deterministic serialization. Use triangulation/TDS serde when
the goal is validated Rust hydration.

## Downstream Extension

The Rust records are generic over optional attribute payloads:

```rust
use delaunay::prelude::export::VisualizationData;

struct CausalVertexAttributes {
    time_slice: usize,
}

type CausalExport<const D: usize> = VisualizationData<D, CausalVertexAttributes>;
```

Downstream crates can also wrap the base export with additional tables for
foliation, simplex types, causal adjacency, simulation step, or observables
without changing the common plotting primitives.

## Non-Goals

- This export is not a native GUI or plotting layer.
- This export is not the canonical hydration format for reconstructing a
  validated Rust `Tds`.
- Arrow or Parquet exports, if added later, should remain optional analytical
  formats and should not affect the default build.

# Mesh Export Schema

`delaunay` exposes a generic simplicial-complex export model for analysis,
visualization, notebooks, ML pipelines, and downstream crates. The API lives in
`delaunay::io::visualization` and is also available through
`delaunay::prelude::export`.

This format is distinct from the internal `Tds` serde snapshot. Use `Tds` /
`DelaunayTriangulation` serde when you need validated Rust round-trip
persistence. Use mesh export when another tool needs stable ids, coordinates,
connectivity, and adjacency without depending on private handle formatting. A
mesh export is an owned, detached interchange snapshot; regenerate it after
mutating the source triangulation.

## Rust API

Call `DelaunayTriangulation::to_mesh_export()` for the default schema, or
`DelaunayTriangulation::to_visualization_data()` when the generic
visualization name communicates the workflow more clearly.

```rust
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::export::{
    MeshExport, MeshExportError, MeshExportValidationError, ValidatedMeshExport,
};
use delaunay::prelude::geometry::CoordinateConversionError;

#[derive(Debug, thiserror::Error)]
enum ExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
    #[error(transparent)]
    Export(#[from] MeshExportError),
    #[error(transparent)]
    Validation(#[from] MeshExportValidationError),
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
    let export = triangulation.to_mesh_export()?;
    let json = serde_json::to_string_pretty(&export)?;
    let decoded: MeshExport<2> = serde_json::from_str(&json)?;
    let validated: ValidatedMeshExport<2> = decoded.into_validated()?;

    assert!(json.contains("delaunay.simplicial_complex"));
    assert_eq!(validated.vertices().len(), 3);
    Ok(())
}
```

## JSON Shape

The v1 schema has four top-level fields:

- `metadata`: schema name, schema version, producer, dimension, counts,
  topology kind, and topology guarantee. In Rust, the topology fields are
  typed schema enums that serde round-trips as the JSON strings shown below.
- `vertices`: stable vertex UUID and coordinate array.
- `simplices`: stable simplex UUID and ordered vertex UUID membership.
- `adjacency`: one record per simplex facet, with nullable neighbor simplex
  UUIDs for boundary facets.

Example, abbreviated:

```json
{
  "metadata": {
    "schema": "delaunay.simplicial_complex",
    "schema_version": 1,
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

`adjacency[*].facet_index` is aligned with the simplex's vertex list: facet
`i` is opposite `simplices[*].vertex_ids[i]`. A `null` neighbor marks a
boundary facet.

After deserializing JSON from another process or file, call `into_validated()`
before handing records to topology-aware code. It parses the raw DTO into a
`ValidatedMeshExport` proof-bearing wrapper; use `validate()` only when a
one-off check is enough. Validation checks schema metadata, topology category
strings, dimensional arity, counts, non-nil and duplicate ids, finite coordinate
values, connectivity references, one adjacency record per simplex facet, and
reciprocal non-boundary adjacency whose neighbor actually contains the source
facet vertices. It does not turn the records into canonical Rust `Tds` storage,
but the validated wrapper does canonicalize detached record order by stable ids
for deterministic serialization. Use triangulation/TDS serde when the goal is
validated Rust hydration.

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

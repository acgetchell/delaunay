struct RawTdsSnapshotMissingDeserializers {
    // ruleid: delaunay.rust.raw-tds-snapshot-uuid-maps-require-duplicate-key-deserializers
    simplex_vertices: FastHashMap<Uuid, Vec<Uuid>>,
    // ruleid: delaunay.rust.raw-tds-snapshot-uuid-maps-require-duplicate-key-deserializers
    simplex_neighbors: FastHashMap<Uuid, Vec<Option<Uuid>>>,
    // ruleid: delaunay.rust.raw-tds-snapshot-uuid-maps-require-duplicate-key-deserializers
    simplex_vertex_offsets: FastHashMap<Uuid, Vec<Vec<i8>>>,
}

struct RawTdsSnapshotDuplicateKeyDeserializersOk {
    // ok: delaunay.rust.raw-tds-snapshot-uuid-maps-require-duplicate-key-deserializers
    #[serde(deserialize_with = "deserialize_simplex_vertices_no_duplicates")]
    simplex_vertices: FastHashMap<Uuid, Vec<Uuid>>,
    // ok: delaunay.rust.raw-tds-snapshot-uuid-maps-require-duplicate-key-deserializers
    #[serde(deserialize_with = "deserialize_simplex_neighbors_no_duplicates")]
    simplex_neighbors: FastHashMap<Uuid, Vec<Option<Uuid>>>,
    // ok: delaunay.rust.raw-tds-snapshot-uuid-maps-require-duplicate-key-deserializers
    #[serde(
        default,
        deserialize_with = "deserialize_simplex_vertex_offsets_no_duplicates"
    )]
    simplex_vertex_offsets: FastHashMap<Uuid, Vec<Vec<i8>>>,
}

// ruleid: delaunay.rust.tds-serialize-must-use-snapshot
impl<U, V, const D: usize> Serialize for Tds<U, V, D>
where
    U: DataSerialize,
    V: DataSerialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.vertices.serialize(serializer)
    }
}

// ok: delaunay.rust.tds-serialize-must-use-snapshot
impl<U, V, const D: usize> Serialize for Tds<U, V, D>
where
    U: DataSerialize,
    V: DataSerialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ValidatedTdsSerialization::try_new(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

// The serialization guard must stop before unrelated impls accessing storage.
impl<U, V, const D: usize> Tds<U, V, D> {
    fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }
}

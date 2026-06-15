#![allow(dead_code, unused_imports)]

use num_traits::NumCast;

// ruleid: delaunay.rust.prefer-prelude-imports-in-examples-benches
use delaunay::core::vertex::Vertex as DeepVertex;
// ok: delaunay.rust.prefer-prelude-imports-in-examples-benches
use delaunay::prelude::Vertex as PreludeVertex;

pub fn production_stdio() {
    // ruleid: delaunay.rust.no-stdio-diagnostics-in-src
    println!("debug output");

    // ruleid: delaunay.rust.no-stdio-diagnostics-in-src
    eprintln!("debug output");
}

pub fn nonfinite_defaults(value: Option<f64>) -> f64 {
    // ruleid: delaunay.rust.no-nonfinite-unwrap-defaults
    value.unwrap_or(f64::NAN)
}

pub fn silent_conversion_fallback(value: u64) -> f64 {
    // ruleid: delaunay.rust.no-silent-conversion-fallbacks, delaunay.rust.no-silent-conversion-fallbacks-in-public-samples
    NumCast::from(value).unwrap_or(0.0)
}

fn safe_f64(_value: u64) -> Option<f64> {
    Some(1.0)
}

pub fn safe_conversion_fallback(value: u64) -> f64 {
    // ruleid: delaunay.rust.no-silent-conversion-fallbacks, delaunay.rust.no-silent-conversion-fallbacks-in-public-samples
    safe_f64(value).unwrap_or(0.0)
}

pub fn public_sample_silent_conversion_fallback(value: u64) -> f64 {
    // ruleid: delaunay.rust.no-silent-conversion-fallbacks, delaunay.rust.no-silent-conversion-fallbacks-in-public-samples
    NumCast::from(value).unwrap_or(0.0)
}

pub fn partial_cmp_ordering_default(left: f64, right: f64) -> std::cmp::Ordering {
    // ruleid: delaunay.rust.no-partial-cmp-ordering-defaults
    left.partial_cmp(&right).unwrap_or(std::cmp::Ordering::Equal)
}

pub fn function_local_use_fixture() {
    // ruleid: delaunay.rust.no-function-local-use-in-src
    use std::cmp::Ordering;

    let _ordering = Ordering::Equal;
}

pub fn deep_crate_path_fixture() {
    // ruleid: delaunay.rust.no-deep-crate-paths-in-functions
    let _buffer = crate::core::collections::SimplexKeyBuffer::new();
}

pub struct RawSimplexSeedFrontier {
    // ruleid: delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers
    seed_simplices: Vec<SimplexKey>,
}

pub struct BufferedSimplexSeedFrontier {
    // ok: delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers
    seed_simplices: SimplexKeyBuffer,
}

pub struct ConflictErrorPayloadFixture {
    // ok: delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers
    extra_simplices: Vec<SimplexKey>,
}

pub fn raw_simplex_frontier_vec_fixture(
    // ruleid: delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers
    pending_seed_simplices: &mut Vec<SimplexKey>,
) {
    // ruleid: delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers
    let mut touched_simplices: Vec<SimplexKey> = Vec::new();
    // ruleid: delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers
    let conflict_preview: Vec<SimplexKey> = pending_seed_simplices.iter().copied().collect();
    // ruleid: delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers
    let mut pending_repair_simplices = Vec::new();
    // ruleid: delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers
    let removed_simplices = Vec::with_capacity(4);

    touched_simplices.extend(conflict_preview);
    pending_repair_simplices.extend(removed_simplices);
}

pub fn simplex_frontier_buffer_fixture(pending_seed_simplices: &mut SimplexKeyBuffer) {
    // ok: delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers
    let mut touched_simplices = SimplexKeyBuffer::new();
    // ok: delaunay.rust.prefer-simplex-key-buffer-for-local-frontiers
    let all_simplices: Vec<SimplexKey> = Vec::new();

    touched_simplices.extend(pending_seed_simplices.iter().copied());
    let _ = all_simplices;
}

pub fn public_unwrap_bypass(value: Option<u8>) -> u8 {
    // ruleid: delaunay.rust.no-production-unwrap-panic, delaunay.rust.no-public-surface-unwrap-panic, delaunay.rust.no-unwrap-expect-in-benches-examples
    value.unwrap()
}

pub fn public_expect_bypass(value: Option<u8>) -> u8 {
    // ruleid: delaunay.rust.no-production-unwrap-panic, delaunay.rust.no-public-surface-unwrap-panic, delaunay.rust.no-unwrap-expect-in-benches-examples
    value.expect("public APIs should return typed errors instead")
}

pub fn public_panic_bypass() {
    // ruleid: delaunay.rust.no-production-unwrap-panic, delaunay.rust.no-public-surface-unwrap-panic
    panic!("public APIs should return typed errors instead");
}

// ruleid: delaunay.rust.no-legacy-coordinate-generic-api
type LegacyPoint = Point<f64, 3>;

// ok: delaunay.rust.no-legacy-coordinate-generic-api
type CurrentPoint = Point<3>;

// ruleid: delaunay.rust.no-legacy-coordinate-generic-api
type LegacyVertex = Vertex<f64, (), 3>;

// ok: delaunay.rust.no-legacy-coordinate-generic-api
type CurrentVertex = Vertex<(), 3>;

// ok: delaunay.rust.no-legacy-coordinate-generic-api
type TuplePayloadVertex = Vertex<(i32, i32), 2>;

// ruleid: delaunay.rust.no-legacy-coordinate-generic-api
type LegacyTds = Tds<f64, (), (), 3>;

// ok: delaunay.rust.no-legacy-coordinate-generic-api
type CurrentTds = Tds<(), (), 3>;

// ruleid: delaunay.rust.no-legacy-coordinate-generic-api
type LegacyConvexHull = ConvexHull<AdaptiveKernel<f64>, (), (), 3>;

// ok: delaunay.rust.no-legacy-coordinate-generic-api
type CurrentConvexHull = ConvexHull<(), (), 3>;

// ruleid: delaunay.rust.no-legacy-coordinate-generic-api
type LegacyConvexHull3D = ConvexHull3D<AdaptiveKernel<f64>, (), ()>;

// ok: delaunay.rust.no-legacy-coordinate-generic-api
type CurrentConvexHull3D = ConvexHull3D<(), ()>;

// ruleid: delaunay.rust.no-legacy-coordinate-generic-api
type LegacyBuilder<'v> = DelaunayTriangulationBuilder<'v, f64, (), 3>;

// ok: delaunay.rust.no-legacy-coordinate-generic-api
type CurrentBuilder<'v> = DelaunayTriangulationBuilder<'v, (), 3>;

pub fn legacy_point_constructor() {
    // ruleid: delaunay.rust.no-legacy-coordinate-generic-api
    let _point = Point::new([1.0, 2.0, 3.0]);
}

pub fn current_point_constructor() {
    // ok: delaunay.rust.no-legacy-coordinate-generic-api
    let _point = Point::try_new([1.0, 2.0, 3.0]);
}

// ruleid: delaunay.rust.no-raw-point-coordinate-storage
pub struct Point<const D: usize> {
    coords: [f64; D],
}

// ok: delaunay.rust.no-raw-point-coordinate-storage
pub struct StrongPoint<const D: usize> {
    coords: ValidatedCoordinates<D>,
}

// ruleid: delaunay.rust.no-coordinate-scalar-trait
pub trait CoordinateScalar: Copy {}

// ok: delaunay.rust.no-coordinate-scalar-trait
pub struct ValidatedCoordinates<const D: usize> {
    values: [f64; D],
}

pub fn vertex_macro_bad() {
    // ruleid: delaunay.rust.no-vertex-macro
    let _vertex = vertex![1.0, 2.0, 3.0];
}

pub fn vertex_try_new_ok() {
    // ok: delaunay.rust.no-vertex-macro
    let _vertex = Vertex::<(), 3>::try_new([1.0, 2.0, 3.0]);
}

pub fn vertex_empty_bad() {
    // ruleid: delaunay.rust.no-vertex-empty-constructor
    let _vertex = Vertex::<(), 3>::empty();
}

pub fn vertex_try_new_replaces_empty_ok() {
    // ok: delaunay.rust.no-vertex-empty-constructor
    let _vertex = Vertex::<(), 3>::try_new([0.0, 0.0, 0.0]);
}

pub fn deserialize_simplex_vertex_keys_bad<M>(mut map: M)
where
    M: MapAccess<'static>,
{
    // ruleid: delaunay.rust.no-slotmap-key-topology-deserialization
    let _vertices = map.next_value::<Vec<VertexKey>>();
}

pub fn deserialize_simplex_vertex_key_buffer_bad<M>(mut map: M)
where
    M: MapAccess<'static>,
{
    // ruleid: delaunay.rust.no-slotmap-key-topology-deserialization
    let _vertices = map.next_value::<SimplexVertexKeyBuffer>();
}

pub fn deserialize_simplex_neighbor_keys_bad<M>(mut map: M)
where
    M: MapAccess<'static>,
{
    // ruleid: delaunay.rust.no-slotmap-key-topology-deserialization
    let _neighbors = map.next_value::<NeighborBuffer<Option<SimplexKey>>>();
}

pub fn deserialize_simplex_vertex_uuids_ok<M>(mut map: M)
where
    M: MapAccess<'static>,
{
    // ok: delaunay.rust.no-slotmap-key-topology-deserialization
    let _vertices = map.next_value::<Vec<Uuid>>();
}

pub fn deserialize_simplex_neighbor_uuids_ok<M>(mut map: M)
where
    M: MapAccess<'static>,
{
    // ok: delaunay.rust.no-slotmap-key-topology-deserialization
    let _neighbors = map.next_value::<Vec<Option<Uuid>>>();
}

pub fn serialize_simplex_vertex_keys_bad<S>(mut state: S)
where
    S: SerializeStruct,
{
    // ruleid: delaunay.rust.no-simplex-slotmap-key-serialization
    let _ = state.serialize_field("vertices", &self.vertex_keys);
}

pub fn serialize_simplex_neighbors_bad<S>(mut state: S)
where
    S: SerializeStruct,
{
    // ruleid: delaunay.rust.no-simplex-slotmap-key-serialization
    let _ = state.serialize_field("neighbors", &self.neighbors);
}

pub fn serialize_simplex_uuid_ok<S>(mut state: S)
where
    S: SerializeStruct,
{
    // ok: delaunay.rust.no-simplex-slotmap-key-serialization
    let _ = state.serialize_field("uuid", &self.uuid);
}

pub fn serialize_tds_uuid_relationships_ok<S>(mut state: S)
where
    S: SerializeStruct,
{
    // ok: delaunay.rust.no-simplex-slotmap-key-serialization
    let _ = state.serialize_field("simplex_vertices", &simplex_vertices);
}

pub fn serialize_tds_vertices_storage_bad<S>(mut state: S, tds: TdsFixture)
where
    S: SerializeStruct,
{
    // ruleid: delaunay.rust.no-tds-storage-map-serde
    let _ = state.serialize_field("vertices", &tds.vertices);
}

pub fn serialize_tds_simplices_storage_bad<S>(mut state: S, tds: TdsFixture)
where
    S: SerializeStruct,
{
    // ruleid: delaunay.rust.no-tds-storage-map-serde
    let _ = state.serialize_field("simplices", &tds.simplices);
}

pub fn deserialize_tds_vertices_storage_bad() {
    // ruleid: delaunay.rust.no-tds-storage-map-serde
    let _vertices: Option<StorageMap<VertexKey, Vertex<(), 3>>> = None;
}

pub fn deserialize_tds_simplices_storage_bad() {
    // ruleid: delaunay.rust.no-tds-storage-map-serde
    let _simplices: Option<StorageMap<SimplexKey, SerializedSimplex<()>>> = None;
}

pub fn rebuild_tds_simplices_storage_bad(
    // ruleid: delaunay.rust.no-tds-storage-map-serde
    serialized_simplices: StorageMap<SimplexKey, SerializedSimplex<()>>,
) {
    let _ = serialized_simplices;
}

pub fn serialize_tds_uuid_records_ok<S>(mut state: S)
where
    S: SerializeStruct,
{
    let vertices: Vec<_> = self.vertices.values().collect();
    let simplices: Vec<_> = self.simplices.values().collect();

    // ok: delaunay.rust.no-tds-storage-map-serde
    let _ = state.serialize_field("vertices", &vertices);
    // ok: delaunay.rust.no-tds-storage-map-serde
    let _ = state.serialize_field("simplices", &simplices);
}

pub fn simplex_new_constructor_bad(vertex_keys: Vec<VertexKey>) {
    // ruleid: delaunay.rust.no-simplex-new-constructor
    let _simplex = Simplex::new(vertex_keys, None);
}

pub fn simplex_try_new_constructor_ok(vertex_keys: Vec<VertexKey>) {
    // ok: delaunay.rust.no-simplex-new-constructor
    let _simplex = Simplex::try_new(vertex_keys);
}

pub fn simplex_try_new_with_data_constructor_ok(vertex_keys: Vec<VertexKey>) {
    // ok: delaunay.rust.no-simplex-new-constructor
    let _simplex = Simplex::try_new_with_data(vertex_keys, Some(()));
}

pub fn facet_new_constructors_bad<TdsType, SimplexKeyType>(
    tds: &TdsType,
    simplex_key: SimplexKeyType,
    facet_map: FacetToSimplicesMap,
) {
    // ruleid: delaunay.rust.no-facet-new-constructors
    let _facet = FacetView::new(tds, simplex_key, 0);
    // ruleid: delaunay.rust.no-facet-new-constructors
    let _all_facets = AllFacetsIter::new(tds);
    // ruleid: delaunay.rust.no-facet-new-constructors
    let _boundary_facets = BoundaryFacetsIter::new(tds, facet_map);
}

pub fn facet_try_new_constructors_ok<TdsType, SimplexKeyType>(
    tds: &TdsType,
    simplex_key: SimplexKeyType,
    facet_map: FacetToSimplicesMap,
) {
    // ok: delaunay.rust.no-facet-new-constructors
    let _facet = FacetView::try_new(tds, simplex_key, 0);
    // ok: delaunay.rust.no-facet-new-constructors
    let _all_facets = AllFacetsIter::try_new(tds);
    // ok: delaunay.rust.no-facet-new-constructors
    let _boundary_facets = BoundaryFacetsIter::try_new(tds, facet_map);
}

pub fn facet_handle_new_constructor_bad(simplex_key: SimplexKey) {
    // ruleid: delaunay.rust.no-facet-new-constructors
    let _handle = FacetHandle::new(simplex_key, 0);
}

pub fn facet_handle_try_new_constructor_ok<TdsType>(tds: &TdsType, simplex_key: SimplexKey) {
    // ok: delaunay.rust.no-facet-new-constructors
    let _handle = FacetHandle::try_new(tds, simplex_key, 0);
}

pub fn ridge_handle_new_constructor_bad(simplex_key: SimplexKey) {
    // ruleid: delaunay.rust.no-ridgehandle-new-constructor
    let _handle = RidgeHandle::new(simplex_key, 0, 1);
}

pub fn ridge_handle_try_new_constructor_ok<TdsType>(tds: &TdsType, simplex_key: SimplexKey) {
    // ok: delaunay.rust.no-ridgehandle-new-constructor
    let _handle = RidgeHandle::try_new(tds, simplex_key, 0, 1);
}

pub fn edgekey_new_constructor_bad(a: VertexKey, b: VertexKey) {
    // ruleid: delaunay.rust.no-edgekey-new-constructor
    let _edge = EdgeKey::new(a, b);
}

pub fn edgekey_try_new_without_tds_bad(a: VertexKey, b: VertexKey) {
    // ruleid: delaunay.rust.no-edgekey-try-new-without-tds
    let _edge = EdgeKey::try_new(a, b);
}

pub fn edgekey_try_new_constructor_ok<TdsType>(tds: &TdsType, a: VertexKey, b: VertexKey) {
    // ok: delaunay.rust.no-edgekey-new-constructor
    // ok: delaunay.rust.no-edgekey-try-new-without-tds
    let _edge = EdgeKey::try_new(tds, a, b);
}

impl PublicVertexUuidConstructorFixture {
    // ruleid: delaunay.rust.no-public-vertex-new-with-uuid
    pub const fn new_with_uuid(point: Point<3>, uuid: Uuid, data: Option<()>) -> Self {
        Self { point, uuid, data }
    }
}

impl CratePrivateVertexUuidConstructorFixture {
    // ok: delaunay.rust.no-public-vertex-new-with-uuid
    pub(crate) const fn new_with_uuid(point: Point<3>, uuid: Uuid, data: Option<()>) -> Self {
        Self { point, uuid, data }
    }

    // ok: delaunay.rust.no-public-vertex-new-with-uuid
    pub fn try_new_with_uuid(
        point: Point<3>,
        uuid: Uuid,
        data: Option<()>,
    ) -> Result<Self, VertexValidationError> {
        validate_uuid(&uuid)?;
        Ok(Self { point, uuid, data })
    }
}

fn private_documented_invariant(value: Option<u8>) -> u8 {
    // ok: delaunay.rust.no-production-unwrap-panic
    // ruleid: delaunay.rust.no-public-surface-unwrap-panic, delaunay.rust.no-unwrap-expect-in-benches-examples
    value.expect("private helper documents an internal invariant")
}

pub fn env_gated_stdio() {
    // ruleid: delaunay.rust.no-env-gated-stdio-diagnostics
    if std::env::var_os("DELAUNAY_DEBUG").is_some() {
        // ruleid: delaunay.rust.no-stdio-diagnostics-in-src
        println!("debug output");
    }
}

// ruleid: delaunay.rust.no-clippy-allow-lints
#[allow(clippy::too_many_lines)]
fn clippy_allow_fixture() {}

// ruleid: delaunay.rust.no-ignored-tests
#[ignore = "Slow (>10s); use the slow-tests feature instead"]
fn slow_ignore_fixture() {}

// ok: delaunay.rust.no-ignored-tests
#[cfg(feature = "slow-tests")]
fn slow_cfg_fixture() {}

// ruleid: delaunay.rust.expect-requires-reason
#[expect(clippy::too_many_lines)]
fn expect_without_reason_fixture() {}

// ok: delaunay.rust.expect-requires-reason
#[expect(clippy::too_many_lines, reason = "fixture documents the suppression")]
fn expect_with_reason_fixture() {}

// ruleid: delaunay.rust.no-box-dyn-error-in-src, delaunay.rust.no-box-dyn-error-in-examples-benches
type ProductionBoxedError = Box<dyn std::error::Error>;

trait ProductionDynamicErrors {
    // ruleid: delaunay.rust.no-box-dyn-error-in-src, delaunay.rust.no-box-dyn-error-in-examples-benches
    fn boxed_error_result(&self) -> Result<(), Box<dyn std::error::Error>>;

    // ruleid: delaunay.rust.no-box-dyn-error-in-src, delaunay.rust.no-box-dyn-error-in-examples-benches
    fn borrowed_error(&self, error: &dyn std::error::Error);

    // ruleid: delaunay.rust.no-box-dyn-error-in-src, delaunay.rust.no-box-dyn-error-in-examples-benches
    fn anyhow_error(&self, error: anyhow::Error);
}

// ruleid: delaunay.rust.public-error-enums-non-exhaustive
pub enum PublicFixtureError {
    Invalid,
}

// ok: delaunay.rust.public-error-enums-non-exhaustive
#[non_exhaustive]
pub enum PublicNonExhaustiveFixtureError {
    Invalid,
}

// ok: delaunay.rust.public-error-enums-non-exhaustive
enum PrivateFixtureError {
    Invalid,
}

#[non_exhaustive]
pub enum CoordinateRangeError<T = f64> {
    // ruleid: delaunay.rust.no-stringly-coordinate-range-error-payloads
    NonIncreasing {
        min: String,
        max: String,
    },
    TypedNonIncreasing {
        // ok: delaunay.rust.no-stringly-coordinate-range-error-payloads
        min: T,
        // ok: delaunay.rust.no-stringly-coordinate-range-error-payloads
        max: T,
    },
}

#[non_exhaustive]
// ruleid: delaunay.rust.no-stringly-generator-error-payloads
pub enum InvalidPositiveScalar<T = f64> {
    NonFinite,
    NonPositive {
        value: String,
    },
    TypedNonPositive {
        // ok: delaunay.rust.no-stringly-generator-error-payloads
        value: T,
    },
}

#[non_exhaustive]
pub enum RandomPointGenerationError<T = f64> {
    // ruleid: delaunay.rust.no-stringly-generator-error-payloads
    PoissonSamplingFailed {
        requested_points: usize,
        generated_points: usize,
        min_distance: String,
        bounds_min: String,
        bounds_max: String,
        attempts: usize,
    },
    TypedPoissonSamplingFailed {
        requested_points: usize,
        generated_points: usize,
        // ok: delaunay.rust.no-stringly-generator-error-payloads
        min_distance: T,
        // ok: delaunay.rust.no-stringly-generator-error-payloads
        bounds: CoordinateRange<T>,
        attempts: usize,
    },
    // ruleid: delaunay.rust.no-stringly-generator-error-payloads
    CoordinateConversionFailed {
        value: String,
    },
    TypedCoordinateConversionFailed {
        // ok: delaunay.rust.no-stringly-generator-error-payloads
        value: usize,
    },
}

#[non_exhaustive]
pub enum NumericDiagnosticError {
    // ruleid: delaunay.rust.no-stringly-numeric-error-payloads
    ConversionFailed {
        coordinate_value: String,
    },
    // ruleid: delaunay.rust.no-stringly-numeric-error-payloads
    NonFiniteValue {
        coordinate_value: String,
    },
    // ruleid: delaunay.rust.no-stringly-numeric-error-payloads
    InvalidCoordinate {
        coordinate_value: String,
    },
    // ruleid: delaunay.rust.no-stringly-numeric-error-payloads
    ValueConversionFailed {
        value: String,
        details: String,
    },
    // ruleid: delaunay.rust.no-stringly-numeric-error-payloads
    DegenerateSimplex {
        observed: String,
        epsilon: String,
        avg_edge_length: Option<String>,
    },
    // ruleid: delaunay.rust.no-stringly-numeric-error-payloads
    MatrixInversionFailed {
        details: String,
    },
    // ruleid: delaunay.rust.no-stringly-numeric-error-payloads
    PerturbationScaleConversion {
        value: String,
    },
    // ruleid: delaunay.rust.no-stringly-numeric-error-payloads
    DuplicateCoordinates {
        coordinates: String,
    },
    TypedConversionFailed {
        // ok: delaunay.rust.no-stringly-numeric-error-payloads
        coordinate_value: CoordinateConversionValue,
    },
    TypedQualityDegeneracy {
        // ok: delaunay.rust.no-stringly-numeric-error-payloads
        observed: CoordinateConversionValue,
        // ok: delaunay.rust.no-stringly-numeric-error-payloads
        epsilon: CoordinateConversionValue,
        // ok: delaunay.rust.no-stringly-numeric-error-payloads
        avg_edge_length: Option<CoordinateConversionValue>,
    },
    TypedPerturbationScaleConversion {
        // ok: delaunay.rust.no-stringly-numeric-error-payloads
        value: CoordinateConversionValue,
    },
    TypedDuplicateCoordinates {
        // ok: delaunay.rust.no-stringly-numeric-error-payloads
        coordinates: CoordinateValues,
    },
}

#[non_exhaustive]
pub enum FlipContextError {
    Invalid,
}

#[non_exhaustive]
pub enum SimplexValidationError {
    Invalid,
}

#[non_exhaustive]
pub enum FlipError {
    BadUnboxedContext {
        // ruleid: delaunay.rust.flip-error-nested-payloads-boxed
        reason: FlipContextError,
    },
    BadUnboxedSimplex(
        // ruleid: delaunay.rust.flip-error-nested-payloads-boxed
        SimplexValidationError,
    ),
    BadBoxedContextMissingSource {
        // ruleid: delaunay.rust.flip-error-boxed-payloads-are-sources
        reason: Box<FlipContextError>,
    },
    GoodBoxedContext {
        // ok: delaunay.rust.flip-error-boxed-payloads-are-sources
        #[source]
        reason: Box<FlipContextError>,
    },
    // ok: delaunay.rust.flip-error-boxed-payloads-are-sources
    GoodBoxedSimplex(#[from] Box<SimplexValidationError>),
    ScalarDiagnostic { found: usize },
}

/// // ruleid: delaunay.rust.no-box-dyn-error-in-doctests
/// # Ok::<(), Box<dyn std::error::Error>>(())
fn doctest_style_error_is_ignored() {}

/// ```rust
/// // ruleid: delaunay.rust.prefer-prelude-imports-in-delaunay-doctests
/// use delaunay::flips::BistellarFlips;
/// // ok: delaunay.rust.prefer-prelude-imports-in-delaunay-doctests
/// use delaunay::prelude::DelaunayTriangulation;
/// // ok: delaunay.rust.prefer-prelude-imports-in-delaunay-doctests
/// # use delaunay::prelude::DelaunayTriangulation as HiddenPreludeImport;
/// // ruleid: delaunay.rust.prefer-prelude-imports-in-delaunay-doctests
/// # use delaunay::flips::BistellarFlips as HiddenDeepImport;
/// ```
fn triangulation_doctest_deep_import_fixture() {}

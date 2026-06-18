# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ⚠️ Breaking Changes

- Validate toroidal domains at parse boundaries [#437](https://github.com/acgetchell/delaunay/pull/437) [#450](https://github.com/acgetchell/delaunay/pull/450)
- Add vertex construction macro [#469](https://github.com/acgetchell/delaunay/pull/469)
- Box nested FlipError payloads [#406](https://github.com/acgetchell/delaunay/pull/406) [#435](https://github.com/acgetchell/delaunay/pull/435)
- Adopt la-stack 0.4.3 API [#424](https://github.com/acgetchell/delaunay/pull/424) [#438](https://github.com/acgetchell/delaunay/pull/438)
- Require refined generator and ordering parameters [#439](https://github.com/acgetchell/delaunay/pull/439)
- Parse coordinate inputs into validated types [#440](https://github.com/acgetchell/delaunay/pull/440) [#444](https://github.com/acgetchell/delaunay/pull/444)
- Require validated generator and Hilbert inputs [#452](https://github.com/acgetchell/delaunay/pull/452)
- Require validated coordinate topology API [#442](https://github.com/acgetchell/delaunay/pull/442) [#455](https://github.com/acgetchell/delaunay/pull/455)
- Return fallible facet iterators [#458](https://github.com/acgetchell/delaunay/pull/458)
- Hydrate TDS through validated UUID snapshots [#454](https://github.com/acgetchell/delaunay/pull/454) [#460](https://github.com/acgetchell/delaunay/pull/460)
- Normalize fallible constructors [#459](https://github.com/acgetchell/delaunay/pull/459) [#464](https://github.com/acgetchell/delaunay/pull/464)
- Reject stale adjacency indexes [#451](https://github.com/acgetchell/delaunay/pull/451) [#463](https://github.com/acgetchell/delaunay/pull/463)
- Update tooling to Rust 1.96.0 [#430](https://github.com/acgetchell/delaunay/pull/430) [#431](https://github.com/acgetchell/delaunay/pull/431)

### Merged Pull Requests

- Add vertex construction macro [#469](https://github.com/acgetchell/delaunay/pull/469)
- Preserve setup failure messages [#468](https://github.com/acgetchell/delaunay/pull/468)
- Refactor/443 329 typed validation errors [#465](https://github.com/acgetchell/delaunay/pull/465)
- Normalize fallible constructors [#459](https://github.com/acgetchell/delaunay/pull/459) [#464](https://github.com/acgetchell/delaunay/pull/464)
- Reject stale adjacency indexes [#451](https://github.com/acgetchell/delaunay/pull/451) [#463](https://github.com/acgetchell/delaunay/pull/463)
- Bump the uv group across 1 directory with 3 updates [#462](https://github.com/acgetchell/delaunay/pull/462)
- Hydrate TDS through validated UUID snapshots [#454](https://github.com/acgetchell/delaunay/pull/454) [#460](https://github.com/acgetchell/delaunay/pull/460)
- Return fallible facet iterators [#458](https://github.com/acgetchell/delaunay/pull/458)
- Bump codecov/codecov-action from 6.0.1 to 7.0.0 [#457](https://github.com/acgetchell/delaunay/pull/457)
- Bump actions/checkout from 6.0.2 to 6.0.3 [#456](https://github.com/acgetchell/delaunay/pull/456)
- Require validated coordinate topology API [#442](https://github.com/acgetchell/delaunay/pull/442) [#455](https://github.com/acgetchell/delaunay/pull/455)
- Require validated generator and Hilbert inputs [#452](https://github.com/acgetchell/delaunay/pull/452)
- Validate toroidal domains at parse boundaries [#437](https://github.com/acgetchell/delaunay/pull/437) [#450](https://github.com/acgetchell/delaunay/pull/450)
- Parse coordinate inputs into validated types [#440](https://github.com/acgetchell/delaunay/pull/440) [#444](https://github.com/acgetchell/delaunay/pull/444)
- Require refined generator and ordering parameters [#439](https://github.com/acgetchell/delaunay/pull/439)
- Adopt la-stack 0.4.3 API [#424](https://github.com/acgetchell/delaunay/pull/424) [#438](https://github.com/acgetchell/delaunay/pull/438)
- Box nested FlipError payloads [#406](https://github.com/acgetchell/delaunay/pull/406) [#435](https://github.com/acgetchell/delaunay/pull/435)
- Harden support tooling for Python 3.13 [#433](https://github.com/acgetchell/delaunay/pull/433)
- Expand public flip benchmark coverage [#432](https://github.com/acgetchell/delaunay/pull/432)
- Update tooling to Rust 1.96.0 [#430](https://github.com/acgetchell/delaunay/pull/430) [#431](https://github.com/acgetchell/delaunay/pull/431)
- Bump starlette in the uv group across 1 directory [#428](https://github.com/acgetchell/delaunay/pull/428)
- Bump taiki-e/install-action from 2.79.1 to 2.81.1 [#427](https://github.com/acgetchell/delaunay/pull/427)
- Bump the dependencies group across 1 directory with 4 updates [#426](https://github.com/acgetchell/delaunay/pull/426)

### Added

- [**breaking**] Validate toroidal domains at parse boundaries [#437](https://github.com/acgetchell/delaunay/pull/437)
  [#450](https://github.com/acgetchell/delaunay/pull/450) [`b1c52b6`](https://github.com/acgetchell/delaunay/commit/b1c52b605c57be5ea58018b787f1fba7244c1ee3)

  - Add ToroidalDomain and ToroidalDomainError so toroidal periods are finite,
    strictly positive, and validated before storage.

  - Add fallible raw-period constructors for GlobalTopology, ToroidalSpace, and
    ToroidalModel while keeping builder toroidal helpers ergonomic.

  - Align shared development tooling pins with causal-triangulations and harden
    benchmark baseline metadata parsing.

- [**breaking**] Add vertex construction macro [#469](https://github.com/acgetchell/delaunay/pull/469)
  [`63228a0`](https://github.com/acgetchell/delaunay/commit/63228a06ffca2ee8d68806823995d92ebfa84525)

  - Add `vertex!` as a fallible constructor for coordinate-only and data-bearing vertices.
  - Export the macro through the root, construction, and triangulation preludes.
  - Migrate public docs, examples, and benchmark setup to prefer `vertex!` for incidental vertex construction.
  - Retire the Semgrep rule that banned `vertex!` and document the new Rust style guidance.

### Changed

- [**breaking**] Box nested FlipError payloads [#406](https://github.com/acgetchell/delaunay/pull/406) [#435](https://github.com/acgetchell/delaunay/pull/435)
  [`2f310d9`](https://github.com/acgetchell/delaunay/commit/2f310d91e0b600fefdc488f2117dc6413b907b25)

  - Box nested typed `FlipError` payloads and the inserted-simplex witness while keeping scalar and key diagnostics inline.
  - Preserve typed inspection through `Error::source` , `reason.as_ref()` , and `source.as_ref()` for flip context, predicate, adjacency, simplex,
    neighbor-wiring, and mutation failures.

  - Add repository Semgrep checks for the boxed-source policy and document the tooling rationale.
  - Preserve padded changelog category headings during post-processing.
- [**breaking**] Adopt la-stack 0.4.3 API [#424](https://github.com/acgetchell/delaunay/pull/424) [#438](https://github.com/acgetchell/delaunay/pull/438)
  [`8e58d57`](https://github.com/acgetchell/delaunay/commit/8e58d57fa2439dd1ec04490a527b8c00b20a091b)

  - Route stack-matrix dispatch, checked access, determinant filters, and
    singular tolerances through the local geometry matrix shim.

  - Preserve typed la-stack solve and factorization errors in geometry error paths
    instead of stringifying backend diagnostics.

  - Use rounded exact solve fallback for circumcenters and reject non-finite
    predicate matrices at construction boundaries.

  - Make `.toroidal()` the periodic image-point constructor and move wrapping-only
    construction to `.canonicalized_toroidal()`.

  - Align pinned just, rumdl, taplo, dprint, and typos setup through cargo installs
    instead of Homebrew.

  - Disable la-stack default features explicitly while selecting exact arithmetic.
- [**breaking**] Require refined generator and ordering parameters [#439](https://github.com/acgetchell/delaunay/pull/439)
  [`ca95380`](https://github.com/acgetchell/delaunay/commit/ca95380b4f6ae1a736f34e3badbc899475feac84)

  - Add `HilbertBitDepth` for validated Hilbert ordering precision and route public ordering helpers through it.
  - Require `NonZeroUsize` for grid and random triangulation generation counts.
  - Remove the random generator's zero-count empty-triangulation path so empty triangulations stay on explicit constructors.
- [**breaking**] Parse coordinate inputs into validated types [#440](https://github.com/acgetchell/delaunay/pull/440)
  [#444](https://github.com/acgetchell/delaunay/pull/444) [`f0252e5`](https://github.com/acgetchell/delaunay/commit/f0252e539ff4ac6c5059bd755ba34ca655977a79)

  - Add CoordinateRange and route generator and Hilbert range inputs through
    typed boundary parsing before internal use.

  - Replace stringly numeric diagnostics with typed coordinate, count, range, and
    error-reason payloads across geometry and generator APIs.

  - Restrict the public coordinate scalar contract to f64 while documenting future
    exact-coordinate support as an explicit API addition.

  - Move geometry and generator error types into their owning modules and update
    prelude exports, docs, examples, and semgrep guardrails accordingly.
- [**breaking**] Require validated generator and Hilbert inputs [#452](https://github.com/acgetchell/delaunay/pull/452)
  [`2319f03`](https://github.com/acgetchell/delaunay/commit/2319f0379a2f1c1bb50b213bfd20ca6bec1822f7)

  - Rename raw-bound generator and Hilbert ordering APIs to `try_*` names so fallible parsing is visible at call sites.
  - Add `CoordinateRange`-based and prevalidated Hilbert batch APIs for callers that already carry validation evidence.
  - Replace Hilbert debug-only invariants with proof-carrying index modes and typed permutation errors.
  - Bound ball rejection sampling with a typed `BallSamplingFailed` error instead of relying on unbounded retry.
  - Update preludes, examples, benches, and tests to exercise the refined API surface.
- [**breaking**] Require validated coordinate topology API [#442](https://github.com/acgetchell/delaunay/pull/442)
  [#455](https://github.com/acgetchell/delaunay/pull/455) [`da82604`](https://github.com/acgetchell/delaunay/commit/da82604480cbf04f6cb7e609a151e0c757d9174d)

  - Store coordinates through validated finite-coordinate types and remove the public coordinate-scalar parameter from core geometry, TDS, hull, and
    triangulation APIs.

  - Replace macro and infallible raw constructors with explicit fallible smart constructors for points, vertices, simplices, edges, facets, and flip handles.
  - Serialize topology relationships through stable vertex and simplex UUIDs instead of process-local slotmap keys.
  - Add semgrep guardrails and update docs, examples, benches, and tests for the validated-coordinate API.
- [**breaking**] Return fallible facet iterators [#458](https://github.com/acgetchell/delaunay/pull/458)
  [`739aba0`](https://github.com/acgetchell/delaunay/commit/739aba043f135f58f3346ec58ee8b20812b234db)

  - Make all-facet and boundary-facet traversal yield `Result&lt;FacetView, FacetError&gt;` so corrupted facet views and invalid boundary incidence are surfaced
    instead of skipped.

  - Route boundary-facet consumers, hull extraction, Euler counting, examples, benches, and prelude coverage through explicit item-error handling.
  - Use `SimplexKeyBuffer` for local repair and topology frontiers, and add a Semgrep guard for future raw `Vec&lt;SimplexKey&gt;` regressions.
  - Add a compact 2D-5D timing summary to `just perf-large-scale-smoke`.
- [**breaking**] Hydrate TDS through validated UUID snapshots [#454](https://github.com/acgetchell/delaunay/pull/454)
  [#460](https://github.com/acgetchell/delaunay/pull/460) [`87eb8b1`](https://github.com/acgetchell/delaunay/commit/87eb8b197fe5f7a826f196f3adde4919d0f60ff9)

  - Route TDS serialization through a validated UUID snapshot boundary that carries vertex, simplex, neighbor, and periodic-offset relationships without
    storage-local slotmap handles.

  - Rebuild runtime TDS storage only from validated snapshots, with fresh slotmap keys and full topology validation before exposing hydrated state.
  - Keep standalone simplex records from becoming an alternate hydration path, so simplex connectivity is resolved only in the TDS snapshot context.
  - Update repository guards and documentation to describe snapshot-based persistence as the serialization boundary.
- [**breaking**] Normalize fallible constructors [#459](https://github.com/acgetchell/delaunay/pull/459) [#464](https://github.com/acgetchell/delaunay/pull/464)
  [`0fb6607`](https://github.com/acgetchell/delaunay/commit/0fb6607551df4d359aa3ae46a92c942f8ef69737)

  - Rename fallible Delaunay triangulation and convex hull constructors to explicit `try_*` forms while keeping infallible constructors for validated inputs.
  - Remove the raw-array validated-coordinate constructor path in favor of `Point::try_new` and validated coordinate proofs.
  - Update documentation, examples, benchmarks, and Semgrep guardrails for the constructor naming contract and panic-free public snippets.
  - Report generated simplex counts in the large-scale smoke benchmark output.
- Refactor/443 329 typed validation errors [#465](https://github.com/acgetchell/delaunay/pull/465)
  [`f6a85e8`](https://github.com/acgetchell/delaunay/commit/f6a85e84d43a81f1d5fdd8ec4c20e65e303dfe40)

### Dependencies

- Bump support tooling and smallvec [`1799d3c`](https://github.com/acgetchell/delaunay/commit/1799d3cbd4a5c01a10c490fffe15cc28b1f3784d)

### Fixed

- [**breaking**] Reject stale adjacency indexes [#451](https://github.com/acgetchell/delaunay/pull/451) [#463](https://github.com/acgetchell/delaunay/pull/463)
  [`bda1cd5`](https://github.com/acgetchell/delaunay/commit/bda1cd508cec667d63e01001900d175a0f661bbe)

  - Validate caller-supplied AdjacencyIndex values against the originating TDS identity and generation before indexed topology queries.
  - Keep AdjacencyIndex internals immutable to downstream callers and route public use through accessor methods.
  - Convert several debug-only invariant checks into typed errors for malformed topology, predicate matrix arity, and cavity replacement mismatches.
  - Avoid release-mode coherent-orientation scans in flip hot paths while preserving structured debug/test diagnostics.
  - Refresh README badges to use Badgen endpoints.
- Preserve setup failure messages [#468](https://github.com/acgetchell/delaunay/pull/468)
  [`fe3ed92`](https://github.com/acgetchell/delaunay/commit/fe3ed92a79684feb6302e03b27887b7ad10f1dfe)

  - Replace benchmark setup unwrap helpers with postfix abort adapters that keep the original Result error text.
  - Keep Option setup failures explicit with caller-provided context.
  - Report abort messages without bench-logging so setup failures are visible in every benchmark build.

### Maintenance

- Bump starlette in the uv group across 1 directory [#428](https://github.com/acgetchell/delaunay/pull/428)
  [`1cde3ee`](https://github.com/acgetchell/delaunay/commit/1cde3ee62e23065d657a5cc9f8a438fe72933632)

  Bumps the uv group with 1 update in the / directory: [starlette](https://github.com/Kludex/starlette).

  Updates `starlette` from 1.0.0 to 1.0.1

  - [Release notes](https://github.com/Kludex/starlette/releases)
  - [Changelog](https://github.com/Kludex/starlette/blob/main/docs/release-notes.md)
  - [Commits](https://github.com/Kludex/starlette/compare/1.0.0...1.0.1)

---

  updated-dependencies:

- dependency-name: starlette
  dependency-version: 1.0.1
  dependency-type: indirect
  dependency-group: uv
  ...

- Bump taiki-e/install-action from 2.79.1 to 2.81.1 [#427](https://github.com/acgetchell/delaunay/pull/427)
  [`3798932`](https://github.com/acgetchell/delaunay/commit/37989328340d1fcc6c8fc0a8be5adf03b31f7fcd)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.79.1 to 2.81.1.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/b550161ef8a7bc4f2a671c0b03a18ac9ccedea1e...e49978b799e49ff429d162b7a30601a569ab6538)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.81.1
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump the dependencies group across 1 directory with 4 updates [#426](https://github.com/acgetchell/delaunay/pull/426)
  [`daabc99`](https://github.com/acgetchell/delaunay/commit/daabc99f012f4c35789f71777d77e3e2b08593aa)

  Bumps the dependencies group with 4 updates in the / directory: [uuid](https://github.com/uuid-rs/uuid) , [pastey](https://github.com/as1100k/pastey) ,
  [serde_json](https://github.com/serde-rs/json) and [sysinfo](https://github.com/GuillaumeGomez/sysinfo) .

  Updates `uuid` from 1.23.1 to 1.23.2

  - [Release notes](https://github.com/uuid-rs/uuid/releases)
  - [Commits](https://github.com/uuid-rs/uuid/compare/v1.23.1...v1.23.2)

  Updates `pastey` from 0.2.2 to 0.2.3

  - [Release notes](https://github.com/as1100k/pastey/releases)
  - [Changelog](https://github.com/AS1100K/pastey/blob/master/CHANGELOG.md)
  - [Commits](https://github.com/as1100k/pastey/compare/v0.2.2...v0.2.3)

  Updates `serde_json` from 1.0.149 to 1.0.150

  - [Release notes](https://github.com/serde-rs/json/releases)
  - [Commits](https://github.com/serde-rs/json/compare/v1.0.149...v1.0.150)

  Updates `sysinfo` from 0.39.2 to 0.39.3

  - [Changelog](https://github.com/GuillaumeGomez/sysinfo/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/GuillaumeGomez/sysinfo/compare/v0.39.2...v0.39.3)

---

  updated-dependencies:

- dependency-name: uuid
  dependency-version: 1.23.2
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies

- dependency-name: pastey
  dependency-version: 0.2.3
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies

- dependency-name: serde_json
  dependency-version: 1.0.150
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies

- dependency-name: sysinfo
  dependency-version: 0.39.3
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies
  ...

- [**breaking**] Update tooling to Rust 1.96.0 [#430](https://github.com/acgetchell/delaunay/pull/430) [#431](https://github.com/acgetchell/delaunay/pull/431)
  [`1ec1d5a`](https://github.com/acgetchell/delaunay/commit/1ec1d5ac6ab02ecefee37495bf9b82fbc434f645)

  - Bump the crate MSRV, pinned toolchain, clippy MSRV, and contributor docs to Rust 1.96.0.
  - Align pinned developer tools and install Cargo tools through cache-cargo-install-action where appropriate.
  - Add the Zizmor workflow and tighten workflow/Semgrep policy for checkout credentials, GitHub script interpolation, doctest error handling, and subprocess
    wrappers.

  - Replace doctest and test assert-matches patterns with std::assert_matches! / assert_matches! diagnostics.

- Harden support tooling for Python 3.13 [#433](https://github.com/acgetchell/delaunay/pull/433)
  [`289b9b7`](https://github.com/acgetchell/delaunay/commit/289b9b76cadda9201748357c536031131a8ddb26)

  - Move Python support scripts to the Python 3.13 baseline and let Ruff/Ty infer the configured target.
  - Parse SARIF, Criterion estimates, benchmark metrics, and baseline fetch options into validated boundary models before use.
  - Align changelog archive and postprocessing helpers with sibling repository behavior for archive links and release-heading detection.
  - Add repository-owned Semgrep guardrails for strict CI JSON handling and positive benchmark metric counts.

- Bump codecov/codecov-action from 6.0.1 to 7.0.0 [#457](https://github.com/acgetchell/delaunay/pull/457)
  [`7b017a0`](https://github.com/acgetchell/delaunay/commit/7b017a041090918efbc2bda0ee786070353a4605)

  Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 6.0.1 to 7.0.0.

  - [Release notes](https://github.com/codecov/codecov-action/releases)
  - [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/codecov/codecov-action/compare/e79a6962e0d4c0c17b229090214935d2e33f8354...fb8b3582c8e4def4969c97caa2f19720cb33a72f)

---

  updated-dependencies:

- dependency-name: codecov/codecov-action
  dependency-version: 7.0.0
  dependency-type: direct:production
  update-type: version-update:semver-major
  ...

- Bump actions/checkout from 6.0.2 to 6.0.3 [#456](https://github.com/acgetchell/delaunay/pull/456)
  [`b41393e`](https://github.com/acgetchell/delaunay/commit/b41393e0f6598e3598e1d533a7ca59f5bb195b46)

  Bumps [actions/checkout](https://github.com/actions/checkout) from 6.0.2 to 6.0.3.

  - [Release notes](https://github.com/actions/checkout/releases)
  - [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/actions/checkout/compare/de0fac2e4500dabe0009e67214ff5f5447ce83dd...df4cb1c069e1874edd31b4311f1884172cec0e10)

---

  updated-dependencies:

- dependency-name: actions/checkout
  dependency-version: 6.0.3
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...
- Harden GHAS security workflows [`add3dea`](https://github.com/acgetchell/delaunay/commit/add3deab592415e28f95faeb197bac099facf792)

  - Restrict manual performance-baseline materialization to main and semver tags instead of checking out arbitrary validated branches.

  - Pin CodeQL actions to the documented v4.36.2 commit so hash comments verify cleanly.

  - Rename benchmark Cargo-mode metadata to avoid CodeQL treating benchmark profile strings as clear-text secret storage.
- Avoid CodeQL benchmark metadata false positive [`9b16822`](https://github.com/acgetchell/delaunay/commit/9b1682299f0254c42f7c305dd4fc889d1ade862e)

  - Rename the internal benchmark build-flavor constant so CodeQL no longer treats the perf label as sensitive data.
  - Preserve baseline metadata output and benchmark command behavior.
- Bump the uv group across 1 directory with 3 updates [#462](https://github.com/acgetchell/delaunay/pull/462)
  [`4fbc838`](https://github.com/acgetchell/delaunay/commit/4fbc8383c7ea07fef943e3e0fe677d819b1d4761)

  Bumps the uv group with 3 updates in the / directory: [cryptography](https://github.com/pyca/cryptography) ,
  [python-multipart](https://github.com/Kludex/python-multipart) and [starlette](https://github.com/Kludex/starlette) .

  Updates `cryptography` from 46.0.7 to 48.0.1

  - [Changelog](https://github.com/pyca/cryptography/blob/main/CHANGELOG.rst)
  - [Commits](https://github.com/pyca/cryptography/compare/46.0.7...48.0.1)

  Updates `python-multipart` from 0.0.27 to 0.0.31

  - [Release notes](https://github.com/Kludex/python-multipart/releases)
  - [Changelog](https://github.com/Kludex/python-multipart/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/Kludex/python-multipart/compare/0.0.27...0.0.31)

  Updates `starlette` from 1.0.1 to 1.3.1

  - [Release notes](https://github.com/Kludex/starlette/releases)
  - [Changelog](https://github.com/Kludex/starlette/blob/main/docs/release-notes.md)
  - [Commits](https://github.com/Kludex/starlette/compare/1.0.1...1.3.1)

---

  updated-dependencies:

- dependency-name: cryptography
  dependency-version: 48.0.1
  dependency-type: indirect
  dependency-group: uv

- dependency-name: python-multipart
  dependency-version: 0.0.31
  dependency-type: indirect
  dependency-group: uv

- dependency-name: starlette
  dependency-version: 1.3.1
  dependency-type: indirect
  dependency-group: uv
  ...

### Performance

- Expand public flip benchmark coverage [#432](https://github.com/acgetchell/delaunay/pull/432)
  [`ea2f580`](https://github.com/acgetchell/delaunay/commit/ea2f5806dfa9fb2b8cb2bc072a7992c71f5fba32)

  - Add stable 2D, 3D, and 5D PL-manifold fixtures for public bistellar flip benchmarks.
  - Extend the benchmark manifest to cover k=1, k=2, and k=3 flip workflows across 2D-5D.
  - Document the all-platform `just ci` timing baseline for future CI-shape evaluation.

#### Performance: Add release baselines and adversarial flips

- Archive perf-profile Criterion baselines as GitHub Release assets and compare CI runs against the latest released Ubuntu baseline.
- Add local ref comparison support with `just perf-vs-ref` and baseline packaging from existing Criterion results.
- Expand public bistellar flip benchmarks with stable and adversarial fixtures that enforce exact n=1 roundtrip recovery.
- Document release benchmark storage, local same-machine comparisons, and the flip ergodicity invariant.

#### Maintenance: Isolate Semgrep test settings

- Route `semgrep-test` through a temporary Semgrep settings file to avoid runner home-directory permission failures.

#### Performance: Harden release benchmarks and ridge flip coverage

- Compare CI performance against the latest release benchmark asset and keep manual baseline artifacts for ad-hoc parity checks.
- Fail release benchmark summaries when fresh runs fall back to reference data, and require ci_performance_suite results before writing baselines.
- Add validated ridge-star support so k=3 flip benchmarks inspect the full incident support with typed ridge errors.
- Align Codacy Markdownlint exclusions with the repository release checklist policy.

## [0.7.8] - 2026-05-21

### ⚠️ Breaking Changes

- Support periodic flip parity for external cells [#391](https://github.com/acgetchell/delaunay/pull/391)
- Split strict and best-effort insertion statistics [#405](https://github.com/acgetchell/delaunay/pull/405)
- Replace public core module with focused facades [#392](https://github.com/acgetchell/delaunay/pull/392)
- Rename public cell APIs to simplex nomenclature [#393](https://github.com/acgetchell/delaunay/pull/393)
- Flatten triangulation modules into focused APIs [#399](https://github.com/acgetchell/delaunay/pull/399)
- Reconcile topology validation policy [#385](https://github.com/acgetchell/delaunay/pull/385) [#404](https://github.com/acgetchell/delaunay/pull/404)
- Box Delaunay repair flip errors [#407](https://github.com/acgetchell/delaunay/pull/407)
- Run slow correctness cases through slow-tests [#412](https://github.com/acgetchell/delaunay/pull/412)

### Merged Pull Requests

- Roll back failed topology repair [#418](https://github.com/acgetchell/delaunay/pull/418)
- Validate compact 3D toroidal quotients [#417](https://github.com/acgetchell/delaunay/pull/417)
- Enforce 10-second default test budget [#415](https://github.com/acgetchell/delaunay/pull/415)
- Run slow correctness cases through slow-tests [#412](https://github.com/acgetchell/delaunay/pull/412)
- Isolate strict insphere consistency control [#383](https://github.com/acgetchell/delaunay/pull/383) [#411](https://github.com/acgetchell/delaunay/pull/411)
- Use typed errors in public examples [#365](https://github.com/acgetchell/delaunay/pull/365) [#410](https://github.com/acgetchell/delaunay/pull/410)
- Prefer builder-based fallible examples [#214](https://github.com/acgetchell/delaunay/pull/214) [#409](https://github.com/acgetchell/delaunay/pull/409)
- Box Delaunay repair flip errors [#407](https://github.com/acgetchell/delaunay/pull/407)
- Split strict and best-effort insertion statistics [#405](https://github.com/acgetchell/delaunay/pull/405)
- Reconcile topology validation policy [#385](https://github.com/acgetchell/delaunay/pull/385) [#404](https://github.com/acgetchell/delaunay/pull/404)
- Localize remove_vertex repair [#401](https://github.com/acgetchell/delaunay/pull/401)
- Bump idna in the uv group across 1 directory [#400](https://github.com/acgetchell/delaunay/pull/400)
- Flatten triangulation modules into focused APIs [#399](https://github.com/acgetchell/delaunay/pull/399)
- Bump codecov/codecov-action from 6.0.0 to 6.0.1 [#398](https://github.com/acgetchell/delaunay/pull/398)
- Bump sysinfo in the dependencies group across 1 directory [#397](https://github.com/acgetchell/delaunay/pull/397)
- Bump taiki-e/install-action from 2.77.5 to 2.79.1 [#396](https://github.com/acgetchell/delaunay/pull/396)
- Surface squash-body commit entries [#395](https://github.com/acgetchell/delaunay/pull/395)
- Replace Node markdown tooling with rumdl [#394](https://github.com/acgetchell/delaunay/pull/394)
- Rename public cell APIs to simplex nomenclature [#393](https://github.com/acgetchell/delaunay/pull/393)
- Replace public core module with focused facades [#392](https://github.com/acgetchell/delaunay/pull/392)
- Support periodic flip parity for external cells [#391](https://github.com/acgetchell/delaunay/pull/391)
- Refactor/387 tds mutation boundaries [#390](https://github.com/acgetchell/delaunay/pull/390)
- Refresh release docs and benchmark guidance [#389](https://github.com/acgetchell/delaunay/pull/389)

### Added

- [**breaking**] Support periodic flip parity for external cells [#391](https://github.com/acgetchell/delaunay/pull/391)
  [`5fb2d4a`](https://github.com/acgetchell/delaunay/commit/5fb2d4a22927231ba3396950cf2bbbc481f69285)

  - Preserve periodic vertex offsets when bistellar flips build replacement cells.
  - Align external-facet parity checks across periodic cell frames instead of rejecting periodic external cells.
  - Surface replacement periodic-offset shape and frame conflicts with typed flip-context errors.
- [**breaking**] Split strict and best-effort insertion statistics [#405](https://github.com/acgetchell/delaunay/pull/405)
  [`71336b5`](https://github.com/acgetchell/delaunay/commit/71336b52ffad7ad6923ea0268928a700e37c43e3)

  - Make insert_with_statistics return typed insertion errors for skipped
    duplicate or retry-exhausted vertices so callers using ? cannot silently
    ignore skipped inputs.

  - Add insert_best_effort_with_statistics for diagnostic and bulk-ingestion
    workflows that intentionally preserve skipped insertions as outcomes with
    telemetry.

  - Derive the default validation policy from the active topology guarantee so
    PLManifold starts in ExplicitOnly mode while Pseudomanifold keeps
    OnSuspicion.

  - Document skipped-input observability across construction, workflows, and
    robustness guidance.

### Changed

- Refactor/387 tds mutation boundaries [#390](https://github.com/acgetchell/delaunay/pull/390)
  [`da30293`](https://github.com/acgetchell/delaunay/commit/da3029302e8484d26d2a88d1291050c332e6b822)
- [**breaking**] Replace public core module with focused facades [#392](https://github.com/acgetchell/delaunay/pull/392)
  [`655ff4c`](https://github.com/acgetchell/delaunay/commit/655ff4c944ac918c1a88ab70b26908bd1ebd329f)

  - Make `crate::core` private and expose low-level APIs through curated
    `tds`, `collections`, `algorithms`, and `query` modules.

  - Add focused prelude/docs coverage for the new public import paths.
  - Update downstream-style tests and doctests to stop relying on
    `delaunay::core`.
- [**breaking**] Rename public cell APIs to simplex nomenclature [#393](https://github.com/acgetchell/delaunay/pull/393)
  [`48935d5`](https://github.com/acgetchell/delaunay/commit/48935d5c1ecbeef1b9b833b1de0d1cbcc2e72938)

  - Replace cell-oriented public types, methods, error variants, maps, and diagnostics with simplex terminology across TDS, triangulation, construction, repair,
    flips, examples, benches, docs, and tests.

  - Preserve typed construction and insertion error context for cavity filling, explicit construction summaries, focused preludes, and Delaunay repair
    diagnostics under the simplex API.

  - Remove redundant integration tests whose coverage is now exercised by unit, property, and public API tests.
  - Move Codacy SARIF filtering into a tested support script and simplify Cargo package metadata.
- [**breaking**] Flatten triangulation modules into focused APIs [#399](https://github.com/acgetchell/delaunay/pull/399)
  [`774fd1b`](https://github.com/acgetchell/delaunay/commit/774fd1b92756a070d945061eff203fbc80563dbe)

  - Move Delaunay-facing APIs from the old triangulation facade to crate-root
    modules and focused preludes such as construction, insertion, flips,
    repair, validation, and delaunayize.

  - Split generic Triangulation behavior into construction, insertion, query,
    orientation, repair, and validation modules with colocated tests and errors.

  - Keep the broad prelude for exploratory use while making focused preludes
    orthogonal and workflow-specific.

  - Refresh docs, examples, benchmarks, and API export tests for the new module
    layout and 7,500-vertex 3D debug-scale default.
- [**breaking**] Reconcile topology validation policy [#385](https://github.com/acgetchell/delaunay/pull/385)
  [#404](https://github.com/acgetchell/delaunay/pull/404) [`e5a74a1`](https://github.com/acgetchell/delaunay/commit/e5a74a10586b162fad56d7f227ab194997f7a87c)

  - Add explicit caller-owned validation mode for PL-manifold topology guarantees.
  - Reject incoherent topology guarantee and validation policy pairings through typed fallible setters.
  - Keep compatibility setters non-committal when a requested pairing is invalid.
  - Derive builder validation policy from the selected topology guarantee and document the compatibility matrix.
- [**breaking**] Box Delaunay repair flip errors [#407](https://github.com/acgetchell/delaunay/pull/407)
  [`1789ceb`](https://github.com/acgetchell/delaunay/commit/1789cebd4f56b4dc09e014482d8d90176fdaeffc)

  - Box `DelaunayRepairError::Flip` sources while preserving typed `FlipError`
    inspection through `From&lt;FlipError&gt;` and `Error::source`.

  - Update repair and construction mappings for the named boxed variant.
  - Cover clone and source behavior for wrapped linear-algebra errors.
- Isolate strict insphere consistency control [#383](https://github.com/acgetchell/delaunay/pull/383) [#411](https://github.com/acgetchell/delaunay/pull/411)
  [`222e572`](https://github.com/acgetchell/delaunay/commit/222e572b4364007cdbf6d1fa6428ea7e670cfb56)

  - Document the strict insphere consistency environment knob as a process-wide
    once-per-process snapshot.

  - Add a thread-local test override guard so strict diagnostic paths can be
    exercised without mutating global environment state.

  - Mark the production review checklist item complete.
- [**breaking**] Run slow correctness cases through slow-tests [#412](https://github.com/acgetchell/delaunay/pull/412)
  [`1498153`](https://github.com/acgetchell/delaunay/commit/149815313530335a98f6951c63924e4774d509c1)

  - Define the slow-test bucket around deterministic correctness tests that exceed the default-suite budget.
  - Move runnable high-dimensional properties out of ignored tests and into either the default suite or the slow-tests feature.
  - Give just test-slow a release-mode nextest profile with a longer watchdog for intentional multi-minute regressions.
  - Add a repository Semgrep guard against reintroducing slow ignored tests.

#### Changed: Enforce explicit test buckets

- Sort correctness tests into default and slow-tests buckets instead of relying on ignored tests.
- Move benchmark-style boundary and UUID iterator measurements into a Criterion benchmark target.
- Replace ignored flaky or known-failure cases with active assertions or slow-tests gating.
- Add a Semgrep guard against reintroducing ignored tests and align docs and helper recipes with the new taxonomy.
- Enforce 10-second default test budget [#415](https://github.com/acgetchell/delaunay/pull/415)
  [`ba15de8`](https://github.com/acgetchell/delaunay/commit/ba15de8e97ab3dace8a2f5e0443f473930d1d344)

  - Gate default-suite cases at or above the 10-second budget behind
    slow-tests and remove obsolete high-dimensional periodic validation from
    routine runs.

  - Move allocation hot-path contracts into a Criterion benchmark over
    calibrated 2D-5D fixtures, leaving allocation_api as wiring smoke coverage.

  - Document the toroidal validation limits and add the bench-allocations
    workflow.

### Documentation

- Refresh release docs and benchmark guidance [#389](https://github.com/acgetchell/delaunay/pull/389)
  [`526583c`](https://github.com/acgetchell/delaunay/commit/526583c627d32d9336910bce1913b7f458ca413c)

  - Update the README pitch, feature list, references, and docs.rs-facing guidance
    around exact predicates, SoS, PL-manifold validation, and bistellar repair.

  - Refresh roadmap, release, limitation, robustness, orientation, invariant,
    diagnostics, property-testing, workflow, and validation docs for the v0.7.8
    cleanup path and v0.8.0 paper-facing work.

  - Align contributor and script docs around non-mutating `just` checks before
    mutating fixes.

  - Update generated benchmark-summary guidance to surface `just bench-perf-summary`,
    current Criterion metadata, and large-scale characterization defaults.
- Prefer builder-based fallible examples [#214](https://github.com/acgetchell/delaunay/pull/214) [#409](https://github.com/acgetchell/delaunay/pull/409)
  [`238dc12`](https://github.com/acgetchell/delaunay/commit/238dc124f6fb930cbeee31ea07c16d017f1822bd)

  - Present `DelaunayTriangulationBuilder` as the primary construction path while retaining `DelaunayTriangulation::new` as a legacy convenience constructor.
  - Replace doctest `unwrap()` patterns with typed `?` propagation and explicit optional guards across construction, validation, repair, geometry, and topology
    examples.

  - Re-export `DelaunayTriangulationBuilder` from `prelude::delaunayize` so single-prelude delaunayize examples can use the builder directly.
- Use typed errors in public examples [#365](https://github.com/acgetchell/delaunay/pull/365) [#410](https://github.com/acgetchell/delaunay/pull/410)
  [`15633e2`](https://github.com/acgetchell/delaunay/commit/15633e2c03d2b2b9af1532676f0ce2c1292a538c)

  - Replace public doctest unwraps and expects with typed Result examples across core and geometry APIs.
  - Use concrete crate errors or small local thiserror enums in guides instead of boxed dynamic errors.
  - Clarify toroidal builder wording and point the README at the workflow docs for periodic construction.

### Fixed

- Surface squash-body commit entries [#395](https://github.com/acgetchell/delaunay/pull/395)
  [`44dc3ed`](https://github.com/acgetchell/delaunay/commit/44dc3ede1012ed4b1f3a560b3c724e2e21d5169e)

  - Mirror conventional pseudo-commit headings from squash merge bodies into their matching changelog sections while preserving the primary PR entry.
  - Regenerate active and archived changelogs so historical squash-body fixes, docs, maintenance, and performance notes appear under the right headings.
  - Resolve production-review hygiene by clarifying Cargo feature gates, relabeling TDS UUID-map invariants, and recording UUID panic paths as test-only.
- Validate compact 3D toroidal quotients [#417](https://github.com/acgetchell/delaunay/pull/417)
  [`3456ca9`](https://github.com/acgetchell/delaunay/commit/3456ca96abeeee5efb2ef413c3eb91d57f3921b4)

  - Validate periodic image-point quotients through final Levels 1-3 topology and Level 4 Delaunay checks before returning them.
  - Preserve lifted periodic vertex identity in Euler and manifold validation so 3D quotient links and ridges are checked in the correct lattice frame.
  - Select periodic quotient candidates by circumcenter ownership instead of barycenter ownership.
  - Add a typed UnsupportedPeriodicDimension guardrail so 4D/5D periodic quotient construction fails fast pending scalable follow-up work in #416.
  - Refresh toroidal periodic docs to state the 2D/compact 3D release boundary and high-dimensional guardrails.
- Roll back failed topology repair [#418](https://github.com/acgetchell/delaunay/pull/418)
  [`8a9cbf3`](https://github.com/acgetchell/delaunay/commit/8a9cbf3510ebc16b884951791f2de20412cf6c36)

  - Restore the incoming triangulation when topology repair fails before fallback rebuild can recover.
  - Add a regression for partial topology-repair mutation that exhausts its iteration budget.
  - Update the v0.7.8 release metadata, roadmap, benchmark summary, changelog, and citation data.
  - Tighten the doctest `.expect(` release guard to the current zero-count baseline.

### Maintenance

- Replace Node markdown tooling with rumdl [#394](https://github.com/acgetchell/delaunay/pull/394)
  [`5654d14`](https://github.com/acgetchell/delaunay/commit/5654d14bc04dba538d776800495846d0a6ea94da)

  - Switch Markdown checks and Codacy configuration from markdownlint/npx to rumdl.
  - Add dprint/pretty_yaml YAML formatting and wire yaml-check/yaml-fix through dprint plus yamllint.
  - Format generated changelog archives with rumdl after git-cliff postprocessing.
  - Add Semgrep guards for check-before-fix docs and pinned, allowlisted GitHub Actions.
- Bump codecov/codecov-action from 6.0.0 to 6.0.1 [#398](https://github.com/acgetchell/delaunay/pull/398)
  [`325aa94`](https://github.com/acgetchell/delaunay/commit/325aa941ba9adaca7973c1f2a3f623c88b9a5d27)

  Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 6.0.0 to 6.0.1.

  - [Release notes](https://github.com/codecov/codecov-action/releases)
  - [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/codecov/codecov-action/compare/57e3a136b779b570ffcdbf80b3bdc90e7fab3de2...e79a6962e0d4c0c17b229090214935d2e33f8354)

---

  updated-dependencies:

- dependency-name: codecov/codecov-action
  dependency-version: 6.0.1
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump sysinfo in the dependencies group across 1 directory [#397](https://github.com/acgetchell/delaunay/pull/397)
  [`8800150`](https://github.com/acgetchell/delaunay/commit/88001505822ed949c5075f9000a7f2e114bcf089)

  Bumps the dependencies group with 1 update in the / directory: [sysinfo](https://github.com/GuillaumeGomez/sysinfo).

  Updates `sysinfo` from 0.39.1 to 0.39.2

  - [Changelog](https://github.com/GuillaumeGomez/sysinfo/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/GuillaumeGomez/sysinfo/compare/v0.39.1...v0.39.2)

---

  updated-dependencies:

- dependency-name: sysinfo
  dependency-version: 0.39.2
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies
  ...

- Bump taiki-e/install-action from 2.77.5 to 2.79.1 [#396](https://github.com/acgetchell/delaunay/pull/396)
  [`d89961c`](https://github.com/acgetchell/delaunay/commit/d89961cfdc4af5c354c78f4e5fe008f1f273c6e2)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.77.5 to 2.79.1.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/fa0dd4cd0a40696e6f9766370614a5ce482e6aa8...b550161ef8a7bc4f2a671c0b03a18ac9ccedea1e)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.79.1
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump idna in the uv group across 1 directory [#400](https://github.com/acgetchell/delaunay/pull/400)
  [`dadb724`](https://github.com/acgetchell/delaunay/commit/dadb7248243e2b320095c647d8314fd952c116e0)

  Bumps the uv group with 1 update in the / directory: [idna](https://github.com/kjd/idna).

  Updates `idna` from 3.13 to 3.15

  - [Release notes](https://github.com/kjd/idna/releases)
  - [Changelog](https://github.com/kjd/idna/blob/master/HISTORY.md)
  - [Commits](https://github.com/kjd/idna/compare/v3.13...v3.15)

---

  updated-dependencies:

- dependency-name: idna
  dependency-version: '3.15'
  dependency-type: indirect
  dependency-group: uv
  ...

- Run Windows tests with cargo-nextest [`bc5d48f`](https://github.com/acgetchell/delaunay/commit/bc5d48f61012d49224d783f6c49f88024948959c)

  - Install and verify the pinned cargo-nextest version across the CI build matrix.
  - Run Windows library and release integration tests through nextest while keeping doctests on cargo test.
  - Document the Windows nextest alignment in the tooling notes.

### Performance

- Localize remove_vertex repair [#401](https://github.com/acgetchell/delaunay/pull/401)
  [`382f9e1`](https://github.com/acgetchell/delaunay/commit/382f9e13cff0f4a325361c056ea1d59be18003c5)

  - Avoid global simplex scans and incident-simplex rebuilds on successful
    vertex removal by using the affected vertex star and scoped repair.

  - Promote and validate orientation over the touched removal scope, with the
    full global orientation path retained as a fallback.

  - Add a Criterion remove_vertex benchmark covering successful removals and
    invalid-remnant rollback across 2D through 5D.

#### Performance: Stress remove_vertex cases

- Add adversarial remove_vertex benchmark fixtures for near-boundary,
  cospherical, near-degenerate, and large-coordinate point sets.

- Include the selected fixture kind in Criterion case names so benchmark
  output shows which geometry each run exercises.

- Name the vertex-removal orientation normalization budget.

#### Changed: Cover vertex-removal helper invariants

  Add focused regression coverage for vertex-removal helper behavior:

- affected-vertex collection and missing-simplex error preservation
- local validation-scope deduplication and live-simplex filtering
- incident-simplex repair, isolated-vertex reporting, and postcondition failure
- fan-boundary filtering for facets that already contain the apex

#### Fixed: Bound local facet repair removals

- Preserve local repair removal budgets before mutating the TDS.
- Reuse scoped neighbor repair after vertex-removal facet cleanup.
- Install and verify cargo-nextest for faster CI test recipes.
- Build examples once before running their compiled binaries.

## [0.7.7] - 2026-05-15

### ⚠️ Breaking Changes

- Harden Delaunay invariants and tooling [#362](https://github.com/acgetchell/delaunay/pull/362)
- Harden typed validation and exact predicate APIs [#375](https://github.com/acgetchell/delaunay/pull/375)
- Harden correctness and performance invariants [#376](https://github.com/acgetchell/delaunay/pull/376)
- Preserve topology and public error invariants [#363](https://github.com/acgetchell/delaunay/pull/363)
- Quiet Codacy code scanning noise [#356](https://github.com/acgetchell/delaunay/pull/356)
- Cadence batch repair for large construction [#369](https://github.com/acgetchell/delaunay/pull/369)

### Merged Pull Requests

- Stabilize large-scale profiling and construction [#377](https://github.com/acgetchell/delaunay/pull/377)
- Harden correctness and performance invariants [#376](https://github.com/acgetchell/delaunay/pull/376)
- Harden typed validation and exact predicate APIs [#375](https://github.com/acgetchell/delaunay/pull/375)
- Bump urllib3 in the uv group across 1 directory [#374](https://github.com/acgetchell/delaunay/pull/374)
- Bump actions-rust-lang/setup-rust-toolchain [#373](https://github.com/acgetchell/delaunay/pull/373)
- Bump sysinfo in the dependencies group [#372](https://github.com/acgetchell/delaunay/pull/372)
- Bump taiki-e/install-action from 2.76.0 to 2.77.5 [#371](https://github.com/acgetchell/delaunay/pull/371)
- Cadence batch repair for large construction [#369](https://github.com/acgetchell/delaunay/pull/369)
- Cover fallback rebuild and flip roundtrips [#368](https://github.com/acgetchell/delaunay/pull/368)
- Harden SARIF tooling [#367](https://github.com/acgetchell/delaunay/pull/367)
- Repair insertion neighbors locally [#335](https://github.com/acgetchell/delaunay/pull/335) [#366](https://github.com/acgetchell/delaunay/pull/366)
- Preserve topology and public error invariants [#363](https://github.com/acgetchell/delaunay/pull/363)
- Harden Delaunay invariants and tooling [#362](https://github.com/acgetchell/delaunay/pull/362)
- Bump python-multipart in the uv group across 1 directory [#361](https://github.com/acgetchell/delaunay/pull/361)
- Bump taiki-e/install-action from 2.75.26 to 2.76.0 [#360](https://github.com/acgetchell/delaunay/pull/360)
- Quiet Codacy code scanning noise [#356](https://github.com/acgetchell/delaunay/pull/356)
- Enable repo Semgrep rules for issue #338 [#354](https://github.com/acgetchell/delaunay/pull/354)
- Type repair diagnostics and harden invariants [#332](https://github.com/acgetchell/delaunay/pull/332) [#352](https://github.com/acgetchell/delaunay/pull/352)
- Harden Python benchmark parsing [#351](https://github.com/acgetchell/delaunay/pull/351)
- Expand profiling benchmarks around public API workflows [#349](https://github.com/acgetchell/delaunay/pull/349)
- Bump taiki-e/install-action from 2.75.18 to 2.75.22 [#348](https://github.com/acgetchell/delaunay/pull/348)

### Changed

- Type repair diagnostics and harden invariants [#332](https://github.com/acgetchell/delaunay/pull/332) [#352](https://github.com/acgetchell/delaunay/pull/352)
  [`a244053`](https://github.com/acgetchell/delaunay/commit/a2440531ae7ee1407e7436379a82fe092f02e7dd)

  - Replace stringified flip-repair skip samples with typed diagnostic context.
  - Make vertex removal transactional across post-removal repair and orientation canonicalization.
  - Deprecate DelaunayTriangulation::as_triangulation_mut ahead of removal in v0.8.0.
  - Use scale-aware degeneracy checks for low-dimensional simplex and facet measures.
  - Add regression and property coverage for rollback behavior, typed diagnostics, and scaled valid measures.
  - Tolerate throughput formatting precision in benchmark baseline round-trip tests.
- [**breaking**] Harden Delaunay invariants and tooling [#362](https://github.com/acgetchell/delaunay/pull/362)
  [`0c1f477`](https://github.com/acgetchell/delaunay/commit/0c1f477658ffbad2198cb55b76089dadfdba0a6d)

  - Enforce checked Delaunay construction and reconstruction so explicit connectivity, deserialization, and TDS wrapping cannot produce invalid
    DelaunayTriangulation values.

  - Replace stringly insertion and construction failures with typed, matchable errors, and route topology/TDS failures without production panics.
  - Tighten public API names, trait bounds, prelude exports, docs, and doctests around mutation, reconstruction, construction statistics, and feature-gated
    behavior.

  - Standardize on DenseSlotMap storage, remove SlotMap backend comparison tooling, and align CI, docs.rs, changelog, benchmark, Semgrep, and release workflows.
  - Harden benchmark comparison parsing so individual regressions and malformed baseline sections fail explicitly.
- Cover fallback rebuild and flip roundtrips [#368](https://github.com/acgetchell/delaunay/pull/368)
  [`aa0de2c`](https://github.com/acgetchell/delaunay/commit/aa0de2cf8892b8caecfc3e8a3b04e3f4eefa23b5)

  - Add dimension-generic flip roundtrip properties for public k=1 and internal k=2/k=3 paths.
  - Exercise fallback rebuilds after supported Delaunay repair failures and duplicate-simplex topology failures.
  - Cover circumradius agreement between direct and precomputed-center paths.
  - Exclude debug-only circumsphere logging branches from LCOV.
- [**breaking**] Harden typed validation and exact predicate APIs [#375](https://github.com/acgetchell/delaunay/pull/375)
  [`e187cf0`](https://github.com/acgetchell/delaunay/commit/e187cf06761308b96a98f524c32bd3cf712250f8)

  - Replace stringly validation and repair contexts with typed summaries and failure categories across insertion, flips, TDS, builder, and Delaunay validation
    paths.

  - Tighten kernel, coordinate, and data trait contracts so exact predicates are dimension-scoped and payload bounds apply only where needed.
  - Split explicit-construction validation into orthogonal typed errors and preserve structured source details through fallback and repair paths.
  - Narrow focused prelude exports, add repository style rules, and document the 10,000-vertex 3D acceptance envelope.
- [**breaking**] Harden correctness and performance invariants [#376](https://github.com/acgetchell/delaunay/pull/376)
  [`4c6d55e`](https://github.com/acgetchell/delaunay/commit/4c6d55e12ff60155c97e117aefabc5ff80d2f47f)

  - Replace scalar-unit duplicate detection with scale-aware tolerances and keep the spatial index usable across rollback-safe insertion and removal paths.
  - Route matrix dispatch, predicate degeneracy, non-finite geometry, and robust insphere overflow through typed errors and exact relative-coordinate
    predicates.

  - Preserve TDS identity across transactional rollback, run required pseudomanifold validation during insertion, and remove raw cell-storage mutation.
  - Use DoS-resistant hashing for coordinate-derived buckets and gate insertion timing telemetry to callers that consume it.
  - Normalize root and archived changelog markdown through shared post-processing.
  - Simplify validation telemetry, predicate helpers, cache invalidation, and prelude coverage without changing the correctness/performance contracts.

### Fixed

- Harden Python benchmark parsing [#351](https://github.com/acgetchell/delaunay/pull/351)
  [`fea5898`](https://github.com/acgetchell/delaunay/commit/fea58987a2fb84e47603be9f0d1960aaa7e0f5f0)

  - Reject non-finite and unordered Criterion timing estimates before using them in summaries, baselines, or backend comparisons.
  - Preserve full Criterion benchmark IDs and normalize timing units when comparing storage backend results.
  - Reuse the shared baseline parser while preserving malformed-section skip behavior and supporting scientific notation.
  - Fall back from unusable lscpu output to /proc CPU core detection on Linux.
  - Add regression and round-trip tests for parser behavior, benchmark IDs, unit normalization, and Linux CPU fallback.
  - Document Python parser/file-format round-trip test expectations.

#### Fixed: Harden Criterion estimate parsing and validation

  Consolidates estimate validation into a single public helper,
  `is_valid_criterion_estimate`, now used by `PerformanceSummaryGenerator`
  and `StorageBackendComparator`. Adds explicit type checks to
  `PerformanceSummaryGenerator` to reject structurally malformed JSON
  data, improving parsing robustness.

- [**breaking**] Preserve topology and public error invariants [#363](https://github.com/acgetchell/delaunay/pull/363)
  [`83d9619`](https://github.com/acgetchell/delaunay/commit/83d96197cc5f7b2b2242904b7bab76ac9e058e6c)

  - Make duplicate-cell removal and bistellar flip mutation fail before committing invalid TDS state.
  - Preserve typed topology, validation, and insertion errors through retry and flip-wiring paths.
  - Enforce typed error handling in public examples, benchmarks, and public API integration tests.

  - Rebuild and validate TDS topology after duplicate-cell removal so neighbor and incident-cell links stay consistent.
  - Preserve flip validation, wiring, and repair failure kinds instead of flattening them into coarse mutation errors.
  - Propagate construction failures from examples and route benchmark diagnostics through feature-gated tracing.
  - Clarify public API docs for collection aliases and facet error paths.

#### Fixed: Make duplicate-cell removal transactional

- Validate duplicate-cell removals on a cloned TDS and commit only after neighbor rebuilding, incident-cell assignment, and full validation succeed.
- Preserve typed construction and mutation errors in examples and doctests, including prelude coverage for TdsMutationError.
- Share feature-gated tracing abort helpers across benchmarks and keep setup error handling out of measured closures.
- Remove committed debug printing from collection tests.

### Maintenance

- Bump taiki-e/install-action from 2.75.18 to 2.75.22 [#348](https://github.com/acgetchell/delaunay/pull/348)
  [`31ec720`](https://github.com/acgetchell/delaunay/commit/31ec720a8638103b9acd7ea58c35b2baa5f571b9)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.75.18 to 2.75.22.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/055f5df8c3f65ea01cd41e9dc855becd88953486...cf525cb33f51aca27cd6fa02034117ab963ff9f1)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.75.22
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Enable repo Semgrep rules for issue #338 [#354](https://github.com/acgetchell/delaunay/pull/354)
  [`9d51d30`](https://github.com/acgetchell/delaunay/commit/9d51d3038ae3f3316102ecaf39429bdfb73ff1cc)

  - Enable project-owned Semgrep rules in local checks, CodeRabbit, and Codacy/OpenGrep scanning.
  - Harden Semgrep execution with strict mode, a higher timeout, and fixture coverage for hot-path hash collection rules.
  - Replace flagged diagnostics and silent numeric fallbacks with explicit tracing, expectations, and typed Hilbert quantization errors.
  - Centralize Delaunay triangulation cache invalidation through the existing repair-cache helper.

#### Maintenance: Enable repository Semgrep rules

- Rename the Semgrep config to semgrep.yaml and wire it into local checks, CodeRabbit, and Codacy/OpenGrep.
- Add strict Semgrep execution plus fixture coverage for hot-path hash collections and targeted panic bypasses.
- Make Hilbert errors non-exhaustive and document quantization-scale conversion failures on APIs that can return them.
- Replace fragile VertexBuilder expect paths with infallible Vertex point constructors.

#### Maintenance: Expand repository Semgrep rules

- Add project-specific Semgrep checks for Rust dynamic errors, lint suppression reasons, Python subprocess mocks, and typed script helpers.
- Add focused Semgrep fixtures for hot-path hash collections, Rust project rules, and Python test conventions.
- Wire the expanded Semgrep fixture suite into `just check`.
- Replace stale Clippy `allow` suppressions with documented `expect` attributes and remove dynamic error trait-object usage from tests.

#### Maintenance: Refresh quality tooling and diagnostics

- Pin GitHub workflow tool versions and update action SHAs for cache, artifact upload, install-action, and SARIF upload.
- Exclude Semgrep fixtures from Codacy analysis so intentional rule-test violations do not surface as production issues.
- Add a cargo-machete backed just unused-deps recipe for checking unused direct dependencies.
- Gate convex hull test diagnostics behind diagnostics tracing instead of unconditional stdout output.
- Add Hilbert ordering and zero-dimensional sort coverage for Codecov patch gaps.

#### Fixed: Harden Hilbert ordering errors and prelude checks [#338](https://github.com/acgetchell/delaunay/pull/338)

- Return typed Hilbert errors for non-finite quantization inputs and failed u32 coordinate conversions instead of silently collapsing values.
- Preserve item order when Hilbert sort key construction fails, and add regression coverage for the new error paths.
- Add the focused ordering prelude and update doctests, examples, benchmarks, and integration tests to use orthogonal prelude imports.
- Add a Semgrep rule and fixture coverage for examples and benchmarks that bypass focused preludes.
- Verify pinned shfmt binaries in CI with explicit SHA256 values instead of downloading a missing upstream checksum file.
- [**breaking**] Quiet Codacy code scanning noise [#356](https://github.com/acgetchell/delaunay/pull/356)
  [`5d8a9ff`](https://github.com/acgetchell/delaunay/commit/5d8a9ff3bfd260636a8fa4209cb00801c134aaa5)

  - Skip empty Codacy SARIF runs before uploading results to GitHub Code Scanning.
  - Limit Codacy code scanning uploads to repository-owned OpenGrep analysis.
  - Migrate triangulation doctests to focused prelude imports for #355.
  - Add Semgrep coverage to keep triangulation doctest imports on focused preludes.
  - Clarify vertex point-constructor docs and avoid UUID-bearing facet assertion output.

#### Added: Add focused preludes for doctest imports

- Add minimal, task-focused preludes for triangulation, TDS, geometry, queries, generators, ordering, collections, and topology workflows.
- Update doctests, examples, integration tests, and benchmarks to use focused preludes instead of deep internal imports.
- Expand the Semgrep doctest import rule to cover all Rust source files and hidden doctest imports.
- Harden Codacy SARIF handling so empty analyses skip upload cleanly instead of creating noisy code-scanning results.
- Document the new prelude structure and update guides to use public prelude imports.

#### Fixed: Gate public diagnostic exports behind diagnostics

- Replace profile-based public diagnostic exports with the documented diagnostics feature.
- Keep debug-only helper definitions and diagnostic call sites aligned with feature-gated API visibility.
- Allow hidden doctest prelude imports in the Semgrep prelude-import rule.

#### Changed: Rename diagnostics feature flag

- Use the diagnostics feature flag for opt-in diagnostic helpers and verbose test diagnostics.
- Add prelude::diagnostics for debug verification helpers while keeping focused preludes narrow and orthogonal.
- Move diagnostic helper exports out of unrelated focused preludes and update cfg gates, tests, docs, and just recipes.
- Document that focused preludes should favor precise taxonomy over backwards compatibility for unrelated re-exports.
- Bump taiki-e/install-action from 2.75.26 to 2.76.0 [#360](https://github.com/acgetchell/delaunay/pull/360)
  [`5b764e0`](https://github.com/acgetchell/delaunay/commit/5b764e0ab8c63223e63835b475d4829081db5e94)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.75.26 to 2.76.0.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/b651345a718c8f44efa2460560b3dbf29cbd7ee1...711e1c3275189d76dcc4d34ddea63bf96ac49090)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.76.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump python-multipart in the uv group across 1 directory [#361](https://github.com/acgetchell/delaunay/pull/361)
  [`af6e210`](https://github.com/acgetchell/delaunay/commit/af6e210cba687606c24c8c2a2ba519470d787abe)

  Bumps the uv group with 1 update in the / directory: [python-multipart](https://github.com/Kludex/python-multipart).

  Updates `python-multipart` from 0.0.26 to 0.0.27

  - [Release notes](https://github.com/Kludex/python-multipart/releases)
  - [Changelog](https://github.com/Kludex/python-multipart/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/Kludex/python-multipart/compare/0.0.26...0.0.27)

---

  updated-dependencies:

- dependency-name: python-multipart
  dependency-version: 0.0.27
  dependency-type: indirect
  dependency-group: uv
  ...
- Harden SARIF tooling [#367](https://github.com/acgetchell/delaunay/pull/367)
  [`e5b20cd`](https://github.com/acgetchell/delaunay/commit/e5b20cd78fd58ba1bc77c20e5076dc778304b917)

  - Add a repository-rule Semgrep SARIF workflow for direct Code Scanning uploads.
  - Harden Clippy SARIF generation with pipefail, cargo lint coverage, and guarded uploads.
  - Discover Semgrep rule fixtures dynamically so new fixtures are tested automatically.
  - Document the tooling parity updates ported from causal-triangulations.
- Bump taiki-e/install-action from 2.76.0 to 2.77.5 [#371](https://github.com/acgetchell/delaunay/pull/371)
  [`b7fa94c`](https://github.com/acgetchell/delaunay/commit/b7fa94cb7611c12c2277b16ac9c215e177e95f41)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.76.0 to 2.77.5.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/711e1c3275189d76dcc4d34ddea63bf96ac49090...fa0dd4cd0a40696e6f9766370614a5ce482e6aa8)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.77.5
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump urllib3 in the uv group across 1 directory [#374](https://github.com/acgetchell/delaunay/pull/374)
  [`6b9716a`](https://github.com/acgetchell/delaunay/commit/6b9716ac2ccd01e979fe10b72303694e362af579)

  Bumps the uv group with 1 update in the / directory: [urllib3](https://github.com/urllib3/urllib3).

  Updates `urllib3` from 2.6.3 to 2.7.0

  - [Release notes](https://github.com/urllib3/urllib3/releases)
  - [Changelog](https://github.com/urllib3/urllib3/blob/main/CHANGES.rst)
  - [Commits](https://github.com/urllib3/urllib3/compare/2.6.3...2.7.0)

---

  updated-dependencies:

- dependency-name: urllib3
  dependency-version: 2.7.0
  dependency-type: indirect
  dependency-group: uv
  ...

- Bump actions-rust-lang/setup-rust-toolchain [#373](https://github.com/acgetchell/delaunay/pull/373)
  [`3a9dabc`](https://github.com/acgetchell/delaunay/commit/3a9dabc1b84f804a86a0ae13ab16b7efc0a06d38)

  Bumps [actions-rust-lang/setup-rust-toolchain](https://github.com/actions-rust-lang/setup-rust-toolchain) from 1.16.0 to 1.16.1.

  - [Release notes](https://github.com/actions-rust-lang/setup-rust-toolchain/releases)
  - [Changelog](https://github.com/actions-rust-lang/setup-rust-toolchain/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/actions-rust-lang/setup-rust-toolchain/compare/2b1f5e9b395427c92ee4e3331786ca3c37afe2d7...46268bd060767258de96ed93c1251119784f2ab6)

---

  updated-dependencies:

- dependency-name: actions-rust-lang/setup-rust-toolchain
  dependency-version: 1.16.1
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump sysinfo in the dependencies group [#372](https://github.com/acgetchell/delaunay/pull/372)
  [`efc2bb7`](https://github.com/acgetchell/delaunay/commit/efc2bb700e51039407794db7e0b9b47591187e02)

  Bumps the dependencies group with 1 update: [sysinfo](https://github.com/GuillaumeGomez/sysinfo).

  Updates `sysinfo` from 0.39.0 to 0.39.1

  - [Changelog](https://github.com/GuillaumeGomez/sysinfo/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/GuillaumeGomez/sysinfo/compare/v0.39.0...v0.39.1)

---

  updated-dependencies:

- dependency-name: sysinfo
  dependency-version: 0.39.1
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies
  ...

### Performance

- Expand profiling benchmarks around public API workflows [#349](https://github.com/acgetchell/delaunay/pull/349)
  [`0acbf65`](https://github.com/acgetchell/delaunay/commit/0acbf651b57c287aecc10bd51eea55fdbcbe2442)

  - Run profiling comparisons with the checked-out crate toolchain by default
  - Add local `just profile` support for comparing code refs and compiler versions
  - Expand `ci_performance_suite` beyond construction to cover hulls, boundary traversal, validation, and bistellar flips
  - Emit a versioned API benchmark manifest so benchmark logs show which public workflows were measured
- Repair insertion neighbors locally [#335](https://github.com/acgetchell/delaunay/pull/335) [#366](https://github.com/acgetchell/delaunay/pull/366)
  [`f09d698`](https://github.com/acgetchell/delaunay/commit/f09d69802fd9ebc554b7df8745db1e6b9ca2126a)

  - Replace post-insertion global neighbor rebuilds with seeded local repair
    over new cells, removed-cell frontiers, and facet-issue survivors.

  - Preserve facet-compatible existing neighbor pointers while repairing
    empty, dangling, or stale local slots.

  - Add a release-mode escape hatch for forcing global neighbor rebuilds
    during A/B isolation.

  - Expose the low-level local repair helper through the insertion prelude
    and preserve typed neighbor-repair error sources.
- [**breaking**] Cadence batch repair for large construction [#369](https://github.com/acgetchell/delaunay/pull/369)
  [`a984b3a`](https://github.com/acgetchell/delaunay/commit/a984b3aa0c57a967b0ae6f33b88042af49ac52cb)

  - Add a batch repair policy option to ConstructionOptions so bulk insertion can repair local Delaunay fronts every N insertions.
  - Carry pending local repair seeds through batch construction and finalization so cadenced repairs cover the accumulated frontier.
  - Report construction telemetry for locate walks, conflict regions, global exterior scans, skipped-vertex budgets, and initial-simplex A/B runs in the
    large-scale debug harness.

#### Performance: Keep bulk repair seeding local

- Return live insertion-created cells as repair seeds so batch construction can
  accumulate local frontiers without scanning vertex stars globally.

- Move repair seed frontier helpers into triangulation locality utilities and
  defer broad repair fallback to final construction repair.

- Extend construction telemetry and large-scale debug output with seed
  accumulation timing.

#### Performance: Keep exterior repair seeding local

- Preserve the terminal locate cell so exterior insertion can seed nearby repair work without scanning the full triangulation.
- Skip global conflict-region scans during exterior hull extension and defer broad Delaunay cleanup to cadenced and final repair.
- Move local repair seed bookkeeping helpers into triangulation locality utilities.

#### Performance: Localize construction repair cadence

- Keep exterior repair seeding local when conflict buffers are empty, avoiding global conflict-region scans during insertion.
- Gate topology-validation telemetry on actual validation work and add typed validation cadences for large-scale diagnostics.
- Preserve typed Delaunay repair failures with construction repair phases and keep mutation repair gates independent of insertion cadence.
- Add focused triangulation construction, repair, and validation preludes across docs, examples, benches, and tests.

#### Performance: Trigger batch repair on seed backlog [#341](https://github.com/acgetchell/delaunay/pull/341)

- Add an adaptive seed-backlog trigger for batch local Delaunay repair.
- Track local repair frontier sizes and cadence versus backlog triggers.
- Expose the 3D large-scale repair interval in the debug just recipe.
- Clarify batch repair policy docs and cover trigger/telemetry behavior with focused tests.

#### Performance: Fast-path retry validation [#341](https://github.com/acgetchell/delaunay/pull/341)

- Use the flip-based Delaunay verifier when accepting shuffled construction
  retries instead of the brute-force property scan.

- Add construction phase telemetry for preprocessing, insertion, finalization,
  and final validation timing.

- Default the large-scale debug repair interval to 2 for the 3K repair sweep.

#### Performance: Default batch repair to EveryN(2) [#341](https://github.com/acgetchell/delaunay/pull/341)

- Default batch construction repair to EveryN(2) while keeping direct incremental repair at EveryInsertion.
- Preserve final repair and validation as the acceptance gate for valid Delaunay output.
- Surface construction skip and slow-insertion diagnostics through the construction prelude.
- Align repair-policy docs with the batch default and the current 500/3000-point proxy results.

#### Fixed: Tolerate degenerate Delaunay retessellation [#341](https://github.com/acgetchell/delaunay/pull/341)

- Compare quality metrics only for cells shared across independently
  transformed triangulations, rejecting cases with no comparable cells.

- Narrow locality repair helper visibility where the current module graph
  allows it while preserving core insertion repair access.

- Clarify that non-substantive PRs may be declined unless justified by
  cleanup or tooling needs.

#### Fixed: Distinguish insertion telemetry [#341](https://github.com/acgetchell/delaunay/pull/341)

- Count only full insertion validation runs in topology validation telemetry while preserving required link checks.
- Exclude caller-provided conflict buffers from discovered conflict-region telemetry.
- Keep large-scale debug validation cadence tied to inserted vertices and tighten triangulation proptest acceptance.
- Add explicit unsafe-code forbids to remaining bench, example, and test crates.

#### Performance: Tune batch construction repair [#341](https://github.com/acgetchell/delaunay/pull/341)

- Default batch construction to the real-vertex max-volume initial simplex and
  every-insertion Delaunay repair cadence.

- Add focused construction diagnostics telemetry for repair phase timing,
  queued work, seed frontiers, and construction phase costs.

- Keep local repair postcondition replay tied to observed repair work so skipped
  flip candidates cannot suppress required validation.

- Update large-scale debug docs and recipes to reflect the new defaults.

- Stabilize large-scale profiling and construction [#377](https://github.com/acgetchell/delaunay/pull/377)
  [`77ba50b`](https://github.com/acgetchell/delaunay/commit/77ba50b23c6f22fbeaa6acf240bb144bbb1458f7)

  - Calibrate the 2D-5D debug-large-scale acceptance runs around maintainer
    hardware and document generated simplex counts.

  - Fold large-scale profiling into profiling_suite with construction, memory,
    validation, and traversal workloads.

  - Compare performance regressions against same-machine main-ref baselines.
  - Harden insertion topology preflight and exact insphere paths used by repair
    and high-dimensional construction.

## [0.7.6] - 2026-04-25

### ⚠️ Breaking Changes

- Remove ScalarAccumulative and ScalarSummable traits [#316](https://github.com/acgetchell/delaunay/pull/316)
  [#318](https://github.com/acgetchell/delaunay/pull/318)

### Merged Pull Requests

- Preserve fallback rebuild cell data [#305](https://github.com/acgetchell/delaunay/pull/305) [#346](https://github.com/acgetchell/delaunay/pull/346)
- Switch coverage reporting to cargo-llvm-cov [#345](https://github.com/acgetchell/delaunay/pull/345)
- Clarify research scope and changelog hygiene [#344](https://github.com/acgetchell/delaunay/pull/344)
- Instrument large-scale 4D debugging and widen local repair seeds [#339](https://github.com/acgetchell/delaunay/pull/339)
- Orient Delaunay repair replacement cells [#307](https://github.com/acgetchell/delaunay/pull/307) [#336](https://github.com/acgetchell/delaunay/pull/336)
- Use dedicated perf profile for consistent benchmark measurement [#334](https://github.com/acgetchell/delaunay/pull/334)
- Periodic-aware Delaunay verification (Level 4) for toroidal tria… [#333](https://github.com/acgetchell/delaunay/pull/333)
- Adopt Rust 1.95.0 MSRV [#330](https://github.com/acgetchell/delaunay/pull/330)
- Bump actions-rust-lang/setup-rust-toolchain [#328](https://github.com/acgetchell/delaunay/pull/328)
- Bump actions/setup-node from 6.3.0 to 6.4.0 [#327](https://github.com/acgetchell/delaunay/pull/327)
- Bump taiki-e/install-action from 2.75.9 to 2.75.18 [#326](https://github.com/acgetchell/delaunay/pull/326)
- Bump astral-sh/setup-uv from 8.0.0 to 8.1.0 [#325](https://github.com/acgetchell/delaunay/pull/325)
- Bump pytest in the uv group across 1 directory [#322](https://github.com/acgetchell/delaunay/pull/322)
- Bump taiki-e/install-action from 2.73.0 to 2.75.9 [#321](https://github.com/acgetchell/delaunay/pull/321)
- Bump actions/github-script from 8.0.0 to 9.0.0 [#320](https://github.com/acgetchell/delaunay/pull/320)
- Unify flip-repair and retry constants across build profiles [#306](https://github.com/acgetchell/delaunay/pull/306)
  [#319](https://github.com/acgetchell/delaunay/pull/319)
- Remove ScalarAccumulative and ScalarSummable traits [#316](https://github.com/acgetchell/delaunay/pull/316)
  [#318](https://github.com/acgetchell/delaunay/pull/318)
- Rename tds file and move delaunay/builder into triangulation/ [#317](https://github.com/acgetchell/delaunay/pull/317)
- Allow explicit cell construction with non-sphere Euler characteristic [#314](https://github.com/acgetchell/delaunay/pull/314)

### Added

- Allow explicit cell construction with non-sphere Euler characteristic [#314](https://github.com/acgetchell/delaunay/pull/314)
  [`c81bb1a`](https://github.com/acgetchell/delaunay/commit/c81bb1a69f50a8c777890c0ac95b07cb99033d83)

  Add `.global_topology()` builder setter so callers can declare the
  intended topology (e.g. Toroidal) for explicit cell construction.
  `validate_topology_core()` uses this metadata to override the heuristic
  classification: closed toroidal meshes expect χ = 0 instead of the
  sphere default χ = 1+(-1)^D.

  - Add `ToroidalConstructionMode::Explicit` variant for explicit builds
  - Add `TopologyClassification::ClosedToroid(D)` with χ(T^d) = 0
  - Add `global_topology` field and setter to `DelaunayTriangulationBuilder`
  - Thread `GlobalTopology` through `build_explicit()`, set before validation
  - Override Euler classification in `validate_topology_core()` when
    `global_topology` is `Toroidal` and heuristic yields `ClosedSphere`

  - Add T² (3×3 grid) and T³ (3×3×3 Freudenthal) integration tests
    validating χ = 0 via explicit construction

- Instrument large-scale 4D debugging and widen local repair seeds [#339](https://github.com/acgetchell/delaunay/pull/339)
  [`3af976e`](https://github.com/acgetchell/delaunay/commit/3af976ec2f7c33d49803b24ab8f1a7da598fea0b)

  - Thread cavity-touched cells through insertion as `repair_seed_cells`
    so post-insertion local Delaunay repair widens its frontier beyond
    the inserted vertex star; cells shrunk out of the conflict region
    during cavity reduction now participate in the next repair pass.

  - Accumulate ridge-fan extras across every fan in a conflict region
    before returning `RidgeFan`, letting one cavity-reduction step
    shrink all detected fans at once instead of peeling them iteration
    by iteration.

  - Add release-visible diagnostic hooks routed through `tracing::debug!`:
    `DELAUNAY_BULK_PROGRESS_EVERY` for periodic batch-construction
    progress, `DELAUNAY_DEBUG_RETRYABLE_SKIP` for retryable
    conflict-region skip traces, `DELAUNAY_DEBUG_CAVITY_REDUCTION_ONCE`
    for the first cavity-reduction chain, `DELAUNAY_DEBUG_RIDGE_FAN_ONCE`
    for the first detected ridge fan, and
    `DELAUNAY_REPAIR_DEBUG_POSTCONDITION_FACET` /
    `DELAUNAY_REPAIR_DEBUG_RIDGE_MIN_MULTIPLICITY` for repair
    postcondition debugging.

  - Thread `last_applied_flip` through repair postcondition verification
    so unresolved k=2 facet and ridge snapshots can relate the violating
    local star to the immediately preceding flip.

  - Replace `ConflictError::InternalInconsistency { context: String }`
    with a typed `InternalInconsistencySite` enum carrying structured
    indices and counts, so callers can `matches!` on specific sites
    instead of parsing prose.

  - Generalize the large-scale incremental prefix bisect over `const D`,
    add a 4D counterpart targeting the seeded 500-point repro
    (`0xD225_B8A0_7E27_4AE6`), and expose it via
    `just debug-large-scale-4d-incremental-bisect`.

  - Switch the large-scale debug just recipes to `--release` and
    document the 2026-04-23 re-verification: historical 35-point 3D and
    100-point 4D correctness repros from #306/#307 now pass, while a
    500-point 4D seed still fails all shuffled retries with
    `Ridge fan detected: 4 facets share ridge with 3 vertices`.

  - Default the large-scale debug harness tracing filter to `debug` when
    any of the new release-visible env vars are present so library-side
    `tracing::debug!` events surface without extra `RUST_LOG` wiring.

  - Broaden `test_perturbation_retry_and_exhaustion_4d` and
    `test_perturbation_retry_seeded_branch_4d` to iterate over 50 seeds
    so the retry-path assertions stay robust to insertion-path
    improvements that make individual well-conditioned seeds less likely
    to trigger retries.

#### Fixed: Close the 4D bulk repair retry collapse

- Raise the D≥4 per-insertion repair budget, add a rate-limited escalation pass, and widen local post-repair validation so the 500-point #204 repro converges
  without skipped vertices.

- Preserve removed-cell snapshots and predecessor context in flip diagnostics, drop stale repair seeds after cavity reduction, and re-export locate conflict
  diagnostics from the prelude.

- Replace committed `eprintln!` diagnostics in production, tests, and benches with `tracing` , using `diagnostics` and `bench-logging` gates and keeping logs
  out of Criterion hot loops.

- Document the #204 investigation, refresh the 4D known-issues and TODO notes, and record the repository logging policy plus release-visible debug environment
  variables.

#### Changed: Harden flip diagnostics and refine large-scale debug workflows

  Refactor flip snapshotting and cavity-reduction bookkeeping to ensure
  diagnostic reliability and accurate repair-seed collection. Update
  documentation and justfile recipes to reflect fixed historical repros
  and transition to monitoring active scalability investigations for 3D,
  4D, and 5D datasets.

- Move removed-cell vertex capturing into fallible internal helpers
- Implement lazy evaluation for cavity-reduction diagnostic logs
- Harden vertex deduplication with fallible epsilon validation
- Update 4D known issues to reflect 100-point and 500-point fixes
- Simplify the large-scale debug harness CLI and documentation

### Changed

- Rename tds file and move delaunay/builder into triangulation/ [#317](https://github.com/acgetchell/delaunay/pull/317)
  [`25faa5b`](https://github.com/acgetchell/delaunay/commit/25faa5b4b2fe54b20e609cd390fe3edb221da29b)

  - Rename src/core/triangulation_data_structure.rs → src/core/tds.rs
  - Move src/core/delaunay_triangulation.rs → src/triangulation/delaunay.rs
  - Move src/core/builder.rs → src/triangulation/builder.rs
  - Widen pub(in crate::core) → pub(crate) for cross-module access
  - Preserve public API via re-exports in core {}
  - Add GlobalTopology::is_euclidean() for API symmetry with is_toroidal()
  - Add doctests for topology_guarantee, global_topology, topology_kind,
    set_global_topology accessors

  - Hoist in-function test imports to module-level per project convention
  - Shorten fully-qualified paths (CellValidationError, ConflictError, etc.)
  - Bump la-stack 0.3.0 → 0.4.0 (integer-only Bareiss, stack-backed exact
    arithmetic, custom f64→BigRational via IEEE 754 bit decomposition)

  - Update README, code_organization.md, debug_env_vars.md, rust.md,
    numerical_robustness_guide.md, COVERAGE.md, and other active docs

  - Archive docs left unchanged (historical state)
- [**breaking**] Remove ScalarAccumulative and ScalarSummable traits [#316](https://github.com/acgetchell/delaunay/pull/316)
  [#318](https://github.com/acgetchell/delaunay/pull/318) [`c612188`](https://github.com/acgetchell/delaunay/commit/c61218858a169108da51067e9b65f6a26baede16)

  - Absorb AddAssign, SubAssign, and Sum bounds into CoordinateScalar
  - Remove ScalarAccumulative and ScalarSummable traits and their blanket impls
  - Replace all ScalarAccumulative bounds with CoordinateScalar across 20 files
  - Simplify operator usage (e.g. coords[axis] += … instead of manual add-assign)
- Update dependencies and synchronize changelog (internal) [`7a1066b`](https://github.com/acgetchell/delaunay/commit/7a1066b4da9229a5148994f8a80f3521df62ec6c)

  Perform a general dependency update, including a patch bump for `uuid`.
  Synchronize `CHANGELOG.md` with recent maintenance tasks, documentation
  updates, and merged pull requests. Add `.kilo/` to the ignored user
  configuration patterns.
- Use dedicated perf profile for consistent benchmark measurement [#334](https://github.com/acgetchell/delaunay/pull/334)
  [`f527c0c`](https://github.com/acgetchell/delaunay/commit/f527c0cf37b76f09222800afcfc138e623957678)

  Introduce a `perf` Cargo profile that inherits from `release` but
  restores ThinLTO and single codegen units. This ensures local, CI, and
  release benchmarks are generated with identical optimization settings.

  Update all benchmark harnesses, `just` recipes, and documentation to
  standardize on `--profile perf` for performance measurements, while
  retaining the default release profile for fast validation in `just ci`.
  A new `bench-smoke` target provides quick harness validation without
  the overhead of high-sample measurements.

  Also deniest warnings via the manifest lint policy to ensure consistent
  repository-wide enforcement.

#### Changed: Standardize benchmark profiles and enhance SARIF analysis

  Standardize benchmark workflows to use the `perf` profile by default
  across local scripts and CI for consistent optimization settings. Add a
  dedicated CodeQL analysis workflow and refactor SARIF reporting for
  cargo-audit, Clippy, and Codacy to improve GitHub Code Scanning
  integration. Update manifest lints to comply with RFC 3389 priority
  requirements and fix the minimum sample size for benchmark smoke tests.

#### Changed: Track sampling metadata and standardize benchmark profiles

  Enhance performance regression testing by embedding sampling configuration
  (Criterion settings and Cargo profile) into baseline files. This enables
  automatic detection of configuration mismatches during comparisons.
  Standardize benchmarking scripts on the trusted perf profile and update
  developer guidelines for naming conventions and local imports.

#### Changed: Enable debug line tables for perf profile and refine validation

  Include `debug = "line-tables-only"` in the perf Cargo profile to
  enable source-level profiling. Update the benchmark comparison logic
  to ensure that legacy baselines with missing or "Unknown" metadata
  trigger configuration mismatch warnings.

#### Changed: Expand benchmark metadata validation tests

  Update the benchmark utility tests to verify that differences or
  omissions in Criterion measurement and warm-up time are correctly
  reported in configuration mismatch warnings.

#### Changed: Enable CodeRabbit request changes workflow

  Enable the request_changes_workflow in the CodeRabbit configuration to
  allow the AI reviewer to formally request changes on pull requests. This
  ensures that identified issues are explicitly addressed during the
  review process rather than appearing as informational comments only.

- Revise SECURITY.md for clarity and completeness [`bd041fa`](https://github.com/acgetchell/delaunay/commit/bd041fa18456abad59a11088ab8a14e7f399adbe)

### Documentation

- Sync documentation with post-v0.7.5 changes [skip ci] [`5fa36aa`](https://github.com/acgetchell/delaunay/commit/5fa36aa67cb99bb3a5781e4c2733c2acec3adea8)

  - Add missing files to code_organization.md directory tree
    (debug_env_vars.md, TODO.md, conflict_region_verification.rs,
    regression_issue_306.rs)

  - Add delaunayize_repair example to README.md examples list
  - Document new test files in tests/README.md (delaunayize_workflow,
    conflict_region_verification, regression_issue_306)

  - Update docs/TODO.md: mark completed items (#288, #302, #306/#307),
    refresh next-release candidates

- Clarify research scope and changelog hygiene [#344](https://github.com/acgetchell/delaunay/pull/344)
  [`f46b7ae`](https://github.com/acgetchell/delaunay/commit/f46b7ae1e64734e441854e9868fae73d5d503c68)

  - Refresh README research positioning with current scope, non-goals,
    predicate limits, and documentation links.

  - Add current limitations and roadmap docs, and archive stale 4D known
    issues and TODO snapshots.

  - Update stale module/workflow references across docs and regenerate the
    active and archived changelogs.

  - Teach changelog post-processing to render squash-body pseudo-headings as
    prose, with tests covering the new normalization helpers.

### Fixed

- Handle geometric degeneracy gracefully in profiling benchmarks
  [`3532624`](https://github.com/acgetchell/delaunay/commit/3532624ed1d669d215b2df2bbcea49236b720d10)

  - Replace .unwrap() on triangulation build with graceful error handling
    across all benchmark functions in profiling_suite.rs

  - Scaling and memory benchmarks skip iterations that hit degeneracy
  - Query latency and algorithmic bottleneck benchmarks skip entries when
    setup construction fails

  - Fixes CI panic: GeometricDegeneracy during 3D memory profiling with
    10,000 random points (seed 42)

- Unify flip-repair and retry constants across build profiles [#306](https://github.com/acgetchell/delaunay/pull/306)
  [#319](https://github.com/acgetchell/delaunay/pull/319) [`14f0d16`](https://github.com/acgetchell/delaunay/commit/14f0d16b7de52d25d64e7d1fb810b82817e5d7b6)

  Several flip-repair and construction-retry constants were split by
  cfg(any(test, debug_assertions)), causing release builds to fail on
  valid 3D inputs that debug/test builds handled correctly.

  Primary fix: RetryPolicy::default() was Disabled in release but
  DebugOnlyShuffled { attempts: 6 } in debug/test. When the initial
  Hilbert-ordered insertion hits a co-spherical flip cycle, shuffled
  retries find a working vertex ordering. Without retries, the first
  failure was fatal. Changed to Shuffled { attempts: 6 } unconditionally.

  Secondary fixes in flips.rs:

  - Unify MAX_REPEAT_SIGNATURE to 128 (was 32 in release)
  - Unify default_max_flips 3D multiplier to 8x (was 4x in release)
  - Make is_connected() postcondition check unconditional

  Also in delaunay.rs:

  - Unify HEURISTIC_REBUILD_ATTEMPTS to 6 (was 2 in release)
  - Remove cfg-gated DELAUNAY_SHUFFLE_ATTEMPTS constant

#### Fixed: Enable construction retries in release builds by default

  Update RetryPolicy::default() to return Shuffled retries in all build
  profiles. Previously, retries were disabled in release mode, causing
  fatal construction failures on 3D configurations that required
  alternative vertex insertion orders to converge. This change ensures
  consistent behavior between debug and release builds and includes a
  regression test for the reported failure case.

- Periodic-aware Delaunay verification (Level 4) for toroidal tria… [#333](https://github.com/acgetchell/delaunay/pull/333)
  [`7c788aa`](https://github.com/acgetchell/delaunay/commit/7c788aa1a7e3b2a94a53193d8d5894718b6afa07)

#### Fixed: Periodic-aware Delaunay verification (Level 4) for toroidal triangulations [#315](https://github.com/acgetchell/delaunay/pull/315)

- Thread GlobalTopologyModelAdapter through all flip-predicate evaluation
  functions (k=2 facets, k=3 ridges, and their inverses) so insphere
  predicates use lifted coordinates for periodic topologies.

- Add verify_delaunay_for_triangulation() as the preferred topology-aware
  Level 4 entry point; is_delaunay_via_flips() now delegates to it.

- Add periodic lifting helpers: vertices_to_points_with_optional_lift,
  vertex_point_lifted_into_cell, align_periodic_offset, and supporting
  functions for offset lookup, validation, and cross-cell alignment.

- Repair paths continue to use GlobalTopology::DEFAULT (non-periodic),
  which is correct since flip repair runs during construction.

- Add integration test reproducing the issue: toroidal_periodic 2D
  triangulation now passes dt.validate() (Levels 1–4).

- Add doctest for verify_delaunay_for_triangulation.

- Add unit tests for align_periodic_offset (identity, delta shifts,
  higher-dimension, overflow).

#### Fixed: Use reference frames for periodic inverse predicates in flip repair

  Update Delaunay violation predicates to accept an optional frame cell,
  ensuring coordinate lifting remains consistent for inverse k=2 and k=3
  moves. This prevents verification failures in toroidal triangulations
  where inverse simplices lack a direct matching cell in the removal
  buffer. Additionally, document the intent and invariants of internal
  functions across the flip and construction modules to align with
  updated developer guidelines.

- Orient Delaunay repair replacement cells [#307](https://github.com/acgetchell/delaunay/pull/307) [#336](https://github.com/acgetchell/delaunay/pull/336)
  [`68deb62`](https://github.com/acgetchell/delaunay/commit/68deb6212a0860cd85776744d29ba7e76f368579)

  - Build flip replacement cell order from oriented cavity-boundary constraints.
  - Keep raw bistellar flips topology-oriented while requiring positive replacement geometry for Delaunay repair.
  - Canonicalize bulk repair results before continuing construction.
  - Add a 4D regression test for the issue #307 bulk construction failure.
  - Document branch naming conventions for contributors and agents.

#### Fixed: Harden flip repair invariants and PR quality gates

- Reject periodic external cells when replacement-cell parity constraints cannot be preserved.
- Canonicalize after D&gt;=4 bulk repair failures before continuing with later insertions.
- Consolidate fixed-bug regression coverage into tests/regressions.rs and document the pattern.
- Narrow CodeQL/Codacy code scanning noise while keeping curated PR quality feedback via Codacy and CodeRabbit.
- Stage repository-owned Semgrep rules for future cleanup without enabling broad default rule packs.

#### Changed: Cover flip repair checks and quality tooling

- Add focused coverage for replacement-cell orientation helper error paths and D&gt;=4 bulk-repair canonicalization branches.
- Strengthen the #307 regression with a longer 4D prefix and explicit geometric-orientation validation.
- Add an opt-in `just semgrep` workflow, pin Semgrep in uv dev dependencies, and refresh Python tool versions.
- Clarify CodeRabbit/Codacy quality-scan documentation and workflow labels.
- Preserve profiling benchmark rustflags behavior under setup-rust-toolchain.
- Preserve fallback rebuild cell data [#305](https://github.com/acgetchell/delaunay/pull/305) [#346](https://github.com/acgetchell/delaunay/pull/346)
  [`7e42be8`](https://github.com/acgetchell/delaunay/commit/7e42be8fba9abe571d0137710fbd7ed0151ebc85)

  - Restore cell payloads during delaunayize fallback rebuilds when a rebuilt cell matches exactly one original vertex UUID signature.
  - Treat duplicate original cell signatures as ambiguous, even when their payloads are identical.
  - Preserve typed snapshot, rebuild, and restore errors for debuggable fallback failure paths.
  - Expose delaunayize error payload types through the focused workflow prelude and add normal-type coverage.
  - Document fallback cell-data preservation and refresh related test/docs utility updates.

### Maintenance

- Bump pytest in the uv group across 1 directory [#322](https://github.com/acgetchell/delaunay/pull/322)
  [`c5e3c3b`](https://github.com/acgetchell/delaunay/commit/c5e3c3bb35c94add75dc1550860b77a0a7e01f32)

  Bumps the uv group with 1 update in the / directory: [pytest](https://github.com/pytest-dev/pytest).

  Updates `pytest` from 8.4.2 to 9.0.3

  - [Release notes](https://github.com/pytest-dev/pytest/releases)
  - [Changelog](https://github.com/pytest-dev/pytest/blob/main/CHANGELOG.rst)
  - [Commits](https://github.com/pytest-dev/pytest/compare/8.4.2...9.0.3)

---

  updated-dependencies:

- dependency-name: pytest
  dependency-version: 9.0.3
  dependency-type: direct:development
  dependency-group: uv
  ...

- Bump actions/github-script from 8.0.0 to 9.0.0 [#320](https://github.com/acgetchell/delaunay/pull/320)
  [`80f5984`](https://github.com/acgetchell/delaunay/commit/80f5984dbe45535226a26a47b2f3eb509b0b9f88)

  Bumps [actions/github-script](https://github.com/actions/github-script) from 8.0.0 to 9.0.0.

  - [Release notes](https://github.com/actions/github-script/releases)
  - [Commits](https://github.com/actions/github-script/compare/ed597411d8f924073f98dfc5c65a23a2325f34cd...3a2844b7e9c422d3c10d287c895573f7108da1b3)

---

  updated-dependencies:

- dependency-name: actions/github-script
  dependency-version: 9.0.0
  dependency-type: direct:production
  update-type: version-update:semver-major
  ...

- Bump taiki-e/install-action from 2.73.0 to 2.75.9 [#321](https://github.com/acgetchell/delaunay/pull/321)
  [`d9fa322`](https://github.com/acgetchell/delaunay/commit/d9fa322eda20cbb4f9b35f3db954618072905613)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.73.0 to 2.75.9.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/7a562dfa955aa2e4d5b0fd6ebd57ff9715c07b0b...d0f23220b09a75c6db730f13bb37c4f8144b4382)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.75.9
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump actions-rust-lang/setup-rust-toolchain [#328](https://github.com/acgetchell/delaunay/pull/328)
  [`b8222f5`](https://github.com/acgetchell/delaunay/commit/b8222f505e7b7980f228d3f75eed3ffcabb43c0e)

  Bumps [actions-rust-lang/setup-rust-toolchain](https://github.com/actions-rust-lang/setup-rust-toolchain) from 1.15.4 to 1.16.0.

  - [Release notes](https://github.com/actions-rust-lang/setup-rust-toolchain/releases)
  - [Changelog](https://github.com/actions-rust-lang/setup-rust-toolchain/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/actions-rust-lang/setup-rust-toolchain/compare/150fca883cd4034361b621bd4e6a9d34e5143606...2b1f5e9b395427c92ee4e3331786ca3c37afe2d7)

---

  updated-dependencies:

- dependency-name: actions-rust-lang/setup-rust-toolchain
  dependency-version: 1.16.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump astral-sh/setup-uv from 8.0.0 to 8.1.0 [#325](https://github.com/acgetchell/delaunay/pull/325)
  [`dc6b5eb`](https://github.com/acgetchell/delaunay/commit/dc6b5ebbfdf4340ad76c96178024c7e076f1ae8d)

  Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 8.0.0 to 8.1.0.

  - [Release notes](https://github.com/astral-sh/setup-uv/releases)
  - [Commits](https://github.com/astral-sh/setup-uv/compare/cec208311dfd045dd5311c1add060b2062131d57...08807647e7069bb48b6ef5acd8ec9567f424441b)

---

  updated-dependencies:

- dependency-name: astral-sh/setup-uv
  dependency-version: 8.1.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump taiki-e/install-action from 2.75.9 to 2.75.18 [#326](https://github.com/acgetchell/delaunay/pull/326)
  [`f14b760`](https://github.com/acgetchell/delaunay/commit/f14b7607b76a7914c31556295005c7b7559932f4)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.75.9 to 2.75.18.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/d0f23220b09a75c6db730f13bb37c4f8144b4382...055f5df8c3f65ea01cd41e9dc855becd88953486)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.75.18
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump actions/setup-node from 6.3.0 to 6.4.0 [#327](https://github.com/acgetchell/delaunay/pull/327)
  [`7de0dc1`](https://github.com/acgetchell/delaunay/commit/7de0dc18fddf65d00fa218361e01d727a71aeefa)

  Bumps [actions/setup-node](https://github.com/actions/setup-node) from 6.3.0 to 6.4.0.

  - [Release notes](https://github.com/actions/setup-node/releases)
  - [Commits](https://github.com/actions/setup-node/compare/53b83947a5a98c8d113130e565377fae1a50d02f...48b55a011bda9f5d6aeb4c2d9c7362e8dae4041e)

---

  updated-dependencies:

- dependency-name: actions/setup-node
  dependency-version: 6.4.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...
- Adopt Rust 1.95.0 MSRV [#330](https://github.com/acgetchell/delaunay/pull/330)
  [`d0c53d9`](https://github.com/acgetchell/delaunay/commit/d0c53d95e748bac33e079a4256222b7bff7fad53)

  Bump MSRV from 1.94 to 1.95 and adopt stabilized features where
  they fit. Coordinates with la-stack 0.4.0 -&gt; 0.4.1, which also
  requires 1.95.

  Toolchain and docs:

  - Cargo.toml (rust-version), rust-toolchain.toml (channel) and
    clippy.toml (msrv) all set to 1.95.

  - AGENTS.md and CONTRIBUTING.md MSRV references refreshed;
    AGENTS.md Design Principles section expanded.

  - CITATION.cff caught up to 0.7.5.
  - la-stack bumped 0.4.0 -&gt; 0.4.1 in Cargo.toml / Cargo.lock.
- Switch coverage reporting to cargo-llvm-cov [#345](https://github.com/acgetchell/delaunay/pull/345)
  [`b5fb221`](https://github.com/acgetchell/delaunay/commit/b5fb22134a07b28b40b45ef8484a4f0c79e4d61e)

  - Replace tarpaulin coverage commands with cargo-llvm-cov for local HTML and CI Cobertura reports.
  - Update Codecov workflow setup, caching, validation, and coverage documentation for the new toolchain.
  - Add GitHub issue templates and clarify private vulnerability reporting guidance.
  - Tighten Python lint/typecheck settings and clean up benchmark/changelog utility diagnostics.
  - Add changelog post-processing coverage for version-heading reset behavior.

## [0.7.5] - 2026-04-10

### ⚠️ Breaking Changes

- Change remove_vertex to accept VertexKey instead of &Vertex [#300](https://github.com/acgetchell/delaunay/pull/300)

### Merged Pull Requests

- V0.7.5 cleanup — impl-block split, builder decomposition, p… [#311](https://github.com/acgetchell/delaunay/pull/311)
- Add diagnostic infrastructure for v0.7.6 investigation (#306, #… [#309](https://github.com/acgetchell/delaunay/pull/309)
- Bump taiki-e/install-action from 2.70.2 to 2.73.0 [#308](https://github.com/acgetchell/delaunay/pull/308)
- Add MVP delaunayize-by-flips workflow [#227](https://github.com/acgetchell/delaunay/pull/227) [#303](https://github.com/acgetchell/delaunay/pull/303)
- Add explicit construction from vertices and cells [#293](https://github.com/acgetchell/delaunay/pull/293)
  [#301](https://github.com/acgetchell/delaunay/pull/301)
- Change remove_vertex to accept VertexKey instead of &Vertex [#300](https://github.com/acgetchell/delaunay/pull/300)
- Bump pygments from 2.19.2 to 2.20.0 [#298](https://github.com/acgetchell/delaunay/pull/298)
- Bump astral-sh/setup-uv from 7.6.0 to 8.0.0 [#297](https://github.com/acgetchell/delaunay/pull/297)
- Bump taiki-e/install-action from 2.69.6 to 2.70.2 [#296](https://github.com/acgetchell/delaunay/pull/296)
- Bump codecov/codecov-action from 5.5.3 to 6.0.0 [#295](https://github.com/acgetchell/delaunay/pull/295)
- Bump the dependencies group with 3 updates [#294](https://github.com/acgetchell/delaunay/pull/294)

### Added

- Add explicit construction from vertices and cells [#293](https://github.com/acgetchell/delaunay/pull/293)
  [#301](https://github.com/acgetchell/delaunay/pull/301) [`458ebae`](https://github.com/acgetchell/delaunay/commit/458ebae0fbbd6c5142f88d24c4ff254f058f9285)

  - Introduce `DelaunayTriangulationBuilder::from_vertices_and_cells` for
    combinatorial construction bypassing Delaunay insertion

  - Add `build_explicit` path that constructs a TDS directly from given
    vertices and cell index lists, with adjacency and orientation repair

  - Define `ExplicitConstructionError` enum with `IndexOutOfBounds`,
    `InvalidCellArity`, `DuplicateVertexInCell`, `EmptyCells`, and
    `IncompatibleTopology` variants

  - Wire `ExplicitConstruction` variant into
    `DelaunayTriangulationConstructionError`

  - Expose `assign_neighbors` as `pub(crate)` in the TDS
  - Add comprehensive integration tests for explicit construction across
    2D/3D, round-trip fidelity, error cases, and topology guarantees

#### Changed: Refine explicit construction validation and error reporting

  Update the triangulation builder to return specific ValidationFailed
  errors when structural invariants are violated. Refine the Delaunay
  property check in the validation suite to use flip-based verification
  for more precise error reporting. Expand the test suite to cover 3D
  round-trip fidelity, non-manifold topology errors, and vertex data
  consistency during explicit construction.

- Add MVP delaunayize-by-flips workflow [#227](https://github.com/acgetchell/delaunay/pull/227) [#303](https://github.com/acgetchell/delaunay/pull/303)
  [`0370070`](https://github.com/acgetchell/delaunay/commit/037007076e20728f0b23e528c89c783a1f5d2a70)

  Add a public `delaunayize_by_flips` entrypoint that performs bounded
  deterministic topology repair followed by flip-based Delaunay repair,
  with an optional fallback rebuild from the vertex set.

  New files:

  - src/core/algorithms/pl_manifold_repair.rs: pub(crate) bounded
    facet over-sharing repair (iterative worst-quality cell removal)

  - src/triangulation/delaunayize.rs: public API with DelaunayizeConfig,
    DelaunayizeOutcome, DelaunayizeError, and delaunayize_by_flips()

  - tests/delaunayize_workflow.rs: 13 integration tests covering
    success paths, fallback behavior, determinism, error variants,
    and flip-then-repair round-trip

  Other changes:

  - Box DelaunayRepairDiagnostics in NonConvergent variant to keep
    DelaunayRepairError small (eliminates downstream boxing)

  - Wire modules in src/lib.rs with self-contained prelude at
    prelude::triangulation::delaunayize

  - Re-export PlManifoldRepairStats, PlManifoldRepairError, and
    DelaunayRepairStats from the delaunayize module

  - Update docs/api_design.md and docs/code_organization.md

#### Changed: Allow manifold repair to proceed on over-shared facets

  Refactor `repair_facet_oversharing` to use partial structural pre-checks
  rather than full validation, as the latter rejects over-shared facets
  before repair can occur. Expand test coverage for budget exhaustion,
  cell removal logic, and repair determinism.

- Add diagnostic infrastructure for v0.7.6 investigation (#306, #… [#309](https://github.com/acgetchell/delaunay/pull/309)
  [`b25dff3`](https://github.com/acgetchell/delaunay/commit/b25dff390f27bc8b34ec99fa0c45a6163b6fd723)

#### Added: Add diagnostic infrastructure for v0.7.6 investigation (#306, #307)

- Enhance conflict-region verifier with neighbor-reachability analysis
- Add BFS boundary logging to find_conflict_region
- Add orientation tracing and post-insertion audit (DELAUNAY_DEBUG_ORIENTATION)
- Add cell creation provenance logging to fill_cavity
- Re-export diagnostic types (DelaunayRepairDiagnostics, etc.) in prelude
- Create comprehensive debug env var reference (docs/dev/debug_env_vars.md)

### Changed

- [**breaking**] Change remove_vertex to accept VertexKey instead of &Vertex [#300](https://github.com/acgetchell/delaunay/pull/300)
  [`71cca10`](https://github.com/acgetchell/delaunay/commit/71cca109607c66fb8af62b3e427078458bcef66d)

  - More ergonomic: callers pass a key directly instead of looking up and
    borrowing the full Vertex struct

  - More efficient: O(1) key dereference replaces UUID-based lookup
  - Consistent with set_vertex_data, set_cell_data, and the Edit API which
    all use keys

  - Aligned with the stated API design in docs/api_design.md
  - Simplifies internal callers in flips.rs (remove get+copy+remove
    pattern) and builder.rs (eliminate intermediate vertex copies)

  - Add stale-key idempotency test for the new VertexKey API
- V0.7.5 cleanup — impl-block split, builder decomposition, p… [#311](https://github.com/acgetchell/delaunay/pull/311)
  [`5bf5c81`](https://github.com/acgetchell/delaunay/commit/5bf5c8192b378d320d35f0a074a57f5b7e8170ff)

#### Changed: V0.7.5 cleanup — impl-block split, builder decomposition, prelude guidance, re-enable 3D proptests

- Split monolithic DelaunayTriangulation impl block into 6 trait-minimal
  blocks per #302 (ScalarAccumulative only where needed)
- Decompose search_closed_2d_selection: extract ClosedSelectionDfs struct
  and sort_candidates_by_rarity_and_domain helper, remove clippy suppressions
- Add "Which import do I need?" prelude guidance table to lib.rs
- Re-enable 43 previously-ignored 3D proptests (all &lt;1s in release mode)
- Audit doctests for builder migration (no changes needed; deferred to #214)
- Update internal performance results for v0.7.5 [`76b2ff7`](https://github.com/acgetchell/delaunay/commit/76b2ff7c4409a8bfa8aa6b9f5362a3dfb8b67999)

  Refresh the performance documentation with updated benchmark metrics,
  timestamps, and commit references to align with the v0.7.5 release.

### Documentation

- Update KNOWN_ISSUES_4D.md with #204 debug run findings [`87d3054`](https://github.com/acgetchell/delaunay/commit/87d3054a3d722029aeedf881f00caeec1bd7e639)

  - 3D: minimal failing prefix is 35 vertices (previously ~130+)
  - 3D: flip cycles confirmed NOT predicate-related (ambiguous=0)
  - 4D: negative-orientation cell causes 88% vertex skip rate
  - Add release-mode reproduction commands
  - Update recommendations section
  - Filed follow-up issues #306 and #307

### Maintenance

- Bump pygments from 2.19.2 to 2.20.0 [#298](https://github.com/acgetchell/delaunay/pull/298)
  [`6228a59`](https://github.com/acgetchell/delaunay/commit/6228a5990cfe38e711cafd2ce851bc3c9494ddc3)

  Bumps [pygments](https://github.com/pygments/pygments) from 2.19.2 to 2.20.0.

  - [Release notes](https://github.com/pygments/pygments/releases)
  - [Changelog](https://github.com/pygments/pygments/blob/master/CHANGES)
  - [Commits](https://github.com/pygments/pygments/compare/2.19.2...2.20.0)

---

  updated-dependencies:

- dependency-name: pygments
  dependency-version: 2.20.0
  dependency-type: indirect
  ...

- Bump the dependencies group with 3 updates [#294](https://github.com/acgetchell/delaunay/pull/294)
  [`4b3ccb3`](https://github.com/acgetchell/delaunay/commit/4b3ccb3da6827bd63b3c76219ea32a1cb5029b64)

  Bumps the dependencies group with 3 updates: [rustc-hash](https://github.com/rust-lang/rustc-hash) ,
  [ordered-float](https://github.com/reem/rust-ordered-float) and [uuid](https://github.com/uuid-rs/uuid) .

  Updates `rustc-hash` from 2.1.1 to 2.1.2

  - [Changelog](https://github.com/rust-lang/rustc-hash/blob/master/CHANGELOG.md)
  - [Commits](https://github.com/rust-lang/rustc-hash/compare/v2.1.1...v2.1.2)

  Updates `ordered-float` from 5.2.0 to 5.3.0

  - [Release notes](https://github.com/reem/rust-ordered-float/releases)
  - [Commits](https://github.com/reem/rust-ordered-float/compare/v5.2.0...v5.3.0)

  Updates `uuid` from 1.22.0 to 1.23.0

  - [Release notes](https://github.com/uuid-rs/uuid/releases)
  - [Commits](https://github.com/uuid-rs/uuid/compare/v1.22.0...v1.23.0)

---

  updated-dependencies:

- dependency-name: rustc-hash
  dependency-version: 2.1.2
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies

- dependency-name: ordered-float
  dependency-version: 5.3.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  dependency-group: dependencies

- dependency-name: uuid
  dependency-version: 1.23.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  dependency-group: dependencies
  ...

- Bump codecov/codecov-action from 5.5.3 to 6.0.0 [#295](https://github.com/acgetchell/delaunay/pull/295)
  [`420a1c5`](https://github.com/acgetchell/delaunay/commit/420a1c50a8e34b6b02ba4b64e460a6d9c6ae30c5)

  Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 5.5.3 to 6.0.0.

  - [Release notes](https://github.com/codecov/codecov-action/releases)
  - [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/codecov/codecov-action/compare/1af58845a975a7985b0beb0cbe6fbbb71a41dbad...57e3a136b779b570ffcdbf80b3bdc90e7fab3de2)

---

  updated-dependencies:

- dependency-name: codecov/codecov-action
  dependency-version: 6.0.0
  dependency-type: direct:production
  update-type: version-update:semver-major
  ...

- Bump astral-sh/setup-uv from 7.6.0 to 8.0.0 [#297](https://github.com/acgetchell/delaunay/pull/297)
  [`0a0c539`](https://github.com/acgetchell/delaunay/commit/0a0c539c743afc5919b3ab6af55f4cec5d780280)

  Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.6.0 to 8.0.0.

  - [Release notes](https://github.com/astral-sh/setup-uv/releases)
  - [Commits](https://github.com/astral-sh/setup-uv/compare/37802adc94f370d6bfd71619e3f0bf239e1f3b78...cec208311dfd045dd5311c1add060b2062131d57)

---

  updated-dependencies:

- dependency-name: astral-sh/setup-uv
  dependency-version: 8.0.0
  dependency-type: direct:production
  update-type: version-update:semver-major
  ...

- Bump taiki-e/install-action from 2.69.6 to 2.70.2 [#296](https://github.com/acgetchell/delaunay/pull/296)
  [`0ab08a7`](https://github.com/acgetchell/delaunay/commit/0ab08a76c46117e633741653bf3e0ab6449671f6)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.69.6 to 2.70.2.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/06203676c62f0d3c765be3f2fcfbebbcb02d09f5...e9e8e031bcd90cdbe8ac6bb1d376f8596e587fbf)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.70.2
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump taiki-e/install-action from 2.70.2 to 2.73.0 [#308](https://github.com/acgetchell/delaunay/pull/308)
  [`839877f`](https://github.com/acgetchell/delaunay/commit/839877f6db184e766fc9f4e3515b5fee3e70918f)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.70.2 to 2.73.0.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/e9e8e031bcd90cdbe8ac6bb1d376f8596e587fbf...7a562dfa955aa2e4d5b0fd6ebd57ff9715c07b0b)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.73.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

## [0.7.4] - 2026-03-27

### ⚠️ Breaking Changes

- Tighten Vertex::data and Cell::data to pub(crate) [#289](https://github.com/acgetchell/delaunay/pull/289)
- Generalize DelaunayTriangulationBuilder::new() over U [#287](https://github.com/acgetchell/delaunay/pull/287)
  [#290](https://github.com/acgetchell/delaunay/pull/290)

### Merged Pull Requests

- Release v0.7.4 [#291](https://github.com/acgetchell/delaunay/pull/291)
- Generalize DelaunayTriangulationBuilder::new() over U [#287](https://github.com/acgetchell/delaunay/pull/287)
  [#290](https://github.com/acgetchell/delaunay/pull/290)
- Tighten Vertex::data and Cell::data to pub(crate) [#289](https://github.com/acgetchell/delaunay/pull/289)
- Add set_vertex_data and set_cell_data methods [#284](https://github.com/acgetchell/delaunay/pull/284) [#285](https://github.com/acgetchell/delaunay/pull/285)

### Added

- Add set_vertex_data and set_cell_data methods [#284](https://github.com/acgetchell/delaunay/pull/284) [#285](https://github.com/acgetchell/delaunay/pull/285)
  [`b398d54`](https://github.com/acgetchell/delaunay/commit/b398d5467114610053902a141980f3583eb71aec)

  - Add `set_vertex_data` and `set_cell_data` to `Tds` for O(1) mutation
    of auxiliary vertex/cell data without affecting geometry or topology

  - Add convenience wrappers on `Triangulation` and `DelaunayTriangulation`
    that delegate to the TDS methods without invalidating caches

  - All doctests use `prelude::triangulation::*` to demonstrate idiomatic
    imports including `DelaunayTriangulationBuilder::from_vertices`

  - 9 unit tests covering replacement, no-data vertices, invalid keys,
    invariant preservation, multi-key mutation, and locate-hint stability

  - Add `set_vertex_data` and `set_cell_data` to `Tds` accepting
    `Option&lt;U&gt;` / `Option&lt;V&gt;` for setting or clearing auxiliary data
    in O(1) without affecting geometry or topology

  - Add convenience wrappers on `Triangulation` and `DelaunayTriangulation`
    that delegate to the TDS methods without invalidating caches

  - All doctests use `prelude::triangulation::*` and demonstrate both
    setting and clearing data paths

  - 11 unit tests covering Tds basics, Triangulation wrappers, invariant
    preservation, multi-key mutation, clearing, and locate-hint stability

### Changed

- [**breaking**] Tighten Vertex::data and Cell::data to pub(crate) [#289](https://github.com/acgetchell/delaunay/pull/289)
  [`07f1565`](https://github.com/acgetchell/delaunay/commit/07f15658aac65e5cce3ecb3b7172502305173b95)

  - Change `Vertex::data` and `Cell::data` from `pub` to `pub(crate)`
  - Add `const fn data() -&gt; Option&lt;&U&gt;` accessor on `Vertex` and
    `const fn data() -&gt; Option&lt;&V&gt;` accessor on `Cell`

  - Update struct-level doc comments with `data()` / `set_vertex_data`
    / `set_cell_data` cross-references

  - Update all external-facing doctests and integration tests to use
    the new accessor
- [**breaking**] Generalize DelaunayTriangulationBuilder::new() over U [#287](https://github.com/acgetchell/delaunay/pull/287)
  [#290](https://github.com/acgetchell/delaunay/pull/290) [`7694a3e`](https://github.com/acgetchell/delaunay/commit/7694a3effba4eefb53238d15323e76452eec56ea)

  - Move `new()` from the `&lt;f64, (), D&gt;` impl to `&lt;f64, U, D&gt;` so it
    accepts any vertex data type — U is inferred from the vertex slice

  - Deprecate `from_vertices()` (now redundant for f64 vertices)
  - Migrate all `from_vertices` call sites to `new`
  - Fix type-inference regression in NaN test by annotating VertexBuilder
- Release v0.7.4 [#291](https://github.com/acgetchell/delaunay/pull/291)
  [`bee1b04`](https://github.com/acgetchell/delaunay/commit/bee1b0409f19631775cbe5bfd1b0618e54380c46)

  - chose(release): release v0.7.4

  - Bump version to v0.7.4
  - Update changelog with latest changes
  - Update documentation for release
  - Add performance results for v0.7.4

#### Changed: Clarify adaptive_tolerance_insphere delegation in 4D docs

  Update the documentation for provable error bounds to clarify that the
  `adaptive_tolerance_insphere` wrapper function remains available but now
  delegates entirely to the provable `insphere_from_matrix` path.

### Fixed

- Fix keyword validation and add publish-check recipe [`676df6b`](https://github.com/acgetchell/delaunay/commit/676df6b36d1ba78a17388baea685f0b4b082a844)

  - Replace `computational-geometry` keyword with `geometry` (crates.io
    enforces a 20-character limit)

  - Add `just publish-check` recipe that validates crates.io metadata
    (keywords, categories, description) and runs `cargo publish --dry-run`

  - Add publish-check to RELEASING.md Step 1.3 so metadata issues are
    caught in the release PR, not at publish time

  - Remove redundant dry-run from RELEASING.md Step 2.7

## [0.7.3] - 2026-03-24

### ⚠️ Breaking Changes

- Add SoS module for deterministic degeneracy resolution [#233](https://github.com/acgetchell/delaunay/pull/233)
  [#251](https://github.com/acgetchell/delaunay/pull/251)

- Remove use_robust_on_ambiguous override from flip repair [#228](https://github.com/acgetchell/delaunay/pull/228)
  [#255](https://github.com/acgetchell/delaunay/pull/255)

- Apply SoS to AdaptiveKernel::orientation() and tolerate degener… [#264](https://github.com/acgetchell/delaunay/pull/264)

- Improve 4D debug harness diagnostics, add capped repair and regression test [#230](https://github.com/acgetchell/delaunay/pull/230)
  [#277](https://github.com/acgetchell/delaunay/pull/277)

- Remove RobustPredicateConfig and config_presets [#259](https://github.com/acgetchell/delaunay/pull/259)
  [#260](https://github.com/acgetchell/delaunay/pull/260)

- Rename TdsValidationError to TdsError [#262](https://github.com/acgetchell/delaunay/pull/262) [#265](https://github.com/acgetchell/delaunay/pull/265)

- Replace custom changelog pipeline with git-cliff [#247](https://github.com/acgetchell/delaunay/pull/247)

### Merged Pull Requests

- Simplify trait bounds and re-export secondary maps [#282](https://github.com/acgetchell/delaunay/pull/282)
- Bump codecov/codecov-action from 5.5.2 to 5.5.3 [#280](https://github.com/acgetchell/delaunay/pull/280)
- Bump taiki-e/install-action from 2.68.34 to 2.69.6 [#279](https://github.com/acgetchell/delaunay/pull/279)
- Bump arc-swap from 1.8.2 to 1.9.0 in the dependencies group [#278](https://github.com/acgetchell/delaunay/pull/278)
- Improve 4D debug harness diagnostics, add capped repair and regression test [#230](https://github.com/acgetchell/delaunay/pull/230)
  [#277](https://github.com/acgetchell/delaunay/pull/277)
- Add progressive scale-invariant perturbation [#209](https://github.com/acgetchell/delaunay/pull/209) [#274](https://github.com/acgetchell/delaunay/pull/274)
- Add ExactPredicates marker trait for flip repair type safety (#… [#273](https://github.com/acgetchell/delaunay/pull/273)
- Identity-based SoS perturbation via canonical vertex ordering (… [#272](https://github.com/acgetchell/delaunay/pull/272)
- Bump astral-sh/setup-uv from 7.3.1 to 7.5.0 [#271](https://github.com/acgetchell/delaunay/pull/271)
- Bump actions-rust-lang/setup-rust-toolchain [#270](https://github.com/acgetchell/delaunay/pull/270)
- Bump taiki-e/install-action from 2.68.25 to 2.68.33 [#269](https://github.com/acgetchell/delaunay/pull/269)
- Bump tracing-subscriber in the dependencies group [#268](https://github.com/acgetchell/delaunay/pull/268)
- Bump actions/download-artifact from 8.0.0 to 8.0.1 [#267](https://github.com/acgetchell/delaunay/pull/267)
- Rename TdsValidationError to TdsError [#262](https://github.com/acgetchell/delaunay/pull/262) [#265](https://github.com/acgetchell/delaunay/pull/265)
- Apply SoS to AdaptiveKernel::orientation() and tolerate degener… [#264](https://github.com/acgetchell/delaunay/pull/264)
- Canonicalize positive orientation after bulk construction repair… [#261](https://github.com/acgetchell/delaunay/pull/261)
- Remove RobustPredicateConfig and config_presets [#259](https://github.com/acgetchell/delaunay/pull/259)
  [#260](https://github.com/acgetchell/delaunay/pull/260)
- Remove use_robust_on_ambiguous override from flip repair [#228](https://github.com/acgetchell/delaunay/pull/228)
  [#255](https://github.com/acgetchell/delaunay/pull/255)
- Add SoS module for deterministic degeneracy resolution [#233](https://github.com/acgetchell/delaunay/pull/233)
  [#251](https://github.com/acgetchell/delaunay/pull/251)
- Replace panicking calls with error propagation [#242](https://github.com/acgetchell/delaunay/pull/242) [#250](https://github.com/acgetchell/delaunay/pull/250)
- Replace derive_builder with hand-written VertexBuilder [#212](https://github.com/acgetchell/delaunay/pull/212)
  [#249](https://github.com/acgetchell/delaunay/pull/249)
- Archive completed changelog minor series into per-minor files [#248](https://github.com/acgetchell/delaunay/pull/248)
- Replace custom changelog pipeline with git-cliff [#247](https://github.com/acgetchell/delaunay/pull/247)

### Added

- Archive completed changelog minor series into per-minor files [#248](https://github.com/acgetchell/delaunay/pull/248)
  [`45e4781`](https://github.com/acgetchell/delaunay/commit/45e47818b93b87aa2c0970d3bf7da0159d299cf1)

  Add archive_changelog.py to split CHANGELOG.md by minor series:

  - Parse version blocks, group by X.Y minor key, write completed
    minors to docs/archive/changelog/X.Y.md (0.2–0.6)

  - Extract and distribute git-cliff reference-style link definitions
    so each output file contains only its own version defs (fixes MD053)

  - Keep Unreleased + active minor (0.7) in root CHANGELOG.md with an
    Archives link section; root shrinks from ~4,000 to ~1,400 lines

  - Pipeline is idempotent: repeated runs produce no diff

  Update tag_release.py to fall back to archive files when a requested
  version is no longer in the root changelog (extract_changelog_section
  and _github_anchor both search docs/archive/changelog/X.Y.md).

  Register archive-changelog CLI entry point in pyproject.toml and wire
  it into the justfile changelog recipe after postprocess-changelog.

  Add 24 tests covering parsing, grouping, link-def distribution,
  archive writing, idempotency, and tag_release archive fallback.

  Other changes:

  - Sort justfile recipes in lexicographic order
  - Update docs/code_organization.md tree (new files, fix NBSP encoding)
  - Update AGENTS.md, README.md, RELEASING.md with archive references
  - Exclude docs/archive/changelog/** from typos checks
- [**breaking**] Add SoS module for deterministic degeneracy resolution [#233](https://github.com/acgetchell/delaunay/pull/233)
  [#251](https://github.com/acgetchell/delaunay/pull/251) [`6ed6f88`](https://github.com/acgetchell/delaunay/commit/6ed6f889b6bd2df7d263d4aae108823ebb71b608)

  - New `src/geometry/sos.rs` with dimension-generic SoS tie-breaking
  - `sos_orientation_sign&lt;D&gt;()` for degenerate orientation predicates
  - `sos_insphere_sign&lt;D&gt;()` for co-spherical insphere predicates
  - Two-stage exact det sign: `det_errbound()` fast filter + Bareiss exact
  - 18 unit tests covering 2D–5D degenerate configurations
  - No coordinate modification — purely a decision rule

  Part of AdaptiveKernel implementation (B2 of #233).
- [**breaking**] Remove use_robust_on_ambiguous override from flip repair [#228](https://github.com/acgetchell/delaunay/pull/228)
  [#255](https://github.com/acgetchell/delaunay/pull/255) [`faf84de`](https://github.com/acgetchell/delaunay/commit/faf84ded97e8343fd36292d38b051be9df39c353)

  Remove the `use_robust_on_ambiguous` flag and `robust_insphere_sign()`
  fallback from flip repair. With AdaptiveKernel providing exact+SoS
  predicates, the old tolerance-based override was the root cause of
  flip-repair non-convergence.

  - Remove override blocks in k2/k3 violation functions
  - Remove `both_positive_artifact` workaround
  - Simplify repair attempts from 3 to 2 (FIFO then LIFO)
  - Remove `used_robust_predicates` from `DelaunayRepairDiagnostics`
  - Fix pre-existing clippy `match_same_arms` in matrix.rs/measures.rs
- [**breaking**] Apply SoS to AdaptiveKernel::orientation() and tolerate degener… [#264](https://github.com/acgetchell/delaunay/pull/264)
  [`526b39e`](https://github.com/acgetchell/delaunay/commit/526b39e4c1f634208db9014793806e2e94f9c0ed)

#### Added: Apply SoS to AdaptiveKernel::orientation() and tolerate degenerate cells [#263](https://github.com/acgetchell/delaunay/pull/263)

- Apply Simulation of Simplicity to AdaptiveKernel::orientation() so
  degenerate ties are broken deterministically (returns ±1 for all
  distinct-point inputs, 0 only for identical f64 points)

- Tolerate degenerate cells (zero exact determinant) in orientation
  normalization after flip-based Delaunay repair:

  - promote_cells_to_positive_orientation: skip instead of error
  - cells_require_positive_orientation_promotion: skip instead of error
  - canonicalize_global_orientation_sign: scan past degenerate cells
  - validate_geometric_cell_orientation: only flag negative orientation

- Add Hilbert-sort preprocessing dedup (Phase 4 in order_vertices_hilbert)
  that removes vertices mapping to the same quantized grid cell

- Add per-cell coordinate uniqueness validation (DuplicateCoordinatesInCell
  variant, CellCoordinateUniqueness invariant kind)

- Document three-layer duplicate vertex handling strategy in
  docs/numerical_robustness_guide.md

- Update convex_hull and quality tests to accept InsufficientVertices
  alongside GeometricDegeneracy for extreme-scale dedup inputs

- Add macro-generated dedup integration tests covering 2D–5D

#### Added: Normalize SoS orientation callers and harden Hilbert dedup [#263](https://github.com/acgetchell/delaunay/pull/263)

- Refactor evaluate_cell_orientation_for_context to use robust_orientation
  as the sole oracle, removing the duplicate kernel call

- Add robust_orientation guard in apply_bistellar_flip_with_k to reject
  degenerate cells before kernel invocation (fixes 4D perf regression)

- Switch find_boundary_edge_split_facet to robust_orientation for true
  collinearity/degeneracy detection instead of kernel SoS

- Use robust_orientation sign directly in build_initial_simplex, removing
  unused kernel variable and redundant orientation call

- Extract hilbert_dedup_sorted from order_vertices_hilbert so the
  ordering function is pure; apply dedup in
  preprocess_vertices_for_construction after Hilbert sort regardless of
  DedupPolicy (safety-critical for SoS identical-point failures)

- Add validate_cell_coordinate_uniqueness to validation_report with
  CoordinateScalar bound

- Fix quantization bits table in numerical_robustness_guide.md (2D/3D
  capped at 31 bits, not 64/42)

- Update rustdoc for cells_require_positive_orientation_promotion to
  reflect skip-degenerate semantics

- Trim inaccurate near-duplicate claim from dedup_batch_construction docs

- Add SoS identical-points regression tests (2D–5D) verifying
  AdaptiveKernel returns 0 when all cofactors vanish

- Add standalone hilbert_dedup_sorted edge-case tests

#### Changed: Address review comments for SoS orientation normalization [#263](https://github.com/acgetchell/delaunay/pull/263)

- Propagate `robust_orientation` errors in `apply_bistellar_flip_with_k`
  instead of silently ignoring `Err` via `matches!`

- Switch `k2_flip_would_create_degenerate_cell` from kernel orientation
  to `robust_orientation` so degenerate cells are detected under SoS;
  remove unused `kernel` parameter

- Delegate `AdaptiveKernel::orientation()` to `robust_orientation` for
  layers 1+2, eliminating duplicated matrix build + exact_det_sign logic

- Remove unused `kernel` parameter from `find_boundary_edge_split_facet`
  and update call site in `extend_hull`

- Update stale rustdoc: `validate_geometric_cell_orientation` and
  `build_initial_simplex` now reference `robust_orientation` instead of
  kernel predicates / `K::default()`

- Add `vkeys[i] == vkeys[j]` guard in `validate_cell_coordinate_uniqueness`
  to avoid misleading error when vertex keys are duplicated

- Replace broad `assert!(result.is_err())` in all-duplicates test with
  precise `InsufficientVertices` match

- Decouple many-duplicates test from magic `$dim + 2` by using the
  helper's returned distinct count

- Update docs: numerical_robustness_guide, KNOWN_ISSUES_4D,
  ORIENTATION_SPEC, validation.md

#### Changed: Remove unused kernel parameter from flip-application functions [#263](https://github.com/acgetchell/delaunay/pull/263)

- Replace `K: Kernel&lt;D&gt;` with `T: CoordinateScalar` in 7 functions:
  apply_bistellar_flip_with_k, apply_bistellar_flip,
  apply_bistellar_flip_dynamic, apply_bistellar_flip_k2/k3/k1,
  apply_bistellar_flip_k1_inverse
- Update ~40 call sites across flips.rs, triangulation/flips.rs,
  delaunay_triangulation.rs
- Delete ZeroOrientationKernel2d test struct (now redundant)
- Update ORIENTATION_SPEC.md flip pseudocode to match new signatures
- Address review round 3 comments: guard coordinate uniqueness
  validation, add #[non_exhaustive] to InvariantKind, extract
  generic flip_would_create_degenerate_cell, add nitpick tests

#### Fixed: Remove Morton and Lexicographic ordering, unconditional Hilbert dedup, strengthen flip assertions

[#263](https://github.com/acgetchell/delaunay/pull/263)

- Strengthen assert_context_has_nonzero_robust_orientation to explicitly
  fail on Err variants instead of only checking for DEGENERATE

- Merge hilbert_dedup_sorted into order_vertices_hilbert via
  dedup_quantized parameter, eliminating redundant re-quantization

- Make Hilbert quantized dedup unconditional (not gated on DedupPolicy)

- Update DedupPolicy docs to frame as performance-tuning knob

- Remove InsertionOrderStrategy::Morton variant and all supporting code
  (~100 lines: morton_bits_per_coord, morton_code, order_vertices_morton)

- Remove InsertionOrderStrategy::Lexicographic variant and associated
  tests; keep internal order_vertices_lexicographic as Hilbert fallback

- Remove Morton/Lexicographic doc references from README, invariants.md,
  numerical_robustness_guide.md, and triangulation_generation.rs

- Identity-based SoS perturbation via canonical vertex ordering (… [#272](https://github.com/acgetchell/delaunay/pull/272)
  [`a125d98`](https://github.com/acgetchell/delaunay/commit/a125d988566bb7196c025212833f3e539665e7de)

#### Added: Identity-based SoS perturbation via canonical vertex ordering [#266](https://github.com/acgetchell/delaunay/pull/266)

- Add canonical_points module with sorted_cell_points and
  sorted_facet_points_with_extra helpers that sort vertices by
  VertexKey identity before resolving to points

- Update all 5 kernel call sites (locate, flips, triangulation) to
  use canonical ordering for consistent SoS tie-breaking

- Fix misleading error variant in find_conflict_region BFS traversal
  (InvalidStartCell → CellDataAccessFailed)

- Add 6 unit tests including 2D/3D permutation-invariance tests

#### Added: Identity-based SoS perturbation via canonical vertex ordering [#266](https://github.com/acgetchell/delaunay/pull/266)

- Add canonical_points module with sorted_cell_points and
  sorted_facet_points_with_extra helpers that sort vertices by
  VertexKey identity before resolving to points

- Update all 5 kernel call sites (locate, flips, triangulation) to
  use canonical ordering for consistent SoS tie-breaking

- Fix misleading error variant in find_conflict_region BFS traversal
  (InvalidStartCell → CellDataAccessFailed)

- Document canonical ordering convention in numerical_robustness_guide

- Add canonical_points.rs to code_organization.md util listing

- Add 14 new tests: permutation invariance (2D all 6, 3D all 24),
  canonical ordering helpers, and error-path coverage for
  is_point_outside_facet and find_conflict_region

#### Changed: Reuse canonical cell ordering for query simplex construction

  Optimize the construction of the query simplex by reusing the
  canonical vertex ordering of the cell. This ensures consistent
  identity-based SoS perturbation and includes a safety check to
  prevent out-of-range facet index access.

- Add ExactPredicates marker trait for flip repair type safety (#… [#273](https://github.com/acgetchell/delaunay/pull/273)
  [`4877151`](https://github.com/acgetchell/delaunay/commit/48771518b2b5a8a6dd9b4036b0903f7c3158b7c9)

#### Added: Add ExactPredicates marker trait for flip repair type safety [#257](https://github.com/acgetchell/delaunay/pull/257)

- Define `ExactPredicates` marker trait in `kernel.rs`, implemented for
  `AdaptiveKernel` and `RobustKernel` but not `FastKernel`

- Add `K: ExactPredicates` bound to flip repair entry points in
  `flips.rs` and propagate through `delaunay_triangulation.rs`,
  `builder.rs`, and `triangulation_generation.rs`

- Add `compile_fail` doctest asserting `FastKernel` cannot satisfy the
  bound, plus positive compile-time assertion tests for the other kernels

- Update test code to use `AdaptiveKernel` where `DelaunayTriangulation`
  construction or flip repair is invoked

- Document the trait and its design rationale in
  `numerical_robustness_guide.md`

#### Changed: Narrow ExactPredicates bound to public repair methods only [#257](https://github.com/acgetchell/delaunay/pull/257)

- Remove K: ExactPredicates from ~20 internal, construction, insertion,
  and removal methods so FastKernel remains usable for all operations
  except explicit Delaunay repair

- Retain ExactPredicates bound on repair_delaunay_with_flips,
  repair_delaunay_with_flips_advanced, and rebuild_with_heuristic

- Replace run_flip_repair_fallbacks with inline robust-only fallback in
  maybe_repair_after_insertion; remove dead code (run_flip_repair_fallbacks,
  remap_vertex_key_by_uuid, HeuristicRebuildRecursionGuard::in_progress)

- Update with_kernel / build_with_kernel docs to note FastKernel is
  accepted but exact kernels are recommended

- Restructure heuristic rebuild test to test_vertex_key_valid_after_explicit_heuristic_rebuild;
  remove test_run_flip_repair_fallbacks_smoke_ok_with_local_seed

#### Fixed: Preserve full fallback-chain context in advanced repair errors [#257](https://github.com/acgetchell/delaunay/pull/257)

- Restore test_with_kernel_fast_kernel to use FastKernel (regressed when
  ExactPredicates bound was added, now possible again after narrowing)

- Restructure repair_delaunay_with_flips_advanced fallback: when primary,
  robust, and heuristic rebuild all fail, the HeuristicRebuildFailed error
  now includes context from all three stages instead of losing the robust
  fallback error

- Add test_advanced_repair_fallback_error_preserves_full_chain_context
  asserting the composed error message includes primary, robust, and
  heuristic failure details

#### Changed: Remove redundant ScalarSummable where clauses [#257](https://github.com/acgetchell/delaunay/pull/257)

- Remove 26 redundant `K::Scalar: ScalarSummable` where clauses from
  `delaunay_triangulation.rs`; the impl block already bounds
  `K::Scalar: ScalarAccumulative` which implies `ScalarSummable` via
  blanket impl
- Remove unused `ScalarSummable` import
- Export `ExactPredicates` from `prelude::query`
- Update docs (numerical_robustness_guide, validation, workflows,
  api_design) to reflect the narrowed `ExactPredicates` bound on
  public repair methods and add `FastKernel` + `EveryN` guidance
- Add tests for non-retryable error pass-through in
  `repair_delaunay_with_flips` and `repair_delaunay_with_flips_advanced`

#### Added: Add test injection for robust-fallback coverage [#257](https://github.com/acgetchell/delaunay/pull/257)

- Add FORCE_REPAIR_NONCONVERGENT thread-local and test guard
- Inject synthetic NonConvergent errors in repair_delaunay_with_flips
  and maybe_repair_after_insertion to exercise robust-fallback paths
- Add tests for advanced robust fallback and insertion robust fallback
- Move test-only helpers into mod tests with pub(super) visibility
- Add progressive scale-invariant perturbation [#209](https://github.com/acgetchell/delaunay/pull/209) [#274](https://github.com/acgetchell/delaunay/pull/274)
  [`4c35028`](https://github.com/acgetchell/delaunay/commit/4c35028537b9011d586be3b9b234925e3ca5bb5a)

  - Replace hardcoded perturbation retry count with
    DEFAULT_PERTURBATION_RETRIES = 3 (4 total attempts)

  - Apply 10^(attempt-1) progressive scaling factor per retry,
    spanning 4 orders of magnitude (e.g. 1e-8 → 1e-5 for f64)

  - Update debug messages with computed max perturbation values
  - Update numerical robustness guide documentation
  - Add tests for scale invariance, f32 base epsilon, and constant value
  - Remove 32 redundant `where K::Scalar: CoordinateScalar` clauses
    (implied by impl-level bounds)

#### Fixed: Correct perturbation exponent off-by-one and improve test coverage [#209](https://github.com/acgetchell/delaunay/pull/209)

- Fix progressive scale factor: use 10^attempt instead of 10^(attempt-1)
  so the retry ladder reaches 1e-5 × local_scale (was capped at 1e-6),
  matching the documented "4 orders of magnitude" range

- Apply the same correction to the debug-message exponent computation

- Update perturbation ladder in numerical_robustness_guide.md to reflect
  the corrected values (1e-7, 1e-6, 1e-5)

- Replace test_perturbation_f32_base_epsilon with
  test_perturbation_epsilon_selection_and_retry: asserts mantissa_digits
  for both f32 and f64, exercises insert_transactional with on-edge
  near-degenerate points for both scalar types

- Add absolute expected counts to test_perturbation_scale_invariance_3d
  to catch regressions that affect all scales equally

- Add test_perturbation_retry_and_exhaustion_4d: 4D random points
  exercise the progressive retry loop body and exhaustion branch

- Add test_perturbation_retry_seeded_branch_4d: calls
  insert_transactional directly with perturbation_seed != 0 to cover
  the seeded sign-selection path

- [**breaking**] Improve 4D debug harness diagnostics, add capped repair and regression test [#230](https://github.com/acgetchell/delaunay/pull/230)
  [#277](https://github.com/acgetchell/delaunay/pull/277) [`0684ec0`](https://github.com/acgetchell/delaunay/commit/0684ec01108cc6a816d5aa5b41179726ee7a51a5)

  Add configurable flip-budget capping, structured debug-harness outcomes,
  enriched orientation diagnostics, and a seeded 4D regression test.

  Flip budget enforcement:

  - Move max_flips check before apply_bistellar_flip in all 5 repair
    paths (k2, k3, dynamic, inverse) so Some(0) means truly zero flips

  - Change guard from `&gt;` to `&gt;=` to eliminate off-by-one
  - Thread max_flips_override through repair_delaunay_with_flips_k2_k3,
    the robust fallback, heuristic rebuild, and per-insertion repair

  Capped repair API:

  - Add max_flips: Option&lt;usize&gt; to DelaunayRepairHeuristicConfig
  - Mark DelaunayRepairHeuristicConfig #[non_exhaustive]
  - Add internal repair_delaunay_with_flips_capped() helper
  - Add DELAUNAY_LARGE_DEBUG_REPAIR_MAX_FLIPS env var to debug harness

  Error taxonomy:

  - Add RepairFailed variant to DelaunayTriangulationValidationError for
    mutating operations (remove_vertex); fix remove_vertex to use it
    instead of VerificationFailed (which is for passive validation)

  Orientation diagnostics:

  - Enrich NegativeOrientation error message with vertex keys
  - Add tracing::warn with cell key, UUID, vertex keys, neighbor keys,
    and orientation sign for negative-orientation cells

  Debug harness:

  - Return DebugOutcome enum instead of panicking (Success,
    ConstructionFailure, SkippedVertices, RepairNonConvergence,
    ValidationFailure)

  - Print seed replay command on every abort path
  - Add per-chunk insertion rate (pts/s) in progress output
  - Harness tests now assert on DebugOutcome::Success

### Changed

- Replace panicking calls with error propagation [#242](https://github.com/acgetchell/delaunay/pull/242) [#250](https://github.com/acgetchell/delaunay/pull/250)
  [`9d640c3`](https://github.com/acgetchell/delaunay/commit/9d640c378bdefd0043c28970b05b7162c47dc1bf)

  - Add `InternalInconsistency` variant to `TriangulationConstructionError`
    for internal bookkeeping failures distinct from geometric degeneracy

  - Replace 5 `.expect()` calls on HashMap lookups in builder.rs periodic
    quotient reconstruction with `?` propagation

  - Reclassify 3 additional internal operations (image vertex removal,
    neighbor assignment, incident-cell rebuild) from `GeometricDegeneracy`
    to `InternalInconsistency`

  - Replace `.expect()` on `initial_points.take()` in
    triangulation_generation.rs with error propagation

  - Update doc comment from `# Panics` to `# Errors`
  - Add single-quote guidance for gh --body in AGENTS.md
- [**breaking**] Remove RobustPredicateConfig and config_presets [#259](https://github.com/acgetchell/delaunay/pull/259)
  [#260](https://github.com/acgetchell/delaunay/pull/260) [`6764bac`](https://github.com/acgetchell/delaunay/commit/6764bacd7a02a9e7c95c88235b68bfcbc447eae3)

  - Delete RobustPredicateConfig struct and config_presets module
  - Remove config parameter from robust_insphere, robust_orientation,
    adaptive_tolerance_insphere, and verify_insphere_consistency

  - Simplify RobustKernel to a zero-size type (remove config field and
    with_config constructor, derive Default)

  - Remove config threading from flips.rs, delaunay_validation.rs, and
    coordinate_conversion_errors.rs

  - Delete tests that only exercised config field values
  - Update numerical_robustness_guide.md: document AdaptiveKernel as the
    default kernel, remove config_presets references
- [**breaking**] Rename TdsValidationError to TdsError [#262](https://github.com/acgetchell/delaunay/pull/262)
  [#265](https://github.com/acgetchell/delaunay/pull/265) [`99b9810`](https://github.com/acgetchell/delaunay/commit/99b9810c7aeeb91f84efba019bb199da8ee4f87a)

  - Rename TdsValidationError -&gt; TdsError across all source files
  - Remove the type alias that bridged the old name
  - Update all references in production code, tests, and examples
  - Part of error hierarchy orthogonalization (Phase 5)

#### Changed: Make TriangulationValidationError purely Level 3 [#262](https://github.com/acgetchell/delaunay/pull/262)

  Remove the Tds variant from TriangulationValidationError so it contains
  only Level 3 (topology) errors. TDS-level errors now flow through
  InvariantError, keeping Levels 1–2 orthogonal from Level 3.

  API changes:

- is_valid, validate, validate_at_completion, validate_after_insertion
  now return InvariantError instead of TriangulationValidationError
- validate_geometric_cell_orientation returns TdsError directly
- remove_vertex returns InvariantError
- Add Tds variant to DelaunayTriangulationValidationError
- TdsMutationError inner field made private; add as_tds_error() and
  into_inner() accessors
- Remove From&lt;TdsMutationError&gt; for TriangulationValidationError
- Add From&lt;ManifoldError&gt; for InvariantError
- Simplify trait bounds and re-export secondary maps [#282](https://github.com/acgetchell/delaunay/pull/282)
  [`04ea024`](https://github.com/acgetchell/delaunay/commit/04ea024f7091b7f1d9406c48d673e58a07a657d3)

  - Re-export VertexSecondaryMap and CellSecondaryMap in prelude and
    prelude::collections; update doc examples to use prelude path

  - Replace ~50 vestigial ScalarSummable bounds with CoordinateScalar
    across geometry and core modules (Sum is never called on T)

  - Remove ~34 redundant K::Scalar: CoordinateScalar where clauses
    already implied by K: Kernel&lt;D&gt; (associated type bound)

  - Remove 8 redundant [T; D]: Copy + Sized bounds already implied
    by T: CoordinateScalar (Float → Copy; arrays are always Sized)

  - Keep ScalarSummable trait definition and blanket impl for future use

### Fixed

- Canonicalize positive orientation after bulk construction repair… [#261](https://github.com/acgetchell/delaunay/pull/261)
  [`dec8df9`](https://github.com/acgetchell/delaunay/commit/dec8df9f07b5e868825f31a64bee85c4a04e984d)

#### Fixed: Canonicalize positive orientation after bulk construction repair [#258](https://github.com/acgetchell/delaunay/pull/258)

- Call normalize_and_promote_positive_orientation() in
  finalize_bulk_construction after the flip-repair block and before
  topology validation, ensuring the global geometric sign is positive

- Update #228 regression test to expect PLManifold (was Pseudomanifold)
  now that orientation is correctly canonicalized

#### Changed: Categorize errors during orientation canonicalization

  Distinguish structural `InsertionError` variants as `InternalInconsistency`
  to separate algorithmic bugs from input-related `GeometricDegeneracy`
  during post-repair orientation canonicalization. Added topology
  validation to the issue #228 regression test to ensure manifold parity.

### Maintenance

- [**breaking**] Replace custom changelog pipeline with git-cliff [#247](https://github.com/acgetchell/delaunay/pull/247)
  [`1b2af41`](https://github.com/acgetchell/delaunay/commit/1b2af41fcb5115d82c1ae6f0ab66651c075fbd52)

  Replace ~4,000 lines of custom Python changelog generation
  (changelog_utils.py, enhance_commits.py) with git-cliff and two
  focused scripts (~640 lines total):

  - postprocess_changelog.py: lightweight markdown hygiene
    (MD004, MD007, MD012, MD013, MD030, MD031, MD032, MD040)
    plus summary section injection (Merged PRs, Breaking Changes)

  - tag_release.py: extract latest version section for git tag messages

  Pipeline changes:

  - cliff.toml: full git-cliff config with conventional commit parsing,
    PR auto-linking, and Tera template for Keep a Changelog format

  - justfile: new changelog-update, changelog-tag, changelog recipes
    replacing the old generate-changelog workflow

  - Idempotent output: postprocessor matches markdownlint --fix exactly,
    so `just changelog-update && just fix` produces zero diff

  Tooling simplification:

  - Remove mypy in favor of ty (Astral) — mypy's permissive config was
    catching nothing that ty doesn't already cover

  - Disable markdownlint MD037 (false positives on cron expressions and
    glob patterns like saturating_*)
- Replace derive_builder with hand-written VertexBuilder [#212](https://github.com/acgetchell/delaunay/pull/212)
  [#249](https://github.com/acgetchell/delaunay/pull/249) [`b017b39`](https://github.com/acgetchell/delaunay/commit/b017b39bc410e96703bc7370972a930b93f6d6be)

  - Add VertexBuilderError enum and VertexBuilder struct with
    point(), data(), and build() methods in src/core/vertex.rs

  - Remove #[derive(Builder)] and #[builder(...)] attributes from Vertex
  - Remove derive_builder dependency from Cargo.toml and lib.rs
  - All ~30 call sites unchanged — builder API is a drop-in replacement
- Bump actions/download-artifact from 8.0.0 to 8.0.1 [#267](https://github.com/acgetchell/delaunay/pull/267)
  [`3727781`](https://github.com/acgetchell/delaunay/commit/37277814a118d100bffed0e6fe48953094936280)

  Bumps [actions/download-artifact](https://github.com/actions/download-artifact) from 8.0.0 to 8.0.1.

  - [Release notes](https://github.com/actions/download-artifact/releases)
  - [Commits](https://github.com/actions/download-artifact/compare/70fc10c6e5e1ce46ad2ea6f2b72d43f7d47b13c3...3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c)

---

  updated-dependencies:

- dependency-name: actions/download-artifact
  dependency-version: 8.0.1
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump tracing-subscriber in the dependencies group [#268](https://github.com/acgetchell/delaunay/pull/268)
  [`7d355f4`](https://github.com/acgetchell/delaunay/commit/7d355f43418a3bad0da1badf5fde3a19d6306d37)

  Bumps the dependencies group with 1 update: [tracing-subscriber](https://github.com/tokio-rs/tracing).

  Updates `tracing-subscriber` from 0.3.22 to 0.3.23

  - [Release notes](https://github.com/tokio-rs/tracing/releases)
  - [Commits](https://github.com/tokio-rs/tracing/compare/tracing-subscriber-0.3.22...tracing-subscriber-0.3.23)

---

  updated-dependencies:

- dependency-name: tracing-subscriber
  dependency-version: 0.3.23
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies
  ...

- Bump taiki-e/install-action from 2.68.25 to 2.68.33 [#269](https://github.com/acgetchell/delaunay/pull/269)
  [`f7fb663`](https://github.com/acgetchell/delaunay/commit/f7fb6638c744f76a92325117160a420eb85e3d0b)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.68.25 to 2.68.33.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/a37010ded18ff788be4440302bd6830b1ae50d8b...cbb1dcaa26e1459e2876c39f61c1e22a1258aac5)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.68.33
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump actions-rust-lang/setup-rust-toolchain [#270](https://github.com/acgetchell/delaunay/pull/270)
  [`73b8d63`](https://github.com/acgetchell/delaunay/commit/73b8d63896386e1f17d14e1b10421059d17a79c3)

  Bumps [actions-rust-lang/setup-rust-toolchain](https://github.com/actions-rust-lang/setup-rust-toolchain) from 1.15.3 to 1.15.4.

  - [Release notes](https://github.com/actions-rust-lang/setup-rust-toolchain/releases)
  - [Changelog](https://github.com/actions-rust-lang/setup-rust-toolchain/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/actions-rust-lang/setup-rust-toolchain/compare/a0b538fa0b742a6aa35d6e2c169b4bd06d225a98...150fca883cd4034361b621bd4e6a9d34e5143606)

---

  updated-dependencies:

- dependency-name: actions-rust-lang/setup-rust-toolchain
  dependency-version: 1.15.4
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump astral-sh/setup-uv from 7.3.1 to 7.5.0 [#271](https://github.com/acgetchell/delaunay/pull/271)
  [`3225d64`](https://github.com/acgetchell/delaunay/commit/3225d64974cb1d7772cf5cde4be6206578ef2da9)

  Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.3.1 to 7.5.0.

  - [Release notes](https://github.com/astral-sh/setup-uv/releases)
  - [Commits](https://github.com/astral-sh/setup-uv/compare/5a095e7a2014a4212f075830d4f7277575a9d098...e06108dd0aef18192324c70427afc47652e63a82)

---

  updated-dependencies:

- dependency-name: astral-sh/setup-uv
  dependency-version: 7.5.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump arc-swap from 1.8.2 to 1.9.0 in the dependencies group [#278](https://github.com/acgetchell/delaunay/pull/278)
  [`da93901`](https://github.com/acgetchell/delaunay/commit/da93901c78a40e8e484e3fb21b117156167ec2a0)

  Bumps the dependencies group with 1 update: [arc-swap](https://github.com/vorner/arc-swap).

  Updates `arc-swap` from 1.8.2 to 1.9.0

  - [Changelog](https://github.com/vorner/arc-swap/blob/master/CHANGELOG.md)
  - [Commits](https://github.com/vorner/arc-swap/compare/v1.8.2...v1.9.0)

---

  updated-dependencies:

- dependency-name: arc-swap
  dependency-version: 1.9.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  dependency-group: dependencies
  ...

- Bump taiki-e/install-action from 2.68.34 to 2.69.6 [#279](https://github.com/acgetchell/delaunay/pull/279)
  [`6c7c8b1`](https://github.com/acgetchell/delaunay/commit/6c7c8b190e563f18555cbfee14e8608be1dadeed)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.68.34 to 2.69.6.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/de6bbd1333b8f331563d54a051e542c7dfef81c3...06203676c62f0d3c765be3f2fcfbebbcb02d09f5)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.69.6
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump codecov/codecov-action from 5.5.2 to 5.5.3 [#280](https://github.com/acgetchell/delaunay/pull/280)
  [`47394f3`](https://github.com/acgetchell/delaunay/commit/47394f3543d4cbd2c69f45a6ce0c54814bd33875)

  Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 5.5.2 to 5.5.3.

  - [Release notes](https://github.com/codecov/codecov-action/releases)
  - [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/codecov/codecov-action/compare/671740ac38dd9b0130fbe1cec585b89eea48d3de...1af58845a975a7985b0beb0cbe6fbbb71a41dbad)

---

  updated-dependencies:

- dependency-name: codecov/codecov-action
  dependency-version: 5.5.3
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

## [0.7.2] - 2026-03-10

### Merged Pull Requests

- Exact insphere predicates with f64 fast filter [#245](https://github.com/acgetchell/delaunay/pull/245)
- Use exact-sign orientation in robust_orientation() [#244](https://github.com/acgetchell/delaunay/pull/244)
- Bump actions/setup-node from 6.2.0 to 6.3.0 [#239](https://github.com/acgetchell/delaunay/pull/239)
- Bump taiki-e/install-action from 2.68.16 to 2.68.25 [#237](https://github.com/acgetchell/delaunay/pull/237)
- Use exact arithmetic for orientation predicates [#235](https://github.com/acgetchell/delaunay/pull/235)
  [#236](https://github.com/acgetchell/delaunay/pull/236)
- Switch FastKernel to insphere_lifted and enable LTO [#234](https://github.com/acgetchell/delaunay/pull/234)
- Deduplicate D&lt;4 repair fallback and improve diagnostics [#232](https://github.com/acgetchell/delaunay/pull/232)
- Resolve 3D seeded bulk construction orientation convergence failure [#228](https://github.com/acgetchell/delaunay/pull/228)
- Bump actions-rust-lang/setup-rust-toolchain [#226](https://github.com/acgetchell/delaunay/pull/226)
- Bump taiki-e/install-action from 2.68.8 to 2.68.16 [#225](https://github.com/acgetchell/delaunay/pull/225)
- Bump astral-sh/setup-uv from 7.3.0 to 7.3.1 [#224](https://github.com/acgetchell/delaunay/pull/224)
- Bump actions/upload-artifact from 6 to 7 [#223](https://github.com/acgetchell/delaunay/pull/223)
- Bump actions/download-artifact from 7.0.0 to 8.0.0 [#222](https://github.com/acgetchell/delaunay/pull/222)
- Introduce GlobalTopology behavior model adapter [#221](https://github.com/acgetchell/delaunay/pull/221)
- Enforce coherent orientation as a first-class invariant [#219](https://github.com/acgetchell/delaunay/pull/219)
- Use bulk Hilbert API in order_vertices_hilbert [#218](https://github.com/acgetchell/delaunay/pull/218)
- Improve Hilbert curve correctness and add bulk API [#207](https://github.com/acgetchell/delaunay/pull/207)
  [#216](https://github.com/acgetchell/delaunay/pull/216)
- Update docs for DelaunayTriangulationBuilder and toroidal topology [#215](https://github.com/acgetchell/delaunay/pull/215)
- Feat/210 toroidalspace periodic [#213](https://github.com/acgetchell/delaunay/pull/213)
- Bump taiki-e/install-action from 2.67.30 to 2.68.8 [#211](https://github.com/acgetchell/delaunay/pull/211)

### Added

- Enforce coherent orientation as a first-class invariant [#219](https://github.com/acgetchell/delaunay/pull/219)
  [`350f614`](https://github.com/acgetchell/delaunay/commit/350f614c3e18d148bfc88809c28fdc2de362dd9a)

  - add Level 2 coherent-orientation validation via `is_coherently_oriented()` and `OrientationViolation` diagnostics
  - preserve/normalize orientation across flips, cavity/neighbor rebuild paths, periodic quotient reconstruction, and vertex-removal retriangulation
  - add orientation coverage in `tests/tds_orientation.rs`, document the invariant, and update related docs/doctest examples

#### Changed: Refine coherent orientation handling and validation

  The flip algorithms now explicitly manage the local orientation of newly created cells, canonicalizing their vertex order to a positive orientation before
  insertion. This ensures consistent orientation within individual cells as they are added.

  Introduced `facet_vertex_identities_in_cell_order` to `TriangulationDataStructure` which derives canonical facet identities (vertex keys and relative periodic
  offsets). This improves the robustness of `normalize_coherent_orientation` and `is_coherently_oriented` by ensuring consistent comparisons across facets,
  especially in periodic domains.

  Added `resolve_facet_handle_for_key` and `resolve_ridge_handle_for_key` to `flips.rs` to allow robust re-resolution of facet/ridge handles. This is necessary
  because vertex reordering within cells (e.g., during canonicalization) can invalidate direct facet/ridge indices.

  Updates documentation to reflect the new explicit orientation management strategy during flips, where post-flip normalization and validation are now standard
  practice.

  Improved error handling by boxing `source` errors in `InsertionError` , `DelaunayTriangulationConstructionErrorWithStatistics` , and
  `TriangulationConstructionError` , reducing struct size and preventing excessive stack usage.

  Enhanced robustness of `Cell::swap_vertex_slots` with additional bounds checks and `Triangulation::build_initial_simplex` to handle cases with insufficient
  vertices more gracefully.

  Added `bump_generation` in `normalize_coherent_orientation` to correctly increment the TDS generation counter when cell orientations are modified.

  Expanded TDS orientation tests to include serialization, manual tampering of cell vertex order, and verification that `is_coherently_oriented` correctly
  detects the violation.

- Use exact arithmetic for orientation predicates [#235](https://github.com/acgetchell/delaunay/pull/235)
  [#236](https://github.com/acgetchell/delaunay/pull/236) [`a62437f`](https://github.com/acgetchell/delaunay/commit/a62437f25c27259f145d3c193ce149ee14b421c7)

  Switch to la-stack v0.2.1 with the `exact` feature to obtain provably
  correct simplex orientation via `det_sign_exact()`.

  Orientation predicates:

  - Add `orientation_from_matrix()` using `det_sign_exact()` with a
    finite-entry guard and adaptive-tolerance fallback for non-finite cases

  - `simplex_orientation()` now delegates to `orientation_from_matrix()`,
    eliminating manual tolerance comparison

  insphere_lifted optimization:

  - Reuse the relative-coordinate block already in the lifted matrix for
    orientation instead of re-converting all D+1 simplex points

  - Combine the dimension-parity sign and (-1)^D orientation correction
    into a single simplified formula: det_norm = −det × rel_sign

  - Remove `dimension_is_even`, `parity_sign`, and `orient_sign` variables

  Stack-matrix dispatch cleanup:

  - Reduce MAX_STACK_MATRIX_DIM from 18 to 7 (matching tested D≤5 range)
  - Replace 19 hand-written match arms with a compact repetition macro
  - Add `matrix_zero_like()` helper for creating same-sized zero matrices
    within macro-dispatched blocks without nested dispatch

- Use exact-sign orientation in robust_orientation() [#244](https://github.com/acgetchell/delaunay/pull/244)
  [`2869cfe`](https://github.com/acgetchell/delaunay/commit/2869cfea111dbca3641e7f88119d67b93a0d4841)

  Replace f64 determinant + adaptive tolerance in `robust_orientation()`
  with `orientation_from_matrix()`, which uses `det_sign_exact()` for
  provably correct sign classification on finite inputs.

  - Make `orientation_from_matrix` pub(crate) so robust_predicates can call it
  - Remove adaptive_tolerance / manual threshold comparison from robust_orientation()
  - Add near-degenerate 2D and 3D tests that exercise the exact-sign path

#### Changed: Ignore slow higher-dimensional Delaunay proptests

  Mark 4D and 5D incremental insertion proptests as ignored. The exact
  arithmetic fallback for orientation matrices larger than 4×4 is
  currently too slow for practical test execution. This is an internal
  change.

- Exact insphere predicates with f64 fast filter [#245](https://github.com/acgetchell/delaunay/pull/245)
  [`fed429f`](https://github.com/acgetchell/delaunay/commit/fed429f281cb2bc2e4a97cd99ac1770ade76a202)

  - Add `insphere_from_matrix` helper in predicates.rs using a 3-stage
    approach: f64 fast filter → exact Bareiss → f64 fallback

  - Update `insphere`, `insphere_lifted`, `adaptive_tolerance_insphere`,
    and `conditioned_insphere` to use exact-sign path

  - Remove dead `interpret_insphere_determinant` function
  - Add near-cocircular and near-cospherical exact-sign tests
  - Switch convex hull performance test to FastKernel to avoid 5×5
    exact Bareiss on cospherical inputs

  - Document lint suppression preference (`expect` over `allow`) in
    docs/dev/rust.md

### Changed

- Feat/210 toroidalspace periodic [#213](https://github.com/acgetchell/delaunay/pull/213)
  [`c172796`](https://github.com/acgetchell/delaunay/commit/c1727967ae2c92440c42413c10e1d5859d4cb561)

- Introduce GlobalTopology behavior model adapter [#221](https://github.com/acgetchell/delaunay/pull/221)
  [`e56265b`](https://github.com/acgetchell/delaunay/commit/e56265bafeb4a1e0e65c72b2037ab0e747af7ffa)

  - add internal `GlobalTopologyModel` abstraction and concrete models
    (euclidean, toroidal, spherical scaffold, hyperbolic scaffold)

  - add `GlobalTopologyModelAdapter` and `GlobalTopology::model()` bridge to keep
    `GlobalTopology&lt;D&gt;` as the stable public metadata/config surface

  - migrate triangulation orientation lifting to model-based behavior calls
  - migrate builder toroidal validation/canonicalization to model-based calls
  - update topology/code-organization docs for metadata-vs-behavior split

#### Changed: Improve global topology model validation and consistency

  Enhances periodic cell offset validation by leveraging `supports_periodic_facet_signatures`.
  Introduces robust checks for non-finite coordinates during point canonicalization
  in toroidal models, preventing invalid states.
  Refactors the `GlobalTopologyModelAdapter` to consistently delegate all trait
  method calls to specific underlying topology model implementations,
  improving maintainability.
  Updates error messages for clarity during topology validation.
  Optimizes `periodic_domain` to return a reference, avoiding data copies.
  Adjusts internal module visibility and re-exports `ToroidalConstructionMode` to prelude.

#### Changed: Add comprehensive documentation and tests for GlobalTopologyModel

  Enhance the global_topology_model module with extensive documentation and unit test coverage:

  Documentation improvements:

- Add module-level overview explaining trait abstraction and concrete implementations
- Enhance trait method documentation for periodic_domain() and supports_periodic_facet_signatures()
- Document public methods: ToroidalModel::new(), GlobalTopology::model(), GlobalTopologyModelAdapter::from_global_topology()

  Test coverage (41 tests added):
- EuclideanModel: comprehensive trait method coverage
- SphericalModel and HyperbolicModel: scaffolded model behavior
- GlobalTopologyModelAdapter: delegation verification for all trait methods
- Error handling: zero/negative/infinite/NaN periods, non-finite coordinates
- Edge cases: large coordinates, exact periods, zero/large offsets, f32 scalars, 5D

  Code quality improvements:
- Fix nitpick: delegate kind() and allows_boundary() in GlobalTopologyModelAdapter
- Fix nitpick: add NaN coordinate test for canonicalize_point_in_place
- Fix nitpick: add finiteness validation to lift_for_orientation with test
- Apply cargo fmt formatting
- Fix clippy warnings (float_cmp, suboptimal_flops)

#### Changed: Optimize facet vertex processing and improve periodic facet key determinism

  Moves the facet vertex buffer initialization to only execute on the non-periodic path,
  avoiding unnecessary work for periodic cells and improving efficiency.

  Enhances the `periodic_facet_key_from_lifted_vertices` function to ensure
  deterministic sorting by considering both the vertex key value and its periodic
  offset. This prevents inconsistencies when multiple lifted vertices share the same
  base key.

- Make Codacy analysis step continue on error [`24a1ad6`](https://github.com/acgetchell/delaunay/commit/24a1ad6a7e17025f40ea9d4f626f302670664c39)

  Configures the Codacy analysis step in the CI workflow to continue on
  error. This prevents the entire workflow from failing due to intermittent
  issues with Codacy's opengrep/semgrep engine, ensuring subsequent steps,
  like SARIF report uploads, can still execute. This improves CI robustness.
  This is an internal CI workflow improvement.

- Enhance Codacy CI reliability and performance [`b980630`](https://github.com/acgetchell/delaunay/commit/b9806304bff5de8d6ee554e36ae5543929c6300f)

  Disables the Semgrep engine within Codacy due to intermittent failures
  and excessively long runtimes observed in CI. Additionally, adds a
  timeout to the Codacy analysis step to prevent hung analyzers from
  consuming the full job timeout, improving overall workflow stability
  and resource utilization. This is an internal CI/CD change.

- Update MSRV to 1.93.1 and `sysinfo` dependency [`a2a42d5`](https://github.com/acgetchell/delaunay/commit/a2a42d58ed913b46bf81489658356dc4d09c3637)

  Increment the Minimum Supported Rust Version (MSRV) to 1.93.1
  across all project configuration and documentation. This updates
  the pinned Rust toolchain, `Cargo.toml` settings, and `clippy.toml`
  MSRV, ensuring consistency and compatibility. Additionally, the
  `sysinfo` dependency is updated to 0.38.3. This is an internal
  maintenance change.

- Improve 3D incremental prefix debug harness [`515dade`](https://github.com/acgetchell/delaunay/commit/515dadeb975250217702a9ae1c0a705fc16f620f)

  Refactors the `run_incremental_prefix_3d` function to use a batch construction
  method, aligning it with other large-scale debug tests and simplifying logic.
  This enhances initial triangulation robustness and error reporting by capturing
  detailed statistics from construction failures.

  Adds new environment variables, `DELAUNAY_LARGE_DEBUG_PREFIX_MAX_PROBES` and
  `DELAUNAY_LARGE_DEBUG_PREFIX_MAX_RUNTIME_SECS`, to control the bisection
  process, enabling more efficient and targeted debugging of failure points.

  Updates the `env_usize` helper to correctly parse environment variables
  provided in a `key=value` format, improving test configuration flexibility.

- Deduplicate D&lt;4 repair fallback and improve diagnostics [#232](https://github.com/acgetchell/delaunay/pull/232)
  [`14ff1b3`](https://github.com/acgetchell/delaunay/commit/14ff1b3aad4ab4c1869018d6cbbb59d2d0456fd3)

  - Extract try_d_lt4_global_repair_fallback helper to eliminate
    duplicated repair-or-abort logic between the stats and non-stats
    branches of insert_remaining_vertices_seeded.

  - Enrich the soft post-condition diagnostic in
    normalize_and_promote_positive_orientation with the count of
    residual negative cells and a sample of up to 5 CellKeys.

  - Add test_construction_options_global_repair_fallback_toggle unit
    test verifying the without_global_repair_fallback builder toggle.

#### Fixed: Switch default kernel from FastKernel to RobustKernel

- Change all convenience constructors (new, new_with_topology_guarantee,
  new_with_options, empty, etc.) to use RobustKernel&lt;f64&gt;
- Change builder build() default to RobustKernel&lt;T&gt;
- Add RobustKernel to prelude::query exports
- Update type annotations across tests, benches, and doc tests
- Preserve FastKernel in tests that explicitly test it via with_kernel()

#### Changed: Use RobustKernel for random generation and 3D examples

  Update examples and random triangulation utilities to use RobustKernel,
  aligning them with the core library's default. FastKernel is now
  explicitly documented as unreliable for 3D workloads due to floating-
  point precision issues in near-degenerate configurations. This change
  also adds topological admissibility checks for flip-based repairs and
  improves error diagnostics for per-insertion failures.

### Documentation

- Update docs for DelaunayTriangulationBuilder and toroidal topology [#215](https://github.com/acgetchell/delaunay/pull/215)
  [`a90526c`](https://github.com/acgetchell/delaunay/commit/a90526cd53be4cbe07c0add0b52ef04bd7243c3d)

  Update all documentation to reflect that toroidal topology is fully
  implemented and accessible via DelaunayTriangulationBuilder.

  Documentation updates:

  - docs/topology.md: Replace "future plumbing" language with current
    implementation status; add complete toroidal triangulation examples

  - docs/api_design.md: Split Builder API section into simple (::new())
    vs advanced (Builder) construction with toroidal examples

  - docs/workflows.md: Add new section for toroidal/periodic triangulations
    with practical examples and construction modes

  - docs/code_organization.md: Update file tree with missing files
    (invariants.md, workflows.md, tests, geometry/util/ subdirectory)

  - README.md: Add toroidal feature to Builder API section with example

  Code updates:

  - src/lib.rs: Export TopologyKind and GlobalTopology from
    prelude::triangulation for ergonomic imports

  - src/core/delaunay_triangulation.rs: Add "Advanced Configuration"
    section to ::new() documentation mentioning Builder for toroidal
    and custom options; fix redundant rustdoc link

  - examples/topology_editing_2d_3d.rs: Migrate to DelaunayTriangulationBuilder
  - benches/profiling_suite.rs: Migrate to DelaunayTriangulationBuilder

#### Changed: Adopt DelaunayTriangulationBuilder and update related documentation

  Migrates benchmark and example code to consistently use the
  DelaunayTriangulationBuilder for creating triangulations.
  This reflects the full implementation and accessibility of toroidal
  topology via the builder, and updates documentation across various
  sections (API design, workflows, topology) to guide users on its
  proper and advanced configuration. Includes internal exports for
  ergonomic usage.

### Fixed

- Ensure deterministic sorting and enforce coherent orientation
  [`c0e4d4f`](https://github.com/acgetchell/delaunay/commit/c0e4d4fffc9e3ae116062b8bd2d89baf58678517)

  Resolves multiple issues to ensure deterministic behavior and strong
  invariants across the triangulation data structure.

  Stabilizes vertex ordering, particularly for Hilbert curve sorts, by
  refining tie-breaking, error handling, and fallback logic. This prevents
  non-deterministic results and corrects inverse permutation calculations,
  addressing previously identified breaking changes related to sorting.

  Enforces a consistent positive geometric orientation for all cells
  throughout the triangulation lifecycle, making coherent orientation a
  first-class invariant. This fixes 4D construction failures and improves
  periodic self-neighbor validation by handling lifted coordinates.

  Enhances the global topology model, especially for toroidal domains,
  by improving validation, canonicalization, and periodic facet key
  derivation. This addresses edge cases related to non-finite coordinates,
  zero/negative periods, and ensures consistent behavior.

- Resolve 3D seeded bulk construction orientation convergence failure [#228](https://github.com/acgetchell/delaunay/pull/228)
  [`c181f28`](https://github.com/acgetchell/delaunay/commit/c181f289c7d36a6668a8441b78aed596b9dae36c)

  - Soften post-condition in normalize_and_promote_positive_orientation:
    add canonicalize_global_orientation_sign before the promote loop to
    prevent oscillation; demote the residual negative-orientation check
    from a hard error to a diagnostic log so near-degenerate but
    structurally valid simplices no longer abort insertion.

  - Replace enlarged local repair fallback with global
    repair_delaunay_with_flips_k2_k3 when D&lt;4 per-insertion local repair
    cycles on co-spherical FP configurations. The multi-attempt global
    repair uses robust predicates and alternate queue orders to break
    cycling.

  - Gate global repair fallback on a new ConstructionOptions field
    (use_global_repair_fallback, default true) threaded through the
    build chain via DelaunayInsertionState. The periodic builder disables
    it (.without_global_repair_fallback()) so global repair cannot
    disrupt the image-point topology; the existing 24-attempt shuffle
    retry finds a working vertex ordering instead.

### Maintenance

- Bump taiki-e/install-action from 2.67.30 to 2.68.8 [#211](https://github.com/acgetchell/delaunay/pull/211)
  [`a133758`](https://github.com/acgetchell/delaunay/commit/a13375884047d06aab280ff5cb17f48910e496f0)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.67.30 to 2.68.8.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/288875dd3d64326724fa6d9593062d9f8ba0b131...cfdb446e391c69574ebc316dfb7d7849ec12b940)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.68.8
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...
- Update docs, remove old files [`18679dc`](https://github.com/acgetchell/delaunay/commit/18679dc73e5764277906ee83d1465cb0134a9780)
- Bump actions-rust-lang/setup-rust-toolchain [#226](https://github.com/acgetchell/delaunay/pull/226)
  [`3e2cd26`](https://github.com/acgetchell/delaunay/commit/3e2cd26f67d4845445f1b876ec31695a20e745be)

  Bumps [actions-rust-lang/setup-rust-toolchain](https://github.com/actions-rust-lang/setup-rust-toolchain) from 1.15.2 to 1.15.3.

  - [Release notes](https://github.com/actions-rust-lang/setup-rust-toolchain/releases)
  - [Changelog](https://github.com/actions-rust-lang/setup-rust-toolchain/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/actions-rust-lang/setup-rust-toolchain/compare/1780873c7b576612439a134613cc4cc74ce5538c...a0b538fa0b742a6aa35d6e2c169b4bd06d225a98)

---

  updated-dependencies:

- dependency-name: actions-rust-lang/setup-rust-toolchain
  dependency-version: 1.15.3
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump actions/download-artifact from 7.0.0 to 8.0.0 [#222](https://github.com/acgetchell/delaunay/pull/222)
  [`c172f09`](https://github.com/acgetchell/delaunay/commit/c172f09dd0a0f533f47e943bfbbbfc54240e40b0)

  Bumps [actions/download-artifact](https://github.com/actions/download-artifact) from 7.0.0 to 8.0.0.

  - [Release notes](https://github.com/actions/download-artifact/releases)
  - [Commits](https://github.com/actions/download-artifact/compare/37930b1c2abaa49bbe596cd826c3c89aef350131...70fc10c6e5e1ce46ad2ea6f2b72d43f7d47b13c3)

---

  updated-dependencies:

- dependency-name: actions/download-artifact
  dependency-version: 8.0.0
  dependency-type: direct:production
  update-type: version-update:semver-major
  ...

- Bump taiki-e/install-action from 2.68.8 to 2.68.16 [#225](https://github.com/acgetchell/delaunay/pull/225)
  [`25a09a4`](https://github.com/acgetchell/delaunay/commit/25a09a46f1c0d46775fd085e6b34053888d69277)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.68.8 to 2.68.16.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/cfdb446e391c69574ebc316dfb7d7849ec12b940...d6e286fa45544157a02d45a43742857ebbc25d12)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.68.16
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump actions/upload-artifact from 6 to 7 [#223](https://github.com/acgetchell/delaunay/pull/223)
  [`0a25bfa`](https://github.com/acgetchell/delaunay/commit/0a25bfa55c5b9d4579e7747e1e7c8661185c4542)

  Bumps [actions/upload-artifact](https://github.com/actions/upload-artifact) from 6 to 7.

  - [Release notes](https://github.com/actions/upload-artifact/releases)
  - [Commits](https://github.com/actions/upload-artifact/compare/v6...v7)

---

  updated-dependencies:

- dependency-name: actions/upload-artifact
  dependency-version: '7'
  dependency-type: direct:production
  update-type: version-update:semver-major
  ...

- Bump astral-sh/setup-uv from 7.3.0 to 7.3.1 [#224](https://github.com/acgetchell/delaunay/pull/224)
  [`1533402`](https://github.com/acgetchell/delaunay/commit/15334025d07c4dfeef43d01bbf3a190eaed54688)

  Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.3.0 to 7.3.1.

  - [Release notes](https://github.com/astral-sh/setup-uv/releases)
  - [Commits](https://github.com/astral-sh/setup-uv/compare/eac588ad8def6316056a12d4907a9d4d84ff7a3b...5a095e7a2014a4212f075830d4f7277575a9d098)

---

  updated-dependencies:

- dependency-name: astral-sh/setup-uv
  dependency-version: 7.3.1
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump taiki-e/install-action from 2.68.16 to 2.68.25 [#237](https://github.com/acgetchell/delaunay/pull/237)
  [`abe7925`](https://github.com/acgetchell/delaunay/commit/abe79250b63aaf695a8a4447250402b6f1cbdcc8)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.68.16 to 2.68.25.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/d6e286fa45544157a02d45a43742857ebbc25d12...a37010ded18ff788be4440302bd6830b1ae50d8b)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.68.25
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump actions/setup-node from 6.2.0 to 6.3.0 [#239](https://github.com/acgetchell/delaunay/pull/239)
  [`eb0000b`](https://github.com/acgetchell/delaunay/commit/eb0000bf27cb583a97370190861c01d4b8902bfe)

  Bumps [actions/setup-node](https://github.com/actions/setup-node) from 6.2.0 to 6.3.0.

  - [Release notes](https://github.com/actions/setup-node/releases)
  - [Commits](https://github.com/actions/setup-node/compare/6044e13b5dc448c55e2357c09f80417699197238...53b83947a5a98c8d113130e565377fae1a50d02f)

---

  updated-dependencies:

- dependency-name: actions/setup-node
  dependency-version: 6.3.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

### Performance

- Improve Hilbert curve correctness and add bulk API [#207](https://github.com/acgetchell/delaunay/pull/207)
  [#216](https://github.com/acgetchell/delaunay/pull/216) [`2d198e7`](https://github.com/acgetchell/delaunay/commit/2d198e7d2f1f41f1b2e47009a1cf7cc12079fe05)

  Implements correctness fixes, API improvements, and comprehensive testing
  for the Hilbert space-filling curve ordering utilities.

#### Correctness Fixes

- Add debug_assert guards in hilbert_index_from_quantized for parameter
  validation (bits range and overflow checks)

- Fix quantization truncation bias by changing from NumCast::from(scaled)
  to scaled.round().to_u32() for fairer spatial distribution across grid
  cells, improving point ordering quality

#### API Design

- Add HilbertError enum with InvalidBitsParameter, IndexOverflow, and
  DimensionTooLarge variants for proper error handling

- Implement hilbert_indices_prequantized() returning Result&lt;Vec&lt;u128&gt;,
  HilbertError&gt; for bulk processing of pre-quantized coordinates

- Bulk API avoids redundant quantization computation, significantly
  improving performance for large insertion batches

#### Testing

- Add 4D continuity test verifying Hilbert curve property on 256-point
  grid (bits=2)

- Add quantization rounding distribution test validating fair cell
  assignment

- Add 5 comprehensive tests for prequantized API covering success cases,
  empty input, and all error conditions

- All 17 Hilbert-specific tests pass (11 existing + 6 new)

#### Known Issue

  Temporarily ignore repair_fallback_produces_valid_triangulation test as
  the rounding change affects insertion order, exposing a latent geometric
  degeneracy issue in triangulation construction. This is properly
  documented and tracked under issue #204 for investigation.

#### Added: Explicitly handle zero-dimensional inputs in Hilbert index calculation

  Ensures correct behavior for `hilbert_indices_prequantized` when
  the dimensionality `D` is zero. In such a space, all points map to
  the single origin, and their Hilbert curve index is always 0.
  This change adds an early return for this specific edge case.

- Use bulk Hilbert API in order_vertices_hilbert [#218](https://github.com/acgetchell/delaunay/pull/218)
  [`4782905`](https://github.com/acgetchell/delaunay/commit/478290556e88f770e0fcda07fb6137a3404b70f5)

  Refactored `order_vertices_hilbert` to use the bulk `hilbert_indices_prequantized` API
  instead of calling `hilbert_index` individually for each vertex. This eliminates
  redundant parameter validation (N validations → 1 validation for N vertices).

- Switch FastKernel to insphere_lifted and enable LTO [#234](https://github.com/acgetchell/delaunay/pull/234)
  [`91e290f`](https://github.com/acgetchell/delaunay/commit/91e290fca1a633f5b084accc367767f235780a49)

  Switch FastKernel::in_sphere() to use insphere_lifted() for 5.3x speedup in 3D.
  Add release profile optimization with thin LTO and codegen-units=1.

  Benchmarks across dimensions (2D-5D):

  - insphere_lifted is 5.3x faster in 3D (15.5 ns vs 81.7 ns)
  - Random query test: 3.75x faster (20.0 µs vs 75.0 µs for 1000 queries)
  - insphere_lifted consistently fastest across all dimensions

  Performance gains attributed to la-stack v0.2.0's closed-form determinants for D=1-4.

## [0.7.1] - 2026-02-20

### Merged Pull Requests

- Prevents timeout in 4D bulk construction [#203](https://github.com/acgetchell/delaunay/pull/203)
- Bump taiki-e/install-action from 2.67.26 to 2.67.30 [#202](https://github.com/acgetchell/delaunay/pull/202)
- Bump the dependencies group with 3 updates [#201](https://github.com/acgetchell/delaunay/pull/201)
- Feat/ball pointgen and debug harness [#200](https://github.com/acgetchell/delaunay/pull/200)
- Ci/perf baselines by tag [#199](https://github.com/acgetchell/delaunay/pull/199)
- Refactors changelog generation to use git-cliff [#198](https://github.com/acgetchell/delaunay/pull/198)
- Bump astral-sh/setup-uv from 7.2.1 to 7.3.0 [#196](https://github.com/acgetchell/delaunay/pull/196)
- Bump taiki-e/install-action from 2.67.18 to 2.67.26 [#195](https://github.com/acgetchell/delaunay/pull/195)
- Improves flip algorithm with topology index [#194](https://github.com/acgetchell/delaunay/pull/194)
- Removes `CoordinateScalar` bound from `Cell`, `Tds`, `Vertex` [#193](https://github.com/acgetchell/delaunay/pull/193)
- Moves `TopologyEdit` to `triangulation::flips` [#192](https://github.com/acgetchell/delaunay/pull/192)
- Correctly wires neighbors after K2 flips [#191](https://github.com/acgetchell/delaunay/pull/191)
- Use borrowed APIs in utility functions [#190](https://github.com/acgetchell/delaunay/pull/190)
- Add ScalarSummable/ScalarAccumulative supertraits [#189](https://github.com/acgetchell/delaunay/pull/189)
- Refactors point access for efficiency (internal) [#187](https://github.com/acgetchell/delaunay/pull/187)
- Corrects kernel parameter passing in triangulation [#186](https://github.com/acgetchell/delaunay/pull/186)
- Examples to error and struct definitions [#185](https://github.com/acgetchell/delaunay/pull/185)
- Validates random triangulations for Euler consistency [#184](https://github.com/acgetchell/delaunay/pull/184)
- Validates ridge links locally after Delaunay repair [#183](https://github.com/acgetchell/delaunay/pull/183)
- Bump astral-sh/setup-uv from 7.2.0 to 7.2.1 [#182](https://github.com/acgetchell/delaunay/pull/182)
- Bump taiki-e/install-action from 2.67.17 to 2.67.18 [#181](https://github.com/acgetchell/delaunay/pull/181)
- Stabilizes Delaunay property tests with bistellar flips [#180](https://github.com/acgetchell/delaunay/pull/180)
- Bump taiki-e/install-action from 2.66.1 to 2.67.11 [#177](https://github.com/acgetchell/delaunay/pull/177)
- Bump actions/checkout from 6.0.1 to 6.0.2 [#176](https://github.com/acgetchell/delaunay/pull/176)
- Bump actions/setup-node from 6.1.0 to 6.2.0 [#175](https://github.com/acgetchell/delaunay/pull/175)
- Feature/bistellar flips [#172](https://github.com/acgetchell/delaunay/pull/172)

### Added

- Examples to error and struct definitions [#185](https://github.com/acgetchell/delaunay/pull/185)
  [`a1bce55`](https://github.com/acgetchell/delaunay/commit/a1bce556cfd9a799f2a6aabb716443a14aaf6772)

  Adds code examples to various error enums and struct definitions
  to improve documentation and provide usage guidance.

  This change enhances the discoverability and understanding of
  various components, such as `AdjacencyIndexBuildError`,
  `BistellarFlipKind`, `FlipDirection`, `FlipError`, `FlipInfo`,
  `TriangleHandle`, `RidgeHandle`, `DelaunayRepairStats`,
  `RepairQueueOrder`, `DelaunayRepairDiagnostics`,
  `DelaunayRepairError`, `HullExtensionReason`, `InsertionError`,
  `LocateResult`, `LocateError`, `ConflictError`, `LocateFallbackReason`,
  `LocateFallback`, `LocateStats`, `CellValidationError`, `EdgeKey`,
  `FacetError`, `TopologicalOperation`, `RepairDecision`,
  `InsertionResult`, `InsertionStatistics`,
  `TriangulationConstructionState`, `TdsConstructionError`,
  `DelaunayValidationError`, `DelaunayRepairPolicy`,
  `DelaunayRepairHeuristicConfig`, `DelaunayRepairHeuristicSeeds`,
  `DelaunayRepairOutcome`, `DelaunayCheckPolicy`,
  `UuidValidationError`, `VertexValidationError`,
  `ConvexHullValidationError`, `ConvexHullConstructionError`,
  `MatrixError`, `InSphere`, `Orientation`, `QualityError`,
  `ConsistencyResult`, `CoordinateConversionError`,
  `CoordinateValidationError`, `CircumcenterError`,
  `SurfaceMeasureError`, `RandomPointGenerationError` and
  `ValueConversionError` to make the crate easier to use.

#### Changed: Improves examples and updates doc tests

  Updates doc tests to use clearer examples and more
  idiomatic syntax, enhancing code readability and
  maintainability. Modifies BistellarFlipKind to use a
  getter method. Addresses issues identified during
  documentation review. (internal)

- Add ScalarSummable/ScalarAccumulative supertraits [#189](https://github.com/acgetchell/delaunay/pull/189)
  [`abdeeb2`](https://github.com/acgetchell/delaunay/commit/abdeeb2f80ab03c998b9a29108c57ff9f0c54393)

  - Add ScalarSummable (CoordinateScalar + Sum) and ScalarAccumulative (CoordinateScalar + AddAssign + SubAssign + Sum)
  - Refactor repeated scalar bounds across geometry/core modules to use the new supertraits
  - Allow “supertrait(s)” in cspell
- Document geometric and topological invariants [`0283bf0`](https://github.com/acgetchell/delaunay/commit/0283bf01cd1860c1a44ff9f645ac304fa44b7345)

  Adds `invariants.md` to document the theoretical background and
  rationale for the topological and geometric invariants enforced by
  the `delaunay` crate. This includes simplicial complexes,
  PL-manifolds, link-based validation, insertion strategies, and
  convergence considerations. Updates `README.md` and `lib.rs` to
  reference the new document. Also adds a `examples/README.md` file.

### Changed

- Feature/bistellar flips [#172](https://github.com/acgetchell/delaunay/pull/172)
  [`66c7028`](https://github.com/acgetchell/delaunay/commit/66c7028d0c3d9dbc00f6b1a9cb791c41d39ab933)

- Refactors point access for efficiency (internal) [#187](https://github.com/acgetchell/delaunay/pull/187)
  [`8020065`](https://github.com/acgetchell/delaunay/commit/8020065afe66fc066ef51307a9de02621f087a54)

  Simplifies vertex coordinate access using `.coords()` instead of `.into()`,
  improving code clarity and potentially performance. This change is
  internal, affecting predicate calculations and geometric algorithms.
  Also, moves the issue 120 investigation document to the archive.

- Use borrowed APIs in utility functions [#190](https://github.com/acgetchell/delaunay/pull/190)
  [`bee065b`](https://github.com/acgetchell/delaunay/commit/bee065bd13f9f7adb8a7767b5b907ea7886248d5)

  Updates `into_hashmap`, `dedup_vertices_exact`,
  `dedup_vertices_epsilon`, and `filter_vertices_excluding`
  functions to accept slices instead of vectors, improving
  performance by avoiding unnecessary cloning.

  This aligns with the Rust agent's preference for borrowed
  APIs, taking references as arguments and returning borrowed
  views when possible, and only taking ownership when required.

- Moves `TopologyEdit` to `triangulation::flips` [#192](https://github.com/acgetchell/delaunay/pull/192)
  [`c491bb9`](https://github.com/acgetchell/delaunay/commit/c491bb913e1b49ff38d4ce52c180d5220e9db9df)

  Moves the `TopologyEdit` trait to `triangulation::flips` and renames it to `BistellarFlips`.

  This change involves updating imports and references throughout the codebase and documentation to reflect the new location and name of the trait.

#### Changed: Refactors prelude modules for clarity (internal)

  Streamlines the prelude modules to provide clearer and more
  focused exports for common triangulation tasks. This change
  affects import statements in documentation and examples,
  requiring more specific paths for certain types.

#### Removed: Topology validation prelude module

  Removes the redundant topology validation prelude module.

  Moves its contents into the main topology prelude, simplifying
  module structure and reducing code duplication. This change
  internally refactors the prelude modules for better organization.

- Removes `CoordinateScalar` bound from `Cell` , `Tds` , `Vertex` [#193](https://github.com/acgetchell/delaunay/pull/193)
  [`e69f3d1`](https://github.com/acgetchell/delaunay/commit/e69f3d153961050e03a520e5b5457a165097c834)

  Relaxes trait bounds on `Cell`, `Tds`, and `Vertex` structs by
  removing the `CoordinateScalar` requirement.

  This change prepares the triangulation data structure for combinatorial
  operations independent of geometry. The `validate` method in `Tds`
  now requires `CoordinateScalar` to perform coordinate validation,
  where applicable. (Internal change).

#### Changed: Clarifies `Vertex` constraints and moves `point`

  Clarifies the `Vertex` struct's constraints, emphasizing
  `CoordinateScalar` requirement for geometric operations and
  serialization but allowing combinatorial use without it.

  Moves the `point` method definition to ensure consistent API
  presentation. (Internal refactoring, no functional change).

- Improves flip algorithm with topology index [#194](https://github.com/acgetchell/delaunay/pull/194)
  [`c4e37ed`](https://github.com/acgetchell/delaunay/commit/c4e37ed9e899170978196af4edad4e2c0a248141)

  Improves flip algorithm by introducing a topology index to
  efficiently check for duplicate cells and non-manifold facets.
  This avoids redundant scans of the triangulation data
  structure, especially during repair operations, by pre-computing
  and storing facet and cell signatures. This change is internal.

- Updates typos-cli installation in CI workflow [`6168e83`](https://github.com/acgetchell/delaunay/commit/6168e83512509123f9c0443f9716752f88cc2aa3)

  Updates the typos-cli installation in the CI workflow to use the
  `taiki-e/install-action` for simpler and more reliable installation.
  This aligns with the switch from cspell to typos in the codebase.

- Refactors changelog generation to use git-cliff [#198](https://github.com/acgetchell/delaunay/pull/198)
  [`63553fa`](https://github.com/acgetchell/delaunay/commit/63553fa170a5603c98c3c5eee87caca756e0bb89)

- Ci/perf baselines by tag [#199](https://github.com/acgetchell/delaunay/pull/199)
  [`0c94dec`](https://github.com/acgetchell/delaunay/commit/0c94dec96ac73004b9b9c924a977499e29dfaf19)

- Feat/ball pointgen and debug harness [#200](https://github.com/acgetchell/delaunay/pull/200)
  [`79bf0e9`](https://github.com/acgetchell/delaunay/commit/79bf0e96b0d9c4f8cbdda8539910adfa18f412a4)

### Documentation

- Refresh docs and add workflows guide [`695e9a0`](https://github.com/acgetchell/delaunay/commit/695e9a0f700b8cb6bf7cb5a3acb6e4510c4b3939)

  - Add a new workflows guide covering Builder/Edit API usage, topology guarantees, and repair
  - Wire README into doctests and refresh crate-level docs/examples (simplify type annotations, link to workflows)
  - Update README Features + references (kernels, insertion ordering, construction options, validation/repair)
  - Reorganize docs index and archive historical/roadmap material; refresh topology/robustness/validation guides
  - Replace debug `eprintln!` with `tracing` in Hilbert utilities/tests
  - Tweak spell-check + release docs (typos-cli, rename handling, release steps) and update CHANGELOG

### Fixed

- Stabilizes Delaunay property tests with bistellar flips [#180](https://github.com/acgetchell/delaunay/pull/180)
  [`e3bd4bf`](https://github.com/acgetchell/delaunay/commit/e3bd4bfa77258484e6dab088a2980139efd0f182)

  Enables previously failing Delaunay property tests by
  implementing bistellar flips for robust Delaunay repair.
  Includes automatic repair and fast validation.

  Updates MSRV to 1.93.0.

- Validates ridge links locally after Delaunay repair [#183](https://github.com/acgetchell/delaunay/pull/183)
  [`7bc4792`](https://github.com/acgetchell/delaunay/commit/7bc4792b33ee1e6ecbeda729834a33fbf06cd0e6)

  Addresses potential topology violations (non-manifold configurations)
  introduced by flip-based Delaunay repair by validating ridge links
  for affected cells post-insertion. This prevents committing
  invalid triangulations and surfaces topology validation failures
  as insertion errors, enabling transactional rollback.

- Validates random triangulations for Euler consistency [#184](https://github.com/acgetchell/delaunay/pull/184)
  [`72eb272`](https://github.com/acgetchell/delaunay/commit/72eb2726f8c683339a23b17a63991ac87a35e412)

  Ensures that random triangulations satisfy Euler characteristic
  validation to prevent construction errors or invalid classifications.

  Adds a validation function to check topology/Euler validity after
  triangulation construction or robust fallback attempts, catching
  potential issues that can lead to incorrect results. Removes
  redundant validation checks.

- Corrects kernel parameter passing in triangulation [#186](https://github.com/acgetchell/delaunay/pull/186)
  [`df3c490`](https://github.com/acgetchell/delaunay/commit/df3c49033dd96558ee5ee0de572913e82c55f210)

  Addresses an issue where the kernel was being passed by value
  instead of by reference in the Delaunay triangulation
  construction. This change ensures that the kernel is correctly
  accessed and used, preventing potential errors and improving
  reliability. The fix involves modifying the `with_kernel` method
  signatures and call sites to accept a kernel reference instead of
  a kernel value. This affects benchmark code, documentation,
  examples, and internal code.

- Correctly wires neighbors after K2 flips [#191](https://github.com/acgetchell/delaunay/pull/191)
  [`5ab686c`](https://github.com/acgetchell/delaunay/commit/5ab686c2177562bdeb852fe843c946181e03753a)

  Fixes an issue where external neighbors across the cavity
  boundary were not being correctly rewired after a K2 flip.

  Introduces `external_facets_for_boundary` to collect the set
  of external facets that are shared with the flip cavity boundary,
  and then uses these to correctly wire up neighbors.

  Adds a test case to verify that external neighbors are correctly
  rewired after the flip, ensuring that the triangulation remains
  valid and consistent.
  Refs: refactor/wire-cavity-neighbors

#### Added: K=3 flip rewiring test

  Adds a test to verify correct rewiring of external neighbors
  after a k=3 flip. This validates the boundary handling and
  neighbor update logic in the bistellar flip implementation.

  This test constructs an explicit k=3 ridge-flip fixture and
  checks neighbor rewiring.

- Prevents timeout in 4D bulk construction [#203](https://github.com/acgetchell/delaunay/pull/203)
  [`b071fb7`](https://github.com/acgetchell/delaunay/commit/b071fb787bf06ff196fade35fd5dab180822985b)

  Addresses a timeout issue in 4D bulk construction
  by implementing per-insertion local Delaunay repair
  (soft-fail) during bulk construction to prevent
  violation accumulation, which slows down the global
  repair process. Also adds a hard wall-clock time
  limit to the test harness.

### Maintenance

- Bump actions/setup-node from 6.1.0 to 6.2.0 [#175](https://github.com/acgetchell/delaunay/pull/175)
  [`8d15114`](https://github.com/acgetchell/delaunay/commit/8d15114553e05ecb9d46cfb9bd78eb9e27379796)

  Bumps [actions/setup-node](https://github.com/actions/setup-node) from 6.1.0 to 6.2.0.

  - [Release notes](https://github.com/actions/setup-node/releases)
  - [Commits](https://github.com/actions/setup-node/compare/395ad3262231945c25e8478fd5baf05154b1d79f...6044e13b5dc448c55e2357c09f80417699197238)

---

  updated-dependencies:

- dependency-name: actions/setup-node
  dependency-version: 6.2.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump actions/checkout from 6.0.1 to 6.0.2 [#176](https://github.com/acgetchell/delaunay/pull/176)
  [`c703f94`](https://github.com/acgetchell/delaunay/commit/c703f944a9928651b639d5e5b2a06db3b1b75e4f)

  Bumps [actions/checkout](https://github.com/actions/checkout) from 6.0.1 to 6.0.2.

  - [Release notes](https://github.com/actions/checkout/releases)
  - [Commits](https://github.com/actions/checkout/compare/v6.0.1...v6.0.2)

---

  updated-dependencies:

- dependency-name: actions/checkout
  dependency-version: 6.0.2
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump taiki-e/install-action from 2.66.1 to 2.67.11 [#177](https://github.com/acgetchell/delaunay/pull/177)
  [`92292e0`](https://github.com/acgetchell/delaunay/commit/92292e01f5777fe4f13ab387a1f9ee4e39930ab3)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.66.1 to 2.67.11.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/3522286d40783523f9c7880e33f785905b4c20d0...887bc4e03483810873d617344dd5189cd82e7b8b)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.67.11
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump astral-sh/setup-uv from 7.2.0 to 7.2.1 [#182](https://github.com/acgetchell/delaunay/pull/182)
  [`cfaaf60`](https://github.com/acgetchell/delaunay/commit/cfaaf600fef814bfe563524fcdbcb0ab83fa9028)

  Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.2.0 to 7.2.1.

  - [Release notes](https://github.com/astral-sh/setup-uv/releases)
  - [Commits](https://github.com/astral-sh/setup-uv/compare/61cb8a9741eeb8a550a1b8544337180c0fc8476b...803947b9bd8e9f986429fa0c5a41c367cd732b41)

---

  updated-dependencies:

- dependency-name: astral-sh/setup-uv
  dependency-version: 7.2.1
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump taiki-e/install-action from 2.67.17 to 2.67.18 [#181](https://github.com/acgetchell/delaunay/pull/181)
  [`068f314`](https://github.com/acgetchell/delaunay/commit/068f314ed00a837678c968734f75b4b69327d9e0)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.67.17 to 2.67.18.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/29feb09ac22f4fde4175fe7b5c3548952234f69a...650c5ca14212efbbf3e580844b04bdccf68dac31)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.67.18
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...
- Bump rand to 0.10 and trim unused deps [`fe3e406`](https://github.com/acgetchell/delaunay/commit/fe3e406f19dcc6c9c72666d89b285626ea00465c)

  - Update rand to v0.10 and fix RNG trait imports (use rand::RngExt) in tests/utilities
  - Move test-only crates to dev-dependencies (approx, serde_json)
  - Remove unused runtime dependencies (anyhow, clap, serde_test)
  - Drop clippy allow for multiple_crate_versions
  - Update Cargo.lock and regenerate CHANGELOG.md
- Bump astral-sh/setup-uv from 7.2.1 to 7.3.0 [#196](https://github.com/acgetchell/delaunay/pull/196)
  [`6c4662a`](https://github.com/acgetchell/delaunay/commit/6c4662a4008e5a43de71d3b83591350dbf4778b0)

  Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.2.1 to 7.3.0.

  - [Release notes](https://github.com/astral-sh/setup-uv/releases)
  - [Commits](https://github.com/astral-sh/setup-uv/compare/803947b9bd8e9f986429fa0c5a41c367cd732b41...eac588ad8def6316056a12d4907a9d4d84ff7a3b)

---

  updated-dependencies:

- dependency-name: astral-sh/setup-uv
  dependency-version: 7.3.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump taiki-e/install-action from 2.67.18 to 2.67.26 [#195](https://github.com/acgetchell/delaunay/pull/195)
  [`1cd6008`](https://github.com/acgetchell/delaunay/commit/1cd6008503d0bc2694e934060f8ed986f2d9e05e)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.67.18 to 2.67.26.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/650c5ca14212efbbf3e580844b04bdccf68dac31...509565405a8a987e73cf742e26b26dcc72c4b01a)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.67.26
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump taiki-e/install-action from 2.67.26 to 2.67.30 [#202](https://github.com/acgetchell/delaunay/pull/202)
  [`e99f0d3`](https://github.com/acgetchell/delaunay/commit/e99f0d3f17cf45e85c51274799b8baee3b28e8e5)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.67.26 to 2.67.30.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/509565405a8a987e73cf742e26b26dcc72c4b01a...288875dd3d64326724fa6d9593062d9f8ba0b131)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.67.30
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump the dependencies group with 3 updates [#201](https://github.com/acgetchell/delaunay/pull/201)
  [`fa69c80`](https://github.com/acgetchell/delaunay/commit/fa69c800e6b01fa2cb4f9b1468473e09ed9f0108)

  Bumps the dependencies group with 3 updates: [arc-swap](https://github.com/vorner/arc-swap) , [uuid](https://github.com/uuid-rs/uuid) and
  [sysinfo](https://github.com/GuillaumeGomez/sysinfo) .

  Updates `arc-swap` from 1.8.1 to 1.8.2

  - [Changelog](https://github.com/vorner/arc-swap/blob/master/CHANGELOG.md)
  - [Commits](https://github.com/vorner/arc-swap/compare/v1.8.1...v1.8.2)

  Updates `uuid` from 1.20.0 to 1.21.0

  - [Release notes](https://github.com/uuid-rs/uuid/releases)
  - [Commits](https://github.com/uuid-rs/uuid/compare/v1.20.0...v1.21.0)

  Updates `sysinfo` from 0.38.1 to 0.38.2

  - [Changelog](https://github.com/GuillaumeGomez/sysinfo/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/GuillaumeGomez/sysinfo/compare/v0.38.1...v0.38.2)

---

  updated-dependencies:

- dependency-name: arc-swap
  dependency-version: 1.8.2
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies

- dependency-name: uuid
  dependency-version: 1.21.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  dependency-group: dependencies

- dependency-name: sysinfo
  dependency-version: 0.38.2
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies
  ...

### Removed

- Replace cspell with typos for spell checking [`4b5e1a1`](https://github.com/acgetchell/delaunay/commit/4b5e1a1641037f137663612c74bb39cd187f90e9)

  Replaces the cspell tool with typos for spell checking across the
  project. This change involves removing cspell configurations and
  dependencies, and integrating typos, including its configuration file.

## [0.7.0] - 2026-01-13

### ⚠️ Breaking Changes

- Add public topology traversal API [#164](https://github.com/acgetchell/delaunay/pull/164)

### Merged Pull Requests

- Refactors topology guarantee and manifold validation [#171](https://github.com/acgetchell/delaunay/pull/171)
- Bump astral-sh/setup-uv from 7.1.6 to 7.2.0 [#170](https://github.com/acgetchell/delaunay/pull/170)
- Bump taiki-e/install-action from 2.65.13 to 2.66.1 [#169](https://github.com/acgetchell/delaunay/pull/169)
- Feature/manifolds [#168](https://github.com/acgetchell/delaunay/pull/168)
- Refactors Gram determinant calculation with LDLT [#167](https://github.com/acgetchell/delaunay/pull/167)
- Bump clap from 4.5.53 to 4.5.54 in the dependencies group [#166](https://github.com/acgetchell/delaunay/pull/166)
- Bump taiki-e/install-action from 2.65.7 to 2.65.13 [#165](https://github.com/acgetchell/delaunay/pull/165)
- Add public topology traversal API [#164](https://github.com/acgetchell/delaunay/pull/164)

### Added

- [**breaking**] Add public topology traversal API [#164](https://github.com/acgetchell/delaunay/pull/164)
  [`3748ebb`](https://github.com/acgetchell/delaunay/commit/3748ebb24ded08154875b2be371128c77d43eed3)

  - Introduce canonical `EdgeKey` and read-only topology traversal helpers on `Triangulation`
  - Add opt-in `AdjacencyIndex` builder for faster repeated adjacency queries
  - Add integration tests for topology traversal and adjacency index invariants
  - Refresh repo tooling/CI configs and supporting scripts/tests

#### Changed: Exposes public topology traversal API

  Makes topology traversal APIs public for external use.

  Exposes `edges()`, `incident_edges()`, and `cell_neighbors()` on the
  `DelaunayTriangulation` struct as convenience wrappers. Updates
  documentation, examples, and benchmarks to use new API.

  This allows external users to traverse the triangulation's topology
  without needing to access internal implementation details.

#### Changed: Expose topology query APIs on DelaunayTriangulation

  Exposes cell and vertex query APIs on `DelaunayTriangulation` for zero-allocation topology traversal.

  Also includes internal refactoring to improve vertex incidence
  validation and ensure more robust handling of invalid key references.
  Now TDS validation detects isolated vertices.

### Changed

- Updates CHANGELOG.md for unreleased changes [`271353f`](https://github.com/acgetchell/delaunay/commit/271353f1b1af82fb45eb619520d62cd7474e4541)

  Updates the changelog to reflect recent changes, including adding
  a new public topology traversal API, refreshing repository
  tooling/CI configurations, and clarifying TDS validation and API
  documentation.

- Refactors Gram determinant calculation with LDLT [#167](https://github.com/acgetchell/delaunay/pull/167)
  [`561a259`](https://github.com/acgetchell/delaunay/commit/561a259d58401d2baa61cd6313dd0ada01179f4a)

  Refactors the Gram determinant calculation to use LDLT factorization from the `la-stack` crate for improved efficiency and numerical stability by exploiting
  symmetry.

  Also, updates the `la-stack` dependency version.

#### Fixed: Improves robustness of incremental insertion

  Addresses rare topological invalidations during incremental
  insertion by:

- Adding connectedness validation to conflict region checks.

- Adding codimension-2 boundary manifoldness validation
  ("no boundary of boundary") to triangulation's `is_valid`
  method.

- Ensuring that strict Level 3 validation is enabled during
  batch construction in debug builds.
  Refs: feat/la-stack-ldlt-factorization

#### Changed: Rename SimplexCounts to FVector for clarity

  Renames the `SimplexCounts` struct to `FVector` to better reflect
  its mathematical meaning as the f-vector in topology, representing
  the counts of simplices of different dimensions.

  This change improves code readability and aligns the naming
  convention with standard topological terminology.
  (Internal refactoring, no API change.)

#### Changed: Improves simplex generation algorithm

  Improves the algorithm for generating simplex combinations in
  the Euler characteristic calculation. This change enhances
  efficiency by using a lexicographic approach to generate
  combinations, reducing unnecessary computations.

- Feature/manifolds [#168](https://github.com/acgetchell/delaunay/pull/168)
  [`10abbe1`](https://github.com/acgetchell/delaunay/commit/10abbe1899a381d1b0d4855727dfef7797952549)

- Refactors topology guarantee and manifold validation [#171](https://github.com/acgetchell/delaunay/pull/171)
  [`dfdba5a`](https://github.com/acgetchell/delaunay/commit/dfdba5a745d6a41b4fa92d66b71c0c3d3dc87e54)

  Refactors manifold validation mode to topology guarantee for
  clarity. Updates Level 3 validation configuration, improves error
  reporting, and adds comprehensive manifold validation tests.
  Also improves robustness of incremental insertion.

#### Changed: Updates topology guarantee defaults and validation

  Updates the default topology guarantee to `Pseudomanifold` for new
  triangulations and deserialized triangulations. Also, clarifies
  validation policy and its relationship to topology guarantees in
  documentation. Introduces a test-only function to repair degenerate
  cells by removing them and clearing dangling references.

#### Fixed: Corrects triangulation perturbation logic

  Fixes a bug in the vertex insertion perturbation logic that
  caused non-equivalent results when translating the input
  point set by using a translation-invariant anchor for
  perturbation scaling.

  Also, preserves the caller-provided vertex UUID across
  perturbation retries to maintain vertex identity.

  Updates documentation on topology guarantees to clarify
  manifoldness invariants.

#### Changed: Improves PL-manifold validation with vertex-link check

  Replaces ridge-link validation with vertex-link validation for
  PL-manifold topology guarantee. This change provides a more
  robust and canonical check for PL-manifoldness, ensuring that
  the link of every vertex is a sphere or ball of the appropriate
  dimension.

### Maintenance

- Bump clap from 4.5.53 to 4.5.54 in the dependencies group [#166](https://github.com/acgetchell/delaunay/pull/166)
  [`ebd0d32`](https://github.com/acgetchell/delaunay/commit/ebd0d32e3552b2e94af021995df8c9c1431ccc1b)

  Bumps the dependencies group with 1 update: [clap](https://github.com/clap-rs/clap).

  Updates `clap` from 4.5.53 to 4.5.54

  - [Release notes](https://github.com/clap-rs/clap/releases)
  - [Changelog](https://github.com/clap-rs/clap/blob/master/CHANGELOG.md)
  - [Commits](https://github.com/clap-rs/clap/compare/clap_complete-v4.5.53...clap_complete-v4.5.54)

---

  updated-dependencies:

- dependency-name: clap
  dependency-version: 4.5.54
  dependency-type: direct:production
  update-type: version-update:semver-patch
  dependency-group: dependencies
  ...

- Bump taiki-e/install-action from 2.65.7 to 2.65.13 [#165](https://github.com/acgetchell/delaunay/pull/165)
  [`6b5f723`](https://github.com/acgetchell/delaunay/commit/6b5f723cf9f71599bedef098b3f226f4786ce539)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.65.7 to 2.65.13.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/4c6723ec9c638cccae824b8957c5085b695c8085...0e76c5c569f13f7eb21e8e5b26fe710062b57b62)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.65.13
  dependency-type: direct:production
  update-type: version-update:semver-patch
  ...

- Bump astral-sh/setup-uv from 7.1.6 to 7.2.0 [#170](https://github.com/acgetchell/delaunay/pull/170)
  [`b50bb8c`](https://github.com/acgetchell/delaunay/commit/b50bb8c4157496c99e9879b5c7214815ddbca633)

  Bumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 7.1.6 to 7.2.0.

  - [Release notes](https://github.com/astral-sh/setup-uv/releases)
  - [Commits](https://github.com/astral-sh/setup-uv/compare/681c641aba71e4a1c380be3ab5e12ad51f415867...61cb8a9741eeb8a550a1b8544337180c0fc8476b)

---

  updated-dependencies:

- dependency-name: astral-sh/setup-uv
  dependency-version: 7.2.0
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

- Bump taiki-e/install-action from 2.65.13 to 2.66.1 [#169](https://github.com/acgetchell/delaunay/pull/169)
  [`44d3add`](https://github.com/acgetchell/delaunay/commit/44d3add45bdac8daff9cff6b686929959ccf84a3)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.65.13 to 2.66.1.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/0e76c5c569f13f7eb21e8e5b26fe710062b57b62...3522286d40783523f9c7880e33f785905b4c20d0)

---

  updated-dependencies:

- dependency-name: taiki-e/install-action
  dependency-version: 2.66.1
  dependency-type: direct:production
  update-type: version-update:semver-minor
  ...

## Archives

Older releases are archived by minor series:

- [0.6.x](docs/archive/changelog/0.6.md)
- [0.5.x](docs/archive/changelog/0.5.md)
- [0.4.x](docs/archive/changelog/0.4.md)
- [0.3.x](docs/archive/changelog/0.3.md)
- [0.2.x](docs/archive/changelog/0.2.md)

[Unreleased]: https://github.com/acgetchell/delaunay/compare/v0.7.8...HEAD
[0.7.8]: https://github.com/acgetchell/delaunay/compare/v0.7.7...v0.7.8
[0.7.7]: https://github.com/acgetchell/delaunay/compare/v0.7.6...v0.7.7
[0.7.6]: https://github.com/acgetchell/delaunay/compare/v0.7.5...v0.7.6
[0.7.5]: https://github.com/acgetchell/delaunay/compare/v0.7.4...v0.7.5
[0.7.4]: https://github.com/acgetchell/delaunay/compare/v0.7.3...v0.7.4
[0.7.3]: https://github.com/acgetchell/delaunay/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/acgetchell/delaunay/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/acgetchell/delaunay/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/acgetchell/delaunay/compare/v0.6.2...v0.7.0

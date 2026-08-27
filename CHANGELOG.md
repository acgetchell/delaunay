# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1] - 2026-08-27

### ⚠️ Breaking Changes

- Automate transactional release preparation [#584](https://github.com/acgetchell/delaunay/pull/584)
- Add verified scientific checkpoint manifests [#587](https://github.com/acgetchell/delaunay/pull/587)
- Name public error variant fields [#567](https://github.com/acgetchell/delaunay/pull/567)
- Separate Level 4 and Delaunay owners [#557](https://github.com/acgetchell/delaunay/pull/557) [#569](https://github.com/acgetchell/delaunay/pull/569)
- Unify proof-bearing builder transitions [#574](https://github.com/acgetchell/delaunay/pull/574)

### Merged Pull Requests

- Add verified scientific checkpoint manifests [#587](https://github.com/acgetchell/delaunay/pull/587)
- Bump the github-actions group with 4 updates [#586](https://github.com/acgetchell/delaunay/pull/586)
- Automate transactional release preparation [#584](https://github.com/acgetchell/delaunay/pull/584)
- Harden release and performance preflight [#582](https://github.com/acgetchell/delaunay/pull/582)
- Feat/550 performance artifacts [#581](https://github.com/acgetchell/delaunay/pull/581)
- Unify proof-bearing builder transitions [#574](https://github.com/acgetchell/delaunay/pull/574)
- Align module ownership with proof layers [#572](https://github.com/acgetchell/delaunay/pull/572)
- Automate repository dependency updates [#571](https://github.com/acgetchell/delaunay/pull/571)
- Separate Level 4 and Delaunay owners [#557](https://github.com/acgetchell/delaunay/pull/557) [#569](https://github.com/acgetchell/delaunay/pull/569)
- Name public error variant fields [#567](https://github.com/acgetchell/delaunay/pull/567)
- Split bistellar flips into focused modules [#565](https://github.com/acgetchell/delaunay/pull/565)
- Align k=1 preflight with periodic realization [#564](https://github.com/acgetchell/delaunay/pull/564)
- Enforce Proptest case-count precedence [#563](https://github.com/acgetchell/delaunay/pull/563)
- Rebalance default and slow test coverage [#562](https://github.com/acgetchell/delaunay/pull/562)
- Restore topology construction repair policy [#561](https://github.com/acgetchell/delaunay/pull/561)
- Accelerate full Delaunay reports [#560](https://github.com/acgetchell/delaunay/pull/560)
- Certify Level 4 realization fast paths [#559](https://github.com/acgetchell/delaunay/pull/559)
- Refresh pins and extend the auto-merge wait [#556](https://github.com/acgetchell/delaunay/pull/556)
- Bump the dependencies group with 2 updates [#555](https://github.com/acgetchell/delaunay/pull/555)
- Reject malformed notebook schema values [#549](https://github.com/acgetchell/delaunay/pull/549)
- Harden version resolution and refresh tool pins [#547](https://github.com/acgetchell/delaunay/pull/547)
- Update setuptools requirement in the dependencies group [#545](https://github.com/acgetchell/delaunay/pull/545)
- Refresh dependencies and update automation [#544](https://github.com/acgetchell/delaunay/pull/544)
- Harden dependency and release automation [#543](https://github.com/acgetchell/delaunay/pull/543)

### Added

- [**breaking**] Automate transactional release preparation [#584](https://github.com/acgetchell/delaunay/pull/584)
  [`484f263`](https://github.com/acgetchell/delaunay/commit/484f2635e5cacb5cdf1aff4f812cdbdcc1ee7966)

  - Synchronize package, lockfile, citation, changelog, and documentation versions through rollback-safe release commands.
  - Update exact Python development pins before lock regeneration and verify release identity against published stable history.
  - Promote validated performance evidence into curated reports and an atomic README snapshot with durable provenance.
  - Fail closed on stale dates, mismatched metadata, unsafe tag input, and partial artifact publication.
- [**breaking**] Add verified scientific checkpoint manifests [#587](https://github.com/acgetchell/delaunay/pull/587)
  [`3ef8e4d`](https://github.com/acgetchell/delaunay/commit/3ef8e4d22439829fea4e6ea028501cc05a4b0387)

  - Add schema-v2 owner checkpoints with versioned SHA-256 digests, complete f-vectors, and derived Euler characteristics.
  - Canonicalize exact coordinate bits, UUID topology, proof context, periodic offsets, and supported user payloads independently of codec ordering.
  - Reject malformed or tampered evidence with typed errors, then rebuild storage and re-establish validation Levels 1–5.
  - Expose focused APIs for manifest inspection, verification, custom-kernel restoration, and explicit schema-v1 migration.

### Changed

- Rebalance default and slow test coverage [#562](https://github.com/acgetchell/delaunay/pull/562)
  [`b318142`](https://github.com/acgetchell/delaunay/commit/b3181424b3d909bc1dbd5c0a077e112737e5ce0d)

  - Promote bounded 3D and 4D invariant coverage into the default suite.
  - Limit debug timeout relief to two platform-sensitive periodic builder tests.
  - Document catalog-based ownership for remaining slow-test cases.
- Split bistellar flips into focused modules [#565](https://github.com/acgetchell/delaunay/pull/565)
  [`e2fc731`](https://github.com/acgetchell/delaunay/commit/e2fc73189a5e4ace4055490843f44c3ea997e52e)

  - Separate move types, contexts, mutation and orientation engines, typed errors, repair queues, and shared support.
  - Keep unit tests with their owning modules and preserve existing public re-export paths.
  - Update documentation and Semgrep coverage for the directory-backed layout.
- [**breaking**] Name public error variant fields [#567](https://github.com/acgetchell/delaunay/pull/567)
  [`eb80aad`](https://github.com/acgetchell/delaunay/commit/eb80aad2d53d91da7f6162c54ad0669f7172743d)

  - Replace positional error payloads with named source fields and typed convex-hull insufficiency reasons.
  - Enforce named public error fields and positive Semgrep rule coverage.
  - Extend release benchmark timeouts and reconcile v0.8.0 artifact metadata.
- [**breaking**] Separate Level 4 and Delaunay owners [#557](https://github.com/acgetchell/delaunay/pull/557)
  [#569](https://github.com/acgetchell/delaunay/pull/569) [`40ae56d`](https://github.com/acgetchell/delaunay/commit/40ae56dfbb6ff1e4b1d8b265901f6915387c1fea)

  - Promote Triangulation to the proof-bearing Levels 1–4 domain owner, with checked TDS restoration and dedicated builder terminals.
  - Reserve DelaunayTriangulation for cumulative Levels 1–5 values, using strict certification or consuming delaunayize conversion from Triangulation.
  - Move bistellar and Pachner mutation workflows onto Triangulation and require explicit demotion before editing a Delaunay value.
  - Split consuming PL-manifold TDS repair from proof-bearing reconstruction and preserve typed boundary failures throughout the composed workflows.
- Align module ownership with proof layers [#572](https://github.com/acgetchell/delaunay/pull/572)
  [`8b10278`](https://github.com/acgetchell/delaunay/commit/8b10278ce7edbaae762a9bf035d89dfa6341c588)

  - Keep Levels 1–2 TDS ownership and validation under core.
  - Move Levels 3–4 triangulation workflows into a dedicated module tree.
  - Isolate the Level 5 Delaunay model and enforce one-way proof-layer dependencies.
  - Codify owner-local unit-test placement and retire duplicate integration coverage.

#### Fixed: Restore validated EdgeIndex construction

- Restrict raw edge-index state to the core layer.
- Route triangulation queries through controlled construction and read-only accessors.
- Align Level 2 evidence and custom-kernel coverage with the intended APIs.

#### Fixed: Make benchmark and notebook evidence trustworthy

- Fail benchmark preflights on predicate errors, disagreements, or invalid topology.
- Keep certification outside timed loops and restrict topology comparisons to valid PL-manifold policies.
- Extract and test notebook rendering with independent validation of visual witnesses.
- Lock workflow tool execution and track all benchmark and paper-producing inputs.
- Preserve core ownership of adjacency internals through validated constructors and read-only accessors.
- Archive the completed production review remediation checklist.

#### Fixed: Harden benchmark and notebook evidence [#568](https://github.com/acgetchell/delaunay/pull/568)

- Publish benchmark results and metadata as a rollback-safe pair so failures preserve the previous baseline.
- Validate notebook witnesses before rendering and report invalid simplex references with contextual diagnostics.
- Parse workflow trigger paths structurally and align their Python dependencies.
- Correct validation documentation and colocate test-only Rust support with its owning module tests.
- [**breaking**] Unify proof-bearing builder transitions [#574](https://github.com/acgetchell/delaunay/pull/574)
  [`d0d5abd`](https://github.com/acgetchell/delaunay/commit/d0d5abdd9a67411a677cdd3c3f8abd5bc83ffdff)

  - Add public drafts for unpublished TDS and Delaunay construction.
  - Publish each invariant-bearing owner through a layer-specific, failure-atomic builder.
  - Separate snapshot-free strict certification from transactional canonicalization and flip repair.
  - Remove redundant conversion paths and encapsulate lower-layer storage.
- Feat/550 performance artifacts [#581](https://github.com/acgetchell/delaunay/pull/581)
  [`c55db1f`](https://github.com/acgetchell/delaunay/commit/c55db1f2a197bedbab33ba0c37f4b1b9f5811968)

### Fixed

- Harden dependency and release automation [#543](https://github.com/acgetchell/delaunay/pull/543)
  [`39d0993`](https://github.com/acgetchell/delaunay/commit/39d09932e1150ae6766cc39ae3406ab4916cb97b)

  - Authenticate CodeRabbit review requests as acgetchell and deduplicate them per PR head.
  - Cancel superseded runs and queue squash auto-merge only for the triggering commit.
  - Include the version title in the suggested GitHub release command.
- Harden version resolution and refresh tool pins [#547](https://github.com/acgetchell/delaunay/pull/547)
  [`a6786b7`](https://github.com/acgetchell/delaunay/commit/a6786b76b95567ab2704aa12ecc64b2a198d4a5d)

  - Fail workflow setup when justfile pins cannot be resolved or are empty.
  - Stagger weekly Dependabot updates for Actions, Cargo, and uv dependencies.
  - Update rumdl, uv, zizmor, and the pinned zizmor action.
- Reject malformed notebook schema values [#549](https://github.com/acgetchell/delaunay/pull/549)
  [`4af5e20`](https://github.com/acgetchell/delaunay/commit/4af5e200df00c9802081b5bf49ca52ce9446bfc9)

  - Reject non-object cell metadata with notebook and cell context.
  - Require nbformat to use the JSON integer 4 rather than numeric-equivalent values.
- Queue approved Dependabot updates for auto-merge [`14c7f36`](https://github.com/acgetchell/delaunay/commit/14c7f36776037c032d6adde977228b584610d3cb)
- Restore topology construction repair policy [#561](https://github.com/acgetchell/delaunay/pull/561)
  [`d1d2f7f`](https://github.com/acgetchell/delaunay/commit/d1d2f7fc95291b1d2d583d0412f660cba774a5dc)

  - Use the default Delaunay repair cadence during construction measurements.
  - Require insertion and completion validation to succeed for every sample.
  - Update the manual suite runtime estimate for the repaired workload.
- Enforce Proptest case-count precedence [#563](https://github.com/acgetchell/delaunay/pull/563)
  [`feeedca`](https://github.com/acgetchell/delaunay/commit/feeedca7332adbb3f18bd219dbd646891c852496)

  - Apply the 32-case repository fallback while preserving suite-specific budgets.
  - Let PROPTEST_CASES override repository and suite fallbacks, including manual runners.
  - Synchronize packaged test support, configuration declarations, and guidance.
  - Refresh the locked UUID dependency and uv tool pin.
- Align k=1 preflight with periodic realization [#564](https://github.com/acgetchell/delaunay/pull/564)
  [`3a66fca`](https://github.com/acgetchell/delaunay/commit/3a66fca44fdbebd61e45b1ccedea6c5aeaeb3738)

  - Evaluate insertion containment in the simplex's lifted periodic frame.
  - Share deterministic validation between feasibility checks and mutation.
  - Report typed exterior-facet and slot diagnostics without mutating topology.
- Harden release and performance preflight [#582](https://github.com/acgetchell/delaunay/pull/582)
  [`c310023`](https://github.com/acgetchell/delaunay/commit/c310023715c3d5631943c0866cfd4cec9ba55626)

  - reject release tags, citation dates, and DOI metadata that diverge from authoritative release metadata
  - fail identical release comparisons before setup and stream progress from long benchmark phases
  - preserve concrete CPU provenance and accept valid ordered Criterion confidence intervals
  - package the Python support tools with their dedicated documentation

### Maintenance

- Refresh dependencies and update automation [#544](https://github.com/acgetchell/delaunay/pull/544)
  [`d3072a4`](https://github.com/acgetchell/delaunay/commit/d3072a447a9630d642f0494957eb52420fcb6d5e)

  - Update Rust, Python, and repository tool pins to compatible releases.
  - Refresh setup-uv, CodeQL, and zizmor action revisions while preserving cache pruning.
  - Add grouped uv Dependabot updates and publish the ruleset-required CodeRabbit status.
- Update setuptools requirement in the dependencies group [#545](https://github.com/acgetchell/delaunay/pull/545)
  [`497e162`](https://github.com/acgetchell/delaunay/commit/497e1627bdbbf8c88327e69dd00ee09b9dd3d00f)

  Updates the requirements on [setuptools](https://github.com/pypa/setuptools) to permit the latest version.

  Updates `setuptools` to 83.0.0

  - [Release notes](https://github.com/pypa/setuptools/releases)
  - [Changelog](https://github.com/pypa/setuptools/blob/main/NEWS.rst)
  - [Commits](https://github.com/pypa/setuptools/compare/v77.0.0...v83.0.0)
- Update tooling pins and Dependabot groups [`24f52ae`](https://github.com/acgetchell/delaunay/commit/24f52ae4c0b26b4b64a30ed91af994dd6bb721ad)
- Bump the dependencies group with 2 updates [#555](https://github.com/acgetchell/delaunay/pull/555)
  [`ac1a1de`](https://github.com/acgetchell/delaunay/commit/ac1a1de9952d0c838b413d5a8ebb17a5e32c83ad)

  Bumps the dependencies group with 2 updates: [ty](https://github.com/astral-sh/ty) and [polars](https://github.com/pola-rs/polars).

  Updates `ty` from 0.0.65 to 0.0.66

  - [Release notes](https://github.com/astral-sh/ty/releases)
  - [Changelog](https://github.com/astral-sh/ty/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/astral-sh/ty/compare/0.0.65...0.0.66)

  Updates `polars` from 1.43.1 to 1.43.2
  - [Release notes](https://github.com/pola-rs/polars/releases)
  - [Commits](https://github.com/pola-rs/polars/compare/py-1.43.1...py-1.43.2)
- Refresh pins and extend the auto-merge wait [#556](https://github.com/acgetchell/delaunay/pull/556)
  [`c76f826`](https://github.com/acgetchell/delaunay/commit/c76f82689c88fa59908c0a11c0158c9e0a18961c)

  - Update cache-cargo-install-action, CodeQL, rumdl, and uv pins.
  - Give required checks 90 minutes to complete before auto-merge times out.
- Automate repository dependency updates [#571](https://github.com/acgetchell/delaunay/pull/571)
  [`36d04a9`](https://github.com/acgetchell/delaunay/commit/36d04a91266925f59dc6b95ee5fe8b7a46b3c0c5)

  - Advance Cargo requirements and every uv lock group through `just update`.
  - Reconcile setup-owned Cargo tool versions into the root justfile atomically.
  - Refresh Rust and Python dependencies and adapt support scripts to stricter type checking.
- Bump the github-actions group with 4 updates [#586](https://github.com/acgetchell/delaunay/pull/586)
  [`b99c3b9`](https://github.com/acgetchell/delaunay/commit/b99c3b93f3c68a7d7b526f8ca9de26355e4e60b0)

  Bumps the github-actions group with 4 updates: [github/codeql-action/upload-sarif](https://github.com/github/codeql-action),
  [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv), [github/codeql-action/init](https://github.com/github/codeql-action) and
  [github/codeql-action/analyze](https://github.com/github/codeql-action).

  Updates `github/codeql-action/upload-sarif` from 4.37.6 to 4.37.7

  - [Release notes](https://github.com/github/codeql-action/releases)
  - [Changelog](https://github.com/github/codeql-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/github/codeql-action/compare/5595ccaf912efad79be6eef63a5619ff05969be3...ff2f1c621b7f889edc0d3c761ac2e6a3f8cdb0dd)

  Updates `astral-sh/setup-uv` from 9.0.0 to 10.0.1
  - [Release notes](https://github.com/astral-sh/setup-uv/releases)
  - [Commits](https://github.com/astral-sh/setup-uv/compare/c771a70e6277c0a99b617c7a806ffedaca235ff9...20cfd1bf945f4377ade1205e4dbc17946fc9a30d)

  Updates `github/codeql-action/init` from 4.37.6 to 4.37.7
  - [Release notes](https://github.com/github/codeql-action/releases)
  - [Changelog](https://github.com/github/codeql-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/github/codeql-action/compare/5595ccaf912efad79be6eef63a5619ff05969be3...ff2f1c621b7f889edc0d3c761ac2e6a3f8cdb0dd)

  Updates `github/codeql-action/analyze` from 4.37.6 to 4.37.7
  - [Release notes](https://github.com/github/codeql-action/releases)
  - [Changelog](https://github.com/github/codeql-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/github/codeql-action/compare/5595ccaf912efad79be6eef63a5619ff05969be3...ff2f1c621b7f889edc0d3c761ac2e6a3f8cdb0dd)

### Performance

- Certify Level 4 realization fast paths [#559](https://github.com/acgetchell/delaunay/pull/559)
  [`3d3b7b0`](https://github.com/acgetchell/delaunay/commit/3d3b7b013312b972ec8c5b891a0b47b44cc9f8ee)

  - Preserve exact active-set classification for degenerate simplex pairs and
    verify optimized agreement across random and boundary cases in 2D–5D.
  - Add focused narrow-phase and whole-realization benchmarks with stable IDs
    to local and release comparison workflows.
  - Add an isolated large-scale Level 4 validation mode.
  - Add scoped Cargo and uv update recipes, refresh dependency locks, and pin
    rumdl 0.2.55.
- Accelerate full Delaunay reports [#560](https://github.com/acgetchell/delaunay/pull/560)
  [`e7fd9b8`](https://github.com/acgetchell/delaunay/commit/e7fd9b8b0050b98abfe3e6c949a406a90d84dde9)

  - Certify complete Euclidean point-set triangulations with robust local flip predicates in O(simplices).
  - Preserve exhaustive empty-sphere diagnostics for unproven connectivity, subset reports, and inconclusive certificates.
  - Add Level 5 performance canaries for well-conditioned and adversarial inputs from 2D through 5D.

#### Fixed: Preserve sound Delaunay report certificates

- Invalidate complete-point-set provenance after deletion, repair access, topology edits, and actual topology changes.
- Preserve certified fast paths across rejected or idempotent topology updates.
- Keep certificate provenance internal and retain exhaustive reporting for unproven connectivity.
- Cover cumulative 2D validation benchmarks and document fast-path and fallback complexity.

#### Changed: Cover report certificate invalidation paths

- Exercise certificate invalidation across topology-scope Pachner moves.
- Verify global topology changes revoke Euclidean report provenance.
- Clarify the complexity difference between Level 5 and cumulative reports.

#### Maintenance: Update uv and build dependency pins

- Pin uv 0.12.4 for local and CI tooling.
- Update cc to 1.4.3 and find-msvc-tools to 0.1.11.

## [0.8.0] - 2026-07-28

### ⚠️ Breaking Changes

- Validate toroidal domains at parse boundaries [#437](https://github.com/acgetchell/delaunay/pull/437) [#450](https://github.com/acgetchell/delaunay/pull/450)
- Add vertex construction macro [#469](https://github.com/acgetchell/delaunay/pull/469)
- Add embedded triangulation validation layer [#449](https://github.com/acgetchell/delaunay/pull/449) [#481](https://github.com/acgetchell/delaunay/pull/481)
- Add Pachner feasibility checks [#489](https://github.com/acgetchell/delaunay/pull/489)
- Add notebook-backed artifact workflows [#508](https://github.com/acgetchell/delaunay/pull/508)
- Add spherical backend and paper workflow [#509](https://github.com/acgetchell/delaunay/pull/509)
- Validate periodic quotient orientation [#521](https://github.com/acgetchell/delaunay/pull/521) [#527](https://github.com/acgetchell/delaunay/pull/527)
- Add certified realization intersection fast paths [#531](https://github.com/acgetchell/delaunay/pull/531)
- Box nested FlipError payloads [#406](https://github.com/acgetchell/delaunay/pull/406) [#435](https://github.com/acgetchell/delaunay/pull/435)
- Adopt la-stack 0.4.3 API [#424](https://github.com/acgetchell/delaunay/pull/424) [#438](https://github.com/acgetchell/delaunay/pull/438)
- Require refined generator and ordering parameters [#439](https://github.com/acgetchell/delaunay/pull/439)
- Parse coordinate inputs into validated types [#440](https://github.com/acgetchell/delaunay/pull/440) [#444](https://github.com/acgetchell/delaunay/pull/444)
- Require validated generator and Hilbert inputs [#452](https://github.com/acgetchell/delaunay/pull/452)
- Require validated coordinate topology API [#442](https://github.com/acgetchell/delaunay/pull/442) [#455](https://github.com/acgetchell/delaunay/pull/455)
- Return fallible facet iterators [#458](https://github.com/acgetchell/delaunay/pull/458)
- Hydrate TDS through validated UUID snapshots [#454](https://github.com/acgetchell/delaunay/pull/454) [#460](https://github.com/acgetchell/delaunay/pull/460)
- Normalize fallible constructors [#459](https://github.com/acgetchell/delaunay/pull/459) [#464](https://github.com/acgetchell/delaunay/pull/464)
- Borrow topology views from canonical storage [#472](https://github.com/acgetchell/delaunay/pull/472) [#474](https://github.com/acgetchell/delaunay/pull/474)
- Make topology views and boundaries owner-aware [#476](https://github.com/acgetchell/delaunay/pull/476)
- Split Pachner moves from vertex lifecycle edits [#477](https://github.com/acgetchell/delaunay/pull/477)
- Make fluent Delaunay builders canonical [#499](https://github.com/acgetchell/delaunay/pull/499)
- Replace raw TDS access with owner topology queries [#502](https://github.com/acgetchell/delaunay/pull/502)
- Tighten invariant-bearing triangulation APIs [#511](https://github.com/acgetchell/delaunay/pull/511)
- Simplify public APIs and repository workflows [#529](https://github.com/acgetchell/delaunay/pull/529)
- Reject stale adjacency indexes [#451](https://github.com/acgetchell/delaunay/pull/451) [#463](https://github.com/acgetchell/delaunay/pull/463)
- Report explicit topology preflight failures precisely [#512](https://github.com/acgetchell/delaunay/pull/512)
- Update tooling to Rust 1.96.0 [#430](https://github.com/acgetchell/delaunay/pull/430) [#431](https://github.com/acgetchell/delaunay/pull/431)
- Require Rust 1.97 for v0.8.0 [#523](https://github.com/acgetchell/delaunay/pull/523)

### Merged Pull Requests

- Harden changelog publication and repository tooling [#540](https://github.com/acgetchell/delaunay/pull/540)
- Bump github/codeql-action/upload-sarif [#539](https://github.com/acgetchell/delaunay/pull/539)
- Bump zizmorcore/zizmor-action from 0.5.7 to 0.6.0 [#538](https://github.com/acgetchell/delaunay/pull/538)
- Bump github/codeql-action/init from 4.37.0 to 4.37.1 [#537](https://github.com/acgetchell/delaunay/pull/537)
- Bump actions/setup-python from 6.3.0 to 7.0.0 [#536](https://github.com/acgetchell/delaunay/pull/536)
- Bump github/codeql-action/analyze from 4.37.0 to 4.37.1 [#535](https://github.com/acgetchell/delaunay/pull/535)
- Bump the dependencies group across 1 directory with 4 updates [#534](https://github.com/acgetchell/delaunay/pull/534)
- Bump actions/checkout from 7.0.0 to 7.0.1 [#533](https://github.com/acgetchell/delaunay/pull/533)
- Add certified realization intersection fast paths [#531](https://github.com/acgetchell/delaunay/pull/531)
- Harden benchmark parsing and correct audit guidance [#530](https://github.com/acgetchell/delaunay/pull/530)
- Simplify public APIs and repository workflows [#529](https://github.com/acgetchell/delaunay/pull/529)
- Align the five-level validation contract [#528](https://github.com/acgetchell/delaunay/pull/528)
- Validate periodic quotient orientation [#521](https://github.com/acgetchell/delaunay/pull/521) [#527](https://github.com/acgetchell/delaunay/pull/527)
- Add intrinsic orientability witnesses [#526](https://github.com/acgetchell/delaunay/pull/526)
- Add simplicial-complex artifact workflows [#525](https://github.com/acgetchell/delaunay/pull/525)
- Adopt la-stack 0.4.4 exact APIs [#524](https://github.com/acgetchell/delaunay/pull/524)
- Require Rust 1.97 for v0.8.0 [#523](https://github.com/acgetchell/delaunay/pull/523)
- Refactor/445 realization validation [#515](https://github.com/acgetchell/delaunay/pull/515)
- Perf/436 release benchmark workflow [#514](https://github.com/acgetchell/delaunay/pull/514)
- Report explicit topology preflight failures precisely [#512](https://github.com/acgetchell/delaunay/pull/512)
- Tighten invariant-bearing triangulation APIs [#511](https://github.com/acgetchell/delaunay/pull/511)
- Add spherical backend and paper workflow [#509](https://github.com/acgetchell/delaunay/pull/509)
- Add notebook-backed artifact workflows [#508](https://github.com/acgetchell/delaunay/pull/508)
- Replace raw TDS access with owner topology queries [#502](https://github.com/acgetchell/delaunay/pull/502)
- Add topology-aware simplex barycenters [#501](https://github.com/acgetchell/delaunay/pull/501)
- Make fluent Delaunay builders canonical [#499](https://github.com/acgetchell/delaunay/pull/499)
- Add Pachner feasibility checks [#489](https://github.com/acgetchell/delaunay/pull/489)
- Make targeted PL repair failure-atomic [#488](https://github.com/acgetchell/delaunay/pull/488)
- Preserve positive orientation after vertex removal [#487](https://github.com/acgetchell/delaunay/pull/487)
- Preserve exact 2D layered strip vertices [#486](https://github.com/acgetchell/delaunay/pull/486)
- Bump actions/checkout from 6.0.3 to 7.0.0 [#485](https://github.com/acgetchell/delaunay/pull/485)
- Bump zizmorcore/zizmor-action from 0.5.6 to 0.5.7 [#484](https://github.com/acgetchell/delaunay/pull/484)
- Add embedded triangulation validation layer [#449](https://github.com/acgetchell/delaunay/pull/449) [#481](https://github.com/acgetchell/delaunay/pull/481)
- Make topology mutations failure-atomic [#480](https://github.com/acgetchell/delaunay/pull/480)
- Add validated mesh export schema [#479](https://github.com/acgetchell/delaunay/pull/479)
- Add 2D edge-to-facet incidence queries [#478](https://github.com/acgetchell/delaunay/pull/478)
- Split Pachner moves from vertex lifecycle edits [#477](https://github.com/acgetchell/delaunay/pull/477)
- Make topology views and boundaries owner-aware [#476](https://github.com/acgetchell/delaunay/pull/476)
- Borrow topology views from canonical storage [#472](https://github.com/acgetchell/delaunay/pull/472) [#474](https://github.com/acgetchell/delaunay/pull/474)
- Refactor!(tds): canonicalize incidence storage [#473](https://github.com/acgetchell/delaunay/pull/473)
- Streamline iterator helpers for diagnostics work [#471](https://github.com/acgetchell/delaunay/pull/471)
- Add common Delaunay result alias [#470](https://github.com/acgetchell/delaunay/pull/470)
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
- Add common Delaunay result alias [#470](https://github.com/acgetchell/delaunay/pull/470)
  [`db46fa2`](https://github.com/acgetchell/delaunay/commit/db46fa2dd1518c1a47defda676453df8829665a6)

  - Add DelaunayError and DelaunayResult for common construction, insertion, validation, coordinate conversion, and toroidal-domain setup workflows.
  - Re-export the aliases from the crate root and construction preludes for downstream examples and applications.
  - Update public docs and examples to use DelaunayResult when workflow-specific errors are not required.
- Add 2D edge-to-facet incidence queries [#478](https://github.com/acgetchell/delaunay/pull/478)
  [`af2620b`](https://github.com/acgetchell/delaunay/commit/af2620b51a29c2fed7d33eabb5cfc0e867f0a419)

  - Add 2D `try_incident_facets_to_edge_2d` and
    `try_interior_facet_for_edge_2d` APIs on triangulation query surfaces.
  - Re-export `FacetHandle` from query-focused surfaces for callers that consume
    simplex-local facet handles outside Pachner-only imports.
  - Optimize `EdgeKey::try_new` to prove live edges from endpoint incidence while
    preserving typed incidence metadata errors.
  - Add edge-key construction benchmarks and document the simplex-local incidence
    query vocabulary.
- Add validated mesh export schema [#479](https://github.com/acgetchell/delaunay/pull/479)
  [`60e99d1`](https://github.com/acgetchell/delaunay/commit/60e99d1007f55a327a70b9e38b55c7223a822f01)

  - Add stable UUID-based visualization and mesh export DTOs with validated wrappers, typed topology schema values, and structured export/validation errors.
  - Expose `to_visualization_data` and `to_mesh_export` through crate-root and focused export prelude APIs, with DelaunayResult-compatible error conversions.
  - Document the mesh export JSON contract, validation boundary, and downstream attribute extension model.
  - Add reusable notebook hygiene tooling and flatten `just ci` into orthogonal validation buckets with one release-profile Rust nextest pass.
- [**breaking**] Add embedded triangulation validation layer [#449](https://github.com/acgetchell/delaunay/pull/449)
  [#481](https://github.com/acgetchell/delaunay/pull/481) [`7906525`](https://github.com/acgetchell/delaunay/commit/7906525ba0511796fd14fdbc46335bd4c851302c)

  Add a Level 4 "embedding" validation layer that certifies a triangulation
  is a faithful geometric embedding independently of the Delaunay predicate:
  maximal simplices are nondegenerate and meet only in shared faces within
  the active affine chart. This renumbers the Delaunay empty-circumsphere
  property to Level 5.

  - Add a public geometry::embedding module (backed by internal
    core::embedding) with the embedding checks and their typed errors.
  - Add Triangulation::is_valid_embedding, validate_embedding, and
    embedding_report for the new Level 4 layer.
  - Move Delaunay property validation to delaunay::property_validation
    (renamed from core::util::delaunay_validation) and expose the Level 5
    check as DelaunayTriangulation::is_valid_delaunay.
  - Refresh the validation guide, invariants, API-design, and prelude docs
    for the five-level stack.
- [**breaking**] Add Pachner feasibility checks [#489](https://github.com/acgetchell/delaunay/pull/489)
  [`be6728c`](https://github.com/acgetchell/delaunay/commit/be6728c4a84c988226a7bf4973ef1d382ff7e63b)

  - Add immutable flip and Pachner feasibility reports for dry-run workflows.
  - Route Pachner feasibility through the shared bistellar preflight used by mutating flips.
  - Export Pachner feasibility from the focused Pachner prelude.
- Add topology-aware simplex barycenters [#501](https://github.com/acgetchell/delaunay/pull/501)
  [`a096ece`](https://github.com/acgetchell/delaunay/commit/a096ececd9bd6bc2fa9046a260ed833fab5e6e7a)

  - Add `DelaunayTriangulation::simplex_barycenter` for Euclidean and periodic triangulations.
  - Surface typed barycenter failures through `SimplexBarycenterError` and `DelaunayError`.
  - Export the barycenter error through root, query, and prelude facades.
  - Add benchmark coverage for the public barycenter query.
  - Refresh dependency locks and align the pinned Markdown linter on `rumdl` 0.2.28.

#### Maintenance: Run Codecov through one llvm-cov nextest pass

- Generate Cobertura coverage and Codecov JUnit results from the same
  instrumented nextest run.
- Keep property tests in the coverage job with a dedicated coverage profile and
  bounded timeout for LLVM instrumentation overhead.
- Remove the stale multiple-crate-versions lint expectation and make the
  barycenter allocation benchmark tolerate exact-float roundoff.

#### Documentation: Document topology-aware barycenter workflows

- Document `simplex_barycenter` as the topology-aware local-editing point for
  k=1 Pachner insert proposals.
- Update query-prelude, benchmark, and coverage docs to match the current
  barycenter and Codecov/nextest surfaces.
- Cover barycenter overflow, periodic lift, canonicalization, and
  point-validation error paths while inheriting CI JUnit settings for coverage.
- [**breaking**] Add notebook-backed artifact workflows [#508](https://github.com/acgetchell/delaunay/pull/508)
  [`69f08d5`](https://github.com/acgetchell/delaunay/commit/69f08d5ca59a1611a3674660c604ea9905b93ddb)

  - Add the opt-in `delaunay` binary for triangulation, convex-hull,
    validation-demo, and Pachner stress JSON/CSV artifact generation.
  - Add Jupyter quickstart, validation visualization, and Pachner stress notebooks that render README imagery and Parquet-backed telemetry under
    `target/notebooks`.
  - Package the active docs, notebooks, and README hero asset for Cargo consumers, and align validation docs on valid affine realization terminology.
  - Add notebook Semgrep guardrails and CLI integration coverage for subcommands, artifact outputs, invalid arguments, and unsupported topology flags.
  - Implement spherical unit-sphere canonicalization for finite nonzero coordinates.
- [**breaking**] Add spherical backend and paper workflow [#509](https://github.com/acgetchell/delaunay/pull/509)
  [`3033088`](https://github.com/acgetchell/delaunay/commit/3033088480b3fcfc634265f29d13c8aa2b3bd461)

  - Add SphericalPoint and SphericalMetric as the spherical coordinate/metric backend for intrinsic S^D points embedded in R^(D+1).
  - Add the bounded S^2/S^3 SphericalDelaunayBuilder, SphericalDelaunayTriangulation, SphericalSimplex, geodesic distances, and typed spherical
    construction/validation errors.
  - Implement spherical validation through Intrinsic PL Topology, Embedding Validity, and empty-cap/ambient-hull Geometric Predicates while keeping D as the
    manifold dimension.
  - Expose spherical construction, validation, and topology backend types through the root and focused preludes.
  - Rename the validation hierarchy across crate docs, README, AGENTS guidance, architecture docs, semgrep fixtures, and developer docs.
  - Document spherical support as a Level 4/5 backend extension, with full 2D-5D integration and mutable-triangulation support still called out as future work.
  - Add the validation paper sources, BibTeX, generated PNG figures, tracked reviewer PDF, and papers README.
  - Rename the validation notebook to notebooks/01_validation.ipynb and make it generate reproducible validation hierarchy and failure figures.
  - Add just papers and `paper-*` recipes plus a dedicated papers.yml CI workflow using Tectonic, tex-fmt, chktex, uv, and notebook-generated figures.
  - Add uv-managed paper support CLIs for PDF sanity checks, deterministic PDF metadata normalization, and TeX source-date extraction.
  - Include paper artifacts in the packaged crate and ignore LaTeX build intermediates.
  - Update local/CI cargo-nextest pins and setup/tooling guidance for the new paper toolchain.
- Adopt la-stack 0.4.4 exact APIs [#524](https://github.com/acgetchell/delaunay/pull/524)
  [`e817406`](https://github.com/acgetchell/delaunay/commit/e817406567d1fe299103babe2aeb8a5e36f31aa3)

  - Use paired certified determinant filters and finite-by-construction matrices.
  - Preserve exact versus numerical singularity handling in circumcenter solves.
  - Add exact-fallback predicate and circumcenter benchmarks with a robustness example.
- Add simplicial-complex artifact workflows [#525](https://github.com/acgetchell/delaunay/pull/525)
  [`de8b861`](https://github.com/acgetchell/delaunay/commit/de8b861e74bbf748785de205d1a788f089d448fc)

  - Expose generic visualization JSON and deterministic S² hero exports through the CLI.
  - Add notebook-generated validation and spherical README imagery, with canonical assets shared by documentation and the paper.
  - Make notebook execution explicit, enforce stable descriptive cell IDs, and guard artifact conventions with Semgrep.
- Add intrinsic orientability witnesses [#526](https://github.com/acgetchell/delaunay/pull/526)
  [`cf93dec`](https://github.com/acgetchell/delaunay/commit/cf93dec02826562d989f22222edf03c1bf12ad54)

  - Expose orientation certificates for pure 2D and 3D simplicial complexes.
  - Report typed non-orientability failures during Level 3 PL-manifold validation.
  - Document intrinsic orientation and refresh the generated validation diagrams.
- [**breaking**] Validate periodic quotient orientation [#521](https://github.com/acgetchell/delaunay/pull/521)
  [#527](https://github.com/acgetchell/delaunay/pull/527) [`b4a2ef3`](https://github.com/acgetchell/delaunay/commit/b4a2ef3cefb31ca0c0b2534c2b593c533c4d1745)

  - Enforce lifted facet parity across periodic neighbors and self-identifications.
  - Include quotient constraints in Level 2 coherence and Level 3 orientability.
  - Propagate contradictory quotient orientation as typed construction failures.
  - Document translation-normalized parity and the distinction from Level 4 realization.

#### Added: Support compact 3D periodic quotients

- construct connected, closed 2D and compact 3D quotients with opposite facet orientations and complete canonical vertex coverage
- preserve distinct lifted vertex identities through simplex storage, incidence maps, snapshots, and realization validation
- recover deterministically from periodic covering-space degeneracies without changing ordinary Euclidean insertion behavior
- document the validated Euclidean, toroidal, and spherical construction scopes
- update the reviewed dprint and rumdl tool pins
- [**breaking**] Add certified realization intersection fast paths [#531](https://github.com/acgetchell/delaunay/pull/531)
  [`36e68ab`](https://github.com/acgetchell/delaunay/commit/36e68ab471371e6801aead28c74be9b3a054ab5b)

  - Add orientation, shared-face, and exact linear-program certificates before the conservative active-set fallback.
  - Dispatch finite floating-point systems through la-stack's exact Bareiss solver while preserving rational fallbacks and typed witnesses.
  - Bound the 4D and 5D release regressions and document the broader evidence deferred to #482 and #483.
  - Align release documentation and tooling with Rust 1.97.1.
  - Refresh pinned development tools and override Semgrep's vulnerable MCP dependency with the patched release.
  - Harden Tectonic dependency discovery across supported development environments.

### Changed

- [**breaking**] Box nested FlipError payloads [#406](https://github.com/acgetchell/delaunay/pull/406) [#435](https://github.com/acgetchell/delaunay/pull/435)
  [`2f310d9`](https://github.com/acgetchell/delaunay/commit/2f310d91e0b600fefdc488f2117dc6413b907b25)

  - Box nested typed `FlipError` payloads and the inserted-simplex witness while keeping scalar and key diagnostics inline.
  - Preserve typed inspection through `Error::source`, `reason.as_ref()`, and `source.as_ref()` for flip context, predicate, adjacency, simplex,
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
- Streamline iterator helpers for diagnostics work [#471](https://github.com/acgetchell/delaunay/pull/471)
  [`1985e6f`](https://github.com/acgetchell/delaunay/commit/1985e6ff95d0a452a09034d27d1bc9b21414befa)

  - Preserve point-generator preallocation while using fallible iterator control flow.
  - Extract Poisson spacing checks and stream geometry test distance calculations.
  - Gate benchmark setup tracing behind bench-logging and update diagnostics examples to use DelaunayResult.
- Refactor!(tds): canonicalize incidence storage [#473](https://github.com/acgetchell/delaunay/pull/473)
  [`9a973df`](https://github.com/acgetchell/delaunay/commit/9a973dfa598692261d9ea475f179862e3e92de4d)
- [**breaking**] Borrow topology views from canonical storage [#472](https://github.com/acgetchell/delaunay/pull/472)
  [#474](https://github.com/acgetchell/delaunay/pull/474) [`3b6d2d4`](https://github.com/acgetchell/delaunay/commit/3b6d2d4a71e0f40a2e8b1be3871696e9b8f13866)

  - Return validated borrowed simplex vertex slices instead of owned or optional detached snapshots.
  - Split convex hull facet access into detached `facet_handles()` and borrowed `facets(triangulation)` views with freshness checks.
  - Make vertex and simplex payload setters checked mutations that report typed stale-key errors.
  - Preserve fallback rebuild payload restoration through typed simplex-data restore errors.
- [**breaking**] Make topology views and boundaries owner-aware [#476](https://github.com/acgetchell/delaunay/pull/476)
  [`db3fcb3`](https://github.com/acgetchell/delaunay/commit/db3fcb3a899db625f3e969c8d2fa098520826479)

  - Replace raw TDS boundary queries with one-sided facet-incidence APIs and keep true boundary classification topology-aware.
  - Add borrowed EdgeView, RidgeQuery, RidgeView, and RidgeLinkView surfaces around detached EdgeKey and RidgeCandidate values.
  - Interpret boundary facets through GlobalTopology so periodic quotient self-identifications remain closed topology.
  - Align Euler validation, focused preludes, docs, examples, benchmarks, and Semgrep rules with the new view/candidate/incidence model.
- [**breaking**] Split Pachner moves from vertex lifecycle edits [#477](https://github.com/acgetchell/delaunay/pull/477)
  [`f6efe10`](https://github.com/acgetchell/delaunay/commit/f6efe10c776e08a52efe1d2a4b4e65cf56271cf2)

  - Add a unified PachnerMove request/result API with attempt_pachner dispatch and a focused prelude::pachner import surface.
  - Move vertex deletion into its own Delaunay module with typed DeleteVertexError and keep insertion/deletion terminology explicit through public docs and
    examples.
  - Keep primitive bistellar flip APIs available from delaunay::flips while hiding them from focused preludes intended for workflow users.
  - Add Pachner stress coverage, delete_vertex benchmarks, and Semgrep rules that require fallible results to be consumed and workflow fixtures to use vertex!.
  - Split long agent/development guidance into focused dev and architecture docs, including release citation and performance-tuning checklists.
- [**breaking**] Make fluent Delaunay builders canonical [#499](https://github.com/acgetchell/delaunay/pull/499)
  [`9cecd49`](https://github.com/acgetchell/delaunay/commit/9cecd49dea00faa0e830ea033774f085e36b0bd4)

  - Make DelaunayTriangulationBuilder the canonical construction API, with
    default simplex storage, typed simplex-data selection, kernel terminals, and
    statistics terminals in one staged workflow.
  - Remove the legacy `DelaunayTriangulation::try_new*` and `try_with_*` batch
    constructor family, and update examples, benches, docs, preludes, and
    semgrep rules to use builder chains.
  - Add post-construction simplex-data filling from closures or secondary maps
    with typed SimplexDataFillError handling.
  - Rework RandomTriangulationBuilder around validated point counts, coordinate
    ranges, vertex/simplex data type selection, and fluent build terminals.
  - Refresh public Pachner, Delaunay repair, and locate workflows around the
    current proposal and builder APIs.
- [**breaking**] Replace raw TDS access with owner topology queries [#502](https://github.com/acgetchell/delaunay/pull/502)
  [`bc53e32`](https://github.com/acgetchell/delaunay/commit/bc53e32eae7a2f5e234cb851cbb17b1833888839)

  - Expose owner-bound topology, identity, ridge, facet, locate, and validation queries on Triangulation and DelaunayTriangulation so callers no longer need raw
    TDS access.
  - Route Pachner moves through the owner transaction API with rollback, orientation repair, and post-move validation while keeping primitive flips as internal
    building blocks.
  - Publish Level 3 topology helpers for PL-manifold checks, including ridge-link and vertex-link validation, and align docs, examples, doctests, and preludes
    around the new API surface.
  - Add MCMC-backed Pachner stress and repair transaction-pressure benchmarks with diagnostics for failed long-run move chains.
  - Extend Semgrep rules to prevent regressions into public TDS/storage access and validation naming drift.
- [**breaking**] Tighten invariant-bearing triangulation APIs [#511](https://github.com/acgetchell/delaunay/pull/511)
  [`6d5020d`](https://github.com/acgetchell/delaunay/commit/6d5020d53e0279c8f8962a01bf021447ad868e1d)

  - Encapsulate flip-kind dimensions and periodic span witnesses behind accessors while adding fluent repair and delaunayize config setters.
  - Relax unnecessary serialization, surface-measure, and coordinate-representation bounds so callers can use narrower payload and kernel types.
  - Precheck explicit simplex imports for duplicate cells and over-shared facets before using the prechecked TDS assembly path.
  - Add explicit-import benchmark coverage, consolidate repeated dimension tests, and quiet ad hoc test logging.
  - Align uv-backed workflows on uv 0.11.27 and run the papers workflow on macOS with Homebrew system dependencies.
- Perf/436 release benchmark workflow [#514](https://github.com/acgetchell/delaunay/pull/514)
  [`ff74d7c`](https://github.com/acgetchell/delaunay/commit/ff74d7c98b5c2a3ab51a6f667b7547e3903a254a)
- Refactor/445 realization validation [#515](https://github.com/acgetchell/delaunay/pull/515)
  [`e6e22f6`](https://github.com/acgetchell/delaunay/commit/e6e22f63045514ea2c806819a4fdcd11932f6353)
- [**breaking**] Simplify public APIs and repository workflows [#529](https://github.com/acgetchell/delaunay/pull/529)
  [`9bfeb2a`](https://github.com/acgetchell/delaunay/commit/9bfeb2a5b528d7800c87c92321154ac2145ec279)

  - Remove redundant public helpers and generation-tracked facet caching while retaining immutable convex-hull memoization.
  - Replace the ambiguous canonicalized-toroidal mode with true periodic construction and consistent T^n/S^n notation.
  - Complete workflow discoverability through focused Rust examples, visualization notebooks, rustdoc idioms, and the v0.8.0 reviewer artifact guide.
  - Make Just recipes orthogonal, composable, sorted, and protected by CI-aligned tool-version guards.

### Dependencies

- Bump support tooling and smallvec [`1799d3c`](https://github.com/acgetchell/delaunay/commit/1799d3cbd4a5c01a10c490fffe15cc28b1f3784d)

### Documentation

- Refresh README and changelog [`b80d25a`](https://github.com/acgetchell/delaunay/commit/b80d25a40df9a06de8554a9900598bdf2a584dbf)

  - Reorganize the README around a concise quickstart, documentation map,
    ecosystem notes, benchmarking guidance, and current limitations.
  - Preserve the citation abstract and detailed feature checklist while updating
    quickstart guidance for the current published crate version.
  - Regenerate the active and archived changelog files with current release-note
    formatting.
- Refresh validation failure artifacts [`282a72d`](https://github.com/acgetchell/delaunay/commit/282a72d3e56e76b345733a063072affa580faacc)

  - Regenerate the validation-failure figure after explicit duplicate-simplex diagnostics switched to input-index wording.
  - Rebuild the tracked reviewer PDF so it embeds the refreshed figure.
- Align the five-level validation contract [#528](https://github.com/acgetchell/delaunay/pull/528)
  [`93fcdce`](https://github.com/acgetchell/delaunay/commit/93fcdce2c42f58c03f24ac07c81bf2622be8aee6)

  - Distinguish stored orientation, intrinsic orientability, and geometric
    orientation across the public documentation.
  - Synchronize validation APIs, layered errors, and backend-specific
    realization guarantees.
  - Refresh the mechanical paper table and author TODO scaffolding without replacing manuscript prose.

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
- Make topology mutations failure-atomic [#480](https://github.com/acgetchell/delaunay/pull/480)
  [`8ef6a95`](https://github.com/acgetchell/delaunay/commit/8ef6a95c8dd6adfaa351f81dd54e16fb8746e64f)

  - Add scoped rollback guards for TDS, generic triangulation, and Delaunay mutation windows.
  - Use the guards across insertion, deletion, local facet repair, and flip-repair retry paths.
  - Restore insertion state and spatial-index caches according to the Delaunay rollback policy.
  - Document rollback ownership rules for future mutation paths.

#### Fixed: Make Delaunay mutation rollback state-aware

- Share owner-bound TDS rollback guards across TDS, Triangulation, and Delaunay wrappers.
- Restore insertion bookkeeping with rollback and restore or invalidate spatial indexes according to the mutation policy.
- Avoid rollback snapshots when local facet repair has no issues or post-insertion repair/check cadence cannot run.
- Preserve exact 2D layered strip vertices [#486](https://github.com/acgetchell/delaunay/pull/486)
  [`2b45995`](https://github.com/acgetchell/delaunay/commit/2b45995fc7470ca326ee6c20a47ea6641a29c63c)

  - Add a non-enforcing construction option that returns valid Levels 1-4 triangulations without automatic Level 5 Delaunay repair.
  - Split exact 2D hull-edge insertions before symbolic perturbation can route them through interior insertion.
  - Allow explicit simplex imports to opt out of Level 5 validation while keeping strict Delaunay validation as the default.

#### Fixed: Preserve exact layered CDT strip vertices

- Preserve collinear 2D boundary vertices by splitting localized boundary-edge facets while keeping exact coordinates and labels intact.
- Allow exact degenerate batch construction to return after Levels 1-4 validation when final Level 5 Delaunay enforcement is disabled.
- Carry explicit simplex validation evidence into construction and reject point-insertion-only options before assembling explicit meshes.
- Enforce short vertex fixture construction across non-constructor tests, examples, and benchmarks.

#### Fixed: Preserve batch finalization policies

- Restore seeded construction finalization from the final Delaunay policy so
  returned triangulations keep the intended repair and global-fallback behavior.
- Keep the workflow vertex-macro guardrail Rust-AST aware so it only flags real
  Vertex constructor calls.

#### Changed: Cover qualified Vertex constructor guard

- Add a workflow vertex-macro fixture for the qualified
  delaunay::prelude::Vertex constructor path.
- Keep the Semgrep rule YAML formatted for the repository config checks.
- Preserve positive orientation after vertex removal [#487](https://github.com/acgetchell/delaunay/pull/487)
  [`d6e9eea`](https://github.com/acgetchell/delaunay/commit/d6e9eea778361aebace37dd451d6189bece9020d)

  - Canonicalize coherent orientation per simplex-neighbor component so disconnected repair states do not keep a negative geometric sign.
  - Surface post-repair orientation failures through the existing typed invariant errors.
  - Avoid redundant full-TDS canonicalization passes on the successful normalization path.
- Make targeted PL repair failure-atomic [#488](https://github.com/acgetchell/delaunay/pull/488)
  [`abd466f`](https://github.com/acgetchell/delaunay/commit/abd466f38da370b58610e7b34cb91aca800abb65)

  - Extend PL-manifold repair from facet over-sharing to bounded boundary-ridge, ridge-link, and vertex-link repair stages with typed diagnostics.
  - Run delaunayize through the Delaunay rollback transaction so failed topology repair, failed Delaunay repair, fallback snapshot failures, and rebuild
    failures restore the pre-call triangulation.
  - Expose PlManifoldRepairStage through the crate root and delaunayize prelude, and add benchmark fixtures for targeted topology repair.
  - Align local and CI tooling on Python 3.14, uv 0.11.26, and the reviewed Rust/Python tool pins.
- [**breaking**] Report explicit topology preflight failures precisely [#512](https://github.com/acgetchell/delaunay/pull/512)
  [`bbae9ed`](https://github.com/acgetchell/delaunay/commit/bbae9ed5898f6e2633f9cb75a08aacfa5e75c25f)

  - Add typed explicit-simplex duplicate and facet-sharing errors that preserve input simplex and vertex indices.
  - Preflight explicit topology from validated simplex specs before constructing TDS simplices.
  - Rename the Delaunay repair flip-budget fluent setters to make their repair scope explicit.
  - Repair the papers workflow TeX dependency setup for macOS runners.
- Harden benchmark parsing and correct audit guidance [#530](https://github.com/acgetchell/delaunay/pull/530)
  [`91efa35`](https://github.com/acgetchell/delaunay/commit/91efa35ef26fa9ad7360651c686578b0591ff3c1)

  - Reject non-finite, non-positive, and misordered benchmark intervals.
  - Align numerical and validation documentation with implemented behavior.
  - Record completed remediation work and correct the Level 4 issue owner.
  - Update the pinned uv version to 0.11.29.
- Harden changelog publication and repository tooling [#540](https://github.com/acgetchell/delaunay/pull/540)
  [`1f15b93`](https://github.com/acgetchell/delaunay/commit/1f15b9352456aca9eade44d116cd8d15d3d1ac20)

  - Reject malformed, duplicated, misordered, or invalidly dated release headings before modifying changelog files.
  - Publish root and archived changelogs transactionally with rollback-safe writes and concise CLI errors.
  - Lock uv-backed workflows and include the complete console-script module closure in Python packages.
  - Consolidate durable tooling guidance, correct documented predicate and validation limits, and archive historical performance observations.
  - Refresh validation-paper metadata and clarify reviewer-artifact comparisons.

### Maintenance

- Bump starlette in the uv group across 1 directory [#428](https://github.com/acgetchell/delaunay/pull/428)
  [`1cde3ee`](https://github.com/acgetchell/delaunay/commit/1cde3ee62e23065d657a5cc9f8a438fe72933632)

  Bumps the uv group with 1 update in the / directory: [starlette](https://github.com/Kludex/starlette).

  Updates `starlette` from 1.0.0 to 1.0.1

  - [Release notes](https://github.com/Kludex/starlette/releases)
  - [Changelog](https://github.com/Kludex/starlette/blob/main/docs/release-notes.md)
  - [Commits](https://github.com/Kludex/starlette/compare/1.0.0...1.0.1)
- Bump taiki-e/install-action from 2.79.1 to 2.81.1 [#427](https://github.com/acgetchell/delaunay/pull/427)
  [`3798932`](https://github.com/acgetchell/delaunay/commit/37989328340d1fcc6c8fc0a8be5adf03b31f7fcd)

  Bumps [taiki-e/install-action](https://github.com/taiki-e/install-action) from 2.79.1 to 2.81.1.

  - [Release notes](https://github.com/taiki-e/install-action/releases)
  - [Changelog](https://github.com/taiki-e/install-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/taiki-e/install-action/compare/b550161ef8a7bc4f2a671c0b03a18ac9ccedea1e...e49978b799e49ff429d162b7a30601a569ab6538)
- Bump the dependencies group across 1 directory with 4 updates [#426](https://github.com/acgetchell/delaunay/pull/426)
  [`daabc99`](https://github.com/acgetchell/delaunay/commit/daabc99f012f4c35789f71777d77e3e2b08593aa)

  Bumps the dependencies group with 4 updates in the / directory: [uuid](https://github.com/uuid-rs/uuid), [pastey](https://github.com/as1100k/pastey),
  [serde_json](https://github.com/serde-rs/json) and [sysinfo](https://github.com/GuillaumeGomez/sysinfo).

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
- Bump actions/checkout from 6.0.2 to 6.0.3 [#456](https://github.com/acgetchell/delaunay/pull/456)
  [`b41393e`](https://github.com/acgetchell/delaunay/commit/b41393e0f6598e3598e1d533a7ca59f5bb195b46)

  Bumps [actions/checkout](https://github.com/actions/checkout) from 6.0.2 to 6.0.3.

  - [Release notes](https://github.com/actions/checkout/releases)
  - [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/actions/checkout/compare/de0fac2e4500dabe0009e67214ff5f5447ce83dd...df4cb1c069e1874edd31b4311f1884172cec0e10)
- Harden GHAS security workflows [`add3dea`](https://github.com/acgetchell/delaunay/commit/add3deab592415e28f95faeb197bac099facf792)

  - Restrict manual performance-baseline materialization to main and semver tags instead of checking out arbitrary validated branches.
  - Pin CodeQL actions to the documented v4.36.2 commit so hash comments verify cleanly.
  - Rename benchmark Cargo-mode metadata to avoid CodeQL treating benchmark profile strings as clear-text secret storage.
- Avoid CodeQL benchmark metadata false positive [`9b16822`](https://github.com/acgetchell/delaunay/commit/9b1682299f0254c42f7c305dd4fc889d1ade862e)

  - Rename the internal benchmark build-flavor constant so CodeQL no longer treats the perf label as sensitive data.
  - Preserve baseline metadata output and benchmark command behavior.
- Bump the uv group across 1 directory with 3 updates [#462](https://github.com/acgetchell/delaunay/pull/462)
  [`4fbc838`](https://github.com/acgetchell/delaunay/commit/4fbc8383c7ea07fef943e3e0fe677d819b1d4761)

  Bumps the uv group with 3 updates in the / directory: [cryptography](https://github.com/pyca/cryptography),
  [python-multipart](https://github.com/Kludex/python-multipart) and [starlette](https://github.com/Kludex/starlette).

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
- Bump actions/checkout from 6.0.3 to 7.0.0 [#485](https://github.com/acgetchell/delaunay/pull/485)
  [`9a73c9d`](https://github.com/acgetchell/delaunay/commit/9a73c9dd5092e18b8d910d63b9f19110ecdeffe7)

  Bumps [actions/checkout](https://github.com/actions/checkout) from 6.0.3 to 7.0.0.

  - [Release notes](https://github.com/actions/checkout/releases)
  - [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/actions/checkout/compare/df4cb1c069e1874edd31b4311f1884172cec0e10...9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0)
- Bump zizmorcore/zizmor-action from 0.5.6 to 0.5.7 [#484](https://github.com/acgetchell/delaunay/pull/484)
  [`f1109b6`](https://github.com/acgetchell/delaunay/commit/f1109b6c2e2aab5c54280058eb8bd1d80c830b2b)

  Bumps [zizmorcore/zizmor-action](https://github.com/zizmorcore/zizmor-action) from 0.5.6 to 0.5.7.

  - [Release notes](https://github.com/zizmorcore/zizmor-action/releases)
  - [Commits](https://github.com/zizmorcore/zizmor-action/compare/5f14fd08f7cf1cb1609c1e344975f152c7ee938d...192e21d79ab29983730a13d1382995c2307fbcaa)
- Install native Tectonic dependencies [`2ba9c64`](https://github.com/acgetchell/delaunay/commit/2ba9c643ba34c107b05cac6991ca1b715e943672)

  - Install the Linux font, Graphite2, HarfBuzz, ICU, libpng, and zlib development headers needed to compile Tectonic in the papers workflow.
  - Document the paper workflow's native dependency boundary for local setup and CI parity.
  - Keep the validation paper outline ChkTeX/tex-fmt clean without changing author-owned prose.
  - Add a progress breadcrumb before benchmark harness compilation so long Windows/MSVC CI runs do not look idle.
- Repair paper workflow and audit lockfile [`703eb1e`](https://github.com/acgetchell/delaunay/commit/703eb1ebb4f8d92b475b2acc8017cdb6eacba3fe)

  - Prebuild the perf-profile `delaunay` CLI before executing paper notebooks so nbconvert timeouts cover the validation demo run instead of Rust compilation.
  - Pass the prebuilt binary through `DELAUNAY_BINARY` when refreshing paper figures.
  - Document the paper CLI prebuild step for local and CI paper workflows.
  - Update transitive `crossbeam-epoch` to 0.9.20 to clear RUSTSEC-2026-0204.
- Stabilize paper dependency setup [`d8f02e5`](https://github.com/acgetchell/delaunay/commit/d8f02e5ce551c632d773475cb3664821c527ede4)

  - Cache Homebrew downloads for paper workflow bottles and casks.
  - Install the native Tectonic bridge dependencies expected by Cargo builds.
  - Update BasicTeX's tlmgr before installing chktex on fresh macOS runners.
- [**breaking**] Require Rust 1.97 for v0.8.0 [#523](https://github.com/acgetchell/delaunay/pull/523)
  [`9592849`](https://github.com/acgetchell/delaunay/commit/95928496a58a4c61a941de80d891ff760f7a073a)

  - align MSRV metadata, tooling, and documentation with Rust 1.97
  - adopt the stabilized Hilbert highest-bit helper
  - apply new Clippy and rustdoc diagnostics and finish assertion cleanup
  - clarify the predicate benchmark workload and cold-path audit evidence
- Bump actions/checkout from 7.0.0 to 7.0.1 [#533](https://github.com/acgetchell/delaunay/pull/533)
  [`faa807d`](https://github.com/acgetchell/delaunay/commit/faa807d0d581ffed208fc293a44cb960ddaded9d)

  Bumps [actions/checkout](https://github.com/actions/checkout) from 7.0.0 to 7.0.1.

  - [Release notes](https://github.com/actions/checkout/releases)
  - [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/actions/checkout/compare/9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0...3d3c42e5aac5ba805825da76410c181273ba90b1)
- Bump zizmorcore/zizmor-action from 0.5.7 to 0.6.0 [#538](https://github.com/acgetchell/delaunay/pull/538)
  [`3a96ffc`](https://github.com/acgetchell/delaunay/commit/3a96ffc6bc82c9a637d9e635948c989b829c8557)

  Bumps [zizmorcore/zizmor-action](https://github.com/zizmorcore/zizmor-action) from 0.5.7 to 0.6.0.

  - [Release notes](https://github.com/zizmorcore/zizmor-action/releases)
  - [Commits](https://github.com/zizmorcore/zizmor-action/compare/192e21d79ab29983730a13d1382995c2307fbcaa...6599ee8b7a49aef6a770f63d261d214911a7ce02)
- Bump actions/setup-python from 6.3.0 to 7.0.0 [#536](https://github.com/acgetchell/delaunay/pull/536)
  [`a536730`](https://github.com/acgetchell/delaunay/commit/a5367308df59d981677032b7d1f2ba51807cb007)

  Bumps [actions/setup-python](https://github.com/actions/setup-python) from 6.3.0 to 7.0.0.

  - [Release notes](https://github.com/actions/setup-python/releases)
  - [Commits](https://github.com/actions/setup-python/compare/ece7cb06caefa5fff74198d8649806c4678c61a1...5fda3b95a4ea91299a34e894583c3862153e4b97)
- Bump github/codeql-action/upload-sarif [#539](https://github.com/acgetchell/delaunay/pull/539)
  [`01f6d8d`](https://github.com/acgetchell/delaunay/commit/01f6d8d06b5aa0326744d969a9338d331fffc450)

  Bumps [github/codeql-action/upload-sarif](https://github.com/github/codeql-action) from 4.37.0 to 4.37.1.

  - [Release notes](https://github.com/github/codeql-action/releases)
  - [Changelog](https://github.com/github/codeql-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/github/codeql-action/compare/99df26d4f13ea111d4ec1a7dddef6063f76b97e9...7188fc363630916deb702c7fdcf4e481b751f97a)
- Bump github/codeql-action/init from 4.37.0 to 4.37.1 [#537](https://github.com/acgetchell/delaunay/pull/537)
  [`a0d15b1`](https://github.com/acgetchell/delaunay/commit/a0d15b1319b97e96522ed79876f62428627d1ded)

  Bumps [github/codeql-action/init](https://github.com/github/codeql-action) from 4.37.0 to 4.37.1.

  - [Release notes](https://github.com/github/codeql-action/releases)
  - [Changelog](https://github.com/github/codeql-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/github/codeql-action/compare/99df26d4f13ea111d4ec1a7dddef6063f76b97e9...7188fc363630916deb702c7fdcf4e481b751f97a)
- Bump github/codeql-action/analyze from 4.37.0 to 4.37.1 [#535](https://github.com/acgetchell/delaunay/pull/535)
  [`4df0e84`](https://github.com/acgetchell/delaunay/commit/4df0e84b4bdf1d96db91c00a2ecbbfce5c3c9d34)

  Bumps [github/codeql-action/analyze](https://github.com/github/codeql-action) from 4.37.0 to 4.37.1.

  - [Release notes](https://github.com/github/codeql-action/releases)
  - [Changelog](https://github.com/github/codeql-action/blob/main/CHANGELOG.md)
  - [Commits](https://github.com/github/codeql-action/compare/99df26d4f13ea111d4ec1a7dddef6063f76b97e9...7188fc363630916deb702c7fdcf4e481b751f97a)
- Bump the dependencies group across 1 directory with 4 updates [#534](https://github.com/acgetchell/delaunay/pull/534)
  [`8836e34`](https://github.com/acgetchell/delaunay/commit/8836e346fb4d997e1a1301f101fe9585ec55d7d8)

  Bumps the dependencies group with 4 updates in the / directory: [clap](https://github.com/clap-rs/clap), [serde](https://github.com/serde-rs/serde),
  [thiserror](https://github.com/dtolnay/thiserror) and [uuid](https://github.com/uuid-rs/uuid).

  Updates `clap` from 4.6.1 to 4.6.2

  - [Release notes](https://github.com/clap-rs/clap/releases)
  - [Changelog](https://github.com/clap-rs/clap/blob/master/CHANGELOG.md)
  - [Commits](https://github.com/clap-rs/clap/compare/clap_complete-v4.6.1...clap_complete-v4.6.2)

  Updates `serde` from 1.0.228 to 1.0.229
  - [Release notes](https://github.com/serde-rs/serde/releases)
  - [Commits](https://github.com/serde-rs/serde/compare/v1.0.228...v1.0.229)

  Updates `thiserror` from 2.0.18 to 2.0.19
  - [Release notes](https://github.com/dtolnay/thiserror/releases)
  - [Commits](https://github.com/dtolnay/thiserror/compare/2.0.18...2.0.19)

  Updates `uuid` from 1.23.5 to 1.24.0
  - [Release notes](https://github.com/uuid-rs/uuid/releases)
  - [Commits](https://github.com/uuid-rs/uuid/compare/v1.23.5...v1.24.0)
- Gate automatic merges on CodeRabbit review [`c30083e`](https://github.com/acgetchell/delaunay/commit/c30083ea1bc4bf1b276a35feb35fe647dec1768e)

  - Request CodeRabbit review and enable squash auto-merge for Dependabot PRs.
  - Fail closed when CodeRabbit cannot review and group GitHub Actions updates.
  - Refresh Rust dependencies and allow the intentional syn 2/syn 3 split.

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

## Archives

Older releases are archived by minor series:

- [0.7.x](docs/archive/changelog/0.7.md)
- [0.6.x](docs/archive/changelog/0.6.md)
- [0.5.x](docs/archive/changelog/0.5.md)
- [0.4.x](docs/archive/changelog/0.4.md)
- [0.3.x](docs/archive/changelog/0.3.md)
- [0.2.x](docs/archive/changelog/0.2.md)

[0.8.1]: https://github.com/acgetchell/delaunay/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/acgetchell/delaunay/compare/v0.7.8...v0.8.0

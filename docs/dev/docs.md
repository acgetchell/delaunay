# Documentation Guidance

Documentation, changelog, citation, and scientific-notation rules for this
repository.

---

## Contents

- [Documentation Ownership](#documentation-ownership)
- [Scientific Notation](#scientific-notation)
- [References](#references)
- [Changelog Maintenance](#changelog-maintenance)
- [Crates.io Documentation](#cratesio-documentation)

---

## Documentation Ownership

- `AGENTS.md` is the coding-agent entry point.
- `docs/dev/README.md` indexes development workflow guidance.
- `docs/code_organization.md` is the required-reading architecture hub.
- `docs/architecture/README.md` indexes focused architecture references.
- `REFERENCES.md` owns literature references.
- `CITATION.cff` owns citation metadata.
- `README.md`'s first Introduction paragraph and `CITATION.cff`'s `abstract`
  field intentionally mirror each other. When one changes, update the other in
  the same patch. `scripts/tests/test_readme_citation_mirror.py` checks the
  normalized text after Markdown links are stripped from the README prose.
  Semgrep also rejects stale public validation-hierarchy wording.
- `docs/archive/` stores historical plans, completed changelog series, and old
  design notes. Do not update archived docs as active guidance unless an
  explicit archive-maintenance task asks for it.

## Scientific Notation

Unicode math is welcome in doc comments and active documentation when it makes
scientific meaning clearer. Examples include:

```text
×, ≤, ≥, ∈, Σ, ², 2^-50, χ(S^d) = 1 + (−1)^d
```

Prefer mathematical invariants where possible instead of prose-only
descriptions.

## References

Algorithms should cite their sources in `REFERENCES.md` and document relevant
conditioning behavior. Typical references include Shewchuk, Bowyer-Watson,
Edelsbrunner, and Preparata-Shamos.

Use numbered `REFERENCES.md` citations in API docs and active design docs when
describing algorithms, numerical robustness, or topology guarantees.

## Changelog Maintenance

- Never edit `CHANGELOG.md` or `docs/archive/changelog/*.md` manually.
- Run `just changelog` to regenerate the root changelog and archive files from
  commits.
- The root `CHANGELOG.md` contains only `Unreleased` plus the active minor
  series.
- Completed minor series are archived in `docs/archive/changelog/X.Y.md`.

Commit-message rules that affect generated changelog text live in
[`git.md`](git.md).

## Crates.io Documentation

Publish documentation changes before bumping the crates.io version. crates.io
does not allow re-publishing documentation without a version bump.

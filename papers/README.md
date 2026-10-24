# Papers

This directory contains publication-facing prose for `delaunay`.

## Optional Maintainer Setup

Paper production uses the repository's root Python environment and `uv.lock`.
The opt-in `papers` dependency group supplies PDF support; `tooling.toml` owns
the Tectonic and `tex-fmt` pins and PDF policy. The root `justfile` imports
`recipes.just`, so every command below runs from the repository root or a
subdirectory. Paper-specific tests live in `tests/papers/` and run through
`just paper-test`, independently of the default Python test suite.

Provision native dependencies before installing the optional tools:

- macOS: install FreeType, Graphite2, ICU, libpng, pkg-config, and zlib, for example
  with `brew install freetype graphite2 icu4c libpng pkgconf zlib`.
- Linux: install pkg-config and the development packages for fontconfig,
  FreeType, Graphite2/HarfBuzz, ICU, libpng, OpenSSL, and zlib.
- Windows MSVC: provision the matching vcpkg libraries and set
  `TECTONIC_DEP_BACKEND=vcpkg`, `VCPKG_ROOT`, and `VCPKGRS_TRIPLET`.
- Install ChkTeX through a TeX distribution or system package manager. macOS
  recipes discover an existing MacTeX/BasicTeX installation at `/Library/TeX/texbin`.

Then explicitly install and check the paper tools:

```bash
just paper-setup
just paper-check
```

`just setup`, `just check`, `just test`, and `just ci` require no TeX tools.
The Papers GitHub Actions workflow runs only through manual dispatch. Its macOS
native setup and bounded Tectonic bundle warmup live in `ci-setup.sh` and
`ci-warmup.sh`; ordinary CI does not provision those dependencies.

## Artifacts and Commands

[`ARTIFACT.md`](ARTIFACT.md) is the reviewer-facing v0.8.0 software artifact
index. It records release identity, reproduction commands, evidence links, and
known limits without replacing the canonical technical documentation or adding
manuscript prose.

The repository keeps three validation artifacts with distinct ownership:

- `docs/construction_and_validation.md` is the canonical developer-facing construction and validation contract.
- `papers/validation.tex` is the mathematical and architecture exposition.
- `notebooks/01_validation.ipynb` is the source of truth for reproducible figures.

Run the paper workflow with:

```bash
just papers
```

That recipe refreshes the canonical validation figures under
`docs/assets/validation/`, checks TeX formatting with `tex-fmt`, lints
TeX with `chktex`, compiles the paper with Tectonic in `target/papers/`, and
copies the final reading copy to `papers/validation.pdf`. The build derives
`SOURCE_DATE_EPOCH` from the paper's explicit `\date{...}` command so CI
rebuilds do not refresh PDF metadata timestamps, then normalizes the remaining
volatile PDF/XMP identifiers before copying the reviewer PDF.

For quick TeX iteration without regenerating notebook-owned figures, run:

```bash
just paper-check
```

That recipe lints the source, rebuilds the target PDF, and runs the uv-managed
`paper-pdf-check` sanity check on that build. Run `just paper-artifact-check`
to compare the target build with the tracked reviewer copy.

`just paper-build` only builds under `target/papers/`; `just paper-refresh`
publishes the checked reviewer PDF. `just paper-clean` removes generated paper
build output. `just tectonic` exposes the shared native-library discovery CLI.

Tracked paper files:

- `*.bib` BibTeX sources for paper-local references
- `*.tex` source files
- `*.pdf` reading copies intended for advisors/reviewers
- `../docs/assets/validation/*.png` figures shared with validation documentation

Untracked paper files:

- Tectonic/TeX auxiliary files such as `*.aux`, `*.bbl`, `*.blg`, `*.fls`,
  `*.log`, `*.out`, `*.synctex.gz`, and `*.toc`
- temporary build output under `target/papers/`

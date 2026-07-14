# Papers

This directory contains publication-facing prose for `delaunay`.

The repository keeps three validation artifacts with distinct ownership:

- `docs/validation.md` is the canonical developer-facing validation contract.
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

That recipe lints the source, rebuilds the PDF, and runs the uv-managed
`paper-pdf-check` sanity check against the reviewer copy.

Tracked paper files:

- `*.bib` BibTeX sources for paper-local references
- `*.tex` source files
- `*.pdf` reading copies intended for advisors/reviewers
- `../docs/assets/validation/*.png` figures shared with validation documentation

Untracked paper files:

- Tectonic/TeX auxiliary files such as `*.aux`, `*.bbl`, `*.blg`, `*.fls`,
  `*.log`, `*.out`, `*.synctex.gz`, and `*.toc`
- temporary build output under `target/papers/`

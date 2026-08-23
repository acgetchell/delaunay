# Performance Evidence Data

This directory owns the exact CSV and provenance JSON pair for every
artifact-backed report promoted to `docs/PERFORMANCE.md`. The filenames use
`vCURRENT-vs-vBASELINE.csv` and `vCURRENT-vs-vBASELINE.provenance.json`.

Only `just performance-release` and `just performance-doc` write these files.
The adjacent Markdown report is a generated view of the validated pair; do not
edit timing values or provenance by hand. The current tracked performance report
predates this retention contract, so this directory contains no historical pair
for it.

CSV is the canonical interchange artifact. A notebook may materialize Parquet
under disposable `target/` output for larger dataframe analysis, but that cache
must be reproducible from a validated CSV/provenance pair and is never a
promotion input.

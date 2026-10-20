# Performance Evidence Data

This directory retains the exact shared Criterion comparison JSON and evidence
envelope for every artifact-backed report promoted to
[`docs/performance.md`](../../../performance.md). The filenames use
`vCURRENT-vs-vBASELINE.comparison.json` and
`vCURRENT-vs-vBASELINE.evidence.json`.

Only `just performance-release` and `just performance-doc` write these files.
The promoted Markdown report is a generated view of the validated pair; do not
edit timing values or provenance by hand. The current tracked performance report
predates this retention contract, so this directory contains no historical pair
for it.

The shared comparison JSON and its evidence envelope are the canonical
promotion inputs. The envelope binds the comparison payload by digest and
records Delaunay's workload and comparability policy. Notebooks may derive
Parquet caches under disposable `target/` output, but those caches must be
reproducible from the validated JSON pair and are never promotion inputs.

See the [benchmark workflow](../../../dev/commands.md#benchmark-profiles)
for measurement, validation, and promotion commands.

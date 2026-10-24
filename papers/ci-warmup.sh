#!/usr/bin/env bash
# Retain bounded bundle-download retries for deliberately requested paper builds.
set -euo pipefail

warmup_dir="$RUNNER_TEMP/tectonic-warmup"
warmup_source="$warmup_dir/warmup.tex"
mkdir -p "$warmup_dir"
printf '%s\n' \
	'\documentclass{article}' \
	'\begin{document}' \
	'Tectonic bundle cache warmup.' \
	'\end{document}' >"$warmup_source"

for attempt in 1 2 3 4; do
	if uv run --locked --group dev --group papers research-repo-tools --root . --config papers/tooling.toml \
		toolchain run -- tectonic --outdir "$warmup_dir" "$warmup_source"; then
		exit 0
	fi
	if [[ "$attempt" -eq 4 ]]; then
		echo "::error::Unable to warm the Tectonic bundle cache after $attempt attempts."
		exit 1
	fi
	delay=$((attempt * 15))
	echo "::warning::Tectonic bundle download failed; retrying in ${delay}s."
	sleep "$delay"
done

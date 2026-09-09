#!/usr/bin/env bash
# Guard the draft-run-upload-publish sequence. All remote writes happen in publish.
set -euo pipefail

mode="${1:?expected preflight or publish}"
tag="${RELEASE_TAG:?missing RELEASE_TAG}"
repo="${GITHUB_REPOSITORY:?missing GITHUB_REPOSITORY}"
asset="delaunay-${tag}-criterion-baseline.tar.gz"

fail() {
	echo "Release benchmarks: $*" >&2
	exit 1
}

[[ "$mode" == preflight || "$mode" == publish ]] || fail "unknown mode: $mode"
[[ "$tag" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || fail "expected a stable vX.Y.Z tag"

# Resolve the remote tag, including annotated tags, without fetching or moving refs.
resolve_commit() {
	local object depth=0
	object="$(gh api "repos/$repo/git/ref/tags/$tag" --jq '.object')"
	while [[ "$(jq -r '.type' <<<"$object")" == tag ]]; do
		((depth += 1))
		((depth <= 10)) || fail "tag nesting exceeds the supported limit"
		object="$(gh api "repos/$repo/git/tags/$(jq -r '.sha' <<<"$object")" --jq '.object')"
	done
	[[ "$(jq -r '.type' <<<"$object")" == commit ]] || fail "tag does not resolve to a commit"
	jq -er '.sha | select(test("^[0-9a-f]{40}$"))' <<<"$object"
}

check_draft() {
	# The by-tag endpoint only promises published releases. Authenticated list
	# pagination includes drafts and also lets us reject ambiguous tag matches.
	releases="$(gh api "repos/$repo/releases?per_page=100" --paginate --slurp)"
	release="$(jq -ce --arg tag "$tag" '
        [ .[][] | select(.tag_name == $tag) ]
        | if length == 1 then .[0] else error("expected exactly one release for tag") end
    ' <<<"$releases")"
	jq -e --arg tag "$tag" '
        .tag_name == $tag and .draft == true and .prerelease == false
        and .immutable == false and .published_at == null
        and (.id | type == "number" and . > 0)
    ' <<<"$release" >/dev/null || fail "target must be an unpublished, mutable stable draft"
	release_id="$(jq -r '.id' <<<"$release")"
	if [[ -n "${EXPECTED_RELEASE_ID:-}" ]]; then
		[[ "$release_id" == "$EXPECTED_RELEASE_ID" ]] || fail "draft release identity changed"
	fi
	commit="$(resolve_commit)"
	local_commit="$(git --no-pager rev-parse --verify "refs/tags/$tag^{commit}")"
	[[ "$commit" == "$local_commit" ]] || fail "remote tag differs from the checked-out tag"
	if [[ -n "${EXPECTED_COMMIT:-}" ]]; then
		[[ "$commit" == "$EXPECTED_COMMIT" ]] || fail "tag moved during benchmarking"
	fi
}

check_draft
# Never replace evidence implicitly, even on a mutable draft. A failed run that
# uploaded successfully requires inspection before the maintainer removes it.
jq -e --arg asset "$asset" 'all(.assets[]; .name != $asset)' <<<"$release" >/dev/null ||
	fail "baseline asset already exists; inspect the draft asset before retrying"

if [[ "$mode" == preflight ]]; then
	{
		echo "release_id=$release_id"
		echo "commit=$commit"
	} >>"${GITHUB_OUTPUT:?missing GITHUB_OUTPUT}"
	exit 0
fi

[[ -n "${EXPECTED_RELEASE_ID:-}" && -n "${EXPECTED_COMMIT:-}" ]] || fail "publish requires preflight identity"
[[ "$(git --no-pager rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "HEAD differs from the benchmarked tag"
[[ -s "$asset" ]] || fail "missing or empty baseline archive: $asset"
digest="sha256:$(sha256sum "$asset" | cut -d ' ' -f 1)"
gh release upload "$tag" "$asset" --repo "$repo"

# Recheck draft identity and tag after upload, then verify GitHub stored these
# exact archive bytes before making the release public.
check_draft
jq -e --arg asset "$asset" --arg digest "$digest" '
    [.assets[] | select(.name == $asset and .state == "uploaded" and .size > 0 and .digest == $digest)]
    | length == 1
' <<<"$release" >/dev/null || fail "uploaded baseline is missing, incomplete, or has a different digest"
gh api "repos/$repo/releases/$release_id" --method PATCH -F draft=false >/dev/null

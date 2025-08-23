# Releasing delaunay

This guide documents the exact commands for performing a clean release using a
dedicated release PR, followed by tagging, publishing to crates.io, and
creating a GitHub release.

Applies to versions vX.Y.Z (example uses v0.3.5). We prefer updating
documentation before publishing to crates.io.

---

## Conventions and environment

Set these variables to avoid repeating the version string:

```bash
# tag has the leading v, version does not
TAG=v0.3.5
VERSION=${TAG#v}
```

Verify your git remotes:

```bash
git remote -v
```

Ensure your local main is up to date before beginning:

```bash
git checkout main
git pull --ff-only
```

---

## Phase 1: Create a clean release PR

This PR should ONLY include: version bumps, changelog updates, and documentation updates. All code changes should already be on main.

1. Create the release branch

```bash
git checkout -b release/$TAG
```

2. Bump versions

Preferred (if cargo-edit is installed):

```bash
# Bump package version in Cargo.toml
cargo set-version $VERSION
```

Alternative (sed fallback):

```bash
# Update version = "..." in the [package] section of Cargo.toml
# Note: Review the change afterward to ensure only the intended line was modified
sed -i.bak -E "s/^version = \"[0-9]+\.[0-9]+\.[0-9]+\"/version = \"$VERSION\"/" Cargo.toml && rm Cargo.toml.bak
```

Update references in documentation (search, then manually edit as needed):

```bash
# List occurrences of version-like strings to review
rg -n "\bv?[0-9]+\.[0-9]+\.[0-9]+\b" README.md docs/ || true
```

3. Generate changelog using a temporary local tag (DO NOT PUSH this tag)

```bash
# Create a temporary annotated tag locally to enable changelog generation
# Do not push this tag; it will be recreated later after merge
git tag -a "$TAG" -m "delaunay $TAG"

# Generate changelog using commit dates
./scripts/generate_changelog.sh
```

4. Stage and commit release artifacts

```bash
git add Cargo.toml CHANGELOG.md docs/

git commit -m "chore(release): release $TAG

- Bump version to $TAG
- Update changelog with latest changes
- Update documentation for release"
```

5. Push the branch and open a PR

```bash
git push -u origin "release/$TAG"
```

PR metadata:

- Title: chore(release): release $TAG
- Description: Clean release PR with version bump, changelog, and documentation updates. No code changes.

Note: Do NOT push the temporary tag created in step 3.

---

## Phase 2: After the PR is merged into main

1. Sync your local main to the merge commit

```bash
git checkout main
git pull --ff-only
```

2. Recreate the final annotated tag using the changelog content

```bash
# Remove the temporary local tag if it exists
git tag -d "$TAG" 2>/dev/null || true

# Create the final annotated tag with the changelog section as the tag message
./scripts/tag-from-changelog.sh "$TAG" --force
```

3. (Optional) Verify tag message content

```bash
git tag -l --format='%(contents)' "$TAG"
```

4. Push the tag

```bash
git push origin "$TAG"
```

5. Create the GitHub release with notes from the tag annotation

```bash
# Requires GitHub CLI (gh) and authenticated session
gh release create "$TAG" --notes-from-tag
```

6. Publish to crates.io

```bash
# Sanity check before publishing
cargo publish --dry-run

# Publish the crate (ensure docs are already updated on main via the PR)
cargo publish
```

---

## Notes and tips

- Never push the temporary tag created for changelog generation; only push the final tag after the PR is merged.
- Keep the release PR strictly to version + changelog + documentation to maintain a clean history.
- If multiple crates or files reference the version, confirm all of them are updated consistently.
- For future convenience, parts of this document can be automated into a release script.

# Releasing delaunay

This guide documents the exact commands for performing a clean release using a
dedicated release PR, followed by tagging, publishing to crates.io, and
creating a GitHub release.

Applies to versions vX.Y.Z. Prefer updating documentation before publishing
to crates.io.

---

## Conventions and environment

Set these variables to avoid repeating the version string:

```bash
# tag has the leading v, version does not
TAG=vX.Y.Z
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

This PR should primarily include: version bumps, changelog updates, and documentation updates. All major code changes should already be on main.

**Exception:** Small, critical fixes discovered during the release process (e.g.,
documentation errors, script bugs, formatting issues) may be included but should
be minimal and release-critical only.

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

# Generate changelog using the Python-based tool (improved error handling)
just changelog
# uv run changelog-utils generate
```

4. Generate performance results with fresh benchmark data

```bash
# Run benchmarks and generate performance summary automatically
# This takes ~30-45 minutes and provides official performance data for the release
echo "Generating performance results for release $TAG..."
echo "This will run fresh benchmarks and update benches/PERFORMANCE_RESULTS.md"

uv run benchmark-utils generate-summary --run-benchmarks

echo "Performance results generated. Review benches/PERFORMANCE_RESULTS.md for accuracy."
```

5. Stage and commit release artifacts

```bash
git add Cargo.toml Cargo.lock CHANGELOG.md docs/ benches/PERFORMANCE_RESULTS.md

git commit -m "chore(release): release $TAG

- Bump version to $TAG
- Update changelog with latest changes
- Update documentation for release
- Add performance results for $TAG"
```

6. Push the branch and open a PR

```bash
git push -u origin "release/$TAG"
```

PR metadata:

- Title: chore(release): release $TAG
- Description: Clean release PR with version bump, changelog, and documentation updates. No code changes.

Note: Do NOT push the temporary tag created in step 3.

### Handling fixes discovered during release process

If you discover issues (bugs, formatting problems, etc.) after creating the changelog:

1. **For critical fixes that must be in this release:**

   ```bash
   # Make your fixes
   # Run code quality tools
   # Commit the fixes
   git add .
   git commit -m "fix: [description of fix]"
   
   # Delete the temporary tag and regenerate changelog
   git tag -d "$TAG"
   git tag -a "$TAG" -m "delaunay $TAG"
   just changelog
   # uv run changelog-utils generate
   
   # Commit updated changelog
   git add CHANGELOG.md
   git commit -m "docs: update changelog with release fixes"
   ```

2. **For non-critical fixes:**
   - Document them as known issues in the release notes
   - Include them in the next release
   - This avoids the changelog regeneration loop

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
# Using the new Python-based tagging tool
just changelog-tag "$TAG"
# uv run changelog-utils tag "$TAG" --force
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

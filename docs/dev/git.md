# Git And GitHub Workflow

Repository Git, GitHub CLI, issue-dependency, branch-name, and commit-message
rules for agents and maintainers.

---

## Contents

- [Git Operations](#git-operations)
- [GitHub CLI](#github-cli)
- [Issue Dependencies](#issue-dependencies)
- [Commit Message Generation](#commit-message-generation)
- [Changelog-Aware Body Text](#changelog-aware-body-text)
- [Breaking Changes](#breaking-changes)

---

## Git Operations

- **NEVER** run `git commit`, `git push`, `git tag`, or any git command that
  modifies version-control state unless the maintainer explicitly asks for that
  operation.
- **ALLOWED**: read-only Git commands such as `git --no-pager status`,
  `git --no-pager diff`, `git --no-pager log`, `git --no-pager show`, and
  `git --no-pager blame`.
- **ALWAYS** use `git --no-pager` when reading Git output.
- Suggest Git commands that mutate version-control state for the maintainer to
  run manually unless they explicitly request that the agent run them.

When suggesting branch names, prefer `{type}/{issue}-descriptor-or-two`, for
example:

```text
fix/307-oriented-flips
perf/315-bench-profile
doc/329-branch-guidance
```

If an environment requires an owner/tool prefix, keep this structure after the
prefix:

```text
codex/fix/307-oriented-flips
```

Typical types are `fix`, `feat`, `perf`, `doc`, `test`, `refactor`, `ci`,
`build`, `chore`, and `style`.

## GitHub CLI

When using the `gh` CLI to view issues, PRs, or other GitHub objects, use
`--json` with `| cat` to avoid pager and scope errors:

```bash
gh issue view 212 --repo acgetchell/delaunay --json title,body | cat
```

To extract fields cleanly, combine `--json` with `--jq`:

```bash
gh issue view 212 --repo acgetchell/delaunay --json title,body --jq '.title + "\n" + .body' | cat
```

Avoid plain `gh issue view N`; it may fail with `read:project` scope errors or
open a pager.

When updating issues, use explicit `comment` or `edit` commands. For arbitrary
Markdown containing backticks, quotes, or special characters, prefer
`--body-file -` with a heredoc:

```bash
gh issue comment 242 --repo acgetchell/delaunay --body-file - <<'EOF'
## Heading

Body with `backticks`, **bold**, and apostrophes that's safe.
EOF
```

For simple text only, single-quoted `--body` is fine:

```bash
gh issue comment 242 --repo acgetchell/delaunay --body 'Simple update text'
```

## Issue Dependencies

To manage GitHub issue dependencies (`Blocks` / `Is blocked by`), use the
GitHub REST API via `gh api`. The endpoint requires the internal issue ID, not
the issue number.

Get an issue's internal ID:

```bash
gh api repos/acgetchell/delaunay/issues/233 --jq '.id'
```

Add a `blocked by` dependency, for example #254 is blocked by #233:

```bash
gh api repos/acgetchell/delaunay/issues/254/dependencies/blocked_by \
  -X POST -F issue_id=<BLOCKING_ISSUE_ID>
```

List existing blocked-by dependencies:

```bash
gh api repos/acgetchell/delaunay/issues/254/dependencies/blocked_by \
  --jq '[.[].number]' | cat
```

Use `-F`, not `-f`, for `issue_id` so the value is sent as an integer. The API
returns HTTP 422 when the dependency already exists.

## Commit Message Generation

When generating commit messages:

1. Run `git --no-pager diff --cached --stat`.
2. Use conventional commits: `<type>: <summary>`.
3. Use one of: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`,
   `style`, `ci`, or `build`.
4. Include a bullet-point body describing key changes.
5. Present the message inside a code block so the maintainer can commit
   manually.

## Changelog-Aware Body Text

Commit bodies appear verbatim in `CHANGELOG.md` through git-cliff's template.
Write them as clean, readable prose:

- Keep the subject line concise; it becomes the changelog entry.
- The type determines the changelog section (`feat` -> Added, `fix` -> Fixed,
  `refactor`/`test`/`style` -> Changed, `perf` -> Performance,
  `docs` -> Documentation, `build`/`chore`/`ci` -> Maintenance).
- Include PR references as `(#N)` in the subject so cliff auto-links them.
- Avoid headings `#` through `###` in the body because they conflict with
  changelog structure. Use `####` only when a heading is truly needed.
- Body text should be plain prose or simple lists.

## Breaking Changes

Breaking changes must use one of these conventional-commit markers so
git-cliff can detect them and generate the breaking-changes section:

- Bang notation: `feat!: remove deprecated API`
- Footer trailer: `BREAKING CHANGE: <description>`

Examples include removing or renaming public API items, changing default
behavior, bumping MSRV, or altering serialization formats.

"""Prepare a stable draft after exact protected-main Cargo and SARIF evidence.

This workflow orchestration never measures, publishes a release, or obtains
registry credentials. A retry can reuse only the same mutable draft and notes.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

from research_repo_tools import changelog, config
from research_repo_tools.measurement import resolve_revision
from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics, run_command
from research_repo_tools.release_assets import preflight_release_target, require_release_checkout, revalidate_release_target
from research_repo_tools.release_evidence import verify_release_evidence
from research_repo_tools.release_interfaces import load_evidence_policy
from research_repo_tools.release_publishing import check_metadata, require_checks
from research_repo_tools.release_validation import api, parse_release_json, redact_release_diagnostic

REPOSITORY = "acgetchell/delaunay"


def require_tag_source(root: Path, tag: str) -> str:
    """Bind the tag event, clean checkout, remote tag and protected main ancestry."""
    if re.fullmatch(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", tag) is None:
        msg = "draft preparation requires a canonical stable vX.Y.Z tag"
        raise ValueError(msg)
    commit = resolve_revision(root, tag)
    if (os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_EVENT_NAME"), os.environ.get("GITHUB_REF"), os.environ.get("GITHUB_SHA")) != (
        REPOSITORY,
        "push",
        f"refs/tags/{tag}",
        commit,
    ):
        msg = "draft preparation requires the exact tag-push ref and source SHA"
        raise ValueError(msg)
    require_release_checkout(root, commit)
    repository = api("", repository=REPOSITORY)
    branch = api("branches/main", repository=REPOSITORY)
    comparison = api(f"compare/{commit}...main", repository=REPOSITORY)
    if repository.get("default_branch") != "main" or branch.get("protected") is not True or comparison.get("status") not in {"ahead", "identical"}:
        msg = "release commit must belong to protected default main"
        raise ValueError(msg)
    refs = run_command("git", ["ls-remote", f"https://github.com/{REPOSITORY}.git", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"], cwd=root).stdout
    observed = dict(line.split()[::-1] for line in refs.splitlines())
    if observed.get(f"refs/tags/{tag}^{{}}", observed.get(f"refs/tags/{tag}")) != commit:
        msg = "remote tag differs from the captured source SHA"
        raise ValueError(msg)
    return commit


def prepare(root: Path, tag: str) -> int:
    """Verify evidence and create or inspect an identical draft without replacing assets."""
    commit = require_tag_source(root, tag)
    settings = config.load(root=root)
    check_metadata(settings, tag)
    policy = load_evidence_policy(root / "tooling/release-evidence.toml")
    read = partial(api, repository=REPOSITORY)
    selection = verify_release_evidence(policy, tag.removeprefix("v"), commit, read)
    if settings.publishing is None:
        msg = "draft preparation requires the publishing check policy"
        raise ValueError(msg)
    checks = parse_release_json(
        run_command("gh", ["api", "--paginate", "--slurp", f"repos/{REPOSITORY}/commits/{commit}/check-runs?filter=latest&per_page=100"], cwd=root).stdout
    )
    require_checks(checks, commit, settings.publishing.required_checks)
    notes = changelog.notes(settings, tag)[0]
    pages = parse_release_json(run_command("gh", ["api", "--paginate", "--slurp", f"repos/{REPOSITORY}/releases?per_page=100"], cwd=root).stdout)
    if not isinstance(pages, list) or not pages or any(not isinstance(page, list) or any(not isinstance(item, dict) for item in page) for page in pages):
        msg = "invalid release inventory"
        raise ValueError(msg)
    matches = [release for page in pages for release in page if release.get("tag_name") == tag]
    if len(matches) > 1:
        msg = "ambiguous release identity"
        raise ValueError(msg)
    # Recheck the frozen selection and source immediately before any write.
    verify_release_evidence(policy, tag.removeprefix("v"), commit, read, expected=selection)
    if require_tag_source(root, tag) != commit:
        msg = "tag source changed during preparation"
        raise ValueError(msg)
    if not matches:
        with tempfile.TemporaryDirectory(prefix="delaunay-draft-notes-") as temporary:
            path = Path(temporary) / "notes.md"
            path.write_text(notes, encoding="utf-8", newline="\n")
            run_command(
                "gh",
                ["release", "create", tag, "--repo", REPOSITORY, "--verify-tag", "--target", commit, "--draft", "--title", tag, "--notes-file", str(path)],
                cwd=root,
            )
    target = preflight_release_target(root, REPOSITORY, tag, commit, expected_title=tag, checkout=root)
    release = api(f"releases/{target.release_id}", repository=REPOSITORY)
    if matches and target.release_id != matches[0].get("id"):
        msg = "draft release identity changed during preparation"
        raise ValueError(msg)
    body = release.get("body")
    if not isinstance(body, str) or body.replace("\r\n", "\n").rstrip("\n") != notes.rstrip("\n"):
        msg = "existing draft notes conflict with generated release notes"
        raise ValueError(msg)
    revalidate_release_target(root, target, checkout=root)
    return target.release_id


def main() -> int:
    """Fail closed on unavailable evidence or a conflicting/published release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    args = parser.parse_args()
    try:
        identifier = prepare(Path.cwd(), args.tag)
        print(f"Verified mutable draft {identifier} for {args.tag}; review it before publication.")
    except (ExecutableNotFoundError, OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        print(f"Draft preparation failed: {redact_release_diagnostic(format_exception_diagnostics(error))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

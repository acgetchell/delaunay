"""Exercise the actual approval step with local transports and runner shell flags."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from subprocess_utils import run_safe_command

if TYPE_CHECKING:
    import subprocess

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "dependabot-auto-merge.yml"
HEAD_SHA = "a" * 40
APPROVAL = {"user": {"login": "coderabbitai[bot]"}, "commit_id": HEAD_SHA, "state": "APPROVED"}


@pytest.fixture(scope="module")
def approval_script() -> str:
    """Load the production step and require GitHub's explicit Bash semantics."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["review-and-enable-auto-merge"]["steps"]
    step = next(step for step in steps if step.get("id") == "coderabbit-approval")
    assert step["shell"] == "bash"
    script = step["run"]
    assert isinstance(script, str)
    return script


@pytest.fixture
def review_env(tmp_path: Path) -> dict[str, str]:
    """Replace GitHub access and polling delays with bounded, observable fakes."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_ROOT/calls"
case "$*" in
    *'/reviews?per_page=100'*)
        cat "$FAKE_ROOT/reviews.json"
        exit "$REVIEWS_STATUS" ;;
    'api repos/owner/repo/pulls/123 --jq .head.sha')
        printf '%s\\n' "$PR_HEAD_SHA" ;;
    *) exit 99 ;;
esac
""",
        encoding="utf-8",
    )
    sleep = fake_bin / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 91\n", encoding="utf-8")
    gh.chmod(0o755)
    sleep.chmod(0o755)
    (tmp_path / "reviews.json").write_text(json.dumps([APPROVAL]), encoding="utf-8")
    return {
        **{key: value for key, value in os.environ.items() if key not in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"}},
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_ROOT": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "PR_HEAD_SHA": HEAD_SHA,
        "PR_NUMBER": "123",
        "REPOSITORY": "owner/repo",
        "REVIEWS_STATUS": "0",
    }


def run_approval(tmp_path: Path, env: dict[str, str], script: str) -> subprocess.CompletedProcess[str]:
    """Model shell: bash on a GitHub runner without executing any live Actions."""
    return run_safe_command("bash", ["--noprofile", "--norc", "-eo", "pipefail", "-c", script], cwd=tmp_path, env=env, check=False, timeout=10)


def test_successful_approval_binds_the_current_head(tmp_path: Path, review_env: dict[str, str], approval_script: str) -> None:
    result = run_approval(tmp_path, review_env, approval_script)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output").read_text(encoding="utf-8") == f"approved_head_sha={HEAD_SHA}\n"


def test_partial_approval_followed_by_api_failure_never_authorizes_merge(tmp_path: Path, review_env: dict[str, str], approval_script: str) -> None:
    review_env["REVIEWS_STATUS"] = "57"
    result = run_approval(tmp_path, review_env, approval_script)
    assert result.returncode == 57
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("response", ["", "[]", "[null]"])
def test_empty_or_missing_review_state_never_authorizes_merge(tmp_path: Path, review_env: dict[str, str], approval_script: str, response: str) -> None:
    (tmp_path / "reviews.json").write_text(response, encoding="utf-8")
    result = run_approval(tmp_path, review_env, approval_script)
    assert result.returncode == 91
    assert not (tmp_path / "output").exists()


def test_malformed_reviews_abort_before_polling(tmp_path: Path, review_env: dict[str, str], approval_script: str) -> None:
    (tmp_path / "reviews.json").write_text("[", encoding="utf-8")
    result = run_approval(tmp_path, review_env, approval_script)
    assert result.returncode not in {0, 91}
    assert not (tmp_path / "output").exists()


def test_later_review_page_overrides_earlier_approval(tmp_path: Path, review_env: dict[str, str], approval_script: str) -> None:
    pages = json.dumps([APPROVAL]) + "\n" + json.dumps([{**APPROVAL, "state": "CHANGES_REQUESTED"}])
    (tmp_path / "reviews.json").write_text(pages, encoding="utf-8")
    result = run_approval(tmp_path, review_env, approval_script)
    assert result.returncode == 91
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("review", [{**APPROVAL, "commit_id": "b" * 40}, {**APPROVAL, "user": {"login": "someone-else"}}])
def test_unrelated_approval_never_authorizes_merge(tmp_path: Path, review_env: dict[str, str], approval_script: str, review: dict[str, object]) -> None:
    (tmp_path / "reviews.json").write_text(json.dumps([review]), encoding="utf-8")
    result = run_approval(tmp_path, review_env, approval_script)
    assert result.returncode == 91
    assert not (tmp_path / "output").exists()

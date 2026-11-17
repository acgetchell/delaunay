"""Exercise the actual tag-planning shell against the installed release CLI."""

import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from research_repo_tools.process import run_command

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/prepare-release.yml"
REPOSITORY = "acgetchell/delaunay"
TAG = "v0.8.4"
SHA = "a" * 40
PLANNER = "Verify tagged source and freeze native evidence and notes"


@pytest.mark.parametrize("external_policy", [False, True])
def test_original_source_shell_uses_checkout_relative_policy(tmp_path: Path, *, external_policy: bool) -> None:
    """The installed planner accepts the consumer shell and guards policy containment."""
    checkout, runner = tmp_path / "tag source café", tmp_path / "runner temp café"
    (checkout / "tooling").mkdir(parents=True)
    runner.mkdir()
    for name in ("Cargo.toml", "pyproject.toml", ".python-version", "tooling/release-evidence.toml"):
        shutil.copyfile(ROOT / name, checkout / name)
    policy = checkout / "tooling/release-evidence.toml"
    (runner / "evidence.toml").write_bytes(policy.read_bytes())
    output = runner / "github-output"
    output.touch()
    fake_provider = runner / "provider.py"
    fake_provider.write_text(
        """
import sys
from research_repo_tools import draft_workflow, release_interfaces
from research_repo_tools.cli import main

# Only provider/source observations are synthetic. The installed CLI still
# resolves paths, loads the real consumer policy, and freezes the handoff.
release_interfaces._checkout = lambda *args, **kwargs: None
draft_workflow.check_release_source = lambda *args, **kwargs: None
draft_workflow.notes = lambda *args: ("Original notes café\u005cn", None, None)
draft_workflow.verify_release_evidence = lambda *args, **kwargs: b'{"original": "native selection"}'
sys.exit(main(sys.argv[1:]))
""",
        encoding="utf-8",
    )
    verify = yaml.safe_load(WORKFLOW.read_bytes())["jobs"]["verify"]["steps"]
    script = next(step["run"] for step in verify if step.get("name") == PLANNER)
    if external_policy:
        script = script.replace("tooling/release-evidence.toml", '"$RUNNER_TEMP/evidence.toml"')
    bootstrap = f"""uvx() {{
    [[ "$*" == '--no-config --isolated --no-cache --python 3.15 --from research-repo-tools==0.2.9 python -I -m research_repo_tools '* ]] || return 91
    shift 11
    {shlex.quote(Path(sys.executable).as_posix())} -I {shlex.quote(fake_provider.as_posix())} "$@"
}}
git() {{
    case "$*" in
        "--no-pager rev-parse "*) printf '%s\\n' "$GITHUB_SHA" ;;
        "--no-pager merge-base --is-ancestor $GITHUB_SHA origin/main") ;;
        *) return 92 ;;
    esac
}}
gh() {{
    [[ "$*" == "api repos/$GITHUB_REPOSITORY --jq .default_branch" ]] || return 96
    printf 'main\\n'
}}
cargo() {{ exit 93; }}
just() {{ exit 94; }}
uv() {{ exit 95; }}
"""
    environment = os.environ | {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": f"refs/tags/{TAG}",
        "GITHUB_SHA": SHA,
        "RELEASE_TAG": TAG,
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_WORKSPACE": checkout.as_posix(),
        "RUNNER_TEMP": runner.as_posix(),
        "GITHUB_OUTPUT": output.as_posix(),
    }
    result = run_command("bash", ["-euc", bootstrap + script], cwd=checkout, env=environment, check=False)
    handoff = runner / "handoff/handoff.json"
    if external_policy:
        assert result.returncode == 1
        assert "relative" in result.stderr
        assert not handoff.exists()
        assert output.read_bytes() == b""
    else:
        assert result.returncode == 0, result.stderr
        data = json.loads(handoff.read_bytes())
        assert data["source"] == {"ref": f"refs/tags/{TAG}", "commit": SHA}
        assert data["evidence_policy"].encode() == policy.read_bytes()
        assert data["selection"] == {"original": "native selection"}
        assert data["notes"] == "Original notes café\n"
        assert output.read_text(encoding="utf-8") == f"commit={SHA}\n"


def test_draft_job_retains_receipt_and_full_recovery_state() -> None:
    """The shared writer has isolated draft authority and retains the original state."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    assert workflow[True] == {"push": {"tags": ["v*"]}}
    assert workflow["concurrency"]["cancel-in-progress"] is False
    job = workflow["jobs"]["prepare"]
    assert job["needs"] == "verify"
    assert job["permissions"] == {"contents": "write", "actions": "read", "checks": "read"}
    assert "environment" not in job
    assert not any(step.get("uses", "").startswith("actions/checkout@") for step in job["steps"])
    receipt, state = job["steps"][-2:]
    assert receipt["if"] == state["if"] == "always()"
    assert "draft-state/draft-receipt.json" in receipt["with"]["path"]
    assert "draft-state/target.json" in receipt["with"]["path"]
    assert state["with"]["path"].endswith("/draft-state/*")

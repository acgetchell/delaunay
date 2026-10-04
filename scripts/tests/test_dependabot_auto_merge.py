"""Check the consumer boundary of the shared Dependabot approval workflow."""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/dependabot-auto-merge.yml"
CONFIG = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
JOB = CONFIG["jobs"]["approve-and-enable-auto-merge"]
POLICY = json.loads(JOB["with"]["policy"])


def test_approval_runs_from_trusted_base_without_consumer_code() -> None:
    """Privileged automation can only call the reviewed shared workflow."""
    assert set(CONFIG["on"]) == {"pull_request_target"}
    event = CONFIG["on"]["pull_request_target"]
    assert event["branches"] == ["main"]
    assert set(event["types"]) == {"opened", "reopened", "ready_for_review", "synchronize"}
    assert CONFIG["permissions"] == {}
    assert set(CONFIG["jobs"]) == {"approve-and-enable-auto-merge"}
    assert set(JOB) == {"permissions", "uses", "with"}
    assert JOB["permissions"] == {"contents": "write", "pull-requests": "write"}
    reference, revision = JOB["uses"].rsplit("@", 1)
    assert reference == "acgetchell/research-repo-tools/.github/workflows/dependabot-approve.yml"
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert set(JOB["with"]) == {"repository", "policy"}
    assert JOB["with"]["repository"] == "acgetchell/delaunay"


def test_dependency_policy_covers_declared_ecosystems_and_resolution_roots() -> None:
    """Dependency approval includes the isolated checkpoint Cargo root."""
    dependabot = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))
    ecosystems = {update["package-ecosystem"].replace("-", "_") for update in dependabot["updates"]}
    assert set(POLICY) == ecosystems
    assert set(POLICY["cargo"]["files"]) == {
        "Cargo.toml",
        "Cargo.lock",
        "tests/fixtures/checkpoint_no_float_roundtrip/Cargo.toml",
        "tests/fixtures/checkpoint_no_float_roundtrip/Cargo.lock",
    }
    assert set(POLICY["uv"]["files"]) == {"pyproject.toml", "uv.lock"}


def test_actions_policy_covers_consumer_workflows_and_composite_actions() -> None:
    """Every Actions dependency is covered by an exact existing file path."""
    paths = {
        path.relative_to(ROOT).as_posix()
        for directory in (ROOT / ".github/workflows", ROOT / ".github/actions")
        for path in directory.rglob("*")
        if path.suffix in {".yml", ".yaml"} and (directory.name == "workflows" or path.stem == "action")
    }
    assert set(POLICY["github_actions"]["files"]) == paths
    for ecosystem in POLICY.values():
        files = ecosystem["files"]
        assert len(files) == len(set(files))
        assert all((ROOT / path).is_file() for path in files)

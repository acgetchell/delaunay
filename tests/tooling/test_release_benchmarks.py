"""Check the active benchmark-attachment workflow boundaries."""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/release-benchmarks.yml"


def test_release_workflow_separates_verification_from_writers() -> None:
    """Only the read-only verifier executes checkout code; writers use pinned RRT."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    assert set(workflow["jobs"]) == {"fetch", "verify", "attach"}
    assert set(workflow[True]["workflow_dispatch"]["inputs"]) == {"tag", "candidate"}
    jobs = workflow["jobs"]
    for name in ("fetch", "attach"):
        job = jobs[name]
        assert job["permissions"]["contents"] == "write"
        assert not any(step.get("uses", "").startswith("actions/checkout@") for step in job["steps"])
        rendered = yaml.safe_dump(job)
        assert "python -I" in rendered
        assert "research-repo-tools==0.2.9" in rendered
        assert "setup-just" not in rendered
        assert "toolchain run" not in rendered
    assert jobs["verify"]["permissions"] == {"contents": "read", "checks": "read", "actions": "read"}
    assert jobs["verify"]["steps"][0]["with"] == {
        "ref": "${{ github.sha }}",
        "fetch-depth": 0,
        "persist-credentials": False,
    }
    assert jobs["attach"]["needs"] == "verify"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert "performance-release" not in yaml.safe_dump(workflow)
    assert "cargo bench" not in yaml.safe_dump(workflow)


def test_release_workflow_binds_both_transport_attempts() -> None:
    """The original dispatch and both immutable artifact identities cross job boundaries."""
    jobs = yaml.safe_load(WORKFLOW.read_bytes())["jobs"]
    verify = yaml.safe_dump(jobs["verify"])
    attach = yaml.safe_dump(jobs["attach"])
    assert "needs.fetch.outputs.artifact-id" in verify
    assert "needs.fetch.outputs.artifact-digest" in verify
    assert "needs.fetch.outputs.validated-attempt" in verify
    assert "needs.verify.outputs.artifact-id" in attach
    assert "needs.verify.outputs.artifact-digest" in attach
    assert "needs.verify.outputs.validated-attempt" in attach
    assert "inputs.candidate" in attach
    assert "github.actor == 'acgetchell'" in jobs["fetch"]["if"]
    assert "github.triggering_actor == 'acgetchell'" in jobs["fetch"]["if"]

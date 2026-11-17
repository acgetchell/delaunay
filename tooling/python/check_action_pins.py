"""Check the shared Actions policy and Delaunay's single CodeQL action release."""

import sys
from pathlib import Path

from research_repo_tools.action_policy import load_policy
from research_repo_tools.cli import main as shared_main
from research_repo_tools.workflow_uses import references, workflow_paths

CODEQL = {"github/codeql-action/analyze", "github/codeql-action/init", "github/codeql-action/upload-sarif"}
POLICY = ".github/action-updates.toml"


def check_action_pins(root: Path, inputs: list[str]) -> int:
    """Compose the installed checker with the cross-subpath CodeQL rule."""
    status = shared_main(["--root", str(root), "actions", "check", "--policy", POLICY, *inputs])
    if status:
        return status
    policy = load_policy(root / POLICY)
    tags = {policy.targets.get(action) for action in CODEQL}
    if not CODEQL.issubset(policy.consistency) or len(tags) != 1 or tags & {None, "latest"}:
        sys.stderr.write(f"{POLICY}: CodeQL actions must target one explicit action-release tag\n")
        return 1
    pins = {
        use.ref.lower()
        for path in workflow_paths(root, inputs)
        for use in references(path, path.read_bytes().decode("utf-8-sig"))
        if use.action in CODEQL and use.ref is not None
    }
    if len(pins) != 1:
        sys.stderr.write(f"{POLICY}: CodeQL actions must use the same release commit SHA\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(check_action_pins(Path.cwd().resolve(), sys.argv[1:]))

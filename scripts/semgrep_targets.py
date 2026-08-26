#!/usr/bin/env python3
"""List the repository-owned Semgrep scan targets shared by local and CI runs."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from subprocess_utils import run_git_command

TRACKED_TARGET_PATHSPECS = (
    "scripts/tests/*.py",
    "tests/*.rs",
    ":(exclude)tests/semgrep/**",
)


@dataclass(frozen=True, slots=True)
class CliArgs:
    """Parsed target-listing options."""

    null_terminated: bool


def tracked_semgrep_targets(repo_root: Path) -> tuple[str, ...]:
    """Return the root plus tracked test files Semgrep ignores by default."""
    result = run_git_command(["ls-files", "-z", *TRACKED_TARGET_PATHSPECS], cwd=repo_root, timeout=30)
    tracked = tuple(path for path in result.stdout.split("\0") if path)
    fixture_paths = tuple(path for path in tracked if path == "tests/semgrep" or path.startswith("tests/semgrep/"))
    if fixture_paths:
        message = f"Semgrep target enumeration included deliberate fixtures: {', '.join(fixture_paths)}"
        raise RuntimeError(message)
    return (".", *dict.fromkeys(tracked))


def parse_args(argv: list[str]) -> CliArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, suggest_on_error=True, color=False)
    parser.add_argument("-0", "--null", action="store_true", help="terminate targets with NUL bytes")
    args = parser.parse_args(argv)
    return CliArgs(null_terminated=args.null)


def main(argv: list[str] | None = None) -> int:
    """Print the shared target set in shell-readable form."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    targets = tracked_semgrep_targets(Path.cwd())
    separator = "\0" if args.null_terminated else "\n"
    sys.stdout.write(separator.join(targets) + separator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Default prospective changelog dates to the prepared citation metadata.

This adapter matches la-stack's command contract while generation, retained
history, formatting and publication remain in research-repo-tools.
"""

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml
from research_repo_tools.cli import main as tooling_main
from research_repo_tools.process import ExecutableNotFoundError, format_exception_diagnostics
from yaml.nodes import MappingNode, ScalarNode


def citation_date(root: Path, tag: str) -> str:
    """Require one ISO date for the requested citation version, without a clock."""
    document = yaml.compose((root / "CITATION.cff").read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    if not isinstance(document, MappingNode):
        msg = "CITATION.cff must contain a top-level mapping"
        raise TypeError(msg)

    def scalar(field: str) -> str:
        values = [value for key, value in document.value if isinstance(key, ScalarNode) and key.value == field]
        message = f"CITATION.cff must contain exactly one nonempty top-level {field} scalar"
        if len(values) != 1 or not isinstance(values[0], ScalarNode):
            raise ValueError(message)
        value: object = values[0].value
        if not isinstance(value, str) or not value:
            raise ValueError(message)
        return value

    version = scalar("version")
    if tag.removeprefix("v") != version:
        raise ValueError(f"CITATION.cff version {version!r} does not match {tag}; run just update-version first")
    released = scalar("date-released")
    if date.fromisoformat(released).isoformat() != released:
        msg = "CITATION.cff date-released must use YYYY-MM-DD form"
        raise ValueError(msg)
    return released


def main(argv: list[str] | None = None) -> int:
    """Resolve an omitted date and delegate generation or preview to the shared CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument("--date")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        released = args.date or (citation_date(Path.cwd(), args.tag) if args.tag else None)
        command = ["changelog", "generate"]
        if args.tag is not None:
            command.extend(["--tag", args.tag])
        if released is not None:
            command.extend(["--date", released])
        if args.dry_run:
            command.append("--dry-run")
        return tooling_main(command)
    except (ExecutableNotFoundError, OSError, TypeError, ValueError, subprocess.SubprocessError, yaml.YAMLError) as error:
        print(f"Release changelog rejected: {format_exception_diagnostics(error, single_line=True)}", file=sys.stderr)
        print("Run just update-version first to record the matching citation version and release date, or provide an explicit date.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

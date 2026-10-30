"""Native subprocess fixtures preserve argument vectors, bytes and exit status."""

import json
import os
from typing import TYPE_CHECKING

import pytest
from research_repo_tools.process import run_command_bytes

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.parametrize("status", [0, 17])
def test_native_launcher_preserves_process_boundaries(tmp_path: Path, native_console_scripts: Callable[[Path, str, dict[str, str]], Path], status: int) -> None:
    """No command shell may interpret arguments or recode an archive payload."""
    root = tmp_path / "tools with spaces"
    root.mkdir()
    scripts = native_console_scripts(
        root,
        "import json, os, sys\n"
        "def main():\n"
        '    sys.stdout.buffer.write(json.dumps(sys.argv[1:]).encode() + b"\\n\\xff\\r\\n")\n'
        '    sys.stderr.buffer.write(b"fixture diagnostic\\n")\n'
        '    return int(os.environ["FIXTURE_STATUS"])\n',
        {"fixture-echo": "main"},
    )
    arguments = ["argument with spaces", "runs?branch=main&status=success", 'literal "quotes"', "caret^value", "$(touch SHOULD_NOT_EXIST)"]
    result = run_command_bytes(
        "fixture-echo",
        arguments,
        cwd=root,
        env=os.environ | {"PATH": f"{scripts}{os.pathsep}{os.environ['PATH']}", "FIXTURE_STATUS": str(status)},
        check=False,
    )
    assert result.returncode == status
    assert result.stdout == json.dumps(arguments).encode() + b"\n\xff\r\n"
    assert result.stderr == b"fixture diagnostic\n"
    assert not (root / "SHOULD_NOT_EXIST").exists()

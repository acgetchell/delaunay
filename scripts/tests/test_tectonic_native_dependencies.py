"""Exercise native dependency discovery without the host's libraries or tools."""

import shutil
from pathlib import Path

import pytest
from research_repo_tools.process import run_command as run_safe_command

SCRIPT = Path(__file__).resolve().parents[1] / "tectonic_native_dependencies.sh"


@pytest.mark.parametrize("sourced", [False, True])
@pytest.mark.parametrize("prerequisite", ["vcpkg", "pkg-config", "libraries", "available"])
def test_bash_propagates_dependency_status_without_exiting_source_caller(prerequisite: str, *, sourced: bool) -> None:
    """Sourcing returns status to the caller; execution returns it to the runner."""
    harness = """
set -eu
uname() { printf '%s\\n' "$dependency_test_platform"; }
export -f uname
dependency_test_platform=Darwin
export TECTONIC_DEP_BACKEND='' VCPKG_ROOT='' VCPKGRS_TRIPLET=x64-windows-static-md
case "$2" in
    vcpkg) dependency_test_platform=MINGW64_NT ;;
    libraries) pkg-config() { return 1; }; export -f pkg-config ;;
    available) pkg-config() { return 0; }; export -f pkg-config ;;
esac
export dependency_test_platform PATH=''
if [ "$3" = sourced ]; then
    if . "$1"; then dependency_test_status=0; else dependency_test_status=$?; fi
    printf 'caller survived: %s\\n' "$dependency_test_status"
else
    "$BASH" "$1"
fi
"""
    result = run_safe_command(
        "bash",
        ["--noprofile", "--norc", "-c", harness, "test", SCRIPT.as_posix(), prerequisite, "sourced" if sourced else "executed"],
        check=False,
    )
    status = 0 if prerequisite == "available" else 1
    assert result.returncode == (0 if sourced else status), result.stderr
    if sourced:
        assert f"caller survived: {status}" in result.stdout


@pytest.mark.parametrize("shell", ["dash", "zsh"])
@pytest.mark.parametrize("sourced", [False, True])
def test_non_bash_shell_is_rejected_before_bash_syntax(shell: str, *, sourced: bool) -> None:
    """The POSIX guard rejects unsupported shells without ending a source caller."""
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} is not installed")
    arguments = [SCRIPT.as_posix()]
    if sourced:
        arguments = [
            "-c",
            'if . "$1"; then dependency_test_status=0; else dependency_test_status=$?; fi; printf "caller survived: %s\\n" "$dependency_test_status"',
            "test",
            SCRIPT.as_posix(),
        ]
    result = run_safe_command(shell, arguments, check=False)
    assert result.returncode == (0 if sourced else 1), result.stderr
    assert "requires Bash" in result.stderr
    assert "syntax error" not in result.stderr.lower()
    if sourced:
        assert "caller survived: 1" in result.stdout

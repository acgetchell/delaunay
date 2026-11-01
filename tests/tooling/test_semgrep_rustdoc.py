"""Native consumer proofs for additive Rustdoc scans and the temporary bridge."""

import json
import os
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

import pytest
import yaml
from research_repo_tools import security, semgrep_scan
from research_repo_tools.process import run_command

import shared_tooling

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
NUMERIC = "delaunay.rust.no-nonfinite-unwrap-defaults"
STDIO = "delaunay.rust.no-stdio-diagnostics-in-src"
TOPOLOGY = "delaunay.docs.topological-space-notation"
SOURCE = "src/nested path/éxample.rs"
GUIDE = "docs/usage μ.md"
CORPUS = {
    SOURCE: (
        'pub fn diagnostics() { println!("native"); }\n'
        '/// ```rust\n/// println!("legitimate example");\n/// ```\n'
        "/// ```rust\n/// # let bad = Some(1.0).unwrap_or(f64::NAN);\n/// ```\n"
        f"/// ```rust\n/// // nosemgrep: {NUMERIC}\n"
        "/// let suppressed = Some(1.0).unwrap_or(f64::INFINITY);\n/// ```\n"
        "pub fn examples() {}\n"
    ),
    GUIDE: "A usage example.\n\n```rust\nlet bad = Some(1.0).unwrap_or(f64::NAN);\n```\n",
    "docs/prose.md": "A 3D toroidal construction.\n",
    # Explanatory fragments retain native prose policy without Rust extraction.
    "docs/api_design.md": "A 3D toroidal construction.\n\n```rust\nlet sketch = /* ... */;\n```\n",
    "docs/archive/history.md": "Historical sketch.\n\n```rust\nlet sketch = /* ... */;\n```\n",
}


def consumer(root: Path, sources: dict[str, str], *, selected_rules: set[str] | None = None) -> tuple[str, ...]:
    """Use unchanged repository rule definitions and consumer extraction policy."""
    rules = yaml.safe_load((ROOT / "semgrep.yaml").read_text(encoding="utf-8"))["rules"]
    selected_ids = selected_rules if selected_rules is not None else {NUMERIC, STDIO, TOPOLOGY, "delaunay.docs.check-before-fix-command-order"}
    selected = [rule for rule in rules if rule["id"] in selected_ids]
    (root / "semgrep.yaml").write_text(yaml.safe_dump({"rules": selected}), encoding="utf-8")
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    exclusions = manifest["tool"]["delaunay"]["semgrep"]["rustdoc-exclude"]
    (root / "pyproject.toml").write_text(
        '[tool.research-repo-tools.semgrep]\nconfig = "semgrep.yaml"\n'
        'fixtures = "fixtures"\nnamespace = "delaunay."\ninline-suppressions = true\n'
        'report-category = "semgrep-repository-rules"\nreport-layout = "aggregate"\n'
        f"[tool.delaunay.semgrep]\nrustdoc-exclude = {json.dumps(exclusions)}\n",
        encoding="utf-8",
    )
    for name, text in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tuple(sources)


def scan(root: Path, names: tuple[str, ...], monkeypatch: pytest.MonkeyPatch) -> int:
    """Model only Git inventory; run the installed extractor, scanner and validators."""
    monkeypatch.setattr(semgrep_scan, "security_inventory", lambda *_args, **_kwargs: names)
    return shared_tooling.main(["--root", str(root), "semgrep", "scan", "--include", "*", "--rust-docs", "--batch-size", "1", "--output", "reports"])


def test_deep_import_policy_covers_triangulation_paths_and_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real rooted scan observes both the moved directory and its crate namespace."""
    rule = "delaunay.rust.no-deep-crate-paths-in-functions"
    corpus = {
        "src/triangulation/probe.rs": (
            "use crate::core::collections::SimplexKeyBuffer;\n"
            "pub fn compliant() { let _buffer = SimplexKeyBuffer::new(); }\n"
            "pub fn violation() {\n"
            "    let _buffer = crate::core::collections::SimplexKeyBuffer::new();\n"
            "}\n"
        ),
        "src/geometry/probe.rs": (
            "use crate::triangulation::query::Query;\n"
            "pub fn compliant() { let _query = Query::new(); }\n"
            "pub fn violation() {\n"
            "    let _query = crate::triangulation::query::Query::new();\n"
            "}\n"
        ),
    }
    names = consumer(tmp_path, corpus, selected_rules={rule})
    assert scan(tmp_path, names, monkeypatch) == 1
    report = json.loads((tmp_path / "reports/semgrep.json").read_bytes())
    assert report["errors"] == []
    assert {(row["check_id"], Path(row["path"]).relative_to(tmp_path).as_posix(), row["start"]["line"]) for row in report["results"]} == {
        (rule, name, 4) for name in corpus
    }
    assert {Path(path).relative_to(tmp_path).as_posix() for path in report["paths"]["scanned"]} == set(corpus)


@pytest.fixture(scope="module")
def native_reports(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, bytes], list[Path]]:
    """Produce one real aggregate pair with native, later-block and prose findings."""
    root = tmp_path_factory.mktemp("native-rustdoc")
    names = consumer(root, CORPUS)
    calls: list[Path] = []
    original = security.run_command_bytes

    def record(  # noqa: PLR0913 - match the installed runner's invocation boundary.
        binary: str | Path,
        arguments: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        input: bytes,  # noqa: A002 - match the installed runner keyword.
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(cwd)
        result = original(binary, arguments, cwd=cwd, env=env, input=input, timeout=timeout, check=check)
        # Native Semgrep omits optional indices. Add equivalent references at
        # its output boundary to prove aggregate reindexing without inventing
        # findings, paths, rules, spans, suppressions or coverage.
        path = Path(arguments[arguments.index("--sarif-output") + 1])
        document = json.loads(path.read_bytes())
        for run in document["runs"]:
            rules = {rule["id"]: index for index, rule in enumerate(run["tool"]["driver"]["rules"])}
            run["artifacts"] = []
            for finding in run["results"]:
                finding["ruleIndex"] = rules[finding["ruleId"]]
                finding["provenance"] = {"invocationIndex": 0}
                location = finding["locations"][0]["physicalLocation"]["artifactLocation"]
                index = len(run["artifacts"])
                run["artifacts"].append({"location": location.copy()})
                location["index"] = index
        path.write_text(json.dumps(document), encoding="utf-8")
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(security, "run_command_bytes", record)
        assert scan(root, names, patch) == 1
    return root, {path.name: path.read_bytes() for path in (root / "reports").iterdir()}, calls


def test_native_aggregate_scopes_mapping_and_suppressions(native_reports: tuple[Path, dict[str, bytes], list[Path]]) -> None:
    root, reports, calls = native_reports
    data = json.loads(reports["semgrep.json"])
    sarif = json.loads(reports["semgrep.sarif"])
    expected = Counter(
        {(STDIO, SOURCE, 1): 1, (NUMERIC, SOURCE, 6): 1, (NUMERIC, GUIDE, 4): 1, (TOPOLOGY, "docs/prose.md", 1): 1, (TOPOLOGY, "docs/api_design.md", 1): 1}
    )
    actual = Counter(
        (row["check_id"], Path(row["path"]).relative_to(root).as_posix(), row["start"]["line"]) for row in data["results"] if not row["extra"].get("is_ignored")
    )
    assert actual == expected
    assert data["errors"] == []
    assert {Path(path).relative_to(root).as_posix() for path in data["paths"]["scanned"]} == set(CORPUS)
    coverage = Counter((row["path"], row["representation"], row.get("block")) for row in data["input_coverage"])
    assert coverage == Counter(
        {
            **{(name, "source", None): 1 for name in CORPUS},
            (SOURCE, "rust-block", 1): 1,
            (SOURCE, "rust-block", 2): 1,
            (SOURCE, "rust-block", 3): 1,
            (GUIDE, "rust-block", 1): 1,
        }
    )
    assert len(data["batches"]) == len(calls) == 9
    assert all(path == root or not path.exists() for path in calls)
    assert {name: (root / name).read_text(encoding="utf-8") for name in CORPUS} == CORPUS
    assert len(sarif["runs"]) == 1
    run = sarif["runs"][0]
    assert run["automationDetails"]["id"] == "semgrep-repository-rules/"
    observed = Counter()
    for result in run["results"]:
        assert run["tool"]["driver"]["rules"][result["ruleIndex"]]["id"] == result["ruleId"]
        assert 0 <= result["provenance"]["invocationIndex"] < len(run["invocations"])
        location = result["locations"][0]["physicalLocation"]
        artifact = location["artifactLocation"]
        if "index" in artifact:
            assert run["artifacts"][artifact["index"]]["location"]["uri"] == artifact["uri"]
        uri = artifact["uri"]
        source = Path.from_uri(uri) if uri.startswith("file:") else Path(unquote(uri))
        path = source.relative_to(root).as_posix()
        if not result.get("suppressions"):
            observed[(result["ruleId"], path, location["region"]["startLine"])] += 1
    assert observed == expected
    ignored = [row for row in data["results"] if row["extra"].get("is_ignored")]
    # Semgrep 1.182.1 omits suppressed findings from JSON while retaining
    # inSource suppressions in SARIF. Active findings must still agree exactly.
    assert ignored == []
    suppressed = [row for row in run["results"] if row.get("suppressions")]
    assert len(suppressed) == 1
    assert suppressed[0]["ruleId"] == NUMERIC
    assert suppressed[0]["locations"][0]["physicalLocation"]["region"]["startLine"] == 10
    assert suppressed[0]["suppressions"] == [{"kind": "inSource"}]
    later = next(row for row in data["results"] if row["check_id"] == NUMERIC and row["start"]["line"] == 6)
    assert later["start"]["col"] == CORPUS[SOURCE].splitlines()[5].index("Some") + 1
    assert json.loads((root / "reports.attempt.json").read_bytes()) == {"published": True, "exit_code": 1}


def damage_reports(report: Path, sarif: Path, failure: str) -> None:
    """Inject invalid output after native execution, keeping validators real."""
    data = json.loads(report.read_bytes())
    if failure == "skip":
        data["paths"]["scanned"] = []
    elif failure == "native-error":
        data["errors"] = [{"type": "Timeout", "message": "injected native failure"}]
    elif failure == "malformed":
        report.write_bytes(b"{incomplete")
        return
    elif failure == "missing-pair":
        sarif.unlink()
        return
    elif failure == "disagree":
        document = json.loads(sarif.read_bytes())
        document["runs"][0]["results"] = []
        sarif.write_text(json.dumps(document), encoding="utf-8")
        return
    else:
        raise AssertionError(f"unhandled report failure: {failure}")
    report.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("failure", ["skip", "native-error", "native-parse", "malformed", "missing-pair", "disagree", "timeout"])
def test_later_block_failure_retains_previous_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], native_reports: tuple[Path, dict[str, bytes], list[Path]], failure: str
) -> None:
    _, previous, _ = native_reports
    names = consumer(tmp_path, {SOURCE: CORPUS[SOURCE]})
    if failure == "native-parse":
        # Keep a rule-relevant token so native prefiltering reaches the parser.
        # A direct native probe confirms this unfinished item emits an error.
        (tmp_path / SOURCE).write_text(
            CORPUS[SOURCE].replace("# let bad = Some(1.0).unwrap_or(f64::NAN);", "let bad = Some(1.0).unwrap_or(f64::NAN);\n/// fn broken("), encoding="utf-8"
        )
    destination = tmp_path / "reports"
    destination.mkdir()
    for name, payload in previous.items():
        (destination / name).write_bytes(payload)
    original = security.run_command_bytes
    damaged: list[Path] = []

    def corrupt(  # noqa: PLR0913 - match the installed runner's invocation boundary.
        binary: str | Path,
        arguments: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        input: bytes,  # noqa: A002 - match the installed runner keyword.
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        result = original(binary, arguments, cwd=cwd, env=env, input=input, timeout=timeout, check=check)
        if cwd.name != "1" or cwd.parent.name != "blocks":
            return result
        damaged.append(cwd)
        if failure == "native-parse":
            return result
        if failure == "timeout":
            raise subprocess.TimeoutExpired(str(binary), timeout)
        report = Path(arguments[arguments.index("--output") + 1])
        sarif = Path(arguments[arguments.index("--sarif-output") + 1])
        damage_reports(report, sarif, failure)
        return result

    monkeypatch.setattr(security, "run_command_bytes", corrupt)
    assert scan(tmp_path, names, monkeypatch) != 0
    assert len(damaged) == 1
    assert not damaged[0].exists()
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == previous
    assert json.loads((tmp_path / "reports.attempt.json").read_bytes())["published"] is False
    diagnostic = capsys.readouterr().err
    assert "no report published" in diagnostic
    if failure in {"skip", "native-error", "native-parse"}:
        assert SOURCE in diagnostic
        assert "Rust block 2" in diagnostic


@pytest.mark.parametrize("published", [True, False, None])
def test_workflow_upload_requires_current_attempt(tmp_path: Path, published: bool | None) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/semgrep-sarif.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["semgrep-sarif"]["steps"]
    command = next(step["run"] for step in steps if step.get("id") == "semgrep")
    assert command.index("rm -f target/semgrep.attempt.json") < command.index("just semgrep-scan")
    code = command.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    marker = tmp_path / "target/semgrep.attempt.json"
    marker.parent.mkdir()
    if published is not None:
        marker.write_text(json.dumps({"published": published}), encoding="utf-8")
    output = tmp_path / "outputs"
    run_command(sys.executable, ["-c", code], cwd=tmp_path, env={**os.environ, "GITHUB_OUTPUT": str(output)})
    assert output.read_text(encoding="utf-8") == f"report_current={str(published is True).lower()}\n"
    upload = next(step for step in steps if step["name"] == "Upload SARIF results")
    assert "steps.semgrep.outputs.report_current == 'true'" in upload["if"]
    assert any(step.get("if") == "steps.semgrep.outputs.exit_code != '0'" and step.get("run") == "exit 1" for step in steps)


@pytest.mark.parametrize("change", ["unchanged", "missing", "unexpected"])
def test_mixed_native_fixture_assertions(tmp_path: Path, capsys: pytest.CaptureFixture[str], change: str) -> None:
    """Positive/negative counts survive mixed source, blocks and prose-only files."""
    example = (
        f"// ruleid: {NUMERIC}\nfn native() {{ let x = Some(1.0).unwrap_or(f64::NAN); }}\n"
        f"/// ```rust\n/// // ruleid: {NUMERIC}\n/// let x = Some(1.0).unwrap_or(f64::NAN);\n/// ```\n"
        f"/// ```rust\n/// // ok: {NUMERIC}\n/// let x = Some(1.0).unwrap_or(0.0);\n/// ```\nfn examples() {{}}\n"
    )
    markdown = f"<!-- ruleid: {TOPOLOGY} -->\nA 3D toroidal construction.\n\n```rust\n// ruleid: {NUMERIC}\nlet x = Some(1.0).unwrap_or(f64::NAN);\n```\n"
    consumer(
        tmp_path,
        {"fixtures/example.rs": example, "fixtures/guide.md": markdown, "fixtures/prose.md": f"<!-- ruleid: {TOPOLOGY} -->\nA 3D toroidal construction.\n"},
    )
    config = tmp_path / "semgrep.yaml"
    rules = yaml.safe_load(config.read_text(encoding="utf-8"))["rules"]
    config.write_text(yaml.safe_dump({"rules": [rule for rule in rules if rule["id"] in {NUMERIC, TOPOLOGY}]}), encoding="utf-8")
    if change == "missing":
        (tmp_path / "fixtures/guide.md").write_text(markdown.replace("f64::NAN", "0.0"), encoding="utf-8")
    elif change == "unexpected":
        (tmp_path / "fixtures/example.rs").write_text(example.replace("unwrap_or(0.0)", "unwrap_or(f64::INFINITY)"), encoding="utf-8")
    assert shared_tooling.main(["--root", str(tmp_path), "semgrep", "check-fixtures", "--rust-docs"]) == int(change != "unchanged")
    errors = capsys.readouterr().err
    if change == "missing":
        assert "expected finding not reported" in errors
    elif change == "unexpected":
        assert "unexpected finding" in errors
        assert "Rust block 2" in errors
    else:
        assert not errors


def test_precondition_failure_clears_current_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_reports: tuple[Path, dict[str, bytes], list[Path]]
) -> None:
    _, previous, _ = native_reports
    consumer(tmp_path, {})
    destination = tmp_path / "reports"
    destination.mkdir()
    for name, payload in previous.items():
        (destination / name).write_bytes(payload)
    marker = tmp_path / "reports.attempt.json"
    marker.write_text('{"published": true}', encoding="utf-8")
    assert scan(tmp_path, (), monkeypatch) != 0
    assert json.loads(marker.read_bytes())["published"] is False
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == previous

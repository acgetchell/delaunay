#!/usr/bin/env python3
"""Filter and split Codacy SARIF before GitHub Code Scanning upload."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs

REPOSITORY_RULE_PREFIX = "delaunay."

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SarifDocument:
    """Validated SARIF root object with a parsed top-level runs array."""

    root: JsonObject
    runs: list[JsonValue]

    def metadata_without_runs(self) -> JsonObject:
        """Return a deep copy of the SARIF metadata used for split outputs."""
        return {key: copy.deepcopy(value) for key, value in self.root.items() if key != "runs"}


@dataclass(frozen=True, slots=True)
class CliArgs:
    """Parsed CLI arguments for filtering a Codacy SARIF file."""

    source: Path
    out_dir: Path
    github_env: Path | None


def is_json_object(value: object) -> TypeIs[JsonObject]:
    """Return whether a JSON value is an object with string keys."""
    return isinstance(value, dict) and all(isinstance(key, str) and is_json_value(item) for key, item in value.items())


def is_json_value(value: object) -> TypeIs[JsonValue]:
    """Return whether a value can be represented as strict JSON."""
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    return is_json_object(value)


def reject_json_constant(value: str) -> object:
    """Reject non-standard JSON constants accepted by Python's decoder."""
    message = f"Codacy SARIF contains non-standard JSON constant: {value}"
    raise ValueError(message)


def slug(value: str) -> str:
    """Return a filesystem-safe SARIF category slug."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower())
    return normalized.strip("-") or "unknown"


def result_rule_id(result: JsonValue) -> str:
    """Return the SARIF rule ID referenced by a result, if present."""
    if not is_json_object(result):
        return ""
    rule_id = result.get("ruleId")
    if isinstance(rule_id, str):
        return rule_id
    rule = result.get("rule")
    if is_json_object(rule):
        nested_rule_id = rule.get("id")
        if isinstance(nested_rule_id, str):
            return nested_rule_id
    return ""


def run_driver(run: JsonObject) -> JsonObject:
    """Return the SARIF run driver object, or an empty mapping."""
    tool = run.get("tool")
    if not is_json_object(tool):
        return {}
    driver = tool.get("driver")
    if not is_json_object(driver):
        return {}
    return driver


def keep_repository_owned_opengrep_rules(run: JsonObject) -> None:
    """Drop default Codacy/OpenGrep results that do not come from semgrep.yaml."""
    driver = run_driver(run)
    tool_name = driver.get("name")
    if "opengrep" not in str(tool_name).lower():
        return

    results = run.get("results")
    if isinstance(results, list):
        run["results"] = [result for result in results if result_rule_id(result).startswith(REPOSITORY_RULE_PREFIX)]

    filtered_results = run.get("results")
    if not isinstance(filtered_results, list):
        filtered_results = []
    used_rule_ids = {result_rule_id(result) for result in filtered_results}
    rules = driver.get("rules")
    if isinstance(rules, list):
        driver["rules"] = [rule for rule in rules if is_json_object(rule) and rule.get("id") in used_rule_ids]


def parse_sarif_document(raw: object) -> SarifDocument:
    """Parse raw JSON into the SARIF shape this splitter requires."""
    if not is_json_object(raw):
        message = "Codacy SARIF root must be a JSON object"
        raise SystemExit(message)

    runs = raw.get("runs")
    if not isinstance(runs, list):
        message = "Codacy SARIF did not contain a runs array"
        raise SystemExit(message)

    return SarifDocument(root=raw, runs=runs)


def load_sarif(source: Path) -> SarifDocument:
    """Load a SARIF document from disk."""
    if not source.is_file() or source.stat().st_size == 0:
        raise SystemExit(f"Codacy did not produce a SARIF file at {source}")

    try:
        sarif: object = json.loads(source.read_text(encoding="utf-8"), parse_constant=reject_json_constant)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Codacy produced invalid SARIF JSON: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    return parse_sarif_document(sarif)


def split_sarif_runs(sarif: SarifDocument, out_dir: Path) -> int:
    """Write one uploadable SARIF file per non-empty run and return the count."""
    if not sarif.runs:
        print("Codacy SARIF did not contain any runs to upload")

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(out_dir.glob("*.sarif")):
        stale.unlink()

    seen_categories: dict[str, int] = {}
    uploadable_count = 0
    for index, run in enumerate(sarif.runs, start=1):
        if not is_json_object(run):
            print(f"Skipping malformed SARIF run {index}")
            continue

        run_copy = copy.deepcopy(run)
        keep_repository_owned_opengrep_rules(run_copy)

        rules = run_driver(run_copy).get("rules")
        results = run_copy.get("results")
        if not rules and not results:
            print(f"Skipping empty SARIF run {index} with no rules or results")
            continue

        tool = run_driver(run_copy).get("name")
        base_category = f"codacy-{slug(str(tool or f'run-{index}'))}"
        seen_categories[base_category] = seen_categories.get(base_category, 0) + 1
        suffix = seen_categories[base_category]
        category = base_category if suffix == 1 else f"{base_category}-{suffix}"

        automation_value = run_copy.get("automationDetails")
        automation: JsonObject = automation_value if is_json_object(automation_value) else {}
        automation["id"] = category
        run_copy["automationDetails"] = automation

        split_sarif = sarif.metadata_without_runs()
        split_sarif.setdefault("$schema", "https://json.schemastore.org/sarif-2.1.0.json")
        split_sarif.setdefault("version", "2.1.0")
        split_sarif["runs"] = [run_copy]

        out_file = out_dir / f"{index:02d}-{category}.sarif"
        out_file.write_text(json.dumps(split_sarif, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote {out_file} with category {category}")
        uploadable_count += 1

    if uploadable_count == 0:
        print("No non-empty Codacy SARIF runs to upload")
    return uploadable_count


def write_github_env(out_dir: Path, uploadable_count: int, env_file: Path | None) -> None:
    """Record split-SARIF state for later GitHub Actions steps."""
    if env_file is None:
        return
    with env_file.open("a", encoding="utf-8") as file:
        file.write(f"CODACY_SPLIT_SARIF_DIR={out_dir}\n")
        file.write(f"CODACY_HAS_UPLOADABLE_SARIF={'true' if uploadable_count else 'false'}\n")


def parse_args(argv: list[str]) -> CliArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, suggest_on_error=True, color=False)
    parser.add_argument("source", type=Path, help="Codacy SARIF file to filter and split")
    parser.add_argument("out_dir", type=Path, help="Directory for split SARIF files")
    parser.add_argument(
        "--github-env",
        type=Path,
        default=Path(os.environ["GITHUB_ENV"]) if "GITHUB_ENV" in os.environ else None,
        help="GitHub Actions environment file to update; defaults to $GITHUB_ENV when set",
    )
    args = parser.parse_args(argv)
    return CliArgs(source=args.source, out_dir=args.out_dir, github_env=args.github_env)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    sarif = load_sarif(args.source)
    uploadable_count = split_sarif_runs(sarif, args.out_dir)
    write_github_env(args.out_dir, uploadable_count, args.github_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

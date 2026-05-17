#!/usr/bin/env python3
"""Filter and split Codacy SARIF before GitHub Code Scanning upload."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, cast

REPOSITORY_RULE_PREFIX = "delaunay."


def slug(value: str) -> str:
    """Return a filesystem-safe SARIF category slug."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower())
    return normalized.strip("-") or "unknown"


def result_rule_id(result: object) -> str:
    """Return the SARIF rule ID referenced by a result, if present."""
    if not isinstance(result, dict):
        return ""
    result_map = cast("dict[str, Any]", result)
    rule_id = result_map.get("ruleId")
    if isinstance(rule_id, str):
        return rule_id
    rule = result_map.get("rule")
    if isinstance(rule, dict):
        rule_map = cast("dict[str, Any]", rule)
        nested_rule_id = rule_map.get("id")
        if isinstance(nested_rule_id, str):
            return nested_rule_id
    return ""


def run_driver(run: dict[str, Any]) -> dict[str, Any]:
    """Return the SARIF run driver object, or an empty mapping."""
    tool = run.get("tool")
    if not isinstance(tool, dict):
        return {}
    driver = tool.get("driver")
    if not isinstance(driver, dict):
        return {}
    return driver


def keep_repository_owned_opengrep_rules(run: dict[str, Any]) -> None:
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
        driver["rules"] = [rule for rule in rules if isinstance(rule, dict) and rule.get("id") in used_rule_ids]


def load_sarif(source: Path) -> dict[str, Any]:
    """Load a SARIF document from disk."""
    if not source.is_file() or source.stat().st_size == 0:
        raise SystemExit(f"Codacy did not produce a SARIF file at {source}")

    try:
        sarif = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Codacy produced invalid SARIF JSON: {exc}") from exc

    if not isinstance(sarif, dict):
        message = "Codacy SARIF root must be a JSON object"
        raise SystemExit(message)
    return sarif


def split_sarif_runs(sarif: dict[str, Any], out_dir: Path) -> int:
    """Write one uploadable SARIF file per non-empty run and return the count."""
    runs = sarif.get("runs")
    if not isinstance(runs, list):
        message = "Codacy SARIF did not contain a runs array"
        raise SystemExit(message)
    if not runs:
        print("Codacy SARIF did not contain any runs to upload")

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(out_dir.glob("*.sarif")):
        stale.unlink()

    seen_categories: dict[str, int] = {}
    uploadable_count = 0
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            print(f"Skipping malformed SARIF run {index}")
            continue

        run_copy = cast("dict[str, Any]", copy.deepcopy(run))
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

        automation = run_copy.get("automationDetails")
        if not isinstance(automation, dict):
            automation = {}
        automation["id"] = category
        run_copy["automationDetails"] = automation

        split_sarif = {key: value for key, value in sarif.items() if key != "runs"}
        split_sarif.setdefault("$schema", "https://json.schemastore.org/sarif-2.1.0.json")
        split_sarif.setdefault("version", "2.1.0")
        split_sarif["runs"] = [run_copy]

        out_file = out_dir / f"{index:02d}-{category}.sarif"
        out_file.write_text(json.dumps(split_sarif, indent=2), encoding="utf-8")
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Codacy SARIF file to filter and split")
    parser.add_argument("out_dir", type=Path, help="Directory for split SARIF files")
    parser.add_argument(
        "--github-env",
        type=Path,
        default=Path(os.environ["GITHUB_ENV"]) if "GITHUB_ENV" in os.environ else None,
        help="GitHub Actions environment file to update; defaults to $GITHUB_ENV when set",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    sarif = load_sarif(args.source)
    uploadable_count = split_sarif_runs(sarif, args.out_dir)
    write_github_env(args.out_dir, uploadable_count, args.github_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

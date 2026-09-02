#!/usr/bin/env python3
"""Validate the sanitized evidence bundle and the allowlisted public snapshot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path


REQUIRED = {"README.md", "LICENSE", "CITATION.cff", "Makefile", "pyproject.toml",
            "evidence/manifest.json", ".github/workflows/ci.yml"}
BANNED_TOP_LEVEL = {"thesis", "logs", "runs", "snapshots"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".npy", ".npz", ".pyc"}
SECRET_RE = re.compile(
    r"(?:sk-ant-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16}|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY)"
)
LOCAL_RE = re.compile(r"(?:/home/[A-Za-z0-9._-]+|/media/[A-Za-z0-9._-]+|[A-Za-z]:\\\\Users\\\\)")
RUN_LABELS = {
    "B_dbpedia_N300_order1": "LLM wiki: DBPedia, order 1",
    "B_dbpedia_N100_order2": "LLM wiki: DBPedia, order 2",
    "B_wikics_N100_order1": "LLM wiki: WikiCS",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_evidence(root: Path) -> list[str]:
    errors: list[str] = []
    evidence = root / "evidence"
    manifest_path = evidence / "manifest.json"
    if not manifest_path.exists():
        return ["evidence/manifest.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in (manifest.get("files") or {}).items():
        path = evidence / name
        if not path.exists():
            errors.append(f"missing evidence/{name}")
            continue
        if sha256(path) != expected.get("sha256"):
            errors.append(f"hash mismatch for evidence/{name}")
        if path.read_bytes().count(b"\n") != expected.get("records"):
            errors.append(f"record mismatch for evidence/{name}")

    cost_table = root / "results" / "tables" / "cost_per_ingest.csv"
    arm_a_table = root / "results" / "tables" / "supp_armA.csv"
    if cost_table.exists():
        expected_points: dict[tuple[str, int], float] = {}
        for path in sorted((evidence / "ingests").glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                label = RUN_LABELS.get(str(row.get("run_id")))
                if (label and row.get("kind") == "ingest" and row.get("status") == "ok"
                        and row.get("cost_usd") is not None):
                    expected_points[(label, int(row["ingest_index"]))] = float(row["cost_usd"])
        if arm_a_table.exists():
            for row in csv.DictReader(arm_a_table.open(encoding="utf-8")):
                if row.get("corpus") == "dbpedia" and row.get("usd"):
                    n = int(row["N"])
                    expected_points[("Pipeline (full rebuild)", n)] = float(row["usd"]) / n
        actual_points = {}
        for row in csv.DictReader(cost_table.open(encoding="utf-8")):
            actual_points[(row["series"], int(row["documents"]))] = float(row["usd_per_document"])
        if set(actual_points) != set(expected_points):
            errors.append("cost_per_ingest.csv keys do not match sanitized evidence")
        else:
            for key, expected in expected_points.items():
                if abs(actual_points[key] - expected) > 5e-7:
                    errors.append(f"cost mismatch at {key}: {actual_points[key]} != {expected}")
                    break
    return errors


def check_snapshot(root: Path) -> list[str]:
    errors: list[str] = []
    present = {p.name for p in root.iterdir()} if root.exists() else set()
    for name in sorted(REQUIRED):
        if not (root / name).exists():
            errors.append(f"required path missing: {name}")
    for name in sorted(BANNED_TOP_LEVEL & present):
        errors.append(f"local-only directory present: {name}/")

    total = 0
    for path in root.rglob("*"):
        if (not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts
                or ".pytest_cache" in path.parts):
            continue
        size = path.stat().st_size
        total += size
        if size > 50 * 1024 * 1024:
            errors.append(f"file exceeds 50 MiB: {path.relative_to(root)}")
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(root)
        if LOCAL_RE.search(text.replace("/home/x", "")):
            errors.append(f"local absolute path in {rel}")
        if SECRET_RE.search(text):
            errors.append(f"possible secret in {rel}")
        if "evidence" in rel.parts and re.search(r'"(?:session_id|apiKeySource)"\s*:', text):
            errors.append(f"private runtime field in {rel}")
    if total > 50 * 1024 * 1024:
        errors.append(f"snapshot is larger than 50 MiB ({total / 1048576:.1f} MiB)")
    return errors


def check_regenerated_tables(root: Path) -> list[str]:
    """Rebuild the standard ledger tables and require byte-identical frozen CSVs."""
    errors: list[str] = []
    sys.path.insert(0, str(root / "src"))
    try:
        from wikiorg.eval.report import write_tables
    except Exception as exc:  # pragma: no cover - exercised in installed CI environments
        return [f"cannot import table generator: {type(exc).__name__}: {exc}"]
    with tempfile.TemporaryDirectory(prefix="wikiorg-public-tables-") as temp:
        written = write_tables(ledger_path=root / "evidence" / "ledger.jsonl", tables_dir=temp)
        for name, generated in sorted(written.items()):
            expected = root / "results" / "tables" / f"{name}.csv"
            if not expected.exists():
                errors.append(f"frozen table missing: results/tables/{name}.csv")
            elif generated.read_bytes() != expected.read_bytes():
                errors.append(f"regenerated table differs: results/tables/{name}.csv")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--regenerate-tables", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    errors = check_evidence(root)
    if not args.evidence_only:
        errors.extend(check_snapshot(root))
    if args.regenerate_tables:
        errors.extend(check_regenerated_tables(root))
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print(f"PASS {'evidence' if args.evidence_only else 'public snapshot'}: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

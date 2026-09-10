#!/usr/bin/env python3
"""Export the explicit, thesis-free file set used for the GitHub release snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT_FILES = (".gitignore", "CITATION.cff", "LICENSE", "Makefile", "README.md", "pyproject.toml")
TREE_DIRS = (".github", "LICENSES", "src", "tests", "evidence")
CONFIG_FILES = (
    "arm_a.yaml", "arm_a_ninit100_posthoc.yaml", "cli_facts.json", "prereg.yaml",
    "prereg_v2.yaml", "prereg_v3.yaml", "prices.yaml",
)
SCRIPT_FILES = (
    "00_env_check.py", "02_reproducibility.py", "10_cost_scenarios.py", "export_public_evidence.py",
    "check_public_release.py", "export_public_snapshot.py",
)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))


def normalized_prereg(source: Path) -> str:
    raw = source.read_text(encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    normalized = re.sub(r"(?m)^(\s*python(?:_executable)?\s*:\s*).+$", r"\1python3.12", raw)
    banner = ("# Public copy: the historical local interpreter path was normalized.\n"
              f"# SHA-256 of the frozen local file: {digest}\n")
    return banner + normalized


def normalized_json(source: Path) -> str:
    """Make repository-local absolute paths portable without changing other values."""
    data = json.loads(source.read_text(encoding="utf-8"))

    def visit(value: object) -> object:
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            try:
                return str(Path(value).relative_to(REPO)) if Path(value).is_absolute() else value
            except ValueError:
                return value
        return value

    return json.dumps(visit(data), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite non-empty directory: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    for name in ROOT_FILES:
        copy_file(REPO / name, out / name)
    for name in TREE_DIRS:
        copy_tree(REPO / name, out / name)
    for name in CONFIG_FILES:
        source = REPO / "configs" / name
        destination = out / "configs" / name
        if name == "prereg.yaml":
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(normalized_prereg(source), encoding="utf-8")
        else:
            copy_file(source, destination)
    for name in SCRIPT_FILES:
        copy_file(REPO / "scripts" / name, out / "scripts" / name)

    for source in sorted((REPO / "results" / "tables").glob("*")):
        if source.is_file() and source.suffix in {".csv", ".md", ".json"}:
            destination = out / "results" / "tables" / source.name
            if source.suffix == ".json":
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(normalized_json(source), encoding="utf-8")
            else:
                copy_file(source, destination)
    for source in sorted((REPO / "figures").glob("*.png")):
        copy_file(source, out / "figures" / source.name)
    for source in sorted((REPO / "data").glob("**/*")):
        if not source.is_file():
            continue
        if source.name in {"manifest.json", "gold.jsonl"} or source.parent.name == "orders":
            destination = out / source.relative_to(REPO)
            if source.name == "manifest.json":
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(normalized_json(source), encoding="utf-8")
            else:
                copy_file(source, destination)

    checker = out / "scripts" / "check_public_release.py"
    proc = subprocess.run([sys.executable, str(checker), "--root", str(out)], text=True)
    if proc.returncode:
        return proc.returncode
    print(f"wrote curated snapshot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

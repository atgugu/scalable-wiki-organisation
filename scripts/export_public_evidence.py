#!/usr/bin/env python3
"""Create or verify the sanitized evidence bundle used by the public repository."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LEDGER_SOURCE = REPO / "results" / "ledger.jsonl"
CALL_SOURCE = REPO / "results" / "extract"
TABLE_MANIFEST = REPO / "results" / "tables" / "manifest.json"
SCHEMA_VERSION = 1

LEDGER_FIELDS = (
    "kind", "arm", "corpus", "n_docs", "replicate", "variant", "k", "k_variant", "seeds",
    "model_id", "cli_version", "effort", "config", "config_hash", "claude_md_sha256",
    "corpus_manifest_sha256", "eval_code_hash", "cost", "metrics", "extra", "package_versions",
    "platform", "python", "gpu", "price_table_date", "git_hash", "git_tag", "run_git_hash",
    "row_id", "supersedes", "timestamp",
)
CALL_FIELDS = (
    "run_id", "arm", "kind", "corpus", "n", "doc_id", "ingest_index", "attempt", "status",
    "subtype", "is_error", "model", "model_reported", "cli_version", "effort",
    "claude_md_sha256", "prompt_sha256", "wiki_pages_before", "tokens", "cost_usd",
    "num_turns", "tool_calls", "duration_ms", "duration_api_ms", "wall_ms", "waited_ms",
    "cache_write_tier", "started_at", "ended_at", "git_commit",
)
DROP_KEYS = {
    "host", "hostname", "session_id", "apiKeySource", "api_key_source", "stdout_file",
    "scratch", "cwd", "order_file", "permission_denials", "prompt", "files_touched",
}
LOCAL_PATH_RE = re.compile(r"(?:/home/|/media/|[A-Za-z]:\\\\Users\\\\)")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_value(value: object, key: str = "") -> object | None:
    if key in DROP_KEYS:
        return None
    if isinstance(value, dict):
        out = {}
        for child_key, child_value in value.items():
            cleaned = clean_value(child_value, str(child_key))
            if cleaned is not None:
                out[str(child_key)] = cleaned
        return out
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := clean_value(item)) is not None]
    if isinstance(value, str):
        repo_prefix = str(REPO.resolve()) + "/"
        if value.startswith(repo_prefix):
            return value[len(repo_prefix):]
        if LOCAL_PATH_RE.search(value):
            return None
    return value


def select(row: dict, fields: tuple[str, ...]) -> dict:
    out = {}
    for key in fields:
        if key not in row:
            continue
        value = clean_value(row[key], key)
        if value is not None:
            out[key] = value
    return out


def jsonl_bytes(rows: list[dict]) -> bytes:
    return ("".join(canonical_json(row) + "\n" for row in rows)).encode("utf-8")


def build_outputs() -> tuple[dict[str, bytes], dict]:
    if not LEDGER_SOURCE.exists():
        raise FileNotFoundError(f"missing local ledger: {LEDGER_SOURCE}")
    source_rows = [json.loads(line) for line in LEDGER_SOURCE.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    table_manifest = json.loads(TABLE_MANIFEST.read_text(encoding="utf-8"))
    cutoff = int(table_manifest["n_ledger_rows"])
    if not 0 < cutoff <= len(source_rows):
        raise ValueError(f"invalid frozen ledger cutoff {cutoff} for {len(source_rows)} source rows")
    ledger_rows = [select(row, LEDGER_FIELDS) for row in source_rows[:cutoff]]
    outputs: dict[str, bytes] = {"ledger.jsonl": jsonl_bytes(ledger_rows)}
    sources = {"ledger_source_rows": len(source_rows), "ledger_rows": len(ledger_rows),
               "table_analysis_cutoff": cutoff, "ingest_files": 0, "ingest_records": 0}

    for source in sorted(CALL_SOURCE.glob("*.calls.jsonl")):
        rows = [select(json.loads(line), CALL_FIELDS)
                for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        name = source.name.removesuffix(".calls.jsonl") + ".jsonl"
        outputs[f"ingests/{name}"] = jsonl_bytes(rows)
        sources["ingest_files"] += 1
        sources["ingest_records"] += len(rows)

    files = {name: {"sha256": sha256_bytes(data), "bytes": len(data),
                    "records": data.count(b"\n")}
             for name, data in sorted(outputs.items())}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "description": "Sanitized evaluation ledger and ingest-level measurements",
        "source_counts": sources,
        "ledger_fields": list(LEDGER_FIELDS),
        "ingest_fields": list(CALL_FIELDS),
        "files": files,
    }
    outputs["manifest.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    return outputs, manifest


def verify_files(out_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema version: {manifest.get('schema_version')}")
    for name, expected in (manifest.get("files") or {}).items():
        path = out_dir / name
        if not path.exists():
            errors.append(f"missing evidence file: {name}")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != expected.get("sha256"):
            errors.append(f"hash mismatch: {name}")
        if data.count(b"\n") != expected.get("records"):
            errors.append(f"record-count mismatch: {name}")
        text = data.decode("utf-8", errors="replace")
        if LOCAL_PATH_RE.search(text):
            errors.append(f"local path remains: {name}")
        if re.search(r'"(?:session_id|host|hostname|apiKeySource|prompt)"\s*:', text):
            errors.append(f"private field remains: {name}")
    return errors


def write_outputs(out_dir: Path, outputs: dict[str, bytes]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "evidence")
    parser.add_argument("--check", action="store_true", help="verify hashes and, when local sources exist, reproducibility")
    args = parser.parse_args(argv)

    if args.check:
        errors = verify_files(args.out)
        if LEDGER_SOURCE.exists() and CALL_SOURCE.exists():
            expected, _ = build_outputs()
            with tempfile.TemporaryDirectory(prefix="wikiorg-evidence-") as temp:
                temp_dir = Path(temp)
                write_outputs(temp_dir, expected)
                for name, data in expected.items():
                    current = args.out / name
                    if not current.exists() or current.read_bytes() != data:
                        errors.append(f"export is stale: {name}")
        if errors:
            for error in errors:
                print(f"FAIL {error}")
            return 1
        print(f"PASS evidence bundle: {args.out}")
        return 0

    outputs, manifest = build_outputs()
    write_outputs(args.out, outputs)
    print(f"wrote {len(manifest['files'])} evidence files to {args.out}")
    print(canonical_json(manifest["source_counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

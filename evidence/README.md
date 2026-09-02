# Sanitized experimental evidence

This directory is generated from the local append-only evaluation ledger and archived ingest
records by `scripts/export_public_evidence.py`. It retains the fields needed to inspect structural
metrics, token accounting, cost growth and run provenance. The exported ledger stops at the
recorded table-analysis cutoff in `results/tables/manifest.json`; later local diagnostic rows are
outside the frozen thesis analysis.

The exporter uses field allowlists. It omits raw prompts, session identifiers, hostnames, absolute
paths, file-operation traces and subscription-specific state. `manifest.json` records the schema,
source counts and SHA-256 digest of every generated JSONL file.

```bash
make evidence-check
python -m wikiorg tables --ledger evidence/ledger.jsonl --out /tmp/wikiorg-tables
```

The sanitized bundle is an inspection artifact. It does not claim that a hosted model will produce
identical generations when rerun.

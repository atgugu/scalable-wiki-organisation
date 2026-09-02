# Scalable wiki organisation

This repository contains the implementation and experimental evidence for a study of wiki
organisation with two different computational designs. The agent baseline rereads and edits a
growing wiki for each new document. The proposed pipeline separates structural organisation from
LLM writing and builds categories and related links with local embeddings, spherical k-means,
c-TF-IDF, cosine neighbours and MMR.

The measured evidence covers 62 successive LLM-wiki ingests and complete pipeline runs through
2,000 documents. Agent ingest cost increased with the state being reread; complete pipeline
rebuilds remained inexpensive over the measured range. Values reported at 1,000 and 10,000
documents for the LLM wiki are extrapolations, not large-corpus LLM experiments.

## Quick start

Python 3.12 or newer is required.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m wikiorg smoke
```

The smoke command runs the deterministic pipeline on a twenty-document fixture without an LLM
account; the embedding model is downloaded on its first run. Agent-backed commands require the
corresponding provider CLI and are not run by CI.

## Main commands

```bash
python -m wikiorg data dbpedia --help
python -m wikiorg data wikics --help
python -m wikiorg run-a --help
python -m wikiorg eval --help
python -m wikiorg tables --help
make evidence-check
```

Generated wikis, downloaded text, embeddings and model caches are deliberately excluded. Set
`WIKIORG_RUNS_ROOT` to place local LLM run repositories outside the project tree.

## Evidence

- `evidence/ledger.jsonl` contains sanitized evaluation rows.
- `evidence/ingests/` contains sanitized ingest-level cost records.
- `evidence/manifest.json` records counts, schemas and SHA-256 hashes.
- `results/tables/` contains the derived numerical tables used in the analysis.
- `figures/` contains plots generated from those tables.
- `data/**/manifest.json`, `gold.jsonl` and `orders/` identify the evaluated corpus subsets without
  redistributing downloaded article text.

Run `make evidence-check` to verify the evidence hashes, privacy boundary and table consistency.
The standard ledger tables can be rebuilt with
`python -m wikiorg tables --ledger evidence/ledger.jsonl --out /tmp/wikiorg-tables`; CI checks
that these regenerated CSV files are byte-identical to the frozen copies. The full local run
directories are not needed to inspect the reported measurements.

## Reproducibility boundaries

Fixed corpus bytes, configuration and seed reproduce the deterministic structural state. Different
hardware may introduce floating-point changes near a cluster boundary when embeddings are
recomputed. Hosted LLM runs reproduce a protocol, not identical generations; the evidence retains
model identifiers, configuration hashes, tokens, prices, timings and observed wiki sizes.

## Data and licences

DBPedia samples derive from `DeveloperOats/DBPedia_Classes`, whose pinned manifest records a CC0
licence. WikiCS identifiers and labels derive from the Wiki-CS dataset; article text was retrieved
from MediaWiki and is not redistributed here. Consult each corpus manifest and upstream source
before rebuilding a dataset.

Software is released under the [MIT License](LICENSE). Original documentation, figures and
author-produced evidence are released under
[Creative Commons Attribution 4.0](LICENSES/CC-BY-4.0.txt). Third-party data remains under its
upstream terms.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff) and cite release `thesis-artifact-v1.0` when
referring to the experimental snapshot.

PY ?= python3
DOCPY ?= $(PY)
PUBLIC_RELEASE_DIR ?= /tmp/scalable-wiki-organisation-release

.PHONY: install env-check smoke test data-dbpedia data-wikics arm-a arm-aplus arm-c arm-b eval tables figures evidence evidence-check public-snapshot public-check thesis pdf-latex pdf-polished

install:
	$(PY) -m pip install -e ".[dev]"

env-check:
	$(PY) scripts/00_env_check.py

smoke:
	$(PY) -m wikiorg smoke

test:
	$(PY) -m pytest -q

data-dbpedia:
	$(PY) -m wikiorg data dbpedia

data-wikics:
	$(PY) -m wikiorg data wikics

arm-a:
	$(PY) -m wikiorg run-a --config configs/arm_a.yaml

arm-aplus:
	$(PY) -m wikiorg run-aplus --config configs/arm_a.yaml

arm-c:
	$(PY) -m wikiorg run-c --config configs/arm_c.yaml

arm-b:
	$(PY) -m wikiorg run-b --config configs/arm_b.yaml

eval:
	$(PY) -m wikiorg eval

tables:
	$(PY) -m wikiorg tables

figures:
	$(PY) -m wikiorg figures

evidence:
	$(PY) scripts/export_public_evidence.py --out evidence

evidence-check:
	$(PY) scripts/export_public_evidence.py --out evidence --check
	$(PY) scripts/check_public_release.py --root . --evidence-only

public-snapshot: evidence-check
	$(PY) scripts/export_public_snapshot.py --output $(PUBLIC_RELEASE_DIR)

public-check:
	$(PY) scripts/check_public_release.py --root $(PUBLIC_RELEASE_DIR) --regenerate-tables

thesis:
	$(DOCPY) thesis/build.py --check

pdf-latex:
	$(DOCPY) thesis/tools/make_pdf.py --check

pdf-polished:
	$(DOCPY) thesis/tools/make_pdf.py --config thesis/thesis-polished.yaml --jobname thesis-latex-polished --check --submission

pdf-revised:
	$(DOCPY) thesis/tools/make_pdf.py --config thesis/thesis-revised.yaml --jobname thesis-latex-revised --check --submission
	cp thesis/build/thesis-latex-revised.pdf thesis/build/Antoniu_Florian_Gugu_Master_Thesis.pdf

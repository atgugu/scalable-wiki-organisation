"""Same seed + cached embeddings -> byte-identical wiki_state.json (sha256), through the full CLI path."""
import hashlib
import json
from pathlib import Path

import numpy as np

from wikiorg.adapters.arm_a import build_wiki_state
from wikiorg.cli import run_arm_a
from wikiorg.cluster import spherical_kmeans
from wikiorg.data.docs import Doc, read_docs
from wikiorg.embed import EmbeddingCache, embedding_input, load_backend_config, text_sha
from wikiorg.label import ctfidf_terms
from wikiorg.link import related_links
from wikiorg.render import dumps_state

FIXTURE = Path(__file__).parent / "fixtures" / "smoke20"


def _synthetic(seed=0, n=48, d=24, k=4):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    words = [[f"w{c}_{j}" for j in range(12)] for c in range(k)]
    docs, X = [], []
    for i in range(n):
        c = i % k
        X.append(centers[c] + 0.3 * rng.normal(size=d))
        body = " ".join(rng.choice(words[c], size=30)) + " " + " ".join(rng.choice(words[(c + 1) % k], size=5))
        docs.append(Doc(id=f"d{i:03d}", title=f"Doc {i}", body=body))
    X = np.asarray(X)
    return docs, X / np.linalg.norm(X, axis=1, keepdims=True)


def _pipeline(docs, X, seed):
    labels, _, _ = spherical_kmeans(X, 4, seed=seed)
    terms, _ = ctfidf_terms([d.body for d in docs], labels, min_df=1)
    links, _ = related_links(X, lam=0.7)
    return dumps_state(build_wiki_state(docs, labels, terms, links))


def test_pipeline_state_is_deterministic():
    docs, X = _synthetic()
    a = hashlib.sha256(_pipeline(docs, X, 3).encode()).hexdigest()
    b = hashlib.sha256(_pipeline(docs, X, 3).encode()).hexdigest()
    assert a == b
    state = json.loads(_pipeline(docs, X, 3))
    assert set(state) == {"arm", "pages", "categories", "extra_pages"}
    assert len(state["pages"]) == 48 and len(state["categories"]) == 4


def _seed_cache(cache_dir: Path, docs, X, cfg):
    cache = EmbeddingCache(cache_dir, cfg)
    ids = [d.id for d in docs]
    shas = {d.id: text_sha(embedding_input(d, cfg["prefix"])) for d in docs}
    cache.store("smoke20-20", ids, shas, X.astype(np.float32),
                {"n_tokens": {i: 100 for i in ids}, "truncated": {i: False for i in ids},
                 "timing": {"wall_s": 0.5, "gpu": False}})


def test_run_arm_a_identical_across_runs(tmp_path):
    docs = read_docs(FIXTURE)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(len(docs), 32))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    cfg = {"embedding": {"env_json": str(tmp_path / "missing-env.json"), "prefix": "clustering: ",
                         "max_seq_length": 2048, "batch_size": 8},
           "clustering": {"n_init": 10, "max_iter": 100}, "labels": {"min_df": 2},
           "links": {"n_candidates": 10, "n_links": 5}, "output": {"write_wiki": True}}
    emb_cfg = load_backend_config(cfg["embedding"]["env_json"], overrides={"prefix": "clustering: ", "max_seq_length": 2048})
    cache_dir = tmp_path / "cache"
    _seed_cache(cache_dir, docs, X, emb_cfg)
    hashes = []
    for run in ("one", "two"):
        out = tmp_path / run
        dirs = run_arm_a(FIXTURE, cfg, "smoke20", seeds=[2], variants=["oracle", "two_level"], lambdas=[0.7, 1.0],
                         out_root=out, cache_dir=cache_dir, device="cpu", log=lambda *_: None)
        assert len(dirs) == 4
        for d in dirs:
            manifest = json.loads((d / "manifest.json").read_text())
            assert manifest["embedding"]["cache_hit"] is True   # no model was loaded
            assert (d / "wiki" / "index.md").exists() and (d / "timing.json").exists()
        hashes.append([hashlib.sha256((d / "wiki_state.json").read_bytes()).hexdigest() for d in dirs])
    assert hashes[0] == hashes[1]
    assert len(set(hashes[0])) >= 2  # two_level adds parent categories, so states differ across variants

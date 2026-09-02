"""Document embeddings with a cached ``.npy`` store.

The embedding input of a document is ``prefix + doc_markdown(doc)`` — the byte-identical text the
other arms consume (frontmatter with id and title, then the body).  The backend is read from
``configs/env.json`` (written by ``scripts/00_env_check.py``); if the file is missing the default
is nomic-embed-text-v1.5 with ``trust_remote_code``, prefix ``"clustering: "``, 2048 tokens.
Fallback chain when a backend cannot be loaded: eager attention -> ONNX -> bge-base-en-v1.5 ->
a local TF-IDF/SVD embedder (no download; loudly flagged in the run manifest).

Cache: ``<cache_dir>/<model-slug>/<key>.npy`` + ``<key>.json`` (ids, sha256 of each input text,
token counts, truncation flags, timing).  A request is served from any cached file whose ids and
text hashes cover it, so nested corpora (100 ⊂ 300 ⊂ 1000) never re-embed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import warnings
from pathlib import Path

import numpy as np

from .data.docs import Doc, doc_markdown

REPO = Path(__file__).resolve().parents[2]
ENV_JSON = REPO / "configs" / "env.json"
DEFAULT_CACHE_DIR = REPO / "runs" / "embeddings"
NOMIC = "nomic-ai/nomic-embed-text-v1.5"
BGE = "BAAI/bge-base-en-v1.5"
DEFAULT_BACKEND = {
    "model": NOMIC,
    "backend": "torch",
    "trust_remote_code": True,
    "model_kwargs": {},
    "device": None,             # None -> cuda if available else cpu
    "max_seq_length": 2048,
    "prefix": "clustering: ",
    "batch_size": 8,
    "revision": None,
}


def load_backend_config(env_json: str | Path | None = ENV_JSON, overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULT_BACKEND)
    p = Path(env_json) if env_json else None
    if p and p.exists():
        try:
            chosen = json.loads(p.read_text(encoding="utf-8")).get("chosen") or {}
        except ValueError:
            chosen = {}
        for k in ("model", "backend", "trust_remote_code", "model_kwargs", "device", "max_seq_length",
                  "prefix", "batch_size", "revision"):
            if k in chosen and chosen[k] is not None:
                cfg[k] = chosen[k]
        cfg["source"] = str(p)
    else:
        cfg["source"] = "defaults (no env.json)"
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def model_slug(cfg: dict) -> str:
    name = re.sub(r"[^A-Za-z0-9.+-]+", "-", cfg["model"].split("/")[-1])
    return f"{name}_L{cfg['max_seq_length']}"


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_input(doc: Doc, prefix: str) -> str:
    return prefix + doc_markdown(doc)


def _fallback_chain(cfg: dict) -> list[dict]:
    """Primary config -> eager attention -> ONNX -> the same model on CPU (keeps the pre-registered
    model when the GPU is merely busy) -> bge-base -> TF-IDF/SVD stand-in."""
    chain = [dict(cfg)]
    if "nomic" in cfg["model"]:
        if not cfg.get("model_kwargs"):
            chain.append(dict(cfg, model_kwargs={"attn_implementation": "eager"}))
        chain.append(dict(cfg, backend="onnx", device="cpu"))
    if (cfg.get("device") or "cuda") != "cpu":
        chain.append(dict(cfg, device="cpu"))
    chain.append(dict(cfg, model=BGE, backend="torch", trust_remote_code=False, model_kwargs={},
                      prefix="", max_seq_length=min(512, cfg["max_seq_length"])))
    chain.append(dict(cfg, model="tfidf-svd", backend="tfidf", prefix=""))
    return chain


class TfidfSvdEmbedder:
    """Deterministic local stand-in used only when no neural backend loads."""

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed
        self.model_name = "tfidf-svd"
        self.max_seq_length = 10**9
        self.device = "cpu"

    def count_tokens(self, texts: list[str]) -> np.ndarray:
        return np.array([len(t.split()) for t in texts])

    def encode(self, texts: list[str], batch_size: int = 8, normalize: bool = True) -> tuple[np.ndarray, dict]:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        X = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(texts)
        k = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        Z = TruncatedSVD(n_components=k, random_state=self.seed).fit_transform(X)
        if normalize:
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        return Z.astype(np.float32), {"batch_size_used": None, "oom_retries": 0}


class Embedder:
    """SentenceTransformer wrapper: fallback chain, OOM-halving batches, token counting."""

    def __init__(self, cfg: dict | None = None, device: str | None = None, allow_fallback: bool = True,
                 log=print) -> None:
        self.cfg = dict(cfg or load_backend_config())
        if device:
            self.cfg["device"] = device
        self.log = log
        self.attempts: list[dict] = []
        self.model = None
        self.effective: dict = {}
        chain = _fallback_chain(self.cfg) if allow_fallback else [dict(self.cfg)]
        last_exc: Exception | None = None
        for cand in chain:
            try:
                self._load(cand)
                break
            except Exception as exc:  # noqa: BLE001 - walk the fallback chain
                last_exc = exc
                self.attempts.append({"model": cand["model"], "backend": cand["backend"],
                                      "model_kwargs": cand.get("model_kwargs"), "ok": False,
                                      "error": f"{type(exc).__name__}: {exc}"[:300]})
                log(f"embed: backend {cand['model']} [{cand['backend']}] failed: {type(exc).__name__}: {str(exc)[:160]}")
        if self.model is None:
            raise RuntimeError(f"no embedding backend could be loaded: {last_exc}")

    def _load(self, cand: dict) -> None:
        if cand["backend"] == "tfidf":
            self.log("embed: WARNING using the TF-IDF/SVD fallback embedder (no neural model available)")
            self.model = TfidfSvdEmbedder()
            self.effective = dict(cand, device="cpu", dim=self.model.dim)
            self.attempts.append({"model": "tfidf-svd", "backend": "tfidf", "ok": True})
            return
        import torch
        from sentence_transformers import SentenceTransformer

        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        device = cand.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        if cand["backend"] == "onnx":
            device = "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        kwargs: dict = {"trust_remote_code": bool(cand.get("trust_remote_code")), "device": device}
        if cand.get("model_kwargs"):
            kwargs["model_kwargs"] = dict(cand["model_kwargs"])
        if cand.get("revision"):
            kwargs["revision"] = cand["revision"]
        if cand["backend"] == "onnx":
            kwargs["backend"] = "onnx"
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SentenceTransformer(cand["model"], **kwargs)
        model.max_seq_length = int(min(cand["max_seq_length"], model.max_seq_length or cand["max_seq_length"]))
        self.model = model
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        self.effective = dict(cand, device=device, dim=get_dim(),
                              max_seq_length=model.max_seq_length, load_seconds=round(time.perf_counter() - t0, 2))
        self.attempts.append({"model": cand["model"], "backend": cand["backend"],
                              "model_kwargs": cand.get("model_kwargs"), "ok": True, "device": device})
        self.log(f"embed: loaded {cand['model']} [{cand['backend']}] on {device}, "
                 f"max_seq_length={model.max_seq_length}, {self.effective['load_seconds']}s")

    # -- helpers --------------------------------------------------------------------------------
    @property
    def max_seq_length(self) -> int:
        return int(self.effective.get("max_seq_length") or self.cfg["max_seq_length"])

    def count_tokens(self, texts: list[str]) -> np.ndarray:
        if isinstance(self.model, TfidfSvdEmbedder):
            return self.model.count_tokens(texts)
        tok = self.model.tokenizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
            enc = tok(texts, add_special_tokens=True, truncation=False, padding=False,
                      return_attention_mask=False, return_token_type_ids=False, verbose=False)
        return np.array([len(ids) for ids in enc["input_ids"]])

    def encode(self, texts: list[str], batch_size: int | None = None, normalize: bool = True) -> tuple[np.ndarray, dict]:
        if isinstance(self.model, TfidfSvdEmbedder):
            return self.model.encode(texts, normalize=normalize)
        import torch

        bs = int(batch_size or self.cfg.get("batch_size") or 8)
        oom = 0
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X = self.model.encode(texts, batch_size=bs, normalize_embeddings=normalize,
                                          convert_to_numpy=True, show_progress_bar=False)
                return np.asarray(X, dtype=np.float32), {"batch_size_used": bs, "oom_retries": oom}
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # type: ignore[attr-defined]
                if "out of memory" not in str(exc).lower() or bs <= 1:
                    raise
                oom += 1
                bs = max(1, bs // 2)
                self.log(f"embed: CUDA OOM -> retrying with batch_size={bs}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


# -------------------------------------------------------------------------------------------------
# cache
# -------------------------------------------------------------------------------------------------
class EmbeddingCache:
    def __init__(self, cache_dir: str | Path, cfg: dict) -> None:
        self.dir = Path(cache_dir) / model_slug(cfg)
        self.cfg = cfg

    def _sidecars(self) -> list[Path]:
        return sorted(self.dir.glob("*.json")) if self.dir.exists() else []

    def lookup(self, ids: list[str], shas: dict[str, str]) -> tuple[np.ndarray | None, dict]:
        """Rows for ``ids`` assembled from any cached files with matching text hashes."""
        need = set(ids)
        rows: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}
        sources: list[str] = []
        served: dict[str, tuple[int, int]] = {}   # source file -> (rows used, rows in file)
        for side in self._sidecars():
            try:
                m = json.loads(side.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if m.get("prefix") != self.cfg["prefix"] or m.get("max_seq_length") != self.cfg["max_seq_length"] \
                    or m.get("model") != self.cfg["model"]:
                continue
            hits = [i for i in need - set(rows) if i in m.get("index", {}) and m["sha"].get(i) == shas[i]]
            if not hits:
                continue
            npy = side.with_suffix(".npy")
            if not npy.exists():
                continue
            X = np.load(npy, mmap_mode="r")
            for i in hits:
                rows[i] = np.array(X[m["index"][i]], dtype=np.float32)
                meta[i] = {"n_tokens": m.get("n_tokens", {}).get(i), "truncated": m.get("truncated", {}).get(i)}
            sources.append(side.name)
            served[side.name] = (len(hits), len(m.get("ids", [])))
            if len(rows) == len(need):
                break
        if len(rows) < len(need):
            return None, {"missing": sorted(need - set(rows)), "sources": sources, "partial": rows, "meta": meta}
        X = np.stack([rows[i] for i in ids])
        info = {"sources": sources, "served": served, "n_tokens": {i: meta[i]["n_tokens"] for i in ids},
                "truncated": {i: meta[i]["truncated"] for i in ids}}
        return X, info

    def store(self, key: str, ids: list[str], shas: dict[str, str], X: np.ndarray, extra: dict) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        npy = self.dir / f"{key}.npy"
        np.save(npy, np.asarray(X, dtype=np.float32))
        meta = {"model": self.cfg["model"], "prefix": self.cfg["prefix"], "max_seq_length": self.cfg["max_seq_length"],
                "ids": list(ids), "index": {i: n for n, i in enumerate(ids)}, "sha": {i: shas[i] for i in ids}, **extra}
        (self.dir / f"{key}.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
        return npy


def embed_docs(docs: list[Doc], cfg: dict | None = None, cache_dir: str | Path = DEFAULT_CACHE_DIR,
               key: str = "corpus", device: str | None = None, timing=None, log=print,
               embedder: Embedder | None = None) -> tuple[np.ndarray, dict]:
    """Embeddings (unit-norm float32, one row per doc in the given order) + info.

    ``info`` carries: backend, truncation fraction, token statistics, cache hit, embed timing
    (measured now, or as recorded when the cache entry was produced)."""
    cfg = dict(cfg or load_backend_config())
    if device:
        cfg["device"] = device
    prefix = cfg["prefix"]
    texts = [embedding_input(d, prefix) for d in docs]
    ids = [d.id for d in docs]
    shas = {d.id: text_sha(t) for d, t in zip(docs, texts)}
    cache = EmbeddingCache(cache_dir, cfg)
    X, hit = cache.lookup(ids, shas)
    if X is not None:
        n_tok = np.array([hit["n_tokens"][i] or 0 for i in ids])
        trunc = np.array([bool(hit["truncated"][i]) for i in ids])
        # timing recorded with the cache entry (from any source file)
        cached_timing = _cached_timing(cache, hit["sources"], hit.get("served"))
        info = {"model": cfg["model"], "backend": cfg.get("backend"), "prefix": prefix,
                "max_seq_length": cfg["max_seq_length"], "device": cfg.get("device"), "cache_hit": True,
                "cache_sources": hit["sources"], "n_docs": len(ids), "dim": int(X.shape[1]),
                "truncation_fraction": round(float(trunc.mean()), 4) if len(ids) else 0.0,
                "n_truncated": int(trunc.sum()), "tokens_mean": round(float(n_tok.mean()), 1) if len(ids) else 0,
                "tokens_median": float(np.median(n_tok)) if len(ids) else 0, "tokens_max": int(n_tok.max()) if len(ids) else 0,
                "embed_input": "prefix + doc_markdown(doc)", "timing": cached_timing,
                "fallback_used": False}   # lookup only accepts entries of the requested model
        if timing is not None and cached_timing:
            timing.add("embed", wall_s=cached_timing.get("wall_s", 0.0), gpu=cached_timing.get("gpu", False),
                       gpu_event_s=cached_timing.get("gpu_event_s"), peak_vram_mib=cached_timing.get("peak_vram_mib"),
                       cached=True, per_doc_wall_s=cached_timing.get("per_doc_wall_s"))
        return X, info

    emb = embedder or Embedder(cfg, device=cfg.get("device"), log=log)
    eff = emb.effective
    is_gpu = str(eff.get("device", "")).startswith("cuda")
    t0 = time.perf_counter()
    n_tok = emb.count_tokens(texts)
    t_tok = time.perf_counter() - t0
    max_len = emb.max_seq_length
    trunc = n_tok > max_len
    local = timing
    if local is None:
        from .timing import Timing

        local = Timing()
    with local.stage("embed", gpu=is_gpu):
        X, enc_info = emb.encode(texts, batch_size=cfg.get("batch_size"), normalize=True)
    st = local.stages["embed"]
    stage_timing = {"wall_s": round(st["wall_s"], 4), "gpu": st["gpu"], "gpu_event_s": st.get("gpu_event_s"),
                    "peak_vram_mib": st.get("peak_vram_mib"), "per_doc_wall_s": round(st["wall_s"] / max(1, len(ids)), 5),
                    "tokenize_s": round(t_tok, 3), "batch_size_used": enc_info.get("batch_size_used"),
                    "oom_retries": enc_info.get("oom_retries", 0), "model_load_s": eff.get("load_seconds")}
    if timing is not None:
        timing.stages["embed"]["per_doc_wall_s"] = stage_timing["per_doc_wall_s"]
    info = {"model": eff.get("model"), "backend": eff.get("backend"), "model_kwargs": eff.get("model_kwargs"),
            "prefix": prefix, "max_seq_length": max_len, "device": eff.get("device"), "dim": int(X.shape[1]),
            "cache_hit": False, "n_docs": len(ids),
            "truncation_fraction": round(float(trunc.mean()), 4) if len(ids) else 0.0, "n_truncated": int(trunc.sum()),
            "tokens_mean": round(float(n_tok.mean()), 1) if len(ids) else 0,
            "tokens_median": float(np.median(n_tok)) if len(ids) else 0, "tokens_max": int(n_tok.max()) if len(ids) else 0,
            "embed_input": "prefix + doc_markdown(doc)", "timing": stage_timing, "attempts": emb.attempts,
            "fallback_used": eff.get("model") != cfg["model"]}
    cache.cfg = dict(cfg, model=eff.get("model"), max_seq_length=max_len, prefix=prefix)
    cache = EmbeddingCache(cache_dir, cache.cfg)
    cache.store(key, ids, shas, X, {"n_tokens": {i: int(n) for i, n in zip(ids, n_tok)},
                                    "truncated": {i: bool(t) for i, t in zip(ids, trunc)},
                                    "timing": stage_timing, "backend": eff.get("backend"),
                                    "device": eff.get("device"), "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"embed: {len(ids)} docs, dim {X.shape[1]}, truncated {int(trunc.sum())} "
        f"({info['truncation_fraction']:.1%}), {stage_timing['wall_s']:.1f}s on {eff.get('device')}")
    return X, info


def _cached_timing(cache: EmbeddingCache, sources: list[str], served: dict | None = None) -> dict:
    """Embed timing attributed to a cache hit: each source file's wall / GPU seconds scaled by the
    fraction of its rows that served the lookup (a 100-doc corpus served from a 2000-doc cache
    file is charged 1/20 of that file's embedding time)."""
    agg = {"wall_s": 0.0, "gpu": False, "gpu_event_s": 0.0, "peak_vram_mib": 0.0, "per_doc_wall_s": None}
    for name in sources:
        try:
            m = json.loads((cache.dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        t = m.get("timing") or {}
        used, total = (served or {}).get(name, (None, None))
        frac = (used / total) if used and total else 1.0
        agg["wall_s"] += float(t.get("wall_s") or 0.0) * frac
        agg["gpu"] = agg["gpu"] or bool(t.get("gpu"))
        agg["gpu_event_s"] += float(t.get("gpu_event_s") or 0.0) * frac
        agg["peak_vram_mib"] = max(agg["peak_vram_mib"], float(t.get("peak_vram_mib") or 0.0))
        if t.get("per_doc_wall_s") is not None:
            agg["per_doc_wall_s"] = float(t["per_doc_wall_s"])
    agg["note"] = "timing attributed from the cache entries that served this run, scaled by rows used"
    return agg

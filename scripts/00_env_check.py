#!/usr/bin/env python
"""Verify the embedding stack on this machine and record what actually works.

Run with: python scripts/00_env_check.py

Tries, in order: nomic-embed-text-v1.5 (sdpa), nomic (eager attention), nomic (ONNX/CPU),
bge-base-en-v1.5. Writes the first configuration that passes a semantic sanity check to
configs/env.json, together with peak VRAM and encode throughput.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "configs" / "env.json"

# Sanity triple: the first two sentences are about the same thing, the third is not.
PROBE = [
    "The Danube is a river in Central Europe that flows into the Black Sea.",
    "Rivers of Europe include the Rhine, the Danube and the Volga.",
    "Association football is a team sport played with a spherical ball.",
]
LONG_DOC = ("Wikipedia is a free content online encyclopedia written and maintained by a community "
            "of volunteers through open collaboration and a wiki-based editing system. ") * 120


def _candidates():
    yield {"model": "nomic-ai/nomic-embed-text-v1.5", "backend": "torch",
           "trust_remote_code": True, "model_kwargs": {}}
    yield {"model": "nomic-ai/nomic-embed-text-v1.5", "backend": "torch",
           "trust_remote_code": True, "model_kwargs": {"attn_implementation": "eager"}}
    yield {"model": "nomic-ai/nomic-embed-text-v1.5", "backend": "onnx",
           "trust_remote_code": True, "model_kwargs": {}}
    yield {"model": "BAAI/bge-base-en-v1.5", "backend": "torch",
           "trust_remote_code": False, "model_kwargs": {}}


def try_candidate(cand: dict, max_seq: int = 2048, batch_size: int = 8) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if (cand["backend"] == "torch" and torch.cuda.is_available()) else "cpu"
    kwargs = dict(trust_remote_code=cand["trust_remote_code"], device=device)
    if cand["model_kwargs"]:
        kwargs["model_kwargs"] = cand["model_kwargs"]
    if cand["backend"] == "onnx":
        kwargs["backend"] = "onnx"
        kwargs["device"] = "cpu"

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = SentenceTransformer(cand["model"], **kwargs)
    model.max_seq_length = min(max_seq, model.max_seq_length or max_seq)

    prefix = "clustering: " if "nomic" in cand["model"] else ""
    emb = model.encode([prefix + t for t in PROBE], normalize_embeddings=True,
                       batch_size=batch_size, show_progress_bar=False)
    same = float(emb[0] @ emb[1])
    diff = float(emb[0] @ emb[2])
    if not same > diff:
        raise RuntimeError(f"semantic sanity failed: same={same:.3f} !> diff={diff:.3f}")

    t0 = time.perf_counter()
    model.encode([prefix + LONG_DOC] * batch_size, normalize_embeddings=True,
                 batch_size=batch_size, show_progress_bar=False)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None
    return {
        **{k: v for k, v in cand.items()},
        "device": device,
        "max_seq_length": model.max_seq_length,
        "dim": model.get_sentence_embedding_dimension(),
        "cos_same": round(same, 4),
        "cos_diff": round(diff, 4),
        "peak_vram_mib": None if peak is None else round(peak, 1),
        "long_batch_seconds": round(dt, 3),
        "batch_size": batch_size,
        "prefix": prefix,
    }


def main() -> int:
    import torch

    report = {
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "attempts": [],
        "chosen": None,
    }
    for cand in _candidates():
        label = f"{cand['model']} [{cand['backend']}{'/eager' if cand['model_kwargs'] else ''}]"
        try:
            res = try_candidate(cand)
            report["attempts"].append({"candidate": label, "ok": True, **res})
            report["chosen"] = res
            print(f"OK   {label}: cos_same={res['cos_same']} cos_diff={res['cos_diff']} "
                  f"vram={res['peak_vram_mib']} MiB seq={res['max_seq_length']}")
            break
        except Exception as exc:  # noqa: BLE001 - we want the whole fallback chain
            report["attempts"].append({"candidate": label, "ok": False,
                                       "error": f"{type(exc).__name__}: {exc}"[:400]})
            print(f"FAIL {label}: {type(exc).__name__}: {str(exc)[:200]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    return 0 if report["chosen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

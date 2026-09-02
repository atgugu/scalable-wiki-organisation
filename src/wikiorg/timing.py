"""Per-stage wall-clock, GPU seconds (CUDA events), peak VRAM and cloud-equivalent USD.

Cloud-equivalent cost (``configs/prices.yaml`` -> ``compute``): stages flagged ``gpu=True`` are
billed on their wall-clock at the GPU rate ($0.35/GPU-h, T4 on-demand reference), every other
stage on its wall-clock at the CPU rate ($0.05/CPU-h).  Event-measured GPU seconds (time between
two CUDA events around the stage, after ``cuda.synchronize``) and peak allocated VRAM are recorded
alongside so the raw numbers can be re-priced.
"""
from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PRICES = REPO / "configs" / "prices.yaml"


def load_prices(path: str | Path | None = DEFAULT_PRICES) -> dict:
    p = Path(path) if path else DEFAULT_PRICES
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    data.setdefault("fetched", None)
    data.setdefault("compute", {})
    data["compute"].setdefault("gpu_usd_per_hour", 0.35)
    data["compute"].setdefault("cpu_usd_per_hour", 0.05)
    return data


class Timing:
    def __init__(self, prices_path: str | Path | None = DEFAULT_PRICES) -> None:
        self.prices = load_prices(prices_path)
        self.stages: dict[str, dict] = {}

    @contextmanager
    def stage(self, name: str, gpu: bool = False):
        torch = None
        use_gpu = False
        if gpu:
            try:
                import torch  # type: ignore

                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        ev0 = ev1 = None
        if use_gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        try:
            yield
        finally:
            gpu_s = peak = None
            if use_gpu:
                ev1.record()
                torch.cuda.synchronize()
                gpu_s = ev0.elapsed_time(ev1) / 1000.0
                peak = torch.cuda.max_memory_allocated() / 2**20
            self.add(name, wall_s=time.perf_counter() - t0, gpu=gpu and use_gpu,
                     gpu_event_s=gpu_s, peak_vram_mib=peak, cpu_process_s=time.process_time() - cpu0)

    def add(self, name: str, wall_s: float, gpu: bool = False, gpu_event_s: float | None = None,
            peak_vram_mib: float | None = None, cpu_process_s: float | None = None,
            cached: bool = False, **extra) -> None:
        """Record (or accumulate into) a stage."""
        rec = self.stages.get(name)
        if rec is None:
            rec = {"wall_s": 0.0, "gpu": bool(gpu), "gpu_event_s": None, "peak_vram_mib": None,
                   "cpu_process_s": None, "cached": cached, "calls": 0}
            self.stages[name] = rec
        rec["wall_s"] += float(wall_s)
        rec["gpu"] = rec["gpu"] or bool(gpu)
        rec["calls"] += 1
        rec["cached"] = rec["cached"] and cached if rec["calls"] > 1 else cached
        if gpu_event_s is not None:
            rec["gpu_event_s"] = (rec["gpu_event_s"] or 0.0) + float(gpu_event_s)
        if peak_vram_mib is not None:
            rec["peak_vram_mib"] = max(rec["peak_vram_mib"] or 0.0, float(peak_vram_mib))
        if cpu_process_s is not None:
            rec["cpu_process_s"] = (rec["cpu_process_s"] or 0.0) + float(cpu_process_s)
        rec.update(extra)

    def clone(self, stages: list[str] | None = None) -> "Timing":
        """Copy (optionally only some stages) so per-run timings share the embed stage."""
        import copy

        t = Timing.__new__(Timing)
        t.prices = self.prices
        t.stages = {k: copy.deepcopy(v) for k, v in self.stages.items() if stages is None or k in stages}
        return t

    def summary(self) -> dict:
        comp = self.prices["compute"]
        gpu_wall = sum(s["wall_s"] for s in self.stages.values() if s["gpu"])
        cpu_wall = sum(s["wall_s"] for s in self.stages.values() if not s["gpu"])
        gpu_event = sum(s["gpu_event_s"] or 0.0 for s in self.stages.values())
        peak = max((s["peak_vram_mib"] or 0.0 for s in self.stages.values()), default=0.0)
        usd = gpu_wall / 3600 * comp["gpu_usd_per_hour"] + cpu_wall / 3600 * comp["cpu_usd_per_hour"]
        return {
            "stages": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                       for k, v in self.stages.items()},
            "total_wall_s": round(gpu_wall + cpu_wall, 4),
            "gpu_stage_wall_s": round(gpu_wall, 4),
            "cpu_stage_wall_s": round(cpu_wall, 4),
            "gpu_event_s": round(gpu_event, 4),
            "peak_vram_mib": round(peak, 1),
            "usd_cloud_equivalent": round(usd, 6),
            "price_basis": {"gpu_usd_per_hour": comp["gpu_usd_per_hour"],
                            "cpu_usd_per_hour": comp["cpu_usd_per_hour"],
                            "price_table_fetched": self.prices.get("fetched"),
                            "rule": "GPU stages billed on wall-clock at the GPU rate, other stages at the CPU rate"},
            "host": platform.node(),
        }

    def write(self, path: str | Path) -> dict:
        s = self.summary()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        return s

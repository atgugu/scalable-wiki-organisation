"""wikiorg command line.

    wikiorg data dbpedia|wikics [...]        build a corpus under data/<corpus>/<N>/
    wikiorg embed --corpus C --n N            compute + cache embeddings, report truncation
    wikiorg run-a [--config configs/arm_a.yaml] [--corpus C] [--n N] [--seeds 1,2] [--variants ...]
    wikiorg smoke                             whole arm-A pipeline + render on tests/fixtures/smoke20 (CPU)
    wikiorg run-aplus | run-c | run-b | extract | eval | stability | tables | figures
                                              dispatched to their owning modules (W2 / W3) when present

Run directory (arm A): runs/A/<corpus>/<N>/s<seed>_<variant>_lam<lambda>/
    config.yaml  manifest.json  wiki_state.json  timing.json  wiki/{index.md,categories/,pages/,graph.json}
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .data.docs import Doc, read_docs, read_gold, read_manifest, sha256_text

REPO = Path(__file__).resolve().parents[2]
OWNED_ELSEWHERE = {
    "run-aplus": ("wikiorg.llm.arm_aplus", "W2 (LLM harness)"),
    "run-c": ("wikiorg.llm.arm_c", "W2 (LLM harness)"),
    "run-b": ("wikiorg.llm.arm_b", "W2 (LLM harness)"),
    "extract": ("wikiorg.llm.extract", "W2 (LLM harness)"),
    "eval": ("wikiorg.eval.report", "W3 (evaluation)"),
    "stability": ("wikiorg.eval.stability", "W3 (evaluation)"),
    "tables": ("wikiorg.eval.report", "W3 (evaluation)"),
    "figures": ("wikiorg.eval.figures", "W3 (evaluation)"),
}
VARIANTS = ("oracle", "auto", "two_level", "k_matched")


def log(msg: str) -> None:
    print(msg, flush=True)


# -------------------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO / p
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def config_hash(cfg: dict) -> str:
    return sha256_text(yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True))


def git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions() -> dict:
    out = {}
    for m in ("torch", "sentence_transformers", "transformers", "sklearn", "numpy", "scipy", "datasets", "networkx"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def _parse_list(s: str | None, cast=str) -> list | None:
    if s is None:
        return None
    return [cast(x) for x in str(s).replace(";", ",").split(",") if x.strip()]


def resolve_corpus_dir(corpus: str, n: int) -> Path:
    p = Path(corpus)
    if p.is_dir() and (p / "docs").is_dir():
        return p
    return REPO / "data" / corpus / str(n)


def gold_levels(gold: dict, cfg_corpus: dict | None) -> tuple[str, str | None]:
    """(gold level used for k*, top level for two-level) inferred from the gold fields."""
    sample = next(iter(gold.values())) if gold else {}
    level = (cfg_corpus or {}).get("gold_level")
    top = (cfg_corpus or {}).get("top_level") if cfg_corpus else None
    if level is None:
        level = "l2" if "l2" in sample else ("class" if "class" in sample else next(iter(sample), None))
    if cfg_corpus is None:
        top = "l1" if "l1" in sample else None
    return level, top


def sanity_metrics(labels: np.ndarray, gold: dict, ids: list[str], level: str) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    y = [str(gold[i].get(level)) for i in ids]
    return {"nmi": round(float(normalized_mutual_info_score(y, labels)), 4),
            "ari": round(float(adjusted_rand_score(y, labels)), 4), "level": level,
            "note": "informational only; wikiorg.eval (W3) is authoritative"}


# -------------------------------------------------------------------------------------------------
# arm A
# -------------------------------------------------------------------------------------------------
def run_arm_a(corpus_dir: str | Path, cfg: dict, corpus_name: str, n: int | None = None,
              seeds: list[int] | None = None, variants: list[str] | None = None, lambdas: list[float] | None = None,
              k_override: int | None = None, device: str | None = None, out_root: str | Path | None = None,
              cache_dir: str | Path | None = None, write_wiki: bool | None = None, arm: str = "A",
              allow_tfidf_fallback: bool = False, log=log) -> list[Path]:
    from .adapters.arm_a import build_wiki_state
    from .cluster import choose_k_silhouette, spherical_kmeans, two_level
    from .embed import embed_docs, load_backend_config
    from .label import ctfidf_terms
    from .link import related_links
    from .render import check_links, write_wiki as render_wiki, write_wiki_state
    from .timing import Timing

    corpus_dir = Path(corpus_dir)
    docs = read_docs(corpus_dir)
    gold = read_gold(corpus_dir)
    corpus_manifest = read_manifest(corpus_dir)
    n = n or len(docs)
    ids = [d.id for d in docs]
    cfg_corpus = (cfg.get("corpora") or {}).get(corpus_name)
    level, top_level = gold_levels(gold, cfg_corpus)
    k_star = len({str(g.get(level)) for g in gold.values()})
    k1 = len({str(g.get(top_level)) for g in gold.values()}) if top_level else None
    ccfg = cfg.get("clustering", {})
    lcfg = cfg.get("labels", {})
    kcfg = cfg.get("links", {})
    ecfg = cfg.get("embedding", {})
    seeds = seeds or list(ccfg.get("seeds", [1]))
    variants = variants or list(ccfg.get("variants", ["oracle"]))
    lambdas = lambdas or list(kcfg.get("lambdas", [0.7]))
    n_init = int(ccfg.get("n_init", 10))
    max_iter = int(ccfg.get("max_iter", 100))
    out_root = Path(out_root or (REPO / cfg.get("output", {}).get("runs_root", "runs")))
    write_wiki = cfg.get("output", {}).get("write_wiki", True) if write_wiki is None else write_wiki
    cache_dir = Path(cache_dir or (REPO / ecfg.get("cache_dir", "runs/embeddings")))
    env_json = ecfg.get("env_json")
    env_json = (REPO / env_json) if env_json and not Path(env_json).is_absolute() else env_json
    emb_cfg = load_backend_config(env_json, overrides={"prefix": ecfg.get("prefix"),
                                                       "max_seq_length": ecfg.get("max_seq_length"),
                                                       "batch_size": ecfg.get("batch_size"), "device": device})
    log(f"arm {arm}: {corpus_name} N={n} ({len(docs)} docs), k*={k_star} on '{level}'"
        + (f", k1={k1} on '{top_level}'" if k1 else "") + f", seeds={seeds}, variants={variants}, lambdas={lambdas}")

    shared = Timing()
    with shared.stage("load"):
        texts = [f"{d.title}\n{d.body}" for d in docs]
    X, emb_info = embed_docs(docs, cfg=emb_cfg, cache_dir=cache_dir, key=f"{corpus_name}-{n}", device=device,
                             timing=shared, log=log)
    log(f"embedding: {emb_info['model']} on {emb_info.get('device')}, truncated {emb_info['n_truncated']}/{len(docs)} "
        f"({emb_info['truncation_fraction']:.1%}), tokens mean {emb_info['tokens_mean']}, cache_hit={emb_info['cache_hit']}")
    if emb_info.get("backend") == "tfidf" and not allow_tfidf_fallback:
        raise RuntimeError("no neural embedding backend could be loaded (see the messages above); refusing to write "
                           "arm-A runs with the TF-IDF/SVD stand-in. Free the GPU or pass --device cpu; "
                           "--allow-tfidf-fallback overrides this for debugging only.")

    km = (ccfg.get("k_matched") or {}).get(corpus_name, {}) or {}
    k_matched = k_override if k_override is not None else km.get(n, km.get(str(n)))
    run_dirs: list[Path] = []
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if variant == "k_matched" and not k_matched:
            log(f"variant k_matched skipped for {corpus_name}/{n}: no k_B value (set clustering.k_matched or --k)")
            continue
        if variant == "two_level" and not k1:
            log(f"variant two_level skipped for {corpus_name}: no top-level gold to size k1")
            continue
        for seed in seeds:
            t = shared.clone()
            parents = parent_terms = None
            curve = None
            with t.stage("cluster"):
                if variant == "oracle":
                    k = k_star
                    labels, _, cinfo = spherical_kmeans(X, k, seed=seed, n_init=n_init, max_iter=max_iter)
                elif variant == "auto":
                    k, curve = choose_k_silhouette(X, k_star, seed=seed, n_init=n_init, max_iter=max_iter)
                    labels, _, cinfo = spherical_kmeans(X, k, seed=seed, n_init=n_init, max_iter=max_iter)
                elif variant == "k_matched":
                    k = int(k_matched)
                    labels, _, cinfo = spherical_kmeans(X, k, seed=seed, n_init=n_init, max_iter=max_iter)
                else:  # two_level
                    labels, top, parents, cinfo = two_level(X, int(k1), k_star, seed=seed, n_init=n_init, max_iter=max_iter)
                    k = int(labels.max()) + 1
            with t.stage("label"):
                terms, linfo = ctfidf_terms(texts, labels, top_n=int(lcfg.get("top_terms", 10)),
                                            ngram_range=tuple(lcfg.get("ngram_range", (1, 2))),
                                            stop_words=lcfg.get("stop_words", "english"), min_df=int(lcfg.get("min_df", 2)))
                if parents is not None:
                    parent_terms, _ = ctfidf_terms(texts, top, top_n=int(lcfg.get("top_terms", 10)),
                                                   ngram_range=tuple(lcfg.get("ngram_range", (1, 2))),
                                                   stop_words=lcfg.get("stop_words", "english"), min_df=int(lcfg.get("min_df", 2)))
            sanity = sanity_metrics(labels, gold, ids, level)
            for lam in lambdas:
                tl = t.clone()
                with tl.stage("link"):
                    links, sims = related_links(X, n_candidates=int(kcfg.get("n_candidates", 10)),
                                                n_links=int(kcfg.get("n_links", 5)), lam=float(lam))
                run_name = f"s{seed}_{variant}_lam{lam}"
                run_dir = out_root / arm / corpus_name / str(n) / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                with tl.stage("render"):
                    state = build_wiki_state(docs, labels, terms, links, parents=parents, parent_terms=parent_terms, arm=arm,
                                             label_terms=int(lcfg.get("label_terms", 3)), top_terms=int(lcfg.get("top_terms", 10)))
                    problems = state.validate()
                    if problems:
                        raise RuntimeError(f"invalid wiki state: {problems[:5]}")
                    write_wiki_state(state, run_dir / "wiki_state.json")
                    if write_wiki:
                        render_wiki(state, docs, run_dir / "wiki", title=f"Arm {arm} wiki — {corpus_name} N={n} ({variant}, seed {seed}, λ={lam})")
                        broken = check_links(run_dir / "wiki")
                        if broken:
                            raise RuntimeError(f"broken wiki links: {broken[:5]}")
                effective = copy.deepcopy(cfg)
                effective.update({"run": {"arm": arm, "corpus": corpus_name, "n": n, "seed": seed, "variant": variant,
                                          "k": int(k), "lambda": float(lam), "k_star": k_star, "k1": k1,
                                          "device": emb_info.get("device"), "embedding_model": emb_info.get("model")}})
                (run_dir / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=True, allow_unicode=True), encoding="utf-8")
                timing_summary = tl.write(run_dir / "timing.json")
                manifest = {
                    "run_id": f"{arm}_{corpus_name}_N{n}_{run_name}",
                    "arm": arm, "corpus": corpus_name, "n": n, "seed": seed, "variant": variant, "k": int(k),
                    "lambda": float(lam), "k_star": k_star, "k1": k1, "gold_level": level, "top_level": top_level,
                    "n_docs": len(docs), "n_categories": len(state.categories),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "git_hash": git_hash(), "config_hash": config_hash(effective),
                    "config_file_hash": config_hash(cfg),
                    "corpus_manifest_sha256": sha256_text(json.dumps(corpus_manifest, sort_keys=True)) if corpus_manifest else None,
                    "corpus_sha256": corpus_manifest.get("sha256"),
                    "embedding": {k_: v_ for k_, v_ in emb_info.items() if k_ not in ("timing",)},
                    "clustering": {"variant": variant, "k": int(k), "n_init": n_init, "max_iter": max_iter,
                                   "info": _jsonable(cinfo), "silhouette_curve": curve},
                    "labels": linfo, "links": {"n_candidates": int(kcfg.get("n_candidates", 10)),
                                               "n_links": int(kcfg.get("n_links", 5)), "lambda": float(lam)},
                    "sanity": sanity, "usd_cloud_equivalent": timing_summary["usd_cloud_equivalent"],
                    "total_wall_s": timing_summary["total_wall_s"],
                    "packages": package_versions(), "python": sys.version.split()[0], "gpu": gpu_name(),
                    "host": platform.node(), "price_table_fetched": tl.prices.get("fetched"),
                    "claude_md_sha256": _file_sha(REPO / "src" / "wikiorg" / "llm" / "schema" / "CLAUDE.md"),
                    "llm_model": None, "llm_calls": 0,
                }
                (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                run_dirs.append(run_dir)
                log(f"  {run_dir.relative_to(out_root)}: k={k} nmi_{level}={sanity['nmi']} ari={sanity['ari']} "
                    f"wall={timing_summary['total_wall_s']:.1f}s usd={timing_summary['usd_cloud_equivalent']:.5f}")
    return run_dirs


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# -------------------------------------------------------------------------------------------------
# commands
# -------------------------------------------------------------------------------------------------
def cmd_data(args) -> int:
    if args.corpus == "dbpedia":
        from .data import dbpedia

        sizes = tuple(_parse_list(args.sizes, int) or dbpedia.SIZES)
        dbpedia.build(args.out or (REPO / "data" / "dbpedia"), seed=args.seed or dbpedia.SEED, sizes=sizes, log=log)
    else:
        from .data import wikics

        sizes = tuple(_parse_list(args.sizes, int) or wikics.SIZES)
        wikics.build(args.raw or (REPO / "data" / "wikics" / "raw"), args.out or (REPO / "data" / "wikics"),
                     sizes=sizes, pool_size=args.pool or wikics.POOL_SIZE, rate=args.rate,
                     seed=args.seed or wikics.SEED, max_pages=args.max_pages, log=log)
    return 0


def cmd_embed(args) -> int:
    from .embed import embed_docs, load_backend_config
    from .timing import Timing

    cfg = load_config(args.config)
    ecfg = cfg.get("embedding", {})
    corpus_dir = resolve_corpus_dir(args.corpus, args.n)
    docs = read_docs(corpus_dir)
    env_json = ecfg.get("env_json")
    env_json = (REPO / env_json) if env_json and not Path(env_json).is_absolute() else env_json
    emb_cfg = load_backend_config(env_json, overrides={"prefix": ecfg.get("prefix"), "max_seq_length": ecfg.get("max_seq_length"),
                                                       "batch_size": ecfg.get("batch_size"), "device": args.device})
    t = Timing()
    X, info = embed_docs(docs, cfg=emb_cfg, cache_dir=args.cache_dir or (REPO / ecfg.get("cache_dir", "runs/embeddings")),
                         key=f"{args.corpus}-{args.n}", device=args.device, timing=t, log=log)
    info["timing_summary"] = t.summary()
    print(json.dumps({k: v for k, v in info.items() if k != "attempts"}, indent=2, default=str))
    return 0


def cmd_run_a(args) -> int:
    cfg = load_config(args.config)
    corpora = cfg.get("corpora", {})
    targets: list[tuple[str, int]] = []
    if args.corpus:
        sizes = _parse_list(args.n, int) or list(corpora.get(args.corpus, {}).get("sizes", []))
        if not sizes:
            raise SystemExit("--n is required for a corpus that is not in the config")
        targets = [(args.corpus, n) for n in sizes]
    else:
        for c, spec in corpora.items():
            targets += [(c, n) for n in spec.get("sizes", [])]
    all_runs = []
    for corpus, n in targets:
        corpus_dir = resolve_corpus_dir(corpus, n)
        if not corpus_dir.is_dir():
            log(f"skip {corpus}/{n}: {corpus_dir} does not exist (run `wikiorg data {corpus}` first)")
            continue
        all_runs += run_arm_a(corpus_dir, cfg, corpus, n, seeds=_parse_list(args.seeds, int),
                              variants=_parse_list(args.variants), lambdas=_parse_list(args.lambdas, float),
                              k_override=args.k, device=args.device, out_root=args.out_root,
                              cache_dir=args.cache_dir, write_wiki=(False if args.no_wiki else None),
                              allow_tfidf_fallback=args.allow_tfidf_fallback)
    log(f"arm A: {len(all_runs)} runs written")
    return 0


def cmd_smoke(args) -> int:
    t0 = time.perf_counter()
    fixture = Path(args.fixture)
    out_root = Path(args.out)
    cfg = load_config(args.config)
    cfg.setdefault("clustering", {})["n_init"] = 10
    run_dirs = run_arm_a(fixture, cfg, "smoke20", n=None, seeds=[1], variants=["oracle"], lambdas=[0.7],
                         device=args.device, out_root=out_root, cache_dir=out_root / "embeddings", write_wiki=True,
                         allow_tfidf_fallback=True)
    run_dir = run_dirs[0]
    from .render import check_links, read_wiki_state

    state = read_wiki_state(run_dir / "wiki_state.json")
    broken = check_links(run_dir / "wiki")
    assert not broken, broken
    assert len(state.pages) == 20 and len(state.categories) == 4, (len(state.pages), len(state.categories))
    for cid, c in state.categories.items():
        log(f"  category {cid}: {c['label']}")
    # optional evaluation (owned by W3)
    eval_note = "skipped: wikiorg.eval.report (W3) has no evaluate_run/main"
    try:
        report = importlib.import_module("wikiorg.eval.report")
        fn = getattr(report, "evaluate_run", None) or getattr(report, "main", None)
        if fn is not None:
            res = fn(str(run_dir)) if fn.__name__ == "evaluate_run" else fn(["--run", str(run_dir)])
            eval_note = f"ok: {str(res)[:200]}"
    except ModuleNotFoundError as exc:
        eval_note = f"skipped: {exc} (wikiorg.eval is workstream W3)"
    except Exception as exc:  # noqa: BLE001 - eval is another workstream; never fail the smoke on it
        eval_note = f"error in wikiorg.eval (W3): {type(exc).__name__}: {exc}"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    log(f"eval: {eval_note}")
    log(f"SMOKE OK: {run_dir} | nmi={manifest['sanity']['nmi']} | wiki links resolve | "
        f"{time.perf_counter() - t0:.1f}s total on {args.device}")
    return 0


def cmd_owned_elsewhere(args) -> int:
    module, owner = OWNED_ELSEWHERE[args.cmd]
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        print(f"wikiorg {args.cmd}: not implemented in this workstream. Owned by {owner}: module `{module}` "
              f"(expected to expose main(argv)).", file=sys.stderr)
        return 2
    fn = getattr(mod, "main", None)
    if fn is None:
        print(f"wikiorg {args.cmd}: module `{module}` exists but has no main(argv); owned by {owner}.", file=sys.stderr)
        return 2
    return int(fn(list(args.args)) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wikiorg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("data", help="build a corpus")
    p.add_argument("corpus", choices=["dbpedia", "wikics"])
    p.add_argument("--out")
    p.add_argument("--raw", help="wikics: directory with data.json and metadata.json")
    p.add_argument("--sizes", help="comma-separated, e.g. 100,300,1000,2000")
    p.add_argument("--seed", type=int)
    p.add_argument("--pool", type=int, help="wikics: snowball pool size (1500)")
    p.add_argument("--rate", type=float, default=1.0, help="wikics: MediaWiki requests per second")
    p.add_argument("--max-pages", type=int, help="wikics: debug, fetch only the first N pool pages")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("embed", help="compute and cache embeddings")
    p.add_argument("--config", default="configs/arm_a.yaml")
    p.add_argument("--corpus", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--device")
    p.add_argument("--cache-dir")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("run-a", help="arm A: cluster, label, link, render")
    p.add_argument("--config", default="configs/arm_a.yaml")
    p.add_argument("--corpus")
    p.add_argument("--n", help="comma-separated sizes")
    p.add_argument("--seeds")
    p.add_argument("--variants", help=f"comma-separated subset of {','.join(VARIANTS)}")
    p.add_argument("--lambdas")
    p.add_argument("--k", type=int, help="k for the k_matched variant")
    p.add_argument("--device")
    p.add_argument("--out-root")
    p.add_argument("--cache-dir")
    p.add_argument("--no-wiki", action="store_true", help="write wiki_state.json only")
    p.add_argument("--allow-tfidf-fallback", action="store_true",
                   help="debugging only: accept the TF-IDF/SVD stand-in when no neural backend loads")
    p.set_defaults(func=cmd_run_a)

    p = sub.add_parser("smoke", help="arm A end-to-end on the 20-document fixture")
    p.add_argument("--config", default="configs/arm_a.yaml")
    p.add_argument("--device", default="cpu")
    p.add_argument("--fixture", default=str(REPO / "tests" / "fixtures" / "smoke20"))
    p.add_argument("--out", default=str(REPO / "runs" / "smoke"))
    p.set_defaults(func=cmd_smoke)

    for name, (module, owner) in OWNED_ELSEWHERE.items():
        p = sub.add_parser(name, help=f"owned by {owner}: {module}")
        p.set_defaults(func=cmd_owned_elsewhere)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # parse_known_args (not parse_args) so pass-through subcommands (OWNED_ELSEWHERE) can forward
    # arbitrary flags -- e.g. `wikiorg eval --run DIR` -- to the owning module's own main(argv)
    # without argparse.REMAINDER's subparser/option-string ambiguity (previously required a
    # `wikiorg eval -- --run DIR` workaround; see configs/prereg.yaml analysis-freeze notes).
    args, extra = parser.parse_known_args(argv)
    if args.cmd in OWNED_ELSEWHERE:
        args.args = extra
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

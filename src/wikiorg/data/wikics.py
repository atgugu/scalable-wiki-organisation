"""WikiCS link corpus (Mernyei & Cangea 2020): hyperlink graph with 10 CS classes.

Raw inputs (``data/wikics/raw/``): ``data.json`` (``labels``, ``links`` = outgoing neighbour
indices, 300-d GloVe features that we ignore) and ``metadata.json`` (``nodes[i].id`` = Wikipedia
pageid, ``title``, ``label`` name).  Text is re-fetched by pageid from the MediaWiki API because
the dataset ships only token lists.

Subset selection (deterministic):
1. undirected union graph, self-loops dropped;
2. seeds: per class the 3 nodes with the highest in-class degree;
3. class-aware snowball to a pool of ``pool_size`` (1500): at each step add the non-pool node with
   the most neighbours already in the pool (ties: total degree, then lower pageid); while some
   class is below the pool floor (pool_size / 10 = 150, i.e. a balanced pool) the choice is
   restricted to the most deficient class;
4. fetch the pool, clean, drop missing pages and texts < 1200 chars;
5. peel by induced degree: repeatedly remove the node with the smallest number of neighbours in
   the current set (ties: lower pageid), never taking a class below its floor
   (1000 -> 40, 500 -> 25, 100 -> 5); while a class holds more than 15 % of the target size its
   lowest-degree node goes first (without the cap, pure min-degree peeling keeps the densely
   linked "Operating systems" hub and pushes seven classes to their floor).  The 500 set is peeled
   from the 1000 set, the 100 set from the 500 set, so the sets are nested.
Gold links of a set are its induced undirected edges.  Acceptance (printed and stored): median
induced degree >= 6 and >= 70 % of nodes with >= 5 in-subset gold links.
"""
from __future__ import annotations

import collections
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .clean import MIN_CHARS, clean_extract
from .docs import Doc, length_stats, sha256_docs, sha256_file, write_docs, write_gold, write_order
from .mediawiki import fetch_pages

SEED = 20260902
SIZES = (1000, 500, 100)
FLOORS = {1000: 40, 500: 25, 100: 5}
CAP_FRACTION = 0.15            # no class above 15 % of a set (binds at N=500 and N=100 only)
POOL_SIZE = 1500
POOL_FLOOR = None              # None -> pool_size // n_classes (balanced pool)
SEEDS_PER_CLASS = 3
ACCEPT_MEDIAN_DEGREE = 6
ACCEPT_FRAC_GE5 = 0.70


def doc_id(pageid: int | str) -> str:
    return f"wcs-{int(pageid)}"


# ---------------------------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------------------------
def load_raw(raw_dir: str | Path) -> tuple[list[dict], list[set[int]], dict[int, str]]:
    raw_dir = Path(raw_dir)
    data = json.loads((raw_dir / "data.json").read_text(encoding="utf-8"))
    meta = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
    label_names = {int(k): v for k, v in meta["labels"].items()}
    labels = data["labels"]
    adj = union_graph(data["links"])
    nodes = []
    for i, nd in enumerate(meta["nodes"]):
        nodes.append({"idx": i, "pageid": int(nd["id"]), "title": nd["title"], "label": int(labels[i]),
                      "label_name": label_names[int(labels[i])]})
        assert nd["label"] == label_names[int(labels[i])], f"label mismatch at node {i}"
    return nodes, adj, label_names


def union_graph(links: list[list[int]]) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in links]
    for u, nbrs in enumerate(links):
        for v in nbrs:
            if v != u:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def select_pool(nodes: list[dict], adj: list[set[int]], pool_size: int = POOL_SIZE,
                seeds_per_class: int = SEEDS_PER_CLASS, pool_floor: int | None = POOL_FLOOR) -> list[int]:
    """Class-aware greedy snowball.  Returns node indices in inclusion order."""
    n = len(nodes)
    label = [nd["label"] for nd in nodes]
    pageid = [nd["pageid"] for nd in nodes]
    deg = [len(a) for a in adj]
    in_class_deg = [sum(1 for u in adj[v] if label[u] == label[v]) for v in range(n)]
    classes = sorted(set(label))
    if pool_floor is None:
        pool_floor = pool_size // len(classes)
    in_pool = [False] * n
    gain = [0] * n
    count: dict[int, int] = {c: 0 for c in classes}
    frontier: set[int] = set()
    pool: list[int] = []

    def add(v: int) -> None:
        in_pool[v] = True
        pool.append(v)
        count[label[v]] += 1
        frontier.discard(v)
        for u in adj[v]:
            gain[u] += 1
            if not in_pool[u]:
                frontier.add(u)

    for c in classes:
        members = [v for v in range(n) if label[v] == c]
        members.sort(key=lambda v: (-in_class_deg[v], -deg[v], pageid[v]))
        for v in members[:seeds_per_class]:
            add(v)

    exhausted: set[int] = set()
    key = lambda v: (gain[v], deg[v], -pageid[v])  # noqa: E731
    while len(pool) < min(pool_size, n):
        deficits = [c for c in classes if count[c] < pool_floor and c not in exhausted]
        cands: list[int] = []
        if deficits:
            c = max(deficits, key=lambda c: (pool_floor - count[c], -c))
            cands = [v for v in frontier if label[v] == c]
            if not cands:
                cands = [v for v in range(n) if not in_pool[v] and label[v] == c]
                if not cands:
                    exhausted.add(c)
                    continue
                v = max(cands, key=lambda v: (in_class_deg[v], deg[v], -pageid[v]))
                add(v)
                continue
        else:
            cands = list(frontier)
            if not cands:
                cands = [v for v in range(n) if not in_pool[v]]
                if not cands:
                    break
        add(max(cands, key=key))
    return pool


def peel(adj: list[set[int]], nodes: list[dict], members: list[int], target: int, floor: int,
         cap: int | None = None) -> list[int]:
    """Remove min-induced-degree nodes (ties: lower pageid) until ``target`` remain, never taking
    a class below ``floor``.  While a class exceeds ``cap`` (default: no cap) its lowest-degree
    node is removed first (largest excess first).  Returns the kept indices sorted."""
    label = {v: nodes[v]["label"] for v in members}
    pageid = {v: nodes[v]["pageid"] for v in members}
    S = set(members)
    deg = {v: len(adj[v] & S) for v in S}
    count = collections.Counter(label[v] for v in S)
    while len(S) > target:
        over = [c for c, n in count.items() if cap is not None and n > cap]
        restrict = None
        if over:
            restrict = max(over, key=lambda c: (count[c] - cap, -c))
        best = None
        best_key = None
        for v in S:
            if count[label[v]] <= floor:
                continue
            if restrict is not None and label[v] != restrict:
                continue
            k = (deg[v], pageid[v])
            if best_key is None or k < best_key:
                best, best_key = v, k
        if best is None:
            break
        S.discard(best)
        count[label[best]] -= 1
        for u in adj[best]:
            if u in S:
                deg[u] -= 1
    return sorted(S)


def class_cap(size: int, cap_fraction: float) -> int | None:
    return None if cap_fraction is None else max(1, int(round(size * cap_fraction)))


def induced_stats(adj: list[set[int]], nodes: list[dict], members: list[int]) -> dict:
    S = set(members)
    degs = [len(adj[v] & S) for v in members]
    per_class = collections.Counter(nodes[v]["label_name"] for v in members)
    n = len(members)
    return {
        "n": n,
        "edges": sum(degs) // 2,
        "degree_median": statistics.median(degs) if degs else 0,
        "degree_mean": round(sum(degs) / n, 2) if n else 0,
        "frac_degree_ge5": round(sum(1 for d in degs if d >= 5) / n, 4) if n else 0,
        "isolated": sum(1 for d in degs if d == 0),
        "class_counts": dict(sorted(per_class.items())),
        "class_min": min(per_class.values()) if per_class else 0,
    }


def acceptance(stats: dict, floor: int) -> dict:
    checks = {
        "median_degree_ge_6": stats["degree_median"] >= ACCEPT_MEDIAN_DEGREE,
        "frac_degree_ge5_ge_0.70": stats["frac_degree_ge5"] >= ACCEPT_FRAC_GE5,
        f"class_floor_{floor}": stats["class_min"] >= floor,
    }
    checks["all"] = all(checks.values())
    return checks


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------
def build(raw_dir: str | Path = "data/wikics/raw", out_root: str | Path = "data/wikics",
          sizes: tuple[int, ...] = SIZES, pool_size: int = POOL_SIZE, rate: float = 1.0,
          seed: int = SEED, max_pages: int | None = None, cap_fraction: float | None = CAP_FRACTION,
          log=print) -> dict[int, Path]:
    t0 = time.perf_counter()
    out_root = Path(out_root)
    nodes, adj, label_names = load_raw(raw_dir)
    pool = select_pool(nodes, adj, pool_size=pool_size)
    if max_pages is not None:
        pool = pool[:max_pages]
    log(f"wikics: {len(nodes)} nodes, pool {len(pool)}; pool class counts "
        f"{dict(sorted(collections.Counter(nodes[v]['label_name'] for v in pool).items()))}")

    fetched = fetch_pages([nodes[v]["pageid"] for v in pool], out_root / "raw_api", rate=rate, log=log)
    t_fetch = time.perf_counter() - t0

    texts: dict[int, tuple[str, str]] = {}
    losses = collections.Counter()
    pool_records = []
    for v in pool:
        pg = fetched[str(nodes[v]["pageid"])]
        rec = {"idx": v, "pageid": nodes[v]["pageid"], "title_dataset": nodes[v]["title"],
               "label": nodes[v]["label_name"]}
        if pg.get("missing"):
            losses["missing"] += 1
            rec["status"] = "missing"
        else:
            body = clean_extract(pg["extract"])
            rec.update(title=pg["title"], revid=pg.get("revid"), timestamp=pg.get("timestamp"),
                       chars_raw=len(pg["extract"]), chars_clean=len(body))
            if len(body) < MIN_CHARS:
                losses["short"] += 1
                rec["status"] = "short"
            elif not pg.get("title"):
                losses["no_title"] += 1
                rec["status"] = "no_title"
            else:
                rec["status"] = "ok"
                texts[v] = (pg["title"], body)
        pool_records.append(rec)
    losses["ok"] = len(texts)
    log(f"wikics fetch/clean: {dict(losses)} ({t_fetch:.0f}s)")

    sets: dict[int, list[int]] = {}
    current = sorted(texts)
    for size in sorted(sizes, reverse=True):
        floor = FLOORS.get(size, max(1, round(size * 0.04)))
        current = peel(adj, nodes, current, size, floor, cap=class_cap(size, cap_fraction))
        sets[size] = current
        st = induced_stats(adj, nodes, current)
        log(f"wikics N={size}: kept {len(current)} (target {size}), median deg {st['degree_median']}, "
            f"frac>=5 {st['frac_degree_ge5']}, class_min {st['class_min']} -> {acceptance(st, floor)}")

    # orders: prefix-consistent (smallest set first)
    order_rng = np.random.default_rng(seed)
    order1: list[str] = []
    prev: set[int] = set()
    for size in sorted(sizes):
        inc = sorted(v for v in sets[size] if v not in prev)
        perm = order_rng.permutation(len(inc))
        order1.extend(doc_id(nodes[inc[i]]["pageid"]) for i in perm)
        prev |= set(inc)
    head = order1[: min(100, len(order1))]
    perm2 = np.random.default_rng(seed + 1).permutation(len(head))
    order2 = [head[i] for i in perm2] + order1[len(head):]

    written: dict[int, Path] = {}
    for size in sorted(sizes):
        members = sets[size]
        d = out_root / str(size)
        d.mkdir(parents=True, exist_ok=True)
        for old in (d / "docs").glob("*.md"):
            old.unlink()
        S = set(members)
        docs = [Doc(id=doc_id(nodes[v]["pageid"]), title=texts[v][0], body=texts[v][1]) for v in members]
        docs.sort(key=lambda dd: dd.id)
        write_docs(d, docs)
        gold_rows = []
        for v in sorted(members, key=lambda v: doc_id(nodes[v]["pageid"])):
            links = sorted(doc_id(nodes[u]["pageid"]) for u in adj[v] if u in S)
            gold_rows.append({"id": doc_id(nodes[v]["pageid"]), "title": texts[v][0],
                              "class": nodes[v]["label_name"], "class_id": nodes[v]["label"],
                              "pageid": nodes[v]["pageid"], "links": links})
        gold_path = write_gold(d, gold_rows)
        n = len(members)
        o1 = write_order(d / "orders" / "order-1.txt", order1[:n])
        o2 = write_order(d / "orders" / "order-2.txt", order2[:n])
        st = induced_stats(adj, nodes, members)
        floor = FLOORS.get(size, max(1, round(size * 0.04)))
        manifest = {
            "corpus": "wikics",
            "n": n,
            "target_n": size,
            "source": {"dataset": "WikiCS (Mernyei & Cangea 2020)",
                       "url": "https://github.com/pmernyei/wiki-cs-dataset (dataset/data.json, dataset/metadata.json)",
                       "text": "MediaWiki API prop=extracts|revisions|info, explaintext, by pageid",
                       "n_nodes_total": len(nodes)},
            "selection": {"seed": seed, "seeds_per_class": SEEDS_PER_CLASS, "pool_size": len(pool),
                          "pool_floor": pool_size // len(label_names), "peel_floor": floor,
                          "peel_cap": class_cap(size, cap_fraction), "cap_fraction": cap_fraction,
                          "nested": True, "sizes": sorted(sizes)},
            "losses": dict(losses),
            "n_classes": len(set(nodes[v]["label"] for v in members)),
            "oracle_k": len(set(nodes[v]["label"] for v in members)),
            "class_names": {str(k): v for k, v in sorted(label_names.items())},
            "graph": st,
            "acceptance": acceptance(st, floor),
            "length": length_stats(docs),
            "orders": {"order-1": "prefix-consistent across N (100 first, then 500 increment, then 1000 increment)",
                       "order-2": "first 100 ids re-shuffled with seed+1, rest as order-1"},
            "sha256": {"gold.jsonl": sha256_file(gold_path), "docs": sha256_docs(docs),
                       "order-1.txt": sha256_file(o1), "order-2.txt": sha256_file(o2)},
            "fetch_seconds": round(t_fetch, 1),
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written[size] = d
        log(f"wrote {d}: {n} docs, acceptance {manifest['acceptance']}")
    (out_root / "pool.json").write_text(json.dumps({"pool": pool_records, "losses": dict(losses)},
                                                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return written

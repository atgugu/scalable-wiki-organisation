"""DBPedia_Classes corpus: pool, filter, dedup, derive titles, nested stratified sampling.

Source: ``DeveloperOats/DBPedia_Classes`` on the Hugging Face Hub (CC0; fields text/l1/l2/l3,
no title).  Pinned to revision ``REVISION``.

Sampling (seed 20260902), all sizes nested 100 ⊂ 300 ⊂ 1000 ⊂ 2000:

* N=2000, all 70 l2 classes: ``clip(round(2000·share), 5, 120)`` per l2 class where ``share`` is the
  class share of the filtered pool.  The clipped allocation sums to fewer than 2000 (1845 on the
  pinned revision), so the remainder is distributed by largest remainder among classes that are
  strictly inside their bounds until the total is exactly 2000.
* N=1000, all 70 classes: proportional to the N=2000 counts, floor 3, cap = the N=2000 count.
* N=300, ≈40 classes: the top 5 l2 classes (by pool count) of every l1, then the largest remaining
  l2 classes until 40; proportional to the N=2000 counts with floor 3.
* N=100, 18 classes: the top 2 l2 classes of every l1, then the largest remaining classes of the
  N=300 set until 18 (three l1 classes have a single l2 subclass, so "2 per l1" alone gives 15);
  proportional with floor 3.

Within a class the documents are shuffled once (seeded) and every size takes a prefix, which is
what makes the sets nested.  ``orders/order-1.txt`` is the ingest order for arms B/C: the N=100
documents first (shuffled), then the N=300 increment, then the N=1000 increment, then the rest, so
the order at any N is a prefix of the order at a larger N and B@300 yields B@100 by snapshot after
ingest #100.  ``orders/order-2.txt`` re-shuffles the first 100 ids (seed+1) and keeps the rest.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from .docs import Doc, length_stats, sha256_docs, sha256_file, write_docs, write_gold, write_order

DATASET = "DeveloperOats/DBPedia_Classes"
REVISION = "4d0aa96069f24063697e4df63b95be78d3f7fb7d"
SPLITS = ("train", "validation", "test")
MIN_CHARS = 1200
SEED = 20260902
SIZES = (100, 300, 1000, 2000)
TITLE_MAX = 80
CLASSES_AT = {300: 40, 100: 18}      # target number of l2 classes for the small sets
TOP_L2_PER_L1 = {300: 5, 100: 2}
FLOOR_SMALL = 3                       # per-class floor for N < 2000
FLOOR_2000, CAP_2000 = 5, 120

# ---------------------------------------------------------------------------------------------
# text normalisation and titles
# ---------------------------------------------------------------------------------------------
_PAREN_RE = re.compile(r"\s*\([^()]*\)")
_IPA_RE = re.compile(r"\s*/[^/\n]{1,80}/")
_WS_RE = re.compile(r"\s+")
_SUBJECT_RE = re.compile(r"^(.{3,80}?)\s(?:is|was|are|were)\s")
_LOWER_AFTER_COMMA_RE = re.compile(r",\s+(?=[a-z])")
_SENTENCE_END_RE = re.compile(r"(?<=[a-z0-9)\"'”’])[.!?]\s+(?=[A-Z\"'“‘(])")


def clean_text(text: str) -> str:
    """Undo the CSV export's JSON-style escapes (``\\"`` in 41 % of long texts, ``\\n`` in 4 %,
    rare ``\\\\``, ``\\uXXXX``, ``\\t``) and normalise whitespace; content is otherwise kept."""
    if "\\" in text:
        try:
            t = json.loads('"' + text.replace('"', '\\"').replace('\\\\"', '\\"') + '"')
        except ValueError:
            t = (text.replace('\\"', '"').replace("\\n", "\n").replace("\\t", " ")
                 .replace("\\'", "'").replace("\\\\", "\\"))
    else:
        t = text
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t\u00a0]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "st", "jr", "sr", "prof", "gen", "col", "lt", "capt", "sgt", "maj", "rev",
    "hon", "vs", "no", "mt", "ft", "pt", "co", "inc", "ltd", "corp", "bros", "op", "vol", "ca", "approx",
    "u.s", "u.k", "i.e", "e.g", "cf", "etc", "fr", "sept", "oct", "nov", "dec", "jan", "feb", "aug",
}


def first_sentence(s: str) -> str:
    """Text up to the first sentence boundary (``.!?`` + space + capital) that does not follow a
    common abbreviation or a single-letter initial."""
    for m in _SENTENCE_END_RE.finditer(s):
        before = s[: m.start()]
        word = before.rsplit(" ", 1)[-1].casefold().strip("(\"'“‘")
        if word in _ABBREVIATIONS or len(word) <= 1:
            continue
        return before
    return s


def strip_parentheticals(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _PAREN_RE.sub("", s)
    return s


def _looks_like_ipa(chunk: str) -> bool:
    return any(ord(ch) > 127 for ch in chunk) or "ˈ" in chunk or "ˌ" in chunk


def strip_pronunciation(s: str) -> str:
    def repl(m: re.Match) -> str:
        return "" if _looks_like_ipa(m.group(0)) else m.group(0)
    return _IPA_RE.sub(repl, s)


def derive_title(text: str, max_len: int = TITLE_MAX) -> str:
    """Subject of the first sentence: ``^(.{3,80}?)\\s(is|was|are|were)\\s`` after stripping
    parentheticals (birth dates, pronunciation guides) and IPA ``/…/`` runs; fallback = first 6
    words.  Appositions introduced by a comma followed by a lowercase word (", also known as …",
    ", the founder of …") are cut; "Springfield, Illinois" is kept.  Cut at 80 characters."""
    s = clean_text(text)
    s = strip_pronunciation(strip_parentheticals(s))
    s = _WS_RE.sub(" ", s).strip()
    first = first_sentence(s).split(";", 1)[0]
    m = _SUBJECT_RE.match(first + " ")
    fallback = m is None
    t = m.group(1) if m else " ".join((first if len(first.split()) >= 2 else s).split()[:6])
    cut = _LOWER_AFTER_COMMA_RE.search(t)
    if cut and cut.start() >= 3:
        head = t[: cut.start()]
        words = head.split()
        if len(words) >= 3 or any(w[:1].isupper() for w in words[1:]):  # looks like a name, not "In 209"
            t = head
    t = t.strip(" ,;:.\"'“”‘’")
    if len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0] if " " in t[:max_len] else t[:max_len]
    t = t.strip(" ,;:.\"'“”‘’")
    if not t:
        t = " ".join(s.split()[:6])[:max_len].strip() or "Untitled"
    derive_title.last_fallback = fallback  # type: ignore[attr-defined]
    return t


def normalise_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.casefold())


def doc_id(text: str) -> str:
    return "dbp-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------------------------
# pool
# ---------------------------------------------------------------------------------------------
def load_pool(min_chars: int = MIN_CHARS, revision: str = REVISION, dataset: str = DATASET,
              log=print) -> tuple[list[dict], dict]:
    """All splits pooled, ``len(text) > min_chars`` (on the raw text), deduplicated by sha1 of the
    cleaned text and by normalised derived title.  Returns rows sorted by id and the pool stats."""
    import datasets  # local import: heavy

    ds = datasets.load_dataset(dataset, revision=revision)
    rows: list[dict] = []
    stats = collections.Counter()
    seen_text: set[str] = set()
    seen_title: set[str] = set()
    for split in SPLITS:
        for r in ds[split]:
            stats["total"] += 1
            if len(r["text"]) <= min_chars:
                stats["dropped_short"] += 1
                continue
            text = clean_text(r["text"])
            if len(text) <= min_chars:
                stats["dropped_short_after_clean"] += 1
                continue
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if h in seen_text:
                stats["dropped_dup_text"] += 1
                continue
            title = derive_title(text)
            nt = normalise_title(title)
            if nt in seen_title:
                stats["dropped_dup_title"] += 1
                continue
            seen_text.add(h)
            seen_title.add(nt)
            if derive_title.last_fallback:  # type: ignore[attr-defined]
                stats["title_fallback"] += 1
            rows.append({"id": "dbp-" + h[:10], "title": title, "text": text,
                         "l1": r["l1"], "l2": r["l2"], "l3": r["l3"], "split": split})
    rows.sort(key=lambda r: r["id"])
    stats["pool"] = len(rows)
    log(f"dbpedia pool: {dict(stats)}")
    return rows, dict(stats)


# ---------------------------------------------------------------------------------------------
# allocation helpers (deterministic)
# ---------------------------------------------------------------------------------------------
def apportion(total: int, weights: dict[str, float], floor: int, caps: dict[str, int]) -> dict[str, int]:
    """Largest-remainder apportionment of ``total`` proportional to ``weights`` with a per-class
    floor (min(floor, cap)) and per-class caps.  Deterministic: ties broken by class name."""
    keys = sorted(weights)
    alloc = {k: min(floor, caps[k]) for k in keys}
    if sum(caps[k] for k in keys) < total:
        raise ValueError(f"caps sum to {sum(caps.values())} < total {total}")
    for _ in range(10_000):
        remaining = total - sum(alloc.values())
        if remaining <= 0:
            break
        room = {k: caps[k] - alloc[k] for k in keys if caps[k] > alloc[k]}
        if not room:
            break
        wsum = sum(weights[k] for k in room) or 1.0
        quota = {k: remaining * weights[k] / wsum for k in room}
        give = {k: min(room[k], int(np.floor(quota[k]))) for k in room}
        left = remaining - sum(give.values())
        for k in sorted(room, key=lambda k: (-(quota[k] - np.floor(quota[k])), k)):
            if left <= 0:
                break
            if give[k] < room[k]:
                give[k] += 1
                left -= 1
        if sum(give.values()) == 0:
            break
        for k, g in give.items():
            alloc[k] += g
    assert sum(alloc.values()) == total, (sum(alloc.values()), total)
    return alloc


def allocate_2000(pool_counts: dict[str, int], total: int = 2000, floor: int = FLOOR_2000,
                  cap: int = CAP_2000) -> dict[str, int]:
    """``clip(round(total·share), floor, min(cap, pool))`` then largest-remainder adjustment to
    exactly ``total`` among classes strictly inside their bounds."""
    pool_total = sum(pool_counts.values())
    keys = sorted(pool_counts)
    caps = {k: min(cap, pool_counts[k]) for k in keys}
    min_total = sum(min(floor, caps[k]) for k in keys)
    if min_total > total:
        raise ValueError(f"largest size {total} is below the sum of class floors ({min_total}: {len(keys)} classes x "
                         f"floor {floor}); the largest size must be at least {min_total}")
    alloc = {k: int(min(caps[k], max(min(floor, caps[k]), round(total * pool_counts[k] / pool_total))))
             for k in keys}
    diff = total - sum(alloc.values())
    exact = {k: max(total * pool_counts[k] / pool_total, 1e-9) for k in keys}
    ratio = lambda k: alloc[k] / exact[k]  # noqa: E731  allocation relative to the proportional share
    while diff != 0:
        if diff > 0:  # give to the class that is most under its share (ties: name)
            cand = [k for k in keys if alloc[k] < caps[k]]
            if not cand:
                raise ValueError("cannot reach total: every class is at its cap")
            k = sorted(cand, key=lambda k: (ratio(k), k))[0]
            alloc[k] += 1
            diff -= 1
        else:  # take from the class that is most over its share
            cand = [k for k in keys if alloc[k] > min(floor, caps[k])]
            k = sorted(cand, key=lambda k: (-ratio(k), k))[0]
            alloc[k] -= 1
            diff += 1
    return alloc


def select_classes(pool_counts_l1l2: dict[str, dict[str, int]], n_classes: int, top_per_l1: int,
                   restrict_to: set[str] | None = None) -> list[str]:
    """Top ``top_per_l1`` l2 classes (by pool count) of every l1, then the largest remaining l2
    classes (optionally restricted to ``restrict_to``) until ``n_classes``.  Sorted output."""
    chosen: list[str] = []
    allowed = lambda c: restrict_to is None or c in restrict_to  # noqa: E731
    for l1 in sorted(pool_counts_l1l2):
        ranked = sorted(pool_counts_l1l2[l1].items(), key=lambda kv: (-kv[1], kv[0]))
        chosen.extend(c for c, _ in ranked[:top_per_l1] if allowed(c))
    if len(chosen) > n_classes:
        # keep the largest classes overall but never below one per l1
        keep = set()
        for l1 in sorted(pool_counts_l1l2):
            ranked = sorted(pool_counts_l1l2[l1].items(), key=lambda kv: (-kv[1], kv[0]))
            for c, _ in ranked:
                if allowed(c):
                    keep.add(c)
                    break
        rest = sorted((c for c in chosen if c not in keep),
                      key=lambda c: (-_count(pool_counts_l1l2, c), c))
        chosen = sorted(keep) + rest[: n_classes - len(keep)]
    remaining = sorted(
        (c for l1 in pool_counts_l1l2 for c in pool_counts_l1l2[l1] if c not in chosen and allowed(c)),
        key=lambda c: (-_count(pool_counts_l1l2, c), c))
    chosen.extend(remaining[: max(0, n_classes - len(chosen))])
    return sorted(chosen)


def _count(pool_counts_l1l2: dict[str, dict[str, int]], l2: str) -> int:
    return sum(d.get(l2, 0) for d in pool_counts_l1l2.values())


# ---------------------------------------------------------------------------------------------
# nested sampling
# ---------------------------------------------------------------------------------------------
def sample_nested(rows: list[dict], seed: int = SEED, sizes: tuple[int, ...] = SIZES,
                  log=print) -> dict:
    """Returns ``{"sets": {N: [row, ...]}, "orders": {N: {"order-1": [ids], "order-2": [ids]}},
    "allocation": {N: {l2: n}}, "classes": {N: [l2]}}`` with nested sets and prefix-consistent
    orders."""
    sizes = tuple(sorted(sizes))
    n_max = sizes[-1]
    by_l2: dict[str, list[dict]] = collections.defaultdict(list)
    l1l2: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for r in rows:
        by_l2[r["l2"]].append(r)
    for l2, rs in by_l2.items():
        l1l2[rs[0]["l1"]][l2] = len(rs)
    pool_counts = {l2: len(rs) for l2, rs in by_l2.items()}

    rng = np.random.default_rng(seed)
    shuffled: dict[str, list[dict]] = {}
    for l2 in sorted(by_l2):
        rs = sorted(by_l2[l2], key=lambda r: r["id"])
        perm = rng.permutation(len(rs))
        shuffled[l2] = [rs[i] for i in perm]

    # allocations from the largest size down
    allocation: dict[int, dict[str, int]] = {}
    classes: dict[int, list[str]] = {}
    if n_max == 2000:
        allocation[n_max] = allocate_2000(pool_counts)
    else:  # generic largest size (tests / other totals)
        allocation[n_max] = allocate_2000(pool_counts, total=n_max)
    classes[n_max] = sorted(allocation[n_max])
    parent_n = n_max
    for n in reversed(sizes[:-1]):
        parent_alloc = allocation[parent_n]
        if n in CLASSES_AT:
            restrict = set(classes[parent_n])
            cls = select_classes(l1l2, CLASSES_AT[n], TOP_L2_PER_L1[n], restrict_to=restrict)
        else:
            cls = list(classes[parent_n])
        weights = {c: float(parent_alloc[c]) for c in cls}
        caps = {c: parent_alloc[c] for c in cls}
        allocation[n] = apportion(n, weights, FLOOR_SMALL, caps)
        classes[n] = sorted(cls)
        parent_n = n

    sets: dict[int, list[dict]] = {}
    for n in sizes:
        chosen = []
        for l2 in classes[n]:
            chosen.extend(shuffled[l2][: allocation[n][l2]])
        chosen.sort(key=lambda r: r["id"])
        assert len(chosen) == n, (n, len(chosen))
        sets[n] = chosen

    # nesting check
    for a, b in zip(sizes, sizes[1:]):
        ida = {r["id"] for r in sets[a]}
        idb = {r["id"] for r in sets[b]}
        assert ida <= idb, f"set {a} not nested in {b}"

    # orders: prefix-consistent across N
    order_rng = np.random.default_rng(seed)
    order1: list[str] = []
    prev: set[str] = set()
    for n in sizes:
        inc = sorted(r["id"] for r in sets[n] if r["id"] not in prev)
        perm = order_rng.permutation(len(inc))
        order1.extend(inc[i] for i in perm)
        prev |= set(inc)
    head = order1[: min(100, len(order1))]
    perm2 = np.random.default_rng(seed + 1).permutation(len(head))
    order2 = [head[i] for i in perm2] + order1[len(head):]
    orders = {n: {"order-1": order1[:n], "order-2": order2[:n]} for n in sizes}
    for n in sizes:
        assert set(orders[n]["order-1"]) == {r["id"] for r in sets[n]}
    log("dbpedia allocation: " + ", ".join(f"N={n}: {len(classes[n])} l2 classes" for n in sizes))
    return {"sets": sets, "orders": orders, "allocation": allocation, "classes": classes,
            "pool_counts": pool_counts}


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------
def build(out_root: str | Path = "data/dbpedia", seed: int = SEED, sizes: tuple[int, ...] = SIZES,
          min_chars: int = MIN_CHARS, log=print) -> dict[int, Path]:
    t0 = time.perf_counter()
    rows, pool_stats = load_pool(min_chars=min_chars, log=log)
    t_pool = time.perf_counter() - t0
    sample = sample_nested(rows, seed=seed, sizes=sizes, log=log)
    out_root = Path(out_root)
    written: dict[int, Path] = {}
    for n in sorted(sizes):
        d = out_root / str(n)
        d.mkdir(parents=True, exist_ok=True)
        for old in (d / "docs").glob("*.md"):
            old.unlink()
        docs = [Doc(id=r["id"], title=r["title"], body=r["text"]) for r in sample["sets"][n]]
        write_docs(d, docs)
        gold_rows = [{"id": r["id"], "title": r["title"], "l1": r["l1"], "l2": r["l2"], "l3": r["l3"]}
                     for r in sample["sets"][n]]
        gold_path = write_gold(d, gold_rows)
        o1 = write_order(d / "orders" / "order-1.txt", sample["orders"][n]["order-1"])
        o2 = write_order(d / "orders" / "order-2.txt", sample["orders"][n]["order-2"])
        counts = {lvl: dict(sorted(collections.Counter(r[lvl] for r in sample["sets"][n]).items()))
                  for lvl in ("l1", "l2", "l3")}
        manifest = {
            "corpus": "dbpedia",
            "n": n,
            "source": {"dataset": DATASET, "revision": REVISION, "splits": list(SPLITS), "license": "CC0"},
            "filter": {"min_chars_exclusive": min_chars, "dedup": ["sha1(clean text)", "normalised derived title"]},
            "pool": pool_stats,
            "seed": seed,
            "sizes": list(sorted(sizes)),
            "nested": True,
            "n_classes": {"l1": len(counts["l1"]), "l2": len(counts["l2"]), "l3": len(counts["l3"])},
            "oracle_k": len(counts["l2"]),
            "class_counts": counts,
            "allocation_rule": ("N=2000: clip(round(2000*share),5,120) + largest-remainder top-up; "
                                "smaller N: proportional to parent counts, floor 3; classes: "
                                "top-5 l2 per l1 -> 40 at N=300, top-2 l2 per l1 -> 18 at N=100"),
            "length": length_stats(docs),
            "title_fallback_fraction_pool": round(pool_stats.get("title_fallback", 0) / max(1, pool_stats["pool"]), 4),
            "orders": {"order-1": "prefix-consistent across N (100 first, then increments)",
                       "order-2": "first 100 ids re-shuffled with seed+1, rest as order-1"},
            "sha256": {"gold.jsonl": sha256_file(gold_path), "docs": sha256_docs(docs),
                       "order-1.txt": sha256_file(o1), "order-2.txt": sha256_file(o2)},
            "build_seconds_pool": round(t_pool, 1),
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written[n] = d
        log(f"wrote {d}: {n} docs, l1={len(counts['l1'])} l2={len(counts['l2'])} l3={len(counts['l3'])}, "
            f"median chars {manifest['length']['chars_median']}")
    return written

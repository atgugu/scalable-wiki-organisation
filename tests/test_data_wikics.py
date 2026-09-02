"""WikiCS: extract cleaning (math residue, section cut, headings) and graph subset selection."""
import numpy as np

from wikiorg.data.clean import clean_extract, convert_headings, cut_trailing_sections, strip_displaystyle
from wikiorg.data.wikics import acceptance, class_cap, induced_stats, peel, select_pool, union_graph

MATH_EXTRACT = (
    "There are FFTs with \n  \n    \n      \n        O\n        (\n        n\n        log\n        ⁡\n"
    "        n\n        )\n      \n    \n    {\\displaystyle O(n\\log n)}\n  \n complexity for all n.\n"
    "Let \n  \n    \n      \n        n\n        =\n        \n          n\n          \n            1\n"
    "          \n        \n      \n    \n    {\\textstyle n=n_{1}n_{2}}\n  \n, where both are coprime.\n\n"
    "== See also ==\nDiscrete Fourier transform\n\n== References ==\n"
)


def test_strip_displaystyle_keeps_inline_latex_and_removes_dump():
    s = strip_displaystyle(MATH_EXTRACT.split("\n\n== See also")[0])
    assert "displaystyle" not in s and "textstyle" not in s and "⁡" not in s
    assert "FFTs with O(n\\log n) complexity for all n." in s
    assert "Let n=n_{1}n_{2}, where both are coprime." in s
    assert not any(ln.startswith("  ") for ln in s.splitlines())


def test_strip_displaystyle_unbalanced_and_long():
    assert strip_displaystyle("a {\\displaystyle x^{2} unbalanced\nnext line") == "a next line"
    assert strip_displaystyle("x = {\\displaystyle y}\n  \n\n\nNew paragraph") == "x = y\n\nNew paragraph"
    long = "{\\displaystyle " + "x" * 200 + "}"
    assert strip_displaystyle(f"before {long} after").strip() == "before after"
    assert strip_displaystyle("no math here") == "no math here"


def test_cut_and_headings():
    text = "Intro.\n\n== History ==\nOld.\n\n=== Early ===\nVery old.\n\n== References ==\nrefs\n\n== Later ==\nx"
    cut = cut_trailing_sections(text)
    assert "refs" not in cut and "Later" not in cut and "Very old." in cut
    assert convert_headings(cut).strip() == "Intro.\n\n## History\nOld.\n\n### Early\nVery old."


def test_clean_extract_end_to_end():
    body = clean_extract(MATH_EXTRACT)
    assert "See also" not in body and "References" not in body
    assert body.startswith("There are FFTs with O(n\\log n) complexity")
    assert "\n\n\n" not in body


def _synthetic_graph(seed=0, classes=3, per_class=40):
    rng = np.random.default_rng(seed)
    n = classes * per_class
    labels = [i % classes for i in range(n)]
    links = [[] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            p = 0.25 if labels[u] == labels[v] else 0.02
            if rng.random() < p:
                links[u].append(v)
    nodes = [{"idx": i, "pageid": 1000 + i, "title": f"T{i}", "label": labels[i], "label_name": f"C{labels[i]}"}
             for i in range(n)]
    return nodes, union_graph(links)


def test_union_graph_symmetric_without_self_loops():
    nodes, adj = _synthetic_graph()
    for u, nb in enumerate(adj):
        assert u not in nb
        for v in nb:
            assert u in adj[v]


def test_select_pool_and_peel_respect_floors_caps_and_nesting():
    nodes, adj = _synthetic_graph()
    pool = select_pool(nodes, adj, pool_size=90, seeds_per_class=3, pool_floor=None)
    assert len(pool) == 90 and len(set(pool)) == 90
    counts = np.bincount([nodes[v]["label"] for v in pool])
    assert counts.min() >= 30  # balanced pool: 90 // 3 per class
    s60 = peel(adj, nodes, pool, 60, floor=15, cap=class_cap(60, 0.4))
    s30 = peel(adj, nodes, s60, 30, floor=5, cap=class_cap(30, 0.4))
    assert len(s60) == 60 and len(s30) == 30 and set(s30) <= set(s60) <= set(pool)
    c60 = np.bincount([nodes[v]["label"] for v in s60], minlength=3)
    c30 = np.bincount([nodes[v]["label"] for v in s30], minlength=3)
    assert c60.min() >= 15 and c60.max() <= 24 and c30.min() >= 5 and c30.max() <= 12
    st = induced_stats(adj, nodes, s60)
    assert st["n"] == 60 and set(st["class_counts"]) == {"C0", "C1", "C2"}
    acc = acceptance(st, 15)
    assert set(acc) == {"median_degree_ge_6", "frac_degree_ge5_ge_0.70", "class_floor_15", "all"}
    assert acc["class_floor_15"] is True


def test_peel_is_deterministic_and_prefers_low_degree():
    nodes, adj = _synthetic_graph(seed=3)
    members = list(range(len(nodes)))
    a = peel(adj, nodes, members, 50, floor=1)
    b = peel(adj, nodes, members, 50, floor=1)
    assert a == b
    kept = set(a)
    deg_kept = np.mean([len(adj[v] & kept) for v in a])
    deg_all = np.mean([len(adj[v]) for v in members])
    assert deg_kept > 0.3 * deg_all

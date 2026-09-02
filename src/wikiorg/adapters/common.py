"""Shared helpers for the arm adapters: title normalisation and title -> id
resolution, slugs, tolerant front-matter parsing.

``norm_title`` is the single normalisation used everywhere a free-text title
written by an LLM (arm C ``related_titles``, arm B page titles) has to be mapped
back to a corpus document id: NFKC Unicode normalisation, ``casefold``, all
punctuation stripped, whitespace collapsed.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Iterable

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_FM_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


def norm_title(s: str) -> str:
    """NFKC -> casefold -> strip punctuation (underscore kept as a word char is
    replaced by a space) -> collapse whitespace."""
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s).replace("_", " ")
    return _WS_RE.sub(" ", s).strip()


def slugify(s: str, max_len: int = 80) -> str:
    """Lower-case ASCII words joined by hyphens (the schema's slug rule)."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:max_len].strip("-") or "untitled"


class TitleIndex:
    """Resolve titles (or ids) to document ids.

    ``resolve`` is an exact match after :func:`norm_title` (also accepting the id
    itself); ``near_miss`` finds the closest known title with a difflib ratio at
    or above ``cutoff`` (0.9 by default), which the link-validity report counts
    separately from exact resolutions and never turns into a link.
    """

    def __init__(self, titles: dict[str, str], cutoff: float = 0.9):
        self.titles = dict(titles)
        self.cutoff = cutoff
        self._by_norm: dict[str, str] = {}
        self.ambiguous: set[str] = set()
        for pid, title in self.titles.items():
            key = norm_title(title)
            if not key:
                continue
            if key in self._by_norm and self._by_norm[key] != pid:
                self.ambiguous.add(key)
            self._by_norm.setdefault(key, pid)
        self._keys = list(self._by_norm)

    def resolve(self, text: str) -> str | None:
        if text in self.titles:
            return text
        key = norm_title(text)
        if not key:
            return None
        return self._by_norm.get(key)

    def near_miss(self, text: str) -> tuple[str | None, float]:
        """Closest title by difflib ratio (id, ratio) if ratio >= cutoff."""
        key = norm_title(text)
        if not key or not self._keys:
            return None, 0.0
        best = difflib.get_close_matches(key, self._keys, n=1, cutoff=self.cutoff)
        if not best:
            return None, 0.0
        ratio = difflib.SequenceMatcher(None, key, best[0]).ratio()
        return self._by_norm[best[0]], ratio

    def classify(self, text: str) -> tuple[str, str | None]:
        """('resolved' | 'near_miss' | 'dangling', id or None)."""
        pid = self.resolve(text)
        if pid is not None:
            return "resolved", pid
        pid, _ = self.near_miss(text)
        if pid is not None:
            return "near_miss", pid
        return "dangling", None


def _parse_scalar(v: str) -> Any:
    v = v.strip()
    if not v:
        return ""
    if (v[0] == v[-1]) and v[0] in "\"'":
        return v[1:-1]
    low = v.lower()
    if low in ("null", "~", "none"):
        return None
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    return v


def _parse_list(v: str) -> list[Any]:
    inner = v.strip()[1:-1]
    if not inner.strip():
        return []
    return [_parse_scalar(x) for x in inner.split(",")]


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(fields, body)``.  Uses PyYAML when the block is valid YAML and
    falls back to a line-wise ``key: value`` parser (LLM-written YAML is often
    slightly malformed: unquoted colons, stray brackets, inline comments)."""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    block, body = m.group(1), text[m.end():]
    fields: dict[str, Any] | None = None
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(block)
        if isinstance(loaded, dict):
            fields = {str(k): v for k, v in loaded.items()}
    except Exception:
        fields = None
    if fields is None:
        fields = {}
        for line in block.splitlines():
            line = re.sub(r"\s+#.*$", "", line)  # trailing comment
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if not key:
                continue
            if val.startswith("[") and val.endswith("]"):
                fields[key] = _parse_list(val)
            else:
                fields[key] = _parse_scalar(val)
    return fields, body


def as_str_list(v: Any) -> list[str]:
    """Front-matter list field -> list of stripped strings (tolerates a scalar,
    a comma-separated string, ``None``)."""
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            return [str(x).strip() for x in _parse_list(s) if str(x).strip()]
        return [x.strip() for x in s.split(",") if x.strip()] if s else []
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    return [str(v)]


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

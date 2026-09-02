"""Polite, resumable MediaWiki API client for full-text extracts by pageid.

One page per request (whole-page ``extracts`` allow ``exlimit=1``), at most ``rate`` requests per
second, User-Agent with a contact address (Wikimedia API etiquette), ``maxlag=5`` so the request
yields to replication lag.  Every raw JSON response is archived as ``<archive_dir>/<pageid>.json``
and reused on the next run, so an interrupted fetch resumes where it stopped.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "wikiorg-thesis/0.1 (antoniugugu@gmail.com)"
PARAMS = {
    "action": "query",
    "format": "json",
    "formatversion": "2",
    "prop": "extracts|revisions|info",
    "explaintext": "1",
    "exsectionformat": "wiki",
    "redirects": "1",
    "rvprop": "ids|timestamp",
    "inprop": "url",
    "maxlag": "5",
}


class RateLimiter:
    def __init__(self, rate: float = 1.0) -> None:
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delay = self._last + self.interval - now
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def fetch_page(pageid: int | str, session: requests.Session | None = None, timeout: float = 60.0,
               retries: int = 6) -> dict:
    """Raw JSON of ``action=query`` for one pageid, with retries on network/5xx/429/maxlag."""
    sess = session or requests.Session()
    params = dict(PARAMS, pageids=str(pageid))
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = sess.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = float(r.headers.get("Retry-After", 0) or 0)
                time.sleep(max(retry_after, 5.0 * 2 ** attempt) + random.uniform(0, 2))
                continue
            r.raise_for_status()
            data = r.json()
            err = data.get("error")
            if err and err.get("code") == "maxlag":
                time.sleep(float(r.headers.get("Retry-After", 5) or 5) + random.uniform(0, 2))
                continue
            return data
        except (requests.RequestException, ValueError) as exc:  # network / JSON
            last_exc = exc
            time.sleep(5.0 * 2 ** attempt + random.uniform(0, 2))
    raise RuntimeError(f"fetch_page({pageid}) failed after {retries} attempts: {last_exc}")


def parse_page(raw: dict) -> dict:
    """Normalise one archived response into {pageid, title, extract, revid, timestamp, missing, ...}."""
    pages = raw.get("query", {}).get("pages", [])
    if not pages:
        return {"missing": True, "error": raw.get("error")}
    pg = pages[0]
    if pg.get("missing") or pg.get("invalid"):
        return {"pageid": pg.get("pageid"), "title": pg.get("title"), "missing": True}
    revs = pg.get("revisions") or [{}]
    return {
        "pageid": pg.get("pageid"),
        "title": pg.get("title"),
        "extract": pg.get("extract", "") or "",
        "revid": revs[0].get("revid"),
        "timestamp": revs[0].get("timestamp"),
        "lastrevid": pg.get("lastrevid"),
        "length": pg.get("length"),
        "fullurl": pg.get("fullurl"),
        "redirected_from": [r.get("from") for r in raw.get("query", {}).get("redirects", [])],
        "missing": False,
    }


def fetch_pages(pageids: list[int | str], archive_dir: str | Path, rate: float = 1.0,
                log=print, progress_every: int = 100) -> dict[str, dict]:
    """Fetch (or load from the archive) every pageid.  Returns {str(pageid): parsed page}."""
    archive = Path(archive_dir)
    archive.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(rate)
    sess = requests.Session()
    out: dict[str, dict] = {}
    n_fetched = n_cached = n_failed = 0
    t0 = time.perf_counter()
    for i, pid in enumerate(pageids, 1):
        pid = str(pid)
        path = archive / f"{pid}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            n_cached += 1
        else:
            limiter.wait()
            try:
                raw = fetch_page(pid, sess)
            except Exception as exc:  # noqa: BLE001 - record and continue
                log(f"  pageid {pid}: FAILED {exc}")
                out[pid] = {"pageid": int(pid), "missing": True, "error": str(exc)}
                n_failed += 1
                continue
            raw["_fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            path.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
            n_fetched += 1
        out[pid] = parse_page(raw)
        if i % progress_every == 0 or i == len(pageids):
            log(f"  mediawiki {i}/{len(pageids)} (fetched {n_fetched}, cached {n_cached}, failed {n_failed}, "
                f"{time.perf_counter() - t0:.0f}s)")
    return out

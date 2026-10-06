"""Acquire preprints: arXiv through its API, bioRxiv/medRxiv through Europe PMC.

WHY THIS EXISTS. Some fields publish mainly in conference proceedings and
preprint servers rather than in the journal channel reached by the default
acquisition path. Seats in those fields will come out thin for exactly that
reason -- a fact about the channel, not about the person, but a fact that makes
those seats abstain when they should be speaking.

Two channels close most of the gap:

    arXiv          conference-oriented work, usually posted before or instead
                   of the proceedings version
    Europe PMC     `SRC:PPR` reaches preprint servers that expose records
                   through the corpus index

THE INTEGRITY RULES DO NOT RELAX FOR PREPRINTS. A preprint is stored with
`provenance: PREPRINT`, so a seat can be asked to weigh it accordingly, and the
same `classify_fulltext` measurement decides whether we actually hold the text.

    A PREPRINT IS NOT PEER REVIEWED AND THE RECORD SAYS SO. It is not excluded:
    in some fields the preprint IS the literature, and a panel that cannot see
    the newest proceedings-era work is not modelling those seats. But `provenance` is carried into the document so
    that a seat which wants to discount it can, and so that a reader can tell.

PREPRINT ACQUISITION HAS THREE INTEGRITY REQUIREMENTS:

  1. Network access goes through the corpus `Fetcher`, which carries the
     User-Agent, per-host throttling, retry with backoff on 429, and the cache.
     A rate-limited channel must not be reported as an empty result set.
  2. Full-text status is DERIVED from the extracted text with
     `classify_fulltext`; callers do not assert it while constructing metadata.
  3. arXiv and Europe PMC are both implemented, so the documented preprint
     channels match the code path.

Usage:
    python ops/acquire_preprints.py --config config/preprints.yaml --dry-run
    python ops/acquire_preprints.py --config config/preprints.yaml --root library_preprints
    python ops/acquire_preprints.py --config config/preprints.yaml --node julia_marino.authored
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from expertwins import paths  # noqa: E402
from expertwins.netenv import console_utf8, use_system_trust_store  # noqa: E402

use_system_trust_store()
console_utf8()

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PAGE = 100
#: How hard to lean on a 429 before giving up. arXiv's export API rate-limits
#: far more aggressively than its stated "one request every three seconds", and
#: the Fetcher's generic backoff is not enough on its own.
ARXIV_RETRIES = 5
ARXIV_BACKOFF = 15.0

#: node id -> the queries that never returned an answer. A floor check counts
#: what arrived and cannot tell a field that published nothing from a server
#: that refused to answer; these are the queries whose silence means nothing at
#: all, and the floor report has to say so.
NODE_FAILURES: dict[str, list[str]] = {}

_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)


@dataclass
class Preprint:
    arxiv_id: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""
    categories: list[str] = field(default_factory=list)
    doi: str = ""

    @property
    def year(self) -> int | None:
        try:
            return int(self.published[:4])
        except (ValueError, TypeError):
            return None

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}"


def _text(block: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, re.S)
    if not m:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()


def parse_arxiv(xml: str) -> list[Preprint]:
    out: list[Preprint] = []
    for block in _ENTRY.findall(xml):
        raw_id = _text(block, "id")
        aid = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        authors = [re.sub(r"\s+", " ", a).strip()
                   for a in re.findall(r"<author>\s*<name>(.*?)</name>", block, re.S)]
        cats = re.findall(r'<category[^>]*term="([^"]+)"', block)
        out.append(Preprint(
            arxiv_id=aid, title=_text(block, "title"), authors=authors,
            abstract=_text(block, "summary"), published=_text(block, "published"),
            categories=cats, doi=_text(block, "arxiv:doi")))
    return out


def _arxiv_get(fetcher, url: str):
    """GET one arXiv page, retrying 429 rather than reporting it as absence.

    A rate-limited preprint channel can look identical to a field with no
    matching preprints. The floor check counts documents and cannot distinguish
    absence from fetch failure, so persistent rate limits must surface as errors.
    """
    delay = ARXIV_BACKOFF
    for attempt in range(1, ARXIV_RETRIES + 1):
        resp = fetcher.get(url)
        if resp.ok:
            return resp
        if str(resp.status) != "429" or attempt == ARXIV_RETRIES:
            return resp
        print(f"    arXiv 429, waiting {delay:.0f}s "
              f"(attempt {attempt}/{ARXIV_RETRIES})", flush=True)
        time.sleep(delay)
        delay *= 2
    return resp


def arxiv_search(fetcher, query: str, max_records: int = 100) -> list[Preprint]:
    """Page the arXiv Atom API through the corpus fetcher.

    Failure is REPORTED, never returned as an empty result: absence and failure
    must be different types, and a silently empty preprint channel looks
    identical to a field that has published nothing.
    """
    found: list[Preprint] = []
    start = 0
    seen: set[str] = set()
    while len(found) < max_records:
        params = {"search_query": query, "start": start,
                  "max_results": min(ARXIV_PAGE, max_records - len(found)),
                  "sortBy": "relevance", "sortOrder": "descending"}
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        resp = _arxiv_get(fetcher, url)
        if not resp.ok:
            raise RuntimeError(f"arXiv query failed for {query!r}: "
                               f"{resp.error or resp.status}")
        batch = parse_arxiv(resp.text)
        if not batch:
            break
        fresh = [p for p in batch if p.arxiv_id and p.arxiv_id not in seen]
        seen |= {p.arxiv_id for p in fresh}
        found.extend(fresh)
        start += len(batch)
    return found[:max_records]


def to_meta(p: Preprint, topics: list[str], n_passages: int, n_chars: int):
    """Build a DocMeta for a preprint.

    `fulltext_status` is DERIVED from the measured text, never asserted: the
    intent is FULL because a PDF was fetched, and `classify_fulltext` downgrades
    it when the bytes do not support the claim.
    """
    from expertwins.corpus.ids import make_doc_id
    from expertwins.corpus.store import classify_fulltext
    from expertwins.corpus.models import DocMeta, DocType, FullTextStatus, Provenance

    year = p.year or 0
    doc_id = make_doc_id(p.authors or ["anon"], year, "arXiv",
                         fallback=p.arxiv_id.replace(".", "").replace("/", ""))
    status = classify_fulltext(n_passages, n_chars, FullTextStatus.FULL)
    return DocMeta(
        doc_id=doc_id, title=p.title, authors=p.authors, year=year or None,
        pub_date=p.published[:10] if p.published else str(year),
        venue="arXiv", doc_type=DocType.PREPRINT,
        # The honest label. A preprint is not peer reviewed and the record says
        # so, so a seat that wants to discount it can and a reader can tell.
        provenance=Provenance.PREPRINT,
        arxiv_id=p.arxiv_id, doi=p.doi, url=f"https://arxiv.org/abs/{p.arxiv_id}",
        text_sha256="pending", fulltext_status=status,
        n_passages=n_passages, n_chars=n_chars, topics=topics, channel="arxiv")


def harvest_arxiv(node: dict, column: str, library, fetcher,
                  max_records: int) -> tuple[int, int]:
    """Fetch, extract and store one node's arXiv queries. Returns (added, seen)."""
    from expertwins.corpus.errors import Absence
    from expertwins.corpus.extract import extract_pdf

    topics = [node["id"], "col:preprint"] + list(node.get("topics", []))
    if column:
        topics.append(column)
    added = seen = 0
    failed: list[str] = []
    for q in node.get("arxiv", []):
        print(f"  arxiv: {q}", flush=True)
        try:
            records = arxiv_search(fetcher, q, max_records)
        except RuntimeError as exc:
            print(f"    QUERY FAILED: {exc}", flush=True)
            failed.append(q)
            continue
        print(f"    {len(records)} record(s) returned", flush=True)
        for p in records:
            seen += 1
            if not p.title or not p.arxiv_id:
                continue
            try:
                raw = fetcher.get(p.pdf_url)
                if not raw.ok or not raw.content:
                    continue
                # The doc_id is needed BEFORE extraction, because passage ids
                # are built from it. Counts are then measured from the
                # extraction and the meta rebuilt -- and the store measures them
                # again and overwrites whatever is claimed here, which is the
                # property that makes the corpus trustworthy.
                probe = to_meta(p, topics, 0, 0)
                doc = extract_pdf(raw.content, probe.doc_id)
                if isinstance(doc, Absence) or not doc.passages:
                    continue
                n_chars = sum(len(x.text) for x in doc.passages)
                meta = to_meta(p, topics, len(doc.passages), n_chars)
                res = library.put(meta, doc.passages, source_bytes=raw.content)
                if str(getattr(res.outcome, "value", res.outcome)) in (
                        "added", "replaced"):
                    added += 1
                    if added % 25 == 0:
                        print(f"    +{added} stored of {seen} seen", flush=True)
            except Exception as exc:                              # noqa: BLE001
                print(f"    {p.arxiv_id}: {type(exc).__name__}: {exc}", flush=True)
    if failed:
        NODE_FAILURES.setdefault(node["id"], []).extend(failed)
    return added, seen


def harvest_europepmc(node: dict, column: str, acq, per_pass: int) -> int:
    """bioRxiv and medRxiv, through the proven Europe PMC pipeline.

    Deliberately NOT a second implementation. `SRC:PPR` is an ordinary Europe
    PMC query, so the preprint channel for toy fieldcraft is the same acquisition path
    as everything else here, with the same full-text measurement and the same
    title-in-text screen.
    """
    topics = [node["id"], "col:preprint"] + list(node.get("topics", []))
    if column:
        topics.append(column)
    added = 0
    failed: list[str] = []
    for q in node.get("europepmc", []):
        print(f"  europepmc: {q}", flush=True)
        before = acq.stats.added + acq.stats.replaced
        try:
            acq.harvest_query(q, topics=topics, per_pass=per_pass)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    QUERY FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed.append(q)
            continue
        gained = acq.stats.added + acq.stats.replaced - before
        added += gained
        print(f"    +{gained} docs | {acq.stats.render()}", flush=True)
    if failed:
        NODE_FAILURES.setdefault(node["id"], []).extend(failed)
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="library_preprints")
    ap.add_argument("--node", default=None, help="restrict to one node id")
    ap.add_argument("--max-records", type=int, default=60)
    ap.add_argument("--per-pass", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(paths.resolve(a.config).read_text(encoding="utf-8"))
    column = cfg.get("column", "")
    nodes = [n for n in cfg["nodes"] if a.node is None or n["id"] == a.node]
    if not nodes:
        raise SystemExit(f"no node matches {a.node!r}")

    if a.dry_run:
        total = 0
        for n in nodes:
            qs = list(n.get("arxiv", [])) + list(n.get("europepmc", []))
            total += len(qs)
            print(f"{n['id']:<34s} floor {n.get('min_full', 0):>4d}  "
                  f"{len(n.get('arxiv', []))} arXiv + "
                  f"{len(n.get('europepmc', []))} Europe PMC")
            for q in qs:
                print(f"    {q}")
        print(f"\n{total} queries. Nothing downloaded.")
        return 0

    from expertwins.corpus.acquire.pipeline import Acquisition
    from expertwins.corpus.store import Library
    from expertwins.corpus.net import Fetcher

    root = paths.resolve(a.root)
    root.mkdir(parents=True, exist_ok=True)
    library = Library(root)
    fetcher = Fetcher(cache_dir=paths.CACHE)
    run_dir = paths.RUNS / f"acquire-{root.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    acq = Acquisition(library, fetcher, run_dir=run_dir)

    summary: dict[str, dict] = {}
    for n in nodes:
        print(f"\n=== {n['id']}  (floor {n.get('min_full', 0)}) ===", flush=True)
        ax, seen = harvest_arxiv(n, column, library, fetcher, a.max_records)
        ep = harvest_europepmc(n, column, acq, a.per_pass)
        summary[n["id"]] = {"arxiv": ax, "europepmc": ep, "seen": seen}
        print(f"  stored {ax + ep} ({ax} arXiv, {ep} Europe PMC)", flush=True)

    print("\nFLOOR CHECK (committed before acquisition)")
    print(f"{'node':<34s} {'added':>7s} {'floor':>7s}  status")
    thin = 0
    for n in nodes:
        s = summary.get(n["id"], {})
        got = s.get("arxiv", 0) + s.get("europepmc", 0)
        floor = n.get("min_full", 0)
        if got < floor:
            thin += 1
        bad = NODE_FAILURES.get(n["id"], [])
        print(f"{n['id']:<34s} {got:>7d} {floor:>7d}  "
              + ("ok" if got >= floor else "THIN")
              + (f"  ({len(bad)} QUERY FAILURE(S))" if bad else ""))
    if NODE_FAILURES:
        # Printed BEFORE the thin report, because it changes what the thin
        # report means. A node whose queries errored is not evidence about the
        # literature; it is evidence about the network.
        print("\nQUERIES THAT NEVER GOT AN ANSWER -- these say NOTHING about "
              "whether the literature exists. Re-run this node before reading "
              "any floor below as a finding:")
        for node_id, qs in NODE_FAILURES.items():
            for q in qs:
                print(f"  {node_id}: {q}")
    if thin:
        print(f"\n{thin} node(s) below floor."
              + (" Some of them had failing queries (above), so the floor is "
                 "not interpretable for those."
                 if NODE_FAILURES else
                 " A FINDING, not a failure: it says this literature is not "
                 "reachable through this channel either."))
    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {paths.relative(out)}")
    print("\nNext: merge, reindex, and RE-ATTRIBUTE every seat whose own corpus "
          "these queries touched:\n"
          "  python ops/people.py install config/people/<seat>.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())

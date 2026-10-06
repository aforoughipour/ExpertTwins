"""Acquire the field's canon by TITLE, through OpenAlex and the open-access PDF.

    python ops/acquire_canon.py --dry-run
    python ops/acquire_canon.py --root library_canon
    python ops/acquire_canon.py --group canon.frontier_architectures

WHY THIS EXISTS. Every other acquisition in this project is anchored on a
person. The field's landmarks are the one part of the corpus that nothing
derives, so nothing notices when they are missing. A title can be absent even
when acquisition floors are met.

    A FLOOR COUNTS DOCUMENTS. IT CANNOT NOTICE WHICH DOCUMENT IS ABSENT.

WHY NOT THROUGH `acquire_preprints.py`. arXiv's export API answers this network
with HTTP 429, 503 and read timeouts on every query and every retry. Its PDF
host answers normally. So this asks OpenAlex for the work by title and fetches
the open-access PDF directly -- the path `neighbourhood.py::_direct_fetch`
already uses for the conference literature.

NOTHING HERE LANDS IN ANYBODY'S OWN CORPUS. These documents are tagged
`field.canon`, never a seat's own topics, so `attribute` is never asked to
adjudicate them. They are what the panel READS, not what it wrote.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from expertwins import paths  # noqa: E402
from expertwins.netenv import console_utf8, use_system_trust_store  # noqa: E402

use_system_trust_store()
console_utf8()

from ops.neighbourhood import (  # noqa: E402
    OA, _direct_fetch, _is_scholarly, _slim,
)

#: How close an OpenAlex hit has to be before it is accepted as the paper asked
#: for. Title search is fuzzy: "Segment Anything" returns "Segment Anything in
#: Medical Images" first, and a canon list that silently accepts the near-miss
#: is worse than one that reports the gap.
def _norm(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum() or c == " ").strip()


def _match(asked: str, got: str) -> bool:
    a, g = _norm(asked), _norm(got)
    if not a or not g:
        return False
    # The asked-for title must BE the hit, or be its leading phrase -- which is
    # how subtitles differ between the preprint and the journal version.
    return g == a or g.startswith(a) or a.startswith(g)


def resolve(fetcher, title: str) -> dict | None:
    """The OpenAlex record for one canonical title, or None with a reason."""
    url = (f"{OA}/works?filter=title.search:{urllib.parse.quote(title)}"
           f"&per-page=10&mailto=your-email@example.org")
    resp = fetcher.get(url)
    if not resp.ok:
        print(f"    LOOKUP FAILED ({resp.status}): {title}")
        return None
    results = json.loads(resp.text).get("results", [])
    exact = [w for w in results
             if _match(title, w.get("display_name") or w.get("title") or "")]
    if not exact:
        near = (results[0].get("display_name") if results else "")
        print(f"    NO MATCH: {title}"
              + (f"   (closest: {near[:60]})" if near else ""))
        return None
    # Among genuine title matches prefer the most-cited: the canonical version
    # of a paper that exists as preprint, workshop paper and journal article is
    # the one the field actually cites.
    best = max(exact, key=lambda w: w.get("cited_by_count", 0))
    if not _is_scholarly(best):
        print(f"    NOT A PAPER ({best.get('type')}): {title}")
        return None
    return best


#: PDF hosts that reliably serve the paper they claim to. ORDER IS PREFERENCE.
#:
#: THE CASE THIS EXISTS FOR. `Attention Is All You Need` resolved to a record
#: whose `best_oa_location` was a mirror on `langtaosha.org.cn` that answers 404,
#: while `https://arxiv.org/pdf/1706.03762` sat eleven locations further down the
#: same record. Taking the first `pdf_url` OpenAlex offers is taking OpenAlex's
#: word for which copy is real, and for the single most-cited paper in this
#: whole list it was wrong.
TRUSTED_PDF_HOSTS = ("arxiv.org", "europepmc.org", "ncbi.nlm.nih.gov",
                     "biorxiv.org", "medrxiv.org", "openreview.net",
                     "thecvf.com", "nature.com", "science.org")


def _pdf_url(work: dict) -> str:
    """The best open PDF for a work, looking past `best_oa_location`.

    `_slim` reads `best_oa_location` and then `primary_location`, which is right
    for a node built from an author's output and wrong here. OpenAlex can choose
    a publisher landing page as "best" while an open PDF appears further down
    `locations`. The canon is a list of specific papers, so it is worth
    searching every location rather than accepting the first answer.

    The arXiv landing-page rewrite is the other half: a location frequently
    carries `abs/2103.14030` and no `pdf_url` at all, and the PDF is at the
    corresponding `pdf/` path. Only arXiv's *API* host refuses this network;
    the PDF host answers normally.
    """
    locs = [loc for loc in (work.get("best_oa_location"),
                            work.get("primary_location"),
                            *(work.get("locations") or []))
            if isinstance(loc, dict)]
    candidates = [loc["pdf_url"] for loc in locs if loc.get("pdf_url")]
    candidates += [loc["landing_page_url"].replace("/abs/", "/pdf/")
                   for loc in locs
                   if "arxiv.org/abs/" in (loc.get("landing_page_url") or "")]
    for host in TRUSTED_PDF_HOSTS:
        for url in candidates:
            if host in url:
                return url
    return candidates[0] if candidates else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/canon.yaml")
    ap.add_argument("--root", default="library_canon")
    ap.add_argument("--group", default=None, help="restrict to one group id")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from expertwins.corpus.store import Library
    from expertwins.corpus.net import Fetcher

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    groups = [g for g in cfg["groups"]
              if not a.group or g["id"] == a.group]
    if not groups:
        raise SystemExit(f"no group {a.group!r} in {a.config}")

    root = paths.ROOT / a.root
    fetcher = Fetcher(cache_dir=paths.CACHE)
    library = None if a.dry_run else Library(root)

    base_topics = list(cfg.get("topics") or [])
    summary: dict[str, dict] = {}
    total_missing: list[str] = []

    for g in groups:
        print(f"\n=== {g['id']} ===", flush=True)
        print(f"  {' '.join((g.get('note') or '').split())}", flush=True)
        works, missing = [], []
        for title in g["titles"]:
            if a.dry_run:
                print(f"    would resolve: {title}")
                continue
            w = resolve(fetcher, title)
            if not w:
                missing.append(title)
                continue
            slim = _slim(w, "field canon (by title)")
            slim["pdf_url"] = _pdf_url(w) or slim["pdf_url"]
            if not slim["pdf_url"]:
                print(f"    NO OPEN PDF: {title}")
                missing.append(title)
                continue
            print(f"    resolved: {slim['title'][:64]} ({slim['year']})")
            works.append(slim)
        if a.dry_run:
            continue
        topics = base_topics + [g["id"]]
        added = _direct_fetch(works, topics, library, fetcher)
        print(f"  stored {added} of {len(works)} resolved "
              f"({len(missing)} unresolved)", flush=True)
        summary[g["id"]] = {"resolved": len(works), "stored": added,
                            "missing": missing}
        total_missing += missing

    if a.dry_run:
        print("\nNothing downloaded.")
        return 0

    run_dir = paths.RUNS / f"acquire-{root.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "canon.json").write_text(json.dumps(summary, indent=2),
                                        encoding="utf-8")
    print(f"\nwrote {paths.relative(run_dir / 'canon.json')}")
    if total_missing:
        # NAMED, not counted. The whole point of this tool is that a number
        # cannot tell you which paper is missing.
        print(f"\n{len(total_missing)} title(s) NOT acquired -- each one is a "
              f"specific hole in what this panel can be expected to know:")
        for t in total_missing:
            print(f"  - {t}")
    print(f"\nNext:\n  python ops/people.py merge {a.root}\n"
          f"  python ops/check_canon.py")
    print("These are tagged `field.canon` and are nobody's own corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

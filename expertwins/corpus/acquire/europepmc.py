"""Europe PMC — the primary discovery channel.

Three acquisition rules are encoded here:

**1. Do not use the ``OPEN_ACCESS:y`` filter.** It can exclude author
manuscripts that have PMC full text. We filter on nothing and resolve full text
per-record against two providers instead.

**2. Neither sort order works alone.** ``CITED desc`` returns famous but
off-topic work. Relevance sort returns on-topic work with few citations, so any
citation floor can delete exactly the papers needed. The fix is a **third
pass**: relevance sort restricted to date windows.

**3. ``fullTextXML`` 404s frequently.** Europe PMC is the *discovery* channel;
NCBI is the second full-text provider because ``efetch`` can recover full text
that Europe PMC does not return (see ncbi.py).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..errors import Absence
from ..net import Fetcher

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# Era slices for the date-windowed pass. A single undated query returns the
# recent literature and almost nothing else, because relevance ranking on any
# large index is dominated by the last decade. Querying each window separately
# and merging gives every era its own quota, so a seat's corpus contains the
# foundational papers as well as the current ones.
#
# RETUNE THESE FOR YOUR FIELD. The right boundaries sit just after the events
# that reshaped what a literature talks about -- a method becoming routine, a
# nomenclature change, a trial that moved the standard of care. Placing a slice
# end just before a date you care about also makes "did we collect enough
# before that date?" answerable by reading one number instead of re-querying.
ERA_WINDOWS: tuple[tuple[int, int], ...] = (
    (1960, 1989),
    (1990, 1999),
    (2000, 2007),
    (2008, 2016),
    (2017, 2020),
    (2021, _dt.date.today().year),
)

# Publication types Europe PMC uses to mark secondary literature. Recorded, not
# excluded: a single review can flip a correct primary-source fact, so reviews
# must be *labelled* so the caller can cap their weight -- but they are still
# the best map of a field.
_REVIEW_TYPES = {"review", "review-article", "systematic-review"}


@dataclass
class Record:
    """One Europe PMC search result, normalised."""

    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: int | None = None
    pub_date: str = ""
    abstract: str = ""
    is_review: bool = False
    cited_by: int = 0
    has_pmc_fulltext: bool = False
    pub_types: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable dedupe key, preferring the most specific identifier."""
        return (
            f"pmcid:{self.pmcid.upper()}" if self.pmcid
            else f"pmid:{self.pmid}" if self.pmid
            else f"doi:{self.doi.lower()}" if self.doi
            else f"title:{self.title.lower()[:120]}"
        )


def _parse_record(item: dict) -> Record:
    authors: list[str] = []
    for a in (item.get("authorList") or {}).get("author", []) or []:
        name = a.get("fullName") or a.get("collectiveName") or ""
        if name:
            authors.append(name)
    if not authors and item.get("authorString"):
        authors = [p.strip() for p in item["authorString"].split(",") if p.strip()]

    pub_types = [
        t.lower()
        for t in (item.get("pubTypeList") or {}).get("pubType", []) or []
        if isinstance(t, str)
    ]

    # Date precision varies wildly across the corpus. Take the most precise
    # available; a document that cannot be dated is refused later, so getting
    # this right is what keeps the temporal holdout possible.
    pub_date = (
        item.get("firstPublicationDate")
        or item.get("electronicPublicationDate")
        or item.get("journalInfo", {}).get("printPublicationDate")
        or ""
    )
    year = None
    if item.get("pubYear"):
        try:
            year = int(str(item["pubYear"])[:4])
        except ValueError:
            year = None
    if not pub_date and year:
        pub_date = str(year)

    return Record(
        pmid=str(item.get("pmid") or ""),
        pmcid=str(item.get("pmcid") or ""),
        doi=str(item.get("doi") or ""),
        title=str(item.get("title") or "").rstrip("."),
        authors=authors,
        journal=str((item.get("journalInfo") or {}).get("journal", {}).get("title") or ""),
        year=year,
        pub_date=pub_date,
        abstract=str(item.get("abstractText") or ""),
        is_review=bool(_REVIEW_TYPES & set(pub_types)),
        cited_by=int(item.get("citedByCount") or 0),
        has_pmc_fulltext=str(item.get("hasTextMinedTerms") or "") == "Y"
        or str(item.get("inEPMC") or "") == "Y"
        or str(item.get("isOpenAccess") or "") == "Y",
        pub_types=pub_types,
        raw=item,
    )


class EuropePMC:
    def __init__(self, fetcher: Fetcher) -> None:
        self.fetcher = fetcher

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        page_size: int = 100,
        max_records: int = 1000,
        sort: str = "",
    ) -> Iterator[Record]:
        """Paginate a query with the cursor API.

        ``sort=''`` is relevance. Note there is deliberately no ``OPEN_ACCESS``
        or ``SRC`` filter here; see the module docstring.
        """
        cursor = "*"
        yielded = 0
        while yielded < max_records:
            params = {
                "query": query,
                "format": "json",
                "pageSize": min(page_size, max_records - yielded),
                "cursorMark": cursor,
                "resultType": "core",
            }
            if sort:
                params["sort"] = sort
            payload = self.fetcher.get_json(f"{BASE}/search", params)
            if isinstance(payload, Absence):
                return
            results = (payload.get("resultList") or {}).get("result", []) or []
            if not results:
                return
            for item in results:
                yield _parse_record(item)
                yielded += 1
                if yielded >= max_records:
                    return
            next_cursor = payload.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                return
            cursor = next_cursor

    def three_pass_search(
        self,
        query: str,
        *,
        per_pass: int = 400,
        window: tuple[int, int] | None = None,
        era_windows: list[tuple[int, int]] | None = None,
    ) -> tuple[list[Record], dict[str, int]]:
        """Ranking recipe: relevance + citation + date-windowed relevance.

        Returns deduplicated records and a per-pass yield breakdown, because
        knowing which pass produced a document is how you find out that a pass
        has stopped earning its keep.
        """
        seen: dict[str, Record] = {}
        yields: dict[str, int] = {}

        def run(label: str, q: str, sort: str, cap: int | None = None) -> None:
            before = len(seen)
            for rec in self.search(q, max_records=cap or per_pass, sort=sort):
                if rec.key not in seen:
                    seen[rec.key] = rec
            yields[label] = len(seen) - before

        run("relevance", query, "")
        run("cited", query, "CITED desc")

        # Pass 3: the one that mattered. Restricting relevance sort to a date
        # window surfaces on-topic work that neither of the first two passes
        # reaches, because relevance alone is dominated by recent papers and
        # citation sort by old famous ones.
        #
        # A single broad window does not fix recency bias; relevance sort inside
        # it still returns mostly recent papers. Slicing into eras gives each
        # period its own relevance-sorted quota, so historical depth is acquired
        # on purpose rather than left to chance.
        if era_windows is None:
            if window is not None:
                era_windows = [window]
            else:
                era_windows = list(ERA_WINDOWS)
        per_era = max(40, per_pass // max(1, len(era_windows)))
        for lo, hi in era_windows:
            windowed = f"({query}) AND (FIRST_PDATE:[{lo} TO {hi}])"
            run(f"era_{lo}_{hi}", windowed, "", cap=per_era)

        return list(seen.values()), yields

    # -- full text ---------------------------------------------------------

    def fulltext_xml(self, pmcid: str) -> bytes | Absence:
        """Fetch JATS full text. Frequently 404s; the caller must fall back."""
        pmcid = pmcid.upper()
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        resp = self.fetcher.get(f"{BASE}/{pmcid}/fullTextXML", accept="application/xml")
        if not resp.ok:
            return Absence(
                what="Europe PMC full text",
                method="GET europepmc /{pmcid}/fullTextXML",
                query=pmcid,
                examined=1,
                detail={"status": resp.status, "note": "NCBI efetch is the fallback"},
            )
        if len(resp.content) < 500:
            return Absence(
                what="substantive Europe PMC full text",
                method="GET europepmc /{pmcid}/fullTextXML",
                query=pmcid,
                examined=len(resp.content),
                detail={"note": "response too small to be an article"},
            )
        return resp.content

    def citations_of(self, pmcid_or_pmid: str, source: str = "MED", max_records: int = 200) -> list[dict]:
        """Papers citing this one. Half of PaperQA2's citation traversal."""
        out: list[dict] = []
        page = 1
        while len(out) < max_records:
            payload = self.fetcher.get_json(
                f"{BASE}/{source}/{pmcid_or_pmid}/citations",
                {"format": "json", "pageSize": 100, "page": page},
            )
            if isinstance(payload, Absence):
                break
            batch = (payload.get("citationList") or {}).get("citation", []) or []
            if not batch:
                break
            out.extend(batch)
            page += 1
        return out[:max_records]

    def references_of(self, pmcid_or_pmid: str, source: str = "MED", max_records: int = 300) -> list[dict]:
        """Papers this one cites. The other half of citation traversal.

        Reference traversal recovers papers that keyword search may miss. Doing
        it against a structured API instead of a PDF reference parser avoids the
        class of parser failure modes caused by malformed or unusual reference
        sections.
        """
        out: list[dict] = []
        page = 1
        while len(out) < max_records:
            payload = self.fetcher.get_json(
                f"{BASE}/{source}/{pmcid_or_pmid}/references",
                {"format": "json", "pageSize": 100, "page": page},
            )
            if isinstance(payload, Absence):
                break
            batch = (payload.get("referenceList") or {}).get("reference", []) or []
            if not batch:
                break
            out.extend(batch)
            page += 1
        return out[:max_records]

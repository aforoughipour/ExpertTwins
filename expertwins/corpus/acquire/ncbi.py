"""NCBI E-utilities — the second full-text provider.

Europe PMC is the better search engine; NCBI has the wider full-text gate.
Europe PMC frequently fails to return full text that NCBI ``efetch`` can
recover, so acquisition treats NCBI as an independent full-text provider rather
than a fallback metadata source.

Set ``NCBI_API_KEY`` to raise the rate limit from 3/s to 10/s. Acquisition runs
roughly three times faster with one and it is free to obtain.
"""

from __future__ import annotations

from ..errors import Absence
from ..net import Fetcher

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class NCBI:
    def __init__(self, fetcher: Fetcher) -> None:
        self.fetcher = fetcher

    def efetch_pmc(self, pmcid: str) -> bytes | Absence:
        """Fetch JATS XML for a PMCID from PMC."""
        numeric = pmcid.upper().replace("PMC", "")
        resp = self.fetcher.get(
            f"{EUTILS}/efetch.fcgi",
            {"db": "pmc", "id": numeric, "retmode": "xml"},
            accept="application/xml",
        )
        if not resp.ok:
            return Absence(
                what="NCBI PMC full text", method="efetch db=pmc",
                query=pmcid, examined=1, detail={"status": resp.status},
            )
        body = resp.content
        # PMC answers a refusal with a 200 and an XML body saying no. Treating
        # that as success is the "record asserts data is present" failure mode.
        lowered = body[:4000].lower()
        if b"does not allow downloading" in lowered or b"<error" in lowered:
            return Absence(
                what="downloadable NCBI full text", method="efetch db=pmc",
                query=pmcid, examined=len(body),
                detail={"note": "publisher blocks bulk download; body is a refusal notice"},
            )
        if len(body) < 800:
            return Absence(
                what="substantive NCBI full text", method="efetch db=pmc",
                query=pmcid, examined=len(body),
                detail={"note": "body too small to be an article"},
            )
        return body

    def efetch_pubmed(self, pmid: str) -> bytes | Absence:
        """Fetch PubMed metadata/abstract XML. Never full text."""
        resp = self.fetcher.get(
            f"{EUTILS}/efetch.fcgi",
            {"db": "pubmed", "id": str(pmid), "retmode": "xml"},
            accept="application/xml",
        )
        if not resp.ok:
            return Absence(
                what="PubMed record", method="efetch db=pubmed",
                query=str(pmid), examined=1, detail={"status": resp.status},
            )
        return resp.content

    def esearch(self, db: str, term: str, retmax: int = 200) -> list[str] | Absence:
        payload = self.fetcher.get_json(
            f"{EUTILS}/esearch.fcgi",
            {"db": db, "term": term, "retmax": retmax, "retmode": "json"},
        )
        if isinstance(payload, Absence):
            return payload
        ids = ((payload.get("esearchresult") or {}).get("idlist")) or []
        if not ids:
            count = int((payload.get("esearchresult") or {}).get("count") or 0)
            return Absence(
                what=f"{db} records", method="esearch",
                query=term, examined=count,
            )
        return list(ids)

    def elink_pmc_from_pmid(self, pmid: str) -> str | Absence:
        """Find the PMC id *of this article*, and never one that merely cites it.

        Europe PMC's search result often omits the PMCID even when PMC holds the
        article. This recovers full text for records that otherwise look
        abstract-only — another instance of a field's emptiness not meaning the
        thing is absent.

        ELink can return *two* link sets with ``dbto == "pmc"``.
        ``pubmed_pmc`` is the article itself.
        ``pubmed_pmc_refs`` is the list of PMC articles that **cite** it, and for
        a paper that is not in PMC it is the only set returned. Taking the first
        ``dbto == "pmc"`` set can therefore fetch a citing paper and file it
        under the target's citation.

        So the link name is matched exactly. An unrecognised link name is an
        absence, not a candidate.
        """
        payload = self.fetcher.get_json(
            f"{EUTILS}/elink.fcgi",
            {"dbfrom": "pubmed", "db": "pmc", "id": str(pmid),
             "linkname": "pubmed_pmc", "retmode": "json"},
        )
        if isinstance(payload, Absence):
            return payload
        for linkset in payload.get("linksets") or []:
            for db in linkset.get("linksetdbs") or []:
                if db.get("linkname") != "pubmed_pmc":
                    continue
                links = db.get("links") or []
                if links:
                    return f"PMC{links[0]}"
        return Absence(
            what="PMC link", method="elink pubmed->pmc (linkname=pubmed_pmc)",
            query=str(pmid), examined=1,
            detail={"note": "no pubmed_pmc link; any pubmed_pmc_refs set lists "
                            "citing articles and must not be used"},
        )

"""The acquisition pipeline: query in, verified documents out.

For each candidate record the pipeline tries full text from two providers in
order (Europe PMC, then NCBI), extracts, measures, and hands the result to the
library — which decides, from the bytes, what we actually hold. Nothing in this
file is allowed to assert that a document has full text; it can only supply
content and let :func:`expertwins.corpus.store.classify_fulltext` judge it.

Every attempt is journaled to ``runs/<run_id>/attempts.jsonl``, including the
failures and the typed absences. This makes failed attempts auditable, including
"did we already try this one, and why did it not work?"
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..errors import Absence, ExtractionError
from ..ids import disambiguate, make_doc_id
from ..extract import extract_jats
from ..store import IngestOutcome, Library, classify_fulltext
from ..models import DocMeta, DocType, FullTextStatus, Provenance
from ..net import Fetcher
from .europepmc import EuropePMC, Record
from .ncbi import NCBI


@dataclass
class AcquireStats:
    considered: int = 0
    already_held: int = 0
    added: int = 0
    replaced: int = 0
    duplicate: int = 0
    quarantined: int = 0
    write_failed: int = 0
    no_fulltext: int = 0
    extract_failed: int = 0
    undatable: int = 0
    fulltext_from: dict[str, int] = field(default_factory=dict)
    absence_reasons: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        return (
            f"considered={self.considered} added={self.added} replaced={self.replaced} "
            f"already_held={self.already_held} dup={self.duplicate} "
            f"abstract_or_none={self.no_fulltext} extract_failed={self.extract_failed} "
            f"undatable={self.undatable} quarantined={self.quarantined}"
            + (f" WRITE_FAILED={self.write_failed}" if self.write_failed else "")
        )


class Acquisition:
    def __init__(
        self,
        library: Library,
        fetcher: Fetcher,
        *,
        run_dir: Path | None = None,
        keep_abstract_only: bool = False,
    ) -> None:
        self.library = library
        self.fetcher = fetcher
        self.epmc = EuropePMC(fetcher)
        self.ncbi = NCBI(fetcher)
        self.stats = AcquireStats()
        self.keep_abstract_only = keep_abstract_only
        self.run_dir = Path(run_dir) if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self._journal = (self.run_dir / "attempts.jsonl") if self.run_dir else None
        self._taken: set[str] = set(self.library.known_ids())
        self._attempted_keys: set[str] = set()

    # -- journalling -------------------------------------------------------

    def _log(self, event: str, record: Record, **extra: object) -> None:
        if not self._journal:
            return
        payload = {
            "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "pmid": record.pmid,
            "pmcid": record.pmcid,
            "doi": record.doi,
            "title": record.title[:200],
            **{k: (asdict(v) if isinstance(v, Absence) else v) for k, v in extra.items()},
        }
        with self._journal.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _note_absence(self, absence: Absence) -> None:
        key = f"{absence.what} via {absence.method}"
        self.stats.absence_reasons[key] = self.stats.absence_reasons.get(key, 0) + 1

    # -- full text resolution ---------------------------------------------

    def _resolve_fulltext(self, record: Record) -> tuple[bytes | None, str]:
        """Try both providers. Returns (xml_bytes, provider) or (None, '').

        Order matters and is empirical, not aesthetic: Europe PMC first because
        it is faster and less rate-limited, NCBI second because it succeeds on
        cases Europe PMC refuses.
        """
        pmcid = record.pmcid
        if not pmcid and record.pmid:
            linked = self.ncbi.elink_pmc_from_pmid(record.pmid)
            if isinstance(linked, Absence):
                self._note_absence(linked)
            else:
                pmcid = linked
        if not pmcid:
            return None, ""

        xml = self.epmc.fulltext_xml(pmcid)
        if not isinstance(xml, Absence):
            return xml, "europepmc"
        self._note_absence(xml)

        xml = self.ncbi.efetch_pmc(pmcid)
        if not isinstance(xml, Absence):
            return xml, "ncbi_efetch"
        self._note_absence(xml)
        return None, ""

    # -- ingestion ---------------------------------------------------------

    def ingest_record(self, record: Record, topics: list[str], channel: str) -> IngestOutcome | None:
        """Acquire one record. Returns None when it was skipped before ingest."""
        self.stats.considered += 1

        if record.key in self._attempted_keys:
            return None
        self._attempted_keys.add(record.key)

        # Already held? Check identity before spending a full-text request.
        probe_keys = set()
        if record.doi:
            probe_keys.add(f"doi:{record.doi.lower()}")
        if record.pmid:
            probe_keys.add(f"pmid:{record.pmid}")
        if record.pmcid:
            probe_keys.add(f"pmcid:{record.pmcid.upper()}")
        existing_map = self.library._identity_map()  # noqa: SLF001 - same package
        hit = next((existing_map[k] for k in probe_keys if k in existing_map), "")
        if hit and self.library.has(hit):
            self.stats.already_held += 1
            self._add_topics(hit, topics)
            return None

        if not record.pub_date and not record.year:
            self.stats.undatable += 1
            self._log("undatable", record)
            return None

        xml, provider = self._resolve_fulltext(record)

        doc_id = disambiguate(
            make_doc_id(record.authors, record.year, record.journal, fallback=record.key),
            self._taken,
        )

        passages = []
        from_abstract = False
        if xml is not None:
            try:
                extracted = extract_jats(xml, doc_id)
            except ExtractionError as exc:
                self.stats.extract_failed += 1
                self._log("extract_failed", record, error=str(exc))
                extracted = None
            if isinstance(extracted, Absence):
                self._note_absence(extracted)
                self._log("no_body", record, absence=extracted)
                extracted = None
            if extracted is not None:
                passages = extracted.passages
                self.stats.fulltext_from[provider] = (
                    self.stats.fulltext_from.get(provider, 0) + 1
                )

        if not passages:
            # No body. Fall back to the abstract only if explicitly allowed —
            # by default we do not store abstract-only records, because
            # PaperQA2/LitQA2 established that the answers live in the body, and
            # abstract-only records can masquerade as read sources.
            if not (self.keep_abstract_only and record.abstract):
                self.stats.no_fulltext += 1
                return None
            from ..extract import extract_text

            extracted = extract_text(record.abstract, doc_id, section="Abstract")
            if isinstance(extracted, Absence):
                self.stats.no_fulltext += 1
                return None
            passages = extracted.passages
            from_abstract = True

        n_passages = len(passages)
        n_chars = sum(len(p.text) for p in passages)
        # Derive, never assert. A short paper is a PARTIAL document we keep, not
        # a failed FULL document we throw away. Asserting FULL here made the
        # DocMeta validator reject legitimate short papers (case reports, early
        # short-format journals) and silently shrink the corpus, with the same
        # net effect as a thin-source error: missing text.
        status = classify_fulltext(
            n_passages,
            n_chars,
            FullTextStatus.ABSTRACT_ONLY if from_abstract else FullTextStatus.FULL,
        )
        try:
            meta = DocMeta(
                doc_id=doc_id,
                title=record.title,
                authors=record.authors,
                year=record.year,
                pub_date=record.pub_date or str(record.year),
                venue=record.journal,
                doc_type=DocType.REVIEW if record.is_review else DocType.RESEARCH_ARTICLE,
                provenance=Provenance.PUBLISHED,
                doi=record.doi,
                pmid=record.pmid,
                pmcid=record.pmcid,
                text_sha256="pending",
                fulltext_status=status,
                n_passages=n_passages,
                n_chars=n_chars,
                topics=sorted(set(topics)),
                channel=channel,
            )
        except ValueError as exc:
            self.stats.quarantined += 1
            self._log("invalid_meta", record, error=str(exc))
            return None

        try:
            result = self.library.put(meta, passages, source_bytes=xml, source_ext=".xml")
        except OSError as exc:
            # One document failing to land on disk must not destroy a run that
            # has already done substantial API work. This is NOT a silent skip:
            # it is counted, journaled with the full error, and reported at the
            # end of the run, and the manifest is never appended for it -- so
            # the library still cannot claim to hold text it does not hold.
            self.stats.write_failed += 1
            self._log("write_failed", record, doc_id=doc_id, error=str(exc))
            return None
        self._taken.add(result.doc_id or doc_id)

        if result.outcome is IngestOutcome.ADDED:
            self.stats.added += 1
        elif result.outcome is IngestOutcome.REPLACED:
            self.stats.replaced += 1
        elif result.outcome is IngestOutcome.DUPLICATE:
            self.stats.duplicate += 1
            self._add_topics(result.doc_id, topics)
        elif result.outcome is IngestOutcome.QUARANTINED:
            self.stats.quarantined += 1
        self._log(result.outcome.value, record, doc_id=result.doc_id, reason=result.reason)
        return result.outcome

    def _add_topics(self, doc_id: str, topics: list[str]) -> None:
        """Union new topic tags onto a document we already hold.

        A paper found under two coverage nodes belongs to both. Recording only
        the first is how a corpus develops invisible holes.
        """
        if not doc_id or not topics:
            return
        try:
            meta = self.library.load_meta(doc_id)
        except (OSError, ValueError) as exc:
            # Failing to tag a topic is minor; failing to NOTICE that a document
            # we believe we hold cannot be read is not. Journal it as a real
            # event rather than returning as though nothing happened.
            self._log("topic_tag_failed", None, doc_id=doc_id, error=str(exc))
            return
        merged = sorted(set(meta.topics) | set(topics))
        if merged == sorted(meta.topics):
            return
        updated = meta.model_copy(update={"topics": merged})
        passages = self.library.load_passages(doc_id)
        source = None
        if meta.source_filename:
            path = self.library.doc_path(doc_id) / meta.source_filename
            if path.exists():
                source = path.read_bytes()
        ext = Path(meta.source_filename).suffix or ".xml"
        self.library._write(  # noqa: SLF001 - deliberate same-package write
            updated, passages, source, ext, IngestOutcome.REPLACED
        )

    # -- high level --------------------------------------------------------

    def harvest_query(
        self,
        query: str,
        topics: list[str],
        *,
        channel: str = "europepmc",
        per_pass: int = 400,
        window: tuple[int, int] | None = None,
        progress: bool = True,
    ) -> dict:
        """Run the three-pass search for one query and ingest everything found."""
        before = self.stats.added + self.stats.replaced
        records, pass_yields = self.epmc.three_pass_search(
            query, per_pass=per_pass, window=window
        )
        for i, record in enumerate(records, 1):
            self.ingest_record(record, topics, channel)
            if progress and i % 100 == 0:
                print(f"    {i}/{len(records)} — {self.stats.render()}", flush=True)
        gained = self.stats.added + self.stats.replaced - before
        return {
            "query": query,
            "topics": topics,
            "candidates": len(records),
            "pass_yields": pass_yields,
            "gained": gained,
        }

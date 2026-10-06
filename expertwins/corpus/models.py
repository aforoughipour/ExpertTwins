"""The schema is the contract.

The models encode the invariants that make stored evidence auditable. Callers
can propose records, but validators decide whether those records are coherent
enough to enter the corpus.

Two rules are enforced here rather than audited afterwards:

1. **A document with no text does not exist.** A record that claims full text
   while holding zero passages can make downstream checks treat unreadable
   evidence as read. Here, ``FullTextStatus`` must agree with the measured
   passage and character counts or the model refuses to build.

2. **Every document is dated.** The pre-registered rediscovery evaluation needs
   an arbitrary year cut to remain possible forever. An undatable document is
   quarantined, never silently admitted.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ids import is_valid_doc_id
from .textnorm import canonical, words

# Floors below which a "full text" claim is not credible. Abstract-sized
# records can pass a zero-only check while still being too thin to support
# body-text claims.
MIN_FULLTEXT_CHARS = 6_000
MIN_FULLTEXT_PASSAGES = 5
MIN_PASSAGE_WORDS = 3


class FullTextStatus(str, Enum):
    """What we actually hold, as opposed to what we wish we held."""

    FULL = "full"
    """Body text present and above the credibility floors."""

    PARTIAL = "partial"
    """Some body text, but below the floors. Usable, flagged, never counted as full."""

    ABSTRACT_ONLY = "abstract_only"
    """Title and abstract only. Structurally incapable of research-grade support
    (PaperQA2/LitQA2: the answers live in the body, not the abstract)."""

    METADATA_ONLY = "metadata_only"
    """A bibliographic record with no readable text. A citation nobody can check."""


class Provenance(str, Enum):
    """Where the text came from. Caps the evidential weight a document can carry."""

    PUBLISHED = "published"
    PREPRINT = "preprint"
    REGISTRY = "registry"
    """Sponsor-authored summaries, e.g. ClinicalTrials.gov. Not peer reviewed."""
    BOOK = "book"
    INSTITUTIONAL = "institutional"
    """Non-public local material. Kept out of networked acquisition paths."""
    WEB = "web"


class DocType(str, Enum):
    RESEARCH_ARTICLE = "research_article"
    REVIEW = "review"
    """Labelled explicitly because reviews can outweigh primary sources unless
    retrieval and analysis can filter them."""
    CASE_REPORT = "case_report"
    GUIDELINE = "guideline"
    TEXTBOOK = "textbook"
    THESIS = "thesis"
    PREPRINT = "preprint"
    TRIAL_RECORD = "trial_record"
    DATASET = "dataset"
    OTHER = "other"


class Passage(BaseModel):
    """One readable unit of a document, in original document order.

    ``order`` is the document's own sequence, and retrieval must restore it
    rather than sorting by relevance — "preserve original document order" is the
    single most replicated practical finding in the retrieval literature
    (OP-RAG 2409.01666, DOS RAG 2506.03989).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_id: str
    order: int = Field(ge=0)
    section: str = ""
    """Publisher section label ('Results', 'Methods'). Empty when unknown —
    empty means unknown, and is never guessed."""
    text: str

    @field_validator("text")
    @classmethod
    def _text_is_substantive(cls, v: str) -> str:
        v = canonical(v)
        if len(words(v)) < MIN_PASSAGE_WORDS:
            raise ValueError(
                f"passage has fewer than {MIN_PASSAGE_WORDS} words; "
                "fragments are dropped at extraction, not stored"
            )
        return v


class DocMeta(BaseModel):
    """The bibliographic record and integrity anchors for one document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    pub_date: str
    """ISO date, ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``. Required: an undatable
    document cannot participate in a temporal holdout and is quarantined."""

    venue: str = ""
    doc_type: DocType = DocType.OTHER
    provenance: Provenance = Provenance.PUBLISHED

    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    arxiv_id: str = ""
    url: str = ""

    # --- integrity -------------------------------------------------------
    source_sha256: str = ""
    """sha256 of ``source.pdf``/``source.xml``. Empty only when text arrived
    without a retrievable original (e.g. a JATS stream we stored as text)."""
    text_sha256: str
    source_filename: str = ""
    source_bytes: int = 0

    # --- what we actually hold -------------------------------------------
    fulltext_status: FullTextStatus
    n_passages: int = Field(ge=0)
    n_chars: int = Field(ge=0)

    # --- classification ---------------------------------------------------
    topics: list[str] = Field(default_factory=list)
    """Coverage-lattice node ids this document was acquired for or matched to."""
    channel: str = ""
    """Which acquisition channel produced it. Used for yield accounting."""
    retrieved_at: str = ""
    notes: str = ""

    @field_validator("doc_id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not is_valid_doc_id(v):
            raise ValueError(f"doc_id {v!r} is not a valid slug")
        return v

    @field_validator("pub_date")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        v = (v or "").strip()
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                _dt.datetime.strptime(v, fmt)
                return v
            except ValueError:
                continue
        raise ValueError(
            f"pub_date {v!r} is not YYYY, YYYY-MM or YYYY-MM-DD. "
            "Undatable documents are quarantined, not admitted."
        )

    @field_validator("title")
    @classmethod
    def _title_present(cls, v: str) -> str:
        v = canonical(v)
        if len(v) < 4:
            raise ValueError("title is missing or too short to identify the document")
        return v

    @model_validator(mode="after")
    def _status_matches_reality(self) -> DocMeta:
        """Refuse to record a full-text claim the byte counts do not support.

        This is the single most important validator in the file. It is the
        structural guard against empty-source and thin-source records asserting
        text they do not have.
        """
        if self.n_passages == 0 and self.fulltext_status is not FullTextStatus.METADATA_ONLY:
            raise ValueError(
                f"{self.doc_id}: fulltext_status={self.fulltext_status.value} with 0 passages. "
                "A document with no text does not exist."
            )
        if self.fulltext_status is FullTextStatus.FULL:
            if self.n_chars < MIN_FULLTEXT_CHARS or self.n_passages < MIN_FULLTEXT_PASSAGES:
                raise ValueError(
                    f"{self.doc_id}: claims FULL but holds {self.n_passages} passages / "
                    f"{self.n_chars} chars, below the floor of {MIN_FULLTEXT_PASSAGES} / "
                    f"{MIN_FULLTEXT_CHARS}. Use PARTIAL or ABSTRACT_ONLY."
                )
        if self.fulltext_status is FullTextStatus.METADATA_ONLY and self.n_passages:
            raise ValueError(
                f"{self.doc_id}: METADATA_ONLY but holds {self.n_passages} passages"
            )
        if self.year is None and self.pub_date:
            self.year = int(self.pub_date[:4])
        return self

    # -- convenience -------------------------------------------------------

    @property
    def is_readable(self) -> bool:
        """True when an agent can actually cite this document's body."""
        return self.fulltext_status in (FullTextStatus.FULL, FullTextStatus.PARTIAL)

    def identity_keys(self) -> set[str]:
        """Keys used to detect that two records are the same document.

        Local aliases can produce two records where one holds metadata and the
        other holds text. Comparing DOIs and titles can miss them because
        aliases need not share either, so sha256 of the source is included here
        as the key that cannot be faked.
        """
        keys: set[str] = set()
        if self.doi:
            keys.add(f"doi:{self.doi.lower()}")
        if self.pmid:
            keys.add(f"pmid:{self.pmid}")
        if self.pmcid:
            keys.add(f"pmcid:{self.pmcid.upper()}")
        if self.arxiv_id:
            keys.add(f"arxiv:{self.arxiv_id.lower()}")
        if self.source_sha256:
            keys.add(f"sha:{self.source_sha256}")
        if self.text_sha256:
            keys.add(f"text:{self.text_sha256}")
        return keys

    def to_manifest_row(self) -> dict[str, Any]:
        """The append-only manifest projection. Deliberately flat and small."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "first_author": self.authors[0] if self.authors else "",
            "n_authors": len(self.authors),
            "year": self.year,
            "pub_date": self.pub_date,
            "venue": self.venue,
            "doc_type": self.doc_type.value,
            "provenance": self.provenance.value,
            "doi": self.doi,
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "arxiv_id": self.arxiv_id,
            "source_sha256": self.source_sha256,
            "text_sha256": self.text_sha256,
            "source_filename": self.source_filename,
            "source_bytes": self.source_bytes,
            "fulltext_status": self.fulltext_status.value,
            "n_passages": self.n_passages,
            "n_chars": self.n_chars,
            "topics": self.topics,
            "channel": self.channel,
            "retrieved_at": self.retrieved_at,
        }

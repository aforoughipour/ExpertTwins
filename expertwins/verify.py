"""The deterministic verifier, plus the one thing an individual seat adds: TIER.

  1. CITATION VERIFICATION. A quote must occur, character-exact after
     typographic folding, as a contiguous span inside a SINGLE passage of the
     SPECIFIC document cited. Not fuzzy, not embedding, not edit distance.
     The failure mode: a near-miss quote can share most of its characters with
     the source while reversing its meaning. Every fuzzy matcher scores that as
     a match, so matching is character-exact.

  2. PERMITTED-SET MEMBERSHIP. A citation whose doc_id is not in the citing
     seat's permitted set is OUT_OF_SCOPE, full stop. This is the deterministic
     kill for provenance laundering: in a topology where agents read each
     other, a seat can cite a document it was never shown, having picked the id
     out of a peer's claim.

WHAT IS NEW HERE, AND WHY IT IS NOT A SECOND JUDGEMENT.

A seat that is a *person* has a property a seat that is a *discipline* does not:
some of the documents in its permitted set were **written by it**. So every
verified citation carries a TIER:

    own    the seat is an author of the cited document
    read   the seat was handed it but did not write it

The tier is derived by set membership against the seat's own-corpus doc_id set,
which is itself derived from the acquisition manifest. It is not a judgement and
no model produces it.

It buys the thing that makes an individual seat honest. Without tiers, asking
whether a claim belongs to a given person-shaped seat is a matter of taste. With
tiers it becomes: *is any part of this claim anchored in something the seat
wrote, and if not, did it say so?* A seat speaking outside its own body of work
is not forbidden -- scientists read -- but it is **recorded**, and a seat whose
self-anchor rate on its own home territory collapses to zero is not that person.

WHAT THIS MODULE STILL REFUSES TO DO. It does not judge whether a quote SUPPORTS
a claim, and it does not judge whether one claim CONTRADICTS another.
Contradiction is a semantic relation; calling it deterministic would be the same
error this module exists to prevent. Semantic judgements go to the transcript
for a human.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    VERIFIED = "verified"
    QUOTE_NOT_FOUND = "quote_not_found"
    DOC_NOT_HELD = "doc_not_held"
    OUT_OF_SCOPE = "out_of_scope"
    #: Outside the permitted set AND absent from the library entirely. A
    #: different failure from laundering a real document: laundering means the
    #: seat is reproducing pretraining, invention means it made the whole thing
    #: up. Collapsing the two lets an invention hide inside a provenance
    #: statistic: a metric can report no quote fabrication while a wholly
    #: invented citation passes through a different failure channel.
    OUT_OF_SCOPE_INVENTED = "out_of_scope_invented"
    QUOTE_TOO_SHORT = "quote_too_short"


class Tier(str, Enum):
    """Whose paper is this, relative to the citing seat."""

    OWN = "own"
    READ = "read"
    #: Not verified, so the question does not arise.
    UNKNOWN = "unknown"


#: A quote shorter than this cannot distinguish a real citation from a
#: coincidence: a 20-character fragment appears in hundreds of documents.
MIN_QUOTE_CHARS = 40

_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
}
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_QUOTES = {
    ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201a"): "'",
    ord("\u201c"): '"', ord("\u201d"): '"', ord("\u201e"): '"',
    ord("\u00a0"): " ",
}


def canonical_fold(text: str) -> str:
    """Fold typography, never meaning.

    Ligatures, dash variants, curly quotes, non-breaking spaces and line-wrap
    hyphenation are artefacts of PDF extraction and are folded. Words, negations
    and numbers are NOT folded, because those differences are the science.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = text.translate(_DASHES).translate(_QUOTES)
    text = re.sub(r"-\s*\n\s*", "", text)      # "lantern-\nmoss" -> "lanternmoss"
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


@dataclass
class Citation:
    doc_id: str
    quote: str
    status: Status = Status.QUOTE_NOT_FOUND
    tier: Tier = Tier.UNKNOWN
    passage_id: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Status.VERIFIED


@dataclass
class Claim:
    text: str
    seat: str = ""
    citations: list[Citation] = field(default_factory=list)
    declared: dict = field(default_factory=dict)
    claim_type: str = ""
    #: Seat-declared: is this inside the territory the seat actually works on?
    #: Self-reported and therefore advisory -- the mechanical counterpart is
    #: `self_anchored`, which nothing but set membership decides.
    territory: str = ""

    @property
    def grounded(self) -> bool:
        """Grounded only if it HAS citations and ALL of them verify.

        Not 'any'. A claim resting on one real and one fabricated citation is
        not half-trustworthy; it is a claim whose author did not check, and the
        fabricated half is precisely the part that cannot be relied on.
        """
        return bool(self.citations) and all(c.ok for c in self.citations)

    @property
    def self_anchored(self) -> bool:
        """Does at least one VERIFIED citation point at the seat's own work?"""
        return any(c.ok and c.tier is Tier.OWN for c in self.citations)


class PassageSource:
    """Anything that can hand back a document's passages.

    Kept abstract so the verifier can be tested without a library, and so an
    air-gapped cluster can swap the backing store without touching the check.
    """

    def passages(self, doc_id: str) -> list[tuple[str, str]]:
        raise NotImplementedError

    def has(self, doc_id: str) -> bool:
        raise NotImplementedError


class DictSource(PassageSource):
    """In-memory source, for tests and for small air-gapped bundles."""

    def __init__(self, docs: dict[str, list[tuple[str, str]]]) -> None:
        self._docs = docs

    def passages(self, doc_id: str) -> list[tuple[str, str]]:
        return self._docs.get(doc_id, [])

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs


class Verifier:
    def __init__(self, source: PassageSource,
                 permitted: set[str] | None = None,
                 own: set[str] | None = None) -> None:
        self.source = source
        #: None means "no restriction". An EMPTY SET means "this seat may cite
        #: nothing", which is a different thing and must not be conflated --
        #: that conflation is how a silent-abstention bug once read as an
        #: honest answer.
        self.permitted = permitted
        #: doc_ids the seat is an author of. Empty is legitimate (a seat may be
        #: a discipline rather than a person); it simply means every verified
        #: citation lands in tier READ.
        self.own = own or set()
        self._cache: dict[str, list[tuple[str, str]]] = {}

    def _folded(self, doc_id: str) -> list[tuple[str, str]]:
        if doc_id not in self._cache:
            self._cache[doc_id] = [
                (pid, canonical_fold(text))
                for pid, text in self.source.passages(doc_id)
            ]
        return self._cache[doc_id]

    def verify(self, citation: Citation) -> Citation:
        quote = (citation.quote or "").strip()
        folded_quote = canonical_fold(quote)

        if len(folded_quote) < MIN_QUOTE_CHARS:
            citation.status = Status.QUOTE_TOO_SHORT
            citation.detail = (
                f"quote folds to {len(folded_quote)} characters, below the "
                f"{MIN_QUOTE_CHARS}-character floor needed to identify a source")
            return citation

        # Membership BEFORE existence: citing outside the permitted set is a
        # provenance failure even if the document is real and the quote genuine.
        if self.permitted is not None and citation.doc_id not in self.permitted:
            exists = self.source.has(citation.doc_id)
            citation.status = (
                Status.OUT_OF_SCOPE if exists else Status.OUT_OF_SCOPE_INVENTED)
            citation.detail = (
                "document is outside this seat's permitted set; the seat was "
                "never shown it and cannot have read it"
                if exists else
                "document is outside this seat's permitted set AND does not "
                "exist anywhere in the library; the citation was invented")
            return citation

        if not self.source.has(citation.doc_id):
            citation.status = Status.DOC_NOT_HELD
            citation.detail = "no readable document with this id"
            return citation

        for passage_id, hay in self._folded(citation.doc_id):
            # Contiguous span, within ONE passage. No cross-passage assembly:
            # A model can fabricate a ratio by fusing two numbers that merely
            # appeared near each other, so assembling a quote from two places is
            # itself the failure mode.
            if folded_quote in hay:
                citation.status = Status.VERIFIED
                citation.passage_id = passage_id
                citation.tier = Tier.OWN if citation.doc_id in self.own else Tier.READ
                citation.detail = ""
                return citation

        citation.status = Status.QUOTE_NOT_FOUND
        citation.detail = (
            "the cited document exists and is permitted, but contains no passage "
            "with this exact text; the quote was altered, paraphrased or invented")
        return citation

    def verify_claim(self, claim: Claim) -> Claim:
        for c in claim.citations:
            self.verify(c)
        return claim


@dataclass
class Report:
    claims: list[Claim] = field(default_factory=list)

    @property
    def all_citations(self) -> list[Citation]:
        return [c for cl in self.claims for c in cl.citations]

    @property
    def n_grounded(self) -> int:
        return sum(1 for c in self.claims if c.grounded)

    @property
    def fabrication_rate(self) -> float:
        """Fraction of citations whose quote is not in the cited document.

        THIS NUMBER MUST NOT BE READ ALONE. Invented citations caught by the
        permitted-set check are counted in `invented_count`, not here, because
        membership is checked first. Always report the pair.

        It is also a FLOOR-CHECK on the instrument rather than an outcome
        measure: a narrower corpus can produce fewer and more grounded claims
        whether or not the partition is appropriate. Read this statistic with
        coverage and out-of-scope counts.
        """
        cites = self.all_citations
        if not cites:
            return 0.0
        bad = sum(1 for c in cites
                  if c.status in (Status.QUOTE_NOT_FOUND, Status.DOC_NOT_HELD))
        return bad / len(cites)

    @property
    def invented_count(self) -> int:
        return sum(1 for c in self.all_citations
                   if c.status is Status.OUT_OF_SCOPE_INVENTED)

    @property
    def ungrounded_citation_rate(self) -> float:
        """The honest headline: any citation that did not verify, for any reason."""
        cites = self.all_citations
        if not cites:
            return 0.0
        return sum(1 for c in cites if not c.ok) / len(cites)

    @property
    def out_of_scope_count(self) -> int:
        return sum(1 for c in self.all_citations
                   if c.status in (Status.OUT_OF_SCOPE, Status.OUT_OF_SCOPE_INVENTED))

    @property
    def self_anchor_rate(self) -> float:
        """Fraction of GROUNDED claims resting on at least one of the seat's own
        papers.

        The fidelity statistic. Read it against the seat's territory, never in
        the abstract: a low rate on a question squarely inside the person's own
        field says the seat has stopped being that person; the same rate on a
        question far outside it is exactly right, because a researcher asked
        outside their field cites other people or declines.
        """
        grounded = [c for c in self.claims if c.grounded]
        if not grounded:
            return 0.0
        return sum(1 for c in grounded if c.self_anchored) / len(grounded)

    def render(self) -> str:
        lines = [
            f"claims {len(self.claims)}  grounded {self.n_grounded}  "
            f"fabrication {self.fabrication_rate:.1%}  "
            f"invented {self.invented_count}  "
            f"out_of_scope {self.out_of_scope_count}  "
            f"self_anchored {self.self_anchor_rate:.0%}"
        ]
        for claim in self.claims:
            if claim.grounded:
                continue
            lines.append(f"  UNGROUNDED [{claim.seat}] {claim.text[:100]}")
            for c in claim.citations:
                if not c.ok:
                    lines.append(f"    - {c.doc_id}: {c.status.value}: {c.detail}")
        return "\n".join(lines)


def repair_prompt(claim: Claim) -> str:
    """The check-before-emit feedback string.

    The agent is shown NO NEW EVIDENCE -- only which of its own citations
    failed and why -- so it cannot gain support it did not already have. It can
    only correct or withdraw.
    """
    lines = [
        "The following citations in your claim did not verify.",
        "You are being shown no new evidence. Correct the quote to the exact text",
        "of the passage you meant, or WITHDRAW the claim. Do not invent support.",
        "",
        f"CLAIM: {claim.text}",
        "",
    ]
    for c in claim.citations:
        if not c.ok:
            lines.append(f"  [{c.status.value}] {c.doc_id}: {c.detail}")
            lines.append(f"      your quote: {c.quote[:160]!r}")
    return "\n".join(lines)

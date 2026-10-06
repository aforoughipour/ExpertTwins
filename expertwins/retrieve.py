"""Retrieval: what a seat may read, what it is handed, and in what proportion.

Two rules are load-bearing:

  * ORIGINAL DOCUMENT ORDER. Passages are returned in source order, never
    re-sorted by relevance score. BM25 decides WHICH passages; the document
    decides in what ORDER they are read (OP-RAG 2409.01666, DOS RAG 2506.03989).

  * THE PERMITTED SET IS DERIVED FROM WHAT WAS ACTUALLY HANDED OVER, never
    declared alongside it. If it were written by hand it could drift from the
    evidence the seat really saw, and the out-of-scope check would then be
    verifying a claim about a claim.

WHAT AN INDIVIDUAL SEAT ADDS: STRATIFICATION.

A discipline seat needs *its field's* literature. A person needs their own
papers first. So the packet is built in strata:

    stratum 0   PRECISION -- a conjunction of the question's topical terms, so
                the packet is anchored on the intersection the question is
                actually about before anything ranked by an OR is added
    stratum 1   the seat's OWN corpus, question-conditioned, up to `own_floor`
    stratum 2   everything else the seat can reach, filling the remainder
    stratum 2b  COVERAGE TOP-UP -- targeted probes for the question's own terms
                that nothing retrieved so far actually contains
    stratum 2c  CONTEXT -- the passages either side of a selected passage, in
                source order, so a hit is read inside its paragraph rather than
                as a fragment
    stratum 3   UNDER EXAMINATION -- documents another seat cited last turn

The floor is a floor, not a quota: if the person has written nothing the query
can reach, stratum 1 is empty and that is the finding -- it is precisely how the
system knows the question is off their territory. What the floor prevents is the
opposite failure, where a generically-worded question outranks the person's own
work on BM25 and the seat ends up arguing from other people's papers while
wearing their name. That is a persona again, arrived at by accident.

WHY STRATA 2b AND 2c EXIST. A single BM25 pass over an OR of the question's
terms is biased twice: towards long documents that repeat one common term, and
towards isolated fragments. A packet can be FULL and still be THIN: enough
passages for the table, but too little term coverage or document breadth to
answer the question. `PacketAudit` says so, per seat, before a token is spent.

WHY STRATUM 0 EXISTS, AND WHY THE BUDGET IS LARGE. An OR query ranks by BM25 and
can be dominated by common question words rather than by the rare terms that
define the question. A seat can then abstain because retrieval missed available
evidence. In the transcript, a retrieval-caused abstention is indistinguishable
from a scientific negative. The conjunction query is the precision half of the
trade; `fts_safe` remains the recall half; and both run inside a budget large
enough to hold the result.

WHY STRATUM 3 EXISTS -- THE DISCLOSURE RULE.

When a seat cites a document in support of a claim, every other seat is handed
that document, with the cited passage marked, on the next turn. A citation is a
public assertion about what a paper says, and a panel in which nobody else can
open the paper is a panel that cannot catch a misreading. This is the
peer-review step made mechanical: the quote was already checked character-exact
against the source, but *that the quote supports the claim* is a judgement, and
only another scientist can make it.

It is deliberately NOT a licence to adopt another seat's position. The documents
arrive in their own section, marked, with the seats that cited them named, and
the packet asks for a verdict on the use of the evidence -- not for agreement.

AND IT PERTURBS THE INSTRUMENT, so it is kept out of the measurement. Documents
handed to everyone for checking are shared BY CONSTRUCTION; counting them as
evidence overlap would let the disclosure rule manufacture the collapse the
guardrail is looking for. `EvidencePacket.evidence_docs` -- permitted minus
under-examination -- is what the heterogeneity measures read, and the panel
records both.

TWO LEVERS THE GUARDRAIL USES, and they are here rather than there because they
are retrieval operations:

    exclude_docs    re-retrieve with the documents everyone has already cited
                    removed. This is the anti-collapse intervention that acts on
                    evidence instead of on tone.
    restrict_docs   re-retrieve inside a fixed set (in practice, the seat's own
                    corpus). This is re-anchoring.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Passage:
    doc_id: str
    passage_id: str
    order: int
    section: str
    year: int | None
    text: str


# FTS5 treats bare words as column references in some positions and punctuation
# as syntax, so a natural-language question CANNOT be passed to MATCH directly:
# "high-risk lanternmoss" makes FTS look for a column named `risk` and raise.
# That is not hypothetical -- it is how the first round table failed, with every
# seat reporting zero evidence for a question the corpus covers thoroughly.
_FTS_KEYWORDS = {"and", "or", "not", "near"}
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "is", "are", "was", "were", "be", "been", "do", "does", "did", "what",
    "why", "how", "when", "which", "who", "whom", "that", "this", "these",
    "those", "would", "could", "should", "can", "may", "might", "will", "shall",
    "as", "it", "its", "their", "there", "than", "then", "and", "or", "but",
    "if", "after", "before", "during", "between", "about", "into", "over",
    "you", "your", "we", "our", "they", "them", "his", "her", "he", "she",
    # DISCOURSE AND FILLER. These are RARE in a toy-corpus corpus, so every
    # frequency-based filter keeps them, and they carry no topical meaning. In
    # dictated questions can make them look MORE discriminative than actual
    # topical terms to any rarity-based selector.
    #
    # `attention` IS DELIBERATELY ABSENT from this list. In attention-based
    # multiple-instance learning `attention` is the subject of the sentence, not
    # filler. A stopword list is a claim about a corpus.
    "thing", "things", "work", "worked", "working", "important", "importantly",
    "extensively", "question", "questions", "panel", "approach", "approaches",
    "useful", "good", "better", "best", "make", "makes", "making", "idea",
    "ideas", "example", "examples", "well", "lot", "lots", "very", "really",
    "just", "also", "some", "such", "like", "want", "need", "think", "know",
    "say", "says", "said", "see", "get", "got", "give", "gives", "given",
    "use", "used", "using", "one", "two", "three", "first", "second", "next",
    "step", "steps", "way", "ways", "case", "cases", "point", "points",
    "happen", "happens", "happened", "bit", "little", "much", "many", "more",
    "most", "less", "least", "same", "different", "additional", "addition",
    "furthermore", "instance", "etc", "sure", "maybe", "perhaps", "rather",
    "quite", "even", "still", "back", "around", "across", "through", "help",
    "helps", "helping", "helpful", "gaining", "obtain", "result", "results",
    "process", "part", "parts", "place", "places",
}


def question_terms(text: str, *, min_len: int = 3) -> list[str]:
    """The content terms of a question, in order, deduplicated.

    Exposed rather than buried inside `fts_safe` because the packet audit has to
    answer a question the search cannot: *which of these terms does the evidence
    actually contain?* A packet that is full of passages and empty of half the
    question's vocabulary is the thin packet this project kept shipping.
    """
    toks: list[str] = []
    for raw in re.split(r"[^A-Za-z0-9]+", text):
        t = raw.strip().lower()
        if len(t) < min_len or t in _STOPWORDS or t in _FTS_KEYWORDS:
            continue
        if t.isdigit():
            continue
        if t not in toks:
            toks.append(t)
    return toks


def fts_safe(text: str, *, min_len: int = 3) -> str:
    """Turn natural language into a valid FTS5 MATCH expression.

    Content terms are quoted as literals and OR-ed, so BM25 ranks by how many of
    the question's terms a passage carries rather than demanding all of them.
    Requiring all terms is what once made an over-specified probe report a
    thoroughly covered topic as absent.
    """
    return " OR ".join(f'"{t}"' for t in question_terms(text, min_len=min_len))


_QUOTED = re.compile(r'"[^"]*"')
_FTS_OPS = {"AND", "OR", "NOT", "NEAR"}
#: A bare term FTS5 cannot parse: it contains a hyphen, slash, plus or dot, or
#: starts with a digit. `COX-2` parses as the column `COX` minus the term `2`.
_UNSAFE_TERM = re.compile(r"(?<![\"\w])([A-Za-z][A-Za-z0-9]*(?:[-+/.][A-Za-z0-9]+)+"
                          r"|\d[A-Za-z0-9\-+/.]*)(?![\"\w])")


def fts_quote_terms(query: str) -> str:
    """Quote the bare terms in a hand-authored FTS5 expression, and nothing else.

    THE BUG THIS EXISTS FOR, and it was silent in the worst way. Seat queries are
    hand-written FTS expressions whose operators are meant -- `AND`, `OR`,
    parentheses, quoted phrases. But a bare scientific term is not always a
    valid FTS5 token: `COX-2`, `HSP-90`, `ApoE-4` and `SARS-CoV-2` all parse as a
    column reference followed by a subtraction, and SQLite raises `no such column`.

    If those query terms are skipped, the packet can still be built from the
    question-only query, so the failure looks like nothing at all in the seat
    table. What was lost was precisely the thing that makes a seat search AS
    ITSELF.

    Operators, parentheses and already-quoted phrases are left untouched: this
    quotes terms, it does not rewrite grammar.
    """
    out: list[str] = []
    pos = 0
    for m in _QUOTED.finditer(query):
        out.append(_quote_bare(query[pos:m.start()]))
        out.append(m.group(0))          # already a phrase; leave it alone
        pos = m.end()
    out.append(_quote_bare(query[pos:]))
    return "".join(out)


def _quote_bare(segment: str) -> str:
    def repl(m: re.Match) -> str:
        term = m.group(1)
        return term if term.upper() in _FTS_OPS else f'"{term}"'
    return _UNSAFE_TERM.sub(repl, segment)


@dataclass
class PacketAudit:
    """Is this packet actually enough to answer the question with?

    THE FAILURE THIS EXISTS FOR. A packet reports sixty passages and looks
    healthy in the seat table, and the seat then abstains or answers from two
    reviews, because those sixty passages came from four documents and never
    mention half the terms in the question. `n_passages` cannot see that;
    nothing in the run could see it. The distinction is between a packet that is
    FULL and a packet that is SUFFICIENT, and only the second one matters.

    Nothing here is a gate. A thin packet can be the correct and honest state of
    the world -- the seat is off its territory, or the corpus does not hold this
    literature -- and in that case the right output is an abstention, not a
    louder search. What the audit removes is the case where nobody could tell
    which of those two things happened.
    """

    n_passages: int
    n_docs: int
    n_own_docs: int
    n_chars: int
    #: fraction of the question's content terms that appear anywhere in the packet
    term_coverage: float
    uncovered_terms: list[str] = field(default_factory=list)
    #: largest share of the packet held by any single document
    max_doc_share: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def thin(self) -> bool:
        return bool(self.reasons)

    def render(self) -> str:
        head = (f"{self.n_passages} passages / {self.n_docs} docs "
                f"({self.n_own_docs} own), coverage {self.term_coverage:.0%}")
        if not self.reasons:
            return head
        return head + "\n" + "\n".join(f"    THIN: {r}" for r in self.reasons)

    def to_json(self) -> dict:
        return {"n_passages": self.n_passages, "n_docs": self.n_docs,
                "n_own_docs": self.n_own_docs, "n_chars": self.n_chars,
                "term_coverage": round(self.term_coverage, 4),
                "uncovered_terms": self.uncovered_terms,
                "max_doc_share": round(self.max_doc_share, 4),
                "thin": self.thin, "reasons": self.reasons}


#: Audit thresholds. Engineering defaults, not measurements -- stated in one
#: place so they can be argued with and changed in one place.
MIN_DOCS_EXPECTED = 8
MIN_COVERAGE_EXPECTED = 0.55
MAX_SINGLE_DOC_SHARE = 0.45


def audit_packet(packet: "EvidencePacket") -> PacketAudit:
    ev = [p for p in packet.passages if p.doc_id not in packet.examined]
    per_doc: dict[str, int] = {}
    for p in ev:
        per_doc[p.doc_id] = per_doc.get(p.doc_id, 0) + 1
    text = " ".join(p.text for p in ev).lower()
    terms = question_terms(packet.question)
    uncovered = [t for t in terms if t not in text]
    coverage = 1.0 if not terms else 1.0 - len(uncovered) / len(terms)
    share = (max(per_doc.values()) / len(ev)) if ev else 0.0

    reasons: list[str] = []
    if not ev:
        reasons.append("no evidence retrieved at all; the seat can only abstain")
    else:
        if len(per_doc) < MIN_DOCS_EXPECTED:
            reasons.append(
                f"{len(per_doc)} documents (expected >= {MIN_DOCS_EXPECTED}); "
                f"a position built on this few papers is one paper's opinion")
        if coverage < MIN_COVERAGE_EXPECTED:
            reasons.append(
                f"{coverage:.0%} of the question's terms appear in the evidence; "
                f"missing: {', '.join(uncovered[:8])}")
        if share > MAX_SINGLE_DOC_SHARE:
            reasons.append(
                f"one document holds {share:.0%} of the packet; the seat will "
                f"answer from it whatever else is here")
    return PacketAudit(
        n_passages=len(ev), n_docs=len(per_doc),
        n_own_docs=len(set(per_doc) & packet.own), n_chars=len(text),
        term_coverage=coverage, uncovered_terms=uncovered,
        max_doc_share=share, reasons=reasons)


@dataclass
class EvidencePacket:
    """Everything one seat may read for one question, and nothing else."""

    seat: str
    question: str
    passages: list[Passage] = field(default_factory=list)
    doc_titles: dict[str, str] = field(default_factory=dict)
    own: set[str] = field(default_factory=set)
    n_own_passages: int = 0
    #: doc_id -> the seats that cited it last turn. These documents are in the
    #: packet because somebody ELSE stood on them, not because this seat's
    #: question retrieved them.
    examined_by: dict[str, list[str]] = field(default_factory=dict)
    #: doc_id -> the exact quotes those seats used, so the passage carrying the
    #: quote is the one shown rather than whatever BM25 liked best.
    examined_quotes: dict[str, list[str]] = field(default_factory=dict)

    @property
    def permitted(self) -> set[str]:
        """Derived, never asserted -- see module docstring."""
        return {p.doc_id for p in self.passages}

    @property
    def examined(self) -> set[str]:
        """Documents present only because another seat cited them."""
        return {d for d in self.examined_by if d in self.permitted}

    @property
    def evidence_docs(self) -> set[str]:
        """What this seat's own retrieval found: permitted minus under-examination.

        This, not `permitted`, is what the heterogeneity measures read. Documents
        handed to every seat for checking are shared by construction, and
        counting them as overlap would let the disclosure rule manufacture the
        collapse the guardrail exists to detect.
        """
        return self.permitted - self.examined

    @property
    def n_docs(self) -> int:
        return len(self.permitted)

    @property
    def own_docs(self) -> set[str]:
        return self.permitted & self.own

    def audit(self) -> PacketAudit:
        return audit_packet(self)

    def render(self) -> str:
        """The evidence block a seat actually sees.

        Grouped by document, passages in source order within each document, so
        the seat reads a paper rather than a relevance-ranked pile of fragments.
        The seat's own papers are marked and placed first: a person reads their
        own results before anyone else's, and the marker is what makes the
        `own`/`read` citation tier meaningful to the seat rather than only to
        the verifier.

        Documents under examination come LAST and in their own section. They are
        another seat's evidence, shown so it can be checked, and putting them
        first would make the packet a summary of the table instead of the seat's
        own reading.
        """
        by_doc: dict[str, list[Passage]] = {}
        for p in self.passages:
            by_doc.setdefault(p.doc_id, []).append(p)

        examined = self.examined
        mine = [d for d in by_doc if d in self.own and d not in examined]
        read = [d for d in by_doc if d not in self.own and d not in examined]
        out: list[str] = []

        def emit(doc_id: str, mark: str, quotes: list[str] | None = None) -> None:
            ps = sorted(by_doc[doc_id], key=lambda x: x.order)
            title = self.doc_titles.get(doc_id, "")
            year = ps[0].year or "n.d."
            out.append(f"### [{doc_id}] {title} ({year}){mark}")
            prev: int | None = None
            for p in ps:
                # THE GAP MARKER IS NOT DECORATION. Passages are shown in source
                # order, and a reader assumes source order means continuous
                # prose. Where it is not continuous the omission is stated,
                # because a seat that reads across a silent gap will believe the
                # second half follows from the first.
                if prev is not None and p.order > prev + 1:
                    out.append(f"[... {p.order - prev - 1} passage(s) of this "
                               f"document not shown ...]")
                line = f"({p.section}) {p.text}" if p.section else p.text
                if quotes and any(q and q in p.text for q in quotes):
                    line = f">> QUOTED >> {line}"
                out.append(line)
                prev = p.order
            out.append("")

        if mine:
            out.append("--- YOUR OWN PAPERS ---")
            for doc_id in mine:
                emit(doc_id, "  [YOURS]")
        if read:
            out.append("--- WHAT YOU HAVE READ (you did not write these) ---")
            for doc_id in read:
                emit(doc_id, "")
        if examined:
            out.append(EXAMINATION_HEADER)
            for doc_id in sorted(examined):
                who = ", ".join(self.examined_by.get(doc_id, []))
                mark = "  [YOURS]" if doc_id in self.own else ""
                emit(doc_id, f"  [CITED BY: {who}]{mark}",
                     self.examined_quotes.get(doc_id, []))
        return "\n".join(out)


#: How many documents the examination stratum carries in one packet. A real
#: budget, not a formality: each one costs `examine_per_doc` passages OUTSIDE
#: the evidence budget, so 16 documents already add ~64 passages a seat did not
#: retrieve. The caller is told when a turn cites more than this -- see
#: `ops/panel.py::_disclosure_order` for why the overflow must be chosen
#: deterministically and reported rather than dropped in dict order.
DEFAULT_MAX_EXAMINE_DOCS = 16

EXAMINATION_HEADER = """\
--- CITED BY ANOTHER SEAT LAST TURN -- YOUR CHECK, NOT YOUR EVIDENCE ---

Each document below was cited by the seat named on its heading, in support of a
claim you were shown. The quote they used is marked `>> QUOTED >>`. The quote
itself has already been machine-checked for being present in the document, word
for word; what has NOT been checked, and cannot be checked mechanically, is
whether it supports the claim it was used for.

That is what you are being asked for, and it is the ordinary obligation of a
colleague at a table: read the passage, and say whether the use made of it holds.
It is not a licence to adopt their position, and a document appearing here does
NOT become part of your own evidence base. If it is outside your field, say so
and stop -- `cannot_tell` is an honest and expected verdict.
"""


class Retriever:
    def __init__(self, index_path: Path) -> None:
        if not Path(index_path).exists():
            raise FileNotFoundError(f"no index at {index_path}")
        self.db = sqlite3.connect(f"file:{Path(index_path).as_posix()}?mode=ro", uri=True)
        self._df_cache: dict[str, int] = {}
        self._title_vocab: dict[str, int] | None = None
        self.n_passages: int = self.db.execute(
            "SELECT count(*) FROM passages").fetchone()[0]

    # -- helpers -----------------------------------------------------------

    def _titles(self, doc_ids: set[str]) -> dict[str, str]:
        if not doc_ids:
            return {}
        out: dict[str, str] = {}
        ids = list(doc_ids)
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT doc_id, title FROM docs WHERE doc_id IN ({marks})", chunk)
            out.update({d: (t or "") for d, t in rows})
        return out

    def docs_with_topic(self, topic: str) -> set[str]:
        return {r[0] for r in self.db.execute(
            "SELECT doc_id FROM doc_topics WHERE topic = ?", (topic,))}

    def search(self, query: str, limit: int,
               before_year: int | None = None) -> list[Passage]:
        sql = (
            "SELECT p.doc_id, p.passage_id, p.ord, p.section, p.year, p.text "
            "FROM passages_fts f "
            "JOIN passages p ON p.rowid = f.rowid "
            "JOIN docs d ON d.doc_id = p.doc_id "
            "WHERE passages_fts MATCH ? ")
        args: list = [query]
        if before_year is not None:
            sql += "AND d.year IS NOT NULL AND d.year < ? "
            args.append(before_year)
        sql += "ORDER BY bm25(passages_fts) LIMIT ?"
        args.append(limit)
        try:
            rows = self.db.execute(sql, args).fetchall()
        except sqlite3.OperationalError as exc:
            # A malformed FTS query is a reportable condition, never an empty
            # result. Absence and failure must be different types.
            raise RuntimeError(f"FTS query failed for {query!r}: {exc}") from exc
        return [Passage(*r) for r in rows]

    # -- the precision query -----------------------------------------------

    def document_frequency(self, term: str) -> int:
        """How many passages contain this term. Cheap: ~2ms against the index."""
        if term in self._df_cache:
            return self._df_cache[term]
        try:
            n = self.db.execute(
                "SELECT count(*) FROM passages_fts WHERE passages_fts MATCH ?",
                (f'"{term}"',)).fetchone()[0]
        except sqlite3.OperationalError:
            n = 0
        self._df_cache[term] = n
        return n

    def _title_vocabulary(self) -> dict[str, int]:
        """How often each word appears in a DOCUMENT TITLE, built once.

        WHY TITLES, AND WHY NOT A LONGER STOPWORD LIST. Selecting a question's
        precision terms by rarity alone is unstable in both directions:

          dictated  "lanternmoss alignment ... in the arm or the leg"
                    -> picked `leg`, `filled`, `die`, `bit`
          polished  "principally in imaginary Peloria plots ... anatomically
                    accessible" -> picked `principally`, `anatomically`,
                    `plausibly`, `characterised`

        Both are rare in a toy-corpus corpus and neither is a subject. Every
        attempt to fix this by extending a hand-written stopword list is a
        losing race against the next question's vocabulary, and it asserts what
        the corpus should be allowed to decide.

        A title is what an author thought their paper was ABOUT. Words that
        appear in titles are the corpus's own declaration of its topics --
        `cloud kelp`, `alignment`, `attention` and `lantern glow` are
        there, `principally` and `plausibly` are not. This is `derive, never
        assert` applied to query construction, and it is falsifiable by
        something that did not produce it: the titles were written by the
        authors of the corpus, not by this retriever.
        """
        if self._title_vocab is None:
            vocab: dict[str, int] = {}
            for (title,) in self.db.execute(
                    "SELECT title FROM docs WHERE title IS NOT NULL"):
                for tok in set(re.split(r"[^A-Za-z0-9]+", title.lower())):
                    if len(tok) >= 3:
                        vocab[tok] = vocab.get(tok, 0) + 1
            self._title_vocab = vocab
        return self._title_vocab

    def focus_query(self, question: str, *, n_terms: int = 3,
                    max_df_fraction: float = 0.05,
                    min_title_docs: int = 25) -> str | None:
        """A PRECISION query: a conjunction of the question's topical terms.

        The recall query (`fts_safe`) is a flat OR, so BM25 ranks it by the
        commonest words in it. This is its counterweight -- an AND over terms
        that are (a) subjects this corpus writes papers about, and (b) rare
        enough to discriminate -- which lands on passages actually about the
        intersection the question asks about.

        Returns None when fewer than two terms qualify, because a one-term AND
        is just the term and adds nothing the broad query did not have. None is
        an ordinary outcome, not a failure: a question phrased entirely in
        common vocabulary has no precision query, and saying so is better than
        manufacturing one out of whatever happened to be rarest.
        """
        total = self.n_passages or 1
        ceiling = max(1, int(total * max_df_fraction))
        titles = self._title_vocabulary()
        scored: list[tuple[int, str]] = []
        for t in question_terms(question):
            if titles.get(t, 0) < min_title_docs:
                continue          # not a thing this corpus writes papers about
            df = self.document_frequency(t)
            if 0 < df <= ceiling:
                scored.append((df, t))
        if len(scored) < 2:
            return None
        scored.sort()
        return " AND ".join(f'"{t}"' for _, t in scored[:n_terms])

    # -- the packet --------------------------------------------------------

    def passages_of(self, doc_id: str, orders: list[int] | None = None,
                    limit: int = 400) -> list[Passage]:
        """Passages of one document, in source order.

        Used for context expansion and for showing a document that is under
        examination. `orders` selects specific positions; the index on
        (doc_id, ord) makes both forms cheap.
        """
        sql = ("SELECT doc_id, passage_id, ord, section, year, text FROM passages "
               "WHERE doc_id = ? ")
        args: list = [doc_id]
        if orders:
            marks = ",".join("?" * len(orders))
            sql += f"AND ord IN ({marks}) "
            args += list(orders)
        sql += "ORDER BY ord LIMIT ?"
        args.append(limit)
        return [Passage(*r) for r in self.db.execute(sql, args).fetchall()]

    def packet_for_seat(
        self,
        seat_name: str,
        seat_queries: list[str],
        question: str,
        *,
        own: set[str] | None = None,
        own_queries: list[str] | None = None,
        own_floor: int = 60,
        per_query: int = 12,
        max_passages: int = 260,
        max_seeds: int = 200,
        before_year: int | None = None,
        exclude_docs: set[str] | None = None,
        restrict_docs: set[str] | None = None,
        max_per_doc: int = 8,
        min_docs: int = 10,
        neighbours: int = 1,
        examine: dict[str, list[str]] | None = None,
        examine_quotes: dict[str, list[str]] | None = None,
        examine_per_doc: int = 4,
        max_examine_docs: int = DEFAULT_MAX_EXAMINE_DOCS,
    ) -> EvidencePacket:
        """Build one seat's evidence for one question.

        The seat's own query terms are combined with the question, which is what
        makes the seat search IN CHARACTER: it retrieves from the seat's own
        vocabulary rather than from the question alone.

        THE CAPS ARE THERE TO WIDEN, NOT TO NARROW. `max_per_doc` stops one
        verbose review eating the packet; `min_docs` forces a breadth pass when
        BM25 has concentrated on a handful of papers; the coverage top-up probes
        for the question's own terms that nothing retrieved so far contains.
        Each of them REMOVES a way for a nominally full packet to be thin, and
        every one of them can be switched off by setting it to 0.

        TWO BUDGETS, NOT ONE. `max_seeds` bounds the RANKED HITS -- how many
        places in the corpus this seat is pointed at. `max_passages` bounds the
        packet after the surrounding context has been filled in around them.
        With a single budget the seeds spend all of it and the context stratum
        has nothing left, so the packet degenerates into one fragment per
        document: a pile of fragments and not a set of papers, whatever the
        render method groups them by.

        NO CONFIGURED QUERY IS SKIPPED. Allocation is ROUND-ROBIN across the
        query plan rather than first-come-first-served. Under sequential
        allocation the
        budget was spent in order, so a seat with nine retrieval queries had
        four of them never execute -- and the ones that never ran were its LAST
        and most specific, which is to say the part of its vocabulary that makes
        it that seat. Nothing reported it. Every query now contributes before
        any query gets a second helping.

        THE BREADTH/DEPTH TRADE. Depth across many skimmed documents spends
        budget quickly and buys little recall; depth on the one document whose
        interpretation is being challenged is the whole point, and that is what
        stratum 3 spends its separate budget on.

        Changing the defaults changes the retrieval regime, and `ops/panel.py`
        records the budget in the run manifest so two regimes are never
        silently pooled.
        """
        own = set(own or ())
        exclude_docs = set(exclude_docs or ())
        examine = dict(examine or {})
        examine_quotes = dict(examine_quotes or {})
        q_terms = fts_safe(question)
        if not q_terms:
            raise ValueError(
                f"question yields no searchable terms after sanitisation: {question!r}")

        chosen: dict[str, Passage] = {}
        pool: dict[str, Passage] = {}          # everything seen, for the relax pass
        per_doc: dict[str, int] = {}

        def admissible(p: Passage) -> bool:
            if p.doc_id in exclude_docs:
                return False
            if restrict_docs is not None and p.doc_id not in restrict_docs:
                return False
            return True

        def take(p: Passage, cap: int, doc_cap: int) -> bool:
            if len(chosen) >= cap or p.passage_id in chosen:
                return False
            if doc_cap and per_doc.get(p.doc_id, 0) >= doc_cap:
                return False
            chosen[p.passage_id] = p
            per_doc[p.doc_id] = per_doc.get(p.doc_id, 0) + 1
            return True

        def run(query: str, limit: int) -> list[Passage]:
            try:
                hits = self.search(query, limit, before_year)
            except RuntimeError as exc:
                print(f"  [{seat_name}] query skipped: {exc}")
                return []
            out = [p for p in hits if admissible(p)]
            for p in out:
                pool.setdefault(p.passage_id, p)
            return out

        def harvest(queries: list[str], cap: int, doc_cap: int,
                    only: set[str] | None = None, depth_mult: int = 1) -> None:
            """Round-robin over every query in the list. See the docstring.

            Every query is EXECUTED even when the cap is already full, so the
            pool -- which the breadth and relax passes draw on -- holds what the
            seat's later vocabulary found, and so a starved query is never
            mistaken for a query that returned nothing.

            `depth_mult` DEEPENS THE DRAW, and the own-work pass needs it. That
            pass filters a ranked list down to one person's papers, so a draw
            deep enough for the open field returns almost nothing after
            filtering. A floor that cannot be reached is not a floor, and it
            fails silently; the binding constraint is often the depth of the
            underlying draw rather than the floor value itself.
            """
            buckets: list[list[Passage]] = []
            for q in queries:
                hits = run(q, per_query * depth_mult * (4 if only else 1))
                if only is not None:
                    hits = [p for p in hits if p.doc_id in only]
                buckets.append(hits)
            for depth in range(max((len(b) for b in buckets), default=0)):
                if len(chosen) >= cap:
                    return
                for b in buckets:
                    if depth < len(b):
                        take(b[depth], cap, doc_cap)
                        if len(chosen) >= cap:
                            return

        # The seed budget: the ranked hits, before any context is filled in.
        seed_cap = min(max_seeds, max_passages)

        # Stratum 0 -- PRECISION. A conjunction of the question's topical terms,
        # run first and into its own bucket, so the packet is anchored on the
        # intersection the question is about before the OR-ranked passes -- which
        # are ranked by the question's COMMONEST words -- add anything.
        focus = self.focus_query(question)
        if focus:
            harvest([focus], min(seed_cap, max(8, seed_cap // 4)), max_per_doc)

        # Stratum 1 -- the person's own work, question-conditioned.
        if own and own_floor > 0:
            oq = [q_terms] + [f"({q_terms}) AND ({fts_quote_terms(sq)})"
                                  for sq in (own_queries or [])]
            if focus:
                oq.insert(0, focus)
            harvest(oq, min(own_floor + len(chosen), seed_cap), max_per_doc,
                        only=own, depth_mult=10)

        # Stratum 2 -- everything else the seat can reach. The question alone
        # goes first, so a seat is never denied the most direct evidence merely
        # because its own vocabulary does not contain the question's words. The
        # seat's BARE vocabulary goes last, as a floor: when the conjunction of
        # a seat query with the question happens to return nothing, the seat
        # should still be handed its own field's literature rather than nothing.
        #
        # A SLICE OF THE BUDGET IS WITHHELD from this stratum. Without it the
        # ranked pass spends the whole packet and the coverage and breadth
        # passes below have nothing left to spend -- which is exactly how the
        # thin packets happened: nominally full, and still missing the term the
        # question turns on.
        terms = question_terms(question)
        reserve = min(max(4, seed_cap // 5), 40) if terms else 0
        main_cap = max(own_floor, seed_cap - reserve)
        harvest([q_terms]
                + [f"({q_terms}) AND ({fts_quote_terms(sq)})" for sq in seat_queries]
                + [fts_quote_terms(sq) for sq in seat_queries],
                main_cap, max_per_doc)

        # Stratum 2b -- COVERAGE TOP-UP. An OR of terms is ranked by BM25, which
        # rewards a passage carrying many occurrences of one common term over a
        # passage carrying the rare one that the question actually turns on. So
        # ask directly for the terms nothing has returned yet.
        if terms:
            for _pass in range(2):
                have = " ".join(p.text for p in chosen.values()).lower()
                missing = [t for t in terms if t not in have]
                if not missing or len(chosen) >= seed_cap:
                        break
                for t in missing:
                        harvest([f'"{t}"'], seed_cap, max(2, max_per_doc // 2))

        # Stratum 2c -- BREADTH. A packet drawn from four documents is one
        # school of thought with four citations. Widen by taking a single
        # passage from documents not yet represented.
        if min_docs and len(per_doc) < min_docs:
            for p in list(pool.values()):
                if len(per_doc) >= min_docs or len(chosen) >= seed_cap:
                        break
                if p.doc_id not in per_doc:
                        take(p, seed_cap, max_per_doc)
            if len(per_doc) < min_docs:
                for p in run(q_terms, per_query * 40):
                    if len(per_doc) >= min_docs or len(chosen) >= seed_cap:
                        break
                    if p.doc_id not in per_doc:
                        take(p, seed_cap, max_per_doc)

        # The relax pass. If the per-document cap has left the packet short of
        # its budget, spend the remainder rather than hand over a thin packet:
        # the cap exists to prevent concentration, not to withhold evidence.
        if len(chosen) < seed_cap:
            for p in pool.values():
                if not take(p, seed_cap, max_per_doc * 2):
                    if len(chosen) >= seed_cap:
                        break

        # Stratum 2d -- CONTEXT. A hit is read inside its paragraph. The
        # neighbours are NOT ranked evidence; they are what makes a quoted
        # sentence legible, and they arrive in source order like everything else.
        #
        # This is what the second budget buys. Neighbours are filled around the
        # BEST-RANKED seeds first -- `chosen` is in the order the passes took
        # them -- and never displace a seed, so widening the radius cannot
        # reduce topical coverage. It can only make what was already selected
        # legible.
        if neighbours:
            wanted: dict[str, set[int]] = {}
            for p in list(chosen.values()):
                for d in range(-neighbours, neighbours + 1):
                    if d:
                        wanted.setdefault(p.doc_id, set()).add(p.order + d)
            budget = max(max_passages, len(chosen))
            for doc_id, orders in wanted.items():
                if len(chosen) >= budget:
                    break
                for p in self.passages_of(doc_id, sorted(o for o in orders if o >= 0)):
                    if p.passage_id not in chosen and admissible(p):
                        chosen[p.passage_id] = p
                        per_doc[p.doc_id] = per_doc.get(p.doc_id, 0) + 1
                    if len(chosen) >= budget:
                        break

        # Stratum 3 -- UNDER EXAMINATION. Documents another seat cited, handed
        # over so their use can be checked. Outside the evidence budget on
        # purpose: this is not this seat's evidence and must not displace it.
        examined_by: dict[str, list[str]] = {}
        used_quotes: dict[str, list[str]] = {}
        for doc_id in list(examine)[:max_examine_docs or None]:
            if exclude_docs and doc_id in exclude_docs:
                # The guardrail withheld this document from this seat on
                # purpose. Disclosure does not get to undo an intervention.
                continue
            quotes = [q for q in examine_quotes.get(doc_id, []) if q]
            shown = self._examination_passages(doc_id, quotes, examine_per_doc)
            if not shown:
                continue
            for p in shown:
                chosen.setdefault(p.passage_id, p)
            examined_by[doc_id] = list(examine[doc_id])
            used_quotes[doc_id] = quotes

        passages = list(chosen.values())
        return EvidencePacket(
            seat=seat_name,
            question=question,
            passages=passages,
            doc_titles=self._titles({p.doc_id for p in passages}),
            own=own,
            n_own_passages=sum(1 for p in passages if p.doc_id in own),
            examined_by=examined_by,
            examined_quotes=used_quotes,
        )

    def _examination_passages(self, doc_id: str, quotes: list[str],
                              per_doc: int) -> list[Passage]:
        """The passages of a cited document that a checking seat must see.

        Anchored on the quote rather than on relevance: the point is to show the
        sentence the other seat stood on, in its own surroundings. If no stored
        passage contains the quote the document is still shown from its start --
        and that disagreement is itself worth seeing, because a quote that
        verified at ingest and cannot be located now means the index and the
        library have diverged.
        """
        all_ps = self.passages_of(doc_id)
        if not all_ps:
            return []
        hit_ords = [p.order for p in all_ps if any(q in p.text for q in quotes)]
        if not hit_ords:
            return all_ps[:per_doc]
        want: set[int] = set()
        for o in hit_ords:
            want |= {o - 1, o, o + 1}
        picked = [p for p in all_ps if p.order in want]
        if len(picked) <= per_doc:
            return picked
        # Keep every passage that carries a quote, then fill with context.
        anchored = [p for p in picked if p.order in hit_ords]
        rest = [p for p in picked if p.order not in hit_ords]
        keep = (anchored + rest)[:max(per_doc, len(anchored))]
        return sorted(keep, key=lambda p: p.order)

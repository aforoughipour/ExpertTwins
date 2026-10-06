"""The packet is the instrument. These tests hold its shape.

Every defect locked down here is silent: the packet is still built, the seat
table still shows a healthy passage count, and the seat answers confidently
from whatever it was handed. A retrieval failure and a scientific negative are
indistinguishable in a transcript, which is why they are tested here rather
than watched for in a run.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expertwins.retrieve import (  # noqa: E402
    EvidencePacket, Passage, Retriever, question_terms,
)


def _index(tmp_path: Path, docs: list[tuple[str, str, list[str]]]) -> Path:
    """A tiny library index: (doc_id, title, [passage texts])."""
    p = tmp_path / "index.sqlite"
    db = sqlite3.connect(p)
    db.executescript(
        "CREATE TABLE docs (doc_id TEXT PRIMARY KEY, title TEXT, year INT);"
        "CREATE TABLE passages (doc_id TEXT, passage_id TEXT PRIMARY KEY,"
        " ord INT, section TEXT, year INT, text TEXT);"
        "CREATE VIRTUAL TABLE passages_fts USING fts5(text);"
        "CREATE TABLE doc_topics (doc_id TEXT, topic TEXT);")
    for doc_id, title, texts in docs:
        db.execute("INSERT INTO docs VALUES (?,?,?)", (doc_id, title, 2020))
        for i, t in enumerate(texts):
            db.execute("INSERT INTO passages VALUES (?,?,?,?,?,?)",
                       (doc_id, f"{doc_id}#p{i}", i, "Body", 2020, t))
            rid = db.execute("SELECT rowid FROM passages WHERE passage_id=?",
                             (f"{doc_id}#p{i}",)).fetchone()[0]
            db.execute("INSERT INTO passages_fts(rowid, text) VALUES (?,?)",
                       (rid, t))
    db.commit()
    db.close()
    return p


# --------------------------------------------------------------------------
# the precision query
# --------------------------------------------------------------------------

def test_focus_query_is_a_conjunction_of_the_corpus_own_subjects(tmp_path):
    """The counterweight to the OR. A flat OR is ranked by the question's
    COMMONEST words, so the packet can drift off the intersection the question
    is about."""
    docs = [(f"d{i}", "Silvering in PMN-prism spores of the Peloria plot", ["common word"] * 2)
            for i in range(30)]
    docs += [(f"c{i}", "A study of plots", ["plots " * 10]) for i in range(80)]
    docs += [("hit", "Silvering in PMN-prism spores",
              ["silvering in prism-spore of plots"])]
    r = Retriever(_index(tmp_path, docs))
    q = r.focus_query("do PMN-prism spores in plots undergo silvering?")
    assert q is not None
    assert " AND " in q
    # `plots` is in 80 titles but in too many passages to discriminate;
    # `silvering` is what the corpus writes papers about.
    assert '"silvering"' in q


def test_focus_query_abstains_rather_than_inventing_one(tmp_path):
    """None is an ordinary outcome. A question phrased entirely in vocabulary
    this corpus has no papers about has no precision query, and saying so beats
    manufacturing one out of whatever happened to be rarest."""
    r = Retriever(_index(tmp_path, [("d", "A title", ["some text here"])]))
    assert r.focus_query("what about the leg and the bit that filled?") is None


def test_a_one_term_conjunction_is_not_a_conjunction(tmp_path):
    docs = [(f"d{i}", "Silvering everywhere", ["silvering"]) for i in range(30)]
    r = Retriever(_index(tmp_path, docs))
    assert r.focus_query("silvering?") is None


# --------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------

def test_no_configured_seat_query_is_starved(tmp_path):
    """THE DEFECT: the budget was spent first-come-first-served, so a seat with
    nine retrieval queries had four never execute -- and the ones that never ran
    were its LAST and most specific, which is the part of its vocabulary that
    makes it that seat. Nothing reported it."""
    docs = ([("alpha", "Alpha", ["alpha topic"] * 40)]
            + [("beta", "Beta", ["beta topic"] * 40)]
            + [("omega", "Omega", ["omega topic"] * 40)])
    r = Retriever(_index(tmp_path, docs))
    pkt = r.packet_for_seat(
        "s", ["alpha", "beta", "omega"], "topic",
        max_passages=12, max_seeds=12, min_docs=0, neighbours=0,
        max_per_doc=8)
    # The last-listed query's document must be represented, not crowded out.
    assert "omega" in pkt.permitted, sorted(pkt.permitted)
    assert {"alpha", "beta", "omega"} <= pkt.permitted


def test_seeds_and_passages_are_separate_budgets(tmp_path):
    """With one budget the seeds spend all of it and the context stratum has
    nothing left, so the packet degenerates into one fragment per document. That
    is a pile of fragments, not a set of papers, whatever the render method
    groups them by."""
    docs = [(f"d{i}", f"Topic {i}", [f"topic passage {j}" for j in range(6)])
            for i in range(40)]
    r = Retriever(_index(tmp_path, docs))
    pkt = r.packet_for_seat("s", [], "topic", max_seeds=20, max_passages=60,
                            min_docs=0, neighbours=1, own_floor=0,
                            per_query=60, max_per_doc=1)
    assert len(pkt.passages) > 20, "context never expanded past the seed cap"
    assert len(pkt.passages) <= 60
    assert len(pkt.passages) / pkt.n_docs > 1.5, "still one fragment per document"


def test_context_never_displaces_a_seed(tmp_path):
    """Widening the radius must not be able to reduce topical coverage: the
    seeds are chosen first and neighbours only fill in around them."""
    docs = [(f"d{i}", f"Topic {i}", [f"topic passage {j}" for j in range(6)])
            for i in range(40)]
    r = Retriever(_index(tmp_path, docs))
    flat = r.packet_for_seat("s", [], "topic", max_seeds=30, max_passages=30,
                             min_docs=0, neighbours=0, own_floor=0)
    deep = r.packet_for_seat("s", [], "topic", max_seeds=30, max_passages=90,
                             min_docs=0, neighbours=1, own_floor=0)
    assert flat.permitted <= deep.permitted


def test_the_own_floor_is_actually_reachable(tmp_path):
    """A FLOOR THAT CANNOT BE REACHED IS NOT A FLOOR, and it fails silently.
    The own-work pass filters a ranked list down to one person's papers, so a
    draw deep enough for the open field can return almost nothing after
    filtering. The binding constraint is the depth of the underlying draw, not
    just the requested floor."""
    docs = [(f"theirs{i}", f"Topic {i}", ["topic " * 20]) for i in range(60)]
    docs += [(f"mine{i}", f"Topic mine {i}", ["topic"]) for i in range(12)]
    r = Retriever(_index(tmp_path, docs))
    pkt = r.packet_for_seat("s", [], "topic", own={f"mine{i}" for i in range(12)},
                            own_floor=10, max_seeds=80, max_passages=80,
                            min_docs=0, neighbours=0)
    assert len(pkt.own_docs) >= 10, sorted(pkt.own_docs)


# --------------------------------------------------------------------------
# what the seat actually reads
# --------------------------------------------------------------------------

def test_a_gap_between_passages_is_stated(tmp_path):
    """Passages are shown in source order, and a reader takes source order to
    mean continuous prose. Where it is not, the omission is stated -- a seat
    that reads across a silent gap will believe the second half follows from
    the first."""
    pkt = EvidencePacket(
        seat="s", question="q", doc_titles={"d": "D"},
        passages=[Passage("d", "d#p0", 0, "", 2020, "first"),
                  Passage("d", "d#p7", 7, "", 2020, "much later")])
    out = pkt.render()
    assert "6 passage(s) of this document not shown" in out


def test_contiguous_passages_are_not_marked(tmp_path):
    pkt = EvidencePacket(
        seat="s", question="q", doc_titles={"d": "D"},
        passages=[Passage("d", "d#p0", 0, "", 2020, "first"),
                  Passage("d", "d#p1", 1, "", 2020, "second")])
    assert "not shown" not in pkt.render()


# --------------------------------------------------------------------------
# the stopword list is a claim about a corpus
# --------------------------------------------------------------------------

def test_dictated_filler_is_not_treated_as_a_subject():
    """These are RARE in a toy-corpus corpus, so every frequency-based filter
    keeps them, and they carry no topical meaning. A question dictated by voice
    -- which is how the questions here are actually written -- is full of them."""
    terms = question_terms(
        "The important thing I want to know is, approximately, how the panel "
        "would approach this question in the first place")
    for filler in ("important", "thing", "want", "know", "panel", "approach",
                   "question", "first", "place"):
        assert filler not in terms, filler


def test_attention_is_a_subject_in_this_corpus_and_is_kept():
    """`attention` is not filler in a computational corpus.

    In attention-based multiple-instance learning, `attention` is the subject of
    the sentence. A stopword list is a claim about a corpus.
    """
    assert "attention" in question_terms(
        "does attention-based multiple instance learning localise the signal?")

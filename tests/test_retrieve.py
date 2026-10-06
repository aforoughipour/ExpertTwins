"""Tests for retrieval, and for the FTS expression a seat actually runs.

The defect these lock down was SILENT, which is why it gets its own file. A seat
query containing a hyphenated identifier -- `COX-2`, `HSP-90`, `SARS-CoV-2` -- is not a valid
FTS5 expression: it parses as a column reference followed by a subtraction. The
query was skipped, the packet was still built from the question-only query, and
the seat table showed a healthy passage count. What had been lost was the thing
that makes a seat search AS ITSELF.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expertwins.retrieve import (  # noqa: E402
    EvidencePacket, Passage, fts_quote_terms, fts_safe, question_terms,
)

SEATS = Path(__file__).resolve().parents[1] / "config" / "seats.yaml"


def test_hyphenated_scientific_terms_are_quoted():
    """`COX-2` parses as column `COX` minus term `2`, and SQLite raises
    `no such column`. Any query containing such a term would be skipped
    unless the bare term is quoted."""
    q = fts_quote_terms("(COX-2 OR COX-1 OR PTGS) AND fever")
    assert '"COX-2"' in q and '"COX-1"' in q
    assert "PTGS" in q and '"PTGS"' not in q      # safe bare word, left alone
    assert " AND " in q and " OR " in q           # operators survive


def test_operators_and_parentheses_are_never_quoted():
    q = fts_quote_terms('(a-b OR c) AND NOT (d/e NEAR f)')
    for op in ("AND", "OR", "NOT", "NEAR"):
        assert f'"{op}"' not in q
    assert q.count("(") == 2 and q.count(")") == 2


def test_already_quoted_phrases_are_left_alone():
    q = fts_quote_terms('("echo pollen" OR HSP-90) AND "heat shock"')
    assert '"echo pollen"' in q
    assert q.count('"echo pollen"') == 1     # not double-quoted
    assert '"HSP-90"' in q


def test_leading_digit_terms_are_quoted():
    assert '"5-HT2A"' in fts_quote_terms("(5-HT2A OR SERT) AND agonist")


def test_quoting_is_idempotent():
    once = fts_quote_terms("(COX-2 OR ApoE-4) AND fever")
    assert fts_quote_terms(once) == once


def test_every_shipped_seat_query_survives_quoting():
    """The registry is generated from the person specs, so this catches a bad
    query the moment it is installed rather than during a run."""
    if not SEATS.exists():                                        # pragma: no cover
        import pytest
        pytest.skip("config/seats.yaml is absent; no shipped seat queries to inspect")
    seats = yaml.safe_load(SEATS.read_text(encoding="utf-8"))["seats"]
    assert seats
    for s in seats:
        for key in ("queries", "own_queries"):
            for q in s.get(key) or []:
                out = fts_quote_terms(q)
                assert out.count("(") == out.count(")"), f"{s['name']}: {q}"
                assert out.count('"') % 2 == 0, f"{s['name']}: {q}"


def test_fts_safe_strips_a_question_to_literals():
    q = fts_safe("Why do high-risk lanternmoss plots dim?")
    assert '"lanternmoss"' in q and '"dim"' in q
    assert '"why"' not in q and '"do"' not in q       # stopwords dropped
    assert " OR " in q                                # OR-ed, never AND-ed


def test_fts_safe_refuses_to_produce_an_empty_expression():
    assert fts_safe("the of and in on at") == ""


def test_packet_puts_own_papers_first_and_marks_them():
    """A person reads their own results before anyone else's, and the marker is
    what makes the own/read citation tier mean anything to the seat."""
    ps = [
        Passage("theirs", "theirs#p0", 0, "Body", 2020, "Someone else wrote this."),
        Passage("mine", "mine#p0", 0, "Body", 2021, "I wrote this one myself."),
    ]
    pkt = EvidencePacket(seat="x", question="q", passages=ps,
                         doc_titles={"mine": "Mine", "theirs": "Theirs"},
                         own={"mine"}, n_own_passages=1)
    text = pkt.render()
    assert text.index("YOUR OWN PAPERS") < text.index("[mine]")
    assert text.index("[mine]") < text.index("[theirs]")
    assert "[YOURS]" in text.split("[theirs]")[0]
    assert "[YOURS]" not in text.split("[theirs]")[1]
    assert pkt.own_docs == {"mine"}


def test_permitted_set_is_derived_from_the_passages_handed_over():
    """Never declared alongside the packet: a hand-written permitted set could
    drift from the evidence actually shown, and the out-of-scope check would
    then be verifying a claim about a claim."""
    ps = [Passage("a", "a#p0", 0, "", 2020, "x"),
          Passage("b", "b#p0", 0, "", 2020, "y")]
    pkt = EvidencePacket(seat="x", question="q", passages=ps)
    assert pkt.permitted == {"a", "b"}
    assert pkt.n_docs == 2


# --------------------------------------------------------------------------
# the disclosure stratum
# --------------------------------------------------------------------------

def _packet(**kw):
    ps = kw.pop("passages", None) or [
        Passage("mine", "mine#p0", 0, "Body", 2021, "I measured this myself."),
        Passage("read", "read#p0", 0, "Body", 2020, "Someone else measured it."),
        Passage("theirs", "theirs#p0", 3, "Body", 2019,
                "Suppression was assayed in co-culture."),
    ]
    args = dict(seat="x", question="does suppression require an assay?",
                passages=ps, own={"mine"},
                doc_titles={"mine": "M", "read": "R", "theirs": "T"})
    args.update(kw)
    return EvidencePacket(**args)


def test_examined_documents_are_permitted_but_are_not_evidence():
    """The distinction the whole disclosure rule rests on. A cross-examined
    document must be CITABLE -- a seat cannot rebut a paper it may not quote --
    and must not count as this seat's evidence, because every seat was handed
    it and the overlap would be an artifact of the rule itself."""
    pkt = _packet(examined_by={"theirs": ["carla_moreau"]})
    assert "theirs" in pkt.permitted
    assert pkt.examined == {"theirs"}
    assert pkt.evidence_docs == {"mine", "read"}


def test_examined_section_names_the_seat_and_marks_the_quote():
    pkt = _packet(examined_by={"theirs": ["carla_moreau"]},
                  examined_quotes={"theirs": ["Suppression was assayed"]})
    text = pkt.render()
    assert "CITED BY ANOTHER SEAT" in text
    assert "[CITED BY: carla_moreau]" in text
    assert ">> QUOTED >>" in text
    # last, never first: the packet is the seat's own reading, not a summary
    # of the table.
    assert text.index("YOUR OWN PAPERS") < text.index("CITED BY ANOTHER SEAT")
    assert text.index("WHAT YOU HAVE READ") < text.index("CITED BY ANOTHER SEAT")


def test_a_document_under_examination_is_not_double_printed():
    pkt = _packet(examined_by={"theirs": ["carla_moreau"]})
    assert pkt.render().count("### [theirs]") == 1


def test_audit_counts_only_the_seats_own_evidence():
    """A packet padded out with documents handed over for checking is still a
    thin packet, and the audit has to say so or the disclosure rule becomes a
    way of hiding thinness."""
    ps = [Passage("mine", "mine#p0", 0, "", 2021, "suppression assay coculture")]
    ps += [Passage(f"ex{i}", f"ex{i}#p0", 0, "", 2020, "text " * 20)
           for i in range(9)]
    pkt = _packet(passages=ps,
                  examined_by={f"ex{i}": ["someone"] for i in range(9)})
    audit = pkt.audit()
    assert audit.n_docs == 1
    assert audit.thin
    assert any("documents" in r for r in audit.reasons)


def test_audit_reports_question_terms_the_evidence_never_mentions():
    """The thin packet that reads as full. Sixty passages, and the term the
    question turns on appears in none of them."""
    ps = [Passage("d1", "d1#p0", 0, "", 2020, "silvering in glasswing voles")]
    pkt = _packet(question="do prism spore clusters undergo silvering after moonlight?",
                  passages=ps, examined_by={})
    audit = pkt.audit()
    assert "moonlight" in audit.uncovered_terms
    assert audit.term_coverage < 1.0
    assert audit.thin


def test_audit_flags_a_packet_dominated_by_one_document():
    ps = [Passage("hog", f"hog#p{i}", i, "", 2020, "silvering moonlight prism spore clusters")
          for i in range(9)]
    ps.append(Passage("other", "other#p0", 0, "", 2020, "undergo"))
    pkt = _packet(question="do prism spore clusters undergo silvering after moonlight?",
                  passages=ps, examined_by={})
    assert any("one document holds" in r for r in pkt.audit().reasons)


def test_a_sufficient_packet_is_not_flagged():
    """The audit must be able to say yes, or it is a warning nobody reads."""
    ps = [Passage(f"d{i}", f"d{i}#p0", 0, "", 2020,
                  "prism spore clusters undergo silvering after moonlight")
          for i in range(10)]
    pkt = _packet(question="do prism spore clusters undergo silvering after moonlight?",
                  passages=ps, examined_by={})
    audit = pkt.audit()
    assert not audit.thin, audit.reasons
    assert audit.term_coverage == 1.0


def test_question_terms_are_the_terms_the_search_actually_uses():
    """Coverage is measured against the same token set the query is built from.
    Two different tokenisers would make the audit measure a question nobody
    asked."""
    q = "Why do high-risk lanternmoss plots dim?"
    assert question_terms(q) == ["high", "risk", "lanternmoss", "plots",
                                 "dim"]
    assert fts_safe(q) == " OR ".join(f'"{t}"' for t in question_terms(q))


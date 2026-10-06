"""Tests for the disclosure rule: cited documents handed over for checking.

WHY THIS IS ITS OWN FILE. The rule deliberately breaks the panel's oldest
contract -- a seat never saw another seat's evidence -- and it breaks it in the
direction that risks the failure the whole project is built to detect. Showing
every seat the same documents is how a table herds. Everything here is a lock on
the three properties that make it safe to do anyway:

  1. what is handed over is CHECKABLE and not adoptable: it is in the permitted
     set (a seat cannot rebut a paper it may not quote) and out of the evidence
     set (so it cannot be counted as this seat's own reading);
  2. it is excluded from every heterogeneity measure, because a document every
     seat was given is shared BY CONSTRUCTION and counting it would let the
     disclosure rule manufacture the collapse the guardrail looks for;
  3. a verdict is only recorded when the seat was actually handed the document,
     and an adverse verdict must quote the passage it convicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.panel import (  # noqa: E402
    _disclosure_set, _ledger_to_observations, _parse_checks,
)


def _ledger():
    return [
        {"seat": "a", "text": "claim A", "grounded": True,
         "citations": [{"doc_id": "d1", "quote": "a real quote from d1",
                        "status": "verified"},
                       {"doc_id": "d9", "quote": "never matched",
                        "status": "quote_not_found"}]},
        {"seat": "b", "text": "claim B", "grounded": True,
         "citations": [{"doc_id": "d1", "quote": "another quote from d1",
                        "status": "verified"},
                       {"doc_id": "d2", "quote": "a quote from d2",
                        "status": "verified"}]},
    ]


def test_only_verified_citations_are_placed_under_examination():
    """There is nothing to examine in a citation that failed the exact-match
    check: no passage to hand over, and the machine has already answered the
    only question that can be answered mechanically."""
    d = _disclosure_set(_ledger())
    assert set(d) == {"d1", "d2"}
    assert "d9" not in d


def test_the_disclosure_set_carries_who_cited_it_and_what_they_quoted():
    """A document arriving without the seat that cited it is an anonymous
    document, and the check being asked for is a check on a USE."""
    d = _disclosure_set(_ledger())
    assert d["d1"]["seats"] == ["a", "b"]
    assert d["d1"]["quotes"] == ["a real quote from d1", "another quote from d1"]
    assert d["d2"]["seats"] == ["b"]


def test_citations_to_examined_documents_are_dropped_from_the_evidence_axis():
    """THE DEFECT THIS PREVENTS, and it is in the safe-looking direction.

    Every seat is handed the same cross-examined documents. If a citation to one
    counted as evidence, two seats quoting a document they were both GIVEN would
    read as convergence, the evidence axis would fall, and the guardrail would
    report a collapse that the disclosure rule itself manufactured.
    """
    ledger = _ledger()
    loose = {t.seat: t.cited_docs for t in _ledger_to_observations(ledger)}
    assert loose["a"] == {"d1"} and loose["b"] == {"d1", "d2"}

    # d1 was under examination for both seats this turn.
    tight = {t.seat: t.cited_docs for t in _ledger_to_observations(
        ledger, None, {"a": {"d1"}, "b": {"d1"}})}
    assert tight["a"] == set()
    assert tight["b"] == {"d2"}


def test_the_claim_itself_is_never_dropped_only_the_axis_contribution():
    """Excluded from the STATISTIC, not from the record. The citation is still
    verified, rendered and attributable; a reader must be able to see what the
    numbers left out."""
    obs = _ledger_to_observations(_ledger(), None, {"a": {"d1"}, "b": {"d1"}})
    texts = {t.seat: t.claim_texts for t in obs}
    assert texts["a"] == ["claim A"] and texts["b"] == ["claim B"]


def test_an_unrecognised_verdict_is_recorded_verbatim_not_rounded():
    """Coercing a malformed verdict to the nearest valid one would hide a broken
    contract behind a plausible value."""
    checks = _parse_checks({"citation_checks": [
        {"doc_id": "d1", "cited_by": "a", "verdict": "probably fine"}]}, "b")
    assert checks[0]["verdict"] == "unrecognised"
    assert checks[0]["raw_verdict"] == "probably fine"


def test_checks_without_a_doc_id_are_dropped_rather_than_guessed():
    checks = _parse_checks({"citation_checks": [
        {"cited_by": "a", "verdict": "misread"},
        {"doc_id": "d2", "verdict": "supports"}]}, "b")
    assert [c["doc_id"] for c in checks] == ["d2"]


def test_a_response_with_no_checks_yields_nothing_rather_than_failing():
    assert _parse_checks({"claims": []}, "b") == []
    assert _parse_checks("not a dict", "b") == []

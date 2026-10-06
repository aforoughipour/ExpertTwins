"""Tests for the verifier and the own/read tier.

Regression docstrings state the failure mode directly so their motivation stays
attached to the assertion.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expertwins.verify import (  # noqa: E402
    MIN_QUOTE_CHARS, Citation, Claim, DictSource, Report, Status, Tier,
    Verifier, canonical_fold, repair_prompt,
)

DOCS = {
    "a2020journal": [("a2020journal#p1",
                      "The receptor was expressed in all differentiation states except the "
                      "quiescent state, which the authors regard as decisive.")],
    "b2021journal": [("b2021journal#p1",
                      "Interstitial fluid pressure rose to 30 mmHg and convective "
                      "transport ceased across the measured region.")],
    "c2022journal": [("c2022journal#p1",
                      "The classifier reached an AUC of 0.91 on the held-out "
                      "cohort assembled from three independent centres.")],
}


def V(permitted=None, own=None):
    return Verifier(DictSource(DOCS), permitted=permitted, own=own)


def test_exact_quote_verifies():
    c = Citation("a2020journal", "expressed in all differentiation states except the quiescent state")
    assert V({"a2020journal"}).verify(c).status is Status.VERIFIED


def test_near_miss_quote_is_rejected():
    """Regression: a near-miss quote can invert the source meaning.

    The model wrote that the gene 'remains unexplored' in quiescent states; the paper
    said it was expressed in all states EXCEPT the quiescent state. Those strings share
    roughly 95% of their characters and mean opposite things. Every fuzzy matcher
    accepts the pair. This is why matching is character-exact and always will be.
    """
    c = Citation("a2020journal",
                 "the receptor in quiescent states remains unexplored, which the authors")
    assert V({"a2020journal"}).verify(c).status is Status.QUOTE_NOT_FOUND


def test_short_quote_is_a_distinct_status():
    c = Citation("a2020journal", "The receptor was")
    out = V({"a2020journal"}).verify(c)
    assert out.status is Status.QUOTE_TOO_SHORT
    assert str(MIN_QUOTE_CHARS) in out.detail


def test_out_of_scope_is_not_fabrication():
    """The provenance-laundering kill.

    When agents read each other, a seat can cite a document it was never shown
    by copying an id from a peer's claim. The quote can be real while the
    attribution is impossible. Membership is checked BEFORE existence for
    exactly this reason.
    """
    c = Citation("b2021journal", "Interstitial fluid pressure rose to 30 mmHg and convective")
    out = V({"a2020journal"}).verify(c)
    assert out.status is Status.OUT_OF_SCOPE
    assert Report(claims=[Claim("x", citations=[out])]).fabrication_rate == 0.0
    assert Report(claims=[Claim("x", citations=[out])]).out_of_scope_count == 1


def test_invented_citation_is_counted_separately():
    """A metric that can read 0% while an invention passes through it needs a
    second column."""
    c = Citation("vandeputte2026biomedicines", "a plausible sentence of sufficient length to pass the floor")
    out = V({"a2020journal"}).verify(c)
    assert out.status is Status.OUT_OF_SCOPE_INVENTED
    assert Report(claims=[Claim("x", citations=[out])]).invented_count == 1


def test_empty_permitted_set_is_not_none():
    """An empty set means 'may cite nothing'. None means 'no restriction'.
    Conflating them is how a silent-abstention bug once read as an honest
    answer."""
    c = Citation("a2020journal", "expressed in all differentiation states except the quiescent state")
    assert V(set()).verify(c).status is Status.OUT_OF_SCOPE
    assert V(None).verify(c).status is Status.VERIFIED


def test_typographic_folding_but_not_semantic_folding():
    assert canonical_fold("ﬁbre—test") == "fibre-test"
    assert canonical_fold("lantern-\nmoss") == "lanternmoss"
    # Negation is meaning, never typography.
    assert canonical_fold("not expressed") != canonical_fold("expressed")


def test_tier_own_vs_read():
    """The tier is what makes 'would this person say that?' mechanical."""
    q = "Interstitial fluid pressure rose to 30 mmHg and convective transport"
    v = V({"b2021journal", "c2022journal"}, own={"b2021journal"})
    assert v.verify(Citation("b2021journal", q)).tier is Tier.OWN
    q2 = "The classifier reached an AUC of 0.91 on the held-out cohort assembled"
    assert v.verify(Citation("c2022journal", q2)).tier is Tier.READ


def test_unverified_citation_has_no_tier():
    v = V({"b2021journal"}, own={"b2021journal"})
    c = v.verify(Citation("b2021journal", "a quote that is not in this document at all"))
    assert c.status is Status.QUOTE_NOT_FOUND
    assert c.tier is Tier.UNKNOWN


def test_self_anchor_rate_counts_grounded_claims_only():
    q_own = "Interstitial fluid pressure rose to 30 mmHg and convective transport"
    q_read = "The classifier reached an AUC of 0.91 on the held-out cohort assembled"
    v = V({"b2021journal", "c2022journal"}, own={"b2021journal"})
    own_claim = Claim("mine", citations=[Citation("b2021journal", q_own)])
    read_claim = Claim("theirs", citations=[Citation("c2022journal", q_read)])
    bad = Claim("bad", citations=[Citation("c2022journal", "not present in the text anywhere here")])
    for c in (own_claim, read_claim, bad):
        v.verify_claim(c)
    r = Report(claims=[own_claim, read_claim, bad])
    assert r.n_grounded == 2
    assert abs(r.self_anchor_rate - 0.5) < 1e-9


def test_grounded_requires_all_citations_not_any():
    """A claim resting on one real and one fabricated citation is not
    half-trustworthy; it is a claim whose author did not check."""
    good = Citation("a2020journal", "expressed in all differentiation states except the quiescent state")
    bad = Citation("a2020journal", "a sentence that does not occur in this document at all")
    c = Claim("mixed", citations=[good, bad])
    V({"a2020journal"}).verify_claim(c)
    assert not c.grounded


def test_claim_with_no_citations_is_ungrounded():
    assert not Claim("bare assertion").grounded


def test_repair_prompt_shows_no_new_evidence():
    bad = Citation("a2020journal", "a sentence that does not occur in this document at all")
    c = Claim("mixed", citations=[bad])
    V({"a2020journal"}).verify_claim(c)
    text = repair_prompt(c)
    assert "WITHDRAW" in text
    # It must not leak the passage the agent failed to quote.
    assert "except the quiescent state" not in text

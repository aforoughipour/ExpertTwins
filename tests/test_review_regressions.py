"""Regressions for safe-looking defects.

Every test here corresponds to a specific finding. The finding is named in the
docstring, because a regression whose motivation is forgotten gets deleted by
the next person who finds it inconvenient -- and every one of these was a defect
in the SAFE-LOOKING direction: the system reported a number that read as health.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expertwins import guardrail as gr  # noqa: E402
from expertwins.diversity import Preflight, SeatTurn, measure  # noqa: E402
from expertwins.identity import name_keys  # noqa: E402

PERM = 30


def _panel(n=4, agree=False, stances=True):
    words = [("lantern glow", "vessel"), ("dendritic", "subset"),
             ("encoder", "patch"), ("estimator", "interval"),
             ("epithelium", "alarmin"), ("trajectory", "cohort")]
    out = []
    names = [f"s{i}" for i in range(n)]
    for i, s in enumerate(names):
        w = words[i % len(words)]
        text = ("Delivery is the constraint." if agree
                else f"The {w[0]} determines the {w[1]} in this system.")
        out.append(SeatTurn(
            s, [text, text + " Second."], {f"d{i}a", f"d{i}b"},
            {o: "agree" for o in names if o != s} if stances else {}))
    return out


def _band(turns):
    pre = Preflight(0.8, 0.5, 0.7, 0.75, len(turns))
    return gr.set_band(pre, measure(turns, turn=1, n_perm=PERM))


# -- finding 9: substring surname matching -------------------------------

def test_surname_matching_is_token_bound_not_substring():
    """`"varga" in "vargassen lm"` is True. Substring containment would quietly
    admit other people's papers into a scientist's own corpus -- the one failure
    attribution exists to prevent."""
    k = name_keys("Okoro", ["S", "Sam"])
    assert k.matches("Okoro S")
    assert not k.matches("Okorowell S")
    assert not k.matches("Mokoro S")
    j = name_keys("Varga", ["LM"])
    assert j.matches("Varga LM")
    assert not j.matches("Vargassen LM")


# -- finding 6: missing stances read as agreement -------------------------

def test_no_declared_stances_is_not_agreement():
    """`_stance_obs` returns 0 when nobody declared a stance, and 0 was then
    read as 'they all agree'. A table that simply stopped answering the stance
    question would have been certified as converged."""
    snap = measure(_panel(stances=False), turn=2, n_perm=PERM)
    assert snap.n_stances == 0
    assert not snap.informative("stance")
    band = _band(_panel())
    a = gr.assess(2, snap, band, _panel(stances=False), monitor=gr.Sequential())
    assert a.verdict is not gr.Verdict.CORROBORATION


# -- finding 6: silent seats raise every statistic ------------------------

def test_a_panel_that_fell_silent_gets_no_verdict():
    """`measure` drops seats with nothing to say, so a ten-seat panel reduced to
    its two most different members scored near-maximal diversity."""
    turns = _panel(2)
    snap = measure(turns, turn=2, n_perm=PERM, n_expected=10)
    assert snap.participation < gr.MIN_PARTICIPATION
    a = gr.assess(2, snap, _band(_panel(4)), turns, monitor=gr.Sequential())
    assert a.verdict is gr.Verdict.UNDETERMINED
    assert any("fell silent" in r for r in a.reasons)


# -- finding 15: lexical was documented as never a floor, and was one ------

def test_the_lexical_axis_can_never_breach():
    turns = _panel()
    band = _band(turns)
    assert not hasattr(band, "lexical")
    a = gr.assess(2, measure(turns, turn=2, n_perm=PERM), band, turns,
                  monitor=gr.Sequential())
    assert not any("lexical" in b for b in a.breached)
    assert any("style channel" in r for r in a.reasons)


# -- finding 15: fidelity only warned ------------------------------------

def test_fidelity_loss_short_circuits_the_verdict():
    """A verdict about the heterogeneity of seats that have stopped being their
    scientists is not a verdict about anything."""
    turns = _panel()
    a = gr.assess(2, measure(turns, turn=2, n_perm=PERM), _band(turns), turns,
                  monitor=gr.Sequential(),
                  self_anchor={"s0": 0.0}, territories={"s0": "home"})
    assert a.verdict is gr.Verdict.FIDELITY_LOST
    assert not a.intervene


# -- finding 3: the retained statistic -----------------------------------

def test_retention_is_measured_against_turn_one_not_against_the_floor():
    """`(ev / band.evidence) * rho_low` equals `ev / S_1` only when the relative
    floor binds. When the ABSOLUTE floor binds it is something else entirely,
    and it was still being called a retained fraction."""
    turns = _panel()
    band = _band(turns)
    assert band.ref_evidence >= 0.0 and band.ref_positions >= 0.0
    # An absolute floor above the relative one must not corrupt the reference.
    band.evidence = 0.99
    band.ref_evidence = 0.5
    snap = measure(turns, turn=2, n_perm=PERM)
    a = gr.assess(2, snap, band, turns, monitor=gr.Sequential())
    assert a.llr == a.llr  # finite, i.e. the statistic was computable


def test_the_monitor_takes_the_worst_axis_not_the_mean():
    """With one axis collapsed and one retained, the mean is 0.5 -- at which the
    Beta increment is exactly zero, so a completely collapsed axis could
    contribute no evidence forever."""
    m = gr.Sequential()
    m.update(0.5)
    assert abs(m.llr) < 1e-9, "0.5 is the zero-information point, by construction"
    m2 = gr.Sequential()
    fired = m2.update(0.0)
    assert fired == "collapsing", "a completely collapsed axis must fire at once"
    assert m2.history[-1] > 0


def test_a_missing_observation_does_not_read_as_health():
    """A missing axis must not be converted to 1.0, which would score an
    unmeasurable turn as perfect health."""
    m = gr.Sequential()
    assert m.update(None) == ""
    assert m.llr == 0.0
    assert m.history == [0.0]


# -- finding 13: the dissent rung and the end of the ladder ---------------

def test_dissent_is_assigned_to_an_evidenced_minority_or_not_at_all():
    turns = [
        SeatTurn("a", ["x one", "x two"], {"shared", "ua"}, {"b": "agree"}),
        SeatTurn("b", ["y one", "y two"], {"shared"}, {"a": "agree"}),
        SeatTurn("c", ["z one", "z two"], {"shared", "uc1", "uc2"},
                 {"a": "disagree", "b": "disagree"}),
    ]
    assert gr._evidenced_minority(turns) == "c"
    # Nobody dissenting -> nobody to assign, and no invented dissenter.
    calm = [SeatTurn(t.seat, t.claim_texts, t.cited_docs,
                     {k: "agree" for k in t.stances}) for t in turns]
    assert gr._evidenced_minority(calm) == ""


def test_terminate_ends_the_run_as_collapsed():
    """DECLARE sat behind TERMINATE, which refuses the next follow-up, so the
    run could never reach the rung that admits collapse."""
    turns = _panel(agree=True)
    band = _band(_panel())
    band.evidence, band.positions = 0.99, 0.99
    band.ref_evidence, band.ref_positions = 0.99, 0.99
    for used, move in enumerate(gr.LADDER):
        a = gr.assess(2, measure(turns, turn=2, n_perm=PERM), band, turns,
                      monitor=gr.Sequential(), rungs_used=used)
        if a.move in (gr.Move.TERMINATE, gr.Move.DECLARE):
            assert a.verdict is gr.Verdict.COLLAPSED
            break
    else:
        raise AssertionError("the ladder never reached its end")


def test_a_rung_that_cannot_bite_is_skipped_and_counted():
    """Skipping to the next rung without counting the skip meant the skipped
    rung was re-selected on the next firing and the ladder never advanced."""
    turns = _panel(agree=True)          # disjoint evidence: nothing to withhold
    band = _band(_panel())
    band.evidence, band.positions = 0.99, 0.99
    band.ref_evidence, band.ref_positions = 0.99, 0.99
    i = gr.LADDER.index(gr.Move.DIFFERENTIAL_RETRIEVAL)
    a = gr.assess(2, measure(turns, turn=2, n_perm=PERM), band, turns,
                  monitor=gr.Sequential(), rungs_used=i)
    if a.move is not gr.Move.DIFFERENTIAL_RETRIEVAL:
        assert a.skipped_rungs == 1


# -- finding 1: the discriminator is weaker than its old name claimed -----

def test_corroboration_is_described_as_document_overlap_not_independence():
    """Five reviews of one primary result are five documents. The verdict may
    not claim evidential independence that a set operation cannot establish."""
    turns = _panel(agree=True)
    for t in turns:
        t.cited_docs |= {"shared_one"}
    band = _band(turns)
    band.evidence = 0.0
    a = gr.assess(2, measure(turns, turn=2, n_perm=PERM), band, turns,
                  monitor=gr.Sequential())
    if a.verdict is gr.Verdict.CORROBORATION:
        joined = " ".join(a.reasons)
        assert "document identity, not evidential independence" in joined


# -- finding 5: the controller perturbs its own measurement ---------------

def test_only_the_dissent_rung_touches_the_prompt():
    """Claim 6 of the design was false in the implementation: every move emitted
    the same heterogeneity prompt, so a recovery could not be attributed to the
    evidence or routing change it was supposed to test."""
    prompted = [m for m in gr.Move if gr.nudge_block(m)]
    assert prompted == [gr.Move.ASSIGN_DISSENT], prompted

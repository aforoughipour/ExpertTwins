"""Tests for the heterogeneity instruments and the guardrail.

These matter more than they look. The guardrail is the component most likely to
be WRONG IN THE SAFE-LOOKING DIRECTION: a guardrail that never fires produces a
clean-reading transcript from a collapsed panel, and a guardrail that always
fires manufactures the dissent it claims to have preserved. So both directions
are tested, and so is the case where the instrument cannot tell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expertwins import guardrail as gr  # noqa: E402
from expertwins.diversity import (  # noqa: E402
    Preflight, SeatTurn, contestedness, js_divergence, jaccard_distance,
    measure, modal_disagreement, preflight, stance_entropy, tfidf_matrix,
    vendi_order2, vendi_truncated,
)

PERM = 40   # keep the tests fast; the pipeline default is 200


def _distinct_panel() -> list[SeatTurn]:
    return [
        SeatTurn("varga", ["Terrace pressure collapses convection in the Peloria hollow.",
                          "Vessel compression precedes hypoxia in this model."],
                 {"d1", "d2"}, {"marchetti": "disagree", "okoro": "out_of_scope"}),
        SeatTurn("marchetti", ["Lanternmoss subsets differ between human blood and tissue.",
                             "Echo pollen uptake requires the prism spore compartment here."],
                 {"d3", "d4"}, {"varga": "disagree", "okoro": "needs_other_evidence"}),
        SeatTurn("okoro", ["Site signature explains most of the reported slide-level gain.",
                          "The encoder rather than the aggregator carries the improvement."],
                 {"d5", "d6"}, {"varga": "out_of_scope", "marchetti": "disagree"}),
        SeatTurn("ndubisi", ["Reimplementation under one protocol removes the reported delta.",
                           "Stain shift degrades the model before any clinic sees it."],
                 {"d7", "d8"}, {"okoro": "agree", "varga": "out_of_scope"}),
    ]


def _collapsed_panel() -> list[SeatTurn]:
    line = "Delivery to the Peloria plot is the dominant constraint here."
    return [SeatTurn(s, [line, line], {"d1", "d2"},
                     {o: "agree" for o in ("varga", "marchetti", "okoro", "ndubisi") if o != s})
            for s in ("varga", "marchetti", "okoro", "ndubisi")]


# -- primitives -----------------------------------------------------------

def test_jaccard_distance_bounds():
    assert jaccard_distance(set(), set()) == 0.0
    assert jaccard_distance({"a"}, {"a"}) == 0.0
    assert jaccard_distance({"a"}, {"b"}) == 1.0


def test_js_divergence_is_symmetric_and_bounded():
    from collections import Counter
    p, q = Counter("aaabbb"), Counter("bbbccc")
    assert abs(js_divergence(p, q) - js_divergence(q, p)) < 1e-12
    assert 0.0 <= js_divergence(p, q) <= 1.0
    assert js_divergence(p, p) < 1e-12


def test_vendi_order2_counts_effective_positions():
    """Four orthogonal items -> four positions. Four identical items -> one.

    The order-2 (RKE) variant is used rather than the plain Vendi Score because
    the plain one does not reliably converge below ~20,000 samples
    (arXiv:2410.21719) and this panel has ten to fifteen seats.
    """
    orth = np.eye(4)
    assert abs(vendi_order2(orth) - 4.0) < 1e-6
    same = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (4, 1))
    assert abs(vendi_order2(same) - 1.0) < 1e-6


def test_vendi_order2_sees_two_camps_where_mean_distance_would_not():
    """Two tight camps must read ~2, not 'somewhat diverse'. This is the state a
    collapsing table passes through, and mean pairwise distance cannot see it."""
    camp = np.array([[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0], [0.0447, 0.999]])
    camp = camp / np.linalg.norm(camp, axis=1, keepdims=True)
    assert 1.8 < vendi_order2(camp) < 2.3


def test_vendi_truncated_is_stable():
    orth = np.eye(6)
    assert 3.5 < vendi_truncated(orth, t=4) <= 4.0


def test_stance_entropy_and_modal_disagreement():
    from collections import Counter
    assert stance_entropy(Counter({"agree": 8})) < 0.2
    assert modal_disagreement(Counter({"agree": 8})) == 0.0
    mixed = Counter({"agree": 2, "disagree": 2, "out_of_scope": 2, "needs_other_evidence": 2})
    assert stance_entropy(mixed) > 0.9
    assert abs(modal_disagreement(mixed) - 0.75) < 1e-9


def test_contestedness_reads_the_literature_not_the_agents():
    calm = "The result was confirmed. The measurement agrees with the model."
    argued = ("However, these findings are controversial and conflicting reports "
              "remain unclear; in contrast, other groups could not be replicated.")
    assert contestedness([argued]) > contestedness([calm])


def test_tfidf_rows_are_unit_norm():
    m = tfidf_matrix(["alpha beta gamma", "gamma delta epsilon"])
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)


# -- the measurement ------------------------------------------------------

def test_distinct_panel_beats_collapsed_panel_on_every_live_axis():
    d = measure(_distinct_panel(), turn=1, n_perm=PERM)
    c = measure(_collapsed_panel(), turn=1, n_perm=PERM)
    assert d.vendi2 > c.vendi2
    assert d.effective_positions > c.effective_positions
    assert d.s("stance") > c.s("stance")
    assert d.stance_entropy > c.stance_entropy


def test_collapsed_panel_has_one_effective_position():
    c = measure(_collapsed_panel(), turn=1, n_perm=PERM)
    assert c.vendi2 < 1.5
    assert c.s("stance") == 0.0


def test_saturated_null_is_marked_uninformative_not_zero():
    """THE BUG THIS TEST EXISTS FOR.

    When seats hold disjoint corpora -- which is the DESIGN -- re-dealing their
    citations still yields disjoint sets, so the Jaccard evidence statistic pins
    at 1.0 under both the observation and the null. The chance-corrected
    separation is then 0. Reading that as collapse would flag the healthiest
    possible configuration as the failure this check is built to catch.
    """
    snap = measure(_distinct_panel(), turn=1, n_perm=PERM)
    ax = snap.axes["evidence"]
    assert ax.observed > 0.95 and ax.null > 0.95
    assert ax.separation == 0.0
    assert not ax.informative


def test_shared_evidence_makes_the_evidence_axis_informative():
    turns = _distinct_panel()
    for t in turns:                       # everyone now stands on one document
        t.cited_docs = {"shared", *t.cited_docs}
    snap = measure(turns, turn=1, n_perm=PERM)
    assert "shared" in {d for t in turns for d in t.cited_docs}
    assert snap.axes["evidence"].observed < 1.0


# -- the band and the ladder ---------------------------------------------

def _band(turns, pre_kwargs=None):
    pre = Preflight(packet_separation=0.8, rubric_separation=0.5,
                    contestedness=0.7, retention=0.75, n_seats=4,
                    **(pre_kwargs or {}))
    return pre, gr.set_band(pre, measure(turns, turn=1, n_perm=PERM))


def test_reference_turn_is_not_judged_against_the_band_it_defines():
    turns = _distinct_panel()
    _, band = _band(turns)
    a = gr.assess(1, measure(turns, turn=1, n_perm=PERM), band, turns)
    assert a.verdict is gr.Verdict.REFERENCE
    assert a.move is gr.Move.NONE


def test_degenerate_reference_panel_is_flagged_rather_than_certified():
    turns = _collapsed_panel()
    _, band = _band(turns)
    a = gr.assess(1, measure(turns, turn=1, n_perm=PERM), band, turns)
    assert a.verdict is gr.Verdict.REFERENCE
    assert a.breached, "a panel degenerate before anyone spoke must not pass silently"
    assert any("degenerate" in r for r in a.reasons)


def test_collapse_after_the_reference_turn_fires_the_ladder():
    ref = _distinct_panel()
    _, band = _band(ref)
    turns = _collapsed_panel()
    snap = measure(turns, turn=2, n_perm=PERM)
    mon = gr.Sequential()
    a = gr.assess(2, snap, band, turns, monitor=mon, rungs_used=0)
    assert a.verdict in (gr.Verdict.COLLAPSING, gr.Verdict.COLLAPSED)
    assert a.move is gr.LADDER[0]
    assert a.intervene


def test_the_ladder_is_climbed_not_repeated():
    ref = _distinct_panel()
    _, band = _band(ref)
    turns = _collapsed_panel()
    snap = measure(turns, turn=2, n_perm=PERM)
    seen = []
    for used in range(len(gr.LADDER)):
        a = gr.assess(2, snap, band, turns, monitor=gr.Sequential(), rungs_used=used)
        seen.append(a.move)
    assert len(set(seen)) > 1
    assert seen[-1] is gr.Move.DECLARE
    a = gr.assess(2, snap, band, turns, monitor=gr.Sequential(),
                  rungs_used=len(gr.LADDER) - 1)
    assert a.verdict is gr.Verdict.COLLAPSED


def test_independent_corroboration_is_not_treated_as_collapse():
    """Agreement on DIFFERENT evidence is the best thing a panel can produce.
    Intervening here would manufacture dissent, which is the worse failure."""
    turns = [
        SeatTurn("a", ["The effect is delivery-limited in this system.",
                       "Pressure gradients, not affinity, set the exposure."],
                 {"d1", "d2"}, {"b": "agree", "c": "agree"}),
        SeatTurn("b", ["Perfusion measurements independently support that reading.",
                       "Oxygenation recovers on the same schedule in our hands."],
                 {"d3", "d4"}, {"a": "agree", "c": "agree"}),
        SeatTurn("c", ["Our imaging cohort reaches the same conclusion by another route.",
                       "Segmented vessel density tracks the same endpoint."],
                 {"d5", "d6"}, {"a": "agree", "b": "agree"}),
    ]
    for t in turns:                       # make the evidence axis informative
        t.cited_docs |= {"shared_one"}
    pre = Preflight(0.8, 0.5, 0.7, 0.75, 3)
    band = gr.set_band(pre, measure(turns, turn=1, n_perm=PERM))
    band.evidence = 0.0                   # evidence separation is above the floor
    a = gr.assess(2, measure(turns, turn=2, n_perm=PERM), band, turns,
                  monitor=gr.Sequential())
    assert a.verdict is gr.Verdict.CORROBORATION
    assert not a.intervene


def test_sequential_monitor_does_not_fire_on_one_noisy_turn():
    """A guardrail that fires on a single turn at this panel size becomes the
    confound it was supposed to detect."""
    m = gr.Sequential()
    assert m.update(1.0) == "healthy"
    m2 = gr.Sequential()
    assert m2.update(0.5) == ""


def test_sequential_monitor_fires_on_sustained_loss():
    m = gr.Sequential()
    fired = [m.update(0.05) for _ in range(4)]
    assert "collapsing" in fired


def test_fidelity_loss_is_reported_even_when_diversity_is_fine():
    turns = _distinct_panel()
    _, band = _band(turns)
    a = gr.assess(2, measure(turns, turn=2, n_perm=PERM), band, turns,
                  monitor=gr.Sequential(),
                  self_anchor={"varga": 0.0}, territories={"varga": "home"})
    assert any("FIDELITY" in r for r in a.reasons)


def test_nudge_never_instructs_the_seat_to_disagree():
    for move in gr.Move:
        text = gr.nudge_block(move)
        if not text:
            continue
        assert "NOT an instruction to disagree" in text
        assert "Manufactured disagreement is worse" in text


def test_preflight_notes_an_undifferentiated_panel():
    packets = {"a": {"d1", "d2"}, "b": {"d1", "d2"}, "c": {"d1", "d2"}}
    pre = preflight(packets, {"a": "", "b": "", "c": ""}, ["a calm settled literature"])
    assert pre.packet_separation == 0.0
    assert any("SAME evidence" in n for n in pre.notes)
    assert any("no seat declared refusals" in n for n in pre.notes)


def test_retention_rises_with_distinctness_and_contestedness():
    calm = preflight({"a": {"1"}, "b": {"1"}}, {"a": "x", "b": "y"},
                     ["the result was confirmed and agrees with the model"])
    hot = preflight({"a": {"1"}, "b": {"2"}}, {"a": "x", "b": "y"},
                    ["however this is controversial and conflicting and unresolved"] * 3)
    assert hot.retention > calm.retention

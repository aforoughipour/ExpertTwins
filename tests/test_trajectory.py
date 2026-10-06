"""The long-run instrument: what it measures, and what it refuses to say.

`ops/trajectory.py` answers a different question from `render`: not what the
panel concluded, but whether the panel is still working at turn 12. It reads
only frozen artifacts, so it cannot be talked into a favourable reading.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

import trajectory as tj  # noqa: E402


def _run(tmp_path: Path, turns: dict[int, dict]) -> Path:
    """turns: n -> {"ledger": [...], "report": {...} or None}"""
    run = tmp_path / "run"
    (run / "turns").mkdir(parents=True)
    man_turns = {}
    for n, payload in turns.items():
        d = run / "turns" / str(n)
        d.mkdir(parents=True, exist_ok=True)
        if payload.get("ledger") is not None:
            (d / "ledger.json").write_text(json.dumps(payload["ledger"]),
                                           encoding="utf-8")
        if payload.get("report") is not None:
            (d / "report.json").write_text(json.dumps(payload["report"]),
                                           encoding="utf-8")
        man_turns[str(n)] = {"question": "q"}
    (run / "manifest.json").write_text(json.dumps({"turns": man_turns}),
                                       encoding="utf-8")
    return run


def _claim(seat: str, text: str, docs: list[str], grounded: bool = True) -> dict:
    return {"seat": seat, "text": text, "grounded": grounded,
            "citations": [{"doc_id": d, "quote": "q", "status": "verified"}
                          for d in docs]}


def test_a_document_cited_again_is_not_a_new_document(tmp_path):
    """F2, evidence exhaustion. A panel arguing from a closed set has stopped
    consulting the literature, and the count of citations cannot see that --
    only the count of citations to documents not already on the table can."""
    run = _run(tmp_path, {
        1: {"ledger": [_claim("a", "one", ["d1", "d2"])]},
        2: {"ledger": [_claim("a", "two", ["d2", "d3"])]},
    })
    rows = tj.trajectory(run)["turns"]
    assert rows[0]["new_docs"] == 2 and rows[0]["new_doc_fraction"] == 1.0
    assert rows[1]["docs_cited"] == 2
    assert rows[1]["new_docs"] == 1
    assert rows[1]["new_doc_fraction"] == 0.5


def test_evidence_exhaustion_fires_when_nothing_new_is_cited(tmp_path):
    run = _run(tmp_path, {
        1: {"ledger": [_claim("a", "one", ["d1", "d2"])]},
        2: {"ledger": [_claim("a", "two", ["d1", "d2"])]},
    })
    out = tj.render(tj.trajectory(run))
    assert "F2 evidence exhaustion: FIRING" in out


def test_a_turn_that_was_never_ingested_is_a_hole_not_a_zero(tmp_path):
    """The distinction this whole project is built on. A turn with no ledger
    has no measurements; reporting zeros for it would put a harness failure on
    the same axis as a scientific result."""
    run = _run(tmp_path, {1: {"ledger": [_claim("a", "one", ["d1"])]},
                          2: {"ledger": None}})
    rows = tj.trajectory(run)["turns"]
    assert rows[1]["status"] == "not ingested"
    assert "n_claims" not in rows[1]
    assert "NOT INGESTED" in tj.render(tj.trajectory(run))


def test_an_uninformative_axis_is_not_read_as_collapse(tmp_path):
    """A permutation null that saturated cannot discriminate. Seats holding
    disjoint corpora saturate it for a legitimate reason, and reading the
    resulting 0.00 as convergence would flag the healthiest possible
    configuration as the failure."""
    rep = {"diversity": {"axes": {"positions": {"separation": 0.0,
                                                "informative": False}}},
           "drift": {}, "guardrail": {}}
    run = _run(tmp_path, {
        1: {"ledger": [_claim("a", "one", ["d1"])], "report": rep},
        2: {"ledger": [_claim("a", "two", ["d2"])], "report": rep},
    })
    rows = tj.trajectory(run)["turns"]
    assert rows[1]["positions_separation"] is None
    assert "F1 convergence collapse: not firing" in tj.render(tj.trajectory(run))


def test_grounding_decay_is_measured_on_every_turn_including_the_first(tmp_path):
    """Unlike the others, F4 does not need a previous turn to compare against:
    a seat reaching past its evidence on turn 1 is already the failure."""
    run = _run(tmp_path, {
        1: {"ledger": [_claim("a", "one", ["d1"]),
                       _claim("a", "two", [], grounded=False)]},
    })
    rows = tj.trajectory(run)["turns"]
    assert rows[0]["grounded_fraction"] == 0.5
    assert "F4 grounding decay: FIRING" in tj.render(tj.trajectory(run))


def test_a_seat_repeating_itself_is_visible_even_when_the_table_moves_on(tmp_path):
    """The sharpest of the four, and the one `drift.novelty` cannot give alone:
    novelty is measured against everything ANYONE said, so a seat restating
    itself while the table moves on is invisible in it. A panel in which every
    seat repeats itself while disagreeing with the others scores well on
    separation, well on lexical diversity, and is not a conversation."""
    text = "prism spores accumulation is driven by chronic granulocyte mobilisation"
    run = _run(tmp_path, {
        1: {"ledger": [_claim("a", text, ["d1"])]},
        2: {"ledger": [_claim("a", text, ["d2"])]},
    })
    rows = tj.trajectory(run)["turns"]
    assert rows[0]["self_restatement"] is None       # nothing earlier to repeat
    assert rows[1]["self_restatement"] == 1.0


def test_a_short_run_is_told_it_is_not_a_trajectory(tmp_path):
    """Two points are a difference, not a trend, and the output must say so
    rather than leave a reader to notice."""
    run = _run(tmp_path, {1: {"ledger": [_claim("a", "one", ["d1"])]}})
    assert "needs a trajectory" in tj.render(tj.trajectory(run))


def test_a_run_with_no_turns_is_refused(tmp_path):
    run = tmp_path / "empty"
    (run / "turns").mkdir(parents=True)
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        tj.trajectory(run)

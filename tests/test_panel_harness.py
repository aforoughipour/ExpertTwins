"""Harness bugs that delete work while reporting success.

They share a shape: nothing crashes, nothing is logged, and the loss is
indistinguishable downstream from a substantive result -- a seat that grounded
nothing, a panel with nothing to say, duplicated turn records.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

import panel  # noqa: E402


# --------------------------------------------------------------------------
# the citation list, under either name a model reaches for
# --------------------------------------------------------------------------

def test_a_citation_list_keyed_evidence_is_still_a_citation_list():
    """THE DEFECT: a seat filed nine well-formed claims whose citation list was
    keyed `evidence`. Every doc_id and quote was correct and every one would
    have verified; the parser looked for one key, found nothing, and reported
    *9 claims, 0 grounded*."""
    payload = {"claims": [{"claim": "X follows from Y",
                           "evidence": [{"doc_id": "d1", "quote": "Y holds"}]}]}
    claims, _, _, _, _ = panel._parse(payload, "seat")
    assert len(claims) == 1
    assert [c.doc_id for c in claims[0].citations] == ["d1"]
    assert claims[0].citations[0].quote == "Y holds"


def test_citations_wins_when_both_keys_are_present():
    payload = {"claims": [{"claim": "X",
                           "citations": [{"doc_id": "right", "quote": "q"}],
                           "evidence": [{"doc_id": "wrong", "quote": "q"}]}]}
    claims, _, _, _, _ = panel._parse(payload, "seat")
    assert [c.doc_id for c in claims[0].citations] == ["right"]


def test_a_claim_with_no_citations_is_still_ungrounded_not_invented():
    """Tolerating shape must never become inventing content."""
    claims, _, _, _, _ = panel._parse({"claims": [{"claim": "bare"}]}, "seat")
    assert claims[0].citations == []


def test_a_citation_check_may_also_use_the_evidence_key():
    checks = panel._parse_checks(
        {"citation_checks": [{"doc_id": "d1", "cited_by": "other",
                              "verdict": "misread",
                              "why": "the next clause reverses it",
                              "evidence": [{"doc_id": "d1", "quote": "but not"}]}]},
        "checker")
    assert checks[0]["citations"][0]["quote"] == "but not"


# --------------------------------------------------------------------------
# a byte-order mark is not a malformed document
# --------------------------------------------------------------------------

def test_a_response_written_with_a_bom_is_readable(tmp_path):
    """An agent writing JSON through a Windows editor or a PowerShell redirect
    emits a UTF-8 BOM. Strict utf-8 turns that into `Unexpected UTF-8 BOM`,
    which `ingest` records as UNPARSEABLE -- silently deleting a seat from the
    turn AND from the next turn's prior context, which is built from the ledger
    ingest writes."""
    p = tmp_path / "seat.json"
    p.write_text('{"claims": []}', encoding="utf-8-sig")
    with pytest.raises(json.JSONDecodeError):
        json.loads(p.read_text(encoding="utf-8"))
    assert json.loads(p.read_text(encoding="utf-8-sig")) == {"claims": []}


def test_a_question_file_with_a_bom_is_read(tmp_path):
    p = tmp_path / "q.md"
    p.write_text("Why do the cells relapse?", encoding="utf-8-sig")
    assert panel._read_question(None, str(p)) == "Why do the cells relapse?"


def test_a_question_must_come_from_exactly_one_place(tmp_path):
    p = tmp_path / "q.md"
    p.write_text("a question", encoding="utf-8")
    with pytest.raises(SystemExit):
        panel._read_question("inline", str(p))
    with pytest.raises(SystemExit):
        panel._read_question(None, None)


def test_an_empty_question_file_is_refused_not_silently_accepted(tmp_path):
    """A partially-applied turn is worse than a failed one."""
    p = tmp_path / "q.md"
    p.write_text("   \n", encoding="utf-8")
    with pytest.raises(SystemExit):
        panel._read_question(None, str(p))


# --------------------------------------------------------------------------
# the panel's memory
# --------------------------------------------------------------------------

def _ledger(run: Path, turn: int, claims: list[dict]) -> None:
    d = run / "turns" / str(turn)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ledger.json").write_text(json.dumps(claims), encoding="utf-8")


def test_prior_context_is_cumulative_not_one_turn_deep(tmp_path):
    """THE DEFECT: the header says "your own record, in full" and it was the
    last turn only. A panel that cannot see its own history cannot build on it,
    so the trajectory flattens -- and the flattening reads as the panel running
    out of things to say when it was the harness deleting the conversation."""
    run = tmp_path / "run"
    _ledger(run, 1, [{"seat": "a", "text": "turn one claim", "grounded": True,
                      "citations": []}])
    _ledger(run, 2, [{"seat": "a", "text": "turn two claim", "grounded": True,
                      "citations": []}])
    by_seat = panel._claims_by_seat(run, 2)
    assert [c["text"] for c in by_seat["a"]] == ["turn one claim", "turn two claim"]
    assert [c["turn"] for c in by_seat["a"]] == [1, 2]


def test_a_seat_not_called_on_last_turn_keeps_its_record(tmp_path):
    """The worst case, and how the bug was found: after a turn restricted to a
    subset, the seats left out would next be told they had filed nothing --
    their entire contribution erased because they were once not asked."""
    run = tmp_path / "run"
    _ledger(run, 1, [{"seat": "quiet", "text": "an early claim", "grounded": True,
                      "citations": []}])
    _ledger(run, 2, [{"seat": "loud", "text": "a later claim", "grounded": True,
                      "citations": []}])
    by_seat = panel._claims_by_seat(run, 2)
    assert "quiet" in by_seat
    assert by_seat["quiet"][0]["text"] == "an early claim"


def test_turn_two_is_unchanged_by_the_cumulative_fix(tmp_path):
    """At turn 2 the only prior turn IS the last one, so turn-2 packets are
    byte-comparable across the fix. Only turn 3 onward differ."""
    run = tmp_path / "run"
    _ledger(run, 1, [{"seat": "a", "text": "one", "grounded": True,
                      "citations": []}])
    by_seat = panel._claims_by_seat(run, 1)
    assert [c["text"] for c in by_seat["a"]] == ["one"]


# --------------------------------------------------------------------------
# the per-seat notebook
# --------------------------------------------------------------------------

class _Report:
    n_grounded = 0
    self_anchor_rate = 0.0


def test_reingesting_a_turn_replaces_its_block_rather_than_growing_it(tmp_path):
    """`ingest --reingest` is supported and deterministic, and the append was
    unconditional. Nothing failed, because no code reads this file -- and an
    audit artifact that silently duplicates itself is worse than none, since a
    reader cannot tell duplicate recorded turns from one turn recorded once."""
    p = tmp_path / "seat.md"
    info = {"permitted": ["d1"], "own": [], "own_fraction": 0.0,
            "territory": "adjacent"}
    for _ in range(3):
        panel._append_notebook(p, "seat", 1, info, [], _Report(), [], [])
    text = p.read_text(encoding="utf-8")
    assert text.count("## Turn 1") == 1
    assert text.count("internal record") == 1


def test_a_later_turn_does_not_disturb_an_earlier_one(tmp_path):
    p = tmp_path / "seat.md"
    info = {"permitted": ["d1"], "own": [], "own_fraction": 0.0,
            "territory": "adjacent"}
    panel._append_notebook(p, "seat", 1, info, [], _Report(), [], [])
    panel._append_notebook(p, "seat", 2, info, [], _Report(), [], [])
    panel._append_notebook(p, "seat", 1, info, [], _Report(), [], [])
    text = p.read_text(encoding="utf-8")
    assert text.count("## Turn 1") == 1
    assert text.count("## Turn 2") == 1


# --------------------------------------------------------------------------
# the standalone citation audit
# --------------------------------------------------------------------------

class _Source:
    def __init__(self, passages):
        self._p = passages

    def passages(self, doc_id):
        return self._p.get(doc_id, [])


def test_the_audit_shows_the_cited_passage_in_context():
    src = _Source({"d1": [("d1#p0", "before"), ("d1#p1", "the quote itself"),
                          ("d1#p2", "after")]})
    ledger = [{"seat": "other", "text": "a claim", "grounded": True,
               "citations": [{"doc_id": "d1", "passage_id": "d1#p1",
                              "quote": "the quote itself", "status": "verified"}]}]
    body, docs = panel._audit_cases(ledger, "reviewer", src, radius=1)
    assert "--- case 1 ---" in body
    assert "CLAIM (other): a claim" in body
    assert ">>> the quote itself" in body
    assert "before" in body and "after" in body
    assert docs == {"d1"}


def test_a_seat_is_never_asked_to_referee_its_own_citation():
    src = _Source({"d1": [("d1#p0", "the quote")]})
    ledger = [{"seat": "reviewer", "text": "mine", "grounded": True,
               "citations": [{"doc_id": "d1", "passage_id": "d1#p0",
                              "quote": "the quote", "status": "verified"}]}]
    body, docs = panel._audit_cases(ledger, "reviewer", src, radius=2)
    assert docs == set()
    assert "no other seat filed" in body


def test_an_unverified_citation_has_nothing_to_referee():
    """Asking a seat to check a quote that failed the exact-match test would be
    asking it to re-run a test the machine already ran and answered."""
    src = _Source({"d1": [("d1#p0", "the quote")]})
    ledger = [{"seat": "other", "text": "a claim", "grounded": True,
               "citations": [{"doc_id": "d1", "passage_id": "d1#p0",
                              "quote": "not present", "status": "unmatched"}]}]
    _, docs = panel._audit_cases(ledger, "reviewer", src, radius=2)
    assert docs == set()


# --------------------------------------------------------------------------
# the leak invariant
# --------------------------------------------------------------------------

def _leaks(run_dir: Path) -> list[tuple[str, int, str]]:
    """Document ids named in a seat's packet that it was never given.

    TWO THINGS THIS GETS RIGHT THAT AN OBVIOUS VERSION GETS WRONG.

    1. `foreign` subtracts the seat's OWN permitted sets from EVERY turn, not
       just this one. A follow-up packet deliberately shows a seat its own prior
       claims with its own citations, and turn-N retrieval is run fresh -- so a
       document the seat itself read and cited in turn 1 is routinely absent
       from its turn-N permitted set. Computing `foreign` from the current turn
       alone would raise a false laundering alarm on any multi-turn run where a
       seat re-sees its own earlier citation.

    2. Ids are matched at a BOUNDARY, not as a bare substring, or `chen2021`
       matches inside `chen2021nature` and the alarm is a spelling coincidence.

    ONE PASS OVER THE TEXT, NOT ONE PASS PER DOCUMENT. The obvious loop --
    `re.search` for each candidate id -- is O(documents x packet bytes), and on
    the local corpus that is roughly a thousand scans of a megabyte-scale packet
    per seat per turn. It took the better part of an hour across the run
    directory, which is the same as not running: a test nobody waits for does
    not defend anything. All candidates go into one alternation instead,
    longest-first so a containing id wins the race, and the packet is read once.
    """
    man_path = run_dir / "manifest.json"
    if not man_path.exists():
        return []
    man = json.loads(man_path.read_text(encoding="utf-8"))
    turns = man.get("turns", {})
    out: list[tuple[str, int, str]] = []
    for key in sorted(turns, key=int):
        n = int(key)
        for seat, info in (turns[key].get("seats") or {}).items():
            mine: set[str] = set()
            foreign: set[str] = set()
            for m in sorted(turns, key=int):
                if int(m) > n:
                    break
                for other, oinfo in (turns[m].get("seats") or {}).items():
                    target = mine if other == seat else foreign
                    target |= set(oinfo.get("permitted", []))
            pkt = run_dir / "turns" / str(n) / "packets" / f"{seat}.md"
            if not pkt.exists():
                continue
            candidates = foreign - mine
            if not candidates:
                continue
            text = pkt.read_text(encoding="utf-8")
            alts = "|".join(re.escape(d) for d in
                            sorted(candidates, key=len, reverse=True))
            found = set(re.findall(
                rf"(?<![A-Za-z0-9])(?:{alts})(?![A-Za-z0-9])", text))
            out.extend((seat, n, d) for d in sorted(found & candidates))
    return out


def _runs() -> list[Path]:
    """Every local run that has a manifest.

    NOT a hard-coded run id. A hard-coded run id cannot notice runs that did not
    exist when the test was written.
    """
    root = Path(__file__).resolve().parents[1] / "runs"
    if not root.exists():
        return []
    return [p for p in sorted(root.iterdir())
            if p.is_dir() and (p / "manifest.json").exists()]


def test_no_packet_names_a_document_the_seat_was_never_given():
    """The single most important property of a multi-turn run. A seat that can
    see another seat's doc_id will cite a paper it was never shown, and the
    claim then rests on a source its author never read.

    THE ROUTE IS REPLAYED PROSE, NOT RETRIEVAL. The ids can arrive inside OTHER
    seats' claims, quoted verbatim into the prior-context block, because an
    agent writing about its evidence may name its own cited document. Retrieval
    never put the document in the packet. Two things close it, in `ops/panel.py`:
    disclosure spends its document budget deterministically and says out loud
    what did not fit, and any id the reading seat holds no passages of is masked
    out of the replayed prose.

    RUNS BUILT BEFORE THE MASK ARE REPORTED, NOT ASSERTED ON. Their packets are
    the record of what those agents were actually shown; editing them so this
    test turns green would falsify that record. They are identified by the
    absence of `leak_mask` in the manifest, the same convention
    `cumulative_context` and `retrieval_regime` already use.
    """
    runs = _runs()
    if not runs:
        pytest.skip("no runs/ manifests to inspect for packet document leaks")
    for run in runs:
        leaked = _leaks(run)
        if leaked:
            raise AssertionError(f"{run.name}: {leaked[:5]}")


def test_the_leak_check_is_not_vacuous(tmp_path):
    """A test that cannot fail is not a test. Inject a genuinely foreign id and
    the checker must name it."""
    run = tmp_path / "run"
    (run / "turns" / "1" / "packets").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"turns": {"1": {"seats": {
        "a": {"permitted": ["doc_a"]},
        "b": {"permitted": ["doc_b"]}}}}}), encoding="utf-8")
    pkt = run / "turns" / "1" / "packets" / "a.md"
    pkt.write_text("### [doc_a] fine\n", encoding="utf-8")
    assert _leaks(run) == []
    pkt.write_text("### [doc_a] fine\nas shown in [doc_b]\n", encoding="utf-8")
    assert _leaks(run) == [("a", 1, "doc_b")]


def test_an_id_that_merely_contains_another_id_is_not_a_leak(tmp_path):
    run = tmp_path / "run"
    (run / "turns" / "1" / "packets").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"turns": {"1": {"seats": {
        "a": {"permitted": ["chen2021nature"]},
        "b": {"permitted": ["chen2021"]}}}}}), encoding="utf-8")
    (run / "turns" / "1" / "packets" / "a.md").write_text(
        "### [chen2021nature] a paper\n", encoding="utf-8")
    assert _leaks(run) == []


def test_a_seats_own_earlier_citation_is_not_a_leak(tmp_path):
    """A seat is shown its own turn-1 citations at turn 2, and turn-2 retrieval
    may not have returned that document again."""
    run = tmp_path / "run"
    (run / "turns" / "2" / "packets").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"turns": {
        "1": {"seats": {"a": {"permitted": ["old_doc"]},
                        "b": {"permitted": ["other"]}}},
        "2": {"seats": {"a": {"permitted": ["new_doc"]},
                        "b": {"permitted": ["other"]}}}}}), encoding="utf-8")
    (run / "turns" / "2" / "packets" / "a.md").write_text(
        "you cited old_doc last turn\n### [new_doc] a paper\n", encoding="utf-8")
    assert _leaks(run) == []


def test_the_mask_removes_an_id_the_reader_was_never_handed():
    """The fix for the replayed-prose leak, tested on the shape it actually
    takes: a peer's claim naming its own cited paper in running prose."""
    text = ("**toma_reed** claimed:\n"
            "  - Her own cited paper, quire2016peloria, reports that "
            "echo pollen dims lanternmoss.\n")
    out, hit = panel._mask_unheld_ids(text, {"quire2016peloria"})
    assert hit == {"quire2016peloria"}
    assert "quire2016peloria" not in out
    assert panel.UNHELD_DOC_MASK in out
    # The argument survives. A mask that ate the sentence would cost the panel
    # the cross-examination the disclosure stratum exists to enable.
    assert "echo pollen dims lanternmoss" in out


def test_the_mask_leaves_a_document_the_reader_does_hold():
    text = "  - as shown in held_doc, the effect is real\n"
    out, hit = panel._mask_unheld_ids(text, set())
    assert out == text and hit == set()


def test_the_mask_matches_at_a_boundary_like_the_leak_check_does():
    """Must agree with `_leaks`, or the mask and the check disagree about what a
    leak is and the test passes for the wrong reason."""
    out, hit = panel._mask_unheld_ids("see chen2021nature here", {"chen2021"})
    assert hit == set()
    assert out == "see chen2021nature here"


def test_the_mask_prefers_the_longer_id_when_one_contains_another():
    out, hit = panel._mask_unheld_ids(
        "see chen2021nature here", {"chen2021", "chen2021nature"})
    assert hit == {"chen2021nature"}
    assert "chen2021" not in out


def test_the_permitted_universe_is_every_seats_every_turn(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"turns": {
        "1": {"seats": {"a": {"permitted": ["d1"]}, "b": {"permitted": ["d2"]}}},
        "2": {"seats": {"a": {"permitted": ["d3"]}}}}}), encoding="utf-8")
    assert panel._permitted_universe(run) == {"d1", "d2", "d3"}


def test_disclosure_spends_its_budget_on_contested_documents_first():
    """The silent bug this replaces: the budget was spent in ledger insertion
    order, so which documents were handed over depended on which seat the
    verifier happened to reach first, and the overflow vanished unrecorded."""
    examine = {
        "solo_z": {"seats": ["a"], "quotes": []},
        "solo_a": {"seats": ["b"], "quotes": []},
        "contested": {"seats": ["a", "b", "c"], "quotes": []},
        "pair": {"seats": ["a", "b"], "quotes": []},
    }
    assert panel._disclosure_order(examine) == [
        "contested", "pair", "solo_a", "solo_z"]


def test_audit_packets_are_kept_out_of_the_packets_directory():
    """The audit bends the isolation rule ON PURPOSE and its packets contain
    foreign doc_ids by design. That is legitimate -- the turn is already frozen
    and the prompt forbids citing what it shows -- but only because the leak
    invariant above guards `packets/`, which the audit must therefore never be
    written into."""
    runs = _runs()
    if not runs:
        pytest.skip("no runs/ manifests to inspect for audit packet placement")
    for run in runs:
        for ad in run.glob("turns/*/audit"):
            assert ad.name == "audit"
            assert (ad.parent / "packets") != ad
            for f in ad.glob("packets/*.md"):
                assert "audit" in f.parts, f

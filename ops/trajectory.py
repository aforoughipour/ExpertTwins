"""Measure the TRAJECTORY of a long panel run, turn by turn.

WHY THIS EXISTS SEPARATELY FROM `render`. `render` produces the transcript,
which is the scientific product. This produces something different and much less
interesting to read: evidence about whether the *panel* is still working at turn
12. Those are different questions, and conflating them is how a study ends up
reporting the quality of an argument as if it were the quality of the machine
that produced it.

WHAT IT DOES NOT DO. It does not read the science. Nothing here knows what an
prism spores is. Every number is computed from the frozen ledgers and reports by set and
string operations, so it cannot be talked into a favourable reading -- the same
reason `ops/panel.py` deliberately calls no model.

THE FOUR PREDICTED FAILURE MODES, named before the data exists, which is the
only thing that stops them being explained away afterwards:

    F1  convergence collapse   disagreement -> 0 and stays there
    F2  evidence exhaustion    NEW documents per turn -> 0; seats recycle the
                               same papers, and a panel arguing from a closed
                               set has stopped consulting the literature
    F3  restatement            claims become near-duplicates of claims already
                               on the table, most damningly the seat's own
    F4  grounding decay        the grounded fraction falls as seats reach past
                               the evidence they were given

F1 and F3 pull in opposite directions and that is the point. A panel can hold
disagreement at 100% forever by repeating its opening position verbatim, which
is not a conversation. A panel can stay novel forever by drifting off the
question. Only reading the four together says anything.

WHAT IS REUSED AND WHAT IS NOT. F1, F3 and herding are already computed during
`ingest` and frozen into `report.json` -- `diversity.measure` and
`diversity.drift`, with a permutation null behind them. They are read here, not
recomputed, because a second implementation of the same quantity is a second
chance to disagree with the record. F2 and the per-seat self-restatement split
are computed here from the ledgers, because nothing computed them before.

Disagreement is not detected from keywords: seats declare stances explicitly and
separation is measured against a shuffle null, which is a better instrument than
counting the word "disagree".

USAGE
    python ops/trajectory.py <run>            # print and write trajectory.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expertwins import paths  # noqa: E402
from expertwins.diversity import embed  # noqa: E402

#: Conventions, not measurements. Stated in one place so they can be argued
#: with. Nothing here gates anything; they only decide when a line is flagged.
NOVELTY_FLOOR = 0.30
NEW_DOC_FLOOR = 0.15
GROUNDED_FLOOR = 0.80
NEAR_DUPLICATE = 0.80


def _turns(run_dir: Path) -> list[int]:
    t = run_dir / "turns"
    if not t.exists():
        return []
    return sorted(int(p.name) for p in t.iterdir() if p.name.isdigit())


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _self_restatement(by_turn: dict[int, list[dict]], n: int) -> float | None:
    """Share of a turn's claims that near-duplicate the SAME seat's own earlier
    claims.

    This is the sharpest of the four signals and the one `drift.novelty` cannot
    give on its own: novelty is measured against everything anyone has said, so
    a seat restating itself while the table as a whole moves on is invisible in
    it. A panel in which every seat is repeating itself while disagreeing with
    the others scores well on separation, well on lexical diversity, and is not
    a conversation.
    """
    cur = by_turn.get(n, [])
    if not cur:
        return None
    scored: list[float] = []
    for seat in {c["seat"] for c in cur}:
        earlier = [c["text"] for m in by_turn if m < n
                   for c in by_turn[m] if c["seat"] == seat]
        now = [c["text"] for c in cur if c["seat"] == seat]
        if not earlier or not now:
            continue
        vecs, _ = embed(earlier + now)
        old, new = vecs[:len(earlier)], vecs[len(earlier):]
        sim = new @ old.T
        scored += [float(x) for x in sim.max(axis=1)]
    if not scored:
        return None
    return sum(1 for s in scored if s >= NEAR_DUPLICATE) / len(scored)


def trajectory(run_dir: Path) -> dict:
    ns = _turns(run_dir)
    if not ns:
        raise SystemExit(f"no turns in {run_dir}")

    by_turn: dict[int, list[dict]] = {}
    for n in ns:
        led = _read(turn_path(run_dir, n) / "ledger.json")
        if led is not None:
            by_turn[n] = led

    rows: list[dict] = []
    seen_docs: set[str] = set()
    for n in ns:
        ledger = by_turn.get(n)
        if ledger is None:
            # A turn that was prepared but never ingested is a HOLE, not a zero.
            rows.append({"turn": n, "status": "not ingested"})
            continue
        rep = _read(turn_path(run_dir, n) / "report.json") or {}
        cited = {ci["doc_id"] for c in ledger for ci in c.get("citations", [])
                 if ci.get("status") == "verified"}
        new_docs = cited - seen_docs
        seen_docs |= cited
        grounded = sum(1 for c in ledger if c.get("grounded"))
        div = rep.get("diversity") or {}
        drift = rep.get("drift") or {}
        # `axes` is a dict keyed by axis name. An axis whose permutation null
        # saturated is marked UNINFORMATIVE, and reading it as a collapse would
        # flag the healthiest possible configuration -- seats whose evidence is
        # disjoint by construction -- as the failure. So it is carried as None.
        pos = (div.get("axes") or {}).get("positions") or {}
        pos_sep = pos.get("separation") if pos.get("informative", True) else None

        checks = _read(turn_path(run_dir, n) / "checks.json") or []
        audit = _read(turn_path(run_dir, n) / "audit" / "checks.json") or []
        adverse = sum(1 for c in checks + audit
                      if c.get("verdict") in ("overstated", "misread", "irrelevant")
                      and c.get("status") == "recorded")

        rows.append({
            "turn": n,
            "status": "ok",
            "n_claims": len(ledger),
            "grounded_fraction": (grounded / len(ledger)) if ledger else None,
            "docs_cited": len(cited),
            "new_docs": len(new_docs),
            "new_doc_fraction": (len(new_docs) / len(cited)) if cited else None,
            "novelty": drift.get("novelty"),
            "herding": drift.get("herding"),
            "self_restatement": _self_restatement(by_turn, n),
            "positions_separation": pos_sep,
            "effective_positions": div.get("effective_positions"),
            "adverse_citation_checks": adverse,
            "applied_move": (rep.get("guardrail") or {}).get("move"),
        })
    return {"run": run_dir.name, "turns": rows,
            "conventions": {"novelty_floor": NOVELTY_FLOOR,
                            "new_doc_floor": NEW_DOC_FLOOR,
                            "grounded_floor": GROUNDED_FLOOR,
                            "near_duplicate": NEAR_DUPLICATE}}


def turn_path(run_dir: Path, n: int) -> Path:
    return run_dir / "turns" / str(n)


def _fmt(v, spec: str = "{:.2f}") -> str:
    return "  --  " if v is None else spec.format(v)


def render(traj: dict) -> str:
    L = [f"TRAJECTORY -- {traj['run']}", ""]
    L.append(f"{'turn':>4s} {'claims':>7s} {'grnd':>6s} {'docs':>5s} {'new':>5s} "
             f"{'new%':>6s} {'novel':>6s} {'self-rst':>8s} {'herd':>6s} "
             f"{'pos-sep':>7s} {'adv':>4s}  move")
    L.append("-" * 92)
    for r in traj["turns"]:
        if r["status"] != "ok":
            L.append(f"{r['turn']:>4d} {'NOT INGESTED -- a hole, not a zero':>7s}")
            continue
        L.append(
            f"{r['turn']:>4d} {r['n_claims']:>7d} "
            f"{_fmt(r['grounded_fraction'], '{:.0%}'):>6s} "
            f"{r['docs_cited']:>5d} {r['new_docs']:>5d} "
            f"{_fmt(r['new_doc_fraction'], '{:.0%}'):>6s} "
            f"{_fmt(r['novelty']):>6s} {_fmt(r['self_restatement'], '{:.0%}'):>8s} "
            f"{_fmt(r['herding'], '{:+.2f}'):>6s} "
            f"{_fmt(r['positions_separation']):>7s} "
            f"{r['adverse_citation_checks']:>4d}  {r['applied_move'] or ''}")

    ok = [r for r in traj["turns"] if r["status"] == "ok"]
    L += ["", "THE FOUR FAILURE MODES", ""]
    fired = []

    late = [r for r in ok if r["turn"] > 1]
    f1 = [r for r in late if r["positions_separation"] is not None
          and r["positions_separation"] < 0.10]
    f2 = [r for r in late if r["new_doc_fraction"] is not None
          and r["new_doc_fraction"] < NEW_DOC_FLOOR]
    f3 = [r for r in late if r["novelty"] is not None and r["novelty"] < NOVELTY_FLOOR]
    f4 = [r for r in ok if r["grounded_fraction"] is not None
          and r["grounded_fraction"] < GROUNDED_FLOOR]

    for key, hits, text in (
        ("F1 convergence collapse", f1,
         "positions separation near zero: the seats are answering as one"),
        ("F2 evidence exhaustion", f2,
         f"under {NEW_DOC_FLOOR:.0%} of citations were to documents not already "
         f"cited: the panel is arguing from a closed set"),
        ("F3 restatement", f3,
         f"novelty under {NOVELTY_FLOOR:.0%}: the claims are already on the table"),
        ("F4 grounding decay", f4,
         f"grounded fraction under {GROUNDED_FLOOR:.0%}: seats are reaching past "
         f"their evidence"),
    ):
        if hits:
            fired.append(key)
            L.append(f"  {key}: FIRING on turn(s) "
                     f"{', '.join(str(r['turn']) for r in hits)} -- {text}")
        else:
            L.append(f"  {key}: not firing")

    L += ["", "HOW TO READ THIS", ""]
    L.append("  F1 and F3 pull in OPPOSITE directions. A panel can hold")
    L.append("  disagreement at 100% forever by repeating its opening position,")
    L.append("  and it can stay novel forever by drifting off the question. A")
    L.append("  single line here says nothing; the four together say something.")
    L.append("")
    L.append("  The thresholds are conventions, not measurements. They decide")
    L.append("  which lines are flagged and nothing else -- no gate reads them.")
    if len(ok) < 4:
        L.append("")
        L.append(f"  {len(ok)} ingested turn(s). A trajectory needs a trajectory:")
        L.append("  with fewer than about four turns these are single differences,")
        L.append("  not trends, and should not be quoted as either.")
    if fired:
        L.append("")
        L.append("  A firing mode is a fact about THIS run with THIS model on")
        L.append("  THIS question. n = 1 is not a measurement of the design.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--json", action="store_true",
                    help="print the JSON instead of the table")
    a = ap.parse_args()

    run_dir = None
    for p in (paths.resolve(a.run_dir), paths.RUNS / a.run_dir):
        if p.exists():
            run_dir = p
            break
    if run_dir is None:
        raise SystemExit(f"no such run: {a.run_dir}")

    traj = trajectory(run_dir)
    out = run_dir / "trajectory.json"
    out.write_text(json.dumps(traj, indent=2), encoding="utf-8")
    print(json.dumps(traj, indent=2) if a.json else render(traj))
    print(f"\nwrote {paths.relative(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

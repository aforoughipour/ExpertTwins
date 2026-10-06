"""Is this seat still the person? Two instruments, one of which has ground truth.

FIDELITY QUESTION: "is this something Elena would
say?" Everywhere else in this system that question is answered by construction --
the seat can only cite what it was handed, and its citations are tiered `own` or
`read`. This module tries to answer it as a MEASUREMENT.

    1. TEMPORAL HOLDOUT -- the one with ground truth.

       Cut the corpus at year Y. Ask the seat a question that one of ITS OWN
       papers from year Y+1 answers. The paper is the ground truth for what that
       scientist actually concluded, and it is not a proxy: it is the person's
       own published position, written by them, after the cut.

       This is the same protocol shape as held-out response prediction in
       individual-simulation work (arXiv:2411.10109, VERIFIED -- agents grounded
       in a person's own long-form self-report reach 83% of that person's own
       two-week test-retest ceiling, against 74% for demographics-only agents).
       Here the "self-report" is the published corpus and the held-out response
       is the next paper.

       The harness builds the pairs and freezes them. It does NOT grade: whether
       a claim matches a paper's conclusion is a semantic judgment, and this
       project does not pretend a semantic judgment is deterministic. Grading is
       a human or a blinded model, and the harness keeps the two apart.

    2. STYLOMETRIC ATTRIBUTION -- model-free, and weaker, and useful anyway.

       Build a tf-idf centroid of every seat's own corpus. Take a claim the seat
       filed and ask which centroid it is nearest to. If a claim attributed to
       Marchetti lands nearest Moreau's corpus, something has gone wrong -- either
       the seat has drifted, or these two seats are not distinguishable in the
       first place, and both are worth knowing.

       HONEST LIMIT: authorship attribution over a handful of short sentences is
       weak, and nearest-centroid over tf-idf is the weakest form of it. A high
       confusion rate is informative; a low one proves very little. It is here
       because it costs nothing and needs no network, no GPU and no model, so it
       can run on the cluster beside the panel.

Usage:
    python ops/fidelity.py holdout elena_marchetti --year 2020 --n 8
    python ops/fidelity.py attribute runs/rt-01 --turn 1
    python ops/fidelity.py centroids
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expertwins import paths  # noqa: E402
from expertwins.diversity import tokens  # noqa: E402
from expertwins.identity import Individual, load_own  # noqa: E402


def _conn() -> sqlite3.Connection:
    if not paths.INDEX.exists():
        raise SystemExit(f"no index at {paths.INDEX}")
    return sqlite3.connect(f"file:{paths.INDEX.as_posix()}?mode=ro", uri=True)


# --------------------------------------------------------------------------
# 1. temporal holdout
# --------------------------------------------------------------------------

def cmd_holdout(seat: str, year: int, n: int) -> int:
    """Build frozen (question, held-out own paper) pairs for one seat.

    The pairs are written, not graded. A run of the panel with
    `--before-year <year>` cannot see any of the held-out papers, so whatever
    the seat says about them is a genuine prediction rather than a recollection.
    """
    conn = _conn()
    own = load_own(seat)
    if not own:
        raise SystemExit(
            f"`{seat}` has no attributed own corpus. Run "
            f"`python ops/people.py attribute config/people/{seat}.yaml` first. "
            f"A fidelity test against a corpus that has not been attributed "
            f"would be testing the search engine, not the scientist.")

    ids = list(own)
    rows: list[tuple] = []
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        rows += conn.execute(
            f"SELECT doc_id, year, title, n_passages FROM docs "
            f"WHERE doc_id IN ({marks}) AND year >= ? ORDER BY year, doc_id",
            (*chunk, year)).fetchall()

    if not rows:
        raise SystemExit(
            f"`{seat}` has no attributed papers from {year} onward, so there is "
            f"nothing to hold out. Lower --year, or accept that this seat cannot "
            f"be tested this way -- which is itself a finding about the corpus.")

    rows = [r for r in rows if (r[3] or 0) >= 5][:n]
    out = {
        "seat": seat, "cut_year": year, "n_pairs": len(rows),
        "protocol": (
            "Run the panel with --before-year {y}. No seat can then see any "
            "paper published in {y} or later, so the seat has never read the "
            "held-out papers below. Ask the question; compare what the seat "
            "concludes with what the paper concluded. GRADE THIS BY HAND, or "
            "with a model that is blind to which seat produced the answer -- "
            "'does this claim match the paper's conclusion' is a semantic "
            "judgment and nothing in this repository adjudicates meaning."
        ).format(y=year),
        "pairs": [{"doc_id": d, "year": y, "title": t,
                   "question": f"What is your current position on: {t}?",
                   "grade": None,
                   "grade_scale": "0 = contradicts the paper, 1 = unrelated, "
                                  "2 = compatible but not the paper's point, "
                                  "3 = the paper's conclusion in this seat's voice"}
                  for d, y, t, _ in rows],
    }
    p = paths.RUNS / "fidelity" / f"{seat}.holdout.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{seat}: {len(rows)} held-out papers from {year} onward")
    for r in out["pairs"]:
        print(f"  {r['year']}  {r['doc_id']:<28s} {r['title'][:64]}")
    print(f"\nwrote {paths.relative(p)}")
    print(f"\nNow run the panel blind to those years:")
    print(f'  python ops/panel.py prepare "<one of the questions above>" \\')
    print(f"      --out runs/fid-{seat} --seats {seat} --task solo "
          f"--before-year {year}")
    return 0


# --------------------------------------------------------------------------
# 2. stylometric attribution
# --------------------------------------------------------------------------

def _centroids(seats: list[str], per_seat_docs: int = 200) -> tuple[dict, dict]:
    """A tf-idf centroid per seat, built from that seat's own corpus.

    Deliberately model-free: this has to be runnable on an air-gapped cluster
    beside the panel, with no download and no GPU.
    """
    conn = _conn()
    vocab: dict[str, int] = {}
    per_seat: dict[str, list[str]] = {}
    for s in seats:
        own = list(load_own(s))[:per_seat_docs]
        if not own:
            continue
        text: list[str] = []
        for i in range(0, len(own), 200):
            chunk = own[i:i + 200]
            marks = ",".join("?" * len(chunk))
            text += [r[0] for r in conn.execute(
                f"SELECT text FROM passages WHERE doc_id IN ({marks}) LIMIT 4000",
                chunk)]
        toks = tokens(" ".join(text))
        per_seat[s] = toks
        for t in toks:
            vocab.setdefault(t, len(vocab))
    return per_seat, vocab


def _vectorise(toks: list[str], vocab: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(vocab))
    for t in toks:
        i = vocab.get(t)
        if i is not None:
            v[i] += 1.0
    v = np.log1p(v)
    n = np.linalg.norm(v)
    return v / n if n else v


def cmd_centroids() -> int:
    """Which seats are distinguishable from their own writing alone?"""
    seats = [p.stem for p in sorted(paths.PEOPLE.glob("*.yaml"))]
    per_seat, vocab = _centroids(seats)
    if len(per_seat) < 2:
        raise SystemExit(
            "fewer than two seats have an attributed own corpus; there is "
            "nothing to distinguish. Run `people.py attribute` first.")
    names = sorted(per_seat)
    M = np.vstack([_vectorise(per_seat[s], vocab) for s in names])
    S = M @ M.T
    print("cosine similarity between seats' OWN corpora "
          f"({len(vocab)} terms, tf-idf-free log-count centroids)\n")
    print(" " * 24 + "".join(f"{n[:7]:>9s}" for n in names))
    for i, a in enumerate(names):
        row = "".join(f"{S[i, j]:9.2f}" for j in range(len(names)))
        print(f"{a:<24s}{row}")
    print("\nHigh off-diagonal values are NOT automatically a fault: co-authors "
          "share a literature, and three of these seats worked in one institute "
          "for two decades. What they do mean is that the evidence axis of the "
          "heterogeneity measure will read lower between those pairs for a real "
          "reason, and that `moves` and `fatal_flaws` are carrying the "
          "separation for them instead of the corpus.")
    return 0


def cmd_attribute(run: str, turn: int) -> int:
    """For each filed claim, which seat's own corpus does it most resemble?"""
    run_dir = paths.resolve(run)
    if not run_dir.exists():
        run_dir = paths.RUNS / run
    ledger = json.loads((run_dir / "turns" / str(turn) / "ledger.json")
                        .read_text(encoding="utf-8"))
    seats = sorted({c["seat"] for c in ledger})
    per_seat, vocab = _centroids(seats)
    have = [s for s in seats if s in per_seat]
    if len(have) < 2:
        raise SystemExit("fewer than two seats have an attributed own corpus")
    C = np.vstack([_vectorise(per_seat[s], vocab) for s in have])

    hit = tot = 0
    confusion: dict[str, dict[str, int]] = {}
    for c in ledger:
        if c["seat"] not in have or not c["grounded"]:
            continue
        v = _vectorise(tokens(c["text"]), vocab)
        if not v.any():
            continue
        pred = have[int(np.argmax(C @ v))]
        tot += 1
        hit += pred == c["seat"]
        confusion.setdefault(c["seat"], {}).setdefault(pred, 0)
        confusion[c["seat"]][pred] += 1

    print(f"stylometric attribution, turn {turn}: {hit}/{tot} claims land "
          f"nearest their own author's corpus"
          + (f" ({hit/tot:.0%}; chance is {1/len(have):.0%})" if tot else ""))
    for seat in sorted(confusion):
        row = ", ".join(f"{k} {v}" for k, v in
                        sorted(confusion[seat].items(), key=lambda kv: -kv[1]))
        print(f"  {seat:<26s} -> {row}")
    print("\nREAD THIS WEAKLY. Nearest-centroid attribution over a handful of "
          "short claims is a weak instrument: a high confusion rate is "
          "informative, a low one proves little. Its value is that it costs "
          "nothing and runs air-gapped.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("holdout"); h.add_argument("seat")
    h.add_argument("--year", type=int, required=True)
    h.add_argument("--n", type=int, default=8)
    a = sub.add_parser("attribute"); a.add_argument("run")
    a.add_argument("--turn", type=int, default=1)
    sub.add_parser("centroids")

    args = ap.parse_args()
    if args.cmd == "holdout":
        return cmd_holdout(args.seat, args.year, args.n)
    if args.cmd == "attribute":
        return cmd_attribute(args.run, args.turn)
    return cmd_centroids()


if __name__ == "__main__":
    sys.exit(main())

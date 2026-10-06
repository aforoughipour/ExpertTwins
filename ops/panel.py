"""Convene the panel: prepare -> dispatch -> ingest -> (followup) -> render.

THE BRIDGE between the corpus and the seats. It deliberately does NOT call a
model. The seats are frontier agents driven externally, for two reasons:

  1. It makes model choice explicit: seats answer through export/ingest, so the
     deterministic layer does not bake in a provider or topology.
  2. It keeps the deterministic layer honest. Nothing in this file can be talked
     into accepting a citation; it either matches the stored bytes or it does
     not.

PROTOCOL

    prepare   each seat gets ONLY its own evidence; the divergence prior is
              computed from the packets before any token is spent
    generate  seats answer IN ISOLATION, in parallel (external)
    ingest    claims verified against THAT SEAT'S permitted set, citations
              tiered own/read, heterogeneity measured, floor fixed at turn 1,
              claim set frozen and hashed
    followup  a new turn. Seats see the new question, their OWN prior claims
              FROM EVERY TURN SO FAR, other seats' CLAIM TEXT ONLY -- never
              their packets, searches or reasoning -- and, in a section of their
              evidence marked as such, THE DOCUMENTS THOSE SEATS CITED. If the
              table collapsed, the guardrail's move is applied HERE, to the
              retrieval, before the packet is written.
    audit     open every document cited in a turn to the whole panel for
              review, asking for nothing but the reading. Needed because
              disclosure reaches a seat on the FOLLOWING turn, so the citations
              of a run's last turn -- the ones a reader quotes -- are otherwise
              never checked by anybody.
    render    the transcript, plus the heterogeneity record

WHY CLAIM-TEXT-ONLY BETWEEN TURNS, AND WHAT IS UNPROVEN ABOUT IT. Agents reading
each other's COMPLETE outputs converge within one round, and the mechanism is
the information exchanged rather than the number of agents. So between turns we
pass claims, not drafts, and never another seat's packet. That is a HYPOTHESIS:
nobody has tested whether claim-only exchange also converges. The guardrail
exists because it might.

THE ONE DELIBERATE EXCEPTION: THE DISCLOSURE RULE. A document a seat CITED is
handed to every other seat on the next turn, with the quoted passage marked, and
each seat files a verdict on whether the use made of it holds. A citation is a
public assertion about what a paper says; if no one else can open the paper, the
exact-match check has verified that the quote exists and nothing has verified
that it means what it was said to mean. That second judgement cannot be made
mechanically, and this is how a table of scientists has always made it.

The cost is honest and is paid explicitly: those documents are shared BY
CONSTRUCTION, so they are excluded from every heterogeneity measure --
`EvidencePacket.evidence_docs` for the packets, and citations to them are
dropped from the evidence axis. Otherwise the disclosure rule would manufacture
exactly the overlap the guardrail reads as collapse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from expertwins import guardrail as gr  # noqa: E402
from expertwins import identity, paths, tasks  # noqa: E402
from expertwins.diversity import SeatTurn, drift, embed, measure, preflight  # noqa: E402
from expertwins.retrieve import Retriever  # noqa: E402
from expertwins import retrieve as retrieve_mod  # noqa: E402
from expertwins.verify import (  # noqa: E402
    Citation, Claim, PassageSource, Report, Tier, Verifier, repair_prompt,
)


class LibrarySource(PassageSource):
    """Passage source backed by the assembled library."""

    def __init__(self, library_root: Path) -> None:
        from expertwins.corpus.store import Library

        self.lib = Library(library_root)
        self._cache: dict[str, list[tuple[str, str]]] = {}

    def passages(self, doc_id: str) -> list[tuple[str, str]]:
        if doc_id not in self._cache:
            try:
                self._cache[doc_id] = [
                    (p.passage_id, p.text) for p in self.lib.load_passages(doc_id)]
            except Exception:                                     # noqa: BLE001
                self._cache[doc_id] = []
        return self._cache[doc_id]

    def has(self, doc_id: str) -> bool:
        return bool(self.passages(doc_id))


# --------------------------------------------------------------------------
# seats
# --------------------------------------------------------------------------

def load_seats(path: Path, only: list[str] | None) -> list[dict]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    seats = doc["seats"]
    if not only:
        return seats
    names = {s["name"] for s in seats}
    missing = [n for n in only if n not in names]
    if missing:
        # A misspelled seat name must never silently produce a smaller panel.
        raise SystemExit(f"unknown seat(s): {', '.join(missing)}\n"
                         f"available: {', '.join(sorted(names))}")
    order = {n: i for i, n in enumerate(only)}
    return sorted((s for s in seats if s["name"] in only),
                  key=lambda s: order[s["name"]])


def turn_dir(run_dir: Path, n: int) -> Path:
    return run_dir / "turns" / str(n)


def latest_turn(run_dir: Path) -> int:
    t = run_dir / "turns"
    if not t.exists():
        return 0
    ns = [int(p.name) for p in t.iterdir() if p.name.isdigit()]
    return max(ns) if ns else 0


def _seat_list(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


# --------------------------------------------------------------------------
# prepare / followup
# --------------------------------------------------------------------------

def _build_turn(run_dir: Path, question: str, seats: list[dict], index: Path,
                task: tasks.Task, *, max_passages: int, own_floor: int,
                before_year: int | None, turn: int, material: str = "",
                prior: dict[str, str] | None = None,
                nudges: dict[str, str] | None = None,
                exclude_docs: set[str] | None = None,
                restrict_own: bool = False, stances: bool = False,
                examine: dict[str, dict] | None = None,
                max_per_doc: int = 8, min_docs: int = 10,
                neighbours: int = 1, max_seeds: int = 200,
                retrieval_text: str | None = None) -> dict:
    """Build one turn: one packet per seat, and the preflight over all of them.

    `question` is what the seat is ASKED. `retrieval_text` is what the corpus is
    SEARCHED with. They are the same thing on turn 1 and must NOT be on later
    turns.

    WHY THEY ARE SEPARATE. A follow-up question is mostly procedure -- "name the
    claim you most disagree with and ground your objection". Fed to the
    retriever, that text produces precision queries like

        "agreement" AND "inference" AND "right"

    which is a query about the English language and not about the subject.
    Without a separate retrieval text, follow-up turns would be retrieved that
    way. The damage is bounded, because a seat's own queries carry most of the
    topicality and they do not change between turns, but the precision bucket --
    whose entire job is to land on the intersection the question is about --
    would be spent on function words.

    Separating the two prevents a silent failure: nothing crashes, the seat
    receives a full packet and answers confidently from it, but the retriever is
    steering by the moderator's grammar rather than by the subject.

    The fix is to retrieve on the ORIGINAL question joined to this turn's
    question. Joined, not replaced: a later turn that really does introduce new
    subject matter must still be able to pull evidence toward itself. Turn 1 is
    unchanged by construction.
    """
    td = turn_dir(run_dir, turn)
    (td / "packets").mkdir(parents=True, exist_ok=True)
    (td / "responses").mkdir(parents=True, exist_ok=True)

    retriever = Retriever(index)
    info: dict[str, dict] = {}
    packet_docs: dict[str, set[str]] = {}
    rubrics: dict[str, str] = {}
    all_text: list[str] = []
    masked_ids: dict[str, list[str]] = {}
    examine = examine or {}
    search_on = retrieval_text or question

    # Every document any seat has ever been permitted in this run. A doc_id in
    # this set that is NOT in the seat reading it is exactly the leak condition
    # the harness tests for, so the mask is built from the same universe the
    # test uses rather than from a guess about what an id looks like.
    universe = _permitted_universe(run_dir)

    print(f"\nturn {turn} [{task.key}]: {question}\n")
    focus = retriever.focus_query(search_on)
    print(f"precision query: {focus or '(none -- too few discriminative terms)'}")
    print(f"budget: {max_seeds} seeds, {max_passages} passages, "
          f"neighbours {neighbours}\n")
    print(f"{'seat':<26s} {'passages':>9s} {'docs':>6s} {'own':>5s} {'chk':>4s} "
          f"{'cov':>5s}  territory")
    print("-" * 78)
    for spec in seats:
        name = spec["name"]
        own = identity.load_own(name)
        # A seat does not cross-examine itself: only documents cited by SOMEBODY
        # ELSE are placed under examination in this seat's packet.
        mine = {d: v for d, v in examine.items() if v["seats"] != [name]}
        by = {d: [s for s in v["seats"] if s != name] for d, v in mine.items()}
        quotes = {d: v["quotes"] for d, v in mine.items()}
        pkt = retriever.packet_for_seat(
            name, spec.get("queries", []), search_on,
            own=own, own_queries=spec.get("own_queries", []),
            own_floor=own_floor,
            max_passages=max_passages, max_seeds=max_seeds,
            before_year=before_year,
            exclude_docs=exclude_docs,
            restrict_docs=own if restrict_own else None,
            max_per_doc=max_per_doc, min_docs=min_docs, neighbours=neighbours,
            examine=by, examine_quotes=quotes)
        terr = identity.territory_of(pkt.evidence_docs, own)
        audit = pkt.audit()

        # THE REPLAYED-PROSE LEAK. The prior-context block quotes other seats'
        # claims verbatim, and an agent writing about its evidence names it:
        # "her own cited paper, quire2016peloria, reports...". The id then
        # sits in THIS seat's packet for a document it holds no passages of.
        # verify.py kills a citation to a non-permitted doc as OUT_OF_SCOPE, so
        # the grounding invariant survives -- but the seat has still been handed
        # an id it can only misuse, and that is the mechanism described in
        # verify.py's docstring (a seat citing a document it was never shown,
        # having picked the id out of a peer's claim).
        # Disclosure hands over the DOCUMENT where it can; where the budget or
        # the index could not, the id is masked rather than left as bait.
        seat_prior, masked = _mask_unheld_ids(
            (prior or {}).get(name, ""), universe - pkt.permitted)
        if masked:
            masked_ids[name] = sorted(masked)

        prompt = tasks.build_packet(
            task=task,
            display=spec.get("display", name),
            question=question,
            material=material,
            evidence=pkt.render(),
            moves=spec.get("moves", ""),
            territory_desc=spec.get("territory", ""),
            fatal_flaws=spec.get("fatal_flaws", ""),
            territory_block=tasks.TERRITORY_BLOCK.format(
                n_own=terr.n_own_docs, n_docs=terr.n_docs,
                own_pct=terr.own_fraction, zone=terr.zone.upper(),
                standard=terr.standard),
            prior=seat_prior,
            nudge=(nudges or {}).get(name, ""),
            stances=stances and task.cross_talk,
            disclosure=bool(pkt.examined))
        (td / "packets" / f"{name}.md").write_text(prompt, encoding="utf-8")

        info[name] = {"permitted": sorted(pkt.permitted),
                      "own": sorted(pkt.own_docs),
                      # Recorded separately BECAUSE it is excluded from every
                      # measure: a reader has to be able to see what was left out
                      # of the statistics and why.
                      "examined": sorted(pkt.examined),
                      "examined_by": {d: v for d, v in pkt.examined_by.items()
                                      if d in pkt.examined},
                      # Ids removed from this seat's replayed prior context
                      # because disclosure could not hand over the document.
                      # Recorded so a reader can see what the seat was NOT able
                      # to check, rather than having it vanish silently.
                      "masked_ids": masked_ids.get(name, []),
                      "n_passages": len(pkt.passages),
                      "territory": terr.zone,
                      "own_fraction": round(terr.own_fraction, 4),
                      "audit": audit.to_json()}
        packet_docs[name] = pkt.evidence_docs
        rubrics[name] = spec.get("fatal_flaws", "")
        all_text += [p.text for p in pkt.passages if p.doc_id not in pkt.examined]

        print(f"{name:<26s} {audit.n_passages:>9d} {audit.n_docs:>6d} "
              f"{terr.n_own_docs:>5d} {len(pkt.examined):>4d} "
              f"{audit.term_coverage:>5.0%}  {terr.zone}")
        if not pkt.passages:
            # Refused loudly. A seat with no evidence answers "no evidence
            # found", which is indistinguishable from a genuine scientific
            # negative and would silently poison the panel.
            print(f"  REFUSED: {name} has no evidence for this question")
        for reason in audit.reasons:
            # A THIN packet is not an error and is not fixed here. It is
            # announced, before any token is spent, because the alternative is a
            # seat answering thinly and a reader assuming it had the literature.
            print(f"  THIN PACKET [{name}]: {reason}")

    # THE SWEEP. The universe above was read from the manifest, which does not
    # yet contain THIS turn's permitted sets -- so a seat whose packet names a
    # document only a CONCURRENT seat was given would slip through. That cannot
    # happen by the prose route (replayed claims only quote earlier turns), but
    # "cannot happen by the route I thought of" is exactly the reasoning that
    # left the leak in place for 22 runs. The packets are now on disk and the
    # full universe is known, so the invariant is enforced rather than argued.
    #
    # Masking the whole file is safe by construction: `unheld` excludes
    # `pkt.permitted`, and every id in a seat's evidence and examination blocks
    # is permitted by definition.
    universe |= {d for i in info.values() for d in i["permitted"]}
    for name, i in info.items():
        pkt_path = td / "packets" / f"{name}.md"
        swept, extra = _mask_unheld_ids(
            pkt_path.read_text(encoding="utf-8"), universe - set(i["permitted"]))
        if extra:
            pkt_path.write_text(swept, encoding="utf-8")
            merged = sorted(set(i["masked_ids"]) | extra)
            i["masked_ids"] = merged
            masked_ids[name] = merged

    pre = preflight(packet_docs, rubrics, all_text)
    print("\nPREFLIGHT -- computed from the packets, before any token is spent")
    print(pre.render())
    if masked_ids:
        n_masked = len({d for v in masked_ids.values() for d in v})
        print(f"  LEAK MASK: {n_masked} doc_id(s) named in other seats' "
              f"replayed claims were masked out of {len(masked_ids)} packet(s), "
              f"because disclosure could not hand those documents over. Without "
              f"the mask a seat is shown an id for a paper it holds no passages "
              f"of -- the measured route to a claim resting on a source its "
              f"author never read.")
    if any(info[s]["examined"] for s in info):
        n_ex = len({d for s in info for d in info[s]["examined"]})
        print(f"  ({n_ex} document(s) under cross-examination this turn are "
              f"excluded from these numbers -- they are shared by construction)")
    (td / "preflight.json").write_text(json.dumps(pre.to_json(), indent=2),
                                       encoding="utf-8")
    return {"seats": info, "preflight": pre.to_json()}


def cmd_prepare(args) -> int:
    task = tasks.TASKS[args.task]
    out = paths.resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seats = load_seats(paths.resolve(args.specs), _seat_list(args.seats))
    material = ""
    if args.material:
        material = paths.resolve(args.material).read_text(encoding="utf-8")

    t = _build_turn(out, args.question, seats, paths.resolve(args.index), task,
                    max_passages=args.max_passages, own_floor=args.own_floor,
                    before_year=args.before_year, turn=1, material=material,
                    max_per_doc=args.max_per_doc, min_docs=args.min_docs,
                    neighbours=args.neighbours, max_seeds=args.max_seeds)

    (out / "manifest.json").write_text(json.dumps({
        "task": task.key,
        "question": args.question,
        "material": paths.relative(paths.resolve(args.material)) if args.material else "",
        "before_year": args.before_year,
        "seat_names": [s["name"] for s in seats],
        "specs": paths.relative(paths.resolve(args.specs)),
        "index": paths.relative(paths.resolve(args.index)),
        "max_passages": args.max_passages,
        # RECORDED SO TWO REGIMES ARE NEVER POOLED. The retrieval budget was 90
        # passages with no separate seed cap and no precision query until
        # 2026-09-13. A run made under one budget is not comparable with a run
        # made under the other, and a run whose manifest lacks `retrieval_regime`
        # predates the change.
        "max_seeds": args.max_seeds,
        "retrieval_regime": 2,
        "own_floor": args.own_floor,
        "max_per_doc": args.max_per_doc,
        "min_docs": args.min_docs,
        "neighbours": args.neighbours,
        "disclosure": not args.no_disclosure,
        # See `cmd_followup` for what this asserts and why older runs lack it.
        "leak_mask": True,
        "rungs_used": 0,
        "turns": {"1": {"question": args.question, **t}},
    }, indent=2), encoding="utf-8")
    _print_next(out, 1)
    return 0


def _claims_by_seat(run_dir: Path, prev: int) -> dict[str, list[dict]]:
    """Every seat's verified claims across ALL turns up to `prev`, in order.

    WHY THIS IS CUMULATIVE. This used to read only `turns/<prev>/ledger.json`,
    so the panel had a ONE-TURN MEMORY: at turn 3 a seat was shown what it said
    at turn 2 and nothing of what it filed at turn 1, under a header that says
    "your own record". For a three-turn run the damage is bounded and invisible.
    For a long run it is fatal in the most misleading way -- a panel that cannot
    see its own history cannot build on it, the trajectory flattens, and the
    flattening reads as the panel running out of things to say when it was the
    harness deleting the conversation.

    It is worse under the guardrail's DISTANT_PEERS and under seat-subset turns:
    a seat not called on last turn would next be told "(you filed no claims last
    turn)" -- its entire contribution erased because it was once not asked.

    Turn 2 is unchanged by this, because at turn 2 the only prior turn IS the
    last one. Only turn 3 onward differ.
    """
    by_seat: dict[str, list[dict]] = {}
    for tn in range(1, prev + 1):
        lp = turn_dir(run_dir, tn) / "ledger.json"
        if not lp.exists():
            continue
        for c in json.loads(lp.read_text(encoding="utf-8")):
            by_seat.setdefault(c["seat"], []).append({**c, "turn": tn})
    return by_seat


def cmd_followup(args) -> int:
    run_dir = _resolve_run(args.run_dir)
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    prev = latest_turn(run_dir)
    if prev == 0:
        raise SystemExit("no turns yet; run `prepare` first")
    ledger_path = turn_dir(run_dir, prev) / "ledger.json"
    if not ledger_path.exists():
        raise SystemExit(
            f"turn {prev} has not been ingested. Run `ingest` before `followup` "
            f"-- a follow-up must build on VERIFIED claims, not raw responses, "
            f"or an unverified claim becomes the premise of the next turn.")

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    by_seat = _claims_by_seat(run_dir, prev)

    names = _seat_list(args.seats) or man["seat_names"]

    # --- the guardrail decides what this turn looks like -------------------
    report = json.loads((turn_dir(run_dir, prev) / "report.json").read_text("utf-8"))
    assess = report.get("guardrail") or {}
    move = gr.Move(assess.get("move", "none"))
    exclude = set(assess.get("excluded", []))
    nudge = gr.nudge_block(move)
    restrict_own = move is gr.Move.REANCHOR
    isolate = move is gr.Move.ISOLATE
    anonymize = move is gr.Move.ANONYMIZE
    dissenter = assess.get("dissenter") or ""
    if move is gr.Move.ASSIGN_DISSENT and not dissenter:
        print("  (no evidenced minority seat was identifiable; the dissent rung "
              "cannot be applied and is skipped)")
        nudge = ""
    if move in (gr.Move.TERMINATE, gr.Move.DECLARE):
        raise SystemExit(
            f"The guardrail called {move.value.upper()} on the previous turn.\n"
            f"{gr.MOVE_DESCRIPTION[move]}\n"
            "Run `render` and read the positions as they stand. If you "
            "deliberately want another turn anyway, that is a choice you are "
            "making against the instrument -- record it in your lab notebook.")

    # Distant-peer routing: each seat sees only the seats furthest from it in
    # embedding space, instead of the whole table. [arXiv:2609.00683]
    peers_for: dict[str, set[str]] | None = None
    if move is gr.Move.DISTANT_PEERS:
        peers_for = _distant_peers(by_seat, k=max(2, len(names) // 3))

    prior: dict[str, str] = {}
    for name in names:
        own_lines: list[str] = []
        last_seen: int | None = None
        for c in by_seat.get(name, []):
            if c.get("turn") != last_seen:
                last_seen = c.get("turn")
                own_lines.append(f"\n*turn {last_seen}:*")
            mark = "" if c["grounded"] else "  [did not verify]"
            own_lines.append(f"- {c['text']}{mark}")
            for ci in c["citations"]:
                if ci["status"] == "verified":
                    tier = ci.get("tier", "read")
                    own_lines.append(f"    (you cited {ci['doc_id']} -- {tier})")
        other_lines: list[str] = []
        shown = 0
        for seat, cs in by_seat.items():
            if seat == name:
                continue
            if peers_for is not None and seat not in peers_for.get(name, set()):
                continue
            shown += 1
            label = f"seat {shown}" if anonymize else seat
            other_lines.append(f"**{label}** claimed:")
            shown_turn: int | None = None
            for c in cs:
                if not c["grounded"]:
                    continue
                if c.get("turn") != shown_turn:
                    shown_turn = c.get("turn")
                    other_lines.append(f"  *turn {shown_turn}:*")
                other_lines.append(f"  - {c['text']}")
            other_lines.append("")
        if anonymize:
            other_lines.insert(0, (
                "The seats below are unlabelled this turn, on purpose. Judge "
                "each claim on its content and against your own evidence, not "
                "on who made it.\n"))
        if peers_for is not None:
            other_lines.insert(0, (
                "You are being shown only the seats whose positions are "
                "furthest from yours this turn. The rest of the table has been "
                "withheld from your context deliberately.\n"))
        prior[name] = tasks.PRIOR_CONTEXT.format(
            turn=prev + 1,
            own="\n".join(own_lines) or "(you have filed no claims in this run)",
            others="\n".join(other_lines) or "(no other seat filed claims)")

    if move is not gr.Move.NONE:
        man["rungs_used"] = (man.get("rungs_used", 0) + 1
                             + int(assess.get("skipped_rungs", 0)))
        print(f"\nGUARDRAIL: applying {move.value}")
        print(gr.MOVE_DESCRIPTION[move])
        print("\nNOTE: this turn is PERTURBED. Interventions change the very "
              "inputs the heterogeneity measure reads -- withholding shared "
              "documents mechanically raises evidence separation, and "
              "restricting to own corpora mechanically raises it further. A "
              "recovery on the next turn is therefore NOT independent evidence "
              "that collapse was prevented. It is recorded in the transcript so "
              "it can be discounted.")
    if isolate:
        prior = {}

    task = tasks.TASKS[man.get("task", "roundtable")]
    seats = load_seats(paths.resolve(man["specs"]), names)

    # THE DISCLOSURE SET. Every document a seat cited and had verified last
    # turn, with the quotes it stood on, is handed to the other seats so the use
    # of it can be checked. Switched off when the guardrail has ISOLATED the
    # table -- isolation means no cross-talk of any kind, and a disclosure
    # packet is cross-talk.
    examine = _disclosure_set(ledger) if (man.get("disclosure", True)
                                          and not isolate) else {}
    if examine:
        # Spent in a deterministic, defensible order. See `_disclosure_order`.
        examine = {d: examine[d] for d in _disclosure_order(examine)}
        print(f"\nDISCLOSURE: {len(examine)} document(s) cited last turn are "
              f"handed to the other seats for checking. They are excluded from "
              f"the heterogeneity measures, because a document every seat is "
              f"given is shared by construction.")
        cap = retrieve_mod.DEFAULT_MAX_EXAMINE_DOCS
        if len(examine) > cap:
            dropped = list(examine)[cap:]
            print(f"  NOT ALL OF THEM FIT. The examination stratum carries at "
                  f"most {cap} documents per packet, so {len(dropped)} of them "
                  f"are not handed over: {', '.join(dropped)}.\n"
                  f"  Their ids are MASKED out of the replayed claims rather "
                  f"than left in the prose, so no seat is shown an id for a "
                  f"paper it holds no passages of. The panel's cross-check of "
                  f"those documents is correspondingly incomplete, and that is "
                  f"a real limit on this turn, not a formality.")

    t = _build_turn(
        run_dir, args.question, seats, paths.resolve(man["index"]), task,
        max_passages=man["max_passages"], own_floor=man.get("own_floor", 20),
        before_year=man["before_year"], turn=prev + 1,
        prior=prior, nudges=({dissenter: nudge} if nudge and dissenter else {}),
        exclude_docs=exclude if move is gr.Move.DIFFERENTIAL_RETRIEVAL else None,
        restrict_own=restrict_own,
        stances=True, examine=examine,
        max_per_doc=man.get("max_per_doc", 8), min_docs=man.get("min_docs", 10),
        neighbours=man.get("neighbours", 1),
        max_seeds=man.get("max_seeds", man["max_passages"]),
        # The original question, carried forward. Without it the retriever
        # steers by the moderator's grammar -- see `_build_turn`.
        retrieval_text=f"{man['question']}\n\n{args.question}")
    man["turns"][str(prev + 1)] = {
        "question": args.question, "applied_move": move.value, **t}
    # Marks this run as built with CUMULATIVE prior context. A run without the
    # key has packets from before 2026-09-13 and carries a one-turn memory; the
    # two must never be pooled, for the same reason `retrieval_regime` exists.
    man["cumulative_context"] = True
    # Marks this run as built with the replayed-prose LEAK MASK in place. A run
    # without the key has packets whose prior-context block may name a document
    # the reading seat was never handed -- see `_mask_unheld_ids`. The flag
    # exists so the harness can hold new runs to the invariant without claiming
    # that older runs met it; rewriting their packets to make a test pass would
    # be falsifying the record of what the agents were actually shown.
    man["leak_mask"] = True
    (run_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    _print_next(run_dir, prev + 1)
    return 0


#: What replaces a doc_id a seat was never handed. Deliberately not an ellipsis
#: or a silent deletion: the seat must be able to tell that its peer stood on
#: something, and that it cannot check what.
UNHELD_DOC_MASK = "[a document you were not given]"


def _permitted_universe(run_dir: Path) -> set[str]:
    """Every doc_id any seat has been permitted anywhere in this run so far.

    Read from the manifest rather than from the index, because the leak the
    mask exists to close is defined relative to what the PANEL has seen, not
    relative to what exists in the library. This is the same universe
    `tests/test_panel_harness.py::_leaks` builds, on purpose: a mask computed
    from a different universe than the check would drift out of agreement with
    it and the test would start passing for the wrong reason.
    """
    man_path = run_dir / "manifest.json"
    if not man_path.exists():
        return set()
    try:
        man = json.loads(man_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    out: set[str] = set()
    for t in (man.get("turns") or {}).values():
        for info in (t.get("seats") or {}).values():
            out |= set(info.get("permitted", []))
    return out


def _mask_unheld_ids(text: str, unheld: set[str]) -> tuple[str, set[str]]:
    """Replace doc_ids the reader holds no passages of with a neutral marker.

    Applied ONLY to the prior-context block -- other seats' replayed claims --
    and never to the seat's own evidence, which is by definition permitted.

    Matched at a token boundary, exactly as the harness matches, so `chen2021`
    does not fire inside `chen2021nature`. Longest ids first, so masking a short
    id cannot destroy a longer one that contains it before the longer one is
    tried.
    """
    if not text or not unheld:
        return text, set()
    hit: set[str] = set()
    for doc_id in sorted(unheld, key=len, reverse=True):
        text, n = re.subn(
            rf"(?<![A-Za-z0-9]){re.escape(doc_id)}(?![A-Za-z0-9])",
            UNHELD_DOC_MASK, text)
        if n:
            hit.add(doc_id)
    return text, hit


def _disclosure_order(examine: dict[str, dict]) -> list[str]:
    """The order disclosure spends its document budget in.

    WHY THIS EXISTS. The budget was spent in dict order -- insertion order of
    the ledger, i.e. whichever seat happened to be verified first -- and
    everything past the cap was dropped with no record. A run whose turn cited
    more documents than the cap therefore replayed every seat's claims while
    handing over an arbitrary subset of the documents those claims stood on.

    Contested documents first: a document two seats cited is a document the
    panel is actually arguing about, and it is worth more as a check than a
    document only one seat touched. Ties broken by doc_id so the order is
    reproducible across runs.
    """
    return sorted(examine, key=lambda d: (-len(examine[d]["seats"]), d))


def _disclosure_set(ledger: list[dict]) -> dict[str, dict]:
    """doc_id -> {seats that cited it, the quotes they used}.

    VERIFIED citations only. An invented or unmatched citation has nothing to
    examine: there is no passage to hand over, and asking a seat to check a
    quote that failed the exact-match test would be asking it to re-run a test
    the machine already ran and answered.
    """
    out: dict[str, dict] = {}
    for c in ledger:
        for ci in c.get("citations", []):
            if ci.get("status") != "verified":
                continue
            e = out.setdefault(ci["doc_id"], {"seats": [], "quotes": []})
            if c["seat"] not in e["seats"]:
                e["seats"].append(c["seat"])
            if ci.get("quote") and ci["quote"] not in e["quotes"]:
                e["quotes"].append(ci["quote"])
    return out


def _distant_peers(by_seat: dict[str, list[dict]], k: int = 3) -> dict[str, set[str]]:
    """For each seat, the k seats whose claims sit furthest from its own.

    Embedding-based peer selection: limiting an agent's context to its most
    semantically distant peers counters majority pull, because the majority is
    simply not in the context to be pulled toward (arXiv:2609.00683, VERIFIED).
    Falls back to showing everyone if there is nothing to embed -- a routing
    intervention that silently shows nothing would be an isolation intervention
    wearing the wrong label.

    SELF IS FILTERED BEFORE THE SLICE, not after. Similarities are sorted
    ascending and self-similarity is the largest, so it falls outside `order[:k]`
    and removing it afterwards left k+1 peers -- the intervention was
    consistently weaker than the number it reported.
    """
    names = [s for s, cs in by_seat.items() if any(c["grounded"] for c in cs)]
    if len(names) < 3:
        return {s: set(by_seat) - {s} for s in by_seat}
    texts = [" ".join(c["text"] for c in by_seat[s] if c["grounded"]) for s in names]
    vecs, _ = embed(texts)
    sim = vecs @ vecs.T
    out: dict[str, set[str]] = {}
    for i, s in enumerate(names):
        order = [j for j in sorted(range(len(names)), key=lambda j: sim[i, j])
                 if j != i]
        out[s] = {names[j] for j in order[:k]}
    for s in by_seat:
        out.setdefault(s, set(by_seat) - {s})
    return out


def _print_next(run_dir: Path, turn: int) -> None:
    td = turn_dir(run_dir, turn)
    print(f"\nwrote {td}/packets/*.md")
    print("Each seat sees ONLY its own packet. Give each packet to ONE agent --")
    print("one agent must never receive two packets; the isolation is the point.")
    print(f"Collect each reply as turns/{turn}/responses/<seat>.json, then run:")
    print(f"  python ops/panel.py ingest {run_dir.name}")
    print("After ingesting the LAST turn of a run, open its citations to the "
          "panel -- disclosure has no following turn to carry them:")
    print(f"  python ops/panel.py audit {run_dir.name}")


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------

def _citation_list(item: dict) -> list:
    """The citation list of a claim, under either of the two names models use.

    THE DEFECT THIS EXISTS FOR. A seat filed nine well-formed claims whose
    citation list was keyed `evidence` rather than `citations`. Every doc_id and
    quote was correct and every one of them would have verified; the parser
    looked for one key, found nothing, and reported *9 claims, 0 grounded* -- a
    seat that appeared to have grounded nothing at all.

    That is a silent failure: a mechanical accident indistinguishable,
    downstream, from a substantive result. Grounding is a claim about whether
    the SOURCE supports the text and it must not be decided by which synonym the
    model reached for. Verification itself is untouched and no less strict:
    every quote is still matched character-exact against the cited document.
    """
    for key in ("citations", "evidence"):
        v = item.get(key)
        if isinstance(v, list) and v:
            return v
    return []


def _parse(payload: object, seat: str):
    """Build Claims from a seat's JSON, tolerating shape but never inventing.

    A missing citation list produces a claim with no citations, reported as
    ungrounded. We never synthesise a citation to make a claim look supported.
    """
    claims: list[Claim] = []
    abstentions: list[dict] = []
    challenges: list[dict] = []
    stances: dict[str, str] = {}
    extras: dict = {}
    if not isinstance(payload, dict):
        return claims, abstentions, challenges, stances, extras

    for item in payload.get("claims", []) or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("claim") or item.get("text") or "").strip()
        if not text:
            continue
        cites = [Citation(doc_id=str(rc.get("doc_id", "")).strip(),
                          quote=str(rc.get("quote", "")).strip())
                 for rc in _citation_list(item) if isinstance(rc, dict)]
        claims.append(Claim(
            text=text, seat=seat, citations=cites,
            declared=item.get("declared") or {},
            claim_type=str(item.get("type", "descriptive")),
            territory=str(item.get("territory", ""))))
        # Captured but deliberately kept OUT of the verified claim object: an
        # audit artifact, not evidence. Models verbalise a hint they actually
        # used under 20% of the time, and larger models produce LESS faithful
        # reasoning. No gate consumes it.
        challenges.append({"claim": text,
                           "self_challenge": str(item.get("self_challenge", "")).strip()})

    for a in payload.get("abstentions", []) or []:
        if isinstance(a, dict):
            abstentions.append(a)

    for s in payload.get("stances", []) or []:
        if isinstance(s, dict) and s.get("seat"):
            key = f"{s['seat']}::{str(s.get('claim',''))[:80]}"
            stances[key] = str(s.get("stance", "")).strip() or "unstated"

    for k in ("outline", "slides", "plan", "files", "objections"):
        if payload.get(k):
            extras[k] = payload[k]
    return claims, abstentions, challenges, stances, extras


#: The verdicts a seat may return on another seat's citation. Anything else is
#: recorded verbatim as `unrecognised` rather than coerced into the nearest
#: valid value: a verdict nobody can interpret is a finding about the response,
#: and silently rounding it to `cannot_tell` would hide a malformed contract.
CHECK_VERDICTS = {"supports", "overstated", "misread", "irrelevant", "cannot_tell"}


def _parse_checks(payload: object, seat: str) -> list[dict]:
    """The disclosure returns: this seat's reading of other seats' citations."""
    out: list[dict] = []
    if not isinstance(payload, dict):
        return out
    for item in payload.get("citation_checks", []) or []:
        if not isinstance(item, dict) or not item.get("doc_id"):
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        out.append({
            "checker": seat,
            "doc_id": str(item["doc_id"]).strip(),
            "cited_by": str(item.get("cited_by", "")).strip(),
            "claim": str(item.get("claim", "")).strip(),
            "verdict": verdict if verdict in CHECK_VERDICTS else "unrecognised",
            "raw_verdict": verdict,
            "why": str(item.get("why", "")).strip(),
            "citations": [{"doc_id": str(rc.get("doc_id", "")).strip(),
                           "quote": str(rc.get("quote", "")).strip()}
                          for rc in _citation_list(item)
                          if isinstance(rc, dict)],
        })
    return out


def _ledger_to_observations(ledger: list[dict],
                            stances: dict[str, dict] | None = None,
                            examined: dict[str, set[str]] | None = None
                            ) -> list[SeatTurn]:
    """The ONE way a frozen ledger becomes a diversity observation.

    There used to be two: `cmd_ingest` built observations from the live claim
    objects and `_observations_of` rebuilt them from the ledger with a different
    rule for citations (grounded claims only, rather than every verified
    citation). A partially-verified claim therefore contributed to the evidence
    axis when a turn was first measured and vanished when the same turn was
    reconstructed for the drift comparison -- so the drift deltas were computed
    against a snapshot that had never existed. One function now, used by both.

    Rule: a claim's TEXT counts only if the claim is grounded; a CITATION counts
    if it verified, whoever it belongs to. Those are different questions --
    "what did this seat assert" and "what evidence is this seat standing on" --
    and they are answered differently on purpose.

    `examined` maps a seat to the documents it was handed for CHECKING rather
    than found by its own retrieval. Citations to those are dropped from the
    evidence axis: every seat was given them, so counting them would let the
    disclosure rule manufacture the overlap the guardrail reads as collapse.
    Dropped from the AXIS only -- the citation is verified, recorded, rendered
    and attributable exactly like any other.
    """
    examined = examined or {}
    by: dict[str, SeatTurn] = {}
    for c in ledger:
        t = by.setdefault(c["seat"], SeatTurn(seat=c["seat"]))
        if c["grounded"]:
            t.claim_texts.append(c["text"])
        skip = examined.get(c["seat"], set())
        t.cited_docs |= {ci["doc_id"] for ci in c["citations"]
                         if ci["status"] == "verified" and ci["doc_id"] not in skip}
    for seat, info in (stances or {}).items():
        if seat in by:
            by[seat].stances = info.get("stances", {}) or {}
    return list(by.values())


def cmd_ingest(args) -> int:
    run_dir = _resolve_run(args.run_dir)
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    n = args.turn or latest_turn(run_dir)
    tinfo = man["turns"][str(n)]
    td = turn_dir(run_dir, n)

    # IDEMPOTENCY. Ingest advances the sequential monitor, so running it twice
    # on one turn feeds the same observation into the monitor twice, and
    # re-ingesting an old turn feeds stale data into state that already contains
    # later turns. Both silently corrupt the guardrail's evidence. Re-ingesting
    # turn 1 would also replace the band, which is supposed to be fixed once.
    if (td / "report.json").exists() and not args.reingest:
        raise SystemExit(
            f"turn {n} has already been ingested and is frozen "
            f"(commit {json.loads((td / 'report.json').read_text('utf-8'))['commit'][:16]}).\n"
            f"Ingest advances the sequential monitor, so running it again would "
            f"count the same observation twice.\n"
            f"If the responses genuinely changed, pass --reingest, which discards "
            f"the monitor and re-derives it chronologically from turn 1.")
    if args.reingest:
        print(f"--reingest: discarding the sequential monitor and the band for "
              f"turns >= {n}. Everything downstream of this turn is now stale.")
        man.pop("monitor", None)
        if n == 1:
            man.pop("band", None)
        man["rungs_used"] = 0

    source = LibrarySource(paths.resolve(args.library))
    (run_dir / "seats").mkdir(exist_ok=True)

    all_claims: list[Claim] = []
    all_checks: list[dict] = []
    per_seat: dict[str, dict] = {}
    observations: list[SeatTurn] = []

    print(f"turn {n}: {tinfo['question']}\n")
    for seat, info in tinfo["seats"].items():
        resp = td / "responses" / f"{seat}.json"
        if not resp.exists():
            per_seat[seat] = {"status": "no response"}
            print(f"{seat:<26s} NO RESPONSE")
            continue
        try:
            # utf-8-sig, NOT utf-8. An agent writing its JSON through a Windows
            # editor or a PowerShell redirect emits a UTF-8 BOM, and strict
            # utf-8 turns that into `Unexpected UTF-8 BOM`, which this function
            # then records as UNPARSEABLE -- silently deleting a seat from the
            # turn, and from the next turn's prior context, which is built from
            # the ledger this writes. A byte-order mark is not a malformed
            # document; otherwise seats are silently dropped from the run.
            payload = json.loads(resp.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            # A parse failure is an ERROR, a different type from an abstention.
            per_seat[seat] = {"status": "unparseable", "error": str(exc)}
            print(f"{seat:<26s} UNPARSEABLE: {exc}")
            continue

        claims, abstentions, challenges, stances, extras = _parse(payload, seat)
        verifier = Verifier(source, permitted=set(info["permitted"]),
                            own=set(info.get("own", [])))
        for c in claims:
            verifier.verify_claim(c)
        report = Report(claims=claims)
        all_claims.extend(claims)

        # THE DISCLOSURE RETURNS. A check is only worth recording if the seat
        # was actually handed the document: a verdict on a paper nobody gave it
        # is a verdict on a memory of that paper, which is the thing this whole
        # system refuses.
        examined_here = set(info.get("examined", []))
        checks = _parse_checks(payload, seat)
        for chk in checks:
            if chk["doc_id"] not in examined_here:
                chk["status"] = "not_under_examination"
                continue
            chk["status"] = "recorded"
            for rc in chk["citations"]:
                c = Claim(text=chk["why"], seat=seat,
                          citations=[Citation(doc_id=rc["doc_id"],
                                              quote=rc["quote"])])
                verifier.verify_claim(c)
                rc["verified"] = bool(c.citations and c.citations[0].ok)
                rc["detail"] = c.citations[0].detail if c.citations else ""
            # A verdict that convicts without quoting is an opinion about a
            # paper rather than a reading of one, and is marked as such.
            if (chk["verdict"] not in ("supports", "cannot_tell")
                    and not any(rc.get("verified") for rc in chk["citations"])):
                chk["status"] = "unsupported_verdict"
        all_checks.extend(checks)
        adverse = [c for c in checks
                   if c["verdict"] in ("overstated", "misread", "irrelevant")
                   and c.get("status") == "recorded"]

        if extras.get("files"):
            _write_artifacts(run_dir, seat, extras["files"])

        per_seat[seat] = {
            "status": "ok", "n_claims": len(claims),
            "n_grounded": report.n_grounded,
            "fabrication_rate": round(report.fabrication_rate, 4),
            "ungrounded_citation_rate": round(report.ungrounded_citation_rate, 4),
            "invented": report.invented_count,
            "out_of_scope": report.out_of_scope_count,
            "self_anchor_rate": round(report.self_anchor_rate, 4),
            "territory": info.get("territory", "?"),
            "n_abstentions": len(abstentions), "abstentions": abstentions,
            # Kept so a past turn's observation can be rebuilt from the frozen
            # record alone -- the drift measure needs the previous turn's
            # stances, and re-deriving them from prose later would be a second
            # judgement about a thing already decided.
            "stances": stances,
            "checks_filed": len(checks),
            "checks_adverse": len(adverse),
            "extras": {k: v for k, v in extras.items() if k != "files"},
        }
        print(f"{seat:<26s} claims {len(claims):>3d}  grounded {report.n_grounded:>3d}"
              f"  fab {report.fabrication_rate:>5.1%}  inv {report.invented_count}"
              f"  oos {report.out_of_scope_count}"
              f"  own-anchored {report.self_anchor_rate:>4.0%}"
              f"  [{info.get('territory','?')}]")
        if examined_here and not checks:
            # Silence here is a contract failure, not agreement. It is reported
            # rather than tolerated, because a disclosure rule nobody answers is
            # a disclosure rule that does not exist.
            print(f"  NO CITATION CHECKS: {seat} was handed "
                  f"{len(examined_here)} document(s) under examination and "
                  f"filed no verdict on any of them")
        elif checks:
            bad = [c for c in checks if c.get("status") != "recorded"]
            print(f"  citation checks: {len(checks)} filed, {len(adverse)} adverse"
                  + (f", {len(bad)} not counted" if bad else ""))
            for c in adverse:
                print(f"    {c['verdict'].upper()}: {c['doc_id']} "
                      f"(cited by {c['cited_by'] or '?'}) -- {c['why'][:90]}")

        _flag_territory(seat, info, report)

        repairs = [repair_prompt(c) for c in claims if not c.grounded and c.citations]
        if repairs:
            (td / "responses" / f"{seat}.repair.md").write_text(
                "\n\n---\n\n".join(repairs), encoding="utf-8")

        observations.append(SeatTurn(
            seat=seat,
            claim_texts=[c.text for c in claims if c.grounded],
            # Documents handed over for CHECKING are excluded: every seat got
            # them, so counting them as this seat's evidence would report the
            # disclosure rule's own footprint as convergence.
            cited_docs={ci.doc_id for c in claims for ci in c.citations
                        if ci.ok and ci.doc_id not in examined_here},
            stances=stances))
        _append_notebook(run_dir / "seats" / f"{seat}.md", seat, n, info, claims,
                         report, abstentions, challenges)

    # --- freeze -----------------------------------------------------------
    payload = json.dumps([
        {"seat": c.seat, "text": c.text, "grounded": c.grounded,
         "type": c.claim_type, "territory": c.territory,
         "self_anchored": c.self_anchored, "declared": c.declared,
         "citations": [{"doc_id": ci.doc_id, "quote": ci.quote,
                        "status": ci.status.value, "tier": ci.tier.value,
                        "passage_id": ci.passage_id, "detail": ci.detail}
                       for ci in c.citations]}
        for c in all_claims], indent=2)
    (td / "ledger.json").write_text(payload, encoding="utf-8")
    commit = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # The disclosure record is frozen beside the ledger, not inside it: a check
    # is a judgement ABOUT a claim, and merging the two would make a seat's
    # verdict on somebody else's citation look like one of its own claims.
    (td / "checks.json").write_text(json.dumps(all_checks, indent=2),
                                    encoding="utf-8")
    if all_checks:
        counts: dict[str, int] = {}
        for c in all_checks:
            counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
        print("\nCITATION CHECKS -- what the table made of each other's evidence")
        print("  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        print("  These are JUDGEMENTS, not verifications. The quote was already "
              "checked mechanically; whether it supports the claim is exactly "
              "the part no machine here can decide.")

    # --- heterogeneity ----------------------------------------------------
    snap = measure(observations, turn=n, n_perm=args.permutations,
                   n_expected=len(tinfo["seats"]))
    print("\nHETEROGENEITY")
    print(snap.render())

    band_json = man.get("band")
    # MONITOR IDENTITY. The band is a statement about one panel on one question
    # measured with one embedding backend. A follow-up may legitimately change
    # the question and may restrict the roster, and comparing a statistic from
    # question B against a band derived from question A is meaningless. So the
    # identity is stored with the band and checked; when it changes, the band is
    # re-fixed from this turn and the monitor is discarded, LOUDLY.
    identity = {"seats": sorted(tinfo["seats"]), "backend": snap.backend}
    if band_json and band_json.get("identity") not in (None, identity):
        old = band_json.get("identity") or {}
        print("\nBAND INVALIDATED -- the panel this band described is not the "
              "panel that just answered.")
        if old.get("seats") != identity["seats"]:
            print(f"  roster was {old.get('seats')}\n  roster now {identity['seats']}")
        if old.get("backend") != identity["backend"]:
            print(f"  embedding backend was {old.get('backend')!r}, now "
                  f"{identity['backend']!r} -- these numbers are not comparable")
        print("  Re-fixing the band from THIS turn and discarding the sequential")
        print("  monitor. This turn was NOT generated in isolation, so the new")
        print("  reference is weaker than a real turn-1 reference. Treat the")
        print("  remainder of this run as a new run that happens to share a file.")
        band_json = None
        man.pop("monitor", None)
        man["rungs_used"] = 0

    if n == 1 or not band_json:
        from expertwins.diversity import Preflight
        pre = Preflight(**tinfo["preflight"])
        band = gr.set_band(pre, snap)
        band.identity = identity
        band.reference_turn = n
        man["band"] = band.to_json()
        print("\nBAND FIXED")
        print(band.render())
        d = None
    else:
        band = gr.Band(**band_json)
        prev_obs = _observations_of(run_dir, n - 1)
        prev_snap = measure(prev_obs, turn=n - 1, n_perm=args.permutations,
                            n_expected=len(man["turns"][str(n - 1)]["seats"]))
        d = drift(prev_obs, observations, prev_snap, snap)
        print(f"\nDRIFT  {d.render()}")

    monitor = gr.Sequential.from_json(man.get("monitor"))
    a = gr.assess(
        n, snap, band, observations, drift=d,
        rungs_used=man.get("rungs_used", 0), monitor=monitor,
        self_anchor={s: i.get("self_anchor_rate", 0.0)
                     for s, i in per_seat.items() if i.get("status") == "ok"},
        territories={s: i.get("territory", "") for s, i in per_seat.items()})
    man["monitor"] = a.monitor
    print("\nGUARDRAIL")
    print(a.render())

    (td / "report.json").write_text(json.dumps(
        {"turn": n, "question": tinfo["question"], "commit": commit,
         "seats": per_seat, "diversity": snap.to_json(),
         "drift": (None if d is None else
                   {"novelty": d.novelty, "herding": d.herding,
                    "delta_separation": d.delta_separation}),
         "guardrail": a.to_json()}, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")

    total = len(all_claims)
    grounded = sum(1 for c in all_claims if c.grounded)
    print(f"\nTOTAL {total} claims, {grounded} grounded"
          + (f" ({grounded/total:.0%})" if total else ""))
    print(f"commit {commit[:16]}  (turn {n} frozen; a later turn may attack a "
          f"claim but never delete it)")
    print("\nNext: render, or ask a follow-up:")
    print(f"  python ops/panel.py render {run_dir.name}")
    print(f'  python ops/panel.py followup {run_dir.name} "your next question"')
    return 0


def _flag_territory(seat: str, info: dict, report: Report) -> None:
    """The fidelity warning. Not a gate -- a flag, and a loud one.

    A seat on its HOME territory that anchored nothing in its own papers has
    stopped being that person and become a competent generalist wearing the
    name. That is the exact failure this project exists to detect, and it is
    invisible in a transcript that reads well.
    """
    if info.get("territory") == identity.HOME and report.n_grounded:
        if report.self_anchor_rate < 0.25:
            print(f"  FIDELITY WARNING: {seat} is on its own territory "
                  f"({info.get('own_fraction', 0):.0%} of its evidence is its own "
                  f"work) but only {report.self_anchor_rate:.0%} of its grounded "
                  f"claims rest on a paper it wrote. This reads as a generalist "
                  f"wearing the name.")


def _write_artifacts(run_dir: Path, seat: str, files: list) -> None:
    """Write a seat's code/artifacts under the run, never outside it."""
    base = (run_dir / "work" / seat).resolve()
    base.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not isinstance(f, dict) or not f.get("path"):
            continue
        target = (base / str(f["path"])).resolve()
        if not str(target).startswith(str(base)):
            # A path that escapes the run directory is refused, not sanitised.
            print(f"  REFUSED artifact outside the run: {f['path']}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(f.get("content", "")), encoding="utf-8")
        print(f"  wrote {paths.relative(target)}")


def _observations_of(run_dir: Path, n: int) -> list[SeatTurn]:
    """Rebuild a past turn's observation from its frozen ledger and report."""
    lp = turn_dir(run_dir, n) / "ledger.json"
    if not lp.exists():
        return []
    ledger = json.loads(lp.read_text(encoding="utf-8"))
    rp = turn_dir(run_dir, n) / "report.json"
    stances = (json.loads(rp.read_text(encoding="utf-8")).get("seats", {})
               if rp.exists() else {})
    mp = run_dir / "manifest.json"
    examined: dict[str, set[str]] = {}
    if mp.exists():
        turns = json.loads(mp.read_text(encoding="utf-8")).get("turns", {})
        for seat, info in (turns.get(str(n), {}).get("seats", {}) or {}).items():
            examined[seat] = set(info.get("examined", []))
    return _ledger_to_observations(ledger, stances, examined)


def _append_notebook(path: Path, seat: str, turn: int, info: dict, claims,
                     report, abstentions, challenges) -> None:
    """The per-seat lab notebook. Re-ingesting a turn REPLACES its block.

    THE HEADER IS NOT DECORATION. Stated reasoning is frequently not real
    reasoning, so this file is an audit and debugging artifact and NOTHING in
    the pipeline consumes it. If a future version starts gating on it, that
    header is the record of why it must not.

    WHY THIS IS NOT A PLAIN APPEND. `ingest --reingest` is supported and
    deterministic. If re-ingest appended unconditionally, every re-ingest would
    add another copy of the same turn. Nothing would fail, because no code reads
    this file -- and an audit artifact that silently duplicates itself is worse
    than none, since a reader cannot tell repeated records from distinct turns.

    Replacing is also the semantics the rest of the system already has: a turn
    is frozen, and re-deriving it must reproduce it, not grow it.
    """
    head = [] if path.exists() else [
        f"# {seat} -- internal record", "",
        "> **This is an audit artifact, not evidence of correctness.**",
        "> Models verbalise a hint they actually used under 20% of the time,",
        "> and larger models produce *less* faithful reasoning. Read this to",
        "> understand what the seat did. Do not accept a claim because the",
        "> reasoning here reads well -- trust flows only from the exact-match",
        "> citation check, permitted-set membership, and the own/read tier.", "",
    ]
    body = [
        f"## Turn {turn}", "",
        f"Evidence: {len(info['permitted'])} documents permitted, "
        f"{len(info.get('own', []))} of them this seat's own "
        f"({info.get('own_fraction', 0):.0%}) -- territory **{info.get('territory','?')}**.",
        f"Claims filed: {len(claims)}; grounded {report.n_grounded}; "
        f"own-anchored {report.self_anchor_rate:.0%}.", "",
        "### Self-challenge, per claim", "",
    ]
    for sc in challenges:
        body += [f"**{sc['claim']}**", "",
                 sc["self_challenge"] or "*(no self-challenge recorded)*", ""]
    if abstentions:
        body += ["### What this seat declined to answer", ""]
        for a in abstentions:
            body.append(f"- **{a.get('cause','?')}** -- {a.get('question_part','')}: "
                        f"{a.get('detail','')}")
        body.append("")
    ung = [c for c in claims if not c.grounded]
    if ung:
        body += ["### Claims that did NOT survive verification", ""]
        for c in ung:
            body.append(f"- {c.text}")
            for ci in c.citations:
                if not ci.ok:
                    body.append(f"  - `{ci.doc_id}` **{ci.status.value}** -- {ci.detail}")
        body.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = f"## Turn {turn}"
    if not path.exists():
        path.write_text("\n".join(head + body) + "\n", encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(body) + "\n")
        return
    # Keep everything that is not a block for THIS turn. The lookahead split
    # keeps each heading attached to its own block, and there may be several
    # blocks for this turn left over from before this fix -- all are dropped.
    parts = re.split(r"(?m)^(?=## Turn \d+$)", text)
    kept = [p for p in parts if not p.startswith(marker + "\n")]
    path.write_text("".join(kept).rstrip("\n") + "\n\n" + "\n".join(body) + "\n",
                    encoding="utf-8")


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def cmd_render(args) -> int:
    run_dir = _resolve_run(args.run_dir)
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    L = [
        f"# {man.get('task','roundtable').title()} -- {man['question']}", "",
        "**This document is the product.** It is not a synthesised answer. It is "
        "attributed disagreement for a human to adjudicate. Every quote below "
        "has been checked by exact string match against the specific document "
        "cited, every citation against the permitted set of the seat that made "
        "it, and every verified citation tagged `own` (the seat wrote that "
        "paper) or `read` (it did not).", "",
    ]
    cited_by: dict[str, list[str]] = {}
    het_rows: list[str] = []

    for n in sorted(int(k) for k in man["turns"]):
        td = turn_dir(run_dir, n)
        lp, rp = td / "ledger.json", td / "report.json"
        if not lp.exists() or not rp.exists():
            continue
        ledger = json.loads(lp.read_text(encoding="utf-8"))
        rep = json.loads(rp.read_text(encoding="utf-8"))
        applied = man["turns"][str(n)].get("applied_move", "none")
        L += [f"## Turn {n} -- {rep['question']}", "",
              f"Commit `{rep['commit'][:16]}`."
              + ("" if applied in ("none", "") else
                 f" **A heterogeneity intervention was applied to this turn: "
                 f"`{applied}`.** {gr.MOVE_DESCRIPTION[gr.Move(applied)]}"), ""]

        by_seat: dict[str, list[dict]] = {}
        for c in ledger:
            by_seat.setdefault(c["seat"], []).append(c)
            for ci in c["citations"]:
                if ci["status"] == "verified":
                    cited_by.setdefault(ci["doc_id"], [])
                    if c["seat"] not in cited_by[ci["doc_id"]]:
                        cited_by[ci["doc_id"]].append(c["seat"])

        for seat, claims in by_seat.items():
            i = rep["seats"].get(seat, {})
            L += [f"### {seat}",
                  f"*{i.get('n_claims',0)} claims, {i.get('n_grounded',0)} grounded, "
                  f"{i.get('self_anchor_rate',0):.0%} resting on its own work; "
                  f"territory **{i.get('territory','?')}**.*", ""]
            for c in claims:
                mark = "" if c["grounded"] else " **[UNGROUNDED]**"
                L.append(f"- {c['text']}{mark}")
                for ci in c["citations"]:
                    if ci["status"] == "verified":
                        tag = "**own**" if ci.get("tier") == "own" else "read"
                        L.append(f"  - `{ci['doc_id']}` ({tag}): \"{ci['quote'][:150]}\"")
                    else:
                        L.append(f"  - `{ci['doc_id']}`: **{ci['status']}** -- {ci['detail']}")
            for a in i.get("abstentions", []) or []:
                L.append(f"- *abstained* ({a.get('cause','?')}): {a.get('detail','')}")
            L.append("")

        g = rep.get("guardrail", {})
        d = rep.get("diversity", {})
        het_rows.append(
            f"| {n} | {d.get('axes',{}).get('evidence',{}).get('separation',0):.2f} "
            f"| {d.get('axes',{}).get('lexical',{}).get('separation',0):.2f} "
            f"| {d.get('axes',{}).get('stance',{}).get('separation',0):.2f} "
            f"| {d.get('effective_positions',0):.2f} "
            f"| {d.get('vendi2',0):.2f}/{d.get('vendi_max',0)} "
            f"| {g.get('verdict','?')} | {g.get('move','none')} |")

    L += ["## Heterogeneity", "",
          "Separation is chance-corrected: 0 means this panel is "
          "indistinguishable from a single agent talking to itself, 1 means the "
          "seats are maximally separated. The floor was fixed from turn 1, which "
          "was generated in isolation and is therefore this panel's free "
          "diversity on this question.", ""]
    if man.get("band"):
        f = man["band"]
        L += [f"Band: rho_low `{f['rho_low']:.2f}`, rho_high "
              f"`{f['rho_high']:.2f}` of turn {f['reference_turn']} -- floors "
              f"evidence `{f['evidence']:.3f}`, positions "
              f"`{f['positions']:.3f}`; ceiling on positions "
              f"`{f['ceiling_positions']:.3f}`.", "", f"> {f['source']}", "",
              "**A turn marked with an applied move was PERTURBED.** The "
              "interventions change the inputs this measure reads -- withholding "
              "shared documents mechanically raises evidence separation -- so a "
              "recovery immediately after one is not independent evidence that "
              "collapse was prevented.", ""]
    L += ["| turn | evidence | lexical | stance | positions | VS2 | verdict | move |",
          "|---|---|---|---|---|---|---|---|", *het_rows, ""]

    L += ["## Contested evidence", "",
          "Documents cited by more than one seat. A **deterministic** locus of "
          "possible disagreement -- the seats are arguing over the same text. "
          "Whether they actually contradict each other is a semantic judgment "
          "and is left to you; this system does not adjudicate meaning and "
          "should not pretend to.", ""]
    contested = {d: s for d, s in cited_by.items() if len(s) > 1}
    if contested:
        for doc, seats in sorted(contested.items(), key=lambda kv: -len(kv[1])):
            L.append(f"- `{doc}` -- cited by {', '.join(seats)}")
    else:
        L.append("*No document was cited by more than one seat. The seats did "
                 "not meet on any single piece of evidence, which is itself a "
                 "finding: they may be talking past each other rather than "
                 "disagreeing.*")

    L += ["", "## Citation checks -- what the seats made of each other's evidence",
          "",
          "Every document a seat cited was handed to the other seats -- on the "
          "following turn through the disclosure stratum of their packet, and, "
          "where `audit` was run, in a standalone review pass that asked for "
          "nothing but the reading. The quoted passage is marked in both, and "
          "each seat was asked whether the use made of it holds. The quote "
          "itself was already checked by exact match; **this section is the "
          "part no machine here can decide.** A verdict is a scientist's "
          "reading, not a verification, and `cannot_tell` from someone outside "
          "the field is a correct answer rather than a missing one.", ""]
    checks_all = _all_checks(run_dir, man)
    adverse = [c for c in checks_all
               if c["verdict"] in ("overstated", "misread", "irrelevant")
               and c.get("status") == "recorded"]
    if not checks_all:
        L.append("*No citation was placed under examination -- either this run "
                 "is a single turn on which no `audit` was run, or no citation "
                 "verified on the turn before.*")
        L.append("")
        L.append("*The last turn of a run is never covered by disclosure alone: "
                 "there is no following turn to carry it. Run "
                 "`python ops/panel.py audit <run>` to open its citations to "
                 "the panel.*")
    else:
        n_audit = sum(1 for c in checks_all if c.get("origin") == "audit")
        L.append(f"{len(checks_all)} verdicts filed ({n_audit} from a standalone "
                 f"audit pass), {len(adverse)} adverse.")
        L.append("")
        for c in adverse:
            L.append(f"- **{c['verdict']}** -- `{c['doc_id']}`, cited by "
                     f"`{c['cited_by'] or '?'}`, challenged by `{c['checker']}`"
                     f"{' (audit)' if c.get('origin') == 'audit' else ''}: "
                     f"{c['why']}")
            for rc in c.get("citations", []):
                if rc.get("verified"):
                    L.append(f"  - quoted back: \"{rc['quote'][:150]}\"")
        if not adverse:
            L.append("*No seat found another seat's use of a document "
                     "overstated, misread or irrelevant. That is either a "
                     "well-cited table or an incurious one, and the two look "
                     "identical from here.*")
        unsupported = [c for c in checks_all
                       if c.get("status") == "unsupported_verdict"]
        if unsupported:
            L += ["", f"{len(unsupported)} adverse verdict(s) were filed without "
                  f"a verifying quote and are NOT counted above. An objection to "
                  f"a reading that does not quote the passage is an opinion "
                  f"about a paper rather than a reading of one.", ""]
            for c in unsupported:
                L.append(f"- `{c['doc_id']}` -- {c['checker']} said "
                         f"*{c['verdict']}*: {c['why']}")

    L += ["", "## What the panel would not commit to", "",
          "Ungrounded claims are shown above rather than deleted, marked as "
          "such. A claim whose citation failed is not evidence; it is a record "
          "of what a seat believed and could not support.", "",
          "Per-seat internal records -- self-challenge, abstentions, withdrawn "
          "claims -- are in `seats/`. **Those are audit artifacts, not "
          "evidence.**", ""]

    (run_dir / "transcript.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {run_dir}/transcript.md")
    return 0


def _all_checks(run_dir: Path, man: dict) -> list[dict]:
    """Every citation check filed across the run, in turn order.

    Two sources, and they are different instruments. `checks.json` holds the
    verdicts seats returned IN THE COURSE OF ANSWERING the next question, from
    the disclosure stratum of their packet. `audit/checks.json` holds the
    verdicts of a standalone review pass, where the seat was asked for nothing
    except a reading of somebody else's source. Both are judgements about the
    same kind of object, so they are reported together and labelled by origin.
    """
    out: list[dict] = []
    for n in sorted(int(k) for k in man.get("turns", {})):
        for cp, origin in ((turn_dir(run_dir, n) / "checks.json", "disclosure"),
                           (turn_dir(run_dir, n) / "audit" / "checks.json", "audit")):
            if cp.exists():
                for c in json.loads(cp.read_text(encoding="utf-8")):
                    out.append({**c, "turn": n, "origin": c.get("origin", origin)})
    return out


AUDIT_INSTRUCTIONS = """\
You are one seat at a scientific round table, and this is a CITATION AUDIT.

You are not being asked for your own view of the question. You are being asked
to do what a referee does: read the actual source another expert cited, and say
whether it supports what they claimed from it.

{rubric}THE CLAIMS UNDER REVIEW

Below are claims filed by OTHER seats. For each one you are given the claim, the
seat that made it, the document it cites, the exact quote it relied on -- which
has already been machine-verified to appear in that document, so the quote is
certainly real -- and the surrounding passages of that document.

A verified quote is not a verified claim. The quote is real by construction;
whether the claim it is attached to follows from the source is a scientific
judgement, and it is the judgement no machine in this system can make. That is
why you are reading it.

{cases}

WHAT TO PRODUCE

A JSON object: {{"citation_checks": [...]}}

Each entry:
{{
  "doc_id":    "the document you assessed",
  "cited_by":  "the seat that cited it",
  "claim":     "the claim text, copied so it can be matched",
  "verdict":   "supports | overstated | misread | irrelevant | cannot_tell",
  "why":       "one or two sentences. If the verdict is anything other than
                `supports`, point at the specific words that show it.",
  "citations": [{{"doc_id": "...", "quote": "an EXACT quote from the passages
                 shown, copied character for character"}}]
}}

RULES

1. Judge only against the passages shown. If the surrounding context is not
   enough to decide, the verdict is `cannot_tell` and you should say what you
   would have needed. Guessing is worse than abstaining.
2. `supports` is a real verdict and you must use it when it is true. An audit
   that finds fault everywhere is as useless as one that finds fault nowhere.
3. AN ADVERSE VERDICT MUST QUOTE. A verdict of `overstated`, `misread` or
   `irrelevant` that carries no verifying quote is an opinion about a paper
   rather than a reading of one, and is recorded as `unsupported_verdict`
   rather than counted.
4. You may not make claims of your own here, and nothing you read in this audit
   becomes evidence you may cite elsewhere. You were handed these documents to
   referee a specific claim, not as a grant of new evidence. This is the rule
   that keeps the audit from becoming a way of acquiring other seats' sources.
5. Your own field's standards apply. If the claim is outside your competence,
   say `cannot_tell` rather than deferring politely.
"""


def _audit_cases(ledger: list[dict], reviewer: str, source: LibrarySource,
                 radius: int) -> tuple[str, set[str]]:
    """Other seats' claims, each with its source in context. Also the doc ids.

    The doc ids come back because `audit-ingest` needs to know what this
    reviewer was actually shown: a verdict on a paper nobody handed it is a
    verdict on a memory of that paper, which is the thing this system refuses.
    """
    out: list[str] = []
    shown: set[str] = set()
    n = 0
    for c in ledger:
        if c["seat"] == reviewer or not c.get("grounded"):
            continue
        for ci in c.get("citations", []):
            if ci.get("status") != "verified":
                continue
            passages = source.passages(ci["doc_id"])
            idx = next((i for i, (pid, _) in enumerate(passages)
                        if pid == ci.get("passage_id")), None)
            if idx is None:
                idx = next((i for i, (_, txt) in enumerate(passages)
                            if ci.get("quote") and ci["quote"] in txt), None)
            if idx is None:
                continue
            n += 1
            shown.add(ci["doc_id"])
            lo, hi = max(0, idx - radius), min(len(passages), idx + radius + 1)
            out.append(f"--- case {n} ---")
            out.append(f"CLAIM ({c['seat']}): {c['text']}")
            out.append(f"CITES: {ci['doc_id']}")
            out.append(f"QUOTE RELIED ON: \"{ci.get('quote','')}\"")
            out.append(f"THE SOURCE, passages {lo}-{hi - 1} of {len(passages)}:")
            for i in range(lo, hi):
                mark = "  >>> " if i == idx else "      "
                out.append(f"{mark}{passages[i][1]}")
            out.append("")
    body = "\n".join(out) if out else (
        "(no other seat filed a grounded, verified citation this turn)")
    return body, shown


def cmd_audit(args) -> int:
    """Open every document cited in a turn to the whole panel for review.

    WHY THIS EXISTS WHEN THE DISCLOSURE STRATUM ALREADY DOES THIS. Disclosure
    reaches a seat on the FOLLOWING turn, inside the packet it answers the next
    question from. Two things follow. The citations of the LAST turn of a run
    are never checked by anybody, which is precisely the turn a reader quotes.
    And a check made while answering a new question competes for the seat's
    attention with the question. This pass asks for nothing but the reading.

    WHY IT DOES NOT BREAK THE ISOLATION INVARIANT, which is the first thing
    anyone reading this will worry about. No seat may see another seat's
    evidence WHILE FORMING ITS OWN CLAIMS, because an agent that can see a
    peer's doc_id will cite it without having been shown it. That failure is
    about PROVENANCE. This is the opposite operation: the turn is already frozen
    and hashed, the claims are already filed, and nothing here can change them.
    What the reviewer is handed is not an evidence grant but a specific
    accusation to test, and rule 4 of the prompt says so.

    Audit packets are written to `turns/<n>/audit/`, never to
    `turns/<n>/packets/`, and the manifest's permitted sets are untouched, so
    the out-of-scope check still means what it meant.
    """
    run_dir = _resolve_run(args.run_dir)
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    n = args.turn or latest_turn(run_dir)
    lp = turn_dir(run_dir, n) / "ledger.json"
    if not lp.exists():
        raise SystemExit(
            f"turn {n} has not been ingested. An audit reviews VERIFIED claims; "
            f"run `ingest` first.")
    ledger = json.loads(lp.read_text(encoding="utf-8"))
    seats = load_seats(paths.resolve(man["specs"]),
                       _seat_list(args.seats) or man["seat_names"])
    source = LibrarySource(paths.resolve(args.library))

    ad = turn_dir(run_dir, n) / "audit"
    (ad / "packets").mkdir(parents=True, exist_ok=True)
    (ad / "responses").mkdir(parents=True, exist_ok=True)

    print(f"turn {n}: opening every cited document to the panel for review\n")
    print(f"{'reviewer':<26s} {'cases':>6s} {'docs':>6s}")
    print("-" * 42)
    shown_by: dict[str, list[str]] = {}
    for spec in seats:
        name = spec["name"]
        cases, docs = _audit_cases(ledger, name, source, args.radius)
        rubric = spec.get("fatal_flaws", "")
        block = ("WHAT YOU REFUSE TO ACCEPT\n\n"
                 "This is your acceptance rubric: the standard below which you "
                 "do not agree, including with yourself. It grants you no "
                 "facts.\n\n" + rubric.strip() + "\n\n") if rubric else ""
        (ad / "packets" / f"{name}.md").write_text(
            AUDIT_INSTRUCTIONS.format(rubric=block, cases=cases),
            encoding="utf-8")
        shown_by[name] = sorted(docs)
        print(f"{name:<26s} {cases.count('--- case '):>6d} {len(docs):>6d}")

    (ad / "shown.json").write_text(json.dumps(shown_by, indent=2),
                                   encoding="utf-8")
    print(f"\nwrote {ad}/packets/*.md")
    print("Give each to one agent; collect JSON into "
          f"turns/{n}/audit/responses/<seat>.json, then run:")
    print(f"  python ops/panel.py audit-ingest {run_dir.name} --turn {n}")
    return 0


def cmd_audit_ingest(args) -> int:
    """Verify and record the audit's verdicts beside the turn they review.

    The quotes a reviewer returns are verified exactly as a claim's are, against
    the documents that reviewer was SHOWN -- `shown.json`, written by `audit`,
    is the permitted set for this pass. A verdict on a document the reviewer was
    never handed is recorded as `not_under_examination` and not counted, for the
    same reason the disclosure returns are filtered that way.
    """
    run_dir = _resolve_run(args.run_dir)
    n = args.turn or latest_turn(run_dir)
    ad = turn_dir(run_dir, n) / "audit"
    shown_path = ad / "shown.json"
    if not shown_path.exists():
        raise SystemExit(f"no audit was prepared for turn {n}; run `audit` first")
    shown_by = json.loads(shown_path.read_text(encoding="utf-8"))
    source = LibrarySource(paths.resolve(args.library))

    out: list[dict] = []
    print(f"turn {n}: audit verdicts\n")
    for seat, docs in shown_by.items():
        resp = ad / "responses" / f"{seat}.json"
        if not resp.exists():
            print(f"{seat:<26s} NO RESPONSE")
            continue
        try:
            payload = json.loads(resp.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            print(f"{seat:<26s} UNPARSEABLE: {exc}")
            continue
        checks = _parse_checks(payload, seat)
        verifier = Verifier(source, permitted=set(docs), own=set())
        for chk in checks:
            chk["origin"] = "audit"
            if chk["doc_id"] not in set(docs):
                chk["status"] = "not_under_examination"
                continue
            chk["status"] = "recorded"
            for rc in chk["citations"]:
                c = Claim(text=chk["why"], seat=seat,
                          citations=[Citation(doc_id=rc["doc_id"],
                                              quote=rc["quote"])])
                verifier.verify_claim(c)
                rc["verified"] = bool(c.citations and c.citations[0].ok)
                rc["detail"] = c.citations[0].detail if c.citations else ""
            if (chk["verdict"] not in ("supports", "cannot_tell")
                    and not any(rc.get("verified") for rc in chk["citations"])):
                chk["status"] = "unsupported_verdict"
        out.extend(checks)
        counts: dict[str, int] = {}
        for c in checks:
            counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
        print(f"{seat:<26s} " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    (ad / "checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {ad}/checks.json ({len(out)} verdicts)")
    print("These are JUDGEMENTS, not verifications. Run `render` to fold them "
          "into the transcript.")
    return 0


def _read_question(inline: str | None, path: str | None) -> str:
    """The question, from a file or from the command line.

    WHY A FILE IS THE PREFERRED PATH. Quoting and encoding failures have the
    same shape: a question is typed at a shell, the shell mangles it, and the
    mangling is not visible until much later. An
    em-dash arriving in a manifest as mojibake; a PowerShell round-trip
    corrupting a document; and the dangerous one -- a question containing a
    double-quoted phrase SPLIT BY THE SHELL into several arguments, so argparse
    rejected the remainder, the turn was never built, and the next command in
    the sequence ran anyway. A partially-applied turn is worse than a failed
    one.

    A file argument has no quoting rules, no argument splitting and an explicit
    encoding, so the whole class disappears. It is read utf-8-sig because a
    byte-order mark is not a malformed document.
    """
    if path and inline:
        raise SystemExit("give either a question or --question-file, not both")
    if path:
        p = paths.resolve(path)
        if not p.exists():
            raise SystemExit(f"no such question file: {p}")
        text = p.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise SystemExit(f"question file is empty: {p}")
        return text
    if not inline:
        raise SystemExit("a question is required: give it inline or with "
                         "--question-file")
    return inline


def _resolve_run(name: str) -> Path:
    for p in (paths.resolve(name), paths.RUNS / name):
        if p.exists():
            return p
    raise SystemExit(f"no such run: {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="start a run (turn 1)")
    p.add_argument("question", nargs="?", default=None)
    p.add_argument("--question-file", default=None,
                   help="read the question from a file instead of the command "
                        "line; ALWAYS prefer this for anything non-trivial")
    p.add_argument("--out", required=True)
    p.add_argument("--task", default="roundtable", choices=sorted(tasks.TASKS))
    p.add_argument("--seats", default=None, help="comma-separated subset")
    p.add_argument("--specs", default=paths.relative(paths.SEATS))
    p.add_argument("--index", default=paths.relative(paths.INDEX))
    p.add_argument("--material", default=None,
                   help="file of material to critique / the data+environment block")
    p.add_argument("--max-passages", type=int, default=260,
                   help="total passages per seat, AFTER context expansion")
    p.add_argument("--max-seeds", type=int, default=200,
                   help="ranked hits per seat, BEFORE context expansion: how "
                        "many places in the corpus the seat is pointed at")
    p.add_argument("--own-floor", type=int, default=60,
                   help="passages reserved for the seat's OWN papers")
    p.add_argument("--max-per-doc", type=int, default=8,
                   help="cap on passages taken from any one document, so a "
                        "single verbose review cannot become the packet. 0 = off")
    p.add_argument("--min-docs", type=int, default=10,
                   help="run a breadth pass until this many distinct documents "
                        "are represented. 0 = off")
    p.add_argument("--neighbours", type=int, default=1,
                   help="passages of context either side of each hit, in source "
                        "order, so a quote is read inside its paragraph. 0 = off")
    p.add_argument("--no-disclosure", action="store_true",
                   help="do NOT hand each seat the documents the others cited. "
                        "Turns off the cross-examination step for this run.")
    p.add_argument("--before-year", type=int, default=None,
                   help="temporal cut: seats see only papers published before this")

    f = sub.add_parser("followup", help="ask a further question")
    f.add_argument("run_dir")
    f.add_argument("question", nargs="?", default=None)
    f.add_argument("--question-file", default=None,
                   help="read the question from a file instead of the command "
                        "line; ALWAYS prefer this for anything non-trivial")
    f.add_argument("--seats", default=None)

    i = sub.add_parser("ingest", help="verify a turn and measure heterogeneity")
    i.add_argument("run_dir")
    i.add_argument("--library", default=paths.relative(paths.LIBRARY))
    i.add_argument("--turn", type=int, default=None)
    i.add_argument("--permutations", type=int, default=200)
    i.add_argument("--reingest", action="store_true",
                   help="re-ingest an already-frozen turn. Discards the "
                        "sequential monitor, because feeding it the same "
                        "observation twice corrupts the guardrail's evidence.")

    r = sub.add_parser("render", help="write the transcript")
    r.add_argument("run_dir")

    au = sub.add_parser(
        "audit", help="open every document cited in a turn to the whole panel")
    au.add_argument("run_dir")
    au.add_argument("--turn", type=int, default=None)
    au.add_argument("--seats", default=None,
                    help="restrict the reviewers to a subset of the panel")
    au.add_argument("--radius", type=int, default=4,
                    help="passages of context shown around each cited quote")
    au.add_argument("--library", default=paths.relative(paths.LIBRARY))

    ai = sub.add_parser("audit-ingest",
                        help="verify and record the audit's verdicts")
    ai.add_argument("run_dir")
    ai.add_argument("--turn", type=int, default=None)
    ai.add_argument("--library", default=paths.relative(paths.LIBRARY))

    a = ap.parse_args()
    if a.cmd in ("prepare", "followup"):
        a.question = _read_question(a.question, a.question_file)
    return {"prepare": cmd_prepare, "followup": cmd_followup,
            "ingest": cmd_ingest, "render": cmd_render,
            "audit": cmd_audit, "audit-ingest": cmd_audit_ingest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

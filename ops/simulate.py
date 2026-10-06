"""A deterministic mock seat, for testing the machinery without spending a model.

THIS IS A TEST FIXTURE AND IT IS NOT AN AGENT. It reads a packet, lifts genuine
quotes out of it, and emits well-formed JSON. It has no opinions, so nothing it
produces says anything about the science. What it does say something about is
the PLUMBING: that quotes verify, that permitted-set membership holds, that
tiers are assigned, that the heterogeneity measure moves in the right direction,
and that the guardrail fires when it should and stays quiet when it should not.

The three modes exist because the guardrail has to be tested in both directions,
and a guardrail that has only ever been tested on the failure it was built for
is not tested at all:

    distinct   each seat cites only from its own packet's private documents and
               writes in its own vocabulary -> a healthy table
    collapse   every seat converges on the documents the whole table shares and
               writes near-identical sentences -> the failure mode
    scatter    every seat emits unrelated fragments -> the OPPOSITE failure,
               which must not be reported as health

Usage:
    python ops/simulate.py runs/smoke-01 --turn 1 --mode distinct
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expertwins import paths  # noqa: E402

_DOC = re.compile(r"^### \[([^\]]+)\](.*)$", re.M)
_CITED_BY = re.compile(r"\[CITED BY: ([^\]]*)\]")


def _packet_docs(text: str) -> tuple[dict[str, list[str]], set[str], dict[str, str]]:
    """Map doc_id -> its passage lines, the set marked [YOURS], and the
    documents under cross-examination mapped to the seat that cited them.

    The own-corpus marker is read out of the packet rather than looked up,
    because the fixture must be able to produce a seat that collapses WITHOUT
    losing fidelity. Those are different failures and the guardrail treats them
    differently -- fidelity loss short-circuits the heterogeneity verdict -- so a
    fixture that always triggers both can only ever exercise one of them.

    The examination marker is read the same way and for the same reason: a
    fixture that cannot tell which documents were handed over for CHECKING
    cannot demonstrate that citing them is herding rather than evidence.
    """
    out: dict[str, list[str]] = {}
    own: set[str] = set()
    examined: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        m = _DOC.match(line)
        if m:
            current = m.group(1)
            out[current] = []
            if "[YOURS]" in m.group(2):
                own.add(current)
            cb = _CITED_BY.search(m.group(2))
            if cb:
                examined[current] = cb.group(1).split(",")[0].strip()
        elif current and line.strip() and not line.startswith("---"):
            out[current].append(line.strip().removeprefix(">> QUOTED >> "))
    return {k: v for k, v in out.items() if v}, own, examined


_SECTION_PREFIX = re.compile(r"^\([A-Za-z][A-Za-z0-9 \-/&,.]{0,40}\)\s")

#: One private vocabulary per simulated seat. Eight slots each:
#: (subject, verb, noun-a, noun-b, noun-c, favoured-cause, rejected-cause, verb2)
_VOCAB = [
    ("lanternmoss", "frames", "prism spores", "moonlit terraces", "luminous bracts",
     "terrace glow", "echo pollen", "explains"),
    ("glasswing voles", "trace", "echo pollen", "hollow reeds", "resonant nests",
     "reed resonance", "silver dew", "drives"),
    ("cloud kelp", "anchors", "silver dew", "floating ponds", "mist anchors",
     "pond drift", "amber gears", "organises"),
    ("clockwork beetles", "turn", "amber gears", "basalt orchards", "tick songs",
     "gear timing", "blue lanterns", "governs"),
    ("velvet lichen", "records", "blue lanterns", "north ravine", "quiet glow",
     "ravine shade", "saffron dust", "determines"),
    ("paperwing moths", "scatter", "saffron dust", "folded leaves", "dawn spirals",
     "leaf folding", "prism spores", "generates"),
]

def _quote(line: str, want: int = 90) -> str | None:
    """A contiguous span of at least the verifier's floor, copied exactly.

    The section prefix is stripped ONLY when it looks like a section label.
    Stripping any leading parenthesis would mangle passages whose own text
    begins with one. A fixture that cannot produce a bad quote cannot
    demonstrate that bad quotes are rejected.
    """
    body = _SECTION_PREFIX.sub("", line)
    if len(body) < 60:
        return None
    return body[:max(want, 60)]


def simulate(run_dir: Path, turn: int, mode: str, seed: int,
             n_claims: int = 4) -> int:
    td = run_dir / "turns" / str(turn)
    packets = sorted((td / "packets").glob("*.md"))
    if not packets:
        raise SystemExit(f"no packets in {td / 'packets'}")
    (td / "responses").mkdir(exist_ok=True)

    parsed, owned, examined = {}, {}, {}
    for p in packets:
        (parsed[p.stem], owned[p.stem],
         examined[p.stem]) = _packet_docs(p.read_text(encoding="utf-8"))
    # Documents every seat holds: the substrate a collapse converges onto.
    # Documents under examination are removed first -- every seat is handed
    # those by construction, so leaving them in would make the fixture's
    # "shared" set mostly an artifact of the disclosure rule.
    everywhere = [set(d) - set(examined[s]) for s, d in parsed.items()]
    shared = set.intersection(*everywhere) if everywhere else set()

    seats = sorted(parsed)
    for si, seat in enumerate(seats):
        docs = parsed[seat]
        under = examined[seat]
        own = sorted(owned[seat] & set(docs))
        rng = random.Random(seed + si)
        private = [d for d in docs if d not in shared and d not in under]
        pool = (sorted(shared) or sorted(docs)) if mode == "collapse" \
            else (private or sorted(docs))
        # In collapse mode a seat still anchors ONE claim in its own work, so the
        # run exhibits collapse WITHOUT fidelity loss. Otherwise the fidelity
        # short-circuit fires first and the collapse ladder is never reached.
        #
        # `distinct` anchors for the mirror-image reason, and it was a defect
        # that it once did not: a seat drawing only from its PRIVATE documents
        # can easily draw nothing it actually wrote, which the loop correctly
        # reports as FIDELITY_LOST -- and a "healthy" fixture that cannot
        # produce a healthy verdict demonstrates nothing. On a small corpus it
        # fired every time.
        if mode in ("collapse", "distinct") and own:
            pool = [own[0]] + [d for d in pool if d != own[0]]

        claims = []
        for ci in range(n_claims):
            doc = pool[ci % len(pool)]
            lines = [ln for ln in docs[doc] if _quote(ln)]
            if not lines:
                continue
            q = _quote(lines[ci % len(lines)])
            if mode == "collapse":
                # Every seat says the same thing about the same documents.
                text = ("The dominant constraint here is delivery to the Peloria plot "
                        "cell and the evidence supports that reading.")
            elif mode == "scatter":
                text = f"Fragment {rng.randint(1000, 9999)} about {doc} unrelated."
            else:
                # Each seat gets its own vocabulary. A fixture whose "distinct"
                # mode writes one template with the seat name substituted is not
                # distinct, and it makes a healthy table read as collapsed.
                lex = _VOCAB[si % len(_VOCAB)]
                text = (f"{lex[0].capitalize()} in {doc} {lex[1]} the "
                        f"{lex[2 + (ci % 3)]}; on this reading the {lex[5]} "
                        f"rather than the {lex[6]} is what {lex[7]} the result.")
            claims.append({
                "claim": text,
                "citations": [{"doc_id": doc, "quote": q}],
                "declared": {"species": "human", "model_system": "in vivo"},
                "type": "mechanism",
                "territory": "home" if si == 0 else "adjacent",
                "self_challenge": "A hostile colleague would ask for the control.",
            })

        payload = {
            "claims": claims,
            "abstentions": [{"question_part": "the rest",
                             "cause": "out_of_my_scope",
                             "detail": "outside this seat's own work"}],
            "stances": [
                {"seat": other, "claim": "their leading claim",
                 "stance": ("agree" if mode == "collapse"
                            else ["disagree", "agree", "needs_other_evidence",
                                  "out_of_scope"][(si + oi) % 4]),
                 "why": "simulated"}
                for oi, other in enumerate(seats) if other != seat],
        }
        checks = _simulate_checks(docs, under, rng)
        if checks:
            payload["citation_checks"] = checks
        (td / "responses" / f"{seat}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

    digest = hashlib.sha256(f"{mode}{seed}{turn}".encode()).hexdigest()[:8]
    print(f"simulated {len(seats)} seats, mode={mode}, fixture {digest}")
    print("THESE ARE NOT OPINIONS. This fixture tests plumbing, not science.")
    print(f"  python ops/panel.py ingest {run_dir.name} --turn {turn}")
    return 0


def _simulate_checks(docs: dict[str, list[str]], under: dict[str, str],
                     rng: random.Random) -> list[dict]:
    """Verdicts on the documents this seat was handed for cross-examination.

    Deliberately produces all four outcomes the ingest path has to tell apart,
    because a fixture that only ever files a clean `supports` cannot show that
    the other three are handled:

      * `supports`          -- no quote required
      * an adverse verdict WITH a verifying quote  -> recorded
      * an adverse verdict WITHOUT one             -> `unsupported_verdict`,
        an opinion about a paper rather than a reading of one
      * a verdict on a document NOT under examination -> `not_under_examination`,
        which is a verdict on a memory of a paper and is exactly what this
        system refuses

    THESE ARE NOT JUDGEMENTS. The fixture has not read anything.
    """
    out: list[dict] = []
    for i, (doc, cited_by) in enumerate(sorted(under.items())):
        verdict = ["supports", "overstated", "cannot_tell", "misread",
                   "supports"][i % 5]
        entry = {"doc_id": doc, "cited_by": cited_by, "verdict": verdict,
                 "why": f"simulated verdict {verdict} on {doc}",
                 "citations": []}
        if verdict in ("overstated", "misread") and i % 2 == 0:
            lines = [ln for ln in docs.get(doc, []) if _quote(ln)]
            if lines:
                entry["citations"] = [
                    {"doc_id": doc, "quote": _quote(lines[0])}]
        out.append(entry)
    if out:
        # One verdict on a document that was never handed over.
        elsewhere = [d for d in docs if d not in under]
        if elsewhere:
            out.append({"doc_id": rng.choice(elsewhere), "cited_by": "nobody",
                        "verdict": "misread",
                        "why": "simulated verdict on a document not under "
                               "examination",
                        "citations": []})
    return out


_CASE = re.compile(r"^--- case \d+ ---$", re.M)
_CLAIM = re.compile(r"^CLAIM \(([^)]+)\): (.*)$", re.M)
_CITES = re.compile(r"^CITES: (\S+)\s*$", re.M)


def _audit_cases(text: str) -> list[dict]:
    """Parse one audit packet back into the cases it puts to the reviewer.

    Read out of the rendered packet rather than out of the ledger, on purpose
    and for the same reason the permitted set is: the fixture must only be able
    to file a verdict on something a reviewer was actually SHOWN. A fixture that
    reads the ledger could file a perfectly-formed verdict on a document that
    never appeared in its packet, and `not_under_examination` -- the check that
    catches a verdict on a *memory* of a paper -- would never fire.
    """
    cases: list[dict] = []
    for block in _CASE.split(text)[1:]:
        claim = _CLAIM.search(block)
        cites = _CITES.search(block)
        if not (claim and cites):
            continue
        shown: list[str] = []
        in_source = False
        for line in block.splitlines():
            if line.startswith("THE SOURCE,"):
                in_source = True
                continue
            if in_source:
                body = line.strip()
                if not body or body.startswith("---") or body.startswith("WHAT TO"):
                    if body.startswith(("---", "WHAT TO")):
                        break
                    continue
                shown.append(body.removeprefix(">>> ").strip())
        cases.append({"cited_by": claim.group(1), "claim": claim.group(2),
                      "doc_id": cites.group(1), "shown": shown})
    return cases


def simulate_audit(run_dir: Path, turn: int, seed: int) -> int:
    """Answer the audit packets for a turn, without a model.

    Every outcome the audit ingest has to tell apart is produced at least once,
    because an audit fixture that only ever files a clean `supports` cannot show
    that the other outcomes are handled:

        supports              no quote required
        overstated / misread  WITH a verifying quote lifted from the passages
                              the reviewer was shown -> recorded
        the same WITHOUT one  -> `unsupported_verdict`
        cannot_tell           the honest abstention

    THESE ARE NOT JUDGEMENTS. The fixture has not read anything.
    """
    ad = run_dir / "turns" / str(turn) / "audit"
    packets = sorted((ad / "packets").glob("*.md"))
    if not packets:
        raise SystemExit(
            f"no audit packets in {ad / 'packets'} -- run "
            f"`python ops/panel.py audit {run_dir.name} --turn {turn}` first")
    (ad / "responses").mkdir(exist_ok=True)

    n_cases = 0
    for p in packets:
        rng = random.Random(seed + len(p.stem))
        checks = []
        for i, case in enumerate(_audit_cases(p.read_text(encoding="utf-8"))):
            verdict = ["supports", "overstated", "cannot_tell", "misread",
                       "supports"][i % 5]
            entry = {"doc_id": case["doc_id"], "cited_by": case["cited_by"],
                     "claim": case["claim"], "verdict": verdict,
                     "why": f"simulated verdict {verdict}; this fixture has not "
                            f"read anything",
                     "citations": []}
            # Adverse verdicts quote -- except one per packet, left unsupported
            # ON PURPOSE so `unsupported_verdict` is exercised.
            if verdict in ("overstated", "misread") and i % 4 != 3:
                quotable = [s for s in case["shown"] if _quote(s)]
                if quotable:
                    entry["citations"] = [
                        {"doc_id": case["doc_id"],
                         "quote": _quote(rng.choice(quotable))}]
            checks.append(entry)
        n_cases += len(checks)
        (ad / "responses" / f"{p.stem}.json").write_text(
            json.dumps({"citation_checks": checks}, indent=2), encoding="utf-8")

    print(f"simulated {len(packets)} audit response(s), {n_cases} case(s)")
    print("THESE ARE NOT JUDGEMENTS. This fixture tests plumbing, not science.")
    print(f"  python ops/panel.py audit-ingest {run_dir.name} --turn {turn}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--turn", type=int, default=1)
    ap.add_argument("--mode", default="distinct",
                    choices=["distinct", "collapse", "scatter"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--audit", action="store_true",
                    help="answer the AUDIT packets for this turn instead of "
                         "the evidence packets")
    a = ap.parse_args()
    rd = paths.resolve(a.run_dir)
    if not rd.exists():
        rd = paths.RUNS / a.run_dir
    if a.audit:
        return simulate_audit(rd, a.turn, a.seed)
    return simulate(rd, a.turn, a.mode, a.seed)


if __name__ == "__main__":
    sys.exit(main())

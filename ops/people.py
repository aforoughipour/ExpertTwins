"""Add a *person* to the table: resolve, acquire, attribute, install, verify.

A discipline seat is created by giving it a literature it alone has read and
then proving the literature is there. A person seat needs one more step, and it
is the step that decides whether the seat is that person or merely someone with
their name on it:

    resolve    WHICH scientist is this? Author strings are ambiguous and the
               ambiguity is not rare. Before downloading anything, look at what
               the search actually returns -- the venues, the years, the
               co-authors, the topics -- and confirm it is the right person.
    plan       what would be fetched, and what floors are committed
    acquire    fetch into a SEPARATE root (parallel runs must not share an
               append-only manifest; two processes appending to one manifest
               interleave lines and tear rows)
    attribute  decide which fetched papers this person ACTUALLY wrote, by
               checking the stored author list -- a different source of truth
               from the query that fetched them. This is the falsification step.
    install    merge into the library, rebuild the index, register the seat
    verify     the literature is really there AND this seat is distinct from
               everyone already at the table

`verify` is a required step, and it can fail. A seat can hold healthy-looking
document counts while its probes return ZERO. Without verification it would
still "work": fluent output, argued from papers that merely contained the right
words, invisible in the transcript.

    A DOCUMENT COUNT IS NOT COVERAGE.

`attribute` is also required, and it can fail. An author query can fetch a paper
written by a different person with the same surname and initials. An
`AUTH:"Varga LM"` query is a claim made by a search engine. The author list
stored on disk adjudicates it.

    AN AUTHOR QUERY IS NOT AUTHORSHIP.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from expertwins import paths  # noqa: E402
from expertwins.identity import (  # noqa: E402
    Individual, attribute_own_corpus, load_own, save_own,
)

TEMPLATE = """\
# Seat definition: {name}  -- an INDIVIDUAL, not a discipline.
#
# WHAT MAKES THIS A PERSON AND NOT A PROFILE.
# Nothing below describes a personality. `moves` says what they DO with a
# problem; `fatal_flaws` says what they REFUSE to accept; `territory` says where
# their authority stops. Every one of those is traceable to a body of published
# work, and none of them grants the seat a single fact -- everything it asserts
# still has to come from retrieved evidence.
#
# TWO QUERY SETS, AND THEY ARE NOT THE SAME THING.
#   retrieval_queries -- SQLite FTS5. Used at question time.
#   acquisition       -- Europe PMC. Used ONCE, to download.
#   own_queries       -- FTS5, applied INSIDE this person's own corpus, so the
#                        packet always leads with their own results.
#
# KEEP PROBES TO 3-5 TOKENS. An FTS MATCH is a conjunction, so a seven-token
# probe demands seven specific words in ONE passage, which real prose rarely
# satisfies. An eight-token probe once reported a covered topic as absent.

name: {name}
display: "CHANGE ME -- the scientist's name as they publish it"
kind: individual
probe: "CHANGE ME -- 3 to 5 words that must appear together in their field"
min_docs: 40

# Identity, for the attribution step. Multi-token surnames are supported and
# necessary ("Del Rio", "Okafor Ndubisi").
surname: "CHANGE ME"
initials: ["CHANGE ME"]
orcid: ""
affiliations: []

# Acquisition nodes whose documents are CANDIDATES for the own corpus.
# Candidacy is asserted by the query; membership is adjudicated by `attribute`.
own_topics: ["{name}.authored"]

territory: >
  CHANGE ME. Where this person's own work lives, and where it stops. Written so
  a reader can tell whether a given question is theirs.

moves: >
  CHANGE ME. What they actually do when handed a problem: the system, the assay,
  the control, the analysis, the order they do it in. This is what makes two
  experts give different answers to one question.

fatal_flaws: >
  CHANGE ME. The conditions under which this person must refuse to agree, each
  traceable to their work. "The mechanism is implausible" is not checkable.
  "A delivery claim with no measurement of what reached the cell" is.

retrieval_queries:
  - 'CHANGE ME AND (another OR term)'

own_queries:
  - 'CHANGE ME'

acquisition:
  - id: {name}.authored
    min_full: 40
    probe: "CHANGE ME short probe"
    queries:
      - 'AUTH:"CHANGE ME"'
"""


# --------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------

def cmd_resolve(spec_path: Path, max_records: int) -> int:
    """Look before you download. Who does this author query actually return?

    Prints the shape of the returned corpus -- years, venues, co-authors, ORCIDs
    -- because those are what distinguish two scientists who share a surname and
    an initial. Nothing is stored. This step exists so that a wrong identity is
    caught in one minute rather than after three hours of acquisition and a
    corpus that quietly belongs to somebody else.
    """
    from expertwins.corpus.acquire.europepmc import EuropePMC
    from expertwins.corpus.net import Fetcher

    person = Individual.load(spec_path)
    epmc = EuropePMC(Fetcher(cache_dir=paths.CACHE))

    queries = [q for node in person.acquisition
               if node["id"] in person.own_topics
               for q in node.get("queries", [])]
    if not queries:
        raise SystemExit(
            f"{spec_path}: no acquisition node listed in `own_topics`, so there "
            f"is no own corpus to resolve. That is the one node a person seat "
            f"cannot do without.")

    print(f"resolving {person.display}  (surname {person.surname!r}, "
          f"initials {person.initials})")
    if person.orcid:
        print(f"  declared ORCID {person.orcid} -- an ORCID query is exact and "
              f"should be preferred over any AUTH string")
    print()

    key = person.key
    for q in queries:
        print(f"query: {q}")
        recs = list(epmc.search(q, max_records=max_records))
        if not recs:
            print("  NOTHING RETURNED. That is a finding: either the name form "
                  "is wrong or this channel does not reach their work.\n")
            continue

        matched = [r for r in recs if any(key.matches(a) for a in r.authors)]
        years = [r.year for r in recs if r.year]
        venues = Counter(r.journal for r in recs if r.journal)
        coauth: Counter = Counter()
        for r in matched:
            for a in r.authors:
                if not key.matches(a):
                    coauth[a] += 1
        full = sum(1 for r in recs if r.has_pmc_fulltext)

        print(f"  {len(recs)} records; {len(matched)} carry an author string this "
              f"seat's name rules accept; {full} report PMC full text")
        if years:
            print(f"  years {min(years)}-{max(years)}")
        print("  venues: " + ", ".join(f"{v} ({n})" for v, n in venues.most_common(6)))
        print("  frequent co-authors: "
              + ", ".join(f"{a} ({n})" for a, n in coauth.most_common(10)))
        print("  most cited:")
        for r in sorted(recs, key=lambda x: -x.cited_by)[:5]:
            print(f"    {r.year} {r.cited_by:>6d} cites  {r.title[:88]}")
        print()

    print("CONFIRM THIS IS THE RIGHT SCIENTIST before acquiring. If the venues, "
          "the co-authors and the topics are not theirs, fix `surname`, "
          "`initials`, `orcid` or the query -- do not proceed and hope the "
          "attribution step will clean it up. It removes wrong papers; it "
          "cannot add missing ones.")
    return 0


# --------------------------------------------------------------------------
# plan / acquire
# --------------------------------------------------------------------------

def cmd_plan(spec_path: Path) -> int:
    person = Individual.load(spec_path)
    print(f"seat: {person.name}   ({person.display})")
    print(f"probe: {person.probe!r}   min_docs: {person.min_docs}")
    print(f"own topics: {', '.join(person.own_topics) or '(none -- not a person seat)'}")
    print(f"\nretrieval ({len(person.retrieval_queries)} queries, FTS syntax):")
    for q in person.retrieval_queries:
        print(f"  {q}")
    print(f"\nown-corpus queries ({len(person.own_queries)}, FTS syntax):")
    for q in person.own_queries:
        print(f"  {q}")
    total = 0
    print(f"\nacquisition ({len(person.acquisition)} nodes, Europe PMC syntax):")
    for node in person.acquisition:
        total += node.get("min_full", 0)
        own = "  [OWN CORPUS]" if node["id"] in person.own_topics else ""
        print(f"  {node['id']:<30s} floor {node.get('min_full',0):>4d}  "
              f"probe {node['probe']!r}{own}")
        for q in node.get("queries", []):
            print(f"    {q}")
    print(f"\ncommitted full-text floor across all nodes: {total}")
    print("Floors are committed BEFORE acquisition so a thin node is a FINDING, "
          "not a disappointment negotiated after the fact.")
    print("\nNothing has been downloaded. Run `acquire` to fetch.")
    return 0


def cmd_acquire(spec_path: Path, per_pass: int, group: str | None) -> int:
    """Write a temporary acquisition config and use the corpus pipeline.

    The corpus layer measures stored bytes to decide whether a document is full
    text (no caller may assert it), and it checks title-in-text consistency so
    bibliographic identity is validated separately from hash integrity.
    """
    person = Individual.load(spec_path)
    root = paths.seat_library(person.name)
    tmp = paths.CONFIG / f"_acquire_{person.name}.yaml"
    nodes = [n for n in person.acquisition
             if group is None or n["id"] == group or n["id"].startswith(f"{group}.")]
    if not nodes:
        raise SystemExit(f"no acquisition node matches {group!r}")
    tmp.write_text(yaml.safe_dump(
        {"version": 1, "column": f"seat:{person.name}",
         "groups": [{"id": person.name, "label": person.display, "nodes": nodes}]},
        sort_keys=False), encoding="utf-8")

    print(f"acquiring {person.display} into its OWN root: {root}")
    r = subprocess.run(
        [sys.executable, str(ROOT / "ops" / "acquire.py"),
         "--config", str(tmp), "--root", str(root), "--per-pass", str(per_pass)],
        cwd=ROOT)
    tmp.unlink(missing_ok=True)
    if r.returncode != 0:
        print("\nacquisition reported a problem -- read the floor check above")
    print(f"\nNext:  python ops/people.py install {paths.relative(spec_path)}")
    return r.returncode


# --------------------------------------------------------------------------
# install / attribute
# --------------------------------------------------------------------------

def cmd_install(spec_path: Path, skip_merge: bool, no_index: bool = False) -> int:
    person = Individual.load(spec_path)
    root = paths.seat_library(person.name)

    if not skip_merge:
        if not (root / "MANIFEST.jsonl").exists():
            raise SystemExit(
                f"{root} has no manifest. Run `acquire` before `install` -- "
                f"installing a seat whose literature was never downloaded is "
                f"exactly the failure this tool exists to prevent.")
        print(f"merging {root} into {paths.LIBRARY} through the ordinary put() path")
        print("(every row is re-derived from the stored bytes, never copied from "
              "a source manifest -- that property is what exposes collisions)\n")
        _merge(root)
        if no_index:
            # Installing several seats at once: rebuilding a 30,000-document
            # index per seat is minutes of pure waste. The caller promises to
            # rebuild once at the end -- and until it does, ATTRIBUTION AND
            # VERIFICATION WILL BE WRONG, because both read the index.
            print("\nindex NOT rebuilt (--no-index). Nothing that reads the "
                  "index is trustworthy until you run:")
            print("  python ops/people.py reindex")
        else:
            print("\nrebuilding the index")
            _rebuild_index()

    doc = (yaml.safe_load(paths.SEATS.read_text(encoding="utf-8"))
           if paths.SEATS.exists() else {"version": 1, "seats": []})
    entry = person.to_seat_entry()
    existing = {s["name"]: i for i, s in enumerate(doc["seats"])}
    if person.name in existing:
        doc["seats"][existing[person.name]] = entry
        print(f"updated `{person.name}` in {paths.relative(paths.SEATS)}")
    else:
        doc["seats"].append(entry)
        print(f"added `{person.name}` to {paths.relative(paths.SEATS)}")
    paths.SEATS.parent.mkdir(parents=True, exist_ok=True)
    paths.SEATS.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8")

    if person.own_topics and not no_index:
        print()
        cmd_attribute(spec_path)
    elif no_index:
        print("\nattribution deferred: it reads the index, which has not been "
              "rebuilt yet.")
    print(f"\nNext:  python ops/people.py verify {person.name}")
    print("       The seat is NOT real until that passes.")
    return 0


def _verified_coauthors(person: Individual, max_authors: int = 30,
                        min_papers: int = 1) -> set[str]:
    """Co-author keys drawn from the seat's OpenAlex-resolved own works.

    The independence is the whole point. These names come from the citation
    neighbourhood's `authored_oa` node, whose membership was decided by an
    OpenAlex author id anchored on the seat's verified papers and then narrowed
    by institution and research topic. That is a different question, asked of a
    different source, from the Europe PMC author query that proposed the
    candidates being adjudicated -- so it can contradict it.

    CONSORTIUM PAPERS ARE EXCLUDED, for the reason `known_collaborators` states:
    two hundred of Sam Okoro's papers yielded 3,416 "collaborators" because
    challenge papers carry a hundred authors each, and a collaborator set that
    large corroborates everything and therefore corroborates nothing.

    A ONE-OFF CO-AUTHOR IS WEAKER EVIDENCE THAN A RECURRING ONE, and
    `min_papers` is the dial for that: taking every co-author of Sam Okoro's
    resolved works gives 689 keys, requiring two papers gives 297. Raising it
    sharpens the set and it does NOT close the hole -- against a set of either
    size, two `surname|initial` matches are reachable by coincidence inside one
    community of common surnames. What actually separated the two Sam Okoros was
    the affiliation and full-name test applied when the plan is built
    (`neighbourhood._affiliation_seeded_coauthors`).

    SO THE DEFAULT IS 1, THE LOOSEST SETTING, ON PURPOSE. Since it buys little
    precision it should not be allowed to cost recall, and the standing
    instruction for this panel is that a seat missing a paper it wrote is a far
    worse failure than a seat holding one it did not: the extra paper is visible
    to a reader and can be argued with, the missing one silently narrows what
    the seat is able to say and nothing in the run reports it.
    """
    from expertwins.identity import coauthor_keys

    plan_path = paths.CONFIG / "neighbourhood" / f"{person.name}.yaml"
    if not plan_path.exists():
        return set()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    node = next((n for n in plan.get("nodes") or [] if n.get("kind") == "own"),
                None)
    if not node:
        return set()
    key = person.key
    seen: dict[str, int] = {}
    for w in node["works"]:
        authors = w.get("authors") or []
        if len(authors) > max_authors:
            continue
        for k in {k for a in authors if not key.matches(a)
                  for k in coauthor_keys(a)}:
            seen[k] = seen.get(k, 0) + 1
    return {k for k, n in seen.items() if n >= min_papers}


def cmd_attribute(spec_path: Path) -> int:
    """Decide which candidate papers this person actually wrote. See identity.py."""
    import sqlite3

    from expertwins.corpus.store import Library

    person = Individual.load(spec_path)
    if not person.own_topics:
        print(f"{person.name}: no own_topics; nothing to attribute "
              "(this is a discipline seat, not a person)")
        return 0
    corroborator = None
    if person.ambiguous_name:
        corroborator = _verified_coauthors(person)
        if not corroborator:
            raise SystemExit(
                f"{person.name} declares `ambiguous_name: true`, which means a "
                f"name check alone cannot decide this seat -- but there is no "
                f"citation-neighbourhood plan to corroborate against. Run "
                f"`python ops/neighbourhood.py plan {person.name}` first. "
                f"Attributing on the name alone here would file another "
                f"scientist's papers under this one, silently.")
        print(f"  ambiguous name: corroborating against {len(corroborator)} "
              f"co-authors of this seat's OpenAlex-resolved works")
    conn = sqlite3.connect(f"file:{paths.INDEX.as_posix()}?mode=ro", uri=True)
    att = attribute_own_corpus(person, Library(paths.LIBRARY), conn,
                               corroborator=corroborator)
    p = save_own(att)

    print(f"own-corpus attribution for {person.display}")
    print(f"  candidates from the author queries : {att.n_candidates}")
    print(f"  accepted (name in the stored author list): {len(att.accepted)}")
    print(f"  REJECTED (someone else's paper)          : {len(att.rejected)}")
    for r in att.rejected[:8]:
        who = ", ".join(r.get("authors", [])[:3])
        print(f"    - {r['doc_id']}: {r.get('title','')[:60]}  [{who}]")
    if len(att.rejected) > 8:
        print(f"    ... {len(att.rejected) - 8} more in {paths.relative(p)}")
    if att.n_candidates and len(att.accepted) / att.n_candidates < 0.5:
        print("\n  WARNING: more than half the candidates were rejected. Either "
              "the author query is too loose, or `surname`/`initials` are too "
              "strict. Read the rejection list before believing either.")
    print(f"\nwrote {paths.relative(p)}")
    return 0


def _merge(root: Path) -> None:
    from expertwins.corpus.store import IngestOutcome, Library

    src, dst = Library(root), Library(paths.LIBRARY)
    added = skipped = failed = retagged = dup = 0
    with dst.writer_lock(f"install seat from {root.name}"):
        for i, doc_id in enumerate(src.known_ids(), 1):
            try:
                passages = src.load_passages(doc_id)
                if not passages:
                    skipped += 1
                    continue
                meta = src.load_meta(doc_id)
                # UNION BEFORE WRITING, NOT AFTER.
                #
                # THE DEFECT THIS EXISTS FOR.
                # `put()` writes a document under its slug. When the destination
                # already holds that slug -- which it does constantly, because
                # these seats co-author and share a literature --
                # the incoming meta OVERWRITES the stored one, topics and all.
                # Whoever merges last wins, and every earlier seat's tag on that
                # paper is gone.
                #
                # A paper found under two nodes belongs to BOTH. The union is
                # taken from whatever the destination currently holds, so the
                # operation is order-independent and idempotent.
                if dst.is_registered(doc_id):
                    try:
                        held = dst.load_meta(doc_id)
                        merged_topics = sorted(set(held.topics) | set(meta.topics))
                        if merged_topics != sorted(meta.topics):
                            meta = meta.model_copy(update={"topics": merged_topics})
                            retagged += 1
                    except Exception:                             # noqa: BLE001
                        pass
                pdf = src.doc_path(doc_id) / "source.pdf"
                res = dst.put(meta, passages,
                              source_bytes=pdf.read_bytes() if pdf.exists() else None)
                if res.outcome is IngestOutcome.DUPLICATE:
                    # Matched an existing document by identity key under a
                    # DIFFERENT slug, so `put` declined to write. Tag it in place.
                    dup += 1
                    retagged += _union_topics(dst, res.existing_doc_id or res.doc_id,
                                              meta.topics)
                else:
                    added += 1
            except Exception as exc:                              # noqa: BLE001
                failed += 1
                if failed <= 3:
                    print(f"  FAILED {doc_id}: {type(exc).__name__}: {exc}")
            if i % 500 == 0:
                print(f"  {i} seen | added {added} dup {dup} retagged {retagged} "
                      f"skipped {skipped} failed {failed}", flush=True)
    print(f"  merged: added {added}, already held {dup} (of which {retagged} "
          f"re-tagged with this seat's topics), skipped {skipped}, failed {failed}")


def _union_topics(library, doc_id: str, topics: list[str]) -> int:
    """Union topic tags onto a document the library already holds.

    Same operation the acquisition pipeline performs when one paper answers two
    coverage nodes; recording only the first is how a corpus develops invisible
    holes. Returns 1 if anything changed.
    """
    from expertwins.corpus.store import IngestOutcome
    from pathlib import Path as _P

    if not doc_id or not topics:
        return 0
    try:
        meta = library.load_meta(doc_id)
    except Exception as exc:                                      # noqa: BLE001
        print(f"  RETAG FAILED {doc_id}: {exc}")
        return 0
    merged = sorted(set(meta.topics) | set(topics))
    if merged == sorted(meta.topics):
        return 0
    updated = meta.model_copy(update={"topics": merged})
    passages = library.load_passages(doc_id)
    source = None
    if meta.source_filename:
        p = library.doc_path(doc_id) / meta.source_filename
        if p.exists():
            source = p.read_bytes()
    ext = _P(meta.source_filename).suffix or ".xml"
    library._write(updated, passages, source, ext,   # noqa: SLF001
                   IngestOutcome.REPLACED)
    return 1


def _rebuild_index() -> None:
    from expertwins.corpus.index import Index
    from expertwins.corpus.store import Library

    Index.rebuild(Library(paths.LIBRARY), paths.INDEX)
    print("  index rebuilt")


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

def cmd_verify(name: str, per_query: int) -> int:
    """The step that decides whether the seat is real.

    Three questions for a person seat, all answered with no model and no GPU:
      1. Is the literature there? (probe, with a fault diagnosis)
      2. Do we actually hold their OWN work, attributed rather than assumed?
      3. Is this seat DIFFERENT from the seats already at the table?

    A seat that duplicates an existing seat's reading is not a new expert; it is
    the same expert wearing a second label, which is the persona failure this
    project exists to escape.
    """
    from expertwins.retrieve import Retriever, fts_quote_terms

    specs = yaml.safe_load(paths.SEATS.read_text(encoding="utf-8"))["seats"]
    if not any(s["name"] == name for s in specs):
        raise SystemExit(f"`{name}` is not in {paths.relative(paths.SEATS)} "
                         f"-- run `install` first")
    r = Retriever(paths.INDEX)

    print(f"{'seat':<26s} {'docs':>6s} {'probe':>6s} {'own':>6s}  status")
    print("-" * 70)
    docsets: dict[str, set[str]] = {}
    ok = True
    for s in specs:
        docs: set[str] = set()
        for q in s.get("queries", []):
            try:
                docs |= {p.doc_id for p in r.search(fts_quote_terms(q),
                                                    per_query, None)}
            except RuntimeError as exc:
                print(f"  [{s['name']}] query failed: {exc}")
        docsets[s["name"]] = docs
        own = load_own(s["name"])
        hits = (len(r.search(fts_quote_terms(s["probe"]), 10, None))
                if s.get("probe") else -1)

        status = "ok"
        if hits == 0:
            # Distinguish an over-specified probe from a genuine coverage gap.
            toks = s["probe"].split()
            fault = ""
            for cut in range(len(toks) - 1, 1, -1):
                if r.search(" ".join(toks[:cut]), 5, None):
                    fault = f"PROBE FAULT: {cut} tokens match, {len(toks)} do not"
                    break
            status = fault or "NOT COVERED"
            if s["name"] == name:
                ok = False
        elif len(docs) < s.get("min_docs", 0):
            status = f"THIN ({len(docs)} < {s['min_docs']})"
            if s["name"] == name:
                ok = False
        elif s.get("kind") == "individual" and not own:
            status = "NO OWN CORPUS -- run `attribute`"
            if s["name"] == name:
                ok = False
        mark = " <--" if s["name"] == name else ""
        print(f"{s['name']:<26s} {len(docs):>6d} {hits:>6d} {len(own):>6d}  "
              f"{status}{mark}")

    print(f"\ndistinctness of `{name}` against the seated panel:")
    mine = docsets[name]
    worst = 0.0
    for other, docs in docsets.items():
        if other == name or not docs or not mine:
            continue
        ov = len(mine & docs) / min(len(mine), len(docs))
        worst = max(worst, ov)
        print(f"  vs {other:<26s} {ov:6.1%}{'  <-- HIGH' if ov > 0.35 else ''}")
    print(f"\n  worst overlap {worst:.1%}")
    print("  reference: personas 100% (inert), a corpus split 2%.")
    print("  Above ~35% this seat is reading what another seat already reads.")
    if worst > 0.35:
        ok = False

    own = load_own(name)
    if own:
        others = {n: load_own(n) for n in docsets if n != name}
        clash = {n: len(own & o) for n, o in others.items() if own & o}
        if clash:
            print("\n  shared own-corpus documents (co-authored papers):")
            for n, k in sorted(clash.items(), key=lambda kv: -kv[1]):
                print(f"    {n:<26s} {k}")
            print("  Co-authorship is real and is NOT a failure. It does mean "
                  "these two seats will sometimes cite the same paper as `own`, "
                  "and the evidence axis of the heterogeneity measure will read "
                  "lower between them for good reason.")

    print("\n" + "=" * 70)
    if ok:
        print(f"VERDICT: `{name}` is a real seat. Invite it with")
        print(f'  python ops/panel.py prepare "..." --out runs/rt-XX --seats {name},...')
    else:
        print(f"VERDICT: `{name}` is NOT ready. Fix the failures above.")
        print("A seat with no probe hits will still produce fluent output. That")
        print("is the failure mode this check exists to catch.")
    print("=" * 70)
    print("\nNOTE: distinctness is not usefulness. A deliberately MISMATCHED "
          "panel was once disjoint, probe-coherent, covered -- and worthless. "
          "Nothing offline can tell you whether a seat has anything to "
          "contribute. Only using it can.")
    return 0 if ok else 1


def cmd_status(_: object) -> int:
    """One line per person: corpus, own corpus, attribution health."""
    import sqlite3

    if not paths.INDEX.exists():
        raise SystemExit(f"no index at {paths.INDEX}")
    conn = sqlite3.connect(f"file:{paths.INDEX.as_posix()}?mode=ro", uri=True)
    print(f"{'seat':<26s} {'own':>6s} {'rej':>6s} {'candidates':>11s}  display")
    print("-" * 84)
    for spec_path in sorted(paths.PEOPLE.glob("*.yaml")):
        person = Individual.load(spec_path)
        cand = 0
        for t in person.own_topics:
            cand += conn.execute(
                "SELECT COUNT(*) FROM doc_topics WHERE topic=?", (t,)).fetchone()[0]
        p = paths.PEOPLE / f"{person.name}.own.json"
        if p.exists():
            j = json.loads(p.read_text(encoding="utf-8"))
            print(f"{person.name:<26s} {j['n_accepted']:>6d} {j['n_rejected']:>6d} "
                  f"{cand:>11d}  {person.display}")
        else:
            print(f"{person.name:<26s} {'-':>6s} {'-':>6s} {cand:>11d}  "
                  f"{person.display}  (not attributed)")
    return 0


def cmd_merge(root: Path, no_index: bool) -> int:
    """Merge any acquisition root into the library.

    `install` merges the root belonging to ONE seat. The citation-neighbourhood
    and preprint tools write roots that belong to no seat -- `library_preprints`
    is the field's canon, `library_nbhd_<seat>` is what one scientist reads --
    and those still have to come in through the same `put()` path, so that every
    row is re-derived from the stored bytes rather than copied from a source
    manifest. That property guards against slug-collision failures, and a
    second, tidier merge path that skipped it would quietly reintroduce them.
    """
    if not (root / "MANIFEST.jsonl").exists():
        raise SystemExit(
            f"{root} has no manifest -- nothing was acquired into it. Merging "
            f"an empty root would report success for a corpus that does not "
            f"exist.")
    print(f"merging {root} into {paths.LIBRARY} through the ordinary put() path\n")
    _merge(root)
    if no_index:
        print("\nindex NOT rebuilt (--no-index). NOTHING that reads the index "
              "is trustworthy until you run:\n  python ops/people.py reindex")
        return 0
    print("\nrebuilding the index")
    _rebuild_index()
    print("\nNow RE-ATTRIBUTE every seat whose own topics this root carried:\n"
          "  python ops/people.py attribute config/people/<seat>.yaml")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template"); t.add_argument("name")
    rs = sub.add_parser("resolve"); rs.add_argument("spec")
    rs.add_argument("--max-records", type=int, default=200)
    p = sub.add_parser("plan"); p.add_argument("spec")
    a = sub.add_parser("acquire"); a.add_argument("spec")
    a.add_argument("--per-pass", type=int, default=100)
    a.add_argument("--group", default=None, help="restrict to one acquisition node")
    i = sub.add_parser("install"); i.add_argument("spec")
    i.add_argument("--skip-merge", action="store_true")
    i.add_argument("--no-index", action="store_true",
                   help="merge but do NOT rebuild the index; run `reindex` after "
                        "installing several seats. Attribution and verification "
                        "are wrong until you do.")
    at = sub.add_parser("attribute"); at.add_argument("spec")
    v = sub.add_parser("verify"); v.add_argument("name")
    v.add_argument("--per-query", type=int, default=40)
    m = sub.add_parser("merge", help="merge ANY acquisition root into the library")
    m.add_argument("root", help="e.g. library_nbhd_julia_marino, library_preprints")
    m.add_argument("--no-index", action="store_true",
                   help="merge several roots, then run `reindex` once")
    sub.add_parser("status")
    sub.add_parser("reindex", help="rebuild the FTS index from the merged library")

    args = ap.parse_args()
    if args.cmd == "template":
        print(TEMPLATE.format(name=args.name))
        return 0
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "merge":
        return cmd_merge(paths.resolve(args.root), args.no_index)
    if args.cmd == "reindex":
        print("rebuilding the index -- wait for this to FINISH before measuring "
              "anything. A mid-write index returns zeros that look exactly like "
              "a coverage gap.")
        _rebuild_index()
        return 0
    if args.cmd == "verify":
        return cmd_verify(args.name, args.per_query)
    spec = paths.resolve(args.spec)
    if not spec.exists():
        alt = paths.PEOPLE / f"{args.spec}.yaml"
        if alt.exists():
            spec = alt
        else:
            raise SystemExit(f"no such spec: {args.spec}")
    return {
        "resolve": lambda: cmd_resolve(spec, args.max_records),
        "plan": lambda: cmd_plan(spec),
        "acquire": lambda: cmd_acquire(spec, args.per_pass, args.group),
        "install": lambda: cmd_install(spec, args.skip_merge, args.no_index),
        "attribute": lambda: cmd_attribute(spec),
    }[args.cmd]()


if __name__ == "__main__":
    sys.exit(main())

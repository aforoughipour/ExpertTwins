"""The citation neighbourhood: the corpus a scientist READS, not only the one they wrote.

THE BRIEF, in the operator's words: *"in order to make sure that the corpus of
each scientist is complete, not only should you include their own papers, but
also the papers they have frequently cited, the papers of their collaborators,
papers which have cited their papers a lot, and important papers of the field.
And when I'm saying important papers of the field, to some extent it is
subjective, so you need to keep that in mind."*

He is right, and the existing design has the gap he describes. A seat was built
from three acquisition nodes -- `authored`, `problems`, `lineage` -- where the
last two are TOPIC queries. A topic query finds papers that talk about the same
subject. It does not find the papers this particular scientist actually stands
on, and those are a different set: they are the ones in their reference lists.

So this tool builds four more nodes per seat, all of them derived from the
citation graph rather than from a keyword:

    reads           the works this scientist cites MOST OFTEN across their own
                    papers. If someone cites a paper in fifteen of their own
                    papers, it is load-bearing for them whatever its topic
    collaborators   the works of the people they publish with. A lab's thinking
                    is shared, and the argument a seat can make is bounded by
                    what its collaborators have shown
    cited_by        the most-cited works that cite THEM. This is the field's
                    reply to their work, and a scientist knows who is building
                    on them and who is arguing with them
    canon           the most-cited works in the topics they work in, whoever
                    wrote them -- the papers you are expected to have read

WHY OPENALEX AND NOT GOOGLE SCHOLAR. The brief asked for Google Scholar. Scholar
has no API, forbids scraping in its terms, and blocks it in practice, so the
honest options were to disobey it or to say so. OpenAlex is the same object seen
through a stable public API: 250M+ works, `referenced_works` and `cited_by` on
every record, author disambiguation, and -- decisively for this panel -- it
indexes CVPR, ICCV, NeurIPS, ECCV and MICCAI, which is exactly the literature
Europe PMC cannot see and exactly why six computational seats came out thin.

AUTHOR IDENTITY IS DERIVED, NEVER SEARCHED. This is the same rule as
`attribute_own_corpus`, applied one level up. The tool does NOT ask OpenAlex for
"the author called Julia R. Marino" -- that is a claim by a search engine, and
`Okoro S` alone would return half a dozen different people. It starts from the
seat's ALREADY-ATTRIBUTED own corpus, looks up each of those papers by DOI, and
counts which OpenAlex author id appears on them under a name this seat's own
matcher accepts. An id that carries several of a scientist's verified papers is
that scientist; an id that carries one is a coincidence and is dropped. The
threshold is stated (`--min-anchor`) rather than hidden.

WHAT IS SUBJECTIVE HERE, STATED PLAINLY. `canon` is a judgement wearing a
number: "most cited in this topic" is a proxy for "important", and it is a poor
one in a young field -- it favours the old, the review, and the big consortium.
The brief says as much. It is included because the alternative (nothing) is
worse, it is tagged `field.canon` so it can be found and removed, and it is
never attributed to anybody's own corpus.

Usage:
    python ops/neighbourhood.py plan julia_marino
    python ops/neighbourhood.py plan --all
    python ops/neighbourhood.py acquire julia_marino
    python ops/neighbourhood.py acquire julia_marino --dry-run
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sqlite3
import sys
import unicodedata
import urllib.parse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from expertwins import paths  # noqa: E402
from expertwins.identity import Individual, load_own  # noqa: E402
from expertwins.netenv import console_utf8, use_system_trust_store  # noqa: E402

use_system_trust_store()
console_utf8()

OA = "https://api.openalex.org"
NBHD = paths.CONFIG / "neighbourhood"

#: How much of each kind to take. Engineering defaults, stated in one place.
N_OWN_WORKS = 400
N_READS = 120
N_COLLABORATORS = 8
N_PER_COLLABORATOR = 12
N_CITED_BY = 80
N_CANON_TOPICS = 3
N_PER_CANON_TOPIC = 30
#: An OpenAlex author id must carry at least this many of the seat's VERIFIED
#: own papers before it is accepted as being that person.
MIN_ANCHOR = 2

#: OpenAlex record types that are a piece of scientific writing someone could
#: read and quote.
#:
#: THE DEFECT THIS EXISTS FOR. OpenAlex indexes a scientist's output record, not
#: only their readable papers. Meeting abstracts, datasets, supplements and
#: other deposits can make a working acquisition look thin if left unfetched, or
#: enter the corpus as quotable documents if fetched indiscriminately.
#:
#: `_slim` never carried `type`, so nothing downstream could tell the difference.
SCHOLARLY_TYPES = frozenset({
    "article", "review", "preprint", "conference-paper", "book-chapter",
    "report", "data-paper",
})


def _is_scholarly(work: dict) -> bool:
    """Is this OpenAlex record a readable piece of writing?

    A MISSING type is accepted. Absence of information is not evidence of a
    defect -- the same rule `_given_name_conflict` applies to an initial -- and
    OpenAlex sets this field on essentially everything, so the permissive branch
    is nearly never taken.
    """
    t = (work.get("type") or "").rsplit("/", 1)[-1].strip().lower()
    return (not t) or t in SCHOLARLY_TYPES


class OpenAlex:
    """A small, polite OpenAlex client over the corpus fetcher.

    The fetcher carries the User-Agent, the per-host throttle, retry with
    backoff and the on-disk cache, so this class is only URL construction and
    paging. `mailto` puts us in OpenAlex's polite pool, which is both the
    courteous thing to do and the fast one.
    """

    def __init__(self, fetcher, mailto: str = "") -> None:
        self.fetcher = fetcher
        self.mailto = mailto or "your-email@example.org"

    def get(self, path: str, **params) -> dict:
        params.setdefault("mailto", self.mailto)
        url = f"{OA}/{path}?{urllib.parse.urlencode(params)}"
        resp = self.fetcher.get(url)
        if not resp.ok:
            # Reported, never returned as an empty page: an API that is refusing
            # us must not look like a field with no publications.
            raise RuntimeError(f"OpenAlex {path} failed: "
                               f"{resp.error or resp.status} ({url[:120]})")
        return resp.json()

    def work_by_doi(self, doi: str) -> dict | None:
        d = doi.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:")
        if not d:
            return None
        try:
            return self.get(f"works/doi:{urllib.parse.quote(d, safe='/')}")
        except RuntimeError:
            return None

    def works(self, filters: str, *, sort: str = "cited_by_count:desc",
              max_records: int = 200, select: str = "") -> list[dict]:
        out: list[dict] = []
        cursor = "*"
        select = select or ("id,doi,title,display_name,publication_year,"
                            "cited_by_count,referenced_works,authorships,"
                            "primary_location,best_oa_location,ids,topics,type")
        while len(out) < max_records:
            page = self.get("works", filter=filters, sort=sort, select=select,
                            per_page=min(200, max_records - len(out)),
                            cursor=cursor)
            results = page.get("results") or []
            out.extend(results)
            cursor = (page.get("meta") or {}).get("next_cursor")
            if not results or not cursor:
                break
        return out[:max_records]


# --------------------------------------------------------------------------
# identity: which OpenAlex author IS this seat
# --------------------------------------------------------------------------

def _given_name_conflict(seat_display: str, candidate: str) -> bool:
    """Do two names disagree on a GIVEN NAME that is spelled out in both?

    THE DEFECT THIS EXISTS FOR. `NameKey.matches` compares a surname and
    INITIALS, which is right for adjudicating a paper: an author list often
    carries only surname and initials, so demanding a full given name would
    reject most of a scientist's real corpus.

    At the author-ID level it is too permissive, because the accepted object is
    not one paper but an entire OpenAlex author entity. A frequent collaborator
    who shares the seat's surname and initials can pull a different scientist's
    complete works into the own-corpus candidates, and `attribute_own_corpus`
    cannot reject papers whose stored author lists genuinely match the initials.

    So: when BOTH names spell a given name out, the given names must match.
    `L M Varga` vs `Lazlo M. Varga` still passes -- an initial is not a
    disagreement, it is an absence of information.
    """
    from expertwins.identity import fold_name

    a = fold_name(seat_display).split()
    b = fold_name(candidate).split()
    if len(a) < 2 or len(b) < 2:
        return False
    given_a, given_b = a[:-1], b[:-1]
    first_a, first_b = given_a[0], given_b[0]
    if _is_initials(first_a, given_b) or _is_initials(first_b, given_a):
        return False            # one side gave initials; no conflict possible
    return first_a != first_b


def _is_initials(token: str, other_given: list[str]) -> bool:
    """Is this token an initialism rather than a name?

    `R` obviously. `RK` less obviously, and it is how a great deal of the
    literature indexes Lazlo M. Varga -- so it is checked against the OTHER
    name's initials rather than by length alone: `rk` is the initials of
    ["lazlo", "m"], while `sayeed` is not the initials of ["sam"].
    """
    if len(token) == 1:
        return True
    initials = "".join(t[0] for t in other_given if t)
    return len(token) <= 3 and initials.startswith(token)


def resolve_author_ids(person: Individual, library, own_ids: set[str],
                       oa: OpenAlex, min_anchor: int,
                       max_probe: int = 60) -> tuple[list[str], dict]:
    """Find this person's OpenAlex author id(s) from their VERIFIED own papers.

    Derived, not searched -- see the module docstring. The evidence for each id
    is returned with it so a wrong resolution is visible rather than silent.
    """
    key = person.key
    hits: Counter = Counter()
    names: dict[str, str] = {}
    anchor_topics: Counter = Counter()
    probed = matched = 0
    for doc_id in sorted(own_ids)[:max_probe]:
        try:
            meta = library.load_meta(doc_id)
        except Exception:                                         # noqa: BLE001
            continue
        doi = (getattr(meta, "doi", "") or "").strip()
        if not doi:
            continue
        probed += 1
        work = oa.work_by_doi(doi)
        if not work:
            continue
        matched += 1
        for a in work.get("authorships") or []:
            author = a.get("author") or {}
            name = author.get("display_name") or ""
            if author.get("id") and key.matches(name):
                hits[author["id"]] += 1
                names[author["id"]] = name
        for t in (work.get("topics") or [])[:3]:
            if t.get("id"):
                anchor_topics[t["id"]] += 1

    candidates = []
    accepted: list[str] = []
    top = hits.most_common(1)[0][1] if hits else 0
    for aid, n in hits.most_common():
        why = ""
        if _given_name_conflict(person.display, names[aid]):
            why = f"given name conflicts with {person.display!r}"
        elif n < min_anchor:
            why = f"anchored on only {n} verified paper(s); need {min_anchor}"
        candidates.append({"id": aid, "name": names[aid], "anchored_papers": n,
                           "rejected_because": why})
        if not why:
            accepted.append(aid)

    # A SECOND ID FOR THE SAME PERSON IS NORMAL -- OpenAlex splits identities --
    # but so is a genuinely different scientist with the same name, and nothing
    # here can tell them apart. Say so rather than pick.
    hazard = len(accepted) > 1
    evidence = {
        "probed_own_papers": probed,
        "found_in_openalex": matched,
        "candidates": candidates,
        "accepted": accepted,
        "min_anchor": min_anchor,
        "top_anchor_count": top,
        "same_name_hazard": hazard,
        # The RESEARCH TOPICS of this scientist's verified papers, as OpenAlex
        # labels them. A third independent signal, and the one that finally
        # separates a merged author entity's foreign work from the real
        # corpus: names collide, communities overlap, institutions are often
        # missing -- but a quantum-algebra paper and a toy-fieldcraft
        # paper do not share a topic.
        "anchor_topics": [t for t, _ in anchor_topics.most_common(12)],
        "hazard_note": (
            "More than one OpenAlex author id passed. That is either a split "
            "identity (harmless) or a different scientist with the same name "
            "and shared initials (not harmless), and this tool cannot tell "
            "which. Read the names, and re-plan with --top-author-only if they "
            "are not obviously the same person." if hazard else ""),
    }
    return accepted, evidence


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def _held_dois() -> set[str]:
    """Every DOI already in the assembled library.

    Used to report how much of a plan is NEW. It is a report, not a filter: the
    store's own duplicate handling is what actually prevents a second copy, and
    duplicating that decision here would give two places that can disagree.
    """
    if not paths.INDEX.exists():
        return set()
    db = sqlite3.connect(f"file:{paths.INDEX.as_posix()}?mode=ro", uri=True)
    return {(r[0] or "").strip().lower().removeprefix("https://doi.org/")
            for r in db.execute("SELECT doi FROM docs WHERE doi != ''")}


def _slim(work: dict, reason: str) -> dict:
    ids = work.get("ids") or {}
    loc = work.get("best_oa_location") or work.get("primary_location") or {}
    doi = (work.get("doi") or "").strip().lower().removeprefix("https://doi.org/")
    return {
        "oa_id": (work.get("id") or "").rsplit("/", 1)[-1],
        "title": work.get("display_name") or work.get("title") or "",
        "year": work.get("publication_year"),
        "doi": doi,
        "pmid": (ids.get("pmid") or "").rsplit("/", 1)[-1],
        "pdf_url": loc.get("pdf_url") or "",
        # CARRIED so that what a node asked for can be audited after the fact.
        # Filtering happens in `_slim_many`; recording it here is what makes the
        # filter's decision inspectable in the plan file instead of invisible.
        "type": (work.get("type") or "").rsplit("/", 1)[-1],
        "venue": ((work.get("primary_location") or {}).get("source") or {})
                 .get("display_name") or "",
        "cited_by": work.get("cited_by_count", 0),
        # CARRIED, AND IT IS LOAD-BEARING. A document fetched without its author
        # list cannot be attributed to anyone: `attribute_own_corpus` asks
        # whether the seat's name is in the STORED author list, so a paper
        # stored with no authors is rejected from its own author's corpus. If
        # this field is dropped, an `authored_oa` node can come back rejected in
        # the safe-looking direction, because a rejection reads as "we were
        # careful" rather than "we lost the metadata".
        "authors": [((a.get("author") or {}).get("display_name") or "")
                    for a in (work.get("authorships") or [])][:60],
        "reason": reason,
    }


def _slim_many(works: list[dict], reason: str) -> list[dict]:
    """Slim a page of OpenAlex works, dropping everything that is not a paper.

    One place, so that every node -- what they wrote, what they read, who they
    work with, who answered them, and the field's canon -- is filtered by the
    same rule. See `SCHOLARLY_TYPES` for why.
    """
    return [_slim(w, reason) for w in works if _is_scholarly(w)]


def _collab_keys(name: str) -> set[str]:
    """Order-agnostic identity keys for a co-author name.

    Moved to `expertwins.identity.coauthor_keys` when attribution needed the same rule:
    the same two sources spell names in opposite orders one level down, and two
    implementations of this would be two things that can disagree. Kept as a
    name here because that is what this module's tests and callers say.
    """
    from expertwins.identity import coauthor_keys

    return coauthor_keys(name)


def known_collaborators(library, own_ids: set[str], person: Individual,
                        max_docs: int = 200, max_authors: int = 30) -> set[str]:
    """Everyone who appears on this scientist's VERIFIED papers, as keys.

    Derived from the library's own stored author lists -- not from OpenAlex --
    so it is independent of the identity resolution it is used to corroborate.
    That independence is the whole point: it is a second source of truth being
    asked a question the first one cannot answer.

    CONSORTIUM PAPERS ARE EXCLUDED, and that is not a detail. Two hundred of
    Sam Okoro's papers yielded 3,416 "collaborators", because challenge and
    consortium papers carry a hundred authors each and half the field has
    stood on one. A collaborator set that large corroborates everything and
    therefore corroborates nothing: it passed 386 of 400 works, including the
    object-detection paper by a different Sam Okoro that the filter exists to
    catch.
    """
    out: set[str] = set()
    key = person.key
    for doc_id in sorted(own_ids)[:max_docs]:
        try:
            meta = library.load_meta(doc_id)
        except Exception:                                         # noqa: BLE001
            continue
        authors = list(getattr(meta, "authors", []) or [])
        if len(authors) > max_authors:
            continue
        for a in authors:
            if a and not key.matches(a):
                out |= _collab_keys(a)
    return out


def _institutions_of(work: dict, key) -> list[str]:
    """The institutions OpenAlex records for THIS SEAT's authorship on a work."""
    out: list[str] = []
    for a in work.get("authorships") or []:
        name = ((a.get("author") or {}).get("display_name") or "")
        if not key.matches(name):
            continue
        for inst in a.get("institutions") or []:
            if inst.get("display_name"):
                out.append(inst["display_name"])
        for raw in a.get("raw_affiliation_strings") or []:
            if raw:
                out.append(raw)
    return out


def _full_name_key(name: str) -> str:
    """An order-agnostic key that needs the WHOLE name, not an initial.

    `coauthor_keys` deliberately reduces a name to `surname|initial` so that
    "Okoro, Sam" and "S. Okoro" can be recognised as the same person across two
    sources that spell names differently. That tolerance is what makes it
    useless for separating two scientists inside one community of common
    surnames: loose surname-initial collisions can make unrelated co-authors
    look shared. Full names avoid that collision where both sources provide
    them.

    So this key is the strict counterpart, used only where both sides come from
    the same source and therefore carry full given names. It normalises
    accents, lowercases, and sorts the tokens, so "Jun-Glass Reed" and
    "Reed, Jun Glass" agree while "Vale O" matches nothing at all -- an initial
    is not a name here, and returning "" for it is the point.
    NOTE THE DASH SUBSTITUTION, WHICH A TEST CAUGHT AND WHICH MATTERED.
    OpenAlex can write names with U+2010 HYPHEN, not an ASCII hyphen. Stripping
    non-ASCII characters DELETES it rather than splitting on it, so equivalent
    names can produce different keys and silently fail to match.
    """
    n = unicodedata.normalize("NFKD", name)
    n = re.sub(r"[\u2010-\u2015\u2212]", "-", n)
    n = n.encode("ascii", "ignore").decode()
    toks = [t for t in re.split(r"[^a-z]+", n.lower()) if len(t) > 1]
    return " ".join(sorted(toks)) if len(toks) >= 2 else ""


def _affiliation_seeded_coauthors(works: list[dict], key,
                                  declared: list[str],
                                  max_authors: int = 30) -> set[str]:
    """Full-name co-authors of the works that PROVABLY belong to this seat.

    The seed has to be clean or the test is circular. Corroborating against the
    co-authors of the whole OpenAlex author entity is exactly that circle: the
    entity may be merged, so foreign papers' own co-authors are in the set that
    is asked to vouch for them.

    Works that positively match a declared affiliation cannot be the other
    scientist's, so their co-authors are the one set that is safe to trust.
    """
    out: set[str] = set()
    for w in works:
        aus = [((a.get("author") or {}).get("display_name") or "")
               for a in (w.get("authorships") or [])]
        if len(aus) > max_authors:
            continue
        got = " | ".join(_institutions_of(w, key)).lower()
        if not got or not any(d.strip().lower() in got
                              for d in declared if d.strip()):
            continue
        out |= {_full_name_key(a) for a in aus if a and not key.matches(a)}
    out.discard("")
    return out


def _affiliation_matches(work: dict, key, declared: list[str]) -> bool:
    """Does this seat's authorship on this work sit at a declared affiliation?

    THE SIGNAL THAT FINALLY SEPARATES TWO PEOPLE WITH ONE NAME. Co-author
    corroboration narrowed `sam_okoro` from 400 works to 338 and still could not
    drop `MirrorMoth`, because the object-detection Sam Okoro and the medical-imaging
    Sam Okoro share co-authors through the computer-vision community. Names
    cannot settle it and co-authors cannot settle it. Institutions can: one of
    them is at the Peloria Institute and the other is not.

    NOTE WHAT THIS RETIRES. `orcid` and `affiliations` are fields a seat spec
    carries that used to be declared but unused -- a promise the code did not
    keep. `affiliations` is
    now load-bearing for exactly the seats whose names are ambiguous, which is
    what it was always for.

    A seat that declares no affiliations is not filtered, and neither is a work
    for which OpenAlex holds no affiliation at all: an absent record is not a
    contradicting one, and older papers frequently carry no institution.
    """
    if not declared:
        return True
    got = " | ".join(_institutions_of(work, key)).lower()
    if not got:
        return True
    return any(d.strip().lower() in got for d in declared if d.strip())


def _corroborated(works: list[dict], collaborators: set[str],
                  min_shared: int = 1) -> list[dict]:
    """Keep works that share at least `min_shared` authors with the verified corpus.

    THE CASE THIS WAS BUILT FOR. `sam_okoro` resolves to an OpenAlex author id
    holding 400 works, and reading them shows it is a MERGED entity: CAMELYON16,
    GlaS, H-DenseUNet and VoxResNet are his, and `MirrorMoth: Fully Convolutional
    One-Stage Object Detection` belongs to a different Sam Okoro entirely. No
    name test can separate those -- the name is identical -- so the question has
    to be asked of something other than the name: *does this paper share authors
    with papers already verified as his?*

    It is deliberately conservative and it WILL drop real papers -- a first
    paper with an entirely new group has no shared author by construction. That
    trade is the standing preference of this project: a thin own corpus is
    visible in `people.py status`, and a contaminated one is invisible forever.
    """
    if not collaborators:
        return works
    kept = []
    for w in works:
        shared = sum(1 for a in (w.get("authors") or [])
                     if _collab_keys(a) & collaborators)
        if shared >= min_shared:
            kept.append(w)
    return kept


def plan_seat(person: Individual, oa: OpenAlex, library, *, min_anchor: int,
              top_only: bool = False, corroborate: bool | None = None,
              verbose: bool = True) -> dict:
    own_ids = load_own(person.name)
    if not own_ids:
        raise SystemExit(
            f"{person.name} has no attributed own corpus. Run "
            f"`python ops/people.py attribute config/people/{person.name}.yaml` "
            f"first -- the neighbourhood is derived FROM the own corpus, and "
            f"deriving it from a name search is the thing this refuses to do.")

    author_ids, evidence = resolve_author_ids(person, library, own_ids, oa,
                                              min_anchor)
    if top_only and author_ids:
        author_ids = author_ids[:1]
        evidence["accepted"] = author_ids
        evidence["top_author_only"] = True
    if verbose:
        print(f"  probed {evidence['probed_own_papers']} own papers by DOI, "
              f"{evidence['found_in_openalex']} found in OpenAlex")
        for c in evidence["candidates"][:8]:
            mark = "ACCEPTED" if c["id"] in author_ids else "dropped"
            tail = f"  ({c['rejected_because']})" if c["rejected_because"] else ""
            print(f"    {mark:<8s} {c['id'].rsplit('/', 1)[-1]:<14s} "
                  f"{c['name'][:32]:<32s} {c['anchored_papers']:>3d} anchored{tail}")
        if evidence["same_name_hazard"] and not top_only:
            print(f"    HAZARD: {evidence['hazard_note']}")
    if not author_ids:
        raise SystemExit(
            f"{person.name}: no OpenAlex author id carries {min_anchor}+ of "
            f"this seat's verified papers. Refusing to guess -- a wrong author "
            f"id would fill this seat's shelves with another person's field.")

    who = "|".join(author_ids)
    own_works = oa.works(f"author.id:{who}", max_records=N_OWN_WORKS)
    # FILTERED HERE, before anything is derived from them, and that ordering is
    # the point. `reads`, `collaborators` and `field.canon` are all computed
    # from `own_works` -- their reference lists, their author lists, their
    # topics. A supplementary figure deposit has no reference list and carries
    # the paper's full author list, so leaving these in does not just pad the
    # own node: it double-counts collaborators and lets a dataset record's
    # topics steer the canon.
    n_raw = len(own_works)
    own_works = [w for w in own_works if _is_scholarly(w)]
    evidence["dropped_not_scholarly"] = n_raw - len(own_works)
    if verbose and n_raw != len(own_works):
        print(f"  {n_raw - len(own_works)} of {n_raw} OpenAlex records are not "
              f"papers (abstracts, datasets, figure deposits, errata) -- dropped")
    if verbose:
        print(f"  {len(own_works)} works by this author in OpenAlex "
              f"(library holds {len(own_ids)} attributed)")

    # CORROBORATION IS THE DEFAULT. A single accepted OpenAlex author id can
    # still be a merged entity. **An unflagged name is not an unambiguous one**;
    # it only means the ambiguity did not show up in the anchoring counts. The
    # recall trade-off from corroborating everyone is visible in `status`, while
    # contamination is not.
    if corroborate is None:
        corroborate = True
    n_before = len(own_works)
    if corroborate:
        collabs = known_collaborators(library, own_ids, person)
        # TWO shared authors, not one. The co-author key is `surname|initial`,
        # and on common surnames a single match is within coincidence. Two is a
        # working relationship. Labs are stable, so a real paper by a seat
        # almost always shares at least two authors with their verified corpus;
        # the exception -- a first paper with an entirely new group -- is the
        # price, and it is visible.
        min_shared = 2
        keep_ids = {w["oa_id"] for w in _corroborated(
            _slim_many(own_works, "authored (OpenAlex)"), collabs,
            min_shared)}
        # ... and then the institution, which is the only signal that separates
        # two scientists who share both a name and a community.
        by_affiliation = {(w.get("id") or "").rsplit("/", 1)[-1] for w in own_works
                          if _affiliation_matches(w, person.key, person.affiliations)}
        n_affil_dropped = sum(1 for w in own_works
                              if (w.get("id") or "").rsplit("/", 1)[-1] in keep_ids
                              and (w.get("id") or "").rsplit("/", 1)[-1]
                              not in by_affiliation)
        keep_ids &= by_affiliation
        # ... and, for a name the spec DECLARES ambiguous, the works for which
        # OpenAlex records no institution at all. The affiliation check abstains
        # on those by design -- an absent record is not a contradicting one --
        # and that abstention is precisely the hole a merged author entity can
        # arrive through. These works must share two FULL co-author names with
        # the works that DID prove their affiliation.
        n_unaffiliated_dropped = 0
        if person.ambiguous_name and person.affiliations:
            seed = _affiliation_seeded_coauthors(
                own_works, person.key, person.affiliations)
            vouched = set()
            for w in own_works:
                oid = (w.get("id") or "").rsplit("/", 1)[-1]
                if _institutions_of(w, person.key):
                    vouched.add(oid)
                    continue
                shared = sum(1 for a in (w.get("authorships") or [])
                             if _full_name_key(
                                 ((a.get("author") or {}).get("display_name") or ""))
                             in seed)
                if shared >= 2:
                    vouched.add(oid)
            n_unaffiliated_dropped = len(keep_ids - vouched)
            keep_ids &= vouched
        evidence["dropped_unaffiliated_uncorroborated"] = n_unaffiliated_dropped
        # ... and finally the research topic. A merged author entity's foreign
        # work survives every name and co-author test when the two scientists
        # publish in overlapping communities; it does not survive being asked
        # whether the paper is in the same FIELD as the papers we have already
        # verified as this person's.
        anchors = set(evidence.get("anchor_topics") or [])
        n_topic_dropped = 0
        if anchors:
            in_field = {(w.get("id") or "").rsplit("/", 1)[-1] for w in own_works
                        if any((t.get("id") or "") in anchors
                               for t in (w.get("topics") or [])[:3])
                        or not (w.get("topics") or [])}
            n_topic_dropped = len(keep_ids - in_field)
            keep_ids &= in_field
        # The RAW records are filtered, not just the slim ones: everything
        # downstream -- what this scientist cites, who they publish with, which
        # topics are theirs -- is computed from these, and a different Sam
        # Okoro's reference list would otherwise become part of what "Sam Okoro
        # reads".
        own_works = [w for w in own_works
                     if (w.get("id") or "").rsplit("/", 1)[-1] in keep_ids]
        evidence["dropped_by_affiliation"] = n_affil_dropped
        evidence["dropped_by_topic"] = n_topic_dropped
        if verbose:
            print(f"  corroboration: {len(own_works)} of {n_before} kept "
                  f"(>= {min_shared} shared author(s) with the VERIFIED corpus; "
                  f"{len(collabs)} known collaborators"
                  + (f"; {n_affil_dropped} dropped on declared affiliation"
                     if person.affiliations else
                     "; NO affiliations declared, so that check is off")
                  + (f"; {n_unaffiliated_dropped} dropped as unaffiliated and "
                     f"not vouched for by a full-name co-author"
                     if n_unaffiliated_dropped else "")
                  + f"; {n_topic_dropped} dropped as outside this scientist's "
                    f"research topics)")
    own_slim = _slim_many(own_works, "authored (OpenAlex)")
    evidence["corroborated"] = bool(corroborate)
    evidence["own_works_before_corroboration"] = n_before
    evidence["own_works_kept"] = len(own_slim)

    # -- reads: what they cite, weighted by how often ----------------------
    ref_counts: Counter = Counter()
    for w in own_works:
        for r in w.get("referenced_works") or []:
            ref_counts[r] += 1
    top_refs = [r for r, n in ref_counts.most_common(N_READS) if n >= 2] \
        or [r for r, _ in ref_counts.most_common(N_READS)]
    reads = _fetch_by_ids(oa, top_refs, "cited by this scientist repeatedly")

    # -- collaborators ------------------------------------------------------
    co: Counter = Counter()
    co_names: dict[str, str] = {}
    for w in own_works:
        for a in w.get("authorships") or []:
            author = a.get("author") or {}
            aid = author.get("id")
            if aid and aid not in author_ids:
                co[aid] += 1
                co_names[aid] = author.get("display_name") or ""
    collaborators = [aid for aid, _ in co.most_common(N_COLLABORATORS)]
    collab_works: list[dict] = []
    for aid in collaborators:
        ws = oa.works(f"author.id:{aid.rsplit('/', 1)[-1]}",
                      max_records=N_PER_COLLABORATOR)
        collab_works += _slim_many(ws, f"collaborator: {co_names[aid]}")

    # -- cited_by: the field's reply ---------------------------------------
    anchor_works = sorted(own_works, key=lambda w: -w.get("cited_by_count", 0))[:10]
    cited_by: list[dict] = []
    for w in anchor_works:
        wid = (w.get("id") or "").rsplit("/", 1)[-1]
        if not wid:
            continue
        ws = oa.works(f"cites:{wid}", max_records=max(8, N_CITED_BY // 10))
        cited_by += _slim_many(ws, f"cites their {w.get('publication_year')} work")

    # -- canon: the topics they work in, most-cited first ------------------
    topics: Counter = Counter()
    topic_names: dict[str, str] = {}
    for w in own_works:
        for t in (w.get("topics") or [])[:2]:
            if t.get("id"):
                topics[t["id"]] += 1
                topic_names[t["id"]] = t.get("display_name") or ""
    canon: list[dict] = []
    for tid, _ in topics.most_common(N_CANON_TOPICS):
        ws = oa.works(f"topics.id:{tid.rsplit('/', 1)[-1]}",
                      max_records=N_PER_CANON_TOPIC)
        canon += _slim_many(ws, f"most-cited in topic: {topic_names[tid]}")

    held = _held_dois()
    nodes = [
        {"id": f"{person.name}.authored_oa",
         "kind": "own",
         # Tagged with the seat's DECLARED own topic as well as its own node id.
         # Without that, these documents are acquired and then never considered
         # for the own corpus, because `attribute_own_corpus` looks up
         # candidates by the topics named in the spec -- the papers would be on
         # the shelves and the seat would still report fifteen.
         "topics": list(person.own_topics),
         "note": "candidates for the OWN corpus; adjudicated by "
                 "attribute_own_corpus against the stored author list",
         "works": own_slim},
        {"id": f"{person.name}.reads", "kind": "neighbourhood",
         "note": "cited repeatedly in this scientist's own reference lists",
         "works": reads},
        {"id": f"{person.name}.collaborators", "kind": "neighbourhood",
         "note": "works of the people they publish with",
         "works": collab_works},
        {"id": f"{person.name}.cited_by", "kind": "neighbourhood",
         "note": "the most-cited works that cite them",
         "works": cited_by},
        {"id": "field.canon", "kind": "field",
         "note": "SUBJECTIVE. Most-cited works in this seat's topics; a proxy "
                 "for 'important' that favours the old, the review and the "
                 "consortium paper. Tagged so it can be removed.",
         "works": canon},
    ]
    for n in nodes:
        seen: set[str] = set()
        unique = []
        for w in n["works"]:
            k = w["doi"] or w["oa_id"]
            if k and k not in seen:
                seen.add(k)
                unique.append(w)
        n["works"] = unique
        n["n_works"] = len(unique)
        n["n_new"] = sum(1 for w in unique if w["doi"] and w["doi"] not in held)
        n["n_fetchable"] = sum(1 for w in unique if w["doi"] or w["pmid"]
                               or w["pdf_url"])
    return {
        "version": 1,
        "seat": person.name,
        "display": person.display,
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "author_resolution": evidence,
        "nodes": nodes,
    }


def _fetch_by_ids(oa: OpenAlex, work_ids: list[str], reason: str) -> list[dict]:
    """Hydrate OpenAlex work ids in batches of 50."""
    out: list[dict] = []
    short = [w.rsplit("/", 1)[-1] for w in work_ids if w]
    for i in range(0, len(short), 50):
        chunk = short[i:i + 50]
        try:
            ws = oa.works(f"openalex_id:{'|'.join(chunk)}", sort="cited_by_count:desc",
                          max_records=len(chunk))
        except RuntimeError as exc:
            print(f"    batch failed: {exc}")
            continue
        out += _slim_many(ws, reason)
    return out


def cmd_plan(args) -> int:
    from expertwins.corpus.store import Library
    from expertwins.corpus.net import Fetcher

    NBHD.mkdir(parents=True, exist_ok=True)
    oa = OpenAlex(Fetcher(cache_dir=paths.CACHE))
    library = Library(paths.LIBRARY)

    specs = (sorted(paths.PEOPLE.glob("*.yaml")) if args.all
             else [paths.PEOPLE / f"{args.seat}.yaml"])
    for spec in specs:
        person = Individual.load(spec)
        print(f"\n=== {person.display} ({person.name}) ===")
        try:
            plan = plan_seat(person, oa, library, min_anchor=args.min_anchor,
                             top_only=args.top_author_only,
                             corroborate=(True if args.corroborate else
                                          False if args.no_corroborate else None))
        except SystemExit as exc:
            print(f"  SKIPPED: {exc}")
            continue
        out = NBHD / f"{person.name}.yaml"
        out.write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True,
                                      width=120), encoding="utf-8")
        print(f"  {'node':<34s} {'works':>6s} {'new':>6s} {'fetchable':>10s}")
        for n in plan["nodes"]:
            print(f"  {n['id']:<34s} {n['n_works']:>6d} {n['n_new']:>6d} "
                  f"{n['n_fetchable']:>10d}")
        print(f"  wrote {paths.relative(out)}")
    print("\nNothing has been downloaded. Read a plan, then:")
    print("  python ops/neighbourhood.py acquire <seat>")
    return 0


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------

def _epmc_batches(works: list[dict], size: int = 20) -> list[str]:
    """Europe PMC queries that fetch a specific list of papers.

    Batched `DOI:"..." OR EXT_ID:...` rather than one query per paper: the
    three-pass search machinery is expensive per call and Europe PMC is happy
    with an OR of twenty identifiers. Anything Europe PMC holds comes back
    through the PROVEN path -- JATS full text where it exists, the same
    full-text measurement, the same title-in-text screen.
    """
    terms: list[str] = []
    for w in works:
        if w.get("pmid"):
            terms.append(f'EXT_ID:{w["pmid"]}')
        elif w.get("doi"):
            terms.append(f'DOI:"{w["doi"]}"')
    return [" OR ".join(terms[i:i + size]) for i in range(0, len(terms), size)]


def _direct_fetch(works: list[dict], topics: list[str], library, fetcher) -> int:
    """Open-access PDFs for what Europe PMC does not hold.

    This is how the conference literature arrives: CVPR, NeurIPS and MICCAI
    papers have no PMID, and their open-access PDF is on arXiv or the publisher.
    Note that only arXiv's *API* host is rate-limited from this network; the PDF
    host answers normally, which is why this path works when
    `acquire_preprints.py` reports HTTP 429.
    """
    from expertwins.corpus.errors import Absence
    from expertwins.corpus.ids import disambiguate, make_doc_id
    from expertwins.corpus.extract import extract_pdf
    from expertwins.corpus.store import classify_fulltext
    from expertwins.corpus.models import DocMeta, DocType, FullTextStatus, Provenance

    added = 0
    for w in works:
        url = w.get("pdf_url")
        if not url:
            continue
        try:
            resp = fetcher.get(url)
            if not resp.ok or not resp.content[:4] == b"%PDF":
                continue
            year = w.get("year") or 0
            authors = w.get("authors") or []
            doc_id = make_doc_id(authors or ["anon"], year,
                                 w.get("venue") or "openalex",
                                 fallback=w["oa_id"].lower())
            # DISAMBIGUATE. `make_doc_id` reduces (first author, year, venue)
            # to a slug, and on arXiv the venue is constant, so unrelated papers
            # can collide. Without disambiguation the store may keep one text
            # while preserving both papers' topics, losing one paper and
            # attributing the other to the wrong author.
            existing = set(library.known_ids())
            if doc_id in existing:
                held = None
                try:
                    held = library.load_meta(doc_id)
                except Exception:                                 # noqa: BLE001
                    held = None
                same = bool(held) and (
                    (w.get("doi") and held.doi and
                     held.doi.lower() == w["doi"].lower())
                    or (held.title or "").strip().lower()
                    == (w.get("title") or "").strip().lower())
                if not same:
                    doc_id = disambiguate(doc_id, existing)
            doc = extract_pdf(resp.content, doc_id)
            if isinstance(doc, Absence) or not doc.passages:
                continue
            n_chars = sum(len(p.text) for p in doc.passages)
            meta = DocMeta(
                doc_id=doc_id, title=w.get("title") or "", authors=authors,
                year=year or None, pub_date=str(year or ""),
                venue=w.get("venue") or "", doc_type=DocType.PREPRINT
                if "arxiv" in url else DocType.RESEARCH_ARTICLE,
                provenance=Provenance.PREPRINT if "arxiv" in url
                else Provenance.PUBLISHED,
                doi=w.get("doi") or "", pmid=w.get("pmid") or "", url=url,
                text_sha256="pending",
                fulltext_status=classify_fulltext(len(doc.passages), n_chars,
                                                  FullTextStatus.FULL),
                n_passages=len(doc.passages), n_chars=n_chars,
                topics=topics, channel="openalex-oa")
            res = library.put(meta, doc.passages, source_bytes=resp.content)
            if str(getattr(res.outcome, "value", res.outcome)) in ("added", "replaced"):
                added += 1
                if added % 25 == 0:
                    print(f"    +{added} via open-access PDF", flush=True)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    {w.get('oa_id')}: {type(exc).__name__}: {exc}")
    return added


def cmd_acquire(args) -> int:
    plan_path = NBHD / f"{args.seat}.yaml"
    if not plan_path.exists():
        raise SystemExit(f"no plan at {paths.relative(plan_path)} -- run "
                         f"`python ops/neighbourhood.py plan {args.seat}` first")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    nodes = [n for n in plan["nodes"]
             if args.node is None or n["id"] == args.node]

    if args.dry_run:
        for n in nodes:
            print(f"{n['id']:<34s} {n['n_works']:>5d} works, "
                  f"{len(_epmc_batches(n['works']))} Europe PMC batch queries, "
                  f"{sum(1 for w in n['works'] if w.get('pdf_url'))} open-access PDFs")
            print(f"    {n['note']}")
        print("\nNothing downloaded.")
        return 0

    from expertwins.corpus.acquire.pipeline import Acquisition
    from expertwins.corpus.store import Library
    from expertwins.corpus.net import Fetcher

    root = paths.resolve(f"library_nbhd_{args.seat}")
    root.mkdir(parents=True, exist_ok=True)
    library = Library(root)
    fetcher = Fetcher(cache_dir=paths.CACHE)
    run_dir = paths.RUNS / f"acquire-{root.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    acq = Acquisition(library, fetcher, run_dir=run_dir)

    summary: dict[str, dict] = {}
    for n in nodes:
        topics = ([n["id"], f"seat:{plan['seat']}", "col:neighbourhood"]
                  + list(n.get("topics") or []))
        print(f"\n=== {n['id']}  ({n['n_works']} works) ===", flush=True)
        print(f"  {n['note']}", flush=True)
        before = acq.stats.added + acq.stats.replaced
        for i, q in enumerate(_epmc_batches(n["works"]), 1):
            try:
                acq.harvest_query(q, topics=topics, per_pass=args.per_pass,
                                  progress=False)
            except Exception as exc:                              # noqa: BLE001
                print(f"    BATCH {i} FAILED: {type(exc).__name__}: {exc}")
            if i % 5 == 0:
                print(f"    {i} batches | {acq.stats.render()}", flush=True)
        epmc = acq.stats.added + acq.stats.replaced - before
        pdfs = 0
        if not args.no_pdf:
            pdfs = _direct_fetch(n["works"], topics, library, fetcher)
        summary[n["id"]] = {"europepmc": epmc, "open_access_pdf": pdfs,
                            "planned": n["n_works"]}
        print(f"  stored {epmc + pdfs} of {n['n_works']} planned "
              f"({epmc} Europe PMC, {pdfs} open-access PDF)", flush=True)

    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {paths.relative(out)}")
    print(f"\nNext, merge and re-attribute:\n"
          f"  python ops/people.py install config/people/{args.seat}.yaml")
    print("Nothing this tool fetched is in anyone's OWN corpus until "
          "`attribute` says so.")
    return 0


def cmd_retag(args) -> int:
    """Re-derive which OA-acquired documents may carry a seat's OWN topic.

    THE REPAIR THIS EXISTS FOR. `kenji_watanabe` was acquired before corroboration was
    the default, from an OpenAlex author entity that turns out to be merged. His
    own corpus came back at 56 papers including hip arthroscopy, quantum groups
    and video compression. The documents are real, they are correctly stored,
    and the only thing wrong with them is a TOPIC -- the claim that this seat
    wrote them.

    So nothing is deleted and nothing is re-downloaded. The current
    corroborated plan says which works are the seat's; every library document tagged
    `<seat>.authored_oa` that is not in it loses that tag and the seat's own
    topic, and gains `<seat>.authored_oa_rejected` in their place. The rejection
    stays visible, because a silently removed tag is indistinguishable from a
    document that was never acquired.

    FOR A SEAT THAT DECLARES ITS NAME AMBIGUOUS, `--all-own-topics` CHECKS THE
    OTHER OWN TOPICS TOO. A contaminated paper may arrive through the seat's own
    Europe PMC query rather than through OpenAlex, so it never carries the
    `<seat>.authored_oa` tag. The tag the repair looks for can be the one tag
    the contamination does not have.

    IT IS OFF BY DEFAULT, AND THAT IS A DELIBERATE PREFERENCE FOR RECALL.
    The OpenAlex plan is not a complete bibliography, and anything it has no
    record of is treated as foreign. The standing instruction for this panel is
    that a seat holding a few papers it did not write is a smaller problem than
    a seat missing papers it did write: the first is visible to a reader and
    correctable by `attribute`, the second is invisible and silently narrows
    what the seat can say. So the wide sweep is a tool to reach for on a named,
    inspected seat, not a default.

    Re-run `attribute` afterwards: this changes what the candidates ARE, and
    attribution is what decides among them.
    """
    from expertwins.corpus.store import IngestOutcome, Library

    plan_path = NBHD / f"{args.seat}.yaml"
    if not plan_path.exists():
        raise SystemExit(f"no plan at {paths.relative(plan_path)}; run `plan` first")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    person = Individual.load(paths.PEOPLE / f"{args.seat}.yaml")
    node = next((n for n in plan["nodes"] if n["kind"] == "own"), None)
    if not node:
        raise SystemExit(f"plan for {args.seat} has no own node")

    keep_doi = {(w.get("doi") or "").lower() for w in node["works"] if w.get("doi")}
    keep_title = {(w.get("title") or "").strip().lower()
                  for w in node["works"] if w.get("title")}
    own_topics = set(person.own_topics)
    tag = f"{args.seat}.authored_oa"

    library = Library(paths.LIBRARY)
    checked = stripped = 0
    for doc_id in library.known_ids():
        try:
            meta = library.load_meta(doc_id)
        except Exception:                                         # noqa: BLE001
            continue
        topics = set(meta.topics)
        wide = person.ambiguous_name and getattr(args, "all_own_topics", False)
        if tag not in topics and not (wide and (topics & own_topics)):
            continue
        checked += 1
        doi = (meta.doi or "").lower()
        title = (meta.title or "").strip().lower()
        if (doi and doi in keep_doi) or (title and title in keep_title):
            continue
        stripped += 1
        print(f"  strip {doc_id}: {meta.title[:64]}")
        if args.dry_run:
            continue
        new_topics = sorted((topics - own_topics - {tag})
                            | {f"{tag}_rejected"})
        updated = meta.model_copy(update={"topics": new_topics})
        passages = library.load_passages(doc_id)
        source = None
        if meta.source_filename:
            p = library.doc_path(doc_id) / meta.source_filename
            if p.exists():
                source = p.read_bytes()
        library._write(updated, passages, source,                 # noqa: SLF001
                       Path(meta.source_filename).suffix or ".xml",
                       IngestOutcome.REPLACED)
    print(f"\n{args.seat}: {checked} document(s) claimed as this seat's own work"
          + (" (by any of its own topics: --all-own-topics)"
             if getattr(args, "all_own_topics", False) and person.ambiguous_name
             else f" via `{tag}`")
          + f", {stripped} no longer corroborated"
          + (" (dry run, nothing written)" if args.dry_run else ""))
    if stripped and not args.dry_run:
        print("Now:\n  python ops/people.py reindex\n"
              f"  python ops/people.py attribute config/people/{args.seat}.yaml")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="ask OpenAlex what this seat reads")
    p.add_argument("seat", nargs="?", default=None)
    p.add_argument("--all", action="store_true", help="every seat with an own corpus")
    p.add_argument("--min-anchor", type=int, default=MIN_ANCHOR,
                   help="verified own papers an OpenAlex author id must carry "
                        "before it is accepted as this person")
    p.add_argument("--top-author-only", action="store_true",
                   help="use only the best-anchored author id. Use this when "
                        "the plan reports a same-name HAZARD and the names are "
                        "not obviously the same person.")
    p.add_argument("--corroborate", action="store_true",
                   help="keep an authored work only if it shares an author with "
                        "this seat's VERIFIED corpus. On by default when the "
                        "name was ambiguous.")
    p.add_argument("--no-corroborate", action="store_true",
                   help="switch corroboration off even for an ambiguous name. "
                        "Only sensible when you have read the works yourself.")

    a = sub.add_parser("acquire", help="fetch a plan")
    a.add_argument("seat")
    a.add_argument("--node", default=None)
    a.add_argument("--per-pass", type=int, default=100)
    a.add_argument("--no-pdf", action="store_true",
                   help="Europe PMC only; skip open-access PDF fetching")
    a.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("retag", help="re-derive which acquired documents may "
                                     "carry this seat's OWN topic")
    r.add_argument("seat")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--all-own-topics", action="store_true",
                   help="for a seat whose spec says `ambiguous_name`, also "
                        "check documents claimed through its OTHER own topics, "
                        "such as a legacy Europe PMC author query. Catches "
                        "contamination that never carried `<seat>.authored_oa` "
                        "-- and WILL strip real papers the OpenAlex plan has no "
                        "record of. Run it with --dry-run first and read the "
                        "list.")

    args = ap.parse_args()
    if args.cmd == "plan" and not args.seat and not args.all:
        raise SystemExit("give a seat name or --all")
    return {"plan": cmd_plan, "acquire": cmd_acquire,
            "retag": cmd_retag}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

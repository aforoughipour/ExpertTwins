"""What an individual seat *is*, and how the machine decides it.

THE PROBLEM THIS MODULE EXISTS TO SOLVE.

Instructions alone do not differentiate seats reliably. Seats are therefore
differentiated by the evidence they hold rather than by persona text alone. A
discipline seat is built from a field's literature. A person seat is built from
five derived objects, four of which are checkable without a model:

    C   OWN CORPUS      the papers they wrote. Derived from the stored author
                        lists, NOT from the query that fetched them (see
                        `attribute_own_corpus` -- the acquisition query asserts
                        authorship; the stored author list adjudicates it, and
                        those are different sources of truth).
    L   LINEAGE         the people who formed them and the collaborators they
                        argue with. A corpus, not a biography.
    P   PROBLEM SPACE   the problems they defined or work inside. A corpus.
    R   REFUSALS        their standing objections, written as conditions under
                        which they must not agree. Each one traceable to a body
                        of their work, never to a personality sketch.
    M   MOVES           what they actually DO when handed a problem: the assay,
                        the model system, the control, the analysis. This is the
                        one that makes two computational scientists give
                        different answers to the same question.

Only R and M are prose. Everything else is set membership.

WHY TERRITORY MATTERS MORE THAN VOICE.

A person-shaped seat should answer from its own coverage area and abstain or
narrow its answer outside it. That is not a stylistic fact; it is a coverage
fact, and coverage is measurable. `territory_of` reports what fraction of a
seat's evidence for a given question comes from its own corpus, which sorts a
question into HOME / ADJACENT / FOREIGN before a token is spent. The seat is
then held to a different standard in each zone -- and a seat that answers a
FOREIGN question with the same confidence it brings to a HOME question has
stopped being the person.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import paths

# --------------------------------------------------------------------------
# Name matching. Deliberately conservative: a false ACCEPT puts someone else's
# paper into a scientist's own corpus, which corrupts the one thing that makes
# the seat that person. A false REJECT only makes the own corpus smaller, and
# is reported as a count so it can be inspected.
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def fold_name(s: str) -> str:
    """Strip diacritics, punctuation and case. 'Núñez, J.' -> 'nunez j'."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _PUNCT.sub(" ", s)
    return " ".join(s.lower().split())


@dataclass(frozen=True)
class NameKey:
    """Acceptable surname forms plus the acceptable given-name initials.

    Multi-token surnames are first-class because several of this panel's seats
    have them -- "Del Rio", "Okafor Ndubisi" -- and a matcher that assumes
    one surname token silently drops their entire own corpus. Worse, indexes
    disagree about which token is the surname: Europe PMC carries the same
    person as both `Okafor Ndubisi N` and `Ndubisi CO`. `surname_aliases` is how a
    seat declares the other forms, deliberately and visibly, rather than the
    matcher guessing.
    """

    surnames: tuple[str, ...]       # folded, space-separated; first is canonical
    initials: frozenset[str]        # lowercase, no dots

    @property
    def surname(self) -> str:
        return self.surnames[0]

    def matches(self, author: str) -> bool:
        a = fold_name(author)
        if not a:
            return False
        toks = a.split()
        for sur in self.surnames:
            st = sur.split()
            # TOKEN-BOUND, not substring. `"okoro" in "chenoweth j"` is True;
            # accepting that would quietly admit other people's papers into a
            # scientist's own corpus, which is the failure this attribution
            # step exists to prevent.
            for i in range(len(toks) - len(st) + 1):
                if toks[i:i + len(st)] == st:
                    rest = toks[:i] + toks[i + len(st):]
                    if self._initials_ok(rest):
                        return True
        return False

    def _initials_ok(self, toks: list[str]) -> bool:
        toks = [t for t in toks if t]
        if not toks:
            # Bare surname. Accept only when the seat declared no initials,
            # otherwise a bare "okoro" would swallow every Okoro in the corpus.
            return not self.initials
        got = "".join(t[0] for t in toks)
        if got in self.initials:
            return True
        # Two directions, both needed:
        #   got="ng", declared "n"      -> the record carries a middle initial
        #   got="n",  declared "chidi" -> the record carries the full given name
        return any(got.startswith(i) or i.startswith(got) for i in self.initials)


def name_keys(surname: str, initials: list[str],
              aliases: list[str] | None = None) -> NameKey:
    forms = [fold_name(surname)] + [fold_name(a) for a in (aliases or [])]
    return NameKey(
        surnames=tuple(dict.fromkeys(f for f in forms if f)),
        initials=frozenset(fold_name(i).replace(" ", "") for i in initials if i),
    )


def coauthor_keys(name: str) -> set[str]:
    """Order-agnostic identity keys for a co-author name.

    THE BUG THIS FIXES, and it made corroboration reject 400 works out of 400
    while reporting 2,372 known collaborators -- a filter that removes
    everything looks exactly like a filter that is working hard.

    The two sources spell names in opposite orders. Europe PMC stores `Okoro S`;
    OpenAlex returns `Sam Okoro`. Folding both to a string and comparing them can
    never match, so every work was "uncorroborated".

    So a name is reduced to `surname|first-initial` -- and because which token is
    the surname depends on a convention neither source declares, BOTH readings
    are produced and a match on either is accepted. That is deliberately
    permissive: this is corroboration of an identity that has already been
    anchored, not attribution of a paper, and the failure that matters here is
    the one that silently empties the set.
    """
    toks = fold_name(name).split()
    if not toks:
        return set()
    if len(toks) == 1:
        return {toks[0]}
    return {f"{toks[-1]}|{toks[0][0]}",        # "Sam Okoro"  -> okoro|h
            f"{toks[0]}|{toks[-1][0]}"}        # "Okoro S"    -> okoro|h


# --------------------------------------------------------------------------
# The seat specification
# --------------------------------------------------------------------------

#: Territory zones, decided by evidence rather than by declaration.
HOME, ADJACENT, FOREIGN = "home", "adjacent", "foreign"


@dataclass
class Individual:
    """One person-shaped seat, loaded from `config/people/<name>.yaml`."""

    name: str
    display: str
    probe: str
    min_docs: int = 40

    surname: str = ""
    initials: list[str] = field(default_factory=list)
    surname_aliases: list[str] = field(default_factory=list)
    orcid: str = ""
    affiliations: list[str] = field(default_factory=list)

    #: Does this seat share a name with another working scientist?
    #:
    #: THE FAILURE THIS EXISTS FOR. `attribute_own_corpus` asks whether the
    #: seat's name is in the stored author list, and that is the right question
    #: for `Varga LM` versus `Varga L` in a toy attribution paper. It
    #: cannot help when the other person's name is spelled identically. In that
    #: case name-only attribution has reached the end of what it can decide: a
    #: broad tag admits contaminants, while a narrow tag can also drop valid
    #: work.
    #:
    #: When this is set, a candidate must ALSO share co-authors with the seat's
    #: independently verified corpus. Declared per seat and not inferred,
    #: because it is a claim about the world -- that this name is contested --
    #: and the project's rule is that such claims are written down where they
    #: can be read and argued with.
    ambiguous_name: bool = False

    #: Acquisition node ids whose documents are CANDIDATES for the own corpus.
    #: Candidacy is asserted by the query; membership is adjudicated later.
    own_topics: list[str] = field(default_factory=list)

    retrieval_queries: list[str] = field(default_factory=list)
    own_queries: list[str] = field(default_factory=list)
    acquisition: list[dict] = field(default_factory=list)

    fatal_flaws: str = ""
    moves: str = ""
    territory: str = ""
    lineage: str = ""
    disagrees_with: list[str] = field(default_factory=list)
    kind: str = "individual"

    @property
    def key(self) -> NameKey:
        return name_keys(self.surname or self.display.split()[-1],
                         self.initials, self.surname_aliases)

    @classmethod
    def load(cls, path: Path) -> "Individual":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if "CHANGE ME" in json.dumps(raw):
            raise SystemExit(
                f"{path} still contains CHANGE ME placeholders. An unedited "
                f"template acquires nothing and then reports a seat that does "
                f"not exist as covered.")
        known = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        unknown = set(raw) - known
        if unknown:
            raise SystemExit(f"{path}: unknown key(s) {sorted(unknown)}")
        return cls(**raw)

    @classmethod
    def load_all(cls, directory: Path | None = None) -> dict[str, "Individual"]:
        d = Path(directory or paths.PEOPLE)
        return {p.stem: cls.load(p) for p in sorted(d.glob("*.yaml"))}

    def to_seat_entry(self) -> dict:
        """The row that goes into `config/seats.yaml` and drives retrieval."""
        entry = {
            "name": self.name,
            "display": self.display,
            "kind": self.kind,
            "probe": self.probe,
            "min_docs": self.min_docs,
            "queries": list(self.retrieval_queries),
            "own_queries": list(self.own_queries),
            "own_topics": list(self.own_topics),
        }
        for k in ("fatal_flaws", "moves", "territory", "lineage"):
            v = str(getattr(self, k) or "").strip()
            if v:
                entry[k] = v
        if self.disagrees_with:
            entry["disagrees_with"] = list(self.disagrees_with)
        return entry


# --------------------------------------------------------------------------
# Own-corpus attribution -- the falsification step
# --------------------------------------------------------------------------

@dataclass
class Attribution:
    seat: str
    accepted: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    @property
    def n_candidates(self) -> int:
        return len(self.accepted) + len(self.rejected)

    def to_json(self) -> dict:
        return {
            "seat": self.seat,
            "n_candidates": self.n_candidates,
            "n_accepted": len(self.accepted),
            "n_rejected": len(self.rejected),
            "accepted": sorted(self.accepted),
            # Kept in full. A rejection list is the only way to notice that a
            # matcher is too strict, and a silent rejection is indistinguishable
            # from a paper that was never fetched.
            "rejected": self.rejected,
        }


def attribute_own_corpus(person: Individual, library, index_conn,
                         corroborator: set[str] | None = None,
                         min_shared: int = 2) -> Attribution:
    """Decide which candidate documents this person actually wrote.

    THE POINT OF THE DESIGN. `AUTH:"Varga LM"` is a *claim* made by a search
    engine, and it can be wrong. The query's claim is adjudicated by a source
    that did not make it: the author list stored on disk with the document.

    This is the same cross-checking principle used elsewhere: ask a question of
    a different source of truth than the one that made the claim.

    `corroborator` IS THE SECOND ADJUDICATOR, AND IT EXISTS BECAUSE THE FIRST
    ONE RAN OUT. The name check settles `Varga L` against `Varga LM`. It cannot
    settle two researchers whose names are spelled identically: the author list
    says yes to both, correctly, and the question is simply not one an author
    list can answer.

    A working relationship can. Given a set of `coauthor_keys` drawn from this
    seat's independently verified papers, a candidate must share at least
    `min_shared` of them. TWO, not one: on common surnames a single shared
    `surname|initial` can be coincidence. Two is a working relationship.

    The tradeoff is a possible rejection of a valid paper written with an
    entirely new group, and it is paid knowingly: **a thin own corpus is visible
    in `people.py status`, and a contaminated one is invisible forever.** Every
    such rejection is recorded with its reason, so the tradeoff is countable
    rather than silent.

    Returns accepted doc_ids and, in full, the rejects with the author string
    that failed, so an over-strict matcher is visible rather than silent.
    """
    key = person.key
    candidates: set[str] = set()
    for topic in person.own_topics:
        rows = index_conn.execute(
            "SELECT doc_id FROM doc_topics WHERE topic = ?", (topic,))
        candidates |= {r[0] for r in rows}

    att = Attribution(seat=person.name)
    for doc_id in sorted(candidates):
        try:
            meta = library.load_meta(doc_id)
        except Exception as exc:                                  # noqa: BLE001
            att.rejected.append({"doc_id": doc_id, "why": f"unreadable: {exc}"})
            continue
        authors = list(getattr(meta, "authors", []) or [])
        if not any(key.matches(a) for a in authors):
            att.rejected.append({
                "doc_id": doc_id,
                "why": "declared name not in the stored author list",
                "authors": authors[:12],
                "title": getattr(meta, "title", "")[:120],
            })
            continue
        if corroborator is not None:
            # COUNT PEOPLE, NOT KEYS. `coauthor_keys` deliberately emits BOTH
            # readings of a name (`okoro|s` and `sam|o`) because neither source
            # declares which token is the surname. Intersecting the key sets and
            # taking the size therefore counts one shared colleague twice, and
            # `min_shared=2` silently degrades to "one co-author in common" --
            # which is the exact threshold that was already measured to be
            # within coincidence on a common surname.
            shared = sum(1 for a in authors
                         if not key.matches(a) and (coauthor_keys(a) & corroborator))
            if shared < min_shared:
                att.rejected.append({
                    "doc_id": doc_id,
                    "why": f"name matches, but only {shared} co-author(s) "
                           f"in common with this seat's verified corpus "
                           f"(need {min_shared}) -- this seat's name is "
                           f"declared ambiguous",
                    "authors": authors[:12],
                    "title": getattr(meta, "title", "")[:120],
                })
                continue
        att.accepted.append(doc_id)
    return att


def own_path(seat: str) -> Path:
    return paths.PEOPLE / f"{seat}.own.json"


def load_own(seat: str) -> set[str]:
    """The verified own-corpus doc_ids for a seat, or an empty set.

    Empty is a legitimate state and must NOT be conflated with "unknown": a
    discipline seat has no own corpus by construction. The caller that needs the
    distinction should test `own_path(seat).exists()`.
    """
    p = own_path(seat)
    if not p.exists():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")).get("accepted", []))


def save_own(att: Attribution) -> Path:
    p = own_path(att.seat)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(att.to_json(), indent=2), encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Territory
# --------------------------------------------------------------------------

@dataclass
class Territory:
    """Where a question sits relative to a person's own body of work."""

    zone: str
    own_fraction: float
    n_own_docs: int
    n_docs: int

    @property
    def standard(self) -> str:
        """What the seat is held to in this zone. Carried into the packet."""
        if self.zone == HOME:
            return (
                "This question is on your own territory: a substantial part of "
                "the evidence you were handed is work you authored. Answer as "
                "yourself, from your own results first, and say plainly where "
                "your own data end.")
        if self.zone == ADJACENT:
            return (
                "This question is ADJACENT to your work: you have written near "
                "it but not on it. Speak to the part that touches your own "
                "results and be explicit about the boundary. Do not extend your "
                "authority past it.")
        return (
            "This question is OUTSIDE your work. You have written nothing the "
            "retrieval could reach on it. A scientist in this position either "
            "declines, or speaks only to the narrow interface between this "
            "question and their own field and says so. Do not answer it as "
            "though it were yours -- an abstention here is the accurate answer, "
            "and it is a first-class output.")


#: Zone boundaries. These are conventions, not measurements, and they are stated
#: here rather than buried so they can be argued with and changed in one place.
HOME_FLOOR = 0.20
ADJACENT_FLOOR = 0.05


def territory_of(packet_doc_ids: set[str], own: set[str]) -> Territory:
    n = len(packet_doc_ids)
    k = len(packet_doc_ids & own)
    frac = (k / n) if n else 0.0
    zone = HOME if frac >= HOME_FLOOR else ADJACENT if frac >= ADJACENT_FLOOR else FOREIGN
    return Territory(zone=zone, own_fraction=frac, n_own_docs=k, n_docs=n)

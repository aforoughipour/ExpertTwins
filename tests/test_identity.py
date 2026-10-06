"""Tests for the identity layer: name matching, attribution, territory.

The attribution matcher is the component where a mistake is most expensive and
least visible. A false ACCEPT puts someone else's paper into a scientist's own
corpus, and from then on the seat can cite it as `own` -- which is to say, the
seat is no longer that person and nothing in the transcript says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expertwins.identity import (  # noqa: E402
    ADJACENT, FOREIGN, HOME, Attribution, Individual, attribute_own_corpus,
    coauthor_keys, fold_name, name_keys, territory_of,
)

PEOPLE = Path(__file__).resolve().parents[1] / "config" / "people"


def test_fold_name_strips_diacritics_and_punctuation():
    assert fold_name("Lefèvre, J.") == "lefevre j"
    assert fold_name("O'Neill  A.") == "o neill a"


def test_simple_surname_with_initials():
    k = name_keys("Varga", ["LM", "L", "Lazlo"])
    assert k.matches("Varga LM")
    assert k.matches("Lazlo M Varga")
    assert k.matches("Varga L")
    assert not k.matches("Varga S")
    assert not k.matches("Varga Sanjay")


def test_multi_token_surname():
    """Three seats on this panel have one. A matcher that assumes a single
    surname token silently drops their entire own corpus."""
    k = name_keys("Del Rio", ["M", "Mira"], ["Delrio"])
    assert k.matches("Del Rio M")
    assert k.matches("Mira Del Rio")
    assert k.matches("Delrio M")
    assert not k.matches("Del Rio B")


def test_inverted_surname_alias():
    """Europe PMC carries the same person as both `Okafor Ndubisi C` and
    `Ndubisi CO`. Indexes disagree about which token is the surname, so the
    other forms are DECLARED rather than guessed."""
    k = name_keys("Okafor Ndubisi", ["C", "Chidi"], ["Ndubisi"])
    assert k.matches("Okafor Ndubisi C")
    assert k.matches("Ndubisi CO")
    assert k.matches("Chidi Okafor Ndubisi")


def test_bare_surname_is_refused_when_initials_are_declared():
    """Otherwise a bare 'Okoro' swallows every Okoro in a 30,000-document corpus."""
    k = name_keys("Okoro", ["S", "Sam"])
    assert not k.matches("Okoro")
    assert k.matches("Okoro S")


def test_attribution_rejects_a_different_person_with_the_same_string():
    """`AUTH:"Varga LM"` is a claim made by a search engine.

    The stored author list adjudicates that claim and rejects a different person
    with the same surname and nearby initials.
    """

    class FakeMeta:
        def __init__(self, authors, title=""):
            self.authors, self.title = authors, title

    class FakeLibrary:
        def __init__(self, m):
            self.m = m

        def load_meta(self, doc_id):
            return self.m[doc_id]

    class FakeConn:
        def __init__(self, rows):
            self.rows = rows

        def execute(self, _sql, params):
            return [(d,) for d in self.rows.get(params[0], [])]

    from expertwins.identity import attribute_own_corpus

    person = Individual(name="rk", display="Lazlo M. Varga", probe="p",
                        surname="Varga", initials=["LM", "L"],
                        own_topics=["varga.authored"])
    lib = FakeLibrary({
        "good": FakeMeta(["Varga LM", "Reed LL"], "Terrace alignment"),
        "bad": FakeMeta(["Varga SK", "Other A"], "Something else entirely"),
    })
    conn = FakeConn({"varga.authored": ["good", "bad"]})
    att = attribute_own_corpus(person, lib, conn)
    assert att.accepted == ["good"]
    assert len(att.rejected) == 1
    assert att.rejected[0]["doc_id"] == "bad"
    # The rejection list is kept in full: a silent rejection is indistinguishable
    # from a paper that was never fetched.
    assert "authors" in att.rejected[0]


def test_attribution_json_round_trip():
    att = Attribution("x", accepted=["a", "b"], rejected=[{"doc_id": "c"}])
    j = att.to_json()
    assert j["n_accepted"] == 2 and j["n_rejected"] == 1 and j["n_candidates"] == 3


# --------------------------------------------------------------------------
# corroboration: when two living scientists spell their names identically
# --------------------------------------------------------------------------

class _FakeMeta:
    def __init__(self, authors, title=""):
        self.authors, self.title = authors, title


class _FakeLibrary:
    def __init__(self, m):
        self.m = m

    def load_meta(self, doc_id):
        return self.m[doc_id]


class _FakeConn:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, _sql, params):
        return [(d,) for d in self.rows.get(params[0], [])]


def _two_sam_okoros():
    """One name, two scientists, both spelled `Sam Okoro`.

    `voxresnet` is the medical-imaging Sam Okoro at the Peloria Institute, with his standing
    co-authors. `sinet` is the object-detection Sam Okoro, with a different
    group. The stored author list says `Sam Okoro` for both, correctly.
    """
    person = Individual(name="sam_okoro", display="Sam Okoro", probe="p",
                        surname="Okoro", initials=["S", "Sam"],
                        ambiguous_name=True, own_topics=["okoro.authored"])
    lib = _FakeLibrary({
        "voxresnet": _FakeMeta(["Sam Okoro", "Mara Pin", "Jun-Glass Reed"],
                               "VoxResNet"),
        "sinet": _FakeMeta(["Oren Vale", "Sam Okoro", "Ida Finch"],
                           "SINet: fast vehicle detection"),
    })
    conn = _FakeConn({"okoro.authored": ["voxresnet", "sinet"]})
    return person, lib, conn


def test_a_name_check_alone_cannot_separate_two_scientists_with_one_name():
    """LOCKS THE REASON THE GATE EXISTS. This is not a matcher that is too
    loose -- it is a question an author list cannot answer, and the honest
    reading of this assertion is that name-only attribution is CORRECT here and
    still wrong."""
    person, lib, conn = _two_sam_okoros()
    att = attribute_own_corpus(person, lib, conn)          # no corroborator
    assert set(att.accepted) == {"sinet", "voxresnet"}


def test_co_authors_separate_them_and_keep_the_real_paper():
    """Both directions in one assertion: the corroborator drops the contaminant
    while retaining the valid paper."""
    person, lib, conn = _two_sam_okoros()
    corroborator = (coauthor_keys("Mara Pin") | coauthor_keys("Jun-Glass Reed")
                    | coauthor_keys("Nola Wren"))
    att = attribute_own_corpus(person, lib, conn, corroborator=corroborator)
    assert att.accepted == ["voxresnet"]
    assert att.rejected[0]["doc_id"] == "sinet"
    assert "co-author" in att.rejected[0]["why"]


def test_one_shared_co_author_is_within_coincidence():
    """On a common surname, a single `surname|initial` match can be coincidence;
    two shared co-author keys indicate a working relationship."""
    person, lib, conn = _two_sam_okoros()
    only_one = coauthor_keys("Mara Pin") | coauthor_keys("Oren Vale")
    assert attribute_own_corpus(person, lib, conn, corroborator=only_one,
                                min_shared=1).accepted == ["sinet", "voxresnet"]
    assert attribute_own_corpus(person, lib, conn, corroborator=only_one,
                                min_shared=2).accepted == []


def test_the_name_check_still_runs_first():
    """Corroboration is an ADDITIONAL gate, never a replacement: a paper by a
    different surname must not be admitted by having the right co-authors."""
    person = Individual(name="sam_okoro", display="Sam Okoro", probe="p",
                        surname="Okoro", initials=["S", "Sam"],
                        ambiguous_name=True, own_topics=["okoro.authored"])
    lib = _FakeLibrary({"x": _FakeMeta(["Mara Pin", "Jun-Glass Reed"], "Not his")})
    conn = _FakeConn({"okoro.authored": ["x"]})
    att = attribute_own_corpus(person, lib, conn,
                               corroborator=coauthor_keys("Mara Pin")
                               | coauthor_keys("Jun-Glass Reed"))
    assert att.accepted == []
    assert "not in the stored author list" in att.rejected[0]["why"]


def test_co_author_keys_survive_the_two_naming_conventions():
    """Europe PMC stores `Okoro S`; OpenAlex returns `Sam Okoro`. Comparing folded
    strings can never match, and a corroborator that matches nothing looks
    exactly like one that is working hard."""
    assert coauthor_keys("Okoro S") & coauthor_keys("Sam Okoro")
    assert coauthor_keys("Pin M") & coauthor_keys("Mara Pin")
    assert not coauthor_keys("Okoro S") & coauthor_keys("Oren Vale")


def test_an_unambiguous_seat_is_not_corroborated_by_default():
    """The gate can reject valid papers from an entirely new group, so it is
    used only where the name actually is contested."""
    person = Individual(name="rk", display="Lazlo M. Varga", probe="p",
                        surname="Varga", initials=["LM", "L"],
                        own_topics=["varga.authored"])
    assert person.ambiguous_name is False


def test_territory_is_measured_not_declared():
    own = {f"o{i}" for i in range(10)}
    home = territory_of({f"o{i}" for i in range(5)} | {"x", "y"}, own)
    assert home.zone == HOME
    adj = territory_of({"o1"} | {f"x{i}" for i in range(15)}, own)
    assert adj.zone == ADJACENT
    foreign = territory_of({f"x{i}" for i in range(20)}, own)
    assert foreign.zone == FOREIGN
    assert foreign.n_own_docs == 0


def test_foreign_territory_makes_abstention_the_accurate_answer():
    t = territory_of({"x", "y", "z"}, {"o1"})
    assert "abstention here is the accurate answer" in t.standard


def test_every_shipped_person_spec_loads_and_is_complete():
    if not PEOPLE.exists():
        import pytest
        pytest.skip("config/people is absent; no shipped person specs to inspect")
    specs = sorted(p for p in PEOPLE.glob("*.yaml") if p.name != "TEMPLATE.yaml")
    assert len(specs) >= 3, "the public tree should ship at least the toy person seats"
    for p in specs:
        person = Individual.load(p)
        assert person.display and person.probe
        assert person.surname, f"{p.name}: no surname, so attribution cannot run"
        assert person.initials, f"{p.name}: no initials -- a bare surname matcher"
        assert person.own_topics, f"{p.name}: no own corpus makes it a discipline"
        assert person.retrieval_queries and person.own_queries
        assert person.acquisition
        for field in ("territory", "moves", "fatal_flaws"):
            v = str(getattr(person, field) or "")
            assert len(v) > 120, f"{p.name}: `{field}` is too thin to differentiate"
        ids = {n["id"] for n in person.acquisition}
        missing = set(person.own_topics) - ids
        assert not missing, f"{p.name}: own_topics {missing} have no acquisition node"
        # Probes are conjunctions: a long one demands every token in ONE passage.
        assert 2 <= len(person.probe.split()) <= 6, f"{p.name}: probe length"


def test_seat_entry_carries_the_differentiating_fields():
    person = Individual.load(PEOPLE / "mira_brindle.yaml")
    entry = person.to_seat_entry()
    for k in ("fatal_flaws", "moves", "territory", "own_queries", "own_topics"):
        assert entry.get(k), f"{k} would be dropped on the way into seats.yaml"


def test_unknown_key_in_a_spec_is_refused(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("name: x\ndisplay: X\nprobe: a b c\nnonsense: 1\n", encoding="utf-8")
    try:
        Individual.load(p)
    except SystemExit as exc:
        assert "nonsense" in str(exc)
    else:                                                        # pragma: no cover
        raise AssertionError("a typo'd key must not be silently ignored")

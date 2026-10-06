"""Tests for citation-neighbourhood author resolution.

The failure mode is accepting a whole OpenAlex author entity because a frequent
co-author shares a surname and an initial with the seat. Accepting one is not
accepting one paper: every paper that person ever wrote would enter the seat's
own-corpus candidates. `attribute_own_corpus` cannot catch that when the short
name really appears in those author lists.

Surname-plus-initials is the right rule for adjudicating ONE PAPER, where an
author list often carries nothing but `Okoro S`. It is the wrong rule for
adjudicating a PERSON.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expertwins.identity import name_keys  # noqa: E402
from ops.neighbourhood import (  # noqa: E402
    _affiliation_matches, _affiliation_seeded_coauthors, _collab_keys,
    _corroborated, _epmc_batches, _full_name_key, _given_name_conflict,
    _is_scholarly, _slim, _slim_many,
)


def test_a_collaborator_who_shares_your_initials_is_rejected():
    assert _given_name_conflict("Sam Okoro", "Sayeed Okoro")
    assert _given_name_conflict("Lazlo M. Varga", "Rudolf Varga")
    assert _given_name_conflict("Julia R. Marino", "Jonah-Marino Marino")


def test_an_initial_is_an_absence_of_information_not_a_disagreement():
    """`L M Varga` is how half the world indexes him. Rejecting it would throw
    away most of a real corpus to avoid a hazard that is not present."""
    assert not _given_name_conflict("Lazlo M. Varga", "L M Varga")
    assert not _given_name_conflict("Lazlo M. Varga", "LM Varga")
    assert not _given_name_conflict("Elena Marchetti", "A. Elena Marchetti")
    assert not _given_name_conflict("Sam Okoro", "Sam Okoro")


def test_the_old_rule_would_have_accepted_both_contaminants():
    """The paper-level matcher is correct for papers but too loose for author
    entities, and says yes to both of these."""
    hao = name_keys("Okoro", ["S", "Sam"])
    varga = name_keys("Varga", ["LM", "L", "Lazlo"])
    assert hao.matches("Sayeed Okoro")
    assert varga.matches("Lars Varga")


def test_a_planned_work_carries_its_author_list():
    """A document stored without authors cannot be attributed to anybody -- so
    an `authored` node whose works lost their author lists would come back
    wholly REJECTED, which reads as care rather than as lost metadata."""
    work = {"id": "https://openalex.org/W1", "display_name": "A paper",
            "publication_year": 2024, "doi": "https://doi.org/10.1/x",
            "authorships": [{"author": {"display_name": "Julia R. Marino"}},
                            {"author": {"display_name": "Fenn Alder"}}],
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123"}}
    w = _slim(work, "authored")
    assert w["authors"] == ["Julia R. Marino", "Fenn Alder"]
    assert w["doi"] == "10.1/x"          # normalised, no resolver prefix
    assert w["pmid"] == "123"


def test_identifier_batches_prefer_pmid_and_skip_the_unfetchable():
    works = [{"pmid": "1", "doi": "10.1/a"}, {"pmid": "", "doi": "10.1/b"},
             {"pmid": "", "doi": ""}]
    q = _epmc_batches(works, size=10)
    assert q == ['EXT_ID:1 OR DOI:"10.1/b"']


def test_batches_are_split_at_the_requested_size():
    works = [{"pmid": str(i), "doi": ""} for i in range(45)]
    qs = _epmc_batches(works, size=20)
    assert len(qs) == 3
    assert qs[0].count(" OR ") == 19


# --------------------------------------------------------------------------
# corroboration: the two signals that are not the name
# --------------------------------------------------------------------------

def test_collaborator_keys_survive_the_two_naming_conventions():
    """The library stores Europe PMC's `Okoro S`; OpenAlex returns `Sam Okoro`.
    Comparing folded strings can never match, so a corroborator can remove
    everything while looking like it is working hard."""
    assert _collab_keys("Okoro S") & _collab_keys("Sam Okoro")
    assert _collab_keys("Alder F") & _collab_keys("Fenn Alder")
    assert not _collab_keys("Okoro S") & _collab_keys("Oren Vale")


def test_a_work_at_the_declared_affiliation_is_kept():
    key = name_keys("Okoro", ["S", "Sam"])
    work = {"authorships": [{"author": {"display_name": "Sam Okoro"},
                             "institutions": [{"display_name":
                                               "Peloria Institute of Fieldcraft"}]}]}
    assert _affiliation_matches(work, key, ["Peloria Institute", "Peloria Institute of Fieldcraft"])


def test_a_work_at_a_different_institution_is_dropped():
    """The only signal that separates the medical-imaging Sam Okoro from the
    object-detection Sam Okoro: they share a name AND a community, so names and
    co-authors both fail, and the institution does not."""
    key = name_keys("Okoro", ["S", "Sam"])
    work = {"authorships": [{"author": {"display_name": "Sam Okoro"},
                             "institutions": [{"display_name": "North Peloria College"}]}]}
    assert not _affiliation_matches(work, key, ["Peloria Institute"])


def test_an_absent_affiliation_is_not_a_contradicting_one():
    """Older papers frequently carry no institution at all. Treating silence as
    a mismatch would delete a scientist's early career."""
    key = name_keys("Okoro", ["S", "Sam"])
    work = {"authorships": [{"author": {"display_name": "Sam Okoro"}}]}
    assert _affiliation_matches(work, key, ["Peloria Institute"])
    assert _affiliation_matches(work, key, [])      # seat declared none


def test_corroboration_can_require_more_than_one_shared_author():
    collabs = _collab_keys("Mara Pin") | _collab_keys("Jun-Glass Reed")
    one = [{"oa_id": "W1", "authors": ["Sam Okoro", "Mara Pin", "Someone Else"]}]
    assert _corroborated(one, collabs, min_shared=1)
    assert not _corroborated(one, collabs, min_shared=2)


# --------------------------------------------------------------------------
# full names: the strict counterpart of `surname|initial`
# --------------------------------------------------------------------------

def test_a_full_name_key_ignores_the_order_and_the_punctuation():
    assert _full_name_key("Jun-Glass Reed") == _full_name_key("Reed, Jun Glass")
    assert _full_name_key("Pavo?Wren Lume") == _full_name_key("Lume, Pavo Wren")


def test_an_initial_is_not_a_name_here():
    """The whole point of this key is that it refuses the match that
    `coauthor_keys` is built to allow. `Vale O` is three different scientists in
    any toy-fieldcraft reference list, and the loose key cannot tell
    them apart, so it must not be given the chance to vouch for anybody."""
    assert _full_name_key("Vale O") == ""
    assert _full_name_key("O. Vale") == ""
    assert _full_name_key("Fenn Alder") != ""
    assert _full_name_key("Fenn Alder") != _full_name_key("Fenna Alder")


def _work(author_insts, others, insts=()):
    return {"authorships": [
        {"author": {"display_name": "Sam Okoro"},
         "institutions": [{"display_name": i} for i in insts],
         "raw_affiliation_strings": list(author_insts)},
        *({"author": {"display_name": o}} for o in others)]}


def test_the_seed_is_taken_only_from_works_that_prove_the_affiliation():
    """THE CIRCLE THIS BREAKS. A merged OpenAlex author entity holds both
    scientists' papers, so corroborating against all of its co-authors lets the
    foreign papers vouch for themselves."""
    key = name_keys("Okoro", ["S", "Sam"])
    works = [
        _work(["Peloria Institute of Fieldcraft"], ["Mara Pin", "Jun-Glass Reed"]),
        _work([], ["Oren Vale", "Ida Finch"]),          # no affiliation
        _work(["North Peloria College"], ["Fenn Alder"]),  # not declared
    ]
    seed = _affiliation_seeded_coauthors(works, key, ["Peloria Institute of Fieldcraft"])
    assert _full_name_key("Mara Pin") in seed
    assert _full_name_key("Jun-Glass Reed") in seed
    assert _full_name_key("Oren Vale") not in seed
    assert _full_name_key("Fenn Alder") not in seed


def test_a_consortium_paper_does_not_seed_the_corroborator():
    """Same reason `known_collaborators` excludes them: a hundred-author paper
    makes half the field a collaborator, and a set that large vouches for
    everything and therefore vouches for nothing."""
    key = name_keys("Okoro", ["S", "Sam"])
    crowd = _work(["Peloria Institute of Fieldcraft"],
                  [f"Author Number{i}" for i in range(40)])
    seed = _affiliation_seeded_coauthors([crowd], key,
                                         ["Peloria Institute of Fieldcraft"])
    assert seed == set()


def test_the_seat_itself_is_never_its_own_corroborator():
    key = name_keys("Okoro", ["S", "Sam"])
    w = _work(["Peloria Institute of Fieldcraft"], ["Sam Okoro", "Mara Pin"])
    seed = _affiliation_seeded_coauthors([w], key, ["Peloria Institute of Fieldcraft"])
    assert _full_name_key("Sam Okoro") not in seed
    assert _full_name_key("Mara Pin") in seed


# --------------------------------------------------------------------------
# record type: an output record is not a bibliography
# --------------------------------------------------------------------------

def test_a_meeting_abstract_is_not_a_paper():
    """THE DEFECT THIS LOCKS, and it presented as its own opposite.
    `mira_del_rio.authored_oa` planned 145 works and stored 32, which reads
    as a broken acquisition. It was a working acquisition asked for the wrong
    things: 95 of his 194 OpenAlex records are `other` (Peloria workshop abstracts in
    make-believe proceedings), 23 `dataset` and 16 `supplementary-materials`,
    against 19 `article`. Fetching a 200-word abstract is worse than not
    fetching it -- it enters the corpus as a document and can be quoted as if it
    were the study."""
    assert not _is_scholarly({"type": "conference-abstract"})
    assert not _is_scholarly({"type": "other"})
    assert not _is_scholarly({"type": "dataset"})
    assert not _is_scholarly({"type": "supplementary-materials"})
    assert not _is_scholarly({"type": "erratum"})
    assert not _is_scholarly({"type": "paratext"})
    assert not _is_scholarly({"type": "peer-review"})
    assert not _is_scholarly({"type": "retraction"})


def test_the_literature_a_computational_seat_actually_publishes_in_is_kept():
    """CVPR, NeurIPS and MICCAI are `conference-paper`, and arXiv is `preprint`.
    These are precisely the records Europe PMC cannot see and the reason the
    computational seats were thin, so a type filter that dropped them would
    reintroduce the problem it was written to help with."""
    for t in ("article", "review", "preprint", "conference-paper",
              "book-chapter", "report", "data-paper"):
        assert _is_scholarly({"type": t}), t


def test_a_missing_type_is_an_absence_of_information():
    """Same rule as an initial in `_given_name_conflict`: silence is not a
    disagreement, so a record OpenAlex has not typed is carried, not dropped."""
    assert _is_scholarly({})
    assert _is_scholarly({"type": ""})


def test_the_type_is_recorded_in_the_plan_so_the_filter_is_inspectable():
    """A filter whose decisions leave no trace is indistinguishable from a
    source that never held the records."""
    w = _slim({"id": "https://openalex.org/W1", "display_name": "A paper",
               "type": "conference-paper"}, "authored")
    assert w["type"] == "conference-paper"


def test_slimming_a_page_drops_the_non_papers():
    page = [{"id": "https://openalex.org/W1", "display_name": "Real study",
             "type": "article"},
            {"id": "https://openalex.org/W2",
             "display_name": "Figure S15 from A Pan-Peloria ...",
             "type": "supplementary-materials"},
            {"id": "https://openalex.org/W3",
             "display_name": "Abstract 5407: A pan-Peloria atlas ...",
             "type": "other"}]
    kept = _slim_many(page, "authored")
    assert [w["title"] for w in kept] == ["Real study"]

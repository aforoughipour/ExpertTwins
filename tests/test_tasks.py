"""Tests for packet assembly.

The packet is the only thing a seat ever sees. A defect here is invisible in the
code and catastrophic in the output: a malformed JSON schema in the prompt
produces malformed JSON in the response, which `ingest` reports as UNPARSEABLE
for every seat at once and which looks like a model problem rather than a
template problem.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expertwins import guardrail as gr  # noqa: E402
from expertwins import tasks  # noqa: E402


def build(task=tasks.ROUNDTABLE, **kw):
    args = dict(task=task, display="Elena Marchetti", question="Why do they relapse?",
                evidence="### [d1] A paper (2020)\nSome text.",
                moves="Start from human tissue.",
                territory_desc="Human lanternmoss toy fieldcraft.",
                fatal_flaws="A mouse result asserted for humans.",
                territory_block="TERRITORY BLOCK", material="MATERIAL")
    args.update(kw)
    return tasks.build_packet(**args)


def test_no_unescaped_format_braces_leak_into_the_packet():
    """THE BUG THIS EXISTS FOR.

    The output contracts are appended RAW but were originally written with
    `.format()`-escaped braces, so every packet carried a JSON schema reading
    `{{"claims": [...]}}`. Caught by looking at a generated packet, not by
    reading the code.
    """
    for task in tasks.TASKS.values():
        p = build(task=task, stances=True)
        assert "{{" not in p, f"{task.key}: doubled opening brace"
        assert "}}" not in p, f"{task.key}: doubled closing brace"


def test_the_json_skeleton_in_each_contract_is_parseable_shape():
    """Braces must balance across every contract. They will not if a template
    is half de-escaped, which is precisely how the bug above would come back."""
    for task in tasks.TASKS.values():
        out = task.output
        assert out.count("{") == out.count("}"), f"{task.key}: unbalanced braces"
        assert "A JSON object:" in out
        assert '"claims"' in out and '"abstentions"' in out


def test_the_objective_inversion_comes_before_the_question():
    """Order is load-bearing: the inversion has to fight the model's default, so
    it cannot appear after the thing it is inverting."""
    p = build()
    assert p.index("NOT being asked for the best answer") < p.index("Why do they relapse?")
    assert "FAILURE of this task" in p


def test_refusals_come_before_the_question():
    """A rubric read after the answer is a rationalisation."""
    p = build()
    assert p.index("WHAT YOU REFUSE TO ACCEPT") < p.index("Why do they relapse?")


def test_evidence_comes_last_so_the_instructions_are_not_buried():
    p = build()
    assert p.index("YOUR EVIDENCE") > p.index("WHAT YOU REFUSE TO ACCEPT")


def test_every_task_carries_the_identity_and_the_enforced_rules():
    for task in tasks.TASKS.values():
        p = build(task=task)
        assert "Elena Marchetti" in p
        assert "A mouse result asserted for humans." in p
        assert "Start from human tissue." in p
        assert "THE RULES THAT ARE ENFORCED, NOT REQUESTED" in p
        assert "exact string match" in p
        assert "Paraphrase fails." in p


def test_material_reaches_only_the_tasks_that_take_it():
    assert "MATERIAL" in build(task=tasks.CRITIQUE)
    assert "MATERIAL" in build(task=tasks.ANALYZE)
    assert "MATERIAL" not in build(task=tasks.ROUNDTABLE)


def test_stances_are_only_requested_when_there_is_cross_talk():
    assert "STANCES -- REQUIRED THIS TURN" in build(task=tasks.ROUNDTABLE, stances=True)
    # A solo task has no other seats, so asking for stances would be incoherent.
    assert "STANCES -- REQUIRED THIS TURN" not in build(task=tasks.SOLO, stances=True)


def test_out_of_scope_is_offered_as_a_first_class_stance():
    """A seat declining outside its territory is being faithful, not evasive,
    and the packet has to say so or the option goes unused."""
    p = build(stances=True)
    assert "out_of_scope" in p
    assert "NOT counted as dissent" in p


def test_the_nudge_never_tells_a_seat_to_disagree():
    p = build(nudge=gr.nudge_block(gr.Move.ASSIGN_DISSENT))
    assert "NOT an instruction to disagree" in p
    assert "Manufactured disagreement is worse" in p
    # And no other rung reaches the prompt at all.
    assert gr.nudge_block(gr.Move.REANCHOR) == ""


def test_analyze_asks_for_the_road_not_taken():
    """The whole value of asking two computational scientists is the difference
    between them, so each is made to state what the other would do."""
    p = build(task=tasks.ANALYZE)
    assert "what_another_expert_would_do_instead" in p
    assert "what_would_make_me_abandon_this_step" in p


def test_present_flags_a_deck_made_of_other_peoples_work():
    p = build(task=tasks.PRESENT)
    assert "somebody else's" in p


def test_prior_context_withholds_reasoning_but_discloses_cited_documents():
    """THE CONTRACT CHANGED, AND NARROWED RATHER THAN LOOSENED.

    It used to say a seat is *not shown their evidence*. That was true and it
    also meant nobody could catch a misread citation: the quote was checked
    character-exact and whether it supported the claim was checked by nobody.
    Now the packet withholds the other seats' packets, searches and reasoning,
    and discloses the specific documents they CITED, marked, for checking.
    """
    block = tasks.PRIOR_CONTEXT.format(turn=2, own="- my claim", others="**x** claimed:")
    assert "CLAIM TEXT ONLY" in block or "claim text only" in block
    assert "NOT shown their packets" in block
    assert "never to borrow their conclusions" in block
    assert "Agreeing to be agreeable" in block


def test_disclosure_block_is_only_added_when_asked_for():
    """A packet that asks for a field the seat cannot fill gets an invented one,
    so the block appears only when documents were actually placed under
    examination -- and never on a task with no other seats."""
    assert "citation_checks" not in build(task=tasks.ROUNDTABLE)
    assert "citation_checks" in build(task=tasks.ROUNDTABLE, disclosure=True)
    assert "citation_checks" not in build(task=tasks.SOLO, disclosure=True)


def test_disclosure_block_refuses_to_be_a_licence_to_adopt():
    """The failure mode of showing a seat another seat's evidence is herding.
    The block has to say so, or the disclosure rule becomes the collapse."""
    p = build(task=tasks.ROUNDTABLE, disclosure=True)
    assert "NOT part of your evidence base" in p
    assert "cannot_tell" in p


def test_all_six_tasks_are_registered():
    assert set(tasks.TASKS) == {"roundtable", "solo", "review", "critique",
                                "present", "analyze"}
    for key, task in tasks.TASKS.items():
        assert task.key == key
        assert task.instruction and task.output and task.summary

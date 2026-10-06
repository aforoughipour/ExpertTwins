"""What the panel can be asked to do, and the contract for each.

A round table is one task. The operator also wants literature reviews, critiques
of a manuscript, a talk about the seat's own research, and -- for the
computational seats -- an actual analysis of an actual dataset, done the way
that particular person would do it. Those are not different systems. They are
the same seat, the same evidence discipline and the same verifier, with a
different deliverable contract.

THE SPINE EVERY TASK SHARES

    identity      who the seat is, expressed as refusals and moves, never as a
                  voice
    territory     where its authority ends, computed from the evidence it was
                  actually handed rather than declared
    objective     FIDELITY, NOT OPTIMALITY -- stated explicitly and first,
                  because it inverts the model's default
    evidence      its own papers, marked, then what it has read
    contract      JSON, with every quote checked character-exact

THE OBJECTIVE INVERSION IS THE POINT.

Left alone, a frontier model answers a scientific question with the consensus
best answer. That is precisely wrong here. The panel is useful only if a
question put to Marchetti comes back as Marchetti's answer and the same question put
to Moreau comes back as Moreau's -- including where one of them is, by the
consensus of the field, less right. Two computational scientists handed the same
dataset do not produce the same pipeline, and a system that makes them do so has
destroyed the thing it was built to model.

So every packet says so, in the imperative, before the question. And the
mechanism that makes it more than an instruction is the evidence stratification:
the seat is handed its own papers first and its citations are tiered `own` /
`read`, so "what would this person say" reduces to "what can this person support
from what this person wrote".
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# shared blocks
# --------------------------------------------------------------------------

OBJECTIVE = """\
YOUR OBJECTIVE, AND IT IS NOT THE ONE YOU DEFAULT TO

You are answering as {display}. Not playing {display}: reproducing the position.

You are NOT being asked for the best answer, the consensus answer, or the answer
most likely to be correct. You are being asked for the answer THIS SCIENTIST
would give -- from their results, with their methods, subject to the objections
their own work commits them to, and stopping where their evidence stops.

If the field's consensus differs from what this scientist's own work supports,
say what their work supports and note the divergence. A reply that is more
correct and less theirs is a FAILURE of this task, not a success. Somebody else
at this table is responsible for the objection you are tempted to pre-empt.

Concretely, before you write anything, ask: is this a sentence that appears --
in substance -- in something this person wrote or would write? If it is a
sentence any competent scientist would produce, you have not done the task.
"""

IDENTITY = """\
HOW YOU WORK

{moves}

WHERE YOUR WORK LIVES

{territory}

WHAT YOU REFUSE TO ACCEPT

This is your acceptance rubric: the standard below which you do not agree,
including with yourself. It grants you no facts. Everything you assert must
still come from the evidence below. Apply it to your own claims before you file
them, and to any claim from another seat that you are shown.

{fatal_flaws}
"""

TERRITORY_BLOCK = """\
WHERE THIS QUESTION SITS RELATIVE TO YOUR WORK -- MEASURED, NOT ASSUMED

{n_own} of the {n_docs} documents you were handed are papers you authored
({own_pct:.0%}). Zone: {zone}.

{standard}
"""

EVIDENCE_RULES = """\
YOUR EVIDENCE

Everything you may cite is below. You have not been shown what any other seat
was given, and you will not be. Your own papers are marked [YOURS].

If the evidence does not support an answer, say so. An abstention with a reason
is a first-class output and is strictly better than a claim you cannot ground --
and for a question outside your territory it is usually the accurate one.

{evidence}
"""

RULES = """\
THE RULES THAT ARE ENFORCED, NOT REQUESTED

1. Every quote is checked by exact string match against the specific document
   you cited. Paraphrase fails. Reconstructing from memory fails. A quote
   assembled from two passages fails. Minimum 40 characters.
2. You may only cite doc_ids that appear above. Citing anything else is recorded
   as a provenance failure; if the document does not exist at all it is recorded
   as an invented citation.
3. Before you emit anything, re-read each quote against the passage above,
   character by character. This one step took a measured system from 27%
   unsupported citations to 15%.
4. Do not soften a disagreement to be agreeable. A panel whose members converge
   after one round has failed, and the failure is measured here every turn.
5. Do not adopt another seat's framing to reach their conclusion. If you end up
   agreeing, agree from YOUR OWN evidence and say which of your own results
   takes you there.
"""

CLAIM_SCHEMA = """\
Each claim:
{
  "claim":     "one falsifiable sentence, in your voice, from your position",
  "citations": [{"doc_id": "<id from above>",
                  "quote": "at least 40 characters, copied EXACTLY, character
                            for character, from a single passage above"}],
  "declared":  {...},
  "type":      "subgroup_effect|survival|predictive_model|
                differential_expression|mechanism|novel_synthesis|descriptive",
  "territory": "home|adjacent|foreign  -- is this inside your own work?",
  "self_challenge": "the strongest objection a hostile colleague in YOUR OWN
                     field would raise, and your answer. If you cannot answer
                     it, withdraw the claim."
}

Each abstention:
{"question_part": "...", "cause": "no_evidence|insufficient_evidence|
                                     out_of_my_scope", "detail": "..."}

DECLARED FIELDS -- checked mechanically; a missing one is a finding against you:
  subgroup_effect          n_subgroup, n_comparisons, correction, interval
  survival                 n_events, censoring, interval
  predictive_model         split_unit, external_validation, n_train, n_test
  differential_expression  n_samples, correction, effect_threshold
  mechanism                species, model_system
"""

STANCE_BLOCK = """\
STANCES -- REQUIRED THIS TURN

For each other seat's claim shown above, file a stance. This is not optional and
it is not a courtesy: an unstated position is how a table collapses without
anyone noticing.

"stances": [
  {"seat": "<other seat>", "claim": "<their claim, quoted or abbreviated>",
    "stance": "agree|disagree|out_of_scope|needs_other_evidence",
    "why": "one sentence, grounded in YOUR evidence or in YOUR refusals"}
]

  agree                 you would make this claim yourself
  disagree              your evidence or your refusals contradict it
  out_of_scope          not your field; you have no standing. Use this freely --
                        it is the honest answer more often than it is used, and
                        it is NOT counted as dissent
  needs_other_evidence  it may be true but the support shown cannot establish it
"""


DISCLOSURE_BLOCK = """\
CITATION CHECKS -- REQUIRED THIS TURN

You were handed the documents other seats cited, with their quoted passage
marked. File a verdict on each one. This is the disclosure obligation of this
table: a citation is a public assertion about what a paper says, and it is only
worth anything if a colleague can open the paper and disagree.

"citation_checks": [
  {"doc_id": "<the document under examination>",
    "cited_by": "<the seat that cited it>",
    "verdict": "supports|overstated|misread|irrelevant|cannot_tell",
    "why": "one sentence. For anything other than `supports`, say what the
            passage actually establishes and where the claim goes beyond it.",
    "citations": [{"doc_id": "...", "quote": "..."}]}
]

  supports      the passage establishes what it was used to establish
  overstated    the passage is real and relevant, and weaker than the claim it
                was made to carry -- the most common honest finding
  misread       the passage says something else, or the opposite
  irrelevant    the passage does not bear on the claim at all
  cannot_tell   outside your competence to judge. Say this freely; a confident
                verdict on a method you do not use is worth less than nothing

Quotes in `citations` are verified exactly as every other quote is, so quote the
passage you are judging rather than describing it. A verdict of `supports` needs
no quote; anything else does, or it is an opinion about a paper rather than a
reading of one.

DO NOT let this turn your answer into a review of other people's evidence. Your
own claims come first and come from your own evidence. A document under
examination is NOT part of your evidence base, and citing it as support for a
position of your own is the herding this table is built to prevent.
"""


@dataclass
class Task:
    key: str
    summary: str
    instruction: str
    output: str
    #: Does this task hand the seat other seats' claims?
    cross_talk: bool = True
    #: Does this task ask the seat to write files as well as claims?
    artifacts: bool = False


ROUNDTABLE = Task(
    key="roundtable",
    summary="a question put to the whole panel; the product is attributed disagreement",
    instruction="""\
THE QUESTION

{question}

Answer it as yourself. Where your own work speaks to it, lead with your own
work. Where it does not, say so and speak only to the interface with your field,
or decline.
""",
    output="""\
WHAT TO PRODUCE

A JSON object: {"claims": [...], "abstentions": [...]}

""" + CLAIM_SCHEMA,
)

SOLO = Task(
    key="solo",
    summary="one seat, one question, no table",
    cross_talk=False,
    instruction="""\
THE QUESTION

{question}

You are answering alone. There is no panel to defer to and none to argue with,
which means nothing will catch an overreach except you. Mark the boundary of
your own evidence explicitly.
""",
    output="""\
WHAT TO PRODUCE

A JSON object: {"claims": [...], "abstentions": [...]}

""" + CLAIM_SCHEMA,
)

REVIEW = Task(
    key="review",
    summary="a literature review written from one scientist's vantage point",
    cross_talk=False,
    instruction="""\
THE REVIEW YOU HAVE BEEN ASKED TO WRITE

{question}

Write it as YOUR review, not as a neutral survey. A review by a named scientist
is a position: it decides what the central problem is, which results are load-
bearing, which are over-cited, and what the field is getting wrong. Two honest
reviews of one literature by two different experts disagree about the shape of
the field, and that disagreement is the value.

Organise around the problem as YOU frame it. Say which results you consider
settled, which you consider unreplicated, and which you think the field has
mis-read. Where the literature you were handed contradicts your own published
position, say so -- do not quietly adopt it.
""",
    output="""\
WHAT TO PRODUCE

A JSON object: {"claims": [...], "abstentions": [...], "outline": [...]}

"outline" is YOUR framing of the field: an ordered list of
  {"section": "...", "why_this_order": "...", "claims": [<indices into claims>]}

Every substantive statement in the review must appear as a claim with a verified
citation. The outline carries the argument; the claims carry the evidence.

""" + CLAIM_SCHEMA,
)

CRITIQUE = Task(
    key="critique",
    summary="review a manuscript, proposal or analysis plan",
    instruction="""\
THE MATERIAL YOU HAVE BEEN ASKED TO CRITIQUE

{material}

WHAT TO DO WITH IT

{question}

Review it the way you review. Apply your own refusals first -- they are the
reason you were asked. Do not produce a balanced summary: produce the objections
that YOU would raise, in the order YOU would raise them, and be explicit about
which are fatal and which are fixable.

You may not reject a claim on taste. Every objection must either (a) name a
condition from your own refusals that the material violates, or (b) cite
evidence you were handed that contradicts it.
""",
    output="""\
WHAT TO PRODUCE

A JSON object: {"claims": [...], "abstentions": [...], "objections": [...]}

"objections": [
  {"target": "the specific statement, method or figure",
    "severity": "fatal|major|minor",
    "basis": "my_refusal|evidence",
    "detail": "...",
    "citations": [{"doc_id": "...", "quote": "..."}],
    "what_would_change_my_mind": "the specific result or control that would
                                  resolve this"}
]

Each objection's citations are verified exactly as claims are.

""" + CLAIM_SCHEMA,
)

PRESENT = Task(
    key="present",
    summary="a talk about the seat's own research",
    cross_talk=False,
    instruction="""\
THE TALK

{question}

Give this talk as yourself, from your own work. You were handed your own papers,
marked [YOURS]; those are the spine of the talk and everything else is context
for them.

A scientist's talk has a shape their competitors' talks do not: it starts from
the problem THEY think is central, it uses THEIR system, and it ends at the
thing THEY think is next. Reproduce that shape. Do not give a review of the
field with your name on it.
""",
    output="""\
WHAT TO PRODUCE

A JSON object: {"claims": [...], "abstentions": [...], "slides": [...]}

"slides": [
  {"title": "...", "beats": ["...", "..."], "claims": [<indices into claims>],
    "figure": "what would be on this slide, from which paper"}
]

Every assertion on a slide must appear as a claim with a verified citation.
Slides whose claims are all tier `read` rather than `own` are somebody else's
talk -- if the whole deck is like that, say so in an abstention instead.

""" + CLAIM_SCHEMA,
)

ANALYZE = Task(
    key="analyze",
    summary="a computational analysis, done the way this scientist would do it",
    artifacts=True,
    instruction="""\
THE ANALYSIS

{question}

THE DATA AND THE ENVIRONMENT

{material}

WHAT MAKES THIS TASK DIFFERENT

Two competent computational scientists handed this dataset produce two different
pipelines, and the difference is not noise -- it is what each of them believes
about where the signal is, what the dominant confounder is, and what a result
has to survive before it is real. Reproduce YOUR difference.

Do not write the pipeline the literature would recommend. Write the pipeline YOU
would write: your unit of analysis, your split, your baseline, your ablation,
your sanity check, your failure mode. Justify every choice that another
competent person would make differently -- and where your own published work
made that choice, cite it.

You are writing code that will run on a cluster. It must be runnable: real file
paths from the environment block, explicit dependencies, deterministic seeds,
and it must fail loudly rather than silently producing a number.
""",
    output="""\
WHAT TO PRODUCE

A JSON object:
{"claims": [...], "abstentions": [...], "plan": [...], "files": [...]}

"plan": [
  {"step": "...", "why_i_do_it_this_way": "...",
    "what_another_expert_would_do_instead": "...",
    "what_would_make_me_abandon_this_step": "...",
    "claims": [<indices into claims>]}
]

"files": [
  {"path": "relative/path.py", "content": "...", "entrypoint": true|false}
]

Files are written under `runs/<run>/work/<seat>/` and are NOT verified by the
citation checker -- code is not a claim. What IS verified is the justification:
every methodological commitment in "plan" that you attribute to prior work must
appear as a claim with a verified citation.

""" + CLAIM_SCHEMA,
)

TASKS: dict[str, Task] = {t.key: t for t in
                          (ROUNDTABLE, SOLO, REVIEW, CRITIQUE, PRESENT, ANALYZE)}


PRIOR_CONTEXT = """\
THE CONVERSATION SO FAR

This is turn {turn}.

WHAT YOU SAID PREVIOUSLY (your own record, in full)
{own}

WHAT THE OTHER SEATS CLAIMED (claim text only)
You were NOT shown their packets, their searches or their reasoning. What you
WERE shown -- at the end of your evidence, in its own section -- is every
document they cited, so that their use of it can be checked. That is a
disclosure rule, not an invitation: read those documents to test their claims,
never to borrow their conclusions. If one of their claims contradicts yours, say
so plainly and ground the objection in YOUR OWN evidence.
Agreeing to be agreeable is the failure mode this table is built against.

{others}

---

"""


def build_packet(*, task: Task, display: str, question: str, evidence: str,
                 moves: str, territory_desc: str, fatal_flaws: str,
                 territory_block: str, prior: str = "", nudge: str = "",
                 material: str = "", stances: bool = False,
                 disclosure: bool = False) -> str:
    """Assemble one seat's complete, self-contained prompt.

    Order is deliberate. The objective inversion comes first because it fights
    the model's default; the refusals come before the question because a rubric
    read after the answer is a rationalisation; the evidence comes last because
    it is the longest block and the instructions must not be buried above it.
    """
    parts = [
        OBJECTIVE.format(display=display),
        IDENTITY.format(moves=moves or "(not specified)",
                        territory=territory_desc or "(not specified)",
                        fatal_flaws=fatal_flaws or "(none declared)"),
        territory_block,
    ]
    if nudge:
        parts.append(nudge)
    if prior:
        parts.append(prior)
    parts.append(task.instruction.format(question=question, material=material))
    parts.append(EVIDENCE_RULES.format(evidence=evidence))
    parts.append(task.output)
    if stances and task.cross_talk:
        # Guarded HERE and not only in the caller. A solo task has no other
        # seats, so asking for stances on them is incoherent, and a packet that
        # asks for an impossible field gets an invented one.
        parts.append(STANCE_BLOCK)
    if disclosure and task.cross_talk:
        # Same guard, same reason: asked for only when documents were actually
        # placed under examination in THIS packet.
        parts.append(DISCLOSURE_BLOCK)
    parts.append(RULES)
    return "\n\n".join(p for p in parts if p)

# 1. Concepts

This page defines the terms used by the pipeline, measurement reports, and
transcripts.

## 1.1 Motivation

ExpertTwins is designed for panels in which multiple LLM seats should preserve
independent evidence trails while answering the same scientific question.
Cross-exposure can reduce that independence by moving seats toward the same
documents, framings, or claims. Prompt-only disagreement can also confound the
identity that a seat is meant to preserve.

The system therefore changes what each seat can see before changing what it is
told. It also measures convergence before interpreting it, because converged
claims can indicate collapse, independent corroboration, or contested evidence
depending on the evidence pattern underneath them.

## 1.2 Seats

A **seat** is one panel participant. There are two kinds.

A **person seat** is modelled on a specific researcher. It is built from:

- *the seat's own corpus* — papers attributed to that researcher, confirmed by
  reading the stored author list of each fetched document rather than trusting
  the query that fetched it;
- *the citation neighbourhood* — the papers cited most often across the seat's
  own work, collaborators' papers, papers that cite that work, and field
  landmarks;
- *a specification* — the seat's territory, characteristic argumentative moves,
  and positions outside its permitted range.

A **discipline seat** (also called a field or domain seat) is modelled on a
literature rather than an individual: ecology, statistics, instrumentation. It
uses the same machinery with a different corpus-definition step. Discipline
seats are cheaper to build and are the appropriate starting point.

A seat is not a persona. The design treats evidence access, rather than prompt
phrasing alone, as the primary source of seat differentiation. Most of the
system therefore concerns corpus construction, packet construction, and
verification.

### Own versus read

Every citation a seat makes is tiered:

- `own` — the seat is citing a paper in its own corpus;
- `read` — the seat is citing something from its wider reading.

The tier distinguishes a source-corpus position from a synthesis drawn from the
wider packet. The tier is computed; it is not asserted by the model.

## 1.3 The corpus

A **library** is a directory of documents plus an SQLite full-text index. Each
document is stored as a list of **passages** — ordered, sectioned chunks of
text — together with metadata.

Two properties are derived from stored bytes and may never be asserted by a
caller:

- `fulltext_status` — whether the store actually holds full text (`full`),
  something less (`partial`), or only a record that the paper exists
  (`metadata_only`). Callers cannot declare full-text status; a declared flag
  can disagree with the stored content, so the store derives it from character
  and passage counts.
- the text hash — recomputed on write.

Hash integrity is not the same property as bibliographic integrity. A document
can hash correctly and still contain the text of a different paper that cites
it. The acquisition pipeline therefore checks that the document's own title
appears in its own text.

Note. A floor counts documents; it cannot identify which document is absent.
Integrity checks therefore test content identity as well as aggregate counts.

## 1.4 The packet

Before generation, `panel.py prepare` writes one packet per seat. A packet is a
markdown file containing the question, the seat's specification, and a
stratified selection of passages drawn from the library.

Three properties distinguish a packet from a context window of search results.

First, the packet is private. Seat A's packet is not seat B's packet. The
overlap between packets is measured and reported before generation as a prior
on the divergence available for the question.

Second, the permitted set is derived from the packet. After the packet is
written, the set of documents that the seat is allowed to cite is read back out
of the packet file. It is not declared alongside it. This prevents disagreement
between packet contents and citation permissions.

Third, packet construction is stratified rather than ranked. Retrieval runs in
layers — a precision conjunction, the seat's own corpus, the wider reachable
literature, a coverage top-up, a breadth pass, context neighbours around what
was selected, and finally documents handed over for cross-examination. Two
separate budgets govern it: how many distinct documents (breadth) and how many
passages (depth). A single ranked list collapses these into one parameter and
can spend the whole budget on one paper.

### Documents under examination

In a multi-turn panel, each seat is also handed the documents the other seats
cited, marked as such, and asked to read them. This lets a panel check work
against documents rather than only against prose.

These documents are permitted but excluded from every heterogeneity statistic.
Every seat is handed them by construction, so counting them as shared evidence
would manufacture collapse from the disclosure rule itself. They remain in the
record and stay out of the measurement.

## 1.5 Verification

A seat returns claims. Each claim carries citations. Each citation is a
`doc_id` and a quote.

`expertwins/verify.py` checks, in this order:

1. the quote is long enough to be a quote rather than a fragment;
2. the `doc_id` is in that seat's permitted set;
3. the document exists in the library;
4. the quote appears exactly in the stored text, after a canonical fold that
   repairs line-wrapped hyphenation and Unicode punctuation but changes nothing
   else;
5. the citation is tiered `own` or `read`.

Membership is checked before existence. A seat citing a real paper it was never
given is a different failure from a seat citing a paper that does not exist, and
the former can look valid in a transcript.

No model is called on this path.

## 1.6 Heterogeneity, measured five ways

A single diversity number is both gameable and ambiguous, so the axes are kept
apart and the verdict is a function of the pattern across them.

| axis | question | notes |
|---|---|---|
| `evidence` | whose paper is each seat standing on? | Jaccard over cited `doc_id`s. Backend-independent. |
| `positions` | how many genuinely distinct positions are on the table? | Order-2 Vendi Score over embedded claim text. Absolute scale. |
| `stance` | who is contradicting whom? | Declared explicitly by each seat, about each other seat. |
| `lexical` | whose words is it using? | Style channel only. Never used as a floor. |
| `herding` | who moved toward the majority between turns? | Cross-turn drift. |

### Chance correction

Raw distance is uninterpretable: 0.6 is high for seats sharing a corpus and low
for seats with disjoint ones. The set-based axes are therefore reported as a
chance-corrected **separation**:

```
S = (H_observed − H_null) / (1 − H_null)
```

where `H_null` is the same statistic recomputed after the claims have been
randomly re-dealt among the seats. Operationally, `H_null` is the value expected
after seat identities are removed. `S = 0` means indistinguishable from a single
agent; `S = 1` means maximally separated; and both ends mean the same thing
regardless of topic or corpus. The formula has the form of Cohen's κ and
Krippendorff's α, with the expected term estimated by permutation rather than
assumed.

### When an axis has no reading

If the seats hold genuinely disjoint corpora, re-dealing their citations still
yields disjoint sets. The evidence statistic can then pin at 1.0 under both the
observation and the null, and the correction would report the healthiest
possible configuration as zero separation.

Such an axis is marked `UNINFORMATIVE` and the guardrail excludes it.

Note. An axis reading `UNINFORMATIVE -- null saturated` is missing information,
not a collapse signal.

## 1.7 The band, and why turn 1 is special

Turn 1 is generated in isolation: no seat has seen any other seat's output. It
is therefore the available measurement of what these particular seats, on this
particular question, produce when they cannot influence each other.

That measurement sets the band:

```
band = [ρ_low · S₁ , ρ_high · S₁]
```

Two consequences follow:

- Turn 1 is never judged against the band it defines. Its verdict is
  `reference_turn`.
- Turn 1 cannot be re-run within the same run. A repeated first turn would no
  longer be isolated. If a different question is needed, start a new run.

It is a two-sided band rather than a floor because there are two failure modes:

- collapse — the seats converge because they read each other, not because the
  evidence converged;
- manufactured dissent — the band forces disagreement on a question where the
  appropriate result is agreement.

Absolute floors sit underneath the relative band, because if turn 1 was itself
degenerate a purely relative band would certify the degeneracy.

## 1.8 The distinction that makes the guardrail possible

| | same evidence | different evidence |
|---|---|---|
| same conclusion | collapse — intervene | independent corroboration — no intervention |
| different conclusions | arguing over one text — no intervention | healthy |

A verdict metric alone cannot distinguish the top-left cell from the top-right
cell. The evidence axis is therefore measured separately from the positions
axis.

### The verdicts

| verdict | meaning |
|---|---|
| `reference_turn` | turn 1; it defines the band and is not judged by it |
| `healthy` | inside the band |
| `independent_corroboration` | converged conclusions, disjoint evidence |
| `contested_evidence` | divergent conclusions, shared evidence |
| `collapsing` | drifting below the band |
| `collapsed` | below the absolute floors |
| `scattered` | above the band — unrelated fragments, not debate |
| `fidelity_lost` | seats have stopped preserving source identity; short-circuits everything else |
| `undetermined` | not enough informative axes to say |

`scattered` prevents noise or manufactured dissent from being reported as
health.

## 1.9 The intervention ladder

The ladder has eight rungs. Rungs that change what a seat reads or sees precede
the rung that changes what a seat is told:

| # | rung | acts on |
|---|---|---|
| 1 | `route_to_most_distant_peers` | routing |
| 2 | `withhold_the_consensus_evidence` | evidence |
| 3 | `reanchor_to_own_corpus` | evidence |
| 4 | `anonymize_the_table` | presentation |
| 5 | `isolate_from_cross_talk` | channel |
| 6 | `assign_dissent_to_the_evidenced_minority` | prompt |
| 7 | `stop_deliberating_and_report` | terminal |
| 8 | `declare_collapse` | terminal |

The next move is selected in `followup`. Retrieval, routing, presentation, or
channel changes are applied before the next packet is written. Rung 6 stamps
the turn as prompt-perturbed so it can be interpreted separately. If the ladder
is exhausted and the table is still collapsed, the run reports collapse rather
than marking a recovery.

One constraint applies across the loop: a diversity gain bought with fidelity
loss is refused. Without that constraint, the band would reward seats for
statements unsupported by the source identity it is meant to preserve.

## 1.10 The turn

```
prepare → dispatch (external, isolated, parallel) → ingest
        → followup → dispatch → ingest → … → audit → audit-ingest → render
```

At `ingest` the turn is frozen: the claim set is hashed and written to a ledger
that later turns read but never rewrite. Reports are derived from the ledger, so
a number in a transcript can be traced to the bytes it came from.

In `followup`, each seat sees:

- the new question;
- its own prior claims, from every turn so far;
- other seats' claim text only — never their packets, their searches, or their
  reasoning;
- the documents those seats cited, in a section of its evidence marked as such.

This asymmetry limits full cross-reading, the exposure channel that the
guardrail measures.

## 1.11 Audit, and the gap it closes

Verification proves that a quote exists. It does not prove that the claim built
on it is correct. A perfectly verified citation can be a misreading.

`panel.py audit` re-opens every document cited in a turn to the whole panel,
with context around each quote, and asks for nothing but the reading. Verdicts:

`supports` · `overstated` · `misread` · `irrelevant` · `cannot_tell`

plus two non-adverse exclusions:

- `unsupported_verdict` — an adverse verdict filed without a verifying quote.
  That is an opinion about a paper, not a reading of one.
- `not_under_examination` — a verdict on a document the reviewer was never
  handed. That is a verdict on memory rather than on the supplied evidence.

## 1.12 Trajectory

`render` produces the transcript. `ops/trajectory.py` reports whether the panel
is still functioning across turns.

Four failure modes are pre-specified before analysis:

| | |
|---|---|
| F1 convergence collapse | disagreement → 0 and stays there |
| F2 evidence exhaustion | new documents per turn → 0; a panel arguing from a closed set has stopped consulting the literature |
| F3 restatement | claims become near-duplicates of claims already made, including the seat's own |
| F4 grounding decay | the grounded fraction falls as seats reach past the evidence they were given |

F1 and F3 pull in opposite directions: a panel can hold disagreement at 100%
forever by repeating its opening position verbatim.

`trajectory.py` does not judge substantive claims. Every number is computed from
the frozen ledgers by set and string operations.

## Next

→ [2. Quickstart](02-quickstart.md) — local offline run procedure.

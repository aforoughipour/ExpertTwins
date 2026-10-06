# 02 — Quickstart: a complete offline round table

This page walks the toy example line by line and explains what each number
means. It calls no model and touches no network. Every reported value is
reproducible from a fixed seed, so local output can be compared directly with
the output quoted here.

The corpus content is deliberately fictional. The purpose of the exercise is to
exercise each deterministic layer of the instrument and to distinguish an
actual failure from an unfamiliar but valid diagnostic message.

The command block in the top-level [README](../README.md) gives the same
sequence without commentary.

## 0. Install

```bash
git clone https://github.com/aforoughipour/ExpertTwins.git
cd ExpertTwins
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

The base install includes PyYAML, numpy and pydantic. Everything in this
walkthrough runs on that environment. The `acquire` and `embed` extras are only
needed for an acquired corpus (see [03](03-building-a-corpus.md)). The toy
corpus is generated locally by a standard-library script. Without
`sentence-transformers`, ingest falls back to a TF-IDF embedding and still
produces every number below. The absolute values shift slightly; the verdicts
do not.

## 1. Build the toy corpus

```bash
python examples/toy/make_toy_corpus.py library --force
```

```
wrote 18 documents, 108 passages, 32960 characters
indexed library/index.sqlite
```

This writes a complete, valid library at `library/`: an append-only
`MANIFEST.jsonl`, a sharded document tree, and the SQLite index that retrieval
actually reads. It is a corpus about lanternmoss, glasswing voles and cloud kelp
on the fictional island of Peloria, written by six fictional researchers.

Three properties are relevant for later procedures:

All documents are classified `partial`, not `full`. A document is only `full`
if it has at least 6000 characters and at least 5 passages. The toy documents
are shorter than that, so they are `partial`; if the generator asserted `full`,
the metadata layer would reject it. The corpus layer derives this status from
stored text rather than accepting a caller declaration.

Every document carries a `<surname>.authored` topic. This makes the attribution
step possible offline.

The corpus contains 18 documents. Corpus-scale runs are usually much larger.
Several defaults are tuned for larger libraries and disengage here; those
messages are identified below and should not be interpreted as faults.

Validate the corpus:

```bash
python ops/doctor.py
```

`doctor.py` checks the installation, then audits the library against the files
on disk rather than trusting the index: each manifest row is checked against its
document directory (text hash, passage and character counts, full-text status,
passage identity and title), the index is compared with the manifest, and a
full-text probe confirms that the index returns the passage it was built from.
On the toy corpus every check passes except the corpus-size check, which is
expected to fail at 18 documents. Downstream results are not interpretable until
any other failure is resolved.

## 2. Work out who wrote what

```bash
python ops/people.py attribute config/people/mira_brindle.yaml
python ops/people.py attribute config/people/toma_reed.yaml
python ops/people.py attribute config/people/lio_kestrel.yaml
```

```
mira_brindle: 3 accepted, 0 rejected
```

Run attribution before packet preparation.

A seat's own corpus — the documents that seat wrote — is not inferred at
retrieval time. It is resolved once, deliberately, and written to
`config/people/<seat>.own.json`. The panel reads that file. If the file is not
there, the seat has no own corpus, every packet comes back `own 0`, every seat's
territory is `foreign`, and own-work anchoring is absent.

The run will not fail, but it is a different experiment. For this reason,
`prepare` prints the own count in its summary.

Attribution is a separate step because author matching can be wrong in a way
that is not visible downstream. Pinning the result to a reviewable JSON file
makes the selected document list explicit. See
[04-building-seats.md](04-building-seats.md) for name-collision cases on an
acquired corpus.

## 3. Prepare turn 1

```bash
python ops/panel.py prepare --question-file examples/toy/question-turn-1.md \
    --out runs/toy --seats mira_brindle,toma_reed,lio_kestrel \
    --min-docs 4 --own-floor 0
```

```
mira_brindle    51 passages   12 docs    3 own   51% coverage   home
toma_reed       80 passages   17 docs    3 own   52% coverage   adjacent
lio_kestrel     79 passages   16 docs    3 own   52% coverage   adjacent

PREFLIGHT
  packet separation   0.201
  rubric separation   0.762
  contestedness       0.000
  retention rho       0.410
```

`prepare` builds one evidence packet per seat and writes the run directory. The
four preflight quantities should be inspected before model dispatch; they
describe the question and retrieved evidence before any seat response exists.

Packet separation (0.201) measures how different the three seats' evidence is.
If this is near zero, the seats have been handed the same papers and later
disagreement is not evidence-separated. 0.201 is low; on an 18-document corpus
there is little room for packets to diverge.

Rubric separation (0.762) measures how different the seats' instructions are.
High values indicate that the seats are being assigned different roles.

Contestedness (0.000) measures whether the corpus contains disagreement about
the question. Zero means it does not. The toy corpus is internally consistent
fiction, so this is correct and expected. On an acquired corpus, zero
contestedness indicates that the corpus does not expose disagreement for the
question as written.

Retention rho (0.410) measures how much of each packet survives the budget.

`prepare` may also report:

```
precision query: (none -- too few discriminative terms)

THIN PACKET [mira_brindle]: 51% of the question's terms appear in the
evidence; missing: disagree, explains, where, found, readings, circulation,
property, plots
```

Both are expected here and neither is an error. The precision stratum of the
retriever needs enough documents for a term to be discriminative; below that
threshold it says so and disables itself rather than silently returning noise.
This is the intended behaviour.

The THIN PACKET warning is about question-term coverage, not document count:
roughly half the question's terms do not appear in the retrieved evidence. On a
larger corpus this is a signal to revise the question or the corpus. Here, most
of the missing terms are connectives (`where`, `found`, `explains`), which is
the floor reached on an 18-document fiction.

The flags `--min-docs 4 --own-floor 0` exist for exactly this situation. The
defaults are `--min-docs 10 --own-floor 60`, which a larger corpus can meet and
an 18-document toy cannot. These flags should not be carried into a
corpus-scale run. They state that the operator accepts a thin packet.

### A note on the question

The toy questions were rewritten until they overlapped the corpus vocabulary;
coverage went from 34% to 52% in the process. This is the normal working loop.
A question whose terms do not appear in the corpus retrieves little useful
evidence, and the coverage number reports that condition before model dispatch.
If `prepare` reports low coverage, revise the question or the corpus before
continuing.

(A question made entirely of stopwords raises an error rather than returning an
empty packet.)

## 4. Answer the packets — without a model

```bash
python ops/simulate.py runs/toy --turn 1 --mode distinct
```

```
wrote 3 response(s) in mode 'distinct'
THESE ARE NOT ANSWERS. This fixture tests plumbing, not science.
```

`simulate.py` is a deterministic mock seat. It reads the real packets, lifts
real quotes from them by exact span, and writes responses in the real schema. It
does not perform semantic inference. It exists so that the deterministic layer
— verification, scoring, the guardrail, the ledger, and the renderer — can be
exercised without an API key, external service, or stochastic model output.

It has three modes:

| mode | what it fabricates |
|---|---|
| `distinct` | seats that cite different documents and take different positions |
| `collapse` | seats that converge onto the same small set of documents |
| `scatter` | seats that disagree about everything, including the question |

A collapse detector that has only ever been run against a collapse has not been
tested. Run all three modes.

In a model-mediated run, this step is replaced by external model dispatch. The
framework does not call a model directly; see
[05-running-a-panel.md](05-running-a-panel.md) for the dispatch interface.

## 5. Ingest turn 1

```bash
python ops/panel.py ingest toy
```

```
VERDICT: REFERENCE_TURN

evidence     obs 0.905   null 0.926   S  0.000
lexical      obs 0.701   null 0.294   S  0.577
stance       obs 0.500   null 0.000   S  0.500
positions    obs 0.455   null 0.113   S  0.455   (VS2 1.91 of 3)

band: floor evidence 0.150 / positions 0.187   ceiling positions 0.546
TOTAL 12 claims, 12 grounded (100%)
commit 70310522d93c1283  (turn 1 frozen)
```

Exact numbers can differ with a different embedding backend. The run above used
`sentence-transformers:all-MiniLM-L6-v2`; with the tf-idf fallback the positions
axis lands elsewhere and the band moves with it. This is handled by
[per-backend floors](06-measurement.md#evidence-is-backend-independent-positions-is-not).
The verdicts are stable.

The `null` columns are sampled permutation values, so they wobble in the third
decimal between runs (the evidence null lands around 0.92 here). The observed
values and the commit hash do not: they are functions of the frozen claims.

This is the central ingest step. In order, `ingest`:

1. Verifies every quote against the corpus by exact span. A claim whose quote is
   not found in the cited document is not grounded. 12 of 12 are grounded here.
2. Scores five heterogeneity axes: evidence, lexical, positions, stance, and
   herding.
3. Compares each against its null, the distribution obtained from seats drawing
   at random from the same packets. This chance correction is why `evidence`
   scores `S 0.000` despite an observed 0.905: random re-dealing of the same
   citations gives approximately the same value. The raw observed value is not
   the measurement.
4. Reads the band: the floors and ceiling that define the admissible range.
5. Issues a verdict and freezes the turn with a commit hash.

`REFERENCE_TURN` is the correct verdict for turn 1. There is no previous turn to
measure drift against, so the guardrail records the baseline rather than judging
it. Collapse is not assessed on the first turn; the first turn establishes the
reference state.

Exit codes: `ingest` returns 0 once the turn is frozen and 1 when it declines to
act, for example when asked to re-ingest a frozen turn without `--reingest`. The
`commit <hash>` line confirms that the turn was frozen.

## 6. Turn 2

```bash
python ops/panel.py followup toy --question-file examples/toy/question-turn-2.md
python ops/simulate.py runs/toy --turn 2 --mode distinct
python ops/panel.py ingest toy --turn 2
```

`followup` builds turn 2's packets with turn 1's transcript in view, and
reports:

```
DISCLOSURE: 10 document(s) cited last turn are handed to the other seats for
checking. They are excluded from the heterogeneity measures, because a document
every seat is given is shared by construction.

packet separation   0.432
retention rho       0.479
```

Separation increased (0.201 → 0.432), which is the expected direction for a
followup: the question became sharper and the packets diverged. The excluded
documents are those now under cross-examination. A document handed to every seat
is shared by construction, and counting it as agreement would be circular.

The ingest:

```
DRIFT  novelty 0.00  herding +0.00
       dS: evidence +0.000  lexical +0.021  stance +0.000  positions +0.032

VERDICT: COLLAPSING
  - the evidence axis is UNINFORMATIVE this turn: its permutation null
    saturated ... excluded from the verdict rather than read as zero
    separation -- counting it would flag the healthiest possible
    configuration as a collapse. NOTE the cost: this is also the case where
    collapse would be hardest to see on this axis.
  - lexical separation 0.60 (style channel; reported, never a floor).
  breached: novelty 0.00 < 0.25 -- this turn restated the table rather than
    adding to it
  sequential LLR +0.00 (fires at +2.94)

TOTAL 12 claims, 12 grounded (100%)
```

Three details matter.

`evidence` is `UNINFORMATIVE`, not zero. The permutation null saturated:
these three seats hold largely disjoint corpora, so re-dealing their citations
at random *still* produces disjoint sets, and the axis can no longer tell a good
panel from a bad one. Reporting it as zero separation would flag a maximally
disjoint configuration as a collapse, so it is excluded from the verdict. The
tool also reports the limitation: this is also the case where collapse would be
hardest to see on this axis. An axis that cannot discriminate is not allowed to
vote.

`COLLAPSING` is the correct verdict. `novelty 0.00` means turn 2 said
nothing turn 1 had not already said. The mock seat is deterministic; given a
near-identical packet it restates itself. That is a genuine restatement breach,
correctly detected on fabricated content. The verdict is about the shape of the
deliberation, not its truth.

The monitor has not fired. `LLR +0.00 of +2.94` — the guardrail says
"axes below the band this turn, but the sequential monitor has not fired.
Watching. One turn at this panel size is too noisy to act on." A single degraded
turn is an observation, not a finding.

## 7. Audit the citations

```bash
python ops/panel.py audit toy --turn 2
python ops/simulate.py runs/toy --turn 2 --audit
python ops/panel.py audit-ingest toy --turn 2
```

```
mira_brindle   8 cases, 6 docs
toma_reed      8 cases, 6 docs
lio_kestrel    8 cases, 6 docs

mira_brindle   cannot_tell 2, misread 1, overstated 2, supports 3
toma_reed      cannot_tell 2, misread 1, overstated 2, supports 3
lio_kestrel    cannot_tell 2, misread 1, overstated 2, supports 3

wrote turns/2/audit/checks.json (24 verdicts)
These are JUDGEMENTS, not verifications.
```

The audit is a different kind of check from the one `ingest` performs.

`ingest` verifies mechanically: is this quote in this document? That is
decidable by exact span matching.

The audit asks something the machine cannot decide: *does the passage actually
support the claim it was cited for?* A quote can be present, exact, and
completely misrepresented. So the audit hands each seat the claims its
colleagues made, the documents cited, and the surrounding passages, and asks for
a verdict: `supports`, `overstated`, `misread`, `irrelevant`, or `cannot_tell`.

Two rules constrain the judgement:

- An adverse verdict must quote. If a claim is marked overstated, the verdict
  must point at the text that shows it. A verdict without a verifying quote is
  recorded as `unsupported_verdict` and does not count against the claim.
- A seat may only rule on material shown in the audit packet. A verdict on a
  document not in the packet is recorded as `not_under_examination`. This check
  detects a reviewer ruling from memory rather than from the supplied text.

The `--audit` mode of `simulate.py` produces all of these outcomes on purpose,
including the unsupported one, because an audit fixture that only ever files a
clean `supports` demonstrates nothing.

The tool says *"These are JUDGEMENTS, not verifications"* for the same reason it
says *"THESE ARE NOT ANSWERS"* earlier. Keeping the two categories apart in the
output is deliberate.

## 8. Render and read the trajectory

```bash
python ops/panel.py render toy
python ops/trajectory.py toy
```

`render` writes `runs/toy/transcript.md`, the human-readable record, with
grounding and audit verdicts folded in beside each claim.

```
turn  claims  grnd  docs  new  new%  novel  self-rst  herd  pos-sep  adv  move
   1      12  100%    10   10  100%    --        --    --     0.46    0  none
   2      12  100%     9    0    0%  0.00      100%  +0.00     0.49    9  none

THE FOUR FAILURE MODES
  F1 convergence collapse: not firing
  F2 evidence exhaustion: FIRING on turn 2 -- under 15% of citations were to
     documents not already cited: the panel is arguing from a closed set
  F3 restatement: FIRING on turn 2 -- novelty under 30%
  F4 grounding decay: not firing
```

F2 and F3 firing is correct here. A deterministic fixture on an 18-document
corpus exhausts its evidence and repeats itself immediately.

`trajectory.py` also prints the following interpretive notes:

```
F1 and F3 pull in OPPOSITE directions. A panel can hold disagreement at 100%
forever by repeating its opening position, and it can stay novel forever by
drifting off the question. A single line here says nothing; the four together
say something.

The thresholds are conventions, not measurements. They decide which lines are
flagged and nothing else — no gate reads them.

2 ingested turns. A trajectory needs a trajectory: with fewer than about four
turns these are single differences, not trends.

A firing mode is a fact about THIS run with THIS model on THIS question.
n = 1 is not a measurement of the design.
```

For measurement limits, see [06-measurement.md](06-measurement.md).

## 9. Run the control modes

```bash
python ops/panel.py prepare --question-file examples/toy/question-turn-1.md \
    --out runs/toy-collapse --seats mira_brindle,toma_reed,lio_kestrel \
    --min-docs 4 --own-floor 0
python ops/simulate.py runs/toy-collapse --turn 1 --mode collapse
python ops/panel.py ingest toy-collapse
```

Then the same with `--mode scatter`. Compare the three verdicts.

This control checks whether the detector distinguishes collapse from other
response shapes. A detector pointed only at the condition it was built to detect
has been demonstrated, not evaluated. The annotated comparison of all three
modes is in
[`examples/toy/README.md`](../examples/toy/README.md).

## Where to go next

- Build a real corpus → [03-building-a-corpus.md](03-building-a-corpus.md)
- Write real seats → [04-building-seats.md](04-building-seats.md)
- Wire in a model → [05-running-a-panel.md](05-running-a-panel.md)
- Understand the numbers → [06-measurement.md](06-measurement.md)
- Hand the setup to an AI agent → [07-agent-setup-guide.md](07-agent-setup-guide.md)

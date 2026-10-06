# 05 — Running a panel

This page specifies the panel execution loop: the function of each command, the
artifacts written to disk, the packet received by each seat, the required
response schema, and the repair path for invalid citations.
[02-quickstart.md](02-quickstart.md) exercises the same loop with a
deterministic fixture in place of model dispatch.

---

## The loop

```
         ┌──────────────────────────────────────────────┐
         │                                              │
  prepare ──▶ dispatch ──▶ ingest ──▶ followup ─────────┘
     │          (you)        │           │
     │                       │           └─▶ audit ─▶ audit-ingest
     │                       │
     └── preflight           └── verify, score, judge, freeze
                                                │
                                   render ──────┴────── trajectory
```

Everything except dispatch is implemented in this repository. Dispatch is
the external model call that sends a packet and records the returned JSON. See
[Dispatch](#dispatch) below.

---

## `prepare` — build the opening packets

```bash
python ops/panel.py prepare \
    --question-file questions/turn-1.md \
    --out runs/my-panel \
    --seats alice,bob,carol \
    --task roundtable
```

Builds one evidence packet per seat, writes the run directory, and reports
preflight.

| flag | meaning |
|---|---|
| `--question-file` | the question, as a file (see [Writing a question](#writing-a-question)) |
| `--out` | the run directory to create |
| `--seats` | comma-separated seat names, resolved against `config/people/` |
| `--specs` | alternate spec directory |
| `--index` | alternate SQLite index |
| `--task` | `analyze`, `critique`, `present`, `review`, `roundtable`, `solo` |
| `--material` | extra material to put in front of every seat |
| `--max-passages` | hard cap on packet size |
| `--max-seeds` | retrieval breadth |
| `--max-per-doc` | cap passages drawn from any one document |
| `--min-docs` | refuse to proceed below this many documents per packet (default 10) |
| `--own-floor` | refuse to proceed below this many own-corpus documents (default 60) |
| `--neighbours` | include the citation neighbourhood |
| `--before-year` | retrieve only from work published before this year |
| `--no-disclosure` | omit the territory disclosure block |

Three flags require additional specification.

`--own-floor` is the guard against a seat answering outside its competence.
A seat whose packet contains little of its own corpus cannot be interpreted as a
corpus-grounded seat. The default of 60 is deliberately high. Lowering it is a
methodological decision that should be recorded.

`--before-year` supports counterfactual retrieval. It convenes the panel
against the corpus available before the specified year. Retrieval is hard-cut at
the year; later documents are not reachable.

`--no-disclosure` removes the "3 of 12 documents are yours, zone HOME"
block from the packet. The block is enabled by default because a seat is
expected to distinguish own-corpus evidence from adjacent or foreign evidence.
Disable it only when the effect of that disclosure is under study.

### Preflight

Four numbers, printed before any model is called, about the question itself:

- packet separation — how different the seats' evidence is. Near zero means
  the seats received the same papers and any disagreement is difficult to
  interpret.
- rubric separation — how different the seats' instructions are. High is
  good.
- contestedness — whether the corpus disagrees about the question. Zero
  means the literature already agrees and a panel is the wrong instrument.
- retention rho — how much of each packet survived the budget.

If these diagnostics are anomalous, revise the question before dispatch.

---

## What a seat receives

A packet is a Markdown file at `runs/<run>/turns/<n>/packets/<seat>.md`. It is
plain text; read one before the first real run. In order:

1. The objective, stated adversarially: the seat is not asked for the best
   answer, the consensus answer, or the answer most likely to be correct; it is
   asked for the answer implied by that seat's corpus and rubric.
2. How the seat works — the seat's method, from its spec.
3. Where the seat's work lives — the seat's territory.
4. What the seat refuses to accept — the acceptance rubric, with the warning
   that it grants no facts. A rubric is a standard, not evidence.
5. The territory disclosure — for example, "3 of the 12 documents you were
   handed are papers you authored (25%). Zone: HOME." Measured, not asserted.
6. The question.
7. The evidence — numbered passages, each with its `doc_id`.

The packet is the entire world the seat is allowed to draw on. There is no
retrieval during the turn, no tool use, no browsing. That constraint is what
makes the quote verification meaningful: when the seat produces a quote, the
quote either is or is not in the text it was handed, and no amount of model
capability changes the answer.

---

## Dispatch

The framework does not call a model. This is deliberate, for three reasons.

Model specification. The model, version, sampling parameters, and system
prompt are part of the experimental method. They should be recorded as run
metadata rather than hidden inside the framework.

Provenance. The deterministic layer in this repository is independent of the
model provider. It verifies, scores, and judges any transcript that satisfies
the response schema, including a transcript written without an API call.

Dependency boundary. The core install is PyYAML, numpy, and pydantic. Model
provider SDKs are not installed by default.

A dispatch implementation can be minimal:

```python
from pathlib import Path

run = Path("runs/my-panel"); turn = 1
pdir = run / "turns" / str(turn) / "packets"
rdir = run / "turns" / str(turn) / "responses"
rdir.mkdir(parents=True, exist_ok=True)

for packet in sorted(pdir.glob("*.md")):
    text = packet.read_text(encoding="utf-8")
    reply = your_model_call(text)          # <- the only line that is yours
    (rdir / f"{packet.stem}.json").write_text(reply, encoding="utf-8")
```

Run the seats independently. Do not let one seat see another's reply within
a turn — cross-talk inside a turn is precisely the thing the heterogeneity
measurement is trying to detect the absence of. If dispatch introduces
cross-talk, the measurement is meaningless. Seats see each other between turns,
through the transcript, which is the channel `followup` controls.

`ops/simulate.py` is a working reference implementation of this loop with a
fixture in place of `your_model_call`.

### The response schema

One JSON object per seat, at `turns/<n>/responses/<seat>.json`:

```json
{
  "claims": [
    {
      "claim": "one falsifiable sentence, in your voice, from your position",
      "citations": [
        {"doc_id": "smith2019method", "quote": "exact text from a passage"}
      ],
      "declared": {"species": "human", "model_system": "in vivo"},
      "type": "mechanism",
      "territory": "home",
      "self_challenge": "the strongest objection to this claim"
    }
  ],
  "abstentions": [
    {"question": "the part not answered", "why": "no evidence in my packet"}
  ]
}
```

- `quote` must be exact. Verification is by span, not by similarity. A
  paraphrase fails.
- `declared` is the claim's scope — species, model system, and so on. It is
  how a mouse result gets caught being discussed as if it were a human one.
- `self_challenge` is required. A seat that cannot state the objection to
  its own claim has not finished thinking about it.
- `abstentions` are first-class. "I have no evidence for that part" is a
  valid, recorded, scored answer. A schema that cannot express abstention
  increases confabulation risk.

---

## `ingest` — verify, score, judge, freeze

```bash
python ops/panel.py ingest my-panel --turn 1
```

| flag | meaning |
|---|---|
| `--turn` | which turn (default 1) |
| `--library` | alternate library root |
| `--permutations` | null-model resamples; raise for a tighter null |
| `--reingest` | re-run over a frozen turn |

In order: verify every quote by exact span → score the five axes → compare each
against its null → read the band → issue a verdict → freeze the turn with a
commit hash.

> `ingest` returns 1 when it declines to act, for example on a turn that is
> already frozen when `--reingest` was not given. The `commit <hash>` line
> confirms that the turn was frozen.

### Repair prompts

When a citation does not verify, `ingest` writes
`turns/<n>/responses/<seat>.repair.md`:

> The following citations in your claim did not verify. You are being shown no
> new evidence. Correct the quote to the exact text of the passage you meant, or
> withdraw the claim. Do not invent support.
>
> `[quote_not_found] quire2022peloria`: the cited document exists and is
> permitted, but contains no passage with this exact text; the quote was
> altered, paraphrased or invented

Send that file back to the same seat and overwrite its response. The "no new
evidence" clause constrains repair to the packet already provided; it is not a
second retrieval step. A seat that cannot find its quote should withdraw the
claim, and withdrawal is a valid outcome.

Unrepaired claims are not deleted. They are recorded as ungrounded and counted
against the turn's grounding rate, which appears in the trajectory as F4.

---

## `followup` — the next turn

```bash
python ops/panel.py followup my-panel --question-file questions/turn-2.md
python ops/panel.py followup my-panel "a question inline"
```

Builds turn *n+1*'s packets with turn *n*'s transcript in view. Seats now see
what the others said and what verified. `--seats` changes panel membership
between turns.

Two behaviours to know:

- Documents currently under cross-examination are excluded from the
  diversity numbers. A paper cannot simultaneously be contested evidence and
  count as agreement.
- The followup question is supplied by the operator. The instrument does not
  generate it. If the panel converged, asking a sharper question is a recorded
  intervention because it changes the experiment.

Watch packet separation across turns. Rising is good.

---

## `audit` / `audit-ingest` — cross-examination

```bash
python ops/panel.py audit my-panel --turn 2 --radius 5
python ops/panel.py audit-ingest my-panel --turn 2
```

`ingest` decides a mechanical question: is this quote in this document?
The audit asks a question that exact matching cannot answer: *does the passage
actually support the
claim?* A quote can be exact and completely misrepresented.

`audit` hands each seat its colleagues' claims, the cited documents, and
`--radius` passages of surrounding context, and asks for a verdict per case:

| verdict | meaning |
|---|---|
| `supports` | the passage says what the claim says it says |
| `overstated` | the passage supports something weaker |
| `misread` | the passage does not say this |
| `irrelevant` | the passage is not about the claim |
| `cannot_tell` | the context given is not enough to decide |

Response schema, at `turns/<n>/audit/responses/<seat>.json`:

```json
{"citation_checks": [
  {"doc_id": "...", "cited_by": "...", "claim": "...",
   "verdict": "overstated", "why": "...",
   "citations": [{"doc_id": "...", "quote": "exact text"}]}
]}
```

Two rules constrain the audit:

- An adverse verdict must quote. No verifying quote ⇒ recorded as
  `unsupported_verdict`, and it does not count against the claim.
- Verdicts are limited to shown material. A verdict on a document outside
  the audit packet is recorded as `not_under_examination` — the check that
  catches a reviewer ruling on memory of a paper.

`audit-ingest` prints the tally and reports that these are judgements, not
verifications.

---

## `render` and `trajectory`

```bash
python ops/panel.py render my-panel      # -> runs/my-panel/transcript.md
python ops/trajectory.py my-panel        # add --json for machine output
```

`render` folds grounding and audit verdicts in beside each claim. `trajectory`
reports the four failure modes across turns — see
[06-measurement.md](06-measurement.md). It requires about four turns before its
output is a trend rather than a difference, and it reports that limitation.

---

## The run directory

```
runs/my-panel/
├── manifest.json              run config, seats, flags, library commit
├── transcript.md              render output
├── trajectory.json            trajectory output
├── seats/<seat>.md            resolved seat briefs
└── turns/<n>/
    ├── packets/<seat>.md      what each seat was shown
    ├── responses/<seat>.json  what each seat returned
    ├── responses/<seat>.repair.md   failed citations, if any
    ├── preflight.json         the four pre-model numbers
    ├── checks.json            per-claim verification
    ├── ledger.json            claims, citations, grounding
    ├── report.json            axes, nulls, band, verdict
    └── audit/
        ├── packets/<seat>.md
        ├── responses/<seat>.json
        ├── shown.json         exactly which documents were handed over
        └── checks.json
```

Everything needed to reconstruct, re-score, or dispute the run is on disk in
plain text and JSON. `shown.json` exists so that `not_under_examination` is
checkable after the fact rather than taken on trust.

`runs/` is gitignored. Retain run directories; they are raw experimental data.

---

## Writing a question

The question is an experimental instrument, not a prompt. Effective question
files have the following properties.

Use corpus vocabulary. `prepare` reports term coverage; if it is low,
the retriever cannot find the relevant work. Rewriting the toy questions against
the corpus vocabulary took coverage from 34% to 52%. Iterate similarly.

Lay out the competing readings explicitly. Name two or three positions
actually in circulation and ask which the seat's evidence supports, undercuts,
or leaves untouched. A question with no alternatives invites agreement.

Ask for the strongest counter-evidence in the seat's own work. This single
sub-question does more than any instruction about tone.

Give explicit permission to abstain. *"Do not answer any part you have no
evidence for. Record it as an abstention."*

Keep the scope answerable from passages. If no passage in any packet could
settle it, the seats will produce prose rather than evidence.

The toy questions in `examples/toy/` are real working examples of this shape.

---

## Operational scale

Each turn is one model call per seat, with a packet that can run to tens of
thousands of tokens, plus repair rounds, plus one audit call per seat per
audited turn. A multi-turn panel of several seats therefore requires explicit
dispatch planning.

Use the offline path to validate the corpus, seats, questions, and dispatch loop
with `ops/simulate.py` before external model dispatch.

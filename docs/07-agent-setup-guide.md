# 07 — Agent setup guide

This protocol is intended for an AI coding agent setting up ExpertTwins from a
clean clone to a working panel over a user-specified literature corpus. It is
also a complete operational checklist.

---

## Instructions to the agent

You are setting up ExpertTwins for a researcher. Work through the phases below
in order. Do not skip a phase; each phase ends in a check, and those checks
detect runs that complete while lacking interpretability.

Rules for this job:

1. Never fabricate corpus content. If a document cannot be acquired, leave
   it out and say so. A seat grounded in invented text is worse than no seat.
2. Never relax a threshold to make a check pass. If `--own-floor` or
   `--min-docs` blocks a run, the finding is that the corpus or the seat is
   thin. Report that. Do not lower the number to get past it.
3. Obtain explicit approval before external model dispatch. The offline path is
   complete; no model call is required until a real run is requested.
4. Stop and report at the end of each phase. Do not chain through all phases
   silently.

---

## Phase 0 — Verify the install

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
python -m pytest tests -q -ra
```

Check: the suite passes. A small number of skips is expected when tests require
run artifacts or seat specs that are not present in a fresh clone.

If `pip install -e .` fails, do not work around it with `PYTHONPATH`. Fix the
install; everything downstream resolves paths through the installed package.

---

## Phase 1 — Run the toy example end to end

Do this before touching the user's data. It is the fastest environment check and
uses only deterministic fixtures.

Follow [02-quickstart.md](02-quickstart.md) exactly. It calls no model and
touches no network.

Check: reach a rendered transcript and a trajectory report, and the
verdicts are `REFERENCE_TURN` on turn 1 and `COLLAPSING` on turn 2.

Then run `--mode collapse` and `--mode scatter` on fresh run directories and
confirm the verdicts differ. If all three modes produce the same verdict,
something is wrong and nothing downstream is trustworthy.

Do not proceed until this works. The toy run isolates environment and wiring
errors before a real corpus is introduced.

---

## Phase 2 — Interview the user

This phase requires information outside the repository. Ask for it and record
the answers; they become the seat specs and the question files.

The question. What is the actual scientific disagreement? Not the topic —
the *disagreement*. A good question has at least two live positions in the
literature. If the user offers a topic ("the role of X in Y"), push for the
contested version ("whether X acts through pathway A or pathway B, or whether
the association is confounded by Z").

The seats. Who is at the table, and are they people or disciplines?

- A *person* seat is grounded in one researcher's publications, speaks from
  their results, and can be checked for fidelity against their own corpus.
- A *discipline* seat is grounded in a field's literature and brings a
  methodological stance rather than a personal one.

Both are supported; they have different specs. See
[04-building-seats.md](04-building-seats.md).

The corpus. What literature grounds each seat, and can it legally be
obtained in full text? Open-access and preprint sources are the normal path; a
subscription corpus requires confirmed access rights.

Scale. How many seats, how many turns, and what compute or provider
constraints apply? Each turn is one model call per seat on a large packet, plus
repairs, plus audit calls. Confirm feasibility before promising a run.

Check: state the question in one paragraph, name every seat, and identify the
source of each seat's corpus.

---

## Phase 3 — Build the corpus

Full detail in [03-building-a-corpus.md](03-building-a-corpus.md). The shape:

```bash
python ops/people.py template <seat_name>      # write a spec skeleton
# ... fill in the spec ...
python ops/people.py resolve  <spec>           # disambiguate the author
python ops/people.py plan     <spec>           # what would be acquired
python ops/people.py acquire  <spec>           # fetch it
python ops/people.py install  <spec>           # into the library
python ops/people.py attribute <spec>          # which docs are this seat's own
python ops/people.py verify   <spec>           # check the result
```

Then:

```bash
python ops/people.py merge      # combine sources
python ops/people.py reindex    # rebuild index.sqlite
python ops/doctor.py            # re-derive every invariant from disk
```

Five checks are required.

`plan` before `acquire`, always. Read the plan. Author name collisions are
common, and the plan is where they are detected before acquisition.

Keep acquisition channels on separate roots, then merge. Mixing a canon
acquisition into a preprint root makes the provenance irrecoverable.

Run the integrity screens.

```bash
python ops/check_canon.py           # are the papers that must be here, here?
python ops/check_contamination.py   # is anything here that must not be?
```

Both are YAML-driven (`config/canon_check.yaml`, `config/contamination.yaml`).
If the config is absent they explain themselves and exit 0; absence of a
complaint is not evidence of a check. Write the configs.
`canon_check` lists the papers the corpus would be incomplete without;
`contamination` lists what must not be in it (for a counterfactual run, that
includes everything after the cut year).

`attribute` is not optional. Without `config/people/<seat>.own.json` the
seat has no own corpus, every packet reports `own 0`, territory is `foreign`,
and own-work anchoring never happens. The run will not fail. It will quietly be
uninterpretable with respect to own-corpus grounding. Open the JSON and read
the list; it is the most error-prone artifact in the pipeline.

Trust `doctor.py` over the index. It re-derives from the files. The index
can be stale; the files cannot.

Check: `doctor.py` reports the library healthy, both screens pass, and
`people.py verify` passes for every seat.

---

## Phase 4 — Write the seats

Full detail in [04-building-seats.md](04-building-seats.md). Use
`config/people/TEMPLATE.yaml` and the three worked toy examples.

Two fields often require attention.

`how_you_work` is a *method*, not a personality. "Read the primary
measurement before the interpretation; prefer effect sizes to significance" is
useful. "Rigorous and insightful" is noise that consumes tokens.

`what_you_refuse_to_accept` is the acceptance rubric — the standard below
which this seat does not agree, including with itself. It grants no facts. Make
it specific enough that it could actually reject a claim.

Then measure, do not assume:

```bash
python ops/fidelity.py holdout    <spec>   # can the seat be told from its corpus?
python ops/fidelity.py attribute  <spec>
python ops/fidelity.py centroids  <spec>
python ops/neighbourhood.py plan  <spec>   # the citation neighbourhood
```

Check: every seat has ≥ `min_docs` documents, a non-empty own corpus, and a
fidelity result the user has looked at.

---

## Phase 5 — Write the question files

One Markdown file is used per turn. The following structure is effective; the
toy questions are working examples.

1. State the disagreement.
2. Enumerate two or three positions actually in circulation, numbered.
3. Ask what the seat's own evidence establishes about its own subject.
4. Ask which position that evidence supports, undercuts, or leaves untouched.
5. Ask for the strongest passage in the seat's own evidence that argues
   against its preferred reading.
6. Give explicit permission to abstain: *"Do not answer any part you have no
   evidence for. Record it as an abstention."*

Use the corpus's vocabulary. `prepare` reports term coverage. If it is low,
the retriever cannot reach the relevant work. Rewriting the toy questions
against the corpus vocabulary moved coverage from 34% to 52%; expect to iterate.

Check: `prepare` reports acceptable coverage and the preflight numbers look
sane (see Phase 6).

---

## Phase 6 — Dry run and read the preflight

```bash
python ops/panel.py prepare --question-file questions/turn-1.md \
    --out runs/<name> --seats <a,b,c> --task roundtable
```

Read the four preflight numbers before calling any model. They are diagnostics
about the question:

| number | bad value | what it means |
|---|---|---|
| packet separation | near 0 | seats were handed the same papers; any disagreement will be theatre |
| rubric separation | low | seats were given the same job |
| contestedness | 0 | the corpus does not disagree about this; a panel is the wrong instrument |
| retention rho | — | sets the band floors |

Also read one packet in full, as text. It is the entire evidence set a seat is allowed
to draw on, and ten minutes reading it will catch problems no number reports.

Check: report the preflight numbers to the user and obtain explicit approval
before dispatch. If contestedness is zero, say so plainly; the recommendation
may be not to run.

---

## Phase 7 — Wire dispatch

The framework does not call a model, by design
([05-running-a-panel.md](05-running-a-panel.md) explains why). Write the loop:

```python
for packet in sorted((run / "turns" / str(turn) / "packets").glob("*.md")):
    reply = your_model_call(packet.read_text(encoding="utf-8"))
    (rdir / f"{packet.stem}.json").write_text(reply, encoding="utf-8")
```

Requirements:

- Run seats independently. No seat sees another's reply within a turn.
  Cross-talk inside a turn destroys the measurement by introducing the exact
  artifact the instrument is trying to detect.
- Record the model, version, and sampling parameters in the run directory.
  They are part of the method.
- Emit the response schema exactly (`claims[]` with `citations[]`, `declared`,
  `type`, `territory`, `self_challenge`; plus `abstentions[]`). Quotes must be
  exact spans — paraphrase fails verification.
- Handle repair: when `ingest` writes `responses/<seat>.repair.md`, send it back
  to the same seat and overwrite the response. The repair prompt shows no
  new evidence on purpose.

Check: `ops/simulate.py` and the dispatch loop produce responses that
`ingest` accepts identically.

---

## Phase 8 — Run, ingest, audit

```bash
python ops/panel.py ingest <name> --turn 1
# dispatch repairs if any, then:
python ops/panel.py followup <name> --question-file questions/turn-2.md
# dispatch turn 2
python ops/panel.py ingest <name> --turn 2
python ops/panel.py audit <name> --turn 2
# dispatch audit packets
python ops/panel.py audit-ingest <name> --turn 2
python ops/panel.py render <name>
python ops/trajectory.py <name>
```

> `ingest` returns 0 when the turn is frozen and 1 when it declines to act, for
> example on a turn that is already frozen when `--reingest` was not given.
> Treat a non-zero exit as a failure and read the message; the `commit <hash>`
> line confirms success.

Watch for, and report rather than route around:

- `FIDELITY_LOST` — a seat stopped sounding like itself. Everything else about
  the turn is uninterpretable until this is addressed. Do not re-run hoping for
  a different draw.
- `UNINFORMATIVE` axes — the null saturated; that axis cannot discriminate and
  is excluded from the verdict. Usually means the corpus is too small or too
  homogeneous.
- `abs_calibrated: false` in the band — the embedding backend has no calibrated
  floor, so the band is relative only and cannot detect a panel that was
  degenerate from the start. Recommend `tfidf-cosine`.
- Grounding rate falling across turns (F4) — the seats are drifting off their
  evidence.

Check: a rendered transcript exists and every verdict can be explained from the
recorded artifacts.

---

## Phase 9 — Report run conditions

When results are summarised for the user, the following must be stated, not
implied:

- Which model, version and sampling parameters produced the text.
- The embedding backend, and whether its absolute floors were calibrated.
- Which axes were uninformative, and why.
- Every intervention the ladder applied, and whether rung 6 (the prompt rung)
  was reached — disagreement produced under it is weaker evidence.
- The grounding rate and the audit tally, kept separate: verification is
  mechanical, audit verdicts are judgements.
- How many turns. Under about four, the trajectory reports single differences,
  not trends.
- That n = 1 is not a measurement of the design. A failure mode firing is a
  fact about this run with this model on this question.

Do not describe a healthy verdict as the panel being correct. It means more than
one position was present, nothing more. See
[06-measurement.md](06-measurement.md#what-these-numbers-are-not).

---

## Quick failure lookup

| symptom | cause | fix |
|---|---|---|
| `own 0` in every packet | `people.py attribute` not run | run it per seat |
| territory `foreign` everywhere | same | same |
| `prepare` refuses: too few docs | packet below `--min-docs` | acquire more; do not just lower the flag |
| `prepare` refuses: own floor | seat's own corpus too thin | acquire that seat's papers |
| "precision query: (none)" | corpus too small for discriminative terms | expected on tiny corpora; harmless |
| `THIN PACKET` warning | under half the question's terms appear in the evidence | fix the question or the corpus; expected on the toy |
| `ValueError` from the question | question is all stopwords | write a real question |
| quotes fail verification | paraphrase, not exact span | dispatch repair prompts |
| `FIDELITY_LOST` | seat anchored nothing in its own work | check `attribute`, check the spec |
| `UNDETERMINED` | too few claims or participation < 60% | check for dispatch failures |
| `ingest` exits 1, no commit printed | the turn is already frozen | use `--reingest` only if the responses changed |
| `check_canon` passes instantly | no `config/canon_check.yaml` | write one; silence is not a check |

---

## What never to do

- Do not commit `library/`, `runs/`, `*.sqlite`, or `MANIFEST.jsonl`. They are
  gitignored for good reasons: size, licensing, and the fact that run
  directories are raw experimental data.
- Do not edit a frozen turn. Re-ingest with `--reingest` if re-scoring is
  required; the commit hash exists so a reader can tell.
- Do not let one seat see another's answer within a turn.
- Do not lower a threshold to get a greener result.
- Do not present verification and audit as the same kind of evidence.

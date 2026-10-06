# The toy example

A complete ExpertTwins round table that runs in about two minutes, calls no
model, and touches no network.

The corpus is fictional: 18 invented papers about lanternmoss, glasswing voles
and cloud kelp on the island of Peloria, written by six invented researchers.
It does not encode scientific claims.

The retrieval, packet construction, exact-span quote verification, five
heterogeneity axes, permutation nulls, band, guardrail, audit, renderer and
trajectory report are the same machinery used on acquired corpora.

For a line-by-line walkthrough with commentary, see
[docs/02-quickstart.md](../../docs/02-quickstart.md). This file is the
comparison across three deterministic response modes.

## Files

| file | what it is |
|---|---|
| `make_toy_corpus.py` | deterministic, stdlib-only corpus generator (seed 1729) |
| `question-turn-1.md` | the opening question |
| `question-turn-2.md` | the followup |

The three seats live in `config/people/`: `mira_brindle.yaml`,
`toma_reed.yaml`, `lio_kestrel.yaml`. Their domain is
`config/domains/peloria.yaml`.

`make_toy_corpus.py` is also the worked example for
[writing a corpus loader](../../docs/03-building-a-corpus.md) — it writes
a valid `MANIFEST.jsonl`, a sharded document tree and the SQLite FTS index from
scratch in about 370 lines of standard library.

## Run it

```bash
python examples/toy/make_toy_corpus.py library --force
python ops/doctor.py

python ops/people.py attribute config/people/mira_brindle.yaml
python ops/people.py attribute config/people/toma_reed.yaml
python ops/people.py attribute config/people/lio_kestrel.yaml

python ops/panel.py prepare --question-file examples/toy/question-turn-1.md \
    --out runs/toy --seats mira_brindle,toma_reed,lio_kestrel \
    --min-docs 4 --own-floor 0

python ops/simulate.py runs/toy --turn 1 --mode distinct
python ops/panel.py ingest toy

python ops/panel.py followup toy --question-file examples/toy/question-turn-2.md
python ops/simulate.py runs/toy --turn 2 --mode distinct
python ops/panel.py ingest toy --turn 2

python ops/panel.py audit toy --turn 2
python ops/simulate.py runs/toy --turn 2 --audit
python ops/panel.py audit-ingest toy --turn 2

python ops/panel.py render toy
python ops/trajectory.py toy
```

Two required adjustments for the toy run:

`people.py attribute`. Without it there is no
`config/people/<seat>.own.json`, so every packet reports `own 0`, every seat's
territory is `foreign`, and own-work anchoring never happens. The command
sequence completes without failure.

`--min-docs 4 --own-floor 0`. The defaults (10 and 60) are written for a larger
corpus and an 18-document toy cannot meet them. Do not carry these flags into
corpus-scale work: they state that the operator accepts a thin packet.

Each `ingest` ends with a `commit <hash>` line, which confirms that the turn was
frozen.

## The three modes

`ops/simulate.py` is a deterministic mock seat. It reads the real packets, lifts
real quotes by exact span, and writes responses in the real schema. It has read
nothing outside the packet and performs no semantic inference; it prints "THESE
ARE NOT ANSWERS" every time it runs.

It fabricates three different *shapes* of table:

| mode | what it fabricates |
|---|---|
| `distinct` | seats citing different documents, taking different positions |
| `collapse` | seats converging onto the same small set of documents |
| `scatter` | seats disagreeing about everything, including the question |

Run all three. A collapse detector that has only ever been pointed at a collapse
has been demonstrated, not evaluated.

```bash
for MODE in distinct collapse scatter; do
  python ops/panel.py prepare --question-file examples/toy/question-turn-1.md \
      --out runs/toy-$MODE --seats mira_brindle,toma_reed,lio_kestrel \
      --min-docs 4 --own-floor 0
  python ops/simulate.py runs/toy-$MODE --turn 1 --mode $MODE
  python ops/panel.py ingest toy-$MODE
done
```

### What turn 1 looks like in each

| | `distinct` | `collapse` | `scatter` |
|---|---|---|---|
| evidence `obs` | 0.905 | 0.267 | 0.867 |
| evidence `null` | ~0.92 | 0.640 | 0.922 |
| positions `S` | 0.455 | 0.000 | 0.260 |
| VS2 (of 3) | 1.91 | 1.00 | 1.52 |
| cos-dispersion | 0.470 | 0.000 | 0.307 |
| stance `S` | 0.500 | 0.000 | 0.500 |
| grounded | 12/12 | 12/12 | 11/12 |
| verdict | `REFERENCE_TURN` | `REFERENCE_TURN` + degenerate-panel warning | `REFERENCE_TURN` |

(Numbers from `sentence-transformers:all-MiniLM-L6-v2`; the `collapse` and
`scatter` rows were measured on the tf-idf fallback. Absolute values move with
the backend. The *structure* does not. The `null` rows are sampled permutation
values and wobble slightly between runs; the observed values do not.)

### Reading the collapse run

```
evidence  obs 0.267  null 0.640  S 0.000  4 doc(s) cited by half the table or more
positions obs 0.000  null 0.000  S 0.000  VS2 1.00 of 3
raw       VS2 1.00 of 3   cos-disp 0.000   stance-H 0.00   modal-dis 0.00
```

`VS2 1.00 of 3` is the primary diagnostic: three seats, one effectively
distinct position. Cosine dispersion is exactly zero — the claims occupy a
single point.

The verdict is still `REFERENCE_TURN`, because turn 1 fixes the band and cannot
breach a band it defines. The guardrail adds the following warning:

```
WARNING: this panel is already below the ABSOLUTE floor before any seat has
seen any other. That is not collapse — nothing has collapsed yet — it is a
degenerate panel: these seats do not produce distinct positions on this
question even in isolation. Re-read the preflight, check whether the right
seats were invited, and treat any later agreement as uninformative. A relative
band built on a degenerate reference will certify the degeneracy.

breached: positions 0.000 < absolute floor 0.200 (tfidf-cosine)
```

Absolute floors distinguish degeneracy from collapse. A purely relative band —
"did the panel keep most of turn 1's diversity?" — would pass this panel because
it kept 100% of nothing. Collapse and degeneracy are different findings and the
instrument distinguishes them.

### Reading the scatter run

```
evidence  obs 0.867  null 0.922  S 0.000
positions obs 0.260  null 0.309  S 0.260   VS2 1.52 of 3
stance    obs 0.500  null 0.000  S 0.500   modal-dis 0.67
grounded  11/12  (lio_kestrel: fab 25.0%)
```

High stance disagreement, low evidence separation after chance correction, and
one seat's citations do not verify. The scatter fixture fabricates a quote, the
exact-span check detects it, and
`ingest` writes a repair prompt to
`runs/toy-scatter/turns/1/responses/lio_kestrel.repair.md`:

```
The following citations in your claim did not verify. **You are being shown no
new evidence.** Correct the quote to the exact text of the passage you meant,
or WITHDRAW the claim. Do not invent support.

`[quote_not_found]`: the cited document exists and is permitted, but contains
no passage with this exact text; the quote was altered, paraphrased or
invented.
```

Disagreement is not rewarded on its own. It must be grounded, and grounding is
decided by exact string matching rather than by model judgement.

### Reading the distinct run across two turns

Turn 1 is `REFERENCE_TURN` and fixes the band. Turn 2 comes back `COLLAPSING`,
with two instructive details:

The evidence axis is `UNINFORMATIVE`. Its permutation null saturated — these
three seats hold largely disjoint corpora, so re-dealing their citations at
random still produces disjoint sets. The axis can no longer discriminate, so it
is excluded from the verdict rather than read as zero. The tool also reports
the limitation: this is also the case where collapse would be hardest to see on
that axis.

The breach is `novelty 0.00 < 0.25`. Turn 2 said nothing turn 1 had not
already said. The mock seat is deterministic; given a near-identical packet it
restates itself, and the restatement breach is detected.

The monitor did not fire. `LLR +0.00 of +2.94` — "axes below the band
this turn, but the sequential monitor has not fired. Watching. One turn at this
panel size is too noisy to act on." One degraded turn is an observation, not a
finding.

## Perturbation checks

Perturb the corpus. Edit a passage in `library/docs/.../text.jsonl` after
indexing, then run `python ops/doctor.py`. It re-derives invariants from the
files rather than trusting the index and reports the mismatch.

Perturb a quote. Hand-edit one quote in `turns/1/responses/<seat>.json` and
re-ingest with `--reingest`. The grounding rate falls and a repair prompt
appears.

Use a low-overlap question. Write a question whose terms are absent from the
corpus and `prepare` reports the coverage collapse before anything is
dispatched. A question of pure stopwords raises an error rather than returning
an empty packet.

Change the backend. Re-run with
`EXPERTWINS_EMBED_MODEL=__force_tfidf_fallback__` and compare. The positions
axis moves a long way; the band moves with it; the verdicts hold. That is
[per-backend calibration](../../docs/06-measurement.md#evidence-is-backend-independent-positions-is-not)
working.

Add a fourth seat. Copy one of the YAML specs, re-run `attribute`, and add it to
`--seats`. Compare VS2 and packet separation with the three-seat run.

## What this example does not show

It does not show the acquisition path; the corpus is generated. See
[docs/03-building-a-corpus.md](../../docs/03-building-a-corpus.md).

It does not show a real model. Every response here is fabricated by a fixture.
See [docs/05-running-a-panel.md](../../docs/05-running-a-panel.md) for model
dispatch wiring.

It does not show the intervention ladder climbing. The ladder fires when the
sequential monitor fires, which takes more turns than two.

It also does not evaluate whether the instrument improves scientific
deliberation; see
[docs/06-measurement.md](../../docs/06-measurement.md#limitations).

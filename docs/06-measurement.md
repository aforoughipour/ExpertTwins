# 06 — Measurement: axes, nulls, bands, verdicts

[01-concepts.md](01-concepts.md) introduces the vocabulary. This page specifies
the measurement axes, null model, band construction, verdicts, constants, and
limitations.

## Measurement objective

The instrument measures whether a panel carries more independent information
than the same claims and citations would carry after seat identities were
removed. The operational question is:

Is this table carrying more independent information than the same seats drawing
at random from the same evidence would?

The distinction matters because fluency and apparent debate do not establish
independence. The measurement therefore targets evidence identity, position
diversity, declared stance, lexical form, and cross-turn movement separately.

## The five axes

Each turn is scored on five axes. Each axis measures a different channel through
which independence can be present or absent.

| axis | statistic | what it sees |
|---|---|---|
| `evidence` | mean pairwise Jaccard distance over cited `doc_id` sets | are the seats reading different papers |
| `lexical` | distinct-*n* / JS divergence over token distributions | are the seats writing differently |
| `positions` | order-2 Vendi score (RKE) over embedded claim text | how many effectively-distinct positions are on the table |
| `stance` | entropy and modal disagreement over declared stances | are the seats taking different sides |
| `herding` | cross-turn drift toward the table's centroid | are the seats moving toward each other |

These axes are not interchangeable. A table can score high on `lexical` and
zero on `evidence` when seats write differently about identical papers. A table
can score high on `evidence` and zero on `positions` when seats read different
papers and reach the same conclusion; that pattern is corroboration, not
collapse. Reporting a single diversity score would remove the distinction the
instrument is designed to preserve. There is no composite score.

### `evidence` is backend-independent; `positions` is not

`evidence` is a set statistic over document identifiers. No embedding touches
it, so one constant floor (`ABS_EVIDENCE_FLOOR = 0.15`) is correct everywhere.

`positions` is `(VS2 − 1) / (m − 1)` over embedded claim text, so its scale is
the scale of the kernel that produced it. Its floor is therefore a property of
the instrument, not only of the construct. The following implementation
calibration trace records why backend-specific floors are used; it is
provenance for engineering constants, not an empirical estimate of panel
performance.

| panel | tfidf-cosine | MiniLM-L6-v2 |
|---|---|---|
| fixture: distinct (short) | 0.996 | 0.901 |
| fixture: COLLAPSED | 0.000 | 0.000 |
| real panel A, turn 1, 9 seats | 0.774 | 0.090 |
| real panel B, turn 1, 9 seats | 0.826 | 0.106 |

Against a single fixed 0.20 threshold, the same row could pass under tf-idf and
fall below threshold under MiniLM. Short fixtures do not expose this compression;
long, concatenated, domain-homogeneous text can place sentence-transformer
embeddings in a narrow cone (cosine dispersion 0.28 against tf-idf's 0.83).
Therefore, short synthetic fixtures are not sufficient to calibrate this floor.

Floors are per backend:

```python
ABS_POSITIONS_FLOORS = {
    "tfidf-cosine": 0.20,
    "sentence-transformers:all-MiniLM-L6-v2": 0.023,
}
```

Note. The MiniLM number is an extrapolation, not a calibration. 0.20 sits at 26%
of panel A's tf-idf score; 0.023 is that same fraction of panel A's MiniLM
score. It separates all four panels above, and that is the whole of its support.
It has not been validated against a labelled collapse, because no such set
exists here.

An unrecognised backend gets no absolute floor:
`abs_positions_floor()` returns `(0.0, False)` and the band records
`abs_calibrated: False`. A band with `abs_calibrated: False` is relative only:
it can still see a panel fall away from its own first turn, and it cannot see a
panel that was degenerate from the start. The band's `render()` reports this in
words.

tf-idf cosine is the recommended backend for this measurement. Under the
calibration trace above, the 0.090 MiniLM row has a factor of four of headroom
against a 0.023 floor, while the corresponding tf-idf row has a factor of nearly
four hundred.

### `lexical` has no floor

`lexical` is reported as a style channel and nothing else. Form-based diversity
metrics assign high diversity to sets of random sentences; a lexical floor
would reward noise and allow form-based variation to affect the verdict.

## Chance correction

A raw diversity number does not distinguish independent reasoning from the
breadth of evidence the seats were handed. Three seats given three disjoint
packets can cite different documents even without independent analysis.

Every axis is therefore scored against a **null model**: shuffle the attribution
of claims across seats, recompute the statistic, repeat (`--permutations`,
default 200). That gives the value the same material would produce with the
seats' identities removed. The reported separation is

```
S = (observed − null) / (1 − null)
```

the share of the available-above-chance headroom actually occupied.

The toy example illustrates the correction. Turn 1 scores `evidence obs 0.867`
— nearly nine tenths of cited documents differ between seats. The null is
`0.920`, so the seats are less diverse than chance. `S = 0.000`. The raw number
is not the measurement.

### Uninformative axes do not vote

When the null saturates (`NULL_SATURATION = 0.95`), the denominator collapses
and the axis can no longer discriminate between a good panel and a bad one. The
axis is marked `UNINFORMATIVE` and excluded from the verdict.

This happens visibly on the toy corpus, and that is one reason the toy corpus is
only 18 documents. An axis that cannot discriminate is not allowed to vote.

The same principle appears in the sequential monitor: an unmeasurable turn feeds
`None`, and the accumulator does not move. Missing information is not scored as
health.

## The band

The band is the set of thresholds for one run, fixed at turn 1 from the
preflight prior and the isolated first turn, and never renegotiated.

```
rho_low  = preflight retention            (from packet separation + contestedness)
rho_high = 1.20                           (fixed convention)

floor   evidence  = max(0.15,  rho_low × S_1(evidence))
floor   positions = max(abs_floor(backend), rho_low × positions_1)
ceiling positions = min(1.0, max(0.35, 1.20 × positions_1))
```

Four design decisions follow from this construction.

First, the band is fixed once. A band recomputed each turn from the current turn
would ratchet downward with the conversation and certify slow collapse.

Second, turn 1 is the reference. Turn 1 is generated in isolation: no seat has
seen another. Its separation is therefore the panel's free diversity on this
question. Later turns are measured as retention of it. The band records one
stochastic draw of that quantity, not a population value.

Third, absolute floors supplement the relative band. A purely relative band
would certify a turn 1 that was already degenerate. Hence `REFERENCE_TURN`
cannot breach the band it defines, but it can start below the absolute floors —
a separate finding: a panel that was degenerate before any cross-exposure.

Fourth, the ceiling captures scatter. More scatter than the isolated turn
produced is not evidence of procedural health; it is assigned its own verdict
(`SCATTERED`) rather than a pass.

The band carries its own identity — panel, question, backend — because a band is
a statement about one panel on one question measured with one backend. Comparing
bands across any of those dimensions is not meaningful.

## Verdicts

| verdict | meaning |
|---|---|
| `healthy` | within band on the axes that could be measured |
| `independent_corroboration` | low position diversity, high evidence diversity: different papers, same conclusion. Independent corroboration. |
| `contested_evidence` | high position diversity, low evidence diversity: same papers, different conclusions |
| `collapsing` | retention falling; the sequential monitor has fired |
| `collapsed` | the ladder was exhausted and the table did not recover |
| `reference_turn` | turn 1; the band is fixed from it |
| `scattered` | above the ceiling |
| `fidelity_lost` | a seat stopped preserving its source identity — short-circuits everything else |
| `undetermined` | too few claims, or participation below 60% |

`independent_corroboration` is the reason there is no composite score. Low
position diversity is the signature of collapse and also the signature of a
case where distinct evidence points to the same conclusion. The 2×2 of position
diversity and evidence diversity separates these cases; a single number cannot.

`fidelity_lost` short-circuits the verdict. If a seat has stopped preserving its
source identity, the turn's heterogeneity is not interpretable, so the verdict
is replaced rather than annotated. A seat on its own territory that anchors
nothing in its own papers trips `FIDELITY_FLOOR = 0.25`. A diversity gain bought
with fidelity loss is refused, because it would reward statements unsupported by
the source identity being preserved.

`undetermined` is reported, never silently treated as healthy. An empty table is
not a diverse one. `MIN_PARTICIPATION = 0.6` exists because dropping the seats
that agree with each other raises every statistic on this page.

## The sequential monitor, and what it is not

Retention is noisy. With a handful of seats and a handful of claims, a per-turn
threshold both fires spuriously and misses slow drift. The band statistic is
therefore accumulated across turns as a log-likelihood ratio under a Beta
family, stopping at Wald boundaries (`log(0.95/0.05)`).

The sequential monitor is not a calibrated Wald SPRT. The shape is borrowed,
but the assumptions required for a calibrated test are violated here:
observations are serially dependent by construction; the Beta parameters are
engineering choices, not fitted likelihoods, so the boundaries are not nominal
error rates; the policy changes when the monitor fires, so post-firing
observations do not come from the pre-firing model; and the statistic is
measured against a single noisy baseline.

The boundaries are tuning constants, not error rates. In practice, one
unambiguous turn fires immediately: a fully collapsed axis sends the increment
to the boundary in a single step, because total collapse is not a noisy
observation. A mildly degraded turn requires several consecutive turns.

Real sequential guarantees would require a calibrated change-point model or an
anytime-valid procedure whose conditional model tolerates adaptive intervention.
The accumulator resets on firing because the regime it was testing has been
deliberately changed; no error-rate claim survives that reset.

## The intervention ladder

When the monitor fires, the guardrail climbs the ladder in a fixed order. Every
rung that changes what a seat reads or sees is tried before the one rung that
changes what a seat is told:

| # | rung | acts on |
|---|---|---|
| 1 | `route_to_most_distant_peers` | routing |
| 2 | `withhold_the_consensus_evidence` | evidence |
| 3 | `reanchor_to_own_corpus` | evidence |
| 4 | `anonymize_the_table` | presentation |
| 5 | `isolate_from_cross_talk` | channel |
| 6 | `assign_dissent_to_the_evidenced_minority` | the prompt |
| 7 | `stop_deliberating_and_report` | the loop |
| 8 | `declare_collapse` | the record |

Rung 6 is the only prompt-level intervention and is placed late because
prompt-level disagreement can trade fidelity for diversity. Disagreement
produced under it is weaker evidence of genuine divergence. The turn is stamped
prompt-perturbed so it can be interpreted separately.

Rung 6 also requires the seat whose own evidence best supports the minority
position to argue it. It is therefore still grounded, rather than assigned to an
arbitrary seat.

Every rung is recorded in the transcript with what it did and what it rests on,
because an unrecorded perturbation is an invisible confound. Where a rung rests
on published work, the citation is given and marked `VERIFIED`; where it is the
project's own construction (rung 2), it says so.

If the ladder is exhausted and the table is still collapsed, the run records
that result rather than marking a recovery.

## Trajectory: the four failure modes

`trajectory.py` reads across turns and reports four modes:

| | mode | fires when |
|---|---|---|
| F1 | convergence collapse | position diversity falling across turns |
| F2 | evidence exhaustion | under 15% of citations to documents not already cited |
| F3 | restatement | novelty under 30% |
| F4 | grounding decay | verified-citation rate falling |

F1 and F3 pull in opposite directions. A panel can hold disagreement at 100%
forever by repeating its opening position, and it can stay novel forever by
drifting off the question. A single trajectory line is insufficient; the four
modes are interpreted together.

The thresholds are conventions, not measurements. They decide which lines are
flagged and nothing else; no gate reads them.

With fewer than about four turns, these outputs are single differences, not
trends, and should not be quoted as either.

## Limitations

A firing failure mode is a fact about one run: one model, one question, and one
corpus. It is not a property of the panel design, the model, or multi-agent
systems in general. `n = 1` is not a measurement of the design. Claims about the
design require many runs, many questions, and a control.

The thresholds are conventions. `0.15`, `0.20`, `0.25`, `0.6`, `1.20` are
engineering defaults chosen so the flags fire where a reader would want to
inspect. They are not estimated from a labelled corpus of collapsed and healthy
panels, because no such corpus exists here. The one extrapolated constant is
labelled as extrapolated in the source.

Heterogeneity is not correctness. A procedurally healthy, maximally
heterogeneous panel can be unanimously wrong, and the instrument will certify
its procedural health while it is. Nothing here measures whether the science is
right. It measures whether more than one position was actually present.

Verification is not validation. `ingest` proves that a quote exists in a cited
document. It cannot prove that the quote supports the claim; that is what the
audit is for. The audit is seats judging seats, so its output is labelled
as judgements rather than verifications.

The seats are not the modeled individuals. They are retrieval-grounded
reconstructions from a corpus. Fidelity instruments measure whether a seat is
drifting from its own corpus; they do not validate identity beyond
corpus-grounding.

The standing null hypothesis is that none of this helps. It has not been shown
that a measured, guard-railed panel produces better science than one careful
analyst with the same papers, or than the same seats with no instrumentation at
all. The instrument is built so that result would be visible if true.

## Improving the measurement

The most useful extension would be a labelled set: panels independently judged
collapsed or healthy by people, against which these thresholds could be
calibrated rather than chosen. Backend calibration is also needed; the
`ABS_POSITIONS_FLOORS` table has two entries and one of them is an
extrapolation.

A new measurement requires a fixture that fails without it.

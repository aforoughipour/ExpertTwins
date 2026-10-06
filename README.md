# ExpertTwins

ExpertTwins is a framework for convening panels of literature-grounded
language-model agents ("expert twins") on a scientific question, and for
measuring whether the disagreement among them reflects the evidence or the
convergence of the panel onto a single voice.

## Overview

A panel is composed of *seats*. A **person seat** is modelled on an individual
scientist, from the papers that scientist wrote and the literature they cite. A
**discipline seat** is modelled on the literature of a field. Each seat receives
a private evidence packet drawn from its own corpus and answers in isolation.
Every quotation a seat produces is verified character by character against the
stored text of a document contained in that seat's packet. Heterogeneity across
seats is then measured on several independent axes, each referenced to a
permutation null, and a lower bound on acceptable heterogeneity (the *floor*) is
fixed from the first, fully isolated turn. When later turns fall below the
floor, a guardrail intervenes on retrieval and routing rather than on the
prompt.

The framework addresses a specific methodological problem. When several
language-model agents discuss a question in a shared context, agreement may
arise because the evidence supports a single position or because the agents
converge for reasons unrelated to the evidence. ExpertTwins is designed to
distinguish these cases: no claim enters the record unless its quotations can
be verified against stored text, and no change in agreement is interpreted
without reference to a measured baseline.

## Repository contents

- **Framework.** Retrieval and packet construction, quote verification, the
  heterogeneity instruments, an eight-rung intervention ladder, the transcript
  renderer, the trajectory monitor, the literature acquisition pipeline, and a
  toolchain for building seats.
- **Toy example.** A generated, fictional corpus and a deterministic mock seat
  with which the complete pipeline can be run offline, without a language
  model, network access, or API cost.
- **Example roundtables.** Interactive idea-flow maps of three discipline-seat
  roundtables on neuroblastoma immunotherapy, which show how ideas arise, are
  contested and are carried forward across turns
  ([`examples/roundtables/`](examples/roundtables/README.md)).

The following are not included in the repository at present:

- Literature corpora. Corpora are built with the included acquisition tools.
- Seat specifications. Those in `config/` are fictional examples.
- Transcripts and results, other than the example roundtables above.

## Installation

Python 3.11 or later is required.

```bash
git clone https://github.com/aforoughipour/ExpertTwins.git
cd ExpertTwins
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

The base installation provides the read path (packet construction,
verification, heterogeneity measurement, the guardrail and the transcript) and
depends only on PyYAML, numpy and pydantic. It requires no GPU, no network
access and no model, so an installation can be copied to and run on an
air-gapped compute node. Optional extras:

```bash
pip install -e ".[acquire]"   # literature acquisition; the only component that uses the network
pip install -e ".[embed]"     # a neural sentence encoder for the semantic axis
pip install -e ".[dev]"       # pytest
```

The installation is checked with:

```bash
python ops/doctor.py
```

`doctor.py` treats a missing, unreadable or undersized index as a failure,
because a seat without evidence returns "no evidence found", which in a
transcript cannot be distinguished from a genuine negative finding. On the toy
corpus the corpus-size check is expected to fail.

## Example: an offline round table

The following builds the fictional corpus, convenes a three-seat panel, runs two
turns with the deterministic mock seat, audits the citations and renders a
transcript. No model is called and no network access is made.

```bash
python examples/toy/make_toy_corpus.py library --force
python ops/doctor.py

# Attribute authored documents to each seat. Without this step a seat has no
# own corpus, and every packet reports `own 0` / territory `foreign`.
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

The mock seat supports two further modes. `--mode collapse` produces a panel
that converges, and the guardrail is expected to fire; `--mode scatter`
produces a panel that disperses without grounding, and it is expected not to be
reported as healthy. Together the three modes test the instruments against both
failure directions. The annotated walkthrough, including the expected output of
every step, is in [`examples/toy/README.md`](examples/toy/README.md).

## Pipeline

```
  config/people/*.yaml          a seat: who, what they wrote, what they read,
  config/domains/*.yaml         what they would never concede
          │
          ▼  ops/people.py  resolve → plan → acquire → attribute → install → verify
  library/                      documents + passages + an FTS5 index
          │
          ▼  ops/panel.py prepare
  runs/<run>/turns/1/packets/   one private packet per seat
          │
          ▼  dispatch to N separate agents, in isolation, in parallel
  runs/<run>/turns/1/responses/ claims, citations, stances, abstentions
          │
          ▼  ops/panel.py ingest
  verified claims · heterogeneity axes · the floor · a frozen, hashed ledger
          │
          ├──▶ healthy  ──▶ ops/panel.py followup  (next turn)
          └──▶ collapsing ─▶ guardrail selects a rung ─▶ followup applies it to retrieval
          │
          ▼  ops/panel.py audit → audit-ingest → render → ops/trajectory.py
  a transcript, and a separate record of whether the panel remained heterogeneous
```

## Design principles

1. **Derived, not asserted.** Whether a document is full text, whether a
   quotation is genuine, and whether a seat was permitted to cite a document
   are each recomputed from stored bytes. No caller can declare them.
2. **Private evidence.** Each seat sees only its own packet. The set of
   documents a seat may cite is derived from the packet as written, not
   declared alongside it; a citation outside that set fails verification
   regardless of its content.
3. **A model-free deterministic layer.** The verification, measurement and
   guardrail code calls no language model and therefore cannot be persuaded to
   accept a citation.
4. **A floor fixed by the isolated turn.** The first turn is generated before
   any seat has seen another seat's output. It provides the baseline estimate
   of between-seat heterogeneity, fixes the floor for subsequent turns, and is
   not itself judged against that floor.
5. **Interventions on evidence before instructions.** The guardrail's ladder
   has eight rungs. The first five change what a seat reads or sees (routing,
   retrieval, re-anchoring to its own corpus, anonymisation, isolation); the
   sixth is the only one that changes what a seat is told; the last two end
   deliberation and record the outcome. Instructing agents to disagree
   produces disagreement in the text without changing the evidence on which
   it rests, and is therefore tried only after every evidence-level rung.

## Documentation

| Document | Contents |
|---|---|
| [docs/01-concepts.md](docs/01-concepts.md) | Terminology and the role of each component |
| [docs/02-quickstart.md](docs/02-quickstart.md) | The offline toy round table, step by step |
| [docs/03-building-a-corpus.md](docs/03-building-a-corpus.md) | Library layout, acquisition and integrity screens |
| [docs/04-building-seats.md](docs/04-building-seats.md) | Seat specifications and the seat-building lifecycle |
| [docs/05-running-a-panel.md](docs/05-running-a-panel.md) | prepare → dispatch → ingest → followup → audit → render |
| [docs/06-measurement.md](docs/06-measurement.md) | Heterogeneity axes, verdicts and the intervention ladder |
| [docs/07-agent-setup-guide.md](docs/07-agent-setup-guide.md) | A protocol for an AI coding agent setting up a new panel |
| [docs/08-reference.md](docs/08-reference.md) | Commands, environment variables, file formats and the test suite |
| [docs/09-hpc.md](docs/09-hpc.md) | Acquisition and panels on a Slurm cluster |

## Limitations

- **Comparison with a single model.** The relevant null hypothesis is that a
  single capable model, given the same corpus and the same verifier, performs
  as well as the panel; if so, the multi-agent structure contributes little
  beyond the corpus and the verifier. The `solo` task (`--task solo`) runs a
  single seat under identical conditions so that this comparison can be made.
- **Seat fidelity.** A seat is a model conditioned on a corpus and a
  specification, not a reproduction of a person. `ops/fidelity.py` implements
  a temporal-holdout protocol that provides ground truth for the resemblance
  question; it is a partial test.
- **Verification scope.** The verifier establishes that a quotation exists in
  the cited document, not that the claim drawn from it is correct. A verified
  citation can still be a misreading; `panel.py audit` addresses this by
  re-opening every cited document to the whole panel for a reading-only
  assessment.
- **Interpretation of the floor.** Heterogeneity above the floor indicates that
  the panel has not collapsed. It does not indicate that the panel's
  conclusions are correct.

## Citation

A manuscript describing ExpertTwins is in preparation. Citation information
will be added here when it becomes available.

## License

MIT. See [LICENSE](LICENSE).

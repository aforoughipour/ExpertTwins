# 4. Building seats

A seat is a YAML file plus a corpus. This page covers the file; the corpus is
[chapter 3](03-building-a-corpus.md).

Start with the shipped examples — [`config/people/mira_brindle.yaml`](../config/people/mira_brindle.yaml),
[`config/domains/peloria.yaml`](../config/domains/peloria.yaml) — and the blank
[`config/people/TEMPLATE.yaml`](../config/people/TEMPLATE.yaml).

## 4.1 Person seat or discipline seat?

| | person seat | discipline seat |
|---|---|---|
| modelled on | a named scientist | a literature |
| needs | `surname`, `initials`, `own_topics`, attribution | queries only |
| extra step | `attribute` — prove they wrote it | none |
| lives in | `config/people/*.yaml` | `config/domains/*.yaml` |
| construction burden | hours, and an attribution risk | minutes |

Build discipline seats first. They exercise the whole pipeline, avoid author
attribution, and test whether the question is answerable from the assembled
literature. Add person seats when a particular reading of the evidence is
required rather than a field-level reading.

Seat specifications for real individuals are user-created local configuration
files and should remain untracked by git.

## 4.2 The fields

```yaml
name: mira_brindle            # the seat id; lowercase, underscores
display: "Mira Brindle"
kind: individual              # or: domain / field
probe: "lanternmoss prism spores"
min_docs: 3
```

### `probe` — keep it to three to five tokens

The probe is a single FTS `MATCH` query used to prove the seat's literature
exists. FTS `MATCH` is a conjunction: every token must appear in the same
passage. A seven-token probe can return zero on a healthy corpus, which indicates
a probe/corpus mismatch rather than a storage failure.

### `min_docs` — commit it before acquisition

The number of full-text documents this seat must hold to be allowed at the
table. Write it down before acquisition, so it cannot be chosen afterwards to
match what arrived.

### Identity fields (person seats only)

```yaml
surname: "Brindle"
initials: ["M", "MB", "Mira"]
surname_aliases: []
orcid: ""
affiliations: ["Fictional Peloria field station"]
own_topics: ["brindle.authored"]
```

These fields drive attribution. Author strings in bibliographic records vary
substantially. The matcher in `expertwins/identity.py` handles, and is tested
on:

- diacritic folding — `Brindlé, M.` and `Brindle M` are the same person;
- initials vs. given names — `Reed TO`, `Toma O Reed` and `Reed T` all
  match; `Reed L` and `Reed Lio` do not;
- prefixes must not match — a `Moss` matcher rejects `Mosswick`;
- two-token surnames — `Van Kelp` needs `surname_aliases: ["Vankelp"]`,
  because indexes disagree about whether to concatenate;
- inverted indexing — a two-token surname sometimes appears with the second
  token promoted to an initial (`Quire Moss E` ↔ `Moss E`), which is why
  aliases exist;
- genuine ambiguity — two people can share a surname and initials. The
  matcher reports the conflict rather than guessing.

For common surnames, treat attribution as unresolved until
`check_contamination.py` exits successfully. A contaminated own corpus means the
seat can cite another author's work as its own position.

### The three prose fields

These are what the seat is *told*, and they matter less than the corpus — but
they are not nothing.

```yaml
territory: >
  What this seat is entitled to speak about, and what it is not. Be specific
  about the boundary. "Pelorian field ecology" is not a territory; "lanternmoss
  prism-spore dispersal in moonlit terrace plots" is.

moves: >
  How this seat characteristically reasons: the question it asks first, the
  control it demands, and the kind of evidence it finds persuasive or discounts.

fatal_flaws: >
  The positions this seat should reject and the arguments it is instructed not
  to accept. This field prevents a seat from agreeing with every premise.
```

Write `fatal_flaws` after the other prose fields and make it specific. A vague
`fatal_flaws` field can make the seat drift toward the majority, creating a
specification-level collapse risk.

### Retrieval queries — FTS5 syntax

```yaml
retrieval_queries:                      # what this seat can reach
  - 'lanternmoss AND "prism spores"'
  - 'lanternmoss AND ("moonlit terraces" OR "luminous bracts")'

own_queries:                            # what this seat wrote
  - 'lanternmoss AND "prism spores"'
  - 'Brindle AND Peloria'
```

Quoted phrases are phrase matches. `AND`/`OR`/`NOT` work. Bare terms are
conjunctive. Validate each query:

```bash
python -c "import sqlite3;c=sqlite3.connect('library/index.sqlite');print(c.execute('select count(*) from passages_fts where passages_fts match ?',['lanternmoss AND \"prism spores\"']).fetchone())"
```

### Acquisition — Europe PMC syntax

```yaml
acquisition:
  - id: brindle.authored
    min_full: 30
    probe: "lanternmoss prism spores"
    queries:
      - 'AUTHOR:"Mira Brindle" AND lanternmoss'
```

The `id` becomes a topic on every document the node fetches, and
`own_topics` refers to those topic names. That is the link between acquisition
and attribution.

`acquisition → queries` is Europe PMC syntax. `retrieval_queries` and
`own_queries` are FTS5 syntax.

## 4.3 The six-step lifecycle

```
template → resolve → plan → acquire → attribute → install → verify
```

```bash
python ops/people.py template "Mira Brindle" > config/people/mira_brindle.yaml
python ops/people.py resolve config/people/mira_brindle.yaml
python ops/people.py plan    config/people/mira_brindle.yaml
python ops/people.py acquire config/people/mira_brindle.yaml --root library_brindle
python ops/people.py attribute config/people/mira_brindle.yaml --root library_brindle
python ops/people.py install config/people/mira_brindle.yaml --root library_brindle
python ops/people.py verify  config/people/mira_brindle.yaml
```

`resolve` — identity and scope. Author strings are ambiguous. Before
downloading anything, inspect what the search returns: venues, years,
co-authors, and topics. Confirm that the results match the intended researcher
or domain.

`plan` — what would be fetched, and what floors are committed. Nothing is
downloaded.

`acquire` — into a separate root, for the manifest-tearing reason in
[chapter 3](03-building-a-corpus.md#33-acquisition).

`attribute` — the falsification step. Decide which fetched papers this
person *actually wrote*, by checking the stored author list of each document.
That is a different source of truth from the query that fetched them, and
that distinction is required. A query that asks for `AUTHOR:"Mira Brindle"` and
gets 200 results has not established that Mira Brindle wrote 200 papers.

`install` — merge, reindex, register.

`verify` — checks retrieval and overlap. See
[chapter 3 §3.6](03-building-a-corpus.md#36-checking-the-corpus-before-model-dispatch).
Two seats can hold healthy-looking document counts while their probes return
zero; such seats are not valid for the intended question.

## 4.4 The citation neighbourhood

A seat built only from what a scientist *wrote* is incomplete. What they *read*
is a different set, and it is the set their arguments actually stand on.

`ops/neighbourhood.py` builds four more acquisition nodes per seat from the
citation graph:

| node | what it fetches |
|---|---|
| `reads` | the works this scientist cites most often across their own papers. Repeatedly cited works are treated as load-bearing. |
| `collaborators` | the works of the people they publish with. Collaborators can share methods and assumptions. |
| `cited_by` | the most-cited works that cite *them* — the field's reply to their work. |
| `canon` | the most-cited works in the topics they work in, whoever wrote them. |

```bash
python ops/neighbourhood.py plan    config/people/mira_brindle.yaml
python ops/neighbourhood.py acquire config/people/mira_brindle.yaml --root library_brindle_nbhd
python ops/neighbourhood.py retag   --root library_brindle_nbhd
```

Nothing acquired this way lands in anybody's `own` corpus. Neighbourhood
documents are tagged separately and are never submitted to `attribute`. They
are what the panel reads, not what it wrote — and the `own`/`read` tier on
every citation keeps the two distinguishable in the transcript.

## 4.5 Fidelity checks

`ops/fidelity.py` offers two instruments. Only the first has ground truth.

Temporal holdout. Cut the corpus at year *Y*. Ask the seat a question that
one of *its own* papers from year *Y+1* answers. That paper is not a proxy for
what the scientist concluded — it is the scientist's own published position,
written by them, after the cut.

```bash
python ops/fidelity.py holdout config/people/mira_brindle.yaml --cut 2020
```

The harness builds the pairs and freezes them. It does not grade. Whether a
claim matches a paper's conclusion is a semantic judgement, and this project
does not pretend a semantic judgement is deterministic. Grading is a human or a
blinded model, and the harness keeps the two apart.

Stylometric attribution. Model-free, weaker, and useful as a diagnostic:

```bash
python ops/fidelity.py attribute runs/<run> --turn 1
python ops/fidelity.py centroids
```

## 4.6 A checklist before the first model-mediated run

- [ ] `python ops/doctor.py` — no failures
- [ ] `python ops/people.py verify <spec>` — passes for every seat
- [ ] pairwise overlap below ~35% for every pair
- [ ] `python ops/check_contamination.py` — exits 0
- [ ] `python ops/check_canon.py` — nothing important missing
- [ ] every seat's `fatal_flaws` is specific enough to be violated
- [ ] the whole loop has been run on the toy corpus at least once

## Next

→ [5. Running a panel](05-running-a-panel.md)

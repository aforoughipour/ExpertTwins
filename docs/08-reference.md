# 08 — Reference

Every command, flag, environment variable and file format. For explanation
rather than enumeration, see [05](05-running-a-panel.md) and
[06](06-measurement.md).

---

## Environment variables

| variable | default | purpose |
|---|---|---|
| `EXPERTWINS_ROOT` | the repo checkout | project root; everything else derives from it |
| `EXPERTWINS_LIBRARY` | `$ROOT/library` | merged corpus root |
| `EXPERTWINS_INDEX` | `$LIBRARY/index.sqlite` | the FTS index actually read by retrieval |
| `EXPERTWINS_RUNS` | `$ROOT/runs` | where run directories are written |
| `EXPERTWINS_CACHE` | `$ROOT/.cache` | acquisition and embedding cache |
| `EXPERTWINS_EMBED_MODEL` | auto | embedding backend; set `__force_tfidf_fallback__` to pin tf-idf |
| `NCBI_API_KEY` | unset | raises the NCBI rate limit from 3/s to 10/s; free, acquisition only |
| `EXPERTWINS_CONTACT` | `nobody@example.org` | contact address sent to Crossref/Unpaywall/NCBI; set before acquiring |
| `EXPERTWINS_OFFLINE` | unset | `1` makes every network path raise `OfflineError` instead of degrading quietly |

`EXPERTWINS_ROOT` is the HPC escape hatch: the tree is meant to be copied onto a
cluster filesystem and run there. Nothing is hard-coded, and `paths.relative()`
/ `paths.resolve()` keep run manifests portable by storing paths relative to the
root.

`EXPERTWINS_INDEX` is useful for staging the index to node-local scratch —
SQLite over a parallel filesystem is slow and occasionally lock-flaky.

`EXPERTWINS_EMBED_MODEL` must be identical on every node. `embed()` records
which backend ran, and a VS-2 computed over tf-idf is not comparable with one
computed over an encoder. Mixing them silently produces incomparable numbers.

```bash
python -c "from expertwins import paths; print(paths.describe())"
```

prints the resolved paths and marks any that are missing. The rows are
`EXPERTWINS_ROOT`, `library`, `index`, `config`, `runs`, and `cache`.
`python ops/doctor.py` also checks the corpus layer and prints
`[ok  ] expertwins.corpus importable` when it can be imported.

---

## `ops/panel.py`

```
panel.py {prepare,followup,ingest,render,audit,audit-ingest}
```

### `prepare`

```
panel.py prepare [question] --out OUT [--question-file FILE]
                 [--task {analyze,critique,present,review,roundtable,solo}]
                 [--seats SEATS] [--specs SPECS] [--index INDEX]
                 [--material MATERIAL] [--max-passages N] [--max-seeds N]
                 [--own-floor N] [--max-per-doc N] [--min-docs N]
                 [--neighbours N] [--no-disclosure] [--before-year YYYY]
```

The question may be given as a positional string or with `--question-file`;
prefer the file, because the question is part of the method and belongs in
version control.

| flag | default | meaning |
|---|---|---|
| `--out` | *required* | run directory to create |
| `--task` | `roundtable` | which task template to use |
| `--seats` | all in `config/seats.yaml` | comma-separated seat names |
| `--specs` | `config/people` | alternate spec directory |
| `--index` | `$EXPERTWINS_INDEX` | alternate SQLite index |
| `--material` | — | extra file put in front of every seat (see `hpc/env_example.md`) |
| `--max-passages` | — | hard cap on packet size |
| `--max-seeds` | — | retrieval breadth |
| `--own-floor` | 60 | refuse below this many own-corpus documents |
| `--max-per-doc` | — | cap passages drawn from any one document |
| `--min-docs` | 10 | refuse below this many documents per packet |
| `--neighbours` | — | include the citation neighbourhood |
| `--no-disclosure` | off | omit the territory disclosure block |
| `--before-year` | — | hard cut: retrieve nothing published at or after this year |

### `followup`

```
panel.py followup run_dir [question] [--question-file FILE] [--seats SEATS]
```

Builds the next turn's packets with the previous turn's transcript in view.
`--seats` changes who is at the table.

### `ingest`

```
panel.py ingest run_dir [--library PATH] [--turn N] [--permutations N] [--reingest]
```

Verifies, scores, judges, freezes. `--permutations` sets null-model resamples
(default 200; raise for a tighter null). `--reingest` re-scores a frozen turn.

> Exits 1 even on success. Check for the `turn frozen` / `commit <hash>`
> line.

### `render`

```
panel.py render run_dir
```

Writes `runs/<run>/transcript.md` with grounding and audit verdicts folded in.

### `audit` / `audit-ingest`

```
panel.py audit        run_dir [--turn N] [--seats SEATS] [--radius N] [--library PATH]
panel.py audit-ingest run_dir [--turn N] [--library PATH]
```

`--radius` is how many surrounding passages each reviewer is shown.

---

## `ops/people.py`

```
people.py {template,resolve,plan,acquire,install,attribute,verify,merge,status,reindex}
```

| subcommand | purpose |
|---|---|
| `template <name>` | write a spec skeleton |
| `resolve <spec>` | disambiguate the author against the source |
| `plan <spec>` | what *would* be acquired — read this before acquiring |
| `acquire <spec>` | fetch into the seat's own root |
| `install <spec>` | install an acquired root into the library |
| `attribute <spec>` | resolve the seat's own corpus → `config/people/<seat>.own.json` |
| `verify <spec>` | check the seat against the installed index |
| `merge` | merge per-seat roots into the library |
| `status` | per-seat document counts |
| `reindex` | rebuild `index.sqlite` from the document tree |

`attribute` takes only the spec path — there is no `--root` flag. It reads
the installed index. Run it after `install`/`merge`, and run it for every seat:
without the resulting `.own.json`, packets report `own 0` and territory
`foreign`.

---

## Other commands

| command | purpose |
|---|---|
| `ops/doctor.py [--quick]` | check the installation, audit the library against the files on disk, and compare the index with it; `--quick` skips the file-level audit |
| `ops/trajectory.py [--json] run_dir` | the four failure modes across turns |
| `ops/simulate.py run_dir [--turn N] [--mode {distinct,collapse,scatter}] [--seed N] [--audit]` | deterministic mock seat; `--audit` answers audit packets |
| `ops/fidelity.py {holdout,attribute,centroids}` | is a seat still its corpus? |
| `ops/neighbourhood.py {plan,acquire,retag}` | the citation neighbourhood |
| `ops/check_canon.py` | are the papers that must be present, present? |
| `ops/check_contamination.py` | is anything present that must not be? |
| `ops/acquire.py --config F [--group G] [--root R] [--per-pass N] [--dry-run]` | generic acquisition |
| `ops/acquire_canon.py --config F [--root R] [--group G] [--dry-run]` | canon acquisition |
| `ops/acquire_preprints.py --config F [--root R] [--node N] [--max-records N] [--per-pass N] [--dry-run]` | preprint acquisition |
| `ops/repair_passage_ids.py [--root R] [--scan] [--apply]` | repair passage id drift |

`check_canon.py` and `check_contamination.py` are YAML-driven. If their config
file is absent they explain themselves and exit 0; absence of a complaint is not
evidence of a check.

---

## Library layout

```
library/
├── MANIFEST.jsonl              append-only; latest row per doc_id wins
├── index.sqlite                the only thing retrieval reads
├── quarantine/                 documents that failed a screen
└── docs/<2-char-shard>/<doc_id>/
    ├── text.jsonl              one JSON object per passage
    └── meta.yaml               document metadata
```

The shard is the first two characters of the `doc_id`. Retrieval reads only
`index.sqlite`; the document tree exists for verification and reindexing, and
`doctor.py` checks one against the other.

### `index.sqlite`

Tables: `docs`, `doc_topics`, `passages`, `meta`, and `passages_fts` — an FTS5
external-content table with `content='passages'`, `content_rowid='rowid'`,
`tokenize="unicode61 remove_diacritics 2"`.

> `passages_fts.rowid` must equal `passages.rowid`. An offset silently
> returns the *wrong passage* — a failure that produces plausible, verifiable,
> incorrect output. `doctor.py` probes for it; the remedy is to rebuild the
> index with `python ops/people.py reindex`.

### Full-text classification

`classify_fulltext` decides `full` vs `partial` by measuring the stored bytes:
`full` requires ≥ 6000 characters *and* ≥ 5 passages. No caller may
assert full-textness; the metadata layer rejects an asserted `full` below the
floors.

### `title_appears_in_text`

The screen detects a stored document that is a different work citing the
intended one. It is applied in addition to hash-based identity checks.

---

## Run directory layout

```
runs/<run>/
├── manifest.json              run config, seats, flags, library commit
├── transcript.md
├── trajectory.json
├── seats/<seat>.md            resolved seat briefs
└── turns/<n>/
    ├── packets/<seat>.md
    ├── responses/<seat>.json
    ├── responses/<seat>.repair.md     failed citations, if any
    ├── preflight.json
    ├── checks.json
    ├── ledger.json
    ├── report.json
    └── audit/
        ├── packets/<seat>.md
        ├── responses/<seat>.json
        ├── shown.json         exactly which documents were handed over
        └── checks.json
```

`shown.json` is what makes `not_under_examination` checkable after the fact
rather than taken on trust.

---

## Response schema (turn)

```json
{
  "claims": [
    {
      "claim": "one falsifiable sentence",
      "citations": [{"doc_id": "...", "quote": "exact text from a passage"}],
      "declared": {"species": "human", "model_system": "in vivo"},
      "type": "mechanism",
      "territory": "home",
      "self_challenge": "the strongest objection to this claim"
    }
  ],
  "abstentions": [{"question": "...", "why": "no evidence in my packet"}]
}
```

`quote` is matched by exact span. Paraphrase fails. `territory` is one of
`home`, `adjacent`, `foreign`.

## Response schema (audit)

```json
{"citation_checks": [
  {"doc_id": "...", "cited_by": "...", "claim": "...",
   "verdict": "supports|overstated|misread|irrelevant|cannot_tell",
   "why": "...",
   "citations": [{"doc_id": "...", "quote": "exact text"}]}
]}
```

An adverse verdict without a verifying quote is recorded as
`unsupported_verdict`. A verdict on a document not in the reviewer's packet is
recorded as `not_under_examination`.

---

## Seat spec

See `config/people/TEMPLATE.yaml` and the three worked examples
(`mira_brindle`, `toma_reed`, `lio_kestrel`). Full field documentation is in
[04-building-seats.md](04-building-seats.md).

---

## Constants

| constant | value | where |
|---|---|---|
| `ABS_EVIDENCE_FLOOR` | 0.15 | `guardrail.py` |
| `ABS_POSITIONS_FLOORS["tfidf-cosine"]` | 0.20 | `guardrail.py` |
| `ABS_POSITIONS_FLOORS["sentence-transformers:all-MiniLM-L6-v2"]` | 0.023 | `guardrail.py` — extrapolated, not calibrated |
| `NOVELTY_FLOOR` | 0.25 | `guardrail.py` |
| `FIDELITY_FLOOR` | 0.25 | `guardrail.py` |
| `MIN_PARTICIPATION` | 0.6 | `guardrail.py` |
| `RHO_HIGH` | 1.20 | `guardrail.py` — engineering default |
| `NULL_SATURATION` | 0.95 | `diversity.py` |
| `RHO_BASE / RHO_W_PACKET / RHO_W_CONTEST` | 0.60 / 0.30 / 0.20 | `diversity.py` |
| full-text floors | 6000 chars, 5 passages | `expertwins/corpus/models.py` |

All of these are conventions chosen so that flags fire where a reader would want
to look. None is estimated from a labelled corpus. See
[06-measurement.md](06-measurement.md#what-these-numbers-are-not).

---

## Verdicts and ladder rungs

Verdicts: `healthy`, `independent_corroboration`, `contested_evidence`,
`collapsing`, `collapsed`, `reference_turn`, `scattered`, `fidelity_lost`,
`undetermined`, `none`.

Ladder, in order: `route_to_most_distant_peers`,
`withhold_the_consensus_evidence`, `reanchor_to_own_corpus`,
`anonymize_the_table`, `isolate_from_cross_talk`,
`assign_dissent_to_the_evidenced_minority` (the only prompt-level rung),
`stop_deliberating_and_report`, `declare_collapse`.

---

## Dependencies

Read path (packets, verification, diversity, guardrail, transcript): PyYAML,
numpy, pydantic. No GPU, no network, no model.

Acquisition only: `requests`, `lxml`, `PyMuPDF`, `truststore`. Not needed on an
air-gapped compute node.

Optional: `sentence-transformers` for a real encoder on the positions axis.
Install it on every node or none, and set `EXPERTWINS_EMBED_MODEL`
identically everywhere.

`truststore` exists for networks that intercept TLS: `requests` verifies against
certifi's fixed bundle of public roots, so a corporate proxy's private CA — which
lives in the OS trust store — fails verification, and the source then looks
*unreachable* rather than *untrusted*. See `expertwins/netenv.py`.

---

## `expertwins.corpus`

`expertwins.corpus` is the corpus layer in the main package. It provides the
acquisition clients for Europe PMC and NCBI, the acquisition pipeline, and the
HTTP fetcher with per-host throttling and caching. It also provides the store
for content-addressed documents with derived full-text status, the FTS5 index,
and PDF/JATS extraction.

The corpus layer has two integrity properties. First, `classify_fulltext` in
`expertwins.corpus.store` derives full-text status from the stored bytes and
passage count; callers do not assert full-textness. Second,
`title_appears_in_text` in `expertwins.corpus.store` screens stored text whose
title evidence does not match the intended document. Only acquisition code uses
the network; packet construction, verification, scoring, rendering, and
trajectory analysis read local files and indexes.

## Test suite

Run the suite with:

```bash
python -m pytest tests -q -ra
```

Skips are expected on a fresh clone. Tests that need a corpus, run artifact, or
seat that is absent call `pytest.skip()` with a reason rather than passing
vacuously.

`tests/conftest.py` pins
`EXPERTWINS_EMBED_MODEL=__force_tfidf_fallback__` at import time with
`os.environ.setdefault`. The calibrated tf-idf backend is used regardless of
what optional packages are installed. The neural and tf-idf backends differ in
scale, so a backend-dependent suite would not be a regression suite. The pin
also keeps the run hermetic and brief (a few seconds on a workstation).

To exercise the neural path deliberately, set the variable explicitly:

```bash
EXPERTWINS_EMBED_MODEL=all-MiniLM-L6-v2 python -m pytest tests/test_diversity.py
```

Expected fresh-clone result: 177 passed, 2 skipped.

---

## Known rough edges

- On a small corpus the precision retrieval stratum disables itself and prints
  `precision query: (none -- too few discriminative terms)`. Harmless.
- `THIN PACKET` warnings report question-term coverage, not document count:
  they fire when under half the question's terms appear in the retrieved
  evidence.
- A question composed entirely of stopwords raises `ValueError` rather than
  returning an empty packet.
- `audit_packet()` expects at least 8 documents (`MIN_DOCS_EXPECTED`).

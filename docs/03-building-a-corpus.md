# 3. Building a corpus

The seat specification shapes retrieval and instructions; the corpus determines
which evidence the seat can cite. Corpus construction is therefore part of the
instrument rather than a preprocessing detail.

## 3.1 What a library looks like on disk

```
library/
  MANIFEST.jsonl          append-only; one JSON row per stored document
  index.sqlite            the FTS5 full-text index; retrieval reads this
  docs/
    br/
      brindle2011peloria/
        meta.yaml         DocMeta, written as JSON (which is valid YAML)
        text.jsonl        one JSON object per passage, in order
  quarantine/             documents that failed an integrity screen
```

Several properties are invariants:

- `MANIFEST.jsonl` is append-only. The latest row for a `doc_id` wins.
  Nothing rewrites it in place.
- The shard directory is the first two characters of the `doc_id`. It exists
  so a corpus of a hundred thousand documents does not put a hundred thousand
  entries in one directory.
- Retrieval reads only `index.sqlite`. The document tree is needed for
  verification (the stored bytes a quote is checked against) and for rebuilding
  the index, not for search.
- `doc_id` must match `^[a-z0-9][a-z0-9_.-]{1,79}$`.

### The passage format

`text.jsonl` contains one object per line:

```json
{"passage_id": "brindle2011peloria#p0000", "order": 0, "section": "Abstract", "text": "..."}
```

`order` starts at 0 and increases. `text` must be at least three words and is
stored canonicalised. There is no automatic splitter: the loader decides where
passages begin and end. Passage boundaries are the units available for
quotation, and character-count chunking can produce quotes that straddle two
ideas.

### Full text is measured, not declared

```
metadata_only   0 passages or 0 characters
full            n_chars >= 6000 AND n_passages >= 5
partial         anything in between
```

`DocMeta` rejects `fulltext_status: full` below those floors. This storage
invariant prevents downstream code from treating an abstract or short excerpt as
complete full text.

Short documents are correctly `partial`, and `partial` is usable: a seat can
quote from a document represented by partial text. The `partial` status prevents
downstream code from mistaking an abstract for complete text.

## 3.2 Integrity screens

`classify_fulltext` implements the full-text classification above.

`title_appears_in_text` checks that a document's own title appears somewhere in
its own text. This detects bibliographic/body mismatches, including cases where
the stored text refers to a different work than the metadata record.

Hash integrity and bibliographic integrity are different properties.

Both functions live in `expertwins.corpus.store`.

A third screen is a repair tool rather than a gate. `ops/repair_passage_ids.py`
fixes documents whose passages carry the `doc_id` of a *different* document — a
slug-disambiguation error that only appears when separate acquisition roots are
merged, because the colliding pair has to coexist in one index before the
`UNIQUE` constraint can notice. Run it if a reindex fails with
`UNIQUE constraint failed: passages.passage_id`.

## 3.3 Acquisition

Acquisition is the only part of the system that touches the network. The read
path — packets, verification, measurement, transcript — does not.

```bash
pip install -e ".[acquire]"
```

### Channels

| tool | source | reaches |
|---|---|---|
| `ops/acquire.py` | Europe PMC | the biomedical journal literature |
| `ops/acquire_preprints.py` | arXiv + Europe PMC `SRC:PPR` | conference work (cs.CV, cs.LG, …), bioRxiv, medRxiv |
| `ops/acquire_canon.py` | OpenAlex + open-access PDF, by exact title | the field's landmarks |
| `ops/neighbourhood.py` | citation graph | what a scientist *reads* |

Each one writes into a separate root:

```bash
python ops/acquire.py --config config/people/mira_brindle.yaml --root library_brindle
```

Separate roots prevent manifest tearing: two processes appending to one
`MANIFEST.jsonl` can interleave lines and tear rows. Acquire in parallel into
separate roots, then merge through the ordinary store path so every row is
re-derived from stored bytes rather than copied from a source manifest.

### Acquisition config

An acquisition node lives inside a seat spec:

```yaml
acquisition:
  - id: brindle.authored        # becomes a topic on every document it fetches
    min_full: 30                # the floor, committed BEFORE anything is fetched
    probe: "lanternmoss prism spores"
    queries:
      - 'AUTHOR:"Mira Brindle" AND lanternmoss'
```

The two query languages look alike and must not be mixed. `queries:` under
`acquisition:` is Europe PMC syntax. `retrieval_queries:` and `own_queries:` are
SQLite FTS5 syntax. They share operators and disagree about most other details.
Mixing them can produce a query that runs without error and returns unintended
records.

### Why a floor is not enough

`min_full` is committed before acquisition so that the number cannot be chosen
afterwards to match what happened to arrive. Commit it, then find out.

But a floor counts documents, and it cannot notice which document is absent.
`ops/check_canon.py` exists for that gap: it reads
`config/canon_check.yaml` — a list of title fragments designated
non-negotiable — and reports which are missing from the index, however healthy
the counts look.

```yaml
# config/canon_check.yaml
foundational_methods:
  "a title fragment that must be present": "short label for the report"
```

Similarly, `ops/check_contamination.py` reads `config/contamination.yaml` and
asserts two things about a seat's own corpus: that specific papers the seat
did *not* write are absent (`contaminants`), and that specific papers it *did*
write are present (`keepers`). It exits non-zero if either fails. Attribution is
the step most likely to be quietly wrong, and this check exposes errors not
visible from document counts alone.

## 3.4 Merging and indexing

```bash
python ops/people.py install config/people/mira_brindle.yaml --root library_brindle
python ops/people.py reindex
python ops/doctor.py
```

`install` merges the per-seat root into the main library, rebuilds the index,
and registers the seat in `config/seats.yaml`.

`config/seats.yaml` is a generated file. Do not hand-edit it. The example
checked into this repository is generated-shaped so the toy example works out
of the box; after seat construction it becomes `people.py install`'s output.

## 3.5 Starting from an existing corpus

The acquisition tools are optional. Locally held text can be written into the
library format directly. The clearest worked example is
[`examples/toy/make_toy_corpus.py`](../examples/toy/make_toy_corpus.py),
which is about 370 lines of pure standard library and builds a complete,
indexed, retrievable corpus from nothing.

Read it top to bottom before writing a loader. It does, in order:

1. build the passage list, giving each passage the id `<doc_id>#p<order>` with
   the order zero-padded to four digits, and canonicalise each passage's text;
2. compute `n_chars` and `n_passages`, then ask `classify_fulltext` what the
   status is;
3. write `meta.yaml` and `text.jsonl` into the sharded path;
4. append one row to `MANIFEST.jsonl`;
5. create the SQLite schema, insert `docs`, `doc_topics` and `passages`, and
   populate the `passages_fts` external-content FTS5 table.

One required detail: `passages_fts.rowid` must equal `passages.rowid`. It is an
external-content table, so an offset there does not raise an error; it returns
the wrong passage for every hit.

## 3.6 Checking the corpus before model dispatch

```bash
python ops/doctor.py
```

It audits every manifest row against its document directory (text hash, passage
and character counts, full-text status, passage identity and title), compares
the index with the manifest, runs a probe query through FTS that must return the
passage it was drawn from, and checks that every registered seat is present.
`--quick` skips the file-level audit on a large corpus. A seat with no evidence
produces the words "no evidence found"; in a transcript, that can be mistaken
for a genuine negative result.

Then, per seat:

```bash
python ops/people.py verify config/people/mira_brindle.yaml
```

`verify` checks two things:

1. the literature is really there — the seat's `probe` actually returns
   documents. A seat can hold a hundred documents and have a probe that returns
   zero, which means its retrieval queries and its corpus are about different
   subjects. Such a seat still "works": fluent output, arguing from papers that
   merely contain the right words.
2. this seat is distinct from everyone already at the table — pairwise
   retrieval overlap above roughly 35% means one seat is duplicating another
   under a second label, and the panel's apparent diversity is an artefact.

## Next

→ [4. Building seats](04-building-seats.md)

# 09 — Running on HPC

Three Slurm scripts are shipped in `hpc/`. They are templates; the dispatch
script requires provider configuration before it will run.

| script | what it does |
|---|---|
| `panel_prepare.sbatch` | `doctor.py` + `prepare`, on a compute node |
| `panel_agents.sbatch` | one array task per packet — the dispatch step |
| `analyze_seat.sbatch` | run the code one seat wrote under `--task analyze` |
| `env_example.md` | template for the `--material` block |

---

## Staging the project

```bash
export EXPERTWINS_ROOT=/scratch/$USER/expertwins
cd "$EXPERTWINS_ROOT"
python -m venv .venv && . .venv/bin/activate
pip install -e .
mkdir -p logs
python -c "from expertwins import paths; print(paths.describe())"
```

Nothing in the tree is hard-coded to a location. `EXPERTWINS_ROOT` is the root
configuration variable, and run manifests store paths relative to it so a run
directory can be copied between machines and still read.

If the compute nodes have no network, perform acquisition on a login node or a
data-transfer node and copy the library across. The read path — packets,
verification, diversity, guardrail, transcript — never touches the network.

### Stage the index to node-local scratch

SQLite over a parallel filesystem is slow and can produce lock contention.
`panel_prepare.sbatch` already picks up a staged index automatically:

```bash
cp library/index.sqlite "${TMPDIR:-/tmp}/index.sqlite"
# or explicitly:
export EXPERTWINS_INDEX="${TMPDIR:-/tmp}/index.sqlite"
```

---

## Acquisition in parallel

Parallel acquisition runs must write to separate roots. Two processes
appending to one `MANIFEST.jsonl` interleave lines and tear rows. The code
provides the separate roots — `paths.seat_library(name)` gives
`library_seat_<name>` — and `people.py merge` combines them afterwards.

```bash
# one job per seat, each on its own root
python ops/people.py acquire config/people/<seat>.yaml
# then, once, serially:
python ops/people.py merge
python ops/people.py reindex
python ops/doctor.py
```

Do not parallelise `merge` or `reindex`.

---

## `panel_prepare.sbatch`

```bash
sbatch hpc/panel_prepare.sbatch runs/rt-01 "your question" seat_a,seat_b
```

Single-core, no GPU, seconds to a minute. It can run on a login node; the script
exists for sites where that is discouraged.

It runs `doctor.py` first and aborts if the library fails its own audit, because
downstream measurements depend on corpus integrity.

It ends by reporting the array size for the next step and printing this warning:

> Read the preflight output before dispatching. A seat showing 0 passages is a
> refusal: it will answer "no evidence found", which reads identically to a
> genuine negative result.

---

## `panel_agents.sbatch` — one array task per packet

```bash
N=$(ls runs/rt-01/turns/1/packets/*.md | wc -l)
sbatch --array=0-$((N-1)) hpc/panel_agents.sbatch runs/rt-01 1
```

The array structure is not only a scheduling convenience. It makes the design
constraint structural:

> One agent must not receive two packets.

The isolation between seats is the only reason turn 1 means anything: it is the
sole measurement of what these scientists produce when they cannot influence
each other, and the entire heterogeneity band is anchored on it. If an agent is
handed two packets, that reference condition is lost even though downstream
numbers still compute.

### Configure this file

The script exits 2 with an error until a provider is configured. This repository
does not call a model ([05-running-a-panel.md](05-running-a-panel.md) explains
why).

The packet is a complete, self-contained prompt. The agent must write valid
JSON to `$OUT`. Tell it, in substance:

> Read the packet in full and follow its instructions exactly. Write your answer
> as valid JSON to `<OUT>`. Write the file before composing any reply. Every
> quote must be copied character-for-character from a single passage in your
> packet — it is checked by exact string match. Paraphrase fails. Reconstruction
> from memory fails.

The "write the file first" instruction prevents loss of completed work when an
agent response is interrupted or truncated.

Example shapes:

```bash
copilot -p "Read $PACKET in full, follow it exactly, and write your JSON answer to $OUT. Write the file before replying." --allow-all-tools

python tools/my_provider.py --prompt-file "$PACKET" --out "$OUT" \
       --model "${EXPERTWINS_MODEL:-your-frontier-model}"
```

### On mixing model families

Using different model families across seats is permissible when the tier is held
constant, for example all frontier-class models.

Avoid mixing a frontier model with a weak one. A strength mismatch can dominate
the axes on this page, measuring capability rather than independence.

Record which model ran in which seat. It is part of the method.

### After the array completes

```bash
python ops/panel.py ingest rt-01
```

The `commit <hash>` line confirms that the turn was frozen; `ingest` returns 1
if it declines to act, for example on a turn that is already frozen.

---

## `analyze_seat.sbatch` — running seat-written code

```bash
sbatch hpc/analyze_seat.sbatch runs/an-01 mira_brindle
```

Under `--task analyze`, seats write code rather than prose. Three warnings are
included in the script.

This runs model-written code. Run it in a container or a restricted account.
The panel refuses to write artifacts outside the run directory, but it cannot
constrain what the code does once it is executed.

Nothing here is verified by the citation checker. Code is not a claim. What is
verified is every methodological commitment the seat's `plan` attributes to
prior work — read `runs/<run>/turns/*/report.json` →
`seats.<seat>.extras.plan` alongside whatever the job produces.

Run each seat separately and compare afterwards. Two scientists handed the
same dataset produce two different pipelines; that is the entire reason for
asking both. Do not merge them into one script "for efficiency" — the difference
between them is the finding.

The script looks for `main.py`, `run.py`, `pipeline.py` or `train.py` as an
entrypoint and fails loudly rather than guessing.

---

## `env_example.md` — the `--material` block

For `--task analyze`, pass a data-and-environment block with `--material`. It is
handed verbatim to every computational seat, and the template in
`hpc/env_example.md` is worth following closely.

Two instructions are central:

Be specific. A seat given vague paths will invent plausible ones, and the
code will not run.

State what is *not* available as plainly as what is. A seat told there is no
external cohort will say so; a seat not told will assume one. Same for internet
access on compute nodes, pre-staged weights, and wall-clock limits.

State the question, not the metric. "Classify outcome status from images" is a
task; "decide whether this cohort supports an image-based triage workflow" is a
question, and the two produce different pipelines.

---

## Embedding backends across nodes

When using `sentence-transformers`, install it on every node or on none, and
set `EXPERTWINS_EMBED_MODEL` to the same value everywhere. `embed()` records
which backend ran, and a VS-2 computed over tf-idf is not comparable with one
computed over an encoder.

On compute nodes without network access, pre-stage the model weights and point
the library at them — a runtime download will fail, and on some setups it fails
*slowly*.

`tfidf-cosine` remains the recommended backend for the positions axis because it
has a calibrated absolute floor; see
[06-measurement.md](06-measurement.md#evidence-is-backend-independent-positions-is-not).

---

## Resource shapes

| step | shape |
|---|---|
| acquisition | network-bound; one job per seat on separate roots |
| `merge` / `reindex` | single job, serial, I/O-bound |
| `prepare` / `ingest` / `render` / `trajectory` | 2 CPU, 8 GB, seconds to minutes |
| dispatch | one array task per packet; wall clock set by the configured provider |
| `analyze` seat code | whatever the seat's pipeline needs; GPU if it trains |

`ingest` runtime scales with `--permutations` (default 200) and with claim
count. It is still minutes, not hours.

---

## Checklist

- [ ] `EXPERTWINS_ROOT` exported, `pip install -e .` done, `logs/` created
- [ ] `paths.describe()` shows nothing missing
- [ ] library acquired on separate roots, merged, reindexed
- [ ] `doctor.py` clean
- [ ] `check_canon.py` and `check_contamination.py` have configs and pass
- [ ] `people.py attribute` run for every seat
- [ ] index staged to node-local scratch if the filesystem is slow
- [ ] `panel_agents.sbatch` edited with the configured provider
- [ ] embedding backend identical on every node
- [ ] preflight read and approved before dispatch

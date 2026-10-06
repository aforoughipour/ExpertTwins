# Data and environment block — template for `--task analyze`

Pass this file with `--material`. Everything below is handed verbatim to each
computational seat. Be specific. A seat given vague paths will invent plausible
ones, and the code will not run.

Two competent computational seats handed this block may produce different
pipelines. The block should specify the task and constraints without
pre-deciding the parts where their analyses may differ.

---

## The data

| item | value |
|---|---|
| cohort | *e.g. cohort-alpha, 594 samples, 618 records* |
| data format | *e.g. pyramidal TIFF, scanner model, 40× base magnification, resolution 0.25 units/pixel* |
| location | `/scratch/$USER/data/cohort-alpha/records/` |
| labels | `/scratch/$USER/data/cohort-alpha/labels.csv` — columns `case_id,record_id,label,site,batch` |
| label balance | *e.g. 87 positive, 507 negative* |
| site / centre column | `site` — 26 submitting sites |
| held-out cohort | `/scratch/$USER/data/external/` — *e.g. 210 records, 1 centre, different acquisition protocol* |
| precomputed features | *e.g. none / encoder features at `/scratch/$USER/feat/` — say which encoder and which resolution* |

State anything that is not available as plainly as what is. A seat that
knows there is no external cohort will say so; a seat that is not told will
assume one.

## The environment

| item | value |
|---|---|
| nodes | *e.g. 1× A100 80GB, 8 CPU, 64 GB RAM, 8 h wall clock* |
| python | 3.12 |
| available | `torch 2.4`, `numpy`, `pandas`, `scikit-learn`, `openslide-python`, `h5py` |
| not available | *e.g. no internet on compute nodes — no model may be downloaded at runtime* |
| pre-staged weights | `/scratch/$USER/weights/` — *list exactly what is there* |
| scratch for outputs | `runs/<run>/work/<seat>/` — write everything here |

## The question

*State the decision, not the metric.* "Classify outcome status from images" is a
task; "decide whether this cohort supports an image-based triage workflow" is a
question, and the two produce different pipelines.

## Imposed constraints

- *e.g. splits must be by plot, and stratified by submitting site*
- *e.g. any reported number must come with variance across ≥3 seeds*
- *e.g. the external cohort may be touched exactly once, at the end*

## Known dataset limitations

*List it.* Withholding a known confounder to test whether the seat detects it is
an experimental choice. Record that choice explicitly; otherwise a seat may
spend its analysis on a batch effect that was already known.

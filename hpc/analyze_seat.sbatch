#!/bin/bash
#SBATCH --job-name=expertwins-analyze
#SBATCH --output=logs/analyze-%j.out
#SBATCH --error=logs/analyze-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#
# Run the code ONE SEAT WROTE under `--task analyze`.
#
# Two scientists handed the same dataset produce two different pipelines. That
# is the entire reason for asking both, so each seat's code is run SEPARATELY,
# in its own directory, with its own environment, and the results are compared
# afterwards. Do not merge them into one script "for efficiency": the
# difference between them is the finding.
#
# NOTHING HERE IS VERIFIED BY THE CITATION CHECKER. Code is not a claim. What
# IS verified is every methodological commitment the seat's `plan` attributes
# to prior work. Read `runs/<run>/turns/*/report.json` -> seats.<seat>.extras.plan
# alongside whatever this job produces.
#
# THIS RUNS MODEL-WRITTEN CODE. Run it in a container or a restricted account.
# The panel refuses to write artifacts outside the run directory, but it cannot
# constrain what the code does once you execute it.
#
# Usage:
#   sbatch hpc/analyze_seat.sbatch runs/an-01 mira_brindle

set -euo pipefail

RUN=${1:?usage: analyze_seat.sbatch <run_dir> <seat>}
SEAT=${2:?usage: analyze_seat.sbatch <run_dir> <seat>}

: "${EXPERTWINS_ROOT:?set EXPERTWINS_ROOT to the project root}"
cd "$EXPERTWINS_ROOT"

WORK="$RUN/work/$SEAT"
[[ -d "$WORK" ]] || { echo "no artifacts at $WORK -- did you ingest the run?" >&2; exit 1; }

echo "=== $SEAT ==="
ls -la "$WORK"

# module load python/3.12 cuda/12.4      # site-specific
# source "$EXPERTWINS_ROOT/.venv/bin/activate"

cd "$WORK"

# The seat is instructed to mark exactly one file as the entrypoint. Prefer it;
# fall back to the conventional names rather than guessing at random.
ENTRY=""
for cand in main.py run.py pipeline.py train.py; do
    [[ -f "$cand" ]] && { ENTRY="$cand"; break; }
done
[[ -n "$ENTRY" ]] || { echo "no entrypoint found in $WORK" >&2; exit 1; }

echo "running $ENTRY"
srun python "$ENTRY" 2>&1 | tee "run.log"

echo
echo "Results are in $WORK. Compare them ACROSS seats, not in isolation:"
echo "  the point of asking two computational scientists is that they disagree."

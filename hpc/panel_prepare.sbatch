#!/bin/bash
#SBATCH --job-name=expertwins-prepare
#SBATCH --output=logs/prepare-%j.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# prepare / ingest / render are single-core, no-GPU and take seconds to a
# minute. They can run on a login node; this file exists for sites where that
# is discouraged.
#
# Usage:
#   sbatch hpc/panel_prepare.sbatch runs/rt-01 "your question" seat_a,seat_b

set -euo pipefail

OUT=${1:?usage: panel_prepare.sbatch <out_dir> <question> [seats]}
QUESTION=${2:?usage: panel_prepare.sbatch <out_dir> <question> [seats]}
SEATS=${3:-}

: "${EXPERTWINS_ROOT:?set EXPERTWINS_ROOT to the project root}"
cd "$EXPERTWINS_ROOT"

# SQLite over a parallel filesystem is slow and occasionally lock-flaky. If the
# index has been staged to node-local scratch, use it.
if [[ -f "${TMPDIR:-/tmp}/index.sqlite" ]]; then
    export EXPERTWINS_INDEX="${TMPDIR:-/tmp}/index.sqlite"
    echo "using node-local index: $EXPERTWINS_INDEX"
fi

python ops/doctor.py || { echo "doctor.py failed -- fix this first" >&2; exit 1; }

if [[ -n "$SEATS" ]]; then
    python ops/panel.py prepare "$QUESTION" --out "$OUT" --seats "$SEATS"
else
    python ops/panel.py prepare "$QUESTION" --out "$OUT"
fi

echo
echo "READ THE PREFLIGHT ABOVE BEFORE DISPATCHING."
echo "A seat showing 0 passages is a REFUSAL: it will answer 'no evidence"
echo "found', which reads identically to a genuine scientific negative."
echo
N=$(ls "$OUT"/turns/1/packets/*.md | wc -l)
echo "next:  sbatch --array=0-$((N-1)) hpc/panel_agents.sbatch $OUT 1"

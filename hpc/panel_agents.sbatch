#!/bin/bash
#SBATCH --job-name=expertwins-agents
#SBATCH --output=logs/agent-%A_%a.out
#SBATCH --error=logs/agent-%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# ONE ARRAY TASK PER PACKET. This is not a scheduling convenience -- it is the
# design constraint made structural:
#
#     ONE AGENT MUST NEVER RECEIVE TWO PACKETS.
#
# The isolation between seats is the only reason turn 1 means anything: it is
# the sole measurement of what these scientists produce when they cannot
# influence each other, and the whole heterogeneity band is anchored on it. An
# agent handed two packets has silently destroyed that reference.
#
# Usage:
#   mkdir -p logs
#   N=$(ls $RUN/turns/$TURN/packets/*.md | wc -l)
#   sbatch --array=0-$((N-1)) hpc/panel_agents.sbatch runs/rt-01 1
#
# WHAT YOU MUST EDIT: the CALL THE AGENT section. This repository deliberately
# does not call a model -- see docs/08-reference -- so the one thing it
# cannot supply is your provider.

set -euo pipefail

RUN=${1:?usage: panel_agents.sbatch <run_dir> <turn>}
TURN=${2:?usage: panel_agents.sbatch <run_dir> <turn>}

: "${EXPERTWINS_ROOT:?set EXPERTWINS_ROOT to the project root}"
cd "$EXPERTWINS_ROOT"

PACKETS=( "$RUN"/turns/"$TURN"/packets/*.md )
PACKET="${PACKETS[$SLURM_ARRAY_TASK_ID]}"
SEAT="$(basename "$PACKET" .md)"
OUT="$RUN/turns/$TURN/responses/$SEAT.json"

mkdir -p "$(dirname "$OUT")"
echo "[$SLURM_ARRAY_TASK_ID] seat=$SEAT packet=$PACKET"

# --------------------------------------------------------------------------
# CALL THE AGENT.
#
# The packet is a complete, self-contained prompt. The agent must write valid
# JSON to $OUT. Tell it, in substance:
#
#   Read the packet in full and follow its instructions exactly. Write your
#   answer as valid JSON to <OUT>. WRITE THE FILE BEFORE COMPOSING ANY REPLY.
#   Every quote must be copied character-for-character from a single passage in
#   your packet -- it is checked by exact string match. Paraphrase fails.
#   Reconstruction from memory fails.
#
# The "write the file first" instruction is a durability requirement: agents can
# lose completed work by holding it until the end of a reply.
#
# Example shapes (pick one, delete the rest):
#
#   copilot -p "Read $PACKET in full, follow it exactly, and write your JSON answer to $OUT. Write the file before replying." --allow-all-tools
#
#   python tools/my_provider.py --prompt-file "$PACKET" --out "$OUT" \
#          --model "${EXPERTWINS_MODEL:-your-frontier-model}"
#
# DIFFERENT MODEL FAMILIES ACROSS SEATS is a reasonable choice, holding the
# TIER constant -- all frontier-class. A strength mismatch dominates everything
# else, so do not mix a frontier model with a weak one.
# --------------------------------------------------------------------------

echo "EDIT hpc/panel_agents.sbatch: no agent call is configured." >&2
exit 2

# After the array completes:
#   python ops/panel.py ingest "$(basename "$RUN")"

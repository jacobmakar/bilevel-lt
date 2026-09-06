#!/bin/bash
#SBATCH --job-name=ladder
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Run one job-list file (one fixed-feature cell command per line) on a CPU node, PAR cells at a
# time, with START/END stamps around every cell so scripts/sweep_eta.py can estimate the
# remaining time while the sweep is live. Cells whose JSON exists are skipped by the
# program itself, so resubmitting a list resumes it.
#
#   sbatch --export=ALL,JOBLIST=$OUT/jobs/seed1_cert.txt,PAR=8,OUT=$OUT slurm/jobs.sbatch
#
# REPO, PYTHON and FEATURES may be overridden through the environment.
set -u
REPO=${REPO:-$HOME/code/bilevel-lt}
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
FEATURES=${FEATURES:-$HOME/scratch/bilevel_lt_data/features/dinov2_s_c10.npz}
cd "$REPO"
export PYTHONPATH=$REPO OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$OUT/headladder"
LOG=$OUT/sweep-${SLURM_JOB_ID}.log
[ -s "$FEATURES" ] || { echo "feature cache missing at $FEATURES (python -m bilevel_lt.data)" | tee -a "$LOG"; exit 1; }
{ echo "host=$(hostname) job=${SLURM_JOB_ID} joblist=${JOBLIST} par=${PAR:-8} date=$(date -Iseconds)"; git log --oneline -1 2>/dev/null; } > "$OUT/env-${SLURM_JOB_ID}.log"
echo "$(wc -l < "$JOBLIST") cells, launched $(date) par=${PAR:-8}" | tee -a "$LOG"
sed "s#\\\$OUT#$OUT#g; s#\\\$FEATURES#$FEATURES#g; s#\\\$PYTHON#$PYTHON#g" "$JOBLIST" \
  | xargs -P "${PAR:-8}" -I{} sh -c 'echo "START $(date +%s) {}" >> '"$LOG"'; {} >> '"$LOG"' 2>&1; rc=$?; echo "END $(date +%s) rc=$rc {}" >> '"$LOG"''
echo "SWEEP DONE $(date)" >> "$LOG"

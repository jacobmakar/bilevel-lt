#!/bin/bash
# The fairness panel: the bilevel method against the closed form as a function of the
# validation set size, everything else equal. Per seed and validation size:
#   ab      Neumann-10, LA init (the reference bilevel configuration)
#   la      tau in {1, 1.5, 2, 2.5}, so scripts/autobalance_tables.py can select tau on
#           validation (la-val) and report the test-selected tau as an oracle
#   ce      tau = 0
# Run ON the cluster. Most informative cells are submitted first.
#   OUT=~/scratch/bilevel_lt_runs/panel SEEDS="1 2 3" VALS="100 20" slurm/autobalance_panel.sh
set -eu
cd "$(dirname "$0")/.."
OUT=${OUT:-$HOME/scratch/bilevel_lt_runs/panel}
SEEDS=${SEEDS:-"1 2 3"}
VALS=${VALS:-"100 20"}
TAUS=${TAUS:-"2 1 1.5 2.5"}
mkdir -p "$OUT"
submit() {  # name, env
  sbatch --job-name="$1" --export="ALL,OUT=$OUT,RUN_NAME=$1,$2" slurm/autobalance.sbatch
}
for seed in $SEEDS; do
  for val in $VALS; do
    submit "ab-neumann10-lainit-val${val}-s${seed}" "METHOD=ab,ESTIMATOR=neumann,COST=10,LA_INIT=1,SEED=${seed},VAL=${val}"
    for tau in $TAUS; do
      submit "la-tau${tau}-val${val}-s${seed}" "METHOD=la,TAU=${tau},SEED=${seed},VAL=${val}"
    done
    submit "ce-val${val}-s${seed}" "METHOD=ce,SEED=${seed},VAL=${val}"
  done
done
squeue -u "$USER" -o "%.10i %.34j %.2t %.10M %R" | head -40

#!/bin/bash
#SBATCH --job-name=ab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu3005
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# One end-to-end run. Arm through the environment:
#   METHOD=ab|la|ce  ESTIMATOR=neumann|cg|nystrom|damped|identity  COST=10  ALPHA=0.01
#   LA_INIT=1  NO_DELTA=1  TAU=2  SEED=1  VAL=100  IMB=100  NUM_ITERS=15000
#   sbatch --export=ALL,METHOD=ab,LA_INIT=1,SEED=1,VAL=100,OUT=$OUT slurm/autobalance.sbatch
set -u
REPO=${REPO:-$HOME/code/bilevel-lt}
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
DATA=${DATA:-$HOME/scratch/bilevel_lt_data/cifar}
OUT=${OUT:-$HOME/scratch/bilevel_lt_runs/autobalance}
cd "$REPO"
export PYTHONPATH=$REPO
"$PYTHON" -m bilevel_lt.autobalance \
  --method "${METHOD:-ab}" --estimator "${ESTIMATOR:-neumann}" --cost "${COST:-10}" --alpha "${ALPHA:-0.01}" \
  ${LA_INIT:+--la_init} ${NO_DELTA:+--no_delta} --tau "${TAU:-2}" \
  --seed "${SEED:-1}" --val_per_class "${VAL:-100}" --imbalance "${IMB:-100}" \
  --num_iters "${NUM_ITERS:-15000}" --data_root "$DATA" --out "$OUT/${RUN_NAME:-${SLURM_JOB_NAME}-${SLURM_JOB_ID}}" \
  --workers 4 ${EXTRA:-}

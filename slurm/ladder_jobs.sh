#!/bin/bash
# Generate job lists for a standard fixed-feature sweep: cert at both leader points, ref,
# and loop with each estimator, per seed.
#
#   OUT=~/scratch/bilevel_lt_runs/ladder HEADS="linear dlin16 relu16" SEEDS="1 2 3" slurm/ladder_jobs.sh
#   for f in $OUT/jobs/*.txt; do sbatch --export=ALL,JOBLIST=$f,PAR=8,OUT=$OUT slurm/jobs.sbatch; done
#
# Lines use the placeholders $PYTHON, $FEATURES and $OUT, which slurm/jobs.sbatch fills in.
set -eu
OUT=${OUT:-runs/ladder}
HEADS=${HEADS:-"linear dlin5 dlin16 dlin64 relu16"}
SEEDS=${SEEDS:-"1 2 3"}
POINTS=${POINTS:-"zero la1"}
LOOP_EST=${LOOP_EST:-"identity neumann10 cg100 damped nystrom10"}
EXTRA=${EXTRA:-""}                      # e.g. "--ridge 1e-2" or "--val_per_class 20"
CERT=${CERT:-"--k_list 200,600 --dense_max 13000"}
mkdir -p "$OUT/jobs"
CMD='$PYTHON -m bilevel_lt.ladder --features $FEATURES --out_root $OUT --threads 2'
for seed in $SEEDS; do
  C=$OUT/jobs/seed${seed}_cert.txt; L=$OUT/jobs/seed${seed}_loop.txt; : > "$C"; : > "$L"
  for head in $HEADS; do
    for point in $POINTS; do
      echo "$CMD --head $head --mode cert --point $point --seed $seed $CERT $EXTRA" >> "$C"
    done
    echo "$CMD --head $head --mode ref --seed $seed $EXTRA" >> "$C"
    for point in $POINTS; do
      for e in $LOOP_EST; do
        echo "$CMD --head $head --mode loop --point $point --estimator $e --seed $seed $EXTRA" >> "$L"
      done
    done
  done
  echo "seed $seed: $(wc -l < "$C") cert/ref cells, $(wc -l < "$L") loop cells"
done

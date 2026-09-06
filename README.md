# bilevel-lt

An implementation of bilevel logit adjustment on long-tailed CIFAR-10. The repository has a
shared library that implements data loading, the heads and the inverse-Hessian estimators,
and two experimental pipelines. The fixed-feature pipeline (`bilevel_lt/ladder.py`) trains
a small head on frozen DINOv2 features, where the hypergradient can be checked against a
ground truth. The end-to-end pipeline (`bilevel_lt/autobalance.py`) is a replication of
AutoBalance (Li et al. 2021) on ResNet-32.

**Setting.** A classifier is trained on a long-tailed training set. A leader adds a
per-class offset `l` (optionally a per-class scale) to the logits inside the training
loss, and is fit to minimize the cross-entropy of the raw logits on a small balanced
validation set. The closed-form baseline is logit adjustment, `l = tau * log(pi)` for the
training class priors `pi`. The leader's gradient is the implicit-function-theorem (IFT)
hypergradient `-B^T H^{-1} g_out`, where `H` is the Hessian of the training loss in the
model parameters, `B` is the Jacobian of the training gradient with respect to the leader,
and `g_out` is the gradient of the validation loss in the model parameters.

## Install

```
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

## Data

CIFAR-10 downloads on first use into `--data_root` (default `data/cifar`). The fixed-feature
pipeline reads a DINOv2 ViT-S/14 feature cache of the full train and test sets. Build the
feature cache with the command below, which downloads the backbone through `torch.hub`:

```
.venv/bin/python -m bilevel_lt.data --data_root data/cifar --out data/features/dinov2_s_c10.npz
```

The long-tail split is applied at load time (`bilevel_lt/data.py`): a balanced validation
set of `val_per_class` images per class is created first, then an exponential train set with
ratio `imbalance` between the largest and the smallest class is sampled from the remaining
data. Both pipelines use this split. The feature cache holds the full dataset, so it does
not need to be regenerated when the seed, imbalance or validation size changes.

## Fixed-feature bilevel logit adjustment (`bilevel_lt/ladder.py`)

A small head (`linear`, `dlin16`, `dlin16d3`, `relu32`, ...) on frozen features. The inner
problem is cross-entropy plus a ridge penalty, solved by gradient descent with momentum.
Everything runs in float64 on CPU. There are three modes. Each run is one cell and writes
one JSON file under `--out_root/headladder/`; a cell whose JSON already exists is skipped
unless `--force`, so a resubmitted sweep resumes.

### `cert`: hypergradient estimators against unrolled differentiation

This mode compares two methods of computing the hypergradient. Unrolled differentiation
differentiates the validation loss through the `k` inner training steps that were actually
run, `l -> L_out(A_k(l))`. The IFT hypergradient `-B^T H^{-1} g_out` replaces the inner
steps with the implicit function theorem at a fixed point, and each estimator approximates
`H^{-1} g_out` differently. `cert` holds the leader fixed at `--point` (`zero` or `la<tau>`),
warm-starts the head for `warm_steps` steps, and compares the two. The unrolled
gradient is computed by central finite differences on the `C` leader coordinates, which
costs `2C` extra `k`-step runs per horizon and avoids backpropagating through the unrolled
steps. Every estimator (`identity`, `cg`, `neumann`, `minres`, `damped`, `nystrom`, and the
dense inverse when `p` is small enough) reports the cosine of its hypergradient to the
unrolled one. `--k_list` sets the horizons `k` (default 200 and 600); the unrolled gradient
is recorded at each horizon, and the estimators are evaluated at the head after the longest
training horizon. The cell also records the inner Hessian spectrum. For `p <= dense_max` the
full Hessian is formed and eigendecomposed. Above that only the extremes are computed: the 6
smallest and 3 largest eigenvalues by Lanczos iteration (scipy's `eigsh`).

```
python -m bilevel_lt.ladder --head dlin16 --mode cert --point la1 --seed 1
```

### `loop`: train the leader with one estimator

`loop` runs the outer loop with one estimator. Each outer step runs `k_loop` warm-started
inner steps, forms the hypergradient with `--estimator`, and takes one Adam (or SGD) step
on `l`. Balanced validation and test accuracy are recorded along the way.

```
python -m bilevel_lt.ladder --head dlin16 --mode loop --point zero --estimator damped --seed 1
```

### `ref`: train the closed-form baseline

`ref` trains the closed-form baseline `l = tau * log(pi)` from scratch for each `tau` in
`--tau_grid` and reports balanced validation and test accuracy, so `tau` can be selected on
validation. Each `tau` gets `--ref_steps` inner steps (default `warm_steps`). For a matched
budget note that the loop trains `warm_steps + outer_steps * k_loop` steps in total.

```
python -m bilevel_lt.ladder --head dlin16 --mode ref --seed 1
```

Cell file names come from `bilevel_lt/tags.py`, which the sweep tooling shares.
`slurm/ladder_jobs.sh` writes the job lists for a sweep, `slurm/jobs.sbatch` runs one list
on a CPU node, and `scripts/sweep_eta.py` estimates the remaining time from the logs.
`scripts/ladder_tables.py <dir>` turns a directory of cells into tables.

## Bilevel logit adjustment on a deep network (`bilevel_lt/autobalance.py`)

A replication of AutoBalance (Li et al. 2021): ResNet-32 trained from scratch on images,
with a per-class offset and a per-class scale on the logits as the leader. The inner
problem is cross-entropy with weight decay, solved by SGD with momentum on minibatches. The
weight decay is applied in the SGD step and is not part of `H`, as in AutoBalance. Every
`unroll_steps` inner steps the leader takes one outer step along the IFT hypergradient,
with `H` and `B` evaluated on one training minibatch and `g_out` on one validation
minibatch. There are three methods (`ab`, `la`, and `ce`), chosen with `--method`. Each
run writes `<out>/metrics.json` (default `runs/autobalance/<name>/`) with the training
history, the final test and validation accuracies after BatchNorm calibration (overall, per
class, balanced) and the leader.

### `ab`: the bilevel method

`ab` trains the leader with `--estimator` (`neumann`, `cg`, `nystrom`, `damped`,
`identity`) at `--cost` iterations or rank, using SGD with momentum (or Adam) and a small
weight decay on the leader. `--la_init` starts the offsets at `l = log(pi)`. With the
per-class scale on (the default, as published) the scale starts at 0.5, so the bilevel
arm's inner objective at step 0 differs from LA's. With `--no_delta --la_init` the leader
is offsets only and starts at `l = log(pi)`, so its inner objective at step 0 is exactly LA
with `tau = 1`.

```
python -m bilevel_lt.autobalance --method ab --estimator neumann --cost 10 --la_init --val_per_class 100 --seed 1
```

### `la`: logit adjustment

`la` freezes the leader at `l = tau * log(pi)` and runs the same inner loop, so the only
difference from `ab` is the outer loop.

```
python -m bilevel_lt.autobalance --method la --tau 2 --val_per_class 100 --seed 1
```

### `ce`: plain cross-entropy

`ce` freezes the leader at `l = 0`.

```
python -m bilevel_lt.autobalance --method ce --val_per_class 100 --seed 1
```

`scripts/autobalance_tables.py <dir>` tabulates runs by arm and validation size.

## Estimators (`bilevel_lt/estimators.py`)

All estimators take a flat HVP oracle `v -> H v` and a right-hand side and return an
approximation of `H^{-1} g_out`: `identity`, `cg` (with breakdown detection on indefinite
directions), `neumann`, `damped` (CG on `H + mu I`), `nystrom_sketch` (Gaussian sketch),
`nystrom_columns` (coordinate columns), `minres`, `dense_inverse`, plus the spectral helpers
`dense_hessian`, `lam_max`, `lanczos_extremes`. The fixed-feature pipeline uses
`nystrom_sketch` and a Neumann step size of `1 / lam_max`; the end-to-end pipeline uses
`nystrom_columns` and the published constant step size, as in AutoBalance. The
fixed-feature pipeline adds `damped_track`, damping proportional to the inner gradient norm.

## Layout

```
bilevel_lt/   data.py heads.py estimators.py problem.py ladder.py tags.py resnet.py autobalance.py
scripts/      ladder_tables.py autobalance_tables.py sweep_eta.py
slurm/        jobs.sbatch ladder_jobs.sh autobalance.sbatch autobalance_panel.sh
tests/
```

## References

- Li, Wang, Oymak, Kuo, Zhang, Chandra, Wang. AutoBalance: Optimized Loss Functions for Imbalanced Data. NeurIPS 2021.
- Menon, Jayasumana, Rawat, Jain, Veit, Kumar. Long-tail learning via logit adjustment. ICLR 2021.
- Hataya, Yamada. Nyström method for accurate and scalable implicit differentiation. AISTATS 2023.
- Lorraine, Vicol, Duvenaud. Optimizing millions of hyperparameters by implicit differentiation. AISTATS 2020.

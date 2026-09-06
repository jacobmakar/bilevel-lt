# bilevel-lt

Bilevel logit adjustment on long-tailed CIFAR-10, in two pipelines that share one data
protocol and one set of inverse-Hessian estimators.

**Setting.** A classifier is trained on a long-tailed training set. A leader adds a
per-class offset `l` (optionally a per-class scale) to the logits inside the training
loss, and is fit to minimize the cross-entropy of the raw logits on a small balanced
validation set. The closed-form reference is logit adjustment, `l = tau * log(pi)` for the
training class priors `pi`. The leader's gradient is the implicit-function-theorem
hypergradient `-B^T H^{-1} g_out`, and the question throughout is what happens to it when
`H`, the training Hessian, stops being positive definite.

## Install

```
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

## Data

CIFAR-10 downloads on first use into `--data_root` (default `data/cifar`). The fixed-feature
pipeline needs one DINOv2 ViT-S/14 feature cache of the full train and test sets (a GPU and
internet access for `torch.hub`, a few minutes):

```
.venv/bin/python -m bilevel_lt.data --data_root data/cifar --out data/features/dinov2_s_c10.npz
```

The long-tail split is applied at load time (`bilevel_lt/data.py`): a balanced validation
set of `val_per_class` images per class is carved first, then an exponential profile with
ratio `imbalance` is sampled from the rest. One cache serves every seed, imbalance and
validation size.

## Fixed-feature ladder (`bilevel_lt/ladder.py`)

A small head (`linear`, `dlin16`, `dlin16d3`, `relu32`, ...) on frozen features, float64 on
CPU, so everything can be certified. Three modes, one JSON per cell under
`--out_root/headladder/`:

```
python -m bilevel_lt.ladder --head dlin16 --mode cert --point la1 --seed 1     # estimators vs finite differences + spectrum
python -m bilevel_lt.ladder --head dlin16 --mode loop --point zero --estimator damped --seed 1
python -m bilevel_lt.ladder --head dlin16 --mode ref --seed 1                  # logit adjustment over a tau grid
```

`cert` differentiates the k-step warm-started training map `l -> L_out(A_k(l))` by central
finite differences, which is well defined even where the training problem has no unique
minimizer, and compares every estimator against it; it also records the inner Hessian
spectrum (dense for `p <= dense_max`, Lanczos extremes above). `loop` runs the outer loop
with one estimator. `ref` trains the closed form for each `tau` and reports validation and
test balanced accuracy, so the temperature can be selected on validation.

`scripts/ladder_tables.py <dir>` turns a directory of cells into tables.

## End to end (`bilevel_lt/autobalance.py`)

ResNet-32 on images, the bilevel method as it is used in practice (AutoBalance, Li et al.
2021): SGD with momentum on the network, an outer step on the leader every `unroll_steps`
inner steps, the estimator evaluated on one training minibatch and one validation
minibatch. The closed-form baselines run through the same inner loop with the leader
frozen, so the only difference is the outer loop:

```
python -m bilevel_lt.autobalance --method ab --estimator neumann --cost 10 --la_init --val_per_class 100 --seed 1
python -m bilevel_lt.autobalance --method la --tau 2 --val_per_class 100 --seed 1
python -m bilevel_lt.autobalance --method ce --val_per_class 100 --seed 1
```

Each run writes `<out>/metrics.json` with the training history, the final calibrated test
and validation accuracies (overall, per class, balanced) and the leader.
`scripts/autobalance_tables.py <dir>` tabulates runs by arm and validation size.

## Comparison protocol

Whether the bilevel method matches or beats logit adjustment depends on how much balanced
validation data it gets, so `val_per_class` is a first-class argument and the tables report
it as an axis. The rules: every arm trains on the same long-tailed set with the validation
images removed; the bilevel leader fits `l` on validation; the closed form's `tau` is
selected on the same validation set (`la-val`), with `tau = 1` as the no-validation
baseline and the test-selected `tau` reported only as an oracle upper bound.
`slurm/autobalance_panel.sh` submits this panel.

## Estimators (`bilevel_lt/estimators.py`)

All on a flat HVP oracle `v -> H v`: `identity`, `cg` (with breakdown detection on
indefinite directions), `neumann`, `damped` (CG on `H + mu I`), `nystrom_sketch` (Gaussian
sketch), `nystrom_columns` (coordinate columns), `minres`, `dense_inverse`, plus the
spectral helpers `dense_hessian`, `lam_max`, `lanczos_extremes`. The ladder adds
`damped_track`, damping proportional to the inner gradient norm.

## Layout

```
bilevel_lt/   data.py heads.py estimators.py problem.py ladder.py resnet.py autobalance.py
scripts/      ladder_tables.py autobalance_tables.py sweep_eta.py
slurm/        jobs.sbatch ladder_jobs.sh autobalance.sbatch autobalance_panel.sh
tests/
```

## References

- Li, Wang, Oymak, Kuo, Zhang, Chandra, Wang. AutoBalance: Optimized Loss Functions for Imbalanced Data. NeurIPS 2021.
- Menon, Jayasumana, Rawat, Jain, Veit, Kumar. Long-tail learning via logit adjustment. ICLR 2021.
- Hataya, Yamada. Nyström method for accurate and scalable implicit differentiation. AISTATS 2023.
- Lorraine, Vicol, Duvenaud. Optimizing millions of hyperparameters by implicit differentiation. AISTATS 2020.

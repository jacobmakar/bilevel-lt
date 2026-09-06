import numpy as np
import torch

from bilevel_lt import estimators as est
from bilevel_lt.problem import FixedFeatureProblem

torch.set_default_dtype(torch.float64)


def synthetic(n_per_class=40, k=6, C=3, seed=0):
    rng = np.random.default_rng(seed)
    means = 2.0 * rng.normal(size=(C, k))
    y = np.repeat(np.arange(C), n_per_class)
    x = means[y] + rng.normal(size=(len(y), k))
    yt = np.repeat(np.arange(C), 30)
    xt = means[yt] + rng.normal(size=(len(yt), k))
    return dict(x_train=x.astype(np.float32), y_train=y, x_test=xt.astype(np.float32), y_test=yt)


def test_hypergradient_sign_and_scale_against_finite_differences():
    """On a strongly convex head the implicit gradient -B^T H^{-1} g_out is the exact
    derivative of L_out(theta*(l)); check sign and scale against central differences."""
    prob = FixedFeatureProblem(synthetic(), 'linear', imbalance=4, val_per_class=5, seed=0, ridge=1e-2)
    l = 0.3 * torch.log(prob.pi)
    theta = prob.init(0)
    lr = 0.5 / est.lam_max(prob.hvp(theta, l), prob.p)
    theta = prob.run_map(theta, l, 4000, lr)
    assert float(prob.g_in(theta, l).norm()) < 1e-6
    v, info = est.cg(prob.hvp(theta, l), prob.g_out(theta), K=prob.p)
    assert not info['breakdown']
    hg = prob.hg_from_v(theta, l, v)
    eps = 1e-4
    fd = torch.zeros(prob.C)
    for j in range(prob.C):
        e = torch.zeros(prob.C)
        e[j] = eps
        fd[j] = (prob.L_out(prob.run_map(theta, l + e, 4000, lr))
                 - prob.L_out(prob.run_map(theta, l - e, 4000, lr))) / (2 * eps)
    assert est.cosine(hg, fd) > 0.999
    assert abs(float(hg.norm() / fd.norm()) - 1.0) < 0.02


def test_split_and_priors_from_arrays():
    prob = FixedFeatureProblem(synthetic(), 'dlin4', imbalance=4, val_per_class=5, seed=1)
    assert len(prob.yv) == 15 and torch.bincount(prob.yv).tolist() == [5, 5, 5]
    assert torch.bincount(prob.y).tolist() == [35, 17, 8]
    assert abs(float(prob.pi.sum()) - 1) < 1e-12 and prob.X.dtype == torch.float64

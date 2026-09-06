import numpy as np
import torch

from bilevel_lt.data import class_priors, lt_split
from bilevel_lt.heads import build_head


def test_lt_split_protocol():
    labels = np.repeat(np.arange(10), 5000)
    tr, va = lt_split(labels, imbalance=100, val_per_class=100, seed=1)
    assert len(va) == 1000 and np.bincount(labels[va]).tolist() == [100] * 10
    assert not set(tr) & set(va)
    counts = np.bincount(labels[tr], minlength=10)
    mu = 0.01 ** (1 / 9)
    assert counts.tolist() == [int(4900 * mu ** c) for c in range(10)]
    assert counts[0] // counts[9] in (100, 101)
    tr2, va2 = lt_split(labels, imbalance=100, val_per_class=100, seed=1)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)
    assert not np.array_equal(va, lt_split(labels, 100, 100, seed=2)[1])
    pi = class_priors(labels[tr])
    assert abs(pi.sum() - 1) < 1e-12 and pi[0] > pi[-1]


def test_heads_parameter_count_and_shapes():
    for kind, p in (('linear', 384 * 10 + 10), ('dlin16', 384 * 16 + 16 + 16 * 10 + 10),
                    ('dlin16d3', 384 * 16 + 16 + 2 * (16 * 16 + 16) + 16 * 10 + 10), ('relu32', 384 * 32 + 32 + 32 * 10 + 10)):
        fwd, init, p_, dims = build_head(kind, 384, 10)
        assert p_ == p
        theta = init(0)
        assert theta.shape == (p,) and theta.dtype == torch.float64
        assert fwd(theta, torch.randn(7, 384, dtype=torch.float64)).shape == (7, 10)

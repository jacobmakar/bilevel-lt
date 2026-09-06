"""The fixed-feature bilevel problem.

    inner:  theta*(l) = argmin_theta  CE(head_theta(X) + l, y) + ridge/2 ||theta||^2   (long-tailed train)
    outer:  min_l  CE(head_theta*(l)(X_val), y_val)                                       (balanced val)

The leader l is a per-class logit offset added inside the inner loss; the outer loss sees
raw logits, so it has no direct dependence on l and the hypergradient is purely the
implicit term. Closed-form reference: logit adjustment l = tau * log(pi).
Everything is float64 on CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import grad, jvp

from .data import load_features, lt_split
from .heads import build_head


class FixedFeatureProblem:
    def __init__(self, features: str, head: str, imbalance: int = 100, val_per_class: int = 100,
                 seed: int = 1, ridge: float = 1e-4, momentum: float = 0.9):
        z = load_features(features)
        y_all = z['y_train']
        C = int(y_all.max()) + 1
        tr, va = lt_split(y_all, imbalance, val_per_class, seed, C)
        f64 = lambda a: torch.tensor(a, dtype=torch.float64)
        i64 = lambda a: torch.tensor(a.astype(np.int64))
        self.X, self.y = f64(z['x_train'][tr]), i64(y_all[tr])
        self.Xv, self.yv = f64(z['x_train'][va]), i64(y_all[va])
        self.Xt, self.yt = f64(z['x_test']), i64(z['y_test'])
        counts = np.bincount(self.y.numpy(), minlength=C).astype(np.float64)
        self.pi = torch.tensor(counts / counts.sum())
        self.C, self.k = C, self.X.shape[1]
        self.lam = ridge
        self.mom = momentum
        self.fwd, self.init, self.p, self.dims = build_head(head, self.k, C)

    # losses ------------------------------------------------------------------
    def L_in(self, theta, l):
        return F.cross_entropy(self.fwd(theta, self.X) + l, self.y) + 0.5 * self.lam * (theta @ theta)

    def L_out(self, theta):
        return F.cross_entropy(self.fwd(theta, self.Xv), self.yv)

    def g_in(self, theta, l):
        return grad(self.L_in)(theta, l)

    def g_out(self, theta):
        return grad(self.L_out)(theta)

    def resid_norm(self, theta, l) -> float:
        """||dL/dlogits||: the output residual that scales the indefinite part of the inner
        Hessian, sum_i (dL/dlogit_i) d^2 f_i. Unlike ||g_in|| it excludes the ridge gradient."""
        with torch.no_grad():
            R = torch.softmax(self.fwd(theta, self.X) + l, 1)
            R[torch.arange(len(self.y)), self.y] -= 1.0
            return float((R / len(self.y)).norm())

    # curvature ---------------------------------------------------------------
    def hvp(self, theta, l):
        """HVP oracle of the inner Hessian at (theta, l), for the estimators."""
        g = lambda t: self.g_in(t, l)
        return lambda u: jvp(g, (theta,), (u,))[1]

    def hg_from_v(self, theta, l, v):
        """-B^T v with B = d_l d_theta L_in: one reverse pass through <g_in(theta, l), v>."""
        f = lambda ll: torch.dot(self.g_in(theta, ll), v)
        return -grad(f)(l)

    # the inner algorithm whose derivative is the ground truth ----------------
    def run_map(self, theta0, l, k, lr):
        """k steps of gradient descent with momentum from theta0 at leader l."""
        theta, v = theta0.clone(), torch.zeros_like(theta0)
        for _ in range(k):
            v = self.mom * v + self.g_in(theta, l)
            theta = theta - lr * v
            if not torch.isfinite(theta).all():
                return theta
        return theta

    # evaluation --------------------------------------------------------------
    def bacc(self, theta, split: str = 'test') -> float:
        X, y = (self.Xt, self.yt) if split == 'test' else (self.Xv, self.yv)
        with torch.no_grad():
            pred = self.fwd(theta, X).argmax(1)
            correct = (pred == y).double()
            cnt = torch.bincount(y, minlength=self.C).double()
            rec = torch.bincount(y, weights=correct, minlength=self.C) / cnt
            return float(rec.mean())

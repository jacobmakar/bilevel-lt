"""Inverse-Hessian-vector-product estimators.

Every estimator maps an HVP oracle `hvp(v) = H v` and a right-hand side `b` to an
approximation of H^{-1} b and an info dict, on flat vectors. Two Nystrom variants are kept
on purpose: `nystrom_sketch` (Gaussian sketch) is used by the fixed-feature pipeline,
`nystrom_columns` (coordinate columns) is the one AutoBalance was published with. The
Neumann step size is chosen by the caller: the fixed-feature pipeline uses 1/lambda_max,
the end-to-end pipeline the published constant.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.func import vmap

HVP = Callable[[Tensor], Tensor]


def identity(hvp: HVP, b: Tensor):
    """No inverse at all: the hypergradient of a one-step inner map."""
    return b, {}


def cg(hvp: HVP, b: Tensor, K: int, tol: float = 1e-12, shift: float = 0.0):
    """Conjugate gradient on (H + shift I) v = b. With shift = 0 this is the plain solver;
    with shift = mu > |lambda_min| the system is positive definite (the damped solve).
    Stops with `breakdown=True` on an indefinite direction p with p^T (H + shift I) p <= 0."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = float(r @ r)
    bnorm = float(b.norm())
    broke, its = False, 0
    for i in range(K):
        Hp = hvp(p) + shift * p
        pHp = float(p @ Hp)
        if pHp <= 0:
            broke = True
            break
        a = rs / pHp
        x = x + a * p
        r = r - a * Hp
        rs_new = float(r @ r)
        its = i + 1
        if math.sqrt(rs_new) < tol * bnorm:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    resid = float((hvp(x) + shift * x - b).norm() / bnorm)
    return x, dict(iters=its, breakdown=broke, resid=resid, shift=shift)


def damped(hvp: HVP, b: Tensor, mu: float, K: int = 2000, tol: float = 1e-8):
    """(H + mu I)^{-1} b by CG. Works iff mu > |lambda_min(H)|."""
    return cg(hvp, b, K, tol=tol, shift=mu)


def neumann(hvp: HVP, b: Tensor, K: int, alpha: float):
    """Truncated Neumann series alpha * sum_{j<=K} (I - alpha H)^j b, which is H^{-1} b as
    K -> inf when 0 < alpha < 2 / lambda_max(H). Every finite K is a spectral filter:
    directions with lambda << 1/(alpha K) are scaled by (K+1) alpha, not 1/lambda."""
    x = b.clone()
    term = b.clone()
    for _ in range(K):
        term = term - alpha * hvp(term)
        x = x + term
    x = alpha * x
    resid = float((hvp(x) - b).norm() / b.norm())
    return x, dict(iters=K, alpha=alpha, resid=resid)


def nystrom_sketch(hvp: HVP, b: Tensor, p: int, k: int, rho: float, seed: int = 0):
    """Rank-k Nystrom approximation Hhat = Y C^{-1} Y^T with a Gaussian sketch Omega
    (Y = H Omega, C = Omega^T Y), then (Hhat + rho I)^{-1} b by Woodbury. Directions outside
    the sketched top subspace are scaled by 1/rho."""
    g = torch.Generator().manual_seed(seed)
    Omega = torch.randn(p, k, generator=g, dtype=b.dtype).to(b.device)
    Y = torch.stack([hvp(Omega[:, j]) for j in range(k)], dim=1)
    C = Omega.T @ Y
    C = 0.5 * (C + C.T)
    M = rho * C + Y.T @ Y
    x = (b - Y @ torch.linalg.solve(M, Y.T @ b)) / rho
    resid = float((hvp(x) - b).norm() / b.norm())
    return x, dict(rank=k, rho=rho, resid=resid)


def nystrom_columns(hvp: HVP, b: Tensor, p: int, k: int, rho: float,
                    generator: torch.Generator | None = None):
    """Nystrom with k random coordinate columns of H (Hataya & Yamada 2023): rows C = H[S, :]
    from k HVPs against unit vectors, M = H[S, S], then Woodbury on (Hhat + rho I)."""
    idx = torch.randperm(p, generator=generator, device=b.device)[:k]
    rows = []
    for i in idx:
        e = torch.zeros_like(b)
        e[i] = 1.0
        rows.append(hvp(e))
    C = torch.stack(rows)                       # k x p
    M = C[:, idx]                               # k x k
    Cb = C @ b
    x = b / rho - (C.T @ torch.linalg.solve(M + C @ C.T / rho, Cb)) / rho ** 2
    resid = float((hvp(x) - b).norm() / b.norm())
    return x, dict(rank=k, rho=rho, resid=resid)


def minres(hvp: HVP, b: Tensor, p: int, tol: float = 1e-10, maxiter: int = 3000):
    """Iterative solve of the symmetric, possibly indefinite system H v = b (scipy MINRES).
    On heads whose Hessian is indefinite it typically hits `maxiter` without converging and
    returns a filtered iterate, not the inverse: check `resid`."""
    from scipy.sparse.linalg import minres as _minres
    x, info = _minres(scipy_operator(hvp, p, b.dtype, b.device), b.detach().cpu().numpy(), rtol=tol, maxiter=maxiter)
    x = torch.as_tensor(np.asarray(x), dtype=b.dtype, device=b.device)
    resid = float((hvp(x) - b).norm() / b.norm())
    return x, dict(info=int(info), resid=resid, maxiter=maxiter)


def dense_inverse(hvp: HVP, b: Tensor, p: int):
    """The true H^{-1} b by a dense eigendecomposition (p up to ~1e4). It is exact, so it
    also inherits every near-zero and negative eigenvalue of H."""
    H = dense_hessian(hvp, p, dtype=b.dtype, device=b.device)
    evals, evecs = torch.linalg.eigh(H)
    c = evecs.T @ b
    x = evecs @ (c / evals)
    lmax = float(evals.max())
    info = dict(lam_min=float(evals.min()), lam_max=lmax,
                n_neg=int((evals < -1e-8 * lmax).sum()),
                neg_mass=float((c[evals < 0] ** 2).sum() / (c ** 2).sum()),
                resid=float((H @ x - b).norm() / b.norm()))
    return x, info


# ---------------------------------------------------------------------------
# spectral helpers
# ---------------------------------------------------------------------------
def dense_hessian(hvp: HVP, p: int, chunk: int = 64, dtype=None, device=None) -> Tensor:
    I = torch.eye(p, dtype=dtype, device=device)
    H = torch.empty(p, p, dtype=dtype, device=device)
    for s in range(0, p, chunk):
        U = I[s:s + chunk]
        try:
            H[s:s + chunk] = vmap(hvp)(U)
        except Exception:                       # oracles that are not vmap-able
            H[s:s + chunk] = torch.stack([hvp(u) for u in U])
    return 0.5 * (H + H.T)


def lam_max(hvp: HVP, p: int, iters: int = 30, seed: int = 0, like: Tensor | None = None) -> float:
    """Largest eigenvalue by power iteration (used to scale Neumann and damping). `like`
    sets the dtype and device of the probe vector."""
    u = torch.randn(p, generator=torch.Generator().manual_seed(seed),
                    dtype=None if like is None else like.dtype)
    if like is not None:
        u = u.to(like.device)
    u = u / u.norm()
    lam = 0.0
    for _ in range(iters):
        Hu = hvp(u)
        lam = float(u @ Hu)
        u = Hu / (Hu.norm() + 1e-300)
    return lam


def scipy_operator(hvp: HVP, p: int, dtype=torch.float64, device=None):
    from scipy.sparse.linalg import LinearOperator
    np_dtype = np.float64 if dtype == torch.float64 else np.float32

    def mv(x):
        u = torch.as_tensor(np.ascontiguousarray(x, dtype=np_dtype), device=device).reshape(-1)
        return hvp(u).detach().cpu().numpy()
    return LinearOperator((p, p), matvec=mv, dtype=np_dtype)


def lanczos_extremes(hvp: HVP, p: int, k_small: int = 6, k_large: int = 3, ncv: int = 64,
                     restarts: int = 180, tol: float = 1e-4, dtype=torch.float64, device=None) -> dict:
    """Smallest-algebraic and largest eigenvalues by implicitly restarted Lanczos, for p too
    large for `dense_hessian`. `restarts` bounds eigsh's maxiter, which counts restarts of
    ~ncv-k matvecs each; a dense cluster at the bottom of the spectrum converges slowly, so
    partial Ritz values are kept (flagged `*_converged=False`) rather than running for hours."""
    from scipy.sparse.linalg import eigsh
    A = scipy_operator(hvp, p, dtype, device)
    out = {}
    for name, k, which in (('smallest', k_small, 'SA'), ('largest', k_large, 'LA')):
        try:
            vals = eigsh(A, k=k, which=which, return_eigenvectors=False,
                         tol=tol, maxiter=restarts, ncv=min(p - 1, ncv))
            out[name] = sorted(float(v) for v in vals)
        except Exception as e:
            vals = getattr(e, 'eigenvalues', None)
            out[name] = (sorted(float(v) for v in vals) if vals is not None and len(vals)
                         else f'failed: {type(e).__name__}')
            out[name + '_converged'] = False
    return out


def cosine(a: Tensor, b: Tensor) -> float:
    return float(a @ b / (a.norm() * b.norm() + 1e-300))

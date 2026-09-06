import torch

from bilevel_lt import estimators as est

torch.set_default_dtype(torch.float64)


def spd(p=30, seed=0, lo=0.5, hi=1.0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(p, p, generator=g))
    evals = torch.linspace(lo, hi, p)
    H = Q @ torch.diag(evals) @ Q.T
    b = torch.randn(p, generator=g)
    return H, b, evals


def test_cg_is_exact_after_p_iterations():
    H, b, _ = spd()
    x, info = est.cg(lambda v: H @ v, b, K=H.shape[0])
    assert torch.allclose(x, torch.linalg.solve(H, b), atol=1e-8)
    assert not info['breakdown']


def test_cg_reports_breakdown_on_indefinite_direction():
    H, b, _ = spd()
    H = H - 2.0 * torch.eye(H.shape[0])        # every direction is negative
    _, info = est.cg(lambda v: H @ v, b, K=10)
    assert info['breakdown'] and info['iters'] == 0


def test_damped_solves_shifted_system():
    H, b, _ = spd()
    mu = 0.3
    x, _ = est.damped(lambda v: H @ v, b, mu)
    assert torch.allclose(x, torch.linalg.solve(H + mu * torch.eye(H.shape[0]), b), atol=1e-6)


def test_neumann_converges_with_alpha_below_two_over_lmax():
    H, b, evals = spd(lo=0.5, hi=1.0)
    x, _ = est.neumann(lambda v: H @ v, b, K=80, alpha=1.0 / float(evals.max()))
    assert torch.allclose(x, torch.linalg.solve(H, b), atol=1e-8)


def test_nystrom_full_rank_equals_damped_inverse():
    H, b, _ = spd()
    p = H.shape[0]
    rho = 0.1
    ref = torch.linalg.solve(H + rho * torch.eye(p), b)
    x, _ = est.nystrom_sketch(lambda v: H @ v, b, p, k=p, rho=rho)
    assert torch.allclose(x, ref, atol=1e-6)
    x, _ = est.nystrom_columns(lambda v: H @ v, b, p, k=p, rho=rho, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(x, ref, atol=1e-6)


def test_dense_and_minres_match_solve():
    H, b, evals = spd()
    ref = torch.linalg.solve(H, b)
    x, info = est.dense_inverse(lambda v: H @ v, b, H.shape[0])
    assert torch.allclose(x, ref, atol=1e-8) and info['n_neg'] == 0
    x, info = est.minres(lambda v: H @ v, b, H.shape[0])
    assert torch.allclose(x, ref, atol=1e-6) and info['info'] == 0


def test_spectral_helpers():
    H, _, evals = spd(p=120, lo=0.1, hi=2.0)
    assert abs(est.lam_max(lambda v: H @ v, H.shape[0], iters=200) - 2.0) < 1e-3
    ext = est.lanczos_extremes(lambda v: H @ v, H.shape[0], k_small=3, k_large=2, ncv=40)
    assert abs(ext['smallest'][0] - 0.1) < 1e-3 and abs(ext['largest'][-1] - 2.0) < 1e-3

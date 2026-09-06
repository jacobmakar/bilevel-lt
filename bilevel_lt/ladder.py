"""The fixed-feature pipeline: a small head on frozen DINOv2 features.

Three modes, one JSON per cell under --out_root/headladder/<tag>.json:

    cert   hold the leader fixed at --point and score every estimator by the cosine of its
           hypergradient to the unrolled gradient (central finite differences through the
           k-step inner map). Also record the inner Hessian spectrum.
    loop   run the outer loop with one estimator.
    ref    train the closed-form baseline l = tau * log(pi) for each tau in --tau_grid.

Every cell prints PROGRESS lines (scripts/sweep_eta.py turns them into a runtime estimate)
and one JSON summary line at the end. A cell whose JSON already exists is skipped unless
--force, so a resubmitted job list resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch

from . import estimators as est
from .problem import FixedFeatureProblem
from .tags import cell_tag

torch.set_default_dtype(torch.float64)


def prog(tag, t0, **kw):
    def f(v):
        if isinstance(v, float):
            return f"{v:.1f}" if abs(v) >= 1 else f"{v:.3g}"
        return str(v)
    print(f"PROGRESS {tag} elapsed={time.time() - t0:.1f} "
          + ' '.join(f"{k}={f(v)}" for k, v in kw.items()), flush=True)


def leader_point(prob, name):
    if name == 'zero':
        return torch.zeros(prob.C)
    m = re.fullmatch(r'la([0-9.]+)', name)
    return float(m.group(1)) * torch.log(prob.pi)


def lbfgs_polish(prob, theta0, l, iters):
    """Drive the inner problem to a critical point with L-BFGS (strong Wolfe), to evaluate
    estimators where the negative curvature of a non-converged point has vanished."""
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=iters, tolerance_grad=1e-12,
                            tolerance_change=1e-16, history_size=50, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = prob.L_in(theta, l)
        loss.backward()
        return loss
    g0 = float(prob.g_in(theta0, l).norm())
    opt.step(closure)
    theta = theta.detach()
    return theta, g0, float(prob.g_in(theta, l).norm())


def solve(name, prob, theta, l, g_out, args, seed=0):
    """Dispatch one estimator by name; returns (v ~ H^{-1} g_out, info)."""
    hvp = prob.hvp(theta, l)
    if name == 'identity':
        return est.identity(hvp, g_out)
    if name.startswith('cg'):
        return est.cg(hvp, g_out, int(name[2:]))
    if name.startswith('neumann'):
        return est.neumann(hvp, g_out, int(name[7:]), 1.0 / est.lam_max(hvp, prob.p, iters=15))
    if name == 'minres':
        return est.minres(hvp, g_out, prob.p, tol=1e-8, maxiter=1000)
    if name == 'damped':
        mu = args.damp_rel * est.lam_max(hvp, prob.p, iters=15)
        return est.damped(hvp, g_out, mu)
    if name == 'damped_track':
        # damping that follows the inner residual: |lambda_min| of the inner Hessian scales
        # with ||g_in|| at a non-converged point, so mu = damp_track * ||g_in|| keeps
        # H + mu I positive definite along the outer trajectory at no extra cost
        mu = args.damp_track * float(prob.g_in(theta, l).norm())
        v, info = est.damped(hvp, g_out, mu)
        return v, dict(info, mu=mu)
    if name.startswith('nystrom'):
        mu = args.damp_rel * est.lam_max(hvp, prob.p, iters=15)
        return est.nystrom_sketch(hvp, g_out, prob.p, int(name[7:]), mu, seed=seed)
    if name == 'dense':
        if prob.p > args.dense_max:
            raise ValueError(f"dense needs p <= dense_max ({prob.p} > {args.dense_max})")
        return est.dense_inverse(hvp, g_out, prob.p)
    raise ValueError(name)


# ---------------------------------------------------------------------------
def mode_cert(prob, args, out):
    l = leader_point(prob, args.point)
    t0 = time.time()
    tag = out['tag']
    theta = prob.init(args.seed)
    L0 = est.lam_max(prob.hvp(theta, l), prob.p)
    lr = args.lr_scale / L0
    out.update(p=prob.p, dims=prob.dims, lam_max_init=L0, lr=lr)
    prog(tag, t0, phase='start', p=prob.p, warm_steps=args.warm_steps)
    theta0 = prob.run_map(theta, l, args.warm_steps, lr)
    if not torch.isfinite(theta0).all():
        out['status'] = 'warm_diverged'
        return
    out['warm'] = dict(grad_norm=float(prob.g_in(theta0, l).norm()), resid_norm=prob.resid_norm(theta0, l),
                       L_in=float(prob.L_in(theta0, l)), bacc=prob.bacc(theta0), seconds=time.time() - t0)
    if args.polish_iters > 0:
        tp = time.time()
        theta0, g_before, g_after = lbfgs_polish(prob, theta0, l, args.polish_iters)
        out['polish'] = dict(iters=args.polish_iters, grad_norm_before=g_before, grad_norm_after=g_after,
                             resid_norm_after=prob.resid_norm(theta0, l), seconds=time.time() - tp)
        prog(tag, t0, phase='polish', grad_before=g_before, grad_after=g_after)
    prog(tag, t0, phase='warm_done', grad_norm=out['warm']['grad_norm'])

    # finite-difference derivative of the k-step map, per horizon k
    ks = [int(k) for k in args.k_list.split(',')]
    fd = {}
    steps_total = 2 * prob.C * sum(ks)
    steps_done = 0
    t_fd0 = time.time()
    for k in ks:
        tk = time.time()
        theta_k = prob.run_map(theta0, l, k, lr)
        g = torch.zeros(prob.C)
        for j in range(prob.C):
            e = torch.zeros(prob.C)
            e[j] = args.fd_eps
            Fp = float(prob.L_out(prob.run_map(theta0, l + e, k, lr)))
            Fm = float(prob.L_out(prob.run_map(theta0, l - e, k, lr)))
            g[j] = (Fp - Fm) / (2 * args.fd_eps)
            steps_done += 2 * k
            rate = (time.time() - t_fd0) / steps_done
            prog(tag, t0, phase='fd', k=k, coord=f"{j + 1}/{prob.C}", eta_fd=rate * (steps_total - steps_done))
        fd[k] = dict(grad=g.tolist(), norm=float(g.norm()),
                     inner_grad_norm=float(prob.g_in(theta_k, l).norm()), resid_norm=prob.resid_norm(theta_k, l),
                     L_out=float(prob.L_out(theta_k)), bacc=prob.bacc(theta_k), seconds=time.time() - tk)
        out.setdefault('fd', {})[str(k)] = fd[k]

    # estimators at the most converged point
    kmax = max(ks)
    theta_k = prob.run_map(theta0, l, kmax, lr)
    g_fd = torch.tensor(fd[kmax]['grad'])
    g_out = prob.g_out(theta_k)
    hvp = prob.hvp(theta_k, l)
    Lk = est.lam_max(hvp, prob.p)
    out['lam_max_at_point'] = Lk
    mu = args.damp_rel * Lk
    out['mu'] = mu
    results = {}

    def record(name, v, info):
        hg = prob.hg_from_v(theta_k, l, v)
        results[name] = dict(cos_fd=est.cosine(hg, g_fd), norm_ratio=float(hg.norm() / g_fd.norm()),
                             grad=hg.tolist(), **info)

    record('identity', *est.identity(hvp, g_out))
    for K in (10, 100):
        record(f'cg{K}', *est.cg(hvp, g_out, K))
    for K in (10, 100):
        record(f'neumann{K}', *est.neumann(hvp, g_out, K, 1.0 / Lk))
    record('minres', *est.minres(hvp, g_out, prob.p))
    record('damped', *est.damped(hvp, g_out, mu))
    for k in (10, 50):
        record(f'nystrom{k}', *est.nystrom_sketch(hvp, g_out, prob.p, k, mu))
    prog(tag, t0, phase='estimators_done')

    if prob.p > args.dense_max:
        tl = time.time()
        out['lanczos'] = est.lanczos_extremes(hvp, prob.p)
        out['lanczos']['seconds'] = time.time() - tl
        prog(tag, t0, phase='lanczos_done', seconds=out['lanczos']['seconds'])
        sm = out['lanczos'].get('smallest')
        if isinstance(sm, list) and sm:
            out['spectrum'] = dict(lam_min=sm[0], lam_max=Lk, n_neg_of_k=sum(1 for v in sm if v < -1e-8 * Lk),
                                   k_small=len(sm), source='lanczos',
                                   converged=out['lanczos'].get('smallest_converged', True))
    else:
        td = time.time()
        H = est.dense_hessian(hvp, prob.p)
        evals, evecs = torch.linalg.eigh(H)
        lmax = float(evals.max())
        lmin = float(evals.min())
        out['spectrum'] = dict(
            lam_min=lmin, lam_max=lmax, lam_min_loss=lmin - prob.lam,
            n_neg=int((evals < -1e-8 * lmax).sum()),
            n_soft=int((evals < 2 * prob.lam).sum()),
            n_below_1e3lmax=int((evals < 1e-3 * lmax).sum()),
            kappa=lmax / lmin if lmin > 0 else None,
            quantiles=[float(q) for q in torch.quantile(evals, torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]))],
            seconds=time.time() - td)
        c = evecs.T @ g_out
        record('dense', evecs @ (c / evals), dict(neg_mass=float((c[evals < 0] ** 2).sum() / (c ** 2).sum())))
        mu_d = 1e-3 * lmax
        record('dense_damped', evecs @ (c / (evals + mu_d)), dict(mu=mu_d))
        record('dense_abs', evecs @ (c / evals.abs().clamp_min(1e-12 * lmax)), {})
        mass = c ** 2 / (c ** 2).sum()
        out['g_out_mass_below_1e-3lmax'] = float(mass[evals < 1e-3 * lmax].sum())
        out['ihvp_mass_below_1e-3lmax'] = float(((c / evals) ** 2)[evals < 1e-3 * lmax].sum() / ((c / evals) ** 2).sum())
    out['estimators'] = results
    out['status'] = 'ok'
    out['seconds'] = time.time() - t0


def mode_ref(prob, args, out):
    t0 = time.time()
    res = {}
    steps = args.ref_steps if args.ref_steps is not None else args.warm_steps
    for tau in (float(t) for t in args.tau_grid.split(',')):
        l = tau * torch.log(prob.pi)
        theta = prob.init(args.seed)
        lr = args.lr_scale / est.lam_max(prob.hvp(theta, l), prob.p)
        theta = prob.run_map(theta, l, steps, lr)
        ok = bool(torch.isfinite(theta).all())
        res[str(tau)] = dict(bacc=prob.bacc(theta) if ok else None, bacc_val=prob.bacc(theta, 'val') if ok else None,
                             lr=lr, grad_norm=float(prob.g_in(theta, l).norm()) if ok else None)
    out.update(p=prob.p, ref=res, ref_steps=steps, status='ok', seconds=time.time() - t0)


def mode_loop(prob, args, out):
    t0 = time.time()
    l = leader_point(prob, args.point).clone()
    theta = prob.init(args.seed)
    lr = args.lr_scale / est.lam_max(prob.hvp(theta, l), prob.p)
    theta = prob.run_map(theta, l, args.warm_steps, lr)
    l_param = torch.nn.Parameter(l)
    opt = (torch.optim.Adam if args.outer_opt == 'adam' else torch.optim.SGD)([l_param], lr=args.outer_lr)
    traj, solver_log = [], []
    for t in range(args.outer_steps):
        ld = l_param.detach()
        theta = prob.run_map(theta, ld, args.k_loop, lr)
        if not torch.isfinite(theta).all():
            out['status'] = 'inner_diverged'
            break
        v, info = solve(args.estimator, prob, theta, ld, prob.g_out(theta), args, seed=t)
        hg = prob.hg_from_v(theta, ld, v)
        if not torch.isfinite(hg).all():
            out['status'] = 'hypergrad_nonfinite'
            break
        opt.zero_grad()
        l_param.grad = hg
        opt.step()
        if info:
            solver_log.append({k: (float(x) if not isinstance(x, bool) else x) for k, x in info.items()})
        if t % args.eval_every == 0 or t == args.outer_steps - 1:
            traj.append(dict(t=t, bacc=prob.bacc(theta), bacc_val=prob.bacc(theta, 'val'),
                             L_out=float(prob.L_out(theta)), l=ld.tolist(), hg_norm=float(hg.norm())))
            el = time.time() - t0
            prog(out['tag'], t0, step=f"{t + 1}/{args.outer_steps}", bacc=traj[-1]['bacc'],
                 eta=el / (t + 1) * (args.outer_steps - t - 1))
    out.update(p=prob.p, lr=lr, traj=traj, final_bacc=traj[-1]['bacc'] if traj else None,
               final_l=l_param.detach().tolist(),
               cg_breakdowns=sum(1 for s in solver_log if s.get('breakdown')),
               mean_resid=(sum(s['resid'] for s in solver_log) / len(solver_log)
                           if solver_log and 'resid' in solver_log[0] else None),
               seconds=time.time() - t0)
    out.setdefault('status', 'ok')


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', default='data/features/dinov2_s_c10.npz')
    ap.add_argument('--head', default='linear', help="linear | dlin<w>[d<L>] | relu<w>[d<L>]")
    ap.add_argument('--mode', choices=('cert', 'loop', 'ref'), default='cert')
    ap.add_argument('--point', default='zero', help="leader point: zero | la<tau>")
    ap.add_argument('--estimator', default='identity',
                    help="loop: identity | cg<K> | neumann<K> | minres | damped | damped_track | nystrom<k> | dense")
    ap.add_argument('--imbalance', type=int, default=100)
    ap.add_argument('--val_per_class', type=int, default=100)
    ap.add_argument('--ridge', type=float, default=1e-4)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--lr_scale', type=float, default=0.5, help="inner lr = lr_scale / lam_max(init)")
    ap.add_argument('--warm_steps', type=int, default=2000, help="inner steps before cert / loop start")
    ap.add_argument('--ref_steps', type=int, default=None,
                    help="ref: inner steps per tau (default warm_steps; the loop trains warm_steps + outer_steps * k_loop)")
    ap.add_argument('--tau_grid', default='0,0.5,1,1.5,2', help="ref: logit-adjustment temperatures")
    ap.add_argument('--k_list', default='200,600', help="cert: horizons of the k-step maps to differentiate")
    ap.add_argument('--fd_eps', type=float, default=1e-3)
    ap.add_argument('--dense_max', type=int, default=8000, help="dense spectrum for p up to this; Lanczos extremes above")
    ap.add_argument('--damp_rel', type=float, default=1e-3, help="damping mu = damp_rel * lam_max (damped, nystrom)")
    ap.add_argument('--damp_track', type=float, default=0.5, help="damping mu = damp_track * ||g_in|| (damped_track)")
    ap.add_argument('--polish_iters', type=int, default=0, help="cert: L-BFGS iterations after the warm start")
    ap.add_argument('--outer_steps', type=int, default=200)
    ap.add_argument('--outer_lr', type=float, default=0.05)
    ap.add_argument('--outer_opt', choices=('adam', 'sgd'), default='adam')
    ap.add_argument('--k_loop', type=int, default=20, help="loop: warm inner steps per outer step")
    ap.add_argument('--eval_every', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out_root', default='runs')
    ap.add_argument('--force', action='store_true', help="rerun even if the cell's JSON exists")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    tag = cell_tag(vars(args))
    out_dir = os.path.join(args.out_root, 'headladder')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, tag + '.json')
    if os.path.exists(path) and not args.force:
        try:
            prev = json.load(open(path)).get('status')
        except Exception:
            prev = None
        if prev == 'ok':
            print(f"SKIP {tag} (exists, status ok)", flush=True)
            return
    prob = FixedFeatureProblem(args.features, args.head, args.imbalance, args.val_per_class,
                               args.seed, args.ridge, args.momentum)
    out = dict(args=vars(args), tag=tag, n_train=int(len(prob.y)), n_val=int(len(prob.yv)), pi=prob.pi.tolist())
    {'cert': mode_cert, 'ref': mode_ref, 'loop': mode_loop}[args.mode](prob, args, out)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    summary = {k: out[k] for k in ('status', 'p', 'seconds') if k in out}
    if 'estimators' in out:
        summary['cos_fd'] = {k: round(v['cos_fd'], 4) for k, v in out['estimators'].items()}
    if 'spectrum' in out:
        summary['spectrum'] = {k: out['spectrum'].get(k) for k in ('lam_min', 'lam_max', 'n_neg', 'n_soft', 'source')
                               if k in out['spectrum']}
    if 'final_bacc' in out:
        summary['final_bacc'] = out['final_bacc']
    if 'ref' in out:
        summary['ref'] = {k: v['bacc'] for k, v in out['ref'].items()}
    print(tag, json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

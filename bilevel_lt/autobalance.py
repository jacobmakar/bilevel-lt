"""A replication of AutoBalance (Li et al. 2021) on ResNet-32.

    inner   SGD with momentum and coupled weight decay, minimizing
            CE(sigma(delta) * f_theta(x) + l, y) on the long-tailed train set
    outer   every `unroll_steps` inner steps, one optimizer step on the leader (l, delta)
            along the hypergradient of the balanced-validation CE of the raw logits, with
            H and B from one train minibatch and g_out from one validation minibatch. The
            weight decay is applied in the SGD step and is not part of H, as in AutoBalance.

The scale sigma(delta) starts at 0.5 (delta = 0), as published, so the bilevel arm's inner
objective at step 0 is not the LA arm's; --no_delta --la_init gives the exactly matched
comparison. The closed-form baselines (--method la, --method ce) run through the same inner
loop with the leader frozen. Minibatches are drawn by epochs without replacement. BatchNorm
uses batch statistics during training (no running-stat updates inside torch.func) and is
calibrated on training batches before every eval-mode measurement. Writes <out>/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, jvp
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import estimators as est
from .data import cifar10_lt
from .resnet import calibrate_bn, evaluate, resnet32, set_bn_track


def infinite(loader, device):
    while True:
        for x, y in loader:
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def sgd_step(params: dict, grads: dict, momentum_buf: dict | None, lr: float, wd: float, momentum: float = 0.9):
    """One step of SGD with momentum and coupled weight decay, functional in the parameters."""
    new_p, new_m = {}, {}
    for k, p in params.items():
        g = grads[k] + wd * p if wd else grads[k]
        m = g if momentum_buf is None else momentum * momentum_buf[k] + g
        new_m[k] = m
        new_p[k] = p - lr * m
    return new_p, new_m


def step_lr(base_lr: float, step: int, num_iters: int) -> float:
    """Divide by 10 at 80% and 90% of training."""
    if step >= 0.9 * num_iters:
        return base_lr * 0.01
    if step >= 0.8 * num_iters:
        return base_lr * 0.1
    return base_lr


class Flattener:
    """Flat vector <-> parameter dict, for estimators that work on vectors."""
    def __init__(self, params: dict):
        self.names = list(params)
        self.shapes = [params[k].shape for k in self.names]
        self.sizes = [params[k].numel() for k in self.names]
        self.p = sum(self.sizes)

    def flat(self, params: dict) -> Tensor:
        return torch.cat([params[k].reshape(-1) for k in self.names])

    def unflat(self, v: Tensor) -> dict:
        return {k: c.view(s) for k, c, s in zip(self.names, torch.split(v, self.sizes), self.shapes)}


def make_estimator(name: str, cost: int, alpha: float, p: int, device, seed: int):
    """Estimator of H^{-1} g on the flat HVP oracle. `cost` = iterations or rank; `alpha` =
    Neumann step size, Nystrom rho, or damping mu."""
    if name == 'neumann':
        return lambda hvp, b: est.neumann(hvp, b, cost, alpha)
    if name == 'cg':
        return lambda hvp, b: est.cg(hvp, b, max(cost, 1))
    if name == 'damped':
        return lambda hvp, b: est.damped(hvp, b, alpha, K=max(cost, 1))
    if name == 'nystrom':
        gen = torch.Generator(device=device).manual_seed(seed)
        return lambda hvp, b: est.nystrom_columns(hvp, b, p, max(cost, 1), alpha, gen)
    if name == 'identity':
        return est.identity
    raise ValueError(name)


class AutoBalance:
    def __init__(self, args, device):
        self.args, self.device = args, device
        torch.manual_seed(args.seed)
        self.train_set, self.val_set, self.test_set, self.pi = cifar10_lt(
            args.data_root, args.imbalance, args.val_per_class, args.seed)
        pin = device.type == 'cuda'
        kw = dict(num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)
        self.train_loader = DataLoader(self.train_set, args.batch_size, shuffle=True, drop_last=True, **kw)
        self.val_loader = DataLoader(self.val_set, args.outer_batch_size, shuffle=True, drop_last=True, **kw)
        self.test_loader = DataLoader(self.test_set, 256, shuffle=False, **kw)
        self.val_eval_loader = DataLoader(self.val_set, 256, shuffle=False, **kw)
        self.cal_loader = DataLoader(self.train_set, 256, shuffle=True, drop_last=True, **kw)

        self.model = resnet32().to(device)
        set_bn_track(self.model, False)          # batch statistics only, no buffer mutation
        self.model.train()
        self.params = {k: v.detach() for k, v in self.model.named_parameters()}
        self.buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        self.flat = Flattener(self.params)

        # leader: per-class offset l, optionally a per-class scale sigma(delta)
        log_pi = torch.tensor(np.log(self.pi), dtype=torch.float32, device=device)
        if args.method == 'ab':
            l0 = log_pi if args.la_init else torch.zeros_like(log_pi)
        elif args.method == 'la':
            l0 = args.tau * log_pi
        else:                                   # ce: no adjustment
            l0 = torch.zeros_like(log_pi)
        self.use_delta = args.method == 'ab' and not args.no_delta
        self.hyper = [nn.Parameter(l0.clone())]
        if self.use_delta:
            self.hyper.append(nn.Parameter(torch.zeros_like(log_pi)))
        self.outer_opt = None
        if args.method == 'ab':
            self.outer_opt = (torch.optim.SGD(self.hyper, lr=args.outer_lr, momentum=0.9, weight_decay=args.outer_wd)
                              if args.outer_opt == 'sgd' else torch.optim.Adam(self.hyper, lr=args.outer_lr, weight_decay=args.outer_wd))
            self.estimator = make_estimator(args.estimator, args.cost, args.alpha, self.flat.p, device, args.seed)

    # objectives -------------------------------------------------------------
    def logits(self, params: dict, x: Tensor) -> Tensor:
        return functional_call(self.model, (params, self.buffers), (x,))

    def adjust(self, z: Tensor, hyper) -> Tensor:
        if self.use_delta:
            return torch.sigmoid(hyper[1]) * z + hyper[0]
        return z + hyper[0]

    def inner_loss(self, params: dict, hyper, x: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(self.adjust(self.logits(params, x), hyper), y)

    def outer_loss(self, params: dict, x: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(self.logits(params, x), y)

    def hypergradient(self, xi, yi, xo, yo):
        """-B^T Hhat^{-1} g_out for the leader, on one train batch (H, B) and one val batch (g_out)."""
        hyper = tuple(h.detach() for h in self.hyper)
        theta = self.flat.flat(self.params)
        g_in = grad(lambda th, hy: self.inner_loss(self.flat.unflat(th), hy, xi, yi))
        hvp = lambda v: jvp(lambda th: g_in(th, hyper), (theta,), (v,))[1]
        g_out = self.flat.flat(grad(self.outer_loss)(self.params, xo, yo))
        v, info = self.estimator(hvp, g_out)
        mixed = grad(lambda hy: torch.dot(g_in(theta, hy), v))(hyper)
        return [-m for m in mixed], info

    # training ----------------------------------------------------------------
    def run(self):
        a = self.args
        train_iter, val_iter = infinite(self.train_loader, self.device), infinite(self.val_loader, self.device)
        momentum_buf = None
        patience = int(a.warmup_frac * a.num_iters)
        history, solver_log = [], []
        skipped = 0
        loss_ema = None
        t0 = time.time()
        inner_grad = grad(self.inner_loss)
        for step in range(a.num_iters):
            lr = step_lr(a.inner_lr, step, a.num_iters)
            x, y = next(train_iter)
            hyper = tuple(h.detach() for h in self.hyper)
            g = inner_grad(self.params, hyper, x, y)
            self.params, momentum_buf = sgd_step(self.params, g, momentum_buf, lr, a.inner_wd)
            with torch.no_grad():
                loss = float(self.inner_loss(self.params, hyper, x, y))
            loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss

            if self.outer_opt is not None and step > 0 and step >= patience and step % a.unroll_steps == 0:
                xi, yi = next(train_iter)
                xo, yo = next(val_iter)
                try:
                    hg, info = self.hypergradient(xi, yi, xo, yo)
                except torch.linalg.LinAlgError as e:      # e.g. a singular Nystrom sketch
                    print(f"step {step}: estimator failed ({e}); outer update skipped", flush=True)
                    skipped += 1
                    continue
                if all(torch.isfinite(h).all() for h in hg):
                    for h, g_ in zip(self.hyper, hg):
                        h.grad = g_
                    self.outer_opt.step()
                    self.outer_opt.zero_grad(set_to_none=True)
                    if info:
                        solver_log.append({k: float(v) for k, v in info.items() if not isinstance(v, bool)}
                                          | {k: v for k, v in info.items() if isinstance(v, bool)})

            if step % a.log_every == 0 or step == a.num_iters - 1:
                el = time.time() - t0
                print(f"step {step}/{a.num_iters} loss {loss_ema:.4f} lr {lr:g} "
                      f"l {[round(v, 3) for v in self.hyper[0].tolist()]} "
                      f"elapsed {el:.0f}s eta {el / (step + 1) * (a.num_iters - step - 1):.0f}s", flush=True)
            if (step > 0 and step % a.eval_every == 0) or step == a.num_iters - 1:
                m = self.measure(max_cal_batches=20)
                history.append(dict(step=step, inner_loss=loss_ema, test_acc=m['test']['acc'],
                                    test_bacc=m['test']['bacc'], val_bacc=m['val']['bacc'],
                                    l=self.hyper[0].tolist(), seconds=time.time() - t0))
                print(f"  eval step {step}: test acc {m['test']['acc']:.4f} bacc {m['test']['bacc']:.4f} "
                      f"val bacc {m['val']['bacc']:.4f}", flush=True)

        final = self.measure(max_cal_batches=50)
        return dict(history=history, final=final, seconds=time.time() - t0,
                    solver=dict(steps=len(solver_log), skipped=skipped,
                                breakdowns=sum(1 for s in solver_log if s.get('breakdown')),
                                mean_resid=(float(np.mean([s['resid'] for s in solver_log]))
                                            if solver_log and 'resid' in solver_log[0] else None)))

    @torch.no_grad()
    def measure(self, max_cal_batches: int) -> dict:
        """Copy the functional parameters into the module, calibrate BN, evaluate on test and
        val, then return the module to the bilevel training state."""
        for k, p in self.model.named_parameters():
            p.copy_(self.params[k])
        calibrate_bn(self.model, self.cal_loader, self.device, max_batches=max_cal_batches)
        out = dict(test=evaluate(self.model, self.test_loader, self.device),
                   val=evaluate(self.model, self.val_eval_loader, self.device),
                   l=self.hyper[0].tolist(),
                   delta=torch.sigmoid(self.hyper[1]).tolist() if self.use_delta else None)
        set_bn_track(self.model, False)
        self.model.train()
        return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', choices=('ab', 'la', 'ce'), default='ab')
    ap.add_argument('--estimator', choices=('neumann', 'cg', 'nystrom', 'damped', 'identity'), default='neumann')
    ap.add_argument('--cost', type=int, default=10, help="estimator iterations (neumann, cg, damped) or rank (nystrom)")
    ap.add_argument('--alpha', type=float, default=0.01, help="neumann step size | nystrom rho | damped mu")
    ap.add_argument('--la_init', action='store_true', help="ab: start the leader at l = log(pi)")
    ap.add_argument('--no_delta', action='store_true',
                    help="ab: offsets only, no per-class scale sigma(delta) (which starts at 0.5)")
    ap.add_argument('--tau', type=float, default=2.0, help="la: l = tau * log(pi)")
    ap.add_argument('--imbalance', type=int, default=100)
    ap.add_argument('--val_per_class', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--num_iters', type=int, default=15000)
    ap.add_argument('--unroll_steps', type=int, default=10, help="inner steps per outer step")
    ap.add_argument('--warmup_frac', type=float, default=0.0, help="fraction of training before outer steps start")
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--outer_batch_size', type=int, default=128)
    ap.add_argument('--inner_lr', type=float, default=0.1)
    ap.add_argument('--inner_wd', type=float, default=2e-4)
    ap.add_argument('--outer_lr', type=float, default=5e-2)
    ap.add_argument('--outer_opt', choices=('sgd', 'adam'), default='sgd')
    ap.add_argument('--outer_wd', type=float, default=1e-4, help="weight decay on the leader (pulls l toward 0)")
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--data_root', default='data/cifar')
    ap.add_argument('--out', default=None, help="run directory (default runs/autobalance/<auto name>)")
    ap.add_argument('--device', default=None)
    ap.add_argument('--save_checkpoint', action='store_true')
    return ap.parse_args(argv)


def run_name(a) -> str:
    if a.method == 'ab':
        name = f"ab-{a.estimator}{a.cost}" + ('-lainit' if a.la_init else '') + ('-nodelta' if a.no_delta else '')
    elif a.method == 'la':
        name = f"la-tau{a.tau:g}"
    else:
        name = 'ce'
    return f"{name}-imb{a.imbalance}-val{a.val_per_class}-s{a.seed}"


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out = args.out or os.path.join('runs', 'autobalance', run_name(args))
    os.makedirs(out, exist_ok=True)
    print(f"device {device}  out {out}", flush=True)
    ab = AutoBalance(args, device)
    print(f"train {len(ab.train_set)}  val {len(ab.val_set)}  test {len(ab.test_set)}  "
          f"priors {[round(float(v), 4) for v in ab.pi]}  params {ab.flat.p}", flush=True)
    result = ab.run()
    result.update(args=vars(args), name=run_name(args), pi=ab.pi.tolist(), p=ab.flat.p)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=1)
    if args.save_checkpoint:
        torch.save(dict(model=ab.model.state_dict(), hyper=[h.detach().cpu() for h in ab.hyper], args=vars(args)),
                   os.path.join(out, 'checkpoint.pt'))
    fin = result['final']
    print(f"FINAL {run_name(args)} test acc {fin['test']['acc']:.4f} bacc {fin['test']['bacc']:.4f} "
          f"val bacc {fin['val']['bacc']:.4f} ({result['seconds'] / 60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()

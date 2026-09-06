"""Heads on top of frozen features, as functions of one flat parameter vector.

    linear          k -> C
    dlin{w}[d{L}]   k -> w -> ... -> w -> C with L hidden layers and no nonlinearity
                    (a product of matrices; the loss is non-convex in the parameters)
    relu{w}[d{L}]   the same shape with ReLU between layers

One flat parameter vector keeps Hessian-vector products and dense Hessians simple.
"""
from __future__ import annotations

import math
import re

import torch


def build_head(kind: str, k: int, C: int):
    """Returns (forward, init, p, dims): forward(theta, X) -> logits, init(seed) -> theta,
    p = number of parameters."""
    if kind == 'linear':
        dims, act = [k, C], None
    else:
        m = re.fullmatch(r'(dlin|relu)(\d+)(?:d(\d+))?', kind)
        if m is None:
            raise ValueError(f"unknown head {kind!r}")
        w, L = int(m.group(2)), int(m.group(3) or 1)
        dims, act = [k] + [w] * L + [C], ('relu' if m.group(1) == 'relu' else None)
    n_layers = len(dims) - 1
    p = sum(dims[i + 1] * dims[i] + dims[i + 1] for i in range(n_layers))

    def forward(theta, X):
        off, h = 0, X
        for i in range(n_layers):
            nout, nin = dims[i + 1], dims[i]
            W = theta[off:off + nout * nin].view(nout, nin)
            off += nout * nin
            b = theta[off:off + nout]
            off += nout
            h = h @ W.T + b
            if act == 'relu' and i < n_layers - 1:
                h = torch.relu(h)
        return h

    def init(seed):
        g = torch.Generator().manual_seed(seed)
        parts = []
        for i in range(n_layers):
            nout, nin = dims[i + 1], dims[i]
            bound = 1.0 / math.sqrt(nin)
            parts.append((torch.rand(nout * nin, generator=g, dtype=torch.float64) * 2 - 1) * bound)
            parts.append((torch.rand(nout, generator=g, dtype=torch.float64) * 2 - 1) * bound)
        return torch.cat(parts)

    return forward, init, p, dims

"""Long-tailed CIFAR-10 splits and the frozen-feature cache.

Split protocol, shared by both pipelines: a balanced validation set of `val_per_class`
images per class is carved from the full training set first (rng seed `seed`), then an
exponential long-tail profile with ratio `imbalance` between the largest and the smallest
class is sampled from what remains (rng seed `seed + 1`). Nobody trains on the validation
images: the bilevel leader fits its class offsets on them, and the closed-form baseline
may select its temperature on them. `val_per_class` is an explicit argument everywhere
because the bilevel-vs-closed-form comparison depends on it.
"""
from __future__ import annotations

import os

import numpy as np
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def lt_sample_indices(labels: np.ndarray, imbalance: int, num_classes: int, seed: int) -> np.ndarray:
    """Exponential long-tail profile: class c keeps n_max * mu**c images, where n_max is
    the size of class 0 and mu = imbalance**(-1/(C-1)), so class C-1 keeps n_max/imbalance.
    Labels equal to -1 are excluded from the pools."""
    rng = np.random.default_rng(seed)
    n_max = int((labels == 0).sum())
    mu = (1.0 / imbalance) ** (1.0 / (num_classes - 1))
    chosen = []
    for c in range(num_classes):
        n = int(n_max * mu ** c)
        pool = np.flatnonzero(labels == c)
        chosen.append(rng.choice(pool, size=n, replace=False))
    return np.concatenate(chosen)


def lt_split(labels: np.ndarray, imbalance: int, val_per_class: int, seed: int,
             num_classes: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, val_idx) for a balanced dataset with `labels`."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    val_idx = np.concatenate([
        rng.choice(np.flatnonzero(labels == c), size=val_per_class, replace=False)
        for c in range(num_classes)])
    remaining = labels.copy()
    remaining[val_idx] = -1
    train_idx = lt_sample_indices(remaining, imbalance, num_classes, seed + 1)
    counts = np.bincount(labels[train_idx], minlength=num_classes)
    if counts.min() < 1:
        raise ValueError(f"imbalance={imbalance} leaves an empty class (min count {counts.min()})")
    return train_idx, val_idx


def class_priors(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# images (end-to-end pipeline)
# ---------------------------------------------------------------------------
def cifar10_lt(root: str, imbalance: int, val_per_class: int, seed: int):
    """(train, val, test, priors): train is long-tailed with augmentation, val is balanced
    without augmentation, test is the standard balanced test set."""
    from torch.utils.data import Subset
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.ToTensor(), norm])
    root = os.path.expanduser(root)
    aug = CIFAR10(root, train=True, download=True, transform=train_tf)
    plain = CIFAR10(root, train=True, download=True, transform=eval_tf)
    test = CIFAR10(root, train=False, download=True, transform=eval_tf)
    labels = np.array(aug.targets)
    train_idx, val_idx = lt_split(labels, imbalance, val_per_class, seed)
    return Subset(aug, train_idx), Subset(plain, val_idx), test, class_priors(labels[train_idx])


# ---------------------------------------------------------------------------
# frozen features (head-ladder pipeline)
# ---------------------------------------------------------------------------
@torch.no_grad()
def cache_dinov2_features(root: str, out: str, device: str | None = None, batch_size: int = 256):
    """DINOv2 ViT-S/14 CLS embeddings (384-d) of the full CIFAR-10 train and test sets,
    saved as one npz. The long-tail split is applied at load time, so one cache serves
    every seed, imbalance ratio and validation size."""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    root = os.path.expanduser(root)
    arrays = {}
    for split, train in (('train', True), ('test', False)):
        ds = CIFAR10(root, train=train, download=True, transform=tf)
        feats, labels = [], []
        for i, (x, y) in enumerate(DataLoader(ds, batch_size=batch_size, num_workers=0)):
            feats.append(model(x.to(device)).float().cpu().numpy())
            labels.append(y.numpy())
            if i % 20 == 0:
                print(f"  {split} batch {i}", flush=True)
        arrays[f'x_{split}'] = np.concatenate(feats).astype(np.float32)
        arrays[f'y_{split}'] = np.concatenate(labels)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, backbone='dinov2_vits14', **arrays)
    print(f"saved {out}: train {arrays['x_train'].shape}, test {arrays['x_test'].shape}")


def load_features(path: str) -> dict[str, np.ndarray]:
    z = np.load(os.path.expanduser(path), allow_pickle=True)
    return {k: z[k] for k in ('x_train', 'y_train', 'x_test', 'y_test')}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="build the DINOv2 feature cache")
    ap.add_argument('--data_root', default='data/cifar')
    ap.add_argument('--out', default='data/features/dinov2_s_c10.npz')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    cache_dinov2_features(a.data_root, a.out, a.device)

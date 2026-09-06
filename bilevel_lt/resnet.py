"""CIFAR ResNet family (He et al. 2016, 6n+2 layers; n=5 is ResNet-32) and the BatchNorm
handling the bilevel pipeline needs.

During bilevel training the network is evaluated through torch.func with
track_running_stats=False on every BN layer: normalization uses batch statistics and no
buffer is mutated inside a transformed function. Running statistics are (re)computed by
`calibrate_bn` before any eval-mode use.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(c_out))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    def __init__(self, n: int = 5, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._stage(16, 16, n, stride=1)
        self.layer2 = self._stage(16, 32, n, stride=2)
        self.layer3 = self._stage(32, 64, n, stride=2)
        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def _stage(c_in, c_out, n, stride):
        return nn.Sequential(BasicBlock(c_in, c_out, stride), *[BasicBlock(c_out, c_out, 1) for _ in range(n - 1)])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def resnet32(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(n=5, num_classes=num_classes)


def set_bn_track(model: nn.Module, track: bool):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = track


@torch.no_grad()
def calibrate_bn(model: nn.Module, loader, device, max_batches: int = 50):
    """Recompute BN running statistics with a train-mode pass over training batches."""
    set_bn_track(model, True)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(x.to(device))


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, num_classes: int = 10) -> dict:
    """Overall accuracy, per-class accuracy and their mean (balanced accuracy) in eval mode."""
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long, device=device)
    total = torch.zeros(num_classes, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(-1)
        total.scatter_add_(0, y, torch.ones_like(y))
        correct.scatter_add_(0, y, (pred == y).long())
    per_class = (correct.double() / total.clamp(min=1).double()).cpu().tolist()
    return dict(acc=float(correct.sum().item() / total.sum().item()),
                bacc=float(sum(per_class) / num_classes), per_class=per_class)

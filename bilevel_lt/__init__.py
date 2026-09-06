"""Bilevel logit adjustment on long-tailed CIFAR-10.

    ladder.py       the fixed-feature pipeline: a small head on frozen DINOv2 features
    autobalance.py  the end-to-end pipeline: a replication of AutoBalance on ResNet-32

Both use the split in data.py and the estimators in estimators.py.
"""

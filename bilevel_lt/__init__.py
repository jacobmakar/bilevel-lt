"""Bilevel logit adjustment on long-tailed CIFAR-10.

Two pipelines share one data protocol and one set of inverse-Hessian estimators:

    ladder.py       fixed DINOv2 features, a small head on top, everything in float64:
                    certify every estimator against finite differences of the inner
                    training map, look at the inner Hessian spectrum, run the outer loop.
    autobalance.py  end-to-end ResNet-32 on images: the bilevel method as it is used in
                    practice, next to the closed-form logit-adjustment baseline.
"""

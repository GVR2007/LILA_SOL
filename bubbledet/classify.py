"""Bubble vs. particle classification from boundary-gradient intensity.

Physics: refraction at a bubble's liquid-air interface produces a bright rim
(high edge gradient). Opaque particles have no such glow (low edge gradient).
A single threshold on the mean boundary gradient separates the two classes.
"""
from __future__ import annotations

from .models import BUBBLE, PARTICLE, DetectedObject


def classify(
    objects: list[DetectedObject], threshold: float
) -> list[DetectedObject]:
    """Label each object: gradient >= threshold -> bubble, else particle."""
    return [
        o.with_label(BUBBLE if o.boundary_gradient >= threshold else PARTICLE)
        for o in objects
    ]


def tune_threshold(features: list[float], labels: list[str]) -> float:
    """Pick the gradient threshold that best separates the labelled classes.

    Sweeps midpoints between sorted feature values and returns the one with the
    highest classification accuracy (bubble == gradient >= threshold).
    """
    if not features:
        raise ValueError("Cannot tune a threshold on empty data.")
    vals = sorted(set(features))
    if len(vals) == 1:
        return vals[0]
    candidates = [(a + b) / 2.0 for a, b in zip(vals, vals[1:])]

    best_t = candidates[0]
    best_acc = -1.0
    for t in candidates:
        correct = sum(
            (f >= t) == (lab == BUBBLE) for f, lab in zip(features, labels)
        )
        acc = correct / len(features)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t

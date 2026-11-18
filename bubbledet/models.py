"""Immutable data structures shared across the pipeline.

Dataclasses are frozen; "updates" return new copies via the with_* helpers,
following the project's immutability rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

PARTICLE = "particle"
BUBBLE = "bubble"


@dataclass(frozen=True)
class GroundTruthObject:
    """A synthetic object whose true position, size, and class are known."""

    cx: float
    cy: float
    diameter_px: float
    kind: str  # PARTICLE or BUBBLE


@dataclass(frozen=True)
class Calibration:
    """Result of detecting the reference square."""

    um_per_pixel: float
    square_side_px: float
    square_bbox: tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True, eq=False)
class DetectedObject:
    """One detected blob and everything measured about it.

    eq=False because the contour is a numpy array (unhashable, ambiguous to
    compare); identity equality is all we need.
    """

    contour: np.ndarray = field(repr=False)
    centroid: tuple[float, float]
    area_px: float
    equiv_diameter_px: float
    boundary_gradient: float
    size_um: float | None = None
    label: str | None = None

    def with_size(self, size_um: float) -> "DetectedObject":
        return replace(self, size_um=size_um)

    def with_label(self, label: str) -> "DetectedObject":
        return replace(self, label=label)

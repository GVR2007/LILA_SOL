"""End-to-end pipeline wiring: image -> calibrated, classified, counted results."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import calibrate, classify, config, detect, measure, report
from .models import Calibration, DetectedObject


@dataclass(frozen=True)
class PipelineResult:
    calibration: Calibration
    objects: list[DetectedObject]        # sized and labelled
    bin_counts: dict[float, int]


def run(
    gray: np.ndarray,
    square_size_um: float = config.SQUARE_SIZE_UM,
    gradient_threshold: float = config.DEFAULT_GRADIENT_THRESHOLD,
    bins: tuple[float, ...] = config.SIZE_BINS_UM,
) -> PipelineResult:
    """Run calibrate -> detect -> measure -> classify -> count."""
    calibration = calibrate.find_reference_square(gray, square_size_um)
    objects = detect.detect_objects(gray, exclude_bbox=calibration.square_bbox)
    objects = measure.assign_sizes(objects, calibration)
    objects = classify.classify(objects, gradient_threshold)
    bin_counts = report.count_bins(objects, bins)
    return PipelineResult(calibration=calibration, objects=objects,
                          bin_counts=bin_counts)

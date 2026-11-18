"""Convert pixel sizes to micrometres using the calibration."""
from __future__ import annotations

from .models import Calibration, DetectedObject


def assign_sizes(
    objects: list[DetectedObject], calibration: Calibration
) -> list[DetectedObject]:
    """Return new objects with size_um set from the equivalent diameter."""
    return [
        o.with_size(o.equiv_diameter_px * calibration.um_per_pixel) for o in objects
    ]

"""Reporting: ISO-style size-bin counts, a text report, and an annotated image."""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .models import BUBBLE, PARTICLE, Calibration, DetectedObject

GREEN = (0, 200, 0)    # counted particle
RED = (0, 0, 255)      # discarded bubble


def count_bins(
    objects: list[DetectedObject], bins: tuple[float, ...] = config.SIZE_BINS_UM
) -> dict[float, int]:
    """Cumulative particle counts at each size threshold (bubbles excluded)."""
    particles = [
        o for o in objects if o.label == PARTICLE and o.size_um is not None
    ]
    return {b: sum(1 for o in particles if o.size_um >= b) for b in bins}


def annotate(gray: np.ndarray, objects: list[DetectedObject]) -> np.ndarray:
    """Draw contours (green=particle, red=bubble) with size labels."""
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for o in objects:
        color = RED if o.label == BUBBLE else GREEN
        cv2.drawContours(canvas, [o.contour], -1, color, 2)
        if o.size_um is not None:
            cx, cy = int(o.centroid[0]), int(o.centroid[1])
            cv2.putText(
                canvas, f"{o.size_um:.0f}um", (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )
    return canvas


def build_report(
    calibration: Calibration,
    objects: list[DetectedObject],
    bin_counts: dict[float, int],
    evaluation_text: str | None = None,
) -> str:
    n_bubbles = sum(1 for o in objects if o.label == BUBBLE)
    n_particles = sum(1 for o in objects if o.label == PARTICLE)
    lines = [
        "ISO 4406 Prototype -- Particle Count Report",
        "===========================================",
        "",
        "Calibration",
        "-----------",
        f"Reference square side : {calibration.square_side_px:.1f} px",
        f"Scale                 : {calibration.um_per_pixel:.4f} um/pixel",
        "",
        "Objects",
        "-------",
        f"Detected total        : {len(objects)}",
        f"Classified particles  : {n_particles}",
        f"Classified bubbles    : {n_bubbles} (discarded)",
        "",
        "Particle counts by size (cumulative, bubbles excluded)",
        "------------------------------------------------------",
    ]
    for b in sorted(bin_counts):
        lines.append(f">= {b:>5.1f} um : {bin_counts[b]}")
    lines += [
        "",
        "NOTE: counts are per image field of view. A true ISO 4406 code "
        "requires counts per mL, which needs the imaged fluid volume "
        "(film thickness x field area).",
    ]
    if evaluation_text:
        lines += ["", evaluation_text]
    return "\n".join(lines)

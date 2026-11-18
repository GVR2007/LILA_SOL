"""Pixel-to-micrometre calibration from a known reference square."""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .models import Calibration


def find_reference_square(
    gray: np.ndarray, square_size_um: float = config.SQUARE_SIZE_UM
) -> Calibration:
    """Detect the largest square-like contour and derive um_per_pixel.

    Raises ValueError if no square is found.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(gray.shape[0] * gray.shape[1])
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.01 * img_area or area > 0.95 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h)
        if 0.8 <= aspect <= 1.2 and area > best_area:
            best = (x, y, w, h)
            best_area = area

    if best is None:
        raise ValueError("No reference square found for calibration.")

    x, y, w, h = best
    side_px = (w + h) / 2.0
    um_per_pixel = square_size_um / side_px
    return Calibration(um_per_pixel=um_per_pixel, square_side_px=side_px,
                       square_bbox=(x, y, w, h))

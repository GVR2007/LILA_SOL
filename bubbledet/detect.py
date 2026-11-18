"""Object detection via gradient-magnitude thresholding.

Gradient-based (not plain intensity) detection catches BOTH dark opaque
particles and bright-rimmed bubbles, since both have strong edges while a plain
intensity threshold would miss a transparent-centred bubble.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .models import DetectedObject


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of the blurred image (float64)."""
    k = config.GAUSSIAN_BLUR_KSIZE
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _boundary_gradient(contour, grad: np.ndarray, ring_px: int) -> float:
    """Mean gradient magnitude in a band straddling the contour outline."""
    mask = np.zeros(grad.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=2 * ring_px + 1)
    vals = grad[mask > 0]
    return float(vals.mean()) if vals.size else 0.0


def _overlaps(cx: float, cy: float, bbox) -> bool:
    x, y, w, h = bbox
    margin = 6
    return (x - margin) <= cx <= (x + w + margin) and (y - margin) <= cy <= (y + h + margin)


def detect_objects(
    gray: np.ndarray, exclude_bbox=None
) -> list[DetectedObject]:
    """Find candidate objects; measure area, centroid, and boundary gradient."""
    grad = gradient_magnitude(gray)
    gnorm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(gnorm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = config.MORPH_CLOSE_KSIZE
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(gray.shape[0] * gray.shape[1])

    objects: list[DetectedObject] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.MIN_OBJECT_AREA_PX:
            continue
        if area > config.MAX_OBJECT_AREA_FRAC * img_area:
            continue  # too big to be a particle (e.g. the reference square)
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        if exclude_bbox is not None and _overlaps(cx, cy, exclude_bbox):
            continue
        equiv = float(np.sqrt(4.0 * area / np.pi))
        grad_mean = _boundary_gradient(c, grad, config.BOUNDARY_RING_PX)
        objects.append(
            DetectedObject(
                contour=c,
                centroid=(cx, cy),
                area_px=float(area),
                equiv_diameter_px=equiv,
                boundary_gradient=grad_mean,
            )
        )
    return objects

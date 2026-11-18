"""Synthetic fluid-film image generator with known ground truth.

Produces a grayscale image containing:
  * a reference square (calibration target),
  * opaque "particles" (dark filled disks, no glow),
  * "bubbles" (bright refraction rim, transparent centre),
plus the exact list of what was drawn, so the classifier can be tuned and
scored against the truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import BUBBLE, PARTICLE, GroundTruthObject


@dataclass(frozen=True)
class SyntheticConfig:
    width: int = 900
    height: int = 700
    background: int = 150
    noise_sigma: float = 3.0
    blur_sigma: float = 1.0

    # Ground-truth scale: micrometres per pixel. 0.5 um/px makes 4/6/14 um
    # objects 8/12/28 px -- comfortably resolvable.
    um_per_pixel: float = 0.5

    # Demo reference square = 200 um reticle (NOT 1 cm; see config.SQUARE_SIZE_UM
    # for why a literal 1 cm square is incompatible with micron particles).
    square_size_um: float = 200.0
    square_origin: tuple[int, int] = (30, 30)
    square_intensity: int = 30
    square_thickness: int = 4

    # Object appearance.
    particle_intensity: int = 85   # dark on 150 bg -> moderate edge gradient
    bubble_rim_intensity: int = 255  # bright glow -> strong edge gradient
    bubble_rim_thickness: int = 3

    # Object sizes to draw (micrometres). Chosen to straddle the ISO bins:
    # 3 (<4), 5 (4-6), 8 & 12 (6-14), 20 & 32 (>=14).
    sizes_um: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0, 20.0, 32.0)
    n_particles: int = 14
    n_bubbles: int = 10
    seed: int = 7


def _square_side_px(cfg: SyntheticConfig) -> int:
    return round(cfg.square_size_um / cfg.um_per_pixel)


def _random_objects(cfg: SyntheticConfig, rng: np.random.Generator):
    """Rejection-sample non-overlapping object placements in the clear region."""
    side = _square_side_px(cfg)
    sx, sy = cfg.square_origin
    # Objects live to the right of the square, with a margin.
    x_lo = sx + side + 40
    x_hi = cfg.width - 50
    y_lo = 40
    y_hi = cfg.height - 50

    specs: list[tuple[str, float]] = (
        [(PARTICLE, 0.0)] * cfg.n_particles + [(BUBBLE, 0.0)] * cfg.n_bubbles
    )
    placed: list[tuple[float, float, float, str]] = []
    for kind, _ in specs:
        diameter_um = float(rng.choice(cfg.sizes_um))
        radius_px = (diameter_um / cfg.um_per_pixel) / 2.0
        for _attempt in range(200):
            cx = rng.uniform(x_lo + radius_px, x_hi - radius_px)
            cy = rng.uniform(y_lo + radius_px, y_hi - radius_px)
            ok = all(
                (cx - px) ** 2 + (cy - py) ** 2 > (radius_px + pr + 10) ** 2
                for px, py, pr, _ in placed
            )
            if ok:
                placed.append((cx, cy, radius_px, kind))
                break
    return placed


def generate(cfg: SyntheticConfig = SyntheticConfig()):
    """Return (image_uint8, ground_truth_list, true_um_per_pixel)."""
    rng = np.random.default_rng(cfg.seed)
    img = np.full((cfg.height, cfg.width), float(cfg.background), dtype=np.float64)

    # Reference square (hollow frame so it is never confused with a filled blob).
    side = _square_side_px(cfg)
    sx, sy = cfg.square_origin
    cv2.rectangle(
        img, (sx, sy), (sx + side, sy + side),
        float(cfg.square_intensity), cfg.square_thickness,
    )

    truth: list[GroundTruthObject] = []
    for cx, cy, radius_px, kind in _random_objects(cfg, rng):
        center = (int(round(cx)), int(round(cy)))
        radius = int(round(radius_px))
        if kind == PARTICLE:
            # Opaque dark disk, no rim.
            cv2.circle(img, center, radius, float(cfg.particle_intensity), -1)
        else:
            # Bright refraction rim, transparent (background) centre.
            cv2.circle(
                img, center, radius, float(cfg.bubble_rim_intensity),
                cfg.bubble_rim_thickness,
            )
        truth.append(
            GroundTruthObject(cx=cx, cy=cy, diameter_px=2.0 * radius_px, kind=kind)
        )

    # Sensor noise, then a mild blur (softens particle edges, keeps rims steep).
    img += rng.normal(0.0, cfg.noise_sigma, img.shape)
    img = cv2.GaussianBlur(img, (3, 3), cfg.blur_sigma)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, truth, cfg.um_per_pixel

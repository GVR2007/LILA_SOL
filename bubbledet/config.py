"""Central configuration: physical constants, ISO bins, and default thresholds.

Every tunable number in the pipeline lives here so it can be audited in one
place (no magic numbers scattered through the code).
"""
from __future__ import annotations

# --- Calibration -----------------------------------------------------------
# Physical length of the reference square's side, in micrometres.
# The ISO 4406 task specifies a 1 cm square == 10_000 um.
#
# IMPORTANT physical note: a literal 1 cm square cannot coexist with 4 um
# particles in a single raster image. 1 cm / 4 um = 2500:1, so if the square is
# ~400 px wide, one pixel is ~25 um and a 4 um particle is 0.16 px (invisible).
# Real particle counters calibrate at the microscope's optical scale, not with
# an in-frame 1 cm square. The synthetic demo therefore overrides this value
# with a small reticle (see synthetic.SyntheticConfig.square_size_um) so that
# 4/6/14 um objects are resolvable. The calibration MATH is identical for any
# square size -- only the number differs.
SQUARE_SIZE_UM: float = 10_000.0

# --- ISO 4406 size bins (cumulative, ">=") ---------------------------------
SIZE_BINS_UM: tuple[float, ...] = (4.0, 6.0, 14.0)

# --- Detection -------------------------------------------------------------
GAUSSIAN_BLUR_KSIZE: int = 3        # odd kernel; suppresses noise before gradients
MIN_OBJECT_AREA_PX: float = 12.0    # drop specks below this (noise floor)
MAX_OBJECT_AREA_FRAC: float = 0.2   # drop blobs larger than this fraction of image
MORPH_CLOSE_KSIZE: int = 5          # connect edge fragments into closed loops

# --- Classification --------------------------------------------------------
BOUNDARY_RING_PX: int = 2           # half-width (px) of the boundary band sampled
# Default gradient threshold separating bubbles (high) from particles (low).
# This is a fallback; prefer a value tuned per-dataset via classify.tune_threshold.
DEFAULT_GRADIENT_THRESHOLD: float = 40.0

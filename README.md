# Vision-Based Bubble vs. Particle Detector (ISO 4406 Prototype)

Detects solid particles vs. air bubbles in images of a thin fluid film, sizes
the particles in micrometres, and reports ISO 4406-style cumulative counts.
Bubbles are told apart by the **bright refraction glow** (high edge gradient)
around their rim; opaque particles have no such glow.

## Pipeline

1. **Calibrate** (`calibrate.py`) — detect the reference square, divide its
   pixel side by its known physical size to get `um_per_pixel`.
2. **Detect** (`detect.py`) — Sobel gradient magnitude -> Otsu threshold ->
   morphological close -> contours. Gradient-based so it catches both dark
   particles and bright-rimmed bubbles. The reference square is excluded.
3. **Measure** (`measure.py`) — equivalent diameter `sqrt(4*area/pi)` x scale.
4. **Classify** (`classify.py`) — mean Sobel gradient along each contour's
   boundary band; `>= threshold` -> bubble, else particle.
5. **Report** (`report.py`) — discard bubbles, count particles at `>=4/6/14 um`,
   annotate the image, write the report.
6. **Evaluate** (`evaluate.py`) — match detections to ground truth; precision
   and recall per class.

`synthetic.py` generates labelled test images (the ground-truth factory).
`pipeline.py` wires stages 1-5; `scripts/run_demo.py` runs the whole thing.

## Setup & run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py     # writes outputs/
pytest -q                      # 11 tests
```

## Two important caveats (physical realism)

* **Scale.** A literal 1 cm square cannot coexist with 4 um particles in one
  raster (2500:1 ratio -> a 4 um particle is sub-pixel). Real counters
  calibrate at the microscope's optical scale. The demo uses a 200 um reticle
  so 4/6/14 um objects are resolvable; the calibration math is identical for
  any square size (`config.SQUARE_SIZE_UM`).
* **Per mL vs. per image.** ISO 4406 counts are per mL; this prototype counts
  per field of view. Converting needs the imaged fluid volume
  (film thickness x field area).

## Current demo result (held-out image)

Calibration error ~1.7%, detection recall 100%, classification accuracy 100%,
bubbles correctly discarded, cumulative counts 14 / 10 / 5 at 4 / 6 / 14 um.

## Known limitations / next steps

* Touching or overlapping objects merge into one contour (add watershed split).
* The gradient threshold is tuned on synthetic glow; real optics need
  re-tuning, possibly a lighting-invariant feature (rim/interior gradient ratio).
* No per-mL conversion yet; no real-image dataset yet.

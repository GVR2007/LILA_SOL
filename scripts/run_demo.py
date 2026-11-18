"""End-to-end demo: generate data, tune, run, evaluate, and save deliverables.

Tunes the gradient threshold on a TRAINING image (seed 1) and evaluates on a
separate HELD-OUT image (seed 2) so the accuracy figures are honest.
"""
from __future__ import annotations

import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running as `python scripts/run_demo.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bubbledet import report
from bubbledet.calibrate import find_reference_square
from bubbledet.classify import classify, tune_threshold
from bubbledet.detect import detect_objects
from bubbledet.evaluate import _match, evaluate
from bubbledet.measure import assign_sizes
from bubbledet.models import BUBBLE, PARTICLE
from bubbledet.pipeline import run
from bubbledet.synthetic import SyntheticConfig, generate

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _features_and_labels(cfg):
    img, truth, _ = generate(cfg)
    cal = find_reference_square(img, square_size_um=cfg.square_size_um)
    objects = detect_objects(img, cal.square_bbox)
    pairs = _match(objects, truth)
    feats = [o.boundary_gradient for o, _ in pairs]
    labs = [gt.kind for _, gt in pairs]
    return feats, labs


def main():
    os.makedirs(OUT, exist_ok=True)
    train_cfg = SyntheticConfig(seed=1)
    test_cfg = SyntheticConfig(seed=2)

    # 1) Tune the classifier threshold on the training image.
    feats, labs = _features_and_labels(train_cfg)
    threshold = tune_threshold(feats, labs)
    print(f"Tuned gradient threshold: {threshold:.2f}")

    # 2) Run the full pipeline on the held-out test image.
    img, truth, _ = generate(test_cfg)
    result = run(
        img,
        square_size_um=test_cfg.square_size_um,
        gradient_threshold=threshold,
    )
    objects = result.objects

    # 3) Evaluate against ground truth.
    ev = evaluate(objects, truth)

    # 4) Save deliverables.
    cv2.imwrite(os.path.join(OUT, "synthetic_test.png"), img)
    annotated = report.annotate(img, objects)
    cv2.imwrite(os.path.join(OUT, "annotated.png"), annotated)

    report_text = report.build_report(
        result.calibration, objects, result.bin_counts, ev.to_text()
    )
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report_text + "\n")

    # Gradient-feature histogram (shows class separation + the threshold).
    train_feats, train_labs = feats, labs
    p = [f for f, l in zip(train_feats, train_labs) if l == PARTICLE]
    b = [f for f, l in zip(train_feats, train_labs) if l == BUBBLE]
    plt.figure(figsize=(7, 4))
    plt.hist(p, bins=12, alpha=0.7, label="particles", color="tab:green")
    plt.hist(b, bins=12, alpha=0.7, label="bubbles", color="tab:red")
    plt.axvline(threshold, color="black", linestyle="--",
                label=f"threshold = {threshold:.1f}")
    plt.xlabel("mean boundary gradient")
    plt.ylabel("count")
    plt.title("Classifier feature separation (training image)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "gradient_histogram.png"), dpi=120)
    plt.close()

    print("\n" + report_text)
    print(f"\nOutputs written to: {OUT}")


if __name__ == "__main__":
    main()

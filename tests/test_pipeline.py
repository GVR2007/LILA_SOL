from bubbledet.calibrate import find_reference_square
from bubbledet.classify import classify, tune_threshold
from bubbledet.detect import detect_objects
from bubbledet.evaluate import evaluate
from bubbledet.measure import assign_sizes
from bubbledet.pipeline import run
from bubbledet.synthetic import SyntheticConfig, generate


def _tuned_threshold(cfg):
    """Tune on a training image using ground-truth labels."""
    img, truth, _ = generate(cfg)
    cal = find_reference_square(img, square_size_um=cfg.square_size_um)
    objects = detect_objects(img, cal.square_bbox)
    from bubbledet.evaluate import _match
    pairs = _match(objects, truth)
    feats = [o.boundary_gradient for o, _ in pairs]
    labs = [gt.kind for _, gt in pairs]
    return tune_threshold(feats, labs)


def test_bin_counts_are_monotonic():
    cfg = SyntheticConfig()
    img, _, _ = generate(cfg)
    t = _tuned_threshold(cfg)
    result = run(img, square_size_um=cfg.square_size_um, gradient_threshold=t)
    counts = [result.bin_counts[b] for b in sorted(result.bin_counts)]
    # Cumulative ">=" bins must be non-increasing as the threshold grows.
    assert counts == sorted(counts, reverse=True)


def test_classification_accuracy_on_held_out_image():
    train_cfg = SyntheticConfig(seed=1)
    test_cfg = SyntheticConfig(seed=2)
    t = _tuned_threshold(train_cfg)

    img, truth, _ = generate(test_cfg)
    cal = find_reference_square(img, square_size_um=test_cfg.square_size_um)
    objects = classify(
        assign_sizes(detect_objects(img, cal.square_bbox), cal), threshold=t
    )
    result = evaluate(objects, truth)
    assert result.overall_accuracy >= 0.9
    assert result.detection_recall >= 0.8

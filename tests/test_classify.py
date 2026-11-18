from bubbledet.classify import classify, tune_threshold
from bubbledet.models import BUBBLE, PARTICLE


def test_tune_threshold_separates_clean_data():
    features = [10, 12, 11, 50, 55, 52]
    labels = [PARTICLE, PARTICLE, PARTICLE, BUBBLE, BUBBLE, BUBBLE]
    t = tune_threshold(features, labels)
    assert 12 < t < 50


def test_classify_applies_threshold():
    from bubbledet.detect import detect_objects
    from bubbledet.synthetic import SyntheticConfig, generate
    from bubbledet.calibrate import find_reference_square

    cfg = SyntheticConfig()
    img, _, _ = generate(cfg)
    cal = find_reference_square(img, square_size_um=cfg.square_size_um)
    objects = detect_objects(img, cal.square_bbox)
    labelled = classify(objects, threshold=10_000)  # absurdly high -> all particles
    assert all(o.label == PARTICLE for o in labelled)

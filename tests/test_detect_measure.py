from bubbledet.calibrate import find_reference_square
from bubbledet.detect import detect_objects
from bubbledet.measure import assign_sizes
from bubbledet.synthetic import SyntheticConfig, generate


def test_detects_most_objects_and_excludes_square():
    cfg = SyntheticConfig()
    img, truth, _ = generate(cfg)
    cal = find_reference_square(img, square_size_um=cfg.square_size_um)
    objects = detect_objects(img, exclude_bbox=cal.square_bbox)
    # Should find most ground-truth objects, and not the big square.
    assert len(objects) >= 0.8 * len(truth)
    assert len(objects) <= len(truth) + 3


def test_sizes_are_positive_micrometres():
    cfg = SyntheticConfig()
    img, _, _ = generate(cfg)
    cal = find_reference_square(img, square_size_um=cfg.square_size_um)
    objects = assign_sizes(detect_objects(img, cal.square_bbox), cal)
    assert objects
    assert all(o.size_um is not None and o.size_um > 0 for o in objects)

import pytest

from bubbledet.calibrate import find_reference_square
from bubbledet.synthetic import SyntheticConfig, generate


def test_calibration_recovers_true_scale():
    cfg = SyntheticConfig()
    img, _, true_upp = generate(cfg)
    cal = find_reference_square(img, square_size_um=cfg.square_size_um)
    # Within 5% of the true micrometres-per-pixel.
    assert cal.um_per_pixel == pytest.approx(true_upp, rel=0.05)


def test_raises_when_no_square():
    import numpy as np
    blank = np.full((200, 200), 150, np.uint8)
    with pytest.raises(ValueError):
        find_reference_square(blank, square_size_um=200.0)

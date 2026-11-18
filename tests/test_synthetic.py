import numpy as np

from bubbledet.models import BUBBLE, PARTICLE
from bubbledet.synthetic import SyntheticConfig, generate


def test_image_shape_and_dtype():
    cfg = SyntheticConfig()
    img, truth, upp = generate(cfg)
    assert img.shape == (cfg.height, cfg.width)
    assert img.dtype == np.uint8
    assert upp == cfg.um_per_pixel


def test_ground_truth_counts_match_config():
    cfg = SyntheticConfig()
    _, truth, _ = generate(cfg)
    n_particles = sum(1 for g in truth if g.kind == PARTICLE)
    n_bubbles = sum(1 for g in truth if g.kind == BUBBLE)
    # Rejection sampling may drop a few if it cannot place them.
    assert n_particles <= cfg.n_particles
    assert n_bubbles <= cfg.n_bubbles
    assert len(truth) >= 0.8 * (cfg.n_particles + cfg.n_bubbles)


def test_generation_is_deterministic():
    img_a, _, _ = generate(SyntheticConfig(seed=42))
    img_b, _, _ = generate(SyntheticConfig(seed=42))
    assert np.array_equal(img_a, img_b)

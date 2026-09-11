"""
Determinism tests for the synthetic data generation.

These pin the Phase 2 property that the whole generation path is reproducible
from a single recorded seed and touches no global numpy state. Before Stage 1
none of this held: `random_irregular_dataset` was called with seed=None,
`generate_random_numbers` defaulted to seed=0, the per-point weights came from
`np.random.dirichlet`, and `random_logcount` from `np.random.uniform`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datageneration import (  # noqa: E402
    generate_random_numbers,
    random_irregular_dataset,
    random_logcount,
)

N_DATASETS = 200
SEED = 20260911


def build(seed, n_datasets=N_DATASETS):
    """Generate a small corpus the way notebook 1 does."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_datasets):
        n = int(random_logcount(rng, lo=4, hi=1000, n=1)[0])
        out.append(random_irregular_dataset(n=n, rng=rng))
    return out


def test_same_seed_reproduces_exactly():
    a = build(SEED)
    b = build(SEED)
    assert len(a) == len(b) == N_DATASETS
    for (da, wa), (db, wb) in zip(a, b):
        assert np.array_equal(da, db)
        assert np.array_equal(wa, wb)


def test_different_seed_differs():
    a = build(SEED)
    b = build(SEED + 1)
    identical = sum(
        1 for (da, _), (db, _) in zip(a, b)
        if da.shape == db.shape and np.array_equal(da, db)
    )
    assert identical == 0, f"{identical} datasets identical across different seeds"


def test_global_numpy_state_does_not_affect_output():
    """The legacy global RNG must have no influence on any generated value."""
    np.random.seed(1)
    a = build(SEED)
    np.random.seed(999)
    _ = np.random.random(1000)
    b = build(SEED)
    for (da, wa), (db, wb) in zip(a, b):
        assert np.array_equal(da, db)
        assert np.array_equal(wa, wb)


def test_generation_does_not_consume_global_state():
    """Generating data must not advance the legacy global stream either."""
    np.random.seed(4)
    before = np.random.random(5)
    np.random.seed(4)
    build(SEED, n_datasets=10)
    after = np.random.random(5)
    assert np.array_equal(before, after)


def test_rng_is_required():
    """No function may silently fall back to a default seed or global state."""
    with pytest.raises(TypeError):
        generate_random_numbers("gauss", 10.0, 1.0, 5)
    with pytest.raises(TypeError):
        random_irregular_dataset(50)
    with pytest.raises(TypeError):
        random_logcount()


def test_component_shapes_are_no_longer_constant():
    """Regression test for the seed=0 collapse.

    With the old default seed, every skew-normal component had a = 3.458732,
    every Student-t df = 3.366328 and every lognormal s = 1.136962, and two
    Gaussian components of equal length were exact affine images of one
    another. Standardizing removed location and scale, leaving identical
    vectors.
    """
    rng = np.random.default_rng(3)
    a = generate_random_numbers("gauss", 5.0, 0.2, 40, rng)
    b = generate_random_numbers("gauss", 19.0, 1.4, 40, rng)
    za = (a - a.mean()) / a.std()
    zb = (b - b.mean()) / b.std()
    assert np.max(np.abs(za - zb)) > 1e-3
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.99


def test_output_contract():
    """Shape, positivity and the normalization the generator promises."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        n = int(random_logcount(rng, lo=4, hi=1000, n=1)[0])
        data, weights = random_irregular_dataset(n=n, rng=rng)
        assert data.shape == (n,)
        assert weights.shape == (n,)
        assert np.all(data > 0)
        assert np.all(weights >= 0)
        assert np.isclose(weights.sum(), 1.0)
        # Normalized by the UNWEIGHTED mean, per Stage 1 amendment A3.
        assert np.isclose(data.mean(), 1.0)

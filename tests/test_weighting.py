"""Stage 2d: the weighting measures.

What these pin is the PROPERTY each measure is claimed to have, because those
properties are what the paper argues from.

The decomposition has to satisfy the inequality it is built on. The relative
measure has to be invariant to rescaling the data, since the whole point of
naming it is that the study's normalization is not doing secret work. And
A_IQR has to be exactly scale invariant and nearly blind to dispersion, which
is the surprising result of this stage and the reason the practitioner
statement is NOT built on it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import weighting as WG          # noqa: E402


def a_dataset(seed=3, n=80, sigma=0.6):
    rng = np.random.default_rng(seed)
    x = np.exp(rng.normal(0.0, sigma, size=n))
    return x / x.mean()


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------
def test_location_is_a_lower_bound_on_w1():
    """W1 >= |difference in means|, which is what makes the split a split."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for seed in range(40):
        x = a_dataset(seed=seed, n=int(rng.integers(3, 400)))
        w = rng.dirichlet(np.ones(len(x)))
        d = WG.location_shape_split(x, w)
        worst = min(worst, d['w1'] - d['location'])
        assert d['shape'] >= 0.0
        assert d['location'] <= d['w1'] + 1e-9
    # The inequality is exact in real arithmetic, so any shortfall is quadrature
    # noise and has to be far below anything the study reports.
    assert worst > -1e-9


def test_uniform_weights_give_a_zero_split():
    x = a_dataset()
    d = WG.location_shape_split(x, WG.uniform_weights(x))
    assert d['w1'] == pytest.approx(0.0, abs=1e-12)
    assert d['location'] == pytest.approx(0.0, abs=1e-12)


def test_a_pure_location_shift_is_all_location():
    """Weights that move mass without changing shape put everything in location.

    Two identical copies of one shape, separated. Reweighting between the two
    copies translates the distribution, so the residual after the mean shift is
    the part W1 charges for the transport being spread over the two blocks.
    A pure two-point dataset is the clean case: with n = 2 there is no shape to
    change, so the location share must be exactly 1.
    """
    x = np.array([1.0, 3.0])
    # p = 0.5 IS the uniform weighting, so the distance is zero and the share is
    # undefined rather than one. Excluded deliberately, not overlooked.
    for p in (0.1, 0.25, 0.75, 0.9):
        d = WG.location_shape_split(x, np.array([p, 1 - p]))
        assert d['location_share'] == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# the relative measure
# ---------------------------------------------------------------------------
def test_relative_measures_are_scale_invariant():
    """Rescaling every value leaves each relative measure unchanged.

    This is the claim that lets the study report a W1 without saying "of the
    mean" every time: the normalization to an unweighted mean of 1.0 is a
    convenience, not a load-bearing step.
    """
    x = a_dataset()
    w = np.random.default_rng(1).dirichlet(np.ones(len(x)))
    base = WG.relativize(WG.location_shape_split(x, w)['w1'],
                         WG.relative_scales(x))
    for c in (1e-6, 0.5, 4.0, 1e7):
        got = WG.relativize(WG.location_shape_split(x * c, w)['w1'],
                            WG.relative_scales(x * c))
        for k in base:
            assert got[k] == pytest.approx(base[k], rel=1e-10)


def test_absolute_w1_is_not_scale_invariant():
    """The control for the test above: the raw distance does move, as it must."""
    x = a_dataset()
    w = np.random.default_rng(1).dirichlet(np.ones(len(x)))
    one = WG.location_shape_split(x, w)['w1']
    ten = WG.location_shape_split(x * 10.0, w)['w1']
    assert ten == pytest.approx(10.0 * one, rel=1e-10)


def test_relative_mean_equals_w1_on_a_normalized_dataset():
    """On the study's normalized datasets the named measure IS the reported W1.

    So naming it changes no published number, which is the point of saying the
    normalization was already doing this.
    """
    x = a_dataset()
    assert np.mean(x) == pytest.approx(1.0, rel=1e-12)
    w = np.random.default_rng(2).dirichlet(np.ones(len(x)))
    d = WG.location_shape_split(x, w)
    rel = WG.relativize(d['w1'], WG.relative_scales(x))
    assert rel['rel_mean'] == pytest.approx(d['w1'], rel=1e-12)


# ---------------------------------------------------------------------------
# A_IQR
# ---------------------------------------------------------------------------
def test_aiqr_is_exactly_scale_invariant():
    """A density carries units of 1/x, so the area of a density band is free of
    them. This is the mechanism behind the stage's main negative result."""
    x = a_dataset(n=50)
    vals = []
    for c in (1.0, 1e-3, 7.5, 1e4):
        xs = x * c
        g = WG.aiqr_grid(xs)
        d = WG.dirichlet_draws(len(xs), np.random.default_rng(11), n_draws=60)
        vals.append(WG.aiqr(WG.density_ensemble(xs, d, g), g))
    assert max(vals) - min(vals) < 1e-9 * max(vals)


def test_aiqr_is_nearly_blind_to_dispersion():
    """A_IQR barely moves when the spread of the data changes by a factor of 20.

    THIS IS THE RESULT THAT NARROWS DECISION 90, which expected A_IQR to track
    the coefficient of variation. It cannot: a measure that is exactly
    invariant to rescaling cannot respond to the scale of the data. The
    mean-relative separation, which is what the flip probability is calibrated
    in, moves by an order of magnitude over the same range, and the test checks
    both so the contrast cannot quietly disappear.
    """
    aiqr, sep, cv = [], [], []
    for sigma in (0.2, 0.6, 1.2, 2.0):
        y = np.exp(np.random.default_rng(3).normal(0.0, sigma, size=60))
        y = y / y.mean()
        g = WG.aiqr_grid(y)
        d = WG.dirichlet_draws(len(y), np.random.default_rng(11), n_draws=200)
        cv.append(float(np.std(y) / np.mean(y)))
        aiqr.append(WG.aiqr(WG.density_ensemble(y, d, g), g))
        sep.append(float(np.median(WG.weighting_separation(y, d, g))))
    assert cv[-1] / cv[0] > 10                      # the spread really varies
    assert max(aiqr) / min(aiqr) < 1.5              # A_IQR does not follow it
    assert sep[-1] / sep[0] > 5                     # the separation does


def test_aiqr_falls_with_dataset_size():
    """More kernels average the weight noise away, so the band narrows."""
    vals = []
    for n in (10, 100, 1000):
        y = np.exp(np.random.default_rng(7).normal(0.0, 0.6, size=n))
        y = y / y.mean()
        g = WG.aiqr_grid(y)
        d = WG.dirichlet_draws(n, np.random.default_rng(11), n_draws=200)
        vals.append(WG.aiqr(WG.density_ensemble(y, d, g), g))
    assert vals[0] > vals[1] > vals[2]


def test_aiqr_is_zero_when_every_draw_is_the_same():
    """A degenerate ensemble has no band. The floor is exact, not approximate."""
    x = a_dataset(n=30)
    g = WG.aiqr_grid(x)
    same = np.tile(WG.uniform_weights(x), (25, 1))
    assert WG.aiqr(WG.density_ensemble(x, same, g), g) == pytest.approx(0.0,
                                                                        abs=1e-12)


def test_density_ensemble_rows_are_densities():
    """Each row integrates to about one on the grid, so the band is comparable
    across datasets rather than carrying an unnormalized mass."""
    x = a_dataset(n=40)
    g = WG.aiqr_grid(x)
    d = WG.dirichlet_draws(len(x), np.random.default_rng(4), n_draws=20)
    mass = np.trapezoid(WG.density_ensemble(x, d, g), g, axis=1)
    assert np.all(mass > 0.97) and np.all(mass <= 1.0 + 1e-9)


def test_weighting_separation_is_zero_at_uniform_weights():
    x = a_dataset(n=40)
    g = WG.aiqr_grid(x)
    u = np.tile(WG.uniform_weights(x), (3, 1))
    assert np.allclose(WG.weighting_separation(x, u, g), 0.0, atol=1e-12)


def test_dirichlet_draws_are_weights():
    d = WG.dirichlet_draws(12, np.random.default_rng(0), n_draws=50)
    assert d.shape == (50, 12)
    assert np.allclose(d.sum(axis=1), 1.0)
    assert (d > 0).all()

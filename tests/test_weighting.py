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


def test_aiqr_tracks_sample_size_while_the_separation_tracks_dispersion():
    """At fixed n, A_IQR barely moves over a 27-fold change in spread. The
    separation moves with it, almost exactly in proportion.

    THIS IS THE RESULT THAT NARROWS DECISION 90, which expected A_IQR to track
    the coefficient of variation.

    AND IT PINS THE MECHANISM, because the obvious explanation is wrong and was
    written into this project before it was checked. A_IQR is exactly invariant
    under rescaling the data -- the test above proves it -- but SO IS the
    mean-relative separation, so invariance is not what separates them. What
    separates them is what each divides by. A_IQR measures the density's
    uncertainty against that curve's own height and width, so the spread cancels
    twice and only the weight sampling noise survives, which is a question of
    how many points there are. The separation is an x-axis distance over the
    mean alone, so the spread-to-mean ratio survives, and that ratio IS the
    coefficient of variation.

    Both halves are asserted, so neither claim can quietly rot.
    """
    aiqr, sep, cv = [], [], []
    n = 60
    for sigma in (0.2, 0.6, 1.2, 2.0):
        y = np.exp(np.random.default_rng(3).normal(0.0, sigma, size=n))
        y = y / y.mean()
        g = WG.aiqr_grid(y)
        d = WG.dirichlet_draws(len(y), np.random.default_rng(11), n_draws=200)
        cv.append(float(np.std(y) / np.mean(y)))
        aiqr.append(WG.aiqr(WG.density_ensemble(y, d, g), g))
        sep.append(float(np.median(WG.weighting_separation(y, d, g))))
    assert cv[-1] / cv[0] > 10                      # the spread really varies
    assert max(aiqr) / min(aiqr) < 1.2              # A_IQR barely follows it
    assert sep[-1] / sep[0] > 10                    # the separation does

    # A_IQR * sqrt(n) is the quantity that is nearly constant here, which is the
    # positive form of the claim rather than the negative one.
    scaled = [a * np.sqrt(n) for a in aiqr]
    assert max(scaled) / min(scaled) < 1.2

    # And the separation is proportional to the coefficient of variation.
    ratio = [s / c for s, c in zip(sep, cv)]
    assert max(ratio) / min(ratio) < 1.3


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


# ---------------------------------------------------------------------------
# clustered market share
# ---------------------------------------------------------------------------
def test_block_weights_are_weights_and_respect_their_groups():
    rng = np.random.default_rng(0)
    for adjacent in (True, False):
        w = WG.block_weights(40, 5, rng, adjacent=adjacent)
        assert w.shape == (40,)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()
        # five groups, so at most five distinct weight levels
        assert len(np.unique(np.round(w, 12))) <= 5


def test_adjacent_blocks_are_contiguous_and_scattered_ones_are_not():
    """The two schemes differ ONLY in placement, which is what the experiment
    comparing them depends on."""
    rng = np.random.default_rng(1)
    adj = WG.block_weights(60, 4, rng, adjacent=True)
    runs = np.sum(np.diff(adj) != 0) + 1
    assert runs <= 4                               # contiguous blocks
    sca = WG.block_weights(60, 4, np.random.default_rng(1), adjacent=False)
    assert np.sum(np.diff(sca) != 0) + 1 > 4       # interleaved
    assert sorted(np.round(adj, 12)) == pytest.approx(sorted(np.round(sca, 12)))


def test_effective_n_is_the_kish_size():
    n = 50
    assert WG.effective_n(np.full(n, 1.0 / n)) == pytest.approx(float(n))
    spike = np.zeros(n)
    spike[0] = 1.0
    assert WG.effective_n(spike) == pytest.approx(1.0)


def test_adjacent_clustering_moves_the_density_more_than_scattered():
    """THE ANSWER TO THE AUTHOR'S QUESTION, pinned so it cannot quietly reverse.

    At a comparable effective sample size, concentrating market share on
    ADJACENT values moves the fitted density further than concentrating it on
    random ones, because a contiguous block shifts the whole distribution one
    way while random concentration partly cancels. That is why the effective
    sample size captures concentration but not coherence, and why a flat
    Dirichlet understates the weighting risk.
    """
    x = np.sort(np.exp(np.random.default_rng(4).normal(0.0, 0.7, size=120)))
    x = x / x.mean()
    g = WG.aiqr_grid(x)
    adj, sca, neff_a, neff_s = [], [], [], []
    rng = np.random.default_rng(6)
    for _ in range(40):
        k = int(rng.integers(3, 30))
        wa = WG.block_weights(len(x), k, rng, adjacent=True)
        ws = WG.block_weights(len(x), k, rng, adjacent=False)
        adj.append(WG.weighting_separation(x, [wa], g)[0])
        sca.append(WG.weighting_separation(x, [ws], g)[0])
        neff_a.append(WG.effective_n(wa))
        neff_s.append(WG.effective_n(ws))
    # the two schemes are matched on concentration ...
    assert np.median(neff_a) == pytest.approx(np.median(neff_s), rel=0.35)
    # ... and differ substantially on what that concentration does
    assert np.median(adj) > 1.5 * np.median(sca)

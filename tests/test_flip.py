"""Stage 2d: the flip-probability calibration.

The deliverable is a curve, so what these pin are the properties that make the
curve readable: that common random numbers really do remove the Monte Carlo
floor, that the isotonic fit is an isotonic fit, that a logistic crossing
inverts its own fit, and that the bootstrap resamples pLCA GROUPS rather than
rows, because the fifteen method pairs inside a group are not fifteen
independent observations.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import fitting as FT            # noqa: E402
import flip as FL               # noqa: E402
import weighting as WG          # noqa: E402


def a_group(seed=0, n=60, k=4):
    """Four datasets and their six fitted models, through the real pipeline."""
    rng = np.random.default_rng(seed)
    names, data, models = [], {}, {}
    for j in range(k):
        x = np.exp(rng.normal(0.0, 0.5 + 0.1 * j, size=n))
        x = x / x.mean()
        w = rng.dirichlet(np.ones(n))
        name = f'd{j}'
        names.append(name)
        data[name] = dict(data=x, weights=w)
        models[name] = FT.fit_pewt(x, w)[0]
    return names, data, models


# ---------------------------------------------------------------------------
# common random numbers
# ---------------------------------------------------------------------------
def test_common_random_numbers_make_a_method_identical_to_itself():
    """The floor is zero BY CONSTRUCTION, which is the claim the stage rests on.

    Under independent streams this comparison flips the top contributor about 5
    percent of the time with no model difference at all. Sharing the uniform
    variates makes the two draws the same array, so there is nothing left to
    measure but the models.
    """
    names, _, models = a_group()
    u = np.random.default_rng(1).random((4000, len(names)))
    first = FL.group_outcomes(models, names, u)
    second = FL.group_outcomes(models, names, u)
    for m in FT.PEWT:
        assert first[m][0] == second[m][0]
        assert first[m][1] == second[m][1]
        assert np.array_equal(first[m][2], second[m][2])


def test_independent_streams_do_flip_the_answer():
    """The control: without shared variates the same models disagree.

    If this ever passed trivially, the test above would be proving nothing.
    """
    names, _, models = a_group(seed=5)
    rng = np.random.default_rng(2)
    flips = 0
    for _ in range(60):
        a = FL.group_outcomes(models, names, rng.random((2000, len(names))))
        b = FL.group_outcomes(models, names, rng.random((2000, len(names))))
        flips += sum(a[m][1] != b[m][1] for m in FT.PEWT)
    assert flips > 0


def test_rank1_frequencies_sum_to_one():
    names, _, models = a_group()
    u = np.random.default_rng(3).random((2000, len(names)))
    for _, (_, _, rank1) in FL.group_outcomes(models, names, u).items():
        assert rank1.sum() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# the tempered calibration set
# ---------------------------------------------------------------------------
def test_temper_zero_is_exactly_uniform_and_never_flips():
    """t = 0 is the control that the whole calibration set is checked against."""
    names, data, _ = a_group(seed=7, n=40)
    combos = [names]
    d = FL.weighting_calibration(data, combos, np.random.default_rng(4),
                                 tempers=(0.0, 0.5), neccs=2000)
    zero = d[d.temper == 0.0]
    assert len(zero) == 1
    assert zero.rel_mean_max.iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert not bool(zero.flip_top.iloc[0])
    assert not bool(zero.flip_order.iloc[0])


def test_separation_grows_with_the_tempering_level():
    """The device has to do what it says: more tempering, more separation."""
    names, data, _ = a_group(seed=8, n=30)
    d = FL.weighting_calibration(data, [names], np.random.default_rng(6),
                                 tempers=(0.0, 0.1, 0.4, 1.0), neccs=1000)
    sep = d.sort_values('temper').rel_mean_max.to_numpy()
    assert np.all(np.diff(sep) > 0)


def test_tempered_weights_interpolate():
    n, draw = 5, np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    assert np.allclose(FL.tempered_weights(n, draw, 0.0), np.full(n, 1 / n))
    assert np.allclose(FL.tempered_weights(n, draw, 1.0), draw)
    mid = FL.tempered_weights(n, draw, 0.5)
    assert mid.sum() == pytest.approx(1.0)
    assert np.allclose(mid, 0.5 * np.full(n, 1 / n) + 0.5 * draw)


# ---------------------------------------------------------------------------
# the curve
# ---------------------------------------------------------------------------
def test_logistic_recovers_a_known_curve():
    rng = np.random.default_rng(0)
    x = np.exp(rng.uniform(-8, 1, size=40_000))
    p = 1.0 / (1.0 + np.exp(-(3.0 + 1.5 * np.log(x))))
    y = (rng.random(len(x)) < p).astype(float)
    a, b = FL.logistic_fit(x, y)
    assert a == pytest.approx(3.0, abs=0.15)
    assert b == pytest.approx(1.5, abs=0.06)


def test_crossing_inverts_the_fit():
    beta = (3.0, 1.5)
    for level in FL.LEVELS:
        x = FL.logistic_crossing(beta, level)
        from scipy.special import expit
        assert float(expit(beta[0] + beta[1] * np.log(x))) == pytest.approx(
            level, rel=1e-10)


def test_isotonic_is_monotone_and_preserves_the_mean():
    rng = np.random.default_rng(1)
    x = np.sort(rng.random(500))
    y = (rng.random(500) < x).astype(float)
    xs, p = FL.isotonic_curve(x, y)
    assert len(p) == len(y)                       # no point may be dropped
    assert np.all(np.diff(p) >= -1e-12)
    assert p.mean() == pytest.approx(y.mean(), rel=1e-12)


def test_isotonic_crossing_is_the_first_x_at_the_level():
    x = np.arange(1.0, 7.0)
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert FL.isotonic_crossing(x, y, 0.5) == 4.0
    assert np.isnan(FL.isotonic_crossing(x, y, 1.5))


def test_cluster_bootstrap_is_wider_than_a_row_bootstrap():
    """THE RESAMPLING UNIT MATTERS, and this is why it is the pLCA.

    Rows inside a group are built to be near-identical here, which is the
    extreme version of what the real data does: fifteen method pairs sharing
    four datasets and one set of uniform variates. Treating them as independent
    understates the spread of every crossing.
    """
    rng = np.random.default_rng(2)
    n_groups, per = 150, 15
    rows = []
    for g in range(n_groups):
        base = np.exp(rng.uniform(-6, 0))
        hit = rng.random() < 1.0 / (1.0 + np.exp(-(3.0 + 1.5 * np.log(base))))
        for _ in range(per):
            rows.append(dict(plca=g, x=base * np.exp(rng.normal(0, 0.01)),
                             y=float(hit)))
    f = pd.DataFrame(rows)
    clustered = FL.bootstrap_crossings(f, 'x', 'y', levels=(0.05,),
                                       resamples=200,
                                       rng=np.random.default_rng(3))
    f2 = f.assign(plca=np.arange(len(f)))         # pretend every row is its own
    naive = FL.bootstrap_crossings(f2, 'x', 'y', levels=(0.05,),
                                   resamples=200, rng=np.random.default_rng(3))
    wide = clustered.ci_hi.iloc[0] - clustered.ci_lo.iloc[0]
    narrow = naive.ci_hi.iloc[0] - naive.ci_lo.iloc[0]
    assert wide > 2.0 * narrow


def test_bootstrap_interval_covers_the_point_estimate():
    rng = np.random.default_rng(4)
    x = np.exp(rng.uniform(-8, 1, size=8000))
    p = 1.0 / (1.0 + np.exp(-(3.0 + 1.5 * np.log(x))))
    f = pd.DataFrame(dict(plca=np.arange(len(x)) // 8, x=x,
                          y=(rng.random(len(x)) < p).astype(float)))
    c = FL.bootstrap_crossings(f, 'x', 'y', resamples=200,
                              rng=np.random.default_rng(5))
    assert (c.ci_lo <= c.crossing).all() and (c.crossing <= c.ci_hi).all()
    assert (c.crossing.diff().dropna() > 0).all()   # higher level, further out


def test_binned_curve_covers_every_row():
    rng = np.random.default_rng(6)
    f = pd.DataFrame(dict(plca=np.arange(600) // 4,
                          x=np.exp(rng.uniform(-6, 0, size=600)),
                          y=(rng.random(600) < 0.3).astype(float)))
    b = FL.binned_curve(f, 'x', 'y', bins=10)
    assert b.n.sum() == len(f)
    assert (b.x_median.diff().dropna() > 0).all()


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------
def test_pair_distance_is_zero_between_a_model_and_itself():
    names, data, models = a_group(seed=9, n=50)
    x = data[names[0]]['data']
    d = FL.pair_distances(models[names[0]], x, WG.relative_scales(x))
    assert len(d) == 15
    for (a, b), v in d.items():
        assert a != b
        assert v['w1'] >= 0.0
    same = FL.pair_distances({'A': models[names[0]]['KDE, Uniform'],
                              'B': models[names[0]]['KDE, Uniform']},
                             x, WG.relative_scales(x), methods=('A', 'B'))
    assert same[('A', 'B')]['w1'] == pytest.approx(0.0, abs=1e-12)


def test_pair_distance_is_scale_invariant_in_relative_units():
    """A distance in units of the dataset's mean does not move when the ECCs are
    reported in different units, which is what makes the calibration portable."""
    rng = np.random.default_rng(10)
    x = np.exp(rng.normal(0, 0.5, size=60))
    x = x / x.mean()
    w = rng.dirichlet(np.ones(len(x)))
    a = FL.pair_distances(FT.fit_pewt(x, w)[0], x, WG.relative_scales(x))
    c = 250.0
    b = FL.pair_distances(FT.fit_pewt(x * c, w)[0], x * c,
                          WG.relative_scales(x * c))
    for k in a:
        assert b[k]['rel_mean'] == pytest.approx(a[k]['rel_mean'], rel=1e-6)

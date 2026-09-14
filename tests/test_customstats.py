"""
Unit tests for the weighted statistics in src/customstats.py.

Hand-computed expectations wherever the arithmetic is small enough to do on
paper, and agreement with an independent reference implementation otherwise.
These are the tests whose absence let the weighted_quantile ordering defect
survive: nothing ever checked the function against a known answer.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customstats import (  # noqa: E402
    weighted_bw,
    weighted_ecdf,
    weighted_kurtosis,
    weighted_mean,
    weighted_quantile,
    weighted_skew,
    weighted_std,
    weighted_var,
    wasserstein1_weighted,
    wasserstein2_weighted,
)


# ---------------------------------------------------------------- quantile

def test_quantile_hand_computed():
    """X=[10,20,30], W=[0.5,0.25,0.25].

    Aggregated CDF knots are x=[10,10,20,30], F=[0, 0.5, 0.75, 1.0], and the
    function linearly interpolates the inverse CDF between them.
    """
    x = np.array([10.0, 20.0, 30.0])
    w = np.array([0.5, 0.25, 0.25])
    for q, expected in [(0.0, 10.0), (0.5, 10.0), (0.625, 15.0),
                        (0.75, 20.0), (0.875, 25.0), (1.0, 30.0)]:
        assert weighted_quantile(x, w, q) == pytest.approx(expected, abs=1e-12)


def test_quantile_is_order_invariant():
    """The defect this function had: the answer depended on input ordering."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(2, 150))
        x = rng.lognormal(0, 1, n)
        w = rng.dirichlet(np.ones(n))
        order = rng.permutation(n)
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            assert weighted_quantile(x, w, q) == pytest.approx(
                weighted_quantile(x[order], w[order], q), rel=1e-12
            )


def test_quantile_order_invariant_with_ties():
    """Ties are the case that survived the first attempt at the fix.

    np.argsort is not stable, so tied values had their weights attached in
    different orders, which moved the intermediate CDF and therefore the
    interpolated quantile.
    """
    x = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    w = np.array([0.3, 0.1, 0.25, 0.05, 0.2, 0.1])
    rng = np.random.default_rng(1)
    base = [weighted_quantile(x, w, q) for q in (0.2, 0.4, 0.6, 0.8)]
    for _ in range(50):
        o = rng.permutation(len(x))
        got = [weighted_quantile(x[o], w[o], q) for q in (0.2, 0.4, 0.6, 0.8)]
        assert np.allclose(base, got, rtol=1e-12)


def test_quantile_endpoints_and_monotonicity():
    x = np.array([5.0, 1.0, 9.0, 3.0])
    w = np.array([1.0, 2.0, 0.0, 3.0])  # a zero weight must not break it
    assert weighted_quantile(x, w, 0.0) == pytest.approx(x.min())
    assert weighted_quantile(x, w, 1.0) == pytest.approx(x.max())
    values = weighted_quantile(x, w, np.linspace(0, 1, 101))
    assert np.all(np.diff(values) >= -1e-12)


def test_quantile_unnormalized_weights():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert weighted_quantile(x, w, 0.5) == pytest.approx(
        weighted_quantile(x, w * 7.5, 0.5)
    )


def test_quantile_rejects_bad_input():
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([1.0]), 0.5)
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([-1.0, 2.0]), 0.5)
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([0.0, 0.0]), 0.5)
    with pytest.raises(ValueError):
        weighted_quantile(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 0.5, output="nope")


# ---------------------------------------------------------------- moments

def test_weighted_mean_and_var_hand_computed():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([0.2, 0.3, 0.5])
    assert weighted_mean(x, w) == pytest.approx(0.2 + 0.6 + 1.5)
    mu = 2.3
    expected = 0.2 * (1 - mu) ** 2 + 0.3 * (2 - mu) ** 2 + 0.5 * (3 - mu) ** 2
    assert weighted_var(x, w) == pytest.approx(expected)
    assert weighted_std(x, w) == pytest.approx(np.sqrt(expected))


def test_uniform_weights_match_scipy():
    """With uniform weights the weighted moments must equal scipy's."""
    rng = np.random.default_rng(2)
    x = rng.lognormal(0, 0.7, 60)
    w = np.ones_like(x) / len(x)
    assert weighted_skew(x, w, bias=True) == pytest.approx(stats.skew(x, bias=True))
    assert weighted_skew(x, w, bias=False) == pytest.approx(stats.skew(x, bias=False))
    assert weighted_kurtosis(x, w, bias=True) == pytest.approx(stats.kurtosis(x, bias=True))
    assert weighted_kurtosis(x, w, bias=False) == pytest.approx(stats.kurtosis(x, bias=False))


def test_weighted_std_with_kernel_variance():
    """Passing bandwidths adds kernel variance, as a KDE's total variance."""
    x = np.array([0.0, 2.0])
    w = np.array([0.5, 0.5])
    bw = np.array([0.5, 0.5])
    assert weighted_std(x, w) == pytest.approx(1.0)
    assert weighted_std(x, w, bw) == pytest.approx(np.sqrt(1.0 + 0.25))


# ---------------------------------------------------------------- bandwidth

def test_bandwidth_formulas():
    """Scott is 1.06*sigma*n_eff^-0.2; Silverman is 0.9*min(sigma, IQR/1.34)*n_eff^-0.2."""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 200)
    w = np.ones_like(x) / len(x)
    n_eff = 1.0 / np.sum(w ** 2)
    sigma = weighted_std(x, w)
    assert weighted_bw(x, w, "scott") == pytest.approx(1.06 * sigma * n_eff ** -0.2)
    iqr = weighted_quantile(x, w, 0.75) - weighted_quantile(x, w, 0.25)
    assert weighted_bw(x, w, "silverman") == pytest.approx(
        0.9 * min(sigma, iqr / 1.34) * n_eff ** -0.2
    )


def test_bandwidth_rejects_unknown_rule():
    x = np.array([1.0, 2.0, 3.0])
    w = np.ones(3) / 3
    with pytest.raises(ValueError):
        weighted_bw(x, w, "sheather-jones")


def test_bandwidth_order_invariant():
    """Regression: the Silverman path reaches weighted_quantile."""
    rng = np.random.default_rng(4)
    for _ in range(200):
        n = int(rng.integers(4, 200))
        x = rng.lognormal(0, 1, n)
        w = rng.dirichlet(np.ones(n))
        o = rng.permutation(n)
        for rule in ("scott", "silverman"):
            assert weighted_bw(x, w, rule) == pytest.approx(
                weighted_bw(x[o], w[o], rule), rel=1e-12
            )


# ---------------------------------------------------------------- distances

def test_wasserstein1_matches_scipy_unweighted():
    rng = np.random.default_rng(5)
    p = rng.normal(0, 1, 100)
    q = rng.normal(1, 2, 150)
    assert wasserstein1_weighted(p, q) == pytest.approx(
        stats.wasserstein_distance(p, q)
    )


def test_wasserstein_identical_samples_is_zero():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([0.2, 0.3, 0.5])
    assert wasserstein1_weighted(x, x, w, w) == pytest.approx(0.0, abs=1e-12)
    assert wasserstein2_weighted(x, x, w, w) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein1_shift_equals_shift():
    """W1 between a sample and the same sample shifted by d is exactly d."""
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 200)
    for d in (0.5, 2.0):
        assert wasserstein1_weighted(x, x + d) == pytest.approx(d, rel=1e-9)


def test_wasserstein_weights_need_not_be_normalized():
    rng = np.random.default_rng(7)
    p = rng.normal(0, 1, 50)
    q = rng.normal(1, 1, 50)
    wp = rng.random(50)
    wq = rng.random(50)
    assert wasserstein1_weighted(p, q, wp, wq) == pytest.approx(
        wasserstein1_weighted(p, q, wp * 3.0, wq * 11.0)
    )


# ---------------------------------------------------------------- ecdf

def test_weighted_ecdf_steps():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([0.5, 0.25, 0.25])
    xs, ys, func = weighted_ecdf(x, w)
    assert func(1.5) == pytest.approx(0.5)
    assert func(2.5) == pytest.approx(0.75)
    assert func(3.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stage 2b: the guarded Silverman bandwidth
# ---------------------------------------------------------------------------
def test_silverman_guarded_falls_back_to_scott_below_the_threshold():
    """Small samples get Scott, large ones get Silverman, and nothing else."""
    from customstats import SILVERMAN_MIN_NEFF, weighted_bw

    rng = np.random.default_rng(0)
    for n in (3, 5, 12, 29):
        x = rng.lognormal(0.0, 0.7, n)
        w = np.ones(n) / n
        assert weighted_bw(x, w, 'silverman_guarded') == weighted_bw(x, w, 'scott')
    for n in (int(SILVERMAN_MIN_NEFF), 60, 400):
        x = rng.lognormal(0.0, 0.7, n)
        w = np.ones(n) / n
        assert weighted_bw(x, w, 'silverman_guarded') == weighted_bw(
            x, w, 'silverman')


def test_silverman_guarded_uses_effective_sample_size_not_raw_n():
    """Concentrated weights reduce n_eff, and the guard must see that.

    A Dirichlet draw can put almost all of a dataset's weight on a handful of
    points. The bandwidth formula already divides by the Kish effective sample
    size, so the guard has to use the same quantity or it would trust quartiles
    that rest on three effective observations.
    """
    from customstats import weighted_bw

    rng = np.random.default_rng(1)
    x = rng.lognormal(0.0, 0.7, 200)
    w = np.full(200, 1e-9)
    w[:4] = 0.25                      # n_eff is about 4 despite n = 200
    w = w / w.sum()
    assert 1.0 / np.sum(w ** 2) < 10
    assert weighted_bw(x, w, 'silverman_guarded') == weighted_bw(x, w, 'scott')


def test_silverman_guarded_never_collapses_the_bandwidth_at_small_n():
    """The failure it exists to prevent, asserted rather than described.

    At n = 3 to 10 the interquartile range is interpolated between two order
    statistics and can land far below the true scale; the bandwidth then
    collapses and the density becomes spikes. Measured over 149 empirical and
    800 synthetic datasets, pure Silverman beats Scott on held-out likelihood
    in only 12.5 percent of empirical fits at n = 3-9.
    """
    from customstats import weighted_bw, weighted_std

    rng = np.random.default_rng(2)
    worst = 1.0
    for seed in range(300):
        r = np.random.default_rng(seed)
        n = int(r.integers(3, 11))
        x = r.lognormal(0.0, 0.9, n)
        w = r.dirichlet(np.ones(n))
        sd = weighted_std(x, w)
        if sd <= 0:
            continue
        worst = min(worst, weighted_bw(x, w, 'silverman_guarded') / sd)
    # Scott's rule is 1.06 * n_eff ** -0.2, which at n_eff = 3 is about 0.85
    # and can only fall with n_eff. It cannot approach zero.
    assert worst > 0.3, f'guarded bandwidth collapsed to {worst:.4g} of sd'


def test_bandwidth_method_names_are_validated():
    from customstats import weighted_bw

    x = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4) / 4
    with pytest.raises(ValueError, match='silverman_guarded'):
        weighted_bw(x, w, 'sheather-jones')

"""Tests for src/modality.py, Silverman's critical bandwidth."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import modality as MD  # noqa: E402


def test_binned_kde_matches_direct_evaluation():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-3, 0.6, 400), rng.normal(3, 0.6, 400)])
    kde = MD._BinnedKDE(x)
    for h in (0.3, 0.8, 1.5):
        _, direct = MD._kde_on_grid(x, h, grid=kde.grid)
        assert np.max(np.abs(kde.density(h) - direct)) < 1e-3 * direct.max()


def test_mode_count_ignores_fft_roundoff():
    """Without a relative-height floor an n = 400 normal sample reads as more
    than a hundred modes, every one of them FFT ripple in the far tail."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 400)
    kde = MD._BinnedKDE(x)
    y = kde.density(0.28 * float(np.std(x)))
    assert MD._count_peaks(y, rel_tol=0.0) > 50
    assert MD._count_peaks(y) <= 2


def test_mode_count_is_non_increasing_in_bandwidth():
    """The fact Silverman's test rests on, for the Gaussian kernel."""
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(-4, 0.5, 200), rng.normal(0, 0.5, 200),
                        rng.normal(4, 0.5, 200)])
    kde = MD._BinnedKDE(x)
    counts = [kde.n_modes(h) for h in np.geomspace(0.02, 5.0, 40)]
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_critical_bandwidth_separates_unimodal_from_multimodal():
    rng = np.random.default_rng(3)
    uni = rng.normal(0, 1, 500)
    bi = np.concatenate([rng.normal(-3, 0.6, 250), rng.normal(3, 0.6, 250)])
    assert MD.critical_bandwidth(uni, 1) < 0.5
    assert MD.critical_bandwidth(bi, 1) > 0.8


def test_silverman_mode_count_recovers_the_truth():
    rng = np.random.default_rng(4)
    cases = {
        1: rng.normal(0, 1, 600),
        2: np.concatenate([rng.normal(-3, 0.5, 300), rng.normal(3, 0.5, 300)]),
        3: np.concatenate([rng.normal(-5, 0.4, 200), rng.normal(0, 0.4, 200),
                           rng.normal(5, 0.4, 200)]),
    }
    for truth, x in cases.items():
        assert MD.n_modes_silverman(x, rng=rng, nboot=80) == truth


def test_critical_bandwidth_is_scale_free():
    """It is reported in units of the data's standard deviation, so rescaling
    the data must not move it."""
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(-3, 0.6, 300), rng.normal(3, 0.6, 300)])
    a = MD.critical_bandwidth(x, 1)
    b = MD.critical_bandwidth(1000.0 * x + 7.0, 1)
    assert a == pytest.approx(b, rel=1e-6)


def test_critical_bandwidth_is_defined_at_small_n():
    """Stratum 1 runs down to n = 3, and this must not raise."""
    rng = np.random.default_rng(6)
    for n in (3, 4, 5, 9):
        v = MD.critical_bandwidth(rng.normal(0, 1, n), 1)
        assert np.isfinite(v)


def test_mixture_fit_recovers_two_components():
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(-3, 0.6, 500), rng.normal(3, 0.6, 500)])
    k, pi, mu, sd = MD.fit_mixture_bic(x, rng)
    assert k == 2
    assert sorted(np.round(mu)) == [-3.0, 3.0]


def test_n_modes_fitted_counts_the_density_the_study_fits():
    """The reported modality figure must come from the density the paper shows,
    not from `gaussian_kde`'s default bandwidth, which is Scott's rule and
    oversmooths this data by about 35 percent."""
    import fitting as FT
    rng = np.random.default_rng(3)
    two = np.concatenate([rng.normal(1.0, 0.08, 120),
                          rng.normal(2.0, 0.08, 120)])
    assert MD.n_modes_fitted(two) == 2
    one = np.abs(rng.lognormal(0.0, 0.5, 300)) + 0.05
    assert MD.n_modes_fitted(one) == 1
    # it is the STUDY's bandwidth, so passing that rule explicitly is a no-op
    assert MD.n_modes_fitted(two) == MD.n_modes_fitted(
        two, bw_method=FT.BW_METHOD)


def test_n_modes_fitted_sees_structure_the_default_bandwidth_smooths_away():
    """The whole reason the corrected measure exists. A wider bandwidth can only
    merge modes, never split them, so the fitted count is never the lower of the
    two and is sometimes higher."""
    rng = np.random.default_rng(11)
    higher = 0
    for _ in range(25):
        x = np.concatenate([rng.lognormal(0.0, 0.25, 90),
                            rng.lognormal(0.95, 0.25, 60)])
        a = MD.n_modes_fitted(x)
        b = MD.n_modes_visible(x)
        assert a >= b
        higher += int(a > b)
    assert higher >= 1


def test_n_modes_fitted_is_defined_at_the_small_end():
    assert MD.n_modes_fitted(np.array([1.0, 2.0, 3.0])) == 1
    assert MD.n_modes_fitted(np.ones(50)) == 1

"""The support, the families and the two estimators. Stage 2b.

What these guard, in one sentence each: that every model is the same object
when scored and when sampled; that the lognormal threshold pathology is caught
rather than walked into; and that the closed-form estimators are the maximum
likelihood estimators they claim to be.
"""
import os
import sys

import numpy as np
import pytest
from scipy.stats import gaussian_kde, gamma as gamma_dist, lognorm, norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import families as F  # noqa: E402
import fitting as FT  # noqa: E402

FAMILIES = ('normal', 'lognormal_2p', 'lognormal_3p', 'lognormal_offset',
            'gamma')


def sample(kind, n, seed):
    rng = np.random.default_rng(seed)
    if kind == 'lognormal':
        x = rng.lognormal(0.0, 0.7, n)
    elif kind == 'gamma':
        x = rng.gamma(2.0, 0.5, n)
    elif kind == 'near_normal':
        x = np.abs(rng.normal(1.0, 0.25, n)) + 1e-6
    elif kind == 'near_zero':
        # A handful of values orders of magnitude below the mean, which is the
        # configuration LOGFIT_OFFSET was patching and the one every fitting
        # method has to survive.
        x = rng.lognormal(0.0, 0.5, n)
        x[:max(1, n // 20)] = 1e-5
    else:
        raise ValueError(kind)
    x = x / x.mean()
    w = rng.dirichlet(np.ones(n))
    return x, w


SHAPES = ['lognormal', 'gamma', 'near_normal', 'near_zero']


# ---------------------------------------------------------------------------
# the support: (0, inf), open at zero
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kind', SHAPES)
@pytest.mark.parametrize('family', FAMILIES)
def test_model_is_supported_only_above_zero(family, kind):
    x, w = sample(kind, 200, 11)
    m, _ = FT.fit_family(family, x, w, 'mle')
    assert m.pdf(np.array([-1.0, -1e-9, 0.0])).tolist() == [0.0, 0.0, 0.0]
    assert m.cdf(np.array([-1.0, 0.0])).tolist() == [0.0, 0.0]


@pytest.mark.parametrize('kind', SHAPES)
@pytest.mark.parametrize('family', FAMILIES)
def test_cdf_inverts_its_own_ppf(family, kind):
    x, w = sample(kind, 200, 12)
    m, _ = FT.fit_family(family, x, w, 'mle')
    q = np.array([1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6])
    assert np.allclose(m.cdf(m.ppf(q)), q, atol=1e-8)


@pytest.mark.parametrize('kind', SHAPES)
@pytest.mark.parametrize('family', FAMILIES)
def test_sampling_never_emits_an_inadmissible_value(family, kind):
    """Zero is not an admissible ECC, so no draw may be zero or negative."""
    x, w = sample(kind, 200, 13)
    m, _ = FT.fit_family(family, x, w, 'mle')
    draws = m.rvs(20_000, np.random.default_rng(0))
    assert draws.min() > 0.0
    assert np.isfinite(draws).all()
    assert m.ppf(np.array([0.0]))[0] > 0.0


@pytest.mark.parametrize('family', FAMILIES)
def test_rvs_from_uniform_is_the_same_map_as_rvs(family):
    """Stage 2e's common random numbers need the map, not just the sampler."""
    x, w = sample('lognormal', 150, 14)
    m, _ = FT.fit_family(family, x, w, 'mle')
    u = np.random.default_rng(1).random(500)
    assert np.array_equal(m.rvs_from_uniform(u), m.ppf(u))
    assert np.array_equal(m.rvs(500, np.random.default_rng(7)),
                          m.ppf(np.random.default_rng(7).random(500)))


@pytest.mark.parametrize('family', FAMILIES)
def test_inverse_cdf_sampling_reproduces_the_model_cdf(family):
    x, w = sample('lognormal', 200, 15)
    m, _ = FT.fit_family(family, x, w, 'mle')
    draws = m.rvs(200_000, np.random.default_rng(3))
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert abs((draws <= m.ppf(np.array([q]))[0]).mean() - q) < 0.01


def test_truncation_renormalizes_rather_than_discarding_mass():
    x, w = sample('near_normal', 200, 16)
    p = F.fit_normal_mle(x, w)
    m = F.make_normal(p)
    assert m.mass_below > 0, 'this fixture exists to have mass below zero'
    grid = np.linspace(1e-9, p['loc'] + 12 * p['scale'], 400_001)
    assert abs(np.trapezoid(m.pdf(grid), grid) - 1.0) < 1e-6
    assert abs(m.cdf(np.array([grid[-1]]))[0] - 1.0) < 1e-6


def test_truncation_refuses_a_parent_with_no_admissible_mass():
    with pytest.raises(ValueError, match='no mass'):
        F.Truncated(norm(loc=-100.0, scale=1.0))


# ---------------------------------------------------------------------------
# the KDE, as a distribution rather than a density plus a resampler
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('n', [5, 50, 500, 3000])
def test_weighted_kde_tabulation_matches_the_direct_kernel_sums(n):
    """pdf, cdf and ppf all read one binned tabulation. It must match the sums.

    Evaluating a kernel sum directly costs O(len(q) * n), and the analysis does
    it on a 1,000-point grid for every dataset on both weighting schemes, with
    datasets up to 9,996 values. The tabulation costs O(n + G log G) once.
    """
    x, w = sample('lognormal', n, 70 + n)
    k = F.WeightedKDE(x, w, FT.weighted_bw(x, w, bw_method=FT.BW_METHOD))
    q = np.linspace(1e-3, float(x.max()) * 2.0, 997)
    assert np.max(np.abs(k.pdf(q) - k.pdf_exact(q))) < 1e-5
    assert np.max(np.abs(k.cdf(q) - k.cdf_exact(q))) < 1e-5
    assert np.all(np.diff(k.cdf(q)) >= -1e-12)


def test_weighted_kde_density_matches_scipy():
    x, w = sample('lognormal', 300, 17)
    bw = FT.weighted_bw(x, w, bw_method=FT.BW_METHOD)
    mine = F.WeightedKDE(x, w, bw)
    ref = gaussian_kde(x, bw_method=1.0, weights=w)
    ref.set_bandwidth(bw / (ref.covariance ** 0.5)[0][0])
    q = np.linspace(0.01, 4.0, 97)
    # The direct sum is the definition and must match scipy exactly; the
    # tabulated `pdf` matches it to interpolation accuracy.
    assert np.allclose(mine.pdf_exact(q), ref.pdf(q), rtol=1e-10, atol=0)
    assert np.allclose(mine.pdf(q), ref.pdf(q), atol=1e-5)


@pytest.mark.parametrize('n', [5, 50, 500, 3000])
def test_weighted_kde_tabulated_inverse_agrees_with_exact_inversion(n):
    """The pLCA samples through the TABLE, so the table is what must be right.

    `ppf` inverts an FFT-binned tabulation because direct inversion costs
    O(iterations * queries * n) and the pLCA draws 10,000 values from each of
    about 20,000 fitted KDEs. The tolerance is stated in PROBABILITY, which is
    the quantity that matters: pushing a uniform through `ppf` and back through
    the exact `cdf` must return the uniform. The quantile-space error is larger
    in the far tails, where the CDF is flat and a large move in x is a small
    move in probability.
    """
    x, w = sample('lognormal', n, 40 + n)
    bw = FT.weighted_bw(x, w, bw_method=FT.BW_METHOD)
    k = F.WeightedKDE(x, w, bw)
    q = np.linspace(1e-4, 1 - 1e-4, 1001)
    assert np.max(np.abs(k.cdf(k.ppf(q)) - q)) < 1e-6
    assert np.all(np.diff(k.ppf(q)) >= 0)


def test_weighted_kde_cdf_is_the_integral_of_its_own_density():
    x, w = sample('lognormal', 120, 18)
    k = F.WeightedKDE(x, w, FT.weighted_bw(x, w, bw_method=FT.BW_METHOD))
    g = np.linspace(-2.0, 8.0, 200_001)
    num = np.concatenate([[0.0], np.cumsum(
        0.5 * (k.pdf_exact(g)[1:] + k.pdf_exact(g)[:-1]) * np.diff(g))])
    assert np.allclose(num, k.cdf_exact(g), atol=1e-6)


# ---------------------------------------------------------------------------
# the estimators
# ---------------------------------------------------------------------------
def test_lognorm2_closed_form_is_the_weighted_mle():
    """The closed form must beat any nearby parameter pair on the likelihood."""
    x, w = sample('lognormal', 400, 19)
    w = w / w.sum()
    p = F.fit_lognorm2_mle(x, w)

    def ll(s, scale):
        return float(w @ lognorm.logpdf(x, s=s, scale=scale))

    best = ll(p['s'], p['scale'])
    for ds in (0.98, 1.02):
        for dsc in (0.98, 1.02):
            assert ll(p['s'] * ds, p['scale'] * dsc) <= best + 1e-12


def test_lognorm2_matches_the_legacy_fit_it_replaces():
    """Same answer as customstats.weighted_lognorm_fit, to optimizer tolerance.

    The legacy 'MLE' branch hands scipy.optimize a problem whose solution is
    closed form. It reaches the same place; the point of this test is that
    replacing it changes no number beyond the optimizer's own tolerance.
    """
    x, w = sample('lognormal', 300, 20)
    s, loc, scale = FT.weighted_lognorm_fit(x, weights=w, method='MLE')
    p = F.fit_lognorm2_mle(x, w)
    assert loc == 0.0
    assert abs(p['s'] - s) < 1e-6 and abs(p['scale'] - scale) < 1e-6


def test_gamma_mle_solves_its_own_score_equation():
    x, w = sample('gamma', 400, 21)
    w = w / w.sum()
    p = F.fit_gamma_mle(x, w)

    def ll(a, scale):
        return float(w @ gamma_dist.logpdf(x, a=a, scale=scale))

    best = ll(p['a'], p['scale'])
    for da in (0.97, 1.03):
        for dsc in (0.97, 1.03):
            assert ll(p['a'] * da, p['scale'] * dsc) <= best + 1e-9


@pytest.mark.parametrize('kind', SHAPES)
def test_profile_threshold_stays_strictly_below_the_minimum(kind):
    """The pathology is the threshold reaching min(x). It must never get there."""
    x, w = sample(kind, 150, 22)
    p = F.fit_lognorm3_profile(x, w)
    assert p['loc'] < x.min()
    assert p['threshold_delta_over_sd'] >= F.PROFILE_DELTA_LO_FRAC * (1 - 1e-9)
    assert p['status'] in ('interior', 'boundary_guard',
                           'boundary_normal_limit')


def test_profile_beats_the_unguarded_optimizer_it_replaces():
    """An unguarded joint fit chases the divergence; the profile does not.

    This is the whole reason `fit_lognorm3_profile` exists, so it is asserted
    rather than described. The unguarded fit drives the threshold onto min(x)
    and sigma through the roof; the guarded one does neither.
    """
    from scipy.optimize import minimize
    # A small lognormal sample whose smallest observation sits well clear of
    # the rest, which is when the divergent term dominates. Found by scanning
    # sizes and seeds: the pathology is a property of the likelihood but
    # whether an optimizer walks into it depends on the sample.
    rng = np.random.default_rng(4)
    x = rng.lognormal(0.0, 0.6, 12)
    x = x / x.mean()
    w = rng.dirichlet(np.ones(12))
    w = w / w.sum()
    xmin = float(x.min())

    def nll(z):
        d = x - (xmin - np.exp(z[0]))
        if np.any(d <= 0) or not np.isfinite(z).all():
            return np.inf
        s = np.exp(z[2])
        v = -float(w @ (-np.log(d) - np.log(s)
                        - 0.5 * ((np.log(d) - z[1]) / s) ** 2))
        return v if np.isfinite(v) else np.inf

    p0 = F.fit_lognorm2_mle(x, w)
    res = minimize(nll, [np.log(xmin * 0.5), np.log(p0['scale']),
                         np.log(p0['s'])], method='Nelder-Mead',
                   options=dict(maxiter=4000, xatol=1e-12, fatol=1e-14))
    naive_gap = xmin - (xmin - np.exp(res.x[0]))
    sd = float(np.sqrt(w @ (x - x @ w) ** 2))
    guarded = F.fit_lognorm3_profile(x, w)
    assert naive_gap / sd < 1e-6, 'the fixture must exhibit the pathology'
    assert (xmin - guarded['loc']) / sd >= F.PROFILE_DELTA_LO_FRAC * (1 - 1e-9)
    assert guarded['s'] < np.exp(res.x[2])


def test_profile_reaches_the_normal_limit_when_the_data_asks_for_it():
    """Decision 10: the threshold is NOT constrained away from the normal limit.

    As the threshold goes to minus infinity the lognormal converges to a normal.
    The author's position is that this is an asset of the family. Symmetric data
    should therefore drive the threshold far below the minimum.
    """
    x, w = sample('near_normal', 400, 24)
    p = F.fit_lognorm3_profile(x, w)
    assert p['threshold_delta_over_sd'] > 3.0
    assert p['s'] < 0.35


# ---------------------------------------------------------------------------
# fitting by the criterion we score by
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kind', SHAPES)
@pytest.mark.parametrize('family', FAMILIES)
def test_w1_optimal_fit_never_scores_worse_than_the_mle_fit(family, kind):
    """It starts from the MLE fit, so it cannot be worse. Guards the fallback."""
    x, w = sample(kind, 120, 25)
    m_mle, _ = FT.fit_family(family, x, w, 'mle')
    m_w1, _ = FT.fit_family(family, x, w, 'w1')
    grid = FT.score_grid_open(x, w)
    assert (FT.score_w1_model(m_w1, x, w, grid)
            <= FT.score_w1_model(m_mle, x, w, grid) + 1e-12)


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------
def test_scoring_grid_is_open_at_zero():
    x, w = sample('lognormal', 100, 26)
    g = FT.score_grid_open(x, w)
    assert g[0] > 0.0
    assert len(g) == FT.SCORE_GRID_POINTS
    assert np.isclose(g[0], g[-1] / FT.SCORE_GRID_POINTS)
    assert FT.score_grid(x, w)[0] == 0.0, 'the Stage 1 grid is kept for comparison'


def test_scoring_grid_upper_bound_is_what_the_docstring_says():
    x, w = sample('lognormal', 100, 27)
    spread = max(np.std(x), FT.weighted_std(x, w))
    assert np.isclose(FT.score_grid_open(x, w)[-1],
                      x.max() + FT.SCORE_GRID_STD_MULTIPLE * spread)


def test_profile_fit_is_a_usable_generative_distribution():
    """The fitted lognormal must have a variance the data could support.

    This is the test that would have caught the defect Stage 2b found in the
    pLCA results rather than in the fit scores. With the guard at 0.01 standard
    deviations, a profile with no interior maximum drove the threshold onto the
    guard, sigma to roughly 2, and the fitted model's standard deviation to
    thousands on data whose own standard deviation is near 0.6. W1 does not see
    that -- a thin far tail costs almost nothing in a distance between CDFs --
    but the pLCA samples from these models.

    The bound of 20 is deliberately loose. It is not a calibration; it is the
    difference between a heavy-tailed model and an unusable one.
    """
    for kind in SHAPES:
        for n in (8, 40, 200):
            x, w = sample(kind, n, 300 + n)
            m, p = FT.fit_family('lognormal_3p', x, w, 'mle')
            draws = m.ppf(np.linspace(1e-9, 1 - 1e-9, 20_001))
            data_sd = float(np.std(x))
            assert np.std(draws) < 20 * max(data_sd, 0.1), (
                f'{kind} n={n}: fitted sd {np.std(draws):.4g} against a data sd '
                f'of {data_sd:.4g}, status {p["status"]}')

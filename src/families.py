"""Parametric families and the KDE on the support (0, inf), with one interface.

Stage 2b. Decision 13 in CLAUDE.md is CONFIRMED BY THE AUTHOR: an embodied
carbon coefficient lives on (0, inf), open at zero, because zero is not an
admissible ECC. That settles a question Stage 1 flagged and four stages built
on, and it has a consequence this module exists to discharge.

WHY THIS MODULE EXISTS. Before Stage 2b the six UQ methods were one object when
JUDGED and a different object when APPLIED. The normal puts mass below zero
directly and the KDE puts mass below zero through Gaussian kernels sitting on
small values. In scoring, that mass vanished because the evaluation grid started
at zero, so the model was implicitly truncated and renormalized without anything
saying so. In the pLCA, it vanished because non-positive draws were rejected and
redrawn. The two happen to agree, but neither was written down, and nothing
would have caught them drifting apart.

Every model here is therefore an EXPLICIT truncation of its parent distribution
to (0, inf), renormalized, exposing:

    .pdf(x)     density, zero at or below the support bound
    .cdf(x)     the truncated, renormalized CDF. This is what W1 scores
    .ppf(q)     its inverse
    .rvs(size, random_state)      inverse-CDF sampling
    .rvs_from_uniform(u)          the same map applied to supplied uniforms

`rvs_from_uniform` is the reason sampling is by inverse CDF and not by
rejection. The two give the same distribution, but Stage 2e installs common
random numbers across UQ methods, which needs ONE uniform variate per material
per iteration pushed through every method's inverse CDF. Rejection sampling
cannot do that: it consumes an unpredictable number of variates per draw.

Nothing here decides how a family's parameters are estimated. That is
`fitting.py`, which has both a maximum-likelihood and a W1-optimal estimator for
every parametric family.
"""

import numpy as np
from scipy import optimize
from scipy.special import digamma
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, norm

#: The support is (0, inf), OPEN at zero. Decision 13, confirmed by the author.
#: `ppf(0)` therefore returns the smallest representable positive double rather
#: than zero, so no sampler can ever emit an inadmissible value.
SUPPORT_LO = 0.0
TINY = np.finfo(float).tiny


class Truncated:
    """A frozen scipy distribution restricted to (lo, inf) and renormalized.

    `dist` must expose pdf, cdf and ppf. The renormalizing constant is
    `1 - dist.cdf(lo)`, the mass the parent puts on the admissible region.
    """

    def __init__(self, dist, lo=SUPPORT_LO, label=None, params=None):
        self.dist = dist
        self.lo = float(lo)
        self.label = label
        self.params = dict(params or {})
        self.mass_below = float(np.ravel(dist.cdf(self.lo))[0])
        self.mass_kept = 1.0 - self.mass_below
        if not self.mass_kept > 0:
            raise ValueError(
                f'{label or dist}: the parent puts no mass on ({lo}, inf), so '
                f'it cannot be renormalized onto the support')

    # -- the three functions every model must answer -----------------------
    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x > self.lo, self.dist.pdf(x) / self.mass_kept, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = (self.dist.cdf(x) - self.mass_below) / self.mass_kept
        return np.clip(np.where(x > self.lo, out, 0.0), 0.0, 1.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        x = self.dist.ppf(self.mass_below + q * self.mass_kept)
        return np.maximum(x, TINY)

    # -- sampling ----------------------------------------------------------
    def rvs(self, size, random_state):
        """Inverse-CDF sampling. `random_state` must be a Generator."""
        return self.ppf(random_state.random(size))

    def rvs_from_uniform(self, u):
        """The inverse-CDF map applied to uniforms supplied by the caller.

        This is the entry point for the common-random-numbers scheme of
        Stage 2e: the same `u` through six models gives six comparable draws.
        """
        return self.ppf(np.asarray(u, dtype=float))

    def __repr__(self):
        p = ', '.join(f'{k}={v:.6g}' for k, v in self.params.items())
        return (f'Truncated({self.label}, {p}, lo={self.lo:g}, '
                f'mass_below={self.mass_below:.3g})')


class WeightedKDE:
    """A weighted Gaussian kernel density estimate, as a distribution object.

    The CDF is a weighted sum of normal CDFs, so it is closed form and needs no
    quadrature; only `ppf` is numerical, by bisection on a monotone function.
    This replaces `scipy.stats.gaussian_kde`, which offers a pdf and a
    resampler but no CDF and no inverse CDF, and therefore cannot be scored and
    sampled as the same object.

    `bandwidth` is supplied rather than chosen here. The rule is
    `customstats.weighted_bw`, and which rule is a Stage 2h question.
    """

    def __init__(self, x, weights, bandwidth):
        self.x = np.asarray(x, dtype=float).ravel()
        w = np.asarray(weights, dtype=float).ravel()
        self.w = w / w.sum()
        self.bandwidth = float(bandwidth)
        self._table = None
        if not self.bandwidth > 0:
            raise ValueError(f'bandwidth must be positive, got {bandwidth}')

    #: Query points per block. The kernel sum is an outer product of the query
    #: points against the data, so evaluating a 20,001-point lattice against
    #: the arm's largest dataset in one go would allocate a 20,001 x 31,025
    #: array, about 5 GB. Blocking bounds the working set without changing a
    #: single value.
    BLOCK = 1 << 20

    def _accumulate(self, q, kernel):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        out = np.empty(q.shape, dtype=float)
        step = max(1, self.BLOCK // max(1, len(self.x)))
        for i in range(0, len(q), step):
            block = q[i:i + step]
            out[i:i + step] = kernel(
                (block[:, None] - self.x[None, :]) / self.bandwidth) @ self.w
        return out

    def pdf(self, q):
        return self._accumulate(q, lambda z: np.exp(-0.5 * z ** 2)) / (
            self.bandwidth * np.sqrt(2.0 * np.pi))

    def cdf(self, q):
        return self._accumulate(q, norm.cdf)

    def ppf_exact(self, p, tol=1e-12, maxiter=200):
        """Bisection on the exact CDF, which is continuous and increasing.

        Correct and O(maxiter * len(p) * n), which is 14 seconds for 2,000
        queries against the arm's largest dataset. It is the reference `ppf` is
        tested against, not the method the pLCA can afford.
        """
        p = np.atleast_1d(np.asarray(p, dtype=float))
        lo = np.full(p.shape, self.x.min() - 40.0 * self.bandwidth)
        hi = np.full(p.shape, self.x.max() + 40.0 * self.bandwidth)
        for _ in range(maxiter):
            mid = 0.5 * (lo + hi)
            go_up = self.cdf(mid) < p
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)
            if np.all(hi - lo < tol * np.maximum(1.0, np.abs(mid))):
                break
        return 0.5 * (lo + hi)

    #: Grid points in the tabulated CDF that `ppf` inverts. The table is built
    #: by linear binning and FFT convolution, so its cost is
    #: O(n + PPF_GRID log PPF_GRID) and does NOT grow with the number of
    #: queries. Direct inversion costs O(iterations * queries * n), which the
    #: pLCA cannot afford: it draws 10,000 values from each of ~20,000 fitted
    #: KDEs.
    PPF_GRID = 1 << 15
    #: How far past the data the table reaches, in bandwidths. The Gaussian
    #: kernel has 1e-16 of its mass beyond 8 sd, so nothing is lost.
    PPF_PAD_BW = 10.0

    def _ppf_table(self):
        """(grid, cdf on that grid), cached. Linear binning plus FFT.

        This is the same construction `modality._BinnedKDE` uses for the
        critical-bandwidth search and for the same reason. Linear binning
        splits each point's weight between its two neighbouring nodes, which is
        second-order accurate where histogram binning is first-order; the
        Gaussian kernel's Fourier transform is itself a Gaussian, so the
        convolution needs no kernel array.

        `tests/test_families.py` pins the result against `ppf_exact`.
        """
        if getattr(self, '_table', None) is not None:
            return self._table
        n = self.PPF_GRID
        pad = self.PPF_PAD_BW * self.bandwidth
        grid = np.linspace(self.x.min() - pad, self.x.max() + pad, n)
        delta = float(grid[1] - grid[0])
        pos = (self.x - grid[0]) / delta
        i0 = np.clip(np.floor(pos).astype(int), 0, n - 2)
        frac = pos - i0
        counts = np.zeros(n)
        np.add.at(counts, i0, self.w * (1.0 - frac))
        np.add.at(counts, i0 + 1, self.w * frac)
        m = 2 * n
        freq = np.fft.rfftfreq(m, d=delta)
        dens = np.fft.irfft(
            np.fft.rfft(counts, m)
            * np.exp(-2.0 * (np.pi * freq * self.bandwidth) ** 2), m)[:n]
        dens = np.maximum(dens, 0.0)
        cdf = np.concatenate([[0.0], np.cumsum(
            0.5 * (dens[1:] + dens[:-1]) * delta)])
        total = cdf[-1]
        if total > 0:
            cdf = cdf / total
        # Strictly increasing, so np.interp inverts it without ties.
        cdf = np.maximum.accumulate(cdf) + np.arange(n) * 1e-15
        self._table = (grid, cdf / cdf[-1])
        return self._table

    def ppf(self, p):
        """Inverse CDF, by interpolating the tabulated CDF.

        See `_ppf_table` for the construction and why direct inversion is not
        affordable here. `ppf_exact` is the reference.
        """
        grid, cdf = self._ppf_table()
        return np.interp(np.atleast_1d(np.asarray(p, dtype=float)), cdf, grid)


# ---------------------------------------------------------------------------
# maximum-likelihood estimators, all weighted
# ---------------------------------------------------------------------------
def _norm_weights(w):
    w = np.asarray(w, dtype=float).ravel()
    return w / w.sum()


def fit_normal_mle(x, w):
    """Weighted mean and weighted standard deviation.

    This is the UQ METHOD under study, not a free choice: the manuscript
    describes the normal fit as the weighted mean and standard deviation of the
    data. It is the MLE for an UNtruncated normal, which is what makes the
    truncation an honest restatement of the same method rather than a different
    one. The W1-optimal estimator in `fitting.py` is the alternative.
    """
    x, w = np.asarray(x, float), _norm_weights(w)
    mu = float(x @ w)
    sd = float(np.sqrt(w @ (x - mu) ** 2))
    return dict(loc=mu, scale=sd)


def fit_lognorm2_mle(x, w):
    """Two-parameter lognormal on strictly positive data. Closed form.

    The weighted MLE of a lognormal is the weighted mean and standard deviation
    of `log x`; there is nothing to optimize. `customstats.weighted_lognorm_fit`
    reaches the same answer by handing that objective to `scipy.optimize`, which
    is why its results shift between scipy versions and why the regression
    fixture needed a tolerance of 4.3e-08 on the lognormal column alone while
    every other column agreed to 1e-13.
    """
    x, w = np.asarray(x, float), _norm_weights(w)
    if np.any(x <= 0):
        raise ValueError('the two-parameter lognormal needs strictly positive '
                         'data; that is the whole point of it')
    lx = np.log(x)
    mu = float(lx @ w)
    sigma = float(np.sqrt(w @ (lx - mu) ** 2))
    return dict(s=sigma, loc=0.0, scale=float(np.exp(mu)))


def _lognorm3_profile_at(x, w, gamma):
    """Concentrated MLE of (mu, sigma) at a fixed threshold, and the profile.

    With gamma fixed the lognormal is a two-parameter lognormal on `x - gamma`,
    so mu and sigma are closed form and the profile log-likelihood is

        l(gamma) = -sum_i w_i log(x_i - gamma) - log sigma(gamma) - 1/2
                   - 1/2 log(2 pi)
    """
    d = x - gamma
    if np.any(d <= 0):
        return None
    ld = np.log(d)
    mu = float(ld @ w)
    sigma = float(np.sqrt(w @ (ld - mu) ** 2))
    if not sigma > 0:
        return None
    ll = float(-(w @ ld) - np.log(sigma) - 0.5 - 0.5 * np.log(2.0 * np.pi))
    return mu, sigma, ll


#: How far below min(x) the threshold grid must stay, as a fraction of the
#: weighted standard deviation. See `fit_lognorm3_profile` for why a guard is
#: required rather than optional, and THE GUARD IS NOT COSMETIC, below, for why
#: this value and not a smaller one.
#:
#: THE GUARD IS NOT COSMETIC. Stage 2b set it to 0.01 first, on the reasoning
#: that a guard only has to stop the divergence. It does stop the divergence,
#: and it leaves a model that is unusable as a generative distribution: where
#: the profile likelihood has no interior maximum, the threshold is driven onto
#: the guard, sigma goes to roughly 1.8 to 2.3, and the fitted lognormal matches
#: the BODY of the data while carrying an enormous right tail. W1 does not see
#: that tail -- it is a distance between CDFs and a thin far tail costs little --
#: but the pLCA SAMPLES from these models, and a model with a standard deviation
#: of 3,000 on data whose standard deviation is 0.6 dominates any Monte Carlo it
#: enters. It was found in the pLCA results, not in the fit scores.
#:
#: Swept over both arms; `outputs/tables/stage2b/TABLE_2b_ProfileGuardSweep.csv`.
#: `max_sd` is the largest standard deviation of any fitted model, on data whose
#: own standard deviation is near 0.6:
#:
#:   frac   empirical max_sd / mean W1     synthetic max_sd / mean W1
#:   0.01        3281 / 0.1815                  5345 / 0.1037
#:   0.05          77 / 0.1699                    99 / 0.0995
#:   0.10          18 / 0.1691                    19 / 0.0979
#:   0.25         3.4 / 0.1778                   2.9 / 0.0975
#:   0.50         1.9 / 0.1941                   1.5 / 0.1010
#:   1.00         1.7 / 0.2203                   1.3 / 0.1087
#:
#: 0.25 is THE SMALLEST GUARD AT WHICH NO FITTED MODEL HAS A VARIANCE THE DATA
#: CANNOT SUPPORT -- zero datasets above a standard deviation of 5 on either
#: arm, against 20.1 percent of the empirical arm at 0.01. W1 is flat from 0.05
#: to 0.25 and better there than at 0.01 on both arms, so the choice costs
#: nothing on the study's own criterion. It is chosen on the bounded-variance
#: criterion rather than on W1 precisely so that it is not a number tuned to the
#: score it is then judged by.
#:
#: WHAT IT MEANS WHEN IT BINDS, and the paper has to say this. At 0.25 the guard
#: determines the threshold for 48 percent of empirical fits and 30 percent of
#: synthetic ones, which is to say: for about half the empirical datasets the
#: likelihood does not identify a threshold at all, and it is set at a fixed
#: fraction of a standard deviation below the smallest observation. That is a
#: SCALE-AWARE version of exactly the heuristic `fitting.LOGFIT_OFFSET` was, and
#: it should be described as one rather than presented as an estimate.
#:
#: Stage 2h sweeps it, where the roadmap had it sweeping the offset.
PROFILE_DELTA_LO_FRAC = 0.25
#: How far below min(x) the grid reaches, in the same units. Far enough that the
#: normal limit is inside the interval rather than beyond its edge.
PROFILE_DELTA_HI_FRAC = 1000.0
PROFILE_GRID_POINTS = 400


def fit_lognorm3_profile(x, w, delta_lo_frac=PROFILE_DELTA_LO_FRAC,
                         delta_hi_frac=PROFILE_DELTA_HI_FRAC,
                         npoints=PROFILE_GRID_POINTS):
    """Three-parameter lognormal by profile likelihood over the threshold.

    THE PATHOLOGY THIS EXISTS TO HANDLE. For a three-parameter lognormal with
    threshold gamma, shape sigma and scale exp(mu), the likelihood is UNBOUNDED.
    As gamma approaches the smallest observation from below, the term
    -w_1 log(x_(1) - gamma) diverges while every other term stays bounded, so
    the likelihood goes to infinity and the global maximum likelihood estimate
    DOES NOT EXIST. A naive optimizer run on the three parameters jointly will
    either drive gamma up against min(x), collapse sigma toward zero, or stop
    wherever its convergence test happened to fire. Any of the three produces a
    density with a spike at the smallest observation, which is what the
    +0.5 offset in `fitting.LOGFIT_OFFSET` was patching.

    THE TREATMENT IMPLEMENTED, which is standard. Restrict gamma to a closed
    interval bounded strictly below min(x), maximize the remaining parameters at
    each grid point -- which is closed form, so the profile is cheap -- and take
    the INTERIOR local maximum of the profile likelihood. An interior maximum is
    a real stationary point of the likelihood; the value at the upper edge of
    the grid is only the pathology reasserting itself.

    THE GUARD IS NOT A TUNING KNOB. `delta_lo_frac` sets how far below min(x)
    the grid stops, as a fraction of the weighted standard deviation. Without
    it the profile has no maximum to find. It is reported, swept in Stage 2h
    alongside the offset, and its effect is visible in `status`.

    Decision 10 in CLAUDE.md governs the other end: the threshold is NOT
    constrained to keep it away from the normal limit. As gamma goes to minus
    infinity the lognormal converges to a normal, and the author's position is
    that this is an asset of the family rather than a defect: "It can
    accommodate more datasets. That's a good thing, not a bug."

    Returns the scipy lognormal parameters plus a `status`:

        interior              an interior local maximum was found
        boundary_normal_limit the profile increases all the way to the far edge,
                              so the family is being used in its normal limit
        boundary_guard        the profile increases all the way to the guard,
                              which is the unbounded likelihood pushing against
                              it. The guard is what is reported, not a fit
    """
    x, w = np.asarray(x, float), _norm_weights(w)
    xmin = float(x.min())
    sd = float(np.sqrt(w @ (x - (x @ w)) ** 2))
    if not sd > 0:
        raise ValueError('a zero-variance dataset has no lognormal fit')

    deltas = np.geomspace(delta_lo_frac * sd, delta_hi_frac * sd, npoints)
    gammas = xmin - deltas                       # increasing delta, decreasing gamma
    rows = [(g, _lognorm3_profile_at(x, w, g)) for g in gammas]
    ok = [(g, r) for g, r in rows if r is not None]
    if not ok:
        raise ValueError('the threshold profile is empty on every grid point')
    g_arr = np.array([g for g, _ in ok])
    ll = np.array([r[2] for _, r in ok])

    # The grid runs from the guard (index 0) down toward the normal limit.
    interior = [i for i in range(1, len(ll) - 1)
                if ll[i] >= ll[i - 1] and ll[i] >= ll[i + 1]]
    if interior:
        i = int(max(interior, key=lambda j: ll[j]))
        status = 'interior'
    elif ll[-1] >= ll[0]:
        i = len(ll) - 1
        status = 'boundary_normal_limit'
    else:
        i = 0
        status = 'boundary_guard'

    gamma = float(g_arr[i])
    mu, sigma, _ = _lognorm3_profile_at(x, w, gamma)
    return dict(s=sigma, loc=gamma, scale=float(np.exp(mu)), status=status,
                threshold_delta_over_sd=float((xmin - gamma) / sd),
                profile_loglik=float(ll[i]))


def fit_gamma_mle(x, w, tol=1e-12):
    """Two-parameter gamma by weighted maximum likelihood.

    Gamma is on (0, inf) with no threshold parameter, so it has no unbounded
    likelihood to guard against, and it is arguably the more natural competitor
    to the lognormal for skewed positive data. The MLE solves

        log(a) - digamma(a) = log(mean_w x) - mean_w(log x)

    by bracketing and Brent, then `scale = mean_w x / a`. The right-hand side is
    non-negative by Jensen and is zero only for constant data.
    """
    x, w = np.asarray(x, float), _norm_weights(w)
    if np.any(x <= 0):
        raise ValueError('the gamma fit needs strictly positive data')
    m = float(x @ w)
    s = float(np.log(m) - (w @ np.log(x)))
    if s <= tol:                       # degenerate: essentially constant data
        a = 1e8
    else:
        def f(a):
            return np.log(a) - digamma(a) - s
        lo, hi = 1e-8, 1.0
        while f(hi) > 0:
            hi *= 2.0
            if hi > 1e12:
                break
        a = float(optimize.brentq(f, lo, hi, xtol=1e-12, rtol=1e-12))
    return dict(a=a, loc=0.0, scale=m / a)


# ---------------------------------------------------------------------------
# the families, as (name -> builder) so every consumer sees the same set
# ---------------------------------------------------------------------------
def make_normal(p):
    return Truncated(norm(loc=p['loc'], scale=p['scale']), label='normal',
                     params={k: p[k] for k in ('loc', 'scale')})


def make_lognorm(p):
    return Truncated(lognorm(s=p['s'], loc=p['loc'], scale=p['scale']),
                     label='lognormal',
                     params={k: p[k] for k in ('s', 'loc', 'scale')})


def make_gamma(p):
    return Truncated(gamma_dist(a=p['a'], loc=p['loc'], scale=p['scale']),
                     label='gamma', params={k: p[k] for k in ('a', 'loc', 'scale')})

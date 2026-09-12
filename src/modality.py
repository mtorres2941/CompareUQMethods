"""Modality on a principled footing: Silverman's critical bandwidth.

The metric this replaces, `customstats.estimate_maxima`, is labelled
"Mode Count" but returns

    (sum of local maxima heights - sum of local minima heights) / max height

of a Gaussian KDE at Scott's bandwidth. That is a continuous modality INDEX,
not a count: it takes values like 1.037, and across the 138 empirical datasets
it spans only 1.000 to 1.140, so nearly all of its variation is in the third
decimal place of a quantity whose name implies integers.

Silverman's (1981) test rests on a fact about the Gaussian kernel specifically:
the number of modes of a Gaussian KDE is a non-increasing function of the
bandwidth. So for each k there is a critical bandwidth

    h_k = inf { h : the KDE at bandwidth h has at most k modes }

and a large h_1 means the data resist being smoothed into one mode, which is
evidence of multimodality. h_1 divided by the data's standard deviation is a
scale-free statistic that needs no p-value to interpret, which is what decision
11 in CLAUDE.md asks for: the author's constraint is to report the statistic,
and the bootstrap significance level is provided here as a secondary
diagnostic, not as the headline.

Using a KDE-based modality measure to characterize datasets in a study that
also evaluates KDE as a fitting method is a circularity worth naming. It is
narrower than it looks: the critical bandwidth is a property of the data (the
smallest smoothing that removes structure), not of any particular bandwidth
rule, and in particular it does not depend on Scott's or Silverman's rule of
thumb, which is what the fitting method under test uses.
"""

import numpy as np

_GRID = 1024

# A local maximum below this fraction of the peak density is not a mode of the
# density, it is round-off. The FFT convolution leaves ripples of relative size
# 1e-16 to 1e-19 in the far tails where the true density has underflowed to
# zero, and without a floor the mode count reads them as structure: an
# n = 400 standard normal sample came out with 116 modes at h = 0.28 sd, of
# which one was real. The threshold sits eight orders of magnitude above the
# round-off it removes and eight below the smallest genuine secondary bump seen
# in the corpus (relative height 1e-2 to 1e-3).
MODE_REL_TOL = 1e-8


class _BinnedKDE:
    """Gaussian KDE evaluated by linear binning and FFT convolution.

    The critical-bandwidth search evaluates the KDE at 30 or more bandwidths
    per dataset, and the bootstrap does that again for every resample. Direct
    evaluation costs O(grid * n) each time, which is 1.5 s for a single
    n = 10,000 dataset and hopeless for a corpus of 10,000. Binning once and
    convolving makes every later evaluation O(grid log grid), independent of n.

    The grid is fixed once, wide enough for the largest bandwidth the search
    will reach, so all evaluations share the same bins. The FFT is zero-padded
    to twice the grid length so the convolution does not wrap around.
    """

    __slots__ = ('grid', 'delta', '_fft', '_m', '_freq')

    def __init__(self, x, weights=None, pad_sd=6.0, grid_n=_GRID):
        x = np.asarray(x, float)
        w = (np.ones(len(x)) / len(x) if weights is None
             else np.asarray(weights, float) / np.sum(weights))
        sd = float(np.std(x))
        span = x.max() - x.min()
        pad = pad_sd * (sd if sd > 0 else 1.0) + 0.05 * (span if span > 0 else 1.0)
        self.grid = np.linspace(x.min() - pad, x.max() + pad, grid_n)
        self.delta = float(self.grid[1] - self.grid[0])
        # linear binning: each point splits its weight between its two
        # neighbouring grid nodes, which is second-order accurate where simple
        # histogram binning is only first-order
        pos = (x - self.grid[0]) / self.delta
        i0 = np.clip(np.floor(pos).astype(int), 0, grid_n - 2)
        frac = pos - i0
        counts = np.zeros(grid_n)
        np.add.at(counts, i0, w * (1.0 - frac))
        np.add.at(counts, i0 + 1, w * frac)
        self._m = 2 * grid_n
        self._fft = np.fft.rfft(counts, self._m)
        self._freq = np.fft.rfftfreq(self._m, d=self.delta)

    def density(self, h):
        """KDE on the grid. The Gaussian kernel's Fourier transform is itself
        a Gaussian, exp(-2 pi^2 f^2 h^2), so no kernel array is needed."""
        k = np.exp(-2.0 * (np.pi * self._freq * h) ** 2)
        y = np.fft.irfft(self._fft * k, self._m)[:len(self.grid)]
        return np.maximum(y, 0.0) / self.delta

    def n_modes(self, h, rel_tol=MODE_REL_TOL):
        return _count_peaks(self.density(h), rel_tol)


def _kde_on_grid(x, h, weights=None, grid=None, pad=4.0):
    """Direct evaluation. Kept for testing _BinnedKDE against it."""
    x = np.asarray(x, float)
    if grid is None:
        grid = np.linspace(x.min() - pad * h, x.max() + pad * h, _GRID)
    w = (np.ones_like(x) / len(x) if weights is None
         else np.asarray(weights, float) / np.sum(weights))
    z = (grid[:, None] - x[None, :]) / h
    return grid, (np.exp(-0.5 * z ** 2) @ w) / (h * np.sqrt(2 * np.pi))


def _count_peaks(y, rel_tol=MODE_REL_TOL):
    peak = float(np.max(y))
    if not np.isfinite(peak) or peak <= 0:
        return 0
    d = np.diff(y)
    is_peak = (d[:-1] > 0) & (d[1:] <= 0)
    return int(np.sum(is_peak & (y[1:-1] > rel_tol * peak)))


def count_modes(x, h, weights=None, grid=None):
    """Number of local maxima of the Gaussian KDE at bandwidth h."""
    return _BinnedKDE(x, weights).n_modes(h)


def critical_bandwidth(x, k=1, weights=None, rel_lo=1e-4, rel_hi=4.0, iters=40):
    """Smallest bandwidth at which the KDE has at most k modes.

    Returned in units of the data's standard deviation, so it is comparable
    across datasets of any scale. Monotonicity of the mode count in h makes a
    bisection exact up to the grid resolution.
    """
    x = np.asarray(x, float)
    sd = float(np.std(x))
    if len(x) < 3 or sd <= 0 or not np.isfinite(sd):
        return np.nan
    kde = _BinnedKDE(x, weights)
    lo, hi = rel_lo * sd, rel_hi * sd
    while kde.n_modes(hi) > k and hi < 1e4 * sd:
        hi *= 2.0
    if kde.n_modes(hi) > k:
        return np.nan
    if kde.n_modes(lo) <= k:
        return lo / sd
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if kde.n_modes(mid) > k:
            lo = mid
        else:
            hi = mid
    return float(hi / sd)


def silverman_pvalue(x, k=1, rng=None, nboot=100, weights=None):
    """Bootstrap significance for 'at most k modes'. A DIAGNOSTIC, not the
    headline statistic; see the module docstring and decision 11.

    Resamples from the KDE smoothed at the critical bandwidth, with Silverman's
    variance correction so the resample has the same variance as the data, and
    reports the fraction of resamples whose own critical bandwidth exceeds the
    observed one.
    """
    x = np.asarray(x, float)
    rng = rng if rng is not None else np.random.default_rng(0)
    h = critical_bandwidth(x, k, weights)
    if not np.isfinite(h):
        return np.nan
    hs = h * float(np.std(x))
    s2 = float(np.var(x))
    xbar = float(np.mean(x))
    bigger = 0
    for _ in range(nboot):
        idx = rng.integers(0, len(x), len(x))
        y = xbar + (x[idx] - xbar + hs * rng.normal(size=len(x))) / np.sqrt(
            1.0 + hs ** 2 / s2)
        hb = critical_bandwidth(y, k)
        if np.isfinite(hb) and hb > h:
            bigger += 1
    return bigger / nboot


def n_modes_silverman(x, weights=None, rng=None, nboot=100, alpha=0.05, kmax=5):
    """Smallest k that the bootstrap does not reject, as an integer count."""
    rng = rng if rng is not None else np.random.default_rng(0)
    for k in range(1, kmax + 1):
        p = silverman_pvalue(x, k, rng, nboot, weights)
        if not np.isfinite(p) or p > alpha:
            return k
    return kmax + 1


# --------------------------------------------------------------------------
# a small 1-D Gaussian mixture, used only to give the 138 empirical datasets a
# component structure so their pairwise overlap can be compared with the
# synthetic corpus on the same footing
# --------------------------------------------------------------------------
def gmm_em_1d(x, k, rng, iters=300, tol=1e-9, reg=1e-6, restarts=4):
    x = np.asarray(x, float)
    n = len(x)
    best = None
    for _ in range(restarts):
        mu = rng.choice(x, size=k, replace=k > n)
        var = np.full(k, np.var(x) / k + reg)
        pi = np.ones(k) / k
        ll_old = -np.inf
        for _ in range(iters):
            r = pi * np.exp(-0.5 * (x[:, None] - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
            tot = r.sum(1, keepdims=True)
            tot = np.where(tot > 0, tot, 1e-300)
            ll = float(np.sum(np.log(tot)))
            r = r / tot
            nk = r.sum(0) + 1e-12
            pi = nk / n
            mu = (r * x[:, None]).sum(0) / nk
            var = (r * (x[:, None] - mu) ** 2).sum(0) / nk + reg
            if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
                break
            ll_old = ll
        if best is None or ll > best[0]:
            best = (ll, pi.copy(), mu.copy(), var.copy())
    return best


def fit_mixture_bic(x, rng, kmax=5):
    """BIC-selected Gaussian mixture. Returns (k, pi, mu, sd)."""
    x = np.asarray(x, float)
    n = len(x)
    best = None
    for k in range(1, min(kmax, max(1, n // 3)) + 1):
        got = gmm_em_1d(x, k, rng)
        if got is None:
            continue
        ll, pi, mu, var = got
        bic = -2 * ll + (3 * k - 1) * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, k, pi, mu, np.sqrt(var))
    if best is None:
        return 1, np.ones(1), np.array([np.mean(x)]), np.array([np.std(x)])
    return best[1], best[2], best[3], best[4]


# ---------------------------------------------------------------------------
# Modes you can see, as distinct from modes a test can detect.
# ---------------------------------------------------------------------------
def n_modes_visible(x, prominence=0.05, grid_n=512):
    """Count local maxima of a default-bandwidth KDE, by prominence.

    This is deliberately NOT Silverman's critical-bandwidth test, and the two
    answer different questions. Silverman asks whether the data are multimodal
    at ANY bandwidth, and is sensitive to fine structure that never appears in
    a plot. This asks how many humps a reader sees in the density they would
    actually be shown.

    They disagree, and the disagreement is the point. On the 2026-08 empirical
    arm, Silverman calls 52 percent of the datasets multimodal while 94.3
    percent have exactly one VISIBLE mode: real ECC categories are single
    right-skewed humps carrying real but small-scale structure. A corpus tuned
    to match the Silverman distribution can therefore be built out of clearly
    separated humps and still report a good match, which is exactly what
    happened through Stage 2a-2: matched on Silverman, 58.9 percent visible-
    unimodal against an empirical 94.3.

    `prominence` is the height a peak must clear above the higher of the two
    valleys flanking it, as a fraction of the tallest peak. 0.05 keeps the
    shoulders a reader would call a second hump and drops ripple.
    """
    from scipy.stats import gaussian_kde
    from scipy.signal import argrelextrema

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.std(x) <= 0:
        return 1
    try:
        kde = gaussian_kde(x)
    except Exception:
        return 1
    grid = np.linspace(x.min(), x.max(), grid_n)
    y = kde(grid)
    idx = argrelextrema(y, np.greater)[0]
    if len(idx) == 0:
        return 1
    peak = y.max()
    if peak <= 0:
        return 1
    kept = 0
    for i in idx:
        left = y[:i].min() if i > 0 else y[i]
        right = y[i + 1:].min() if i < len(y) - 1 else y[i]
        if (y[i] - max(left, right)) / peak >= prominence:
            kept += 1
    return max(kept, 1)

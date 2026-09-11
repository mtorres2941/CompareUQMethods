"""The synthetic ECC parent: a truncated, renormalized finite mixture.

Stage 2c scores fitted models against the distribution a synthetic dataset was
actually drawn from, so that distribution has to be an object, not a procedure.
This module is that object.

A dataset is produced in four steps, and every one of them is closed form:

  1. a mixture  sum_k pi_k f_k  of components from src/components.py, each with
     a target mean, standard deviation, skewness and excess kurtosis;
  2. truncation to [lo, hi], where lo = max(Q1 - 3 * IQR, 0) and
     hi = Q3 + 3 * IQR are quantiles of the POPULATION mixture, not of a
     realized sample, and the truncated mixture is renormalized;
  3. n draws by inverse CDF, with the per-component counts drawn from the
     TRUNCATED component proportions, which is exactly sampling the truncated
     mixture;
  4. division by the realized unweighted sample mean.

Step 4 is the one place a realized quantity enters, and it has to: decision 6
(Stage 1 amendment A3) fixes normalization on the unweighted sample mean in
both the synthetic and the empirical path, because a practitioner holding a set
of EPDs can compute that and cannot compute anything else. The divisor is
recorded per dataset, so the parent is still exactly specified: conditional on
the recorded normalizer c, the parent of the normalized values is the truncated
mixture evaluated at c * x. Conditioning on the sample mean induces an O(1/n)
dependence between the values, which `audits/stage2a/a8_parent_verification.py`
measures rather than assumes.

What this replaces, and why (Stage 2a Part 1):

  - the truncation loop, which computed lo and hi from one realized draw and
    then refilled from a randomly chosen component until the length came back
    to n. That made the parent a truncated mixture with conditional redrawing,
    reachable only by simulation. It also kept `data[:n]` after filtering,
    which preferentially discarded the LATER components in concatenation order;
  - the power transform `data ** rng.uniform(0.9, 4.0)`, a tuning knob for
    skew;
  - the reflection `np.max(data) - data + np.min(data)`, applied 25 percent of
    the time, the only sample-dependent transform in the generator.

Skewness now comes from the component families, where it is a specified target.
"""

import numpy as np
from scipy import optimize

import components as C

TRUNC_IQR_MULT = 3.0
_EPS = 1e-12


# --------------------------------------------------------------------------
class MixtureParent:
    """A truncated, renormalized mixture with both weighting schemes attached.

    Attributes
    ----------
    comps            frozen component distributions, in order
    pi               sampling weights, which drive how many points fall in each
                     mode
    market           mode-level market shares, drawn independently of pi
    lo, hi           population truncation bounds
    pi_trunc         component proportions of the TRUNCATED mixture,
                     proportional to pi_k * P_k([lo, hi]); these are what a
                     draw actually realizes
    market_effective the mode weights of the market-weighted parent at the
                     configured coupling: (1 - c) * pi_trunc + c * market
    normalizer       the realized unweighted sample mean, set by `sample`
    """

    __slots__ = ('comps', 'pi', 'market', 'lo', 'hi', 'pi_trunc',
                 'market_effective', 'coupling', 'normalizer', '_mass')

    def __init__(self, comps, pi, market, lo, hi, coupling):
        self.comps = tuple(comps)
        self.pi = np.asarray(pi, float)
        self.market = np.asarray(market, float)
        self.lo, self.hi = float(lo), float(hi)
        self.coupling = float(coupling)
        self._mass = np.array([float(d.cdf(self.hi) - d.cdf(self.lo))
                               for d in self.comps])
        w = self.pi * self._mass
        self.pi_trunc = w / w.sum()
        self.market_effective = ((1.0 - self.coupling) * self.pi_trunc
                                 + self.coupling * self.market)
        self.market_effective = self.market_effective / self.market_effective.sum()
        self.normalizer = 1.0

    # ---------------------------------------------------------------- parent
    def _weights(self, scheme):
        if scheme == 'uniform':
            return self.pi_trunc
        if scheme == 'market':
            return self.market_effective
        raise ValueError(f'scheme must be uniform or market, not {scheme!r}')

    def cdf(self, x, scheme='uniform'):
        """CDF of the parent, in NORMALIZED units (divided by `normalizer`)."""
        x = np.asarray(x, float) * self.normalizer
        w = self._weights(scheme)
        out = np.zeros(np.shape(x), float)
        for wk, d, mk in zip(w, self.comps, self._mass):
            lo_k = float(d.cdf(self.lo))
            out += wk * np.clip((np.asarray(d.cdf(x), float) - lo_k) / mk, 0.0, 1.0)
        return np.clip(out, 0.0, 1.0)

    def pdf(self, x, scheme='uniform'):
        x = np.asarray(x, float) * self.normalizer
        w = self._weights(scheme)
        out = np.zeros(np.shape(x), float)
        inside = (x > self.lo) & (x < self.hi)
        for wk, d, mk in zip(w, self.comps, self._mass):
            out += wk * np.asarray(d.pdf(x), float) / mk
        return np.where(inside, out, 0.0) * self.normalizer

    def ppf(self, q, scheme='uniform'):
        """Quantile function, by bisection on the closed-form CDF."""
        q = np.atleast_1d(np.asarray(q, float))
        lo = np.full(q.shape, self.lo / self.normalizer)
        hi = np.full(q.shape, self.hi / self.normalizer)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            go_up = self.cdf(mid, scheme) < q
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)
            if np.max(hi - lo) < 1e-14 * max(1.0, abs(self.hi)):
                break
        return 0.5 * (lo + hi)

    def truncated_mass(self):
        """Probability mass the truncation step removes, per component and
        overall. Stage 2a Part 1 asks for this to be stated."""
        return float(1.0 - np.sum(self.pi * self._mass))

    # -------------------------------------------------------------- sampling
    def sample(self, n, rng, normalize=True):
        """Draw n values by inverse CDF, exactly, with no rejection loop.

        Sampling a truncated mixture is: choose component k with probability
        proportional to pi_k * P_k([lo, hi]), then draw from f_k truncated to
        [lo, hi]. Multinomial counts give the same joint law as n independent
        choices, and they also tell us which mode each point came from, which
        Part 3 needs in order to attach market share at the mode level.

        Returns (values, mode_index), values sorted by nothing in particular
        but shuffled so mode membership carries no positional information.
        """
        n = int(n)
        counts = rng.multinomial(n, self.pi_trunc)
        vals, modes = [], []
        for k, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            d = self.comps[k]
            a, b = float(d.cdf(self.lo)), float(d.cdf(self.hi))
            u = a + (b - a) * rng.uniform(_EPS, 1.0 - _EPS, size=int(cnt))
            vals.append(np.asarray(d.ppf(u), float))
            modes.append(np.full(int(cnt), k, dtype=np.int16))
        x = np.concatenate(vals)
        m = np.concatenate(modes)
        order = rng.permutation(n)
        x, m = x[order], m[order]
        if normalize:
            self.normalizer = float(np.mean(x))
            x = x / self.normalizer
        return x, m


# --------------------------------------------------------------------------
# overlap, after Maitra and Melnykov (2010) generalized to one dimension
# --------------------------------------------------------------------------
def pairwise_overlap(comps, pi, grid_n=4001):
    """Matrix of pairwise overlaps omega_ij = omega_{j|i} + omega_{i|j}.

    Maitra and Melnykov define omega_{j|i} as the probability that a draw from
    component i is assigned to component j by the Bayes rule,

        omega_{j|i} = Pr[ pi_i f_i(X) < pi_j f_j(X) | X ~ f_i ],

    the misclassification probability. Their closed forms are Gaussian-only,
    but the DEFINITION is distribution free, and in one dimension the integral
    is a quadrature over a common grid. That matters here because the
    components are Johnson SU, beta, beta-prime and lognormal, not Gaussians,
    so the Gaussian formulas would not apply anyway.

    Overlap is the right generation parameter for this study because the thing
    that varies between real material categories is how far apart the product
    groups are, and the current generator has no knob for it: locations on
    (5, 20) with scales on (0.2, 1.5) place components tens of standard
    deviations apart, giving well-separated clusters rather than the partially
    merged shoulders real ECC data shows.
    """
    k = len(comps)
    om = np.zeros((k, k))
    if k < 2:
        return om
    lo = min(float(d.ppf(1e-6)) for d in comps)
    hi = max(float(d.ppf(1 - 1e-6)) for d in comps)
    pad = 0.05 * (hi - lo)
    x = np.linspace(lo - pad, hi + pad, grid_n)
    dens = np.array([np.asarray(d.pdf(x), float) for d in comps])
    dens = np.where(np.isfinite(dens), dens, 0.0)
    wd = dens * np.asarray(pi, float)[:, None]
    for i in range(k):
        for j in range(i + 1, k):
            # Ties are split rather than dropped. Maitra and Melnykov write the
            # definition with a strict inequality because for two distinct
            # Gaussians the tie set has probability zero, but two components
            # that coincide tie everywhere and the strict form then reports
            # zero overlap for the most overlapping case there is. Splitting
            # ties makes the function continuous as components merge, which the
            # bisection in solve_spread_for_overlap relies on.
            tie = 0.5 * (wd[i] == wd[j])
            om_ji = np.trapezoid(np.where(wd[i] < wd[j], 1.0, tie) * dens[i], x)
            om_ij = np.trapezoid(np.where(wd[j] < wd[i], 1.0, tie) * dens[j], x)
            om[i, j] = om[j, i] = float(np.clip(om_ji + om_ij, 0.0, 2.0))
    return om


def average_overlap(comps, pi, grid_n=4001):
    k = len(comps)
    if k < 2:
        return 0.0
    om = pairwise_overlap(comps, pi, grid_n)
    iu = np.triu_indices(k, 1)
    return float(np.mean(om[iu]))


def solve_spread_for_overlap(build, target, lo=1e-3, hi=1e3, tol=1e-3, grid_n=2001):
    """Find the location spread giving the requested average overlap.

    `build(c)` returns the component list at spread multiplier c, with c small
    meaning the components sit on top of one another (overlap near its maximum)
    and c large meaning they are far apart (overlap near zero). Average overlap
    is monotone decreasing in c, so a bisection is enough.

    This is Maitra and Melnykov's step 3, with the roles of scale and location
    exchanged: they hold locations fixed and scale the covariances, which for
    non-Gaussian components would change the component shapes' relationship to
    their own moment targets. Moving the locations instead leaves every
    component's standardized shape, skewness and kurtosis exactly as specified.

    Returns (components, realized_overlap, status).
    """
    def f(c):
        comps, pi = build(c)
        return average_overlap(comps, pi, grid_n)

    f_lo, f_hi = f(lo), f(hi)
    if target > f_lo:
        comps, pi = build(lo)
        return comps, f_lo, 'clipped_max_overlap'
    if target < f_hi:
        comps, pi = build(hi)
        return comps, f_hi, 'clipped_min_overlap'
    a, b = lo, hi
    for _ in range(60):
        mid = np.sqrt(a * b)
        fm = f(mid)
        if abs(fm - target) < tol * max(target, 1e-4):
            comps, pi = build(mid)
            return comps, fm, 'ok'
        if fm > target:
            a = mid
        else:
            b = mid
    comps, pi = build(np.sqrt(a * b))
    return comps, f(np.sqrt(a * b)), 'tolerance_not_met'


# --------------------------------------------------------------------------
def population_truncation_bounds(comps, pi, mult=TRUNC_IQR_MULT):
    """[max(Q1 - mult * IQR, 0), Q3 + mult * IQR] of the POPULATION mixture.

    The rule is the one the old generator used; what changes is that Q1 and Q3
    are quantiles of the mixture rather than of one realized draw, so the
    bounds are a property of the parent and the same for every sample from it.
    """
    def mix_cdf(x):
        return float(np.sum([w * float(d.cdf(x)) for w, d in zip(pi, comps)]))

    lo_b = min(float(d.ppf(1e-9)) for d in comps)
    hi_b = max(float(d.ppf(1 - 1e-9)) for d in comps)
    span = hi_b - lo_b
    lo_b, hi_b = lo_b - 0.5 * span, hi_b + 0.5 * span

    def q(p):
        return optimize.brentq(lambda x: mix_cdf(x) - p, lo_b, hi_b,
                               xtol=1e-12 * max(1.0, abs(hi_b)), rtol=1e-13)

    q1, q3 = q(0.25), q(0.75)
    iqr = q3 - q1
    return max(q1 - mult * iqr, 0.0), q3 + mult * iqr

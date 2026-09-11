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

# A component holding less than this share of its own probability inside the
# truncation bounds is dropped from the parent. See MixtureParent.__init__.
MIN_COMPONENT_MASS = 1e-9


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
                 'market_effective', 'coupling', 'normalizer', '_mass',
                 'n_components_dropped')

    def __init__(self, comps, pi, market, lo, hi, coupling):
        comps = list(comps)
        pi = np.asarray(pi, float)
        market = np.asarray(market, float)
        self.lo, self.hi = float(lo), float(hi)
        self.coupling = float(coupling)

        # A component can fall almost entirely outside [lo, hi]: the truncation
        # bounds come from the MIXTURE's quartiles, so a far-flung component
        # contributes essentially nothing to the truncated mixture. Such a
        # component must be dropped, not kept with a near-zero mass. Keeping it
        # divides by that mass in cdf() and in sample(), which turns a
        # rounding-level number into the dominant term: two parents in a
        # 1,600-draw check came back with a population mean of 1e-12 because
        # one component's clipped CDF saturated at 1 for every x.
        #
        # The mass genuinely is not there, so the honest treatment is to drop
        # the component and renormalize both weight vectors over the
        # survivors. How many were dropped is recorded rather than hidden.
        mass = np.array([float(d.cdf(self.hi)) - float(d.cdf(self.lo)) for d in comps])
        mass = np.where(np.isfinite(mass), np.clip(mass, 0.0, 1.0), 0.0)
        contrib = pi * mass
        total = contrib.sum()
        keep = (mass > MIN_COMPONENT_MASS) & (contrib > MIN_COMPONENT_MASS * max(total, _EPS))
        if not keep.any():
            keep = np.zeros(len(comps), bool)
            keep[int(np.argmax(contrib))] = True
        self.n_components_dropped = int((~keep).sum())
        comps = [d for d, k in zip(comps, keep) if k]
        pi = pi[keep] / pi[keep].sum()
        market = market[keep] / market[keep].sum()
        mass = mass[keep]

        self.comps = tuple(comps)
        self.pi = pi
        self.market = market
        self._mass = mass
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
def _log_ratio(di, dj, pi_i, pi_j):
    """g(x) = log(pi_j f_j(x)) - log(pi_i f_i(x)). Positive where a draw is
    assigned to j."""
    def g(x):
        x = np.asarray(x, float)
        with np.errstate(divide='ignore', invalid='ignore'):
            fi = np.asarray(di.pdf(x), float)
            fj = np.asarray(dj.pdf(x), float)
            a = np.log(np.where(fj > 0, fj, 1e-300)) + np.log(pi_j)
            b = np.log(np.where(fi > 0, fi, 1e-300)) + np.log(pi_i)
        return a - b
    return g


def _mass_where_positive(g, d, lo, hi, grid_n):
    """Probability under d of the region where g > 0, from d's CDF.

    Evaluating the misclassification probability as a sum of CDF differences
    over root-found crossings, rather than as a quadrature of the density,
    matters here. The targeted overlaps run down to 1e-4, and at that size the
    region is a thin tail: a trapezoid rule on a grid that has to span every
    component at once puts only a handful of nodes inside it, and the resulting
    value is quantized rather than small. That quantization is not monotone in
    the component spread, which broke the bisection in
    solve_spread_for_overlap - 23 percent of multi-component datasets came back
    'tolerance_not_met' before this was changed. CDF differences are exact
    given the crossings, and the crossings are found to machine precision.
    """
    x = np.linspace(lo, hi, grid_n)
    v = g(x)
    v = np.where(np.isfinite(v), v, -np.inf)
    # Two components that coincide tie everywhere. The strict inequality in the
    # definition then reports zero overlap for the most overlapping case there
    # is, so ties are split. Splitting keeps the function continuous as
    # components merge, which the bisection in solve_spread_for_overlap needs.
    if np.all(np.abs(v) < 1e-12):
        return 0.5
    sign = v > 0
    total = 0.0
    # locate every sign change, refine it, and add the CDF mass between
    # consecutive crossings wherever g is positive
    changes = np.flatnonzero(sign[:-1] != sign[1:])
    edges = [lo]
    for c in changes:
        a, b = x[c], x[c + 1]
        try:
            edges.append(optimize.brentq(lambda t: g(np.array([t]))[0], a, b,
                                         xtol=1e-13 * max(1.0, abs(b)), rtol=1e-14))
        except Exception:
            edges.append(0.5 * (a + b))
    edges.append(hi)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        mid = 0.5 * (a + b)
        if g(np.array([mid]))[0] > 0:
            total += float(d.cdf(b)) - float(d.cdf(a))
    return float(np.clip(total, 0.0, 1.0))


def pairwise_overlap(comps, pi, grid_n=2001):
    """Matrix of pairwise overlaps omega_ij = omega_{j|i} + omega_{i|j}.

    Maitra and Melnykov define omega_{j|i} as the probability that a draw from
    component i is assigned to component j by the Bayes rule,

        omega_{j|i} = Pr[ pi_i f_i(X) < pi_j f_j(X) | X ~ f_i ],

    the misclassification probability. Their closed forms are Gaussian-only,
    but the DEFINITION is distribution free, and in one dimension the assigned
    region is a handful of intervals whose endpoints can be found exactly. That
    matters here because the components are Johnson SU, beta, beta-prime and
    lognormal, not Gaussians, so the Gaussian formulas would not apply anyway.

    Overlap is the right generation parameter for this study because the thing
    that varies between real material categories is how far apart the product
    groups are, and the old generator had no knob for it: locations on (5, 20)
    with scales on (0.2, 1.5) placed components a measured 6.5 pooled standard
    deviations apart on average, giving well-separated clusters rather than the
    partially merged shoulders real ECC data shows.
    """
    k = len(comps)
    om = np.zeros((k, k))
    if k < 2:
        return om
    for i in range(k):
        for j in range(i + 1, k):
            di, dj = comps[i], comps[j]
            lo = min(float(di.ppf(1e-10)), float(dj.ppf(1e-10)))
            hi = max(float(di.ppf(1 - 1e-10)), float(dj.ppf(1 - 1e-10)))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            pad = 0.02 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            g_ij = _log_ratio(di, dj, pi[i], pi[j])
            om_ji = _mass_where_positive(g_ij, di, lo, hi, grid_n)

            def g_ji(x, g_ij=g_ij):
                return -g_ij(x)
            om_ij = _mass_where_positive(g_ji, dj, lo, hi, grid_n)
            om[i, j] = om[j, i] = float(np.clip(om_ji + om_ij, 0.0, 2.0))
    return om


def average_overlap(comps, pi, grid_n=2001):
    k = len(comps)
    if k < 2:
        return 0.0
    om = pairwise_overlap(comps, pi, grid_n)
    iu = np.triu_indices(k, 1)
    return float(np.mean(om[iu]))


def solve_spread_for_overlap(build, target, lo=1e-4, hi=1e4, tol=1e-3, grid_n=1501):
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

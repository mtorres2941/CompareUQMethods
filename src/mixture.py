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

    def truncated_mean(self, grid_n=4001):
        """Population mean of the truncated mixture, in RAW units.

        Computed as lo + integral of (1 - F) over [lo, hi], which needs only
        the closed-form mixture CDF and no quantile inversion. Shifting every
        component by s shifts this by exactly s, which is what lets
        `generator.draw_parent` solve for the shift that places the lower
        support bound at a requested fraction of the mean.
        """
        return self.truncated_moments(grid_n)[0]

    def truncated_moments(self, grid_n=4001):
        """(mean, sd) of the truncated mixture, in RAW units.

        Both from the closed-form CDF alone, with no quantile inversion, and
        both computed about Y = X - lo rather than about the origin:

            E[Y]   = integral of (1 - F) over [0, hi - lo]
            E[Y^2] = integral of 2y (1 - F)
            var    = E[Y^2] - E[Y]^2

        Working in Y matters. The generator places a mixture by shifting it,
        and the shift can be many orders of magnitude larger than the spread.
        Computed about the origin, the variance is then the difference of two
        numbers of order lo ** 2, and cancels catastrophically: the solve for a
        target coefficient of variation was coming out 59 percent off at the
        median because of it, and low targets were driving the shift to 1e14
        times the span, which left the normalized values identical to float
        precision and the datasets rejected as degenerate.
        """
        # The nodes are the COMPONENTS' OWN QUANTILES, not a uniform grid.
        #
        # A uniform grid over [0, hi - lo] silently fails when the truncation
        # bounds are wide relative to the body, because the body then gets
        # almost no nodes. With hi - lo about 4,800 and the mass near 1, a
        # 4,001-point uniform grid has a spacing of 1.2 and lands roughly one
        # node on the entire distribution: the survival function reads as zero
        # everywhere, e1 and e2 collapse, and the variance clamps to exactly 0.
        # That reported sd = 0 for every parent, which made the
        # coefficient-of-variation solve believe no shift could ever reach its
        # target and clip 93 percent of them.
        #
        # The additive truncation rule never triggered it, because its upper
        # bound sits a few interquartile ranges from the body. Any wider rule
        # does, so this was a live trap for the multiplicative rule and for any
        # Stage 2h sweep that raises trunc_iqr_mult.
        #
        # Quantile nodes put resolution where the probability is, by
        # construction, and cost one ppf evaluation per component.
        probs = np.linspace(0.0, 1.0, max(grid_n // max(len(self.comps), 1), 64))
        nodes = [np.asarray(d.ppf(np.clip(probs, 1e-12, 1 - 1e-12)), float)
                 for d in self.comps]
        x = np.unique(np.concatenate(nodes + [np.array([self.lo, self.hi])]))
        x = x[(x >= self.lo) & (x <= self.hi)]
        if len(x) < 8 or not np.all(np.isfinite(x)):
            x = np.linspace(self.lo, self.hi, grid_n)
        y = x - self.lo
        F = np.zeros_like(x)
        for wk, d, mk in zip(self.pi_trunc, self.comps, self._mass):
            lo_k = float(d.cdf(self.lo))
            F += wk * np.clip((np.asarray(d.cdf(x), float) - lo_k) / mk, 0.0, 1.0)
        S = 1.0 - F
        e1 = np.trapezoid(S, y)
        e2 = np.trapezoid(2.0 * y * S, y)
        var = max(e2 - e1 ** 2, 0.0)
        return float(self.lo + e1), float(np.sqrt(var))

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
def _log_assignment_margin(x, di, dj, pi_i, pi_j):
    """log(pi_j f_j(x)) - log(pi_i f_i(x)). Positive where the Bayes rule
    assigns x to component j rather than to component i."""
    with np.errstate(divide='ignore', invalid='ignore'):
        fi = np.asarray(di.pdf(x), float)
        fj = np.asarray(dj.pdf(x), float)
        v = (np.log(np.where(fj > 0, fj, 1e-300)) + np.log(pi_j)
             - np.log(np.where(fi > 0, fi, 1e-300)) - np.log(pi_i))
    return np.where(np.isfinite(v), v, -np.inf)


def _assigned_to_other(x, di, dj, pi_i, pi_j):
    """True where the Bayes rule assigns x to component j rather than i."""
    return _log_assignment_margin(x, di, dj, pi_i, pi_j) > 0


def _pair_overlap(di, dj, pi_i, pi_j, n_u=120, refine=20):
    """omega_ij = omega_{j|i} + omega_{i|j} for one pair of components.

    Maitra and Melnykov define omega_{j|i} as the probability that a draw from
    component i is assigned to component j by the Bayes rule, the
    misclassification probability. Their closed forms are Gaussian-only, but
    the DEFINITION is distribution free; in one dimension the assigned region
    is a handful of intervals, and the answer is the probability each component
    puts on them.

    Both directions are computed together, because they share everything that
    costs anything. The assignment margin for j given i is the negative of the
    margin for i given j, so the crossings are the SAME points; only which
    intervals count, and which component's CDF measures them, differ. Computing
    them once halves the number of scipy density evaluations, which is what
    this function's runtime is made of.

    The scan grid is the UNION of both components' quantiles, not a uniform
    grid in x and not a dense quantile grid of one component. Each choice fixes
    a specific failure:

      - a uniform x-grid has to span both components at once, so a narrow
        component gets very few nodes. On one pair that put the answer 0.070 in
        absolute probability away from the truth at 2,001 nodes while reporting
        the correct NUMBER of crossings: the crossings were being located
        inside cells across which f_i's CDF moved a great deal.
      - a dense quantile grid of one component fixes the accuracy and costs too
        much: scipy inverts numerically for the Johnson SU, beta and beta-prime
        families, and 2,001 ppf evaluations per pair per bisection step made
        generation slower than the version it replaced.

    An earlier version located each crossing with scipy.optimize.brentq and
    spent 65 percent of the entire generation run inside it.

    The defaults come from measurement against a 500,001-point brute-force
    count of the definition: 120 quantiles per component with 20 bisection
    refinements agrees to 7.5e-07. More of either buys no accuracy (400 with 44
    refinements is also 7.5e-07, at twice the cost); fewer refinements cost
    accuracy quickly (8 gives 9.9e-06, none gives 1.6e-03).
    """
    eps = 1e-9
    u = np.linspace(eps, 1.0 - eps, n_u)
    x = np.unique(np.concatenate([np.asarray(di.ppf(u), float),
                                  np.asarray(dj.ppf(u), float)]))
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0

    margin = _log_assignment_margin(x, di, dj, pi_i, pi_j)
    # Two components that coincide tie everywhere. The strict inequality in the
    # definition then reports zero overlap for the most overlapping case there
    # is, so an exact tie is split. Splitting keeps overlap continuous as
    # components merge, which the bisection in solve_spread_for_overlap needs.
    if np.all(np.abs(margin) < 1e-12):
        return 1.0
    to_j = margin > 0
    idx = np.flatnonzero(to_j[:-1] != to_j[1:])
    if len(idx) == 0:
        # One component dominates everywhere. Every draw from the loser is then
        # misassigned and none from the winner is, so the two directions sum to
        # exactly 1 whichever component wins. Returning the winner-dependent
        # 1.0-or-0.0 here, as the single-direction version correctly did, is
        # wrong once both directions are added together.
        return 1.0

    xa, xb = x[idx].copy(), x[idx + 1].copy()
    left = to_j[idx]
    for _ in range(refine):
        xm = 0.5 * (xa + xb)
        same_as_left = _assigned_to_other(xm, di, dj, pi_i, pi_j) == left
        xa = np.where(same_as_left, xm, xa)
        xb = np.where(same_as_left, xb, xm)
    edges = np.concatenate(([x[0]], 0.5 * (xa + xb), [x[-1]]))

    assigned_to_j = np.empty(len(edges) - 1, bool)
    assigned_to_j[0] = bool(to_j[0])
    assigned_to_j[1:] = ~to_j[idx]

    total = 0.0
    for d, want_j in ((di, True), (dj, False)):
        F = np.asarray(d.cdf(edges), float)
        F[0], F[-1] = 0.0, 1.0      # the scan grid spans both supports
        seg = assigned_to_j if want_j else ~assigned_to_j
        total += float(np.clip(np.sum(np.diff(F)[seg]), 0.0, 1.0))
    return float(np.clip(total, 0.0, 2.0))


def pairwise_overlap(comps, pi, grid_n=120):
    """Matrix of pairwise overlaps omega_ij = omega_{j|i} + omega_{i|j}.

    Maitra and Melnykov define omega_{j|i} as the probability that a draw from
    component i is assigned to component j by the Bayes rule, the
    misclassification probability. Their closed forms are Gaussian-only, but
    the DEFINITION is distribution free, and in one dimension it is a measure
    of a set of quantiles. That matters here because the components are Johnson
    SU, beta, beta-prime and lognormal, not Gaussians, so the Gaussian formulas
    would not apply anyway.

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
            om[i, j] = om[j, i] = _pair_overlap(comps[i], comps[j], pi[i], pi[j],
                                                grid_n)
    return om


def average_overlap(comps, pi, grid_n=120):
    k = len(comps)
    if k < 2:
        return 0.0
    om = pairwise_overlap(comps, pi, grid_n)
    iu = np.triu_indices(k, 1)
    return float(np.mean(om[iu]))

def min_adjacent_overlap(comps, pi, grid_n=120):
    """Smallest overlap between NEIGHBOURING components, by location.

    The quantity to control, in place of the average, once k > 2.

    Average pairwise overlap is what Maitra and Melnykov parameterize and what
    this generator targeted through Stage 2a-2, and for k = 2 the two are the
    same number. Beyond that the average stops constraining the thing a reader
    of a density plot actually sees. With five components, a couple of heavily
    overlapping pairs carry the average while another pair sits at zero, so a
    mixture can hit an average overlap of 0.09 and still show two sharp peaks
    with empty space between them. Measured on corpus_2026-09-12c, the median
    SMALLEST pairwise overlap is 0.00000 at k = 3, 4 and 5 while the median
    average is 0.028, 0.060 and 0.091.

    Neighbouring rather than all pairs, because in one dimension the outermost
    pair of a five-component mixture is legitimately far apart; that is not what
    makes a density look wrong. What makes it look wrong is a GAP, and a gap is
    a consecutive pair with no overlap. The empirical arm agrees: its smallest
    pairwise overlap has a median of 0.0000 too, because of exactly this
    outermost-pair effect, so the all-pairs minimum cannot distinguish the two
    arms and the adjacent minimum can.

    Components are ordered by median rather than by mean, because a component
    with heavy skew and finite-but-huge kurtosis can have a mean far outside its
    own body.
    """
    k = len(comps)
    if k < 2:
        return 0.0
    order = np.argsort([float(d.ppf(0.5)) for d in comps])
    om = pairwise_overlap(comps, pi, grid_n)
    return float(min(om[order[i], order[i + 1]] for i in range(k - 1)))


def overlap_statistic(comps, pi, statistic='average', grid_n=120):
    """Dispatch for the quantity the spread solve targets."""
    if statistic == 'average':
        return average_overlap(comps, pi, grid_n)
    if statistic == 'min_adjacent':
        return min_adjacent_overlap(comps, pi, grid_n)
    raise ValueError(f'unknown overlap statistic {statistic!r}')


def log_truncation_bounds(q1, q3, mult):
    """Multiplicative interquartile bounds, the rule the empirical arm uses.

    In log space the interquartile rule is a ratio rather than a difference:

        lo = exp(log q1 - mult * (log q3 - log q1)) = q1 / (q3 / q1) ** mult
        hi = exp(log q3 + mult * (log q3 - log q1)) = q3 * (q3 / q1) ** mult

    Both bounds are strictly positive whenever q1 is, so no clip at zero is
    needed, which is the whole reason for preferring it: the ADDITIVE form of
    the same rule has a lower bound of q1 - mult * (q3 - q1), which is negative
    for any right-skewed distribution on the positive half line and therefore
    never binds. `src/datageneration.clean_empirical_symmetric` applies exactly
    this rule to the empirical data, so the two arms are now truncated the same
    way. Through Stage 2a-2 they were not: the empirical arm was multiplicative
    and the synthetic arm additive.

    Quantiles translate under a shift, so a caller solving for a shift can pass
    `q1 + shift` and `q3 + shift` rather than recomputing the mixture quantiles
    at every step.
    """
    if not (q1 > 0) or not (q3 > q1):
        return None
    r = q3 / q1
    return q1 / r ** mult, q3 * r ** mult



def solve_spread_for_overlap(build, target, lo=1e-4, hi=1e4, tol=1e-3,
                             grid_n=120, statistic='average'):
    """Find the location spread giving the requested overlap.

    `build(c)` returns the component list at spread multiplier c, with c small
    meaning the components sit on top of one another (overlap near its maximum)
    and c large meaning they are far apart (overlap near zero). Both overlap
    statistics are monotone decreasing in c, so a bisection is enough.

    `statistic` selects what is held to the target: 'average' is the
    Maitra-Melnykov average pairwise overlap, and 'min_adjacent' is the smallest
    overlap between neighbouring components. See min_adjacent_overlap for why
    the average stops being the right quantity once k > 2.

    This is Maitra and Melnykov's step 3, with the roles of scale and location
    exchanged: they hold locations fixed and scale the covariances, which for
    non-Gaussian components would change the component shapes' relationship to
    their own moment targets. Moving the locations instead leaves every
    component's standardized shape, skewness and kurtosis exactly as specified.

    Returns (components, realized_overlap, status).
    """
    def f(c):
        comps, pi = build(c)
        return overlap_statistic(comps, pi, statistic, grid_n)

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
def mixture_quantile(comps, pi, p):
    """p-quantile of the untruncated mixture, by bisection on its CDF."""
    def mix_cdf(x):
        return float(np.sum([w * float(d.cdf(x)) for w, d in zip(pi, comps)]))

    lo_b = min(float(d.ppf(1e-12)) for d in comps)
    hi_b = max(float(d.ppf(1 - 1e-12)) for d in comps)
    span = hi_b - lo_b
    lo_b, hi_b = lo_b - 0.5 * span, hi_b + 0.5 * span
    return optimize.brentq(lambda x: mix_cdf(x) - p, lo_b, hi_b,
                           xtol=1e-12 * max(1.0, abs(hi_b)), rtol=1e-13)


def population_truncation_bounds(comps, pi, mult=TRUNC_IQR_MULT, clip_at_zero=True):
    """Q1 - mult * IQR and Q3 + mult * IQR of the POPULATION mixture.

    The rule is the one the old generator used; what changes is that Q1 and Q3
    are quantiles of the mixture rather than of one realized draw, so the
    bounds are a property of the parent and the same for every sample from it.

    `clip_at_zero` reproduces the old `max(Q1 - mult * IQR, 0)`. The caller
    needs the UNCLIPPED lower bound to position the mixture, because the two
    are very different objects: for any right-skewed distribution on the
    positive half line, Q1 - 3 * IQR is comfortably negative and the low end is
    therefore not truncated at all. Shifting a mixture so that Q1 - 3 * IQR
    lands on zero is not the same as shifting it so its SUPPORT starts at zero,
    and the difference is large: the first forces the mean to sit about
    3.5 * IQR above the lower bound with the upper bound at about 7 * IQR,
    which caps the coefficient of variation near 0.5 whatever the component
    shapes. The empirical ECC datasets reach 2.08.
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
    lo = q1 - mult * iqr
    return (max(lo, 0.0) if clip_at_zero else lo), q3 + mult * iqr

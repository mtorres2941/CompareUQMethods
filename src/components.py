"""Component families for the synthetic ECC mixture generator.

Every family here has a closed-form CDF and quantile function in scipy, and
closed-form standardized moments, which is what lets Stage 2a write the parent
CDF down exactly and lets a component be specified by a moment TARGET rather
than by a shape knob tuned until the output looked right.

Four families tile the feasible (skewness, excess kurtosis) plane, following
the Pearson system's own partition of it:

    johnsonsu   above the lognormal line
    lognorm     on the lognormal line
    betaprime   between the gamma line (excess kurtosis = 1.5 * skewness ** 2)
                and the lognormal line, which is Pearson type VI
    beta        below the gamma line, which is Pearson type I, and covers the
                whole platykurtic region

Three families are not enough: beta stops at the gamma line, Johnson SU starts
at the lognormal line, and the band between them is a real gap. A first version
of this module left that gap open and silently failed on 13 percent of moment
targets, all of them inside it.

betaprime and lognorm have support on one side only, so for negative skewness
they are reflected about a fixed centre. That reflection is a property of the
component, computed from its own parameters, not from a realized sample.

The feasible region is bounded below by

    excess kurtosis >= skewness ** 2 - 2

which is the standard moment inequality beta2 >= beta1 + 1. A target below it
cannot be met by any distribution; `solve_component` reports that rather than
silently returning the nearest thing, because Stage 2a Part 6 asks what the
generator did when a requested combination could not exist.

Parameterization. Each family's shape parameters fix the STANDARDIZED shape;
loc and scale then place it. So a component is (family, shape, loc, scale) and
its skewness and excess kurtosis depend on shape alone.
"""

import numpy as np
from scipy import optimize, stats

FAMILIES = ('johnsonsu', 'lognorm', 'betaprime', 'beta')

# Moment targets closer to the boundary than this are treated as infeasible.
# The solvers lose conditioning there, and a component pinned against the
# boundary is nearly degenerate in any case.
BOUNDARY_MARGIN = 0.05


# --------------------------------------------------------------------------
# feasibility
# --------------------------------------------------------------------------
def min_excess_kurtosis(skew):
    """Lowest excess kurtosis any distribution can have at this skewness."""
    return np.asarray(skew, float) ** 2 - 2.0


def lognormal_excess_kurtosis(skew):
    """Excess kurtosis of the lognormal with this (signed) skewness.

    The lognormal line separates the Johnson SU region above it from the
    Johnson SB / beta region below it. Solving skew = (w + 2) * sqrt(w - 1) for
    w = exp(s ** 2) gives the matching kurtosis
    w ** 4 + 2 w ** 3 + 3 w ** 2 - 6.
    """
    s = np.abs(np.asarray(skew, float))
    out = np.empty_like(s)
    flat = out.reshape(-1)
    for i, si in enumerate(s.reshape(-1)):
        if si < 1e-9:
            flat[i] = 0.0          # the s -> 0 limit is the normal
            continue
        w = optimize.brentq(lambda w: (w + 2) * np.sqrt(w - 1) - si,
                            1 + 1e-15, 1e4, xtol=1e-14, rtol=1e-14)
        flat[i] = w ** 4 + 2 * w ** 3 + 3 * w ** 2 - 6
    return out if out.shape else float(out)


def gamma_excess_kurtosis(skew):
    """The Pearson type III line, excess kurtosis = 1.5 * skewness ** 2.

    Separates the beta (type I) region below it from the beta-prime (type VI)
    region above it.
    """
    return 1.5 * np.asarray(skew, float) ** 2


def is_feasible(skew, exkurt, margin=BOUNDARY_MARGIN):
    return np.asarray(exkurt, float) >= min_excess_kurtosis(skew) + margin


# --------------------------------------------------------------------------
# Johnson SU: closed-form moments, so the solve is a pair of bisections
# --------------------------------------------------------------------------
# scipy parameterizes johnsonsu(a, b) as X = sinh((Z - a) / b), which is the
# classical (gamma, delta) with gamma = a and delta = b. Writing w = exp(1/b^2)
# and W = a / b, the standardized moments are the textbook expressions. They are
# evaluated below in terms of r = exp(-2|W|) so that nothing overflows: every
# cosh and sinh is factored by exp(2|W|) to the appropriate power first, and the
# factors cancel between numerator and denominator.
W_MAX = 100.0          # r underflows to 0 far below this, so it is the W -> inf limit
W_TOL = 1e-13
LOGW_MAX = np.log(1e8)


def _su_shape_from_wW(w, W):
    """(w, W) -> scipy johnsonsu (a, b)."""
    b = 1.0 / np.sqrt(np.log(w))
    return float(W * b), float(b)


def _su_moments_wW(w, W):
    """Skewness and excess kurtosis at (w, W), overflow-free."""
    sgn = -1.0 if W < 0 else 1.0
    r = np.exp(-2.0 * abs(W))                     # exp(2|W|) ** -1
    cA = 0.5 * w * (1.0 + r * r) + r              # c2 / exp(2|W|)
    n3A = -np.sqrt(w) * np.sqrt(w - 1.0) * sgn * 0.5 * (
        w * (w + 2.0) * (1.0 - r ** 3) + 3.0 * r * (1.0 - r))
    skew = n3A / (np.sqrt(2.0) * cA ** 1.5)
    n4A = (0.5 * w ** 2 * (w ** 4 + 2 * w ** 3 + 3 * w ** 2 - 3) * (1.0 + r ** 4)
           + 2.0 * w ** 2 * (w + 2.0) * r * (1.0 + r * r)
           + 3.0 * (2 * w + 1.0) * r * r)
    return float(skew), float(n4A / (2.0 * cA ** 2) - 3.0)


def _su_max_abs_skew(w):
    """|skewness| in the W -> inf limit, the most skew this w can produce."""
    return abs(_su_moments_wW(w, W_MAX)[0])


def _su_W_for_skew(w, target_abs_skew):
    """|W| giving this |skewness| at this w, or None if w cannot reach it.

    |skewness| is 0 at W = 0 and rises monotonically to a finite limit, so a
    plain bisection is both correct and robust here.
    """
    if _su_max_abs_skew(w) < target_abs_skew:
        return None
    lo, hi = 0.0, W_MAX
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if abs(_su_moments_wW(w, mid)[0]) < target_abs_skew:
            lo = mid
        else:
            hi = mid
        if hi - lo < W_TOL:
            break
    return 0.5 * (lo + hi)


def _solve_su(skew, exkurt):
    """scipy johnsonsu (a, b) matching the standardized moments, or None.

    Excess kurtosis at a fixed skewness decreases as w falls toward 1, where SU
    approaches the lognormal line from above, and rises without bound as w
    grows. Bisecting on log w with |W| re-solved for the skewness at each step
    is monotone in both variables and needs no starting guess.
    """
    target = abs(float(skew))
    sgn_skew = 1.0 if skew >= 0 else -1.0

    def kurt_at(logw):
        w = np.exp(logw)
        W = _su_W_for_skew(w, target)
        if W is None:
            return None
        return _su_moments_wW(w, W)[1]

    lo = 1e-12                      # w -> 1, the lognormal line
    hi = LOGW_MAX
    # the smallest w that can produce this much skew at all
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _su_max_abs_skew(np.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    lo = hi                         # everything below this cannot reach the skew

    k_lo, k_hi = kurt_at(lo), kurt_at(LOGW_MAX)
    if k_lo is None or k_hi is None or not (k_lo <= exkurt <= k_hi):
        return None
    a_, b_ = lo, LOGW_MAX
    for _ in range(300):
        mid = 0.5 * (a_ + b_)
        km = kurt_at(mid)
        if km is None or km < exkurt:
            a_ = mid
        else:
            b_ = mid
        if b_ - a_ < 1e-14:
            break
    w = np.exp(0.5 * (a_ + b_))
    W = _su_W_for_skew(w, target)
    if W is None:
        return None
    s, k = _su_moments_wW(w, -sgn_skew * W)
    if abs(s - skew) > 1e-7 * max(1.0, target) or \
       abs(k - exkurt) > 1e-6 * max(1.0, abs(exkurt)):
        return None
    return _su_shape_from_wW(w, -sgn_skew * W)


# --------------------------------------------------------------------------
# Beta: closed-form moments too
# --------------------------------------------------------------------------
def _beta_standardized_moments(a, b):
    # A root finder legitimately probes a + b large enough to overflow on its
    # way to the solution; the resulting inf/nan is rejected by the caller.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        return _beta_standardized_moments_raw(a, b)


def _beta_standardized_moments_raw(a, b):
    s = a + b
    skew = 2.0 * (b - a) * np.sqrt(s + 1.0) / ((s + 2.0) * np.sqrt(a * b))
    exk = (6.0 * ((a - b) ** 2 * (s + 1.0) - a * b * (s + 2.0))
           / (a * b * (s + 2.0) * (s + 3.0)))
    return float(skew), float(exk)


def _solve_beta(skew, exkurt):
    def resid(p):
        a, b = np.exp(p)
        s, k = _beta_standardized_moments(a, b)
        return [s - skew, k - exkurt]

    for a0, b0 in ((2.0, 2.0), (1.0, 1.0), (0.6, 0.6), (5.0, 5.0),
                   (1.0, 4.0), (4.0, 1.0), (0.3, 1.0), (1.0, 0.3)):
        try:
            sol = optimize.root(resid, [np.log(a0), np.log(b0)], method='hybr',
                                options=dict(xtol=1e-12))
        except Exception:
            continue
        if sol.success:
            a, b = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
            s, k = _beta_standardized_moments(a, b)
            if abs(s - skew) < 1e-6 and abs(k - exkurt) < 1e-6 * max(1, abs(exkurt)):
                return a, b
    return None


# --------------------------------------------------------------------------
# Beta-prime (Pearson VI): raw moments are a ratio of rising factorials
# --------------------------------------------------------------------------
def _betaprime_standardized_moments(a, b):
    """Skewness and excess kurtosis of betaprime(a, b). Needs b > 4."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        return _betaprime_standardized_moments_raw(a, b)


def _betaprime_standardized_moments_raw(a, b):
    if b <= 4.0 + 1e-9:
        return np.nan, np.nan
    m1 = a / (b - 1.0)
    m2 = a * (a + 1.0) / ((b - 1.0) * (b - 2.0))
    m3 = a * (a + 1.0) * (a + 2.0) / ((b - 1.0) * (b - 2.0) * (b - 3.0))
    m4 = (a * (a + 1.0) * (a + 2.0) * (a + 3.0)
          / ((b - 1.0) * (b - 2.0) * (b - 3.0) * (b - 4.0)))
    var = m2 - m1 ** 2
    if var <= 0:
        return np.nan, np.nan
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return float(mu3 / var ** 1.5), float(mu4 / var ** 2 - 3.0)


def _solve_betaprime(abs_skew, exkurt):
    """betaprime (a, b) for a POSITIVE skewness target, or None.

    Solved on (log a, log(b - 4)) so the b > 4 constraint that the fourth
    moment needs is enforced by the parameterization rather than by a penalty.
    """
    def resid(p):
        a = np.exp(p[0])
        b = 4.0 + np.exp(p[1])
        s, k = _betaprime_standardized_moments(a, b)
        if not np.isfinite(s) or not np.isfinite(k):
            return [1e6, 1e6]
        return [s - abs_skew, np.log1p(k + 3.0) - np.log1p(exkurt + 3.0)]

    for a0 in (0.5, 1.0, 2.0, 5.0, 15.0, 50.0, 0.15):
        for db0 in (0.3, 1.0, 3.0, 10.0, 40.0, 0.05):
            try:
                sol = optimize.root(resid, [np.log(a0), np.log(db0)], method='hybr',
                                    options=dict(xtol=1e-13))
            except Exception:
                continue
            if not sol.success:
                continue
            a = float(np.exp(sol.x[0]))
            b = 4.0 + float(np.exp(sol.x[1]))
            s, k = _betaprime_standardized_moments(a, b)
            if (np.isfinite(s) and np.isfinite(k)
                    and abs(s - abs_skew) < 1e-8 * max(1.0, abs_skew)
                    and abs(k - exkurt) < 1e-8 * max(1.0, abs(exkurt))):
                return a, b
    return None


# --------------------------------------------------------------------------
# the public entry point
# --------------------------------------------------------------------------
def solve_component(skew, exkurt, mean=0.0, sd=1.0):
    """Return (family, shape, loc, scale, status) hitting the moment target.

    The returned component has exactly the requested mean, standard deviation,
    skewness and excess kurtosis whenever status == 'ok'. status is one of:

        ok                     target met
        infeasible_boundary    below excess kurtosis = skewness ** 2 - 2
        degenerate             a solution exists but is numerically unusable:
                               the fitted shape puts essentially all its mass
                               at two points, so its CDF cannot be inverted.
                               Extreme skewness with modest kurtosis does this,
                               driving beta toward a U shape with a, b << 1
        unsolved               inside the feasible region but no solver
                               converged; the caller decides what to do

    Nothing here silently substitutes a different target. A 'degenerate' or
    'unsolved' status is returned for the caller to record and act on, which is
    what Stage 2a Part 6 needs in order to say what the generator did when a
    combination could not be produced.
    """
    skew = float(skew)
    exkurt = float(exkurt)
    if not is_feasible(skew, exkurt):
        return None, None, None, None, 'infeasible_boundary'

    line = float(lognormal_excess_kurtosis(skew))
    if abs(exkurt - line) < 1e-9:
        s = optimize.brentq(
            lambda s: (np.exp(s ** 2) + 2) * np.sqrt(np.exp(s ** 2) - 1) - abs(skew),
            1e-12, 10, xtol=1e-15) if abs(skew) > 1e-12 else 1e-6
        return _place('lognorm', (s,), skew, mean, sd)

    if exkurt > line:
        got = _solve_su(skew, exkurt)
        if got is None:
            return None, None, None, None, 'unsolved'
        return _place('johnsonsu', got, 1.0, mean, sd)

    if exkurt > gamma_excess_kurtosis(skew):
        got = _solve_betaprime(abs(skew), exkurt)
        if got is None:
            return None, None, None, None, 'unsolved'
        return _place('betaprime', got, skew, mean, sd)

    got = _solve_beta(skew, exkurt)
    if got is None:
        return None, None, None, None, 'unsolved'
    return _place('beta', got, 1.0, mean, sd)


# A component must invert its own CDF to this accuracy across the body of the
# distribution, or it is useless for inverse-CDF sampling and for a parent CDF.
#
# The check runs on the PLACED component, loc + scale * X, not on the
# standardized shape. That is deliberate: a shape whose bulk sits near 0 while
# its lower quantiles run down to 1e-35 inverts perfectly in standardized form
# and then loses the entire lower tail to floating point once a location of
# order 1 is added to it. The object the generator actually uses is the placed
# one, so that is the one that has to work.
#
# The grid stops at the 0.1 and 99.9 percentiles because the truncation step
# discards more than that at both ends, so resolution further out is never
# needed.
ROUNDTRIP_Q = np.array([1e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
                        0.99, 1 - 1e-3])
ROUNDTRIP_TOL = 1e-8


def _roundtrips(d):
    try:
        x = d.ppf(ROUNDTRIP_Q)
        if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            return False
        return bool(np.max(np.abs(d.cdf(x) - ROUNDTRIP_Q)) < ROUNDTRIP_TOL)
    except Exception:
        return False


def _place(family, shape, skew_sign, mean, sd):
    """Give a solved standardized shape the requested mean and sd.

    Families with one-sided support (lognorm, betaprime) are solved for
    POSITIVE skewness and reflected when the target is negative; `sign` in the
    stored shape records which. Reflection here is about the component's own
    mean, so the component is fully determined by its parameters.
    """
    sign = 1.0 if skew_sign >= 0 else -1.0
    if family in ('lognorm', 'betaprime'):
        shape = tuple(shape) + (sign,)
    base = _standard_frozen(family, shape)
    m, v = base.stats(moments='mv')
    if not np.isfinite(v) or v <= 0:
        return None, None, None, None, 'degenerate'
    scale = sd / np.sqrt(float(v))
    loc = float(mean - scale * float(m))
    if not _roundtrips(_Affine(base, loc, scale)):
        return None, None, None, None, 'degenerate'
    return family, shape, loc, float(scale), 'ok'


def _standard_frozen(family, shape):
    if family == 'johnsonsu':
        return stats.johnsonsu(shape[0], shape[1])
    if family == 'beta':
        return stats.beta(shape[0], shape[1])
    if family == 'lognorm':
        d = stats.lognorm(shape[0])
        return d if shape[-1] >= 0 else _Reflected(d)
    if family == 'betaprime':
        d = stats.betaprime(shape[0], shape[1])
        return d if shape[-1] >= 0 else _Reflected(d)
    raise ValueError(f'unknown family {family!r}')


def frozen(family, shape, loc, scale):
    """A frozen, shifted and scaled distribution for a solved component.

    Exposes pdf, cdf, ppf and sf. Reflected families are wrapped rather than
    special-cased at every call site.
    """
    return _Affine(_standard_frozen(family, shape), loc, scale)


class _Reflected:
    """Y = -X, so a right-skewed family supplies left skew as a population
    property. This replaces the old sample-dependent flip
    `np.max(data) - data + np.min(data)`, which used realized order statistics
    and was the single step that made the parent impossible to write down."""

    def __init__(self, d):
        self._d = d

    def pdf(self, x):
        return self._d.pdf(-np.asarray(x, float))

    def cdf(self, x):
        return self._d.sf(-np.asarray(x, float))

    def sf(self, x):
        return self._d.cdf(-np.asarray(x, float))

    def ppf(self, q):
        return -self._d.isf(np.asarray(q, float))

    def stats(self, moments='mv'):
        m, v = self._d.stats(moments='mv')
        return -float(m), float(v)


class _Affine:
    """Z = loc + scale * X."""

    __slots__ = ('_d', 'loc', 'scale')

    def __init__(self, d, loc, scale):
        self._d, self.loc, self.scale = d, float(loc), float(scale)

    def pdf(self, x):
        return self._d.pdf((np.asarray(x, float) - self.loc) / self.scale) / self.scale

    def cdf(self, x):
        return np.asarray(self._d.cdf((np.asarray(x, float) - self.loc) / self.scale),
                          dtype=float)

    def sf(self, x):
        return np.asarray(self._d.sf((np.asarray(x, float) - self.loc) / self.scale),
                          dtype=float)

    def ppf(self, q):
        return self.loc + self.scale * np.asarray(self._d.ppf(np.asarray(q, float)),
                                                  dtype=float)

"""
Fitting and scoring for the six UQ methods.

One implementation of the PEWT fit. Before Stage 1 this block existed as three
verbatim copies, in notebook 2 cells 12 and 20 and notebook 3 cell 15, so any
change to the method had to be made in three places and kept in step by hand.
Stage 2 changes the lognormal threshold and the bandwidth rule, which is
exactly the kind of edit that goes wrong when it has to be repeated.

The default constants below reproduce the behaviour as of Stage 1. Both are
under review in Stage 2; see reports/MANUSCRIPT_discrepancies.md entries 6 and
10.
"""

import numpy as np
from scipy.stats import gaussian_kde, lognorm, norm

import families
from customstats import (
    wasserstein1_weighted,
    weighted_bw,
    weighted_ecdf,
    weighted_lognorm_fit,
    weighted_std,
)

PE_METHODS = ["Normal", "Lognormal", "KDE"]
WT_METHODS = ["Uniform", "Variable"]
PEWT = [f"{pe}, {wt}" for pe in PE_METHODS for wt in WT_METHODS]

# The data are offset by this much before the lognormal fit and the location is
# shifted back afterwards, which is a 3-parameter lognormal with the threshold
# fixed at -LOGFIT_OFFSET rather than estimated. It exists because a handful of
# near-zero values otherwise drag the log-space mean down and inflate sigma,
# collapsing the fitted mode toward zero. That problem is real and is worse in
# the empirical data than in the synthetic data: 28.3% of the 138 empirical ECC
# datasets have a minimum below 1% of their mean, against 1.8% of the synthetic
# datasets. The fixed value is nonetheless arbitrary and is under review.
LOGFIT_OFFSET = 0.5

# The KDE bandwidth rule. Author decision, 2026-09-14: Silverman's robust rule
# of thumb, guarded by a minimum EFFECTIVE sample size.
#
#   'scott'              1.06 * sigma * n_eff ** -0.2, Scott (1992). What this
#                        study used through Stage 2a.
#   'silverman'          0.9 * min(sigma, IQR/1.34) * n_eff ** -0.2. The rule
#                        Torres et al. (2026), the KL2 paper, uses and defends.
#   'silverman_guarded'  Silverman above customstats.SILVERMAN_MIN_NEFF = 30
#                        effective observations, Scott below it.
#
# WHY GUARDED, and the reason is not the one it looks like. Silverman's
# min(sigma, IQR/1.34) protects against outliers inflating the bandwidth, and it
# works hardest exactly where it looks most alarming: on the heavy-tailed
# categories where (IQR/1.34)/sigma falls below 0.2 it beats Scott on held-out
# likelihood 100 percent of the time. It fails at SMALL n, where the quartiles
# are interpolated between two order statistics and a low estimate collapses the
# bandwidth into a set of spikes. Guarding on effective sample size beats BOTH
# pure rules on leave-one-out likelihood and repairs the worst cases.
#
# THE THRESHOLD IS CALIBRATED ON HELD-OUT LIKELIHOOD, NOT ON W1, deliberately:
# W1 falls monotonically as the bandwidth shrinks, so it cannot choose one.
# See customstats.weighted_bw, audits/bandwidth_rules.py, entry 45.
#
# Note that scipy.stats.gaussian_kde uses the words 'scott' and 'silverman' for
# different formulas: its 'scott' carries no 1.06 factor. Do not describe the
# method by pointing at a scipy keyword. Decision 9.
BW_METHOD = "silverman_guarded"

# The scoring grid runs from 0 to max(data) + this many standard deviations.
SCORE_GRID_POINTS = 1_000
SCORE_GRID_STD_MULTIPLE = 10


def uniform_weights(x):
    """Equal weight on every data point, summing to one."""
    return np.ones_like(x) / len(x)


def fit_pewt_models(x, weights_variable, logfit_offset=LOGFIT_OFFSET,
                    bw_method=BW_METHOD):
    """Fit all six probability-estimation / weighting combinations.

    Parameters
    ----------
    x : array-like
        One ECC dataset, strictly positive.
    weights_variable : array-like
        Per-point weights for the variable weighting scheme.
    logfit_offset : float
        Positive shift applied before the lognormal fit, undone in the fitted
        location, so the model has support (-logfit_offset, inf).
    bw_method : {'scott', 'silverman', 'silverman_guarded'}
        Passed to customstats.weighted_bw.

    Returns
    -------
    dict
        Keys are the six PEWT labels; values are frozen scipy distributions or
        a gaussian_kde, each exposing .pdf().
    """
    x = np.asarray(x, dtype=float)
    weights_variable = np.asarray(weights_variable, dtype=float)

    models = {}
    for weights, wt in zip([uniform_weights(x), weights_variable], WT_METHODS):
        # Normal: weighted mean and weighted standard deviation.
        mean = np.sum(x * weights) / np.sum(weights)
        models[f"Normal, {wt}"] = norm(loc=mean, scale=weighted_std(x, weights))

        # Lognormal: weighted MLE on the offset data, location shifted back.
        shape, loc, scale = weighted_lognorm_fit(
            x + logfit_offset, weights=weights, method="MLE"
        )
        models[f"Lognormal, {wt}"] = lognorm(
            s=shape, loc=loc - logfit_offset, scale=scale
        )

        # KDE: bandwidth computed here, then imposed on the scipy estimator.
        # gaussian_kde is constructed with bw_method=1.0 and rescaled, because
        # its own rules do not account for weights.
        bandwidth = weighted_bw(x, weights, bw_method=bw_method)
        kde = gaussian_kde(x, bw_method=1.0, weights=weights)
        kde.set_bandwidth(bandwidth / (kde.covariance ** 0.5)[0][0])
        models[f"KDE, {wt}"] = kde

    return models


def score_grid(x, weights):
    """The x-grid a fitted model is discretized onto for scoring.

    Runs from 0, so every model is implicitly truncated at zero and
    renormalized. That matches the rejection sampling used in the pLCA, which
    discards non-positive draws, so the model that is scored is the model that
    is sampled.
    """
    x = np.asarray(x, dtype=float)
    spread = np.max([np.std(x), weighted_std(x, weights)])
    return np.linspace(
        0, np.max(x) + SCORE_GRID_STD_MULTIPLE * spread, SCORE_GRID_POINTS
    )


def score_w1(model, x, weights):
    """Wasserstein-1 between a fitted model and the weighted empirical data.

    Every model is scored against the variable-weighted empirical CDF,
    including the uniform-weighted fits.
    """
    grid = score_grid(x, weights)
    return wasserstein1_weighted(x, grid, weights, model.pdf(grid))


def score_all(models, x, weights):
    """W1 for each of the six fits, as a dict keyed by PEWT label."""
    return {label: score_w1(model, x, weights) for label, model in models.items()}


# ===========================================================================
# Stage 2b. The support, the families, and fitting by the criterion we score by
# ===========================================================================
#
# Everything above this line is the Stage 1 implementation, kept verbatim so
# that the change from it can be measured rather than asserted. Everything below
# is the Stage 2b replacement. `PEWT_METHODS` selects which is used.

#: The scoring grid's lower bound is the first point of the same lattice
#: strictly above zero, so zero is excluded without introducing a new free
#: parameter. See `score_grid_open` for why that, and not some invented epsilon.
GRID_OPEN_AT_ZERO = True

#: Families available to `fit_family`. The value is (estimator, builder).
#: `lognormal_offset` is the Stage 1 method, kept so it can be compared rather
#: than remembered; see LOGFIT_OFFSET above for what it is.
FAMILIES = {
    'normal': (families.fit_normal_mle, families.make_normal),
    'lognormal_2p': (families.fit_lognorm2_mle, families.make_lognorm),
    'lognormal_3p': (families.fit_lognorm3_profile, families.make_lognorm),
    'lognormal_offset': (None, families.make_lognorm),
    'gamma': (families.fit_gamma_mle, families.make_gamma),
}

PARAMETRIC_FAMILIES = tuple(FAMILIES)


def score_grid_open(x, weights, npoints=SCORE_GRID_POINTS,
                    std_multiple=SCORE_GRID_STD_MULTIPLE):
    """The W1 evaluation grid, on (0, hi] rather than [0, hi].

    THE GRID, STATED EXPLICITLY, because the paper has to state it. `npoints`
    equally spaced points running from `hi / npoints` to
    `hi = max(x) + std_multiple * spread`, where `spread` is the larger of the
    unweighted and the weighted standard deviation. With the defaults that is
    1,000 points and 10 standard deviations.

    HOW THE LOWER BOUND WAS CHOSEN. It is the first point of the same lattice
    above zero, `hi / npoints`, and it is chosen that way precisely so that it
    is NOT a choice. Decision 13 says an ECC of exactly zero is not admissible,
    so zero must leave the grid; discrepancy entry 18 records that the old grid
    included it while the pLCA sampler excluded it, which made the scored object
    and the sampled object different sets. Any invented epsilon -- 1e-6, a
    fraction of min(x), a quantile of the fitted model -- would be a new
    parameter that a reviewer could ask about and that would have to be swept.
    The lattice already exists and already sets the resolution of the whole
    calculation, so its own first point is the bound that adds nothing.

    WHAT THIS DOES NOT FIX, and it is worth stating. The lattice is LINEAR, so
    its resolution near zero is `hi / npoints` for every dataset. On a dataset
    whose values span several orders of magnitude the interval below the first
    grid point can contain real data, and everything the model puts there is
    lumped onto one point. That is a property of the linear grid, it was present
    before this change and is unchanged by it, and it falls to Stage 2c, which
    owns the evaluation target. `w1_grid_error` in
    `audits/fitting_method_comparison.py` measures it.
    """
    x = np.asarray(x, dtype=float)
    spread = np.max([np.std(x), weighted_std(x, weights)])
    hi = np.max(x) + std_multiple * spread
    return np.linspace(hi / npoints, hi, npoints)


def score_w1_model(model, x, weights, grid=None):
    """W1 between a truncated model and the variable-weighted empirical CDF.

    The model is discretized onto `grid` with weight proportional to its
    density, which is what `wasserstein1_weighted` consumes, and is the same
    calculation Stage 1 performed. What has changed is the object being
    discretized: it is now an explicit `families.Truncated`, renormalized onto
    (0, inf), rather than an untruncated parent that the grid happened to
    truncate by starting at zero.
    """
    if grid is None:
        grid = score_grid_open(x, weights)
    return wasserstein1_weighted(x, grid, weights, model.pdf(grid))


def score_w1_exact(model, x, weights, npoints=200_001):
    """W1 computed from the model's own CDF on a dense lattice.

    Not the study's criterion, and not a replacement for it. This exists to
    measure how much of a reported W1 is the 1,000-point discretization in
    `score_w1_model` rather than the fit, which is a question `score_grid_open`
    raises and cannot answer.
    """
    grid = score_grid_open(x, weights, npoints=npoints)
    e = weighted_ecdf(x, weights)[2](grid)
    return float(np.trapezoid(np.abs(model.cdf(grid) - e), grid))


# ---------------------------------------------------------------------------
# estimation: maximum likelihood, and minimum W1
# ---------------------------------------------------------------------------
def fit_family(name, x, weights, method='mle', logfit_offset=LOGFIT_OFFSET,
               **kw):
    """Fit one parametric family, by `mle` or by `w1`.

    `w1` minimizes the SAME quantity the model is then scored by. Every
    parametric family in this study is estimated by maximum likelihood and
    judged by W1, and those are different criteria: a family can lose the
    comparison because it was never fitted under the rule it is judged by. That
    is a real vulnerability in the paper and this is how it is answered. The
    W1-optimal fit starts from the maximum-likelihood fit, so it can never be
    worse by more than the optimizer's tolerance.
    """
    if name not in FAMILIES:
        raise ValueError(f'unknown family {name!r}; have {sorted(FAMILIES)}')
    estimator, builder = FAMILIES[name]
    if name == 'lognormal_offset':
        p = families.fit_lognorm2_mle(np.asarray(x, float) + logfit_offset,
                                      weights)
        p = dict(p, loc=p['loc'] - logfit_offset)
    else:
        p = estimator(x, weights, **kw)
    if method == 'mle':
        return builder(p), p
    if method != 'w1':
        raise ValueError(f"method must be 'mle' or 'w1', got {method!r}")
    return _fit_w1(name, x, weights, p, builder, logfit_offset)


#: Nelder-Mead rather than a gradient method: W1 on a finite grid is piecewise
#: linear in the model parameters, so it has no useful derivative. The starting
#: point is always the maximum-likelihood fit.
W1_OPT_MAXITER = 600
W1_OPT_XATOL = 1e-8
W1_OPT_FATOL = 1e-12


def _w1_pack(name, p, x):
    """Family parameters -> an unconstrained vector, and the inverse.

    Positive parameters are carried as logs and the lognormal threshold as
    `log(min(x) - gamma)`, so no optimizer step can produce a shape, a scale or
    a threshold that the family does not admit. That is what keeps the search
    from wandering into the unbounded-likelihood region that
    `families.fit_lognorm3_profile` exists to avoid.
    """
    xmin = float(np.min(x))
    if name == 'normal':
        v = np.array([p['loc'], np.log(p['scale'])])
        return v, (lambda z: dict(loc=z[0], scale=float(np.exp(z[1]))))
    if name in ('lognormal_2p', 'lognormal_offset'):
        loc = p['loc']
        v = np.array([np.log(p['scale']), np.log(p['s'])])
        return v, (lambda z: dict(s=float(np.exp(z[1])), loc=loc,
                                  scale=float(np.exp(z[0]))))
    if name == 'lognormal_3p':
        d = max(xmin - p['loc'], 1e-12)
        v = np.array([np.log(d), np.log(p['scale']), np.log(p['s'])])
        return v, (lambda z: dict(s=float(np.exp(z[2])),
                                  loc=float(xmin - np.exp(z[0])),
                                  scale=float(np.exp(z[1]))))
    if name == 'gamma':
        v = np.array([np.log(p['a']), np.log(p['scale'])])
        return v, (lambda z: dict(a=float(np.exp(z[0])), loc=0.0,
                                  scale=float(np.exp(z[1]))))
    raise ValueError(name)


def _fit_w1(name, x, weights, p_mle, builder, logfit_offset):
    from scipy.optimize import minimize as _minimize
    grid = score_grid_open(x, weights)
    v0, unpack = _w1_pack(name, p_mle, x)

    def objective(z):
        if not np.all(np.isfinite(z)):
            return np.inf
        try:
            m = builder(unpack(z))
        except (ValueError, FloatingPointError):
            return np.inf
        d = wasserstein1_weighted(x, grid, weights, m.pdf(grid))
        return d if np.isfinite(d) else np.inf

    res = _minimize(objective, v0, method='Nelder-Mead',
                    options=dict(maxiter=W1_OPT_MAXITER, xatol=W1_OPT_XATOL,
                                 fatol=W1_OPT_FATOL))
    # The maximum-likelihood fit is the starting point, so the W1-optimal fit is
    # accepted only if it actually improved on it. Nelder-Mead can return a
    # worse vertex when the objective is flat.
    if res.fun < objective(v0):
        p = unpack(res.x)
    else:
        p = dict(p_mle)
    p = {k: v for k, v in p.items() if k in ('s', 'a', 'loc', 'scale')}
    p['w1_opt_iterations'] = int(res.nit)
    p['w1_opt_improved'] = bool(res.fun < objective(v0))
    keep = {k: v for k, v in p.items() if k in ('s', 'a', 'loc', 'scale')}
    return builder(keep), p


def fit_kde(x, weights, bw_method=BW_METHOD):
    """The KDE, as a truncated, renormalized object with a CDF and an inverse.

    `scipy.stats.gaussian_kde` offers a density and a resampler and no CDF, so
    a KDE scored through it cannot be the same object that is sampled from it.
    `families.WeightedKDE` supplies the closed-form CDF; the density is
    identical to gaussian_kde's to machine precision, which
    `tests/test_families.py` checks.
    """
    bw = weighted_bw(x, weights, bw_method=bw_method)
    return families.Truncated(families.WeightedKDE(x, weights, bw), label='kde',
                              params=dict(bandwidth=bw)), dict(bandwidth=bw)


# ---------------------------------------------------------------------------
# the six UQ methods, on the settled support
# ---------------------------------------------------------------------------
#: Which lognormal the study's "Lognormal" method is. Author instruction,
#: Stage 2b: the three-parameter lognormal is fitted by profile likelihood over
#: a threshold restricted to a closed interval strictly below min(x), taking the
#: interior local maximum. See `families.fit_lognorm3_profile` for the pathology
#: that makes this necessary, and `audits/lognormal_offset.py` for
#: what the +0.5 offset it replaces was actually doing.
LOGNORMAL_FAMILY = 'lognormal_3p'

#: How every parametric family is estimated in the production pipeline.
#: 'mle' is the study's method. 'w1' fits by the criterion the model is then
#: scored by, and is reported alongside rather than instead; see `fit_family`.
FIT_METHOD = 'mle'

PE_FAMILY = {'Normal': 'normal', 'Lognormal': LOGNORMAL_FAMILY, 'KDE': 'kde'}


def fit_pewt(x, weights_variable, lognormal_family=None, method=None,
             bw_method=BW_METHOD):
    """The six probability-estimation / weighting combinations, on (0, inf).

    Replaces `fit_pewt_models`, which is kept above so the change from it can be
    measured. Three things differ and each is a Stage 2b decision.

    1. EVERY MODEL IS AN EXPLICIT TRUNCATION to (0, inf), renormalized, with a
       CDF and an inverse CDF. Before this the normal and the KDE were scored as
       untruncated objects and sampled as truncated ones, and neither the
       scoring grid starting at zero nor the pLCA's rejection loop said so.
       Decision 13, confirmed by the author.
    2. THE LOGNORMAL IS FITTED BY PROFILE LIKELIHOOD over its threshold, and the
       +0.5 offset is gone. `LOGNORMAL_FAMILY` names the family; pass
       `lognormal_family='lognormal_offset'` to recover the Stage 1 method.
    3. SAMPLING IS BY INVERSE CDF, through `.rvs` or `.rvs_from_uniform`, and
       not by rejection.

    Returns (models, params): models keyed by the six PEWT labels, params by the
    same keys, so a fit that hit a boundary is visible rather than assumed.
    """
    lognormal_family = lognormal_family or LOGNORMAL_FAMILY
    method = method or FIT_METHOD
    x = np.asarray(x, dtype=float)
    weights_variable = np.asarray(weights_variable, dtype=float)

    models, params = {}, {}
    for weights, wt in zip([uniform_weights(x), weights_variable], WT_METHODS):
        for pe in PE_METHODS:
            fam = lognormal_family if pe == 'Lognormal' else PE_FAMILY[pe]
            label = f'{pe}, {wt}'
            if fam == 'kde':
                models[label], params[label] = fit_kde(x, weights,
                                                       bw_method=bw_method)
            else:
                models[label], params[label] = fit_family(fam, x, weights,
                                                          method)
    return models, params


def score_all_models(models, x, weights):
    """W1 for each fit, against the VARIABLE-weighted empirical CDF.

    One grid for all six, so the six numbers are comparable. That was already
    true and stays true; what changed is that the grid is open at zero.
    """
    grid = score_grid_open(x, weights)
    return {label: score_w1_model(m, x, weights, grid=grid)
            for label, m in models.items()}

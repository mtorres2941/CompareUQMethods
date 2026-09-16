"""The evaluation target: what a fitted model is scored AGAINST.

Stage 2c. Through Stage 2b every model was scored by W1 against the
variable-weighted empirical CDF of the same data it was fitted to. That target
has two defects and neither is small.

  1. IT IS THE TRAINING DATA. A flexible method scored on its own training data
     is rewarded for flexibility with no complexity penalty, and the KDE is the
     most flexible method here by a wide margin: in-sample W1 falls
     monotonically as its bandwidth shrinks, to about 2 percent of any standard
     rule, because a KDE with a vanishing bandwidth IS the empirical
     distribution it is being compared with.
  2. IT IS THE VARIABLE-WEIGHTED CDF, which makes "variable weighting improves
     fit" close to true by construction, since the variable-weighted empirical
     CDF is itself the target.

The fix differs by arm, and both are implemented here.

SYNTHETIC: score against the KNOWN PARENT. A synthetic dataset is drawn from a
`mixture.MixtureParent` that this project can write down exactly, so the truth
is available and the question becomes estimator recovery rather than curve
fitting. Uniform-weighted methods are scored against the SAMPLING mixture, the
distribution the values were actually drawn from; variable-weighted methods
against the MARKET-weighted mixture, which is a real population object because
market share attaches at the mode level (decision 21, `mode_coupling = 1.0`).
Both are closed form. There is no training data in the target and no Dirichlet
draw noise in it either, which matters: Stage 2b measured the weight-draw noise
floor of the empirical target at a median of 0.1344 against a best-method
median W1 of 0.0984, so the old target's own noise exceeded the best score.

EMPIRICAL: there is no parent, so CROSS-VALIDATE. Fit on part of the data and
score against the weighted empirical CDF of the part the model has not seen.
That removes the training-data defect without inventing a truth.

The two lines of evidence are independent and are reported side by side. The
synthetic one is exact but is only as relevant as the generator is realistic;
the empirical one is real data but is noisy and cannot separate estimation
error from the sampling error of the evaluation half. Neither on its own would
settle the comparison.

WHAT IS NOT HERE. The uniform-to-variable W1 is decomposed into location and
shape by Stage 2d, which owns that quantity. This module decomposes a different
one: a model's total error against the variable-weighted target, into the error
against its own weighting scheme and the definitional gap between the two
weightings.
"""

import numpy as np
import pandas as pd

import fitting as FT
from customstats import weighted_ecdf, weighted_std

#: Which parent a method is scored against. A uniform-weighted method estimates
#: the distribution the VALUES came from, so its target is the sampling mixture;
#: a variable-weighted method estimates the market-weighted distribution, so its
#: target is the market-weighted mixture. Scoring both against one parent would
#: charge one of them for answering the question it was asked.
PARENT_SCHEME = {'Uniform': 'uniform', 'Variable': 'market'}

#: Points on the recovery grid. Ten times the study's in-sample grid, because
#: nothing here is limited by the number of data points and the integrand is two
#: smooth CDFs rather than a step function against a curve.
RECOVERY_GRID_POINTS = 10_000

#: Standard deviations of headroom above the data, as in `fitting.score_grid_open`.
RECOVERY_GRID_STD_MULTIPLE = FT.SCORE_GRID_STD_MULTIPLE

#: Upper quantile of the fitted model the tail charge integrates out to. The
#: mass beyond it is reported, so what the number omits is stated rather than
#: assumed.
TAIL_QUANTILE = 1.0 - 1e-9

#: Points in the log-spaced tail extension.
TAIL_GRID_POINTS = 2_000


# ---------------------------------------------------------------------------
# the synthetic arm: recovery against the known parent
# ---------------------------------------------------------------------------
def parent_support(parent):
    """(lo, hi) of the parent in NORMALIZED units, the units the data is in.

    A parent is stated in raw units and the dataset was divided by its realized
    unweighted sample mean, which the generation record carries as
    `normalizer`. `MixtureParent.cdf` already applies it; its `lo` and `hi`
    attributes do not, so they are converted here rather than at each call site.
    """
    c = float(parent.normalizer)
    return float(parent.lo) / c, float(parent.hi) / c


def recovery_grid(x, weights, parent, npoints=RECOVERY_GRID_POINTS,
                  std_multiple=RECOVERY_GRID_STD_MULTIPLE):
    """The grid the recovery score integrates on.

    Built exactly like `fitting.score_grid_open` -- `npoints` equally spaced
    points from `hi / npoints` to `hi`, open at zero because the support is
    (0, inf) -- with one addition: `hi` also covers the parent's upper support
    bound. The data-driven bound `max(x) + 10 * spread` is almost always the
    larger of the two, but a small sample from a long-tailed parent can fall
    entirely inside it, and a grid that stopped short of the truth would score
    every method against a truncated version of it.
    """
    x = np.asarray(x, dtype=float)
    spread = np.max([np.std(x), weighted_std(x, weights)])
    hi = max(float(np.max(x)) + std_multiple * spread, parent_support(parent)[1])
    return np.linspace(hi / npoints, hi, npoints)


def w1_against_parent(model, parent, scheme, grid):
    """W1 between a fitted model and the parent: the area between two CDFs.

    Both objects are continuous and both expose a CDF, so this is the integral
    of |F_model - F_parent| by the trapezoid rule, which is W1's definition
    directly. The in-sample criterion cannot be computed that way because its
    target is a set of atoms, so it discretizes the model onto the grid instead;
    the two agree to the grid's resolution and `fitting.score_w1_exact` measures
    that gap on the in-sample side.

    Note what the grid leaves out: mass the MODEL puts above the grid's top.
    That is charged separately by `tail_charge`, which is kept separate on
    purpose -- see its docstring.
    """
    d = np.abs(np.asarray(model.cdf(grid), float) - parent.cdf(grid, scheme))
    return float(np.trapezoid(d, grid))


def tail_charge(model, parent, grid, quantile=TAIL_QUANTILE,
                npoints=TAIL_GRID_POINTS):
    """W1 the recovery grid does not see, because the model's tail runs past it.

    Above the grid the parent's CDF is 1 -- the parent has bounded support and
    the grid covers it -- so the integrand is the model's survival function and
    the missing contribution is the mean excess of the model above the grid's
    top. IT IS REPORTED SEPARATELY RATHER THAN FOLDED IN, because that split is
    the answer to a question Stage 2b left open (section 4.9): W1 is nearly
    blind to tail mass, and a model with a standard deviation of 3,281 on data
    whose own is 0.6 passed the fit score and was caught only in the pLCA. This
    column says, per fit, how much of the total distance lives out there.

    Integrated on a log-spaced grid to the model's `quantile`, with the mass
    beyond it returned as `residual_mass` so the omission is stated.
    """
    top = float(grid[-1])
    hi_model = float(np.ravel(model.ppf(quantile))[0])
    if not np.isfinite(hi_model) or hi_model <= top:
        return 0.0, 0.0
    g = np.geomspace(top, hi_model, npoints)
    sf = 1.0 - np.asarray(model.cdf(g), float)
    return float(np.trapezoid(sf, g)), float(1.0 - quantile)


def overlap_area(model, parent, scheme, grid):
    """Overlap area between the fitted density and the parent's: min(f, g).

    The main alternative to W1 in the comparative-LCA literature
    (Prado-Lopez et al. 2014), and the carried-forward item Stage 2c owns. It is
    bounded in [0, 1] with 1 for a perfect fit, so it answers the objection that
    W1 is an unbounded distance whose magnitude depends on the units of the
    dataset. Reported as `1 - OVL` in tables, so that lower is better for both
    it and W1 and a rank means the same thing in either column.

    IT NEEDS A REFERENCE DENSITY, so it exists on the synthetic arm only. The
    empirical target is a set of atoms with no density, and supplying one would
    mean choosing a bin width or a kernel -- and a kernel would score the KDE
    against a KDE. That limitation is the reason W1 remains the study's
    criterion; what this column can do is check that the two criteria rank the
    methods the same way where both are computable.
    """
    f = np.asarray(model.pdf(grid), float)
    g = np.asarray(parent.pdf(grid, scheme), float)
    return float(np.trapezoid(np.minimum(f, g), grid))


def mean_shape_split(model, parent, scheme, grid):
    """(location, shape): how much of the error is the mean, and how much is not.

    `location` is |E[model] - E[parent]|, which is a lower bound on W1. `shape`
    is W1 between the model shifted onto the parent's mean and the parent, so it
    is the error that survives getting the mean right. The two are not additive
    -- the triangle inequality gives total <= location + shape -- and they are
    reported as two columns rather than a partition for that reason.

    It is here to answer one question and not to become a headline: if a
    method's advantage is concentrated in visibly UNIMODAL datasets, which is
    94.9 percent of the empirical arm, it is not coming from multimodality, and
    this says whether it is coming from the mean or from the shape around it.
    """
    Fm = np.asarray(model.cdf(grid), float)
    Fp = parent.cdf(grid, scheme)
    # E[X] = integral of (1 - F) over a support bounded below by zero, which is
    # the grid's own construction, so both means come from the same quadrature
    # and their difference is not contaminated by two different ones.
    mu_m = float(np.trapezoid(1.0 - Fm, grid))
    mu_p = float(np.trapezoid(1.0 - Fp, grid))
    shift = mu_p - mu_m
    Fs = np.asarray(model.cdf(grid - shift), float)
    return abs(shift), float(np.trapezoid(np.abs(Fs - Fp), grid))


def score_recovery(models, x, weights, parent, grid=None, tail=True,
                   scheme_of=None):
    """Every recovery column for one dataset's six fits.

    TWO COMPARISONS, AND THEY ANSWER DIFFERENT QUESTIONS. Reporting only the
    first would be a serious error and reporting only the second would waste the
    parent.

      `w1_parent`  each method against the parent IT IS ESTIMATING: the sampling
                   mixture for a uniform-weighted method, the market-weighted
                   mixture for a variable-weighted one. This says how well each
                   method does its own job, and it is the only fair way to judge
                   the ESTIMATION method, because a uniform-weighted fit was
                   never asked to know anything about market share.

      `w1_market`  every method, both weightings, against the MARKET-WEIGHTED
                   parent. This is the decision-relevant comparison and the only
                   one that can say whether variable weighting helps, because
                   the six are then estimating the SAME quantity. A pLCA of what
                   actually gets built is a statement about the market-weighted
                   population, so that is the target a practitioner needs
                   recovered. A uniform-weighted model pays a BIAS here -- it is
                   estimating the wrong distribution -- and a variable-weighted
                   model pays VARIANCE, because the weights are a noisy
                   Dirichlet draw. Which one wins is the bias-variance question
                   the paper exists to answer, and it cannot be read off
                   `w1_parent`.

      `w1_sampling` the same six against the SAMPLING mixture, for symmetry, so
                   the two common-target comparisons can be read together.

    Returns {method: {...}}.
    """
    scheme_of = scheme_of or PARENT_SCHEME
    grid = recovery_grid(x, weights, parent) if grid is None else grid
    out = {}
    for label, m in models.items():
        scheme = scheme_of[FT_weighting(label)]
        body = w1_against_parent(m, parent, scheme, grid)
        t, residual = tail_charge(m, parent, grid) if tail else (0.0, 0.0)
        loc, shape = mean_shape_split(m, parent, scheme, grid)
        out[label] = dict(w1_parent=body, w1_parent_tail=t,
                          w1_parent_total=body + t,
                          overlap=overlap_area(m, parent, scheme, grid),
                          w1_parent_location=loc, w1_parent_shape=shape,
                          tail_residual_mass=residual,
                          parent_scheme=scheme,
                          w1_market=w1_against_parent(m, parent, 'market', grid),
                          w1_sampling=w1_against_parent(m, parent, 'uniform',
                                                        grid),
                          overlap_market=overlap_area(m, parent, 'market', grid))
    return out


def parent_separation(parent, grid):
    """W1 between the sampling parent and the market-weighted parent.

    The synthetic arm's version of the definitional gap: how far apart the two
    populations are for this dataset, before any model is fitted. A method that
    ignores the weights cannot do better than this against the market parent, so
    it is the floor on a uniform-weighted method's `w1_market` and the scale
    every weighting result has to be read against.
    """
    return float(np.trapezoid(
        np.abs(parent.cdf(grid, 'uniform') - parent.cdf(grid, 'market')), grid))


def FT_weighting(label):
    """'Uniform' or 'Variable' from a PEWT label."""
    return label.split(', ')[1]


def score_recovery_arm(datasets, parents, rng=None, progress=None, **fit_kw):
    """Tidy recovery scores for a whole arm. One row per (dataset, method).

    `datasets` is {name: (values, weights)} and `parents` is
    {name: MixtureParent}. The in-sample W1 is computed here too, on the study's
    own grid, so the old score and the new one sit in the same row and the
    change can be read off rather than joined together later.
    """
    rows = []
    for i, (name, (x, w)) in enumerate(datasets.items()):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        parent = parents[name]
        models, params = FT.fit_pewt(x, w, **fit_kw)
        in_grid = FT.score_grid_open(x, w)
        grid = recovery_grid(x, w, parent)
        rec = score_recovery(models, x, w, parent, grid=grid)
        sep = parent_separation(parent, grid)
        for label in FT.PEWT:
            row = dict(arm='synthetic', dataset=name, n=len(x), method=label,
                       w1=FT.score_w1_model(models[label], x, w, grid=in_grid),
                       parent_separation=sep)
            row.update(rec[label])
            rows.append(row)
        if progress and (i + 1) % progress == 0:
            print(f'  {i + 1}/{len(datasets)}', flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the empirical arm: cross-validation
# ---------------------------------------------------------------------------
#: Below this many values a fit half cannot support six fits, so the dataset has
#: no cross-validated score and is reported as absent rather than as a number
#: computed from two points. Stage 2b used the same threshold for `w1_heldout`.
CV_MIN_N = 10

#: Repeats of the split. Each repeat scores BOTH directions, so the number of
#: fits per dataset per method is twice this. Stage 2b used 2 repeats and
#: reported only the mean; this stage reports the spread across splits, which
#: needs enough of them for the spread to mean something.
CV_REPEATS = 10

#: Share of the values used to FIT. A half-and-half split maximizes the
#: information in the evaluation target, which is the noisier of the two sides;
#: `cv_w1` takes it as an argument so the dependence can be swept.
CV_FIT_FRACTION = 0.5


def cv_splits(n, rng, repeats=CV_REPEATS, fit_fraction=CV_FIT_FRACTION):
    """Index pairs (fit, evaluate) for `repeats` random splits, both directions.

    Both directions of each partition are used, so every value appears in the
    fitting half of exactly half the fits. That halves the variance for a given
    number of fits and costs nothing, because the two directions of one
    permutation are free.
    """
    n_fit = int(round(fit_fraction * n))
    n_fit = min(max(n_fit, 1), n - 1)
    out = []
    for _ in range(repeats):
        order = rng.permutation(n)
        a, b = order[:n_fit], order[n_fit:]
        out.append((a, b))
        out.append((b, a))
    return out


def cv_w1(x, weights, rng, repeats=CV_REPEATS, min_n=CV_MIN_N,
          fit_fraction=CV_FIT_FRACTION, **fit_kw):
    """Cross-validated W1 for all six methods, one row per split.

    All six methods share each split, so the comparison is PAIRED: a difference
    between two methods is a difference between methods and not between two
    random partitions of the data. The evaluation half keeps its own weights,
    renormalized, so the target is the same weighted empirical CDF the in-sample
    score uses, restricted to values the model has not seen.

    Returns a list of dicts, one per (split, method), carrying `split`,
    `direction`, `n_fit`, `n_eval` and `w1_cv`. An empty list below `min_n`.

    HOW TO READ IT, and this is a restriction rather than a caveat. A
    cross-validated score may be compared ACROSS ESTIMATION METHODS WITHIN ONE
    WEIGHTING SCHEME and not across weighting schemes. The market-share weights
    are an exchangeable Dirichlet draw, so they carry no information that
    generalizes from one half of a dataset to the other: the expected
    variable-weighted empirical CDF of a random half IS the unweighted one, and
    a uniform-weighted fit is therefore the better predictor of the held-out
    target by construction. `comparison.add_ranks(within_weighting=True)` is
    what enforces it. The in-sample comparison does not have this problem and
    the paper's weighting claim rests on that one.
    """
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(x)
    if n < min_n:
        return []
    rows = []
    for k, (fit_idx, score_idx) in enumerate(
            cv_splits(n, rng, repeats, fit_fraction)):
        xf, wf = x[fit_idx], weights[fit_idx]
        xs, ws = x[score_idx], weights[score_idx]
        if len(xf) < 3 or len(xs) < 3 or np.ptp(xf) <= 0:
            continue
        try:
            models, _ = FT.fit_pewt(xf, wf / wf.sum(), **fit_kw)
        except Exception:
            continue
        ws = ws / ws.sum()
        grid = FT.score_grid_open(xs, ws)
        for label in FT.PEWT:
            try:
                v = FT.score_w1_model(models[label], xs, ws, grid=grid)
            except Exception:
                continue
            rows.append(dict(split=k // 2, direction=k % 2, method=label,
                             n_fit=len(xf), n_eval=len(xs), w1_cv=float(v)))
    return rows


def cv_loglik(x, weights, rng, repeats=CV_REPEATS, min_n=CV_MIN_N,
              fit_fraction=CV_FIT_FRACTION, floor=1e-300, **fit_kw):
    """Cross-validated mean log density on the held-out half, per method.

    A DENSITY-BASED companion to cross-validated W1, on the arm where the
    overlap area cannot be computed. W1 compares CDFs and is nearly blind to how
    much mass a model puts where the data is not; a held-out log density is
    exactly the opposite, it is dominated by the points the model assigns least
    mass to, and it is unbounded below so one badly placed point can decide a
    dataset. Two criteria with opposite failure modes agreeing is worth more
    than either alone, and where they disagree the disagreement is the finding.

    Weighted by the evaluation half's own weights, so it answers the same
    question the weighted W1 does.
    """
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(x)
    if n < min_n:
        return []
    rows = []
    for k, (fit_idx, score_idx) in enumerate(
            cv_splits(n, rng, repeats, fit_fraction)):
        xf, wf = x[fit_idx], weights[fit_idx]
        xs, ws = x[score_idx], weights[score_idx]
        if len(xf) < 3 or len(xs) < 3 or np.ptp(xf) <= 0:
            continue
        try:
            models, _ = FT.fit_pewt(xf, wf / wf.sum(), **fit_kw)
        except Exception:
            continue
        ws = ws / ws.sum()
        for label in FT.PEWT:
            try:
                d = np.asarray(models[label].pdf(xs), float)
            except Exception:
                continue
            rows.append(dict(split=k // 2, direction=k % 2, method=label,
                             n_fit=len(xf), n_eval=len(xs),
                             loglik_cv=float(np.sum(
                                 ws * np.log(np.maximum(d, floor))))))
    return rows


def cv_arm(datasets, rng, arm, repeats=CV_REPEATS, min_n=CV_MIN_N,
           fit_fraction=CV_FIT_FRACTION, loglik=True, progress=None, **fit_kw):
    """Cross-validate a whole arm. One row per (dataset, method, split, direction).

    The per-split rows are kept rather than averaged here, because the spread
    across splits is a result this stage has to report: it says whether a
    difference between two methods on one dataset is real or is the split.
    """
    rows = []
    for i, (name, (x, w)) in enumerate(datasets.items()):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        got = cv_w1(x, w, rng, repeats, min_n, fit_fraction, **fit_kw)
        if loglik:
            ll = {(r['split'], r['direction'], r['method']): r['loglik_cv']
                  for r in cv_loglik(x, w, rng, repeats, min_n, fit_fraction,
                                     **fit_kw)}
        for r in got:
            row = dict(arm=arm, dataset=name, n=len(x), **r)
            if loglik:
                row['loglik_cv'] = ll.get((r['split'], r['direction'],
                                           r['method']), np.nan)
            rows.append(row)
        if progress and (i + 1) % progress == 0:
            print(f'  {i + 1}/{len(datasets)}', flush=True)
    return pd.DataFrame(rows)


def cv_summary(cv_rows, value='w1_cv'):
    """Per (arm, dataset, method): the mean over splits and the spread across them.

    `sd_across_splits` is the standard deviation of the per-split scores and
    `se` is that over sqrt(number of splits). The point of carrying both is that
    a gap between two methods smaller than `sd_across_splits` is a property of
    the partition and not of the methods, and the paper has to be able to say
    which of its gaps are which.
    """
    g = cv_rows.groupby(['arm', 'dataset', 'n', 'method'], observed=True)[value]
    out = g.agg(['mean', 'std', 'count', 'median']).reset_index()
    out = out.rename(columns={'mean': value, 'std': f'{value}_sd_across_splits',
                              'count': f'{value}_n_splits',
                              'median': f'{value}_median'})
    out[f'{value}_se'] = (out[f'{value}_sd_across_splits']
                          / np.sqrt(out[f'{value}_n_splits'].clip(lower=1)))
    return out


# ---------------------------------------------------------------------------
# decomposition: the model's own error, and the definitional gap
# ---------------------------------------------------------------------------
def decompose_weighting(models, x, weights, grid=None):
    """Split a model's score against the variable-weighted target in two.

    Every model in this study, including the three uniform-weighted ones, is
    scored against the VARIABLE-weighted empirical CDF. For a uniform-weighted
    model that total confounds two different things:

      `w1_own`          the model against the empirical CDF under ITS OWN
                        weighting scheme. This is fit error, and it is the only
                        part the estimation method is responsible for.
      `w1_definitional` the distance between the uniform-weighted and the
                        variable-weighted empirical CDFs of the same values.
                        It is a property of the WEIGHTS and of nothing else, it
                        is identical for all three uniform-weighted methods, and
                        no estimation method can reduce it. It is the study's
                        `w_v_uw_wasserstein` characteristic computed on the
                        scoring grid.

    For a variable-weighted model `w1_own` IS the total and `w1_definitional`
    is zero, which is the point: the comparison between the two weighting
    schemes under this target is a comparison between a model that is charged
    the definitional gap and a model that is not.

    The two are related by the triangle inequality rather than by addition,
    total <= own + definitional, so `w1_slack` is reported as
    own + definitional - total rather than the three being presented as a
    partition.
    """
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    uni = FT.uniform_weights(x)
    grid = FT.score_grid_open(x, weights) if grid is None else grid
    e_var = weighted_ecdf(x, weights)[2](grid)
    e_uni = weighted_ecdf(x, uni)[2](grid)
    definitional = float(np.trapezoid(np.abs(e_uni - e_var), grid))
    out = {}
    for label, m in models.items():
        target_is_uniform = FT_weighting(label) == 'Uniform'
        e_own = e_uni if target_is_uniform else e_var
        F = np.asarray(m.cdf(grid), float)
        own = float(np.trapezoid(np.abs(F - e_own), grid))
        total = float(np.trapezoid(np.abs(F - e_var), grid))
        d = definitional if target_is_uniform else 0.0
        out[label] = dict(w1_total=total, w1_own=own, w1_definitional=d,
                          w1_slack=own + d - total)
    return out


def decompose_arm(datasets, arm, progress=None, **fit_kw):
    """`decompose_weighting` over a whole arm. One row per (dataset, method)."""
    rows = []
    for i, (name, (x, w)) in enumerate(datasets.items()):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        models, _ = FT.fit_pewt(x, w, **fit_kw)
        for label, d in decompose_weighting(models, x, w).items():
            rows.append(dict(arm=arm, dataset=name, n=len(x), method=label, **d))
        if progress and (i + 1) % progress == 0:
            print(f'  {i + 1}/{len(datasets)}', flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# regret
# ---------------------------------------------------------------------------
def add_regret(scores, value='w1', suffix=None, within_weighting=False):
    """Per dataset, a method's score minus the best score on that dataset.

    WHY REGRET AND NOT A WIN RATE. A win rate answers "how often is this method
    best", which is not the question a practitioner has. They will use ONE
    method on every dataset, so what they need to know is what that costs them
    when it is not the best one -- and a method that is second by a hair on
    every dataset is a better default than one that wins half the time and is
    catastrophic on the rest. The upper tail of the regret distribution is where
    that difference shows up, and neither a mean rank nor a win share can see
    it.

    Reported in the score's own units. Divided by the dataset's own mean score
    it would be comparable across datasets but would no longer be a cost; both
    are wanted, so `add_regret_relative` is the other one.
    """
    suffix = suffix or f'{value}_regret'
    keys = ['arm', 'dataset']
    frame = scores.copy()
    if within_weighting:
        frame['_wt'] = frame.method.map(FT_weighting)
        keys = keys + ['_wt']
    best = frame.groupby(keys, observed=True)[value].transform('min')
    frame[suffix] = frame[value] - best
    return frame.drop(columns='_wt') if within_weighting else frame


def add_regret_relative(scores, value='w1', suffix=None,
                        within_weighting=False):
    """Regret over the best score on that dataset, so it reads as a percentage.

    1.0 means the method cost twice the best method's distance on that dataset.
    Scale free, so it can be averaged over datasets whose W1 magnitudes differ
    by orders of magnitude, which the absolute regret cannot.
    """
    suffix = suffix or f'{value}_regret_relative'
    keys = ['arm', 'dataset']
    frame = scores.copy()
    if within_weighting:
        frame['_wt'] = frame.method.map(FT_weighting)
        keys = keys + ['_wt']
    best = frame.groupby(keys, observed=True)[value].transform('min')
    frame[suffix] = (frame[value] - best) / best.replace(0.0, np.nan)
    return frame.drop(columns='_wt') if within_weighting else frame


REGRET_QUANTILES = (0.5, 0.9, 0.95, 1.0)


def regret_table(scores, value='w1', within_weighting=False,
                 quantiles=REGRET_QUANTILES, by=None):
    """Mean, median and upper tail of regret per method.

    `by` adds grouping columns, so the same call gives the arm-level table and
    the size-banded one.
    """
    frame = add_regret(scores, value, within_weighting=within_weighting)
    frame = add_regret_relative(frame, value, within_weighting=within_weighting)
    keys = ['arm', 'method'] + list(by or [])
    rows = []
    for k, g in frame.groupby(keys, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k), n_datasets=int(g.dataset.nunique()))
        for col, tag in ((f'{value}_regret', 'regret'),
                         (f'{value}_regret_relative', 'regret_relative')):
            v = g[col].replace([np.inf, -np.inf], np.nan).dropna()
            row[f'{tag}_mean'] = float(v.mean()) if len(v) else np.nan
            for q in quantiles:
                name = 'max' if q == 1.0 else f'p{int(round(q * 100))}'
                row[f'{tag}_{name}'] = float(v.quantile(q)) if len(v) else np.nan
            row[f'{tag}_zero_share'] = (float((v <= 0).mean()) if len(v)
                                        else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# post-stratification
# ---------------------------------------------------------------------------
#: Size bands, as (label, lo, hi) with hi inclusive. The same four the corpus is
#: stratified by, so a synthetic aggregate can be reweighted to the empirical
#: size mix without either arm being re-binned.
SIZE_BANDS = (('s1_3_9', 3, 9), ('s2_10_99', 10, 99),
              ('s3_100_999', 100, 999), ('s4_1000_9999', 1000, 9999))


def size_band(n, bands=SIZE_BANDS):
    """The band a dataset of `n` values falls in, or None above the last one."""
    for label, lo, hi in bands:
        if lo <= n <= hi:
            return label
    return None


def add_size_band(frame, column='n', bands=SIZE_BANDS):
    return frame.assign(size_band=frame[column].map(lambda v: size_band(v, bands)))


def empirical_size_shares(empirical_scores, bands=SIZE_BANDS):
    """The share of the EMPIRICAL arm in each size band, from the arm itself.

    Measured rather than taken from `genconfig.EMPIRICAL_STRATUM_SHARE`, which
    is a constant recorded when the arm had 149 datasets and is stale whenever
    the arm changes. A reweighting that silently used the wrong denominator
    would move every post-stratified number and leave nothing to catch it.
    """
    one = empirical_scores.drop_duplicates('dataset')
    b = one.n.map(lambda v: size_band(v, bands))
    counts = b.value_counts()
    total = float(counts.sum())
    return {label: float(counts.get(label, 0)) / total for label, _, _ in bands}


def post_stratify(scores, value, shares, by=('arm', 'method'),
                  bands=SIZE_BANDS, statistic='mean'):
    """One aggregate two ways: equal allocation across bands, and reweighted.

    EQUAL ALLOCATION IS A PRECISION CHOICE, NOT A CLAIM ABOUT FREQUENCY. The
    corpus draws 2,500 datasets in each of four size bands so that every size
    regime is estimated with the same precision; the empirical arm's own mix is
    nothing like that. Stage 2b measured 25 percent of the corpus above
    n = 1,000 against 7.4 percent of the empirical arm, and since the KDE
    improves monotonically with n and the lognormal degrades, the unweighted
    corpus mean is a statement about the allocation as much as about the
    methods. Reweighting the per-band aggregates by the empirical shares removes
    that, and the two arms then agree where they appeared to disagree.

    Both columns are reported everywhere. Neither is the true one: the equal
    allocation says what happens in each regime with equal confidence, the
    reweighted one says what happens on a population of datasets that looks like
    the EC3 categories.
    """
    frame = add_size_band(scores, bands=bands)
    keys = list(by)
    per = (frame.groupby(keys + ['size_band'], observed=True)[value]
           .agg(statistic).reset_index())
    rows = []
    for k, g in per.groupby(keys, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        w, v = [], []
        for label, _, _ in bands:
            hit = g[g.size_band == label]
            row[f'{statistic}__{label}'] = (float(hit[value].iloc[0]) if len(hit)
                                            else np.nan)
            if len(hit) and np.isfinite(hit[value].iloc[0]):
                w.append(shares.get(label, 0.0))
                v.append(float(hit[value].iloc[0]))
        sub = frame
        for name, val in zip(keys, k):
            sub = sub[sub[name] == val]
        agg = sub[value].replace([np.inf, -np.inf], np.nan).dropna()
        row[f'{statistic}_equal_allocation'] = (float(getattr(agg, statistic)())
                                                if len(agg) else np.nan)
        w = np.asarray(w, float)
        row[f'{statistic}_post_stratified'] = (
            float(np.sum(np.asarray(v) * w) / w.sum()) if w.sum() > 0 else np.nan)
        row['bands_present'] = int(len(v))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def win_share(scores, value='w1', by=('arm',), within_weighting=False):
    """The share of datasets on which each method has the lowest score.

    STATE THE EMPIRICAL HEADLINE THIS WAY AND NOT AS A MEAN RANK. Stage 2b
    measured the empirical target's own weight-draw noise at a median of 0.1344
    against the best method's median W1 of 0.0984, and over five independent
    Dirichlet realizations `KDE, Variable` and `Lognormal, Variable` sat within
    that noise of each other on mean rank -- 2.243 against 2.353, with the
    lognormal ahead in one of the five. On win share the KDE led in every
    realization, 39 to 45 percent against 25 to 31. A mean rank averages a
    signed distance in rank space and inherits the noise of every dataset; a win
    share is a count, and the noise has to flip a dataset's winner to move it.
    """
    frame = scores.copy()
    keys = list(by)
    if within_weighting:
        frame['_wt'] = frame.method.map(FT_weighting)
        keys = keys + ['_wt']
    idx = (frame.groupby(keys + ['dataset'], observed=True)[value]
           .transform('min') == frame[value])
    wins = frame[idx]
    n = wins.groupby(keys, observed=True)['dataset'].nunique().rename('n_datasets')
    c = (wins.groupby(keys + ['method'], observed=True)['dataset'].nunique()
         .rename('n_wins').reset_index())
    out = c.merge(n.reset_index(), on=keys, how='left')
    out['win_share'] = out.n_wins / out.n_datasets
    if within_weighting:
        out = out.rename(columns={'_wt': 'weighting'})
    return out.sort_values(list(out.columns[:len(keys)]) + ['win_share'],
                           ascending=[True] * len(keys) + [False]
                           ).reset_index(drop=True)

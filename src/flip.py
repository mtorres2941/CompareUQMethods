"""When does a difference between two fitted models change the ANSWER? Stage 2d.

The study scores a UQ method by how far its fitted CDF sits from a target. That
is a statement about fit and not about consequence. This module calibrates the
one against the other: for every pair of UQ methods inside every probabilistic
LCA, how far apart are the two models, and did the downstream decision change?

The deliverable is the curve P(decision flips) against the relative W1 between
the two models, and the relative W1 at which it crosses 1, 5 and 10 percent.

    THE MONTE CARLO NOISE FLOOR IS WHY THIS USES COMMON RANDOM NUMBERS, AND IT
    IS NOT AN OPTIONAL REFINEMENT.

The study's pLCA draws each method's Monte Carlo sample from its own stretch of
one shared stream, so two methods are compared under INDEPENDENT randomness.
Stage 2d measured what that costs by running the same method twice, with the
same fitted models and two independent streams, over 400 groups: the top
contributor changed in 5.33 percent of cases and the full rank ordering in 34.2
percent, with NO model difference whatever. A curve asked to resolve a 1 percent
flip probability cannot be read off data with a 5.33 percent floor under it.

Under common random numbers -- one uniform variate per material per iteration,
pushed through every method's inverse CDF -- two identical models produce
identical draws and the floor is exactly zero by construction. Everything the
curve then measures is the effect of the models differing, which is the question.

`families.rvs_from_uniform` was built in Stage 2b for exactly this and has been
unused since. Note for the next Claude Code session: the roadmap gives common
random numbers to Stage 2e, which owns installing them in the STUDY's pLCA and
moving its published numbers. Nothing here writes or replaces
`outputs/tables/TABLE_PLCAResults.csv`; this is a separate calibration run, and
what Stage 2e inherits is a tested implementation rather than a decision.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit

import fitting

#: Monte Carlo draws per material per pLCA, matching the study's pLCA.
NECCS = 10_000

#: Flip-probability levels the curve is read at.
LEVELS = (0.01, 0.05, 0.10)

#: Resamples for the cluster bootstrap. See `bootstrap_crossings` for why the
#: resampling unit is the pLCA and not the row.
RESAMPLES = 1_000
ALPHA = 0.05


# ---------------------------------------------------------------------------
# the calibration pLCA
# ---------------------------------------------------------------------------
def group_outcomes(models, names, uniforms, methods=None):
    """Run one pLCA group under every method on ONE set of uniform variates.

    Parameters
    ----------
    models : dict
        `models[dataset][method]`, each exposing `rvs_from_uniform`.
    names : sequence
        The datasets in this group, in a fixed order.
    uniforms : ndarray, shape (neccs, len(names))
        Column j is dataset j's uniform stream, SHARED across methods. Each
        column is independent of the others, so the materials are sampled
        independently as the study's pLCA samples them; what is shared is the
        comparison between methods, which is the whole point.

    Returns
    -------
    dict
        `method -> (top_contributor, ordering, rank1_frequencies)`.

    Material use intensity is 1.0 for every material, as everywhere in this
    study, so the contribution of a material IS its sampled ECC.
    """
    methods = methods or fitting.PEWT
    names = list(names)
    out = {}
    for m in methods:
        draws = np.column_stack([
            np.asarray(models[d][m].rvs_from_uniform(uniforms[:, j]), float)
            for j, d in enumerate(names)])
        # Rank 1 is the LARGEST contributor, which is what "ECI Rank #1
        # Frequency" counts.
        order_within = (-draws).argsort(axis=1).argsort(axis=1)
        rank1 = (order_within == 0).mean(axis=0)
        top = names[int(np.argmax(rank1))]
        ordering = tuple(np.asarray(names)[np.argsort(-draws.mean(axis=0))])
        out[m] = (top, ordering, rank1)
    return out


def run_calibration(models, combos, rng, neccs=NECCS, methods=None,
                    progress=None):
    """Every pLCA group under every method, on common random numbers.

    Returns `{plca_index: {method: (top, ordering, rank1)}}`.
    """
    methods = methods or fitting.PEWT
    out = {}
    it = enumerate(combos)
    if progress is not None:
        it = progress(it, total=len(combos))
    for i, g in it:
        u = rng.random((int(neccs), len(g)))
        out[i] = group_outcomes(models, list(g), u, methods=methods)
    return out


# ---------------------------------------------------------------------------
# model-to-model distance
# ---------------------------------------------------------------------------
def pair_distances(models, x, scales, methods=None, npoints=None):
    """Relative W1 between every pair of fitted models on one dataset.

    The grid is `fitting.score_grid_open`, the study's own scoring lattice at
    its converged resolution, so a distance here is measured the same way every
    other W1 in this study is. The empirical CDF does not appear: both objects
    are models.
    """
    from itertools import combinations

    import weighting as WG

    methods = methods or fitting.PEWT
    x = np.asarray(x, dtype=float)
    grid = fitting.score_grid_open(
        x, WG.uniform_weights(x),
        npoints=npoints or fitting.SCORE_GRID_POINTS)
    cdfs = {m: np.asarray(models[m].cdf(grid), dtype=float) for m in methods}
    out = {}
    for a, b in combinations(methods, 2):
        w1 = float(np.trapezoid(np.abs(cdfs[a] - cdfs[b]), grid))
        out[(a, b)] = dict(w1=w1, **WG.relativize(w1, scales))
    return out


def calibration_frame(outcomes, distances, combos, methods=None):
    """One row per (pLCA, method pair): how far apart, and did the answer change.

    `distances[dataset][(a, b)]` comes from `pair_distances`.

    THE PREDICTOR HAS TO BE AGGREGATED OVER THE FOUR MATERIALS, because the
    decision is a property of the group while the distance is a property of a
    material. Both the maximum and the mean are carried, and
    `audits/flip_calibration.py` reports which separates the flips better rather
    than assuming. The maximum is the natural candidate: a ranking changes
    because ONE material moved past another, so the largest single disagreement
    is what has the chance to do it.
    """
    from itertools import combinations

    methods = methods or fitting.PEWT
    rows = []
    for i, g in enumerate(combos):
        names = list(g)
        if i not in outcomes:
            continue
        per = outcomes[i]
        for a, b in combinations(methods, 2):
            d = [distances[n][(a, b)] for n in names]
            row = dict(plca=i, method_a=a, method_b=b,
                       flip_top=per[a][0] != per[b][0],
                       flip_order=per[a][1] != per[b][1],
                       rank1_shift=float(np.abs(per[a][2] - per[b][2]).max()))
            for key in d[0]:
                v = np.array([e[key] for e in d], dtype=float)
                row[f'{key}_max'] = float(v.max())
                row[f'{key}_mean'] = float(v.mean())
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the curve
# ---------------------------------------------------------------------------
#: Distances at or below this are treated as zero when taking a logarithm. Two
#: methods can be numerically identical -- the two normals coincide when the
#: variable weights happen to be near-uniform -- and log(0) is not a data point.
LOG_FLOOR = 1e-12


def logistic_fit(x, y):
    """Logistic regression of a binary outcome on log(distance).

    LOG, NOT THE RAW DISTANCE, and it is not a cosmetic choice. The relative W1
    between two fitted models spans about four orders of magnitude across the
    corpus, and the crossings being asked for -- 1, 5 and 10 percent -- all sit
    in the lower part of that range. On a linear predictor a handful of very
    separated pairs set the slope and the fit is worthless exactly where it is
    read.

    Returns (intercept, slope) on the logit scale, by Newton-Raphson on the
    exact gradient and Hessian. Written out rather than taken from a library so
    that the model is visible in the file the paper is defended from.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & (x > LOG_FLOOR) & np.isfinite(y)
    z = np.log(x[keep])
    t = y[keep]
    X = np.column_stack([np.ones_like(z), z])
    beta = np.zeros(2)
    for _ in range(100):
        p = expit(X @ beta)
        w = np.clip(p * (1 - p), 1e-12, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (t - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def logistic_crossing(beta, level):
    """The distance at which the fitted logistic curve reaches `level`."""
    a, b = beta
    if not np.isfinite(b) or b == 0:
        return np.nan
    return float(np.exp((logit(level) - a) / b))


def isotonic_curve(x, y):
    """Pool-adjacent-violators: the monotone step function closest to the data.

    The logistic is a two-parameter shape imposed on the data; this imposes only
    that flip probability does not DECREASE as the two models move further
    apart, which is the one thing the mechanism guarantees. Reporting both is
    what says whether the crossings are a property of the data or of the link
    function.

    Returns (x_sorted, fitted_probability), both of length len(x).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    o = np.argsort(x, kind='mergesort')
    x, y = x[o], y[o]
    # Each block carries (sum of y, number of points). Merge backwards while the
    # previous block's level exceeds this one's.
    #
    # Pop into locals FIRST. `total[-2] += total.pop()` evaluates the pop before
    # the assignment, so the target index is taken on the ALREADY-SHORTENED list
    # and the merge lands on the wrong block, silently dropping a point. Written
    # that way this returned a curve one point short of its input; the length
    # and mean-preservation assertions in `tests/test_flip.py` are there because
    # of it.
    total, count = [], []
    for yi in y:
        total.append(float(yi))
        count.append(1)
        while len(total) > 1 and total[-2] / count[-2] > total[-1] / count[-1]:
            t, c = total.pop(), count.pop()
            total[-1] += t
            count[-1] += c
    fitted = np.repeat([t / c for t, c in zip(total, count)], count)
    return x, fitted


def isotonic_crossing(x, y, level):
    """The smallest distance at which the isotonic fit reaches `level`."""
    xs, p = isotonic_curve(x, y)
    hit = np.flatnonzero(p >= level)
    if len(hit) == 0:
        return np.nan
    return float(xs[hit[0]])


def bootstrap_crossings(frame, predictor, outcome, levels=LEVELS,
                        resamples=RESAMPLES, alpha=ALPHA, rng=None,
                        cluster='plca'):
    """Crossings with a percentile interval, resampling pLCAs and not rows.

    THE RESAMPLING UNIT IS THE pLCA GROUP. The fifteen method pairs inside one
    group share four datasets, four fitted model sets and one set of uniform
    variates, so their outcomes are strongly dependent. Resampling rows would
    treat fifteen dependent observations as fifteen independent ones and return
    an interval roughly a factor of four too narrow. `tests/test_flip.py` pins
    the difference.
    """
    rng = rng or np.random.default_rng(0)
    frame = frame[np.isfinite(frame[predictor]) &
                  (frame[predictor] > LOG_FLOOR)]
    groups = frame[cluster].to_numpy()
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}

    def crossings(sub):
        beta = logistic_fit(sub[predictor], sub[outcome])
        return [logistic_crossing(beta, lv) for lv in levels]

    point = crossings(frame)
    draws = np.empty((resamples, len(levels)), dtype=float)
    for r in range(resamples):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[g] for g in pick])
        draws[r] = crossings(frame.iloc[rows])
    lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    iso = [isotonic_crossing(frame[predictor], frame[outcome], lv)
           for lv in levels]
    return pd.DataFrame(dict(level=list(levels), crossing=point,
                             ci_lo=lo, ci_hi=hi, crossing_isotonic=iso,
                             n_rows=len(frame), n_clusters=len(uniq)))


def binned_curve(frame, predictor, outcome, bins=30, cluster='plca'):
    """The observed flip rate in equal-count bins of the predictor.

    What the figure plots under the fitted curve, so a reader can see whether
    the logistic shape is doing any work. Plain binomial standard errors on the
    bin, which understate the spread for the same clustering reason
    `bootstrap_crossings` documents; they are a visual guide and no claim rests
    on them.
    """
    f = frame[np.isfinite(frame[predictor]) &
              (frame[predictor] > LOG_FLOOR)].copy()
    f['_bin'] = pd.qcut(f[predictor].rank(method='first'), bins, labels=False)
    g = f.groupby('_bin', observed=True)
    out = g.agg(x_median=(predictor, 'median'), x_lo=(predictor, 'min'),
                x_hi=(predictor, 'max'), rate=(outcome, 'mean'),
                n=(outcome, 'size'), n_clusters=(cluster, 'nunique'))
    out['se'] = np.sqrt(out.rate * (1 - out.rate) / out.n)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# the calibration set, and why the six method pairs are not enough on their own
# ---------------------------------------------------------------------------
#: How far a calibration draw is moved from uniform weights toward a Dirichlet
#: draw: `w(t) = (1 - t) * uniform + t * dirichlet`. t = 1 is the study's own
#: variable weighting and t = 0 is no reweighting at all.
#:
#: WHY THIS EXISTS, AND IT IS NOT A CHOICE OF CONVENIENCE. The curve is asked for
#: the distance at which the flip probability crosses 1, 5 and 10 percent. The
#: six UQ methods of this study cannot answer that, because they never get close
#: enough to each other: across the corpus the SMALLEST relative W1 between any
#: two of the six is about 0.012, and the flip rate is already 15 percent there.
#: Every level being asked about lies below the observed data, so a logistic
#: read at 1 percent would be pure extrapolation, and the isotonic fit reports
#: the same crossing for all three levels because its first block is already
#: above the top of them.
#:
#: Tempering supplies the missing region. It is a calibration DEVICE and not a
#: claim about reality: it generates pairs of fitted models at controlled
#: separations running continuously down to zero, so the flip probability can be
#: estimated where it is small. Whether the resulting curve is about the
#: DISTANCE rather than about how the distance was produced is then a testable
#: question, and `audits/flip_calibration.py` tests it by checking that the six
#: real method pairs fall on the curve in the range where the two overlap.
TEMPER_LEVELS = (0.0, 0.02, 0.04, 0.07, 0.12, 0.2, 0.35, 0.6, 1.0)


def tempered_weights(n, draw, t):
    """Uniform weights moved a fraction `t` of the way toward a Dirichlet draw."""
    return (1.0 - t) / n + t * np.asarray(draw, dtype=float)


def weighting_calibration(datasets, combos, rng, tempers=TEMPER_LEVELS,
                          neccs=NECCS, bw_method=None, progress=None,
                          alpha=1.0):
    """Flip probability against separation, over a controlled range of separations.

    For each pLCA group: fit the KDE under UNIFORM weights on all four
    materials, draw one flat-Dirichlet market share per material, and then, for
    each tempering level, fit the KDE under the tempered weights and ask whether
    the pLCA's answer changed. Every comparison inside a group shares ONE set of
    uniform variates, so the Monte Carlo floor is zero and `t = 0` is a check on
    that rather than a data point.

    The uniform-weighted model is the one a practitioner builds when they do not
    know the market shares, and `t = 1` is a market share this study considers
    possible. So the largest tempering level is not a device at all: it is
    exactly the question "how safe is my uniform assumption for this pLCA".
    """
    import weighting as WG

    bw_method = bw_method or fitting.BW_METHOD
    rows = []
    it = enumerate(combos)
    if progress is not None:
        it = progress(it, total=len(combos))
    for i, g in it:
        names = list(g)
        xs = [np.asarray(datasets[d]['data'], dtype=float) for d in names]
        scales = [WG.relative_scales(x) for x in xs]
        grids = [fitting.score_grid_open(x, WG.uniform_weights(x)) for x in xs]
        # `fitting.fit_kde` and not a bare WeightedKDE: the study's KDE is an
        # explicit truncation to (0, inf), renormalized, and `rvs_from_uniform`
        # lives on that wrapper. Building the kernel estimate directly here
        # would calibrate against an object the study never samples from.
        uni = [fitting.fit_kde(x, WG.uniform_weights(x),
                               bw_method=bw_method)[0] for x in xs]
        ucdf = [np.asarray(m.cdf(gr), float) for m, gr in zip(uni, grids)]
        draws = [rng.dirichlet(np.full(len(x), float(alpha))) for x in xs]
        u = rng.random((int(neccs), len(names)))
        base = _outcome_from(uni, names, u)
        for t in tempers:
            w = [tempered_weights(len(x), d, t) for x, d in zip(xs, draws)]
            mods = [fitting.fit_kde(x, wi, bw_method=bw_method)[0]
                    for x, wi in zip(xs, w)]
            sep = np.array([
                float(np.trapezoid(np.abs(np.asarray(m.cdf(gr), float) - c), gr))
                for m, gr, c in zip(mods, grids, ucdf)])
            got = _outcome_from(mods, names, u)
            row = dict(plca=i, temper=t, n_min=int(min(len(x) for x in xs)),
                       flip_top=base[0] != got[0], flip_order=base[1] != got[1],
                       rank1_shift=float(np.abs(base[2] - got[2]).max()),
                       w1_max=float(sep.max()), w1_mean=float(sep.mean()))
            for k in WG.SCALES:
                rel = np.array([s / sc[k] if sc[k] > 0 else np.nan
                                for s, sc in zip(sep, scales)])
                row[f'rel_{k}_max'] = float(np.nanmax(rel))
                row[f'rel_{k}_mean'] = float(np.nanmean(rel))
            rows.append(row)
    return pd.DataFrame(rows)


def _outcome_from(models, names, uniforms):
    draws = np.column_stack([
        np.asarray(m.rvs_from_uniform(uniforms[:, j]), float)
        for j, m in enumerate(models)])
    rank1 = ((-draws).argsort(axis=1).argsort(axis=1) == 0).mean(axis=0)
    return (names[int(np.argmax(rank1))],
            tuple(np.asarray(names)[np.argsort(-draws.mean(axis=0))]), rank1)


# ---------------------------------------------------------------------------
# what the curve settled
# ---------------------------------------------------------------------------
#: The relative W1 at which the probability of a changed top contributor
#: crosses 1, 5 and 10 percent. In units of the dataset's own unweighted mean,
#: which is what every W1 this study reports is already in.
#:
#: MEASURED, not chosen. 2,500 pLCA groups by nine tempering levels, under
#: common random numbers so the Monte Carlo floor is zero, with a logistic fit
#: on log distance and a percentile interval from a bootstrap that resamples
#: pLCA GROUPS. The intervals are [0.00109, 0.00199], [0.00831, 0.01177] and
#: [0.02049, 0.02652]; an isotonic fit, which assumes only that the probability
#: does not fall as the models separate, gives 0.0026, 0.0133 and 0.0226.
#:
#: WHY THEY LIVE HERE AS CONSTANTS. Notebook 1 needs them to turn a per-dataset
#: weighting risk into a probability, and notebook 3 is what computes them.
#: Hard-coding the calibrated value and having notebook 3 print the recomputed
#: crossings beside it breaks that circularity and makes any drift visible,
#: which is the same pattern `customstats.SILVERMAN_MIN_NEFF` follows.
#:
#: READ THE LEVELS AS A PROPERTY OF THIS STUDY'S pLCA, NOT OF A BUILDING. Every
#: material here is normalized to a mean of 1.0 and carries a material use
#: intensity of 1.0, so the four contributions are nearly exchangeable and their
#: ranking is as fragile as it can be made. A real building, where materials
#: differ by orders of magnitude in contribution, is harder to flip. These
#: numbers are therefore an upper bound on how often a weighting choice changes
#: an answer, which is the conservative direction for a practitioner rule.
FLIP_THRESHOLDS = {0.01: 0.00149, 0.05: 0.00991, 0.10: 0.02334}

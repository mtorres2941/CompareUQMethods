"""Which downstream metric the paper should lead with. Stage 2g.

The study's headline has always been ECI Rank #1 Frequency: the share of Monte
Carlo iterations in which a material is the largest contributor. Three separate
arguments now say that is the wrong thing to lead with, and this module is what
replaces it.

    IT IS FRAGILE. Four materials each normalized to a mean of 1.0 and each
    carrying a use intensity of 1.0 are exchangeable, so the four frequencies
    sit near 0.25 and their ORDER is decided by the tails. Stage 2d.

    IT CARRIES A NOISE FLOOR. Under independent random streams the argmax of
    four near-equal frequencies lands elsewhere 3.67 percent of the time with
    no model difference at all. Common random numbers set that floor to zero
    for a COMPARISON and cannot help a statement made from one run. Stage 2e.

    IT IS NOT A PROPERTY OF THE MATERIAL. The error in a material's rank-1
    frequency is 9 percent predictable from that material's own dataset,
    against 62 to 66 percent for the error in its estimated contribution,
    because a rank-1 frequency is a property of the GROUP the material was
    placed in. Stage 2f.

    AND NONE OF THOSE IS THE DECIDING ONE.

Stage 2e ran the pLCA against the TRUE parent distributions, so the question a
metric has to answer is no longer "is this sensitive" but "does this recover
the right answer". `recovery_table` is that measurement and it is what ranks
the candidates here. A metric earns the headline by being one a fitted model
gets close to the truth on, not by being stable, and the two are different: a
metric every method agrees on and every method is wrong about is worse than a
metric the methods disagree about and one of them gets right.

    TWO STATISTICS, AND THEY ARE THE SAME DIVISION ON TWO NUMERATORS.

`nrmse`, which this study already reports, is the root mean squared difference
BETWEEN two methods over the spread of the metric across every material and
method. The recovery error here is the mean absolute difference between a
method and the TRUTH over that same spread. So they are directly comparable and
the pair says something neither says alone: a low NRMSE with a high recovery
error is a metric on which all six methods agree and all six are wrong.

    AND A DECISION AGREEMENT BESIDE BOTH.

Every one of these metrics is read as an argmax at some point -- which material
leads, which contributes most, which holds the most share when the building
lands at its bad end. `decision_agreement` is how often the method's argmax is
the truth's argmax, which is the form a practitioner actually uses and the form
in which two metrics can be compared without any units at all.

    THE MAGNITUDE COMPANIONS.

`eci_perc_mean`, each material's mean share of the building total, already
existed and was never compared against the rank metric. `share_at_total_quantile`
is new: each material's share of the total in the iterations where the TOTAL
sits at its 95th percentile, which is the attribution question asked at the end
of the distribution a carbon budget is written against. It is not the same as
`eci_p95`, which is the 95th percentile of the material's OWN contribution and
says nothing about whether that material is what puts the BUILDING at its bad
end.

    THE STRATEGY RANK FREQUENCIES, AND THE DIVISOR THAT NORMALIZED THEM.

`strategy_rank_frequencies` replaces a `1 / (1 - capecc)` divisor that was
exact only while the specification cap was each method's own 75th percentile
and therefore bound in exactly 25 percent of iterations for every material.
Stage 2e made the cap absolute and had to drop the divisor with it, leaving a
plain count over all iterations whose four columns sum to something between
0.31 and 1.00 rather than to 1. The correct normalization is the one the old
divisor was reaching for: divide by the iterations in which the strategy
APPLIES, and report the applicability as its own quantity, because under an
absolute cap it is a property of the material and the method rather than a
constant.

    THE TAIL FAILURE MODE.

W1 is an area between CDFs and is nearly blind to a thin far tail; a Monte
Carlo samples from the model and is not. `tail_stress` contaminates a fitted
model with a small amount of mass a long way out and reports what that costs in
W1 beside what it costs in every pLCA output, so a recommended metric is
checked against the failure mode rather than assumed immune to it.
"""
import numpy as np
import pandas as pd

import fitting as FT
import plca as PL


# ---------------------------------------------------------------------------
# the new magnitude companion
# ---------------------------------------------------------------------------
# `share_at_total_quantile` itself lives in `src/plca.py`, because
# `plca.outputs` has to compute it and this module already imports that one.
# It is named here so a reader of this module can find it.
share_at_total_quantile = PL.share_at_total_quantile
TOTAL_QUANTILE = PL.TOTAL_QUANTILE
TOTAL_WINDOW = PL.TOTAL_WINDOW


# ---------------------------------------------------------------------------
# the strategy rank frequencies, and the divisor
# ---------------------------------------------------------------------------
def strategy_rank_frequencies(reduction):
    """Rank frequencies for a reduction strategy that does not always apply.

    Parameters
    ----------
    reduction : ndarray, shape (neccs, k)
        The change in the building total from applying the strategy to material
        j in iteration i. NEGATIVE is a saving. **NaN where the strategy does
        not apply to that material in that iteration**, which for a
        specification cap means no draw reached the cap.

    Returns
    -------
    dict with
        `applies`     the share of ALL iterations in which the strategy applies
                      to at least one material. This is the denominator, and it
                      is reported rather than assumed.
        `rank_k`      shape (k,), for k = 1 .. k: among the iterations in which
                      the strategy applies to something, the share in which
                      material j is the k-th most effective of the materials it
                      applies to. **`rank_1` sums to exactly 1.0 across the
                      materials**, which is what makes it a frequency
                      comparable with `eci_rank_1` and `matred_rank_1`.
        `bound_share` shape (k,), the share of ALL iterations in which the
                      strategy applies to material j itself.

    WHY THE DENOMINATOR IS THE APPLICABLE ITERATIONS AND NOT ALL OF THEM.

    The question the metric answers is "if I can pursue one specification cap,
    which material should I cap". In an iteration where no material's cap binds
    there is no answer: every choice delivers nothing, and counting that
    iteration against all four materials makes the columns depend on how often
    the strategy applies rather than on which material is the right one. That
    is the same conflation the old `1 / (1 - capecc)` divisor was written to
    remove; what was wrong with the divisor is that it used a constant 0.25,
    which was the applicable share only while the cap was each method's own
    75th percentile and so bound in exactly a quarter of iterations for every
    material by construction. Under the absolute cap Stage 2e installed, the
    applicable share runs from about 0.31 to 0.40 per material, it differs
    between methods, and that difference is signal: a method that understates
    the upper tail should find the cap binding less often.

    SO THE APPLICABILITY IS NOT DIVIDED AWAY, IT IS REPORTED. `applies` and
    `bound_share` carry it, and they are the quantities the old construction
    destroyed by forcing them to a constant.

    WHY THE RANKING IS AMONG THE MATERIALS THE STRATEGY APPLIES TO. A material
    the cap does not bind delivers a saving of exactly zero, and any material
    it does bind delivers a strictly negative one, so the best of the binding
    materials IS the best of all of them and `rank_1` is unambiguous. The lower
    ranks are read "second best of those that bound", which is stated rather
    than hidden: filling the non-binding materials with zero instead puts two
    or three exact ties at the bottom, and an averaged tied rank of 3.5 belongs
    to no integer column, so those iterations would vanish from every column
    rather than appear as the shortfall.
    """
    reduction = np.asarray(reduction, dtype=float)
    neccs, k = reduction.shape
    ok = np.isfinite(reduction)
    any_ok = ok.any(axis=1)
    n_applicable = int(any_ok.sum())
    out = dict(applies=float(n_applicable) / neccs if neccs else np.nan,
               n_applicable=n_applicable, n_iterations=int(neccs),
               bound_share=ok.mean(axis=0))
    ranks = np.full((k,), np.nan)
    freq = np.zeros((k, k), dtype=float)  # [rank index, material]
    if n_applicable:
        sub = reduction[any_ok]
        # Ascending rank: the most negative change is the largest saving and
        # takes rank 1. NaN never wins a comparison, so a non-applicable
        # material is pushed past every applicable one and then masked out.
        filled = np.where(np.isfinite(sub), sub, np.inf)
        order = filled.argsort(axis=1, kind='stable').argsort(axis=1)
        valid = np.isfinite(sub)
        for r in range(k):
            freq[r] = ((order == r) & valid).sum(axis=0) / n_applicable
    for r in range(k):
        out[f'rank_{r + 1}'] = freq[r]
    out['ranks'] = ranks
    return out


# ---------------------------------------------------------------------------
# does the metric recover the right answer
# ---------------------------------------------------------------------------
#: Every candidate the stage judges. The first is the study's current headline.
CANDIDATES = ('eci_rank_1', 'eci_mean', 'eci_perc_mean', 'eci_perc_p95tot',
              'eci_p95', 'eci_std', 'ui')

#: Which of them are magnitude-based companions, as opposed to rank-based.
MAGNITUDE_CANDIDATES = ('eci_mean', 'eci_perc_mean', 'eci_perc_p95tot',
                        'eci_p95', 'eci_std', 'ui')


#: The dataset-size bands the corpus is stratified over, as (low, high) with
#: None for "no upper bound". These are the generator's own strata, so each
#: holds 2,500 of the 10,000 datasets by construction rather than by a cut
#: chosen here.
SIZE_BANDS = (('3-9', 3, 9), ('10-99', 10, 99), ('100-999', 100, 999),
              ('1000+', 1000, None))


def size_band_recovery(truth, outputs=CANDIDATES, method='method', n='n',
                       truth_parent='truth_parent', parent='market',
                       bands=SIZE_BANDS):
    """Per (size band, method, output): the error as a pct of the true LEVEL.

    WHY THIS EXISTS, and it is the figure's own caveat made measurable. The
    claim scorecard pools every dataset size, so counting which method is
    closest on the most rows reads as a verdict between the families. It is
    not: the corpus allocates 2,500 datasets to each of four size bands
    (decision 19), so half of every pLCA sits below 100 declarations, which is
    where a three-parameter lognormal is already established to beat a kernel
    estimate. Split by band, the ordering INVERTS -- and so does the weighting.

    The divisor is the output's true level taken over the WHOLE arm, not within
    the band, so the four rows of a column are on one scale and can be read
    down as well as across. Using each band's own level would make a band with
    a smaller true value look better for free.

    Returns a tidy frame with one row per (band, method, output) carrying
    `rel_error`, plus `n_materials` so a thin band is visible as one.
    """
    work = truth[truth[truth_parent] == parent] if truth_parent in truth else truth
    rows = []
    for out in outputs:
        col, tcol = f'{out}__error', f'{out}__truth'
        if col not in work.columns or tcol not in work.columns:
            continue
        level = abs(float(np.nanmean(
            pd.to_numeric(work[tcol], errors='coerce').to_numpy(float))))
        if not level > 0:
            continue
        for label, lo, hi in bands:
            pick = work[n] >= lo
            if hi is not None:
                pick &= work[n] <= hi
            band = work[pick]
            if band.empty:
                continue
            for name, sub in band.groupby(method, sort=True):
                err = pd.to_numeric(sub[col], errors='coerce').abs()
                rat = unit_ratios(sub, out)
                rows.append(dict(band=label, method=name, output=out,
                                 rel_error=float(err.mean() / level),
                                 median_error=float(np.nanmedian(rat))
                                 if rat.notna().any() else np.nan,
                                 truth_level=level, n_materials=int(len(sub))))
    return pd.DataFrame(rows)


def recovery_table(truth, outputs=CANDIDATES, method='method',
                   cluster='plca', truth_parent='truth_parent',
                   resamples=400, rng=None):
    """Per (truth parent, output, method): how close the answer is to the truth.

    `truth` is the tidy run against the true parents, one row per
    (pLCA, material, method), carrying `<output>`, `<output>__truth` and
    `<output>__error` for every output.

    Returns one row per (truth parent, output, method) with

        `abs_error`      mean |method - truth| over materials, with a cluster
                         bootstrap interval over pLCA groups
        `truth_sd`       the standard deviation of the TRUE value across every
                         material in the arm, which is the spread the metric
                         exists to reveal
        `recovery`       `abs_error / truth_sd`. **This is the statistic that
                         ranks the candidates.** Below 1 the method's error is
                         smaller than the differences between materials the
                         metric is being used to detect; at or above 1 it is
                         not, and the metric cannot distinguish two materials
                         at all under that method.
        `bias`           mean signed error over `truth_sd`, so it is in the
                         same units as `recovery` and the two can be read
                         together: a `recovery` made mostly of `bias` is a
                         systematic error that ADDS over the materials of a
                         building, which is the failure Stage 2e measured.
        `truth_mean`     the mean of the TRUE value over every material in the
                         arm, which is the LEVEL of the thing being estimated
                         rather than its spread
        `rel_error`      `abs_error / abs(truth_mean)`, with `rel_error_lo`
                         and `rel_error_hi`. **This is the statistic the claim
                         scorecard uses**, because it is the one definition
                         that can also be written for a building total, a
                         strategy's saving and a design comparison, none of
                         which has a between-material spread at all.

    TWO DIVISORS, TWO QUESTIONS, AND THEY MUST NOT APPEAR ON ONE AXIS.

    `recovery` divides by the SPREAD and answers "can this metric tell two
    materials apart under this method", which is what ranks a candidate metric
    for the headline. `rel_error` divides by the LEVEL and answers "how wrong
    is this number", which is what compares one claim against another. They
    are not the same unit and the ratio between them is not a constant: across
    the seven per-material outputs here the level is 1.17 to 6.57 times the
    spread, so a figure mixing the two is not comparing like with like even
    within one block of rows.

    THE SPREAD DIVISOR IS THE TRUTH'S AND NOT THE METHOD'S. Dividing by the
    method's own spread would let a method that compresses every material
    toward the mean improve its score by being less informative, which is
    exactly backwards. The truth's spread is one number per (arm, output) and
    is the same for all six methods, so the six are comparable and the column
    is a pure measure of error.

    `rel_error` IS A RATIO OF MEANS AND NOT A MEAN OF RATIOS, which is the
    ordinary mean absolute percentage error and is not usable here. The true
    uncertainty index reaches -0.000671 and 2,904 of 60,000 materials carry a
    true value below a hundredth of the mean, so a per-material ratio is
    unbounded and, where the truth is negative, signless. Dividing the mean
    absolute error by the mean true level is stable, is defined for every row,
    and is what the magnitude and action rows were already doing.
    """
    rng = rng or np.random.default_rng(0)
    rows = []
    for parent, block in truth.groupby(truth_parent, sort=True):
        for out in outputs:
            col, tcol = out, f'{out}__truth'
            if col not in block.columns or tcol not in block.columns:
                continue
            tvals = pd.to_numeric(block[tcol], errors='coerce').to_numpy(float)
            sd = float(np.nanstd(tvals))
            lvl = float(np.nanmean(tvals))
            for name, sub in block.groupby(method, sort=True):
                err = pd.to_numeric(sub[f'{out}__error'], errors='coerce')
                work = sub.assign(_abs=err.abs(), _signed=err)
                got = PL.cluster_bootstrap(work, '_abs', cluster=cluster,
                                           statistic='mean',
                                           resamples=resamples, rng=rng)
                rows.append(dict(
                    truth_parent=parent, output=out, method=name,
                    abs_error=got['statistic'], abs_error_lo=got['ci_lo'],
                    abs_error_hi=got['ci_hi'],
                    bias_raw=float(np.nanmean(work['_signed'])),
                    truth_sd=sd, truth_mean=lvl,
                    n=got['n'], n_clusters=got['n_clusters']))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for c, src in (('recovery', 'abs_error'), ('recovery_lo', 'abs_error_lo'),
                   ('recovery_hi', 'abs_error_hi'), ('bias', 'bias_raw')):
        frame[c] = frame[src] / frame['truth_sd']
    for c, src in (('rel_error', 'abs_error'), ('rel_error_lo', 'abs_error_lo'),
                   ('rel_error_hi', 'abs_error_hi')):
        frame[c] = frame[src] / frame['truth_mean'].abs()
    return frame


def decision_agreement(truth, outputs=CANDIDATES, method='method',
                       cluster='plca', truth_parent='truth_parent',
                       resamples=400, rng=None):
    """How often the method's argmax of a metric is the TRUTH's argmax.

    Every metric here is read at some point as "which material is the biggest",
    and this is that reading scored against the right answer. It needs no units
    and no divisor, so it is the one comparison between two candidate metrics
    that carries nothing of either metric's scale.

    Returns one row per (truth parent, output, method) with the share of pLCA
    groups on which the two agree, a cluster bootstrap interval, and the chance
    level `1 / k` for the group size, so an agreement can be read against what
    guessing would give.
    """
    rng = rng or np.random.default_rng(0)
    rows = []
    for parent, block in truth.groupby(truth_parent, sort=True):
        for out in outputs:
            col, tcol = out, f'{out}__truth'
            if col not in block.columns or tcol not in block.columns:
                continue
            for name, sub in block.groupby(method, sort=True):
                per = []
                for pl, grp in sub.groupby(cluster, sort=True):
                    v = pd.to_numeric(grp[col], errors='coerce').to_numpy(float)
                    t = pd.to_numeric(grp[tcol], errors='coerce').to_numpy(float)
                    if not (np.isfinite(v).any() and np.isfinite(t).any()):
                        continue
                    per.append((pl, float(np.nanargmax(v) == np.nanargmax(t)),
                                len(v)))
                if not per:
                    continue
                frame = pd.DataFrame(per, columns=[cluster, 'agree', 'k'])
                got = PL.cluster_bootstrap(frame, 'agree', cluster=cluster,
                                           statistic='mean',
                                           resamples=resamples, rng=rng)
                rows.append(dict(
                    truth_parent=parent, output=out, method=name,
                    agreement=got['statistic'], agreement_lo=got['ci_lo'],
                    agreement_hi=got['ci_hi'],
                    chance=float(1.0 / frame['k'].mean()),
                    n_plca=got['n_clusters']))
    return pd.DataFrame(rows)


def metric_verdict(recovery, agreement, nrmse_table=None,
                   truth_parent='market'):
    """One row per candidate metric: recovery, agreement and spread together.

    The three answer different questions and the paper needs all three in one
    place, because the argument for demoting a metric is never one of them
    alone:

        `recovery_best` / `recovery_worst`   how close to the truth the best
            and worst of the six methods get, in units of the metric's own
            between-material spread. **This is the column that decides.**
        `agreement_best` / `agreement_worst` how often the argmax reading is
            the truth's, against `chance`
        `nrmse`  how far apart the six methods sit on the metric, which is what
            the study reported before it could compare anything with the truth

    A metric with a low NRMSE and a high recovery error is one every method
    agrees on and every method gets wrong, and reporting it as stable would be
    the most misleading thing this study could do. The join is what makes that
    case visible.
    """
    rec = recovery[recovery['truth_parent'] == truth_parent]
    agr = agreement[agreement['truth_parent'] == truth_parent]
    rows = []
    for out in sorted(set(rec['output'])):
        r = rec[rec['output'] == out]
        a = agr[agr['output'] == out]
        row = dict(
            output=out,
            truth_sd=float(r['truth_sd'].iloc[0]),
            recovery_best=float(r['recovery'].min()),
            recovery_worst=float(r['recovery'].max()),
            recovery_best_method=str(
                r.loc[r['recovery'].idxmin(), 'method']),
            bias_abs_max=float(r['bias'].abs().max()))
        if len(a):
            row.update(agreement_best=float(a['agreement'].max()),
                       agreement_worst=float(a['agreement'].min()),
                       agreement_best_method=str(
                           a.loc[a['agreement'].idxmax(), 'method']),
                       chance=float(a['chance'].iloc[0]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if nrmse_table is not None and len(frame):
        frame = frame.merge(
            nrmse_table[['output', 'nrmse']].drop_duplicates('output'),
            on='output', how='left')
    return frame.sort_values('recovery_best').reset_index(drop=True)


# ---------------------------------------------------------------------------
# the tail failure mode
# ---------------------------------------------------------------------------
class Contaminated:
    """A fitted model with a little mass moved a long way out.

    `base` is any model in this study -- pdf, cdf, ppf, rvs_from_uniform -- and
    the contaminated model is `base` with probability `1 - weight` and a narrow
    lognormal centered at `factor * anchor` with probability `weight`. It
    exposes the same interface, so it can be scored by the study's own W1 and
    sampled by the study's own pLCA without either knowing it is not a fit.

    THE POINT IS THAT IT IS CHEAP IN W1 AND EXPENSIVE IN A MONTE CARLO. W1 is
    the area between two CDFs, so moving a fraction `w` of the mass out to any
    distance at all costs at most `w` times that distance in the far region and
    the CDF is already within `w` of 1 there. A mean, a standard deviation and
    a high quantile are all dominated by it. This class is what lets the two
    costs be measured against each other instead of argued about.
    """

    def __init__(self, base, factor=100.0, weight=1e-3, anchor=1.0,
                 spread=0.1):
        self.base = base
        self.weight = float(weight)
        self.factor = float(factor)
        self.anchor = float(anchor)
        self.spread = float(spread)
        if not 0.0 <= self.weight < 1.0:
            raise ValueError('weight must be in [0, 1)')
        self.mu = np.log(self.factor * self.anchor)
        self.sigma = float(spread)
        self.label = f'{getattr(base, "label", base)} + {weight:g} at ' \
                     f'{factor:g}x'

    # The far component, as a two-parameter lognormal in closed form.
    def _far_cdf(self, x):
        from scipy.stats import norm
        x = np.asarray(x, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (np.log(np.where(x > 0, x, np.nan)) - self.mu) / self.sigma
        return np.where(x > 0, norm.cdf(z), 0.0)

    def _far_pdf(self, x):
        from scipy.stats import norm
        x = np.asarray(x, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (np.log(np.where(x > 0, x, np.nan)) - self.mu) / self.sigma
            d = norm.pdf(z) / (x * self.sigma)
        return np.where(x > 0, d, 0.0)

    def _far_ppf(self, q):
        from scipy.stats import norm
        return np.exp(self.mu + self.sigma * norm.ppf(np.asarray(q, float)))

    def pdf(self, x):
        return ((1 - self.weight) * np.asarray(self.base.pdf(x), float)
                + self.weight * self._far_pdf(x))

    def cdf(self, x):
        return ((1 - self.weight) * np.asarray(self.base.cdf(x), float)
                + self.weight * self._far_cdf(x))

    def ppf(self, q):
        """Inverted by bisection, because a mixture CDF has no closed inverse.

        Monotone and bounded below by the base model's own quantile and above
        by the far component's, so 200 bisection steps put it at machine
        precision.
        """
        q = np.atleast_1d(np.asarray(q, dtype=float))
        lo = np.full_like(q, 1e-12)
        hi = np.maximum(np.asarray(self.base.ppf(np.minimum(q, 1 - 1e-12)),
                                   float),
                        self._far_ppf(np.clip(q, 1e-12, 1 - 1e-12)))
        hi = np.maximum(hi * 1.000001, lo * 10)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            go = self.cdf(mid) < q
            lo = np.where(go, mid, lo)
            hi = np.where(go, hi, mid)
        return 0.5 * (lo + hi)

    def rvs_from_uniform(self, u):
        return self.ppf(np.asarray(u, dtype=float))

    def rvs(self, size, random_state):
        return self.ppf(random_state.random(size))


#: How far out the contamination sits, as a multiple of the dataset mean, and
#: how much mass goes there. The weights are deliberately small: the point of
#: the failure mode is that a model can carry a tail the data does not support
#: while still looking like a good fit.
#:
#: THE DISTANCES ARE A CONTINUOUS LOG SWEEP AND THEY START INSIDE THE DATA.
#: An earlier version used three round decades, which drew as three points and
#: could not show WHERE the criterion goes blind. It goes blind at the top of
#: its own scoring grid, `max(x) + 10 sd`, which on a dataset normalized to a
#: mean of 1.0 is of order ten times the mean -- so the sweep has to run from
#: inside the data, through that boundary, and out, or the transition is not in
#: the picture.
TAIL_FACTORS = tuple(np.round(np.logspace(0.0, np.log10(3000.0), 25), 4))
TAIL_WEIGHTS = (1e-4, 1e-3, 1e-2)


def tail_stress(models, names, x_by_dataset, w_by_dataset, u, method,
                factors=TAIL_FACTORS, weights=TAIL_WEIGHTS, target=0,
                outputs_=None, score=None):
    """What a thin far tail costs in W1, beside what it costs in every output.

    One material of a pLCA group has its fitted model replaced by the same
    model with `weight` of its mass moved to `factor` times the dataset mean.
    Everything else -- the other three models, the uniform variates, the group
    -- is held, so the difference in every output is caused by the
    contamination and by nothing else.

    `score` is the study's own W1 scorer, `fitting.score_w1_model`, passed in
    rather than imported so this function can be tested without a fit.

    Returns one row per (factor, weight): the relative change in W1 and the
    relative change in each output for the contaminated material.

    READ THE RATIO, NOT EITHER COLUMN. A tail that costs 1 percent of a W1 and
    40 percent of a metric is a tail the goodness-of-fit criterion cannot see
    and the metric cannot survive; a tail that costs both the same is one the
    criterion is already charging for.
    """
    outputs_ = outputs_ or PL.OUTPUTS
    names = list(names)
    d = names[int(target)]
    base_model = models[d][method]
    x, w = np.asarray(x_by_dataset[d], float), np.asarray(w_by_dataset[d], float)
    anchor = float(np.mean(x))
    base_out = PL.outputs(PL.draw_contributions(models, names, method, u))
    base_w1 = float(score(base_model, x, w)) if score is not None else np.nan
    # WHERE THE SCORING GRID ENDS, in the same units the sweep is reported in.
    # Beyond this the body of W1 cannot see the contamination at all, so a
    # figure of the sweep has to be able to mark it.
    grid = FT.score_grid_open(x, w)
    grid_top = float(np.max(grid)) / anchor if anchor > 0 else np.nan
    rows = []
    for factor in factors:
        for weight in weights:
            bad = Contaminated(base_model, factor=factor, weight=weight,
                               anchor=anchor)
            patched = {k: dict(v) for k, v in models.items()}
            patched[d][method] = bad
            got = PL.outputs(PL.draw_contributions(patched, names, method, u))
            row = dict(dataset=d, method=method, factor=factor, weight=weight,
                       w1_base=base_w1, grid_top=grid_top,
                       beyond_grid=bool(factor > grid_top))
            row['w1'] = (float(score(bad, x, w)) if score is not None
                         else np.nan)
            row['w1_rel'] = (abs(row['w1'] - base_w1) / base_w1
                             if base_w1 and np.isfinite(base_w1) else np.nan)
            for key in outputs_:
                b = float(base_out[key][int(target)])
                g = float(got[key][int(target)])
                row[f'{key}_base'] = b
                row[f'{key}'] = g
                row[f'{key}_rel'] = abs(g - b) / abs(b) if b else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def tail_exposure(stress, outputs_=None, w1='w1_rel'):
    """Each output's relative move per unit of relative W1 move.

    The number to read is `exposure`: how many percent a metric moves for each
    percent the study's own goodness-of-fit criterion moves. **An exposure near
    1 means the criterion sees what the metric sees.** A large exposure names
    an output a model can wreck while scoring well, which is the failure mode
    this stage was told to check a recommended metric against rather than
    assume it immune to.
    """
    outputs_ = outputs_ or PL.OUTPUTS
    rows = []
    for key in outputs_:
        col = f'{key}_rel'
        if col not in stress.columns:
            continue
        ok = np.isfinite(stress[col]) & np.isfinite(stress[w1]) \
            & (stress[w1] > 0)
        if not ok.any():
            continue
        ratio = (stress.loc[ok, col] / stress.loc[ok, w1]).to_numpy(float)
        rows.append(dict(output=key,
                         exposure=float(np.nanmedian(ratio)),
                         exposure_max=float(np.nanmax(ratio)),
                         rel_median=float(np.nanmedian(stress.loc[ok, col])),
                         rel_max=float(np.nanmax(stress.loc[ok, col])),
                         n=int(ok.sum())))
    return (pd.DataFrame(rows).sort_values('exposure', ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# the claim scorecard, and the two errors it must not confuse
# ---------------------------------------------------------------------------
#: The FIFTEEN claims a probabilistic LCA makes, grouped by the five questions
#: a reader asks. Each entry is
#:
#:     (question, label, source, output)
#:
#: where `source` names which of the four tidy frames the claim is read from.
#: `'recovery'` rows come from `recovery_table`, which is already per material;
#: the other three are ROW-LEVEL frames and the per-unit error is taken here.
#:
#: IT WAS SIXTEEN UNTIL STAGE 4, AND ONE OF THE SIXTEEN WAS A COPY OF ANOTHER.
#: "using 25 pct less: its mean saving" (`qty_reduction_mean`) is an exact
#: algebraic function of "a material: its share of the total"
#: (`eci_perc_mean`): using 25 percent less of a material removes exactly a
#: quarter of that material's share of the building total, with no
#: distribution entering, so
#:
#:     qty_reduction_mean__error  =  0.25 * eci_perc_mean__error
#:
#: exactly, and the two true LEVELS stand in the same ratio, so the two
#: RELATIVE errors are equal to within floating point. Verified at 7e-16
#: across every policy (decisions 186 and 238). Keeping both made the
#: scorecard count one claim twice and gave every count a denominator of 16
#: where it should be 15. `DUPLICATE_CLAIMS` records what was removed and why,
#: so a later stage adding a claim can check it is not a third copy; the
#: underlying column is untouched in `TABLE_PLCATruthIntervention.csv.gz`.
SCORECARD_CLAIMS = (
    ('magnitude', 'the total: its mean', 'building', 'total_mean'),
    ('magnitude', 'the total: its standard deviation', 'building', 'total_sd'),
    ('magnitude', 'the total: its 90th percentile', 'building', 'total_q0.9'),
    ('magnitude', 'the chance of meeting a budget', 'building',
     'p_below_q0.9'),
    ('attribution', 'a material: its mean contribution', 'recovery',
     'eci_mean'),
    ('attribution', 'a material: its standard deviation', 'recovery',
     'eci_std'),
    ('attribution', 'a material: its 95th percentile', 'recovery', 'eci_p95'),
    ('attribution', 'a material: its share of the total', 'recovery',
     'eci_perc_mean'),
    ('attribution', 'a material: its share at the building 95th', 'recovery',
     'eci_perc_p95tot'),
    ('attribution', 'a material: its chance of being largest', 'recovery',
     'eci_rank_1'),
    ('information', 'the uncertainty index', 'recovery', 'ui'),
    ('action', 'a cap: how often it binds', 'intervention',
     'cap_share_touched'),
    ('action', 'a cap: its mean saving', 'intervention', 'cap_reduction_mean'),
    ('action', 'a cap: its chance of saving 5 pct', 'intervention',
     'cap_p_reduction_over_5'),
    ('comparison', 'the probability B beats A', 'swap', 'discernibility'),
)

#: Claims removed from `SCORECARD_CLAIMS` because they are algebraic copies of
#: a claim that stays, with the factor relating the two and the claim they
#: duplicate. Kept as a record rather than deleted outright: the finding --
#: that a quantity reduction is a deterministic fraction of a material's own
#: share, so its accuracy IS that share's accuracy -- is worth a sentence in
#: the paper, and a later stage adding a claim can check against this list.
DUPLICATE_CLAIMS = (
    ('action', 'using 25 pct less: its mean saving', 'intervention',
     'qty_reduction_mean', 0.25,
     'a material: its share of the total'),
)

#: Below this fraction of the true level the six methods are treated as
#: agreeing. Ranking a row whose whole spread is arithmetic noise invites the
#: misreading the win-share leaders did in Stage 2g.
DIFFERENCE_FLOOR = 1e-3


#: THE STATISTIC EVERY SCORECARD CELL, RANK AND BEST-METHOD BOX IS READ OFF.
#: Author decision 2026-10-06 (decision 253): the MEDIAN OF PER-UNIT RATIOS,
#: median over units of |method - truth| / |truth|. It is the error one
#: typical building, material or design comparison carries as a percentage of
#: its OWN true value, which is the median absolute percentage error of the
#: forecasting literature. The ratio of means, `total_error`, stays in every
#: table beside it and is what a sentence about the average building quotes.
#: The per-unit errors are strongly right skewed on all fifteen claims, so the
#: two differ by about a factor of 1.4 pooled and the mean overstates what a
#: typical building sees.
HEADLINE_ERROR = 'median_error'


def unit_ratios(frame, output):
    """|error| / |truth| per row, NaN where the truth is zero or not finite.

    A per-unit ratio is undefined at a zero truth -- a specification cap that
    never binds, a material whose true variance share is exactly zero -- and
    unbounded near one. The median is robust to the second and the first is
    dropped and COUNTED rather than silently imputed; see `median_error`.
    """
    e = pd.to_numeric(frame[f'{output}__error'], errors='coerce').abs()
    t = pd.to_numeric(frame[f'{output}__truth'], errors='coerce').abs()
    r = e / t
    return r.where(np.isfinite(r) & (t > 0))


def median_error(frame, output, method='method'):
    """Per method: the median over units of |error| / |truth|.

    Returns a DataFrame indexed by method with `median_error` and
    `ratio_dropped`, the number of units whose truth was zero or not finite
    and so carry no ratio. Every unit is weighted equally, which is what
    "the typical building" means; a large miss on a large material counts the
    same as a small miss on a small one, which is the trade decision 253
    accepted against the ratio of means.
    """
    r = unit_ratios(frame, output)
    work = pd.DataFrame({'m': frame[method].to_numpy(), 'r': r.to_numpy()})
    rows = []
    for name, sub in work.groupby('m', sort=True):
        rows.append(dict(method=name,
                         median_error=float(np.nanmedian(sub['r']))
                         if sub['r'].notna().any() else np.nan,
                         ratio_dropped=int(sub['r'].isna().sum())))
    return pd.DataFrame(rows).set_index('method')


def per_unit_error(frame, output, method='method', block=None):
    """Both errors a set of signed per-unit errors can be summarized into.

    THE TWO ARE DIFFERENT QUANTITIES AND FIVE OF THE SIXTEEN SCORECARD ROWS
    USED THE WRONG ONE. Stage 2g read four reduction-strategy rows and the
    design comparison off summary tables that averaged the SIGNED error over
    pLCA groups before taking the absolute value, so a method that is too high
    on one group and too low on the next reported almost no error at all. The
    other eleven rows were already per unit, so the figure put two statistics
    on one color scale and the five flattered themselves: the best method on
    "how often a cap binds" read 0.48 percent of the true level where the
    per-unit figure is 33.01, and on "what using 25 percent less saves" it read
    0.00 against 10.02.

    `error`
        mean |signed error| over the units. **This is the error in a single
        decision** -- one building, one design comparison, one specification
        cap -- which is what a practitioner making one choice experiences, and
        it is what every scorecard row now reports.

    `error_portfolio`
        |mean signed error| over the units, taken within `block` first where a
        block is given. **This is the error in the AVERAGE claim over many
        decisions**, which is the right quantity for a portfolio of buildings
        or a stock model and the wrong one for a single design. It is kept and
        labeled rather than dropped, because it is a real quantity that a
        different reader wants; it is not a worse version of the first.

    The gap between them is the extent to which a method's error cancels
    across units, so `error_portfolio` at or near zero beside a large `error`
    says the method is unbiased on average and wrong case by case.

    Parameters
    ----------
    frame : DataFrame
        Row-level, one row per unit per method, carrying `<output>__error` and
        `<output>__truth`.
    output : str
        The output's column stem.
    method : str
        The column naming the UQ method.
    block : str or None
        A column to average within before taking the absolute value, for the
        portfolio form. The design comparison is reported per claimed saving,
        so its portfolio error averages pairs within a saving and then across
        savings; passing None averages every unit at once.

    Returns
    -------
    DataFrame indexed by method with `error`, `error_portfolio` and `n_units`,
    plus the claim's `truth_level` as an attribute-free column repeated per row.
    """
    ecol, tcol = f'{output}__error', f'{output}__truth'
    if ecol not in frame.columns or tcol not in frame.columns:
        raise KeyError(f'{output}: need both {ecol} and {tcol}')
    work = frame.assign(
        _e=pd.to_numeric(frame[ecol], errors='coerce'),
        _t=pd.to_numeric(frame[tcol], errors='coerce'))
    level = abs(float(np.nanmean(work['_t'].to_numpy(float))))
    rows = []
    for name, sub in work.groupby(method, sort=True):
        if block is None or block not in sub.columns:
            portfolio = abs(float(np.nanmean(sub['_e'].to_numpy(float))))
        else:
            per_block = sub.groupby(block)['_e'].mean().abs()
            portfolio = float(np.nanmean(per_block.to_numpy(float)))
        rows.append(dict(
            method=name,
            error=float(np.nanmean(sub['_e'].abs().to_numpy(float))),
            error_portfolio=portfolio,
            n_units=int(sub['_e'].notna().sum()),
            truth_level=level))
    return pd.DataFrame(rows).set_index('method')


def claim_scorecard(recovery, building_rows, intervention_rows, swap_rows,
                    claims=SCORECARD_CLAIMS, method='method',
                    swap_block='saving', difference_floor=DIFFERENCE_FLOOR,
                    recovery_rows=None):
    """The fifteen claims by the six methods, every row on ONE definition.

    ONE DEFINITION FOR EVERY ROW IS THE WHOLE POINT OF THE TABLE, and it has
    now been established twice. Stage 2g's decision 157 put every row on the
    same DIVISOR -- the mean true LEVEL of the quantity, never the spread
    between materials, because a signal-to-noise ratio and a relative error are
    not one unit however both are printed as percentages. This function adds
    the other half: every row on the same NUMERATOR, the mean absolute error
    per unit, because five rows were averaging the signed error over groups
    first and so reported a cancellation rather than an error.

    Both statistics are returned for all fifteen rows and both are labeled:

        `total_error`       the per-unit relative error. **What the figure
                            draws**, and what a single design decision carries.
        `portfolio_error`   the relative error in the AVERAGE claim over many
                            decisions. The right quantity for a stock model.

    and, derived from `total_error` alone,

        `best_error`        what the closest of the six still gets wrong
        `stakes`            worst minus best, which is what the CHOICE costs
        `excess`            this method's excess over the best
        `methods_differ`    whether `stakes` clears `difference_floor`

    `best_error` and `stakes` answer different questions and reporting only one
    misleads: a small spread can mean every method is right or every method is
    equally wrong.

    Parameters
    ----------
    recovery : DataFrame
        `recovery_table` output, already filtered to one truth parent. Its
        `abs_error` is per material and its `bias_raw` is the signed mean, so
        both statistics are read straight off it.
    building_rows, intervention_rows, swap_rows : DataFrame
        The row-level truth-run frames, one row per pLCA, per material and per
        design pair respectively.
    recovery_rows : DataFrame or None
        The ROW-LEVEL per-material truth run, filtered to the same truth
        parent as `recovery`. Needed for `median_error` on the seven
        per-material claims, which the aggregated `recovery` cannot supply.
        When it is given every row gains `median_error` and `ratio_dropped`,
        and the derived columns are ranked on `HEADLINE_ERROR`; when it is not,
        they are ranked on `total_error` as before and `ranked_on` says so.
    """
    sources = {'building': (building_rows, None),
               'intervention': (intervention_rows, None),
               'swap': (swap_rows, swap_block)}
    rows = []
    medians = {}
    row_frames = {'recovery': recovery_rows, 'building': building_rows,
                  'intervention': intervention_rows, 'swap': swap_rows}
    if recovery_rows is not None:
        for _, label, source, output in claims:
            f = row_frames[source]
            if f is not None and f'{output}__error' in f.columns:
                medians[label] = median_error(f, output, method=method)
    for group, label, source, output in claims:
        if source == 'recovery':
            sub = recovery[recovery['output'] == output]
            if sub.empty:
                continue
            level = abs(float(sub['truth_mean'].iloc[0]))
            for _, r in sub.iterrows():
                rows.append(dict(
                    group=group, claim=label, method=r[method],
                    error=float(abs(r['abs_error'])),
                    error_portfolio=float(abs(r['bias_raw'])),
                    scale=level, n_units=int(r.get('n', 0))))
        else:
            frame, block = sources[source]
            got = per_unit_error(frame, output, method=method, block=block)
            for name, r in got.iterrows():
                rows.append(dict(
                    group=group, claim=label, method=name,
                    error=float(r['error']),
                    error_portfolio=float(r['error_portfolio']),
                    scale=float(r['truth_level']), n_units=int(r['n_units'])))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame['total_error'] = frame['error'] / frame['scale']
    frame['portfolio_error'] = frame['error_portfolio'] / frame['scale']
    if medians:
        med = pd.concat({k: v for k, v in medians.items()},
                        names=['claim', 'method']).reset_index()
        frame = frame.merge(med, on=['claim', 'method'], how='left')
        frame = rescore(frame, difference_floor=difference_floor,
                        value=HEADLINE_ERROR)
    else:
        frame = rescore(frame, difference_floor=difference_floor)
    order = {lab: i for i, (_, lab, _, _) in enumerate(claims)}
    return (frame.assign(_o=frame['claim'].map(order))
            .sort_values(['_o', 'method']).drop(columns='_o')
            .reset_index(drop=True))


def rescore(frame, difference_floor=DIFFERENCE_FLOOR, value='total_error'):
    """Recompute the derived columns over whatever methods `frame` holds.

    `rank`, `stakes`, `best_error`, `excess`, `best_method` and
    `methods_differ` are properties of the SET of methods compared, not of any
    one method. So a scorecard that gains a seventh policy -- Stage 3 adds the
    feasible size rule as a column, decision 211 -- cannot keep the six-method
    values for them: the best method on a row may now be the rule, and `stakes`
    then measures a different choice.

    Split out of `claim_scorecard` so there is ONE implementation of those six
    columns. Doing it a second time in a notebook cell is how two tables end up
    disagreeing about which method is best.

    `value` names the statistic they are read off, and it is written into the
    frame as `ranked_on` so a table cannot be read under the wrong one. The
    scorecard ranks on `HEADLINE_ERROR` from decision 253; `total_error`, the
    ratio of means, is the default only so that a caller holding no row-level
    frames still gets a table.
    """
    frame = frame.copy()
    piv = frame.pivot(index='claim', columns='method', values=value)
    lo, hi = piv.min(axis=1), piv.max(axis=1)
    frame['rank'] = frame.groupby('claim')[value].rank(
        method='min').astype(int)
    frame['stakes'] = frame['claim'].map(hi - lo)
    frame['best_error'] = frame['claim'].map(lo)
    frame['excess'] = frame[value] - frame['claim'].map(lo)
    frame['ranked_on'] = value
    frame['best_method'] = frame['claim'].map(piv.idxmin(axis=1))
    frame['methods_differ'] = frame['stakes'] > difference_floor
    return frame


#: The unit a claim is made about, per `SCORECARD_CLAIMS` source. A claim's
#: per-decision statistics are taken by aligning the methods on these keys, so
#: that two methods are compared on the SAME building, material or design pair
#: rather than on two averages.
CLAIM_UNIT_KEYS = {
    'recovery': ('plca', 'dataset'),
    'building': ('plca',),
    'intervention': ('plca', 'dataset'),
    'swap': ('pair', 'saving'),
}


def choice_cost(sources, claims=SCORECARD_CLAIMS, method='method',
                unit_keys=CLAIM_UNIT_KEYS):
    """What choosing one method over another costs IN ONE DECISION.

    THIS EXISTS BECAUSE `stakes` IS AN AVERAGED QUANTITY AND THE SCORECARD'S
    CELLS ARE NOT. `rescore` computes `stakes` as

        max over methods of (mean error)  -  min over methods of (mean error)

    which is the spread of the AVERAGE error across methods. Every cell of the
    same figure is `total_error`, the mean error in ONE decision, which is what
    decision 174 put all fifteen rows onto. The two differ by more than an
    order of magnitude where a method's error cancels across units: on the
    uncertainty index `stakes` is 2.2 percent of the true level where the mean
    per-pair per-decision difference is 27.9 and the worst pair is 40.2, and
    the study's own published NRMSE of 0.5504 for that output implies 44.0. A
    bar drawn from `stakes` beside those cells tells a reader the choice of
    method is nearly free on that claim, when what is nearly free is the
    average over many buildings.

    Two methods are differenced UNIT BY UNIT, which is the whole point: a
    method that runs high on one building and low on the next differs from its
    rival on both, and averaging first hides exactly that.

    Returns one row per claim with

        `pair_mean`     mean over method PAIRS of the mean |difference| per
                        unit. **What a reader choosing between two methods at
                        random should expect one decision to move.**
        `unit_p05` ..   the 5th, 10th, 25th, 50th, 75th, 90th and 95th
        `unit_p95`      percentile of the per-unit, per-pair absolute
                        difference POOLED over both, which is the
                        distribution `pair_mean` is the mean of. A
                        claim where every building moves a little and one
                        where most move nothing and a few move a lot have the
                        same `pair_mean` and different spreads.
        `ratio_p05` ..  the same ladder with each per-unit difference
        `ratio_p95`     divided by THAT unit's own true value rather than by
                        the claim's mean level, so it is on the scale of
                        `median_error`, the headline the cells show (decision
                        253). Units with a zero truth are dropped.
        `pair_worst`    the same for the worst pair, which is what the choice
                        costs between the two methods that disagree most.
        `shared`        mean over units of |mean signed error over methods|:
                        the part of the error EVERY method makes, which no
                        choice of method can remove.
        `worst_cell`    the worst method's own `total_error`, recomputed here
                        so the stack below closes exactly.
        `specific`      `worst_cell - shared`, so `shared + specific` is the
                        worst method's cell by construction. It is a residual
                        and NOT a mean deviation from the cross-method mean;
                        the two differ because a mean of absolute values does
                        not decompose additively.
        `stakes_mean`   `max(mean error) - min(mean error)` over the same
                        methods and the same frames, which is `rescore`'s
                        `stakes`. Carried so the two readings can be shown
                        together and labeled, never so one can stand in for
                        the other.

    All six are divided by the claim's mean TRUE LEVEL, which is the divisor
    decision 157 put every scorecard row onto, so they are directly comparable
    with `total_error` and with each other.

    `pair_mean` AND `specific` ARE NOT THE SAME QUANTITY AND MUST NOT BE
    QUOTED FOR EACH OTHER. `pair_mean` is a per-case distance between two
    methods; `specific` is a difference of two averaged magnitudes. On the
    uncertainty index they read 27.9 and 5.2 percent of the true level.

    Parameters
    ----------
    sources : mapping
        `{'recovery': frame, 'building': frame, 'intervention': frame,
        'swap': frame}`, each row-level and carrying `<output>__error` and
        `<output>__truth`. The recovery frame must already be filtered to one
        truth parent. A missing source drops its claims rather than raising,
        which is what lets a smoke run produce a partial table.
    claims : sequence
        `(question, label, source, output)` tuples, defaulting to the fifteen.
    method : str
        The column naming the UQ method or policy.
    unit_keys : mapping
        Source name to the columns identifying one decision.

    Returns
    -------
    DataFrame, one row per claim, in `claims` order.
    """
    rows = []
    for group, label, source, output in claims:
        frame = sources.get(source)
        if frame is None or frame.empty:
            continue
        ecol, tcol = f'{output}__error', f'{output}__truth'
        if ecol not in frame.columns or tcol not in frame.columns:
            continue
        keys = [k for k in unit_keys[source] if k in frame.columns]
        if not keys:
            raise KeyError(f'{label}: none of {unit_keys[source]} present')
        piv = frame.pivot_table(index=keys, columns=method, values=ecol)
        names = list(piv.columns)
        if len(names) < 2:
            continue
        err = piv.to_numpy(float)
        # each unit's OWN true value, for the ratio ladder below. The truth
        # repeats identically across methods, so the first column is it.
        tru = np.abs(frame.pivot_table(index=keys, columns=method,
                                       values=tcol).reindex(piv.index)
                     .to_numpy(float)[:, 0])
        level = abs(float(np.nanmean(
            pd.to_numeric(frame[tcol], errors='coerce').to_numpy(float))))
        diffs = [np.abs(err[:, i] - err[:, j])
                 for i in range(len(names))
                 for j in range(i + 1, len(names))]
        pairs = [float(np.nanmean(d)) for d in diffs]
        # THE DISTRIBUTION BEHIND `pair_mean`, NOT ANOTHER SUMMARY OF IT.
        # `pair_mean` averages twice -- over units within a pair, then over
        # pairs -- so it cannot say whether a claim is one where every
        # building moves a little or one where most move nothing and a few
        # move a lot. These are the quantiles of the per-unit, per-pair
        # absolute difference POOLED over both, which is the spread a
        # practitioner with one building actually faces.
        pooled = np.concatenate(diffs)
        pooled = pooled[np.isfinite(pooled)]
        # A FULL LADDER, not the three the first figure happened to need.
        # Recomputing this costs a 110-minute notebook run, so the table
        # carries enough quantiles that a change of figure style never needs
        # another one.
        QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        qq = (np.nanquantile(pooled, QS) if pooled.size
              else np.full(len(QS), np.nan))
        # THE SAME DISTRIBUTION ON THE HEADLINE'S OWN SCALE (decision 253):
        # each per-unit difference divided by THAT unit's true value, so the
        # bar is a percentage of the case's own size exactly as the cells
        # are. Units with a zero truth carry no ratio and are dropped.
        with np.errstate(divide='ignore', invalid='ignore'):
            rat = np.concatenate([d / tru for d in diffs])
        rat = rat[np.isfinite(rat)]
        rq = (np.nanquantile(rat, QS) if rat.size
              else np.full(len(QS), np.nan))
        cells = np.nanmean(np.abs(err), axis=0)
        shared = float(np.nanmean(np.abs(np.nanmean(err, axis=1))))
        worst_cell = float(np.nanmax(cells))
        rows.append(dict(
            group=group, claim=label, output=output, source=source,
            n_methods=len(names), n_units=int(piv.shape[0]),
            pair_mean=float(np.mean(pairs)) / level,
            pair_worst=float(np.max(pairs)) / level,
            pair_best=float(np.min(pairs)) / level,
            unit_p05=float(qq[0]) / level,
            unit_p10=float(qq[1]) / level,
            unit_p25=float(qq[2]) / level,
            unit_p50=float(qq[3]) / level,
            unit_p75=float(qq[4]) / level,
            unit_p90=float(qq[5]) / level,
            unit_p95=float(qq[6]) / level,
            **{f'ratio_p{int(round(100 * q)):02d}': float(v)
               for q, v in zip(QS, rq)},
            shared=shared / level,
            worst_cell=worst_cell / level,
            specific=(worst_cell - shared) / level,
            stakes_mean=float(np.nanmax(cells) - np.nanmin(cells)) / level,
            worst_method=names[int(np.nanargmax(cells))],
            truth_level=level))
    return pd.DataFrame(rows)

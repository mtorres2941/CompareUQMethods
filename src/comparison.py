"""Comparing the six UQ methods: the tidy score table, and the curves drawn from it.

Stage 2b. These are the comparisons that belong to the PAPER, so they live here
under test rather than in `audits/`, which is for one-off measurements that get
decided once and never rerun. Notebook 2 calls these, writes the tables, and
draws every figure from the tables on disk.

THREE SCORES PER DATASET PER METHOD, and the second and third exist because the
first has a known bias.

  `w1`            Wasserstein-1 between the fitted model and the
                  variable-weighted empirical CDF. The study's criterion, in
                  sample, and what every reported number has always been.

  `w1_heldout`    The same distance, but the model is fitted on half the values
                  and scored against the other half, averaged over both
                  directions and several splits. IN-SAMPLE W1 REWARDS
                  FLEXIBILITY: it falls monotonically as a KDE bandwidth
                  shrinks, because a KDE with a vanishing bandwidth IS the
                  empirical distribution it is being scored against
                  (`audits/why_kde_loses.py`). The families compared
                  here run from 2 parameters to effectively n, so the in-sample
                  number cannot settle the comparison on its own. Held-out W1
                  keeps the same units and removes the reward.

                  WHAT IT IS AND IS NOT FOR. It is the right instrument for
                  comparing FAMILIES of different parameter counts. It is the
                  WRONG one for choosing a bandwidth: measured over the
                  empirical arm it turns at about 0.1 of Scott's bandwidth on
                  the mean and 0.5 on the median, and the median moves only from
                  0.233 to 0.243 across a 50-fold range. It removes most of
                  W1's bandwidth sensitivity rather than replacing it with a
                  sharp optimum. Leave-one-out likelihood is the sharp
                  instrument for bandwidth, and decision 54 used it.

  `model_sd_ratio`  The fitted model's own standard deviation over the data's.
                  W1 is nearly blind to tail mass -- a thin far tail is a small
                  area between two CDFs -- while the pLCA SAMPLES from these
                  models. A lognormal that scored well and had a standard
                  deviation of 3,000 got through Stage 2b's fit scores and was
                  caught only in the pLCA results; see discrepancy entry 43.
                  One column catches that class of failure.

NOT scored against the known parent. The corpus stores the exact generating
distribution in `parents.json.gz` and comparing fits to it is the cleanest test
of all, but it is available on the SYNTHETIC arm only, so it cannot carry a
comparison the paper has to make on both. Stage 2c owns it, and its job there is
to confirm that held-out W1 behaves, not to become a third headline.
"""

import numpy as np
import pandas as pd

import fitting as FT

#: Values below this cannot be split into two halves that both support a fit,
#: so `w1_heldout` is not defined for them and is returned as NaN.
HELDOUT_MIN_N = 10

#: Random 50/50 splits per dataset. Both directions of each split are scored, so
#: this is 2 * HELDOUT_REPEATS fits per dataset per method.
HELDOUT_REPEATS = 2

#: Rolling window for the characteristic curves, as a fraction of the number of
#: datasets, so the same call works on a 149-dataset arm and a 9,999-dataset
#: corpus. The Stage 1 figure hard-coded 501, which exceeds the whole empirical
#: arm and would have drawn a flat line. The minimum is 15 rather than the 7 the
#: fraction gives on 149 datasets, because below that the empirical curve is too
#: jagged to read.
CURVE_WINDOW_FRAC = 0.05
CURVE_WINDOW_MIN = 15


def model_sd_ratio(model, x, weights, npoints=20_001):
    """The fitted model's standard deviation over the data's weighted one.

    Near 1 means the model carries the spread the data has. Large means a tail
    the data does not support, which W1 will not charge for and a Monte Carlo
    will be dominated by.
    """
    sd = FT.weighted_std(x, weights)
    if not sd > 0:
        return np.nan
    q = np.linspace(1e-9, 1.0 - 1e-9, npoints)
    return float(np.std(model.ppf(q)) / sd)


def heldout_w1_all(x, weights, rng, repeats=HELDOUT_REPEATS,
                   min_n=HELDOUT_MIN_N, **fit_kw):
    """Held-out W1 for ALL SIX methods at once, from the same splits.

    Both halves are used as the fitting half in turn, and the result is the mean
    over `2 * repeats` fits. The held-out half keeps its own weights,
    renormalized, so the scoring target is the same weighted empirical CDF the
    in-sample score uses, restricted to the values the model has not seen.

    All six methods share each split, which matters twice: it is six times
    cheaper than splitting per method, and it makes the comparison PAIRED, so a
    difference between two methods is not confounded with a difference between
    two random partitions.

    Returns NaN below `min_n`, where a half cannot support a fit. That is the
    honest answer rather than a number computed from two points, and it is why
    the empirical arm's small categories are absent from this column.
    """
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(x)
    if n < min_n:
        return {label: np.nan for label in FT.PEWT}
    acc = {label: [] for label in FT.PEWT}
    for _ in range(repeats):
        order = rng.permutation(n)
        halves = (order[: n // 2], order[n // 2:])
        for fit_idx, score_idx in (halves, halves[::-1]):
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
                    acc[label].append(
                        FT.score_w1_model(models[label], xs, ws, grid=grid))
                except Exception:
                    continue
    return {label: (float(np.mean(v)) if v else np.nan)
            for label, v in acc.items()}


def heldout_w1(x, weights, label, rng, repeats=HELDOUT_REPEATS,
               min_n=HELDOUT_MIN_N, **fit_kw):
    """Held-out W1 for one method. See `heldout_w1_all`, which this calls."""
    return heldout_w1_all(x, weights, rng, repeats, min_n, **fit_kw)[label]


def score_methods(datasets, rng, arm, heldout=True, repeats=HELDOUT_REPEATS,
                  progress=None, **fit_kw):
    """Tidy scores for every dataset and every method. One row per pair.

    `datasets` is {name: (values, weights)}, the shape both arms already use.
    Columns: arm, dataset, n, method, w1, w1_heldout, model_sd_ratio.
    """
    rows = []
    for i, (name, (x, w)) in enumerate(datasets.items()):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        models, _ = FT.fit_pewt(x, w, **fit_kw)
        grid = FT.score_grid_open(x, w)
        ho = (heldout_w1_all(x, w, rng, repeats, **fit_kw) if heldout
              else {label: np.nan for label in FT.PEWT})
        for label in FT.PEWT:
            rows.append(dict(
                arm=arm, dataset=name, n=len(x), method=label,
                w1=FT.score_w1_model(models[label], x, w, grid=grid),
                w1_heldout=ho[label],
                model_sd_ratio=model_sd_ratio(models[label], x, w)))
        if progress and (i + 1) % progress == 0:
            print(f'  {i + 1}/{len(datasets)}', flush=True)
    return pd.DataFrame(rows)


def weighting_of(method):
    """'Uniform' or 'Variable' from a PEWT label."""
    return method.split(', ')[1]


def add_ranks(scores, value='w1', within_weighting=False, suffix=None):
    """Rank the methods within each dataset, 1 = best. Scale-free by design.

    W1 magnitudes vary by orders of magnitude across datasets, so a mean over
    datasets is dominated by a handful of them. A rank is bounded by the number
    of methods and answers the question the paper asks, which is which method
    wins, not by how much.

    `within_weighting=True` ranks the three probability-estimation methods
    separately inside each weighting scheme, and IT IS THE ONLY HONEST WAY TO
    READ A HELD-OUT SCORE HERE.

    WHY. The market-share weights are an exchangeable Dirichlet draw, so they
    carry no information that generalizes from one half of a dataset to the
    other: the expected variable-weighted empirical CDF of a random half IS the
    unweighted one. A variable-weighted fit is therefore fitted partly to the
    weight realization of its own half, and a held-out score correctly penalizes
    that -- which makes a held-out comparison BETWEEN weighting schemes a
    statement about the Dirichlet draw rather than about weighting. Comparing
    the three estimation methods inside a fixed weighting scheme is unaffected.

    The in-sample comparison does not have this problem: there the weights are a
    stated property of the dataset being described, not something being
    predicted. Discrepancy entry 32 is the same issue seen from another angle.
    """
    suffix = suffix or f'{value}_rank'
    keys = ['arm', 'dataset']
    frame = scores.assign(_wt=scores.method.map(weighting_of))
    if within_weighting:
        keys = keys + ['_wt']
    wide = frame.pivot_table(index=keys, columns='method', values=value)
    ranks = wide.rank(axis=1).stack().rename(suffix).reset_index()
    out = frame.merge(ranks, on=keys + ['method'], how='left')
    return out.drop(columns='_wt')


def add_relative(scores, value='w1', suffix=None):
    """Each method's score over the mean score across methods, per dataset.

    RANK IS THE WRONG READOUT ON ITS OWN, and this is the column that says so.
    A rank turns any gap into 1, 2, 3 regardless of size. Measured on the
    empirical arm, the relative gap between the best and worst of the three
    estimation methods has a median of **0.21 at n = 3-9** and **7.21 at
    n >= 1000**: at small n the methods are practically indistinguishable and a
    mean-rank curve implies a separation that is not there, while at large n
    they differ by a factor of seven and the rank understates it.

    Scale free like a rank, because it is divided by the dataset's own mean, so
    it can be averaged across datasets whose W1 magnitudes differ by orders of
    magnitude. Unlike a rank it keeps the size of the difference. 1.0 means a
    method scored exactly the average of the methods on that dataset.
    """
    suffix = suffix or f'{value}_relative'
    mean = (scores.groupby(['arm', 'dataset'])[value].transform('mean'))
    return scores.assign(**{suffix: scores[value] / mean})


def curve_window(n_datasets, frac=CURVE_WINDOW_FRAC, minimum=CURVE_WINDOW_MIN):
    """Rolling window scaled to the arm, always odd and at least `minimum`."""
    w = max(minimum, int(round(frac * n_datasets)))
    return w if w % 2 else w + 1


def characteristic_curves(scores, characteristics, value='w1',
                          frac=CURVE_WINDOW_FRAC):
    """Smoothed `value` against each characteristic, per method, unbinned.

    Returns one tidy row per (arm, characteristic, method, dataset) carrying the
    characteristic's value, the raw score and the rolling mean of the score over
    datasets ordered by that characteristic. A figure drawn from this shows
    every dataset as a point and the trend as a line, which is what a binned
    table cannot do: the crossover between two methods happens at a value, not
    in a bin.

    `characteristics` is a frame indexed by dataset name.
    """
    out = []
    for arm, g in scores.groupby('arm'):
        window = curve_window(g.dataset.nunique(), frac)
        for metric in characteristics.columns:
            vals = characteristics[metric]
            for method, h in g.groupby('method'):
                h = h.assign(x=h.dataset.map(vals)).dropna(subset=['x', value])
                h = h[np.isfinite(h.x) & np.isfinite(h[value])].sort_values('x')
                if len(h) < 3:
                    continue
                out.append(pd.DataFrame(dict(
                    arm=arm, characteristic=metric, method=method,
                    value=value, dataset=h.dataset.values, x=h.x.values,
                    y=h[value].values,
                    y_smooth=h[value].rolling(window, min_periods=1,
                                              center=True).mean().values)))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=['arm', 'characteristic', 'method',
                                       'value', 'dataset', 'x', 'y',
                                       'y_smooth']))


def bandwidth_comparison(datasets, arm, rules=('scott', 'silverman',
                                               'silverman_guarded')):
    """W1 for the two KDE methods under each bandwidth rule. One row per fit.

    The bandwidth rule is a stated methodological decision (CLAUDE.md decision
    54), so the paper has to show what it does. Note that W1 alone cannot choose
    a rule -- it falls monotonically as the bandwidth shrinks -- which is why
    the rule was chosen on held-out likelihood in
    `audits/bandwidth_rules.py` and why `model_sd_ratio` is reported here
    beside it.
    """
    rows = []
    for name, (x, w) in datasets.items():
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        for wt, ww in (('Uniform', FT.uniform_weights(x)), ('Variable', w)):
            for rule in rules:
                m, p = FT.fit_kde(x, ww, bw_method=rule)
                rows.append(dict(
                    arm=arm, dataset=name, n=len(x), weighting=wt, rule=rule,
                    bandwidth=p['bandwidth'],
                    h_over_sd=p['bandwidth'] / FT.weighted_std(x, ww),
                    w1=FT.score_w1_model(m, x, w),
                    mass_below_zero=m.mass_below,
                    model_sd_ratio=model_sd_ratio(m, x, w)))
    return pd.DataFrame(rows)


#: Characteristics whose values span orders of magnitude, so a linear x-axis
#: puts almost every dataset in the leftmost sliver of the panel.
LOG_X_CHARACTERISTICS = ('n', 'coeffvar', 'crit_bw_1')
#: The suffix a characteristic carries when it was measured on the
#: UNIFORM-weighted version of the dataset. Without it, the characteristic was
#: measured on the variable-weighted version.
UNIFORM_SUFFIX = '_uw'
#: Characteristics that take both signs and so need a symmetric log axis.
SYMLOG_X_CHARACTERISTICS = ('kurtosis', 'skewness')


def base_characteristic(name):
    """`coeffvar_uw` -> `coeffvar`. The characteristic without its weighting."""
    return (name[:-len(UNIFORM_SUFFIX)]
            if name.endswith(UNIFORM_SUFFIX) else name)


def base_label(label):
    """Strip the trailing weighting tag from a metric label.

    The stored labels end in `(Var)` or `(Uni)` because the overview figures
    show the two as separate panels and need to say which is which. A
    per-characteristic figure says it once in the column headers instead, so
    repeating it in the title and on all four x axes is noise. Newlines in the
    stored label are flattened, because these are used in running text.
    """
    text = ' '.join(str(label).split())
    for tag in (' (Var)', ' (Uni)'):
        if text.endswith(tag):
            return text[:-len(tag)]
    return text


def point_style(n_points, alpha_budget=60.0, size_budget=900.0,
                alpha_range=(0.03, 0.55), size_range=(1.0, 14.0)):
    """Marker alpha and size for a scatter of `n_points`, so both arms read.

    A fixed alpha that works for 149 points is invisible at 9,999 and a fixed
    alpha that works for 9,999 hides the curve at 149. Both are set so that the
    total ink is roughly constant, then clipped to a range that stays legible.
    """
    alpha = float(np.clip(alpha_budget / max(n_points, 1), *alpha_range))
    size = float(np.clip(size_budget / max(n_points, 1), *size_range))
    return alpha, size


def x_scale_for(characteristic, values):
    """('linear' | 'log' | 'symlog', linthresh) for one characteristic.

    Chosen from the characteristic rather than from the data, so the same panel
    is drawn the same way on both arms and the two figures can be laid side by
    side.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    # Match on the base name, so `coeffvar` and `coeffvar_uw` get the SAME
    # axis. Keying off the full name put the two weightings of one
    # characteristic on a log and a linear axis, which makes the pair
    # impossible to read side by side -- and the pair is the comparison.
    characteristic = base_characteristic(characteristic)
    if characteristic in LOG_X_CHARACTERISTICS and len(v) and v.min() > 0:
        return 'log', None
    if characteristic in SYMLOG_X_CHARACTERISTICS and len(v):
        pos = np.abs(v[v != 0])
        return 'symlog', (float(np.percentile(pos, 10)) if len(pos) else 1.0)
    return 'linear', None


def symlog_ticks(values, linthresh, max_ticks=5):
    """Decade tick positions for a symmetric-log axis, thinned to fit a panel.

    Matplotlib's default symlog locator puts a tick on every decade on both
    sides of zero AND on the linear region's edges. In a panel three inches
    wide that is a dozen labels, and the ones either side of zero overlap each
    other because the linear region is narrow by construction. Here the ticks
    are the decades the data actually spans, on each side, thinned by a common
    stride until at most `max_ticks` remain including zero.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.array([0.0])

    def decades(side):
        m = np.abs(side).max() if len(side) else 0.0
        if m <= linthresh:
            return []
        lo = int(np.ceil(np.log10(linthresh)))
        hi = int(np.floor(np.log10(m)))
        return list(range(lo, hi + 1))

    pos = decades(v[v > 0])
    neg = decades(v[v < 0])
    for stride in range(1, 9):
        # Thin from the outside in, so the largest decade on each side -- the
        # one that sets the axis limit -- is always labelled.
        kept_pos = pos[::-1][::stride][::-1]
        kept_neg = neg[::-1][::stride][::-1]
        if 1 + len(kept_pos) + len(kept_neg) <= max_ticks:
            break
    ticks = [-10.0 ** k for k in kept_neg[::-1]] + [0.0] + [10.0 ** k for k in kept_pos]
    return np.array(ticks)


def win_share_by_percentile(scores, characteristics, value='w1',
                            frac=CURVE_WINDOW_FRAC):
    """How often each method wins, against the PERCENTILE of a characteristic.

    A rolling mean plotted against a characteristic's VALUE gives the sparse end
    of a skewed characteristic as much axis as the dense middle, so a handful of
    datasets can set the shape of a whole panel. Ranking the datasets and
    plotting against percentile gives every dataset equal width.

    A share also answers a question a mean cannot: not "which method is lower on
    average" but "for datasets like this, how often is each one best". Its
    uncertainty is the binomial one, about 0.5 / sqrt(window), which is reported
    beside it rather than left implicit.

    Methods come back ordered by overall win share on that arm, best FIRST, so
    a stacked plot drawn in that order puts the best-fitting method at the
    bottom and the reader can follow one band along the axis.

    Returns one tidy row per (arm, characteristic, method, dataset) with the
    percentile, the characteristic's value there, the rolling win share and the
    method's stack position.
    """
    out = []
    for arm, g in scores.groupby('arm'):
        wide = g.pivot_table(index='dataset', columns='method', values=value)
        winner = wide.idxmin(axis=1)
        window = curve_window(len(winner), frac)
        order = (winner.value_counts().reindex(wide.columns).fillna(0)
                 .sort_values(ascending=False).index.tolist())
        for metric in characteristics.columns:
            v = characteristics[metric].reindex(winner.index)
            d = pd.DataFrame({'winner': winner, 'x': v})
            d = d[np.isfinite(d.x)].sort_values('x')
            if len(d) < 3:
                continue
            pct = np.linspace(0.0, 100.0, len(d))
            for position, method in enumerate(order):
                share = ((d.winner == method).astype(float)
                         .rolling(window, min_periods=1, center=True).mean())
                out.append(pd.DataFrame(dict(
                    arm=arm, characteristic=metric, method=method,
                    dataset=d.index, percentile=pct, x=d.x.values,
                    win_share=share.values, window=window,
                    stack_position=position)))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=['arm', 'characteristic', 'method',
                                       'dataset', 'percentile', 'x',
                                       'win_share', 'window',
                                       'stack_position']))


def characteristic_pairs(characteristics):
    """Group characteristic names into (base, variable_name, uniform_name).

    A characteristic is measured twice, once on each weighting of the same
    dataset, and the two are named `x` and `x_uw`. Reading them side by side is
    the comparison the study is about, so they belong in one figure rather than
    in two panels several rows apart. Three characteristics have no pair -- the
    dataset size is the same under either weighting, the unweighted mean is 1.0
    by construction, and the uniform-to-variable W1 is a single quantity
    describing the pair itself -- and they come back with `uniform_name` None.

    Returns a list of (base, variable_name_or_None, uniform_name_or_None),
    ordered by base name.
    """
    names = set(characteristics)
    bases = {}
    for name in names:
        base = (name[:-len(UNIFORM_SUFFIX)]
                if name.endswith(UNIFORM_SUFFIX) else name)
        bases.setdefault(base, [None, None])
        bases[base][1 if name.endswith(UNIFORM_SUFFIX) else 0] = name
    return [(base, var, uni) for base, (var, uni) in sorted(bases.items())]


def win_share_headline(curves, arm, characteristic, rank_value='w1_rank',
                       frac=0.1):
    """The leading method's win share overall and in each tail decile.

    This is the figure's caption reduced to numbers. It is counted, not
    smoothed: the share is the fraction of datasets on which that method has
    rank 1, over all datasets and then over the lowest and highest `frac` of
    them by the characteristic. The rolling curve a panel draws is a smoothed
    version of the same quantity, so the two agree in the middle and the
    counted one is what a reader can check against the table.

    Returns None when the characteristic is not on that arm, and otherwise a
    dict with the method, the three shares and the dataset count behind each.
    """
    g = curves[(curves.arm == arm) & (curves.characteristic == characteristic)
               & (curves.value == rank_value)]
    if not len(g):
        return None
    wide = g.pivot_table(index='dataset', columns='method', values='y')
    x = g.drop_duplicates('dataset').set_index('dataset')['x']
    best = wide.idxmin(axis=1)
    order = best.value_counts()
    if not len(order):
        return None
    leader = order.index[0]

    x = x.reindex(best.index)
    ok = np.isfinite(x)
    best, x = best[ok], x[ok]
    n = len(best)
    if n < 3:
        return None
    k = max(1, int(round(frac * n)))
    by_x = x.sort_values().index

    def share(index):
        return float((best.reindex(index) == leader).mean())

    return dict(method=leader, overall=share(best.index),
                bottom=share(by_x[:k]), top=share(by_x[-k:]),
                n=n, n_tail=k)


def headline_sentence(headline):
    """One plain sentence stating the three shares, for a figure subtitle.

    The characteristic is not named: this goes under a title that already names
    it, and repeating it there is what made the line too long to fit.
    """
    if headline is None:
        return ''
    return (f"{headline['method']} is best on {headline['overall']:.0%} of "
            f"datasets overall, {headline['bottom']:.0%} in the lowest decile "
            f"and {headline['top']:.0%} in the highest (n = {headline['n']:,})")

"""Which dataset characteristics actually carry signal. Stage 2f.

WHAT THIS REPLACES. The study showed goodness-of-fit against about ten
statistical characteristics as rolling averages over 250 datasets each side,
in a 21-panel figure per arm. Three things are wrong with that presentation and
all three are fixed here. It carries no uncertainty band, so a wiggle and a
result look the same. It shows no data density, so a curve drawn through four
datasets in a sparse tail looks exactly like one drawn through four hundred.
And the characteristics are strongly correlated with each other, so twenty-one
MARGINAL views overstate how many independent effects there are.

THE TWO TARGETS, AND RUNNING THE REDUCTION TWICE IS THE POINT. A characteristic
that predicts the distance between a fitted curve and its target, but not the
error in the ANSWER a probabilistic LCA gives, is not worth keeping. Stage 2e
ran the pLCA against the true distributions the synthetic datasets were drawn
from, so the per-material error against truth is on disk and this is a direct
test rather than an argument. `TARGETS` names both families. A characteristic
that survives one and not the other is reported as such, because those are the
interesting ones.

DATASET SIZE IS A CONFOUND IN EVERYTHING HERE, and it is treated as a
first-class predictor rather than as one metric among twenty-one.
`size_confounding` reports how much of each characteristic log(n) alone
explains, and `incremental_over_size` reports what each characteristic adds
ONCE log(n) is already in the model, which is the only form of the question
that cannot be answered by n wearing another name.

MISSINGNESS IS EXPLICIT AND NOT LEFT TO THE DEFAULT. Unbiased excess kurtosis
divides by (n-1)(n-2)(n-3), so it is undefined below n = 4, and the smallest
size stratum runs n = 3 to 9. A complete-case model silently drops those
datasets, which is exactly the regime where a parametric family is expected to
beat a kernel estimate, so a reduction run that way would be blind to the one
place the answer is known to change. Every model here reports the rows it
actually used, per stratum, and the flexible model takes missing values
natively rather than dropping the row.

TWO MODEL FAMILIES, BECAUSE NEITHER ALONE IS ENOUGH. A penalized additive model
on spline bases is interpretable and assumes additivity; gradient boosting is
flexible and assumes nothing. Both are scored out of sample and both are ranked
by the same instrument, permutation importance on held-out folds, so the two
rankings are comparable. Where they disagree, the disagreement is the finding.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


# --------------------------------------------------------------------------
# the candidate set
# --------------------------------------------------------------------------

#: Every characteristic the study computes, as a candidate predictor.
#:
#: Eight are computed under both weightings and carry a `_uw` twin; `n` and
#: `w_v_uw_wasserstein` have one version each; `mean` has one because `mean_uw`
#: is identically 1.0 by construction, every dataset being divided by its own
#: unweighted mean. That is the 21 the current figure draws.
#:
#: THE THREE MODALITY MEASURES ARE THREE CANDIDATES, NOT THREE READINGS OF ONE
#: THING, and they disagree by design. `modality_index` is a continuous index,
#: the summed height of the maxima less the minima of a Scott's-bandwidth
#: kernel density, over the tallest peak. `crit_bw_1` is Silverman's critical
#: bandwidth, the smallest bandwidth at which the density has one mode, in
#: units of the data's own standard deviation. `modes_fitted` counts the local
#: maxima of the density the study ACTUALLY FITS. On real data they disagree
#: profoundly -- 94.9 percent of categories have exactly one visible mode at
#: Scott's bandwidth while only about half are unimodal by Silverman -- so
#: which of them predicts anything is a question, and it is this stage's.
CANDIDATE_METRICS = (
    'n',
    'coeffvar', 'coeffvar_uw',
    'skewness', 'skewness_uw',
    'kurtosis', 'kurtosis_uw',
    'entropy', 'entropy_uw',
    'crit_bw_1', 'crit_bw_1_uw',
    'modality_index', 'modality_index_uw',
    'fit_norm_SF', 'fit_norm_SF_uw',
    'fit_lognorm_SF', 'fit_lognorm_SF_uw',
    'weight_outliers', 'weight_outliers_uw',
    'mean',
    'w_v_uw_wasserstein',
)

#: The visible-mode counts, joined in from the modality table when available.
#: `modes_fitted` is the one decision 82 says to report, because it counts the
#: modes of the density a reader is shown and the pLCA samples from;
#: `modes_scipy_default` is at Scott's bandwidth and is the generator's tuning
#: target, kept as a third candidate so the disagreement can be measured.
MODE_METRICS = ('modes_fitted', 'modes_scipy_default')

#: The author's modality index computed at the bandwidth the study FITS, added
#: in Stage 2f. `modality_index` is the same index at Scott's rule, which is
#: what it was written with and what the study stopped using in Stage 2b; the
#: two are kept side by side because only the second is the author's measure
#: applied to the density a reader is shown. Decision 134.
EXTRA_METRICS = ('modality_index_fitted', 'modality_index_fitted_uw')

#: How each candidate enters a LINEAR model. The flexible model needs none of
#: this -- a tree does not care about monotone reparameterization -- but knots
#: placed on a raw scale that spans four orders of magnitude all land in the
#: first percent of the range, so the additive model needs the transform to be
#: given a fair hearing. Each is the standard one for its quantity:
#:
#:  `log`        strictly positive and spanning orders of magnitude
#:  `signed_log` signed and heavy tailed; sign(x) * log1p(|x|)
#:  `log1p`      non-negative, bounded below by zero, often exactly zero
#:  `cloglog`    a statistic on (0, 1] near its ceiling; log(1 - W), which is
#:               the scale Royston's own normalizing transform works on
#:  `identity`   already on a sensible scale
METRIC_TRANSFORM = {
    'n': 'log',
    'coeffvar': 'log', 'coeffvar_uw': 'log',
    'skewness': 'signed_log', 'skewness_uw': 'signed_log',
    'kurtosis': 'signed_log', 'kurtosis_uw': 'signed_log',
    'entropy': 'identity', 'entropy_uw': 'identity',
    'crit_bw_1': 'log', 'crit_bw_1_uw': 'log',
    'modality_index': 'identity', 'modality_index_uw': 'identity',
    'fit_norm_SF': 'cloglog', 'fit_norm_SF_uw': 'cloglog',
    'fit_lognorm_SF': 'cloglog', 'fit_lognorm_SF_uw': 'cloglog',
    'weight_outliers': 'log1p', 'weight_outliers_uw': 'log1p',
    'mean': 'log',
    'w_v_uw_wasserstein': 'log',
    'modes_fitted': 'identity', 'modes_scipy_default': 'identity',
    'modality_index_fitted': 'identity',
    'modality_index_fitted_uw': 'identity',
}

#: The targets the reduction is run against, and what each one is FOR.
#:
#: The first family is the FIT score: how far the fitted curve sits from its
#: target. The second is the DOWNSTREAM error: how wrong the answer is. They
#: are different questions and the whole design of this stage is to ask both.
TARGETS = {
    # fit score, in sample. What every panel of the current figure plots.
    'w1': dict(family='fit', scale='log', label='W1, in sample'),
    # fit score, out of sample. The non-circular version: cross-validated on
    # the empirical arm, against the known parent on the synthetic one.
    'w1_cv': dict(family='fit', scale='log', label='W1, cross-validated'),
    'w1_market': dict(family='fit', scale='log',
                      label='W1 against the market-weighted parent'),
    # downstream error, synthetic arm only, from the run against the truth.
    'err_eci_mean': dict(family='answer', scale='log',
                         label="error in a material's estimated contribution"),
    'err_eci_rank_1': dict(family='answer', scale='log',
                           label="error in a material's chance of leading"),
    'err_ui': dict(family='answer', scale='log',
                   label="error in a material's share of total variance"),
    'err_eci_p95': dict(family='answer', scale='log',
                        label="error in a material's 95th percentile"),
    # the CHOICE between two families, which is the question the paper asks.
    # Already a log ratio, so it is modeled on its own scale.
    'log_ratio': dict(family='choice', scale='identity',
                      label='log(W1_KDE / W1_lognormal), same weighting'),
}

#: Size bands, matching the corpus strata, so every count can be reported per
#: stratum as the stage requires.
SIZE_BANDS = ((3, 9), (10, 99), (100, 999), (1000, 10 ** 9))
SIZE_BAND_LABELS = ('n 3-9', 'n 10-99', 'n 100-999', 'n >= 1000')


def size_band(n):
    for label, (lo, hi) in zip(SIZE_BAND_LABELS, SIZE_BANDS):
        if lo <= n <= hi:
            return label
    return SIZE_BAND_LABELS[-1]


def transform(values, how):
    """Apply one of the named transforms, propagating non-finite values.

    Non-finite input stays non-finite: an undefined kurtosis must arrive at the
    model as missing, not as a number the transform invented.
    """
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    v = x[ok]
    if how == 'identity':
        out[ok] = v
    elif how == 'log':
        with np.errstate(divide='ignore', invalid='ignore'):
            out[ok] = np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), np.nan)
    elif how == 'log1p':
        out[ok] = np.where(v >= 0, np.log1p(np.clip(v, 0, None)), np.nan)
    elif how == 'signed_log':
        out[ok] = np.sign(v) * np.log1p(np.abs(v))
    elif how == 'cloglog':
        # A Shapiro statistic lives on (0, 1] and piles up near 1. log(1 - W)
        # is the scale its own normalizing transform uses. W = 1 exactly is a
        # perfect fit and has no image here, so it is clipped rather than
        # dropped: one dataset scoring exactly 1.0 should not remove a row.
        out[ok] = np.log(np.clip(1.0 - v, 1e-12, None))
    else:
        raise ValueError(f'unknown transform {how!r}')
    return out


def transform_frame(frame, metrics):
    """Every candidate on its modeling scale, as a frame with the same index."""
    return pd.DataFrame(
        {m: transform(frame[m].to_numpy(dtype=float),
                      METRIC_TRANSFORM.get(m, 'identity'))
         for m in metrics},
        index=frame.index)


# --------------------------------------------------------------------------
# assembling the modeling frame
# --------------------------------------------------------------------------

def assemble(characteristics, scores, truth=None, modes=None,
             metrics=CANDIDATE_METRICS):
    """One row per (arm, dataset, method): the characteristics and every target.

    `characteristics` is one row per (arm, dataset); `scores` is
    `TABLE_TargetComparison.csv`; `truth` is the Stage 2e run against the true
    parents, which exists on the synthetic arm only; `modes` is the visible
    mode counts.
    """
    chars = characteristics.copy()
    if modes is not None:
        keep = ['arm', 'dataset'] + [m for m in MODE_METRICS if m in modes]
        # `modes` is the source of truth for these columns, so any copy already
        # sitting in `characteristics` is dropped rather than merged alongside.
        # Merging both leaves pandas' `_x`/`_y` suffixes and every later lookup
        # by the bare name raises. It cannot happen with the production tables,
        # where the mode counts live only in `modes`, which is exactly why it
        # would have surfaced first in somebody's new caller.
        dupes = [m for m in keep[2:] if m in chars.columns]
        if dupes:
            chars = chars.drop(columns=dupes)
        chars = chars.merge(modes[keep], on=['arm', 'dataset'], how='left')

    cols = ['arm', 'dataset', 'method', 'w1', 'w1_cv', 'w1_market', 'w1_parent']
    sc = scores[[c for c in cols if c in scores.columns]].copy()
    frame = sc.merge(chars, on=['arm', 'dataset'], how='left',
                     suffixes=('', '__chars'))

    if truth is not None:
        t = truth[truth.truth_parent == 'market'].copy()
        ren = {f'{o}__error': f'err_{o}' for o in
               ('eci_mean', 'eci_rank_1', 'ui', 'eci_p95', 'eci_std')}
        t = t.rename(columns=ren)
        keep = ['dataset', 'method'] + [v for v in ren.values() if v in t.columns]
        t = t[keep].copy()
        t['arm'] = 'synthetic'
        frame = frame.merge(t, on=['arm', 'dataset', 'method'], how='left')
        # The truth run reports a SIGNED error; what a reduction models is how
        # WRONG the answer is, so the magnitude is the target. The direction is
        # a separate finding and Stage 2e reports it as bias.
        for v in ren.values():
            if v in frame.columns:
                frame[v] = frame[v].abs()

    frame['size_band'] = [size_band(v) for v in frame['n']]
    present = [m for m in metrics if m in frame.columns]
    return frame, present


# --------------------------------------------------------------------------
# what the models are allowed to forget: missingness and redundancy
# --------------------------------------------------------------------------

def missingness_by_band(frame, metrics):
    """How many datasets each metric is DEFINED on, per size band.

    Reported before any model is fitted, because a complete-case model drops
    these rows without saying so. The expected offender is excess kurtosis at
    n = 3.
    """
    rows = []
    per_dataset = frame.drop_duplicates(['arm', 'dataset'])
    for (arm, band), g in per_dataset.groupby(['arm', 'size_band']):
        row = dict(arm=arm, size_band=band, n_datasets=len(g))
        for m in metrics:
            if m in g:
                v = pd.to_numeric(g[m], errors='coerce')
                row[f'defined__{m}'] = int(np.isfinite(v).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    band_order = {b: i for i, b in enumerate(SIZE_BAND_LABELS)}
    return out.sort_values(
        ['arm', 'size_band'],
        key=lambda s: s.map(band_order) if s.name == 'size_band' else s
    ).reset_index(drop=True)


def complete_case_cost(frame, metrics):
    """What a complete-case model would throw away, per arm and size band.

    The number the stage was told to report: a model that drops any row with a
    missing predictor is not modeling the corpus, it is modeling the corpus
    minus its smallest datasets.
    """
    rows = []
    per_dataset = frame.drop_duplicates(['arm', 'dataset'])
    M = transform_frame(per_dataset, metrics)
    complete = np.isfinite(M.to_numpy(float)).all(axis=1)
    per_dataset = per_dataset.assign(_complete=complete)
    for (arm, band), g in per_dataset.groupby(['arm', 'size_band']):
        rows.append(dict(arm=arm, size_band=band, n_datasets=len(g),
                         n_complete=int(g._complete.sum()),
                         share_dropped=float(1.0 - g._complete.mean())))
    for arm, g in per_dataset.groupby('arm'):
        rows.append(dict(arm=arm, size_band='all', n_datasets=len(g),
                         n_complete=int(g._complete.sum()),
                         share_dropped=float(1.0 - g._complete.mean())))
    return pd.DataFrame(rows)


def redundancy(frame, metrics, method='spearman'):
    """The correlation structure, its effective dimension, and the worst pairs.

    Stage 2a Part 6 established that these characteristics are correlated; this
    is the same measurement on the current metric set, carried forward rather
    than rediscovered, and extended with the clustering that decides which
    survivors are redundant with each other.
    """
    per_dataset = frame.drop_duplicates(['arm', 'dataset'])
    out = {}
    for arm, g in per_dataset.groupby('arm'):
        M = transform_frame(g, metrics)
        C = M.corr(method=method, min_periods=10)
        lam = np.linalg.eigvalsh(np.nan_to_num(C.to_numpy(float), nan=0.0))
        lam = np.clip(lam, 0, None)
        out[arm] = dict(
            correlation=C,
            effective_dimension=float(lam.sum() ** 2 / (lam ** 2).sum()),
            n_metrics=len(metrics),
        )
    return out


def redundancy_table(frame, metrics, method='spearman', threshold=0.8):
    """The correlation matrix in long form, plus the effective dimension."""
    res = redundancy(frame, metrics, method)
    rows = []
    for arm, d in res.items():
        C = d['correlation']
        for i, a in enumerate(metrics):
            for b in metrics[i + 1:]:
                if a in C.index and b in C.columns:
                    rows.append(dict(arm=arm, metric_a=a, metric_b=b,
                                     correlation=float(C.loc[a, b]),
                                     effective_dimension=d['effective_dimension'],
                                     n_metrics=d['n_metrics']))
    out = pd.DataFrame(rows)
    out['redundant'] = out.correlation.abs() >= threshold
    return out.sort_values(['arm', 'correlation'],
                           key=lambda s: s.abs() if s.name == 'correlation' else s,
                           ascending=[True, False]).reset_index(drop=True)


def size_confounding(frame, metrics):
    """How much of each characteristic is dataset size wearing another name.

    A metric that predicts which method wins, and is itself largely a function
    of n, is not a second mechanism. This reports the share of each metric's
    variance that log(n) alone explains, by a spline fit so a curved
    relationship is not missed.
    """
    rows = []
    per_dataset = frame.drop_duplicates(['arm', 'dataset'])
    for arm, g in per_dataset.groupby('arm'):
        M = transform_frame(g, metrics)
        logn = transform(g['n'].to_numpy(float), METRIC_TRANSFORM['n'])
        for m in metrics:
            if m == 'n':
                continue
            y = M[m].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(logn)
            if ok.sum() < 30 or np.nanstd(y[ok]) == 0:
                rows.append(dict(arm=arm, metric=m, n_used=int(ok.sum()),
                                 r2_on_log_n=np.nan, spearman_with_log_n=np.nan))
                continue
            X = SplineTransformer(n_knots=5, degree=3).fit_transform(
                logn[ok].reshape(-1, 1))
            X = np.column_stack([np.ones(ok.sum()), X])
            beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
            resid = y[ok] - X @ beta
            ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
            r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
            rho = float(pd.Series(y[ok]).corr(pd.Series(logn[ok]), method='spearman'))
            rows.append(dict(arm=arm, metric=m, n_used=int(ok.sum()),
                             r2_on_log_n=r2, spearman_with_log_n=rho))
    return pd.DataFrame(rows).sort_values(
        ['arm', 'r2_on_log_n'], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------
# the two models
# --------------------------------------------------------------------------

def additive_model(n_knots=5, l1_ratio=(0.1, 0.5, 0.9, 1.0), cv=5,
                   random_state=0):
    """A penalized ADDITIVE model: natural splines per predictor, elastic net.

    This is the interpretable half. It assumes the characteristics act
    additively on the log of the target and it shrinks, so a predictor earns
    its coefficients or loses them. Missing values are median-imputed WITH an
    indicator column, so `kurtosis is undefined` is itself a predictor the
    model can use rather than a row it drops.
    """
    return Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('spline', SplineTransformer(n_knots=n_knots, degree=3,
                                     extrapolation='linear')),
        ('scale', StandardScaler()),
        ('net', ElasticNetCV(l1_ratio=list(l1_ratio), cv=cv, alphas=40,
                             max_iter=20000, random_state=random_state)),
    ])


def boosted_model(random_state=0, **kw):
    """The flexible half: gradient boosting, which takes NaN natively.

    No imputation and no dropped row: the tree sends a missing value down
    whichever branch the training data says is better, so the smallest stratum
    stays in the model with its kurtosis undefined.
    """
    params = dict(max_depth=None, max_leaf_nodes=15, learning_rate=0.06,
                  max_iter=400, min_samples_leaf=20, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.15,
                  random_state=random_state)
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


#: Cores for permutation importance. It is bit-identical across `n_jobs` for a
#: fixed `random_state` -- sklearn derives each feature's permutation seed
#: deterministically -- so parallelising changes the wall clock and nothing
#: else. `tests/test_reduction.py` pins that.
PERMUTATION_JOBS = -1


def _fit_score_permute(model, X, y, rng, n_repeats=10, n_splits=5):
    """Out-of-sample R2 and permutation importance, averaged over CV folds.

    Permutation importance is computed on the HELD-OUT fold of each split, so
    it measures what a predictor is worth for prediction rather than how much
    the model leaned on it while memorizing. It permutes the RAW column, so for
    the additive model a metric's whole spline basis moves together and a
    metric is credited once rather than once per basis function.
    """
    seed = int(rng.integers(0, 2 ** 31 - 1))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    imps, r2s = [], []
    for tr, te in kf.split(X):
        m = model()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        r2s.append(r2_score(y[te], pred))
        pi = permutation_importance(
            m, X[te], y[te], scoring='r2', n_repeats=n_repeats,
            random_state=seed, n_jobs=PERMUTATION_JOBS)
        imps.append(pi.importances_mean)
    return np.mean(r2s), np.std(r2s), np.mean(imps, axis=0), np.std(imps, axis=0)


def _prepare_xy(frame, metrics, target, scale):
    M = transform_frame(frame, metrics)
    y = pd.to_numeric(frame[target], errors='coerce').to_numpy(float)
    if scale == 'log':
        # A distance of exactly zero cannot happen here -- no fitted model is
        # its own target -- but a floor is kept so one pathological row cannot
        # remove itself from the model silently.
        y = np.log(np.clip(y, 1e-12, None))
    keep = np.isfinite(y)
    return M.to_numpy(float)[keep], y[keep], keep


def importance(frame, metrics, target, arm=None, method=None,
               rng=None, n_repeats=10, n_splits=5, models=('additive', 'boosted')):
    """Rank the candidates by independent predictive contribution.

    Returns one row per (model, metric) with the permutation importance, its
    spread across folds, and the model's own out-of-sample R2, so an importance
    can never be read without knowing whether the model predicts anything at
    all. A large importance inside a model with R2 near zero means nothing.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    g = frame
    if arm is not None:
        g = g[g.arm == arm]
    if method is not None:
        g = g[g.method == method]
    scale = TARGETS.get(target, {}).get('scale', 'identity')
    X, y, keep = _prepare_xy(g, metrics, target, scale)
    if len(y) < 40:
        return pd.DataFrame()

    builders = {'additive': lambda: additive_model(random_state=int(rng.integers(1 << 30))),
                'boosted': lambda: boosted_model(random_state=int(rng.integers(1 << 30)))}
    rows = []
    for name in models:
        r2m, r2s, imp, ispread = _fit_score_permute(
            builders[name], X, y, rng, n_repeats=n_repeats, n_splits=n_splits)
        for m, v, s in zip(metrics, imp, ispread):
            rows.append(dict(arm=arm, method=method, target=target, model=name,
                             metric=m, importance=float(v),
                             importance_sd_across_folds=float(s),
                             model_r2=float(r2m), model_r2_sd=float(r2s),
                             n_rows=int(len(y))))
    return pd.DataFrame(rows)


def incremental_over_size(frame, metrics, target, arm=None, method=None):
    """What each characteristic adds ONCE log(n) is already in the model.

    This is the form of the question Stage 2c asked of one targeted quantity
    and this stage asks of everything. A spline in log(n) is the base model;
    each candidate is then added, as a spline, and the increment in adjusted R2
    with an F test is reported. Complete cases on the pair only, so a metric is
    not penalized for another metric's missingness, and the row count is
    reported with every increment.
    """
    from scipy import stats as sps
    g = frame
    if arm is not None:
        g = g[g.arm == arm]
    if method is not None:
        g = g[g.method == method]
    scale = TARGETS.get(target, {}).get('scale', 'identity')
    y_all = pd.to_numeric(g[target], errors='coerce').to_numpy(float)
    if scale == 'log':
        y_all = np.log(np.clip(y_all, 1e-12, None))
    M = transform_frame(g, metrics)
    # log(n) is the base model, so it has to come from the FRAME and not from
    # the candidate list: `modality_head_to_head` offers only the modality
    # measures, and taking size from the list would then ask the frame for a
    # column the caller never named. It cost an audit run.
    logn = transform(g['n'].to_numpy(float), METRIC_TRANSFORM['n'])

    def basis(v):
        return SplineTransformer(n_knots=5, degree=3).fit_transform(
            np.asarray(v, float).reshape(-1, 1))

    rows = []
    for m in metrics:
        if m == 'n':
            continue
        x = M[m].to_numpy(float)
        ok = np.isfinite(y_all) & np.isfinite(logn) & np.isfinite(x)
        nobs = int(ok.sum())
        if nobs < 40 or np.std(x[ok]) == 0:
            rows.append(dict(arm=arm, method=method, target=target, metric=m,
                             n_rows=nobs, r2_size_only=np.nan,
                             r2_with_metric=np.nan, incremental_r2=np.nan,
                             f_stat=np.nan, p_value=np.nan))
            continue
        y = y_all[ok]
        B0 = np.column_stack([np.ones(nobs), basis(logn[ok])])
        B1 = np.column_stack([B0, basis(x[ok])])

        def rss(B):
            beta, *_ = np.linalg.lstsq(B, y, rcond=None)
            r = y - B @ beta
            return float((r ** 2).sum()), np.linalg.matrix_rank(B)

        rss0, k0 = rss(B0)
        rss1, k1 = rss(B1)
        sst = float(((y - y.mean()) ** 2).sum())
        r2_0 = 1 - rss0 / sst if sst > 0 else np.nan
        r2_1 = 1 - rss1 / sst if sst > 0 else np.nan
        df1, df2 = max(k1 - k0, 1), max(nobs - k1, 1)
        f = ((rss0 - rss1) / df1) / (rss1 / df2) if rss1 > 0 else np.nan
        p = float(sps.f.sf(f, df1, df2)) if np.isfinite(f) and f > 0 else np.nan
        rows.append(dict(arm=arm, method=method, target=target, metric=m,
                         n_rows=nobs, r2_size_only=r2_0, r2_with_metric=r2_1,
                         incremental_r2=r2_1 - r2_0, f_stat=f, p_value=p))
    return pd.DataFrame(rows).sort_values('incremental_r2', ascending=False)


# --------------------------------------------------------------------------
# which method wins
# --------------------------------------------------------------------------

def winner_frame(frame, value='w1', within_weighting=False):
    """One row per (arm, dataset): which method scored best on `value`.

    `within_weighting` reduces it to the question the paper actually asks --
    the FAMILY, kernel estimate against three-parameter lognormal against
    normal, holding the weighting scheme fixed -- because comparing across
    weightings on a cross-validated score is invalid (decision 65) and the
    six-way label mixes the two questions.
    """
    g = frame.dropna(subset=[value]).copy()
    if within_weighting:
        g['weighting'] = np.where(g.method.str.endswith('Variable'),
                                  'Variable', 'Uniform')
        g['family'] = g.method.str.split(',').str[0]
        keys = ['arm', 'dataset', 'weighting']
        idx = g.groupby(keys)[value].idxmin()
        out = g.loc[idx, keys + ['family', value]].rename(
            columns={'family': 'winner', value: 'winning_score'})
    else:
        keys = ['arm', 'dataset']
        idx = g.groupby(keys)[value].idxmin()
        out = g.loc[idx, keys + ['method', value]].rename(
            columns={'method': 'winner', value: 'winning_score'})
    chars = frame.drop_duplicates(['arm', 'dataset'])
    carry = [c for c in chars.columns
             if c in CANDIDATE_METRICS + MODE_METRICS + EXTRA_METRICS
             or c == 'size_band']
    return out.merge(chars[['arm', 'dataset'] + carry], on=['arm', 'dataset'],
                     how='left')


def winner_importance(winners, metrics, arm=None, weighting=None, rng=None,
                      n_repeats=10, n_splits=5):
    """Permutation importance for predicting WHICH method wins.

    Scored by accuracy against the majority-class baseline, which is printed
    beside it: on a question where one method wins 70 percent of the time, an
    accuracy of 0.70 is a model that has learned nothing, and an importance
    ranking under it means nothing either.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    g = winners
    if arm is not None:
        g = g[g.arm == arm]
    if weighting is not None and 'weighting' in g:
        g = g[g.weighting == weighting]
    if len(g) < 40 or g.winner.nunique() < 2:
        return pd.DataFrame()

    X = transform_frame(g, metrics).to_numpy(float)
    y = g.winner.to_numpy()
    counts = pd.Series(y).value_counts(normalize=True)
    baseline = float(counts.iloc[0])

    seed = int(rng.integers(0, 2 ** 31 - 1))
    n_splits = min(n_splits, int(pd.Series(y).value_counts().min()))
    if n_splits < 2:
        return pd.DataFrame()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    for name, build in (
        ('additive', lambda: Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True)),
            ('spline', SplineTransformer(n_knots=5, degree=3,
                                         extrapolation='linear')),
            ('scale', StandardScaler()),
            # l1_ratio=0 is ridge; `penalty=` is deprecated in this sklearn.
            ('logit', LogisticRegression(l1_ratio=0.0, C=0.5,
                                         max_iter=5000))])),
        ('boosted', lambda: HistGradientBoostingClassifier(
            max_leaf_nodes=15, learning_rate=0.06, max_iter=300,
            min_samples_leaf=20, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, random_state=seed)),
    ):
        accs, imps = [], []
        for tr, te in skf.split(X, y):
            m = build()
            m.fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], m.predict(X[te])))
            pi = permutation_importance(m, X[te], y[te], scoring='accuracy',
                                        n_repeats=n_repeats,
                                        random_state=seed,
                                        n_jobs=PERMUTATION_JOBS)
            imps.append(pi.importances_mean)
        imp = np.mean(imps, axis=0)
        spread = np.std(imps, axis=0)
        for mm, v, s in zip(metrics, imp, spread):
            rows.append(dict(arm=arm, weighting=weighting, model=name,
                             metric=mm, importance=float(v),
                             importance_sd_across_folds=float(s),
                             accuracy=float(np.mean(accs)),
                             majority_baseline=baseline,
                             lift_over_baseline=float(np.mean(accs)) - baseline,
                             n_rows=int(len(y))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the survivors, and the curves that replace the rolling averages
# --------------------------------------------------------------------------

def rank_survivors(importances, top=5, min_r2=0.02):
    """Pool the per-(arm, method, target) importances into one ranking.

    A metric's score is its MEAN RANK across every model that predicted
    anything at all, which is deliberately robust: one arm, one method or one
    target cannot promote a metric on its own, and a model whose out-of-sample
    R2 is below `min_r2` contributes nothing because its importances are
    ranking noise.
    """
    usable = importances[importances.model_r2 >= min_r2].copy()
    if not len(usable):
        return pd.DataFrame()
    keys = ['arm', 'method', 'target', 'model']
    # A MISSING KEY WOULD SILENTLY DROP A WHOLE TARGET FAMILY. pandas' groupby
    # discards a NaN key by default, so concatenating importances whose key
    # columns do not all line up loses rows with no error at all -- the choice
    # family disappeared this way once. Fail loudly instead.
    missing = [k for k in keys if k not in usable.columns]
    if missing:
        raise ValueError(f'importances lack the grouping keys {missing}')
    blank = usable[keys].isna().any(axis=1)
    if blank.any():
        raise ValueError(
            f'{int(blank.sum())} importance rows have a missing grouping key; '
            f'groupby would drop them silently')
    usable['rank_within_model'] = (usable.groupby(keys).importance
                                   .rank(ascending=False, method='average'))
    agg = (usable.groupby('metric')
           .agg(mean_rank=('rank_within_model', 'mean'),
                median_rank=('rank_within_model', 'median'),
                mean_importance=('importance', 'mean'),
                share_top5=('rank_within_model', lambda s: float((s <= 5).mean())),
                n_models=('rank_within_model', 'size'))
           .reset_index()
           .sort_values('mean_rank'))
    agg['survivor'] = np.arange(len(agg)) < top
    return agg.reset_index(drop=True)


def binned_curve(x, y, n_bins=12, n_boot=1000, rng=None, equal_count=True):
    """Binned means with a bootstrap band and the count in every bin.

    THIS IS WHAT REPLACES THE ROLLING AVERAGE, and the three things it adds are
    the three that were missing. The band is a percentile bootstrap WITHIN each
    bin, so a bin holding four datasets gets a band wide enough to say so. The
    count travels with the bin, so a reader can see where the data are. And the
    bins are equal-COUNT by default rather than equal-width, so no bin is drawn
    from a handful of points in the first place.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < n_bins * 2:
        n_bins = max(2, len(x) // 4)
    if equal_count:
        q = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(x, q))
    else:
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
    if len(edges) < 3:
        return pd.DataFrame()
    idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        k = int(sel.sum())
        if k == 0:
            continue
        yy = y[sel]
        if k > 1:
            draws = rng.integers(0, k, size=(n_boot, k))
            boots = yy[draws].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
        else:
            lo = hi = float(yy[0])
        rows.append(dict(bin=b, x_lo=float(edges[b]), x_hi=float(edges[b + 1]),
                         x_center=float(np.median(x[sel])), count=k,
                         mean=float(yy.mean()), median=float(np.median(yy)),
                         lo=float(lo), hi=float(hi)))
    return pd.DataFrame(rows)


def lowess_curve(x, y, frac=0.3, n_out=100, n_boot=0, rng=None,
                 bounds=None):
    """A LOWESS smooth, optionally with a bootstrap band.

    **THE BAND COMES FROM `binned_curve`, NOT FROM HERE, AND THAT IS A COST
    DECISION MADE ON A MEASUREMENT.** statsmodels' LOWESS runs three
    robustifying iterations by default, and on 10,000 datasets one fit takes
    about 600 ms, so a 400-replicate band for five characteristics by six
    methods by three targets would be some 12,000 fits and several hours.
    Turning the iterations off makes a fit 255 times faster and is NOT
    available: measured on this data it moves the curve by 54 percent of its
    own range on the target's scale and by 14 percent on the log scale, which
    is six times the width of the band it would be drawn inside. A curve that
    changes that much is a different curve, not a faster one.

    So the division of labour is: `binned_curve` carries the uncertainty, with
    a within-bin percentile bootstrap and equal-count bins that is exact,
    cheap and reports its own counts; this function is the smooth read through
    those bins, fitted once with the robustifying iterations intact.
    `n_boot > 0` still gives a band, for a small arm where it is affordable.

    **`bounds` STOPS THE SMOOTHER EXTRAPOLATING PAST ITS DATA.** A local linear
    fit at the edge of the sample projects the local slope outward, and on the
    empirical arm that carried the curve for dataset size from 0.026 at the last
    populated bin down through zero to **-0.0067**, a negative Wasserstein
    distance. Passing the first and last bin centers confines the curve to the
    span the binned summary actually covers, which is where the band is drawn
    and so the only place the two can be read together.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20:
        return pd.DataFrame()
    if bounds is not None and np.isfinite(bounds).all():
        glo, ghi = float(bounds[0]), float(bounds[1])
    else:
        glo, ghi = np.percentile(x, 1), np.percentile(x, 99)
    if not (ghi > glo):
        return pd.DataFrame()
    grid = np.linspace(glo, ghi, n_out)
    fit = _lowess(y, x, frac=frac, xvals=grid, return_sorted=False)

    if n_boot:
        boots = np.empty((n_boot, n_out))
        for b in range(n_boot):
            s = rng.integers(0, len(x), len(x))
            try:
                boots[b] = _lowess(y[s], x[s], frac=frac, xvals=grid,
                                   return_sorted=False)
            except Exception:
                boots[b] = np.nan
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    else:
        lo = hi = np.full(n_out, np.nan)
    # Local density, for the rug: how many datasets sit within one smoothing
    # window of each grid point. It is what makes a sparse region visible.
    half = frac * (x.max() - x.min()) / 2
    dens = np.array([int(np.sum(np.abs(x - g) <= half)) for g in grid])
    return pd.DataFrame(dict(x=grid, fit=fit, lo=lo, hi=hi, local_count=dens))


def curves_for(frame, metrics, methods, value='w1', arm=None, rng=None,
               n_bins=12, frac=0.35, lowess_boot=0):
    """Binned and smoothed curves for every (metric, method) pair on one arm.

    The BINNED rows carry the uncertainty and the counts; the LOWESS rows are
    the smooth read through them. See `lowess_curve` for why the band is not
    taken from the smoother.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    g = frame if arm is None else frame[frame.arm == arm]
    scale = TARGETS.get(value, {}).get('scale', 'identity')
    out = []
    for metric in metrics:
        if metric not in g:
            continue
        xt = transform(g[metric].to_numpy(float),
                       METRIC_TRANSFORM.get(metric, 'identity'))
        for method in methods:
            sel = (g.method == method).to_numpy()
            y = pd.to_numeric(g[value], errors='coerce').to_numpy(float)
            b = binned_curve(xt[sel], y[sel], n_bins=n_bins, rng=rng)
            span = ((b.x_center.min(), b.x_center.max()) if len(b) else None)
            if len(b):
                b.insert(0, 'method', method)
                b.insert(0, 'characteristic', metric)
                b.insert(0, 'arm', arm)
                b['kind'] = 'binned'
                b['scale'] = scale
                out.append(b)
            l = lowess_curve(xt[sel], y[sel], frac=frac, rng=rng,
                             n_boot=lowess_boot, bounds=span)
            if len(l):
                l = l.rename(columns={'x': 'x_center', 'fit': 'mean',
                                      'local_count': 'count'})
                l.insert(0, 'method', method)
                l.insert(0, 'characteristic', metric)
                l.insert(0, 'arm', arm)
                l['kind'] = 'lowess'
                l['scale'] = scale
                out.append(l)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# does the corpus cover the metrics that matter?
# --------------------------------------------------------------------------

def coverage_versus_importance(coverage, survivors):
    """THE GENERALIZATION QUESTION, answered as a join rather than an opinion.

    The generator's tuning objective matches the SHAPE of the synthetic
    characteristic distribution to the empirical one, while the study also
    needs to SPAN that space with margin so its conclusions generalize past
    the categories EC3 happens to hold. Those two goals pull apart wherever
    matching concentrates the corpus where real data is dense.

    If the characteristics that predict which method wins are the ones where
    the corpus is densest and reaches least far, the generalization claim is
    narrower than the coverage figure suggests. This puts the per-metric
    importance next to that metric's coverage margin so the question has an
    answer with a number in it.
    """
    cov = coverage.copy()
    cov['metric'] = cov.metric.astype(str)
    out = survivors.merge(cov, on='metric', how='left')
    keep = ['metric', 'mean_rank', 'mean_importance', 'survivor',
            'empirical_covered', 'margin_below', 'margin_above',
            'synthetic_beyond_empirical']
    out = out[[c for c in keep if c in out.columns]]
    if {'mean_rank', 'margin_above'} <= set(out.columns):
        sub = out.dropna(subset=['mean_rank', 'margin_above'])
        if len(sub) > 3:
            out.attrs['spearman_rank_vs_margin_above'] = float(
                sub.mean_rank.corr(sub.margin_above, method='spearman'))
            out.attrs['spearman_rank_vs_beyond'] = float(
                sub.mean_rank.corr(sub.synthetic_beyond_empirical,
                                   method='spearman'))
    return out


# --------------------------------------------------------------------------
# post-stratification, and the three modality measures head to head
# --------------------------------------------------------------------------

def empirical_size_shares(frame):
    """The share of the EMPIRICAL arm's datasets in each size band.

    Measured from the arm rather than taken from a stored constant, which is
    the mistake Stage 2c had to correct once: a reweighting that silently used
    a stale denominator would move every post-stratified number and leave
    nothing to catch it.
    """
    one = frame[frame.arm == 'empirical'].drop_duplicates('dataset')
    counts = one.size_band.value_counts()
    total = float(counts.sum())
    return {b: float(counts.get(b, 0)) / total for b in SIZE_BAND_LABELS}


def size_mix_resample(frame, shares=None, rng=None, n_datasets=None):
    """The corpus resampled to the EMPIRICAL size mix, so an aggregate can be
    reported both ways.

    WHY THIS AND NOT A WEIGHTED FIT. The corpus allocates 2,500 datasets to
    each of four size bands for equal PRECISION, while the empirical arm is
    nothing like that -- about 14 / 53 / 26 / 7 percent. The KDE improves with
    dataset size and the lognormal degrades, so an aggregate over the corpus is
    partly a statement about the allocation. Permutation importance has no
    sample-weight argument that means what is wanted here, so the honest
    reweighting is to resample DATASETS to the empirical mix and refit. Every
    row of a resampled dataset travels with it, so a method's six rows stay
    together and the pairing is preserved.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    shares = shares or empirical_size_shares(frame)
    syn = frame[frame.arm == 'synthetic']
    per_dataset = syn.drop_duplicates('dataset')[['dataset', 'size_band']]
    if n_datasets is None:
        # The largest corpus size that can supply every band at the empirical
        # mix without sampling any band with replacement.
        avail = per_dataset.size_band.value_counts()
        n_datasets = int(min(avail.get(b, 0) / s
                             for b, s in shares.items() if s > 0))
    picked = []
    for band, share in shares.items():
        pool = per_dataset[per_dataset.size_band == band].dataset.to_numpy()
        k = int(round(share * n_datasets))
        if k == 0 or len(pool) == 0:
            continue
        picked.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    keep = set(np.concatenate(picked)) if picked else set()
    return syn[syn.dataset.isin(keep)].copy()


def modality_head_to_head(frame, targets, methods, arm, rng=None,
                          n_repeats=10, n_splits=5):
    """The three modality measures as three candidates, judged against each
    other and against log(n).

    THE QUESTION STAGE 2A-2 COULD NOT ANSWER. `modality_index` is a continuous
    index off a Scott's-bandwidth density; `crit_bw_1` is Silverman's critical
    bandwidth; `modes_fitted` counts the modes of the density the study
    actually fits. They disagree profoundly on real data, so "is the dataset
    multimodal" is not one predictor with three readings, and which reading
    predicts anything is a separate question from whether modality matters.

    Each is offered ALONE over a spline in log(n), so the three are compared on
    equal terms and none of them can borrow credit from another.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    measures = [m for m in ('modality_index', 'modality_index_uw',
                            'crit_bw_1', 'crit_bw_1_uw',
                            'modes_fitted', 'modes_scipy_default')
                if m in frame.columns]
    rows = []
    for target in targets:
        if target not in frame.columns:
            continue
        for method in methods:
            inc = incremental_over_size(frame, measures, target, arm=arm,
                                        method=method)
            if len(inc):
                rows.append(inc)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    summary = (out.groupby('metric')
               .agg(mean_incremental_r2=('incremental_r2', 'mean'),
                    max_incremental_r2=('incremental_r2', 'max'),
                    median_p_value=('p_value', 'median'),
                    share_p_below_05=('p_value', lambda s: float((s < 0.05).mean())),
                    mean_r2_size_only=('r2_size_only', 'mean'),
                    n_models=('incremental_r2', 'size'))
               .reset_index()
               .sort_values('mean_incremental_r2', ascending=False))
    summary.insert(0, 'arm', arm)
    return summary


def modality_agreement(frame):
    """How far apart the three modality measures actually are, per arm.

    Reported as a correlation and as the share of datasets each calls
    multimodal, because "they disagree" is a claim that needs a number: at
    Scott's bandwidth 94.9 percent of real categories have one visible mode
    while only about half are unimodal by Silverman.
    """
    rows = []
    per_dataset = frame.drop_duplicates(['arm', 'dataset'])
    for arm, g in per_dataset.groupby('arm'):
        measures = [m for m in ('modality_index', 'crit_bw_1',
                                'modes_fitted', 'modes_scipy_default')
                    if m in g.columns]
        M = transform_frame(g, measures)
        for i, a in enumerate(measures):
            for b in measures[i + 1:]:
                ok = M[a].notna() & M[b].notna()
                rows.append(dict(
                    arm=arm, measure_a=a, measure_b=b, n=int(ok.sum()),
                    spearman=float(M.loc[ok, a].corr(M.loc[ok, b],
                                                     method='spearman'))))
        for m in ('modes_fitted', 'modes_scipy_default'):
            if m in g:
                # OVER THE DATASETS THE MEASURE IS DEFINED ON, which is not the
                # arm: a mode count needs at least 8 values, so 17 of the 147
                # real categories and about 2,000 of the 10,000 synthetic ones
                # have none. Dividing by the arm counts an undefined dataset as
                # multimodal and understates the share -- it read 0.605 against
                # a true 0.685 on the empirical arm before this was fixed. The
                # count is returned beside it so the denominator is visible.
                defined = g[m].notna()
                rows.append(dict(arm=arm, measure_a=m,
                                 measure_b='share_unimodal',
                                 n=int(defined.sum()),
                                 spearman=np.nan,
                                 share_unimodal=(
                                     float((g.loc[defined, m] == 1).mean())
                                     if defined.any() else np.nan),
                                 n_arm=int(len(g))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the tautology guard
# --------------------------------------------------------------------------

#: A candidate that is DEFINITIONALLY tied to a fit target rather than
#: predictive of it. `w_v_uw_wasserstein` is the Wasserstein distance between
#: the uniform-weighted and the variable-weighted version of the same dataset,
#: and the study scores every model against the VARIABLE-weighted target, so
#: for a uniform-weighted method it is exactly the part of the score that no
#: estimator can remove. Its Spearman correlation with `w1_definitional` is
#: 1.000 for all three uniform methods, by construction. Ranking it first on a
#: fit target is therefore an identity being rediscovered, not a finding; its
#: standing on the DOWNSTREAM error, where no such identity exists, is a real
#: result and is reported separately.
DEFINITIONAL_CANDIDATES = ('w_v_uw_wasserstein',)


def definitional_check(frame, scores, metrics=CANDIDATE_METRICS,
                       targets=('w1', 'w1_market', 'w1_parent',
                                'err_eci_mean', 'err_eci_rank_1')):
    """Is a candidate predicting a target, or IS it part of that target?

    A fit score decomposes into the part an estimator can remove and a
    definitional part it cannot. A candidate that reproduces the definitional
    part exactly is not a predictor of the score, and a reduction that ranks
    it first has rediscovered an identity. This reports, per method and
    candidate, the rank correlation with the definitional term and with each
    target, so the identity is visible in a table instead of being argued
    about.
    """
    defn = scores[['arm', 'dataset', 'method']].copy()
    for c in ('w1_definitional',):
        if c in scores.columns:
            defn[c] = scores[c]
    g = frame.merge(defn, on=['arm', 'dataset', 'method'], how='left',
                    suffixes=('', '__d'))
    rows = []
    for (arm, method), h in g.groupby(['arm', 'method']):
        M = transform_frame(h, metrics)
        for m in metrics:
            row = dict(arm=arm, method=method, metric=m,
                       is_declared_definitional=m in DEFINITIONAL_CANDIDATES)
            for col, tag in [('w1_definitional', 'definitional')] + \
                            [(t, t) for t in targets]:
                src = h[col] if col in h.columns else None
                if src is None or src.notna().sum() < 30:
                    row[f'spearman_{tag}'] = np.nan
                    continue
                ok = M[m].notna() & src.notna()
                row[f'spearman_{tag}'] = (
                    float(M.loc[ok, m].corr(src[ok], method='spearman'))
                    if ok.sum() >= 30 else np.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
    # A correlation of 1.000 with the definitional term is an identity, not a
    # strong relationship, and the flag says so without a threshold argument.
    if 'spearman_definitional' in out:
        out['is_an_identity'] = out.spearman_definitional.abs() > 0.9999
    return out


def rows_used_by_band(frame, targets, methods=None):
    """How many datasets each model ACTUALLY uses, per size band.

    THE EXPLICIT ANSWER TO "report how many datasets each model uses, per
    stratum", and it catches a second exclusion that the predictor-side
    missingness report does not see. A row is used when its TARGET is defined:
    the predictors never remove one, because the additive model imputes with an
    indicator and the boosted model splits on missingness natively.

    **The cross-validated empirical target is undefined below n = 10**, because
    a half of a nine-value dataset is four values and a held-out score on four
    values is not a measurement. So the out-of-sample reduction on the real arm
    cannot speak about the smallest size band AT ALL -- 20 of 147 categories --
    which is exactly the band where a parametric family is expected to beat a
    kernel estimate. That is a limitation of the target and not of the models,
    the in-sample target covers all 147, and the two must be read together.
    """
    rows = []
    methods = methods if methods is not None else sorted(frame.method.unique())
    for target in targets:
        if target not in frame.columns:
            continue
        for (arm, method), g in frame.groupby(['arm', 'method']):
            if method not in methods:
                continue
            row = dict(arm=arm, method=method, target=target,
                       rows_total=len(g),
                       rows_used=int(g[target].notna().sum()))
            for band in SIZE_BAND_LABELS:
                h = g[g.size_band == band]
                row[f'used__{band}'] = int(h[target].notna().sum())
                row[f'available__{band}'] = int(len(h))
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# partial dependence: what a characteristic is worth with the others held
# --------------------------------------------------------------------------

def partial_dependence_table(frame, metrics, target, which, arm=None,
                             method=None, rng=None, grid_points=30,
                             models=('additive', 'boosted')):
    """Partial dependence of the target on each of `which`, on one fitted model.

    THE MARGINAL CURVE AND THIS ONE ANSWER DIFFERENT QUESTIONS, AND THE GAP
    BETWEEN THEM IS THE POINT OF THE WHOLE STAGE. A binned mean, or the rolling
    average it replaces, shows how the score varies ACROSS datasets that differ
    in this characteristic -- and those datasets differ in every other
    characteristic too, because the characteristics are correlated. Partial
    dependence averages the fitted model over the observed joint distribution
    of everything else, so it shows what moving this characteristic ALONE is
    worth. A characteristic with a steep marginal curve and a flat partial
    dependence is one whose apparent effect belongs to something it travels
    with, and dataset size is what it usually travels with here.

    The curve is on the MODELING scale of the characteristic, and the target is
    on its own (log for every target here), so a slope reads as an elasticity.
    `grid_points` values are taken between the 5th and 95th percentiles, which
    keeps the curve inside the data rather than extrapolating a spline past its
    last knot.

    **TWO NEARLY COLLINEAR PREDICTORS SPLIT THE EFFECT RATHER THAN ONE TAKING
    IT, and a reader of these curves has to know that.** Given a predictor and
    a noisy copy of it, neither model can tell which one the target depends on,
    so both come back with a partial dependence of intermediate size --
    measured on a planted example, the copy keeps about two thirds of the
    original's range rather than collapsing to zero. So a SMALL partial
    dependence is evidence that a characteristic carries nothing, while a
    moderate one is not evidence that it carries something of its own if it is
    strongly correlated with a survivor. Read this table beside
    `redundancy_table`, and treat a pair above about 0.9 as one quantity.
    `tests/test_metricreduction.py` pins the behaviour on a planted copy.
    """
    from sklearn.inspection import partial_dependence

    rng = rng if rng is not None else np.random.default_rng(0)
    g = frame
    if arm is not None:
        g = g[g.arm == arm]
    if method is not None:
        g = g[g.method == method]
    scale = TARGETS.get(target, {}).get('scale', 'identity')
    X, y, _ = _prepare_xy(g, metrics, target, scale)
    if len(y) < 60:
        return pd.DataFrame()

    builders = {
        'additive': lambda: additive_model(random_state=int(rng.integers(1 << 30))),
        'boosted': lambda: boosted_model(random_state=int(rng.integers(1 << 30))),
    }
    index = {m: i for i, m in enumerate(metrics)}
    rows = []
    for name in models:
        model = builders[name]()
        model.fit(X, y)
        for m in which:
            if m not in index:
                continue
            col = X[:, index[m]]
            finite = col[np.isfinite(col)]
            if len(finite) < 30 or np.ptp(finite) == 0:
                continue
            grid = np.linspace(np.percentile(finite, 5),
                               np.percentile(finite, 95), grid_points)
            try:
                pd_res = partial_dependence(
                    model, X, [index[m]], grid_resolution=grid_points,
                    percentiles=(0.05, 0.95), kind='average')
            except Exception:
                continue
            xs = np.asarray(pd_res['grid_values'][0], dtype=float)
            ys = np.asarray(pd_res['average'][0], dtype=float)
            rows.append(pd.DataFrame(dict(
                arm=arm, method=method, target=target, model=name,
                characteristic=m, x=xs, partial_dependence=ys,
                n_rows=len(y))))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Centered, because only the SHAPE of a partial dependence is interpretable:
    # its level absorbs the model intercept and every other predictor's mean.
    out['partial_dependence_centered'] = (
        out.groupby(['model', 'characteristic']).partial_dependence
        .transform(lambda s: s - s.mean()))
    return out


def marginal_versus_partial(curves, pd_table):
    """How much of a characteristic's marginal slope survives holding the rest.

    BOTH RANGES ARE IN LOG UNITS OF THE TARGET, and that is not a detail. The
    partial dependence is fitted on `log(target)`, while `curves_for` draws the
    marginal curve on the target's own scale, so ranging the two as they come
    compares a distance in W1 with a distance in log W1 and produces a ratio
    that means nothing. An earlier version did exactly that and returned
    "fractions surviving" above 7. The marginal curve is therefore logged here
    before it is ranged.

    One row per (characteristic, model): the log range the marginal curve
    covers, the log range the partial dependence covers, and the ratio. A ratio
    near zero says the marginal picture was borrowed from the other
    characteristics; near one says the effect is the characteristic's own.

    A ratio ABOVE one is possible and is not an error: holding the other
    characteristics fixed can expose an effect that the marginal view masks,
    when two correlated characteristics push the target in opposite directions.
    """
    rows = []
    marg = curves[curves.kind == 'lowess']
    for char, g in marg.groupby('characteristic'):
        # Over METHODS as well, since every method is drawn on the panel and
        # the widest one is what a reader sees.
        spans = []
        for _, gg in g.groupby('method'):
            v = pd.to_numeric(gg['mean'], errors='coerce').to_numpy(float)
            v = v[np.isfinite(v) & (v > 0)]
            if len(v) > 1:
                spans.append(float(np.log(v.max()) - np.log(v.min())))
        m_range = float(np.nanmax(spans)) if spans else np.nan
        h = pd_table[pd_table.characteristic == char]
        for model, hh in h.groupby('model'):
            p_range = float(hh.partial_dependence.max()
                            - hh.partial_dependence.min())
            rows.append(dict(characteristic=char, model=model,
                             marginal_range_log=m_range,
                             partial_range_log=p_range,
                             partial_over_marginal=(
                                 p_range / m_range
                                 if m_range and np.isfinite(m_range) else np.nan)))
    return pd.DataFrame(rows).sort_values('partial_range_log',
                                          ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# the target the paper actually asks about: WHICH METHOD, not how big the score
# --------------------------------------------------------------------------

#: The pair whose comparison the paper makes. The normal is excluded because
#: every arm and every criterion already agrees it loses; the live question is
#: kernel estimate against three-parameter lognormal.
CHOICE_PAIR = ('KDE', 'Lognormal')


def choice_frame(frame, value, pair=CHOICE_PAIR):
    """One row per (arm, dataset, weighting): log(W1_first / W1_second).

    **THIS IS A DIFFERENT QUESTION FROM THE ONE `importance` ASKS, AND THE
    ANSWERS DIFFER.** `importance` predicts the LEVEL of a single method's
    score, and the level is dominated by dispersion and dataset size because
    every method gets worse on spread data and on small samples. What the paper
    asks is which method to USE, which is the DIFFERENCE between two of them,
    and a difference is about whose shape assumption fits -- so it is where
    skewness, kurtosis, lognormality and the weight of outliers live. Ranking
    characteristics on the level and then reporting the ranking as an answer to
    the choice question is a mistake this module made until decision 135.

    TWO PROPERTIES OF THE RATIO WORTH KNOWING. It is taken WITHIN a weighting
    scheme, so the definitional part of the score -- the distance a
    uniform-weighted model cannot remove, which `definitional_check` shows is
    exactly `w_v_uw_wasserstein` -- is common to both terms and CANCELS. That
    makes `w_v_uw_wasserstein` a legitimate predictor here, where on the level
    target it was an identity. And a ratio is scale free, so it does not
    inherit the level's dependence on how spread the data happen to be.
    """
    g = frame.dropna(subset=[value]).copy()
    g['weighting'] = np.where(g.method.str.endswith('Variable'),
                              'Variable', 'Uniform')
    g['family'] = g.method.str.split(',').str[0]
    keys = ['arm', 'dataset', 'weighting']
    wide = (g[g.family.isin(pair)]
            .pivot_table(index=keys, columns='family', values=value))
    wide = wide.dropna(subset=list(pair))
    with np.errstate(divide='ignore', invalid='ignore'):
        wide['log_ratio'] = np.log(wide[pair[0]] / wide[pair[1]])
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_ratio'])
    out = wide.reset_index()[keys + ['log_ratio']]
    chars = frame.drop_duplicates(['arm', 'dataset'])
    carry = [c for c in chars.columns
             if c in CANDIDATE_METRICS + MODE_METRICS + EXTRA_METRICS
             or c == 'size_band']
    return out.merge(chars[['arm', 'dataset'] + carry], on=['arm', 'dataset'],
                     how='left')


def choice_importance(choice, metrics, arm=None, weighting=None, rng=None,
                      n_repeats=10, n_splits=5, models=('additive', 'boosted'),
                      pair=CHOICE_PAIR):
    """Permutation importance for predicting WHICH of two families fits better.

    The target is already a log ratio, so it is modeled on its own scale.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    g = choice
    if arm is not None:
        g = g[g.arm == arm]
    if weighting is not None:
        g = g[g.weighting == weighting]
    if len(g) < 40:
        return pd.DataFrame()
    X = transform_frame(g, metrics).to_numpy(float)
    y = g.log_ratio.to_numpy(float)
    keep = np.isfinite(y)
    X, y = X[keep], y[keep]
    builders = {
        'additive': lambda: additive_model(random_state=int(rng.integers(1 << 30))),
        'boosted': lambda: boosted_model(random_state=int(rng.integers(1 << 30))),
    }
    rows = []
    for name in models:
        r2m, r2s, imp, spread = _fit_score_permute(
            builders[name], X, y, rng, n_repeats=n_repeats, n_splits=n_splits)
        for m, v, s in zip(metrics, imp, spread):
            rows.append(dict(arm=arm, weighting=weighting, target='log_ratio',
                             target_family='choice', model=name, metric=m,
                             # A `method` value so these rows SURVIVE being
                             # concatenated with the per-method importances and
                             # grouped: pandas' groupby drops a NaN key by
                             # default, so without this the whole choice family
                             # vanished from `rank_survivors` without an error.
                             method=f'{pair[0]} vs {pair[1]}',
                             importance=float(v),
                             importance_sd_across_folds=float(s),
                             model_r2=float(r2m), model_r2_sd=float(r2s),
                             n_rows=int(len(y))))
    return pd.DataFrame(rows)


def choice_increments(choice, metrics, arm, weighting, base=('n',),
                      bonferroni=True):
    """What each characteristic adds to the CHOICE once `base` is in the model.

    Reported over two bases, because the two answer different questions: over
    dataset size alone, which is the mechanism every earlier stage found, and
    over size AND dispersion together, which is what the level-target reduction
    said was all that mattered. A characteristic that still adds something over
    the second base is one the level target was hiding.

    **A BONFERRONI COLUMN, BECAUSE FIFTEEN CANDIDATES ARE TESTED ON 127
    DATASETS.** Fifteen tests at the five percent level expect about one false
    positive, so the raw p-value is not the right threshold and the corrected
    one is carried beside it rather than left to the reader.
    """
    from scipy import stats as sps
    from sklearn.preprocessing import SplineTransformer

    g = choice[(choice.arm == arm) & (choice.weighting == weighting)]
    cand = [m for m in metrics if m not in base]
    M = transform_frame(g, list(base) + cand)
    y = g.log_ratio.to_numpy(float)

    def basis(v):
        return SplineTransformer(n_knots=5, degree=3).fit_transform(
            np.asarray(v, float).reshape(-1, 1))

    rows = []
    for m in cand:
        ok = np.isfinite(y)
        for c in list(base) + [m]:
            ok &= np.isfinite(M[c].to_numpy(float))
        n = int(ok.sum())
        if n < 40:
            continue
        yy = y[ok]
        B0 = np.column_stack([np.ones(n)]
                             + [basis(M[c].to_numpy(float)[ok]) for c in base])
        B1 = np.column_stack([B0, basis(M[m].to_numpy(float)[ok])])

        def rss(B):
            b, *_ = np.linalg.lstsq(B, yy, rcond=None)
            r = yy - B @ b
            return float((r ** 2).sum()), np.linalg.matrix_rank(B)

        r0, k0 = rss(B0)
        r1, k1 = rss(B1)
        sst = float(((yy - yy.mean()) ** 2).sum())
        df1, df2 = max(k1 - k0, 1), max(n - k1, 1)
        F = ((r0 - r1) / df1) / (r1 / df2) if r1 > 0 else np.nan
        p = float(sps.f.sf(F, df1, df2)) if np.isfinite(F) and F > 0 else np.nan
        rows.append(dict(arm=arm, weighting=weighting, base='+'.join(base),
                         metric=m, n_rows=n,
                         r2_base=1 - r0 / sst if sst else np.nan,
                         incremental_r2=(r0 - r1) / sst if sst else np.nan,
                         f_stat=F, p_value=p))
    out = pd.DataFrame(rows)
    if len(out) and bonferroni:
        out['n_tests'] = len(out)
        out['bonferroni_threshold'] = 0.05 / len(out)
        out['survives_bonferroni'] = out.p_value < out.bonferroni_threshold
    return out.sort_values('incremental_r2', ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# what the independent directions ARE
# --------------------------------------------------------------------------

def principal_components(frame, metrics, n_report=5):
    """Name the independent directions in the characteristic set.

    The effective dimension says HOW MANY independent quantities there are; it
    does not say what they are, and "about four or five" is not something a
    reader can act on. This gives each component its variance share and the
    characteristics that load on it, so the axes can be named.

    Correlations, not covariances, because the characteristics are on
    incommensurate scales, and on the MODELING scale of each so that a
    quantity spanning four orders of magnitude does not dominate by virtue of
    its units.
    """
    rows = []
    per_dataset = frame.drop_duplicates(['arm', 'dataset'])
    for arm, g in per_dataset.groupby('arm'):
        M = transform_frame(g, metrics).replace([np.inf, -np.inf], np.nan).dropna()
        if len(M) < 30:
            continue
        names = list(M.columns)
        sd = M.std()
        names = [n for n in names if sd[n] > 0]
        M = M[names]
        Z = ((M - M.mean()) / M.std()).to_numpy(float)
        lam, vec = np.linalg.eigh(np.corrcoef(Z, rowvar=False))
        order = np.argsort(lam)[::-1]
        lam, vec = lam[order], vec[:, order]
        share = lam / lam.sum()
        for i in range(min(n_report, len(lam))):
            load = pd.Series(vec[:, i], index=names)
            top = load.reindex(load.abs().sort_values(ascending=False).index)
            rows.append(dict(
                arm=arm, component=f'PC{i+1}', n_datasets=len(M),
                n_characteristics=len(names),
                eigenvalue=float(lam[i]), variance_share=float(share[i]),
                cumulative_share=float(share[:i + 1].sum()),
                participation_ratio=float(lam.sum() ** 2 / (lam ** 2).sum()),
                eigenvalues_above_one=int((lam > 1).sum()),
                loadings=', '.join(f'{k} {v:+.2f}' for k, v in top.head(5).items()),
            ))
    return pd.DataFrame(rows)

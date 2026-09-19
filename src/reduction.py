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
        logn = M['n'].to_numpy(float)
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
            random_state=seed, n_jobs=1)
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
    logn = M['n'].to_numpy(float)

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
             if c in CANDIDATE_METRICS + MODE_METRICS + ('size_band',)]
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
                                        random_state=seed, n_jobs=1)
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


def lowess_curve(x, y, frac=0.3, n_out=100, n_boot=400, rng=None):
    """LOWESS with a bootstrap band, as the smooth companion to the bins.

    The band resamples the DATASETS and refits, so it carries the uncertainty
    of the smoother itself and not only the scatter at a point. Where the
    datasets thin out the band opens, which is the artifact the rolling average
    hid.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20:
        return pd.DataFrame()
    grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_out)
    fit = _lowess(y, x, frac=frac, xvals=grid, return_sorted=False)

    boots = np.empty((n_boot, n_out))
    for b in range(n_boot):
        s = rng.integers(0, len(x), len(x))
        try:
            boots[b] = _lowess(y[s], x[s], frac=frac, xvals=grid,
                               return_sorted=False)
        except Exception:
            boots[b] = np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    # Local density, for the rug: how many datasets sit within one smoothing
    # window of each grid point. It is what makes a sparse region visible.
    half = frac * (x.max() - x.min()) / 2
    dens = np.array([int(np.sum(np.abs(x - g) <= half)) for g in grid])
    return pd.DataFrame(dict(x=grid, fit=fit, lo=lo, hi=hi, local_count=dens))


def curves_for(frame, metrics, methods, value='w1', arm=None, rng=None,
               n_bins=12, frac=0.35):
    """Binned and smoothed curves for every (metric, method) pair on one arm."""
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
            if len(b):
                b.insert(0, 'method', method)
                b.insert(0, 'characteristic', metric)
                b.insert(0, 'arm', arm)
                b['kind'] = 'binned'
                b['scale'] = scale
                out.append(b)
            l = lowess_curve(xt[sel], y[sel], frac=frac, rng=rng)
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

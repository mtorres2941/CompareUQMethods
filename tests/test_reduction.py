"""Tests for src/reduction.py, the Stage 2f metric reduction.

The properties worth pinning here are the ones that would let a reduction be
WRONG QUIETLY rather than fail: a model that silently drops the smallest
stratum, an importance ranking taken from a model that predicts nothing, a
transform that invents a value where a metric is undefined, a bootstrap band
that does not widen where the data thin out, and a "which method wins" model
whose accuracy is only the majority class.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import reduction as R  # noqa: E402


# ---------------------------------------------------------------- fixtures

def synthetic_frame(n_datasets=500, seed=0, planted=('n', 'coeffvar')):
    """A corpus-shaped frame with a KNOWN signal planted in two metrics.

    Everything else is noise, so a reduction that works must recover exactly
    these two and rank the rest below them. Kurtosis is undefined below n = 4,
    as it is in the real corpus, so the missingness path is exercised.
    """
    rng = np.random.default_rng(seed)
    n = np.concatenate([
        rng.integers(3, 10, n_datasets // 4),
        rng.integers(10, 100, n_datasets // 4),
        rng.integers(100, 1000, n_datasets // 4),
        rng.integers(1000, 9999, n_datasets - 3 * (n_datasets // 4)),
    ])
    rng.shuffle(n)
    N = len(n)
    cv = rng.lognormal(-0.7, 0.6, N)
    chars = pd.DataFrame(dict(
        arm='synthetic', dataset=[f'd{i}' for i in range(N)], n=n,
        coeffvar=cv, coeffvar_uw=cv * rng.lognormal(0, 0.05, N),
        skewness=rng.normal(1, 1, N), skewness_uw=rng.normal(1, 1, N),
        kurtosis=np.where(n < 4, np.nan, rng.normal(2, 2, N)),
        kurtosis_uw=np.where(n < 4, np.nan, rng.normal(2, 2, N)),
        entropy=rng.normal(3, 0.5, N), entropy_uw=rng.normal(3, 0.5, N),
        crit_bw_1=rng.lognormal(-2, 0.5, N), crit_bw_1_uw=rng.lognormal(-2, 0.5, N),
        modality_index=rng.uniform(1, 1.2, N),
        modality_index_uw=rng.uniform(1, 1.2, N),
        fit_norm_SF=rng.uniform(0.5, 0.999, N),
        fit_norm_SF_uw=rng.uniform(0.5, 0.999, N),
        fit_lognorm_SF=rng.uniform(0.6, 0.999, N),
        fit_lognorm_SF_uw=rng.uniform(0.6, 0.999, N),
        weight_outliers=rng.uniform(0, 0.2, N),
        weight_outliers_uw=rng.uniform(0, 0.2, N),
        mean=rng.lognormal(0, 0.1, N),
        w_v_uw_wasserstein=rng.lognormal(-2, 0.5, N),
        modes_fitted=rng.integers(1, 4, N),
        modes_scipy_default=rng.integers(1, 3, N)))

    signal = np.zeros(N)
    if 'n' in planted:
        signal = signal - 0.5 * np.log(n)
    if 'coeffvar' in planted:
        signal = signal + 0.8 * np.log(cv)
    scores = []
    for meth in ('KDE, Uniform', 'Lognormal, Uniform'):
        s = chars[['arm', 'dataset', 'n']].copy()
        s['method'] = meth
        s['w1'] = np.exp(signal + rng.normal(0, 0.3, N))
        s['w1_cv'] = s.w1 * 1.1
        s['w1_market'] = s.w1 * 0.9
        s['w1_parent'] = s.w1
        scores.append(s)
    scores = pd.concat(scores, ignore_index=True)
    modes = chars[['arm', 'dataset'] + list(R.MODE_METRICS)]
    return R.assemble(chars, scores, modes=modes)


# ---------------------------------------------------------------- transforms

def test_transform_propagates_undefined_rather_than_inventing_a_value():
    # An undefined kurtosis must arrive at the model as missing. A transform
    # that mapped it to a number would put a fabricated value in the corpus's
    # smallest stratum, which is where the answer is known to change.
    x = np.array([1.0, np.nan, np.inf, -np.inf, 4.0])
    for how in ('identity', 'log', 'log1p', 'signed_log', 'cloglog'):
        out = R.transform(x, how)
        assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])
    assert R.transform(x, 'identity')[0] == 1.0


def test_transform_log_refuses_non_positive_input():
    out = R.transform(np.array([-1.0, 0.0, 1.0]), 'log')
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(0.0)


def test_cloglog_is_monotone_and_finite_at_a_perfect_fit():
    # A Shapiro statistic of exactly 1.0 is a perfect fit. It must not remove
    # the dataset from the model.
    v = R.transform(np.array([0.5, 0.9, 0.99, 1.0]), 'cloglog')
    assert np.all(np.isfinite(v))
    assert np.all(np.diff(v) < 0)


def test_signed_log_keeps_the_sign():
    v = R.transform(np.array([-100.0, 0.0, 100.0]), 'signed_log')
    assert v[0] < 0 < v[2]
    assert v[1] == pytest.approx(0.0)
    assert v[0] == pytest.approx(-v[2])


def test_every_candidate_has_a_declared_transform():
    # A metric added to the candidate list without a transform would silently
    # fall through to identity, and a raw n spanning four orders of magnitude
    # on a spline basis is not a fair hearing for it.
    for m in R.CANDIDATE_METRICS + R.MODE_METRICS:
        assert m in R.METRIC_TRANSFORM, m


# ------------------------------------------------------------- missingness

def test_missingness_is_reported_and_kurtosis_is_the_offender():
    frame, metrics = synthetic_frame(400, seed=1)
    miss = R.missingness_by_band(frame, metrics)
    small = miss[miss.size_band == 'n 3-9'].iloc[0]
    assert small['defined__kurtosis'] < small['n_datasets']
    # and nothing else in that band is missing
    assert small['defined__coeffvar'] == small['n_datasets']


def test_complete_case_would_drop_the_smallest_band_and_says_so():
    """The stage's explicit instruction: handle the missingness rather than
    let a default drop the stratum silently."""
    frame, metrics = synthetic_frame(400, seed=2)
    cost = R.complete_case_cost(frame, metrics)
    small = cost[(cost.size_band == 'n 3-9')].iloc[0]
    assert small.share_dropped > 0.0
    overall = cost[cost.size_band == 'all'].iloc[0]
    assert overall.n_complete < overall.n_datasets


def test_the_flexible_model_keeps_the_rows_the_complete_case_model_drops():
    """Gradient boosting takes NaN natively, which is why it is here.

    The additive model imputes with an indicator and the boosted model splits
    on missingness, so BOTH use every row. This asserts the row count the
    models report is the full one and not the complete-case one.
    """
    frame, metrics = synthetic_frame(400, seed=3)
    cost = R.complete_case_cost(frame, metrics)
    n_complete = int(cost[cost.size_band == 'all'].iloc[0].n_complete)
    n_total = int(cost[cost.size_band == 'all'].iloc[0].n_datasets)
    assert n_complete < n_total
    imp = R.importance(frame, metrics, 'w1', arm='synthetic',
                       method='KDE, Uniform', rng=np.random.default_rng(4),
                       n_repeats=2, n_splits=3)
    assert set(imp.n_rows) == {n_total}


# ---------------------------------------------------------------- recovery

def test_importance_recovers_a_planted_signal_and_ranks_noise_below_it():
    frame, metrics = synthetic_frame(500, seed=5)
    imp = R.importance(frame, metrics, 'w1', arm='synthetic',
                       method='KDE, Uniform', rng=np.random.default_rng(6),
                       n_repeats=3, n_splits=3)
    assert len(imp)
    for model, g in imp.groupby('model'):
        top2 = set(g.nlargest(2, 'importance').metric)
        assert top2 == {'n', 'coeffvar'}, (model, top2)
        assert g.model_r2.iloc[0] > 0.5


def test_importance_finds_nothing_when_there_is_nothing():
    """The control. With no planted signal every importance must be near zero
    and the model's own out-of-sample R2 must be too, which is why
    `rank_survivors` refuses to rank a model below `min_r2`."""
    frame, metrics = synthetic_frame(400, seed=7, planted=())
    imp = R.importance(frame, metrics, 'w1', arm='synthetic',
                       method='KDE, Uniform', rng=np.random.default_rng(8),
                       n_repeats=3, n_splits=3)
    assert imp.model_r2.max() < 0.15
    assert R.rank_survivors(imp, min_r2=0.15).empty


def test_incremental_over_size_credits_the_second_predictor_not_the_first():
    frame, metrics = synthetic_frame(500, seed=9)
    inc = R.incremental_over_size(frame, metrics, 'w1', arm='synthetic',
                                  method='KDE, Uniform')
    assert inc.iloc[0].metric in ('coeffvar', 'coeffvar_uw')
    assert inc.iloc[0].incremental_r2 > 0.1
    assert inc.iloc[0].p_value < 1e-6
    # size alone already explains a lot, and that is reported beside it
    assert 0.1 < inc.iloc[0].r2_size_only < 0.95
    # a pure-noise metric adds nothing detectable
    noise = inc[inc.metric == 'entropy'].iloc[0]
    assert noise.incremental_r2 < 0.02


def test_size_confounding_finds_a_metric_that_is_size_in_disguise():
    frame, metrics = synthetic_frame(400, seed=10)
    # plant a metric that IS log(n) plus a little noise
    rng = np.random.default_rng(11)
    frame = frame.copy()
    frame['entropy'] = np.log(frame.n) + rng.normal(0, 0.05, len(frame))
    sc = R.size_confounding(frame, metrics)
    row = sc[(sc.arm == 'synthetic') & (sc.metric == 'entropy')].iloc[0]
    assert row.r2_on_log_n > 0.95
    other = sc[(sc.arm == 'synthetic') & (sc.metric == 'skewness')].iloc[0]
    assert other.r2_on_log_n < 0.1


# ------------------------------------------------------------- redundancy

def test_redundancy_sees_a_duplicated_metric_and_the_effective_dimension_falls():
    frame, metrics = synthetic_frame(400, seed=12)
    tbl = R.redundancy_table(frame, list(metrics))
    pair = tbl[((tbl.metric_a == 'coeffvar') & (tbl.metric_b == 'coeffvar_uw'))]
    assert len(pair) == 1 and pair.iloc[0].correlation > 0.9
    assert pair.iloc[0].redundant
    # the effective dimension is below the metric count whenever anything is
    # correlated with anything
    assert pair.iloc[0].effective_dimension < pair.iloc[0].n_metrics


# ---------------------------------------------------------------- winners

def test_winner_frame_picks_the_smallest_score_within_each_weighting():
    frame, metrics = synthetic_frame(200, seed=13)
    w = R.winner_frame(frame, 'w1', within_weighting=True)
    assert set(w.weighting) == {'Uniform'}
    one = frame[frame.dataset == w.iloc[0].dataset]
    assert w.iloc[0].winning_score == pytest.approx(one.w1.min())


def test_winner_importance_reports_the_majority_baseline_beside_the_accuracy():
    """An accuracy of 0.70 on a question one method wins 70 percent of the
    time is a model that has learned nothing, and the importances under it are
    noise. The baseline must travel with the number."""
    frame, metrics = synthetic_frame(400, seed=14)
    w = R.winner_frame(frame, 'w1', within_weighting=True)
    wi = R.winner_importance(w, list(metrics), arm='synthetic',
                             weighting='Uniform', rng=np.random.default_rng(15),
                             n_repeats=2, n_splits=3)
    if len(wi):
        assert {'accuracy', 'majority_baseline', 'lift_over_baseline'} <= set(wi.columns)
        assert np.allclose(wi.lift_over_baseline,
                           wi.accuracy - wi.majority_baseline)
        # the two methods here differ only by noise, so nothing should be
        # learnable and the lift must be near zero or negative
        assert wi.lift_over_baseline.max() < 0.15


# ------------------------------------------------------------------ curves

def test_binned_curve_bands_widen_where_the_data_thin_out():
    """The defect in the rolling average, asserted directly.

    A region with four datasets and a region with four hundred must not look
    the same. Equal-count bins plus a within-bin bootstrap make the sparse
    region's band wide and put the count on the row.
    """
    rng = np.random.default_rng(16)
    # dense on the left, sparse on the right, same noise everywhere
    x = np.concatenate([rng.uniform(0, 1, 800), rng.uniform(1, 2, 20)])
    y = rng.normal(0, 1, len(x))
    c = R.binned_curve(x, y, n_bins=8, n_boot=400,
                       rng=np.random.default_rng(17), equal_count=False)
    widths = c.hi - c.lo
    assert c['count'].iloc[0] > c['count'].iloc[-1]
    assert widths.iloc[-1] > widths.iloc[0]


def test_equal_count_bins_hold_about_the_same_number_of_datasets():
    rng = np.random.default_rng(18)
    x = rng.lognormal(size=600)
    y = rng.normal(size=600)
    c = R.binned_curve(x, y, n_bins=10, n_boot=200, rng=np.random.default_rng(19))
    assert c['count'].max() <= c['count'].min() + 2


def test_binned_band_covers_the_truth_for_a_flat_relationship():
    rng = np.random.default_rng(20)
    x = rng.uniform(0, 1, 1200)
    y = rng.normal(0.0, 1.0, 1200)
    c = R.binned_curve(x, y, n_bins=10, n_boot=600, rng=np.random.default_rng(21))
    covered = ((c.lo <= 0.0) & (c.hi >= 0.0)).mean()
    assert covered >= 0.7


def test_lowess_carries_a_band_and_a_local_count():
    rng = np.random.default_rng(22)
    x = np.concatenate([rng.uniform(0, 1, 500), rng.uniform(2, 3, 30)])
    y = 2 * x + rng.normal(0, 0.3, len(x))
    c = R.lowess_curve(x, y, frac=0.4, n_out=40, n_boot=80,
                       rng=np.random.default_rng(23))
    assert len(c) == 40
    assert (c.hi >= c.lo).all()
    assert c.local_count.min() < c.local_count.max()
    # it recovers a straight line it was given
    assert np.corrcoef(c.x, c.fit)[0, 1] > 0.98


def test_curves_for_produces_both_kinds_per_metric_and_method():
    frame, metrics = synthetic_frame(400, seed=24)
    cur = R.curves_for(frame, ['n', 'coeffvar'], ['KDE, Uniform'], 'w1',
                       arm='synthetic', rng=np.random.default_rng(25))
    got = set(map(tuple, cur[['characteristic', 'kind']].drop_duplicates().values))
    assert got == {('n', 'binned'), ('n', 'lowess'),
                   ('coeffvar', 'binned'), ('coeffvar', 'lowess')}


# ------------------------------------------------------------- aggregation

def test_rank_survivors_ignores_models_that_predict_nothing():
    """A metric must not be promoted by a model whose out-of-sample R2 is
    noise, because inside such a model the importance ordering is arbitrary."""
    good, metrics = synthetic_frame(400, seed=26)
    imp_good = R.importance(good, metrics, 'w1', arm='synthetic',
                            method='KDE, Uniform', rng=np.random.default_rng(27),
                            n_repeats=2, n_splits=3)
    junk, _ = synthetic_frame(400, seed=28, planted=())
    imp_junk = R.importance(junk, metrics, 'w1', arm='synthetic',
                            method='Lognormal, Uniform',
                            rng=np.random.default_rng(29), n_repeats=2, n_splits=3)
    both = pd.concat([imp_good, imp_junk], ignore_index=True)
    surv = R.rank_survivors(both, top=3, min_r2=0.3)
    assert set(surv[surv.survivor].metric) >= {'n', 'coeffvar'}
    assert surv.n_models.max() <= 2


def test_permutation_importance_is_identical_across_cores():
    """Parallelising must change the wall clock and nothing else.

    `reduction.PERMUTATION_JOBS` is -1 so a notebook run finishes in an hour
    instead of three. sklearn derives each feature's permutation seed
    deterministically from `random_state`, so the result does not depend on
    how the work is split, and this pins that rather than assuming it.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = 2 * X[:, 0] + X[:, 1] + rng.normal(0, 0.3, 600)
    m = HistGradientBoostingRegressor(random_state=0, max_iter=50).fit(X, y)
    one = permutation_importance(m, X, y, scoring='r2', n_repeats=6,
                                 random_state=7, n_jobs=1).importances_mean
    many = permutation_importance(m, X, y, scoring='r2', n_repeats=6,
                                  random_state=7,
                                  n_jobs=R.PERMUTATION_JOBS).importances_mean
    np.testing.assert_array_equal(one, many)

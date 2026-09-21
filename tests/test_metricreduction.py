"""Tests for src/metricreduction.py, the Stage 2f metric reduction.

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

import metricreduction as R  # noqa: E402


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


def test_rows_used_by_band_sees_a_target_that_is_undefined_at_small_n():
    """The second exclusion, which the predictor-side report cannot see.

    The empirical cross-validated target is undefined below n = 10, so the
    out-of-sample reduction on the real arm silently has nothing to say about
    the smallest size band. A model that used 127 of 147 datasets must report
    WHICH 20 it did not use, and per band, or the reduction reads as though it
    covered the arm.
    """
    frame, metrics = synthetic_frame(400, seed=30)
    frame = frame.copy()
    frame.loc[frame.n < 10, 'w1_cv'] = np.nan
    used = R.rows_used_by_band(frame, ['w1', 'w1_cv'])
    cv = used[used.target == 'w1_cv'].iloc[0]
    insample = used[used.target == 'w1'].iloc[0]
    assert cv['used__n 3-9'] == 0
    assert cv['available__n 3-9'] > 0
    assert insample['used__n 3-9'] == insample['available__n 3-9']
    assert cv.rows_used < cv.rows_total


def test_partial_dependence_separates_a_real_effect_from_a_borrowed_one():
    """The measurement the whole stage turns on.

    A characteristic that is a copy of a real predictor has a steep MARGINAL
    curve and a flat PARTIAL DEPENDENCE, because holding the real predictor
    fixed leaves it nothing to explain. This plants exactly that: `entropy` is
    made a noisy copy of log(n), and `n` is one of the two predictors the
    target actually depends on.
    """
    frame, metrics = synthetic_frame(500, seed=31)
    frame = frame.copy()
    rng = np.random.default_rng(32)
    frame['entropy'] = np.log(frame.n) + rng.normal(0, 0.05, len(frame))
    pdt = R.partial_dependence_table(
        frame, metrics, 'w1', which=['n', 'entropy'], arm='synthetic',
        method='KDE, Uniform', rng=np.random.default_rng(33),
        models=('boosted',))
    assert len(pdt)
    span = pdt.groupby('characteristic').partial_dependence.apply(
        lambda s: s.max() - s.min())
    # The genuine predictor keeps the larger partial dependence -- but NOT by
    # a wide margin, and that is the caveat the docstring records rather than
    # a defect. Two nearly collinear predictors SPLIT the effect, because no
    # model can tell which of them the target depends on, so a copy retains a
    # substantial fraction instead of collapsing to zero.
    assert span['n'] > span['entropy']
    assert span['entropy'] > 0.3 * span['n'], (
        'a near-copy is expected to retain a large share; if it now collapses, '
        'the docstring caveat is wrong and should be rewritten')


def test_partial_dependence_is_centered_and_stays_inside_the_data():
    frame, metrics = synthetic_frame(400, seed=34)
    pdt = R.partial_dependence_table(
        frame, metrics, 'w1', which=['coeffvar'], arm='synthetic',
        method='KDE, Uniform', rng=np.random.default_rng(35),
        models=('boosted',))
    assert len(pdt)
    assert abs(pdt.partial_dependence_centered.mean()) < 1e-9
    # The grid must stay INSIDE the observed data, which is the property that
    # matters: a spline extrapolated past its last knot is not a measurement.
    # It is not asserted against numpy's percentiles, because sklearn places
    # the grid with a different quantile convention and lands a hair outside
    # them; the two disagree by about 0.006 on this fixture and agreeing with
    # sklearn's choice is not the point.
    sub = frame[(frame.arm == 'synthetic') & (frame.method == 'KDE, Uniform')
                & frame.w1.notna()]
    x = R.transform_frame(sub, ['coeffvar']).coeffvar
    assert pdt.x.min() > np.nanmin(x)
    assert pdt.x.max() < np.nanmax(x)
    # and it covers the body rather than a sliver of it
    assert (pdt.x.max() - pdt.x.min()) > 0.5 * (np.nanmax(x) - np.nanmin(x))


def test_incremental_over_size_does_not_need_n_in_the_candidate_list():
    """log(n) is the BASE model, so it comes from the frame, not the list.

    `modality_head_to_head` offers only the modality measures, so asking the
    transformed candidate frame for a size column the caller never named
    raises a KeyError. It cost an audit run to find.
    """
    frame, metrics = synthetic_frame(300, seed=36)
    out = R.incremental_over_size(frame, ['crit_bw_1', 'modality_index'], 'w1',
                                  arm='synthetic', method='KDE, Uniform')
    assert len(out) == 2
    assert out.r2_size_only.notna().all()


def test_modality_head_to_head_runs_on_the_modality_measures_alone():
    frame, metrics = synthetic_frame(300, seed=37)
    out = R.modality_head_to_head(frame, ('w1',), ['KDE, Uniform'],
                                  'synthetic', rng=np.random.default_rng(38))
    assert len(out)
    assert set(out.metric) <= {'modality_index', 'modality_index_uw',
                               'crit_bw_1', 'crit_bw_1_uw',
                               'modes_fitted', 'modes_scipy_default'}
    assert out.mean_incremental_r2.notna().all()


def test_modality_agreement_shares_use_the_defined_denominator():
    """A mode count needs at least 8 values, so it is undefined on the
    smallest datasets. Dividing the unimodal share by the whole arm counts an
    undefined dataset as multimodal, which understated the empirical share as
    0.605 against a true 0.685.
    """
    frame, metrics = synthetic_frame(400, seed=39)
    frame = frame.copy()
    # built as float from the start: assigning NaN into an integer column is
    # not a silent widening in this pandas, and the fixture must exercise the
    # denominator, not pandas' dtype rules
    frame['modes_fitted'] = np.where(frame.n.to_numpy() < 8, np.nan, 1.0)
    out = R.modality_agreement(frame)
    row = out[(out.measure_a == 'modes_fitted')
              & (out.measure_b == 'share_unimodal')].iloc[0]
    # every dataset it is DEFINED on is unimodal here, so the share is exactly 1
    assert row.share_unimodal == pytest.approx(1.0)
    assert row.n < row.n_arm


def test_marginal_versus_partial_compares_like_with_like():
    """Both ranges must be in LOG units of the target.

    The partial dependence is fitted on log(target) while the marginal curve
    is drawn on the target's own scale, so ranging them as they come compares
    a distance in W1 with a distance in log W1. An earlier version did that
    and returned "fractions surviving" above 7.
    """
    frame, metrics = synthetic_frame(400, seed=40)
    curves = R.curves_for(frame, ['n', 'coeffvar'], ['KDE, Uniform'], 'w1',
                          arm='synthetic', rng=np.random.default_rng(41))
    pdt = R.partial_dependence_table(
        frame, metrics, 'w1', which=['n', 'coeffvar'], arm='synthetic',
        method='KDE, Uniform', rng=np.random.default_rng(42),
        models=('boosted',))
    out = R.marginal_versus_partial(curves, pdt)
    assert 'marginal_range_log' in out.columns
    assert out.marginal_range_log.notna().all()
    # both planted effects are real, so the partial range is the same order of
    # magnitude as the marginal one rather than several times it
    assert out.partial_over_marginal.between(0.2, 3.0).all(), \
        out[['characteristic', 'partial_over_marginal']].to_string()


def test_lowess_does_not_extrapolate_past_the_bins():
    """A local linear fit projects its edge slope outward.

    On the empirical arm that carried the dataset-size curve from 0.026 at the
    last populated bin down through zero to a NEGATIVE Wasserstein distance.
    `curves_for` bounds the smooth to the span the binned summary covers.
    """
    frame, metrics = synthetic_frame(400, seed=43)
    cur = R.curves_for(frame, ['n'], ['KDE, Uniform'], 'w1', arm='synthetic',
                       rng=np.random.default_rng(44))
    b = cur[cur.kind == 'binned']
    l = cur[cur.kind == 'lowess']
    assert l.x_center.min() >= b.x_center.min() - 1e-9
    assert l.x_center.max() <= b.x_center.max() + 1e-9
    # and the target is a distance, so a fitted value below zero is a failure
    assert (l['mean'] > 0).all()


# -------------------------------------------- the CHOICE target, decision 135

def choice_fixture(n_datasets=500, seed=50):
    """A frame where the LEVEL of both methods is driven by dispersion and size,
    and which one WINS is driven by skewness alone.

    That is the situation the level-target reduction cannot see, and it is why
    the choice target exists: ranking characteristics on the level would put
    skewness last, while the question the paper asks is exactly the one
    skewness answers.
    """
    rng = np.random.default_rng(seed)
    frame, metrics = synthetic_frame(n_datasets, seed=seed)
    per = frame.drop_duplicates('dataset')
    level = (-0.5 * np.log(per.n.to_numpy(float))
             + 0.8 * np.log(per.coeffvar.to_numpy(float)))
    # The tilt is SMALL against the level, which is the real situation: every
    # method gets worse on spread, small data by far more than any one family
    # is favoured by shape. So skewness explains about 2 percent of the level
    # and essentially all of the ratio, because the shared level cancels.
    tilt = 0.15 * per.skewness.to_numpy(float)
    rows = []
    for meth, sign in (('KDE, Uniform', +0.5), ('Lognormal, Uniform', -0.5)):
        s = per[['arm', 'dataset', 'n']].copy()
        s['method'] = meth
        s['w1'] = np.exp(level + sign * tilt
                         + rng.normal(0, 0.05, len(per)))
        rows.append(s)
    scores = pd.concat(rows, ignore_index=True)
    modes = per[['arm', 'dataset']].copy()
    for m in R.MODE_METRICS:
        modes[m] = per[f'{m}_x'] if f'{m}_x' in per else per[m]
    per = per.rename(columns={f'{m}_x': m for m in R.MODE_METRICS})
    chars = per.drop(columns=['method', 'w1', 'w1_cv', 'w1_market',
                              'w1_parent', 'size_band'], errors='ignore')
    return R.assemble(chars, scores, modes=modes)


def test_choice_frame_is_one_row_per_dataset_and_weighting():
    frame, metrics = choice_fixture(200, seed=51)
    ch = R.choice_frame(frame, 'w1')
    assert set(ch.weighting) == {'Uniform'}
    assert len(ch) == ch.dataset.nunique()
    assert np.isfinite(ch.log_ratio).all()


def test_the_choice_target_finds_what_the_level_target_hides():
    """The test that justifies decision 135.

    Both methods' LEVELS are built from dispersion and size; which one WINS is
    built from skewness. So the level reduction must rank skewness low and the
    choice reduction must rank it first. If these two ever agree on this
    fixture, the choice target has stopped being a separate question.
    """
    frame, metrics = choice_fixture(500, seed=52)
    level = R.importance(frame, metrics, 'w1', arm='synthetic',
                         method='KDE, Uniform', rng=np.random.default_rng(53),
                         n_repeats=3, n_splits=3, models=('boosted',))
    top_level = set(level.nlargest(2, 'importance').metric)
    assert 'skewness' not in top_level, top_level

    ch = R.choice_frame(frame, 'w1')
    choice = R.choice_importance(ch, metrics, arm='synthetic',
                                 weighting='Uniform',
                                 rng=np.random.default_rng(54),
                                 n_repeats=3, n_splits=3, models=('boosted',))
    assert choice.metric.iloc[choice.importance.argmax()] == 'skewness'
    assert choice.model_r2.iloc[0] > 0.5


def test_choice_increments_carry_a_bonferroni_threshold():
    """Fifteen candidates on 127 datasets is a multiple-comparison problem and
    the corrected threshold travels with the p-value rather than being left to
    the reader."""
    frame, metrics = choice_fixture(400, seed=55)
    ch = R.choice_frame(frame, 'w1')
    inc = R.choice_increments(ch, metrics, 'synthetic', 'Uniform')
    assert {'bonferroni_threshold', 'survives_bonferroni', 'n_tests'} <= set(inc.columns)
    assert inc.bonferroni_threshold.iloc[0] == pytest.approx(0.05 / inc.n_tests.iloc[0])
    assert inc.iloc[0].metric == 'skewness'
    assert bool(inc.iloc[0].survives_bonferroni)


def test_principal_components_name_the_independent_directions():
    """The effective dimension says how many; this says what they are."""
    frame, metrics = synthetic_frame(400, seed=56)
    pcs = R.principal_components(frame, metrics)
    assert len(pcs) == 5
    assert pcs.cumulative_share.is_monotonic_increasing
    assert 0 < pcs.variance_share.iloc[0] < 1
    # coeffvar and its uniform twin are near-copies in the fixture, so they
    # must load on the SAME component
    first = pcs.iloc[0].loadings
    assert 'coeffvar' in first


def test_rank_survivors_refuses_rows_with_a_missing_grouping_key():
    """A missing key would drop a whole target family without an error.

    pandas' groupby discards a NaN key by default, so concatenating the
    per-method importances with the choice importances -- which have no
    `method` of their own -- lost the entire choice family silently. The
    choice rows now carry a method label, and this refuses the shape that
    caused it in case a future caller reintroduces it.
    """
    frame, metrics = synthetic_frame(300, seed=57)
    imp = R.importance(frame, metrics, 'w1', arm='synthetic',
                       method='KDE, Uniform', rng=np.random.default_rng(58),
                       n_repeats=2, n_splits=3, models=('boosted',))
    broken = imp.copy()
    broken.loc[broken.index[:5], 'method'] = np.nan
    with pytest.raises(ValueError, match='missing grouping key'):
        R.rank_survivors(broken)


def test_choice_importance_rows_survive_a_concat_and_groupby():
    frame, metrics = choice_fixture(300, seed=59)
    ch = R.choice_frame(frame, 'w1')
    ci = R.choice_importance(ch, metrics, arm='synthetic', weighting='Uniform',
                             rng=np.random.default_rng(60), n_repeats=2,
                             n_splits=3, models=('boosted',))
    lvl = R.importance(frame, metrics, 'w1', arm='synthetic',
                       method='KDE, Uniform', rng=np.random.default_rng(61),
                       n_repeats=2, n_splits=3, models=('boosted',))
    both = pd.concat([lvl, ci], ignore_index=True)
    ranked = R.rank_survivors(both[both.target_family == 'choice'])
    assert len(ranked), 'the choice family was dropped by the groupby'
    assert set(ci.method) == {'KDE vs Lognormal'}


# ------------------------------------------------- out-of-sample selection
# Stage 2f review, decision 136. These pin the properties that make the
# cross-validated selection an improvement on the in-sample one rather than a
# rearrangement of it: a gain that cannot be bought by adding terms, a fold
# spread that actually widens when the data thin, selection that refuses a
# redundant copy, and a policy curve read as a region rather than an argmin.

def _choice_frame(n=1200, seed=3, noise=0.4):
    """A choice-shaped frame: log_ratio driven by n and one shape metric."""
    rng = np.random.default_rng(seed)
    nn = 10 ** rng.uniform(0.5, 4.0, n)
    shape = rng.normal(0, 1, n)
    junk = rng.normal(0, 1, n)
    log_ratio = (-0.5 * (np.log10(nn) - 2.0) + 0.6 * shape
                 + rng.normal(0, noise, n))
    return pd.DataFrame(dict(
        arm='synthetic', dataset=[f'd{i}' for i in range(n)],
        weighting='Uniform', log_ratio=log_ratio, n=nn,
        coeffvar=np.abs(rng.normal(0.6, 0.2, n)),
        skewness=shape, kurtosis=junk,
        skewness_copy=shape + rng.normal(0, 1e-6, n)))


def test_cv_gain_finds_the_planted_metric_and_not_the_junk():
    ch = _choice_frame()
    out = R.cv_gain(ch, ['skewness', 'kurtosis'], base=('n', 'coeffvar'))
    top = out.iloc[0]
    assert top.metric == 'skewness'
    assert top.cv_gain > 0.1
    junk = out[out.metric == 'kurtosis'].iloc[0]
    assert junk.cv_gain < 0.02


def test_cv_gain_cannot_be_bought_by_adding_a_useless_term():
    """The whole reason for cross-validating. In sample, adding a spline basis
    always raises R2; out of sample a useless term must not."""
    ch = _choice_frame()
    out = R.cv_gain(ch, ['kurtosis'], base=('n', 'coeffvar'))
    assert out.iloc[0].cv_gain < 0.02


def test_cv_gain_reports_a_fold_spread_that_grows_when_data_thin():
    big = R.cv_gain(_choice_frame(n=2000), ['skewness'], base=('n',))
    small = R.cv_gain(_choice_frame(n=120), ['skewness'], base=('n',))
    assert small.iloc[0].fold_sd > big.iloc[0].fold_sd


def test_cv_r2_refuses_to_report_below_the_row_floor():
    ch = _choice_frame(n=30)
    r2, sd, rows = R.cv_r2(R.transform_frame(ch, ['n']), ['n'],
                           ch.log_ratio.to_numpy(float))
    assert rows == 30 and np.isnan(r2)


def test_cv_r2_can_be_negative_when_the_model_predicts_worse_than_the_mean():
    """The empirical arm's actual behaviour: base R2 -0.724. A clipped-at-zero
    score would have hidden it."""
    rng = np.random.default_rng(0)
    n = 60
    ch = pd.DataFrame(dict(arm='synthetic', dataset=[f'd{i}' for i in range(n)],
                           weighting='Uniform',
                           log_ratio=rng.normal(0, 1, n),
                           n=10 ** rng.uniform(0.5, 4, n),
                           coeffvar=np.abs(rng.normal(0.6, 0.2, n))))
    r2, _sd, rows = R.cv_r2(R.transform_frame(ch, ['n', 'coeffvar']),
                            ['n', 'coeffvar'], ch.log_ratio.to_numpy(float))
    assert rows == n
    assert r2 < 0.0


def test_forward_select_refuses_a_redundant_copy():
    """`skewness_copy` is `skewness` to six decimals. A one-at-a-time table
    credits both; selection must take one and stop."""
    ch = _choice_frame()
    steps = R.forward_select(ch, ['skewness', 'skewness_copy', 'kurtosis'],
                             base=('n', 'coeffvar'), max_steps=4)
    added = list(steps.added)
    assert 'skewness' in added or 'skewness_copy' in added
    assert not ('skewness' in added and 'skewness_copy' in added)
    assert 'kurtosis' not in added


def test_forward_select_gains_are_positive_and_shrink():
    ch = _choice_frame()
    steps = R.forward_select(ch, ['skewness', 'kurtosis'],
                             base=('n',), min_gain=0.001, max_steps=3)
    assert (steps.gain > 0).all()
    assert steps.cv_r2.is_monotonic_increasing


# ------------------------------------------------------- policy thresholds

def _policy_scores(n=900, seed=5):
    """Two methods crossing over at n = 100, plus a bad third."""
    rng = np.random.default_rng(seed)
    nn = 10 ** rng.uniform(0.5, 4.0, n)
    ds = [f'd{i}' for i in range(n)]
    kde = 0.30 * (nn / 100.0) ** -0.45 * np.exp(rng.normal(0, .15, n))
    logn = 0.30 * (nn / 100.0) ** -0.12 * np.exp(rng.normal(0, .15, n))
    norm = logn * 1.8
    rows = []
    for name, v in (('KDE, Variable', kde), ('Lognormal, Uniform', logn),
                    ('Normal, Uniform', norm)):
        rows.append(pd.DataFrame(dict(arm='synthetic', dataset=ds,
                                      method=name, w1_market=v)))
    return pd.concat(rows, ignore_index=True), pd.Series(nn, index=ds)


def test_policy_curve_puts_its_flat_region_around_the_true_crossover():
    sc, sizes = _policy_scores()
    curve = R.policy_curve(sc, sizes, 'KDE, Variable', 'Lognormal, Uniform')
    reg = R.flat_region(curve)
    assert reg['lo'] <= 100 <= reg['hi']
    assert reg['best_cost'] >= 0.0


def test_policy_curve_beats_both_fixed_policies():
    """A threshold rule must cost less than always using either method, or it
    is not worth printing."""
    sc, sizes = _policy_scores()
    curve = R.policy_curve(sc, sizes, 'KDE, Variable', 'Lognormal, Uniform')
    best = curve.mean_cost_pct.min()
    always_high = curve.iloc[0].mean_cost_pct
    always_low = curve.iloc[-1].mean_cost_pct
    assert best < always_high and best < always_low


def test_best_method_share_rows_sum_to_one_hundred():
    sc, sizes = _policy_scores()
    tab = R.best_method_share(sc, sizes)
    cols = [c for c in tab.columns if ', ' in str(c)]
    assert np.allclose(tab[cols].sum(axis=1), 100.0)
    assert tab.n_datasets.sum() == len(sizes)


# -------------------------------------------------- effective sample size

def test_effective_sample_fraction_matches_the_closed_forms():
    """Uniform weights are worth n; a weight vector on one point is worth 1."""
    vals = pd.DataFrame(dict(
        dataset_id=['a'] * 4 + ['b'] * 4,
        weight=[0.25, 0.25, 0.25, 0.25, 1.0, 1e-12, 1e-12, 1e-12]))
    eff = R.effective_sample_fraction(vals).set_index('dataset')
    assert np.isclose(eff.loc['a', 'n_eff'], 4.0)
    assert np.isclose(eff.loc['a', 'eff_frac'], 1.0)
    assert eff.loc['b', 'n_eff'] < 1.01


def test_weighting_by_concentration_is_taken_inside_a_size_band():
    """If the split leaked dataset size it would report a gradient on data
    where concentration is assigned at random."""
    rng = np.random.default_rng(11)
    n = 800
    ds = [f'd{i}' for i in range(n)]
    nn = rng.integers(100, 999, n)
    eff = pd.DataFrame(dict(dataset=ds, n=nn,
                            n_eff=nn * rng.uniform(.2, .9, n)))
    eff['eff_frac'] = eff.n_eff / eff.n
    base = np.abs(rng.normal(0.1, 0.02, n))
    sc = pd.concat([
        pd.DataFrame(dict(arm='synthetic', dataset=ds, method='KDE, Uniform',
                          w1_market=base)),
        pd.DataFrame(dict(arm='synthetic', dataset=ds, method='KDE, Variable',
                          w1_market=base * np.exp(rng.normal(0, .05, n))))],
        ignore_index=True)
    out = R.weighting_by_concentration(sc, eff)
    assert len(out) == 4
    assert out.variable_closer_pct.between(25, 75).all()
    assert out.n_datasets.sum() == n

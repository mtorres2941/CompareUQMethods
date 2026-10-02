"""Stage 2g: which downstream metric the paper leads with.

What these pin are the four properties the stage's conclusions rest on.

That the new magnitude companion is a SHARE read at the building's own bad end
and not at the material's -- a planted case where one material is the only
source of the building's upper tail must give it nearly the whole share at the
total's 95th percentile while leaving its mean share alone, because if the two
could not be made to disagree there would be no reason to add the second one.

That the corrected strategy rank frequencies sum to exactly 1.0 across the
materials, which is the property the old `1 / (1 - capecc)` divisor was reaching
for and got only by assuming a constant applicability the absolute cap made
false.

That the recovery statistic is a measure of ERROR and not of informativeness: a
method that is the truth scores exactly zero, and a method that compresses every
material toward the mean cannot improve its score by being less useful, because
the divisor is the truth's spread rather than its own.

And that the tail contamination is cheap in W1 and expensive in a Monte Carlo,
which is the failure mode the stage was told to check a recommended metric
against rather than assume it immune to.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import fitting as FT            # noqa: E402
import metricset as MS          # noqa: E402
import plca as PL               # noqa: E402

NECCS = 4_000


def a_group(seed=0, n=60, k=4):
    """`k` datasets and their six fitted models, through the real pipeline."""
    rng = np.random.default_rng(seed)
    names, data, models = [], {}, {}
    for j in range(k):
        x = np.exp(rng.normal(0.0, 0.5 + 0.1 * j, size=n))
        x = x / x.mean()
        w = rng.dirichlet(np.ones(n))
        name = f'd{j}'
        names.append(name)
        data[name] = dict(data=x, weights=w)
        models[name] = FT.fit_pewt(x, w)[0]
    return names, data, models


# ---------------------------------------------------------------------------
# the magnitude companion
# ---------------------------------------------------------------------------
def test_share_at_total_quantile_is_a_share():
    """Bounded in [0, 1] and summing to one, which `eci_p95` is not."""
    rng = np.random.default_rng(0)
    draws = rng.lognormal(0.0, 0.4, size=(NECCS, 4))
    got = PL.share_at_total_quantile(draws)
    assert got.shape == (4,)
    assert np.all(got >= 0) and np.all(got <= 1)
    assert np.isclose(got.sum(), 1.0)


def test_share_at_total_quantile_finds_the_material_driving_the_tail():
    """THE REASON THE METRIC EXISTS, as a planted case.

    Three steady materials and one that is usually small and occasionally
    enormous. Its MEAN share is modest, so `eci_perc_mean` barely notices it;
    but it is the only thing that can put the BUILDING at its 95th percentile,
    so its share there is far larger. If these two could not be made to
    disagree, the second metric would be the first one under another name.
    """
    rng = np.random.default_rng(3)
    steady = rng.normal(1.0, 0.02, size=(NECCS, 3))
    # The spike has to be commoner than the 5 percent tail being read, or the
    # 95th percentile of the total sits below it and the metric correctly
    # reports that the spiky material is NOT what puts the building there.
    spike = np.where(rng.random(NECCS) < 0.15, 40.0, 0.2)
    draws = np.column_stack([steady, spike])
    perc_mean = (draws / draws.sum(axis=1, keepdims=True)).mean(axis=0)
    at_tail = PL.share_at_total_quantile(draws)
    assert perc_mean[3] < 0.35
    assert at_tail[3] > 0.85
    assert at_tail[3] > 2.5 * perc_mean[3]


def test_share_at_total_quantile_equals_the_mean_share_when_nothing_varies():
    """The control. With a deterministic total every iteration is at every
    quantile, so the two metrics must coincide."""
    draws = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (500, 1))
    at_tail = PL.share_at_total_quantile(draws)
    perc_mean = (draws / draws.sum(axis=1, keepdims=True)).mean(axis=0)
    assert np.allclose(at_tail, perc_mean)


def test_share_at_total_quantile_rejects_a_window_off_the_unit_interval():
    draws = np.random.default_rng(0).random((100, 3)) + 1.0
    with pytest.raises(ValueError):
        PL.share_at_total_quantile(draws, quantile=0.995, window=0.02)


def test_the_new_output_is_in_the_output_set_and_in_outputs():
    """It has to reach the sweep, the pair table and the truth run, and all
    three read `OUTPUTS`."""
    assert 'eci_perc_p95tot' in PL.OUTPUTS
    draws = np.random.default_rng(0).lognormal(0, 0.3, size=(NECCS, 4))
    got = PL.outputs(draws)
    assert 'eci_perc_p95tot' in got
    assert np.isclose(got['eci_perc_p95tot'].sum(), 1.0)


def test_adding_the_output_moved_none_of_the_others():
    """A pure addition: it consumes no randomness and touches no other column.

    Checked by recomputing every other output from the same draws with the
    original arithmetic written out here.
    """
    rng = np.random.default_rng(7)
    draws = rng.lognormal(0.0, 0.5, size=(NECCS, 4))
    got = PL.outputs(draws)
    total = draws.sum(axis=1)
    perc = draws / total[:, None]
    order = (-draws).argsort(axis=1).argsort(axis=1)
    assert np.allclose(got['eci_mean'], draws.mean(axis=0))
    assert np.allclose(got['eci_std'], draws.std(axis=0))
    assert np.allclose(got['eci_perc_mean'], perc.mean(axis=0))
    assert np.allclose(got['eci_perc_std'], perc.std(axis=0))
    assert np.allclose(got['eci_rank_1'], (order == 0).mean(axis=0))
    assert np.allclose(got['eci_p95'], np.quantile(draws, 0.95, axis=0))


# ---------------------------------------------------------------------------
# the strategy rank frequencies, and the divisor
# ---------------------------------------------------------------------------
def test_strategy_rank_1_sums_to_one_across_materials():
    """THE PROPERTY THE OLD DIVISOR WAS REACHING FOR.

    Conditional on the strategy applying to something, exactly one material is
    the best one to act on, so the four rank-1 frequencies are a partition.
    """
    rng = np.random.default_rng(1)
    red = -rng.random((2_000, 4))
    red[rng.random((2_000, 4)) < 0.6] = np.nan
    got = MS.strategy_rank_frequencies(red)
    assert np.isclose(got['rank_1'].sum(), 1.0)


def test_strategy_rank_frequencies_report_applicability_rather_than_divide_it_away():
    """`applies` and `bound_share` are the quantities a constant divisor
    destroyed by forcing them to 0.25 for every material and every method."""
    red = np.full((10, 3), np.nan)
    red[0:6, 0] = -1.0          # material 0 applies in 6 of 10 iterations
    red[0:2, 1] = -2.0          # material 1 in 2, and is better in both
    got = MS.strategy_rank_frequencies(red)
    assert np.isclose(got['applies'], 0.6)
    assert np.allclose(got['bound_share'], [0.6, 0.2, 0.0])
    # Of the 6 applicable iterations, material 1 is best in 2 and material 0
    # in the other 4.
    assert np.allclose(got['rank_1'], [4 / 6, 2 / 6, 0.0])
    assert np.allclose(got['rank_2'], [2 / 6, 0.0, 0.0])


def test_strategy_ranks_of_one_material_sum_to_its_own_applicability():
    """Across the four RANKS of one material: the share of applicable
    iterations in which its own strategy applied. Both sums are quantities,
    which is what the plain count over all iterations could not manage."""
    rng = np.random.default_rng(5)
    red = -rng.random((3_000, 4))
    red[rng.random((3_000, 4)) < 0.5] = np.nan
    got = MS.strategy_rank_frequencies(red)
    ranks = np.vstack([got[f'rank_{r}'] for r in range(1, 5)]).sum(axis=0)
    applicable = np.isfinite(red)[np.isfinite(red).any(axis=1)]
    assert np.allclose(ranks, applicable.mean(axis=0))


def test_strategy_frequencies_are_the_old_count_divided_by_applicability():
    """The correction is exactly a change of denominator, which is the claim
    the stage makes, so it is asserted rather than described."""
    rng = np.random.default_rng(11)
    red = -rng.random((2_000, 4))
    red[rng.random((2_000, 4)) < 0.4] = np.nan
    got = MS.strategy_rank_frequencies(red)
    old = pd.DataFrame(red).rank(axis=1)
    old_r1 = (old == 1).sum(axis=0).to_numpy(float) / red.shape[0]
    assert np.allclose(got['rank_1'] * got['applies'], old_r1)


def test_a_strategy_that_never_applies_is_reported_and_not_crashed_on():
    got = MS.strategy_rank_frequencies(np.full((50, 3), np.nan))
    assert got['applies'] == 0.0
    assert np.allclose(got['rank_1'], 0.0)
    assert np.allclose(got['bound_share'], 0.0)


# ---------------------------------------------------------------------------
# does the metric recover the right answer
# ---------------------------------------------------------------------------
def a_truth_frame(seed=0, n_plca=40, k=4, offset=0.0, noise=0.0):
    """A tidy truth frame with a known error, for the recovery statistics."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_plca):
        truth = rng.lognormal(0.0, 0.5, size=k)
        for m in ('A', 'B'):
            shift = offset if m == 'B' else 0.0
            got = truth + shift + noise * rng.normal(size=k)
            for j in range(k):
                rows.append(dict(
                    plca=p, dataset=f'd{p}_{j}', method=m,
                    truth_parent='market',
                    eci_mean=got[j], eci_mean__truth=truth[j],
                    eci_mean__error=got[j] - truth[j]))
    return pd.DataFrame(rows)


def test_a_method_that_is_the_truth_has_zero_recovery_error():
    """The control every recovery number is read against."""
    frame = a_truth_frame()
    got = MS.recovery_table(frame, outputs=('eci_mean',), resamples=40,
                            rng=np.random.default_rng(0))
    assert np.allclose(got['recovery'], 0.0)
    assert np.allclose(got['bias'], 0.0)


def test_recovery_grows_with_the_error_and_is_in_units_of_the_truth_spread():
    small = MS.recovery_table(a_truth_frame(offset=0.05),
                              outputs=('eci_mean',), resamples=40,
                              rng=np.random.default_rng(0))
    big = MS.recovery_table(a_truth_frame(offset=0.50),
                            outputs=('eci_mean',), resamples=40,
                            rng=np.random.default_rng(0))
    b_small = float(small.loc[small.method == 'B', 'recovery'].iloc[0])
    b_big = float(big.loc[big.method == 'B', 'recovery'].iloc[0])
    assert b_big > 5 * b_small
    # The divisor really is the truth's own spread, so the same absolute error
    # on a metric with twice the spread reads half as large.
    sd = float(small['truth_sd'].iloc[0])
    assert np.isclose(b_small, 0.05 / sd, rtol=0.05)


def test_recovery_cannot_be_bought_by_being_less_informative():
    """THE REASON THE DIVISOR IS THE TRUTH'S SPREAD AND NOT THE METHOD'S.

    A method that reports the same number for every material has learned
    nothing, and must not score better than one that tracks the truth with a
    little noise. Dividing by the method's own spread would reward it.
    """
    rng = np.random.default_rng(2)
    rows = []
    for p in range(60):
        truth = rng.lognormal(0.0, 0.5, size=4)
        flat = np.full(4, truth.mean())
        noisy = truth + 0.05 * rng.normal(size=4)
        for m, got in (('flat', flat), ('noisy', noisy)):
            for j in range(4):
                rows.append(dict(plca=p, dataset=f'd{p}_{j}', method=m,
                                 truth_parent='market', eci_mean=got[j],
                                 eci_mean__truth=truth[j],
                                 eci_mean__error=got[j] - truth[j]))
    got = MS.recovery_table(pd.DataFrame(rows), outputs=('eci_mean',),
                            resamples=40, rng=np.random.default_rng(0))
    rec = got.set_index('method')['recovery']
    assert rec['noisy'] < rec['flat']


def test_decision_agreement_is_one_for_the_truth_and_chance_for_a_shuffle():
    rng = np.random.default_rng(4)
    rows = []
    for p in range(300):
        truth = rng.lognormal(0.0, 0.6, size=4)
        for m, got in (('exact', truth), ('shuffled', rng.permutation(truth))):
            for j in range(4):
                rows.append(dict(plca=p, dataset=f'd{p}_{j}', method=m,
                                 truth_parent='market', eci_mean=got[j],
                                 eci_mean__truth=truth[j],
                                 eci_mean__error=got[j] - truth[j]))
    got = MS.decision_agreement(pd.DataFrame(rows), outputs=('eci_mean',),
                                resamples=40, rng=np.random.default_rng(0))
    a = got.set_index('method')
    assert np.isclose(a.loc['exact', 'agreement'], 1.0)
    assert np.isclose(a.loc['exact', 'chance'], 0.25)
    assert abs(a.loc['shuffled', 'agreement'] - 0.25) < 0.08


def test_metric_verdict_puts_the_best_recovered_metric_first():
    frame = pd.concat([
        a_truth_frame(seed=1, offset=0.01),
        a_truth_frame(seed=2, offset=0.40).rename(
            columns={c: c.replace('eci_mean', 'eci_rank_1')
                     for c in ('eci_mean', 'eci_mean__truth',
                               'eci_mean__error')})], ignore_index=True)
    rec = MS.recovery_table(frame, outputs=('eci_mean', 'eci_rank_1'),
                            resamples=40, rng=np.random.default_rng(0))
    agr = MS.decision_agreement(frame, outputs=('eci_mean', 'eci_rank_1'),
                                resamples=40, rng=np.random.default_rng(0))
    got = MS.metric_verdict(rec, agr)
    assert list(got['output'])[0] == 'eci_mean'


# ---------------------------------------------------------------------------
# the tail failure mode
# ---------------------------------------------------------------------------
def test_contamination_inverts_its_own_cdf():
    """It is sampled by inverse CDF like every other model in this study, so
    the bisection has to invert the mixture CDF it is scored on."""
    names, data, models = a_group()
    base = models['d0']['KDE, Variable']
    bad = MS.Contaminated(base, factor=100.0, weight=1e-3, anchor=1.0)
    q = np.array([0.001, 0.05, 0.25, 0.5, 0.9, 0.99, 0.9995])
    assert np.allclose(bad.cdf(bad.ppf(q)), q, atol=1e-6)


def test_contamination_is_a_mixture_and_reduces_to_the_base_at_zero_weight():
    names, data, models = a_group()
    base = models['d0']['Lognormal, Uniform']
    same = MS.Contaminated(base, factor=1000.0, weight=0.0, anchor=1.0)
    x = np.linspace(0.01, 5.0, 200)
    assert np.allclose(same.cdf(x), base.cdf(x), atol=1e-12)


def _stress(seed=9, **kw):
    names, data, models = a_group()
    u = np.random.default_rng(seed).random((NECCS, len(names)))
    return MS.tail_stress(
        models, names, {k: v['data'] for k, v in data.items()},
        {k: v['weights'] for k, v in data.items()}, u, 'KDE, Variable', **kw)


def test_the_body_of_w1_goes_blind_exactly_where_its_own_grid_ends():
    """THE FAILURE MODE THE STAGE WAS TOLD TO CHECK, as an assertion.

    W1 integrated over the scoring grid alone charges for the MASS a model
    moves and, once that mass is past the top of the grid, stops charging for
    the DISTANCE entirely, because the integrand is clipped away there. The
    sweep runs from inside the data out to three thousand times its mean, so
    the assertion is two-sided: the score still moves while the contamination
    is inside the grid, and it is constant to a part in a million once the
    contamination is beyond it.
    """
    got = _stress(score=lambda m, x, w: FT.score_w1_model(m, x, w, tail=False))
    for weight, block in got.groupby('weight'):
        inside = block[~block.beyond_grid]
        # CLEARLY beyond, not merely past the boundary: the contamination has
        # its own width, so the first point past the grid top still has a
        # little of itself inside and is a transitional case rather than a
        # counterexample.
        outside = block[block.factor > 1.5 * block.grid_top]
        assert len(inside) >= 3 and len(outside) >= 10, weight
        # Seven significant figures across two and a half orders of magnitude
        # in the contamination distance; the residual is float noise at the
        # grid's far end, not a response to the distance.
        assert outside['w1'].std() / outside['w1'].mean() < 1e-7, weight
        assert inside['w1'].std() / inside['w1'].mean() > 1e-4, weight


def test_the_grid_top_is_of_order_ten_times_the_dataset_mean():
    """Which is why the blindness starts just past the data, and why the sweep
    has to begin inside it."""
    got = _stress(score=FT.score_w1_model)
    top = float(got['grid_top'].iloc[0])
    assert 3.0 < top < 60.0
    assert got['beyond_grid'].any() and (~got['beyond_grid']).any()


def test_the_tail_term_stage_2c_added_is_what_closes_that_blindness():
    """The same contamination, scored the way the study now scores. The tail
    term integrates the model's survival function beyond the grid, so the score
    rises with the distance instead of ignoring it."""
    got = _stress(score=FT.score_w1_model)
    block = got[(got['weight'] == 1e-4) & got['beyond_grid']].sort_values('factor')
    rel = block['w1_rel'].to_numpy(float)
    assert (np.diff(rel) > 0).all()
    assert rel[-1] > 10 * rel[0]


def test_a_share_is_immune_to_the_tail_and_a_level_is_not():
    """THE PROPERTY THAT DECIDES WHICH COMPANION IS SAFE.

    A share and a rank frequency are bounded in [0, 1] and saturate: once a
    material's draw is enormous it holds the whole share and takes rank one,
    and making it a thousand times more enormous changes neither. A mean, a
    standard deviation and a variance share have no such ceiling.
    """
    got = _stress(score=FT.score_w1_model)
    # THE CLAIM IS ABOUT A FAR TAIL AND THE TEST HAS TO SAY SO. Contamination
    # INSIDE the data moves everything, shares included, and that is not the
    # failure mode: it is a badly fitting model, which the criterion charges
    # for in full. The immunity claim is about mass the criterion cannot see.
    got = got[got.factor > 1.5 * got.grid_top]
    exp = MS.tail_exposure(got).set_index('output')
    for bounded in ('eci_perc_mean', 'eci_perc_p95tot', 'eci_rank_1'):
        assert exp.loc[bounded, 'rel_max'] < 0.2, bounded
    for level in ('eci_mean', 'eci_std', 'ui'):
        assert exp.loc[level, 'rel_max'] > 1.0, level
    assert exp.loc['eci_std', 'exposure'] > exp.loc['eci_perc_mean', 'exposure']


def test_a_bounded_metric_stops_moving_once_the_tail_is_far_enough_out():
    """Saturation, asserted directly rather than through a ratio: past a
    hundred times the dataset mean the share metrics are the same number at
    every further distance, while the level metrics keep climbing."""
    got = _stress(score=FT.score_w1_model)
    block = got[(got['weight'] == 1e-3) & (got['factor'] > 100)]
    assert len(block) >= 5
    for bounded in ('eci_perc_p95tot_rel', 'eci_rank_1_rel'):
        assert block[bounded].std() < 1e-12, bounded
    assert block['eci_perc_mean_rel'].max() < 0.01
    assert block['eci_std_rel'].max() / block['eci_std_rel'].min() > 5.0


def test_rel_error_divides_by_the_level_and_recovery_by_the_spread():
    """THE TWO DIVISORS ARE DIFFERENT STATISTICS, which is why both are kept.

    The claim scorecard needs one definition it can write for a building total
    and for a material's share alike, and only the LEVEL exists for both. The
    candidate ranking needs to know whether a metric can tell two materials
    apart, and only the SPREAD says that. Their ratio is not a constant, so a
    figure mixing them is not comparing like with like.
    """
    got = MS.recovery_table(a_truth_frame(offset=0.20), outputs=('eci_mean',),
                            resamples=40, rng=np.random.default_rng(0))
    row = got[got.method == 'B'].iloc[0]
    assert np.isclose(row['rel_error'], row['abs_error'] / abs(row['truth_mean']))
    assert np.isclose(row['recovery'], row['abs_error'] / row['truth_sd'])
    # A lognormal(0, 0.5) has a mean of about 1.13 and a standard deviation of
    # about 0.60, so the two readings of the same error differ by about two.
    assert row['truth_mean'] > 1.5 * row['truth_sd']
    assert row['rel_error'] < 0.8 * row['recovery']


def test_rel_error_is_a_ratio_of_means_and_not_a_mean_of_ratios():
    """WHY IT IS NOT THE ORDINARY MEAN ABSOLUTE PERCENTAGE ERROR.

    A per-material ratio is unbounded where the true value approaches zero, and
    the real truth run has 2,904 materials of 60,000 whose true uncertainty
    index is below a hundredth of the mean and some that are negative. One
    near-zero material must not be able to move the statistic.
    """
    frame = a_truth_frame(offset=0.10)
    near_zero = frame.eci_mean__truth.idxmin()
    frame.loc[near_zero, 'eci_mean__truth'] = 1e-9
    got = MS.recovery_table(frame, outputs=('eci_mean',), resamples=40,
                            rng=np.random.default_rng(0))
    row = got[got.method == 'B'].iloc[0]
    assert np.isfinite(row['rel_error'])
    assert 0.05 < row['rel_error'] < 0.5


def test_display_method_renames_only_the_weighting_and_only_for_display():
    """THE STORED `method` VALUE IS THE JOIN KEY AND MUST NOT MOVE.

    "Variable" means the market shares were drawn from a flat Dirichlet because
    nobody publishes them. Read as "market shares accounted for" it makes a
    result where uniform weighting wins look like a modeling error, which is
    what happened. The display label says what the method does; the data keeps
    the key, because every table this study writes and every regression fixture
    joins on it.

    THE LABEL IS "market weights" FROM 2026-09-25, by author decision, and the
    overclaim decision 160 guarded against is handled in the TEXT rather than
    the label: the methods section says at first use that they are drawn from a
    Dirichlet because production volumes are not published, and the oracle
    scheme is "known market shares" so the contrast is visible wherever both
    appear. A label cannot carry a caveat; a sentence can.
    """
    assert FT.display_method('KDE, Variable') == 'KDE, market weights'
    assert FT.display_method('KDE, Uniform') == 'KDE, uniform weights'
    assert FT.display_method('KDE, Oracle') == 'KDE, known market shares'
    assert FT.display_method('KDE, Variable', short=True) == 'KDE, market'
    # The family is never touched, and an unknown scheme passes through rather
    # than raising, so a sweep that invents one still plots.
    assert FT.display_method('Lognormal, Somethingelse') == 'Lognormal, Somethingelse'
    assert FT.display_method('NoComma') == 'NoComma'
    # The canonical list itself is unchanged: this is a view, not a rename.
    assert FT.PEWT == ['Normal, Uniform', 'Normal, Variable',
                       'Lognormal, Uniform', 'Lognormal, Variable',
                       'KDE, Uniform', 'KDE, Variable']


def a_size_banded_truth_frame(seed=1, per_band=30, k=4):
    """A truth frame where the ORDERING of two methods flips with dataset size.

    Method 'small_is_good' is accurate below 100 and poor above; 'big_is_good'
    is the reverse. Pooling the two bands hides both facts, which is exactly
    the misreading `size_band_recovery` exists to prevent.
    """
    rng = np.random.default_rng(seed)
    rows, plca = [], 0
    for n, small_err, big_err in ((10, 0.02, 0.30), (5000, 0.30, 0.02)):
        for _ in range(per_band):
            truth = rng.lognormal(0.0, 0.4, size=k)
            for m, e in (('small_is_good', small_err), ('big_is_good', big_err)):
                got = truth + e * rng.normal(size=k)
                for j in range(k):
                    rows.append(dict(
                        plca=plca, dataset=f'd{plca}_{j}', method=m, n=n,
                        truth_parent='market',
                        eci_mean=got[j], eci_mean__truth=truth[j],
                        eci_mean__error=got[j] - truth[j]))
            plca += 1
    return pd.DataFrame(rows)


def test_size_band_recovery_shows_an_ordering_the_pooled_table_hides():
    frame = a_size_banded_truth_frame()
    banded = MS.size_band_recovery(frame, outputs=('eci_mean',))
    got = banded.pivot(index='band', columns='method', values='rel_error')
    # Each band picks the method built to win it.
    assert got.loc['10-99'].idxmin() == 'small_is_good'
    assert got.loc['1000+'].idxmin() == 'big_is_good'
    # Pooled, the two are indistinguishable, which is the whole point: a single
    # scorecard row cannot report a flip.
    pooled = MS.recovery_table(frame, outputs=('eci_mean',), resamples=40,
                               rng=np.random.default_rng(0))
    a, b = pooled.set_index('method')['rel_error']
    assert abs(a - b) < 0.25 * max(a, b)
    # Every band divides by the SAME true level, so the four rows of a column
    # are comparable with each other rather than each being self-normalized.
    assert banded.truth_level.nunique() == 1
    # And a band reports how many materials it holds, so a thin one is visible.
    assert set(banded.n_materials) == {30 * 4}


def test_size_band_recovery_skips_an_output_with_no_usable_level():
    """A distance has a true value of zero, so it has no level to divide by and
    must be left out rather than dividing by something near zero."""
    frame = a_size_banded_truth_frame()
    frame['w1__truth'] = 0.0
    frame['w1__error'] = 0.1
    got = MS.size_band_recovery(frame, outputs=('eci_mean', 'w1'))
    assert set(got.output) == {'eci_mean'}


# ---------------------------------------------------------------------------
# Stage 2h: the two errors a scorecard row can report, and the one it must
# ---------------------------------------------------------------------------
def a_cancelling_frame(n_units=200):
    """One method that is wrong by the same amount every time, in alternating
    directions, and one that is right.

    This is the exact shape of the defect Stage 2h corrects: averaging the
    SIGNED error over units before taking the absolute value reports the
    cancelling method as perfect, and it is wrong on every single unit.
    """
    unit = np.arange(n_units)
    sign = np.where(unit % 2 == 0, 1.0, -1.0)
    rows = []
    for name, err in (('cancels', 0.2 * sign), ('exact', np.zeros(n_units))):
        rows.append(pd.DataFrame({
            'unit': unit, 'saving': unit % 4, 'method': name,
            'q__truth': 1.0, 'q__error': err, 'q': 1.0 + err}))
    return pd.concat(rows, ignore_index=True)


def test_per_unit_error_separates_a_cancelling_method_from_an_exact_one():
    """The defect, planted. A method wrong by 0.2 on every unit must not be
    reported as having no error because its signs alternate."""
    got = MS.per_unit_error(a_cancelling_frame(), 'q')
    # The per-unit error is the truth about a SINGLE decision.
    assert got.loc['cancels', 'error'] == pytest.approx(0.2)
    assert got.loc['exact', 'error'] == pytest.approx(0.0)
    # The portfolio error is the truth about the AVERAGE over many, and it is
    # near zero for the cancelling method. That is not wrong, it is a different
    # question -- which is why both are reported and both are labeled.
    assert got.loc['cancels', 'error_portfolio'] == pytest.approx(0.0,
                                                                  abs=1e-12)
    # The two must not be interchangeable, or the correction would be empty.
    assert got.loc['cancels', 'error'] > 100 * got.loc['cancels',
                                                       'error_portfolio']
    assert (got['truth_level'] == 1.0).all()
    assert (got['n_units'] == 200).all()


def test_per_unit_error_blocks_the_portfolio_form_but_never_the_per_unit_one():
    """Averaging within a block before taking the absolute value is what the
    design comparison does, reporting per claimed saving. It changes the
    portfolio form and must leave the per-unit form alone."""
    frame = a_cancelling_frame()
    plain = MS.per_unit_error(frame, 'q')
    blocked = MS.per_unit_error(frame, 'q', block='saving')
    assert blocked['error'].equals(plain['error'])
    # Blocks 0 and 2 hold only positive errors and 1 and 3 only negative, so
    # blocking recovers the per-unit magnitude where the pooled form cancels.
    assert blocked.loc['cancels', 'error_portfolio'] == pytest.approx(0.2)


def test_per_unit_error_refuses_an_output_it_cannot_score():
    with pytest.raises(KeyError):
        MS.per_unit_error(a_cancelling_frame(), 'not_an_output')


def a_scorecard_fixture():
    """The four frames the scorecard reads, each carrying one usable claim."""
    rng = np.random.default_rng(11)
    n = 120
    building = pd.DataFrame({
        'plca': np.arange(n).repeat(2),
        'method': np.tile(['a', 'b'], n),
        'total_mean__truth': 4.0,
        'total_mean__error': np.tile([0.4, -0.4], n) * np.where(
            np.arange(2 * n) % 4 < 2, 1.0, -1.0)})
    building['total_mean'] = 4.0 + building['total_mean__error']
    intervention = building.rename(columns={
        'total_mean__truth': 'cap_share_touched__truth',
        'total_mean__error': 'cap_share_touched__error',
        'total_mean': 'cap_share_touched'}).assign(cap_share_touched__truth=0.5)
    swap = pd.DataFrame({
        'pair': np.arange(n).repeat(2), 'saving': (np.arange(n) % 3).repeat(2),
        'method': np.tile(['a', 'b'], n), 'discernibility__truth': 0.6,
        'discernibility__error': rng.normal(0, 0.05, 2 * n)})
    recovery = pd.DataFrame({
        'output': ['eci_mean', 'eci_mean'], 'method': ['a', 'b'],
        'abs_error': [0.10, 0.30], 'bias_raw': [0.01, -0.02],
        'truth_mean': [1.0, 1.0], 'n': [n, n]})
    return recovery, building, intervention, swap


def test_claim_scorecard_puts_every_row_on_the_per_unit_error():
    recovery, building, intervention, swap = a_scorecard_fixture()
    claims = (('magnitude', 'the total: its mean', 'building', 'total_mean'),
              ('attribution', 'a material: its mean contribution', 'recovery',
               'eci_mean'),
              ('action', 'a cap: how often it binds', 'intervention',
               'cap_share_touched'),
              ('comparison', 'the probability B beats A', 'swap',
               'discernibility'))
    got = MS.claim_scorecard(recovery, building, intervention, swap,
                             claims=claims)
    assert len(got) == 8
    assert set(got.claim) == {c[1] for c in claims}
    # The action row's errors cancel exactly across groups, so the per-unit and
    # portfolio forms must disagree by the whole error rather than agreeing.
    cap = got[got.claim == 'a cap: how often it binds']
    assert (cap.total_error > 0.7).all()          # 0.4 / 0.5
    assert (cap.portfolio_error < 1e-9).all()
    # And the magnitude row, which was already per unit, is unchanged by this.
    tot = got[got.claim == 'the total: its mean']
    assert tot.total_error.to_numpy() == pytest.approx(0.1)   # 0.4 / 4.0


def test_claim_scorecard_derives_stakes_and_best_from_the_per_unit_error():
    recovery, building, intervention, swap = a_scorecard_fixture()
    got = MS.claim_scorecard(
        recovery, building, intervention, swap,
        claims=(('attribution', 'a material: its mean contribution',
                 'recovery', 'eci_mean'),))
    row = got.set_index('method')
    assert row.loc['a', 'total_error'] == pytest.approx(0.10)
    assert row.loc['b', 'total_error'] == pytest.approx(0.30)
    # best_error is what is left however you choose; stakes is what the choice
    # costs; the two add to the worst method's total.
    assert row['best_error'].to_numpy() == pytest.approx(0.10)
    assert row['stakes'].to_numpy() == pytest.approx(0.20)
    assert row['best_error'].iloc[0] + row['stakes'].iloc[0] == pytest.approx(
        row['total_error'].max())
    assert (row['best_method'] == 'a').all()
    assert row.loc['a', 'rank'] == 1 and row.loc['b', 'rank'] == 2
    assert row['methods_differ'].all()


def test_claim_scorecard_marks_a_row_the_six_agree_on():
    """A row whose whole spread is arithmetic noise must be flagged, so that
    naming a best method on it is visibly not a ranking."""
    recovery, building, intervention, swap = a_scorecard_fixture()
    recovery = recovery.assign(abs_error=[0.10, 0.1000001])
    got = MS.claim_scorecard(
        recovery, building, intervention, swap,
        claims=(('attribution', 'a', 'recovery', 'eci_mean'),))
    assert not got.methods_differ.any()


def test_adding_a_method_leaves_every_other_method_s_error_bit_identical():
    """A seventh policy on the scorecard cannot move the six.

    THIS IS THE STATEMENT THE NOTEBOOK'S CONTROL CANNOT MAKE EXACTLY, so it is
    made here. Stage 4 added the feasible size rule to notebook 3's MAIN truth
    pass rather than scoring it in a pass of its own, because the scorecard
    figure's title is a COUNT of claims and comparing a column measured in one
    Monte Carlo experiment against six measured in another decided one of them
    on the experiment rather than on the method (decision 237). What licenses
    that is: the error of a method is computed WITHIN that method, so adding
    another cannot touch it.

    `scale` is excluded deliberately and is not a hole. It is the mean TRUE
    level over every row of a claim, and the truth repeats identically once
    per method, so with a seventh it is a mean over seven identical blocks
    instead of six: the same number by a different summation order, which can
    differ in the last bit. `total_error` is the ratio and inherits it. The
    notebook prints both gaps.
    """
    rng = np.random.default_rng(11)
    base = a_truth_frame(seed=3, noise=0.3, offset=0.2)
    extra = base[base.method == 'A'].copy()
    extra['eci_mean'] = extra['eci_mean'] + 0.37      # a genuinely new column
    extra['eci_mean__error'] = extra['eci_mean'] - extra['eci_mean__truth']
    extra['method'] = 'C'
    with_extra = pd.concat([base, extra], ignore_index=True)

    kw = dict(outputs=('eci_mean',), resamples=40, rng=rng)
    six = MS.recovery_table(base, **kw).set_index('method')
    seven = MS.recovery_table(with_extra, **dict(kw, rng=np.random.default_rng(11))
                              ).set_index('method')
    for m in ('A', 'B'):
        assert seven.loc[m, 'abs_error'] == six.loc[m, 'abs_error']
        assert seven.loc[m, 'bias_raw'] == six.loc[m, 'bias_raw']

    # and the same for a row-level claim, which is where `per_unit_error` runs
    rows = pd.DataFrame(dict(
        plca=list(range(30)) * 2,
        method=['A'] * 30 + ['B'] * 30,
        total_mean=rng.normal(4.0, 0.3, 60),
        total_mean__truth=np.tile(rng.normal(4.0, 0.3, 30), 2)))
    rows['total_mean__error'] = rows.total_mean - rows.total_mean__truth
    more = rows[rows.method == 'A'].copy()
    more['method'] = 'C'
    more['total_mean'] = more['total_mean'] * 1.15
    more['total_mean__error'] = more.total_mean - more.total_mean__truth
    u6 = MS.per_unit_error(rows, 'total_mean')
    u7 = MS.per_unit_error(pd.concat([rows, more], ignore_index=True),
                           'total_mean')
    for m in ('A', 'B'):
        assert u7.loc[m, 'error'] == u6.loc[m, 'error']
        assert u7.loc[m, 'error_portfolio'] == u6.loc[m, 'error_portfolio']

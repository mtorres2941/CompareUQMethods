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


def test_the_body_of_w1_is_exactly_blind_to_how_far_out_the_tail_goes():
    """THE FAILURE MODE THE STAGE WAS TOLD TO CHECK, as an assertion.

    W1 integrated over the scoring grid alone charges for the MASS a model
    moves and not for the DISTANCE it moves it, because above the grid's top
    the integrand is clipped away. Moving a thousandth of the mass to ten, a
    hundred or a thousand times the dataset mean gives the SAME body score to
    five decimal places, while a Monte Carlo that samples the model is wrecked
    by the third case and not by the first.
    """
    got = _stress(score=lambda m, x, w: FT.score_w1_model(m, x, w, tail=False))
    for weight, block in got.groupby('weight'):
        # Relative to the score itself, because the grid's own top point picks
        # up a difference of order 1e-9 from the far component's CDF there.
        assert block['w1'].std() / block['w1'].mean() < 1e-6, weight
        assert block['factor'].nunique() == 3


def test_the_tail_term_stage_2c_added_is_what_closes_that_blindness():
    """The same contamination, scored the way the study now scores. The tail
    term integrates the model's survival function beyond the grid, so the score
    rises with the distance instead of ignoring it."""
    got = _stress(score=FT.score_w1_model)
    block = got[got['weight'] == 1e-4].sort_values('factor')
    rel = block['w1_rel'].to_numpy(float)
    assert rel[0] < rel[1] < rel[2]
    assert rel[2] > 10 * rel[0]


def test_a_share_is_immune_to_the_tail_and_a_level_is_not():
    """THE PROPERTY THAT DECIDES WHICH COMPANION IS SAFE.

    A share and a rank frequency are bounded in [0, 1] and saturate: once a
    material's draw is enormous it holds the whole share and takes rank one,
    and making it a thousand times more enormous changes neither. A mean, a
    standard deviation and a variance share have no such ceiling.
    """
    got = _stress(score=FT.score_w1_model)
    exp = MS.tail_exposure(got).set_index('output')
    for bounded in ('eci_perc_mean', 'eci_perc_p95tot', 'eci_rank_1'):
        assert exp.loc[bounded, 'rel_max'] < 0.2, bounded
    for level in ('eci_mean', 'eci_std', 'ui'):
        assert exp.loc[level, 'rel_max'] > 1.0, level
    assert exp.loc['eci_std', 'exposure'] > exp.loc['eci_perc_mean', 'exposure']


def test_a_bounded_metric_does_not_move_when_the_tail_moves_further_out():
    """Saturation, asserted directly rather than through a ratio: at a fixed
    contamination weight the share metrics are the same number at ten times the
    mean and at a thousand times it."""
    got = _stress(score=FT.score_w1_model)
    block = got[got['weight'] == 1e-3]
    # Exactly flat: once a material is the whole tail, every further factor of
    # ten leaves its share at the total's 95th percentile and its rank-1
    # frequency untouched to the last digit.
    for bounded in ('eci_perc_p95tot_rel', 'eci_rank_1_rel'):
        assert block[bounded].std() < 1e-12, bounded
    # The mean share is not exactly flat, because at ten times the mean the
    # material does not yet hold the whole of every iteration it dominates,
    # but it is three orders of magnitude steadier than the level metrics.
    assert block['eci_perc_mean_rel'].max() < 0.01
    assert block['eci_std_rel'].std() > 1.0

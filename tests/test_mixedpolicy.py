"""Stage 2j: the mixed policy is a SELECTION, not a seventh estimator.

What these pin are the four properties every number in the stage rests on.
That the rule reads exactly one input, the dataset's size, so it cannot have
acquired a second selector by accident. That adding it refits nothing and
consumes no randomness, so the six fixed policies it is compared against come
back bit-identical. That a group whose materials all sit on one side of the
threshold reproduces that fixed policy EXACTLY, which is the control that says
the mixed rows are the fixed rows wherever the rule does not act. And that the
fit-score selection cannot disagree with the six-method table it is taken from.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import fitting as FT            # noqa: E402
import mixedpolicy as MP        # noqa: E402
import plca as PL               # noqa: E402

NECCS = 2_000


def a_group(seed=0, sizes=(6, 30, 300, 3000)):
    """Four datasets straddling the threshold, fitted through the real path."""
    rng = np.random.default_rng(seed)
    names, data, models = [], {}, {}
    for j, n in enumerate(sizes):
        x = np.exp(rng.normal(0.0, 0.5 + 0.1 * j, size=int(n)))
        x = x / x.mean()
        w = rng.dirichlet(np.ones(int(n)))
        name = f'd{j}'
        names.append(name)
        data[name] = dict(data=x, weights=w)
        models[name] = FT.fit_pewt(x, w)[0]
    return names, data, models, {n: s for n, s in zip(names, sizes)}


# ---------------------------------------------------------------------------
# the rule is one number
# ---------------------------------------------------------------------------
def test_the_rule_switches_exactly_at_the_threshold():
    assert MP.select_method(MP.MIXED_THRESHOLD - 1) == MP.SMALL_METHOD
    assert MP.select_method(MP.MIXED_THRESHOLD) == MP.LARGE_METHOD
    assert MP.select_method(MP.MIXED_THRESHOLD + 1) == MP.LARGE_METHOD


def test_the_rule_reads_nothing_but_the_size():
    """It takes one scalar, so there is no second selector to have crept in.

    A rule with two numbers in it is not the deliverable (decisions 88, 139),
    and the cheapest guarantee of that is an interface that cannot express one.
    """
    import inspect
    sig = inspect.signature(MP.select_method)
    required = [p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty]
    assert [p.name for p in required] == ['n']


def test_the_threshold_is_the_studys_own_and_is_not_re_derived_here():
    """81 declarations, decision 142, reproduced unchanged by decision 198.

    Pinned so that a later edit moving it has to move this line too and say so.
    """
    assert MP.MIXED_THRESHOLD == 81
    assert MP.LARGE_METHOD == 'KDE, Variable'
    assert MP.SMALL_METHOD == 'Lognormal, Uniform'


# ---------------------------------------------------------------------------
# nothing is refitted
# ---------------------------------------------------------------------------
def test_the_mixed_key_is_the_selected_model_object_itself():
    names, _, models, sizes = a_group()
    models, choice = MP.add_mixed(models, sizes)
    for name in names:
        assert models[name][MP.MIXED] is models[name][choice[name]]
        assert choice[name] == MP.select_method(sizes[name])


def test_adding_the_mixed_key_leaves_the_six_fixed_models_untouched():
    names, _, models, sizes = a_group()
    before = {d: dict(models[d]) for d in names}
    MP.add_mixed(models, sizes)
    for d in names:
        for m in FT.PEWT:
            assert models[d][m] is before[d][m]


def test_a_group_entirely_above_the_threshold_reproduces_that_fixed_policy():
    """The control. Where the rule does not act, the mixed draw IS the fixed
    draw, bit for bit, so any difference elsewhere is the rule and not noise.
    """
    names, _, models, _ = a_group(sizes=(200, 400, 800, 1600))
    MP.add_mixed(models, {n: 200 for n in names})
    u = np.random.default_rng(3).random((NECCS, len(names)))
    mixed = PL.draw_contributions(models, names, MP.MIXED, u)
    fixed = PL.draw_contributions(models, names, MP.LARGE_METHOD, u)
    assert np.array_equal(mixed, fixed)


def test_a_group_entirely_below_the_threshold_reproduces_the_other_one():
    names, _, models, _ = a_group(sizes=(6, 8, 12, 20))
    MP.add_mixed(models, {n: 10 for n in names})
    u = np.random.default_rng(3).random((NECCS, len(names)))
    mixed = PL.draw_contributions(models, names, MP.MIXED, u)
    fixed = PL.draw_contributions(models, names, MP.SMALL_METHOD, u)
    assert np.array_equal(mixed, fixed)


def test_a_split_group_matches_neither_fixed_policy_and_matches_both_in_part():
    """And where it DOES act, it is each material's own selected model."""
    names, _, models, sizes = a_group()
    MP.add_mixed(models, sizes)
    u = np.random.default_rng(4).random((NECCS, len(names)))
    mixed = PL.draw_contributions(models, names, MP.MIXED, u)
    big = PL.draw_contributions(models, names, MP.LARGE_METHOD, u)
    small = PL.draw_contributions(models, names, MP.SMALL_METHOD, u)
    assert not np.array_equal(mixed, big)
    assert not np.array_equal(mixed, small)
    for j, name in enumerate(names):
        want = big if sizes[name] >= MP.MIXED_THRESHOLD else small
        assert np.array_equal(mixed[:, j], want[:, j])


def test_the_truth_run_carries_the_mixed_policy_through_unchanged():
    """And the six fixed methods come back identical whether or not it is
    present, which is what licenses reading them as controls."""
    names, _, models, sizes = a_group()
    MP.add_mixed(models, sizes)

    class _Parent:
        def rvs_from_uniform(self, u):
            return np.asarray(u, float) + 0.5

    samplers = {d: _Parent() for d in names}
    u = np.random.default_rng(5).random((NECCS, len(names)))
    six = pd.DataFrame(PL.truth_rows(models, samplers, names, u,
                                     methods=FT.PEWT, plca=0))
    seven = pd.DataFrame(PL.truth_rows(models, samplers, names, u,
                                       methods=MP.methods_with_mixed(),
                                       plca=0))
    assert set(seven.method) == set(FT.PEWT) | {MP.MIXED}
    cols = [c for c in six.columns if six[c].dtype.kind in 'fb']
    pd.testing.assert_frame_equal(
        six[six.method.isin(FT.PEWT)].reset_index(drop=True)[cols],
        seven[seven.method.isin(FT.PEWT)].reset_index(drop=True)[cols])


# ---------------------------------------------------------------------------
# the fit selection cannot disagree with the table it is taken from
# ---------------------------------------------------------------------------
def test_fit_scores_are_selected_and_never_computed():
    scores = pd.DataFrame({
        'dataset': ['a', 'a', 'b', 'b'],
        'n': [10, 10, 5000, 5000],
        'method': [MP.SMALL_METHOD, MP.LARGE_METHOD] * 2,
        'w1_market': [0.11, 0.22, 0.33, 0.04]})
    out = MP.fit_scores(scores)
    mixed = out[out.method == MP.MIXED].set_index('dataset')
    assert mixed.loc['a', 'w1_market'] == 0.11
    assert mixed.loc['b', 'w1_market'] == 0.04
    assert mixed.loc['a', 'picked'] == MP.SMALL_METHOD
    assert mixed.loc['b', 'picked'] == MP.LARGE_METHOD
    # the six original rows are untouched
    pd.testing.assert_frame_equal(
        out[out.method != MP.MIXED].drop(columns='picked')
        .reset_index(drop=True), scores)


def test_fit_scores_leave_the_other_four_methods_alone():
    scores = pd.DataFrame({
        'dataset': ['a'] * 6, 'n': [10] * 6, 'method': FT.PEWT,
        'w1_market': [0.5, 0.6, 0.11, 0.7, 0.8, 0.22]})
    out = MP.fit_scores(scores)
    assert (out.method == MP.MIXED).sum() == 1
    assert float(out.loc[out.method == MP.MIXED, 'w1_market'].iloc[0]) == 0.11


# ---------------------------------------------------------------------------
# group composition, which is what would make the stage a null
# ---------------------------------------------------------------------------
def test_group_composition_counts_what_the_rule_moves():
    combos = [['a', 'b', 'c', 'd'], ['a', 'a', 'a', 'a'], ['d', 'd', 'd', 'd']]
    sizes = {'a': 5, 'b': 50, 'c': 500, 'd': 5000}
    got = MP.group_composition(combos, sizes)
    assert list(got.n_above) == [2, 0, 4]
    assert list(got.n_min) == [5, 5, 5000]
    assert list(got.split_group) == [True, False, False]
    assert list(got.all_below) == [False, True, False]
    assert list(got.all_above) == [False, False, True]


def test_provenance_is_stamped_on_a_table():
    frame = pd.DataFrame({'x': [1, 2]})
    out = MP.stamp(frame, corpus_label='corpus_2026-09-25', weight_rho=0.5)
    assert list(out.corpus.unique()) == ['corpus_2026-09-25']
    assert list(out.weight_rho.unique()) == [0.5]
    assert list(out.mixed_threshold.unique()) == [81]


def test_display_labels_never_use_the_retired_vocabulary():
    """"Variable", "Dirichlet shares" and "sampled market shares" are retired
    by the author's decision of 2026-09-25 and must not reach a reader."""
    banned = ('variable', 'dirichlet', 'sampled')
    for name in MP.methods_with_mixed() + [MP.ORACLE]:
        shown = MP.display_method(name).lower()
        assert not any(b in shown for b in banned), shown


# ---------------------------------------------------------------------------
# the paired comparison, and the pivot that makes it paired
# ---------------------------------------------------------------------------
def _fake_truth(n_groups=120, k=4, seed=0, mixed_sd=0.15, fixed_sd=0.20):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for j in range(k):
            for m in MP.methods_with_mixed():
                sd = mixed_sd if m == MP.MIXED else fixed_sd
                rows.append(dict(plca=g, dataset=f'd{j}', method=m,
                                 eci_mean__error=rng.normal(0.0, sd),
                                 eci_mean__truth=1.0))
    return pd.DataFrame(rows)


def test_claim_errors_gives_one_row_per_unit_per_policy():
    got = MP.claim_errors({'recovery': _fake_truth(n_groups=5)})
    assert got.unit.nunique() == 5 * 4
    assert set(got.method) == set(MP.methods_with_mixed())
    assert (got.groupby(['claim', 'unit']).size() == 7).all()


def test_the_oracle_is_a_per_unit_minimum_and_not_a_grand_mean():
    """The defect this pins is real and was in the first draft of the stage: a
    pivot on the frame's own index leaves one value per row, so a row minimum
    returns that value and the 'oracle' comes back as the mean over ALL
    policies -- worse than the best fixed one, which an oracle cannot be.
    """
    errors = MP.claim_errors({'recovery': _fake_truth()})
    got = MP.oracle_ceiling(errors).iloc[0]
    assert got['oracle'] < got['best_fixed']
    assert got['available'] > 0


def test_the_gain_is_paired_so_a_policy_identical_to_a_fixed_one_scores_zero():
    """The control. If the mixed policy IS a fixed policy on every unit, the
    paired difference is exactly zero on every resample and the interval
    closes on it."""
    frame = _fake_truth(n_groups=60)
    twin = frame[frame.method == MP.LARGE_METHOD].copy()
    twin['method'] = MP.MIXED
    frame = pd.concat([frame[frame.method != MP.MIXED], twin],
                      ignore_index=True)
    errors = MP.claim_errors({'recovery': frame})
    got = MP.claim_gain(errors, reference=MP.LARGE_METHOD,
                        rng=np.random.default_rng(2), resamples=200).iloc[0]
    assert got['gain'] == pytest.approx(0.0, abs=1e-12)
    assert got['gain_lo'] == pytest.approx(0.0, abs=1e-12)
    assert got['gain_hi'] == pytest.approx(0.0, abs=1e-12)
    assert not got['beats_reference']


def test_the_gain_finds_a_real_improvement_and_signs_it_correctly():
    errors = MP.claim_errors({'recovery': _fake_truth(mixed_sd=0.10)})
    got = MP.claim_gain(errors, rng=np.random.default_rng(3),
                        resamples=400).iloc[0]
    assert got['gain'] > 0                      # positive means mixed is closer
    assert got['beats_reference']
    assert got['mixed_error'] < got['reference_error']


def test_the_gain_does_not_claim_an_improvement_that_is_not_there():
    errors = MP.claim_errors({'recovery': _fake_truth(mixed_sd=0.20)})
    got = MP.claim_gain(errors, rng=np.random.default_rng(4),
                        resamples=400).iloc[0]
    assert got['gain_lo'] < 0 < got['gain_hi']
    assert not got['beats_reference']


def test_the_reference_defaults_to_the_best_fixed_policy_on_that_claim():
    """Not to an arbitrary one, because the best fixed policy is the comparator
    a reader would otherwise use."""
    frame = _fake_truth()
    tight = frame.method == MP.SMALL_METHOD
    frame.loc[tight, 'eci_mean__error'] *= 0.25
    errors = MP.claim_errors({'recovery': frame})
    got = MP.claim_gain(errors, rng=np.random.default_rng(5),
                        resamples=200).iloc[0]
    assert got['reference'] == MP.SMALL_METHOD


def test_the_paired_bootstrap_agrees_with_the_studys_cluster_bootstrap():
    """Written as a ratio of per-cluster sums for speed; it must still be the
    same estimator `plca.cluster_bootstrap` uses everywhere else."""
    import plca as PL
    rng = np.random.default_rng(7)
    clusters = np.repeat(np.arange(40), 3)
    values = rng.normal(size=(len(clusters), 1))
    lo, hi = MP._paired_boot(values, clusters, np.random.default_rng(11), 4000)
    got = PL.cluster_bootstrap(
        pd.DataFrame({'v': values[:, 0], 'plca': clusters}), 'v',
        resamples=4000, rng=np.random.default_rng(11))
    assert lo == pytest.approx(got['ci_lo'], abs=5e-3)
    assert hi == pytest.approx(got['ci_hi'], abs=5e-3)


# ---------------------------------------------------------------------------
# the composition join, and the id collision that made it wrong
# ---------------------------------------------------------------------------
def test_a_design_pair_is_not_given_a_plca_groups_composition():
    """THE DEFECT THIS PINS WAS IN THE STAGE'S FIRST FULL RUN. The design
    comparison's cluster is a design PAIR from its own resampling and its ids
    run over the same integers the pLCA groups use, so a join on the id alone
    hands every pair an unrelated group's composition. It surfaced as a cell
    in which the rule cannot act -- all four materials above the threshold, so
    the mixed policy IS the fixed one -- reporting a two percent gain.
    """
    recovery = pd.DataFrame({
        'plca': [0, 0, 1, 1], 'dataset': ['a', 'a', 'b', 'b'],
        'method': [MP.MIXED, MP.LARGE_METHOD] * 2,
        'eci_mean__error': [0.1, 0.2, 0.3, 0.4],
        'eci_mean__truth': [1.0] * 4})
    swap = pd.DataFrame({
        'pair': [0, 0, 1, 1], 'saving': [0.0, 0.0, 0.05, 0.05],
        'method': [MP.MIXED, MP.LARGE_METHOD] * 2,
        'discernibility__error': [0.1, 0.2, 0.3, 0.4],
        'discernibility__truth': [0.5] * 4})
    errors = MP.claim_errors({'recovery': recovery, 'swap': swap})
    assert set(errors.cluster_kind) == {'plca', 'pair'}
    comp = MP.group_composition([['a', 'a', 'a', 'a'], ['b', 'b', 'b', 'b']],
                                {'a': 5, 'b': 5000})
    got = MP.attach_composition(errors, comp)
    joined = got[got.n_above.notna()]
    assert set(joined.cluster_kind) == {'plca'}
    assert got[got.cluster_kind == 'pair'].n_above.isna().all()
    assert (got[got.cluster_kind == 'pair'].n_min_band == '').all()


def test_a_group_the_rule_cannot_act_on_shows_exactly_zero_gain():
    """The built-in control on the composition split. Where every material is
    on one side of the threshold the mixed policy IS that fixed policy, so its
    gain against that policy must be exactly zero and the interval must close
    on it. Anything else means the split joined the wrong rows."""
    rng = np.random.default_rng(9)
    rows = []
    for g in range(60):
        for j in range(4):
            e_fixed = rng.normal(0.0, 0.2)
            for m in MP.methods_with_mixed():
                e = e_fixed if m in (MP.MIXED, MP.LARGE_METHOD) \
                    else rng.normal(0.0, 0.2)
                rows.append(dict(plca=g, dataset=f'd{j}', method=m,
                                 eci_mean__error=e, eci_mean__truth=1.0))
    errors = MP.claim_errors({'recovery': pd.DataFrame(rows)})
    comp = MP.group_composition([[f'd{j}' for j in range(4)]] * 60,
                                {f'd{j}': 5000 for j in range(4)})
    got = MP.attach_composition(errors, comp)
    assert (got.n_above == 4).all()
    out = MP.gain_by_group(got, 'n_min_band', reference=MP.LARGE_METHOD,
                           rng=np.random.default_rng(3), resamples=200)
    assert len(out) == 1
    assert float(out.gain.iloc[0]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# the sweep: every candidate is still ONE number, and the range is measured
# ---------------------------------------------------------------------------
def test_every_swept_policy_reads_only_the_size():
    """The variants change WHAT the cutoff switches, never how many numbers a
    reader needs. Each one is still a single threshold on a single input."""
    for p in MP.all_policies():
        assert isinstance(p.threshold, int)
        assert p.choose(p.threshold - 1) == p.below
        assert p.choose(p.threshold) == p.above
        assert p.above in FT.PEWT and p.below in FT.PEWT


def test_the_study_rule_keeps_its_bare_name_inside_the_sweep():
    """So that every table and figure written before the sweep still joins."""
    names = [p.name for p in MP.sweep_policies()]
    assert MP.MIXED in names
    study = next(p for p in MP.sweep_policies() if p.name == MP.MIXED)
    assert study.threshold == MP.MIXED_THRESHOLD
    assert sum(n == MP.MIXED for n in names) == 1


def test_add_policies_points_every_key_at_an_existing_model_object():
    names, _, models, sizes = a_group()
    policies = MP.all_policies()
    models, choice = MP.add_policies(models, sizes, policies)
    for p in policies:
        for d in names:
            assert models[d][p.name] is models[d][choice[p.name][d]]
            assert choice[p.name][d] == p.choose(sizes[d])
    for d in names:
        for m in FT.PEWT:
            assert m in models[d]


def test_the_variant_that_uses_market_weights_below_the_cutoff_differs():
    """The author's question: what if the lognormal below the cutoff uses
    market weights too. It must be a genuinely different policy, not a
    relabelling."""
    names, _, models, sizes = a_group(sizes=(6, 30, 300, 3000))
    MP.add_policies(models, sizes, MP.all_policies())
    u = np.random.default_rng(6).random((NECCS, len(names)))
    study = PL.draw_contributions(models, names, MP.MIXED, u)
    market = PL.draw_contributions(models, names, 'MixedMarket@81', u)
    assert not np.array_equal(study, market)
    # they agree exactly on the materials ABOVE the cutoff and differ below
    for j, d in enumerate(names):
        same = np.array_equal(study[:, j], market[:, j])
        assert same == (sizes[d] >= MP.MIXED_THRESHOLD)


def _sweep_errors(n_groups=200, seed=0, best=81, steepness=0.02,
                  with_fixed=True):
    """A planted curve whose minimum is at a known cutoff.

    The six FIXED methods are planted too, worse than every policy, because
    `claim_gain` and `policy_table` compare against them and a frame without
    them is not the frame the stage builds.
    """
    rng = np.random.default_rng(seed)
    policies = MP.all_policies()
    rows = []
    for g in range(n_groups):
        for j in range(4):
            for p in policies:
                hurt = steepness * abs(np.log(p.threshold / best))
                rows.append(dict(plca=g, dataset=f'd{j}', method=p.name,
                                 eci_mean__error=rng.normal(hurt, 0.05),
                                 eci_mean__truth=1.0))
            if not with_fixed:
                continue
            for k, m in enumerate(FT.PEWT):
                rows.append(dict(plca=g, dataset=f'd{j}', method=m,
                                 eci_mean__error=rng.normal(0.10 + 0.01 * k,
                                                            0.05),
                                 eci_mean__truth=1.0))
    return MP.claim_errors({'recovery': pd.DataFrame(rows)}), policies


def test_the_threshold_curve_finds_a_planted_minimum_and_a_range_around_it():
    errors, policies = _sweep_errors()
    frame, summary = MP.threshold_curve(errors, policies,
                                        rng=np.random.default_rng(1),
                                        resamples=300)
    assert summary['best_threshold'] in (70, 81, 90)
    assert summary['range_lo'] <= summary['best_threshold'] <= summary['range_hi']
    # the range is a RANGE, not the argmin dressed up
    assert summary['range_lo'] < summary['range_hi']
    # and it is a contiguous run of the swept cutoffs
    inside = frame[frame.in_range].threshold.tolist()
    assert inside == sorted(inside)
    assert inside[0] == summary['range_lo'] and inside[-1] == summary['range_hi']


def test_a_flat_curve_gives_a_wider_range_than_a_steep_one():
    """The range has to respond to how much the cutoff actually matters,
    which is the whole reason for publishing one rather than a point."""
    flat, pol = _sweep_errors(seed=3, steepness=0.002)
    steep, _ = _sweep_errors(seed=3, steepness=0.30)
    ff, fs = MP.threshold_curve(flat, pol, rng=np.random.default_rng(5),
                                resamples=300)
    sf, ss = MP.threshold_curve(steep, pol, rng=np.random.default_rng(5),
                                resamples=300)
    assert int(ff.in_range.sum()) > int(sf.in_range.sum())
    assert ss['range_lo'] <= 81 <= ss['range_hi']


def test_the_gain_comparator_is_a_fixed_method_and_never_another_policy():
    """With a sweep in the frame, 'everything except me' would pick the
    neighbouring cutoff as the thing to beat and collapse every gain."""
    errors, policies = _sweep_errors(n_groups=60)
    got = MP.claim_gain(errors, name=MP.MIXED, rng=np.random.default_rng(2),
                        resamples=100)
    assert got.empty or got.reference.isin(FT.PEWT).all()


def test_the_policy_table_ranks_policies_and_marks_the_fixed_ones():
    errors, policies = _sweep_errors(n_groups=80)
    out = MP.policy_table(errors, policies)
    assert set(out.kind) <= {'fixed', 'threshold', 'variant'}
    assert out.pooled_error.is_monotonic_increasing
    assert (out.loc[out.kind == 'threshold', 'threshold'] > 0).all()


def test_pooling_covers_the_design_comparison_and_not_only_the_plca_claims():
    """FIFTEEN of the sixteen claims belong to a pLCA group and the design
    comparison belongs to a design PAIR from its own resampling. A single
    array indexed by pLCA group leaves it as a column of NaN, so a number
    described as 'pooled over sixteen claims' would quietly be over fifteen.
    """
    errors, policies = _sweep_errors(n_groups=40, with_fixed=False)
    swap_rows = []
    rng = np.random.default_rng(8)
    for pair in range(40):
        for sv in (0.0, 0.05):
            for p in policies:
                swap_rows.append(dict(
                    pair=pair, saving=sv, method=p.name,
                    discernibility__error=rng.normal(0.0, 0.05),
                    discernibility__truth=0.5))
    both = pd.concat([errors, MP.claim_errors(
        {'swap': pd.DataFrame(swap_rows)})], ignore_index=True)
    assert set(both.cluster_kind) == {'plca', 'pair'}
    blocks, claims, _, levels = MP.claim_blocks(
        both, [p.name for p in policies])
    assert len(blocks) == 2
    assert len(claims) == 2
    assert np.isfinite(levels).all()
    pooled = MP.pooled_from_blocks(blocks, levels, len(claims))
    assert np.isfinite(pooled).all()
    # and the swap claim really moves the pooled number
    only_plca = MP.pooled_from_blocks(blocks[:1], levels, len(claims))
    assert not np.allclose(pooled, only_plca, equal_nan=True)


def test_the_threshold_curve_reports_the_claims_it_actually_used():
    errors, policies = _sweep_errors(n_groups=60)
    _, summary = MP.threshold_curve(errors, policies,
                                    rng=np.random.default_rng(1),
                                    resamples=150)
    assert summary['n_claims'] == errors['claim'].nunique()

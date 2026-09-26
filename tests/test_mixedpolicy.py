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

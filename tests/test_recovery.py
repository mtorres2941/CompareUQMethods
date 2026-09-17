"""Stage 2c: the evaluation target.

What these pin is the PROPERTY that makes each new column worth having, not the
value it happens to take. A recovery score has to be zero when the model IS the
parent and has to rise as the model moves away from it; a cross-validated score
has to be worse than an in-sample one for the flexible method and not for the
rigid one; a decomposition has to satisfy the inequality it claims; a regret has
to be zero for the winner. Those are the claims the paper makes, so those are
what a test can protect.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import comparison as CMP        # noqa: E402
import components as C          # noqa: E402
import fitting as FT            # noqa: E402
import genconfig as G           # noqa: E402
import generator as GEN         # noqa: E402
import mixture as M             # noqa: E402
import recovery as R            # noqa: E402


def a_parent(seed=11, n=200):
    """One drawn parent and one dataset from it, through the real pipeline."""
    rng = np.random.default_rng(seed)
    for _ in range(40):
        x, w, rec = GEN.generate_dataset(G.DEFAULT, n, rng)
        if x is not None:
            break
    parent, _, x, w, _m = _replay(seed, n)
    return parent, x, w


def _replay(seed, n):
    import corpus
    rng = np.random.default_rng(seed)
    return corpus._replay_one(G.DEFAULT, n, rng)


# ---------------------------------------------------------------- spec round trip
def test_parent_spec_round_trips_exactly():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(25):
        parent, rec = GEN.draw_parent(G.DEFAULT, 200, rng)
        if parent is None:
            continue
        parent.normalizer = 1.7
        q = M.parent_from_spec(parent.spec())
        lo, hi = R.parent_support(parent)
        g = np.linspace(lo, hi, 401)
        for scheme in ('uniform', 'market'):
            assert np.allclose(parent.cdf(g, scheme), q.cdf(g, scheme),
                               rtol=0, atol=1e-13)
            assert np.allclose(parent.pdf(g, scheme), q.pdf(g, scheme),
                               rtol=1e-12, atol=1e-12)
        assert q.normalizer == parent.normalizer
        checked += 1
    assert checked >= 10


def test_spec_survives_json():
    import json
    rng = np.random.default_rng(5)
    parent = None
    while parent is None:
        parent, _ = GEN.draw_parent(G.DEFAULT, 100, rng)
    s = json.loads(json.dumps(parent.spec()))
    q = M.parent_from_spec(s)
    lo, hi = R.parent_support(parent)
    g = np.linspace(lo, hi, 201)
    assert np.allclose(parent.cdf(g), q.cdf(g), rtol=0, atol=1e-13)


def test_an_affine_without_a_family_refuses_to_describe_itself():
    d = C._Affine(C._standard_frozen('johnsonsu', (0.2, 1.1)), 0.0, 1.0)
    with pytest.raises(ValueError):
        d.spec()


def test_the_overlap_displacement_is_not_in_the_record():
    """The reason `rebuild_parents` has to replay rather than read.

    If this ever fails because the record gained the component locations, the
    replay can be replaced by a read and this test should be deleted with it.
    """
    rng = np.random.default_rng(9)
    parent, rec = GEN.draw_parent(G.DEFAULT, 300, rng)
    while parent is None or rec['k'] < 2:
        parent, rec = GEN.draw_parent(G.DEFAULT, 300, rng)
    solved = [C.solve_component(c['skew'], c['exkurt'], mean=0.0, sd=c['sd'])
              for c in rec['components']]
    from_record = np.array([s[2] + rec['shift'] for s in solved])
    actual = np.array([c.spec()['loc'] for c in parent.comps])
    assert not np.allclose(from_record, actual), (
        'the record now determines the component locations; see the docstring')


# ------------------------------------------------------------------ recovery
def test_recovery_is_zero_when_the_model_is_the_parent():
    """A recovery score is a distance, so the truth has to score zero."""
    parent, x, w = a_parent(seed=21, n=300)

    class AsModel:
        def __init__(self, p, scheme):
            self.p, self.scheme = p, scheme

        def cdf(self, g):
            return self.p.cdf(g, self.scheme)

        def pdf(self, g):
            return self.p.pdf(g, self.scheme)

        def ppf(self, q):
            return self.p.ppf(q, self.scheme)

    grid = R.recovery_grid(x, w, parent)
    for scheme in ('uniform', 'market'):
        m = AsModel(parent, scheme)
        assert R.w1_against_parent(m, parent, scheme, grid) < 1e-9
        assert R.tail_charge(m, parent, grid)[0] == 0.0
        assert R.overlap_area(m, parent, scheme, grid) == pytest.approx(1.0,
                                                                        abs=1e-3)
        loc, shape = R.mean_shape_split(m, parent, scheme, grid)
        assert loc < 1e-9 and shape < 1e-9


def test_recovery_rises_as_the_model_moves_away():
    parent, x, w = a_parent(seed=22, n=300)
    grid = R.recovery_grid(x, w, parent)

    class Shifted:
        def __init__(self, p, s):
            self.p, self.s = p, s

        def cdf(self, g):
            return self.p.cdf(np.asarray(g, float) - self.s, 'uniform')

        def pdf(self, g):
            return self.p.pdf(np.asarray(g, float) - self.s, 'uniform')

        def ppf(self, q):
            return self.p.ppf(q, 'uniform') + self.s

    prev = -1.0
    prev_ovl = 2.0
    for s in (0.0, 0.05, 0.1, 0.2):
        d = R.w1_against_parent(Shifted(parent, s), parent, 'uniform', grid)
        o = R.overlap_area(Shifted(parent, s), parent, 'uniform', grid)
        assert d > prev
        assert o < prev_ovl
        prev, prev_ovl = d, o


def test_the_grid_always_covers_the_parent():
    parent, x, w = a_parent(seed=23, n=20)
    grid = R.recovery_grid(x, w, parent)
    assert grid[-1] >= R.parent_support(parent)[1]
    assert grid[0] > 0.0


def test_the_two_weightings_are_scored_against_different_parents():
    """The whole point of the recovery target: a variable-weighted method is
    asked a different question and is scored against a different truth."""
    parent, x, w = a_parent(seed=24, n=400)
    models, _ = FT.fit_pewt(x, w)
    rec = R.score_recovery(models, x, w, parent)
    assert set(rec) == set(FT.PEWT)
    grid = R.recovery_grid(x, w, parent)
    # the two parents differ, so scoring one model against both differs too
    m = models['KDE, Uniform']
    a = R.w1_against_parent(m, parent, 'uniform', grid)
    b = R.w1_against_parent(m, parent, 'market', grid)
    assert a != b


def test_the_tail_charge_catches_what_w1_does_not():
    """Stage 2b, section 4.9: a model can score well on W1 and be unusable
    because it puts mass far past anything the data supports. The body score is
    nearly blind to that; the tail charge is not."""
    parent, x, w = a_parent(seed=25, n=300)
    grid = R.recovery_grid(x, w, parent)
    import families as FAM
    lo, hi = R.parent_support(parent)
    tight = FAM.Truncated(FAM.make_normal(dict(loc=1.0, scale=0.3)).dist)
    wide = FAM.Truncated(FAM.make_normal(dict(loc=1.0, scale=0.3)).dist)

    class Heavy:
        """Same body, a far tail that carries 1 percent of the mass."""
        def __init__(self, base, far, p=0.01):
            self.base, self.far, self.p = base, far, p

        def cdf(self, g):
            return (1 - self.p) * self.base.cdf(g) + self.p * self.far.cdf(g)

        def pdf(self, g):
            return (1 - self.p) * self.base.pdf(g) + self.p * self.far.pdf(g)

        def ppf(self, q):
            q = np.asarray(q, float)
            return np.where(q > 1 - self.p / 2, self.far.ppf(q), self.base.ppf(q))

    far = FAM.Truncated(FAM.make_normal(dict(loc=1e4, scale=1e3)).dist)
    heavy = Heavy(tight, far)
    body_tight = R.w1_against_parent(tight, parent, 'uniform', grid)
    body_heavy = R.w1_against_parent(heavy, parent, 'uniform', grid)
    tail_heavy = R.tail_charge(heavy, parent, grid)[0]
    assert R.tail_charge(wide, parent, grid)[0] < tail_heavy
    # the body scores are close; the tail charge is orders of magnitude apart
    assert abs(body_heavy - body_tight) < tail_heavy / 10


# ------------------------------------------------------------ cross-validation
def test_cv_splits_use_every_value_in_both_roles():
    rng = np.random.default_rng(1)
    splits = R.cv_splits(20, rng, repeats=3)
    assert len(splits) == 6
    for fit, ev in splits:
        assert len(set(fit) & set(ev)) == 0
        assert sorted(np.concatenate([fit, ev])) == list(range(20))
    # both directions of each permutation are present
    for i in range(0, 6, 2):
        assert set(splits[i][0]) == set(splits[i + 1][1])


def test_cv_is_undefined_below_the_minimum():
    rng = np.random.default_rng(2)
    x = np.array([0.8, 1.0, 1.2, 1.1, 0.9])
    w = np.ones(5) / 5
    assert R.cv_w1(x, w, rng) == []


def test_cv_penalizes_the_flexible_method_relative_to_in_sample():
    """The defect the whole stage exists to fix, stated as a test: in sample the
    KDE is scored against the data it was fitted to, so it is rewarded for
    flexibility. Out of sample it is not."""
    rng = np.random.default_rng(4)
    x = np.abs(rng.lognormal(0.0, 0.5, 400)) + 0.05
    w = np.ones_like(x) / len(x)
    models, _ = FT.fit_pewt(x, w)
    grid = FT.score_grid_open(x, w)
    ins = {k: FT.score_w1_model(m, x, w, grid=grid) for k, m in models.items()}
    rows = pd.DataFrame(R.cv_w1(x, w, rng, repeats=4))
    cv = rows.groupby('method').w1_cv.mean()
    kde_penalty = cv['KDE, Uniform'] / ins['KDE, Uniform']
    normal_penalty = cv['Normal, Uniform'] / ins['Normal, Uniform']
    assert kde_penalty > normal_penalty


def test_cv_rows_are_paired_across_methods():
    rng = np.random.default_rng(6)
    x = np.abs(rng.lognormal(0.0, 0.4, 120)) + 0.05
    w = rng.dirichlet(np.ones(120))
    rows = pd.DataFrame(R.cv_w1(x, w, rng, repeats=3))
    per = rows.groupby(['split', 'direction']).method.nunique()
    assert set(per.unique()) == {6}
    assert rows.groupby(['split', 'direction']).n_fit.nunique().max() == 1


def test_cv_summary_reports_the_spread_across_splits():
    rng = np.random.default_rng(8)
    x = np.abs(rng.lognormal(0.0, 0.4, 80)) + 0.05
    w = np.ones_like(x) / len(x)
    rows = pd.DataFrame(R.cv_w1(x, w, rng, repeats=5))
    rows['arm'] = 'test'
    rows['dataset'] = 'd0'
    rows['n'] = len(x)
    s = R.cv_summary(rows)
    assert (s.w1_cv_n_splits == 10).all()
    assert (s.w1_cv_sd_across_splits > 0).all()
    assert np.allclose(s.w1_cv_se,
                       s.w1_cv_sd_across_splits / np.sqrt(s.w1_cv_n_splits))


# ------------------------------------------------------------- decomposition
def test_decomposition_satisfies_its_own_inequality():
    rng = np.random.default_rng(10)
    x = np.abs(rng.lognormal(0.0, 0.5, 150)) + 0.05
    w = rng.dirichlet(np.ones(150))
    models, _ = FT.fit_pewt(x, w)
    d = R.decompose_weighting(models, x, w)
    for label, r in d.items():
        assert r['w1_slack'] >= -1e-9, label
        assert r['w1_own'] <= r['w1_total'] + r['w1_definitional'] + 1e-9


def test_the_definitional_term_is_the_same_for_every_uniform_method_and_zero_otherwise():
    rng = np.random.default_rng(12)
    x = np.abs(rng.lognormal(0.0, 0.5, 150)) + 0.05
    w = rng.dirichlet(np.ones(150))
    models, _ = FT.fit_pewt(x, w)
    d = R.decompose_weighting(models, x, w)
    uni = {r['w1_definitional'] for k, r in d.items() if k.endswith('Uniform')}
    assert len(uni) == 1 and uni.pop() > 0
    assert all(d[k]['w1_definitional'] == 0.0 for k in d if k.endswith('Variable'))
    assert all(d[k]['w1_own'] == d[k]['w1_total']
               for k in d if k.endswith('Variable'))


def test_a_variable_weighted_target_is_what_makes_uniform_look_worse():
    """The circularity, measured. Under the study's target a uniform-weighted
    model is charged a distance no estimation method can remove."""
    rng = np.random.default_rng(14)
    x = np.abs(rng.lognormal(0.0, 0.6, 200)) + 0.05
    w = rng.dirichlet(np.ones(200) * 0.5)
    models, _ = FT.fit_pewt(x, w)
    d = R.decompose_weighting(models, x, w)
    assert d['Normal, Uniform']['w1_definitional'] > 0.01
    assert (d['Normal, Uniform']['w1_total']
            > d['Normal, Uniform']['w1_own'])


# -------------------------------------------------------------------- regret
def test_regret_is_zero_for_the_winner_and_positive_for_the_rest():
    s = pd.DataFrame(dict(
        arm=['a'] * 6, dataset=['d1'] * 3 + ['d2'] * 3,
        method=['m1', 'm2', 'm3'] * 2, n=[10] * 6,
        w1=[1.0, 2.0, 4.0, 5.0, 3.0, 3.5]))
    r = R.add_regret(s)
    assert list(r.w1_regret) == [0.0, 1.0, 3.0, 2.0, 0.0, 0.5]
    rel = R.add_regret_relative(s)
    assert rel.w1_regret_relative.tolist() == pytest.approx(
        [0.0, 1.0, 3.0, 2 / 3, 0.0, 1 / 6])


def test_regret_table_reports_the_upper_tail():
    rng = np.random.default_rng(16)
    s = pd.DataFrame(dict(
        arm='a', dataset=np.repeat([f'd{i}' for i in range(200)], 3),
        method=np.tile(['m1', 'm2', 'm3'], 200), n=30,
        w1=rng.random(600) + 0.1))
    t = R.regret_table(s)
    assert set(t.method) == {'m1', 'm2', 'm3'}
    for _, row in t.iterrows():
        assert row.regret_mean <= row.regret_p90 <= row.regret_p95 <= row.regret_max
        assert 0.0 <= row.regret_zero_share <= 1.0


# --------------------------------------------------------- post-stratification
def test_post_stratification_moves_an_aggregate_toward_the_common_band():
    """A method that is good only on large datasets must lose ground when the
    corpus is reweighted to an empirical mix that is mostly small ones."""
    rows = []
    for band, n, good in (('s1_3_9', 5, 1.0), ('s2_10_99', 50, 1.0),
                          ('s3_100_999', 500, 1.0), ('s4_1000_9999', 5000, 0.0)):
        for i in range(10):
            rows.append(dict(arm='synthetic', dataset=f'{band}_{i}', n=n,
                             method='big_data_method', w1=good))
    s = pd.DataFrame(rows)
    shares = {'s1_3_9': 0.14, 's2_10_99': 0.54, 's3_100_999': 0.27,
              's4_1000_9999': 0.05}
    out = R.post_stratify(s, 'w1', shares)
    assert out.mean_equal_allocation.iloc[0] == pytest.approx(0.75)
    assert out.mean_post_stratified.iloc[0] == pytest.approx(0.95)


def test_empirical_shares_are_measured_not_assumed():
    s = pd.DataFrame(dict(arm='empirical', dataset=[f'd{i}' for i in range(10)],
                          n=[5, 5, 20, 20, 20, 20, 200, 200, 2000, 5],
                          method='m', w1=1.0))
    sh = R.empirical_size_shares(s)
    assert sh['s1_3_9'] == pytest.approx(0.3)
    assert sh['s2_10_99'] == pytest.approx(0.4)
    assert sh['s3_100_999'] == pytest.approx(0.2)
    assert sh['s4_1000_9999'] == pytest.approx(0.1)
    assert sum(sh.values()) == pytest.approx(1.0)


def test_size_band_is_none_above_the_last_band():
    assert R.size_band(3) == 's1_3_9'
    assert R.size_band(9999) == 's4_1000_9999'
    assert R.size_band(20000) is None
    assert R.size_band(2) is None


# ----------------------------------------------------------------- win share
def test_win_share_counts_datasets_and_sums_to_one():
    s = pd.DataFrame(dict(
        arm='a', dataset=np.repeat(['d1', 'd2', 'd3'], 3),
        method=np.tile(['m1', 'm2', 'm3'], 3), n=30,
        w1=[1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 1.0, 5.0, 6.0]))
    out = R.win_share(s)
    assert out.set_index('method').win_share.to_dict() == pytest.approx(
        {'m1': 2 / 3, 'm2': 1 / 3})
    assert out.win_share.sum() == pytest.approx(1.0)


def test_win_share_only_moves_when_the_WINNER_moves():
    """Why the empirical headline is a win share and not a mean rank.

    Stage 2b measured the empirical target's own weight-draw noise at a median
    of 0.1344 against a best-method median W1 of 0.0984, and over five Dirichlet
    realizations `KDE, Variable` and `Lognormal, Variable` swapped places on mean
    rank while the KDE led on win share in every one. The mechanism is pinned
    here rather than the anecdote: a mean rank responds to noise on datasets
    where the method was never going to win, and a win share does not.

    The construction makes that exact. On half the datasets the leader is far
    ahead of everything and wins under any perturbation; on the other half a
    different method is far ahead, so the leader cannot win there but its
    position among the three losers is decided entirely by noise. The win share
    is therefore constant across realizations by construction and the mean rank
    is not.
    """
    rng = np.random.default_rng(20)
    lead = 'KDE, Variable'
    others = ['Lognormal, Variable', 'Normal, Variable', 'Normal, Uniform']
    methods = [lead] + others
    n_each = 40
    ranks, shares = [], []
    for _ in range(12):
        rows = []
        for i in range(n_each):                      # the leader is far ahead
            rows.append(dict(dataset=f'win{i}', method=lead, w1=0.01))
            for m in others:
                rows.append(dict(dataset=f'win{i}', method=m,
                                 w1=float(np.exp(rng.normal(0, 0.5)))))
        for i in range(n_each):                      # someone else is far ahead
            rows.append(dict(dataset=f'lose{i}', method=others[0], w1=0.01))
            for m in [lead] + others[1:]:
                rows.append(dict(dataset=f'lose{i}', method=m,
                                 w1=float(np.exp(rng.normal(0, 0.5)))))
        s = pd.DataFrame(rows).assign(arm='a', n=50)
        ranks.append(float(CMP.add_ranks(s).groupby('method')
                           .w1_rank.mean()[lead]))
        shares.append(float(R.win_share(s).set_index('method')
                            .win_share.get(lead, 0.0)))
    assert np.allclose(shares, 0.5)          # exactly constant, by construction
    assert np.std(ranks) > 0.02              # and the mean rank is not


# ------------------------------------------------------------ paired bootstrap
def test_paired_bootstrap_finds_a_real_gap_and_not_an_imaginary_one():
    rng = np.random.default_rng(31)
    n = 400
    ds = [f'd{i}' for i in range(n)]
    # A per-dataset difficulty of the size this study's scores actually have,
    # orders of magnitude apart, so an UNPAIRED test would see nothing.
    hard = np.exp(rng.normal(0, 1.5, n))
    rows = []
    for i, d in enumerate(ds):
        rows.append(dict(arm='a', dataset=d, method='A', w1=hard[i] * 1.00))
        rows.append(dict(arm='a', dataset=d, method='B', w1=hard[i] * 1.05))
        rows.append(dict(arm='a', dataset=d, method='C', w1=hard[i] * 1.00))
    s = pd.DataFrame(rows)
    out = R.paired_bootstrap(s, 'w1', 'A', rng=np.random.default_rng(0))
    out = out.set_index('method')
    assert out.loc['B', 'distinguishable']          # a real 5 percent gap
    assert out.loc['B', 'mean_difference'] > 0      # A is lower, so better
    assert not out.loc['C', 'distinguishable']      # an identical method
    assert out.loc['C', 'mean_difference'] == pytest.approx(0.0, abs=1e-12)


def test_paired_bootstrap_groups_and_is_reproducible():
    rng = np.random.default_rng(32)
    rows = []
    for band, effect in (('small', -0.05), ('large', 0.05)):
        for i in range(200):
            base = float(np.exp(rng.normal(0, 0.5)))
            rows.append(dict(arm='a', dataset=f'{band}{i}', size_band=band,
                             method='A', w1=base))
            rows.append(dict(arm='a', dataset=f'{band}{i}', size_band=band,
                             method='B', w1=base * (1 + effect)))
    s = pd.DataFrame(rows)
    a = R.paired_bootstrap(s, 'w1', 'A', by=['size_band'],
                           rng=np.random.default_rng(1)).set_index('size_band')
    b = R.paired_bootstrap(s, 'w1', 'A', by=['size_band'],
                           rng=np.random.default_rng(1)).set_index('size_band')
    pd.testing.assert_frame_equal(a, b)
    # the sign of the effect is recovered separately in each band, and an
    # aggregate over both would have canceled them
    assert a.loc['small', 'mean_difference'] < 0
    assert a.loc['large', 'mean_difference'] > 0
    assert bool(a.distinguishable.all())


# --------------------------------------------------------- the quadrature route
def test_both_w1_routes_converge_to_the_same_number():
    """`atoms` and `trapezoid` approximate one integral, so a dense enough grid
    has to make them agree. If this ever fails they are computing different
    things and the choice between them is not a numerical one."""
    rng = np.random.default_rng(41)
    x = np.abs(rng.lognormal(0.0, 0.5, 300)) + 0.05
    w = rng.dirichlet(np.ones(300))
    models, _ = FT.fit_pewt(x, w)
    for label, m in models.items():
        coarse = [FT.score_w1_model(m, x, w, grid=FT.score_grid_open(
            x, w, npoints=n), route=r)
            for n in (50_000,) for r in ('atoms', 'trapezoid')]
        assert coarse[0] == pytest.approx(coarse[1], rel=2e-3), label


def test_the_two_routes_trade_typical_error_against_tail_error():
    """NOT "trapezoid is better", which a first version of this test asserted and
    which is false. The atom route is better TYPICALLY, because the data's
    empirical CDF is a step function and a discrete-to-discrete distance handles
    it exactly; the trapezoid route is better in the TAIL, where one grid cell
    spans a large change in the model's CDF. The real finding is that 1,000
    points is not converged."""
    rng = np.random.default_rng(43)
    err = {'atoms': [], 'trapezoid': []}
    for _ in range(25):
        x = np.abs(rng.lognormal(0.0, 0.7, 120)) + 0.05
        w = rng.dirichlet(np.ones(120))
        models, _ = FT.fit_pewt(x, w)
        m = models['KDE, Variable']
        ref = FT.score_w1_exact(m, x, w)
        for r in err:
            err[r].append(abs(FT.score_w1_model(m, x, w, route=r) - ref) / ref)
    a, t = np.array(err['atoms']), np.array(err['trapezoid'])
    assert np.median(a) < np.median(t)          # atoms wins typically
    # The trapezoid route's advantage is in the TAIL and it needs a dataset
    # spanning orders of magnitude to appear, which the well-behaved sample here
    # does not. It is measured on the real arms instead, where the atom route's
    # p99 relative error is 0.136 against 0.023 and its worst case 1.385 against
    # 0.200: `audits/scoring_grid_error.py`. Not asserted here, because a test
    # that needs its input tuned to make the effect show is not pinning anything.
    assert np.median(a) < 0.01 and np.median(t) < 0.01


def test_the_scoring_settings_are_the_decided_ones():
    """A guard, not a preference. These three move every reported number in the
    paper, so none of them may change as a side effect of another edit. Author
    decisions of 2026-09-16: trapezoid quadrature on 20,000 points, and a
    bandwidth guard at 20 effective observations."""
    import customstats as CS
    assert FT.W1_ROUTE == 'trapezoid'
    assert FT.SCORE_GRID_POINTS == 20_000
    assert CS.SILVERMAN_MIN_NEFF == 20.0

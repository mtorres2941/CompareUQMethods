"""Stage 2e: the pLCA construction.

What these pin are the three properties the stage's numbers rest on. That
common random numbers make a method identical to itself and leave each method's
own marginal distribution untouched, so installing them is a refinement and not
a change of estimand. That the intensity sweep's control -- an infinite
Dirichlet concentration -- reproduces the equal-intensity case EXACTLY, so the
sweep is anchored to the study's own construction rather than approaching it.
And that the outputs computed here are the notebook's own definitions, because
a sweep measured with different arithmetic from the table it is compared
against would be measuring the arithmetic.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import components as CP         # noqa: E402
import fitting as FT            # noqa: E402
import mixture as MX            # noqa: E402
import plca as PL               # noqa: E402

NECCS = 2_000


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
# common random numbers
# ---------------------------------------------------------------------------
def test_common_variates_make_a_method_identical_to_itself():
    """The control. Two runs of one method on one uniform block agree exactly."""
    names, _, models = a_group()
    rng = np.random.default_rng(1)
    u = rng.random((NECCS, len(names)))
    a = PL.draw_contributions(models, names, 'KDE, Uniform', u)
    b = PL.draw_contributions(models, names, 'KDE, Uniform', u)
    assert np.array_equal(a, b)


def test_independent_variates_do_not():
    """Which is what the control is worth something against."""
    names, _, models = a_group()
    rng = np.random.default_rng(1)
    a = PL.draw_contributions(models, names, 'KDE, Uniform',
                              rng.random((NECCS, len(names))))
    b = PL.draw_contributions(models, names, 'KDE, Uniform',
                              rng.random((NECCS, len(names))))
    assert not np.allclose(a, b)


def test_sharing_variates_does_not_change_a_methods_own_distribution():
    """CRN is a refinement, not a change of estimand.

    Pushing one uniform block through six models gives each model the same
    marginal sample it would have had from its own block: a uniform is a
    uniform. So no per-method result changes in distribution, and only the
    PAIRING between methods does. This checks it on the quantiles rather than
    asserting it.
    """
    names, _, models = a_group()
    rng = np.random.default_rng(2)
    shared = rng.random((20_000, len(names)))
    own = rng.random((20_000, len(names)))
    q = np.linspace(0.02, 0.98, 25)
    for m in ('Normal, Uniform', 'Lognormal, Variable', 'KDE, Variable'):
        a = PL.draw_contributions(models, names, m, shared)[:, 0]
        b = PL.draw_contributions(models, names, m, own)[:, 0]
        assert np.allclose(np.quantile(a, q), np.quantile(b, q), rtol=0.05)


def test_materials_are_independent_within_an_iteration():
    """Columns of the uniform block must not induce rank correlation.

    Sharing ONE variate across materials would make every material's draw move
    together, which is a modelling claim this study does not make. The variates
    are shared across METHODS and independent across MATERIALS.
    """
    names, _, models = a_group(k=3)
    rng = np.random.default_rng(3)
    d = PL.draw_contributions(models, names, 'Normal, Uniform',
                              rng.random((20_000, len(names))))
    c = np.corrcoef(d, rowvar=False)
    off = c[~np.eye(len(names), dtype=bool)]
    assert np.abs(off).max() < 0.05


# ---------------------------------------------------------------------------
# the outputs are the notebook's own
# ---------------------------------------------------------------------------
def test_rank_1_frequency_matches_the_notebooks_pandas_ranking():
    rng = np.random.default_rng(4)
    draws = rng.lognormal(0.0, 0.4, size=(5_000, 4))
    got = PL.outputs(draws)['eci_rank_1']
    df = pd.DataFrame(draws)
    want = (df.rank(axis=1, ascending=False) == 1).mean().to_numpy()
    assert np.allclose(got, want)


def test_uncertainty_index_matches_its_definition():
    rng = np.random.default_rng(5)
    draws = rng.lognormal(0.0, 0.4, size=(4_000, 3))
    total = draws.sum(axis=1)
    med = np.median(draws, axis=0)
    want = np.array([1.0 - np.var(total - draws[:, j] + med[j]) / np.var(total)
                     for j in range(3)])
    assert np.allclose(PL.outputs(draws)['ui'], want)


def test_nrmse_matches_the_notebooks_arithmetic():
    """The notebook computes it inside a plotting function; this is that."""
    rng = np.random.default_rng(6)
    rows = []
    for g in range(12):
        for d in range(4):
            base = rng.normal(1.0, 0.3)
            for m in ('A', 'B', 'C'):
                rows.append(dict(plca=g, dataset=f'{g}_{d}', method=m,
                                 v=base + rng.normal(0, 0.05)))
    frame = pd.DataFrame(rows)
    wide = frame.pivot_table(index=['plca', 'dataset'], columns='method',
                             values='v')
    se = []
    for a in wide.columns:
        for b in wide.columns:
            if a != b:
                se.append(((wide[a] - wide[b]) ** 2).to_numpy())
    want = np.mean(np.concatenate(se)) ** 0.5 / np.std(wide.to_numpy())
    assert PL.nrmse(frame, 'v') == pytest.approx(want)


# ---------------------------------------------------------------------------
# material use intensity
# ---------------------------------------------------------------------------
def test_infinite_concentration_is_exactly_the_equal_case():
    """The sweep's anchor. Not close to the study's construction: identical."""
    rng = np.random.default_rng(7)
    for k in (2, 4, 12):
        assert np.array_equal(PL.mui_dirichlet(k, np.inf, rng), np.ones(k))


def test_equal_intensity_leaves_the_pLCA_untouched():
    """Passing the equal vector must change no draw, or the sweep's own
    baseline would differ from the study's results for no reason."""
    names, _, models = a_group()
    rng = np.random.default_rng(8)
    u = rng.random((NECCS, len(names)))
    a = PL.draw_contributions(models, names, 'KDE, Variable', u)
    b = PL.draw_contributions(models, names, 'KDE, Variable', u,
                              PL.mui_equal(len(names)))
    assert np.array_equal(a, b)


def test_intensity_vectors_average_to_one():
    """So a swept case is comparable with the equal case rather than comparable
    up to a building-sized constant."""
    rng = np.random.default_rng(9)
    for k in (2, 4, 8):
        for c in (200.0, 3.0, 0.4):
            assert PL.mui_dirichlet(k, c, rng).mean() == pytest.approx(1.0)
        for r in PL.MUI_RATIOS:
            assert PL.mui_from_ratio(k, r).mean() == pytest.approx(1.0)


def test_lower_concentration_is_more_concentrated():
    """The sweep's direction, measured rather than assumed."""
    rng = np.random.default_rng(10)
    got = [np.median([PL.contribution_profile(PL.mui_dirichlet(4, c, rng))
                      ['top_share'] for _ in range(400)])
           for c in (200.0, 20.0, 3.0, 0.4)]
    assert got == sorted(got)


def test_the_profile_is_a_property_of_the_design_not_of_the_method():
    """`top2_ratio` is computed from intensities and dataset means, both fixed
    before any distribution is fitted, so it cannot move with the method."""
    p = PL.contribution_profile(PL.mui_from_ratio(4, 10.0))
    v = PL.mui_from_ratio(4, 10.0)
    assert p['top2_ratio'] == pytest.approx(v[0] / v[1])
    assert p['top_share'] == pytest.approx(v[0] / v.sum())
    assert PL.contribution_profile(PL.mui_equal(4))['top2_ratio'] == 1.0


def test_concentration_makes_the_top_contributor_stop_moving():
    """The mechanism the whole intensity sweep is about: once one material
    leads by enough, no difference between fitted models can reorder them."""
    names, _, models = a_group(seed=11)
    rng = np.random.default_rng(11)
    u = rng.random((NECCS, len(names)))
    flips = {}
    for r in (1.0, 100.0):
        per = PL.run_group(models, names, u, PL.mui_from_ratio(len(names), r))
        tops = {PL.top_contributor(per[m], names) for m in FT.PEWT}
        flips[r] = len(tops)
    assert flips[100.0] == 1
    assert flips[100.0] <= flips[1.0]


# ---------------------------------------------------------------------------
# groupings
# ---------------------------------------------------------------------------
def test_resampled_groups_hold_distinct_datasets():
    rng = np.random.default_rng(12)
    g = PL.resample_groups([f'd{i}' for i in range(50)], 6, 200, rng)
    assert g.shape == (200, 6)
    assert all(len(set(row)) == 6 for row in g)


def test_resampled_groups_are_drawn_with_replacement_across_groups():
    """Which is what gives a bootstrap over groupings; a single partition
    cannot say how much of a headline is the grouping."""
    rng = np.random.default_rng(13)
    g = PL.resample_groups([f'd{i}' for i in range(20)], 4, 50, rng)
    assert len(set(g.ravel())) <= 20 and g.size == 200


def test_a_group_cannot_be_larger_than_the_pool():
    with pytest.raises(ValueError):
        PL.resample_groups(['a', 'b'], 4, 3, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# the bootstrap resamples clusters
# ---------------------------------------------------------------------------
def test_cluster_bootstrap_is_wider_than_a_row_bootstrap():
    """The fifteen pairs inside a pLCA are not fifteen independent
    observations, and an interval that treats them as such is too narrow."""
    rng = np.random.default_rng(14)
    rows = []
    for g in range(60):
        level = rng.normal(0.0, 1.0)
        for _ in range(15):
            rows.append(dict(plca=g, v=level + rng.normal(0, 0.05)))
    frame = pd.DataFrame(rows)
    clustered = PL.cluster_bootstrap(frame, 'v', cluster='plca',
                                     resamples=400, rng=np.random.default_rng(0))
    frame = frame.assign(row=np.arange(len(frame)))
    row_wise = PL.cluster_bootstrap(frame, 'v', cluster='row',
                                    resamples=400, rng=np.random.default_rng(0))
    assert ((clustered['ci_hi'] - clustered['ci_lo'])
            > 2 * (row_wise['ci_hi'] - row_wise['ci_lo']))


def test_cluster_bootstrap_covers_its_own_point_estimate():
    rng = np.random.default_rng(15)
    frame = pd.DataFrame(dict(plca=np.repeat(np.arange(80), 4),
                              v=rng.normal(0.3, 0.1, 320)))
    got = PL.cluster_bootstrap(frame, 'v', resamples=400,
                               rng=np.random.default_rng(1))
    assert got['ci_lo'] < got['statistic'] < got['ci_hi']
    assert got['n_clusters'] == 80 and got['n'] == 320


# ---------------------------------------------------------------------------
# the truth
# ---------------------------------------------------------------------------
def a_parent(seed=0):
    """A two-component truncated mixture, built the way the generator builds one."""
    rng = np.random.default_rng(seed)
    comps = []
    for skew, exkurt, mean, sd in ((0.6, 0.6, 1.0, 0.30), (0.9, 1.2, 1.8, 0.50)):
        fam, shape, loc, scale, status = CP.solve_component(
            skew, exkurt, mean=mean, sd=sd)
        assert status == 'ok', status
        comps.append(CP.frozen(fam, shape, loc, scale))
    pi = np.array([0.6, 0.4])
    lo, hi = MX.population_truncation_bounds(comps, pi)
    market = rng.dirichlet(np.ones(2))
    return MX.MixtureParent(comps, pi, market, lo, hi, 1.0)


def test_parent_sampler_inverts_the_parents_own_cdf():
    """The tabulation is a speed device and must not be a second model."""
    p = a_parent()
    s = PL.ParentSampler(p, scheme='market')
    q = np.linspace(0.001, 0.999, 60)
    exact = np.asarray(p.ppf(q, 'market'), dtype=float)
    assert np.max(np.abs(s.ppf(q) - exact)) < 1e-3 * (s.hi - s.lo)


def test_parent_sampler_reproduces_the_parent_cdf():
    p = a_parent(1)
    s = PL.ParentSampler(p, scheme='market')
    x = np.linspace(s.lo, s.hi, 200)
    assert np.max(np.abs(s.cdf(x) - p.cdf(x, 'market'))) < 1e-6


def test_parent_sampler_stays_inside_the_parents_support():
    p = a_parent(2)
    s = PL.ParentSampler(p)
    u = np.random.default_rng(0).random(5_000)
    d = s.rvs_from_uniform(u)
    assert d.min() >= s.lo - 1e-12 and d.max() <= s.hi + 1e-12


def test_the_two_truth_schemes_are_different_populations():
    """Scoring against the market parent and against the sampling parent are
    different questions, and the sweep reports both."""
    p = a_parent(3)
    a = PL.ParentSampler(p, scheme='market')
    b = PL.ParentSampler(p, scheme='uniform')
    q = np.linspace(0.05, 0.95, 20)
    assert not np.allclose(a.ppf(q), b.ppf(q))


def test_truth_error_is_zero_when_the_model_is_the_parent():
    """The control for the whole truth experiment: a method that IS the truth
    must have exactly zero error, which only holds under common variates."""
    p = a_parent(4)
    s = PL.ParentSampler(p)
    names = ['d0', 'd1']
    models = {d: {'M': s} for d in names}
    u = np.random.default_rng(0).random((NECCS, 2))
    rows = PL.truth_rows(models, {d: s for d in names}, names, u,
                         methods=['M'])
    assert all(abs(r['eci_rank_1__error']) == 0.0 for r in rows)
    assert all(abs(r['eci_mean__error']) == 0.0 for r in rows)


def test_truth_error_grows_as_a_model_leaves_the_parent():
    p = a_parent(5)
    s = PL.ParentSampler(p)
    names = ['d0', 'd1']

    class Shifted:
        def __init__(self, base, factor):
            self.base, self.factor = base, factor

        def rvs_from_uniform(self, u):
            return self.base.rvs_from_uniform(u) * self.factor

    models = {d: {'near': Shifted(s, 1.02), 'far': Shifted(s, 1.5)}
              for d in names}
    u = np.random.default_rng(1).random((NECCS, 2))
    rows = pd.DataFrame(PL.truth_rows(models, {d: s for d in names}, names, u,
                                      methods=['near', 'far']))
    err = rows.groupby('method')['eci_mean__error'].apply(
        lambda v: v.abs().mean())
    assert err['far'] > err['near']


# ---------------------------------------------------------------------------
# the capped-reduction strategy under shared variates
# ---------------------------------------------------------------------------
def test_cap_reduction_leaves_every_draw_below_the_cap():
    names, _, models = a_group(seed=16)
    rng = np.random.default_rng(16)
    u = rng.random((NECCS, len(names)))
    col = PL.draw_contributions(models, names, 'KDE, Uniform', u)[:, 0]
    passes = PL.PassUniforms(rng, NECCS, len(names))
    red, touched, cap = PL.cap_reduction(models[names[0]]['KDE, Uniform'], col,
                                         0.75, passes.for_material(0))
    assert red.max() < cap
    # Every draw at or above the cap was redrawn, and nothing else was.
    assert np.array_equal(touched, col >= cap)
    assert np.array_equal(red[~touched], col[~touched])


def test_pass_uniforms_give_two_methods_the_same_variate():
    """Which is what makes the capped strategy paired across methods too."""
    rng = np.random.default_rng(17)
    passes = PL.PassUniforms(rng, 50, 4)
    first = passes(2, 0).copy()
    assert np.array_equal(passes(2, 0), first)
    assert not np.array_equal(passes(2, 1), first)
    assert not np.array_equal(passes(3, 0), first)


# ---------------------------------------------------------------------------
# the pair table
# ---------------------------------------------------------------------------
def test_pair_rows_report_no_change_for_a_method_against_itself():
    names, _, models = a_group(seed=18)
    rng = np.random.default_rng(18)
    u = rng.random((NECCS, len(names)))
    per = PL.run_group(models, names, u)
    per['copy'] = per['KDE, Uniform']
    rows = PL.pair_rows(per, names, methods=['KDE, Uniform', 'copy'])
    assert len(rows) == 1
    assert rows[0]['flip_top'] is False
    assert rows[0]['eci_mean_max'] == 0.0


def test_pair_rows_cover_every_pair():
    names, _, models = a_group(seed=19)
    rng = np.random.default_rng(19)
    per = PL.run_group(models, names, rng.random((NECCS, len(names))))
    assert len(PL.pair_rows(per, names)) == 15


# ---------------------------------------------------------------------------
# the committed pLCA table is a full run
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, 'outputs', 'tables', 'TABLE_PLCAResults.csv')
RUNMETA = os.path.join(ROOT, 'outputs', 'tables', 'TABLE_PLCAResults_runmeta.json')


@pytest.mark.skipif(not os.path.exists(RUNMETA), reason='no pLCA table on disk')
def test_the_committed_plca_table_is_not_a_smoke_run():
    """The second half of the smoke guard, and the half that catches a table
    that got in by some other route than notebook 3's own writer.

    In Stage 2d a smoke run reached a commit and replaced the 60,000-row
    results table with a 960-row one. It was caught by eye. This asserts the
    group count in the run metadata, checks that the table on disk has the
    number of rows that metadata implies, and refuses a table whose metadata
    says it came from a smoke run.
    """
    import json
    meta = json.load(open(RUNMETA))
    assert not meta.get('smoke', False), 'the committed pLCA table is a smoke run'
    assert meta['n_combos'] >= 2_000, (
        f"only {meta['n_combos']} pLCA groups: this is a smoke or truncated run")
    assert meta['n_rows'] == meta['n_combos'] * meta['n_pewt'] * meta['nmats']
    with open(TABLE) as fh:
        lines = sum(1 for _ in fh)
    assert lines - 1 == meta['n_rows'], (
        f"{TABLE} holds {lines - 1} rows, metadata says {meta['n_rows']}")


# ---------------------------------------------------------------------------
# the sweep, the summary and the truth tables
# ---------------------------------------------------------------------------
def test_sweep_covers_every_cell_and_carries_its_observables():
    names, data, models = a_group(seed=20, n=40, k=8)
    rng = np.random.default_rng(20)
    f = PL.sweep(models, names, rng, sizes=(2, 4), neccs=500, n_groups=3,
                 cases=[('1:1', 'ratio', 1.0), ('10:1', 'ratio', 10.0)])
    assert set(f.nmats) == {2, 4}
    assert set(f.mui_case) == {'1:1', '10:1'}
    # 2 sizes x 2 cases x 3 groups x 15 pairs
    assert len(f) == 2 * 2 * 3 * 15
    assert f[f.mui_case == '1:1'].top2_ratio.eq(1.0).all()
    assert f[f.mui_case == '10:1'].top2_ratio.eq(10.0).all()
    assert np.allclose(f.inv_top2_ratio, 1.0 / f.top2_ratio)


def test_sweep_groups_are_distinct_clusters_per_cell():
    """The bootstrap resamples pLCA groups, so two cells must not share a
    cluster label or a resample would pool them."""
    names, _, models = a_group(seed=21, n=40, k=6)
    rng = np.random.default_rng(21)
    f = PL.sweep(models, names, rng, sizes=(2, 3), neccs=400, n_groups=2,
                 cases=[('1:1', 'ratio', 1.0)])
    per_cell = f.groupby('nmats').plca.nunique()
    assert (per_cell == 2).all()
    assert f.plca.nunique() == 4


def test_sweep_summary_reports_an_interval_on_every_headline():
    names, _, models = a_group(seed=22, n=40, k=6)
    rng = np.random.default_rng(22)
    f = PL.sweep(models, names, rng, sizes=(3,), neccs=400, n_groups=8,
                 cases=[('1:1', 'ratio', 1.0)])
    s = PL.sweep_summary(f, resamples=100, rng=np.random.default_rng(0))
    assert len(s) == 1
    row = s.iloc[0]
    assert row.flip_top_lo <= row.flip_top <= row.flip_top_hi
    assert row.eci_mean_max_lo <= row.eci_mean_max <= row.eci_mean_max_hi


def test_nrmse_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(23)
    rows = []
    for g in range(40):
        for d in range(4):
            base = rng.normal(1.0, 0.3)
            for m in ('A', 'B', 'C'):
                rows.append(dict(plca=g, dataset=f'{g}_{d}', method=m,
                                 v=base + rng.normal(0, 0.05)))
    frame = pd.DataFrame(rows)
    got = PL.nrmse_ci(frame, 'v', resamples=200, rng=np.random.default_rng(0))
    assert got['ci_lo'] < got['nrmse'] < got['ci_hi']
    assert got['n_clusters'] == 40


def test_nrmse_bootstrap_keeps_repeated_groups_apart():
    """A resampled group can be drawn twice, and averaging the two copies
    together would shrink the very variation the bootstrap is measuring."""
    rng = np.random.default_rng(24)
    rows = []
    for g in range(15):
        for d in range(4):
            for m in ('A', 'B'):
                rows.append(dict(plca=g, dataset=f'{g}_{d}', method=m,
                                 v=rng.normal(1.0, 0.3)))
    frame = pd.DataFrame(rows)
    got = PL.nrmse_ci(frame, 'v', resamples=200, rng=np.random.default_rng(1))
    assert got['ci_hi'] > got['ci_lo'] > 0


def test_truth_run_and_summary_round_trip():
    p = a_parent(6)
    s = PL.ParentSampler(p)
    names = ['d0', 'd1', 'd2', 'd3']
    samplers = {d: s for d in names}

    class Shifted:
        def __init__(self, base, factor):
            self.base, self.factor = base, factor

        def rvs_from_uniform(self, u):
            return self.base.rvs_from_uniform(u) * self.factor

    models = {d: {'exact': s, 'off': Shifted(s, 1.3)} for d in names}
    combos = np.array([names, names[::-1]])
    frame = PL.truth_run(models, samplers, combos, np.random.default_rng(0),
                         neccs=1_000, methods=['exact', 'off'],
                         sizes_of={d: 50 for d in names})
    assert len(frame) == 2 * 4 * 2
    summ = PL.truth_summary(frame, resamples=50,
                            rng=np.random.default_rng(0))
    exact = summ[summ.method == 'exact'].iloc[0]
    off = summ[summ.method == 'off'].iloc[0]
    assert exact.eci_mean_abs_error == 0.0
    assert off.eci_mean_abs_error > 0
    assert exact.names_true_top == 1.0
    wins = PL.truth_win_share(frame)
    assert wins[wins.method == 'exact'].win_share.iloc[0] >= 0.5

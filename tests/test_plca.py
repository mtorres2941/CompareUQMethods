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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    together, which is a modeling claim this study does not make. The variates
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
    names, data, models = a_group(seed=16)
    rng = np.random.default_rng(16)
    u = rng.random((NECCS, len(names)))
    col = PL.draw_contributions(models, names, 'KDE, Uniform', u)[:, 0]
    cap = PL.specification_cap(data[names[0]]['data'])
    red, touched, got = PL.cap_reduction(models[names[0]]['KDE, Uniform'], col,
                                         cap, rng.random(NECCS))
    assert got == cap
    assert np.nanmax(red) < cap
    # Every draw at or above the cap was replaced, and nothing else was.
    assert np.array_equal(touched, col >= cap)
    assert np.array_equal(red[~touched], col[~touched])


def test_the_capped_draw_is_the_model_conditioned_on_being_below_the_cap():
    """Exact, by inverse CDF. Redrawing until a value lands below the cap gives
    the same distribution, which is what makes the one-step form a replacement
    and not a different intervention."""
    names, data, models = a_group(seed=43, n=120)
    model = models[names[0]]['Lognormal, Uniform']
    cap = PL.specification_cap(data[names[0]]['data'])
    rng = np.random.default_rng(43)
    col = np.asarray(model.rvs(40_000, random_state=rng), dtype=float)
    exact, _, _ = PL.cap_reduction(model, col, cap, rng.random(40_000))
    # the rejection version, written out here so the two can be compared
    loop = col.copy()
    above = loop >= cap
    while above.any():
        loop[above] = np.asarray(model.rvs(int(above.sum()), random_state=rng),
                                 dtype=float)
        above = loop >= cap
    q = np.linspace(0.02, 0.98, 25)
    assert np.allclose(np.quantile(exact, q), np.quantile(loop, q), rtol=0.03)


def test_a_cap_below_the_models_whole_support_is_reported_not_looped():
    """The failure a bounded redraw loop cannot report: with no mass below the
    cap there is nothing to redraw, and a loop returns values still above it."""

    class AllHigh:
        def cdf(self, x):
            return np.zeros(np.shape(x))

        def ppf(self, q):
            return np.full(np.shape(q), 99.0)

    col = np.full(100, 50.0)
    red, touched, _ = PL.cap_reduction(AllHigh(), col, 10.0,
                                       np.random.default_rng(0).random(100))
    assert touched.all()
    assert np.isnan(red).all()


def test_a_reduction_statement_survives_an_impossible_cap():
    base = np.full(50, 4.0)
    got = PL.reduction_statement(base, np.full(50, np.nan), prefix='cap_')
    assert np.isnan(got['cap_reduction_mean'])
    assert np.isnan(got['cap_p_reduction_over_5'])


def test_the_cap_is_paired_across_methods_by_one_uniform_block():
    """Two methods capping the same iteration use the same variate, which is
    what the pass cache used to do and what one block now does."""
    names, data, models = a_group(seed=44)
    rng = np.random.default_rng(44)
    u = rng.random((NECCS, len(names)))
    u_cap = rng.random(NECCS)
    cap = PL.specification_cap(data[names[0]]['data'])
    out = {}
    for m in ('KDE, Uniform', 'KDE, Variable'):
        col = PL.draw_contributions(models, names, m, u)[:, 0]
        out[m] = PL.cap_reduction(models[names[0]][m], col, cap, u_cap)[0]
    # different models, so different values, but the same variate drives both
    again = PL.cap_reduction(models[names[0]]['KDE, Uniform'],
                             PL.draw_contributions(models, names,
                                                   'KDE, Uniform', u)[:, 0],
                             cap, u_cap)[0]
    assert np.array_equal(out['KDE, Uniform'], again)


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


def test_lazy_samplers_bound_their_memory_and_agree_with_eager_ones():
    """A ParentSampler is about half a megabyte and the corpus has nearly ten
    thousand parents, so the truth run builds them four at a time."""
    parents = {f'd{i}': a_parent(i) for i in range(6)}
    lazy = PL.LazySamplers(parents, maxsize=2)
    eager = PL.parent_samplers(parents)
    q = np.linspace(0.05, 0.95, 20)
    for d in parents:
        assert np.allclose(lazy[d].ppf(q), eager[d].ppf(q))
    assert len(lazy._cache) <= 2
    assert len(lazy) == 6 and 'd3' in lazy


# ---------------------------------------------------------------------------
# the five statements a probabilistic LCA makes
# ---------------------------------------------------------------------------
def test_w1_between_samples_matches_scipy():
    """The output-side distance is the study's own criterion and nothing new."""
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(30)
    a, b = rng.normal(0, 1, 3_000), rng.normal(0.4, 1.2, 4_000)
    assert PL.w1_samples(a, b) == pytest.approx(wasserstein_distance(a, b))


def test_both_distances_are_zero_for_a_sample_against_itself():
    rng = np.random.default_rng(31)
    a = rng.normal(size=500)
    assert PL.w1_samples(a, a) == 0.0
    assert PL.cramer_distance(a, a) == 0.0


def test_cramer_is_the_l2_sibling_and_orders_the_same_way_here():
    """Reported beside W1 as the robustness check, so what matters is that it
    agrees about which of two candidates is closer to the truth."""
    rng = np.random.default_rng(32)
    truth = rng.normal(0, 1, 5_000)
    near, far = rng.normal(0.1, 1, 5_000), rng.normal(0.8, 1, 5_000)
    assert PL.w1_samples(near, truth) < PL.w1_samples(far, truth)
    assert PL.cramer_distance(near, truth) < PL.cramer_distance(far, truth)


def test_the_compliance_statement_is_read_at_the_truths_own_quantile():
    """A threshold at the truth's q is met by the truth with probability q, so
    a method that IS the truth must report exactly zero error."""
    rng = np.random.default_rng(33)
    t = rng.lognormal(0, 0.3, 20_000)
    got = PL.building_statement(t, t)
    for q in PL.COMPLIANCE_QUANTILES:
        assert got[f'p_below_q{q:g}__truth'] == q
        assert abs(got[f'p_below_q{q:g}__error']) < 1e-12
    assert got['total_w1'] == 0.0


def test_a_method_that_understates_the_tail_overstates_compliance():
    """Which is the whole point of the statement: a thin-tailed model tells a
    practitioner they are likelier to meet a budget than they are."""
    rng = np.random.default_rng(34)
    truth = rng.lognormal(0, 0.6, 20_000)
    thin = rng.lognormal(0, 0.3, 20_000) * np.exp(0.6 ** 2 / 2 - 0.3 ** 2 / 2)
    got = PL.building_statement(thin, truth)
    assert got['p_below_q0.9__error'] > 0
    assert got['total_q0.9__error'] < 0


# ---------------------------------------------------------------------------
# the interventions
# ---------------------------------------------------------------------------
def test_the_cap_is_a_property_of_the_data_not_of_the_method():
    """The defect this replaced: a cap taken from each method's own draws gave
    the six methods six different interventions."""
    rng = np.random.default_rng(35)
    x = rng.lognormal(0, 0.5, 400)
    assert PL.specification_cap(x) == pytest.approx(np.quantile(x, 0.75))
    assert PL.specification_cap(x, scale=2.0) == pytest.approx(
        2.0 * np.quantile(x, 0.75))


def test_the_share_capped_is_free_to_differ_between_methods():
    """Under the old form it was 25 percent for every method by construction,
    which forced the signal to zero. A method that puts more mass above the cap
    must now be able to say so."""
    names, data, models = a_group(seed=36, n=80)
    rng = np.random.default_rng(36)
    u = rng.random((NECCS, len(names)))
    cap = PL.specification_cap(data[names[0]]['data'])
    shares = []
    u_cap = rng.random(NECCS)
    for m in ('Normal, Uniform', 'Lognormal, Uniform', 'KDE, Uniform'):
        col = PL.draw_contributions(models, names, m, u)[:, 0]
        _, touched, _ = PL.cap_reduction(models[names[0]][m], col, cap, u_cap)
        shares.append(float(touched.mean()))
    assert len(set(np.round(shares, 3))) > 1
    assert all(0.0 < s < 1.0 for s in shares)


def test_a_reduction_statement_carries_its_confidence():
    """The half the study threw away: not the mean saving but the chance of
    achieving at least a stated one."""
    rng = np.random.default_rng(37)
    base = rng.lognormal(1.0, 0.3, 20_000)
    got = PL.reduction_statement(base, base * 0.9)
    assert got['reduction_mean'] == pytest.approx(0.1)
    assert got['p_reduction_over_5'] == 1.0
    assert got['p_reduction_over_20'] == 0.0


def test_a_reduction_that_only_sometimes_lands_reports_a_middling_chance():
    rng = np.random.default_rng(38)
    base = rng.lognormal(1.0, 0.3, 20_000)
    factor = rng.uniform(0.8, 1.0, 20_000)
    got = PL.reduction_statement(base, base * factor)
    assert 0.2 < got['p_reduction_over_10'] < 0.8


# ---------------------------------------------------------------------------
# the design swap
# ---------------------------------------------------------------------------
def test_the_two_options_share_variates_for_the_materials_they_share():
    """Dependent sampling. Without it the comparison carries a sampling
    difference that has nothing to do with the design."""
    names, _, models = a_group(seed=39, n=60, k=5)
    rng = np.random.default_rng(39)
    u = rng.random((NECCS, 5))
    a, b = PL.swap_totals(models, names[:3], names[3], names[3], u,
                          'KDE, Uniform', saving=0.0, n_materials=4)
    # same alternative on BOTH sides but different variate columns, so only
    # the shared part is identical; with the same column it is exact.
    a2, b2 = PL.swap_totals(models, names[:3], names[3], names[4], u,
                            'KDE, Uniform', saving=0.0, n_materials=4)
    shared = a - np.asarray(
        models[names[3]]['KDE, Uniform'].rvs_from_uniform(u[:, 3]), float)
    shared2 = a2 - np.asarray(
        models[names[3]]['KDE, Uniform'].rvs_from_uniform(u[:, 3]), float)
    assert np.allclose(shared, shared2)


def test_the_saving_lands_where_it_is_asked_for():
    """Every dataset has a mean of 1.0, so the intensity carries the design
    difference and it must land exactly."""
    names, _, models = a_group(seed=40, n=200, k=5)
    rng = np.random.default_rng(40)
    u = rng.random((40_000, 5))
    for saving in (0.0, 0.05, 0.10):
        a, b = PL.swap_totals(models, names[:3], names[3], names[4], u,
                              'KDE, Uniform', saving=saving, n_materials=4)
        assert (a.mean() - b.mean()) / a.mean() == pytest.approx(
            saving, abs=0.02)


def test_discernibility_is_a_coin_flip_when_the_options_are_equivalent():
    rng = np.random.default_rng(41)
    a = rng.lognormal(0, 0.4, 20_000)
    b = rng.lognormal(0, 0.4, 20_000)
    got = PL.comparison_statement(a, b)
    assert 0.45 < got['discernibility'] < 0.55
    assert got['mci_1.2'] > got['discernibility']


def test_the_comparison_margin_is_harder_to_clear_in_the_right_direction():
    """A margin above 1 asks whether A beats B by enough to act on, so it must
    be EASIER to satisfy than plain dominance when A is the smaller."""
    rng = np.random.default_rng(42)
    a = rng.lognormal(0, 0.3, 20_000)
    b = a * 1.1
    got = PL.comparison_statement(a, b)
    assert got['discernibility'] > 0.9
    assert got['mci_1.2'] >= got['discernibility']


def test_the_figure_style_writes_ascii_minus_signs():
    """FIGURE_STYLE.md requires plain ASCII on every figure and matplotlib's
    default negative tick label is U+2212, which nothing had ever turned off."""
    import matplotlib as mpl
    import figstyle
    figstyle.apply()
    assert mpl.rcParams['axes.unicode_minus'] is False
    fig, ax = plt.subplots()
    ax.plot([-1.0, 0.0, 1.0], [0, 1, 0])
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    plt.close(fig)
    assert all(lab.isascii() for lab in labels), labels


def test_the_overlap_check_sees_a_LEFT_aligned_panel_title():
    """The clash detector had never checked a single title in this project.

    matplotlib keeps a separate Text artist for the center, left and right
    title; `ax.get_title()` reads the CENTER one, and `figstyle.apply` sets
    `axes.titlelocation` to 'left'. So every title this project draws lives in
    `_left_title`, the old guard `if ax.get_title()` was always false, and the
    function reported no overlap on a five-column figure whose titles plainly
    collided. Found in Stage 2f.
    """
    import figstyle
    figstyle.apply()
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.6))
    for ax in axes:
        ax.set_title('A Very Long Panel Title That Will Certainly Collide',
                     fontsize=8)
    hits = figstyle.check_overlaps(fig, verbose=False)
    plt.close(fig)
    assert hits, 'colliding left-aligned titles were not reported'


def test_the_overlap_check_is_quiet_when_nothing_collides():
    """The control: a detector that always fires is not a detector."""
    import figstyle
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.6))
    for ax in axes:
        ax.set_title('Short', fontsize=8)
    hits = figstyle.check_overlaps(fig, verbose=False)
    plt.close(fig)
    assert not hits, hits


def test_the_overlap_check_sees_a_label_sitting_on_the_data():
    """The check claimed to do this from the day it was written and did not.

    `check_overlaps` compared text against text only, so a legend label
    squarely on a data point passed -- the exact fault FIGURE_STYLE.md section
    5 names, and it happened on a Stage 2f figure. It now renders the canvas
    with the text hidden and counts plotted ink inside each label's box.
    """
    import figstyle
    import numpy as np
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(4, 2.5))
    x = np.linspace(0, 1, 50)
    ax.fill_between(x, 0, 1, color='#0072B2')
    ax.text(0.5, 0.5, 'LABEL ON INK', ha='center')
    hits = figstyle.check_overlaps(fig, verbose=False)
    plt.close(fig)
    assert hits, 'a label on a solid band was not reported'


def test_the_overlap_check_is_quiet_for_a_label_in_white_space():
    """The control, and the reason the background is the figure facecolor and
    not the median pixel: a median over a canvas the data fill IS the data's
    color, so every label would read as sitting on background."""
    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot([0, 1], [0, 1])
    ax.text(0.05, 0.9, 'clear label')
    hits = figstyle.check_overlaps(fig, verbose=False)
    plt.close(fig)
    assert not hits, hits


def test_finish_does_not_delete_deliberate_tick_labels():
    """A categorical axis loses its labels if `finish` thins the locator, and
    it fails silently: the figure just comes back with some bands unnamed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figstyle

    figstyle.apply()
    fig, ax = plt.subplots()
    bands = ['3-9', '10-99', '100-999', '1000+']
    ax.plot(range(len(bands)), [1, 2, 3, 4])
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands)
    figstyle.finish(ax, title='t', xlabel='x', ylabel='y')
    fig.canvas.draw()
    got = [t.get_text() for t in ax.get_xticklabels()]
    plt.close(fig)
    assert got == bands


# ---------------------------------------------------------------------------
# Stage 2h: which way a comparison margin points
# ---------------------------------------------------------------------------
def test_comparison_margin_above_one_loosens_and_below_one_tightens():
    """THE DIRECTION IS NOT OBVIOUS AND THE DOCSTRING HAD IT BACKWARDS.

    `mci_g` is `P(a < g * b)`. A margin ABOVE one is a TOLERANCE -- "a is
    better, or worse by less than g" -- and a margin BELOW one is the credit
    form, "a beats b by at least 1 - g". Caught on the study's own output,
    where mci_1.2 read 0.9993 at a true 20 percent saving against a
    discernibility of 0.9628.
    """
    rng = np.random.default_rng(0)
    base = rng.lognormal(0.0, 0.25, 40000)
    # A proposal 10 percent better in expectation, with its own spread, which
    # is the realistic case: an exact multiple makes every margin degenerate.
    prop = base * 0.90 * rng.lognormal(0.0, 0.10, 40000)
    got = PL.comparison_statement(prop, base, margins=(1.0, 1.2, 0.90, 0.80))
    # A TOLERANCE is satisfied at least as often as plain superiority.
    assert got['mci_1.2'] >= got['discernibility']
    # A CREDIT margin is satisfied less often, and a stricter credit less
    # often still. This is the ordering the docstring had backwards.
    assert got['mci_0.9'] < got['discernibility']
    assert got['mci_0.8'] < got['mci_0.9']
    # And all four are genuine interior probabilities rather than 0 or 1, so
    # the ordering is not an artifact of a degenerate case.
    for key in ('discernibility', 'mci_1.2', 'mci_0.9', 'mci_0.8'):
        assert 0.01 < got[key] < 0.999, (key, got[key])


# ---------------------------------------------------------------------------
# Stage 2h: the sampler must resolve the BODY, not the truncation bounds
# ---------------------------------------------------------------------------
class _WideParent:
    """A parent whose mass is near 1 and whose support runs to 1e8.

    This is the shape that broke Stage 2h: the truncation rule's width grows
    exponentially in the data's log spread, so `hi` can be enormous while the
    distribution itself is ordinary. A linear grid across [lo, hi] then puts
    the whole body inside its first cell.
    """
    normalizer = 1.0
    lo, hi = 0.0, 1e8

    def __init__(self, mu=0.0, sigma=0.6):
        from scipy.stats import lognorm
        self._d = lognorm(s=sigma, scale=np.exp(mu))

    def cdf(self, x, scheme='market'):
        return self._d.cdf(np.asarray(x, dtype=float))

    def ppf(self, q, scheme='market'):
        return self._d.ppf(np.asarray(q, dtype=float))


def test_parent_sampler_resolves_a_body_inside_an_enormous_support():
    """THE DEFECT THAT COST A WHOLE REGENERATION.

    With `hi` at 1e8, a linear 20,001-point grid has a spacing of ~5,000, so
    every value the distribution actually produces falls between the first two
    points. The tabulated CDF becomes a step, inverting it returns draws spread
    over the whole support, and the "true" mean came back as 6,624 against data
    normalized to 1.0. Nothing upstream was wrong: the parent was sound and
    every check on the generator passed.
    """
    p = _WideParent()
    s = PL.ParentSampler(p)
    u = np.linspace(1e-6, 1 - 1e-6, 20001)
    got = s.ppf(u)
    exact = p.ppf(u)
    # The mean of a lognormal(0, 0.6) is exp(0.18) = 1.197.
    assert float(np.mean(got)) == pytest.approx(float(np.mean(exact)), rel=1e-3)
    assert float(np.median(got)) == pytest.approx(1.0, rel=1e-3)
    # And the quantiles agree across the body, not just on average.
    for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.999):
        a = float(np.ravel(s.ppf(q))[0]); b = float(np.ravel(p.ppf(q))[0])
        assert a == pytest.approx(b, rel=1e-3), q


def test_parent_sampler_tail_points_are_quantile_spaced_not_linear():
    """A LINEAR run of points from the body out to `hi` spreads real
    probability across a range the parent never puts mass in. The grid's top
    must track the parent's quantiles instead."""
    p = _WideParent()
    s = PL.ParentSampler(p)
    # The parent's 1 - 1e-9 quantile is about 7.6; nothing in the grid should
    # sit orders of magnitude above that except the closing bound itself.
    inner = s.grid[s.grid < p.hi]
    assert inner.max() < 1e3, inner.max()
    # The support is still closed at the true bound, so ppf(1) cannot fall off.
    assert s.grid[-1] == pytest.approx(p.hi)


def test_the_hoisted_swap_draws_reproduce_swap_totals_exactly():
    """Stage 2j split the saving-independent draw blocks out of `swap_totals`
    so that six savings do not redraw the same five columns six times. The
    saving enters only through option B's use intensity, so the two routes
    must agree to the last bit, and the committed design-swap table must not
    move because of it."""
    names, _, models = a_group(seed=11, k=5)
    u = np.random.default_rng(12).random((NECCS, 5))
    for method in ('KDE, Uniform', 'Lognormal, Variable'):
        base, a, b, k = PL.swap_components(models, names[:3], names[3],
                                           names[4], u, method)
        for saving in PL.SWAP_SAVINGS:
            want_a, want_b = PL.swap_totals(models, names[:3], names[3],
                                            names[4], u, method, saving)
            got_a, got_b = PL.swap_from_components(base, a, b, k, saving)
            assert np.array_equal(got_a, want_a)
            assert np.array_equal(got_b, want_b)

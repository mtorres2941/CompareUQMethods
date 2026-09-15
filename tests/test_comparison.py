"""The paper's method comparison: held-out W1, the tail check, and the curves.

Stage 2b. These guard the three things the comparison rests on: that the
held-out score really is out of sample, that the tail check notices a tail W1
does not, and that the curve smoothing adapts to the size of the arm instead of
assuming the corpus.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import comparison as C  # noqa: E402
import families as F  # noqa: E402
import fitting as FT  # noqa: E402


def dataset(n, seed, kind='lognormal'):
    rng = np.random.default_rng(seed)
    x = (rng.lognormal(0.0, 0.7, n) if kind == 'lognormal'
         else np.abs(rng.normal(1.0, 0.3, n)) + 1e-6)
    x = x / x.mean()
    return x, rng.dirichlet(np.ones(n))


# ---------------------------------------------------------------------------
# held-out W1
# ---------------------------------------------------------------------------
def test_heldout_is_undefined_for_datasets_too_small_to_split():
    """NaN, not a number computed from two points."""
    rng = np.random.default_rng(0)
    for n in (3, 5, 9):
        x, w = dataset(n, n)
        assert np.isnan(C.heldout_w1(x, w, 'KDE, Variable', rng))
    x, w = dataset(C.HELDOUT_MIN_N, 11)
    assert np.isfinite(C.heldout_w1(x, w, 'KDE, Variable', rng))


def test_heldout_is_worse_than_in_sample_for_the_flexible_method():
    """The whole point: in-sample W1 rewards flexibility and held-out does not.

    A KDE interpolates the values it was fitted on, so scoring it against those
    same values flatters it. Scoring it against values it has not seen must not.
    """
    x, w = dataset(400, 12)
    models, _ = FT.fit_pewt(x, w)
    label = 'KDE, Variable'
    ins = FT.score_w1_model(models[label], x, w)
    out = C.heldout_w1(x, w, label, np.random.default_rng(1), repeats=4)
    assert out > ins


def test_heldout_removes_most_of_w1s_bandwidth_sensitivity():
    """In-sample W1 collapses as the bandwidth shrinks. Held-out barely moves.

    THE CLAIM AT THE RIGHT STRENGTH, because the obvious stronger one is false.
    In-sample W1 falls monotonically to a few percent of any standard rule, so
    it rewards a KDE for becoming the sample. Held-out W1 does not: across a
    50-fold bandwidth range its spread is a small fraction of the in-sample
    spread. But it does NOT have a sharp interior optimum either -- measured on
    the 149-dataset empirical arm the held-out mean turns at about 0.1 of
    Scott's bandwidth and the held-out median at 0.5, and the median moves only
    from 0.233 to 0.243 across the whole range.

    So held-out W1 is the right instrument for comparing FAMILIES of different
    parameter counts, which is what this study needs it for, and the WRONG one
    for choosing a bandwidth. Leave-one-out likelihood is the sharp instrument
    for that; see `audits/stage2b/r9_bandwidth.py` and decision 54.
    """
    rng = np.random.default_rng(2)
    mults = (1.0, 0.25, 0.05, 0.02)
    ins = {m: [] for m in mults}
    out = {m: [] for m in mults}
    for seed in range(10):
        x, w = dataset(120, 200 + seed)
        base = FT.weighted_bw(x, w, bw_method='scott')
        for mult in mults:
            m = F.Truncated(F.WeightedKDE(x, w, base * mult), label='kde')
            ins[mult].append(FT.score_w1_model(m, x, w))
            vals = []
            for _ in range(4):
                idx = rng.permutation(len(x))
                a, b = idx[: len(x) // 2], idx[len(x) // 2:]
                ma = F.Truncated(F.WeightedKDE(x[a], w[a] / w[a].sum(),
                                               base * mult), label='kde')
                vals.append(FT.score_w1_model(ma, x[b], w[b] / w[b].sum()))
            out[mult].append(np.mean(vals))

    def spread(d):
        means = np.array([np.mean(d[m]) for m in mults])
        return float((means.max() - means.min()) / means.mean())

    assert np.mean(ins[0.02]) < 0.3 * np.mean(ins[1.0]), (
        'in-sample W1 must collapse as the bandwidth shrinks')
    assert spread(out) < 0.5 * spread(ins), (
        f'held-out spread {spread(out):.3f} should be far below the in-sample '
        f'spread {spread(ins):.3f}')


# ---------------------------------------------------------------------------
# the tail check
# ---------------------------------------------------------------------------
def test_model_sd_ratio_is_near_one_for_a_sane_fit():
    x, w = dataset(500, 14)
    models, _ = FT.fit_pewt(x, w)
    for label in FT.PEWT:
        r = C.model_sd_ratio(models[label], x, w)
        assert 0.5 < r < 2.0, f'{label} gave {r:.4g}'


def test_model_sd_ratio_catches_a_tail_that_w1_does_not():
    """The failure mode of discrepancy entry 43, in one assertion.

    A lognormal whose threshold sits just below the smallest value matches the
    body and carries an enormous tail. W1 barely moves; the ratio does.
    """
    x, w = dataset(200, 15)
    good, _ = FT.fit_family('lognormal_3p', x, w, 'mle')
    heavy = F.make_lognorm(dict(s=2.6, loc=float(x.min()) - 1e-4,
                                scale=float(np.median(x))))
    w1_good = FT.score_w1_model(good, x, w)
    w1_heavy = FT.score_w1_model(heavy, x, w)
    r_good = C.model_sd_ratio(good, x, w)
    r_heavy = C.model_sd_ratio(heavy, x, w)
    assert r_heavy > 10 * r_good, 'the ratio must see the tail'
    assert w1_heavy < 30 * w1_good, 'W1 barely charges for it, which is the point'


# ---------------------------------------------------------------------------
# ranks and curves
# ---------------------------------------------------------------------------
def test_ranks_are_within_dataset_and_scale_free():
    rng = np.random.default_rng(3)
    ds = {f'd{i}': dataset(60, 20 + i) for i in range(4)}
    scores = C.score_methods(ds, rng, 'test', heldout=False)
    ranked = C.add_ranks(scores)
    per = ranked.groupby('dataset').w1_rank
    assert per.min().eq(1).all() and per.max().eq(len(FT.PEWT)).all()
    # Rescaling one dataset must not move its ranks: W1 is linear in a scaling
    # and every method is fitted to the same values.
    ds2 = dict(ds)
    x, w = ds2['d0']
    ds2['d0'] = (x * 1000.0, w)
    r2 = C.add_ranks(C.score_methods(ds2, np.random.default_rng(3), 'test',
                                     heldout=False))
    a = ranked[ranked.dataset == 'd0'].set_index('method').w1_rank
    b = r2[r2.dataset == 'd0'].set_index('method').w1_rank
    assert (a == b[a.index]).all()


def test_curve_window_scales_to_the_arm():
    """The Stage 1 figure hard-coded 501, which exceeds the empirical arm."""
    assert C.curve_window(9999) < 9999
    assert C.curve_window(149) <= 149
    assert C.curve_window(149) >= C.CURVE_WINDOW_MIN
    for n in (12, 149, 1000, 9999):
        assert C.curve_window(n) % 2 == 1, 'window must be odd to centre'


def test_characteristic_curves_are_unbinned_and_ordered():
    rng = np.random.default_rng(4)
    ds = {f'd{i}': dataset(40 + 3 * i, 30 + i) for i in range(12)}
    scores = C.score_methods(ds, rng, 'test', heldout=False)
    chars = pd.DataFrame({'n': {k: len(v[0]) for k, v in ds.items()}})
    curves = C.characteristic_curves(scores, chars)
    assert set(curves.method) == set(FT.PEWT)
    # One row per dataset, not per bin.
    for _, g in curves.groupby(['characteristic', 'method']):
        assert len(g) == len(ds)
        assert g.x.is_monotonic_increasing
        assert np.isfinite(g.y_smooth).all()


def test_bandwidth_comparison_covers_every_rule():
    ds = {f'd{i}': dataset(50, 40 + i) for i in range(3)}
    b = C.bandwidth_comparison(ds, 'test')
    assert set(b.rule) == {'scott', 'silverman', 'silverman_guarded'}
    assert len(b) == 3 * 2 * 3
    # Scott is never the smaller bandwidth: it uses sd where Silverman uses
    # min(sd, IQR/1.34).
    wide = b.pivot_table(index=['dataset', 'weighting'], columns='rule',
                         values='h_over_sd')
    assert (wide['scott'] >= wide['silverman'] - 1e-12).all()


def test_heldout_uses_one_set_of_splits_for_all_six_methods():
    """Paired by construction: the same partitions score every method.

    Six independent splits would confound a difference between methods with a
    difference between random partitions, and would cost six times as much.
    """
    x, w = dataset(200, 99)
    a = C.heldout_w1_all(x, w, np.random.default_rng(5))
    for label in FT.PEWT:
        b = C.heldout_w1(x, w, label, np.random.default_rng(5))
        assert a[label] == pytest.approx(b, rel=1e-12)


def test_within_weighting_ranks_compare_only_the_estimation_methods():
    """Three methods per weighting scheme, so ranks run 1 to 3, not 1 to 6.

    The held-out score must be read this way: the weights are an exchangeable
    Dirichlet draw, so a held-out comparison between weighting schemes measures
    the draw rather than the weighting.
    """
    rng = np.random.default_rng(6)
    ds = {f'd{i}': dataset(60, 60 + i) for i in range(4)}
    scores = C.score_methods(ds, rng, 'test', heldout=False)
    ranked = C.add_ranks(scores, 'w1', within_weighting=True,
                         suffix='w1_rank_within')
    assert ranked.w1_rank_within.between(1, 3).all()
    for (_, _), g in ranked.groupby(
            ['dataset', ranked.method.map(C.weighting_of)]):
        assert sorted(g.w1_rank_within) == [1.0, 2.0, 3.0]
    # The six-way rank is still available and still runs 1 to 6.
    assert C.add_ranks(scores, 'w1').w1_rank.between(1, 6).all()


def test_relative_score_keeps_the_size_of_the_gap_that_a_rank_discards():
    """Two datasets with identical RANKS and very different separations.

    The relative column must tell them apart; the rank cannot. This is the
    defect it exists to fix: at n < 10 the three estimation methods differ by a
    median of 21 percent and a mean-rank curve implies a separation that is not
    there.
    """
    scores = pd.DataFrame({
        'arm': ['e'] * 6,
        'dataset': ['tight'] * 3 + ['wide'] * 3,
        'method': ['KDE, Variable', 'Lognormal, Variable', 'Normal, Variable'] * 2,
        'w1': [0.100, 0.101, 0.102, 0.100, 0.500, 0.900],
    })
    out = C.add_relative(C.add_ranks(scores, 'w1'), 'w1')
    tight = out[out.dataset == 'tight']
    wide = out[out.dataset == 'wide']
    assert sorted(tight.w1_rank) == sorted(wide.w1_rank), 'ranks are identical'
    assert tight.w1_relative.max() - tight.w1_relative.min() < 0.05
    assert wide.w1_relative.max() - wide.w1_relative.min() > 1.0
    # A method that scores the average of the methods sits at exactly 1.0.
    assert out.groupby('dataset').w1_relative.mean().eq(1.0).all()

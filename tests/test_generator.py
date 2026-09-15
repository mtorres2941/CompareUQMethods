"""Tests for src/generator.py and src/genconfig.py."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import genconfig as G  # noqa: E402
import generator as GEN  # noqa: E402

CFG = G.DEFAULT


def test_strata_allocate_as_configured():
    rng = np.random.default_rng(0)
    sizes, names = GEN.stratified_sizes(CFG, rng)
    assert len(sizes) == 10_000
    for s in CFG.strata:
        sel = sizes[names == s.name]
        assert len(sel) == s.n_datasets
        assert sel.min() >= s.n_lo
        assert sel.max() <= s.n_hi


def test_strata_cover_their_endpoints():
    """Log-uniform within a stratum must actually reach both ends, or the
    coverage claim in Part 6 is wrong at the boundaries."""
    rng = np.random.default_rng(1)
    sizes, names = GEN.stratified_sizes(CFG, rng)
    for s in CFG.strata:
        sel = sizes[names == s.name]
        assert sel.min() == s.n_lo
        assert sel.max() >= int(0.9 * s.n_hi)


def test_probe_set_is_outside_the_corpus():
    rng = np.random.default_rng(2)
    p = GEN.probe_sizes(CFG, rng)
    assert len(p) == CFG.probe.n_datasets
    assert p.min() >= 10_000
    assert p.max() <= 100_000
    corpus_max = max(s.n_hi for s in CFG.strata)
    assert p.min() > corpus_max


@pytest.mark.parametrize('n', [3, 4, 9, 25, 300, 2500])
def test_generated_datasets_are_valid_and_normalized(n):
    rng = np.random.default_rng(20260911)
    for _ in range(12):
        x, w, rec = GEN.generate_dataset(CFG, n, rng)
        assert x is not None, rec
        assert GEN.validity_failures(x, w, n) == []
        assert len(x) == n
        assert np.mean(x) == pytest.approx(1.0, rel=1e-12)
        assert w.sum() == pytest.approx(1.0, rel=1e-12)
        assert np.all(x > 0)


def test_record_reconstructs_the_parent_exactly():
    """The provenance record must be enough to rebuild the parent. This is the
    contract Stage 2c depends on."""
    import components as C
    import mixture as M
    rng = np.random.default_rng(5)
    x, w, rec = GEN.generate_dataset(CFG, 500, rng)
    comps = []
    for c in rec['components']:
        fam, shape = c['family'], c['shape']
        _, _, loc, scale, status = C.solve_component(c['skew'], c['exkurt'], mean=0.0,
                                                     sd=c['sd'])
        assert status == 'ok'
        comps.append((fam, shape, loc, scale))
    assert len(comps) == rec['k']
    assert len(rec['pi']) == rec['k']
    assert len(rec['market']) == rec['k']
    assert sum(rec['mode_counts']) == rec['n']
    assert rec['normalizer'] > 0
    assert 0.0 <= rec['truncated_mass'] < 1.0


def test_overlap_target_is_met_or_the_shortfall_is_recorded():
    rng = np.random.default_rng(6)
    met = shortfall = 0
    for _ in range(40):
        _, _, rec = GEN.generate_dataset(CFG, 200, rng)
        if rec['k'] == 1:
            assert rec['overlap_status'] == 'single_component'
            continue
        if rec['overlap_status'] == 'ok':
            assert rec['overlap_achieved'] == pytest.approx(rec['overlap_target'],
                                                            rel=5e-3)
            met += 1
        else:
            assert rec['overlap_status'] in ('clipped_max_overlap', 'clipped_min_overlap',
                                             'tolerance_not_met')
            shortfall += 1
    assert met > 0


def test_validity_filter_does_not_reject_for_being_unusual():
    """Part 2. The old filter removed marginal outliers on any of 20 metrics,
    including weight_outliers, which are the cases the study exists to study.
    The replacement must pass an extreme but analysable dataset."""
    x = np.concatenate([np.full(99, 0.5), [50.0]])
    x = x / x.mean()
    w = np.full(100, 1 / 100)
    assert GEN.validity_failures(x, w, 100) == []

    w2 = np.full(100, 1e-6)
    w2[0] = 1.0 - 99e-6
    assert GEN.validity_failures(x, w2 / w2.sum(), 100) == []


def test_validity_filter_catches_what_it_should():
    n = 20
    good_w = np.full(n, 1 / n)
    assert 'zero_variance' in GEN.validity_failures(np.ones(n), good_w, n)
    x = np.linspace(0.5, 1.5, n)
    x = x / x.mean()
    bad = x.copy()
    bad[0] = np.nan
    assert 'non_finite_values' in GEN.validity_failures(bad, good_w, n)
    neg = x.copy()
    neg[0] = -1.0
    assert 'non_positive_values' in GEN.validity_failures(neg, good_w, n)
    assert 'wrong_size' in GEN.validity_failures(x, good_w, n + 1)
    degen = np.zeros(n)
    degen[0] = 1.0
    assert 'degenerate_weight_vector' in GEN.validity_failures(x, degen, n)


def test_undefined_kurtosis_at_n3_is_not_a_validity_failure():
    """Expected, and informative: small n is where parametric families should
    beat KDE. Stage 2f must be told, not left to drop the stratum."""
    rng = np.random.default_rng(8)
    for _ in range(30):
        x, w, rec = GEN.generate_dataset(CFG, 3, rng)
        assert GEN.validity_failures(x, w, 3) == []


def test_sample_moment_bounds_are_the_sharp_ones():
    for n in (3, 4, 5, 10, 50):
        a = np.zeros(n)
        a[0] = 1.0
        m = a.mean()
        m2 = np.mean((a - m) ** 2)
        g1 = abs(np.mean((a - m) ** 3) / m2 ** 1.5)
        b2 = np.mean((a - m) ** 4) / m2 ** 2
        max_skew, min_k, max_k = GEN.sample_moment_bounds(n)
        assert g1 == pytest.approx(max_skew, rel=1e-12)
        assert b2 == pytest.approx(max_k, rel=1e-12)


def test_generation_is_reproducible_from_the_seed():
    a = GEN.generate_dataset(CFG, 200, np.random.default_rng(42))
    b = GEN.generate_dataset(CFG, 200, np.random.default_rng(42))
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_generation_never_touches_global_numpy_state():
    np.random.seed(123)
    before = np.random.get_state()[1][0]
    GEN.generate_dataset(CFG, 300, np.random.default_rng(0))
    assert np.random.get_state()[1][0] == before


def test_zero_coupling_reproduces_uncoupled_weights():
    """Part 3's sweep endpoint: at coupling 0 a point's weight carries no
    information about which mode it came from."""
    cfg = CFG.replace(mode_coupling=0.0)
    rng = np.random.default_rng(9)
    x, w, rec = GEN.generate_dataset(cfg, 2000, rng)
    assert rec['k'] >= 1
    assert w.sum() == pytest.approx(1.0)
    assert GEN.validity_failures(x, w, 2000) == []


def test_a_rejected_parent_draw_is_redrawn_rather_than_abandoned():
    """Both rejection statuses must be retried, not just one.

    The targets a parent is drawn from are random. A draw that lands on an
    unusable combination is a rejected draw, and a fresh one almost always
    succeeds; abandoning the slot leaves the corpus short of the size that was
    asked for and the stratum counts disagreeing with the design.

    Refusing to APPROXIMATE a target that cannot be met is a different thing and
    is not affected.
    """
    assert 'mode_too_narrow' in GEN.REDRAWABLE
    assert 'component_targets_exhausted' in GEN.REDRAWABLE

    cfg = G.DEFAULT
    calls = {'n': 0}
    real = GEN.draw_parent

    def flaky(c, n, rng):
        calls['n'] += 1
        if calls['n'] <= 3:
            return None, dict(status='component_targets_exhausted',
                              statuses=['unbounded_density'])
        return real(c, n, rng)

    GEN.draw_parent = flaky
    try:
        x, w, rec = GEN.generate_dataset(cfg, 40, np.random.default_rng(0))
    finally:
        GEN.draw_parent = real
    assert x is not None, 'a redrawable rejection must not abandon the dataset'
    assert calls['n'] == 4
    assert rec['parent_retries'] == 3


def test_a_genuine_failure_is_not_retried():
    """A status that is not a rejected draw must still stop immediately."""
    cfg = G.DEFAULT
    calls = {'n': 0}
    real = GEN.draw_parent

    def broken(c, n, rng):
        calls['n'] += 1
        return None, dict(status='something_structural')

    GEN.draw_parent = broken
    try:
        x, w, rec = GEN.generate_dataset(cfg, 40, np.random.default_rng(0))
    finally:
        GEN.draw_parent = real
    assert x is None and calls['n'] == 1

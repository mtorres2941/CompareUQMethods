"""Tests for src/mixture.py, the truncated-mixture parent.

The deliverable of Stage 2a Part 1 is that the parent CDF is exact, so these
tests check it against large samples drawn through the real pipeline, and check
that the inverse-CDF sampler agrees with the truncation loop it replaces.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import components as C  # noqa: E402
import mixture as M  # noqa: E402


def build_parent(specs, pi, market, coupling=1.0):
    comps = []
    for skew, exkurt, mean, sd in specs:
        fam, shape, loc, scale, status = C.solve_component(skew, exkurt, mean=mean, sd=sd)
        assert status == 'ok', (skew, exkurt, status)
        comps.append(C.frozen(fam, shape, loc, scale))
    pi = np.asarray(pi, float)
    lo, hi = M.population_truncation_bounds(comps, pi)
    return M.MixtureParent(comps, pi, np.asarray(market, float), lo, hi, coupling)


SPECS = [(1.2, 3.0, 5.0, 1.0), (-0.8, 1.0, 6.5, 1.2), (2.5, 12.0, 8.0, 0.8)]


def test_parent_cdf_matches_the_pipeline_it_describes():
    """A large sample drawn through the real generator must have the parent's
    CDF. This is the Stage 2c contract."""
    rng = np.random.default_rng(20260911)
    p = build_parent(SPECS, [0.5, 0.3, 0.2], [0.6, 0.1, 0.3])
    n = 400_000
    x, modes = p.sample(n, rng)
    xs = np.sort(x)
    grid = np.quantile(xs, np.linspace(0.002, 0.998, 400))
    empirical = np.searchsorted(xs, grid, side='right') / n
    assert np.max(np.abs(empirical - p.cdf(grid))) < 5.0 / np.sqrt(n)


def test_market_weighted_parent_is_a_real_population_object():
    """Part 3. Weighting the realized points by their mode's market share must
    reproduce the market-weighted parent, which is what gives the
    variable-weighted methods something to be right or wrong about."""
    rng = np.random.default_rng(7)
    p = build_parent(SPECS, [0.5, 0.3, 0.2], [0.6, 0.1, 0.3], coupling=1.0)
    n = 400_000
    x, modes = p.sample(n, rng)
    counts = np.bincount(modes, minlength=len(p.comps))
    w = p.market_effective[modes] / counts[modes]
    w = w / w.sum()
    order = np.argsort(x)
    grid = np.quantile(x, np.linspace(0.01, 0.99, 300))
    empirical = np.interp(grid, x[order], np.cumsum(w[order]))
    assert np.max(np.abs(empirical - p.cdf(grid, 'market'))) < 5.0 / np.sqrt(n)


def test_zero_coupling_collapses_the_two_parents():
    """At coupling 0 the point weights carry no mode information, so the
    market-weighted parent IS the uniform-weighted parent. That is the
    circularity Stage 2c addresses, stated as an identity."""
    p = build_parent(SPECS, [0.5, 0.3, 0.2], [0.6, 0.1, 0.3], coupling=0.0)
    assert np.allclose(p.market_effective, p.pi_trunc)
    grid = np.linspace(p.lo + 1e-6, p.hi - 1e-6, 200)
    assert np.allclose(p.cdf(grid, 'market'), p.cdf(grid, 'uniform'))


def test_ppf_inverts_cdf_for_both_schemes():
    p = build_parent(SPECS, [0.5, 0.3, 0.2], [0.6, 0.1, 0.3])
    q = np.array([1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-3])
    for scheme in ('uniform', 'market'):
        assert np.max(np.abs(p.cdf(p.ppf(q, scheme), scheme) - q)) < 1e-10


def test_inverse_cdf_sampling_agrees_with_the_truncation_loop_it_replaces():
    """Part 1 asks for this before the loop is deleted.

    The old loop drew counts from the UNtruncated proportions, filtered, kept
    `data[:n]` and refilled from a component chosen with the untruncated
    weights. Where truncation removes little mass the two must agree closely;
    the test pins that they do, on a parent whose truncation is mild.
    """
    rng = np.random.default_rng(3)
    p = build_parent(SPECS, [0.5, 0.3, 0.2], [0.6, 0.1, 0.3])
    assert p.truncated_mass() < 0.01

    n = 60_000
    exact, _ = p.sample(n, rng, normalize=False)

    # the old procedure, on the same parent and the same bounds
    loop = []
    while len(loop) < n:
        counts = rng.multinomial(2000, p.pi)
        chunk = []
        for k, cnt in enumerate(counts):
            if cnt > 0:
                u = rng.uniform(1e-12, 1 - 1e-12, int(cnt))
                chunk.append(np.asarray(p.comps[k].ppf(u), float))
        c = np.concatenate(chunk)
        loop.append(c[(c > p.lo) & (c < p.hi)])
    loop = np.concatenate(loop)[:n]

    qs = np.linspace(0.01, 0.99, 99)
    a, b = np.quantile(exact, qs), np.quantile(loop, qs)
    spread = float(np.quantile(exact, 0.99) - np.quantile(exact, 0.01))
    assert np.max(np.abs(a - b)) < 0.02 * spread


def test_truncated_mass_is_reported_not_hidden():
    p = build_parent(SPECS, [0.5, 0.3, 0.2], [0.6, 0.1, 0.3])
    m = p.truncated_mass()
    assert 0.0 <= m < 1.0
    # mass inside the bounds, from the parent's own component masses
    assert np.isfinite(m)


def test_overlap_is_symmetric_and_monotone_in_separation():
    """Two components moved apart must overlap less. This is what makes the
    bisection in solve_spread_for_overlap valid."""
    prev = None
    for sep in (0.5, 1.0, 2.0, 4.0, 8.0):
        comps = []
        for mean in (0.0, sep):
            fam, shape, loc, scale, status = C.solve_component(0.0, 0.0, mean=mean, sd=1.0)
            assert status == 'ok'
            comps.append(C.frozen(fam, shape, loc, scale))
        om = M.pairwise_overlap(comps, np.array([0.5, 0.5]), grid_n=2001)
        assert om[0, 1] == pytest.approx(om[1, 0])
        if prev is not None:
            assert om[0, 1] < prev
        prev = om[0, 1]


def test_overlap_of_identical_components_is_one():
    """Two identical equally weighted components misclassify every draw, so
    each conditional overlap is 1/2 and their sum is 1."""
    fam, shape, loc, scale, status = C.solve_component(0.0, 0.0, mean=0.0, sd=1.0)
    d = C.frozen(fam, shape, loc, scale)
    om = M.pairwise_overlap([d, d], np.array([0.5, 0.5]), grid_n=20001)
    assert om[0, 1] == pytest.approx(1.0, abs=2e-3)


def test_overlap_matches_the_definition_counted_directly():
    """The crossings are located by linear interpolation, not by brentq.

    brentq was exact and cost 65 percent of the whole generation run, because
    each of its iterations evaluated two scipy pdfs on a one-element array.
    This pins that the cheap version lands on the same number.

    The oracle is the definition counted directly on 500,001 of component i's
    own quantiles: no interpolation, no refinement, no crossing location at
    all. An earlier version of this test compared against a second hand-rolled
    brentq loop, which was itself wrong -- a reference implementation is only a
    reference if it is simpler than the thing it checks.
    """
    def fine_reference(di, dj, pi_i, pi_j, grid_n=500_001):
        """Brute force: the share of a uniform grid of component i's own
        quantiles that the Bayes rule hands to component j. No interpolation,
        no refinement, no crossing location - just the definition, counted."""
        u = np.linspace(1e-12, 1 - 1e-12, grid_n)
        x = np.asarray(di.ppf(u), float)
        return float(M._assigned_to_other(x, di, dj, pi_i, pi_j).mean())

    rng = np.random.default_rng(11)
    worst = 0.0
    errors = []
    checked = 0
    # 30 trials, not 12. Since components.has_bounded_density began refusing
    # J-shaped beta and beta-prime solutions, about 45 percent of drawn moment
    # targets are rejected, so fewer components survive per trial and 12 trials
    # no longer accumulate the 15 pairs this test requires of itself. Raising
    # the trial count keeps the anti-vacuity guard doing its job; lowering the
    # guard would have hidden the change instead.
    for _ in range(30):
        comps = []
        for _ in range(int(rng.integers(2, 5))):
            skew = rng.uniform(-3, 3)
            exk = rng.uniform(max(skew ** 2 - 1.9, -1.1), 15)
            fam, shape, loc, scale, status = C.solve_component(
                skew, exk, mean=rng.uniform(0, 6), sd=10 ** rng.uniform(-0.7, 0.3))
            if status == 'ok':
                comps.append(C.frozen(fam, shape, loc, scale))
        if len(comps) < 2:
            continue
        pi = rng.dirichlet(np.ones(len(comps)) * 10)
        fast = M.pairwise_overlap(comps, pi)
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                di, dj = comps[i], comps[j]
                ref = (fine_reference(di, dj, pi[i], pi[j])
                       + fine_reference(dj, di, pi[j], pi[i]))
                err = abs(fast[i, j] - min(ref, 2.0))
                errors.append(err)
                worst = max(worst, err)
                checked += 1
    assert checked > 15
    median = float(np.median(errors))
    # the solver's own tolerance is 1e-3 relative, so this is far inside it
    # Median agreement is 2.2e-06. The residual sits entirely on NEARLY
    # DISJOINT pairs, where the crossing lies far out in a tail and the two
    # methods disagree about a region carrying about 2e-04 of probability. The
    # counting oracle is itself unreliable at that magnitude, and the solver's
    # own tolerance is 1e-03 relative, so this bound is not the binding
    # constraint on anything.
    assert worst < 5e-4, f'overlap off by {worst:.2e}'
    assert median < 1e-5, f'typical overlap error {median:.2e}'



def test_spread_solver_hits_the_requested_overlap():
    def build(c):
        comps = []
        for z, (skew, exkurt, sd) in enumerate([(0.5, 1.0, 1.0), (-0.3, 0.5, 1.0),
                                                (1.0, 2.0, 1.0)]):
            fam, shape, loc, scale, status = C.solve_component(
                skew, exkurt, mean=c * z, sd=sd)
            assert status == 'ok'
            comps.append(C.frozen(fam, shape, loc, scale))
        return comps, np.array([1 / 3, 1 / 3, 1 / 3])

    for target in (0.02, 0.1, 0.3):
        comps, got, status = M.solve_spread_for_overlap(build, target)
        assert status == 'ok', (target, status, got)
        assert got == pytest.approx(target, rel=2e-3)

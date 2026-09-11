"""Unit tests for src/components.py, the moment-targeted component families.

The claim these tests defend is the one Stage 2c depends on: a component
returned with status 'ok' has EXACTLY the requested mean, standard deviation,
skewness and excess kurtosis, and its CDF can be inverted.
"""
import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import components as C  # noqa: E402


def exact_mvsk(family, shape):
    """Moments from scipy directly, never from a sample."""
    if family == 'johnsonsu':
        m, v, s, k = stats.johnsonsu.stats(shape[0], shape[1], moments='mvsk')
    elif family == 'beta':
        m, v, s, k = stats.beta.stats(shape[0], shape[1], moments='mvsk')
    elif family == 'lognorm':
        m, v, s, k = stats.lognorm.stats(shape[0], moments='mvsk')
        if shape[-1] < 0:
            m, s = -m, -s
    elif family == 'betaprime':
        m, v, s, k = stats.betaprime.stats(shape[0], shape[1], moments='mvsk')
        if shape[-1] < 0:
            m, s = -m, -s
    else:
        raise AssertionError(family)
    return float(m), float(v), float(s), float(k)


TARGETS = [(sk, ek) for sk in (-4.0, -2.0, -0.7, 0.0, 0.7, 2.0, 4.0)
           for ek in (-1.0, -0.3, 0.5, 2.0, 8.0, 25.0)]


@pytest.mark.parametrize('skew,exkurt', TARGETS)
def test_moment_target_is_hit_exactly(skew, exkurt):
    mean, sd = 3.25, 0.8
    fam, shape, loc, scale, status = C.solve_component(skew, exkurt, mean=mean, sd=sd)
    if status != 'ok':
        assert status in ('infeasible_boundary', 'degenerate', 'unsolved')
        if status == 'infeasible_boundary':
            assert exkurt < skew ** 2 - 2 + C.BOUNDARY_MARGIN
        return
    m, v, s, k = exact_mvsk(fam, shape)
    assert loc + scale * m == pytest.approx(mean, abs=1e-10)
    assert scale * np.sqrt(v) == pytest.approx(sd, rel=1e-12)
    assert s == pytest.approx(skew, abs=1e-9)
    assert k == pytest.approx(exkurt, rel=1e-9, abs=1e-9)


def test_infeasible_targets_are_refused_not_approximated():
    """Below excess kurtosis = skewness ** 2 - 2 nothing exists, and the
    generator must say so rather than return the nearest feasible thing."""
    for skew in (0.5, 1.5, 3.0, -2.5):
        target = skew ** 2 - 2 - 0.5
        fam, shape, loc, scale, status = C.solve_component(skew, target)
        assert status == 'infeasible_boundary'
        assert fam is None


def test_accepted_components_invert_their_own_cdf():
    rng = np.random.default_rng(20260911)
    checked = 0
    for _ in range(400):
        skew = rng.uniform(-5, 5)
        exkurt = rng.uniform(-1.5, 40)
        fam, shape, loc, scale, status = C.solve_component(skew, exkurt, mean=5.0, sd=1.0)
        if status != 'ok':
            continue
        d = C.frozen(fam, shape, loc, scale)
        x = d.ppf(C.ROUNDTRIP_Q)
        assert np.all(np.diff(x) > 0)
        assert np.max(np.abs(d.cdf(x) - C.ROUNDTRIP_Q)) < 1e-7
        checked += 1
    assert checked > 200


def test_families_partition_the_plane_as_documented():
    """Each family is used in its own Pearson region and nowhere else."""
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(600):
        skew = rng.uniform(-5, 5)
        exkurt = rng.uniform(-1.5, 40)
        fam, shape, loc, scale, status = C.solve_component(skew, exkurt)
        if status != 'ok':
            continue
        seen.add(fam)
        line = float(C.lognormal_excess_kurtosis(skew))
        gamma = float(C.gamma_excess_kurtosis(skew))
        if fam == 'johnsonsu':
            assert exkurt > line
        elif fam == 'betaprime':
            assert gamma < exkurt <= line
        elif fam == 'beta':
            assert exkurt <= gamma
    assert {'johnsonsu', 'betaprime', 'beta'} <= seen


def test_reflection_is_a_population_property():
    """The left-skewed component is a reflected family, not a flipped sample.

    Its CDF is available in closed form at any x, which the old
    `np.max(data) - data + np.min(data)` never was.
    """
    fam, shape, loc, scale, status = C.solve_component(-2.0, 8.0, mean=1.0, sd=0.3)
    assert status == 'ok'
    d = C.frozen(fam, shape, loc, scale)
    m, v, s, k = exact_mvsk(fam, shape)
    assert s < 0
    grid = np.linspace(d.ppf(1e-3), d.ppf(1 - 1e-3), 50)
    cdf = d.cdf(grid)
    assert np.all(np.diff(cdf) >= -1e-12)
    assert cdf[0] == pytest.approx(1e-3, abs=1e-8)
    assert cdf[-1] == pytest.approx(1 - 1e-3, abs=1e-8)


def test_lognormal_line_is_consistent_with_the_lognormal():
    """A lognormal's own (skew, kurtosis) must land on the lognormal line."""
    for s in (0.2, 0.5, 0.9, 1.3):
        _, _, skew, exkurt = stats.lognorm.stats(s, moments='mvsk')
        assert float(C.lognormal_excess_kurtosis(float(skew))) == pytest.approx(
            float(exkurt), rel=1e-8)


def test_gamma_line_is_consistent_with_the_gamma():
    for a in (0.5, 2.0, 9.0):
        _, _, skew, exkurt = stats.gamma.stats(a, moments='mvsk')
        assert float(C.gamma_excess_kurtosis(float(skew))) == pytest.approx(
            float(exkurt), rel=1e-9)

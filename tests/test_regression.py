"""
Regression tests for the Stage 1 refactor.

Purpose
-------
Prove that a change labelled NEUTRAL did not move any reported number. These
tests are a change detector, not a correctness check: the fixtures were
produced by code with known defects, and Stage 2 will deliberately change many
of these values. When a number is meant to move, the fixture is re-frozen in
the same commit that moves it, and the commit message records the delta.

Layers
------
1. Fixture integrity. The frozen files have not themselves been edited.
2. Output comparison. Whatever the notebooks last wrote to outputs/tables
   matches the fixtures. This is the gate to run after executing the
   notebooks.
3. Independent recomputation. The metric and fitting code in src/ is driven
   directly, without running a notebook, and its output is compared to the
   fixtures. This is what actually exercises the library during a refactor.

Tolerance
---------
The environment that produced the fixtures no longer exists, so the fixtures
cannot be reproduced bit for bit. Measured agreement between the shipped
tables and a rebuild under the pinned compareuq environment (Python 3.11.16,
numpy 2.4.6, scipy 1.17.1, pandas 3.0.5) was:

    metric columns            max relative difference 1.3e-14
    Normal and KDE W1         max relative difference 2.0e-13
    Lognormal W1              max relative difference 4.3e-08

The lognormal term is the largest because weighted_lognorm_fit runs
scipy.optimize.minimize, whose convergence path differs slightly between scipy
versions. RTOL is set an order of magnitude above the worst observed value, so
it tolerates a compiler or library change but still catches any real change in
method.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = ROOT / "tests" / "fixtures"
TABLES = ROOT / "outputs" / "tables"
PROCESSED = ROOT / "data" / "processed"

RTOL = 1e-6
ATOL = 1e-12

# The seed every notebook uses. The empirical weights are drawn from it, so the
# frozen empirical table is only reproducible at this value.
NOTEBOOK_SEED = 42

# Stage 2a renamed one metric and added another. `mode_count_est` became
# `modality_index`, which is what it measures: a continuous KDE-based modality
# index, not a count. `crit_bw_1`, Silverman's critical bandwidth, is new.
# Neither change moves a value, and the recomputation tests below prove it by
# comparing every column the frozen fixtures and the current code have in
# common, under this mapping.
RENAMED_SINCE_FIXTURES = {
    "mode_count_est": "modality_index",
    "mode_count_est_uw": "modality_index_uw",
}
ADDED_SINCE_FIXTURES = ("crit_bw_1", "crit_bw_1_uw")


def align_to_fixture(actual, expected):
    """Rename current columns back to the fixture's names and drop new ones.

    Returns (expected_subset, actual_subset) over the shared columns, so a
    renamed metric is still checked value for value rather than silently
    skipped.
    """
    actual = actual.rename(columns={v: k for k, v in RENAMED_SINCE_FIXTURES.items()})
    actual = actual.drop(columns=[c for c in ADDED_SINCE_FIXTURES
                                  if c in actual.columns], errors="ignore")
    shared = [c for c in expected.columns if c in actual.columns]
    missing = [c for c in expected.columns if c not in actual.columns]
    assert not missing, f"fixture columns no longer produced: {missing}"
    return expected[shared], actual[shared]

TABLE_NAMES = [
    "TABLE_EmpiricalECCMetrics.xlsx",
    "TABLE_EmpiricalECCMetricsAndW1.xlsx",
    "TABLE_SyntheticECCMetricsAndW1.xlsx",
]

# Fitting constants, matching NB2 cells 12 and 20 and NB3 cell 15.
PE_METHODS = ["Normal", "Lognormal", "KDE"]
WT_METHODS = ["Uniform", "Variable"]
PEWT = [f"{pe}, {wt}" for pe in PE_METHODS for wt in WT_METHODS]
LOGFIT_OFFSET = 0.5
SCORE_GRID_POINTS = 1_000
SCORE_GRID_STD_MULTIPLE = 10
BW_METHOD = "scott"

# Number of synthetic datasets to recompute in the independent check. The full
# set of 10,000 takes about 30 s; a deterministic slice keeps the suite quick
# while still covering every code path.
N_SYNTHETIC_SAMPLE = 500


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def compare_frames(expected, actual, label):
    """Compare two frames, allowing matching non-finite values."""
    assert list(expected.columns) == list(actual.columns), f"{label}: columns differ"
    assert list(expected.index) == list(actual.index), f"{label}: index differs"

    numeric = expected.select_dtypes(include=[np.number]).columns
    exp = expected[numeric].to_numpy(dtype=float)
    act = actual[numeric].to_numpy(dtype=float)

    exp_finite = np.isfinite(exp)
    act_finite = np.isfinite(act)
    if not (exp_finite == act_finite).all():
        bad = np.argwhere(exp_finite != act_finite)[:5]
        rows = [(expected.index[i], numeric[j]) for i, j in bad]
        raise AssertionError(f"{label}: finite/non-finite pattern differs at {rows}")

    # Matching non-finite cells (the n=3 infinite kurtosis, for example) are
    # treated as equal; only the finite cells are compared numerically.
    np.testing.assert_allclose(
        act[exp_finite], exp[exp_finite], rtol=RTOL, atol=ATOL, err_msg=label
    )


def fit_pewt(x, weights_variable):
    """Fit all six PEWT models to one dataset.

    Mirrors NB2 cell 12 exactly. When the Phase 4 consolidation lands this
    should be replaced by a call to the shared src implementation, so that the
    test drives production code rather than a copy of it.
    """
    from scipy.stats import gaussian_kde, lognorm, norm

    from customstats import weighted_bw, weighted_lognorm_fit, weighted_std

    weights_uniform = np.ones_like(x) / len(x)
    models = {}
    for weights, wt in zip([weights_uniform, weights_variable], WT_METHODS):
        mean = np.sum(x * weights) / np.sum(weights)
        models[f"Normal, {wt}"] = norm(loc=mean, scale=weighted_std(x, weights))

        shape, loc, scale = weighted_lognorm_fit(
            x + LOGFIT_OFFSET, weights=weights, method="MLE"
        )
        models[f"Lognormal, {wt}"] = lognorm(
            s=shape, loc=loc - LOGFIT_OFFSET, scale=scale
        )

        bandwidth = weighted_bw(x, weights, bw_method=BW_METHOD)
        kde = gaussian_kde(x, bw_method=1.0, weights=weights)
        kde.set_bandwidth(bandwidth / (kde.covariance ** 0.5)[0][0])
        models[f"KDE, {wt}"] = kde
    return models


def score_w1(model, data, weights):
    """Wasserstein-1 between a fitted model and the weighted empirical data.

    Mirrors NB2 cell 32. The model is discretized onto a uniform grid from 0 to
    max(data) + 10 standard deviations, which implicitly truncates it at zero
    and renormalizes. That matches the rejection sampling used in NB3.
    """
    from customstats import wasserstein1_weighted, weighted_std

    std = np.max([np.std(data), weighted_std(data, weights)])
    grid = np.linspace(0, np.max(data) + SCORE_GRID_STD_MULTIPLE * std,
                       SCORE_GRID_POINTS)
    return wasserstein1_weighted(data, grid, weights, model.pdf(grid))


def load_json(name):
    with open(PROCESSED / name) as handle:
        return json.load(handle)


# ----------------------------------------------------------------------------
# layer 1: fixture integrity
# ----------------------------------------------------------------------------

def test_fixtures_present():
    for name in TABLE_NAMES:
        assert (FIXTURES / name).exists(), f"missing fixture {name}"


def test_fixture_checksums_unchanged():
    """The frozen fixtures must not be edited in place.

    A fixture may only change in a commit that deliberately moves a number,
    and that commit must update SHA256SUMS.txt alongside it.
    """
    import hashlib

    recorded = {}
    with open(FIXTURES / "SHA256SUMS.txt") as handle:
        for line in handle:
            digest, name = line.split()
            recorded[name] = digest

    for name, expected in recorded.items():
        actual = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        assert actual == expected, (
            f"fixture {name} was modified without updating SHA256SUMS.txt"
        )


# ----------------------------------------------------------------------------
# layer 2: notebook outputs against fixtures
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("name", TABLE_NAMES)
def test_output_table_matches_fixture(name):
    if not (TABLES / name).exists():
        pytest.skip(f"{name} not present in outputs/tables; run the notebooks first")
    expected = pd.read_excel(FIXTURES / name, index_col=0)
    actual = pd.read_excel(TABLES / name, index_col=0)
    compare_frames(expected, actual, name)


# ----------------------------------------------------------------------------
# layer 3: independent recomputation from src/
# ----------------------------------------------------------------------------

def test_empirical_metrics_recomputed():
    """Drive the empirical preparation path directly and compare to the frozen
    table.

    This recomputes through `empirical.prepare`, which is what notebook 1 runs:
    log-space trimming of near-zero values, then flat-Dirichlet weights from the
    notebook's seed, then division by the unweighted mean. Reading the stored
    weights out of `dct_realeccs_trimmed.json` instead, as an earlier version
    did, tested a path nothing uses any more.
    """
    import empirical
    from customstats import empirical_metadata

    expected = pd.read_excel(FIXTURES / "TABLE_EmpiricalECCMetrics.xlsx", index_col=0)

    # the notebook passes a spawned sub-stream, so this must too
    datasets, _ = empirical.prepare(np.random.default_rng(NOTEBOOK_SEED).spawn(1)[0])
    computed = {mat: empirical_metadata(x, w) for mat, (x, w) in datasets.items()}

    actual = pd.DataFrame(computed).T.loc[expected.index]
    compare_frames(expected, actual[expected.columns], "empirical metrics recomputed")


def test_synthetic_fits_and_w1_recomputed():
    """Drive the fitting and scoring path directly on a deterministic slice."""
    expected_full = pd.read_excel(
        FIXTURES / "TABLE_SyntheticECCMetricsAndW1.xlsx", index_col=0
    )

    data_all = load_json("DATA_all.json")
    sample = list(expected_full.index[:N_SYNTHETIC_SAMPLE])

    rows = {}
    for name in sample:
        x = np.array(data_all[name]["data"], dtype=float)
        weights = np.array(data_all[name]["weights"], dtype=float)
        models = fit_pewt(x, weights)
        # Every PEWT model is scored against the variable-weighted empirical
        # CDF, including the uniform-weighted fits.
        rows[name] = {p: score_w1(models[p], x, weights) for p in PEWT}

    actual = pd.DataFrame(rows).T
    expected = expected_full.loc[sample, PEWT]
    compare_frames(expected, actual, "synthetic W1 recomputed")


def test_synthetic_metrics_recomputed():
    """Recompute empirical_metadata for the same synthetic slice."""
    from customstats import empirical_metadata

    expected_full = pd.read_excel(
        FIXTURES / "TABLE_SyntheticECCMetricsAndW1.xlsx", index_col=0
    )
    metric_columns = [c for c in expected_full.columns if c not in PEWT]

    data_all = load_json("DATA_all.json")
    sample = list(expected_full.index[:N_SYNTHETIC_SAMPLE])

    rows = {}
    for name in sample:
        x = np.array(data_all[name]["data"], dtype=float)
        weights = np.array(data_all[name]["weights"], dtype=float)
        rows[name] = empirical_metadata(x, weights)

    actual = pd.DataFrame(rows).T
    expected = expected_full.loc[sample, metric_columns]
    expected, actual = align_to_fixture(actual, expected)
    compare_frames(expected, actual, "synthetic metrics recomputed")

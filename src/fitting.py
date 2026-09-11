"""
Fitting and scoring for the six UQ methods.

One implementation of the PEWT fit. Before Stage 1 this block existed as three
verbatim copies, in notebook 2 cells 12 and 20 and notebook 3 cell 15, so any
change to the method had to be made in three places and kept in step by hand.
Stage 2 changes the lognormal threshold and the bandwidth rule, which is
exactly the kind of edit that goes wrong when it has to be repeated.

The default constants below reproduce the behaviour as of Stage 1. Both are
under review in Stage 2; see reports/MANUSCRIPT_discrepancies.md entries 6 and
10.
"""

import numpy as np
from scipy.stats import gaussian_kde, lognorm, norm

from customstats import (
    wasserstein1_weighted,
    weighted_bw,
    weighted_lognorm_fit,
    weighted_std,
)

PE_METHODS = ["Normal", "Lognormal", "KDE"]
WT_METHODS = ["Uniform", "Variable"]
PEWT = [f"{pe}, {wt}" for pe in PE_METHODS for wt in WT_METHODS]

# The data are offset by this much before the lognormal fit and the location is
# shifted back afterwards, which is a 3-parameter lognormal with the threshold
# fixed at -LOGFIT_OFFSET rather than estimated. It exists because a handful of
# near-zero values otherwise drag the log-space mean down and inflate sigma,
# collapsing the fitted mode toward zero. That problem is real and is worse in
# the empirical data than in the synthetic data: 28.3% of the 138 empirical ECC
# datasets have a minimum below 1% of their mean, against 1.8% of the synthetic
# datasets. The fixed value is nonetheless arbitrary and is under review.
LOGFIT_OFFSET = 0.5

# 1.06 * sigma * n_eff ** -0.2, which is Scott (1992). Note that
# scipy.stats.gaussian_kde uses the words 'scott' and 'silverman' for different
# formulas: its 'scott' carries no 1.06 factor. Do not describe the method by
# pointing at a scipy keyword.
BW_METHOD = "scott"

# The scoring grid runs from 0 to max(data) + this many standard deviations.
SCORE_GRID_POINTS = 1_000
SCORE_GRID_STD_MULTIPLE = 10


def uniform_weights(x):
    """Equal weight on every data point, summing to one."""
    return np.ones_like(x) / len(x)


def fit_pewt_models(x, weights_variable, logfit_offset=LOGFIT_OFFSET,
                    bw_method=BW_METHOD):
    """Fit all six probability-estimation / weighting combinations.

    Parameters
    ----------
    x : array-like
        One ECC dataset, strictly positive.
    weights_variable : array-like
        Per-point weights for the variable weighting scheme.
    logfit_offset : float
        Positive shift applied before the lognormal fit, undone in the fitted
        location, so the model has support (-logfit_offset, inf).
    bw_method : {'scott', 'silverman'}
        Passed to customstats.weighted_bw.

    Returns
    -------
    dict
        Keys are the six PEWT labels; values are frozen scipy distributions or
        a gaussian_kde, each exposing .pdf().
    """
    x = np.asarray(x, dtype=float)
    weights_variable = np.asarray(weights_variable, dtype=float)

    models = {}
    for weights, wt in zip([uniform_weights(x), weights_variable], WT_METHODS):
        # Normal: weighted mean and weighted standard deviation.
        mean = np.sum(x * weights) / np.sum(weights)
        models[f"Normal, {wt}"] = norm(loc=mean, scale=weighted_std(x, weights))

        # Lognormal: weighted MLE on the offset data, location shifted back.
        shape, loc, scale = weighted_lognorm_fit(
            x + logfit_offset, weights=weights, method="MLE"
        )
        models[f"Lognormal, {wt}"] = lognorm(
            s=shape, loc=loc - logfit_offset, scale=scale
        )

        # KDE: bandwidth computed here, then imposed on the scipy estimator.
        # gaussian_kde is constructed with bw_method=1.0 and rescaled, because
        # its own rules do not account for weights.
        bandwidth = weighted_bw(x, weights, bw_method=bw_method)
        kde = gaussian_kde(x, bw_method=1.0, weights=weights)
        kde.set_bandwidth(bandwidth / (kde.covariance ** 0.5)[0][0])
        models[f"KDE, {wt}"] = kde

    return models


def score_grid(x, weights):
    """The x-grid a fitted model is discretized onto for scoring.

    Runs from 0, so every model is implicitly truncated at zero and
    renormalized. That matches the rejection sampling used in the pLCA, which
    discards non-positive draws, so the model that is scored is the model that
    is sampled.
    """
    x = np.asarray(x, dtype=float)
    spread = np.max([np.std(x), weighted_std(x, weights)])
    return np.linspace(
        0, np.max(x) + SCORE_GRID_STD_MULTIPLE * spread, SCORE_GRID_POINTS
    )


def score_w1(model, x, weights):
    """Wasserstein-1 between a fitted model and the weighted empirical data.

    Every model is scored against the variable-weighted empirical CDF,
    including the uniform-weighted fits.
    """
    grid = score_grid(x, weights)
    return wasserstein1_weighted(x, grid, weights, model.pdf(grid))


def score_all(models, x, weights):
    """W1 for each of the six fits, as a dict keyed by PEWT label."""
    return {label: score_w1(model, x, weights) for label, model in models.items()}

"""How much does the weighting scheme matter, for one dataset? Stage 2d.

Three things live here, and they answer one question at three levels of
commitment.

1. THE DECOMPOSITION. `w_v_uw_wasserstein`, the study's uniform-to-variable
   distance, is bounded below by the absolute difference in the two weighted
   means. `location_shape_split` separates it into that bound and the residual.
   If the bound carries most of it, the practitioner rule collapses to a
   weighted mean, which needs no distributional machinery at all.

2. THE RELATIVE MEASURE. Every dataset in this study is divided by its own
   unweighted mean, so every W1 already reported IS a W1 divided by a mean.
   `relative_scales` makes that explicit and offers two robust alternatives, so
   the choice of denominator is a stated one rather than an accident of
   normalization.

3. A_IQR. The measure from the author's own KL2 paper (Torres, Lupton, Marsh,
   Srubar and Allen, Resources, Conservation and Recycling 234, 109022, 2026):
   sample market-share vectors from the Dirichlet, fit a density under each, and
   take the area between the pointwise 75th and 25th percentile density curves.
   One number per dataset, in density space, where a practitioner reads an error
   band. Using KL2's measure rather than a new one keeps this paper consistent
   with the author's published work, which CLAUDE.md treats as a constraint.

WHAT MAKES THIS DIFFERENT FROM THE SINGLE REALIZATION IT REPLACES. The
characteristic `w_v_uw_wasserstein` is ONE draw from the Dirichlet. Stage 2a-3
measured its draw-to-draw spread at up to 1.02 per dataset, which is larger than
most of the differences the study reports. Everything here is a property of the
DISTRIBUTION of possible weightings instead, so it does not move when the seed
does.
"""
import numpy as np

import fitting
from customstats import (wasserstein1_weighted, weighted_quantile,
                         weighted_std)

# ---------------------------------------------------------------------------
# 1. the decomposition
# ---------------------------------------------------------------------------


def uniform_weights(x):
    n = len(np.asarray(x))
    return np.full(n, 1.0 / n)


def location_shape_split(x, weights):
    """Split W1(uniform-weighted, variable-weighted) into location and shape.

    W1 between two distributions is bounded below by the absolute difference of
    their means, because the optimal transport plan must at minimum move the
    mass far enough to move the mean:

        W1(P, Q) = integral |F_P - F_Q| >= |mean(P) - mean(Q)|

    Here P and Q are the SAME values under two weightings, so the bound is
    |weighted mean - unweighted mean|. Call that the LOCATION component. The
    residual is everything reweighting did that a shift of the mean does not
    describe, and it is non-negative by the inequality.

    WHY THE SPLIT IS WORTH MAKING. If the location component carries most of the
    distance, then a practitioner asking "do market shares matter for my
    category" needs only a weighted mean, which is a spreadsheet column. If the
    residual carries most of it, they need the whole distribution and the
    question is genuinely distributional. The two answers imply very different
    guidance, and nothing in the study had separated them.

    Returns
    -------
    dict with `w1`, `location`, `shape` and `location_share`.

    `shape` is clipped at zero. The inequality is exact in real arithmetic, so a
    negative value is floating-point noise in the quadrature and never a real
    result; `tests/test_weighting.py` pins the size of it.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    u = uniform_weights(x)
    w1 = float(wasserstein1_weighted(x, x, u, w))
    location = abs(float(np.sum(x * w)) - float(np.mean(x)))
    shape = max(0.0, w1 - location)
    return dict(w1=w1, location=location, shape=shape,
                location_share=(location / w1) if w1 > 0 else np.nan)


# ---------------------------------------------------------------------------
# 2. the relative measure
# ---------------------------------------------------------------------------

#: The denominators a W1 on one dataset can be divided by, to make it a relative
#: measure a practitioner can compare across categories.
#:
#: ALL THREE ARE COMPUTED WITH UNIFORM WEIGHTS, and that is the important
#: decision here rather than which of the three is used. A denominator taken
#: under the VARIABLE weights would move when the weights move, which is the
#: quantity being measured, so the ratio would confound the numerator with its
#: own scale. It would also be uncomputable in practice: decision 6 normalizes
#: by the unweighted mean precisely because a practitioner holding a set of EPDs
#: can compute one and cannot compute a market-weighted mean without already
#: knowing the market shares, which is what they lack.
#:
#: `mean` is the study's existing implicit choice. Every dataset is divided by
#: its unweighted mean before anything else happens, so every W1 this study has
#: ever reported is already a W1 divided by a mean, and naming it changes no
#: number.
SCALES = ('mean', 'iqr', 'sd')


def relative_scales(x, weights=None):
    """The three candidate denominators for one dataset.

    `weights` defaults to uniform and should be left that way for anything
    reported; see `SCALES` for why. It is exposed so the weighted denominators
    can be measured and rejected rather than asserted to be worse.
    """
    x = np.asarray(x, dtype=float)
    w = uniform_weights(x) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    q1 = float(weighted_quantile(x, w, 0.25, output='perc2val'))
    q3 = float(weighted_quantile(x, w, 0.75, output='perc2val'))
    return dict(mean=float(np.sum(x * w)), iqr=q3 - q1,
                sd=float(weighted_std(x, w)))


def relativize(w1, scales):
    """One absolute W1 as three relative measures. Non-positive scale -> nan."""
    return {f'rel_{k}': (w1 / scales[k] if scales[k] > 0 else np.nan)
            for k in SCALES}


# ---------------------------------------------------------------------------
# 3. A_IQR, and the ensemble it comes from
# ---------------------------------------------------------------------------

#: Dirichlet draws per dataset. KL2 uses 1,000 -- "Fig. 3a shows 1000 iterations
#: illustrating the range of viable solutions", and the steel proof-of-concept
#: likewise generates 1,000 viable PDFs -- and this study matches it so that the
#: two papers report the same measure and not merely the same name.
#:
#: Measured convergence is in `audits/aiqr_convergence.py`.
N_DRAWS = 1_000

#: Points in the density grid A_IQR is integrated on. A_IQR is an area under a
#: difference of two smooth density curves, so it needs far fewer points than
#: the W1 criterion's grid, which has to resolve a step function.
AIQR_GRID_POINTS = 2_000

#: How far past the data the grid reaches, in standard deviations. The same
#: convention as `fitting.score_grid_open`, so the density is integrated over
#: the same support the study scores on, and the grid is open at zero for the
#: same reason: an ECC of exactly zero is not admissible (decision 13).
AIQR_GRID_STD_MULTIPLE = fitting.SCORE_GRID_STD_MULTIPLE


def aiqr_grid(x, npoints=AIQR_GRID_POINTS, std_multiple=AIQR_GRID_STD_MULTIPLE):
    """The x-lattice A_IQR is integrated on, open at zero."""
    x = np.asarray(x, dtype=float)
    hi = float(np.max(x) + std_multiple * np.std(x))
    return np.linspace(hi / npoints, hi, npoints)


def dirichlet_draws(n, rng, n_draws=N_DRAWS, alpha=1.0):
    """`n_draws` market-share vectors over `n` points, from a flat Dirichlet.

    The same prior the study already uses for its variable weights, so every row
    is an allocation this study considers possible. `alpha` is the CONCENTRATION
    parameter and is 1.0 everywhere in this project (decision 16): a SMALLER
    alpha gives MORE dispersed shares, so a flat Dirichlet is a lower bound on
    how concentrated real market shares are. Marsh, Hattam and Allen (2025)
    report Rest-of-World BOF steel at 63.75 percent of global production against
    an expected top share of 5.2 percent at n = 100 under this prior.
    """
    return rng.dirichlet(np.full(n, float(alpha)), size=int(n_draws))


def density_ensemble(x, draws, grid, bw_method=None):
    """The density each weight vector implies, on a common grid. (n_draws, n_grid).

    One bandwidth per draw, not one for the dataset. That is KL2's construction
    and it is also the only self-consistent one: the bandwidth rule reads the
    weighted standard deviation, the weighted interquartile range and the KISH
    effective sample size, all three of which change when the weights change. A
    concentrated draw genuinely is a smaller effective sample and should be
    smoothed more.

    THE DENSITY IS THE STUDY'S KDE, TRUNCATED TO (0, inf) AND RENORMALIZED, and
    not a bare kernel sum. `fitting.fit_kde` builds the same object the study
    fits, scores and samples from, which decision 13 requires and decision 50
    implemented. It matters here rather than being a formality: a Gaussian
    kernel sitting on a small ECC puts real mass below zero, and on small
    datasets with a concentrated weight draw the bandwidth is wide enough that a
    bare kernel sum loses up to 10 percent of its mass off the bottom of the
    grid. An ensemble of curves integrating to between 0.90 and 1.00 would make
    A_IQR partly a measure of how much mass each draw spilled.
    """
    x = np.asarray(x, dtype=float)
    bw_method = bw_method or fitting.BW_METHOD
    out = np.empty((len(draws), len(grid)), dtype=float)
    for k, w in enumerate(draws):
        out[k] = fitting.fit_kde(x, w, bw_method=bw_method)[0].pdf(grid)
    return out


def aiqr(densities, grid):
    """A_IQR: the area of the interquartile range of an ensemble of densities.

    KL2's definition, quoted: the uncertainty of the mean PDF "is represented by
    the area of the IQR across all viable PDFs, AIQR, which is calculated by
    subtracting the 25th percentile density curve from the 75th percentile
    density curve at each point along the x-axis". The quartiles are therefore
    POINTWISE IN x, which the author confirmed on 2026-09-17, and the result is
    an area under that band.

    IT IS NOT NORMALIZED, AND IT DOES NOT NEED TO BE. Stage 2d read KL2 to
    settle this: the paper reports A_IQR as a bare number (0.40, 0.22 and 0.12
    for its three scenarios) with no divisor. Nor should there be one. A density
    has units of 1 / x, so integrating a difference of two densities over x is
    already dimensionless, and A_IQR is therefore invariant under rescaling the
    data -- multiply every ECC by a constant and the densities shrink by exactly
    the factor the lattice stretches. `tests/test_weighting.py` pins that.

    HOW TO READ IT. A_IQR is the expected width of the error band around the
    density a practitioner would have drawn from uniform weights, integrated
    over the support. Zero means every possible market-share allocation gives
    the same density and the uniform assumption is free; large means the density
    is mostly a statement about a weight vector nobody has measured.
    """
    lo, hi = np.percentile(np.asarray(densities, dtype=float), [25.0, 75.0],
                           axis=0)
    return float(np.trapezoid(hi - lo, grid))


# ---------------------------------------------------------------------------
# what a draw costs, in the units the flip curve is calibrated in
# ---------------------------------------------------------------------------
def model_w1(model_a, model_b, grid):
    """W1 between two fitted models, as the area between their CDFs.

    The same quadrature `fitting.score_w1_model` uses on its trapezoid route,
    on a grid supplied by the caller. There is no empirical CDF in it: this is a
    distance between two things the pLCA could sample from, which is what the
    flip curve needs, and it is the quantity Stage 2d calibrates against.
    """
    fa = np.asarray(model_a.cdf(grid), dtype=float)
    fb = np.asarray(model_b.cdf(grid), dtype=float)
    return float(np.trapezoid(np.abs(fa - fb), grid))


def weighting_separation(x, draws, grid, bw_method=None):
    """How far each possible weighting moves the fitted density, as a W1.

    For every Dirichlet draw, the absolute W1 between the KDE fitted under
    UNIFORM weights -- the model a practitioner builds when they do not know the
    market shares -- and the KDE fitted under that draw. Divide by a scale from
    `relative_scales` to get the relative measure the flip curve is calibrated
    in, then count the share above the calibrated threshold.

    THE COMPARISON IS BETWEEN MODELS, NOT BETWEEN EMPIRICAL CDFs.
    `audits/weighting_risk.py`, the Stage 2c feasibility probe, compared the two
    weighted empirical CDFs instead. That answers a slightly different question
    and, more to the point, it is not the quantity the flip probability is
    calibrated against: the pLCA samples from the fitted models and never from
    the data.
    """
    x = np.asarray(x, dtype=float)
    bw_method = bw_method or fitting.BW_METHOD
    base = fitting.fit_kde(x, uniform_weights(x), bw_method=bw_method)[0]
    fu = np.asarray(base.cdf(grid), dtype=float)
    out = np.empty(len(draws), dtype=float)
    for k, w in enumerate(draws):
        m = fitting.fit_kde(x, w, bw_method=bw_method)[0]
        out[k] = float(np.trapezoid(
            np.abs(np.asarray(m.cdf(grid), dtype=float) - fu), grid))
    return out


def dataset_risk(x, rng, n_draws=N_DRAWS, alpha=1.0, thresholds=(),
                 npoints=AIQR_GRID_POINTS, bw_method=None, scales=None):
    """A_IQR and the weighting risk for one dataset, in a single pass.

    `thresholds` are RELATIVE, in units of the scale each is paired with, and
    come from the flip-probability calibration. Returns one flat dict, so an arm
    is a list comprehension over this.
    """
    x = np.asarray(x, dtype=float)
    grid = aiqr_grid(x, npoints=npoints)
    draws = dirichlet_draws(len(x), rng, n_draws=n_draws, alpha=alpha)
    dens = density_ensemble(x, draws, grid, bw_method=bw_method)
    sc = scales or relative_scales(x)
    sep = weighting_separation(x, draws, grid, bw_method=bw_method)
    row = dict(n=len(x), aiqr=aiqr(dens, grid))
    for k in SCALES:
        rel = sep / sc[k] if sc[k] > 0 else np.full(len(sep), np.nan)
        row[f'sep_median_{k}'] = float(np.median(rel))
        row[f'sep_p90_{k}'] = float(np.quantile(rel, 0.90))
        for name, thr in thresholds:
            row[f'P_{name}_{k}'] = float(np.mean(rel > thr))
    return row


# ---------------------------------------------------------------------------
# is a flat Dirichlet an adequate stand-in for real market shares?
# ---------------------------------------------------------------------------
def block_weights(n, k, rng, adjacent=True):
    """Market share concentrated in `k` groups rather than spread over n points.

    Author's question, 2026-09-17: a flat Dirichlet explores the simplex
    uniformly, but real market share probably arrives in CLUSTERS -- a few
    related products carrying most of the volume. Is a flat draw the right
    model, and if share does cluster, is that just a dataset with fewer points?

    THE SECOND HALF OF THAT IS RIGHT AND THE FIRST HALF IS NOT, which is what
    this function exists to show. Draw group shares from a flat Dirichlet over
    `k` groups and split each equally inside its group.

    `adjacent=True` puts each group on a contiguous run of the SORTED values, so
    products with similar coefficients share their volume, which is what
    clustering means in practice. `adjacent=False` puts the same group sizes on
    randomly chosen points, so the concentration is identical and only the
    coherence is removed. Comparing the two isolates adjacency from
    concentration.

    Measured in `audits/weight_clustering.py` and in notebook 1, at MATCHED
    effective sample size: random concentration is indistinguishable from a flat
    draw (ratio 0.90 to 0.99 across bands), while adjacent concentration
    produces separations 1.5 to 3.1 times larger. So the effective sample size
    does capture concentration -- the author's mechanism is correct -- and it
    does NOT capture coherence. A contiguous block shifts the whole distribution
    one way, which lands in the location term that already carries most of the
    uniform-to-variable distance, whereas random concentration moves mass in
    directions that partly cancel.

    **The consequence for this study is conservative**: a flat Dirichlet
    UNDERSTATES how far real market shares would move a fitted density, so the
    reported weighting risk is a lower bound.
    """
    k = int(max(1, min(k, n)))
    share = rng.dirichlet(np.ones(k))
    order = np.arange(n) if adjacent else rng.permutation(n)
    w = np.zeros(n, dtype=float)
    for s, group in zip(share, np.array_split(order, k)):
        if len(group):
            w[group] = s / len(group)
    total = w.sum()
    return w / total if total > 0 else np.full(n, 1.0 / n)


def effective_n(weights):
    """Kish effective sample size, the denominator concentration actually acts on."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w ** 2))


# ---------------------------------------------------------------------------
# the counterfactual: weights that carry the signal and none of the noise
# ---------------------------------------------------------------------------
def oracle_weights(parent, modes):
    """Mode-level market share, split equally inside each mode.

    NOT A METHOD AND NOT A PROPOSAL. A practitioner cannot compute this: it
    needs the mode each value was drawn from, which only the generator knows.
    It exists to separate two things the study's variable arm confounds. A
    synthetic dataset's weights are built in two steps -- mode k is given its
    true market share, which is signal, and that share is then split among the
    points inside mode k by a flat Dirichlet, which is noise the real world
    does not have, because a real market share is a property of a product and
    not a random draw. This removes the second step and keeps the first.

    **CORRECTED 2026-09-29, and the old wording is the one to unlearn.** This
    docstring used to say the comparison is between KNOWING market shares and
    GUESSING them. It is not, and that phrasing has misled three stages.
    A synthetic dataset's market weights already carry the TRUE market share of
    every product group exactly -- the weight mass on group k equals
    `parent.market[k]` to machine precision, which
    `tests/test_mixedpolicy.py` pins -- so the study's variable arm is USING a
    known market share, not guessing one. What this function removes is only
    the arbitrary division of a group's share among the products inside it.

    **So the comparison it licenses is narrow**: the same known group-level
    share, divided evenly within a group against divided at random. Decision 79
    already recorded that this division is uninformative BY CONSTRUCTION in
    this generator, so a difference here measures a generator artifact and not
    a property of weighting. **The comparison that matters -- what ignoring a
    known market share costs -- is uniform against variable, and needs nothing
    from this function.**

    The same total weight per mode that `generator.draw_weights` targets, with
    the within-mode flat Dirichlet replaced by equal shares. A mode that drew no
    points contributes nothing and the rest are renormalized, which is what the
    realized weights do too.
    """
    modes = np.asarray(modes, int)
    k = len(parent.comps)
    counts = np.bincount(modes, minlength=k).astype(float)
    # `generator.draw_weights` targets `parent.market`, which equals
    # `market_effective` at the configured coupling of 1.0. Mirror the
    # source rather than the identity, so this stays right if coupling moves.
    share = np.asarray(parent.market, float).copy()
    share[counts == 0] = 0.0
    total = share.sum()
    if not total > 0:
        return np.ones(len(modes)) / len(modes)
    share = share / total
    per_point = np.zeros(k)
    nz = counts > 0
    per_point[nz] = share[nz] / counts[nz]
    w = per_point[modes]
    return w / w.sum()


# ---------------------------------------------------------------------------
# ONE market-share rule for BOTH arms, Stage 2h
# ---------------------------------------------------------------------------
#: How many market-share groups a category is cut into, when `k` is not given.
#: THESE ARE THE GENERATOR'S OWN, `genconfig.k_min` and `k_max`: the synthetic
#: arm draws its mixture component count uniformly from 1 to 5, INDEPENDENT of
#: dataset size, and porting that rule means porting that distribution.
BLOCKS_MIN, BLOCKS_MAX = 1, 5


def draw_blocks(n, rng, lo=BLOCKS_MIN, hi=BLOCKS_MAX):
    """How many market-share groups a category is cut into.

    Uniform on 1 to 5 and INDEPENDENT OF n, which is exactly what the
    generator does when it draws a mixture's component count. A real market
    holds a handful of product routes whether the category has nine
    declarations or thirty thousand, and nothing about publishing more
    declarations creates more routes.

    A FIRST VERSION OF THIS GREW THE BLOCK COUNT WITH n, up to 12, and that was
    wrong: it is a different weight model, not the synthetic arm's ported. It
    also broke the thing the port exists to fix, because more groups at large n
    means more dilution at large n, which is the artifact the flat draw already
    had. Measured, it left the two arms further apart on the paper's central
    quantity than they started.

    Stage 2h sweeps `k` directly as well, because the block count changes
    CONCENTRATION while `rho` changes COHERENCE and decision 97 requires the
    two to be separated.
    """
    return int(min(rng.integers(int(lo), int(hi) + 1), max(1, int(n))))


def blocks_for_n(n, per_block=8, lo=2, hi=12):
    """SUPERSEDED by `draw_blocks`, kept so the sweep can ask for it by name.

    A size-growing block count. Not the synthetic arm's rule; see `draw_blocks`
    for why that matters and what it cost.
    """
    return int(np.clip(round(n / float(per_block)), lo, min(hi, max(1, n))))


def coherent_weights(values, rng, k=None, rho=1.0, block_alpha=1.0,
                     point_alpha=1.0, per_block=8, return_blocks=False):
    """Market-share weights that may be CORRELATED with the coefficients.

    THE DEFECT THIS EXISTS TO REMOVE. Until Stage 2h the two halves of this
    study drew market shares by different rules, on the exact dimension the
    paper is built on. The synthetic arm attached a share to each mixture
    component and split it inside the component, so shares were correlated with
    the carbon coefficients. The real categories drew a flat Dirichlet over
    every individual declaration, so shares were INDEPENDENT of them.

    That is not a neutral default, and it is not a small one. Weights drawn
    independently of the values are exchangeable, so the weighted CDF converges
    to the unweighted one and the measured weighting effect MUST decay like
    n^-1/2 whatever the market does. Measured decay of the median
    uniform-to-variable separation on log(n): **-0.397 on the real categories
    against -0.167 on the synthetic**, and above a thousand declarations the
    synthetic arm shows ten times the effect. Real market share does not become
    more uniform as more manufacturers publish declarations, so the decay is a
    property of the WEIGHT MODEL rather than of markets.

    THE RULE, which is the synthetic arm's ported to both. Cut the declarations
    into `k` groups, draw each group's share from a Dirichlet, and split that
    share inside the group by a second Dirichlet. Groups stand in for the
    mixture components the real data does not label.

    HOW A GROUP IS FORMED, and this is the swept axis. Rank the values to
    `r` in (0, 1], draw `u` uniform, and sort by

        s = rho * r + (1 - rho) * u

    then cut `s` into `k` contiguous runs.

        rho = 1   groups are runs of ADJACENT coefficients: share tracks
                  technology, which is what clustering means in practice
        rho = 0   group membership is random: concentration without coherence,
                  which Stage 2d measured to be indistinguishable from a flat
                  draw at matched effective sample size

    **rho = 0 IS NOT THE AGNOSTIC CHOICE and must not be treated as one.** It
    is the specific claim that market share is uncorrelated with carbon
    intensity, and the published production volumes say otherwise: Marsh,
    Hattam and Allen (2025) put 63.75 percent of world steel on the
    higher-carbon Rest-of-World BOF route while the lower-carbon Austrian EAF
    route is 0.03 percent, and KL2's steel example puts 54 percent of global
    production in China alone. Share tracking technology is exactly the
    correlation `rho` measures.

    **A BIGGER SEPARATION IS NOT EVIDENCE OF A BETTER MODEL.** The separation
    measures what unknown shares do; it is not a target. What is a defect,
    and worth fixing at whatever rho is defensible, is that the two arms
    differed at all.

    NO FITTING, AND NO FAILURE MODE AT SMALL n. A mixture model was rejected
    for this: it cannot be estimated at three to nine declarations, mode counts
    on real data swing from 95 to 68 percent unimodal on one smoothing choice,
    and it would put a fitted model inside the paper's central quantity. A
    contiguous cut of the sorted values needs none of that.

    Parameters
    ----------
    values : array
        The declarations. Only their ORDER is used, so the rule is invariant
        under any increasing rescaling, which `tests/test_weighting.py` pins.
    rng : Generator
    k : int or None
        Number of market-share groups. None draws from `draw_blocks`, which is
        the generator's own rule: uniform on 1 to 5, independent of n.
    rho : float in [0, 1]
        Coherence. 1 is maximal clustering by coefficient, 0 is random
        membership.
    block_alpha, point_alpha : float
        Dirichlet concentration between groups and within a group. Smaller is
        more concentrated. `point_alpha` at 1.0 with `k = n` reproduces the old
        flat draw exactly.
    per_block : int
        Unused unless `k='size'`, which asks for the superseded size-growing
        rule so a sweep can measure it.
    return_blocks : bool
        Also return the group index of every declaration, so a caller can sum
        the realized weight per GROUP. That is the quantity a published
        production volume reports -- Marsh, Hattam and Allen (2025)'s 63.75
        percent is a ROUTE's share of world steel, not one declaration's -- so
        the anchor needs it.

    Returns
    -------
    ndarray of weights summing to 1, aligned with `values`; or, with
    `return_blocks`, the pair (weights, block_index).
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return (np.zeros(0), np.zeros(0, dtype=int)) if return_blocks else np.zeros(0)
    if n == 1:
        return (np.ones(1), np.zeros(1, dtype=int)) if return_blocks else np.ones(1)
    if k is None:
        k = draw_blocks(n, rng)
    elif k == 'size':
        k = blocks_for_n(n, per_block=per_block)
    k = int(max(1, min(int(k), n)))
    rho = float(np.clip(rho, 0.0, 1.0))
    # Ranks scaled to (0, 1]. `mergesort` so that ties keep a stable order and
    # the rule is deterministic given the stream.
    r = (np.argsort(np.argsort(v, kind='mergesort'),
                    kind='mergesort') + 1.0) / n
    u = rng.random(n)
    order = np.argsort(rho * r + (1.0 - rho) * u, kind='mergesort')
    share = rng.dirichlet(np.full(k, float(block_alpha)))
    w = np.zeros(n, dtype=float)
    block = np.zeros(n, dtype=int)
    for j, (s, group) in enumerate(zip(share, np.array_split(order, k))):
        if not len(group):
            continue
        within = (rng.dirichlet(np.full(len(group), float(point_alpha)))
                  if len(group) > 1 else np.ones(1))
        w[group] = s * within
        block[group] = j
    total = w.sum()
    w = w / total if total > 0 else np.full(n, 1.0 / n)
    return (w, block) if return_blocks else w


def weight_effect(values, weights):
    """The uniform-to-variable separation: the paper's central quantity.

    W1 between the equal-weighted and the supplied-weight empirical CDFs of the
    SAME values, divided by the dataset's own unweighted mean, which is the
    relative measure decision 93 named and which every W1 in this study has
    always silently been, since every dataset is divided by that mean first.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = float(np.mean(v))
    if not mean > 0:
        return np.nan
    uni = np.full(len(v), 1.0 / len(v))
    return float(wasserstein1_weighted(v, v, uni, w) / mean)

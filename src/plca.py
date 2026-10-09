"""The pLCA construction: common random numbers, group size and material use
intensity. Stage 2e.

Everything the study says about a UQ method's CONSEQUENCE is read off a
probabilistic LCA of four synthetic materials, each normalized to a mean of 1.0
and each carrying a material use intensity of 1.0. Three properties of that
construction were never tested, and all three bound the numbers the paper
reports:

    the SAMPLING        each method drew its own Monte Carlo sample, so two
                        methods were compared under two independent streams
    the GROUP SIZE      four materials, fixed by one shuffle of the corpus
    the INTENSITIES     all equal, which makes the four contributions
                        exchangeable and a ranking as fragile as it can be made

This module supplies the machinery for all three, and for the experiment that
converts a goodness-of-fit score into an error in the answer: running the same
pLCA against the TRUE parent distributions.

    COMMON RANDOM NUMBERS ARE ONE UNIFORM PER MATERIAL PER ITERATION.

`draw_contributions` takes a (neccs, k) block of uniform variates and pushes
column j through material j's fitted model. Independent ACROSS materials within
an iteration, because materials in a building are not rank-correlated;
identical ACROSS methods, which is the whole point. `families.rvs_from_uniform`
is the inverse-CDF map every model in this study exposes, built in Stage 2b for
exactly this and used by `src/flip.py` since Stage 2d.

    MATERIAL USE INTENSITY IS A SHARE VECTOR, NOT A SET OF ABSOLUTE QUANTITIES.

A building total is arbitrary: multiplying every intensity by ten changes no
ranking, no frequency and no share. What matters is each material's share of
the total mean contribution, so the object being swept is a point on the
simplex and the sweep is over a symmetric Dirichlet's concentration. Large
concentration reproduces the equal-intensity case exactly, concentration 1 is
flat on the simplex, and small concentration puts nearly everything on one
material. `mui_dirichlet` and `mui_from_ratio`.

    AND IT IS REPORTED AGAINST AN OBSERVABLE, NOT AGAINST THE CONCENTRATION.

A Dirichlet concentration means nothing to a reader, and what it implies about
dominance changes with the number of materials, which would confound the two
sweeps with each other. Every outcome is reported against the ratio of the
largest mean contribution to the second largest, and against the largest
material's share of the total. Both are one line of a quantity take-off, and
the first is the quantity that decides whether a flip is possible at all.
`contribution_profile`.
"""
import numpy as np
import pandas as pd

import fitting as FT

#: Monte Carlo draws per material per pLCA, as everywhere in this study.
NECCS = 10_000

#: Materials per pLCA, swept. Four is the study's own construction.
GROUP_SIZES = (2, 3, 4, 6, 8, 12)

#: Concentration of the symmetric Dirichlet the intensity shares are drawn
#: from. Large is nearly equal, 1.0 is flat on the simplex, small puts nearly
#: all of a building's impact on one material. `mui_dirichlet` also accepts
#: `np.inf`, which returns the equal vector exactly; it is not listed here
#: because the sweep gets that case from the named 1:1 checkpoint instead, and
#: running it twice under two labels would put one cell in the table twice.
MUI_CONCENTRATIONS = (200.0, 50.0, 20.0, 8.0, 3.0, 1.0, 0.4, 0.15)

#: Deterministic checkpoints, as the leading material's intensity against every
#: other material at 1. Reproducible, interpretable, and they bracket the
#: random draws: 1 is the study's own case and 100 is total dominance.
MUI_RATIOS = (1.0, 2.0, 10.0, 100.0)

#: Outputs compared between methods. Every one is a quantity notebook 3 already
#: computes; they are gathered in one place so they can be compared on equal
#: terms. The names match the columns of `TABLE_PLCAResults.csv`.
#:
#: `eci_perc_p95tot` was added in Stage 2g as a magnitude-based companion to
#: the rank metric: each material's share of the total in the iterations where
#: the BUILDING sits at its 95th percentile. Adding it consumes no randomness,
#: so it is a new column beside the existing ones and moves none of them.
OUTPUTS = ('eci_mean', 'eci_std', 'eci_cov', 'eci_perc_mean', 'eci_perc_std',
           'eci_perc_p95tot', 'eci_rank_1', 'eci_rank_4', 'eci_meanrank',
           'eci_p95', 'ui')

#: Where on the BUILDING TOTAL's own distribution `eci_perc_p95tot` is read.
TOTAL_QUANTILE = 0.95

#: Half-width of the window around that quantile, in quantile units. One
#: iteration is one draw and would be pure noise; +/- 0.01 holds 200 of this
#: study's 10,000 iterations, which is enough to average over and narrow enough
#: that the total inside it really is at its 95th percentile.
TOTAL_WINDOW = 0.01


def share_at_total_quantile(draws, quantile=TOTAL_QUANTILE,
                            window=TOTAL_WINDOW):
    """Each material's share of the total, in the iterations where the TOTAL is
    at its `quantile`. Stage 2g.

    Parameters
    ----------
    draws : ndarray, shape (neccs, k)
        One pLCA sample: material j's contribution in each iteration.
    quantile, window : float
        The iterations used are those whose total falls between the
        `quantile - window` and `quantile + window` quantiles of the total.

    Returns
    -------
    ndarray, shape (k,), summing to 1.0

    WHY THIS AND NOT `eci_p95`. `eci_p95` is the 95th percentile of a
    material's OWN contribution, taken over its own marginal distribution, and
    the iteration that puts material A at its 95th percentile is usually not
    the iteration that puts the BUILDING at its 95th percentile. A carbon
    budget is written against the building, so the attribution question at the
    bad end of the building's distribution has to read the shares in the
    iterations where the building is actually there. Those are different
    iterations and, where the materials differ in spread, a different answer.

    It is a SHARE, so it is bounded in [0, 1] and sums to one across the
    materials of a pLCA, which is what makes it comparable across group sizes
    in a way `eci_p95` is not.
    """
    draws = np.asarray(draws, dtype=float)
    total = draws.sum(axis=1)
    lo, hi = float(quantile - window), float(quantile + window)
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f'the window [{lo}, {hi}] is not inside [0, 1]')
    a, b = np.quantile(total, [lo, hi])
    pick = (total >= a) & (total <= b)
    if not pick.any():
        # Only reachable if the total is degenerate, in which case every
        # iteration is at every quantile and the whole sample is the window.
        pick = np.ones(len(total), dtype=bool)
    got = draws[pick].sum(axis=0)
    denom = got.sum()
    return got / denom if denom > 0 else np.full(draws.shape[1], np.nan)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def draw_contributions(models, names, method, u, mui=None):
    """One pLCA sample: material j's contribution in each Monte Carlo iteration.

    Parameters
    ----------
    models : dict
        `models[dataset][method]`, each exposing `rvs_from_uniform`.
    names : sequence
        The datasets in this group, in a fixed order.
    method : str
        Which of the six UQ methods to push the variates through.
    u : ndarray, shape (neccs, len(names))
        Column j is material j's uniform stream. SHARED across methods.
    mui : ndarray or None
        Material use intensity per material. None is 1.0 for every material,
        which is the study's own construction.

    Returns
    -------
    ndarray, shape (neccs, len(names))

    Every dataset is normalized to an unweighted mean of 1.0 (decision 6), so a
    material's mean contribution is its intensity and nothing else. That is
    what makes the intensity vector the ONLY source of between-material
    variation in contribution, and therefore the axis worth sweeping.
    """
    u = np.asarray(u, dtype=float)
    draws = np.column_stack([
        np.asarray(models[d][method].rvs_from_uniform(u[:, j]), dtype=float)
        for j, d in enumerate(names)])
    if mui is None:
        return draws
    return draws * np.asarray(mui, dtype=float)[None, :]


def outputs(draws):
    """Every pLCA output this study reports, from one (neccs, k) sample.

    Returns a dict of name -> per-material vector, one entry per name in
    `OUTPUTS`. The definitions are notebook 3's, moved here so that a sweep and
    the study's own table cannot drift apart; `tests/test_plca.py` checks
    `eci_rank_1` against the pandas ranking the notebook uses.

    `ui` is the uncertainty index: one minus the share of the building total's
    variance that survives replacing this material's draw by its median. It is
    the output Stage 2d measured as the least sensitive to the choice of UQ
    method, and the one a practitioner would act on when deciding where to
    collect better data.

    `eci_perc_p95tot` is Stage 2g's magnitude companion to the rank metric:
    each material's share of the total in the iterations where the BUILDING
    total sits at its 95th percentile. It is the attribution question asked at
    the end of the distribution a carbon budget is written against, and it is
    not `eci_p95`, which is the 95th percentile of the material's own
    contribution taken over its own marginal.
    """
    draws = np.asarray(draws, dtype=float)
    k = draws.shape[1]
    total = draws.sum(axis=1)
    order = (-draws).argsort(axis=1).argsort(axis=1)
    var_total = np.var(total)
    med = np.median(draws, axis=0)
    ui = np.array([1.0 - np.var(total - draws[:, j] + med[j]) / var_total
                   if var_total > 0 else np.nan for j in range(k)])
    perc = draws / total[:, None]
    return {
        'eci_mean': draws.mean(axis=0),
        'eci_std': draws.std(axis=0),
        'eci_cov': draws.std(axis=0) / draws.mean(axis=0),
        'eci_perc_mean': perc.mean(axis=0),
        'eci_perc_std': perc.std(axis=0),
        'eci_perc_p95tot': share_at_total_quantile(draws),
        'eci_rank_1': (order == 0).mean(axis=0),
        'eci_rank_4': (order == k - 1).mean(axis=0),
        'eci_meanrank': order.mean(axis=0) + 1.0,
        'eci_p95': np.quantile(draws, 0.95, axis=0),
        'ui': ui,
    }


def run_group(models, names, u, mui=None, methods=None):
    """`{method: outputs(...)}` for one group, every method on the same `u`."""
    methods = methods or FT.PEWT
    return {m: outputs(draw_contributions(models, names, m, u, mui))
            for m in methods}


def top_contributor(out, names):
    """The material with the highest rank-1 frequency: the pLCA's answer.

    STATE THE NOISE FLOOR WITH ANY RESULT READ OFF THIS. It is an argmax over
    four nearly equal frequencies, so under independent random streams it can
    land elsewhere without any number moving meaningfully (decision 103). Under
    common random numbers two identical models give identical draws and the
    floor is exactly zero, which is why every comparison here shares `u`.
    """
    return names[int(np.argmax(out['eci_rank_1']))]


# ---------------------------------------------------------------------------
# material use intensity
# ---------------------------------------------------------------------------
def mui_equal(k):
    """The study's own construction: every material at 1.0."""
    return np.ones(int(k), dtype=float)


def mui_dirichlet(k, concentration, rng):
    """Intensity shares from a symmetric Dirichlet, scaled to a mean of 1.0.

    The shares are what matter, so the vector is drawn on the simplex and then
    multiplied by `k`, which leaves the average material at 1.0 and makes the
    swept cases directly comparable with the study's equal case rather than
    comparable up to a building-sized constant.

    `concentration = np.inf` returns the equal vector exactly, so the sweep's
    own control needs no separate code path and cannot drift from the study.
    """
    k = int(k)
    if not np.isfinite(concentration):
        return mui_equal(k)
    share = rng.dirichlet(np.full(k, float(concentration)))
    return share * k


def mui_from_ratio(k, ratio):
    """One leading material at `ratio`, every other at 1, mean scaled to 1.0.

    The deterministic checkpoints of `MUI_RATIOS`. Reported as named cases
    because a reader can picture 10:1:1:1 and cannot picture a Dirichlet draw.
    """
    v = np.ones(int(k), dtype=float)
    v[0] = float(ratio)
    return v * len(v) / v.sum()


def contribution_profile(mui, means=None):
    """What a practitioner can compute from a quantity take-off, in one line.

    Returns `top2_ratio`, the largest mean contribution divided by the second
    largest, and `top_share`, the largest material's share of the total.

    THE PROFILE IS A PROPERTY OF THE DESIGN, NOT OF THE UQ METHOD, and that is
    deliberate. It is computed from the intensities and the datasets' own
    unweighted means, both of which are fixed before any distribution is
    fitted, so it cannot move with the method being tested. Every dataset in
    this study has an unweighted mean of exactly 1.0 (decision 6), so `means`
    defaults to that and the profile is a property of the intensity vector
    alone.

    `top2_ratio` is the quantity that decides whether a flip is possible: if
    the leading material's mean contribution is well clear of the next, no
    plausible difference between two fitted distributions can reorder them.
    """
    mui = np.asarray(mui, dtype=float)
    c = mui if means is None else mui * np.asarray(means, dtype=float)
    s = np.sort(c)[::-1]
    return dict(top2_ratio=float(s[0] / s[1]) if len(s) > 1 and s[1] > 0
                else np.inf,
                top_share=float(s[0] / s.sum()) if s.sum() > 0 else np.nan)


# ---------------------------------------------------------------------------
# groupings
# ---------------------------------------------------------------------------
def resample_groups(names, k, n_groups, rng):
    """`n_groups` groups of `k` distinct datasets, drawn WITH replacement.

    The study partitions the corpus once, by a single shuffle, into disjoint
    groups of four. That is one grouping, so every headline it produces has no
    uncertainty attached to the grouping itself; and it fixes the group size at
    whatever divides the corpus.

    Resampling instead gives a bootstrap over groupings. Datasets are distinct
    WITHIN a group -- the same material twice is not a building -- and groups
    are independent of each other, so a dataset may appear in several. The
    resampling unit downstream is therefore the GROUP, as it is in
    `flip.bootstrap_crossings` and for the same reason.
    """
    names = np.asarray(list(names), dtype=str)
    k, n_groups = int(k), int(n_groups)
    if k > len(names):
        raise ValueError(f'cannot draw {k} distinct datasets from {len(names)}')
    out = np.empty((n_groups, k), dtype=names.dtype)
    for i in range(n_groups):
        out[i] = rng.choice(names, size=k, replace=False)
    return out


# ---------------------------------------------------------------------------
# uncertainty on every headline
# ---------------------------------------------------------------------------
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_ALPHA = 0.05


def cluster_bootstrap(frame, value, cluster='plca', statistic='mean',
                      resamples=BOOTSTRAP_RESAMPLES, alpha=BOOTSTRAP_ALPHA,
                      rng=None):
    """A percentile interval for one aggregate, resampling CLUSTERS not rows.

    Every headline this study reports is an average over rows that are not
    independent: the fifteen method pairs inside one pLCA share its materials
    and its random variates, and the four materials share its total. Resampling
    rows would treat those as independent observations and return an interval
    several times too narrow; `tests/test_flip.py` pins that difference for the
    flip calibration and the same argument applies to every percentage here.

    `statistic` is 'mean', 'median' or a callable on the value array.
    """
    rng = rng or np.random.default_rng(0)
    v = pd.to_numeric(frame[value], errors='coerce').to_numpy(dtype=float)
    g = frame[cluster].to_numpy()
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if len(v) == 0:
        return dict(statistic=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0,
                    n_clusters=0)
    fn = (statistic if callable(statistic)
          else {'mean': np.mean, 'median': np.median}[statistic])
    uniq, inverse = np.unique(g, return_inverse=True)
    index = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    point = float(fn(v))
    draws = np.empty(int(resamples), dtype=float)
    for r in range(int(resamples)):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([index[p] for p in pick])
        draws[r] = fn(v[rows])
    return dict(statistic=point,
                ci_lo=float(np.nanpercentile(draws, 100 * alpha / 2)),
                ci_hi=float(np.nanpercentile(draws, 100 * (1 - alpha / 2))),
                n=int(len(v)), n_clusters=int(len(uniq)))


def nrmse(frame, value, by='method', cluster='plca', unit='dataset'):
    """The study's NRMSE between UQ methods, from a tidy table.

    Notebook 3 computes this inside a plotting function: for one output, the
    root mean squared difference between every ORDERED pair of methods on the
    same material, divided by the standard deviation of that output over every
    material and method. It is the single number the paper uses to say how far
    apart the six methods are on a given output.

    Reproduced here so it can be computed from a table rather than from a
    figure, and so an interval can be attached to it. `tests/test_plca.py`
    checks it against the notebook's own arithmetic.
    """
    return _nrmse_from(_wide(frame, value, by, cluster, unit)[0])


def _wide(frame, value, by, cluster, unit):
    """(values, group labels, method names), one row per material."""
    w = frame.pivot_table(index=[cluster, unit], columns=by, values=value)
    return (w.to_numpy(dtype=float),
            w.index.get_level_values(cluster).to_numpy(), list(w.columns))


def _nrmse_from(wide):
    """Root mean squared difference over ORDERED method pairs, over the spread.

    The spread is the standard deviation of the whole table -- every material
    under every method -- which is what makes the number comparable across
    outputs on different scales.
    """
    if wide.shape[1] < 2:
        return np.nan
    d = wide[:, :, None] - wide[:, None, :]
    off = ~np.eye(wide.shape[1], dtype=bool)
    sd = np.nanstd(wide)
    return float(np.sqrt(np.nanmean(d[:, off] ** 2)) / sd) if sd > 0 else np.nan


# ---------------------------------------------------------------------------
# the study's own pLCA, under common random numbers
# ---------------------------------------------------------------------------
#: Where a specifier sets the cap, as a quantile of the values they hold.
CAP_QUANTILE = 0.75


def specification_cap(x, quantile=CAP_QUANTILE, scale=1.0):
    """The ABSOLUTE ECC cap for one material, read off the data and nothing else.

    THIS REPLACED A CAP TAKEN FROM EACH METHOD'S OWN DRAWS, AND THE OLD FORM
    COULD NOT MEASURE WHAT THE STRATEGY IS FOR. Capping every method at the
    75th percentile of its OWN sample gives each method a different absolute
    cap, so the six are no longer being asked about the same intervention; and
    because each is capped at its own 75th percentile, exactly 25 percent of
    iterations are capped under every method by construction. That forces the
    signal to zero. A method that understates the upper tail SHOULD conclude
    that capping buys less, and under the old form it could not.

    One absolute cap per material, applied to every method and to the true
    parent, restores it. The quantile is taken on the VALUES a practitioner
    holds, unweighted, because that is what a specifier can compute -- "I will
    accept no product above the 75th percentile of the declarations I have" --
    and because it exists on the empirical arm, where no parent does.

    `scale` is the material's use intensity, since the draws carry it.
    """
    return float(np.quantile(np.asarray(x, dtype=float), quantile) * scale)


def cap_reduction(model, col, cap, u, scale=1.0):
    """The specification strategy: every draw at or above an ABSOLUTE cap is
    replaced by one from the model CONDITIONED on being below the cap.

    EXACTLY, BY INVERSE CDF, AND NOT BY REDRAWING UNTIL IT LANDS. Redrawing
    until a value falls below the cap samples from the model conditioned on
    being below it, so `ppf(u * F(cap))` is the same distribution in one step.
    Three things follow, and the third is why the loop is gone rather than
    merely tidied:

        it is the project's own rule. Decision 50 settled that sampling is by
        inverse CDF and never by rejection, and this redraw loop was the last
        rejection sampler left in the study

        it is paired across methods with ONE uniform block, where the loop
        needed a cache of variates indexed by redraw pass

        it cannot fail. A model with little mass below the cap needs
        unboundedly many redraws, and a bounded loop returns values that are
        still above the cap; this returns the conditional draw whenever the
        conditioning event has any probability at all, and says so when it does
        not. A full run hit exactly that case

    `cap` is an absolute value from `specification_cap`, NOT a quantile of this
    method's own draws. `scale` is the material's use intensity, because `col`
    carries it and the model does not.

    Returns `(reduced, touched, cap)`. Where the model puts NO mass below the
    cap the conditional distribution does not exist, and those entries come
    back as NaN rather than as a value above a cap that is labeled as capped.
    """
    col = np.asarray(col, dtype=float)
    cap = float(cap)
    scale = float(scale)
    u = np.asarray(u, dtype=float)
    touched = col >= cap
    red = col.copy()
    if not touched.any():
        return red, touched, cap
    mass_below = float(np.ravel(model.cdf(cap / scale))[0])
    if mass_below <= 0.0:
        red[touched] = np.nan
        return red, touched, cap
    red[touched] = scale * np.asarray(
        model.ppf(u[touched] * mass_below), dtype=float)
    return red, touched, cap


# ---------------------------------------------------------------------------
# the pLCA against the TRUTH
# ---------------------------------------------------------------------------
#: Which parent is the truth for a pLCA. The MARKET-weighted mixture, because a
#: probabilistic LCA of what actually gets built is a statement about the
#: population of products weighted by how much of each is produced, and that is
#: the one population all six methods can be scored against on equal terms
#: (decision 65). The sampling distribution is reported beside it as
#: `SAMPLING_SCHEME`, which is what a uniform-weighted method is
#: estimating, so a reader can see how much of a method's error is definitional
#: rather than an error of estimation.
TRUTH_SCHEME = 'market'
SAMPLING_SCHEME = 'uniform'


class ParentSampler:
    """A `mixture.MixtureParent` as a sampler with the study's model interface.

    The parent's own `ppf` is a 200-step bisection on a closed-form mixture
    CDF, which costs a scipy CDF evaluation per component per step. Drawing
    10,000 values from 10,000 parents that way is hours. This tabulates the CDF
    once on a fixed grid across the parent's own truncated support and inverts
    by interpolation, which is what `families.WeightedKDE` does for the same
    reason and to the same end: the object that is SAMPLED is the object that
    is scored.

    Everything is in NORMALIZED units, the units the datasets and the fitted
    models live in, because `MixtureParent.cdf` divides by the realized
    unweighted sample mean and its `lo` and `hi` attributes do not.
    """

    #: Points across the parent's support. The parent is a smooth truncated
    #: mixture, so linear interpolation of its CDF is second-order accurate;
    #: `tests/test_plca.py` pins the error against the exact bisection.
    GRID = 20_001

    def __init__(self, parent, scheme=TRUTH_SCHEME, npoints=GRID):
        c = float(parent.normalizer)
        lo, hi = float(parent.lo) / c, float(parent.hi) / c
        grid = self._grid(parent, scheme, lo, hi, int(npoints))
        cdf = np.asarray(parent.cdf(grid, scheme), dtype=float)
        # Strictly increasing, so np.interp inverts it without ties. Same
        # construction as WeightedKDE._tabulate.
        cdf = np.maximum.accumulate(cdf)
        cdf = cdf + np.arange(len(cdf)) * 1e-15
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        self.grid, self._cdf = grid, cdf
        self.lo, self.hi = float(grid[0]), float(grid[-1])
        self.parent, self.scheme = parent, scheme

    #: Probability mass to keep OUTSIDE the concentrated part of the grid. The
    #: body gets most of the points and the two tails keep enough to stay
    #: accurate where a Monte Carlo will actually land. At 1e-4 a 10,000-draw
    #: run expects one value beyond each end.
    TAIL_MASS = 1e-4

    #: Grid points reserved for each tail, placed at the parent's own
    #: quantiles between TAIL_MASS and TAIL_MIN_MASS.
    TAIL_POINTS = 128

    #: How far into each tail the quantile-spaced points reach. Beyond this the
    #: parent carries less probability than any Monte Carlo in this study would
    #: ever draw, so the two true bounds close the grid and nothing is lost.
    TAIL_MIN_MASS = 1e-12

    @staticmethod
    def _grid(parent, scheme, lo, hi, npoints, tail_mass=None):
        """A grid that resolves the BODY, whatever the truncation bounds are.

        **A PLAIN `linspace(lo, hi)` IS WHAT THIS REPLACES, AND IT FAILED
        SILENTLY.** The parent's truncation bounds are set by a multiplicative
        rule whose width grows exponentially in the data's log spread, so `hi`
        is usually a small multiple of the body and occasionally enormous.
        Stage 2h generated a corpus whose parents had `hi` near 1e8; a linear
        grid of 20,001 points across that has a spacing of about 18,000, so the
        entire body -- every value between roughly 0.1 and 10 -- fell between
        the first two grid points. The tabulated CDF became a step function,
        inverting it returned draws spread over the whole support, and the
        resulting "true" mean was 6,624 against data normalized to 1.0.

        **NOTHING UPSTREAM WAS WRONG.** The parents were mathematically sound
        under both weighting schemes, with means near 1.02 and maxima near 2.3.
        Only this approximation of them broke, which is why every check on the
        generator and on the parents themselves came back clean.

        The fix is to spend the points where the mass is. One coarse pass finds
        where the CDF leaves 0 and reaches 1; the fine grid then concentrates
        there and keeps a few points beyond, so the far tail is still
        representable but no longer consumes the whole budget.
        """
        tail_mass = ParentSampler.TAIL_MASS if tail_mass is None else tail_mass
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            return np.linspace(lo, hi, npoints)
        # THE ENDPOINTS COME FROM THE PARENT'S OWN QUANTILES, not from its
        # truncation bounds. `lo` and `hi` bound the SUPPORT; the parent can
        # carry essentially no mass near `hi` while `hi` itself is enormous,
        # because the truncation rule's width grows exponentially in the data's
        # log spread. Spending the grid on [lo, hi] then spends it on emptiness.
        # `ppf` is a bisection and costs a CDF evaluation per step, which is why
        # this class exists at all -- but TWO calls to set the endpoints is
        # nothing against the 20,001 the tabulation saves.
        try:
            q_lo = float(np.ravel(parent.ppf(tail_mass, scheme))[0])
            q_hi = float(np.ravel(parent.ppf(1.0 - tail_mass, scheme))[0])
        except Exception:
            return np.linspace(lo, hi, npoints)
        if not (np.isfinite(q_lo) and np.isfinite(q_hi) and q_hi > q_lo):
            return np.linspace(lo, hi, npoints)
        # THE TAILS ARE QUANTILE-SPACED TOO, and that is not a refinement. A
        # LINEAR run of points from `q_hi` out to `hi` spreads the remaining
        # `tail_mass` of real probability evenly across a range that can be
        # eight orders of magnitude wide, so inverting it hands back draws of
        # order 1e8 that the parent never places there. Putting the points at
        # the parent's own quantiles instead puts them where the mass is: the
        # parent's 1 - 1e-6 quantile is about 33, not 1e8.
        n_body = int(npoints) - 2 * ParentSampler.TAIL_POINTS
        p_tail = np.geomspace(tail_mass, ParentSampler.TAIL_MIN_MASS,
                              ParentSampler.TAIL_POINTS)
        try:
            low_tail = np.ravel(parent.ppf(p_tail[::-1], scheme)).astype(float)
            high_tail = np.ravel(parent.ppf(1.0 - p_tail, scheme)).astype(float)
        except Exception:
            low_tail = np.linspace(lo, q_lo, ParentSampler.TAIL_POINTS)
            high_tail = np.linspace(q_hi, hi, ParentSampler.TAIL_POINTS)
        grid = np.concatenate([[lo], low_tail,
                               np.linspace(q_lo, q_hi, n_body),
                               high_tail, [hi]])
        grid = grid[np.isfinite(grid)]
        return np.unique(np.clip(grid, lo, hi))
        self.lo, self.hi = lo, hi

    def cdf(self, x):
        return np.interp(np.atleast_1d(np.asarray(x, dtype=float)),
                         self.grid, self._cdf, left=0.0, right=1.0)

    def ppf(self, q):
        return np.interp(np.atleast_1d(np.asarray(q, dtype=float)),
                         self._cdf, self.grid)

    def rvs_from_uniform(self, u):
        return self.ppf(u)

    def rvs(self, size, random_state):
        return self.ppf(random_state.random(size))


def parent_samplers(parents, scheme=TRUTH_SCHEME, npoints=ParentSampler.GRID):
    """`{dataset: ParentSampler}` for every parent supplied.

    For a handful of parents. Use `LazySamplers` for a whole corpus: see its
    docstring for the arithmetic that makes the difference matter.
    """
    return {d: ParentSampler(p, scheme=scheme, npoints=npoints)
            for d, p in parents.items()}


class LazySamplers:
    """`ParentSampler` objects built on demand and held in a bounded cache.

    WHY THIS IS NOT A PREMATURE OPTIMIZATION. A ParentSampler holds three
    arrays of `GRID` doubles, which is about half a megabyte; the corpus has
    just under 10,000 parents and the truth run wants two schemes, so building
    them all would ask for roughly ten gigabytes on top of the twenty thousand
    fitted models the notebook is already holding. A pLCA group needs four of
    them at a time, and the study's groups are disjoint, so a small cache costs
    nothing and bounds the memory.

    Indexing is by dataset name, so this is a drop-in for the dict every other
    entry point takes.
    """

    def __init__(self, parents, scheme=TRUTH_SCHEME, npoints=ParentSampler.GRID,
                 maxsize=64):
        self.parents = parents
        self.scheme = scheme
        self.npoints = int(npoints)
        self.maxsize = int(maxsize)
        self._cache = {}
        self._order = []

    def __getitem__(self, dataset):
        got = self._cache.get(dataset)
        if got is None:
            got = ParentSampler(self.parents[dataset], scheme=self.scheme,
                                npoints=self.npoints)
            self._cache[dataset] = got
            self._order.append(dataset)
            while len(self._order) > self.maxsize:
                del self._cache[self._order.pop(0)]
        return got

    def __contains__(self, dataset):
        return dataset in self.parents

    def __len__(self):
        return len(self.parents)


def truth_rows(models, samplers, names, u, mui=None, methods=None,
               plca=None, extra=None):
    """One row per (material, method): every output, and the true value.

    The truth is the same pLCA run with each material's TRUE parent in place of
    its fitted model, on the SAME uniform variates, so the difference between a
    method's answer and the truth contains no Monte Carlo noise at all: it is
    the error the fitted model causes and nothing else.

    This is the experiment that converts every score in this study from "how
    far is the fitted curve from a target" into "how wrong is the answer a
    practitioner gets".
    """
    methods = methods or FT.PEWT
    names = list(names)
    truth = outputs(draw_contributions(
        {d: {'__truth__': samplers[d]} for d in names},
        names, '__truth__', u, mui))
    rows = []
    for m in methods:
        got = outputs(draw_contributions(models, names, m, u, mui))
        for j, d in enumerate(names):
            row = dict(plca=plca, dataset=d, method=m, material_index=j)
            if extra:
                row.update(extra)
            row['is_top'] = bool(np.argmax(got['eci_rank_1']) == j)
            row['is_top__truth'] = bool(np.argmax(truth['eci_rank_1']) == j)
            for key in OUTPUTS:
                row[key] = float(got[key][j])
                row[f'{key}__truth'] = float(truth[key][j])
                row[f'{key}__error'] = float(got[key][j] - truth[key][j])
            rows.append(row)
    return rows


def pair_rows(per_method, names, methods=None, plca=None, extra=None):
    """One row per method pair: did the answer change, and by how much.

    `per_method` is what `run_group` returns. Two figures per output are
    carried and they answer different questions. `<output>_max` is the change
    for the MOST-AFFECTED material, which is the one a practitioner is
    deciding about and the figure Stage 2d reports; `<output>_mean` averages
    over the materials in the group, which is the figure that stays comparable
    as the group grows, because a maximum over twelve materials is drawn from
    more chances than a maximum over two.

    `flip_top` is whether the two methods name a different largest contributor.
    Both methods share the group's uniform variates, so it carries no Monte
    Carlo noise: it moves only when the models differ.
    """
    from itertools import combinations
    methods = methods or FT.PEWT
    names = list(names)
    rows = []
    for a, b in combinations(methods, 2):
        row = dict(plca=plca, method_a=a, method_b=b)
        if extra:
            row.update(extra)
        row['flip_top'] = bool(top_contributor(per_method[a], names)
                               != top_contributor(per_method[b], names))
        for key in OUTPUTS:
            d = np.abs(np.asarray(per_method[a][key], float)
                       - np.asarray(per_method[b][key], float))
            row[f'{key}_max'] = float(np.nanmax(d))
            row[f'{key}_mean'] = float(np.nanmean(d))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# the crossed sweep
# ---------------------------------------------------------------------------
def mui_cases(ratios=MUI_RATIOS, concentrations=MUI_CONCENTRATIONS):
    """The intensity settings the sweep runs, as (label, kind, parameter).

    The deterministic checkpoints come first, because a reader can picture
    10:1:1:1 and cannot picture a Dirichlet draw, and because they bracket the
    random cases at both ends: 1:1 is the study's own construction and 100:1 is
    one material carrying almost the whole building.
    """
    out = [(f'{r:g}:1', 'ratio', float(r)) for r in ratios]
    out += [(f'dirichlet {c:g}', 'dirichlet', float(c)) for c in concentrations]
    return out


def make_mui(kind, parameter, k, rng):
    """One intensity vector for one group, from a case of `mui_cases`."""
    if kind == 'ratio':
        return mui_from_ratio(k, parameter)
    if kind == 'dirichlet':
        return mui_dirichlet(k, parameter, rng)
    raise ValueError(f'unknown intensity case {kind!r}')


def sweep(models, pool, rng, sizes=GROUP_SIZES, cases=None, n_groups=400,
          neccs=NECCS, methods=None, sizes_of=None, cv_of=None, progress=None):
    """Every (group size, intensity) cell, one row per pLCA and method pair.

    THE TWO SWEEPS ARE CROSSED AND NOT RUN SEPARATELY, because they interact:
    adding a material raises the chance of a near-tie when the contributions
    are even, and barely moves it when the added material is small.

    `pool` is the datasets groups are resampled from, `sizes_of` maps a dataset
    to its number of values so that a row can be post-stratified afterwards.
    Every group at every cell gets a fresh uniform block, and every method in a
    group shares it.

    Returns a frame whose predictor columns are the two OBSERVABLES --
    `top2_ratio` and `top_share` -- rather than the Dirichlet concentration,
    which means nothing to a reader and whose implied dominance changes with
    the number of materials.

    `cv_of` maps a dataset to its coefficient of variation, and the group's
    dispersion is carried through so the ratio is not asked to do the work
    alone. IT ALMOST CERTAINLY CANNOT: the ratio is built from MEAN
    contributions, and two materials whose means sit two to one apart can still
    trade places if their spreads are wide enough. The columns recorded are the
    dispersion of the leading material, of the runner-up, and of the group.
    """
    methods = methods or FT.PEWT
    cases = cases if cases is not None else mui_cases()
    rows = []
    cells = [(k, c) for k in sizes for c in cases]
    it = enumerate(cells)
    if progress is not None:
        it = progress(it, total=len(cells))
    for cell_index, (k, (label, kind, parameter)) in it:
        groups = resample_groups(pool, k, n_groups, rng)
        for gi, g in enumerate(groups):
            names = list(g)
            mui = make_mui(kind, parameter, k, rng)
            u = rng.random((int(neccs), k))
            per = run_group(models, names, u, mui, methods)
            extra = dict(nmats=k, mui_case=label, mui_kind=kind,
                         mui_parameter=parameter, **contribution_profile(mui))
            if sizes_of is not None:
                extra['n_min'] = int(min(sizes_of[d] for d in names))
            if cv_of is not None:
                # Ordered by mean contribution, so `cv_lead` and `cv_second`
                # belong to the two materials whose order a flip exchanges.
                order = np.argsort(-np.asarray(mui, dtype=float))
                cvs = np.array([float(cv_of[d]) for d in names])
                extra['cv_lead'] = float(cvs[order[0]])
                extra['cv_second'] = float(cvs[order[1]]) if k > 1 else np.nan
                extra['cv_mean'] = float(np.mean(cvs))
                extra['cv_max'] = float(np.max(cvs))
                # What decides a trade of places is the spread of the two
                # materials relative to the gap between their means, so the
                # pair's dispersion is carried as one number too.
                extra['cv_pair'] = float(np.sqrt(
                    cvs[order[0]] ** 2 + cvs[order[1]] ** 2)) if k > 1 else (
                    float(cvs[order[0]]))
            rows += pair_rows(per, names, methods,
                              plca=f'{cell_index}_{gi}', extra=extra)
    frame = pd.DataFrame(rows)
    # The flip probability FALLS as the leading material pulls away, and every
    # curve-fitting tool in this project -- the logistic on a log predictor, the
    # isotonic fit, the bootstrap that inverts a crossing -- is written for a
    # probability that RISES with its predictor. The reciprocal of the ratio
    # turns one into the other exactly, so the same tested machinery reads this
    # curve, and a crossing is reported back as `1 / crossing`.
    frame['inv_top2_ratio'] = 1.0 / frame.top2_ratio
    return frame


def sweep_summary(frame, by=('nmats', 'mui_case'), outputs_=None,
                  resamples=BOOTSTRAP_RESAMPLES, rng=None):
    """One row per cell: the flip rate and each output's shift, with intervals.

    Every percentage this study reports should carry an interval, and none of
    the pLCA ones did. The interval is a cluster bootstrap over pLCA groups,
    because the fifteen method pairs inside a group share its materials and its
    variates.
    """
    rng = rng or np.random.default_rng(0)
    outputs_ = outputs_ or ('eci_mean_max', 'eci_p95_max', 'eci_rank_1_max',
                            'ui_max', 'eci_perc_mean_max')
    rows = []
    for key, g in frame.groupby(list(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key), n_pairs=len(g), n_plca=g.plca.nunique(),
                   top2_ratio=float(g.top2_ratio.median()),
                   top_share=float(g.top_share.median()))
        b = cluster_bootstrap(g, 'flip_top', resamples=resamples, rng=rng)
        row.update(flip_top=b['statistic'], flip_top_lo=b['ci_lo'],
                   flip_top_hi=b['ci_hi'])
        for col in outputs_:
            if col not in g:
                continue
            b = cluster_bootstrap(g, col, statistic='median',
                                  resamples=resamples, rng=rng)
            row[col] = b['statistic']
            row[f'{col}_lo'] = b['ci_lo']
            row[f'{col}_hi'] = b['ci_hi']
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(by)).reset_index(drop=True)


def nrmse_ci(frame, value, by='method', cluster='plca', unit='dataset',
             resamples=BOOTSTRAP_RESAMPLES, alpha=BOOTSTRAP_ALPHA, rng=None):
    """`nrmse` with a percentile interval, resampling pLCA GROUPS.

    The study reports an NRMSE for every pLCA output and attaches no
    uncertainty to any of them, which is awkward in a paper about uncertainty.
    The resampling unit is the group because the four materials of a pLCA share
    its total and its variates.

    The table is pivoted ONCE and the bootstrap resamples rows of the result. A
    group drawn twice contributes its rows twice, which is what a replicate is;
    pivoting inside the loop would cost 2,000 pivots and would average the
    duplicate away instead.
    """
    rng = rng or np.random.default_rng(0)
    wide, groups, _ = _wide(frame, value, by, cluster, unit)
    point = _nrmse_from(wide)
    uniq, inverse = np.unique(groups, return_inverse=True)
    index = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    draws = np.empty(int(resamples), dtype=float)
    for r in range(int(resamples)):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([index[p] for p in pick])
        draws[r] = _nrmse_from(wide[rows])
    return dict(nrmse=point,
                ci_lo=float(np.nanpercentile(draws, 100 * alpha / 2)),
                ci_hi=float(np.nanpercentile(draws, 100 * (1 - alpha / 2))),
                n_clusters=int(len(uniq)), n_units=int(len(wide)))


def nrmse_table(frame, values, by='method', cluster='plca', unit='dataset',
                resamples=BOOTSTRAP_RESAMPLES, rng=None, progress=None):
    """`nrmse_ci` for every output, as one table."""
    rng = rng or np.random.default_rng(0)
    it = values
    if progress is not None:
        it = progress(values, total=len(values))
    rows = []
    for v in it:
        rows.append(dict(output=v, **nrmse_ci(frame, v, by=by, cluster=cluster,
                                              unit=unit, resamples=resamples,
                                              rng=rng)))
    return pd.DataFrame(rows).sort_values(
        'nrmse', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# the truth run
# ---------------------------------------------------------------------------
#: Fraction of a material's quantity a reduction strategy removes, as in the
#: study's own `matred`.
MATERIAL_REDUCTION = 0.25


def intervention_rows(draws, caps, u_cap, models, names, method, mui,
                      matred=MATERIAL_REDUCTION):
    """Statement 4 for one group: what each intervention delivers, per material.

    Two strategies, both expressed as a fraction of the WHOLE BUILDING's total,
    because that is the number a designer commits to -- "capping the concrete
    cuts the building by six percent" -- and not as a fraction of the material.

        specification  every draw at or above an ABSOLUTE cap is redrawn. The
                       cap is the same value for every method and for the
                       truth, so the six are asked about one intervention
        quantity       the material's contribution is reduced by a fixed
                       fraction, which needs no redraw

    A third intervention is already computed elsewhere and is not repeated
    here: collapsing a material's uncertainty by obtaining a supplier-specific
    declaration is exactly what the uncertainty index measures, and it is the
    only one of the family that reduces the VARIANCE of the answer rather than
    its level.
    """
    base = draws.sum(axis=1)
    out = []
    for j, d in enumerate(names):
        scale = 1.0 if mui is None else float(np.asarray(mui, float)[j])
        red, touched, cap = cap_reduction(models[d][method], draws[:, j],
                                          caps[j], u_cap[:, j], scale=scale)
        capped_total = base - draws[:, j] + red
        row = dict(cap_value=caps[j],
                   cap_share_touched=float(np.mean(touched)),
                   cap_unreachable=float(np.mean(~np.isfinite(red))))
        row.update(reduction_statement(base, capped_total, prefix='cap_'))
        row.update(reduction_statement(
            base, base - float(matred) * draws[:, j], prefix='qty_'))
        out.append(row)
    return out


def truth_run(models, samplers, combos, rng, neccs=NECCS, methods=None,
              mui_of=None, progress=None, sizes_of=None, data=None,
              statements=False):
    """Every pLCA group, run with the fitted models and with the TRUE parents.

    One row per (group, material, method), carrying each output, the value the
    true parent gives for it, and the difference. Both are drawn from the SAME
    uniform block, so the difference is the error the fitted model causes and
    contains no Monte Carlo noise whatever.

    `mui_of(k, rng)` supplies the intensity vector if the run is not at equal
    intensity; `sizes_of` maps a dataset to its number of values so a row can
    be post-stratified.
    """
    methods = methods or FT.PEWT
    rows, group_rows = [], []
    it = enumerate(combos)
    if progress is not None:
        it = progress(it, total=len(combos))
    for i, g in it:
        names = list(g)
        mui = None if mui_of is None else mui_of(len(names), rng)
        u = rng.random((int(neccs), len(names)))
        extra = dict(nmats=len(names))
        if mui is not None:
            extra.update(contribution_profile(mui))
        rows += truth_rows(models, samplers, names, u, mui, methods,
                           plca=i, extra=extra)
        if not statements:
            continue
        # Statements 1 and 4, which are properties of the BUILDING and of an
        # intervention on it rather than of a material's own numbers.
        if data is None:
            raise ValueError('statements=True needs `data` for the cap, which '
                             'is read off the values a specifier holds')
        scale = np.ones(len(names)) if mui is None else np.asarray(mui, float)
        caps = [specification_cap(data[d]['data'], scale=scale[j])
                for j, d in enumerate(names)]
        tmod = {d: {'__truth__': samplers[d]} for d in names}
        truth_draws = draw_contributions(tmod, names, '__truth__', u, mui)
        u_cap = rng.random((int(neccs), len(names)))
        truth_iv = intervention_rows(truth_draws, caps, u_cap, tmod,
                                     names, '__truth__', mui)
        truth_total = truth_draws.sum(axis=1)
        for m in methods:
            d_m = draw_contributions(models, names, m, u, mui)
            row = dict(plca=i, method=m, **extra)
            row.update(building_statement(d_m.sum(axis=1), truth_total))
            group_rows.append(row)
            got = intervention_rows(d_m, caps, u_cap, models, names, m, mui)
            for j, dname in enumerate(names):
                iv = dict(plca=i, dataset=dname, method=m)
                for key, val in got[j].items():
                    iv[key] = val
                    iv[f'{key}__truth'] = truth_iv[j][key]
                    iv[f'{key}__error'] = val - truth_iv[j][key]
                group_rows[-1].setdefault('_interventions', []).append(iv)
    frame = pd.DataFrame(rows)
    if sizes_of is not None:
        frame['n'] = frame.dataset.map(sizes_of)
    if not statements:
        return frame
    iv_rows = [iv for r in group_rows for iv in r.pop('_interventions', [])]
    gframe = pd.DataFrame(group_rows)
    iframe = pd.DataFrame(iv_rows)
    if sizes_of is not None and len(iframe):
        iframe['n'] = iframe.dataset.map(sizes_of)
    return frame, gframe, iframe


def truth_summary(frame, outputs_=None, rng=None,
                  resamples=BOOTSTRAP_RESAMPLES, by=('method',)):
    """Per method: how far its answer sits from the truth, with intervals.

    The mean ABSOLUTE error is reported rather than the mean error, because a
    method that is too high on half the materials and too low on the other half
    is not accurate; and the mean SIGNED error is carried beside it, because a
    method that is systematically high is a different failure from one that is
    merely noisy.
    """
    rng = rng or np.random.default_rng(0)
    outputs_ = outputs_ or ('eci_rank_1', 'eci_mean', 'eci_p95', 'ui',
                            'eci_perc_mean')
    rows = []
    for key, g in frame.groupby(list(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key), n_rows=len(g), n_plca=g.plca.nunique())
        row['names_true_top'] = float((g.is_top == g.is_top__truth)[
            g.is_top__truth].mean())
        for k in outputs_:
            e = g[f'{k}__error']
            sub = g.assign(_abs=e.abs())
            b = cluster_bootstrap(sub, '_abs', resamples=resamples, rng=rng)
            row[f'{k}_abs_error'] = b['statistic']
            row[f'{k}_abs_error_lo'] = b['ci_lo']
            row[f'{k}_abs_error_hi'] = b['ci_hi']
            row[f'{k}_signed_error'] = float(e.mean())
            row[f'{k}_rmse'] = float(np.sqrt(np.mean(e.to_numpy() ** 2)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(by)).reset_index(drop=True)


def truth_win_share(frame, value='eci_rank_1', by=(), rng=None,
                    resamples=BOOTSTRAP_RESAMPLES):
    """How often each method is CLOSEST to the truth, per material.

    A win share rather than a mean error, for the reason Stage 2c gives for
    the empirical headline: a mean of a signed distance inherits the noise of
    every material, while a count moves only when the winner moves.

    Carries a cluster-bootstrap interval over pLCA groups, because a win share
    is a headline percentage and every headline percentage in this stage has
    one.
    """
    f = frame.assign(_err=frame[f'{value}__error'].abs())
    keys = list(by)
    idx = (f.groupby(keys + ['plca', 'dataset'], observed=True)['_err']
           .transform('min') == f['_err'])
    f['_win'] = idx.astype(float)
    wins = f[idx]
    c = wins.groupby(keys + ['method'], observed=True).size().rename('n_wins')
    c = c.reset_index()
    total = (float(len(wins)) if not keys
             else c.groupby(keys, observed=True)['n_wins'].transform('sum'))
    c['win_share'] = c.n_wins / total
    if rng is not None:
        lo, hi = [], []
        for _, row in c.iterrows():
            sub = f[f.method == row['method']]
            for k in keys:
                sub = sub[sub[k] == row[k]]
            b = cluster_bootstrap(sub, '_win', resamples=resamples, rng=rng)
            lo.append(b['ci_lo'])
            hi.append(b['ci_hi'])
        c['win_share_lo'], c['win_share_hi'] = lo, hi
    return c.sort_values(keys + ['win_share'],
                         ascending=[True] * len(keys) + [False]
                         ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# what a probabilistic LCA is FOR: five statements, and how wrong each one is
# ---------------------------------------------------------------------------
#: Fractional reductions a designer would ask an intervention to deliver, for
#: the "and how likely is it" half of the statement.
REDUCTION_TARGETS = (0.05, 0.10, 0.20)

#: Quantiles of the TRUE building total at which the compliance statement is
#: read. A threshold set at the truth's own q is one the truth meets with
#: probability q exactly, so a method's answer can be compared against a number
#: that needs no external budget: "you believe you have a 90 percent chance of
#: coming in under this figure; you actually have X".
COMPLIANCE_QUANTILES = (0.5, 0.9)

#: Multipliers for the modified comparison index. 1.0 is the plain
#: discernibility index, "how often is A below B"; above 1.0 asks how often A
#: beats B by a margin worth acting on. Marsh et al. (in press) use 1.2 and
#: discuss 1.05.
COMPARISON_MARGINS = (1.0, 1.05, 1.2)


def _step_integral(a, b, power):
    """Integral of |F_a - F_b| ** power over x, for two empirical samples.

    Both CDFs are right-continuous step functions that only change at the
    pooled sample points, so the integral is exact as a sum of rectangles and
    needs no grid. Written out rather than taken from a library because two
    different powers are wanted and the second has no library function.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    pooled = np.union1d(a, b)
    if len(pooled) < 2:
        return 0.0
    fa = np.searchsorted(a, pooled, side='right') / len(a)
    fb = np.searchsorted(b, pooled, side='right') / len(b)
    width = np.diff(pooled)
    return float(np.sum(np.abs(fa[:-1] - fb[:-1]) ** power * width))


def w1_samples(a, b):
    """W1 between two samples: the area between their empirical CDFs.

    The study's own criterion, applied to the pLCA OUTPUT rather than to the
    ECC data. That is the point of using it here rather than anything else: it
    puts the input-side score and the output-side error in the same units, so
    the paper can ask whether a better fit to the data produces a better
    building total.
    """
    return _step_integral(a, b, 1.0)


def cramer_distance(a, b):
    """Integral of (F_a - F_b) ** 2: the L2 sibling of W1.

    WHY THIS AND NOT CRPS. The continuous ranked probability score grades a
    predictive distribution against a single observed VALUE, and the truth here
    is itself a distribution. Taking the expectation of CRPS over outcomes drawn
    from the true distribution G gives the integral of (F - G) ** 2 plus a term
    that depends only on G, so ranking methods by expected CRPS against a
    distributional truth IS ranking them by this. It is reported beside W1 as
    the robustness check, the way overlap area was in Stage 2c.
    """
    return _step_integral(a, b, 2.0)


def building_statement(method_total, truth_total,
                       quantiles=COMPLIANCE_QUANTILES):
    """Statement 1: the building total, and how wrong the method is about it.

    Both samples come from the same uniform variates, so nothing here contains
    Monte Carlo noise. Returns the distance between the two distributions, the
    error at two readable quantiles, and the error in the COMPLIANCE statement:
    a threshold placed at the truth's own q is met by the truth with
    probability q, so `p_error_q` is the gap between what a practitioner would
    believe and what is true.
    """
    m = np.asarray(method_total, dtype=float)
    t = np.asarray(truth_total, dtype=float)
    out = dict(total_w1=w1_samples(m, t), total_cramer=cramer_distance(m, t),
               total_mean=float(m.mean()), total_mean__truth=float(t.mean()),
               total_sd=float(m.std()), total_sd__truth=float(t.std()))
    out['total_mean__error'] = out['total_mean'] - out['total_mean__truth']
    out['total_sd__error'] = out['total_sd'] - out['total_sd__truth']
    for q in quantiles:
        mq, tq = float(np.quantile(m, q)), float(np.quantile(t, q))
        out[f'total_q{q:g}'] = mq
        out[f'total_q{q:g}__truth'] = tq
        out[f'total_q{q:g}__error'] = mq - tq
        # The compliance statement, read at a threshold the TRUTH meets with
        # probability q by construction.
        out[f'p_below_q{q:g}'] = float((m <= tq).mean())
        out[f'p_below_q{q:g}__truth'] = q
        out[f'p_below_q{q:g}__error'] = float((m <= tq).mean()) - q
    return out


def reduction_statement(base_total, new_total, targets=REDUCTION_TARGETS,
                        prefix=''):
    """Statement 4: what an intervention delivers, and how likely it is to.

    The mean reduction is what the study already reports. The rest is the half
    it throws away: the probability that the intervention delivers AT LEAST a
    given fraction, which is the form a designer commits to.
    """
    base = np.asarray(base_total, dtype=float)
    new = np.asarray(new_total, dtype=float)
    ok = np.isfinite(base) & np.isfinite(new) & (base != 0)
    rel = (base[ok] - new[ok]) / base[ok]
    if rel.size == 0:
        # The model puts no mass below the cap, so the intervention has no
        # distribution to report. NaN rather than an exception, and the share
        # of materials in this state is carried beside it as `cap_unreachable`.
        out = {f'{prefix}reduction_mean': np.nan,
               f'{prefix}reduction_sd': np.nan,
               f'{prefix}reduction_p05': np.nan}
        for t in targets:
            out[f'{prefix}p_reduction_over_{int(round(100 * t))}'] = np.nan
        return out
    out = {f'{prefix}reduction_mean': float(rel.mean()),
           f'{prefix}reduction_sd': float(rel.std()),
           f'{prefix}reduction_p05': float(np.quantile(rel, 0.05))}
    for t in targets:
        out[f'{prefix}p_reduction_over_{int(round(100 * t))}'] = float(
            (rel >= t).mean())
    return out


def comparison_statement(total_a, total_b, margins=COMPARISON_MARGINS):
    """Statement 5: is option A better than option B, and by enough to act on.

    `margin` 1.0 is the discernibility index of Heijungs (2021), the share of
    Monte Carlo iterations in which A comes out below B.

    **READ THE DIRECTION BEFORE USING A MARGIN.** The quantity is
    `P(a < g * b)`, so `g` ABOVE one LOOSENS the test -- "A is better, or worse
    by less than g" -- and `g` BELOW one tightens it to "A beats B by at least
    `1 - g`". The study's own margins of 1.05 and 1.2 are therefore
    TOLERANCES, which is what Marsh et al. (in press) report at 1.2, and a
    certification credit of the form "demonstrate a 10 percent reduction" is
    `g = 0.90`. An earlier version of this docstring called the g > 1 case "the
    share in which A beats B by a margin worth acting on", which describes
    g < 1; Stage 2h caught it on the study's own output, where `mci_1.2` reads
    0.9993 at a true 20 percent saving against a discernibility of 0.9628 --
    the looser condition, not the stricter one.

    THE TWO OPTIONS MUST BE DRAWN ON THE SAME VARIATES for their shared
    materials, which is dependent sampling and is what Henriksson et al. (2015)
    and Heijungs (2021) require of a comparative probabilistic LCA. This
    function takes the totals; `swap_options` is what builds them that way.
    """
    a = np.asarray(total_a, dtype=float)
    b = np.asarray(total_b, dtype=float)
    out = dict(mean_difference=float((a - b).mean()),
               mean_difference_relative=float((a - b).mean() / b.mean()))
    for g in margins:
        key = 'discernibility' if g == 1.0 else f'mci_{g:g}'
        out[key] = float((a < g * b).mean())
    return out


#: Expected savings a design swap is asked to deliver, as a fraction of the
#: whole building's mean total. Zero is the control: two options that differ in
#: which product they use but not in expected impact, where any discernibility
#: a method reports is coming from shape alone.
SWAP_SAVINGS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)


def swap_totals(models, shared, alt_a, alt_b, u, method, saving=0.0,
                n_materials=None):
    """Two design options differing in ONE material, drawn on shared variates.

    Option A uses `alt_a` as its last material and option B uses `alt_b`. The
    materials they have in common take the SAME uniform variates in both
    options, which is dependent sampling: it is what Henriksson et al. (2015)
    and Heijungs (2021) require of a comparative probabilistic LCA and what
    Marsh et al. (in press) do, and without it the comparison inherits a
    sampling difference that has nothing to do with the design.

    WHY THE INTENSITY CARRIES THE SAVING AND NOT THE DATASET. Every dataset in
    this study is normalized to a mean of 1.0, so replacing one material with
    another changes the expected total by NOTHING and the two options would
    differ only in shape. That is not a design decision anyone makes. The
    replacement's use intensity therefore carries the difference: option B's
    alternative is set so that B's expected total is `saving` lower than A's,
    as a fraction of the whole building. This is also the realistic reading,
    since a substitute product generally needs a different quantity.

    `u` has one column per shared material plus one for each alternative, so
    the two alternatives are independent of each other as two different
    products must be.
    """
    base, a, b, k = swap_components(models, shared, alt_a, alt_b, u, method,
                                    n_materials)
    return swap_from_components(base, a, b, k, saving)


def swap_components(models, shared, alt_a, alt_b, u, method, n_materials=None):
    """The three draw blocks a swap needs, which do NOT depend on the saving.

    Split out of `swap_totals` in Stage 2j. The saving enters only through
    option B's use intensity, so drawing the blocks once and applying every
    saving to them is bit-identical at a sixth of the cost -- and the sweep
    that stage runs scores twenty-three policies where the study scored six,
    which made the redundancy worth removing. `swap_totals` still exists and
    still returns what it always did; `tests/test_plca.py` pins that the two
    routes agree exactly.
    """
    shared = list(shared)
    k = int(n_materials or (len(shared) + 1))
    u = np.asarray(u, dtype=float)
    cols = [np.asarray(models[d][method].rvs_from_uniform(u[:, j]), float)
            for j, d in enumerate(shared)]
    a = np.asarray(models[alt_a][method].rvs_from_uniform(u[:, len(shared)]),
                   float)
    b = np.asarray(models[alt_b][method].rvs_from_uniform(u[:, len(shared) + 1]),
                   float)
    base = np.sum(cols, axis=0) if cols else np.zeros(len(a))
    return base, a, b, k


def swap_from_components(base, a, b, k, saving=0.0):
    """Option A's and option B's totals at one claimed saving.

    Every material sits at an intensity of 1.0 except option B's alternative,
    which is lowered so that B's expected total is `saving` below A's as a
    share of the k-material building.
    """
    return base + a, base + (1.0 - float(saving) * int(k)) * b


def swap_run(models, groups, rng, savings=SWAP_SAVINGS, neccs=NECCS,
             methods=None, samplers=None, progress=None,
             margins=COMPARISON_MARGINS):
    """Statement 5, over many option pairs and many claimed savings.

    Each row of `groups` is `k + 1` datasets: the first `k - 1` are shared
    between the two options, then option A's distinctive material, then option
    B's. Returns one row per (pair, saving, method) carrying the discernibility
    index and the modified comparison index, and, when `samplers` is given, the
    same quantities under the TRUE parents and the error in each.
    """
    methods = methods or FT.PEWT
    groups = np.asarray(groups)
    k = groups.shape[1] - 1
    rows = []
    it = enumerate(groups)
    if progress is not None:
        it = progress(it, total=len(groups))
    for i, g in it:
        names = list(g)
        shared, alt_a, alt_b = names[:k - 1], names[k - 1], names[k]
        u = rng.random((int(neccs), k + 1))
        truth = {}
        if samplers is not None:
            tmod = {d: {'__truth__': samplers[d]} for d in names}
            gbase, ga, gb, gk = swap_components(tmod, shared, alt_a, alt_b, u,
                                                '__truth__', n_materials=k)
            for sv in savings:
                ta, tb = swap_from_components(gbase, ga, gb, gk, sv)
                truth[sv] = comparison_statement(tb, ta, margins=margins)
        for m in methods:
            # The draw blocks do not depend on the saving, so they are drawn
            # ONCE per (pair, policy) and every saving is applied to them.
            cbase, ca, cb, ck = swap_components(models, shared, alt_a, alt_b,
                                                u, m, n_materials=k)
            for sv in savings:
                ta, tb = swap_from_components(cbase, ca, cb, ck, sv)
                # B against A, so a HIGH discernibility means the substitution
                # is judged an improvement, which is the direction a designer
                # reads.
                row = dict(pair=i, method=m, saving=sv, nmats=k,
                           **comparison_statement(tb, ta, margins=margins))
                if sv in truth:
                    for key, val in truth[sv].items():
                        row[f'{key}__truth'] = val
                        row[f'{key}__error'] = row[key] - val
                rows.append(row)
    return pd.DataFrame(rows)

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
OUTPUTS = ('eci_mean', 'eci_std', 'eci_cov', 'eci_perc_mean', 'eci_perc_std',
           'eci_rank_1', 'eci_rank_4', 'eci_meanrank', 'eci_p95', 'ui')


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
    wide = frame.pivot_table(index=[cluster, unit], columns=by, values=value)
    cols = list(wide.columns)
    se = []
    for a in cols:
        for b in cols:
            if a == b:
                continue
            se.append((wide[a] - wide[b]).to_numpy(dtype=float) ** 2)
    se = np.concatenate(se)
    sd = np.nanstd(wide.to_numpy(dtype=float))
    return float(np.sqrt(np.nanmean(se)) / sd) if sd > 0 else np.nan


# ---------------------------------------------------------------------------
# the study's own pLCA, under common random numbers
# ---------------------------------------------------------------------------
def cap_reduction(model, col, capecc, uniform_for_pass, scale=1.0):
    """The ECC-cap strategy, with the redraws taken from supplied uniforms.

    The study's cap strategy replaces every Monte Carlo draw at or above the
    material's own `capecc` sample quantile, redrawing until the column lies
    entirely below its own cap. The algorithm is unchanged here; only where the
    variates come from is. `uniform_for_pass(p)` must return the same
    (neccs,) vector for pass `p` however many times it is asked and whichever
    method is asking, so that two methods redrawing the same iteration on the
    same pass use the same variate.

    `scale` is the material's use intensity, because `col` carries it and a
    redraw off the model does not. Returns `(reduced, touched, cap)`.

    WHY THE PASSES ARE INDEXED. Within one pass the variate for an iteration is
    fixed, so a draw that lands above the cap again must be given a DIFFERENT
    variate or the loop cannot terminate. Each pass therefore has its own
    vector, shared across methods.
    """
    col = np.asarray(col, dtype=float)
    cap = float(np.quantile(col, capecc))
    red = col.copy()
    touched = np.zeros(len(col), dtype=bool)
    above = red >= cap
    p = 0
    while above.any():
        touched |= above
        u = np.asarray(uniform_for_pass(p), dtype=float)
        red[above] = np.asarray(model.rvs_from_uniform(u[above]),
                                dtype=float) * float(scale)
        above = red >= cap
        p += 1
    return red, touched, cap


class PassUniforms:
    """A per-group cache of uniform vectors, one per redraw pass.

    Handed to `cap_reduction` so that every method in a group draws the same
    variate for the same (iteration, pass). Vectors are created on demand and
    kept, because how many passes a method needs depends on its own fitted
    model and the methods must not consume each other's stream.
    """

    def __init__(self, rng, neccs, n_materials):
        self.rng = rng
        self.neccs = int(neccs)
        self.k = int(n_materials)
        self._cache = {}

    def __call__(self, material, pass_index):
        key = (int(material), int(pass_index))
        if key not in self._cache:
            self._cache[key] = self.rng.random(self.neccs)
        return self._cache[key]

    def for_material(self, material):
        return lambda p: self(material, p)


# ---------------------------------------------------------------------------
# the pLCA against the TRUTH
# ---------------------------------------------------------------------------
#: Which parent is the truth for a pLCA. The MARKET-weighted mixture, because a
#: probabilistic LCA of what actually gets built is a statement about the
#: population of products weighted by how much of each is produced, and that is
#: the one population all six methods can be scored against on equal terms
#: (decision 65). The sampling mixture is reported beside it as
#: `TRUTH_SCHEME_SAMPLING`, which is what a uniform-weighted method is
#: estimating, so a reader can see how much of a method's error is definitional
#: rather than an error of estimation.
TRUTH_SCHEME = 'market'
TRUTH_SCHEME_SAMPLING = 'uniform'


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
        grid = np.linspace(lo, hi, int(npoints))
        cdf = np.asarray(parent.cdf(grid, scheme), dtype=float)
        # Strictly increasing, so np.interp inverts it without ties. Same
        # construction as WeightedKDE._tabulate.
        cdf = np.maximum.accumulate(cdf)
        cdf = cdf + np.arange(len(cdf)) * 1e-15
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        self.grid, self._cdf = grid, cdf
        self.parent, self.scheme = parent, scheme
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
    """`{dataset: ParentSampler}` for every parent supplied."""
    return {d: ParentSampler(p, scheme=scheme, npoints=npoints)
            for d, p in parents.items()}


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
          neccs=NECCS, methods=None, sizes_of=None, progress=None):
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
    """
    rng = rng or np.random.default_rng(0)
    groups = frame[cluster].to_numpy()
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    point = nrmse(frame, value, by=by, cluster=cluster, unit=unit)
    draws = np.empty(int(resamples), dtype=float)
    for r in range(int(resamples)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[g] for g in pick])
        sub = frame.iloc[rows].copy()
        # A resampled group may appear several times, so the (group, material)
        # key is no longer unique and the pivot would average the copies
        # together. Numbering the copies keeps them as separate pLCAs, which is
        # what a bootstrap replicate is.
        sub['_copy'] = np.concatenate(
            [np.full(len(index[g]), i) for i, g in enumerate(pick)])
        sub['_cluster'] = sub[cluster].astype(str) + '_' + sub['_copy'].astype(str)
        draws[r] = nrmse(sub, value, by=by, cluster='_cluster', unit=unit)
    return dict(nrmse=point,
                ci_lo=float(np.nanpercentile(draws, 100 * alpha / 2)),
                ci_hi=float(np.nanpercentile(draws, 100 * (1 - alpha / 2))),
                n_clusters=int(len(uniq)))


# ---------------------------------------------------------------------------
# the truth run
# ---------------------------------------------------------------------------
def truth_run(models, samplers, combos, rng, neccs=NECCS, methods=None,
              mui_of=None, progress=None, sizes_of=None):
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
    rows = []
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
    frame = pd.DataFrame(rows)
    if sizes_of is not None:
        frame['n'] = frame.dataset.map(sizes_of)
    return frame


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


def truth_win_share(frame, value='eci_rank_1', by=()):
    """How often each method is CLOSEST to the truth, per material.

    A win share rather than a mean error, for the reason Stage 2c gives for
    the empirical headline: a mean of a signed distance inherits the noise of
    every material, while a count moves only when the winner moves.
    """
    f = frame.assign(_err=frame[f'{value}__error'].abs())
    keys = list(by)
    idx = (f.groupby(keys + ['plca', 'dataset'], observed=True)['_err']
           .transform('min') == f['_err'])
    wins = f[idx]
    tot = wins.groupby(keys, observed=True).size().rename('n') if keys else None
    c = wins.groupby(keys + ['method'], observed=True).size().rename('n_wins')
    c = c.reset_index()
    total = (len(wins) if not keys
             else c.groupby(keys, observed=True)['n_wins'].transform('sum'))
    c['win_share'] = c.n_wins / (total if keys else float(len(wins)))
    return c.sort_values(keys + ['win_share'],
                         ascending=[True] * len(keys) + [False]
                         ).reset_index(drop=True)

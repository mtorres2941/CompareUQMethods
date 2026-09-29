"""Stage 2j: let the UQ method vary BY MATERIAL, on one number.

WHY THIS STAGE EXISTS. Every simulated building in this study fits ONE method
to all four of its materials, so a material's own goodness-of-fit advantage is
averaged against three neighbours drawn at random from the whole corpus, most
of them below the size where that advantage exists. Stage 2h measured the
consequence: the kernel estimate overtakes the three-parameter lognormal on FIT
at about 81 declarations and does not pull clear on the downstream CLAIMS until
about ten times that (decision 166). Letting the method vary by material should
recover much of the gap, and unlike every other policy this study evaluates it
is one a practitioner can follow one material at a time, without knowing
anything about the other three.

THE RULE IS ONE NUMBER AND THE STUDY ALREADY HAD IT.

    at or above 81 declarations   kernel estimate, market weights
    below 81 declarations         three-parameter lognormal, uniform weights

81 is the practitioner threshold of decision 142, calibrated on the superseded
corpus and reproduced UNCHANGED on the current one with the band of equally
good choices at 68 to 106 (decision 198). Both the family and the weighting
switch at the same line, because decision 161 found both orderings invert at
about 100 declarations: equal weights win every size band below and market
weights win every band above.

THERE IS NO SECOND SELECTOR AND THERE MUST NOT BE ONE. Every other
characteristic was tested across two stages and none yields a usable
threshold; modality as a selector is measurably WORSE than not selecting at all
(decisions 88 and 139). A rule with two numbers in it is not the deliverable.

NOTHING IS REFITTED. The mixed policy is a NEW KEY on each dataset's existing
`models[dataset]` dict, pointing at whichever of the six already-fitted models
the rule selects, so the existing pLCA, truth run and scorecard machinery carry
it through unchanged.
"""

import numpy as np
import pandas as pd

import fitting as FT

# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------
#: Declarations at or above which the kernel estimate is used. Decision 142,
#: reproduced unchanged on the regenerated corpus by decision 198. **It is not
#: re-derived here.** What this stage does instead is SWEEP it, because the
#: author's instruction is that the paper publish a range rather than a point:
#: "we made assumptions, so we shouldn't claim 81 is a precisely correct
#: cutoff value".
MIXED_THRESHOLD = 81

#: The two fixed policies the study's rule switches between, as stored `method`
#: values. The stored spellings keep "Uniform" and "Variable" because they are
#: the join key for every table and fixture in this project; the DISPLAY
#: vocabulary is "uniform weights" and "market weights", from
#: `fitting.WT_DISPLAY`.
LARGE_METHOD = 'KDE, Variable'
SMALL_METHOD = 'Lognormal, Uniform'

#: What the study's rule is called in a stored `method` column. Kept as a bare
#: name rather than folded into the swept labels, because every table and
#: figure written before the sweep existed joins on it.
MIXED = 'Mixed'
MIXED_DISPLAY = f'size rule (n >= {MIXED_THRESHOLD})'

#: An unreachable per-material ORACLE, carried as a CEILING and never as a
#: policy. It picks, for each material and each output, whichever of the six
#: FIXED methods happens to be closest to the truth, which needs the answer in
#: order to choose. It exists so that "the rule recovers X of what is
#: available" is a measured fraction rather than an assertion.
ORACLE = 'Oracle, per material'

#: The cutoffs the sweep runs. Spaced closely through the region the fit-level
#: work already pointed at -- decision 142 puts the fit optimum at 81 with 68
#: to 106 indistinguishable -- and carried out to BOTH DEGENERATE ENDS, which
#: is what makes the sweep self-checking: the corpus holds 3 to 9,999
#: declarations, so a cutoff of 3 assigns every dataset the kernel estimate
#: with market weights and a cutoff of 10,000 assigns every dataset the
#: three-parameter lognormal with uniform weights. **Those two points must
#: reproduce those two fixed methods exactly**, and if they do not, the sweep
#: is wrong. Widened 2026-09-29 at the author's request, from a range that
#: stopped at 20 and 220 and so could not show either end.
SWEEP_THRESHOLDS = (3, 10, 20, 30, 50, 70, 81, 100, 130, 200, 300, 1000,
                    3000, 10000)


def select_method(n, threshold=MIXED_THRESHOLD, large=LARGE_METHOD,
                  small=SMALL_METHOD):
    """Which of the six fixed methods a rule picks for a dataset of size n.

    One number and nothing else, by the author's instruction. `n` is the count
    of declarations a practitioner holds, which is the only input the rule has
    and the only one they can compute before fitting anything.
    """
    return large if int(n) >= int(threshold) else small


def method_column(sizes, threshold=MIXED_THRESHOLD, **kw):
    """`{dataset: chosen fixed method}` for a mapping of dataset -> n."""
    return {d: select_method(n, threshold=threshold, **kw)
            for d, n in dict(sizes).items()}


# ---------------------------------------------------------------------------
# the policies the sweep compares, and they are all ONE number
# ---------------------------------------------------------------------------
#: A candidate policy: a stored name, the cutoff, the method used at or above
#: it, the method used below it, and how it is shown to a reader. **Every one
#: of these reads exactly one input, the dataset's size.** They differ in WHAT
#: the cutoff switches and in whether a practitioner could follow them, never
#: in how many numbers the reader needs.
class Policy:
    __slots__ = ('name', 'threshold', 'above', 'below', 'display', 'kind',
                 'family', 'feasible')

    def __init__(self, name, threshold, above, below, display, kind,
                 family='mixed', feasible=False):
        self.name, self.threshold = name, int(threshold)
        self.above, self.below = above, below
        self.display, self.kind = display, kind
        self.family, self.feasible = family, bool(feasible)

    def choose(self, n):
        return self.above if int(n) >= self.threshold else self.below

    def __repr__(self):
        return (f'Policy({self.name!r}, {self.threshold}, {self.family!r}, '
                f'feasible={self.feasible})')


#: The two rule families the sweep compares, and the distinction between them
#: is the author's, 2026-09-29: "a weighting switch isn't feasible. Nobody will
#: ever know weights like that."
#:
#:   'feasible'  uniform weights throughout, the FAMILY switches at the cutoff.
#:               A practitioner can do this with a set of EPDs and nothing else,
#:               so it is the rule the paper recommends.
#:   'mixed'     the same family switch, plus market weights above the cutoff.
#:               On the synthetic arm those are the TRUE market shares, so this
#:               is not a method a reader can follow -- it is the value of
#:               knowing market share, and the gap between the two curves is
#:               what that knowledge is worth.
FEASIBLE_ABOVE = 'KDE, Uniform'
FEASIBLE_BELOW = 'Lognormal, Uniform'


def sweep_policies(thresholds=SWEEP_THRESHOLDS, study_threshold=MIXED_THRESHOLD,
                   above=LARGE_METHOD, below=SMALL_METHOD):
    """The KNOWN-SHARE rule at each cutoff: market weights above it.

    The cutoff the study settled on keeps the bare name `Mixed`, so the tables
    and the figure written before the sweep existed still join on it.
    """
    out = []
    for t in thresholds:
        name = MIXED if int(t) == int(study_threshold) else f'Mixed@{int(t)}'
        out.append(Policy(name, t, above, below,
                          f'known shares above {int(t)}', 'threshold',
                          family='mixed', feasible=False))
    return out


def feasible_policies(thresholds=SWEEP_THRESHOLDS, above=FEASIBLE_ABOVE,
                      below=FEASIBLE_BELOW):
    """The FEASIBLE rule at each cutoff: uniform weights throughout, the family
    switches.

    **This is the one a reader can act on.** Nobody publishes market shares, so
    uniform weighting is not a choice a practitioner makes -- it is the only
    option -- and what a practitioner CAN choose is which family to fit. The
    two degenerate ends are in the six fixed methods, so this family is
    self-checking in the same way the other is.
    """
    return [Policy(f'Feasible@{int(t)}', t, above, below,
                   f'uniform weights, switch at {int(t)}', 'threshold',
                   family='feasible', feasible=True)
            for t in thresholds]


def variant_policies(threshold=MIXED_THRESHOLD):
    """The one-axis variants at a fixed cutoff, which say which half of the
    known-share rule's switch does the work.

    `MixedUniform` is not here any more: it IS the feasible rule, and it has
    its own swept family.
    """
    t = int(threshold)
    return [
        Policy(f'MixedMarket@{t}', t, 'KDE, Variable', 'Lognormal, Variable',
               f'known shares throughout, family switches at {t}', 'variant',
               family='variant'),
        Policy(f'MixedKDE@{t}', t, 'KDE, Variable', 'KDE, Uniform',
               f'kernel throughout, weighting switches at {t}', 'variant',
               family='variant'),
        Policy(f'MixedLognormal@{t}', t, 'Lognormal, Variable',
               'Lognormal, Uniform',
               f'lognormal throughout, weighting switches at {t}', 'variant',
               family='variant'),
    ]


def all_policies(thresholds=SWEEP_THRESHOLDS, threshold=MIXED_THRESHOLD):
    """Every candidate this stage scores: the feasible rule swept, the
    known-share rule swept, then the variants."""
    return (feasible_policies(thresholds) + sweep_policies(thresholds, threshold)
            + variant_policies(threshold))


def add_policies(models, sizes, policies):
    """Add each policy as a NEW KEY on every dataset's fitted-model dict.

    Mutates `models` in place and returns `(models, choice)`, where `choice`
    maps a policy name to `{dataset: the fixed method it selected}`.

    **Nothing is refitted and no randomness is consumed.** Each new key is a
    reference to one of the six model objects already in the dict, so the
    object SAMPLED under a policy is bit-for-bit the object that fixed policy
    samples, and a group whose materials all land on one side of a cutoff
    reproduces that fixed policy exactly.
    """
    choice = {}
    for p in policies:
        picked = {}
        for dataset, n in dict(sizes).items():
            if dataset not in models:
                continue
            m = p.choose(n)
            models[dataset][p.name] = models[dataset][m]
            picked[dataset] = m
        choice[p.name] = picked
    return models, choice


def add_mixed(models, sizes, name=MIXED, threshold=MIXED_THRESHOLD, **kw):
    """The study's rule alone, as a single new key. Kept for the tests and for
    any caller that wants one policy rather than the whole sweep."""
    choice = {}
    for dataset, n in dict(sizes).items():
        if dataset not in models:
            continue
        picked = select_method(n, threshold=threshold, **kw)
        models[dataset][name] = models[dataset][picked]
        choice[dataset] = picked
    return models, choice


def methods_with_mixed(methods=None, name=MIXED):
    """The six fixed policies, then the study's rule, in a stable order."""
    base = list(methods or FT.PEWT)
    return base + ([name] if name not in base else [])


def methods_with_policies(policies, methods=None):
    """The six fixed policies, then every candidate, in a stable order."""
    base = list(methods or FT.PEWT)
    return base + [p.name for p in policies if p.name not in base]


#: Filled by `display_method` for any policy the caller has built.
_DISPLAY = {}


def register_display(policies):
    """Teach `display_method` the labels of a set of policies."""
    _DISPLAY.update({p.name: p.display for p in policies})
    return _DISPLAY


def display_method(name, short=False):
    """Display label for any policy label, the swept ones included."""
    if name == MIXED:
        return MIXED_DISPLAY
    if name == ORACLE:
        return 'per-material oracle'
    if name in _DISPLAY:
        return _DISPLAY[name]
    return FT.display_method(name, short=short)


def is_policy(name):
    """Is this a per-material policy rather than one of the six fixed methods."""
    return name == MIXED or name.startswith('Mixed')

# ---------------------------------------------------------------------------
# provenance, stamped on every table this stage writes
# ---------------------------------------------------------------------------
def provenance(corpus_label, weight_rho, threshold=MIXED_THRESHOLD, **extra):
    """The two facts Stage 2h could not answer about its own output.

    Which corpus a result ran on and which market-share weight rule the real
    arm was built under. Stamped as columns rather than left to be
    reconstructed from file timestamps, which is what a stage had to do.
    """
    stamp = dict(corpus=str(corpus_label), weight_rho=float(weight_rho),
                 mixed_threshold=int(threshold))
    stamp.update(extra)
    return stamp


def stamp(frame, **kw):
    """Return `frame` with the provenance columns appended."""
    out = frame.copy()
    for key, value in provenance(**kw).items():
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# what the rule does to the FIT, which costs nothing to measure
# ---------------------------------------------------------------------------
def fit_scores(scores, sizes=None, score='w1_market', method='method',
               dataset='dataset', size='n', name=MIXED,
               threshold=MIXED_THRESHOLD, **kw):
    """The mixed policy's fit score, selected per dataset from existing scores.

    `scores` is a long frame with one row per (dataset, method), such as
    `TABLE_MethodScores.csv`. The mixed policy's score for a dataset is simply
    the score already recorded for whichever fixed method the rule picks, so
    this is a selection and not a computation and it cannot disagree with the
    six-method table it is taken from.

    Returns the input frame with the mixed rows appended.
    """
    frame = scores.copy()
    if sizes is None:
        sizes = (frame[[dataset, size]].drop_duplicates()
                 .set_index(dataset)[size].to_dict())
    choice = method_column(sizes, threshold=threshold, **kw)
    want = frame[dataset].map(choice)
    mixed = frame[frame[method] == want].copy()
    mixed['picked'] = mixed[method]
    mixed[method] = name
    return pd.concat([frame, mixed], ignore_index=True)


# ---------------------------------------------------------------------------
# the group-composition effect, which is what would make this a null
# ---------------------------------------------------------------------------
def group_composition(combos, sizes, threshold=MIXED_THRESHOLD):
    """Per pLCA group: how many materials the rule moves, and how small the
    smallest one is.

    A probabilistic LCA claim belongs to the GROUP of four materials and the
    group's worst-fitted member sets much of its error, so improving one
    material of four cannot improve the group by more than that member's share
    of it. `n_switched` is how many of the group's materials the rule assigns a
    method other than the one the best fixed policy would give them, and
    `n_min` is the smallest dataset in the group, which Stage 2h found moves
    the kernel estimate's advantage by a factor of nearly two (decision 163).
    """
    rows = []
    for i, g in enumerate(np.asarray(combos)):
        ns = np.array([int(sizes[d]) for d in g])
        above = ns >= int(threshold)
        rows.append(dict(
            plca=i,
            nmats=int(len(ns)),
            n_min=int(ns.min()),
            n_max=int(ns.max()),
            n_median=float(np.median(ns)),
            n_above=int(above.sum()),
            all_above=bool(above.all()),
            all_below=bool(not above.any()),
            split_group=bool(above.any() and not above.all())))
    return pd.DataFrame(rows)


#: How the smallest material in a group is banded when the result is split by
#: group composition. Same edges as `metricset.SIZE_BANDS`, so the two tables
#: can be read side by side.
COMPOSITION_BANDS = (('3-9', 3, 9), ('10-99', 10, 99), ('100-999', 100, 999),
                     ('1000+', 1000, 10 ** 9))


def band(values, bands=COMPOSITION_BANDS):
    """Label each value with the band it falls in."""
    v = np.asarray(values, dtype=float)
    out = np.array([''] * len(v), dtype=object)
    for label, lo, hi in bands:
        out[(v >= lo) & (v <= hi)] = label
    return out


# ---------------------------------------------------------------------------
# is the gain real: a PAIRED bootstrap, because the comparison is paired
# ---------------------------------------------------------------------------
#: Resamples and interval width, matching `plca.BOOTSTRAP_RESAMPLES` so a gain
#: here and an NRMSE there carry the same kind of interval.
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_ALPHA = 0.05


def claim_errors(frames, claims=None, method='method', cluster='plca'):
    """Row-level absolute error for every scorecard claim, one frame.

    `frames` maps each source name used by `metricset.SCORECARD_CLAIMS`
    -- 'recovery', 'building', 'intervention', 'swap' -- to the ROW-LEVEL frame
    it is read from. The 'recovery' claims are read from the per-material truth
    frame rather than from `recovery_table`, because a paired comparison needs
    the units and not their mean.

    Returns one row per (claim, unit, method) carrying `abs_error`, the signed
    `error`, and the claim's `truth` value, plus the group the unit belongs to
    so a cluster bootstrap can resample it.
    """
    import metricset as MS
    claims = claims or MS.SCORECARD_CLAIMS
    out = []
    for group, label, source, output in claims:
        frame = frames.get(source)
        if frame is None:
            continue
        ecol, tcol = f'{output}__error', f'{output}__truth'
        if ecol not in frame.columns or tcol not in frame.columns:
            continue
        clus = next((c for c in (cluster, 'pair') if c in frame.columns), None)
        keys = [c for c in (clus, 'dataset', 'saving') if c in frame.columns]
        unit = (frame[keys].astype(str).agg('|'.join, axis=1).to_numpy()
                if keys else np.arange(len(frame)).astype(str))
        sub = pd.DataFrame({
            'question': group,
            'claim': label,
            'output': output,
            'source': source,
            'method': frame[method].to_numpy(),
            'cluster': (frame[clus].to_numpy() if clus is not None
                        else np.arange(len(frame))),
            # WHAT THE CLUSTER ID COUNTS, and it is not always a pLCA group.
            # The design comparison's cluster is a design PAIR drawn from its
            # own resampling, and its ids run 0 to 2,499 exactly as the pLCA
            # groups' do. Without this column a join on the cluster id silently
            # hands every design pair the composition of the same-numbered pLCA
            # group, which is what the first version of this stage did.
            'cluster_kind': (clus if clus is not None else 'row'),
            'unit': unit,
            'error': pd.to_numeric(frame[ecol], errors='coerce').to_numpy(),
            'truth': pd.to_numeric(frame[tcol], errors='coerce').to_numpy()})
        sub['abs_error'] = sub['error'].abs()
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _wide_units(sub, method='method', unit='unit', value='abs_error'):
    """One row per UNIT, one column per policy, plus the unit's cluster.

    Every policy sees the same units on the same uniform variates, so a unit's
    row is a paired observation and a difference taken along it carries no
    Monte Carlo noise. Pivoting on the unit rather than on the frame's own
    index is what makes that true; taking the row minimum of an unpivoted
    frame would return each row's own single value and call it an oracle.
    """
    wide = sub.pivot_table(index=unit, columns=method, values=value,
                           aggfunc='mean', observed=True)
    clusters = sub.drop_duplicates(unit).set_index(unit)['cluster']
    wide['cluster'] = clusters.reindex(wide.index).to_numpy()
    return wide


def _paired_boot(values, clusters, rng, resamples, alpha=BOOTSTRAP_ALPHA):
    """Percentile interval for the mean of each column, resampling CLUSTERS.

    Same estimator as `plca.cluster_bootstrap`: a resample draws whole clusters
    with replacement and the statistic is the mean over the rows they carry.
    It is written as a ratio of per-cluster sums to per-cluster counts rather
    than by materializing the resampled rows, because this stage bootstraps
    sixteen claims over 2,500 clusters and the row-materializing form spends
    minutes doing it. `tests/test_mixedpolicy.py` pins that the two agree.
    """
    values = np.asarray(values, dtype=float)
    uniq, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    n_groups, n_cols = len(uniq), values.shape[1]
    finite = np.isfinite(values)
    sums = np.zeros((n_groups, n_cols), dtype=float)
    counts = np.zeros((n_groups, n_cols), dtype=float)
    np.add.at(sums, inverse, np.where(finite, values, 0.0))
    np.add.at(counts, inverse, finite.astype(float))
    stats = np.empty((int(resamples), n_cols), dtype=float)
    for b in range(int(resamples)):
        pick = rng.integers(0, n_groups, size=n_groups)
        s, c = sums[pick].sum(axis=0), counts[pick].sum(axis=0)
        stats[b] = np.where(c > 0, s / np.where(c > 0, c, 1.0), np.nan)
    lo = np.nanpercentile(stats, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(stats, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def claim_gain(errors, name=MIXED, reference=None, rng=None,
               resamples=BOOTSTRAP_RESAMPLES, alpha=BOOTSTRAP_ALPHA,
               fixed=None):
    """What the mixed policy buys on each claim, against the noise of the
    comparison rather than against zero.

    The comparison is PAIRED: every policy sees the same units on the same
    uniform variates, so the difference between two policies on one unit has no
    Monte Carlo noise in it and an unpaired interval would be far too wide.
    The resampling unit is the pLCA GROUP and never the row, because the
    materials of a group share its total and its variates.

    `reference` names the fixed policy to compare against; the default is
    whichever of the six is closest to the truth ON THAT CLAIM, which is the
    honest comparator because it is the one a reader would otherwise use.

    Returns one row per claim with the mixed policy's error, the reference's,
    the paired gain as a percentage of the reference, and a 95 percent
    interval on the gain. `gain_pct` is positive when the mixed policy is
    CLOSER to the truth.
    """
    rng = rng or np.random.default_rng(0)
    rows = []
    for claim, sub in errors.groupby('claim', sort=False):
        wide = _wide_units(sub).groupby('cluster', observed=True).mean()
        if name not in wide.columns:
            continue
        # THE COMPARATOR IS ONE OF THE SIX FIXED METHODS AND NEVER ANOTHER
        # POLICY. With a sweep in the frame, "everything except me" would pick
        # a neighbouring cutoff as the thing to beat, which is not the
        # comparison a reader would otherwise make and would collapse every
        # gain to nearly zero.
        pool = list(fixed) if fixed is not None else list(FT.PEWT)
        pool = [c for c in pool if c in wide.columns and c != name]
        means = wide.mean()
        ref = reference or means[pool].idxmin()
        level = abs(float(np.nanmean(sub['truth'].to_numpy(float))))
        cols = [name, ref]
        mat = wide[cols].to_numpy(float)
        diff = mat[:, 1] - mat[:, 0]          # positive: mixed is closer
        stacked = np.column_stack([mat, diff])
        lo, hi = _paired_boot(stacked, wide.index.to_numpy(), rng, resamples,
                              alpha)
        rows.append(dict(
            question=sub['question'].iloc[0],
            claim=claim,
            output=sub['output'].iloc[0],
            reference=ref,
            n_clusters=int(len(wide)),
            truth_level=level,
            mixed_error=float(means[name]),
            reference_error=float(means[ref]),
            mixed_rel=float(means[name]) / level if level else np.nan,
            reference_rel=float(means[ref]) / level if level else np.nan,
            gain=float(np.nanmean(diff)),
            gain_lo=float(lo[2]),
            gain_hi=float(hi[2]),
            gain_pct=100.0 * float(np.nanmean(diff)) / float(means[ref])
            if means[ref] else np.nan,
            gain_pct_lo=100.0 * float(lo[2]) / float(means[ref])
            if means[ref] else np.nan,
            gain_pct_hi=100.0 * float(hi[2]) / float(means[ref])
            if means[ref] else np.nan,
            beats_reference=bool(lo[2] > 0.0),
            rank_of_mixed=int((means.rank(method='min')[name])),
            n_policies=int(len(means))))
    return pd.DataFrame(rows)


def oracle_ceiling(errors, name=MIXED, fixed=None):
    """The unreachable per-material ceiling, as a fraction the rule recovers.

    For each unit the oracle takes whichever of the SIX fixed policies is
    closest to the truth, which needs the answer in order to choose and is
    therefore a bound rather than a policy. `recovered` is how much of the
    distance between the best fixed policy and that bound the size rule
    closes; a value near zero means the single-threshold rule captures what a
    per-material choice can capture, and a value near one means the rule is
    almost as good as knowing.
    """
    rows = []
    for claim, sub in errors.groupby('claim', sort=False):
        wide = _wide_units(sub)
        cols = list(fixed) if fixed is not None else list(FT.PEWT)
        cols = [c for c in cols if c in wide.columns and c != name]
        unit = wide[cols].dropna(how='all')
        best_fixed = float(unit.mean().min())
        oracle = float(unit.min(axis=1).mean())
        mixed = float(wide[name].mean()) if name in wide.columns else np.nan
        span = best_fixed - oracle
        rows.append(dict(
            question=sub['question'].iloc[0],
            claim=claim,
            output=sub['output'].iloc[0],
            best_fixed_method=str(unit.mean().idxmin()),
            best_fixed=best_fixed,
            oracle=oracle,
            mixed=mixed,
            available=span,
            recovered=(best_fixed - mixed) / span if span > 0 else np.nan))
    return pd.DataFrame(rows)


def attach_composition(errors, composition, cluster='cluster', on='plca'):
    """Join each error row to the composition of the pLCA group it came from.

    ONLY ROWS WHOSE CLUSTER REALLY IS A pLCA GROUP ARE JOINED, and that has to
    be checked rather than assumed. The design comparison's cluster is a design
    PAIR drawn from its own resampling and its ids run over the same integers
    the pLCA groups use, so a join on the id alone gives every design pair the
    composition of an unrelated pLCA group -- which is what the first version
    of this stage did, and it showed up as a group in which the rule cannot
    act reporting a two percent gain. `claim_errors` records `cluster_kind`
    for exactly this, and rows of any other kind come back with the
    composition columns empty and are dropped by any split that uses them.
    That is correct rather than a gap: a design pair has no pLCA group whose
    composition could be read.
    """
    comp = composition.set_index(on)
    out = errors.copy()
    ok = (out['cluster_kind'] == on if 'cluster_kind' in out.columns
          else pd.Series(True, index=out.index))
    for col in ('n_min', 'n_above', 'n_median', 'split_group'):
        if col not in comp.columns:
            continue
        out[col] = out[cluster].map(comp[col]).where(ok)
    if 'n_min' in out.columns:
        out['n_min_band'] = np.where(out['n_min'].notna(),
                                     band(out['n_min'].fillna(-1)), '')
    return out


def gain_by_group(errors, split, name=MIXED, reference=None, rng=None,
                  resamples=BOOTSTRAP_RESAMPLES, min_clusters=25, fixed=None):
    """`claim_gain` computed separately within each level of a split.

    THIS IS THE MEASUREMENT THAT MAKES A SMALL OVERALL GAIN ATTRIBUTABLE
    RATHER THAN MERELY DISAPPOINTING. A probabilistic LCA claim belongs to the
    GROUP of four materials and the group's worst-fitted member sets much of
    its error, so improving one material of four cannot improve the group by
    more than that member's share of it. Splitting by the SMALLEST dataset in
    each group is the direct test: Stage 2h found that split moves the kernel
    estimate's advantage by a factor of nearly two (decision 163).

    The REFERENCE is held fixed across levels when one is given, because
    letting each level pick its own best fixed policy would compare the mixed
    policy against a different comparator in every row of the table.
    """
    rng = rng or np.random.default_rng(0)
    out = []
    for level, sub in errors.groupby(split, sort=True, observed=True):
        if not str(level):
            continue
        if sub['cluster'].nunique() < int(min_clusters):
            continue
        got = claim_gain(sub, name=name, reference=reference, rng=rng,
                         resamples=resamples, fixed=fixed)
        if got.empty:
            continue
        got.insert(0, split, level)
        out.append(got)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def pooled_error(errors, questions=('attribution', 'information'),
                 name=MIXED):
    """Mean RELATIVE error across several claims, per policy.

    Each claim is divided by its own true level first, because the claims are
    in incomparable units and an unweighted mean over them would be dominated
    by whichever has the largest level. This is the same divisor the scorecard
    uses (decision 157) and the same one `metricset.size_band_recovery` uses,
    so the three tables can be read against each other.
    """
    sub = errors[errors['question'].isin(questions)] if questions else errors
    per = (sub.groupby(['claim', 'method'], observed=True)
           .agg(err=('abs_error', 'mean'), lvl=('truth', 'mean'))
           .reset_index())
    per['rel'] = per['err'] / per['lvl'].abs()
    return (per.groupby('method', observed=True)['rel'].mean()
            .rename('mean_rel_error').reset_index()
            .sort_values('mean_rel_error').reset_index(drop=True))


# ---------------------------------------------------------------------------
# how precisely does the cutoff have to be set: a RANGE, not a point
# ---------------------------------------------------------------------------
def claim_blocks(errors, policies, claims=None):
    """Per-unit-universe arrays of mean absolute error, one block per cluster
    kind, plus each claim's true level.

    **THERE ARE TWO UNIT UNIVERSES AND POOLING NEEDS BOTH.** Fifteen of the
    sixteen claims belong to a pLCA GROUP; the design comparison belongs to a
    design PAIR drawn from its own resampling. A single array indexed by pLCA
    group silently leaves the design comparison as a column of NaN, so a
    "pooled over sixteen claims" number would quietly be over fifteen. Each
    block carries its own units and is resampled in its own universe, which is
    also correct: the two experiments are independent.

    Returns `(blocks, claims, policies, levels)` where each block is
    `(array of shape (units, claims_in_block, policies), positions)` and
    `positions` indexes into `claims`.
    """
    policies = list(policies)
    claims = list(claims if claims is not None
                  else errors['claim'].drop_duplicates())
    cpos = {c: i for i, c in enumerate(claims)}
    levels = np.full(len(claims), np.nan)
    blocks = []
    for kind, part in errors.groupby('cluster_kind', sort=True, observed=True):
        names = [c for c in claims if c in set(part['claim'])]
        if not names:
            continue
        units = np.sort(part['cluster'].unique())
        upos = {u: i for i, u in enumerate(units)}
        arr = np.full((len(units), len(names), len(policies)), np.nan)
        for ci, claim in enumerate(names):
            sub = part[part['claim'] == claim]
            levels[cpos[claim]] = abs(float(np.nanmean(
                sub['truth'].to_numpy(float))))
            piv = sub.pivot_table(index='cluster', columns='method',
                                  values='abs_error', aggfunc='mean',
                                  observed=True)
            rows = np.array([upos[u] for u in piv.index])
            for pi, pol in enumerate(policies):
                if pol in piv.columns:
                    arr[rows, ci, pi] = piv[pol].to_numpy(float)
        blocks.append((arr, np.array([cpos[c] for c in names])))
    return blocks, claims, policies, levels


def pooled_from_blocks(blocks, levels, n_claims, draws=None):
    """Mean relative error over every claim, for each policy.

    `draws` is one row-index array per block, for a bootstrap resample; None
    uses every unit. Each claim is divided by its own true level first, because
    the claims are in incomparable units.
    """
    n_pol = blocks[0][0].shape[2]
    per_claim = np.full((n_claims, n_pol), np.nan)
    for bi, (arr, pos) in enumerate(blocks):
        block = arr if draws is None else arr[draws[bi]]
        with np.errstate(invalid='ignore'):
            per_claim[pos] = np.nanmean(block, axis=0) / levels[pos][:, None]
    return np.nanmean(per_claim, axis=0)


def threshold_curve(errors, policies, rng=None,
                    resamples=BOOTSTRAP_RESAMPLES, alpha=BOOTSTRAP_ALPHA,
                    family=None):
    """The cost of the rule at every cutoff, and the cutoffs that cannot be
    told apart from the best one.

    **THE DELIVERABLE IS A RANGE AND NOT A POINT**, at the author's
    instruction: the paper should say "the cutoff above which a kernel
    estimate performs best is 60 to 100 declarations" rather than name 81,
    because the study made assumptions and 81 is not precisely correct.

    Same instrument as `metricreduction.threshold_interval`, which answers the
    same question one level up at the FIT, so the two ranges are comparable.
    Two bootstraps over pLCA GROUPS: `best_threshold` resamples and takes the
    argmin, so its spread says how well the data pin the cutoff down; the
    PENALTY is the excess over whichever cutoff won on that same resample, so
    the variation common to both cancels and the interval is about the
    difference rather than the level. The cutoffs whose penalty interval
    reaches zero cannot be told apart from the best, and the longest UNBROKEN
    run of those is the range to print.
    """
    rng = rng or np.random.default_rng(0)
    sweep = [p for p in policies
             if p.kind == 'threshold' and (family is None or p.family == family)]
    seen = [p.threshold for p in sweep]
    if len(set(seen)) != len(seen):
        raise ValueError(
            'two rule families share a cutoff grid, so a curve over both would '
            'have two points at each cutoff. Pass `family=` to choose one; '
            f'got families {sorted({p.family for p in sweep})}')
    names = [p.name for p in sweep]
    thresholds = np.array([p.threshold for p in sweep])
    blocks, claims, _, levels = claim_blocks(errors, names)
    point = pooled_from_blocks(blocks, levels, len(claims))
    sizes = [arr.shape[0] for arr, _ in blocks]
    best = np.empty(int(resamples))
    penalty = np.empty((int(resamples), len(names)))
    for b in range(int(resamples)):
        draws = [rng.integers(0, n, n) for n in sizes]
        cost = pooled_from_blocks(blocks, levels, len(claims), draws)
        j = int(np.nanargmin(cost))
        best[b] = thresholds[j]
        penalty[b] = cost - cost[j]
    lo = np.percentile(penalty, 100 * alpha / 2, axis=0)
    hi = np.percentile(penalty, 100 * (1 - alpha / 2), axis=0)
    ok = lo <= 0.0
    runs, cur = [], []
    for t, good in zip(thresholds, ok):
        if good:
            cur.append(t)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    run = max(runs, key=len) if runs else []
    frame = pd.DataFrame(dict(
        threshold=thresholds, policy=names,
        pooled_error=point,
        penalty=point - float(np.nanmin(point)),
        penalty_lo=lo, penalty_hi=hi,
        indistinguishable=ok,
        in_range=[t in run for t in thresholds]))
    flagged = thresholds[ok]
    summary = dict(
        family=family,
        best_threshold=int(thresholds[int(np.nanargmin(point))]),
        best_error=float(np.nanmin(point)),
        range_lo=int(run[0]) if run else None,
        range_hi=int(run[-1]) if run else None,
        # THE SPAN OF CUTOFFS THAT ARE INDIVIDUALLY INDISTINGUISHABLE, which
        # is wider than the unbroken run whenever an INTERIOR cutoff falls out
        # on bootstrap jitter. The run rule exists to stop one lucky far-away
        # point widening the band (decision 142); it was not written for a hole
        # in the middle, and reporting only the run understates the answer.
        indistinguishable_lo=int(flagged.min()) if len(flagged) else None,
        indistinguishable_hi=int(flagged.max()) if len(flagged) else None,
        argmin_lo=float(np.percentile(best, 100 * alpha / 2)),
        argmin_hi=float(np.percentile(best, 100 * (1 - alpha / 2))),
        n_claims=len(claims), n_groups=int(max(sizes)))
    return frame, summary


def policy_table(errors, policies, fixed=None):
    """Pooled relative error over every claim, for the six fixed methods and
    every candidate policy, in one table.

    This is the small table the report prints: one number per policy, so a
    reader can see what each variant of the rule is worth without reading
    sixteen rows of a scorecard.
    """
    pool = list(fixed) if fixed is not None else list(FT.PEWT)
    names = pool + [p.name for p in policies if p.name not in pool]
    blocks, claims, _, levels = claim_blocks(errors, names)
    value = pooled_from_blocks(blocks, levels, len(claims))
    kind = {p.name: p.kind for p in policies}
    disp = {p.name: p.display for p in policies}
    rows = []
    for name, v in zip(names, value):
        rows.append(dict(
            method=name,
            display=disp.get(name, display_method(name)),
            kind=kind.get(name, 'fixed'),
            threshold=next((p.threshold for p in policies
                            if p.name == name), np.nan),
            pooled_error=float(v),
            n_claims=len(claims)))
    out = pd.DataFrame(rows)
    out['excess_over_best_fixed'] = (
        out.pooled_error - out.loc[out.kind == 'fixed', 'pooled_error'].min())
    return out.sort_values('pooled_error').reset_index(drop=True)

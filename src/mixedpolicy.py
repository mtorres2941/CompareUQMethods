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
#: reproduced unchanged on the regenerated corpus by decision 198. NOT to be
#: re-derived here: this stage measures what the rule buys downstream, and a
#: stage that re-tunes its own threshold on the outcome it reports is tuning on
#: the criterion it is judged by.
MIXED_THRESHOLD = 81

#: The two fixed policies the rule switches between, as stored `method` values.
#: The stored spellings keep "Uniform" and "Variable" because they are the join
#: key for every table and fixture in this project; the DISPLAY vocabulary is
#: "uniform weights" and "market weights" and comes from `fitting.WT_DISPLAY`.
LARGE_METHOD = 'KDE, Variable'
SMALL_METHOD = 'Lognormal, Uniform'

#: What the mixed policy is called in a stored `method` column, and to a reader.
MIXED = 'Mixed'
MIXED_DISPLAY = f'size rule (n >= {MIXED_THRESHOLD})'

#: An unreachable per-material ORACLE, carried as a CEILING and never as a
#: policy. It picks, for each material and each output, whichever of the six
#: fixed methods happens to be closest to the truth, which needs the answer in
#: order to choose. It exists so that "the size rule recovers X of what is
#: available" is a measured fraction rather than an assertion.
ORACLE = 'Oracle, per material'


def select_method(n, threshold=MIXED_THRESHOLD, large=LARGE_METHOD,
                  small=SMALL_METHOD):
    """Which of the six fixed methods the rule picks for a dataset of size n.

    One number and nothing else, by the author's instruction. `n` is the count
    of declarations a practitioner holds, which is the only input the rule has
    and the only one they can compute before fitting anything.
    """
    return large if int(n) >= int(threshold) else small


def method_column(sizes, threshold=MIXED_THRESHOLD, **kw):
    """`{dataset: chosen fixed method}` for a mapping of dataset -> n."""
    return {d: select_method(n, threshold=threshold, **kw)
            for d, n in dict(sizes).items()}


def add_mixed(models, sizes, name=MIXED, threshold=MIXED_THRESHOLD, **kw):
    """Add the mixed policy as a NEW KEY on each dataset's fitted-model dict.

    Mutates `models` in place and returns `(models, choice)`, where `choice`
    maps each dataset to the fixed method the rule selected for it.

    **Nothing is refitted and no randomness is consumed.** The new key is a
    reference to one of the six model objects already in the dict, so the
    object that is SAMPLED under the mixed policy is bit-for-bit the object
    that fixed policy samples, and a group in which every material lands on the
    same side of the threshold reproduces that fixed policy exactly.
    """
    choice = {}
    for dataset, n in dict(sizes).items():
        if dataset not in models:
            continue
        picked = select_method(n, threshold=threshold, **kw)
        models[dataset][name] = models[dataset][picked]
        choice[dataset] = picked
    return models, choice


def methods_with_mixed(methods=None, name=MIXED):
    """The six fixed policies, then the mixed one, in a stable order."""
    base = list(methods or FT.PEWT)
    return base + ([name] if name not in base else [])


def display_method(name, short=False):
    """Display label for any policy label, mixed one included."""
    if name == MIXED:
        return MIXED_DISPLAY
    if name == ORACLE:
        return 'per-material oracle'
    return FT.display_method(name, short=short)


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
               resamples=BOOTSTRAP_RESAMPLES, alpha=BOOTSTRAP_ALPHA):
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
        fixed = [c for c in wide.columns if c != name]
        means = wide.mean()
        ref = reference or means[fixed].idxmin()
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
        cols = list(fixed) if fixed is not None else [
            c for c in wide.columns if c not in ('cluster', name, ORACLE)]
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

    Only the frames whose cluster IS a pLCA group can be joined; the design
    comparison's cluster is a design PAIR drawn from a different grouping, so
    its rows come back with the composition columns missing and are dropped by
    any split that uses them. That is correct rather than a gap: a design pair
    has no pLCA group whose composition could be read.
    """
    comp = composition.set_index(on)
    out = errors.copy()
    for col in ('n_min', 'n_above', 'n_median', 'split_group'):
        if col in comp.columns:
            out[col] = out[cluster].map(comp[col])
    if 'n_min' in out.columns:
        out['n_min_band'] = np.where(out['n_min'].notna(),
                                     band(out['n_min'].fillna(-1)), '')
    return out


def gain_by_group(errors, split, name=MIXED, reference=None, rng=None,
                  resamples=BOOTSTRAP_RESAMPLES, min_clusters=25):
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
                         resamples=resamples)
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

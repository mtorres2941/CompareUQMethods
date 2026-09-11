"""Build synthetic ECC datasets from a GeneratorConfig.

One function, `generate_dataset`, turns (config, rng, n) into a dataset, its
weights, and the parent it came from. Everything it does is determined by the
configuration; nothing is tuned inside.

The provenance record returned alongside each dataset is the point of the
exercise. It holds the realized value of every drawn parameter, the status of
every moment solve, the overlap asked for and the overlap achieved, and the
normalizer. A reader can reconstruct the parent exactly from it, which is what
Stage 2c needs and what the old generator could not provide at any price.

Stage 2a Part 2: the validity filter. The old notebook generated 15,000
datasets and discarded every one that was a marginal outlier on any of 20
metrics, keeping the first 10,000 survivors. That removed 27.5 percent of what
was generated, was keyed to the synthetic corpus's own metric spread rather
than to anything empirical, and is not described in the manuscript. It is
replaced by `validity_failures`, which rejects a dataset only when it cannot be
analysed at all. Nothing is filtered for being statistically unusual.
"""

import numpy as np

import components as C
import genconfig as G
import mixture as M

_EPS = 1e-12


# --------------------------------------------------------------------------
def stratified_sizes(cfg, rng):
    """Dataset sizes for the whole corpus, log-uniform within each stratum.

    Returns (sizes, stratum_names), both length cfg.n_datasets, in the order
    the datasets are generated.
    """
    sizes, names = [], []
    for s in cfg.strata:
        u = rng.uniform(np.log10(s.n_lo), np.log10(s.n_hi + 1), s.n_datasets)
        n = np.clip(np.floor(10 ** u).astype(int), s.n_lo, s.n_hi)
        sizes.append(n)
        names.extend([s.name] * s.n_datasets)
    return np.concatenate(sizes), np.array(names)


def probe_sizes(cfg, rng):
    s = cfg.probe
    u = rng.uniform(np.log10(s.n_lo), np.log10(s.n_hi + 1), s.n_datasets)
    return np.clip(np.floor(10 ** u).astype(int), s.n_lo, s.n_hi)


# --------------------------------------------------------------------------
def _draw_component_targets(cfg, k, rng):
    """Draw k (skewness, excess kurtosis, sd) targets, retrying the ones that
    cannot be met. Returns targets, solved components, and a status record."""
    specs, solved, retries, statuses = [], [], 0, []
    for _ in range(k):
        for attempt in range(cfg.max_component_retries):
            skew = float(rng.uniform(cfg.comp_skew_lo, cfg.comp_skew_hi))
            exk = float(rng.uniform(cfg.comp_exkurt_lo, cfg.comp_exkurt_hi))
            # lift the target to the feasible side of the moment boundary
            floor = skew ** 2 - 2.0 + C.BOUNDARY_MARGIN
            if exk < floor:
                exk = floor + abs(exk - floor) * 0.0 + 1e-9
            sd = float(10 ** rng.uniform(cfg.comp_sd_log10_lo, cfg.comp_sd_log10_hi))
            fam, shape, loc, scale, status = C.solve_component(skew, exk, mean=0.0, sd=sd)
            if status == 'ok':
                specs.append(dict(skew=skew, exkurt=exk, sd=sd, family=fam,
                                  shape=tuple(float(v) for v in shape),
                                  attempts=attempt + 1))
                solved.append((fam, shape, loc, scale))
                statuses.append('ok')
                retries += attempt
                break
            statuses.append(status)
        else:
            return None, None, retries, statuses
    return specs, solved, retries, statuses


def draw_parent(cfg, n, rng):
    """Draw one parent: components, overlap-solved locations, truncation
    bounds, sampling weights and mode-level market shares."""
    k = int(rng.integers(cfg.k_min, cfg.k_max + 1))
    pi = rng.dirichlet(np.ones(k) * cfg.mode_share_alpha)
    market = rng.dirichlet(np.ones(k) * cfg.market_share_alpha)
    specs, solved, retries, statuses = _draw_component_targets(cfg, k, rng)
    if specs is None:
        return None, dict(status='component_targets_exhausted', statuses=statuses)

    # unit-spaced ordinates; the spread multiplier is what the overlap solve moves
    z = np.sort(rng.uniform(0.0, 1.0, k)) if k > 1 else np.zeros(1)

    def build(c):
        comps = []
        for (fam, shape, loc, scale), zi in zip(solved, z):
            comps.append(C.frozen(fam, shape, loc + c * float(zi), scale))
        return comps, pi

    if k > 1:
        target = float(10 ** rng.uniform(cfg.overlap_log10_lo, cfg.overlap_log10_hi))
        comps, overlap, ov_status = M.solve_spread_for_overlap(build, target)
    else:
        target, overlap, ov_status = 0.0, 0.0, 'single_component'
        comps, _ = build(0.0)

    # shift the whole mixture positive: ECC support is (0, inf), open at zero
    # (decision 13). The shift is a property of the parent, applied to every
    # component identically, so it changes no shape and no overlap.
    lo_supp = min(float(d.ppf(1e-12)) for d in comps)
    shift = 0.0
    if not np.isfinite(lo_supp) or lo_supp <= 0:
        finite_lo = min(float(d.ppf(1e-9)) for d in comps)
        shift = max(0.0, 1e-6 - finite_lo) + abs(finite_lo) * 0.05 + 1.0
        comps = [_Shifted(d, shift) for d in comps]

    lo, hi = M.population_truncation_bounds(comps, pi, cfg.trunc_iqr_mult)
    parent = M.MixtureParent(comps, pi, market, lo, hi, cfg.mode_coupling)
    record = dict(status='ok', k=k, overlap_target=target, overlap_achieved=overlap,
                  overlap_status=ov_status, component_retries=retries,
                  components=specs, pi=pi.tolist(), market=market.tolist(),
                  shift=shift, lo=lo, hi=hi,
                  truncated_mass=parent.truncated_mass(),
                  n_components_dropped=parent.n_components_dropped,
                  k_effective=len(parent.comps))
    return parent, record


class _Shifted:
    """X + shift. Applied identically to every component of a mixture, so it
    moves the whole parent and changes neither shape nor overlap."""

    __slots__ = ('_d', 'shift')

    def __init__(self, d, shift):
        self._d, self.shift = d, float(shift)

    def pdf(self, x):
        return self._d.pdf(np.asarray(x, float) - self.shift)

    def cdf(self, x):
        return self._d.cdf(np.asarray(x, float) - self.shift)

    def sf(self, x):
        return self._d.sf(np.asarray(x, float) - self.shift)

    def ppf(self, q):
        return self._d.ppf(q) + self.shift


# --------------------------------------------------------------------------
def draw_weights(parent, modes, cfg, rng):
    """Market-share weights for the realized points (Part 3).

    Mode-level share v_k is distributed among that mode's own points by a
    Dirichlet, so the market-weighted distribution has a population object:
    sum_k v_k f_k. The coupling parameter blends that against the old
    uncoupled draw, in which a point's weight carried no information about
    which mode it came from.
    """
    n = len(modes)
    k = len(parent.comps)
    coupled = np.zeros(n)
    for j in range(k):
        idx = np.flatnonzero(modes == j)
        if len(idx) == 0:
            continue
        within = rng.dirichlet(np.ones(len(idx)) * cfg.point_weight_alpha)
        coupled[idx] = parent.market[j] * within
    uncoupled = rng.dirichlet(np.ones(n) * cfg.point_weight_alpha)
    c = cfg.mode_coupling
    w = (1.0 - c) * uncoupled + c * coupled
    return w / w.sum()


def generate_dataset(cfg, n, rng):
    """One synthetic ECC dataset. Returns (values, weights, record).

    `values` has unweighted mean exactly 1.0, by decision 6 (Stage 1 amendment
    A3): the normalizer is the unweighted SAMPLE mean in both the synthetic and
    the empirical path, because a practitioner holding a set of EPDs can
    compute that and cannot compute a market-weighted mean. The divisor is in
    the record, so the parent is still exactly specified.
    """
    parent, record = draw_parent(cfg, n, rng)
    if parent is None:
        return None, None, record
    x, modes = parent.sample(n, rng)
    w = draw_weights(parent, modes, cfg, rng)
    record = dict(record)
    record['n'] = int(n)
    record['normalizer'] = parent.normalizer
    record['mode_counts'] = np.bincount(modes, minlength=len(parent.comps)).tolist()
    return x, w, record


# --------------------------------------------------------------------------
# Part 2: the validity filter
# --------------------------------------------------------------------------
def validity_failures(x, w, n_expected=None):
    """Reasons this dataset cannot be analysed. Empty list means it is valid.

    This rejects on analysability alone. It does NOT reject a dataset for being
    statistically unusual, and in particular it never looks at
    weight_outliers -- the old filter preferentially removed high
    weight_outliers datasets, which are exactly the cases the study exists to
    examine. Empirical plausibility is a reported coverage statistic in Part 6,
    not an enforced criterion.

    Undefined kurtosis at n = 3 is EXPECTED and is not a failure: the unbiased
    estimator divides by (n - 1)(n - 2)(n - 3). Stratum 1 runs from n = 3, so
    that column is legitimately missing there, and Stage 2f's complete-case
    models must be told rather than left to drop the stratum silently.
    """
    out = []
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    if x.size == 0 or w.size != x.size:
        out.append('length_mismatch')
        return out
    if n_expected is not None and x.size != int(n_expected):
        out.append('wrong_size')
    if not np.all(np.isfinite(x)):
        out.append('non_finite_values')
    if not np.all(np.isfinite(w)):
        out.append('non_finite_weights')
    if np.any(x <= 0):
        out.append('non_positive_values')
    if np.ptp(x) <= 0 or not np.isfinite(np.std(x)) or np.std(x) <= 0:
        out.append('zero_variance')
    if np.any(w < 0):
        out.append('negative_weights')
    if not np.isclose(w.sum(), 1.0, atol=1e-9):
        out.append('weights_do_not_sum_to_one')
    if np.max(w) >= 1.0 - 1e-12:
        out.append('degenerate_weight_vector')
    # Skewness and kurtosis are hard-bounded by n alone. Both bounds are
    # attained by the same configuration, n - 1 points equal and one apart:
    #
    #     |m3| / m2 ** 1.5  <=  (n - 2) / sqrt(n - 1)
    #      m4 / m2 ** 2     <=  (n ** 2 - 3n + 3) / (n - 1) = n - 2 + 1/(n - 1)
    #
    # and m4 / m2 ** 2 >= 1 always, which is beta2 >= beta1 + 1 with beta1 = 0.
    # A value outside these means the statistic was computed wrongly, not that
    # the data are unusual, which is why this belongs in a validity filter.
    # The 1/(n - 1) matters: an earlier version of this check used n - 2 and
    # rejected 12 percent of valid stratum 1 datasets.
    n = x.size
    if n >= 3:
        m = x.mean()
        m2 = np.mean((x - m) ** 2)
        if m2 > 0:
            g1 = np.mean((x - m) ** 3) / m2 ** 1.5
            if abs(g1) > (n - 2) / np.sqrt(n - 1) * (1 + 1e-9) + 1e-9:
                out.append('skewness_outside_sample_bound')
            g2 = np.mean((x - m) ** 4) / m2 ** 2
            hi_b = (n * n - 3 * n + 3) / (n - 1)
            if g2 > hi_b * (1 + 1e-9) + 1e-9 or g2 < 1 - 1e-9:
                out.append('kurtosis_outside_sample_bound')
    return out


def sample_moment_bounds(n):
    """The (|skewness|, kurtosis) box attainable by any n points.

    Returned as (max_abs_skew, min_kurt, max_kurt) with kurtosis in the
    non-excess convention. Used by the validity filter and reported in Part 6,
    where it explains the artifact at high negative excess kurtosis: at n = 3
    the excess kurtosis cannot exceed -1.5, so the whole stratum is pinned
    against a bound that has nothing to do with the generator.
    """
    n = int(n)
    if n < 3:
        return np.nan, np.nan, np.nan
    return ((n - 2) / np.sqrt(n - 1), 1.0, (n * n - 3 * n + 3) / (n - 1))

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
analyzed at all. Nothing is filtered for being statistically unusual.
"""

import numpy as np
from scipy import stats

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
def _truncated_normal(rng, mean, sd, lo, hi):
    """One draw from N(mean, sd) conditioned to [lo, hi], by inverse CDF.

    Inverse CDF rather than reject-and-redraw so the number of random values
    consumed does not depend on where the draw lands, which keeps a run
    reproducible from its seed regardless of the bounds.
    """
    a = stats.norm.cdf((lo - mean) / sd)
    b = stats.norm.cdf((hi - mean) / sd)
    u = a + (b - a) * rng.uniform()
    return float(mean + sd * stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)))


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


#: Spread multipliers the separation-driven solve scans. Geometric from very
#: blended to very separated, plus 0 for "no separation at all". A SCAN and not
#: a bisection because the coefficient of variation is not monotone in this
#: multiplier; see `genconfig.separation_dispersion_frac`.
SEPARATION_GRID = (0.0, 0.25, 0.5, 1.0, 1.75, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0,
                   55.0, 89.0)


def _floor_state(cs, pi, market, cfg):
    """(parent, shift) for a mixture placed at its POSITIVITY FLOOR.

    The floor is the smallest shift that keeps the support strictly positive,
    which is where a separation-driven parent sits: its dispersion is supposed
    to come from the spacing, so the shift does no work beyond admissibility.
    Returns None if the truncation bounds cannot be formed.
    """
    q1 = M.mixture_quantile(cs, pi, 0.25)
    q3 = M.mixture_quantile(cs, pi, 0.75)
    lo_raw, hi_raw = M.population_truncation_bounds(
        cs, pi, cfg.trunc_iqr_mult, clip_at_zero=False)
    span = hi_raw - lo_raw
    if not np.isfinite(span) or span <= 0:
        return None
    if cfg.trunc_rule == 'log':
        shift = max(0.0, cfg.min_q1_over_iqr * (q3 - q1) - q1)
        bounds = M.log_truncation_bounds(q1 + shift, q3 + shift,
                                         cfg.trunc_iqr_mult)
    else:
        q_low = M.mixture_quantile(cs, pi, cfg.max_low_tail_truncated)
        shift = max(0.0, 1e-4 * span - q_low)
        bounds = None
    if bounds is None:
        bounds = (max(lo_raw + shift, 1e-9 * span), hi_raw + shift)
    shifted = [_Shifted(d, shift) for d in cs] if shift > 0 else cs
    return M.MixtureParent(shifted, pi, market, bounds[0], bounds[1],
                           cfg.mode_coupling), shift


def _solve_spread_for_cv(build, pi, market, cfg, cv_target):
    """Spread multiplier whose mixture has `cv_target` at its positivity floor.

    Returns (comps, multiplier, achieved_cv, status). `status` is 'ok' when the
    scan lands within tolerance, 'clipped_max_cv' or 'clipped_min_cv' when the
    target is outside what any spacing can deliver for these components -- which
    is reported rather than silently approximated, the same contract the shift
    solve keeps.
    """
    best = None
    for c in SEPARATION_GRID:
        cs, _ = build(c)
        state = _floor_state(cs, pi, market, cfg)
        if state is None:
            continue
        parent, _shift = state
        mean, sd = parent.truncated_moments()
        if not (mean > 0 and np.isfinite(sd)):
            continue
        cv = sd / mean
        gap = abs(cv - cv_target)
        if best is None or gap < best[0]:
            best = (gap, c, cv, cs)
    if best is None:
        return None, None, None, 'separation_scan_failed'
    gap, c, cv, cs = best
    if gap <= 2e-3 * max(cv_target, 1e-9):
        status = 'ok'
    else:
        status = 'clipped_max_cv' if cv < cv_target else 'clipped_min_cv'
    return cs, c, cv, status


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
    z = (np.sort(rng.uniform(0.0, 1.0, k)) ** cfg.position_skew
         if k > 1 else np.zeros(1))

    # SHOULDER OR SEPARATED HUMPS. `z` is ascending, and `pi` and `solved` are
    # drawn independently of it, so by default the biggest component is as
    # likely to sit at the top of the range as at the bottom. Pairing the
    # largest weight and the widest component with the LOWEST position instead
    # makes the parent a dominant body with small narrow components on its
    # upper tail, which is the shape a real multimodal ECC category has. See
    # `genconfig.shoulder_frac` for the arithmetic; at 0.0 nothing changes and
    # the draw below still consumes no randomness, so a corpus generated with
    # shoulder_frac = 0 is bit-identical to one generated before this existed.
    shoulder = False
    if k > 1 and cfg.shoulder_frac > 0.0:
        shoulder = bool(rng.random() < cfg.shoulder_frac)
        if shoulder:
            order = np.argsort(-pi)               # heaviest mode first
            pi = pi[order]
            market = market[order]
            # widest component first, so the body is wide and the shoulders
            # are narrow. `solved` is (family, shape, loc, scale).
            solved = [solved[i] for i in order]
            # scale is element 3 of (family, shape, loc, scale). 'narrow' puts
            # the tightest component at the lowest position, which is the
            # sharp-peak-with-a-tail shape real categories have; 'wide' is the
            # direction that was tried first and measured worse.
            reverse = (cfg.shoulder_body == 'wide')
            solved = sorted(solved, key=lambda c: float(c[3]), reverse=reverse)
            specs = [specs[i] for i in order]

    def build(c):
        comps = []
        for (fam, shape, loc, scale), zi in zip(solved, z):
            comps.append(C.frozen(fam, shape, loc + c * float(zi), scale))
        return comps, pi

    # DISPERSION FROM SEPARATION, OR FROM THE SHIFT. The default path solves
    # the spacing for a drawn OVERLAP target and then solves the shift for a
    # drawn COEFFICIENT OF VARIATION, which means separating components can
    # only ever cost dispersion: a wider mixture has a wider interquartile
    # range, so its positivity floor is higher, so its achievable coefficient
    # of variation is lower. The alternative path solves the SPACING for the
    # coefficient of variation and leaves the shift at that floor, so a
    # dispersed parent is dispersed BECAUSE its components are far apart.
    # `genconfig.separation_dispersion_frac`; at 0.0 no randomness is consumed
    # here and the default path is bit-identical.
    separation = False
    if k > 1 and cfg.separation_dispersion_frac > 0.0:
        separation = bool(rng.random() < cfg.separation_dispersion_frac)

    sep_cv_target = sep_cv_status = None
    if separation:
        sep_cv_target = float(10 ** _truncated_normal(
            rng, cfg.cv_log10_mean, cfg.cv_log10_sd,
            cfg.cv_log10_lo, cfg.cv_log10_hi))
        comps, spread, sep_cv, sep_cv_status = _solve_spread_for_cv(
            build, pi, market, cfg, sep_cv_target)
        if comps is None:
            return None, dict(status='separation_scan_failed', k=k)
        target = float('nan')
        overlap = (float(M.min_adjacent_overlap(comps, pi))
                   if cfg.overlap_statistic == 'min_adjacent'
                   else float(M.average_overlap(comps, pi)))
        ov_status = f'separation_driven(c={spread:g})'
    elif k > 1:
        target = float(10 ** rng.uniform(cfg.overlap_log10_lo, cfg.overlap_log10_hi))
        comps, overlap, ov_status = M.solve_spread_for_overlap(
            build, target, statistic=cfg.overlap_statistic)
    else:
        target, overlap, ov_status = 0.0, 0.0, 'single_component'
        comps, _ = build(0.0)

    # Place the mixture relative to zero. ECC support is (0, inf), open at
    # zero (decision 13), so the mixture has to be shifted up; the question is
    # by how much, and it is the most consequential number in the generator.
    #
    # Shifting a distribution changes its mean and leaves its standard
    # deviation alone, so the shift IS the coefficient of variation: for a
    # parent normalized to mean 1, cv(s) = sd(s) / mean(s), decreasing in s.
    # The target is drawn from the configuration and solved for by bisection.
    #
    # Three earlier attempts got this wrong in instructive ways. Shifting above
    # the 1e-9 quantile of the heaviest-tailed component put the median
    # dataset's support at 0.65 of its own mean and drove the median
    # coefficient of variation to 0.049. Shifting so that Q1 - 3 * IQR landed
    # on zero capped it near 0.5 whatever the component shapes, because it
    # forces the mean to about 3.5 * IQR with the upper bound at 7 * IQR.
    # Neither is a statement about the data; both are artifacts of choosing a
    # shift for a reason unrelated to what the shift controls. The third was
    # keeping the ADDITIVE interquartile rule here while the empirical arm
    # moved to the multiplicative one, which left the two arms truncated
    # differently on a dimension the study is about; see below.
    #
    # Quartiles TRANSLATE under a shift, so both truncation rules can be
    # evaluated at any shift from the unshifted quartiles alone, and the
    # bisection stays cheap.
    q1_0 = M.mixture_quantile(comps, pi, 0.25)
    q3_0 = M.mixture_quantile(comps, pi, 0.75)
    lo0_raw, hi0 = M.population_truncation_bounds(comps, pi, cfg.trunc_iqr_mult,
                                                  clip_at_zero=False)
    span = hi0 - lo0_raw

    if cfg.trunc_rule == 'log':
        # The multiplicative rule needs a strictly positive first quartile,
        # which is also exactly the condition for its lower bound to be
        # positive, so positivity is enforced here rather than by a separate
        # tail-mass budget.
        shift_min = max(0.0, cfg.min_q1_over_iqr * (q3_0 - q1_0) - q1_0)
    else:
        q_low = M.mixture_quantile(comps, pi, cfg.max_low_tail_truncated)
        shift_min = max(0.0, 1e-4 * span - q_low)

    # On the separation path the target was drawn before the spacing solve,
    # because the spacing is what has to hit it.
    cv_target = sep_cv_target if separation else float(10 ** _truncated_normal(
        rng, cfg.cv_log10_mean, cfg.cv_log10_sd, cfg.cv_log10_lo, cfg.cv_log10_hi))

    def bounds_at(sh):
        """Truncation bounds of the mixture shifted by sh."""
        if cfg.trunc_rule == 'log':
            b = M.log_truncation_bounds(q1_0 + sh, q3_0 + sh, cfg.trunc_iqr_mult)
            if b is not None:
                return b
        return max(lo0_raw + sh, 1e-9 * span), hi0 + sh

    def parent_at(sh):
        # The COMPONENTS move with the bounds. Truncating the unshifted mixture
        # to shifted bounds is a different distribution, and doing that made
        # the solve miss its target by 59 percent at the median.
        cs = [_Shifted(d, sh) for d in comps] if sh > 0 else comps
        lo, h = bounds_at(sh)
        return M.MixtureParent(cs, pi, market, lo, h, cfg.mode_coupling)

    def cv_at(sh):
        m, sd = parent_at(sh).truncated_moments()
        return (sd / m) if m > 0 else np.inf

    if separation:
        # The spacing already carries the dispersion, so the shift does only
        # what admissibility requires and the scan's verdict is the status.
        shift, cv_status = shift_min, sep_cv_status
    else:
        shift, cv_status = _solve_shift_for_cv(cv_at, cv_target, shift_min, span)
    if shift > 0:
        comps = [_Shifted(d, shift) for d in comps]

    lo, hi = bounds_at(shift)

    parent = M.MixtureParent(comps, pi, market, lo, hi, cfg.mode_coupling)
    m_fin, sd_fin = parent.truncated_moments()
    cv_achieved = (sd_fin / m_fin) if m_fin > 0 else np.nan

    # A mode much narrower than the dataset it sits in draws as a spike. The
    # overlap solve moves LOCATIONS and leaves component widths alone, so a low
    # overlap target spreads fixed-width components over a wider parent and can
    # produce exactly that. Reject and let the caller redraw; see
    # genconfig.min_mode_sd_frac.
    # No getattr fallback here. If a component cannot report its own standard
    # deviation that is a defect to surface, not a NaN to skip the check with:
    # an earlier version swallowed it and the floor silently never fired.
    comp_sds = [float(d.std()) for d in parent.comps]
    narrowest = min(comp_sds) if comp_sds and np.all(np.isfinite(comp_sds)) else np.nan
    mode_sd_frac = (narrowest / sd_fin) if (sd_fin and sd_fin > 0) else np.nan
    if np.isfinite(mode_sd_frac) and len(parent.comps) > 1 \
            and mode_sd_frac < cfg.min_mode_sd_frac:
        return None, dict(status='mode_too_narrow', k=k,
                          mode_sd_frac=float(mode_sd_frac),
                          overlap_achieved=overlap)

    # A PARENT WHOSE MEAN IS DOMINATED BY MASS ITS OWN SAMPLE NEVER SEES.
    #
    # THIS IS THE CHECK STAGE 2h WAS MISSING AND IT COST A WHOLE REGENERATION.
    # Every characteristic the tuning objective scores is computed on the
    # SAMPLE, and the sample is divided by its own mean. A parent with a very
    # long thin upper tail produces samples that look entirely reasonable --
    # they almost never draw from that tail -- while the parent's OWN mean is
    # set by it. Stage 2h widened the truncation and generated a corpus whose
    # parents had a mean of 6,624 against data normalized to 1.0, improving
    # every sample statistic by 8.4 standard errors on the way. Nothing in the
    # loop could see it, because nothing in the loop looked at the parent.
    #
    # THE PARENT IS THE TRUTH EVERY RECOVERY SCORE IS MEASURED AGAINST, so it
    # is checked here, in the generation path, rather than downstream.
    #
    # The test is scale free by construction: the ratio of the parent's own
    # mean to its own median. For a lognormal that ratio is exp(sigma^2 / 2),
    # so it is about 1.6 at sigma = 1 and 90 at sigma = 3, and it grows without
    # bound as the tail lengthens. It cannot be fooled by a change of units,
    # which the raw mean can.
    med = float(np.ravel(parent.ppf(0.5))[0])
    mean_over_median = (m_fin / med) if med > 0 else np.inf
    if not np.isfinite(mean_over_median) \
            or mean_over_median > cfg.max_parent_mean_over_median:
        return None, dict(status='parent_tail_dominated', k=k,
                          mean_over_median=float(mean_over_median),
                          parent_mean=float(m_fin), parent_median=med,
                          overlap_achieved=overlap)
    record = dict(status='ok', k=k, overlap_target=target, overlap_achieved=overlap,
                  overlap_statistic=cfg.overlap_statistic,
                  overlap_avg=float(M.average_overlap(comps, pi)) if k > 1 else 0.0,
                  overlap_min_adjacent=float(M.min_adjacent_overlap(comps, pi))
                  if k > 1 else 0.0,
                  trunc_rule=cfg.trunc_rule, shoulder=shoulder,
                  separation_driven=separation,
                  overlap_status=ov_status, component_retries=retries,
                  components=specs, pi=pi.tolist(), market=market.tolist(),
                  shift=shift, lo=lo, hi=hi,
                  cv_target=cv_target, cv_achieved=cv_achieved,
                  mode_sd_frac=float(mode_sd_frac) if np.isfinite(mode_sd_frac)
                  else None,
                  cv_status=cv_status, shift_min=shift_min,
                  floor_ratio_achieved=(lo / m_fin) if m_fin > 0 else np.nan,
                  truncated_mass=parent.truncated_mass(),
                  n_components_dropped=parent.n_components_dropped,
                  k_effective=len(parent.comps))
    return parent, record


def _solve_shift_for_cv(cv_at, target, shift_min, span, iters=40, tol=2e-3):
    """Smallest shift whose parent has the requested coefficient of variation.

    cv is decreasing in the shift, so a bisection is enough. `shift_min` is the
    positivity floor; if even that gives less spread than asked for, the target
    is unreachable for this mixture and the status records it rather than the
    generator quietly returning something else.
    """
    cv_lo_shift = cv_at(shift_min)
    if cv_lo_shift <= target * (1 + tol):
        # even the smallest allowed shift is not spread out enough
        return shift_min, ('ok' if abs(cv_lo_shift - target) <= tol * target
                           else 'clipped_max_cv')

    # expand upward until the coefficient of variation drops below the target
    a = shift_min
    b = shift_min + max(span, 1e-12)
    for _ in range(200):
        if cv_at(b) < target:
            break
        b = shift_min + (b - shift_min) * 2.0
        if b - shift_min > 1e14 * max(span, 1e-12):
            return b, 'clipped_min_cv'
    else:
        return b, 'clipped_min_cv'

    for _ in range(iters):
        mid = 0.5 * (a + b)
        if cv_at(mid) > target:
            a = mid
        else:
            b = mid
        if (b - a) <= 1e-10 * max(span, 1.0):
            break
    sh = 0.5 * (a + b)
    got = cv_at(sh)
    return sh, ('ok' if abs(got - target) <= tol * max(target, 1e-9)
                else 'tolerance_not_met')


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

    def std(self):
        # A shift moves the distribution and leaves its spread alone.
        return float(self._d.std())

    def spec(self):
        """The wrapped component with the shift folded into its location.

        A shift and an affine location are the same operation, so a shifted
        component is exactly `components.frozen(family, shape, loc + shift,
        scale)`. Folding them here means a serialized parent needs no wrapper
        of its own to rebuild.
        """
        d = self._d.spec()
        return dict(d, loc=d['loc'] + self.shift)


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


#: Parent-draw outcomes that are a REJECTED DRAW rather than a failure, so a
#: fresh draw of the random targets is the right response.
#:
#:   parent_tail_dominated       the parent's own mean is more than
#:                               max_parent_mean_over_median times its own
#:                               median, so the mean is set by mass its own
#:                               sample will essentially never draw. Added in
#:                               Stage 2h on a diagnosis later shown to be
#:                               wrong -- it does not fire on the corpus that
#:                               prompted it, whose parents had a mean over
#:                               median of 1.31 -- and kept because the
#:                               failure mode it names is real and invisible
#:                               at the sample level. See the field's
#:                               docstring in genconfig.py for what actually
#:                               broke that run
#:   mode_too_narrow             a mode came out narrower than min_mode_sd_frac
#:   component_targets_exhausted the skewness and excess kurtosis drawn for a
#:                               component could only be met by a J-shaped
#:                               density, which is refused because it has
#:                               infinite density at an endpoint and is drawn
#:                               as a spike
#:
#: Both describe targets that happened to land somewhere unusable. The targets
#: are themselves random, so drawing new ones almost always succeeds; abandoning
#: the slot instead leaves the corpus one dataset short of what was asked for,
#: and the stratum counts then disagree with the design. Refusing to APPROXIMATE
#: a target that cannot be met is a different thing and still holds.
REDRAWABLE = frozenset({'mode_too_narrow', 'component_targets_exhausted',
                        'parent_tail_dominated'})


def generate_dataset(cfg, n, rng):
    """One synthetic ECC dataset. Returns (values, weights, record).

    `values` has unweighted mean exactly 1.0, by decision 6 (Stage 1 amendment
    A3): the normalizer is the unweighted SAMPLE mean in both the synthetic and
    the empirical path, because a practitioner holding a set of EPDs can
    compute that and cannot compute a market-weighted mean. The divisor is in
    the record, so the parent is still exactly specified.
    """
    parent, record, parent_retries = None, None, 0
    for parent_retries in range(cfg.max_parent_retries):
        parent, record = draw_parent(cfg, n, rng)
        if parent is not None:
            break
        if record.get('status') not in REDRAWABLE:
            break          # not a rejected draw: retrying cannot help
    if parent is None:
        return None, None, record
    x, modes = parent.sample(n, rng)
    w = draw_weights(parent, modes, cfg, rng)
    record = dict(record)
    record['n'] = int(n)
    record['normalizer'] = parent.normalizer
    record['parent_retries'] = int(parent_retries)
    record['mode_counts'] = np.bincount(modes, minlength=len(parent.comps)).tolist()
    return x, w, record


# --------------------------------------------------------------------------
# Part 2: the validity filter
# --------------------------------------------------------------------------
def validity_failures(x, w, n_expected=None):
    """Reasons this dataset cannot be analyzed. Empty list means it is valid.

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
    elif np.ptp(x) < 1e-9 * max(abs(float(np.mean(x))), 1e-300):
        # Not literally constant, but too narrow for the metrics to mean
        # anything: a 256-bin histogram cannot be formed, the Shapiro-Wilk
        # statistic is dominated by float spacing, and a KDE bandwidth is
        # meaningless. The configuration can request a coefficient of variation
        # this small; such datasets are rejected here rather than silently
        # producing metrics that describe rounding.
        out.append('degenerate_range')
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

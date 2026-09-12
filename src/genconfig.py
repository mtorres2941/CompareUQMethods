"""Generation configuration: every knob in one place, none of them hand-set.

Stage 2a's charge is a generator whose every parameter comes from a
configuration rather than from a tuning history. This module is that
configuration. `DEFAULT` is the one the corpus is generated from; the fields
are the only things a sweep in Stage 2h needs to vary.

Where a range derives from a measured property of the 138 empirical ECC
datasets, the docstring for that field says which. The audit tables under
`outputs/tables/stage2a/` hold the measurements.

Nothing here is a tuning knob in the sense the removed steps were. The power
transform `data ** rng.uniform(0.9, 4.0)` and the 25 percent reflection existed
because they made the output look right; they had no interpretation and no
target. Every field below is a distributional property with a name, and the
generator reports what it achieved against what was asked.
"""

from dataclasses import dataclass, field, asdict, replace
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Stratum:
    """One dataset-size stratum. Sizes are drawn log-uniformly inside it."""
    name: str
    n_lo: int
    n_hi: int
    n_datasets: int


# Equal allocation across four size strata, so every size regime is estimated
# with the same precision. The empirical sizes run 3 to 77,548 with a median of
# 37, and a single log-uniform draw either leaves the large regime too sparse to
# analyse or lets large datasets dominate every aggregate. Because equal
# allocation does not match the empirical size distribution, every headline
# aggregate is reported twice: per stratum, and reweighted by the empirical
# frequency of each stratum (EMPIRICAL_STRATUM_SHARE below).
STRATA = (
    Stratum('s1_3_9',        3,    9, 2500),
    Stratum('s2_10_99',     10,   99, 2500),
    Stratum('s3_100_999',  100,  999, 2500),
    Stratum('s4_1000_9999', 1000, 9999, 2500),
)

# The probe set sits OUTSIDE the corpus and is excluded from every aggregate.
# Its only job is to establish whether results have plateaued by n = 10 ** 4.
# Six empirical datasets exceed 9,999, reaching 77,548, and they include the two
# largest and most carbon-significant materials.
PROBE = Stratum('probe_10k_100k', 10_000, 100_000, 50)

# Share of the 138 empirical datasets falling in each stratum, measured in
# audits/stage2a/a5_empirical_envelope.py. Used for post-stratification
# reweighting, never for generation.
EMPIRICAL_STRATUM_SHARE = {
    's1_3_9': 31 / 138,
    's2_10_99': 68 / 138,
    's3_100_999': 33 / 138,
    's4_1000_9999': 5 / 138,
}


@dataclass(frozen=True)
class GeneratorConfig:
    """Every parameter of the synthetic ECC generator."""

    # ---- mixture structure -------------------------------------------------
    strata: Tuple[Stratum, ...] = STRATA
    probe: Stratum = PROBE
    k_min: int = 1
    k_max: int = 5
    """Number of mixture components. Unchanged from the old generator's
    rng.integers(1, 6)."""

    # ---- component separation, as overlap ----------------------------------
    overlap_log10_lo: float = -3.5
    overlap_log10_hi: float = np.log10(0.75)
    """Target average pairwise overlap, drawn log-uniformly in this range and
    then solved for by moving the component locations (Maitra and Melnykov
    2010, step 3). Overlap replaces the old locs ~ U(5, 20) with
    scales ~ U(0.2, 1.5), which placed components tens of standard deviations
    apart and so produced well-separated clusters rather than the partially
    merged shoulders real material categories show.

    The upper end is set from measurement, not taste. Fitting a BIC-selected
    Gaussian mixture to each of the 138 empirical datasets and computing the
    same overlap functional gives a median of 0.0218, a 95th percentile of
    0.4528 and a maximum of 0.6719; the same estimator on the shipped synthetic
    corpus gives a median of 0.0037, about six times less overlap than the
    empirical data at the median. 0.75 covers the empirical maximum with
    margin. A log-uniform draw over [1e-4, 0.75] has median 0.0087, which
    brackets the empirical median from below. See
    outputs/tables/stage2a/TABLE_2a_OverlapComparison.csv."""

    # ---- component shapes, as moment targets -------------------------------
    comp_skew_lo: float = -3.0
    comp_skew_hi: float = 8.0
    """Component skewness target, drawn uniformly.

    Deliberately asymmetric. Real ECC datasets are predominantly right skewed
    (the 138 empirical datasets have a median skewness of 1.055 and run from
    -1.44 to 4.62), and a symmetric component range produces a corpus with a
    median skewness near zero, which is not what the data look like. The range
    keeps a substantial negative arm anyway, because the corpus has to contain
    left-skewed datasets for a generalizability claim even though the empirical
    set has few. A MIXTURE can be more skewed than any of its components, and
    small-sample noise widens the realized range further."""

    comp_exkurt_lo: float = -1.2
    comp_exkurt_hi: float = 60.0
    """Component excess kurtosis target, drawn uniformly and then clipped up to
    the feasible boundary skewness ** 2 - 2 plus a margin. -1.2 is the uniform
    distribution, the platykurtic limit of any unimodal shape."""

    position_skew: float = 5.0
    """How component locations are spread between the two ends of the mixture.

    Locations are placed at z ** position_skew for z uniform on (0, 1), before
    the overlap solve scales the whole arrangement. At 1 the components sit
    uniformly across the range; above 1 they cluster toward the low end with
    the occasional far-out one, which is what a material category looks like
    when most products are similar and a few are much more carbon intensive.

    It matters because it, not the component shapes, sets the typical
    achievable coefficient of variation. Components spread uniformly over a
    range of width c give a mixture with mean about c/2 and standard deviation
    about c/3.5, so a coefficient of variation near 0.57 whatever the
    components are. That is why 47 percent of targets were unreachable at
    position_skew = 1, and why the corpus came out centred at 0.24 against an
    empirical 0.600."""

    comp_sd_log10_lo: float = -0.7
    comp_sd_log10_hi: float = 0.3
    """Component standard deviations, drawn log-uniformly relative to a common
    unit. Only ratios matter: the dataset is divided by its own mean at the
    end, so the overall scale is unidentified."""

    # ---- how many points fall in each mode ---------------------------------
    mode_share_alpha: float = 10.0
    """Dirichlet concentration for the SAMPLING weights pi, which set how many
    points fall in each mode. 10 is the old generator's cpv = ones(k) * 10.

    It gives modes near-equal shares and so leaves mode dominance almost
    constant across the corpus: at k = 2 the larger mode holds between 0.501
    and 0.760 of the points, median 0.569, and at k = 5 the smallest mode never
    falls below 0.086. Kept at 10 here so that this stage changes one thing at
    a time and the new corpus stays comparable to the old on this axis; Stage
    2h owns the sweep, and alpha = 1 is the obvious other end. See
    outputs/tables/stage2a/TABLE_2a_ModeShares.csv."""

    # ---- market share, Part 3 ----------------------------------------------
    market_share_alpha: float = 1.0
    """Dirichlet concentration for the MODE-level market shares, drawn
    independently of the sampling weights. alpha = 1 is the flat Dirichlet, the
    maximum-entropy prior over unknown market shares."""

    point_weight_alpha: float = 1.0
    """Dirichlet concentration for the point weights within a mode. alpha = 1
    in both arms, per the Part 0 item 2 decision; the empirical arm previously
    used 5."""

    mode_coupling: float = 1.0
    """How much of a point's market weight is determined by which mode it is
    in. 0 reproduces the old uncoupled behaviour, in which the market-weighted
    distribution existed only on the realized sample and had no population to
    be right or wrong about. 1 makes market share fully mode-determined, so the
    market-weighted parent is the mixture sum_k v_k f_k. Swept."""

    # ---- spread, as a target rather than a side effect ---------------------
    cv_log10_mean: float = -0.2641
    cv_log10_sd: float = 0.2913 * 2.0
    cv_log10_lo: float = np.log10(0.004)
    cv_log10_hi: float = np.log10(3.2)
    """Target coefficient of variation of the population parent, drawn from a
    normal in log10 truncated to [lo, hi], and then solved for exactly.

    The centre and spread come from the 138 empirical datasets, whose log10
    coefficient of variation has mean -0.2641 and standard deviation 0.2913,
    a median coefficient of variation of 0.544. The standard deviation is
    DOUBLED, which is what "margin beyond the empirical envelope" means here:
    the corpus is centred where the real data are and reaches roughly twice as
    far in each direction, so a generalizability claim has something to stand
    on.

    A log-UNIFORM draw over the same range was tried first and rejected. It
    gives even coverage of every regime, which is attractive for the
    metric-versus-W1 modelling in Stage 2f, but its geometric centre is 0.113
    and the corpus came out with a median coefficient of variation of 0.071
    against an empirical 0.600. Range coverage was 98.6 percent and the corpus
    still did not look like the data, which is the actual requirement.

    This is the parameter that replaces the removed power transform. That step
    raised every value to a power drawn from U(0.9, 4.0) with an inline comment
    saying the purpose was to align with empirical ECC data; its actual job was
    to manufacture spread, and nothing else in the generator produced any. With
    it gone and nothing in its place, the first Stage 2a regeneration came out
    with a median coefficient of variation of 0.049 against an empirical 0.600,
    which is not a corpus with margin around the empirical data, it is a
    different population.

    The difference from the power transform is that this is a named
    distributional property with a target, solved for and reported, not a knob
    whose effect is discovered afterwards. For data on (0, inf) normalized to
    mean 1, the coefficient of variation is set by how far the distribution
    sits from the origin relative to its own spread, and shifting a mixture
    changes its mean while leaving its standard deviation alone, so the
    solution is a single bisection on the shift.

    The range brackets the 138 empirical datasets, which run from 0.0066 to
    2.40, with margin at both ends."""

    max_low_tail_truncated: float = 0.15
    """The largest share of the parent's probability the generator may discard
    below zero when placing the mixture.

    Positivity is enforced by the truncation at `lo`, not by the shift, so this
    is not a correctness requirement: it is a limit on how far a distribution
    may be pushed toward the origin in pursuit of its target coefficient of
    variation. Setting it to 1e-4 capped the achievable spread and left 39
    percent of datasets unable to reach their target; 0.15 leaves the median
    dataset discarding 0.0002 of its mass and the 95th percentile 0.157."""

    # ---- truncation --------------------------------------------------------
    trunc_iqr_mult: float = 3.0
    """Bounds are max(Q1 - mult * IQR, 0) and Q3 + mult * IQR of the POPULATION
    mixture. Same rule as the old generator; what changed is that the quantiles
    are the parent's rather than one realized sample's."""

    # ---- reproducibility ---------------------------------------------------
    seed: int = 42
    max_component_retries: int = 12
    """A moment target can be infeasible or numerically degenerate. The
    generator redraws the target that many times and records how often it had
    to, rather than silently substituting a different shape."""

    def stratum_by_name(self, name):
        for s in self.strata:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def n_datasets(self):
        return sum(s.n_datasets for s in self.strata)

    def to_dict(self):
        d = asdict(self)
        d['strata'] = [asdict(s) for s in self.strata]
        d['probe'] = asdict(self.probe)
        return d

    def replace(self, **kw):
        return replace(self, **kw)


DEFAULT = GeneratorConfig()

"""Generation configuration: every knob in one place, none of them hand-set.

Stage 2a's charge is a generator whose every parameter comes from a
configuration rather than from a tuning history. This module is that
configuration. `DEFAULT` is the one the corpus is generated from; the fields
are the only things a sweep in Stage 2h needs to vary.

Where a range derives from a measured property of the empirical ECC datasets,
the docstring for that field says which. The audit tables under
`outputs/tables/stage2a/` and `outputs/tables/stage2a2/` hold the measurements.

Every empirical figure quoted below was remeasured in Stage 2a-2 against the
2026-08 raw extract. The 2026-03 figures the ranges were originally set from
were distorted: that file had been trimmed additively at the high end before it
was stored, which cuts the right tail of every dataset and so understates the
spread, the skewness and the multimodality of the real data.

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
# with the same precision. The empirical sizes run 3 to 86,770 with a median of
# 53, and a single log-uniform draw either leaves the large regime too sparse to
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
# One empirical dataset exceeds 9,999, ReadyMix at 86,770, and it is the largest
# and most carbon-significant material category in the set.
PROBE = Stratum('probe_10k_100k', 10_000, 100_000, 50)

# Share of the empirical datasets falling in each stratum, measured in
# audits/stage2a2/p6_empirical_envelope.py on the 2026-08 arm. Used for
# post-stratification reweighting, never for generation.
#
# These moved when the empirical extract was rebuilt from raw values: the
# smallest stratum fell from 0.2246 to 0.0956 and the second rose from 0.4928 to
# 0.5588, because categories the old additive high-end trim had cut to a handful
# of values now keep more of them. A further 0.0074 of the arm sits above 9,999
# and is covered by the probe set rather than by a stratum.
EMPIRICAL_STRATUM_SHARE = {
    's1_3_9': 13 / 136,
    's2_10_99': 76 / 136,
    's3_100_999': 40 / 136,
    's4_1000_9999': 6 / 136,
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
    overlap_log10_lo: float = -2.5
    overlap_log10_hi: float = np.log10(1.4)
    """Target average pairwise overlap, drawn log-uniformly in this range and
    then solved for by moving the component locations (Maitra and Melnykov
    2010, step 3).

    Overlap is the probability that a value drawn from one mode would be
    attributed to another, so a target near 1 means the modes are barely
    distinguishable and a target near 1e-3 means they are separate clusters.

    This is the single most consequential parameter in the configuration, and it
    is set against a DISTRIBUTION rather than a range. Two independent
    measurements of the empirical arm set it, and they agree.

    The first is the overlap itself. Fitting a BIC-selected Gaussian mixture to
    each empirical dataset and computing the same Maitra-Melnykov overlap from
    the fit puts both arms on one footing. On the 2026-08 arm that gives
    quartiles 0.0020, 0.0474 and 0.1078, a 95th percentile of 0.2671 and a
    maximum of 0.5956. The range above brackets that with margin at both ends.

    The second is how many modes a dataset appears to have, by Silverman's test.
    Empirical, on 136 datasets: 49.3 percent unimodal, 39.7 bimodal, 9.6
    trimodal, 1.4 with four or more.

    HISTORY, because this parameter has now been wrong in both directions and
    the reason is instructive. Stage 2a set it to [0.3, 1.4], reasoning that
    components should merge, and tuned that against an empirical arm which was
    81.9 percent unimodal. That figure was itself an artifact: the 2026-03
    extract had been trimmed additively at the high end before it was stored,
    which cuts the right tail and suppresses the modality the test can see. On
    raw 2026-08 data cleaned symmetrically the empirical arm is 49.3 percent
    unimodal, and the fitted empirical overlap distribution sits ENTIRELY BELOW
    the [0.3, 1.4] range that had been tuned in. Two independent measurements
    say the same thing, and both were wrong before for the same reason.

    Measured sweep of the lower bound on the 2026-08 arm, at the coefficient of
    variation centre below, reporting the weighted objective, the mode-count
    total variation distance and the unimodal share:

        lower bound   objective   mode TV   unimodal
        0.3 (2a)        0.4953      0.358      85.1%
        1e-1.5          0.4524      0.167      63.1%
        1e-2.5          0.4261      0.088      49.5%

    THE COST, stated because it is real. Driving the lower bound down improves
    every weighted characteristic but makes the corpus less lognormal-looking
    than the data: fit_lognorm_SW rises from 0.93 to 1.83 standardized W1. Real
    ECC datasets manage to be 49 percent multimodal while still having a median
    Shapiro-lognormal statistic of 0.937, so their modes are gentle shoulders on
    a lognormal body rather than separate clusters. The generator reproduces the
    mode COUNT without reproducing that gentleness.

    See audits/stage2a/b5_tune_configuration.py, which is the script that
    produced these numbers and the one to re-run after any change."""

    # ---- component shapes, as moment targets -------------------------------
    comp_skew_lo: float = -3.0
    comp_skew_hi: float = 8.0
    """Component skewness target, drawn uniformly.

    Deliberately asymmetric. Real ECC datasets are predominantly right skewed
    (the 2026-08 empirical arm has a median dataset skewness of 2.06, a 95th
    percentile of 8.15 and a range of -1.24 to 20.65), and a symmetric component
    range produces a corpus with a median skewness near zero, which is not what
    the data look like. The range keeps a substantial negative arm anyway,
    because the corpus has to contain left-skewed datasets for a generalizability
    claim even though the empirical set has few. A MIXTURE can be more skewed
    than any of its components, and small-sample noise widens the realized range
    further.

    Raising the upper bound to 14 was tried in Stage 2a-2 and rejected: it made
    the corpus skewness distribution WORSE, not better (standardized W1 0.540 to
    0.631), because the components then need placing further apart to hit their
    overlap target and the mixture stops looking like the data. The empirical
    median moved from 1.055 to 2.06 when the arm was rebuilt from raw values, and
    the mixture reaches that without a wider component range."""

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
    falls below 0.086. Kept at 10 through Stage 2a-2 as well, so that the
    retune changes the separation of the modes and not also how many points
    land in each; Stage 2h owns the sweep, and alpha = 1 is the obvious other
    end. See outputs/tables/stage2a/TABLE_2a_ModeShares.csv."""

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
    cv_log10_mean: float = 0.211
    cv_log10_sd: float = 0.3752 * 2.0
    cv_log10_lo: float = np.log10(0.004)
    cv_log10_hi: float = np.log10(16.0)
    """Target coefficient of variation of the population parent, drawn from a
    normal in log10 truncated to [lo, hi], and then solved for exactly.

    The spread comes from the empirical datasets, whose log10 coefficient of
    variation has a standard deviation of 0.3752 on the 2026-08 arm; it is
    DOUBLED here, which is what "margin beyond the empirical envelope" means:
    the corpus reaches about twice as far in each direction as the real data do.

    The centre is NOT the empirical centre. It sits above it, because this is a
    target for the POPULATION coefficient of variation while the characteristic
    being matched is the SAMPLE one, and the sample value of a right-skewed
    distribution runs systematically low: a finite sample rarely contains the far
    tail. The offset was measured, not assumed.

    All three of these numbers moved when the empirical arm was rebuilt from raw
    values in Stage 2a-2, and they moved a long way, because the 2026-03 extract
    had been trimmed additively at the high end before it was stored and the
    right tail is what carries the spread:

                                 2026-03 arm   2026-08 arm
        median coefficient of variation  0.600        0.782
        log10 standard deviation        0.2913       0.3752
        maximum                           2.40        13.40

    The upper truncation was raised from 3.2 to 16 for the same reason: at 3.2 it
    no longer bracketed the empirical maximum, so the draw was being clipped
    inside the range the corpus is meant to cover with margin. The lower bound of
    0.004 still sits below the empirical minimum of 0.0081.

    Measured sweep of the centre on the 2026-08 arm, reporting the standardized
    W1 of the coefficient of variation and the weighted objective:

        centre        coeffvar W1   objective
        0.011 (2a)       0.391        0.5087
        0.211            0.265        0.4261
        0.361            0.253        0.4400

    0.211 is kept rather than 0.361: the two are within 0.012 on the
    characteristic being targeted, and the higher centre is worse overall,
    because pushing the mixture further from the origin costs skewness and
    lognormality.

    This is the parameter that replaces the removed power transform. That step
    raised every value to a power drawn from U(0.9, 4.0) with an inline comment
    saying the purpose was to align with empirical ECC data; its actual job was
    to manufacture spread, and nothing else in the generator produced any. The
    difference is that this is a named distributional property with a target,
    solved for and reported, not a knob whose effect is discovered afterwards.
    For data on (0, inf) normalized to mean 1, the coefficient of variation is
    set by how far the distribution sits from the origin relative to its own
    spread, and shifting a mixture changes its mean while leaving its standard
    deviation alone, so the solution is a single bisection on the shift.

    A log-UNIFORM draw over the same range was tried in Stage 2a and rejected: it
    gives even coverage of every regime, which is attractive for the
    metric-versus-W1 modelling in Stage 2f, but the corpus came out with a median
    coefficient of variation of 0.071 against an empirical 0.600. Range coverage
    was 98.6 percent and the corpus still did not look like the data, which is
    the actual requirement."""

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

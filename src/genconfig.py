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
# of values now keep more of them. A further 0.0070 of the arm sits above 9,999
# and is covered by the probe set rather than by a stratum.
#
# Remeasured in Stage 2a-3 on the 149-dataset arm, after the categories were
# resolved into specifiable products: EC3 residual bins dropped, concrete split
# by specified compressive strength, insulation by material type. The smallest
# stratum rises from 0.0956 to 0.1342, because splitting produces smaller
# datasets, and three datasets now sit above 9,999 rather than one. This is a
# post-stratification weight and NOT a generation parameter, so it moves
# reported aggregates but requires no regeneration.
EMPIRICAL_STRATUM_SHARE = {
    's1_3_9': 20 / 149,
    's2_10_99': 79 / 149,
    's3_100_999': 39 / 149,
    's4_1000_9999': 8 / 149,
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
    overlap_log10_lo: float = np.log10(0.3)
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

    THIS IS STAGE 2A'S ORIGINAL RANGE, RESTORED. Stage 2a-2 lowered it, through
    several values down to 1e-3.0, and every one of those was a regression. The
    reason is worth stating plainly because it cost a whole stage.

    The retune was steered by the SILVERMAN mode-count distribution, which the
    2026-08 empirical arm puts at 49.3 percent unimodal. Driving the overlap
    down did match that. But Silverman's critical-bandwidth test detects
    structure at ANY bandwidth, including fine structure that never appears in
    a plot, and the empirical datasets that it calls multimodal are single
    right-skewed humps to look at: 94.9 percent of them have exactly one mode
    visible in a default-bandwidth KDE. So the corpus was rebuilt out of clearly
    separated humps in order to match a count that, in the real data, comes from
    something else entirely.

    Sweep on the 2026-08 arm, scoring both measures (modality.n_modes_visible
    and modality.n_modes_silverman):

        range              objective   visible TV   visible unimodal
        [1e-2.5, 0.9]        0.4584       0.280          66.8%
        [0.05, 1.0]          0.4754       0.273          67.6%
        [0.15, 1.4]          0.4486       0.074          87.4%
        [0.3, 1.4]           0.4504       0.022          92.7%   <- chosen
        [0.5, 2.0]           0.4475       0.029          97.8%

    against an empirical 94.9 percent visible-unimodal. [0.5, 2.0] scores
    marginally better on the objective and overshoots to 97.8; [0.3, 1.4] is the
    closest match on the measure that corresponds to what the data look like,
    and the four objectives span 2 percent, which is inside the seed-to-seed
    noise measured in audits/stage2a2/p10_config_noise.py.

    The lesson, recorded because it is the expensive one: a modality statistic
    that is matched can coexist with a modality mismatch that is obvious in a
    figure, and only the measure taken at the bandwidth a reader sees will catch
    it. Tune against n_modes_visible; keep n_modes_silverman as a
    characteristic, not as a steering signal.

        An earlier sweep of the lower bound while targeting the AVERAGE overlap is
    kept for the record, because it is what made the average's inadequacy
    visible: 0.3 gave 85.1 percent unimodal, 1e-1.5 gave 63.1 and 1e-2.5 gave
    49.5, but the last of those reached the right mode COUNT by opening gaps,
    and cost fit_lognorm_SW 1.828 and six-or-more modes in 6.5 percent of the
    corpus against an empirical 0.7.

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
    cv_log10_mean: float = 0.129
    cv_log10_sd: float = 0.3919 * 2.0
    cv_log10_lo: float = np.log10(0.004)
    cv_log10_hi: float = np.log10(16.0)
    """Target coefficient of variation of the population parent, drawn from a
    normal in log10 truncated to [lo, hi], and then solved for exactly.

    The spread comes from the empirical datasets, whose log10 coefficient of
    variation has a standard deviation of 0.3919 on the 2026-08 arm once its
    categories are resolved into specifiable products (Stage 2a-3); it is DOUBLED here, which is what "margin beyond the empirical
    envelope" means: the corpus reaches about twice as far in each direction as
    the real data do.

    It was 0.3752 through Stage 2a-2, measured on the same extract UNSPLIT.
    Splitting the six categories that are not one product population narrows the
    arm's spread, because the widest categories were the heterogeneous ones: the
    maximum coefficient of variation falls from 14.34 to 11.20. This is the ONLY
    generation parameter whose cited measurement moved when the arm was split.

    Honest note on how much it is worth. Updating it improves the tuning
    objective from 0.2307 to 0.2274 at the 440-dataset pre-flight scale, a
    movement of 0.0033 against a seed-to-seed standard deviation of 0.0066
    measured in audits/stage2a2/p10_config_noise.py. That is HALF the noise: the
    change is adopted because it is the measurement the parameter cites, not
    because the improvement is distinguishable from a different seed.

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

    Both numbers moved together in Stage 2a-3 when the categories were resolved
    into specifiable products, and both track the same measurement. The arm's
    log10 centre moved from -0.1387 to -0.2207, so the centre here moves by the
    same 0.082 and KEEPS its measured offset of 0.350 above the arm; the arm's
    log10 standard deviation moved from 0.3800 to 0.3919.

    Honest note on the gain, as for every retune in this stage. At the
    440-dataset pre-flight scale the objective goes 0.2125 to 0.2118, and moving
    the standard deviation WITHOUT the centre makes it worse, 0.2186. The three
    candidates span 0.0068, about one seed-to-seed standard deviation, so the
    pair is adopted because it is what the parameters cite, not because the
    improvement is measurable.

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

    min_mode_sd_frac: float = 0.15
    """Narrowest component standard deviation, as a fraction of the parent's
    own standard deviation. A parent below this is redrawn.

    This is a floor on how tight a mode may be RELATIVE TO THE DATASET IT SITS
    IN, which is what makes a mode look like a spike in a density plot. An
    absolute floor would be meaningless, since every dataset is divided by its
    own mean and the spreads differ by orders of magnitude.

    It exists because the author looked at the generation figure and said the
    modes were unreasonably tight, twice. Two things were producing that, and
    they are different:

      - components with an UNBOUNDED density, beta with a < 1 or b < 1 and
        beta-prime with a < 1, which are genuine singularities rather than
        narrow modes. Those are refused outright in
        components.has_bounded_density, not by this parameter.
      - components that are legitimately narrow but placed far apart, which is
        what this parameter catches.

    Measured on corpus_2026-09-12g_draft1k, which had the singularities fixed
    but no width floor, the narrowest mode as a fraction of the dataset's own
    spread ran p01 0.040, p05 0.085, p25 0.210, median 0.475. It is entirely a
    low-overlap effect, because the overlap solve moves the component LOCATIONS
    and leaves their widths alone, so a low overlap target spreads fixed-width
    components across a wider dataset:

        achieved overlap   median ratio   share under 0.10
        under 0.01             0.143           29.3%
        0.01 to 0.03           0.218           15.1%
        0.03 to 0.1            0.272            8.2%
        0.1 to 0.3             0.442            0.6%
        over 0.3               0.820            0.0%

    Rejecting on the ratio is preferred to raising overlap_log10_lo because it
    removes only the offending cases: a low-overlap parent whose components
    happen to be wide is fine and is kept.

    There is NO reliable empirical target for this. The comparable empirical
    quantity has to come from a fitted mixture, and that estimate is
    unusable here for two compounding reasons: gmm_em_1d floors every variance
    at reg = 1e-6, and 39.7 percent of the empirical datasets sit exactly on
    that floor because real EPD data contains piles of identical declarations,
    which a Gaussian mixture answers with a degenerate component. So 0.15 is a
    judgment, not a measurement, and it is recorded as one. Stage 2h should
    sweep it."""

    # ---- truncation --------------------------------------------------------
    overlap_statistic: str = 'min_adjacent'
    """Which overlap statistic the spread solve holds to the target.

    'average' is the Maitra-Melnykov average pairwise overlap and is what Stage
    2a and Stage 2a-2 used. 'min_adjacent' is the smallest overlap between
    NEIGHBOURING components.

    The average stops constraining what a reader of a density plot sees once
    there are more than two components: a couple of heavily overlapping pairs
    carry the average while another pair sits at zero, so a mixture can report
    an average overlap of 0.09 and still show two sharp peaks with empty space
    between them. Measured on corpus_2026-09-12c, which targeted the average:

        k    median smallest pairwise overlap    median average
        2                 0.00094                    0.00094
        3                 0.00000                    0.02793
        4                 0.00000                    0.06046
        5                 0.00000                    0.09082

    This was recorded as a known defect of corpus_2026-09-12 in
    data/INPUTS.sha256 and then reintroduced by the Stage 2a-2 retune, which
    lowered the average-overlap target to match the mode-count distribution and
    so pushed the already-unconstrained pairs further apart. Controlling the
    minimum adjacent overlap targets the gaps directly.

    Neighbouring rather than all pairs: in one dimension the outermost pair of a
    five-component mixture is legitimately far apart, and the empirical arm
    shows the same thing, with a median smallest all-pairs overlap of 0.0000.
    The all-pairs minimum therefore cannot distinguish the two arms; the
    adjacent minimum can."""

    trunc_rule: str = 'log'
    """Whether the population truncation bounds are additive or multiplicative.

    'additive' is max(Q1 - mult * IQR, 0) and Q3 + mult * IQR, the rule the old
    generator used and the one the empirical arm used before Stage 2a-2.
    'log' is the same rule in log space, lo = Q1 / (Q3/Q1) ** mult and
    hi = Q3 * (Q3/Q1) ** mult.

    The empirical arm moved to the multiplicative rule in Stage 2a-2, because an
    ECC is strictly positive and right skewed and the additive lower bound is
    therefore negative in most categories and never binds. The synthetic arm was
    left on the additive rule in that stage, which is a defect: it left the two
    arms truncated by different rules on a dimension the study is about, in the
    same way the Dirichlet concentration differed between the arms before Stage
    2a. The notebook text claiming "the same rule applied to the empirical data"
    was false for the whole of Stage 2a-2.

    The practical difference is entirely at the low end. Under the additive rule
    the synthetic parent keeps its full left tail down to the positivity floor;
    under the log rule the left tail is cut at a fixed ratio below the first
    quartile. That is the region the lognormal offset in Stage 2b exists to
    handle, so the two stages interact.

    BOTH ARMS USE THE MULTIPLICATIVE RULE. They must: an ECC is strictly
    positive and right skewed in both arms, and trimming the two by different
    rules would put a difference between them on a dimension the study is about,
    in the same way the Dirichlet concentration differed before Stage 2a.

    An earlier attempt in Stage 2a-2 set this to 'log', appeared to fail, and
    was reverted with a docstring asserting that a multiplicative truncation is
    incompatible with using an additive shift to control the coefficient of
    variation, because q3/q1 tends to 1 as the shift grows and the bounds
    collapse onto the interquartile range. THAT WAS WRONG, in two ways, and the
    author pushed back on it three times before it was rechecked.

    First the algebra: as the shift grows, q1 / (q3/q1) ** mult converges to
    q1 - mult * IQR, so the multiplicative rule converges to the additive one
    from ABOVE and never collapses. Verified numerically over shifts from 0.6 to
    1000.

    Second the real cause: `MixtureParent.truncated_moments` integrated on a
    UNIFORM grid over [0, hi - lo], which silently fails when the bounds are
    wide relative to the body. Under the log rule hi reached about 5,000 while
    the mass sat near 1, so a 4,001-point grid put roughly one node on the whole
    distribution, the survival function read as zero everywhere and the variance
    clamped to exactly 0. Every parent then reported sd = 0, the
    coefficient-of-variation solve concluded no shift could reach any target,
    and 93 percent clipped. The grid now uses the components' own quantiles.
    That bug was latent under the additive rule, whose upper bound sits a few
    interquartile ranges from the body, and it would have bitten any Stage 2h
    sweep that raised trunc_iqr_mult.

    With it fixed the multiplicative rule is BETTER than the additive one on
    this generator: 41.7 percent of coefficient-of-variation targets met against
    37.5, and a median truncated mass of 0.097 against 0.154."""

    min_q1_over_iqr: float = 0.5
    """Positivity floor for the log truncation rule, as a multiple of the
    interquartile range: the shift must leave Q1 >= this times IQR.

    It replaces `max_low_tail_truncated` when `trunc_rule` is 'log'. Under the
    multiplicative rule the lower bound is positive whenever Q1 is, so a
    separate tail-mass budget is unnecessary and this single condition does the
    whole job.

    It is also what sets how wide the multiplicative bounds can get, because it
    caps the quartile ratio: at the floor, q3/q1 = 1 + 1/min_q1_over_iqr. At
    0.05 that is 21, which is the 95th percentile of the empirical ratio and
    sends the upper bound to about 9,000 times Q3. At 0.5 it is 3, which sits
    at the empirical median of 2.48. The empirical distribution of q3/q1 is
    p05 1.33, p25 1.69, median 2.48, p75 3.80, p95 20.3, so 0.5 places the
    tightest synthetic parents where most real datasets are.

    Measured effect on the generator, at trunc_iqr_mult = 3:

        min_q1_over_iqr   CV targets met   median truncated mass   median hi
             0.05              45.8%              0.209              1270
             0.5               41.7%              0.097                16
             1.0               35.0%              0.064              8.06
             2.0               18.3%              0.038              6.17

    0.5 keeps more of the parent than the additive rule did (0.154) while
    meeting more of its targets (37.5 percent)."""

    trunc_iqr_mult: float = 3.0
    """Bounds are max(Q1 - mult * IQR, 0) and Q3 + mult * IQR of the POPULATION
    mixture. Same rule as the old generator; what changed is that the quantiles
    are the parent's rather than one realized sample's."""

    # ---- reproducibility ---------------------------------------------------
    seed: int = 42
    max_parent_retries: int = 20
    """How many times a parent may be redrawn when it is rejected for having a
    mode narrower than `min_mode_sd_frac`. A rejection is a redraw of the whole
    parent, not a nudge of the one that failed, so the accepted parents are a
    clean conditional sample rather than a distorted one."""
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

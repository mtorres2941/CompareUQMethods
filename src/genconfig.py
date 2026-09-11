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
    overlap_log10_lo: float = -4.0
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
    comp_skew_lo: float = -3.5
    comp_skew_hi: float = 3.5
    """Component skewness target, drawn uniformly. Covers the empirical dataset
    skewness range of -5.73 to 4.09 with margin once the mixture and the
    small-sample noise are applied on top; a MIXTURE of components can be more
    skewed than any of its components."""

    comp_exkurt_lo: float = -1.2
    comp_exkurt_hi: float = 20.0
    """Component excess kurtosis target, drawn uniformly and then clipped up to
    the feasible boundary skewness ** 2 - 2 plus a margin. -1.2 is the uniform
    distribution, the platykurtic limit of any unimodal shape."""

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

    # ---- truncation --------------------------------------------------------
    trunc_iqr_mult: float = 3.0
    """Bounds are max(Q1 - mult * IQR, 0) and Q3 + mult * IQR of the POPULATION
    mixture. Same rule as the old generator; what changed is that the quantiles
    are the parent's rather than one realized sample's."""

    # ---- reproducibility ---------------------------------------------------
    seed: int = 20260911
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

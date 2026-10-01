"""Models built from JUDGMENT rather than from data. Stage 2h.

Every method this study compares is data driven: it turns a set of
environmental product declarations into a distribution. The probabilistic LCA
methods in general use are not. The pedigree matrix, which is what ecoinvent
applies and what most practitioners meet, is formulaic expert judgment applied
where data is absent; a uniform or a triangular over a plausible range is what
someone reaches for when they have two or three numbers and no dataset.

    THESE ARE NOT COMPETITORS IN THE MAIN COMPARISON, AND PUTTING THEM THERE
    WOULD BE A STRAW MAN THAT FLATTERS THIS PAPER'S OWN METHOD.

A uniform fitted to n declarations is exactly the smallest and the largest of
them and nothing else -- its maximum likelihood fit discards every value in
between -- so it would lose a goodness-of-fit comparison by a distance, and
that distance would say nothing. Decision 151 is why they live here.

    WHAT MAKES THE COMPARISON POSSIBLE ANYWAY.

Stage 2e's yardstick -- the error against the TRUE distribution the data was
drawn from -- does not care how a model was built. So a judgment-driven model
can be placed on the same axis as a data-driven one without any claim that the
two approaches are comparable in kind, which is the claim that would not
survive review. Decision 124.

    TWO DIMENSIONS, NOT ONE, AND THE SECOND IS THE ONE THAT MATTERS.

A pedigree model is a SPREAD applied around a POINT ESTIMATE the practitioner
already holds, and there is no reason that point sits where the category's
market-weighted mean sits. A practitioner working without data centers on a
single declaration they happened to obtain, a generic database value, or an
industry average, and each is offset from the truth by an amount nobody can
see. Sweeping only the spread would miss that and would flatter the pedigree
approach, because this project has twice established that BIAS adds across a
building's materials while random error cancels (decisions 122b and 120): a
judgment model with a well-chosen spread and a displaced center fails the way
the normal fit does.

    THE PRIMARY LOCATION MODEL HAS NO FREE PARAMETER.

A practitioner with no dataset has ONE declaration. `center_one_declaration`
draws exactly that, so the offset distribution falls out of the data rather
than being chosen. `center_offset` sweeps a deliberate displacement beside it
so the sensitivity is mapped rather than only sampled.

    THE SPREAD IS SWEPT RELATIVE TO THE DATA'S OWN, AND THAT IS DELIBERATE.

Decision 124 asks for a sweep over the range the pedigree matrix produces
rather than a choice of pedigree scores, because the scores describe a
data-collection context a generated dataset does not have and selecting them
would be inventing a provenance. So the primary axis here is `gsd_ratio`, the
model's geometric standard deviation over the DATA'S own, which answers
decision 124's deliverable sentence directly: how far from the data's own
spread can a judgment-driven model sit before it changes the answer.

    THE MATRIX IS NOW SOURCED, AND MOST OF THIS SWEEP IS UNREACHABLE.

The factor table was not in this repository when the sweep was designed, so the
range was chosen to bracket generously. The author has since supplied Muller,
Lesage, Ciroth, Mutel, Weidema and Samson (2016), Int J Life Cycle Assess
21:1185-1196, and `audits/pedigree_range.py` enumerates all 3,125 score
combinations from its Table 3 and Table 4. The matrix spans a geometric
standard deviation of **1.025 at the best scores to 1.587 at the worst** -- a
factor of 1.55 end to end -- against a median real ECC category at **1.871**.

**So a pedigree model is systematically NARROWER than the data it stands for,
and 61.9 percent of real categories are wider than its worst score can
reach.** Taken the way this sweep takes it -- on the EXCESS over 1, since
`gsd = 1 + (gsd_data - 1) * ratio` and a GSD of 1 is no spread at all -- the
reachable ratio on the median category is **0.028 to 0.674**, so of the six
ratios swept here only 0.5 is attainable and 0.75, 1.0, 1.5, 2.0 and 3.0 are
not. Across the whole arm, ratio 0.5 is reachable on 62.6 percent of
categories, 1.0 on 38.1 and 3.0 on 5.4. The wide end of this sweep is a
sensitivity and not a pedigree model.

**The two ratio definitions are not interchangeable and mixing them misplaces
the range badly**: the STRAIGHT ratio, model GSD over data GSD, puts the same
reachable band at 0.55 to 0.85, which would wrongly report 0.75 as attainable
and 0.5 as unreachably tight. The excess form is the one the sweep uses and the
one these numbers are in.

That is a property of what the matrix is FOR rather than a defect in it: it
quantifies uncertainty about one datum for one process, not the spread of
products within a material category, which is what an ECC dataset measures.
The manuscript should say so rather than present the two as rival estimates of
one quantity.

**EVERY FACTOR IN THAT TABLE IS A CONTRIBUTOR TO THE SQUARE of the geometric
standard deviation**, which is the detail that decides the arithmetic:

    sigma_95 = sqrt(sum over indicators of [ln(UF_i)] ** 2, plus the basic)
    GSD      = exp(sigma_95 / 2)

Quoting the combined factor AS a geometric standard deviation would double the
spread. A first reading of that paper here also mistook its Table 5 values of
1.26 and 1.69 for a total range; they are the posterior factors for ONE
indicator, the further technological correlation, at scores 2 and 3 for the
manufacturing sector.
"""
import numpy as np
from scipy.stats import lognorm, triang, uniform as uniform_dist

import families as FAM

#: The model's geometric standard deviation as a MULTIPLE of the data's own.
#: 1.0 is a judgment model that happens to get the spread exactly right.
#:
#: ONLY 0.75 IS REACHABLE BY ANY PEDIGREE SCORE ON THE MEDIAN CATEGORY; see
#: PEDIGREE_GSD below and the module docstring. The values at and above 1.0 are
#: kept because a flat sensitivity is only informative if it is measured over a
#: range wide enough to have shown a slope, and they must be labeled as a
#: sensitivity rather than as pedigree models.
GSD_RATIOS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

#: Absolute geometric standard deviations. The first three bracket what the
#: pedigree matrix can produce and the last three are above its worst score.
GSD_ABSOLUTE = (1.05, 1.2, 1.5, 2.0, 2.5, 3.0)

#: What the pedigree matrix spans, end to end, from all 3,125 score
#: combinations of ecoinvent's published factors: best (1,1,1,1,1) to worst
#: (5,5,5,5,5), with the median combination between them. Computed by
#: `audits/pedigree_range.py` from Muller et al. (2016); not quoted from
#: memory, which decision 49's amendment is a standing warning against.
PEDIGREE_GSD = dict(best=1.0247, median=1.2416, worst=1.5873)

#: Deliberate displacement of the model's center, as a fraction of the true
#: mean. Swept beside the no-free-parameter location model so the sensitivity
#: is mapped rather than only sampled.
CENTER_OFFSETS = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)


def data_gsd(x, w=None):
    """The data's own geometric standard deviation, weighted if asked.

    `exp(sd of log x)`, which is the quantity a pedigree geometric standard
    deviation is directly comparable with, because a pedigree model IS a
    lognormal parameterization.
    """
    x = np.asarray(x, dtype=float)
    keep = x > 0
    if keep.sum() < 2:
        return np.nan
    lx = np.log(x[keep])
    if w is None:
        return float(np.exp(np.std(lx, ddof=0)))
    ww = np.asarray(w, dtype=float)[keep]
    ww = ww / ww.sum()
    mu = float(ww @ lx)
    return float(np.exp(np.sqrt(max(0.0, float(ww @ (lx - mu) ** 2)))))


def center_one_declaration(x, rng):
    """THE PRIMARY LOCATION MODEL: one declaration, drawn at random.

    A practitioner without a dataset holds one environmental product
    declaration. This is that, and it has no free parameter -- the offset
    distribution is whatever the category's own spread makes it, which is the
    point: nobody can see how far their one declaration sits from the mean.
    """
    x = np.asarray(x, dtype=float)
    return float(rng.choice(x[x > 0]))


def center_offset(x, frac, w=None):
    """A DELIBERATE displacement, as a fraction of the mean.

    `frac = 0` centers on the category's own mean, which is the best a
    judgment model could possibly do and is the control the sweep needs.
    """
    x = np.asarray(x, dtype=float)
    if w is None:
        mean = float(np.mean(x))
    else:
        ww = np.asarray(w, dtype=float)
        mean = float((ww / ww.sum()) @ x)
    return float(mean * (1.0 + float(frac)))


def pedigree_model(center, gsd):
    """A lognormal specified by a center and a geometric standard deviation.

    THIS IS WHAT A PEDIGREE MATRIX PRODUCES. The matrix turns six quality
    scores into a geometric standard deviation and applies it around a point
    estimate; a geometric standard deviation IS a lognormal parameterization,
    which is why the lognormal is ecoinvent's default.

    `center` is the model's geometric mean, so `exp(mu) = center`. It is NOT
    the mean of the distribution: a lognormal's mean is `center * gsd ** (ln
    gsd / 2)`, which is above the center and is part of what the sweep
    measures, because a practitioner reading a point estimate off a declaration
    does not usually intend it as a median.
    """
    gsd = float(gsd)
    if not gsd > 1.0:
        gsd = 1.0 + 1e-9
    return FAM.Truncated(lognorm(s=float(np.log(gsd)), loc=0.0,
                                 scale=float(center)),
                         label='pedigree',
                         params=dict(center=float(center), gsd=gsd))


def uniform_model(lo, hi):
    """A uniform over a plausible range, truncated to (0, inf).

    In the judgment arm because it is SPECIFIED from two numbers rather than
    fitted: its maximum likelihood fit to n declarations is exactly their
    smallest and largest, discarding everything between. Decision 151.
    """
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    if not hi > lo:
        hi = lo + 1e-12
    return FAM.Truncated(uniform_dist(loc=lo, scale=hi - lo), label='uniform',
                         params=dict(lo=lo, hi=hi))


def triangular_model(lo, mode, hi):
    """A triangular over a range with a most-likely value. Judgment arm."""
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    if not hi > lo:
        hi = lo + 1e-12
    mode = float(np.clip(mode, lo, hi))
    return FAM.Truncated(triang(c=(mode - lo) / (hi - lo), loc=lo,
                                scale=hi - lo), label='triangular',
                         params=dict(lo=lo, mode=mode, hi=hi))


def plausible_range(center, gsd, coverage=0.95):
    """The low and high a practitioner would quote, from a center and a spread.

    A uniform and a triangular are specified from BOUNDS, so they need the same
    two judgment inputs the pedigree model needs, turned into a range. Taking
    the central `coverage` interval of the lognormal those inputs describe
    keeps all three judgment models on ONE pair of inputs, so the comparison
    between them is about the SHAPE a practitioner assumes and not about
    feeding them different information.
    """
    q = (1.0 - float(coverage)) / 2.0
    d = lognorm(s=float(np.log(max(gsd, 1.0 + 1e-9))), loc=0.0,
                scale=float(center))
    return float(d.ppf(q)), float(d.ppf(1.0 - q))


#: How a deliberate center offset is applied ACROSS the materials of a
#: building, which turns out to matter more than its size.
#:
#: `common`       every material is displaced by the SAME fraction. Measured:
#:                this cancels EXACTLY in a design comparison, because both
#:                options' totals scale by the same factor, so it costs nothing
#:                at the decision level while costing a great deal at the fit
#:                level.
#: `independent`  each material is displaced by its own draw of that
#:                magnitude, random in sign. This is what a practitioner
#:                actually suffers, because they obtain a different arbitrary
#:                declaration for each material, and it does NOT cancel.
OFFSET_MODES = ('common', 'independent')


def judgment_models(x, rng, gsd, center=None, offset_frac=None, w=None,
                    coverage=0.95, offset_mode='common'):
    """The three judgment models on one pair of inputs.

    `center` may be given directly; otherwise it is built from `offset_frac`,
    and from one random declaration when that is None.

    **`offset_mode` is not a detail.** A displacement applied identically to
    every material of a building cancels in any comparison of two designs, so
    sweeping only that would report a null that is an artifact of the sweep.
    `independent` draws the sign per material, which is what having obtained a
    different arbitrary declaration for each material actually produces.
    """
    if center is None:
        if offset_frac is None:
            center = center_one_declaration(x, rng)
        elif offset_mode == 'independent':
            sign = 1.0 if rng.random() < 0.5 else -1.0
            center = center_offset(x, sign * abs(offset_frac), w=w)
        else:
            center = center_offset(x, offset_frac, w=w)
    lo, hi = plausible_range(center, gsd, coverage=coverage)
    return {
        'pedigree': pedigree_model(center, gsd),
        'uniform': uniform_model(lo, hi),
        'triangular': triangular_model(lo, center, hi),
    }

"""Synthetic ECC dataset generation.

All randomness enters through an explicitly passed numpy Generator. No
function in this module touches global numpy random state, and none creates a
Generator of its own, so a caller can always reproduce a run from a single
recorded seed.
"""

import numpy as np
import scipy.stats as stats
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d


########################################################################
def random_samples(X, Y, rng, n=1):
    """Inverse-transform sample n values from the density (X, Y)."""
    pdf = Y / trapezoid(Y, X)
    cdf = np.cumsum(pdf) * np.diff(X, prepend=X[0])
    inverse_cdf = interp1d(cdf, X, kind='linear', fill_value="extrapolate")
    return inverse_cdf(rng.uniform(0, 1, n))


########################################################################
def generate_random_numbers(t, loc, scale, cnt, rng):
    """Draw cnt values from one mixture component of type t.

    The shape parameter of the skew-normal, Student-t and lognormal components
    is drawn from rng, so it varies between components and between datasets.

    Before Stage 1 this function took a `seed` argument that defaulted to 0 and
    built a fresh Generator from it on every call. That made the shape
    parameters constants across the entire study (skew-normal a = 3.458732,
    Student-t df = 3.366328, lognormal s = 1.136962) and made the draws
    themselves a deterministic function of (t, cnt): two Gaussian components of
    equal length were exact affine images of one another. Passing a shared
    Generator is the fix. Whether the previously generated datasets need
    regenerating as a result is a Stage 2 decision.
    """
    if t == "gauss":
        samples = rng.normal(loc=loc, scale=scale, size=cnt)

    elif t == "skewnorm":
        # skew parameter
        a = float(rng.uniform(-1, 6))
        samples = stats.skewnorm.rvs(a=a, loc=loc, scale=scale, random_state=rng, size=cnt)

    elif t == "studentt":
        # df of 1.0 is Cauchy dist, df of inf is normal dist
        df = float(rng.uniform(0.5, 5.0))
        samples = stats.t.rvs(df, loc=loc, scale=scale, random_state=rng, size=cnt)

    elif t == 'lognorm':
        s = float(rng.uniform(0.5, 1.5))
        # random_state was omitted here before Stage 1, so this branch alone
        # drew from global numpy state.
        samples = stats.lognorm.rvs(s=s, loc=loc, scale=scale, random_state=rng, size=cnt)

    else:
        raise ValueError('Check that all distribution types are included in random_irregular_dataset function')

    return samples


########################################################################
def random_irregular_dataset(n: int, rng) -> tuple:
    """Generate one synthetic ECC dataset of exactly n values, with weights.

    Procedure:
      - 1 to 5 mixture components, each Gaussian, skew-normal, Student-t or
        lognormal, with random locations, scales and Dirichlet mixture weights
      - values outside [max(0, Q1 - 3*IQR), Q3 + 3*IQR] are rejected and
        redrawn until exactly n values remain in range
      - every value is raised to a power drawn from U(0.9, 4.0), which inflates
        the spread to resemble empirical ECC data
      - with probability 0.25 the dataset is reflected, producing left skew
      - per-point weights are drawn from a flat Dirichlet
      - the data are offset by 1 and divided by their unweighted mean, so the
        unweighted mean is exactly 1.0

    Args:
        n: number of values to generate
        rng: numpy Generator, required

    Returns:
        (data, weights), each a 1-D array of length n
    """
    n = int(n)

    # choose number of modes
    k = int(rng.integers(1, 6))  # 1 to 5 modes

    # component locations spread across a range
    locs = rng.uniform(5, 20, size=k)

    # component base scales
    scales = rng.uniform(0.2, 1.5, size=k)

    # mixture weights
    cpv = np.ones(k) * 10
    weights = rng.dirichlet(cpv)

    # choose component types
    types = rng.choice(["gauss", "skewnorm", "studentt", "lognorm"], size=k,
                       p=[0.40, 0.25, 0.25, 0.10])

    # sample counts per component
    counts = rng.multinomial(n, weights)

    pieces = []
    for i in range(k):
        cnt = int(counts[i])
        if cnt <= 0:
            continue
        samples = generate_random_numbers(types[i], float(locs[i]), float(scales[i]), cnt, rng)
        pieces.append(samples)

    data = np.concatenate(pieces)

    # get rid of extreme outliers and ensure data is positive
    q1, q3 = np.quantile(data, [0.25, 0.75])
    iqr = q3 - q1
    lo = np.max([q1 - 3 * iqr, 0])
    hi = q3 + 3 * iqr
    while np.min(data) <= lo or np.max(data) >= hi or len(data) != n:
        data = data[data > lo]
        data = data[data < hi]
        if len(data) > n:
            data = data[:n]
        i = rng.choice(range(k), p=weights, size=1)[0]
        cnt = n - len(data)
        add = generate_random_numbers(types[i], float(locs[i]), float(scales[i]), cnt, rng)
        data = np.concatenate([data, add])

    # increase standard deviation of data to align with empirical ECC data
    exp = rng.uniform(0.9, 4.0)
    data = data ** exp
    lo = lo ** exp
    hi = hi ** exp

    # flip data 25% of the time to get more left skewed data
    if rng.uniform(0, 1) < 0.25:
        data = np.max(data) - data + np.min(data)

    if data.size != n:
        raise ValueError(f'random_irregular_dataset is not outputting data of the right size. Should be {n}, but instead is {data.size}')
    if np.max(data) >= hi:
        raise ValueError(f'there is too large of a value in the dataset {hi}<{np.min(data)}')
    if np.min(data) <= lo:
        raise ValueError(f'there is too small of a value in the dataset {lo}>{np.max(data)}')

    # randomly generate weights
    weights = rng.dirichlet(np.ones_like(data))

    # normalize data. Note this is the UNWEIGHTED mean: see Stage 1 amendment
    # A3 and entry 2 of reports/MANUSCRIPT_discrepancies.md. The threshold this
    # study builds must be computable by a practitioner who has a set of EPDs
    # but does not know the market shares, so the unweighted mean is the
    # correct normalizer and the manuscript text is what needs changing.
    data = data + 1  # to provide a buffer
    data = data / np.mean(data)

    return data, weights


########################################################################
def random_logcount(rng, lo=4, hi=1_000, n=1):
    """Sample n integers between lo and hi, uniform on a log scale.

    INPUT
        rng     numpy Generator, required
        lo      Lower bound
        hi      Upper bound
        n       Number of values to be output

    OUTPUT
        An array of n rounded values
    """
    # np.log(x)/np.log(10) is kept rather than np.log10(x): the two differ in
    # the last bit, and there is no reason to perturb the generation stream.
    return np.round(10 ** rng.uniform(np.log(lo) / np.log(10), np.log(hi) / np.log(10), n), 0)


########################################################################
def clean_empirical_low_end(data, mult=3.0):
    """Remove near-zero empirical ECC values with a MULTIPLICATIVE bound.

    The extraction trims at Q1 - 3 * IQR and Q3 + 3 * IQR. The low bound is
    negative in 128 of the 138 empirical datasets, so it never binds: high
    outliers are removed and near-zero values are not. `ReadyMix` retains a
    value at 3.1e-17 of its mean, which is a data error rather than a product,
    and 39 of the 138 datasets hold a value below 1 percent of their mean.

    In log space the same rule is a ratio rather than a difference and does
    bind. A value is kept when

        log(x) > Q1(log x) - mult * IQR(log x)

    Decision 12 in CLAUDE.md records the multiplicative direction as the
    author's, with the specific filter left to this stage.

    Only the LOW end is treated here, deliberately. The high end was already
    trimmed additively when `dct_realeccs_trimmed.json` was written, and the
    directory that extraction read no longer exists, so the empirical data
    cannot be re-cleaned from source on this machine. Applying a log-space high
    bound on top of the additive one would trim the same tail twice. A
    symmetric re-clean needs a fresh EC3 extraction; see
    reports/MANUSCRIPT_discrepancies.md.
    """
    data = np.asarray(data, float)
    pos = data[data > 0]
    if len(pos) < 4:
        return data
    L = np.log(pos)
    l1, l3 = np.quantile(L, [0.25, 0.75])
    li = l3 - l1
    if li <= 0:
        return data
    return data[(data > 0) & (np.log(np.where(data > 0, data, 1e-300)) > l1 - mult * li)]

from typing import Optional
import numpy as np
import scipy.stats as stats
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

# last edited 2026-03-11

# import (random_samples, generate_random_numbers, random_irregular_dataset, random_logcount)
########################################################################
def random_samples(X, Y, n=1):
    pdf = Y/trapezoid(Y, X)
    cdf = np.cumsum(pdf) * np.diff(X, prepend=X[0])
    inverse_cdf = interp1d(cdf, X, kind='linear', fill_value="extrapolate")
    return inverse_cdf(np.random.uniform(0,1,n))

########################################################################
def generate_random_numbers(t, loc, scale, cnt, seed=0):
    rng = np.random.default_rng(seed)
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
        s = float(rng.uniform(0.5,1.5))
        samples = stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=cnt)

    # elif t == "beta":
    #     # small localized bump mapped to [loc - w, loc + w]
    #     a = float(rng.uniform(0.5, 5.0))
    #     b = float(rng.uniform(0.5, 5.0))
    #     width = max(0.25, scale)  # width of the beta bump
    #     u = stats.beta.rvs(a, b, random_state=rng, size=cnt)
    #     left = loc - width
    #     right = loc + width
    #     samples = left + (right - left) * u

    # elif t == "uniform":
    #     samples = rng.uniform(loc - scale, loc + scale, size=cnt)

        # samples = rng.normal(loc=loc, scale=scale, size=cnt)

    else:
        raise ValueError('Check that all distribution types are included in random_irregular_dataset function')

    
    return samples

########################################################################
def random_irregular_dataset(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a 1-D irregular dataset with `n` samples.

    Behavior (randomized each call):
      - Random number of modes between 1 and 5
      - Each mode is one of: Gaussian, skew-normal, Student-t, Beta (local bump), Uniform
      - Random mixture weights (Dirichlet)
      - Random component locations and scales
      - Small chance of contamination (Pareto-style heavy outliers)
      - Returns a 1-D numpy array of length n

    Args:
        n: int, number of samples to generate
        seed: optional int for reproducibility

    Returns:
        np.ndarray of shape (n,)
    """
    rng = np.random.default_rng(seed)
    n = int(n)

    # choose number of modes
    k = int(rng.integers(1, 6))  # 1-6 modes

    # component locations spread across a range
    locs = rng.uniform(5, 20, size=k)

    # component base scales
    scales = rng.uniform(0.2, 1.5, size=k)

    # mixture weights. using inverse of location for concentration parameter vector
    # so lower values are more likely to have higher weights. Normalized so min(cpv)=1
    # cpv = 1/locs
    # cpv = cpv/np.min(cpv)
    # cpv = np.array([5*ele**0.3 for ele in cpv]) # adjust so values are closer together to increase multimodality
    cpv = np.ones(k)*10
    weights = rng.dirichlet(cpv)

    # choose component types
    # types = rng.choice(["gauss", "skewnorm", "studentt", "beta", "uniform"], size=k,
    #                    p=[0.35, 0.25, 0.2, 0.1, 0.1])
    types = rng.choice(["gauss", "skewnorm", "studentt", "lognorm"], size=k,
                       p=[0.40, 0.25, 0.25, 0.10,])
    
    # sample counts per component
    counts = rng.multinomial(n, weights)

    pieces = []
    for i in range(k):
        cnt = int(counts[i])
        if cnt <= 0:
            continue

        t = types[i]
        loc = float(locs[i])
        scale = float(scales[i])
        samples = generate_random_numbers(t, loc, scale, cnt)
        pieces.append(samples)

    data = np.concatenate(pieces)
    
    # get rid of extreme outliers and ensure data is positive
    q1, q3 = np.quantile(data, [0.25, 0.75])
    iqr = q3-q1
    lo = np.max([q1-3*iqr, 0])
    hi = q3 + 3*iqr
    while np.min(data) <= lo or np.max(data) >= hi or len(data) != n:
        data = data[data>lo]
        data = data[data<hi]
        if len(data) > n:
            data = data[:n]
        i = rng.choice(range(k), p=weights, size=1)[0]
        t = types[i]
        loc = float(locs[i])
        scale = float(scales[i])
        cnt = n-len(data)
        add = generate_random_numbers(t, loc, scale, cnt)
        data = np.concatenate([data, add])
    
    # increase standard deviation of data to align with empirical ECC data
    exp = rng.uniform(0.9,4.0)
    data = np.array([ele**exp for ele in data])
    lo = lo**exp
    hi = hi**exp
    
    # flip data 25% of the time to get more left skewed data
    if rng.uniform(0,1) < 0.25:
        data = np.max(data)-data + np.min(data)
    
    # shuffle and trim/pad to exactly n
    # rng.shuffle(data)
    # if data.size > n:
    #     data = data[:n]
    # while len(data) < n:
    #     i = rng.choice(range(k), p=weights, size=1)[0]
    #     t = types[i]
    #     loc = float(locs[i])
    #     scale = float(scales[i])
    #     cnt = n-len(data)
    #     add = generate_random_numbers(t, loc, scale, cnt)
    #     data = np.concatenate([data, add])
    #     data = data[data>0]
    if data.size != n:
        raise ValueError(f'random_irregular_dataset is not outputting data of the right size. Should be {n}, but instead is {data.size}')
    if np.max(data) >= hi:
        raise ValueError(f'there is too large of a value in the dataset {hi}<{np.min(data)}')
    if np.min(data) <= lo:
        raise ValueError(f'there is too small of a value in the dataset {lo}>{np.max(data)}')

    # randomly generate weights
    weights = np.random.dirichlet(np.ones_like(data))

    # normalize data
    data = data + 1 # to provide a buffer
    data = data / np.mean(data)
    
    return data, weights


########################################################################
def random_logcount(lo=4, hi=1_000, n=1):
    """
    Samples a random integer between two values, distributed normally on a log scale
    
    INPUT
        lo      Lower bound
        hi      Upper bound
        n       Number of values to be output
        
    OUTPUT
        size    An integer
    """
    return np.round(10**np.random.uniform(np.log(lo)/np.log(10), np.log(hi)/np.log(10), n),0)
########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

########################################################################


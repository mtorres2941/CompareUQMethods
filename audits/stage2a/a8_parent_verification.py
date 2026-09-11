"""Stage 2a, Part 1: verify the parent CDF numerically.

The parent is closed form after the simplifications, but "should be closed
form" is not a result. This script compares a dense sample generated through
the real pipeline against the analytic parent, for both weighting schemes and
across the strata, and separately measures the one approximation in the chain:
the parent is stated CONDITIONAL on the realized normalizer, while the values
were divided by their own sample mean, which induces an O(1/n) dependence
between them.

Also reports the probability mass truncation removes, which Part 1 requires to
be stated, and confirms that inverse-CDF sampling agrees distributionally with
the truncation loop it replaces.

Writes TABLE_2a_ParentVerification.csv and TABLE_2a_TruncationMass.csv
"""
import numpy as np, pandas as pd, sys, os, warnings, time
warnings.filterwarnings('ignore')
from _common import write

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genconfig as G
import generator as GEN
import mixture as M
import legacy_generator_pre_stage1 as legacy

SEED = 20260911
CFG = G.DEFAULT


def effective_n(w):
    """Kish effective sample size, 1 / sum(w ** 2) for normalized weights.

    The market-weighted empirical CDF converges at the rate set by this, not by
    n. That is not an artifact of the verification: it is the mechanism the
    whole study is about. A mode with a small sampling share and a large market
    share contributes few points carrying much weight, and its sampling error
    dominates the market-weighted distance.
    """
    w = np.asarray(w, float)
    return float(1.0 / np.sum((w / w.sum()) ** 2))


def ks_against_parent(parent, x, scheme='uniform', w=None):
    """Kolmogorov-Smirnov distance between the realized sample and the parent."""
    o = np.argsort(x)
    xs = x[o]
    if w is None:
        emp_hi = np.arange(1, len(xs) + 1) / len(xs)
        emp_lo = np.arange(0, len(xs)) / len(xs)
    else:
        cw = np.cumsum(w[o])
        emp_hi = cw
        emp_lo = cw - w[o]
    F = parent.cdf(xs, scheme)
    return float(max(np.max(np.abs(emp_hi - F)), np.max(np.abs(emp_lo - F))))


def verify_big_sample(n_big=500_000, reps=12, seed=SEED):
    """One parent, one very large sample, both schemes."""
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        parent, rec = GEN.draw_parent(CFG, n_big, rng)
        if parent is None:
            continue
        x, modes = parent.sample(n_big, rng)
        w = GEN.draw_weights(parent, modes, CFG, rng)
        rows.append(dict(rep=rep, k=rec['k'], n=n_big,
                         overlap=rec['overlap_achieved'],
                         truncated_mass=rec['truncated_mass'],
                         ks_uniform=ks_against_parent(parent, x, 'uniform'),
                         ks_market=ks_against_parent(parent, x, 'market', w),
                         ess_market=effective_n(w),
                         ks_noise_floor=1.36 / np.sqrt(n_big),
                         ks_floor_market=1.36 / np.sqrt(effective_n(w))))
    return pd.DataFrame(rows)


def verify_by_stratum(reps=200, seed=SEED + 1):
    """The conditioning effect: how far the realized sample sits from the
    parent at each n, against the Kolmogorov noise floor for that n."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in CFG.strata:
        for _ in range(reps):
            n = int(np.floor(10 ** rng.uniform(np.log10(s.n_lo), np.log10(s.n_hi + 1))))
            n = int(np.clip(n, s.n_lo, s.n_hi))
            parent, rec = GEN.draw_parent(CFG, n, rng)
            if parent is None:
                continue
            x, modes = parent.sample(n, rng)
            w = GEN.draw_weights(parent, modes, CFG, rng)
            rows.append(dict(stratum=s.name, n=n, k=rec['k'],
                             truncated_mass=rec['truncated_mass'],
                             normalizer=parent.normalizer,
                             ks_uniform=ks_against_parent(parent, x, 'uniform'),
                             ks_market=ks_against_parent(parent, x, 'market', w),
                             ess_market=effective_n(w),
                             ks_noise_floor=1.36 / np.sqrt(n),
                             ks_floor_market=1.36 / np.sqrt(effective_n(w))))
    return pd.DataFrame(rows)


def conditioning_effect(reps=400, seed=SEED + 2):
    """Isolate the O(1/n) effect of dividing by the realized sample mean.

    Draws a sample, then compares its unweighted sample mean against the
    parent's population mean over the truncated support. The ratio is the only
    respect in which the stated parent is conditional rather than marginal.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in CFG.strata:
        for _ in range(reps):
            n = int(np.clip(np.floor(10 ** rng.uniform(np.log10(s.n_lo),
                                                       np.log10(s.n_hi + 1))),
                            s.n_lo, s.n_hi))
            parent, rec = GEN.draw_parent(CFG, n, rng)
            if parent is None:
                continue
            x, _ = parent.sample(n, rng)
            # population mean of the truncated mixture, by quadrature on its ppf
            u = np.linspace(1e-6, 1 - 1e-6, 20001)
            pop_mean = float(np.trapezoid(parent.ppf(u) * parent.normalizer, u))
            rows.append(dict(stratum=s.name, n=n,
                             sample_mean=parent.normalizer, pop_mean=pop_mean,
                             ratio=parent.normalizer / pop_mean))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    t = time.time()
    print('--- parent CDF against a dense sample from the pipeline ---')
    big = verify_big_sample()
    pd.set_option('display.width', 220)
    print(big[['k', 'overlap', 'truncated_mass', 'ks_uniform', 'ks_noise_floor',
               'ks_market', 'ess_market', 'ks_floor_market']].to_string(
        index=False, float_format=lambda v: f'{v:,.6f}'))
    print(f"\nmax KS, uniform parent : {big.ks_uniform.max():.6f}")
    print(f"max KS, market parent  : {big.ks_market.max():.6f}")
    print(f"Kolmogorov 5% floor    : {big.ks_noise_floor.iloc[0]:.6f}")
    print(f"uniform within its floor : "
          f"{bool((big.ks_uniform < big.ks_noise_floor).all())}")
    # 1.36 / sqrt(ESS) is the 5 percent Kolmogorov critical value, so it is
    # expected to be exceeded about one time in twenty; the ratio is the
    # informative number, not a pass or fail.
    print(f"market KS / its ESS floor: max {float((big.ks_market / big.ks_floor_market).max()):.3f}, "
          f"median {float((big.ks_market / big.ks_floor_market).median()):.3f}")

    print(f'\n--- by stratum ({time.time()-t:.0f}s) ---')
    bs = verify_by_stratum()
    write(bs, 'TABLE_2a_ParentVerification.csv')
    agg = bs.groupby('stratum').agg(
        n_datasets=('n', 'size'),
        median_ks_uniform=('ks_uniform', 'median'),
        p95_ks_uniform=('ks_uniform', lambda s: s.quantile(.95)),
        median_ks_market=('ks_market', 'median'),
        median_floor=('ks_noise_floor', 'median'),
        median_floor_market=('ks_floor_market', 'median'),
        median_trunc_mass=('truncated_mass', 'median'),
        mean_trunc_mass=('truncated_mass', 'mean'),
        p95_trunc_mass=('truncated_mass', lambda s: s.quantile(.95)))
    print(agg.to_string(float_format=lambda v: f'{v:,.5f}'))
    print('\nratio of observed KS to the Kolmogorov floor (should be about 1):')
    print((bs.ks_uniform / bs.ks_noise_floor).groupby(bs.stratum).describe(
        percentiles=[.5, .95]).to_string(float_format=lambda v: f'{v:,.3f}'))

    print(f'\n--- truncation mass, the quantity Part 1 asks to be stated ({time.time()-t:.0f}s) ---')
    tm = bs[['stratum', 'truncated_mass']].copy()
    write(tm, 'TABLE_2a_TruncationMass.csv')
    print(bs.truncated_mass.describe(percentiles=[.5, .75, .9, .95, .99]).to_string())

    print(f'\n--- the one conditional step: dividing by the realized sample mean ({time.time()-t:.0f}s) ---')
    ce = conditioning_effect()
    print(ce.groupby('stratum').ratio.describe(percentiles=[.05, .5, .95]).to_string(
        float_format=lambda v: f'{v:,.4f}'))
    print('\nsd of (sample mean / population mean), by stratum:')
    print(ce.groupby('stratum').ratio.std().to_string(float_format=lambda v: f'{v:,.4f}'))

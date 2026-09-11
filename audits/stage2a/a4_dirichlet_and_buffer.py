"""Stage 2a, Part 0 items 2 and 3.

Item 2  Dirichlet concentration mismatch. Empirical weights were drawn at
        alpha = 5, synthetic at alpha = 1, on the exact dimension the paper is
        about. Decided: alpha = 1 in both arms. Measures what moves.

Item 3  The undocumented +1 buffer. Measures what removing it does to the
        achievable coefficient of variation, the metric it compresses.

Writes TABLE_2a_DirichletConcentration.csv and TABLE_2a_BufferRemoval.csv
"""
import numpy as np, pandas as pd, sys, os
from _common import load_empirical, write

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datageneration as current
from customstats import empirical_metadata

SEED = 20260911
METRICS = ['coeffvar', 'entropy', 'skewness', 'kurtosis', 'mode_count_est',
           'weight_outliers', 'fit_norm_SW', 'fit_lognorm_SW', 'w_v_uw_wasserstein']


# ---------------------------------------------------------------- item 2 ----
def empirical_metrics_at_alpha(alpha, seed=SEED):
    d = load_empirical()
    rng = np.random.default_rng(seed)
    out = {}
    for mat in d:
        data = d[mat]['data']
        w = rng.dirichlet(np.ones_like(data) * alpha)
        out[mat] = empirical_metadata(data / np.mean(data), w)
    return pd.DataFrame(out).T


def dirichlet_summary():
    a5 = empirical_metrics_at_alpha(5)
    a1 = empirical_metrics_at_alpha(1)
    rows = []
    for m in METRICS:
        s5 = a5[m].astype(float).replace([np.inf, -np.inf], np.nan)
        s1 = a1[m].astype(float).replace([np.inf, -np.inf], np.nan)
        both = pd.concat([s5, s1], axis=1).dropna()
        rows.append(dict(metric=m,
                         alpha5_mean=float(s5.mean()), alpha1_mean=float(s1.mean()),
                         alpha5_sd=float(s5.std()), alpha1_sd=float(s1.std()),
                         mean_abs_change=float((both.iloc[:, 1] - both.iloc[:, 0]).abs().mean()),
                         max_abs_change=float((both.iloc[:, 1] - both.iloc[:, 0]).abs().max()),
                         change_in_sd_units=float((both.iloc[:, 1] - both.iloc[:, 0]).abs().mean()
                                                  / s5.std()) if s5.std() else np.nan))
    # the concentration itself
    rng = np.random.default_rng(SEED)
    for alpha in (1, 5):
        for n in (10, 100, 1000):
            w = rng.dirichlet(np.ones(n) * alpha, size=2000)
            rows.append(dict(metric=f'__max_weight_share_n{n}', alpha5_mean=np.nan,
                             alpha1_mean=np.nan, alpha5_sd=np.nan, alpha1_sd=np.nan,
                             mean_abs_change=np.nan, max_abs_change=np.nan,
                             change_in_sd_units=np.nan,
                             **{f'alpha{alpha}_max_share': float(w.max(1).mean())}))
    return pd.DataFrame(rows), a5, a1


# ---------------------------------------------------------------- item 3 ----
def _gen_raw(n, rng):
    """random_irregular_dataset stopped one step short of normalization, so the
    same realized values can be normalized with and without the +1 buffer."""
    n = int(n)
    k = int(rng.integers(1, 6))
    locs = rng.uniform(5, 20, size=k)
    scales = rng.uniform(0.2, 1.5, size=k)
    weights = rng.dirichlet(np.ones(k) * 10)
    types = rng.choice(['gauss', 'skewnorm', 'studentt', 'lognorm'], size=k,
                       p=[0.40, 0.25, 0.25, 0.10])
    counts = rng.multinomial(n, weights)
    pieces = [current.generate_random_numbers(types[i], float(locs[i]), float(scales[i]),
                                             int(counts[i]), rng)
              for i in range(k) if counts[i] > 0]
    data = np.concatenate(pieces)
    q1, q3 = np.quantile(data, [0.25, 0.75])
    iqr = q3 - q1
    lo = max(q1 - 3 * iqr, 0); hi = q3 + 3 * iqr
    while np.min(data) <= lo or np.max(data) >= hi or len(data) != n:
        data = data[(data > lo) & (data < hi)][:n]
        i = rng.choice(range(k), p=weights, size=1)[0]
        data = np.concatenate([data, current.generate_random_numbers(
            types[i], float(locs[i]), float(scales[i]), n - len(data), rng)])
    exp = rng.uniform(0.9, 4.0)
    data = data ** exp
    if rng.uniform(0, 1) < 0.25:
        data = np.max(data) - data + np.min(data)
    return data


def buffer_summary(ndatasets=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(ndatasets):
        n = int(current.random_logcount(rng, lo=4, hi=1000, n=1)[0])
        raw = _gen_raw(n, rng)
        w = rng.dirichlet(np.ones_like(raw))
        with_b = (raw + 1) / np.mean(raw + 1)
        without = raw / np.mean(raw)
        cv_w = float(np.std(with_b) / np.mean(with_b))
        cv_o = float(np.std(without) / np.mean(without))
        rows.append(dict(dataset=i, n=n, raw_mean=float(np.mean(raw)),
                         raw_min=float(np.min(raw)),
                         cv_with_buffer=cv_w, cv_without_buffer=cv_o,
                         cv_compression=cv_w / cv_o if cv_o else np.nan,
                         min_over_mean_with=float(np.min(with_b)),
                         min_over_mean_without=float(np.min(without))))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print('--- item 2: Dirichlet concentration, empirical arm ---')
    dsum, a5, a1 = dirichlet_summary()
    write(dsum, 'TABLE_2a_DirichletConcentration.csv')
    pd.set_option('display.width', 240)
    print(dsum[dsum.metric.isin(METRICS)].to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
    print()
    ms = dsum[dsum.metric.str.startswith('__')]
    print(ms[['metric', 'alpha1_max_share', 'alpha5_max_share']].groupby('metric').first()
          .to_string(float_format=lambda v: f'{v:,.4f}'))

    print('\n--- item 3: the +1 buffer ---')
    b = buffer_summary()
    write(b, 'TABLE_2a_BufferRemoval.csv')
    q = b[['cv_with_buffer', 'cv_without_buffer', 'cv_compression', 'raw_mean',
           'min_over_mean_with', 'min_over_mean_without']].describe(
        percentiles=[.01, .05, .25, .5, .75, .95, .99])
    print(q.to_string(float_format=lambda v: f'{v:,.5f}'))
    print(f"\ndatasets where the buffer compresses CV by more than 1 pct : "
          f"{(b.cv_compression < 0.99).mean()*100:.1f}%")
    print(f"datasets where it compresses CV by more than 10 pct        : "
          f"{(b.cv_compression < 0.90).mean()*100:.1f}%")
    print(f"max CV with buffer {b.cv_with_buffer.max():.4f}  "
          f"without {b.cv_without_buffer.max():.4f}")

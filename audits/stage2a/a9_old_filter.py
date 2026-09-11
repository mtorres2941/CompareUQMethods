"""Stage 2a, Part 2: what the old metric-outlier filter actually removed.

The notebook generated 15,000 datasets, removed every dataset that was a
marginal outlier on ANY of the 20 metrics using Q1 - 1.5 * IQR and
Q3 + 1.5 * IQR with the IQR widened to max(q3 - q1, std * 1.35), and kept the
first 10,000 survivors. The manuscript does not describe this step, so the
manuscript has to be able to describe what the old corpus was.

Two questions:
  - what fraction did the filter remove per metric?
  - did the removed datasets share a structural signature?

Writes TABLE_2a_OldFilterPerMetric.csv and TABLE_2a_OldFilterSignature.csv
"""
import numpy as np, pandas as pd, sys, os, warnings
warnings.filterwarnings('ignore')
from _common import load_shipped, metrics_frame, write

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    DATA, keep, outliers, trim = load_shipped()
    df = metrics_frame(DATA).astype(float)
    print(f'{len(df)} generated, {len(outliers)} flagged as metric outliers, '
          f'{len(trim)} further dropped to reach 10,000, {len(keep)} analysed')
    print(f'filter removed {len(outliers)/len(df)*100:.1f}% of what was generated')

    rows = []
    flagged_by = {}
    for metric in df.columns:
        srs = df[metric]
        finite = srs.replace([np.inf, -np.inf], np.nan).dropna()
        q1, q2, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
        std = np.std(finite)
        iqr = max(q3 - q1, std * 1.35)
        hi, lo = q3 + 1.5 * iqr, q1 - 1.5 * iqr
        flagged = set(srs.index[(srs > hi) | (srs < lo)])
        flagged_by[metric] = flagged
        rows.append(dict(metric=metric, n_flagged=len(flagged),
                         pct_flagged=100 * len(flagged) / len(df),
                         q1=q1, median=q2, q3=q3, raw_iqr=q3 - q1,
                         std_times_1p35=std * 1.35,
                         widened=bool(std * 1.35 > q3 - q1),
                         cutoff_lo=lo, cutoff_hi=hi,
                         data_min=float(finite.min()), data_max=float(finite.max()),
                         is_constant_by_construction=bool(finite.nunique() <= 2)))
    per = pd.DataFrame(rows).sort_values('n_flagged', ascending=False)
    write(per, 'TABLE_2a_OldFilterPerMetric.csv')
    pd.set_option('display.width', 250)
    print('\n--- per metric ---')
    print(per[['metric', 'n_flagged', 'pct_flagged', 'raw_iqr', 'std_times_1p35',
               'widened', 'cutoff_lo', 'cutoff_hi']].to_string(
        index=False, float_format=lambda v: f'{v:,.4f}'))

    # the mean_uw column is 1.0 by construction
    mu = df['mean_uw']
    print(f"\nmean_uw is 1.0 by construction: min {mu.min():.17g}, max {mu.max():.17g}")
    print(f"datasets it flagged anyway (floating-point noise): {len(flagged_by['mean_uw'])}")

    # structural signature of the removed set
    removed = sorted(outliers)
    kept = [k for k in df.index if k not in outliers]
    sig = []
    for metric in df.columns:
        a = df.loc[removed, metric].replace([np.inf, -np.inf], np.nan).dropna()
        b = df.loc[kept, metric].replace([np.inf, -np.inf], np.nan).dropna()
        pooled = np.sqrt((a.var() + b.var()) / 2) if len(a) > 1 and len(b) > 1 else np.nan
        sig.append(dict(metric=metric,
                        removed_mean=float(a.mean()), kept_mean=float(b.mean()),
                        removed_median=float(a.median()), kept_median=float(b.median()),
                        removed_p95=float(a.quantile(.95)), kept_p95=float(b.quantile(.95)),
                        std_diff=float((a.mean() - b.mean()) / pooled) if pooled else np.nan))
    sig = pd.DataFrame(sig).sort_values('std_diff', key=abs, ascending=False)
    write(sig, 'TABLE_2a_OldFilterSignature.csv')
    print('\n--- structural signature of the removed datasets ---')
    print('standardized mean difference, removed minus kept, in pooled sd units:')
    print(sig[['metric', 'removed_mean', 'kept_mean', 'removed_median', 'kept_median',
               'std_diff']].to_string(index=False, float_format=lambda v: f'{v:,.4f}'))

    print('\n--- the n cap ---')
    print(f"n over the analysed 10,000 : min {df.loc[kept,'n'].min():.0f}, "
          f"max {df.loc[kept,'n'].max():.0f}, median {df.loc[kept,'n'].median():.0f}")
    print(f"n over all 15,000 generated: min {df['n'].min():.0f}, "
          f"max {df['n'].max():.0f}")
    print(f"datasets discarded for being large (flagged on n): "
          f"{len(flagged_by['n'])}, all with n >= {df.loc[sorted(flagged_by['n']),'n'].min():.0f}")

    print('\n--- how many datasets each metric was solely responsible for ---')
    sole = {}
    for m, fl in flagged_by.items():
        others = set().union(*[v for k, v in flagged_by.items() if k != m])
        sole[m] = len(fl - others)
    ss = pd.Series(sole).sort_values(ascending=False)
    print(ss[ss > 0].to_string())

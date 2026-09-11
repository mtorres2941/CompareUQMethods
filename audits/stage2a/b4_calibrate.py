"""Stage 2a: calibrate the generator configuration against the empirical envelope.

The goal, stated by the author: the synthetic datasets should "look like" the
empirical ECC datasets as measured by the statistical metrics, with margin on
both sides so a generalizability claim is supportable. Margin means the
synthetic range should EXTEND BEYOND the empirical range at both ends, including
into regions the empirical data do not occupy at all, such as left skew.

What it must not mean is a synthetic distribution centred somewhere else
entirely. The first regenerated corpus had a median coefficient of variation of
0.049 against an empirical 0.600, which is not margin, it is a different
population.

This script samples a few hundred datasets per candidate configuration and
scores it, so a configuration can be chosen on measurement rather than on
inspection of a figure.

Writes TABLE_2a_Calibration.csv
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from _common import write, TABLES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genconfig as G
import generator as GEN
from customstats import empirical_metadata

METRICS = ['coeffvar', 'skewness', 'kurtosis', 'entropy', 'crit_bw_1',
           'weight_outliers', 'fit_norm_SW', 'fit_lognorm_SW', 'w_v_uw_wasserstein']
PER_STRATUM = 70


def sample_config(cfg, seed, per_stratum=PER_STRATUM):
    rng = np.random.default_rng(seed)
    rows = []
    for st in cfg.strata:
        for _ in range(per_stratum):
            n = int(np.clip(np.floor(10 ** rng.uniform(np.log10(st.n_lo),
                                                       np.log10(st.n_hi + 1))),
                            st.n_lo, st.n_hi))
            x, w, rec = GEN.generate_dataset(cfg, n, rng)
            if x is None or GEN.validity_failures(x, w, n):
                continue
            m = empirical_metadata(x, w)
            m['stratum'] = st.name
            rows.append(m)
    return pd.DataFrame(rows)


def score(syn, emp):
    """Per metric: does the synthetic distribution cover the empirical one, and
    is it centred in the same place?

    `median_ratio_log10` is the log10 ratio of medians, so 0 is perfect
    agreement and +/-1 is an order of magnitude off. `covers` asks whether the
    synthetic range brackets the empirical range with margin at both ends.
    """
    rows = []
    for m in METRICS:
        a = syn[m].replace([np.inf, -np.inf], np.nan).dropna()
        b = emp[m].replace([np.inf, -np.inf], np.nan).dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        sa, sb = a.median(), b.median()
        if m in ('coeffvar', 'crit_bw_1', 'entropy', 'w_v_uw_wasserstein'):
            ratio = np.log10(max(sa, 1e-9) / max(sb, 1e-9))
        else:
            spread = b.quantile(.75) - b.quantile(.25)
            ratio = (sa - sb) / spread if spread else np.nan
        rows.append(dict(
            metric=m,
            syn_p05=a.quantile(.05), syn_median=sa, syn_p95=a.quantile(.95),
            syn_min=a.min(), syn_max=a.max(),
            emp_p05=b.quantile(.05), emp_median=sb, emp_p95=b.quantile(.95),
            emp_min=b.min(), emp_max=b.max(),
            centre_offset=ratio,
            covers_low=bool(a.min() <= b.min()), covers_high=bool(a.max() >= b.max()),
            emp_inside_syn=float(((b >= a.min()) & (b <= a.max())).mean()),
        ))
    return pd.DataFrame(rows)


def report(name, syn, emp):
    sc = score(syn, emp).assign(config=name)
    print(f'\n=== {name}  ({len(syn)} datasets) ===')
    print(sc[['metric', 'syn_p05', 'syn_median', 'syn_p95', 'emp_p05', 'emp_median',
              'emp_p95', 'centre_offset', 'covers_low', 'covers_high',
              'emp_inside_syn']].to_string(index=False,
                                           float_format=lambda v: f'{v:,.3f}'))
    return sc


if __name__ == '__main__':
    emp = pd.read_csv(os.path.join(TABLES, 'TABLE_2a_EmpiricalMetrics_final.csv'))
    pd.set_option('display.width', 260)

    base = G.DEFAULT
    candidates = {
        'A_default': base,
        'B_floor_to_0.99': base.replace(floor_ratio_hi=0.99),
        'C_skew_right_biased': base.replace(comp_skew_lo=-2.5, comp_skew_hi=5.0),
        'D_both': base.replace(floor_ratio_hi=0.99, comp_skew_lo=-2.5,
                               comp_skew_hi=5.0),
    }
    out = []
    for name, cfg in candidates.items():
        t = time.time()
        syn = sample_config(cfg, seed=4242)
        out.append(report(f'{name} [{time.time()-t:.0f}s]', syn, emp))
    write(pd.concat(out), 'TABLE_2a_Calibration.csv')

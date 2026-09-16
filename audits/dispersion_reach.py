"""Can the generator reach the empirical upper tail of dispersion, and at what cost?

The empirical arm has 6.0 percent of datasets above a coefficient of variation of
2 and 3.4 percent above 3, reaching 14.34. `corpus_2026-09-14d` has 0.08 percent
above 2 and NONE above 3, topping out at 2.58. That is both a coverage failure
and a distribution mismatch in the upper tail, so widening the generator would
serve the tuning objective and the coverage claim at once -- if it works.

This sweeps the three parameters that control dispersion and reports, for each
candidate, the achieved sample-CV distribution beside the full objective. The
question is whether the gap is a parameter setting or a structural limit: the
coefficient of variation is specified as a POPULATION target and measured as a
SAMPLE statistic, and a finite sample of a heavy-tailed distribution
systematically understates it.

    conda run -n compareuq python audits/dispersion_reach.py
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'audits'))

import genconfig as G  # noqa: E402
from tune_configuration import empirical_arm, sample_config, score  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
PER_STRATUM = 110

#: Candidates. `cv_log10_mean` is the centre of the population target,
#: `cv_log10_sd` its spread, `cv_log10_hi` the upper truncation of the draw.
CANDIDATES = {
    'current': {},
    'centre +0.2': dict(cv_log10_mean=0.329),
    'spread x1.4': dict(cv_log10_sd=0.3919 * 2.8),
    'centre +0.2, spread x1.4': dict(cv_log10_mean=0.329,
                                     cv_log10_sd=0.3919 * 2.8),
    'centre +0.4, spread x1.8, hi 60': dict(cv_log10_mean=0.529,
                                            cv_log10_sd=0.3919 * 3.6,
                                            cv_log10_hi=np.log10(60.0)),
    # None of the above move the achieved sample CV at all, because the binding
    # constraint is not the target: it is the positivity floor of the log
    # truncation rule, which caps the parent's quartile ratio at
    # 1 + 1/min_q1_over_iqr. At 0.5 that is 3, the empirical MEDIAN, while the
    # five uncovered categories have quartile ratios of 4.5 to 285.
    'min_q1_over_iqr 0.1': dict(min_q1_over_iqr=0.1),
    'min_q1_over_iqr 0.05': dict(min_q1_over_iqr=0.05),
    'min_q1_over_iqr 0.01': dict(min_q1_over_iqr=0.01),
    'min_q1_over_iqr 0.01 + centre +0.2': dict(min_q1_over_iqr=0.01,
                                               cv_log10_mean=0.329),
}


def main():
    os.makedirs(TABLES, exist_ok=True)
    emp_met, emp_modes, emp_vis = empirical_arm()
    ecv = emp_met['coeffvar'].replace([np.inf, -np.inf], np.nan).dropna()
    print(f'empirical arm: {len(emp_met)} datasets, '
          f'{(ecv > 2).mean()*100:.1f}% above CV 2, '
          f'{(ecv > 3).mean()*100:.1f}% above 3, max {ecv.max():.2f}\n')

    rows = []
    for name, kw in CANDIDATES.items():
        cfg = G.DEFAULT.replace(**kw) if kw else G.DEFAULT
        syn, modes, vis = sample_config(cfg, per_stratum=PER_STRATUM)
        d, _, s = score(emp_met, emp_modes, syn, modes, emp_vis, vis)
        scv = syn['coeffvar'].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(
            config=name, n=len(syn),
            objective=s['weighted_objective'],
            mean_w1=s['mean_w1_unweighted'],
            coeffvar_w1=s['coeffvar_w1'],
            pct_above_2=float((scv > 2).mean() * 100),
            pct_above_3=float((scv > 3).mean() * 100),
            max_cv=float(scv.max()),
            visible_tv=s['visible_tv'],
            worst=s['worst_metric'], worst_w1=s['worst_w1'],
        ))
        r = rows[-1]
        print(f'{name:<34} objective {r["objective"]:.4f}  meanW1 {r["mean_w1"]:.4f}'
              f'  cvW1 {r["coeffvar_w1"]:.3f}  above2 {r["pct_above_2"]:5.2f}%'
              f'  above3 {r["pct_above_3"]:5.2f}%  maxCV {r["max_cv"]:6.2f}'
              f'  visTV {r["visible_tv"]:.3f}  worst {r["worst"]}')

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TABLES, 'TABLE_DispersionReach.csv'), index=False)
    print(f"\nempirical targets: above2 {(ecv > 2).mean()*100:.2f}%  "
          f"above3 {(ecv > 3).mean()*100:.2f}%  max {ecv.max():.2f}")


if __name__ == '__main__':
    main()

"""Is a standardized worsening real, or is it the denominator? Stage 2h.

THE AUTHOR ASKED WHAT VALUE JUDGMENT "worst characteristic" CARRIES. None:
`tune_configuration.score` reports the first row of
`coverage.distribution_comparison` sorted by standardized distance descending,
so it means "the characteristic on which the two arms' DISTRIBUTIONS sit
furthest apart" and not "the least important characteristic".

But the question exposes a real hazard. That standardization divides by the
EMPIRICAL standard deviation of the characteristic across categories, and
decision 63 records a case in this project where a standardized WORSENING was
the denominator shrinking rather than the corpus degrading. So a claim that
rests on a standardized move is checked here in ABSOLUTE terms.

THE STANDARDIZATION IS NECESSARY AND IS NOT ITSELF A JUDGMENT. The
characteristics are in incomparable units -- the distance in dataset SIZE runs
to 848 while the one in a Shapiro statistic runs to 0.04 -- so an
unstandardized mean over them is meaningless and is dominated by `n` alone.
What the check establishes is only that a SINGLE characteristic's move has the
same sign either way.

Run on the configuration this stage needed it for:

`coverage.distribution_comparison` standardizes by the EMPIRICAL standard
deviation of each characteristic across categories. Decision 63 records a case
where a standardized WORSENING was an artifact of that denominator shrinking.
The empirical arm does not change between these two configurations, so the
denominator is FIXED here -- but the check is one line and the claim is load
bearing, so it is made rather than assumed.
"""
import sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src'); sys.path.insert(0,'audits')
import numpy as np, pandas as pd
from scipy import stats as st
import coverage, genconfig as G
import tune_configuration as TC
pd.set_option('display.width',200)

emp_met, emp_modes, emp_vis = TC.empirical_arm()

def per_characteristic(cfg, label):
    syn_met, syn_modes, syn_vis = TC.sample_config(cfg, per_stratum=TC.PER_STRATUM)
    d = coverage.distribution_comparison(emp_met, syn_met)
    # the ABSOLUTE distance, undivided
    rows=[]
    for m in d.metric:
        e = coverage._clean(emp_met[m]); s = coverage._clean(syn_met[m])
        rows.append(dict(metric=m, absolute_w1=float(st.wasserstein_distance(e,s)),
                         empirical_sd=float(e.std())))
    a = pd.DataFrame(rows).merge(d[['metric','w1_standardized']], on='metric')
    a['config']=label
    return a

base = per_characteristic(G.DEFAULT, 'default')
cand = per_characteristic(G.DEFAULT.replace(min_q1_over_iqr=0.05), 'min_q1_over_iqr 0.05')
j = base.merge(cand, on='metric', suffixes=('_base','_cand'))
j['abs_change'] = j.absolute_w1_cand - j.absolute_w1_base
j['std_change'] = j.w1_standardized_cand - j.w1_standardized_base
print('PER CHARACTERISTIC: absolute Wasserstein distance between the two arms,')
print('and the standardized version. The empirical arm is IDENTICAL between')
print('the two configurations, so empirical_sd is the same and the two columns')
print('must agree in SIGN. If they do, the claim is not a denominator artifact.')
print()
print(j[['metric','absolute_w1_base','absolute_w1_cand','abs_change',
         'w1_standardized_base','w1_standardized_cand','std_change']]
      .sort_values('abs_change', ascending=False)
      .to_string(index=False, float_format=lambda v:f'{v:.4f}'))
print()
print(f"mean ABSOLUTE distance: default {j.absolute_w1_base.mean():.4f} -> "
      f"candidate {j.absolute_w1_cand.mean():.4f}")
print(f"signs agree on {int((np.sign(j.abs_change)==np.sign(j.std_change)).sum())}"
      f" of {len(j)} characteristics")

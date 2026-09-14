"""The empirical envelope before and after the Stage 2a-3 category split.

Every range in src/genconfig.py cites a measurement of the empirical arm, so
splitting the arm invalidates the calibration if any cited measurement moves.
This reports each of them on the unsplit and the split arm side by side, which
is the input to the decision about regenerating.

Silverman's unimodal share carries a few points of estimator noise, so nboot is
quoted with it. The visible-mode distribution is the measure the generator is
actually steered by; see CONTEXT.md section 5.

    conda run -n compareuq python audits/stage2a3/q2_envelope_before_after.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import empirical  # noqa: E402
import genconfig as G  # noqa: E402
import modality as MD  # noqa: E402
from customstats import empirical_metadata  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a3')
SEED = 42
NBOOT = 100


def arm(split):
    ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0],
                              split=split)
    met = pd.DataFrame({m: empirical_metadata(x, w)
                        for m, (x, w) in ds.items()}).T.astype(float)
    met.index.name = 'dataset'
    rng = np.random.default_rng(0)
    met['n_modes_silverman'] = [MD.n_modes_silverman(x, rng=rng, nboot=NBOOT)
                                for x, _ in ds.values()]
    met['n_modes_visible'] = [MD.n_modes_visible(x) for x, _ in ds.values()]
    return met


def summarise(met):
    def clean(c):
        return met[c].replace([np.inf, -np.inf], np.nan).dropna()
    cv, sk, ku = clean('coeffvar'), clean('skewness'), clean('kurtosis')
    lcv = np.log10(cv[cv > 0])
    n = met['n']
    out = {
        'datasets': len(met),
        'coeffvar median': cv.median(),
        'coeffvar log10 sd': lcv.std(),
        'coeffvar log10 mean': lcv.mean(),
        'coeffvar min': cv.min(),
        'coeffvar max': cv.max(),
        'skewness median': sk.median(),
        'skewness min': sk.min(),
        'skewness max': sk.max(),
        'kurtosis median': ku.median(),
        'kurtosis max': ku.max(),
        'n median': n.median(),
        'n max': n.max(),
        'crit_bw_1 median': clean('crit_bw_1').median(),
        'crit_bw_1 max': clean('crit_bw_1').max(),
        f'silverman unimodal share (nboot={NBOOT})':
            (met.n_modes_silverman == 1).mean(),
        'visible unimodal share': (met.n_modes_visible == 1).mean(),
        'visible 2 modes': (met.n_modes_visible == 2).mean(),
        'visible 3+ modes': (met.n_modes_visible >= 3).mean(),
        'entropy median': clean('entropy').median(),
        'weight_outliers median': clean('weight_outliers').median(),
        'fit_norm_SW median': clean('fit_norm_SW').median(),
        'fit_lognorm_SW median': clean('fit_lognorm_SW').median(),
        'w_v_uw_wasserstein median': clean('w_v_uw_wasserstein').median(),
        'modality_index median': clean('modality_index').median(),
    }
    for st in G.DEFAULT.strata:
        out[f'stratum share {st.name}'] = float(
            ((n >= st.n_lo) & (n <= st.n_hi)).mean())
    out['share above 9999'] = float((n > G.DEFAULT.strata[-1].n_hi).mean())
    return pd.Series(out)


def main():
    os.makedirs(TABLES, exist_ok=True)
    before, after = arm(split=False), arm(split=True)
    before.to_csv(os.path.join(TABLES, 'TABLE_2a3_EnvelopeUnsplit.csv'))
    after.to_csv(os.path.join(TABLES, 'TABLE_2a3_EnvelopeSplit.csv'))

    a, b = f'unsplit ({len(before)})', f'split ({len(after)})'
    tab = pd.DataFrame({a: summarise(before), b: summarise(after)})
    tab['change'] = tab[b] - tab[a]
    tab.index.name = 'quantity'
    tab.to_csv(os.path.join(TABLES, 'TABLE_2a3_EnvelopeBeforeAfter.csv'))
    pd.set_option('display.width', 120)
    print(tab.to_string(float_format=lambda v: f'{v:10.4f}'))

    print('\n=== what genconfig cites, and whether it still brackets the arm ===')
    cv = after['coeffvar'].replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = 10 ** G.DEFAULT.cv_log10_lo, 10 ** G.DEFAULT.cv_log10_hi
    print(f'  cv range {lo:.4f} to {hi:.4f} brackets [{cv.min():.4f}, '
          f'{cv.max():.4f}]: {"YES" if lo < cv.min() and hi > cv.max() else "NO"}')
    print(f'  cv_log10_mean  {G.DEFAULT.cv_log10_mean:.4f}  vs arm '
          f'{np.log10(cv[cv > 0]).mean():.4f}')
    print(f'  cv_log10_sd    {G.DEFAULT.cv_log10_sd:.4f}  vs arm '
          f'{np.log10(cv[cv > 0]).std() * 2:.4f} (config is 2x the arm sd)')
    print('  EMPIRICAL_STRATUM_SHARE:')
    for st in G.DEFAULT.strata:
        s = float(((after.n >= st.n_lo) & (after.n <= st.n_hi)).mean())
        print(f'    {st.name:<14} config {G.EMPIRICAL_STRATUM_SHARE[st.name]:.4f}'
              f'   split arm {s:.4f}')


if __name__ == '__main__':
    main()

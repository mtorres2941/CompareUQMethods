"""The corpus's multimodal datasets are the WRONG SHAPE, not merely too few.

WHY THIS EXISTS. Decision 82 established that the corpus under-represents
multimodal datasets and then measured that reweighting it to the empirical mode
mix moves the kernel-estimate-minus-lognormal difference by 0.0004, and closed
the question on that basis. The author reopened it -- "that's one of the main
purposes of the corpus" -- and the reweighting argument turns out not to answer
it, for a reason worth writing down: REWEIGHTING CAN ONLY REWEIGHT DATASETS
THAT EXIST. If the corpus's multimodal datasets are a different object from real
multimodal ones, no weighting of them reproduces the real population.

THE TEST. For each arm, correlate the visible mode count with every other
characteristic. If the corpus's multimodality is the real thing in smaller
numbers, the signs agree. They do not: the sign is OPPOSITE on all six.

THE MECHANISM, which the same script measures. The generator makes a second
visible mode by SEPARATING components -- overlap falls as the mode count rises.
Separated components are each individually tidy, so a multi-mode synthetic
dataset ends up LESS skewed, LESS dispersed and LESS heavy-tailed than a
unimodal one. Real multimodality is a shoulder on a long-tailed body, so it
comes with MORE of all three. Decision 37 predicted exactly this and it was
never measured against the mode count.

    conda run -n compareuq python audits/corpus_modality_shape.py
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

TABLES = os.path.join(ROOT, 'outputs', 'tables')
OUT = os.path.join(TABLES, 'audits')
CHARS = ['coeffvar', 'skewness', 'kurtosis', 'fit_lognorm_SF', 'fit_norm_SF',
         'crit_bw_1']


def arms():
    vm = pd.read_csv(os.path.join(TABLES, 'TABLE_VisibleModes.csv'))
    e = (pd.read_excel(os.path.join(TABLES,
                                    'TABLE_EmpiricalECCMetricsAndW1.xlsx'))
         .rename(columns={'Unnamed: 0': 'dataset'}))
    s = pd.read_excel(os.path.join(TABLES,
                                   'TABLE_SyntheticECCMetricsAndW1.xlsx'))
    e = e.merge(vm[vm.arm == 'empirical'][['dataset', 'modes_fitted', 'n']],
                on='dataset')
    s = s.merge(vm[vm.arm == 'synthetic'][['dataset', 'modes_fitted']],
                on='dataset')
    return e.assign(arm='empirical'), s.assign(arm='synthetic')


def main():
    e, s = arms()
    rows = []
    for c in CHARS:
        if c not in e.columns or c not in s.columns:
            continue
        re_ = spearmanr(e.modes_fitted, e[c])[0]
        rs = spearmanr(s.modes_fitted, s[c])[0]
        rows.append(dict(characteristic=c, empirical=re_, synthetic=rs,
                         same_sign=bool(np.sign(re_) == np.sign(rs))))
    corr = pd.DataFrame(rows)
    print('HOW MODALITY CO-VARIES WITH EVERYTHING ELSE, by arm.')
    print('Spearman of the visible mode count with each characteristic.\n')
    print(corr.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print(f'\nopposite sign on {int((~corr.same_sign).sum())} of {len(corr)} '
          'characteristics.\n')

    both = pd.concat([e[CHARS + ['modes_fitted', 'arm']],
                      s[CHARS + ['modes_fitted', 'arm']]])
    both['modes'] = both.modes_fitted.clip(upper=3)
    print('MEDIAN CHARACTERISTIC BY MODE COUNT AND ARM:')
    for c in CHARS:
        piv = both.pivot_table(index='modes', columns='arm', values=c,
                               aggfunc='median')
        print(f'\n  {c}')
        print(piv.to_string(float_format=lambda v: f'{v:+.3f}'))

    # the generator-side mechanism: modes come from SEPARATION, not shape.
    metrics = pd.read_parquet(
        os.path.join(__import__('corpus').active_dir(), 'metrics.parquet'))
    vm = pd.read_csv(os.path.join(TABLES, 'TABLE_VisibleModes.csv'))
    g = metrics.merge(vm[vm.arm == 'synthetic'][['dataset', 'modes_fitted']],
                      on='dataset')
    g = g[g.k > 1].copy()
    g['modes'] = g.modes_fitted.clip(upper=3)
    print('\n\nWHERE THE CORPUS GETS ITS MODES (multi-component datasets only):')
    print(g.groupby('modes').agg(datasets=('dataset', 'count'),
                                 overlap=('overlap_achieved', 'median'),
                                 coeffvar=('coeffvar', 'median'),
                                 skewness=('skewness', 'median'),
                                 kurtosis=('kurtosis', 'median'))
          .to_string(float_format=lambda v: f'{v:.3f}'))
    print(f"\n  Spearman(overlap achieved, visible modes) = "
          f"{spearmanr(g.overlap_achieved, g.modes_fitted)[0]:+.3f}")

    # how much of the real multimodal population the corpus can reach
    em = e[e.modes_fitted >= 2]
    sm = g[g.modes_fitted >= 2]
    print(f'\n\nTHE HOLE. {len(em)} real multimodal categories, '
          f'{len(sm)} corpus multimodal datasets.')
    hole = []
    for c in ('coeffvar', 'skewness', 'kurtosis'):
        med = em[c].median()
        share = float((sm[c] > med).mean())
        hole.append(dict(characteristic=c, real_median=med,
                         corpus_share_above=share))
        print(f'  {c:10s} real median {med:7.3f} | corpus share above it '
              f'{100 * share:5.1f} pct  (50 pct if the arms matched)')
    os.makedirs(OUT, exist_ok=True)
    corr.to_csv(os.path.join(OUT, 'TABLE_CorpusModalityShape.csv'), index=False)
    pd.DataFrame(hole).to_csv(
        os.path.join(OUT, 'TABLE_CorpusModalityHole.csv'), index=False)
    print(f'\nwrote {OUT}/TABLE_CorpusModalityShape.csv and '
          'TABLE_CorpusModalityHole.csv')


if __name__ == '__main__':
    main()

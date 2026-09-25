"""Does the corpus's dispersion shortfall bias the method comparison?

THE QUESTION THAT DECIDES WHETHER TO CHASE IT. The corpus is less dispersed
than the real categories -- median coefficient of variation 0.51 against 0.64,
p90 0.89 against 1.49. Three stages have failed to close that with any
generator parameter. The question is not how big the gap is; it is whether any
conclusion depends on it.

THE TEST, which is decision 82's applied to dispersion rather than modality.
Reweight the corpus so its DISPERSION distribution matches the real one, and
see whether the method comparison moves. Reweighting can only reweight
datasets that exist, so this is a fair test exactly where the corpus HAS
datasets at a given dispersion and understates how many; it cannot speak to
dispersion the corpus never reaches, which is reported separately as the share
of empirical weight that falls outside the corpus range.

IT IS NOW GENERAL OVER THE CHARACTERISTIC, because Stage 2h needed the same
question asked of two more. The bounded widening candidates improve dispersion
and the weighting distance together and WORSEN Silverman's critical bandwidth,
from 0.158 to 0.259 standardized, on a characteristic carrying weight 3 in the
calibration objective (decision 193). "A characteristic moved" is not a reason
to act until it is shown to move a conclusion, which is what this measures.

    conda run -n compareuq python audits/dispersion_matters.py
    conda run -n compareuq python audits/dispersion_matters.py --metric crit_bw_1
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus                  # noqa: E402
import empirical               # noqa: E402
from customstats import empirical_metadata   # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
OUT = os.path.join(ROOT, 'outputs', 'tables')

#: Dispersion bins to match on. Equal-count over the REAL categories, so each
#: carries the same weight of real evidence.
N_BINS = 8


def main(metric='coeffvar'):
    os.makedirs(TABLES, exist_ok=True)
    pd.set_option('display.width', 210)
    scores = pd.read_csv(os.path.join(OUT, 'TABLE_TargetComparison.csv'))
    scores = scores[scores.arm == 'synthetic'].copy()
    scores['dataset'] = scores.dataset.astype(str)
    met, _, _ = corpus.load_corpus(with_values=False)
    met = met[met.dataset.astype(str).str.startswith('dataset')].copy()
    met['dataset'] = met.dataset.astype(str)

    emp, _ = empirical.prepare(np.random.default_rng(20260912))
    ecv = np.array([float(empirical_metadata(np.asarray(x), np.asarray(w))
                          [metric]) for x, w in emp.values()])
    ecv = ecv[np.isfinite(ecv)]

    piv = (scores.pivot_table(index='dataset', columns='method',
                              values='w1_market')
           .merge(met.set_index('dataset')[[metric, 'n']],
                  left_index=True, right_index=True))
    piv = piv.dropna()
    k, lg = 'KDE, Uniform', 'Lognormal, Uniform'
    kv, lv = 'KDE, Variable', 'Lognormal, Variable'

    # Equal-count bins over the REAL dispersion distribution.
    edges = np.unique(np.quantile(ecv, np.linspace(0, 1, N_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    emp_share = np.histogram(ecv, bins=edges)[0] / len(ecv)
    syn_bin = np.digitize(piv[metric], edges) - 1
    syn_share = np.bincount(syn_bin, minlength=len(emp_share)) / len(piv)

    print(f'{metric.upper()}, by equal-count bins of the REAL categories:')
    rows = []
    for i in range(len(emp_share)):
        rows.append(dict(bin=f'{edges[i]:.2f}-{edges[i+1]:.2f}',
                         real_share=emp_share[i], corpus_share=syn_share[i],
                         corpus_n=int((syn_bin == i).sum())))
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f'{v:.4f}'))
    empty = [r for r in rows if r['corpus_n'] == 0]
    outside = sum(r['real_share'] for r in empty)
    print()
    print(f'REAL WEIGHT THE CORPUS CANNOT REPRESENT AT ALL: {outside:.4f}')
    print('  (bins where the corpus holds no datasets; reweighting cannot')
    print('   reach these and they are the honest limit of this test)')

    # The reweighting: each corpus dataset gets the real share of its bin over
    # the corpus share of its bin, so the weighted corpus matches the real
    # dispersion distribution wherever it has datasets.
    w = np.zeros(len(piv))
    for i in range(len(emp_share)):
        pick = syn_bin == i
        if pick.sum() and syn_share[i] > 0:
            w[pick] = emp_share[i] / syn_share[i]
    w = w / w.sum() if w.sum() > 0 else w

    print()
    print('=' * 74)
    print('DOES THE METHOD COMPARISON MOVE?')
    print('=' * 74)
    out = []
    for a, b, label in ((k, lg, 'equal weights'), (kv, lv, 'Dirichlet shares')):
        if a not in piv or b not in piv:
            continue
        wins = (piv[a] < piv[b]).to_numpy(float)
        gap = (piv[a] - piv[b]).to_numpy(float)
        out.append(dict(
            comparison=label,
            kde_wins_asis=float(wins.mean()),
            kde_wins_reweighted=float(np.sum(w * wins)),
            mean_gap_asis=float(gap.mean()),
            mean_gap_reweighted=float(np.sum(w * gap))))
    d = pd.DataFrame(out)
    d['wins_shift'] = d.kde_wins_reweighted - d.kde_wins_asis
    d['gap_shift'] = d.mean_gap_reweighted - d.mean_gap_asis
    d.insert(0, 'metric', metric)
    suffix = '' if metric == 'coeffvar' else f'_{metric}'
    d.to_csv(os.path.join(TABLES,
                          f'TABLE_DispersionMatters{suffix}.csv'),
             index=False)
    print(d.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print()
    print('`kde_wins` is the share of datasets on which the kernel estimate is')
    print('closer to the TRUE parent than the three-parameter lognormal. If')
    print('reweighting the corpus to the real dispersion mix barely moves it,')
    print('the shortfall bounds COVERAGE and not the conclusion.')
    print()
    print('AND PER METHOD, mean distance from the truth:')
    m = pd.DataFrame({
        'as_is': {c: float(piv[c].mean()) for c in piv.columns
                  if c not in ('coeffvar', 'n')},
        'reweighted': {c: float(np.sum(w * piv[c])) for c in piv.columns
                       if c not in ('coeffvar', 'n')}})
    m['shift'] = m.reweighted - m.as_is
    print(m.sort_values('as_is').to_string(float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', default='coeffvar',
                    help='the characteristic to match the corpus '
                         'to the real arm on')
    main(ap.parse_args().metric)

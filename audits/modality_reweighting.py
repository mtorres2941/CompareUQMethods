"""Does the corpus's modality shortfall change the answer? Reweight and see.

The author asked the right follow-up to the finding that "95 percent of ECC
datasets are visibly unimodal" is an artifact of Scott's bandwidth: if the corpus
under-represents multimodal datasets, and multimodality is what a kernel estimate
is for, does the corpus need rebuilding?

**REWEIGHTING ANSWERS IT WITHOUT REBUILDING ANYTHING**, and it is the same trick
post-stratification already uses for dataset size. If the corpus is reweighted to
the empirical distribution of visible modes and the conclusion does not move, the
shortfall is a stated limitation. If it moves a lot, regeneration is warranted.
Reweighting is reversible and costs minutes; regeneration reopens decisions 47,
48 and 55 and costs the whole of Stage 2a.

THE MODE COUNT IS TAKEN AT THE BANDWIDTH THE STUDY ACTUALLY FITS, which is
`silverman_guarded`, rather than at scipy's default or at an invented multiple.
That is the defensible choice: the density whose modes are counted is the density
the study puts in front of a reader and samples from in the pLCA. It needs no new
parameter and it is not tuned to anything.

    conda run -n compareuq python audits/modality_reweighting.py [n_synth]
"""
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus  # noqa: E402
import empirical  # noqa: E402
import families as F  # noqa: E402
import fitting as FT  # noqa: E402
import mixture as M  # noqa: E402
import recovery as R  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'audits')
SEED = 42
N_SYNTH = 3_000
MIN_N = 8
PROMINENCE = 0.05


def modes_of_fitted_kde(x, w, grid_n=1024, prominence=PROMINENCE):
    """Visible modes of the density the STUDY FITS, at its own bandwidth.

    Not `gaussian_kde`'s default, which is Scott's rule and oversmooths this data
    by about 35 percent, and not an arbitrary multiple of it.
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    w = w / w.sum()
    if len(x) < MIN_N or np.std(x) <= 0:
        return 1
    m, _ = FT.fit_kde(x, w)
    g = np.linspace(float(x.min()), float(x.max()), grid_n)
    y = np.asarray(m.pdf(g), float)
    idx = argrelextrema(y, np.greater)[0]
    if not len(idx) or y.max() <= 0:
        return 1
    kept = 0
    for i in idx:
        left = y[:i].min() if i > 0 else y[i]
        right = y[i + 1:].min() if i < len(y) - 1 else y[i]
        if (y[i] - max(left, right)) / y.max() >= prominence:
            kept += 1
    return max(kept, 1)


def mode_class(k):
    return '1' if k <= 1 else ('2' if k == 2 else '3+')


def main(n_synth=N_SYNTH):
    os.makedirs(TABLES, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    emp, _ = empirical.prepare(rng.spawn(1)[0])
    met, vals, _ = corpus.load_corpus()
    ids = sorted(met.sample(n_synth, random_state=0).dataset.astype(str))
    syn = corpus.as_dict(vals[vals.dataset_id.isin(set(ids))])
    specs = corpus.load_parent_specs()

    emp_rows = []
    for name, (x, w) in emp.items():
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        if len(x) < MIN_N:
            continue
        emp_rows.append(dict(arm='empirical', dataset=name, n=len(x),
                             modes=modes_of_fitted_kde(x, w)))
    emp_d = R.add_size_band(pd.DataFrame(emp_rows))
    emp_d['mode_class'] = emp_d.modes.map(mode_class)

    rows = []
    for i, name in enumerate(ids):
        x, w = syn[name]
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        if len(x) < MIN_N:
            continue
        parent = M.parent_from_spec(specs[name])
        grid = R.recovery_grid(x, w, parent)
        models, _ = FT.fit_pewt(x, w)
        r = dict(arm='synthetic', dataset=name, n=len(x),
                 modes=modes_of_fitted_kde(x, w))
        for label, m in models.items():
            r[label] = R.w1_against_parent(
                m, parent, R.OWN_TARGET_SCHEME[R.FT_weighting(label)], grid)
        rows.append(r)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(ids)}  {time.time()-t0:.0f}s', flush=True)
    syn_d = R.add_size_band(pd.DataFrame(rows))
    syn_d['mode_class'] = syn_d.modes.map(mode_class)

    pd.concat([emp_d, syn_d], ignore_index=True).to_csv(
        os.path.join(TABLES, 'TABLE_ModalityReweighting.csv'), index=False)
    report(emp_d, syn_d)


def report(emp_d, syn_d):
    pd.set_option('display.width', 220)
    print()
    print('=' * 78)
    print('VISIBLE MODES OF THE DENSITY THE STUDY FITS, n >= 8')
    print('=' * 78)
    ce = emp_d.mode_class.value_counts(normalize=True).sort_index()
    cs = syn_d.mode_class.value_counts(normalize=True).sort_index()
    t = pd.DataFrame({'empirical': ce, 'synthetic': cs}).fillna(0.0)
    print((100 * t).round(2).to_string())
    print(f'  total variation between the arms: '
          f'{0.5 * float((t.empirical - t.synthetic).abs().sum()):.4f}')
    print(f'  empirical datasets: {len(emp_d)}, synthetic: {len(syn_d)}')

    print()
    print('=' * 78)
    print('DOES REWEIGHTING THE CORPUS TO THE EMPIRICAL MODE MIX MOVE THE ANSWER?')
    print('=' * 78)
    shares = ce.to_dict()
    for wt in ('Uniform', 'Variable'):
        kde, logn = f'KDE, {wt}', f'Lognormal, {wt}'
        print(f'--- {wt} ---')
        per = syn_d.groupby('mode_class').apply(
            lambda g: pd.Series({
                'n_datasets': len(g),
                'corpus_share': len(g) / len(syn_d),
                'empirical_share': shares.get(g.name, 0.0),
                'kde': g[kde].mean(), 'lognormal': g[logn].mean(),
                'kde_minus_logn': (g[kde] - g[logn]).mean(),
                'kde_wins': float((g[kde] < g[logn]).mean())}),
            include_groups=False)
        print(per.to_string(float_format=lambda v: f'{v:.4f}'))
        d_equal = float((syn_d[kde] - syn_d[logn]).mean())
        w = np.array([shares.get(c, 0.0) for c in per.index], float)
        d_rw = float((per.kde_minus_logn.to_numpy() * w).sum() / w.sum())
        print(f'  KDE minus lognormal, corpus as drawn   {d_equal:+.4f}')
        print(f'  KDE minus lognormal, mode-reweighted   {d_rw:+.4f}'
              f'   (change {d_rw - d_equal:+.4f})')
        print()
    print('IF THE CHANGE IS SMALL, the modality shortfall is a limitation to')
    print('state and NOT a reason to reopen generation, which decisions 47, 48')
    print('and 55 close. Reweighting is reversible; regeneration is not.')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SYNTH)

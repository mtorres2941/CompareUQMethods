"""Fast parent-level probe for the modality-shape defect. Seconds, not minutes.

WHY THIS EXISTS. Judging a generator change by building a 1,000-dataset draft
corpus costs about 100 seconds per candidate, which is too slow to search a
design space. This draws parents directly, samples each one, and reports the
two things that matter -- the share of datasets that are multimodal AND
dispersed, and the SIGN of the modality-shape correlations -- for any set of
configuration overrides. Use it to find candidates; use
`audits/corpus_joint_structure.py` to confirm one on a real draft.

    conda run -n compareuq python audits/shoulder_probe.py [n_parents]
"""
import dataclasses
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import skew as sample_skew, kurtosis as sample_kurt, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import genconfig as G      # noqa: E402
import generator           # noqa: E402
import modality as MO      # noqa: E402

OUT = os.path.join(ROOT, 'outputs', 'tables', 'audits')
#: measured on the 130 real categories that carry a visible-mode count.
#: the real arm, 130 categories carrying a visible-mode count. THREE numbers,
#: and they are not equally wrong: the multimodal share is 31.5 against the
#: corpus's 24.1, which is a gap and not a crisis; the DISPERSED share is 36.2
#: against 11.4; and conditional on being dispersed a real category is
#: multimodal 44.7 percent of the time against the corpus's 16.9.
TARGET = dict(multimodal=0.315, dispersed=0.362, multi_dispersed=0.162,
              multimodal_given_dispersed=0.447, rho_cv=0.163,
              rho_skew=0.211, rho_kurt=0.230)
DISPERSED = 0.889          # the real multimodal median coefficient of variation

VARIANTS = {
    'as shipped': {},
    'unequal modes': dict(mode_share_alpha=1.0),
    'wider ceiling': dict(trunc_iqr_mult=5.0, min_q1_over_iqr=0.02),
    'shoulder, wide body': dict(shoulder_frac=1.0, shoulder_body='wide',
                                mode_share_alpha=1.0),
    'shoulder, narrow body': dict(shoulder_frac=1.0, shoulder_body='narrow',
                                  mode_share_alpha=1.0),
    'narrow + wider': dict(shoulder_frac=1.0, shoulder_body='narrow',
                           mode_share_alpha=1.0, trunc_iqr_mult=5.0,
                           min_q1_over_iqr=0.02),
    'narrow + widest': dict(shoulder_frac=1.0, shoulder_body='narrow',
                            mode_share_alpha=1.0, trunc_iqr_mult=8.0,
                            min_q1_over_iqr=0.01),
    'half narrow + wider': dict(shoulder_frac=0.5, shoulder_body='narrow',
                                mode_share_alpha=1.0, trunc_iqr_mult=5.0,
                                min_q1_over_iqr=0.02),
    # THE FLOOR IS THE BINDING CONSTRAINT. Dispersion is produced by the SHIFT
    # -- Spearman(shift, cv) = -0.654 -- and separation contributes nothing,
    # -0.017. The smallest admissible shift is
    # `min_q1_over_iqr * (q3 - q1) - q1`, so a WIDER mixture is forced to shift
    # MORE, which caps its coefficient of variation lower. That is the causal
    # chain behind the negative modality-dispersion correlation, and
    # `min_q1_over_iqr` is the parameter in it.
    'floor 0.005': dict(min_q1_over_iqr=0.005, trunc_iqr_mult=8.0),
    'floor 0.001': dict(min_q1_over_iqr=0.001, trunc_iqr_mult=12.0),
    'floor 0.001 + modes': dict(min_q1_over_iqr=0.001, trunc_iqr_mult=12.0,
                                mode_share_alpha=1.0, overlap_log10_hi=-0.10),
}


def probe(overrides, n_parents, seed=11):
    rng = np.random.default_rng(seed)
    cfg = dataclasses.replace(G.DEFAULT, **overrides) if overrides else G.DEFAULT
    rows = []
    for _ in range(n_parents):
        n = int(10 ** rng.uniform(1.5, 3.5))
        parent = None
        for _ in range(cfg.max_parent_retries):
            parent, rec = generator.draw_parent(cfg, n, rng)
            if parent is not None:
                break
        if parent is None:
            continue
        x, _m = parent.sample(n, rng)
        x = x / np.mean(x)
        if len(x) < 8 or np.std(x) == 0:
            continue
        rows.append(dict(n=n, cv=float(np.std(x) / np.mean(x)),
                         skewness=float(sample_skew(x)),
                         kurt=float(sample_kurt(x)),
                         modes=int(MO.n_modes_fitted(x))))
    return pd.DataFrame(rows)


def summarize(name, d):
    mm = d.modes >= 2
    disp = d.cv > DISPERSED
    return dict(variant=name, parents=len(d),
                multimodal=float(mm.mean()),
                dispersed=float(disp.mean()),
                multi_dispersed=float((mm & disp).mean()),
                multimodal_given_dispersed=float(mm[disp].mean())
                if disp.any() else float('nan'),
                median_cv=float(d.cv.median()),
                rho_cv=spearmanr(d.modes, d.cv)[0],
                rho_skew=spearmanr(d.modes, d['skewness'])[0],
                rho_kurt=spearmanr(d.modes, d['kurt'])[0])


def main(argv):
    n_parents = int(argv[1]) if len(argv) > 1 else 500
    rows = []
    for name, ov in VARIANTS.items():
        t0 = time.time()
        d = probe(ov, n_parents)
        row = summarize(name, d)
        row['seconds'] = round(time.time() - t0)
        rows.append(row)
        print(f"  {name:24s} {row['seconds']:4d}s  multimodal "
              f"{100*row['multimodal']:5.1f}  multi+disp "
              f"{100*row['multi_dispersed']:5.2f}  rho_cv {row['rho_cv']:+.3f}",
              flush=True)
    out = pd.DataFrame(rows)
    print('\n' + out.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print('\nREAL ARM TARGET: multimodal 0.315, dispersed 0.362, '
          'multi+dispersed 0.162, multimodal|dispersed 0.447,')
    print('                 rho_cv +0.163, rho_skew +0.211, rho_kurt +0.230')
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, 'TABLE_ShoulderProbe.csv'), index=False)
    print(f'\nwrote {OUT}/TABLE_ShoulderProbe.csv')


if __name__ == '__main__':
    main(sys.argv)

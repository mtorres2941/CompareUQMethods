"""Candidate generator configurations, judged on the JOINT structure.

WHY THIS EXISTS. `audits/corpus_modality_shape.py` established that the corpus
gets the joint structure of modality and shape BACKWARDS: in the real arm more
visible modes come with more spread, skew and kurtosis, and in the corpus they
come with less, on all six characteristics. The calibration objective matches
MARGINAL distributions one characteristic at a time (decision 35) and never
looks at a correlation, so a configuration can match every margin and still be
wrong this way -- which is what happened.

Stage 2f's `dispersion_reach` sweep (decision 138) already tried widening the
dispersion ceiling and judged the candidates on the MARGINAL objective alone.
This re-runs a short version of that sweep and adds the measurement that was
missing: does the candidate also fix the SIGN of the modality-shape
correlations, and at what cost to the margins that currently match?

DRAFT CORPORA ONLY. Each candidate generates a small corpus under a label that
says `draft` (decision 41), and a draft must never supply a paper number.

    conda run -n compareuq python audits/corpus_joint_structure.py [n_datasets]
"""
import dataclasses
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import corpus                       # noqa: E402
import coverage                     # noqa: E402
import genconfig as G               # noqa: E402
import modality as MO               # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables')
OUT = os.path.join(TABLES, 'audits')
CHARS = ['coeffvar', 'skewness', 'kurtosis', 'fit_norm_SF', 'crit_bw_1']

#: The candidates. `trunc_iqr_mult` with `min_q1_over_iqr` is the dispersion
#: ceiling decision 138 identified as the binding constraint -- the achieved
#: coefficient of variation tops out at 1.447 against a real arm reaching 6.93,
#: and 57 percent of drawn targets come back clipped. `overlap_log10_hi` is the
#: modality lever: lower means more separated components and more visible modes.
CANDIDATES = {
    'current': {},
    'wider_dispersion': dict(trunc_iqr_mult=5.0, min_q1_over_iqr=0.02),
    'wider_and_blended': dict(trunc_iqr_mult=5.0, min_q1_over_iqr=0.02,
                              overlap_log10_hi=0.30),
    'widest': dict(trunc_iqr_mult=8.0, min_q1_over_iqr=0.01),
    # THE SECOND ROUND. The first four all widened the DISPERSION ceiling and
    # all four left the sign of the modality-shape correlation negative, so the
    # ceiling is not what flips it. What makes a synthetic multimodal dataset
    # tidy is that its modes are the SAME SIZE: `mode_share_alpha = 10` puts
    # the larger of two modes between 0.50 and 0.76 of the points (decision
    # 27), which is a symmetric two-humped shape, and symmetric shapes have low
    # skew by construction. Real multimodality is a small shoulder on a big
    # skewed body. Dropping the concentration to 1 makes the mode sizes a flat
    # Dirichlet -- the larger mode spans 0.52 to 0.97 -- which is the author's
    # own proposal already carried to Stage 2h in decision 141.
    'unequal_modes': dict(mode_share_alpha=1.0),
    'unequal_and_wider': dict(mode_share_alpha=1.0, trunc_iqr_mult=5.0,
                              min_q1_over_iqr=0.02),
    'unequal_wider_separated': dict(mode_share_alpha=1.0, trunc_iqr_mult=5.0,
                                    min_q1_over_iqr=0.02,
                                    overlap_log10_hi=-0.10),
    # THE THIRD ROUND, and the first that changes the CONSTRUCTION rather than
    # a range. `shoulder_frac` pairs the largest weight and the widest
    # component with the lowest position, so a multi-component parent becomes
    # one dominant body with small narrow components on its upper tail instead
    # of a symmetric blend of separated humps. See genconfig.shoulder_frac.
    'shoulder_half': dict(shoulder_frac=0.5, mode_share_alpha=1.0),
    'shoulder_all': dict(shoulder_frac=1.0, mode_share_alpha=1.0),
    'shoulder_all_wider': dict(shoulder_frac=1.0, mode_share_alpha=1.0,
                               trunc_iqr_mult=5.0, min_q1_over_iqr=0.02),
    'shoulder_all_widest': dict(shoulder_frac=1.0, mode_share_alpha=1.0,
                                trunc_iqr_mult=8.0, min_q1_over_iqr=0.01),
}


def empirical_targets():
    """The signs the corpus has to reproduce, measured on the real arm."""
    vm = pd.read_csv(os.path.join(TABLES, 'TABLE_VisibleModes.csv'))
    e = (pd.read_excel(os.path.join(TABLES,
                                    'TABLE_EmpiricalECCMetricsAndW1.xlsx'))
         .rename(columns={'Unnamed: 0': 'dataset'}))
    e = e.merge(vm[vm.arm == 'empirical'][['dataset', 'modes_fitted']],
                on='dataset')
    return e, {c: spearmanr(e.modes_fitted, e[c])[0] for c in CHARS
               if c in e.columns}


def visible_modes(values, metrics):
    """The mode count at the bandwidth the study fits, per dataset."""
    out = {}
    for ds, g in values.groupby('dataset_id'):
        x = g.value.to_numpy(float)
        if len(x) < 8 or np.std(x) == 0:
            continue
        try:
            out[ds] = MO.n_modes_fitted(x)
        except Exception:
            continue
    return pd.Series(out, name='modes_fitted')


def main(argv):
    n_total = int(argv[1]) if len(argv) > 1 else 1000
    emp, targets = empirical_targets()
    print('EMPIRICAL TARGET SIGNS, Spearman(visible modes, characteristic):')
    print(' ', {k: round(v, 3) for k, v in targets.items()})
    print(f'\ngenerating {len(CANDIDATES)} draft corpora at {n_total} datasets '
          'each. drafts only -- never a paper number.\n')

    rows, corr_rows = [], []
    for name, overrides in CANDIDATES.items():
        cfg = corpus.scaled_config(G.DEFAULT, n_total,
                                   n_probe=max(5, n_total // 200))
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        t0 = time.time()
        # A corpus is never overwritten (decision 26), so an existing draft for
        # this candidate is REUSED rather than relabelled. That also makes the
        # script cheap to re-run when only one candidate is added.
        label = f'draft_joint_{name}'
        d = os.path.join(ROOT, 'data', 'processed', f'corpus_{label}')
        if os.path.isdir(d):
            print(f'  {name:24s} reusing existing draft', flush=True)
        else:
            d = corpus.generate_corpus(cfg, label, progress=False)
        metrics, values, _meta = corpus.load_corpus(d)
        modes = visible_modes(values, metrics)
        m = metrics.merge(modes, left_on='dataset', right_index=True,
                          how='inner')
        row = dict(candidate=name, datasets=len(m),
                   seconds=round(time.time() - t0),
                   median_cv=m.coeffvar.median(), max_cv=m.coeffvar.max(),
                   share_cv_over_089=float((m.coeffvar > 0.889).mean()),
                   multimodal=float((m.modes_fitted >= 2).mean()),
                   multimodal_and_dispersed=float(
                       ((m.modes_fitted >= 2) & (m.coeffvar > 0.889)).mean()))
        signs_right = 0
        for c in CHARS:
            if c not in m.columns:
                continue
            rho = spearmanr(m.modes_fitted, m[c])[0]
            corr_rows.append(dict(candidate=name, characteristic=c,
                                  synthetic=rho, empirical=targets.get(c),
                                  sign_matches=bool(
                                      np.sign(rho) == np.sign(targets.get(c, 0)))))
            signs_right += int(np.sign(rho) == np.sign(targets.get(c, 0)))
        row['signs_matching'] = f'{signs_right} of {len(CHARS)}'
        # The MARGINAL objective this project has always tuned on: mean
        # standardized Wasserstein distance between the two arms, one
        # characteristic at a time. Kept so a candidate that fixes the joint
        # structure by wrecking the margins is visible as such.
        cmp = coverage.distribution_comparison(emp, m)
        row['marginal_objective'] = float(cmp.w1_standardized.mean())
        row['coeffvar_margin'] = float(
            cmp.loc[cmp.metric == 'coeffvar', 'w1_standardized'].iloc[0])
        row['weighting_margin'] = float(
            cmp.loc[cmp.metric == 'w_v_uw_wasserstein',
                    'w1_standardized'].iloc[0])
        rows.append(row)
        print(f"  {name:20s} {row['seconds']:4d}s  multimodal "
              f"{100*row['multimodal']:5.1f} pct  "
              f"multimodal+dispersed {100*row['multimodal_and_dispersed']:5.2f} pct"
              f"  signs {row['signs_matching']}", flush=True)

    summary = pd.DataFrame(rows)
    corr = pd.DataFrame(corr_rows)
    print('\n\nSUMMARY (real arm: 31.5 pct multimodal, 16.2 pct multimodal AND '
          'dispersed, max CV 6.93)')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n\nSIGN OF Spearman(visible modes, characteristic) BY CANDIDATE')
    print(corr.pivot(index='characteristic', columns='candidate',
                     values='synthetic')
          .join(corr.drop_duplicates('characteristic')
                .set_index('characteristic')['empirical'])
          .to_string(float_format=lambda v: f'{v:+.3f}'))
    os.makedirs(OUT, exist_ok=True)
    summary.to_csv(os.path.join(OUT, 'TABLE_CorpusJointCandidates.csv'),
                   index=False)
    corr.to_csv(os.path.join(OUT, 'TABLE_CorpusJointCorrelations.csv'),
                index=False)
    print(f'\nwrote {OUT}/TABLE_CorpusJointCandidates.csv and '
          'TABLE_CorpusJointCorrelations.csv')


if __name__ == '__main__':
    main(sys.argv)

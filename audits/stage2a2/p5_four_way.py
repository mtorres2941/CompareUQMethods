"""Both corpora against both empirical arms, so cause can be attributed.

Two things changed in this stage: the empirical data (a newer, raw extract
cleaned symmetrically) and the generator configuration (retuned against it).
Reporting only the new corpus against the new data cannot separate them. All
four combinations are scored on the same footing, so the reader can see how much
of any movement came from the pull and how much from the retune.

  old arm = the Stage 2a empirical arm: the 2026-03 file, already trimmed
            additively at the high end, with a log-space LOW-end bound added.
  new arm = the 2026-08 raw extract, cleaned symmetrically in log space.

The old arm is rebuilt here rather than read from a file, because src/empirical
now describes the method as it stands and carries no second rule. This is the
only place the superseded rule survives, and it exists to make the comparison
possible.

    conda run -n compareuq python p5_four_way.py <old_corpus> <new_corpus>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import coverage  # noqa: E402
import empirical  # noqa: E402
import modality as MD  # noqa: E402
from customstats import empirical_metadata  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, 'audits', 'stage2a'))
from b5_tune_configuration import WEIGHTS, MODE_TV_WEIGHT  # noqa: E402

TABLES = os.path.join(ROOT, 'outputs', 'tables', 'stage2a2')
PROCESSED = os.path.join(ROOT, 'data', 'processed')
SEED = 42


def old_arm(rng, mult=3.0, alpha=1.0, min_n=3):
    """The Stage 2a empirical arm: 2026-03 file, log-space LOW bound only."""
    src = os.path.join(PROCESSED, 'dct_realeccs_trimmed.json')
    raw = {m: np.asarray(v['data'], float) for m, v in json.load(open(src)).items()}
    out = {}
    for mat in sorted(raw):
        d = raw[mat]
        pos = d[np.isfinite(d) & (d > 0)]
        if len(pos) >= 4:
            L = np.log(pos)
            l1, l3 = np.quantile(L, [0.25, 0.75])
            li = l3 - l1
            kept = pos[L > l1 - mult * li] if li > 0 else pos
        else:
            kept = pos
        if len(kept) < min_n:
            continue
        out[mat] = (kept / np.mean(kept), rng.dirichlet(np.ones(len(kept)) * alpha))
    return out


def arm_stats(datasets, label):
    met = pd.DataFrame({m: empirical_metadata(x, w)
                        for m, (x, w) in datasets.items()}).T.astype(float)
    met.index.name = 'material'
    rng = np.random.default_rng(0)
    modes = np.array([MD.n_modes_silverman(x, rng=rng, nboot=100)
                      for x, _ in datasets.values()])
    print(f'  {label:<10} {len(met):>4} datasets   '
          f'{(modes == 1).mean()*100:5.1f}% unimodal   '
          f'median coeffvar {met.coeffvar.median():.3f}   '
          f'median skewness {met.skewness.median():.3f}   '
          f'median n {met.n.median():.0f}')
    return met.reset_index(), modes


def corpus_stats(label):
    d = os.path.join(PROCESSED, f'corpus_{label}')
    met = pd.read_parquet(os.path.join(d, 'metrics.parquet'))
    met = met[~met.is_probe]
    vals = pd.read_parquet(os.path.join(d, 'values.parquet'))
    ids = set(met.dataset.astype(str))
    rng = np.random.default_rng(1)
    modes = []
    for ds, g in vals.groupby('dataset_id', observed=True):
        if str(ds) not in ids:
            continue
        x = g['value'].to_numpy()
        if len(x) >= 4:
            modes.append(MD.n_modes_silverman(x, rng=rng, nboot=60))
    return met, np.array(modes)


def objective(emp_met, emp_modes, syn_met, syn_modes):
    d = coverage.distribution_comparison(emp_met, syn_met)
    d['weight'] = d.metric.map(WEIGHTS).fillna(1.0)
    tv = sum(abs(float((emp_modes == k).mean()) - float((syn_modes == k).mean()))
             for k in range(1, 7)) / 2.0
    num = float((d.w1_standardized * d.weight).sum()) + MODE_TV_WEIGHT * tv
    den = float(d.weight.sum()) + MODE_TV_WEIGHT
    return d, dict(weighted_objective=num / den,
                   mean_w1_unweighted=float(d.w1_standardized.mean()),
                   mode_tv=tv,
                   unimodal_synthetic=float((syn_modes == 1).mean()),
                   unimodal_empirical=float((emp_modes == 1).mean()),
                   worst_metric=d.iloc[0].metric,
                   worst_w1=float(d.iloc[0].w1_standardized))


def main(old_label, new_label):
    print('=== the two empirical arms ===')
    rng = np.random.default_rng(SEED).spawn(1)[0]
    old_ds = old_arm(rng)
    old_met, old_modes = arm_stats(old_ds, 'old arm')
    new_ds, _ = empirical.prepare(np.random.default_rng(SEED).spawn(1)[0])
    new_met, new_modes = arm_stats(new_ds, 'new arm')

    print('\n=== the two corpora ===')
    corpora = {}
    for lab in [old_label, new_label]:
        met, modes = corpus_stats(lab)
        corpora[lab] = (met, modes)
        print(f'  {lab:<22} {len(met):>6} datasets   '
              f'{(modes == 1).mean()*100:5.1f}% unimodal   '
              f'median coeffvar {met.coeffvar.median():.3f}   '
              f'median skewness {met.skewness.median():.3f}')

    print('\n=== four combinations ===')
    rows, detail = [], []
    for aname, (amet, amodes) in [('old arm', (old_met, old_modes)),
                                  ('new arm', (new_met, new_modes))]:
        for cname in [old_label, new_label]:
            smet, smodes = corpora[cname]
            d, s = objective(amet, amodes, smet, smodes)
            rows.append(dict(empirical_arm=aname, corpus=cname, **s))
            detail.append(d.assign(empirical_arm=aname, corpus=cname))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, 'TABLE_2a2_FourWayComparison.csv'), index=False)
    pd.concat(detail).to_csv(
        os.path.join(TABLES, 'TABLE_2a2_FourWayPerCharacteristic.csv'), index=False)
    pd.set_option('display.width', 220)
    print(out.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))

    print('\n=== acceptance ===')
    ref = out[(out.empirical_arm == 'old arm') & (out.corpus == old_label)].iloc[0]
    new = out[(out.empirical_arm == 'new arm') & (out.corpus == new_label)].iloc[0]
    print(f'  reference: {old_label} vs old arm   objective '
          f'{ref.weighted_objective:.4f}  mean W1 {ref.mean_w1_unweighted:.4f}  '
          f'mode TV {ref.mode_tv:.4f}')
    print(f'  now:       {new_label} vs new arm   objective '
          f'{new.weighted_objective:.4f}  mean W1 {new.mean_w1_unweighted:.4f}  '
          f'mode TV {new.mode_tv:.4f}')
    print(f'  objective at least as good : '
          f'{new.weighted_objective <= ref.weighted_objective}')
    print(f'  modality closer            : {new.mode_tv <= ref.mode_tv}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '2026-09-12b',
         sys.argv[2] if len(sys.argv) > 2 else '2026-09-12c')
